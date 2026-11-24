"""Ejecuta el pipeline de preprocesado CT/CBCT por tramos.

El ajuste fino del registro se hace A MANO en SPM, así que el pipeline no puede
ejecutarse de un tirón. Se divide en dos tramos:

    python ejecutar_pipeline.py pre-spm
        inventario -> paso1 (DICOM a NIfTI) -> paso2 (registro)

        El paso 2 se lanza en modo landmarks para los pacientes que tienen
        XXXX_CT.mrk.json y XXXX_CBCT.mrk.json en la carpeta LANDMARKS, y en
        modo automático para el resto. No hay listas de pacientes en el código.

    >>> ajuste manual en SPM ("Reorient images") de todos los pacientes,
        guardando en PACIENTES_REGISTERED/"XXXX nuevo"/ <<<

    python ejecutar_pipeline.py post-spm
        paso2c (SPM a registrado) -> paso3 (normalización, leyendo de
        PACIENTES_REGISTERED_FINAL) -> paso4 (redimensionado) -> paso5
        (máscaras de vóxeles válidos) -> validación -> división
        entrenamiento/validación

Con --simular solo se muestran los comandos, sin ejecutar nada.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes

CARPETA_LANDMARKS = "LANDMARKS"
CARPETA_REGISTRADO_FINAL = "PACIENTES_REGISTERED_FINAL"


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta el preprocesado CT/CBCT por tramos (antes y después de SPM)"
    )
    parser.add_argument(
        "tramo",
        choices=("pre-spm", "post-spm"),
        help=(
            "pre-spm: inventario, paso 1 y paso 2. "
            "post-spm: pasos 2c, 3, 4, validación y división."
        ),
    )
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help=(
            "Filtro opcional de pacientes. En post-spm solo afecta al paso 2c: "
            "los pasos 3 y 4, la validación y la división usan siempre toda la cohorte."
        ),
    )
    parser.add_argument("--sobrescribir", action="store_true")
    parser.add_argument(
        "--sin-inventario",
        action="store_true",
        help="(pre-spm) Empieza directamente por la conversión DICOM -> NIfTI.",
    )
    parser.add_argument(
        "--landmarks-dir",
        help=f"(pre-spm) Carpeta de landmarks (default: {CARPETA_LANDMARKS} junto al código).",
    )
    parser.add_argument(
        "--figuras",
        action="store_true",
        help="(post-spm) El paso 2c guarda figuras de fusión y tablero de ajedrez.",
    )
    parser.add_argument(
        "--continuar-con-errores",
        action="store_true",
        help="Continúa con el siguiente paso aunque uno devuelva error.",
    )
    parser.add_argument(
        "--simular",
        action="store_true",
        help="Muestra los comandos que se ejecutarían, sin ejecutarlos.",
    )
    return parser


class Ejecutor:
    """Lanza los scripts uno a uno y recuerda cuáles han fallado."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.script_dir = Path(__file__).resolve().parent
        self.fallos: list[tuple[str, int]] = []

    def comunes(self, *, pacientes: list[str] | None, sobrescribir: bool) -> list[str]:
        extra: list[str] = []
        if self.args.project_dir:
            extra += ["--project-dir", self.args.project_dir]
        if pacientes:
            extra += ["--pacientes", *pacientes]
        if sobrescribir and self.args.sobrescribir:
            extra.append("--sobrescribir")
        return extra

    def ejecutar(self, nombre: str, fichero: str, extra: list[str]) -> bool:
        """Devuelve True si se puede seguir con el siguiente paso."""

        comando = [sys.executable, str(self.script_dir / fichero), *extra]
        print(f"\n{'#' * 80}")
        print(f"EJECUTANDO {nombre}: {fichero} {' '.join(extra)}")
        print(f"{'#' * 80}")
        if self.args.simular:
            return True
        codigo = subprocess.run(comando, check=False).returncode
        if codigo != 0:
            self.fallos.append((nombre, codigo))
            print(f"\n✗ {nombre} terminó con código {codigo}")
            return self.args.continuar_con_errores
        return True


def _pacientes_con_landmarks(landmarks_dir: Path, pacientes: list[str]) -> list[str]:
    if not landmarks_dir.is_dir():
        return []
    return [
        pid
        for pid in pacientes
        if (landmarks_dir / f"{pid}_CT.mrk.json").is_file()
        and (landmarks_dir / f"{pid}_CBCT.mrk.json").is_file()
    ]


def tramo_pre_spm(ej: Ejecutor) -> None:
    args = ej.args
    rutas = obtener_rutas(args.project_dir)

    if not args.sin_inventario:
        if not ej.ejecutar(
            "inventario",
            "inventario_dicom.py",
            ej.comunes(pacientes=args.pacientes, sobrescribir=False),
        ):
            return

    if not ej.ejecutar(
        "paso1",
        "paso1_dicom_a_nifti.py",
        ej.comunes(pacientes=args.pacientes, sobrescribir=True),
    ):
        return

    # El reparto automático/landmarks se decide con lo que hay en disco tras el
    # paso 1. En una simulación el paso 1 no se ha ejecutado: si aún no hay
    # NIfTI, se usan los pacientes de PACIENTES para poder enseñar el plan.
    try:
        pacientes = detectar_pacientes(rutas.nifti, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        if not args.simular:
            print(f"ERROR: {exc}")
            ej.fallos.append(("paso2", 2))
            return
        pacientes = detectar_pacientes(rutas.pacientes, args.pacientes)

    landmarks_dir = (
        Path(args.landmarks_dir).expanduser().resolve()
        if args.landmarks_dir
        else ej.script_dir / CARPETA_LANDMARKS
    )
    con_landmarks = _pacientes_con_landmarks(landmarks_dir, pacientes)
    automaticos = [pid for pid in pacientes if pid not in con_landmarks]
    print(f"\nPaso 2 automático: {texto_pacientes(automaticos)}")
    print(f"Paso 2 con landmarks ({landmarks_dir}): {texto_pacientes(con_landmarks)}")

    if automaticos:
        if not ej.ejecutar(
            "paso2 (automático)",
            "paso2_registro.py",
            ej.comunes(pacientes=automaticos, sobrescribir=True),
        ):
            return
    if con_landmarks:
        if not ej.ejecutar(
            "paso2 (landmarks)",
            "paso2_registro.py",
            ej.comunes(pacientes=con_landmarks, sobrescribir=True)
            + ["--landmarks-dir", str(landmarks_dir)],
        ):
            return

    print(
        "\nSIGUIENTE (manual): ajusta cada paciente en SPM y guarda el resultado en\n"
        f"  {rutas.registered / 'XXXX nuevo' / 'XXXX_cbct_registered.nii'}  (y XXXX_ct.nii)\n"
        "Después ejecuta:  python ejecutar_pipeline.py post-spm"
    )


def tramo_post_spm(ej: Ejecutor) -> None:
    args = ej.args
    rutas = obtener_rutas(args.project_dir)
    registrado_final = rutas.proyecto / CARPETA_REGISTRADO_FINAL

    extra_2c = ej.comunes(pacientes=args.pacientes, sobrescribir=True)
    if args.figuras:
        extra_2c.append("--figuras")
    if not ej.ejecutar("paso2c", "paso2c_spm_a_registrado.py", extra_2c):
        return

    # El spacing común del paso 4 depende de TODA la cohorte, por eso los pasos
    # 3 y 4 no reciben el filtro --pacientes.
    if not ej.ejecutar(
        "paso3",
        "paso3_normalizacion.py",
        ej.comunes(pacientes=None, sobrescribir=True)
        + ["--registered-root", str(registrado_final)],
    ):
        return
    if not ej.ejecutar(
        "paso4",
        "paso4_redimensionado.py",
        ej.comunes(pacientes=None, sobrescribir=True),
    ):
        return
    # Las máscaras se calculan sobre la rejilla del paso 4: si este se repite,
    # hay que regenerarlas.
    if not ej.ejecutar(
        "paso5",
        "paso5_mascaras.py",
        ej.comunes(pacientes=None, sobrescribir=True),
    ):
        return
    if not ej.ejecutar(
        "validación",
        "validar_dataset_final.py",
        ej.comunes(pacientes=None, sobrescribir=False),
    ):
        return
    # Nunca se pasa --sobrescribir: una división ya creada se conserva para no
    # cambiar el paciente de validación sin querer.
    ej.ejecutar(
        "división",
        "dividir_dataset.py",
        ej.comunes(pacientes=None, sobrescribir=False),
    )


def main() -> int:
    args = construir_parser().parse_args()
    ej = Ejecutor(args)

    if args.tramo == "pre-spm":
        tramo_pre_spm(ej)
    else:
        tramo_post_spm(ej)

    if ej.fallos:
        print("\nTramo terminado con fallos:")
        for nombre, codigo in ej.fallos:
            print(f"  - {nombre}: código {codigo}")
        return 1

    if args.simular:
        print("\nSimulación terminada: no se ha ejecutado ningún script.")
    else:
        print(f"\n✓ Tramo {args.tramo} finalizado sin errores comunicados por los scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
