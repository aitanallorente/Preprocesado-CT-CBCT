"""División del dataset final en entrenamiento y validación (6 a 1 por defecto).

Lee ``PACIENTES_FINAL`` (salida del paso 4) y reparte los PACIENTES, nunca los
cortes sueltos: todos los cortes de un paciente van al mismo conjunto, para que
la validación no contenga anatomía ya vista en el entrenamiento.

Con tan pocos pacientes no hay test independiente: un paciente se aparta para
validación y el resto se usa para entrenar.

El paciente de validación se elige al azar con una semilla fija (``--semilla``,
42 por defecto), de modo que el reparto es reproducible. También se puede fijar
a mano con ``--validacion 0012``.

No mueve ni copia ningún volumen: solo escribe
``PACIENTES_FINAL/division_entrenamiento_validacion.json`` con las listas de
pacientes y las rutas relativas de cada par. El código de entrenamiento debe
leer ese archivo.

Al terminar, el script imprime qué pacientes van a cada conjunto y cuántos
pares hay en cada uno.

Para no cambiar el conjunto de validación sin querer, si la división ya existe
NO se recalcula. Si se añaden pacientes hay que regenerarla con
``--sobrescribir`` (y, por tanto, volver a entrenar).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes

NOMBRE_DIVISION = "division_entrenamiento_validacion.json"
SEMILLA_DEFAULT = 42
N_VALIDACION_DEFAULT = 1


def elegir_validacion(
    pacientes: list[str],
    n_validacion: int = N_VALIDACION_DEFAULT,
    semilla: int = SEMILLA_DEFAULT,
    validacion_manual: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Devuelve (entrenamiento, validacion), ambas listas ordenadas."""

    pacientes = sorted(pacientes)
    if validacion_manual:
        desconocidos = sorted(set(validacion_manual) - set(pacientes))
        if desconocidos:
            raise ValueError(
                "Pacientes de validación que no están en el dataset: "
                + ", ".join(desconocidos)
            )
        validacion = sorted(set(validacion_manual))
    else:
        if not 1 <= n_validacion < len(pacientes):
            raise ValueError(
                f"n_validacion={n_validacion} no es válido para {len(pacientes)} pacientes"
            )
        validacion = sorted(random.Random(semilla).sample(pacientes, n_validacion))

    entrenamiento = [pid for pid in pacientes if pid not in validacion]
    if not entrenamiento:
        raise ValueError("No queda ningún paciente para entrenamiento")
    return entrenamiento, validacion


def _entradas(final_root: Path, pacientes: list[str]) -> list[dict[str, str]]:
    entradas = []
    for pid in pacientes:
        ct = final_root / pid / f"{pid}_ct_final.nii.gz"
        cbct = final_root / pid / f"{pid}_cbct_final.nii.gz"
        if not ct.is_file() or not cbct.is_file():
            raise FileNotFoundError(f"Paciente {pid}: faltan CT y/o CBCT final")
        entradas.append(
            {
                "patient_id": pid,
                "ct": ct.relative_to(final_root).as_posix(),
                "cbct": cbct.relative_to(final_root).as_posix(),
            }
        )
    return entradas


def _imprimir_resumen(
    ruta: Path,
    entrenamiento: list[str],
    validacion: list[str],
    nuevo: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print("DIVISIÓN ENTRENAMIENTO / VALIDACIÓN")
    print(f"{'=' * 60}")
    print("Reparto por paciente (todos los cortes de un paciente juntos).")
    print(f"  Entrenamiento ({len(entrenamiento)}): {texto_pacientes(entrenamiento)}")
    print(f"  Validación    ({len(validacion)}): {texto_pacientes(validacion)}")
    print(f"  Proporción: {len(entrenamiento)} a {len(validacion)}")
    estado = "guardada" if nuevo else "ya existía y se conserva"
    print(f"Archivo ({estado}): {ruta}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Divide PACIENTES_FINAL en entrenamiento y validación por paciente"
    )
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional. Sin él se usan todos los de PACIENTES_FINAL.",
    )
    parser.add_argument(
        "--n-validacion",
        type=int,
        default=N_VALIDACION_DEFAULT,
        help="Número de pacientes de validación (default: 1, es decir, 6 a 1 con 7 pacientes).",
    )
    parser.add_argument("--semilla", type=int, default=SEMILLA_DEFAULT)
    parser.add_argument(
        "--validacion",
        nargs="+",
        help="Fija a mano los pacientes de validación (ignora --n-validacion y --semilla).",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Recalcula la división aunque ya exista.",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    rutas = obtener_rutas(args.project_dir)
    ruta_division = rutas.final / NOMBRE_DIVISION

    try:
        pacientes = detectar_pacientes(rutas.final, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if len(pacientes) < 2:
        print(f"ERROR: hacen falta al menos 2 pacientes en {rutas.final}")
        return 2

    if ruta_division.is_file() and not args.sobrescribir:
        existente = json.loads(ruta_division.read_text(encoding="utf-8"))
        previos = sorted(existente.get("entrenamiento", []) + existente.get("validacion", []))
        if previos != pacientes:
            print(
                "ERROR: ya existe una división, pero con otros pacientes.\n"
                f"  En la división: {texto_pacientes(previos)}\n"
                f"  En el dataset : {texto_pacientes(pacientes)}\n"
                "Regenérela con --sobrescribir (el conjunto de validación puede cambiar)."
            )
            return 1
        _imprimir_resumen(
            ruta_division,
            existente["entrenamiento"],
            existente["validacion"],
            nuevo=False,
        )
        return 0

    try:
        entrenamiento, validacion = elegir_validacion(
            pacientes,
            n_validacion=args.n_validacion,
            semilla=args.semilla,
            validacion_manual=args.validacion,
        )
        manual = bool(args.validacion)
        contenido = {
            "criterio": "por paciente; "
            + ("validación fijada a mano" if manual else "validación al azar con semilla"),
            "semilla": None if manual else args.semilla,
            "n_entrenamiento": len(entrenamiento),
            "n_validacion": len(validacion),
            "entrenamiento": entrenamiento,
            "validacion": validacion,
            "pares_entrenamiento": _entradas(rutas.final, entrenamiento),
            "pares_validacion": _entradas(rutas.final, validacion),
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    temporal = ruta_division.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporal.replace(ruta_division)

    _imprimir_resumen(ruta_division, entrenamiento, validacion, nuevo=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
