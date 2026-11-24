"""Paso 3: normalización de intensidades de CT y CBCT.

Criterio (acordado con el tutor)
--------------------------------
El mismo para las dos modalidades, con límites FIJOS en unidades Hounsfield:

1. recorte (clipping) al rango [-1024, 3071] HU;
2. escalado min-max a [0, 1] con esos mismos límites:

       valor_norm = (HU + 1024) / (3071 + 1024)

Así -1024 HU (aire y relleno) queda en 0 y 3071 HU en 1. Al ser límites fijos,
la normalización de un paciente no depende del resto de la cohorte: se puede
añadir o rehacer un paciente sin repetir los demás.

El relleno -1024 que el registro pone fuera del CBCT queda en 0 de forma
natural. En el informe se cuenta aparte (``porcentaje_padding``) y no entra en
las estadísticas del CBCT.

La versión anterior (CT en HU fijo a [-1, 1] y CBCT con percentiles globales de
la cohorte) se descartó y no se incluye en el repositorio.

Entrada
-------
En el flujo del TFG se lee la salida del paso 2c:
``--registered-root PACIENTES_REGISTERED_FINAL`` (``ejecutar_pipeline.py
post-spm`` lo pasa automáticamente). Sin esa opción el script lee
``PACIENTES_REGISTERED``, que es el registro ANTERIOR al ajuste en SPM.
Los pacientes se detectan solos, sin listas de IDs.

Salida
------
``PACIENTES_NORMALIZED/XXXX/XXXX_ct_norm.nii.gz`` y ``XXXX_cbct_norm.nii.gz``,
``informe_normalizacion.csv`` (acumulativo) y ``config_normalizacion.json``.

La geometría no se modifica. CT y CBCT deben compartir exactamente tamaño,
spacing, origen y dirección antes y después de la normalización.

Siguiente paso: ``paso4_redimensionado.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from escritura_segura import escribir_grupo_atomico
from imagen_utils import (
    diferencias_geometria,
    geometrias_coinciden,
    validar_geometria_3d,
)
from informes import guardar_informe_acumulado

VERSION_NORMALIZACION = "hu_fijo_minmax_0_1_v3"

HU_MIN_DEFAULT = -1024.0
HU_MAX_DEFAULT = 3071.0
SALIDA_MIN = 0.0
SALIDA_MAX = 1.0
PADDING_ENTRADA_DEFAULT = -1024.0
PADDING_TOLERANCIA_DEFAULT = 1e-3
# Si se recorta más de este porcentaje de vóxeles válidos se avisa por consola:
# puede indicar que la imagen no está en HU (típico en algunos CBCT).
AVISO_CLIPPING_PORCENTAJE = 1.0

CAMPOS_INFORME = [
    "patient_id",
    "modalidad",
    "estado",
    "fuente_entrada",
    "metodo",
    "limite_inferior",
    "limite_superior",
    "voxeles_totales",
    "voxeles_validos",
    "porcentaje_padding",
    "min_antes",
    "max_antes",
    "media_antes",
    "std_antes",
    "porcentaje_clipped_bajo",
    "porcentaje_clipped_alto",
    "min_despues",
    "max_despues",
    "media_despues",
    "std_despues",
    "ruta_salida",
    "mensaje",
]


def _resolver_ruta(valor: str | None, predeterminada: Path, proyecto: Path) -> Path:
    if valor is None:
        return predeterminada.resolve()
    ruta = Path(valor).expanduser()
    if not ruta.is_absolute():
        ruta = proyecto / ruta
    return ruta.resolve()


def _validar_par(ct: sitk.Image, cbct: sitk.Image, patient_id: str) -> None:
    validar_geometria_3d(ct, f"CT {patient_id}")
    validar_geometria_3d(cbct, f"CBCT {patient_id}")
    if not geometrias_coinciden(ct, cbct):
        raise ValueError(
            f"Paciente {patient_id}: el par registrado no comparte geometría: "
            + "; ".join(diferencias_geometria(ct, cbct))
        )


def normalizar_array(
    array: np.ndarray,
    hu_min: float = HU_MIN_DEFAULT,
    hu_max: float = HU_MAX_DEFAULT,
    padding_entrada: float | None = None,
    tolerancia_padding: float = PADDING_TOLERANCIA_DEFAULT,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Recorta a [hu_min, hu_max] y escala a [0, 1]. Devuelve array y estadísticas.

    ``padding_entrada`` solo afecta a las ESTADÍSTICAS: los vóxeles con ese valor
    (relleno del registro) no cuentan como válidos. La normalización es la misma
    para todos los vóxeles; el relleno -1024 queda en 0.
    """

    if not np.isfinite([hu_min, hu_max]).all() or hu_min >= hu_max:
        raise ValueError("Los límites HU deben ser finitos y cumplir min < max")
    if array.size == 0:
        raise ValueError("La imagen está vacía")

    entrada = np.asarray(array, dtype=np.float32)
    if not np.isfinite(entrada).all():
        raise ValueError("La imagen contiene NaN o infinitos")

    if padding_entrada is None:
        validos = entrada.reshape(-1)
        n_padding = 0
    else:
        es_padding = np.isclose(
            entrada, np.float32(padding_entrada), rtol=0.0, atol=float(tolerancia_padding)
        )
        n_padding = int(np.count_nonzero(es_padding))
        validos = entrada[~es_padding]
    if validos.size == 0:
        raise ValueError("No quedan vóxeles válidos tras excluir el relleno")

    salida = np.clip(entrada, hu_min, hu_max)
    salida -= np.float32(hu_min)
    salida /= np.float32(hu_max - hu_min)
    # Protege frente a errores de redondeo en los extremos.
    np.clip(salida, SALIDA_MIN, SALIDA_MAX, out=salida)

    validos_salida = salida.reshape(-1) if padding_entrada is None else salida[~es_padding]
    n_validos = int(validos.size)
    stats: dict[str, float | int] = {
        "voxeles_totales": int(entrada.size),
        "voxeles_validos": n_validos,
        "porcentaje_padding": 100.0 * n_padding / entrada.size,
        "min_antes": float(validos.min()),
        "max_antes": float(validos.max()),
        "media_antes": float(validos.mean(dtype=np.float64)),
        "std_antes": float(validos.std(dtype=np.float64)),
        "porcentaje_clipped_bajo": 100.0 * int(np.count_nonzero(validos < hu_min)) / n_validos,
        "porcentaje_clipped_alto": 100.0 * int(np.count_nonzero(validos > hu_max)) / n_validos,
        "min_despues": float(validos_salida.min()),
        "max_despues": float(validos_salida.max()),
        "media_despues": float(validos_salida.mean(dtype=np.float64)),
        "std_despues": float(validos_salida.std(dtype=np.float64)),
    }
    return salida, stats


def normalizar_imagen(
    imagen: sitk.Image,
    hu_min: float,
    hu_max: float,
    padding_entrada: float | None = None,
    tolerancia_padding: float = PADDING_TOLERANCIA_DEFAULT,
) -> tuple[sitk.Image, dict[str, float | int]]:
    """Normaliza una imagen SimpleITK conservando su geometría."""

    validar_geometria_3d(imagen, "imagen a normalizar")
    array, stats = normalizar_array(
        sitk.GetArrayFromImage(imagen),
        hu_min=hu_min,
        hu_max=hu_max,
        padding_entrada=padding_entrada,
        tolerancia_padding=tolerancia_padding,
    )
    salida = sitk.GetImageFromArray(array)
    salida.CopyInformation(imagen)
    return salida, stats


def validar_imagen_normalizada(
    imagen: sitk.Image,
    nombre: str,
    tolerancia: float = 1e-5,
) -> None:
    validar_geometria_3d(imagen, nombre)
    array = sitk.GetArrayViewFromImage(imagen)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{nombre}: imagen vacía o con NaN/infinitos")
    minimo = float(array.min())
    maximo = float(array.max())
    if minimo < SALIDA_MIN - tolerancia or maximo > SALIDA_MAX + tolerancia:
        raise ValueError(f"{nombre}: rango [{minimo}, {maximo}] fuera de [0, 1]")


def _imprimir_estadisticas(nombre: str, stats: dict[str, float | int]) -> None:
    print(
        f"    {nombre} antes  : min={float(stats['min_antes']):.2f} "
        f"max={float(stats['max_antes']):.2f} "
        f"media={float(stats['media_antes']):.2f} "
        f"std={float(stats['std_antes']):.2f}"
    )
    print(
        f"    {nombre} después: min={float(stats['min_despues']):.5f} "
        f"max={float(stats['max_despues']):.5f} "
        f"media={float(stats['media_despues']):.5f} "
        f"std={float(stats['std_despues']):.5f}"
    )
    bajo = float(stats["porcentaje_clipped_bajo"])
    alto = float(stats["porcentaje_clipped_alto"])
    print(
        f"    {nombre} clipping: {bajo:.3f}% bajo, {alto:.3f}% alto, "
        f"relleno={float(stats['porcentaje_padding']):.2f}%"
    )
    if bajo + alto > AVISO_CLIPPING_PORCENTAJE:
        print(
            f"    AVISO {nombre.strip()}: se recorta el {bajo + alto:.2f}% de los vóxeles. "
            "Compruebe que la imagen está realmente en HU."
        )


def normalizar_paciente(
    patient_id: str,
    registered_root: Path,
    output_root: Path,
    hu_min: float,
    hu_max: float,
    padding_entrada: float | None,
    tolerancia_padding: float,
    sobrescribir: bool,
) -> list[dict[str, object]]:
    print(f"\n{'=' * 72}")
    print(f"Normalizando paciente: {patient_id}")
    print(f"{'=' * 72}")

    entrada_dir = registered_root / patient_id
    ruta_ct = entrada_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct = entrada_dir / f"{patient_id}_cbct_registered.nii.gz"
    salida_dir = output_root / patient_id
    ruta_ct_salida = salida_dir / f"{patient_id}_ct_norm.nii.gz"
    ruta_cbct_salida = salida_dir / f"{patient_id}_cbct_norm.nii.gz"

    base_error: dict[str, object] = {
        "patient_id": patient_id,
        "modalidad": "PAR",
        "estado": "ERROR",
        "fuente_entrada": registered_root.name,
        "metodo": "HU_fijo_minmax_0_1",
        "limite_inferior": hu_min,
        "limite_superior": hu_max,
        "mensaje": "",
    }

    faltantes = [ruta for ruta in (ruta_ct, ruta_cbct) if not ruta.is_file()]
    if faltantes:
        base_error["mensaje"] = "Faltan entradas: " + ", ".join(str(r) for r in faltantes)
        print(f"  ✗ {base_error['mensaje']}")
        return [base_error]

    if ruta_ct_salida.exists() and ruta_cbct_salida.exists() and not sobrescribir:
        try:
            ct_existente = sitk.ReadImage(str(ruta_ct_salida))
            cbct_existente = sitk.ReadImage(str(ruta_cbct_salida))
            validar_imagen_normalizada(ct_existente, "CT normalizado existente")
            validar_imagen_normalizada(cbct_existente, "CBCT normalizado existente")
            if not geometrias_coinciden(ct_existente, cbct_existente):
                raise ValueError(
                    "El par existente no comparte geometría: "
                    + "; ".join(diferencias_geometria(ct_existente, cbct_existente))
                )
        except Exception as exc:
            fila = dict(base_error)
            fila["mensaje"] = (
                "Las salidas existentes no son válidas; use --sobrescribir: "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"  ✗ {fila['mensaje']}")
            return [fila]
        fila = dict(base_error)
        fila.update(estado="OMITIDO", mensaje="Salidas existentes validadas")
        print("  ↷ Las salidas válidas ya existen; se omite.")
        return [fila]

    try:
        ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
        cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
        _validar_par(ct, cbct, patient_id)

        ct_norm, stats_ct = normalizar_imagen(ct, hu_min, hu_max)
        cbct_norm, stats_cbct = normalizar_imagen(
            cbct,
            hu_min,
            hu_max,
            padding_entrada=padding_entrada,
            tolerancia_padding=tolerancia_padding,
        )

        _imprimir_estadisticas("CT  ", stats_ct)
        _imprimir_estadisticas("CBCT", stats_cbct)

        if not geometrias_coinciden(ct_norm, cbct_norm):
            raise RuntimeError("La normalización alteró la geometría del par")
        validar_imagen_normalizada(ct_norm, "CT normalizado")
        validar_imagen_normalizada(cbct_norm, "CBCT normalizado")

        escribir_grupo_atomico(
            [
                (
                    ruta_ct_salida,
                    lambda temporal: sitk.WriteImage(ct_norm, str(temporal), True),
                ),
                (
                    ruta_cbct_salida,
                    lambda temporal: sitk.WriteImage(cbct_norm, str(temporal), True),
                ),
            ]
        )
    except Exception as exc:
        base_error["mensaje"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ {base_error['mensaje']}")
        return [base_error]

    filas: list[dict[str, object]] = []
    for modalidad, stats, ruta_salida in (
        ("CT", stats_ct, ruta_ct_salida),
        ("CBCT", stats_cbct, ruta_cbct_salida),
    ):
        fila = dict(base_error)
        fila.update(
            modalidad=modalidad,
            estado="OK",
            ruta_salida=str(ruta_salida),
            mensaje="Normalización completada",
        )
        fila.update(stats)
        filas.append(fila)

    print(f"  ✓ Par normalizado guardado en {salida_dir}")
    return filas


def guardar_informe(output_root: Path, filas: list[dict[str, object]]) -> Path:
    """Informe acumulativo: conserva los pacientes no procesados en esta ejecución."""

    output_root.mkdir(parents=True, exist_ok=True)
    return guardar_informe_acumulado(
        output_root / "informe_normalizacion.csv",
        filas,
        CAMPOS_INFORME,
    )


def _guardar_configuracion(
    output_root: Path,
    registered_root: Path,
    hu_min: float,
    hu_max: float,
    padding_entrada: float | None,
) -> Path:
    ruta = output_root / "config_normalizacion.json"
    contenido = {
        "version": VERSION_NORMALIZACION,
        "metodo": "clipping en HU fijo y min-max, igual para CT y CBCT",
        "entrada": str(registered_root),
        "hu_min": hu_min,
        "hu_max": hu_max,
        "rango_salida": [SALIDA_MIN, SALIDA_MAX],
        "formula": "(clip(HU, hu_min, hu_max) - hu_min) / (hu_max - hu_min)",
        "padding_entrada_cbct": padding_entrada,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporal.replace(ruta)
    return ruta


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paso 3: clipping a [-1024, 3071] HU y min-max a [0, 1], "
            "igual para CT y CBCT"
        )
    )
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional. Sin él se detectan todos en la carpeta de entrada.",
    )
    parser.add_argument(
        "--registered-root",
        help=(
            "Carpeta con los pares registrados XXXX/XXXX_ct.nii.gz y "
            "XXXX/XXXX_cbct_registered.nii.gz (default: PACIENTES_REGISTERED). "
            "Para el flujo con SPM usar PACIENTES_REGISTERED_FINAL."
        ),
    )
    parser.add_argument("--hu-min", type=float, default=HU_MIN_DEFAULT)
    parser.add_argument("--hu-max", type=float, default=HU_MAX_DEFAULT)
    parser.add_argument(
        "--padding-entrada",
        type=float,
        default=PADDING_ENTRADA_DEFAULT,
        help=(
            "Valor de relleno que el registro pone fuera del CBCT. Solo se usa "
            "para excluirlo de las estadísticas del informe."
        ),
    )
    parser.add_argument(
        "--tolerancia-padding",
        type=float,
        default=PADDING_TOLERANCIA_DEFAULT,
    )
    parser.add_argument("--sobrescribir", action="store_true")
    return parser


def main() -> int:
    args = construir_parser().parse_args()

    if not np.isfinite([args.hu_min, args.hu_max]).all() or args.hu_min >= args.hu_max:
        print("ERROR: --hu-min y --hu-max deben ser finitos y cumplir min < max")
        return 2
    if not math.isfinite(args.tolerancia_padding) or args.tolerancia_padding < 0:
        print("ERROR: --tolerancia-padding debe ser finita y no negativa")
        return 2

    rutas = obtener_rutas(args.project_dir)
    registered_root = _resolver_ruta(
        args.registered_root,
        rutas.registered,
        rutas.proyecto,
    )

    try:
        pacientes = detectar_pacientes(registered_root, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if not pacientes:
        print(f"ERROR: no hay pacientes con formato XXXX en {registered_root}")
        return 2

    hu_min = float(args.hu_min)
    hu_max = float(args.hu_max)
    padding_entrada = float(args.padding_entrada)

    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")
    print(f"Entrada: {registered_root}")
    print(f"CT y CBCT: clipping [{hu_min:g}, {hu_max:g}] HU -> min-max [0, 1]")

    filas: list[dict[str, object]] = []
    for patient_id in pacientes:
        filas.extend(
            normalizar_paciente(
                patient_id=patient_id,
                registered_root=registered_root,
                output_root=rutas.normalized,
                hu_min=hu_min,
                hu_max=hu_max,
                padding_entrada=padding_entrada,
                tolerancia_padding=float(args.tolerancia_padding),
                sobrescribir=args.sobrescribir,
            )
        )

    informe = guardar_informe(rutas.normalized, filas)
    config = _guardar_configuracion(
        rutas.normalized,
        registered_root,
        hu_min,
        hu_max,
        padding_entrada,
    )

    errores = [fila for fila in filas if fila.get("estado") == "ERROR"]
    print(f"\nPaso 3 terminado. Informe: {informe}")
    print(f"Configuración reproducible: {config}")
    if errores:
        print("Pacientes con error:")
        for fila in errores:
            print(f"  - {fila['patient_id']}: {fila['mensaje']}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
