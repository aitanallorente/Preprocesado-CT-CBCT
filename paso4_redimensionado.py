"""Paso 4: homogeneización espacial final de los pares CT/CBCT.

Entrada: ``PACIENTES_NORMALIZED`` (salida del paso 3).

Genera volúmenes de tamaño fijo (por defecto 256 x 256 x 128) en
``PACIENTES_FINAL``. Todos los pacientes usan el MISMO spacing final, calculado
para cubrir el mayor campo físico de la cohorte seleccionada. De esta forma:

* CT y CBCT de cada paciente permanecen en la misma rejilla;
* todos los tensores tienen el mismo tamaño;
* la escala física por vóxel es común entre pacientes;
* ningún volumen se recorta: la rejilla final contiene el campo completo de
  todos los pacientes.

Las zonas fuera del volumen de entrada se rellenan con 0, valor coherente con
la salida normalizada.

Salida: ``PACIENTES_FINAL/XXXX/XXXX_ct_final.nii.gz`` y
``XXXX_cbct_final.nii.gz``, ``config_redimensionado.json`` e
``informe_redimensionado.csv``, que es ACUMULATIVO (una fila CT y otra CBCT por
paciente): una ejecución sustituye solo las filas de los pacientes procesados y
conserva las del resto.

IMPORTANTE: el spacing común depende de TODA la cohorte. Si se añade o se
rehace un paciente, hay que volver a ejecutar este paso con todos los pacientes
y ``--sobrescribir``; si no, los pares ya generados quedarían con otro spacing.

Siguiente paso: ``validar_dataset_final.py``.
"""

from __future__ import annotations

import argparse
import itertools
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

VERSION_REDIMENSIONADO = "rejilla_comun_cohorte_v3"

TARGET_X_DEFAULT = 256
TARGET_Y_DEFAULT = 256
TARGET_Z_DEFAULT = 128
PADDING_VALUE_DEFAULT = 0.0
MARGEN_FOV_MM_DEFAULT = 2.0
REDONDEO_SPACING_MM_DEFAULT = 0.05

CAMPOS_INFORME = [
    "patient_id",
    "modalidad",
    "estado",
    "size_x_original",
    "size_y_original",
    "size_z_original",
    "spacing_x_original",
    "spacing_y_original",
    "spacing_z_original",
    "extent_x_original_mm",
    "extent_y_original_mm",
    "extent_z_original_mm",
    "target_x",
    "target_y",
    "target_z",
    "spacing_x_final",
    "spacing_y_final",
    "spacing_z_final",
    "origin_x_final",
    "origin_y_final",
    "origin_z_final",
    "padding_value",
    "interpolador_cbct",
    "contenido_completo",
    "min",
    "max",
    "media",
    "std",
    "porcentaje_relleno_geometrico",
    "ruta_salida",
    "mensaje",
]


def extension_fisica(imagen: sitk.Image) -> tuple[float, float, float]:
    """Extensión centro-a-centro de la rejilla a lo largo de sus ejes."""

    validar_geometria_3d(imagen, "imagen para calcular extensión")
    return tuple(
        max(0.0, (int(n) - 1) * float(s))
        for n, s in zip(imagen.GetSize(), imagen.GetSpacing())
    )


def centro_fisico(imagen: sitk.Image) -> np.ndarray:
    validar_geometria_3d(imagen, "imagen para calcular centro")
    indice = [(int(n) - 1.0) / 2.0 for n in imagen.GetSize()]
    return np.asarray(
        imagen.TransformContinuousIndexToPhysicalPoint(indice),
        dtype=float,
    )


def _redondear_hacia_arriba(valor: float, paso: float) -> float:
    if paso <= 0:
        return float(valor)
    return float(math.ceil((valor - 1e-12) / paso) * paso)


def calcular_spacing_comun(
    imagenes: list[sitk.Image],
    target_size: tuple[int, int, int],
    margen_fov_mm: float,
    redondeo_spacing_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Calcula un spacing único que contiene toda la cohorte sin recortar."""

    if not imagenes:
        raise ValueError("No hay imágenes para calcular el spacing común")
    if min(target_size) < 2:
        raise ValueError("Cada dimensión objetivo debe ser al menos 2")

    extensiones = np.asarray([extension_fisica(img) for img in imagenes], dtype=float)
    max_extent = np.max(extensiones, axis=0)
    requerido = max_extent + 2.0 * float(margen_fov_mm)
    denominador = np.asarray(target_size, dtype=float) - 1.0
    spacing = requerido / denominador
    spacing = np.asarray(
        [
            _redondear_hacia_arriba(float(v), float(redondeo_spacing_mm))
            for v in spacing
        ],
        dtype=float,
    )
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Spacing común no válido: {spacing}")

    fov_final = spacing * denominador
    return tuple(float(v) for v in spacing), tuple(float(v) for v in fov_final)


def crear_referencia_centrada(
    imagen: sitk.Image,
    target_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> sitk.Image:
    """Crea una rejilla fija centrada en la imagen y con la misma dirección."""

    validar_geometria_3d(imagen, "imagen de referencia")
    if min(target_size) < 1:
        raise ValueError("El tamaño objetivo debe ser positivo")
    if not np.isfinite(target_spacing).all() or min(target_spacing) <= 0:
        raise ValueError("El spacing objetivo debe ser finito y positivo")

    direction = np.asarray(imagen.GetDirection(), dtype=float).reshape(3, 3)
    centro = centro_fisico(imagen)
    semiextension_indices = (
        (np.asarray(target_size, dtype=float) - 1.0)
        / 2.0
        * np.asarray(target_spacing, dtype=float)
    )
    nuevo_origen = centro - direction @ semiextension_indices

    referencia = sitk.Image(list(target_size), sitk.sitkFloat32)
    referencia.SetSpacing(tuple(float(v) for v in target_spacing))
    referencia.SetOrigin(tuple(float(v) for v in nuevo_origen))
    referencia.SetDirection(tuple(float(v) for v in direction.ravel()))
    return referencia


def verificar_imagen_dentro_referencia(
    imagen: sitk.Image,
    referencia: sitk.Image,
    tolerancia_voxel: float = 0.51,
) -> tuple[bool, tuple[float, float, float], tuple[float, float, float]]:
    """Comprueba que las ocho esquinas de entrada caben en la salida."""

    size_in = tuple(int(v) for v in imagen.GetSize())
    size_out = np.asarray(referencia.GetSize(), dtype=float)
    indices_salida: list[np.ndarray] = []

    for esquina in itertools.product(*[(0, n - 1) for n in size_in]):
        punto = imagen.TransformIndexToPhysicalPoint([int(v) for v in esquina])
        indice = np.asarray(
            referencia.TransformPhysicalPointToContinuousIndex(punto),
            dtype=float,
        )
        indices_salida.append(indice)

    matriz = np.vstack(indices_salida)
    minimo = np.min(matriz, axis=0)
    maximo = np.max(matriz, axis=0)
    cabe = bool(
        np.all(minimo >= -float(tolerancia_voxel))
        and np.all(maximo <= size_out - 1.0 + float(tolerancia_voxel))
    )
    return (
        cabe,
        tuple(float(v) for v in minimo),
        tuple(float(v) for v in maximo),
    )


def resamplear_a_referencia(
    imagen: sitk.Image,
    referencia: sitk.Image,
    interpolador: int,
    padding_value: float,
) -> sitk.Image:
    identidad = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(
        imagen,
        referencia,
        identidad,
        interpolador,
        float(padding_value),
        sitk.sitkFloat32,
    )


def _estadisticas(
    imagen: sitk.Image,
    porcentaje_relleno_geometrico: float,
) -> dict[str, float]:
    array = sitk.GetArrayViewFromImage(imagen)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("La imagen está vacía o contiene NaN/infinitos")
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "media": float(array.mean()),
        "std": float(array.std()),
        "porcentaje_relleno_geometrico": float(porcentaje_relleno_geometrico),
    }


def calcular_porcentaje_relleno_geometrico(
    imagen: sitk.Image,
    referencia: sitk.Image,
) -> float:
    """Porcentaje de la rejilla final que queda fuera del volumen de entrada."""

    unos = sitk.Image(imagen.GetSize(), sitk.sitkUInt8)
    unos.CopyInformation(imagen)
    unos = unos + 1
    identidad = sitk.Transform(3, sitk.sitkIdentity)
    cobertura = sitk.Resample(
        unos,
        referencia,
        identidad,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    array = sitk.GetArrayViewFromImage(cobertura)
    if array.size == 0:
        raise ValueError("No se pudo calcular la cobertura geométrica")
    dentro = float(np.count_nonzero(array)) / float(array.size)
    return 100.0 * (1.0 - dentro)


def _validar_normalizada(imagen: sitk.Image, nombre: str) -> None:
    validar_geometria_3d(imagen, nombre)
    array = sitk.GetArrayViewFromImage(imagen)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{nombre}: imagen vacía o con NaN/infinitos")
    minimo = float(array.min())
    maximo = float(array.max())
    if minimo < -0.00001 or maximo > 1.00001:
        raise ValueError(
            f"{nombre}: rango [{minimo}, {maximo}] fuera de [0,1]. "
            "Ejecute antes paso3_normalizacion.py."
        )


def _validar_salida_final(
    imagen: sitk.Image,
    nombre: str,
    esperado: tuple[int, int, int],
    spacing_esperado: tuple[float, float, float],
    porcentaje_relleno_geometrico: float,
) -> dict[str, float]:
    _validar_normalizada(imagen, nombre)
    if tuple(int(v) for v in imagen.GetSize()) != esperado:
        raise ValueError(
            f"{nombre}: tamaño {imagen.GetSize()} distinto del esperado {esperado}"
        )
    if not np.allclose(
        np.asarray(imagen.GetSpacing(), dtype=float),
        np.asarray(spacing_esperado, dtype=float),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{nombre}: spacing {imagen.GetSpacing()} distinto de {spacing_esperado}"
        )
    return _estadisticas(imagen, porcentaje_relleno_geometrico)


def redimensionar_paciente(
    patient_id: str,
    normalized_root: Path,
    output_root: Path,
    target_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
    padding_value: float,
    interpolador_cbct: str,
    sobrescribir: bool,
) -> list[dict[str, object]]:
    print(f"\n{'=' * 72}")
    print(f"Redimensionando paciente: {patient_id}")
    print(f"{'=' * 72}")

    entrada_dir = normalized_root / patient_id
    salida_dir = output_root / patient_id
    ruta_ct = entrada_dir / f"{patient_id}_ct_norm.nii.gz"
    ruta_cbct = entrada_dir / f"{patient_id}_cbct_norm.nii.gz"
    ruta_ct_salida = salida_dir / f"{patient_id}_ct_final.nii.gz"
    ruta_cbct_salida = salida_dir / f"{patient_id}_cbct_final.nii.gz"

    fila_error: dict[str, object] = {
        "patient_id": patient_id,
        "modalidad": "PAR",
        "estado": "ERROR",
        "target_x": target_size[0],
        "target_y": target_size[1],
        "target_z": target_size[2],
        "spacing_x_final": target_spacing[0],
        "spacing_y_final": target_spacing[1],
        "spacing_z_final": target_spacing[2],
        "interpolador_cbct": interpolador_cbct,
        "padding_value": padding_value,
        "mensaje": "",
    }

    faltantes = [ruta for ruta in (ruta_ct, ruta_cbct) if not ruta.is_file()]
    if faltantes:
        fila_error["mensaje"] = "Faltan entradas: " + ", ".join(str(r) for r in faltantes)
        print(f"  ✗ {fila_error['mensaje']}")
        return [fila_error]

    if ruta_ct_salida.exists() and ruta_cbct_salida.exists() and not sobrescribir:
        try:
            ct_existente = sitk.ReadImage(str(ruta_ct_salida))
            cbct_existente = sitk.ReadImage(str(ruta_cbct_salida))
            # En la validación de salidas preexistentes no se dispone ya de la
            # máscara de cobertura original. Se conserva el campo vacío en el CSV.
            _validar_salida_final(
                ct_existente,
                "CT final existente",
                target_size,
                target_spacing,
                float("nan"),
            )
            _validar_salida_final(
                cbct_existente,
                "CBCT final existente",
                target_size,
                target_spacing,
                float("nan"),
            )
            if not geometrias_coinciden(ct_existente, cbct_existente):
                raise ValueError(
                    "El par final existente no comparte geometría: "
                    + "; ".join(diferencias_geometria(ct_existente, cbct_existente))
                )
        except Exception as exc:
            fila = dict(fila_error)
            fila["mensaje"] = (
                "Las salidas existentes no son válidas; use --sobrescribir: "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"  ✗ {fila['mensaje']}")
            return [fila]
        fila = dict(fila_error)
        fila.update(estado="OMITIDO", mensaje="Salidas existentes validadas")
        print("  ↷ Las salidas válidas ya existen; se omite.")
        return [fila]

    try:
        ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
        cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
        _validar_normalizada(ct, "CT normalizado")
        _validar_normalizada(cbct, "CBCT normalizado")
        if not geometrias_coinciden(ct, cbct):
            raise ValueError(
                "El par normalizado no comparte geometría: "
                + "; ".join(diferencias_geometria(ct, cbct))
            )

        size_original = tuple(int(v) for v in ct.GetSize())
        spacing_original = tuple(float(v) for v in ct.GetSpacing())
        extent_original = extension_fisica(ct)
        referencia = crear_referencia_centrada(ct, target_size, target_spacing)

        cabe, indice_min, indice_max = verificar_imagen_dentro_referencia(ct, referencia)
        if not cabe:
            raise ValueError(
                "La rejilla objetivo recortaría el volumen. "
                f"Índices de las esquinas en salida: min={indice_min}, max={indice_max}. "
                "Aumente el tamaño o el spacing objetivo."
            )

        if interpolador_cbct == "lineal":
            interp_cbct = sitk.sitkLinear
        elif interpolador_cbct == "vecino":
            interp_cbct = sitk.sitkNearestNeighbor
        else:
            raise ValueError(f"Interpolador CBCT no soportado: {interpolador_cbct}")

        porcentaje_relleno = calcular_porcentaje_relleno_geometrico(ct, referencia)

        ct_final = resamplear_a_referencia(
            ct, referencia, sitk.sitkLinear, padding_value
        )
        cbct_final = resamplear_a_referencia(
            cbct, referencia, interp_cbct, padding_value
        )

        if not geometrias_coinciden(ct_final, cbct_final):
            raise RuntimeError(
                "La geometría final del par no coincide: "
                + "; ".join(diferencias_geometria(ct_final, cbct_final))
            )

        stats_ct = _validar_salida_final(
            ct_final,
            "CT final",
            target_size,
            target_spacing,
            porcentaje_relleno,
        )
        stats_cbct = _validar_salida_final(
            cbct_final,
            "CBCT final",
            target_size,
            target_spacing,
            porcentaje_relleno,
        )

        escribir_grupo_atomico(
            [
                (
                    ruta_ct_salida,
                    lambda temporal: sitk.WriteImage(ct_final, str(temporal), True),
                ),
                (
                    ruta_cbct_salida,
                    lambda temporal: sitk.WriteImage(cbct_final, str(temporal), True),
                ),
            ]
        )
    except Exception as exc:
        fila_error["mensaje"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ {fila_error['mensaje']}")
        return [fila_error]

    print(f"  Tamaño/spacing original: {size_original} / {spacing_original}")
    print(
        "  Extensión física original (mm): "
        + str(tuple(round(v, 2) for v in extent_original))
    )
    print(
        f"  ✓ Tamaño final: {target_size} | spacing común: "
        f"{tuple(round(v, 5) for v in target_spacing)}"
    )

    filas: list[dict[str, object]] = []
    for modalidad, stats, ruta_salida in (
        ("CT", stats_ct, ruta_ct_salida),
        ("CBCT", stats_cbct, ruta_cbct_salida),
    ):
        fila: dict[str, object] = {
            "patient_id": patient_id,
            "modalidad": modalidad,
            "estado": "OK",
            "size_x_original": size_original[0],
            "size_y_original": size_original[1],
            "size_z_original": size_original[2],
            "spacing_x_original": spacing_original[0],
            "spacing_y_original": spacing_original[1],
            "spacing_z_original": spacing_original[2],
            "extent_x_original_mm": extent_original[0],
            "extent_y_original_mm": extent_original[1],
            "extent_z_original_mm": extent_original[2],
            "target_x": target_size[0],
            "target_y": target_size[1],
            "target_z": target_size[2],
            "spacing_x_final": target_spacing[0],
            "spacing_y_final": target_spacing[1],
            "spacing_z_final": target_spacing[2],
            "origin_x_final": float(ct_final.GetOrigin()[0]),
            "origin_y_final": float(ct_final.GetOrigin()[1]),
            "origin_z_final": float(ct_final.GetOrigin()[2]),
            "padding_value": padding_value,
            "interpolador_cbct": interpolador_cbct,
            "contenido_completo": "SI",
            "ruta_salida": str(ruta_salida),
            "mensaje": "Redimensionado completado sin recorte físico",
        }
        fila.update(stats)
        filas.append(fila)
    return filas


def guardar_informe(output_root: Path, filas: list[dict[str, object]]) -> Path:
    """Informe acumulativo: conserva los pacientes no procesados en esta ejecución."""

    output_root.mkdir(parents=True, exist_ok=True)
    return guardar_informe_acumulado(
        output_root / "informe_redimensionado.csv",
        filas,
        CAMPOS_INFORME,
    )


def _guardar_configuracion(
    output_root: Path,
    pacientes: list[str],
    target_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
    fov_final: tuple[float, float, float],
    metodo_spacing: str,
    margen_fov_mm: float,
    redondeo_spacing_mm: float,
    padding_value: float,
    interpolador_cbct: str,
) -> Path:
    ruta = output_root / "config_redimensionado.json"
    contenido = {
        "version": VERSION_REDIMENSIONADO,
        "pacientes": pacientes,
        "target_size": list(target_size),
        "target_spacing_mm": list(target_spacing),
        "fov_final_centro_a_centro_mm": list(fov_final),
        "metodo_spacing": metodo_spacing,
        "margen_fov_mm": margen_fov_mm,
        "redondeo_spacing_mm": redondeo_spacing_mm,
        "padding_value": padding_value,
        "interpolador_ct": "lineal",
        "interpolador_cbct": interpolador_cbct,
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
            "Paso 4: tamaño fijo y spacing común para toda la cohorte, sin "
            "recorte centrado de anatomía."
        )
    )
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help=(
            "Filtro opcional. Sin él se detectan todos en PACIENTES_NORMALIZED. "
            "Con un subconjunto el spacing se calcularía solo con ese subconjunto: "
            "úselo junto con --target-spacing."
        ),
    )
    parser.add_argument("--target-x", type=int, default=TARGET_X_DEFAULT)
    parser.add_argument("--target-y", type=int, default=TARGET_Y_DEFAULT)
    parser.add_argument("--target-z", type=int, default=TARGET_Z_DEFAULT)
    parser.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        help=(
            "Spacing común manual en mm. Si se omite, se calcula con el mayor "
            "campo físico de la cohorte para no recortar ningún paciente."
        ),
    )
    parser.add_argument(
        "--margen-fov-mm",
        type=float,
        default=MARGEN_FOV_MM_DEFAULT,
        help="Margen físico añadido a cada lado al calcular el spacing común.",
    )
    parser.add_argument(
        "--redondeo-spacing-mm",
        type=float,
        default=REDONDEO_SPACING_MM_DEFAULT,
        help="Redondeo hacia arriba del spacing automático para reproducibilidad.",
    )
    parser.add_argument(
        "--padding-value",
        type=float,
        default=PADDING_VALUE_DEFAULT,
    )
    parser.add_argument(
        "--interpolador-cbct",
        choices=("lineal", "vecino"),
        default="lineal",
        help="Lineal es el recomendado para imágenes de intensidad.",
    )
    parser.add_argument("--sobrescribir", action="store_true")
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    target_size = (int(args.target_x), int(args.target_y), int(args.target_z))

    if min(target_size) < 2:
        print("ERROR: las dimensiones objetivo deben ser >= 2")
        return 2
    if not np.isfinite(args.padding_value) or not 0.0 <= args.padding_value <= 1.0:
        print("ERROR: --padding-value debe ser finito y estar en [0,1]")
        return 2
    if not np.isfinite(args.margen_fov_mm) or args.margen_fov_mm < 0:
        print("ERROR: --margen-fov-mm debe ser finito y no negativo")
        return 2
    if not np.isfinite(args.redondeo_spacing_mm) or args.redondeo_spacing_mm < 0:
        print("ERROR: --redondeo-spacing-mm debe ser finito y no negativo")
        return 2

    rutas = obtener_rutas(args.project_dir)
    try:
        pacientes = detectar_pacientes(rutas.normalized, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if not pacientes:
        print(f"ERROR: no hay pacientes con formato XXXX en {rutas.normalized}")
        return 2

    if args.pacientes and args.target_spacing is None:
        print(
            "AVISO: usa --pacientes sin --target-spacing. El spacing común se "
            "calculará SOLO con esos pacientes y puede no coincidir con el de los "
            "demás. Para un dataset coherente ejecute todos los pacientes."
        )

    imagenes_ct: list[sitk.Image] = []
    direccion_referencia: np.ndarray | None = None
    try:
        for patient_id in pacientes:
            carpeta = rutas.normalized / patient_id
            ruta_ct = carpeta / f"{patient_id}_ct_norm.nii.gz"
            ruta_cbct = carpeta / f"{patient_id}_cbct_norm.nii.gz"
            if not ruta_ct.is_file() or not ruta_cbct.is_file():
                raise FileNotFoundError(
                    f"Paciente {patient_id}: faltan entradas normalizadas"
                )
            ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
            cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
            _validar_normalizada(ct, f"CT {patient_id}")
            _validar_normalizada(cbct, f"CBCT {patient_id}")
            if not geometrias_coinciden(ct, cbct):
                raise ValueError(
                    f"Paciente {patient_id}: CT/CBCT no comparten geometría: "
                    + "; ".join(diferencias_geometria(ct, cbct))
                )

            direccion = np.asarray(ct.GetDirection(), dtype=float)
            if direccion_referencia is None:
                direccion_referencia = direccion
            elif not np.allclose(
                direccion,
                direccion_referencia,
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(
                    f"Paciente {patient_id}: dirección distinta del resto. "
                    "Todos los volúmenes deben estar reorientados de forma homogénea."
                )
            imagenes_ct.append(ct)
    except Exception as exc:
        print(f"ERROR al revisar la cohorte: {type(exc).__name__}: {exc}")
        return 2

    if args.target_spacing is not None:
        target_spacing = tuple(float(v) for v in args.target_spacing)
        if not np.isfinite(target_spacing).all() or min(target_spacing) <= 0:
            print("ERROR: --target-spacing debe contener tres valores positivos")
            return 2
        fov_final = tuple(
            (target_size[d] - 1) * target_spacing[d] for d in range(3)
        )
        metodo_spacing = "manual"
    else:
        try:
            target_spacing, fov_final = calcular_spacing_comun(
                imagenes_ct,
                target_size=target_size,
                margen_fov_mm=float(args.margen_fov_mm),
                redondeo_spacing_mm=float(args.redondeo_spacing_mm),
            )
        except Exception as exc:
            print(f"ERROR al calcular spacing común: {type(exc).__name__}: {exc}")
            return 2
        metodo_spacing = "automatico_maximo_fov_cohorte"

    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")
    print(f"Tamaño fijo: {target_size[0]} x {target_size[1]} x {target_size[2]}")
    print(
        "Spacing común final (mm): "
        + " x ".join(f"{v:.5g}" for v in target_spacing)
    )
    print(
        "FOV final centro-a-centro (mm): "
        + " x ".join(f"{v:.3f}" for v in fov_final)
    )
    print(f"Método de spacing: {metodo_spacing}")

    filas: list[dict[str, object]] = []
    for patient_id in pacientes:
        filas.extend(
            redimensionar_paciente(
                patient_id=patient_id,
                normalized_root=rutas.normalized,
                output_root=rutas.final,
                target_size=target_size,
                target_spacing=target_spacing,
                padding_value=float(args.padding_value),
                interpolador_cbct=args.interpolador_cbct,
                sobrescribir=args.sobrescribir,
            )
        )

    informe = guardar_informe(rutas.final, filas)
    config = _guardar_configuracion(
        rutas.final,
        pacientes,
        target_size,
        target_spacing,
        fov_final,
        metodo_spacing,
        float(args.margen_fov_mm),
        float(args.redondeo_spacing_mm),
        float(args.padding_value),
        args.interpolador_cbct,
    )

    errores = [fila for fila in filas if fila.get("estado") == "ERROR"]
    print(f"\nPaso 4 terminado. Dataset final: {rutas.final}")
    print(f"Informe: {informe}")
    print(f"Configuración reproducible: {config}")
    if errores:
        print("Pacientes con error:")
        for fila in errores:
            print(f"  - {fila['patient_id']}: {fila['mensaje']}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
