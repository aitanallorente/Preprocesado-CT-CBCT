"""Validación final de los pares CT/CBCT preparados para el modelo.

Último eslabón del pipeline: lee ``PACIENTES_FINAL`` (salida del paso 4) y
comprueba por paciente:
* existencia de CT y CBCT finales;
* tamaño fijo;
* geometría idéntica dentro del par;
* spacing y dirección comunes en toda la cohorte;
* intensidades finitas y dentro de [0,1].

Genera ``PACIENTES_FINAL/informe_validacion_final.csv`` (ACUMULATIVO: una
ejecución sustituye solo las filas de los pacientes validados y conserva las
del resto). Para validar la cohorte completa, ejecútelo sin ``--pacientes``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from imagen_utils import (
    diferencias_geometria,
    geometrias_coinciden,
    validar_geometria_3d,
)
from informes import guardar_informe_acumulado

CAMPOS_INFORME = [
    "patient_id",
    "modalidad",
    "estado",
    "size",
    "spacing",
    "origin",
    "direction",
    "min",
    "max",
    "media",
    "std",
    "porcentaje_valor_cero",
    "mensaje",
]


def estadisticas(imagen: sitk.Image, nombre: str) -> dict[str, float]:
    validar_geometria_3d(imagen, nombre)
    array = sitk.GetArrayViewFromImage(imagen)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{nombre}: volumen vacío o con NaN/infinitos")
    minimo = float(array.min())
    maximo = float(array.max())
    if minimo < -0.00001 or maximo > 1.00001:
        raise ValueError(f"{nombre}: rango [{minimo}, {maximo}] fuera de [0,1]")
    return {
        "min": minimo,
        "max": maximo,
        "media": float(array.mean()),
        "std": float(array.std()),
        "porcentaje_valor_cero": 100.0
        * float(np.count_nonzero(np.isclose(array, 0.0, rtol=0.0, atol=1e-6)))
        / float(array.size),
    }


def guardar_informe(root: Path, filas: list[dict[str, object]]) -> Path:
    return guardar_informe_acumulado(
        root / "informe_validacion_final.csv",
        filas,
        CAMPOS_INFORME,
    )


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida PACIENTES_FINAL")
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional. Sin él se detectan todos en PACIENTES_FINAL.",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    rutas = obtener_rutas(args.project_dir)

    try:
        pacientes = detectar_pacientes(rutas.final, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if not pacientes:
        print(f"ERROR: no hay pacientes en {rutas.final}")
        return 2

    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")

    filas: list[dict[str, object]] = []
    errores: list[str] = []
    size_ref: tuple[int, int, int] | None = None
    spacing_ref: np.ndarray | None = None
    direction_ref: np.ndarray | None = None

    for patient_id in pacientes:
        carpeta = rutas.final / patient_id
        ruta_ct = carpeta / f"{patient_id}_ct_final.nii.gz"
        ruta_cbct = carpeta / f"{patient_id}_cbct_final.nii.gz"
        try:
            if not ruta_ct.is_file() or not ruta_cbct.is_file():
                raise FileNotFoundError("faltan CT y/o CBCT final")

            ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
            cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
            if not geometrias_coinciden(ct, cbct):
                raise ValueError(
                    "el par no comparte geometría: "
                    + "; ".join(diferencias_geometria(ct, cbct))
                )

            size = tuple(int(v) for v in ct.GetSize())
            spacing = np.asarray(ct.GetSpacing(), dtype=float)
            direction = np.asarray(ct.GetDirection(), dtype=float)

            if size_ref is None:
                size_ref = size
                spacing_ref = spacing
                direction_ref = direction
            else:
                if size != size_ref:
                    raise ValueError(f"size {size} distinto del común {size_ref}")
                if not np.allclose(spacing, spacing_ref, rtol=0.0, atol=1e-6):
                    raise ValueError(
                        f"spacing {tuple(spacing)} distinto del común {tuple(spacing_ref)}"
                    )
                if not np.allclose(direction, direction_ref, rtol=0.0, atol=1e-5):
                    raise ValueError("dirección distinta de la cohorte")

            stats_ct = estadisticas(ct, f"CT {patient_id}")
            stats_cbct = estadisticas(cbct, f"CBCT {patient_id}")

            for modalidad, imagen, stats in (
                ("CT", ct, stats_ct),
                ("CBCT", cbct, stats_cbct),
            ):
                fila: dict[str, object] = {
                    "patient_id": patient_id,
                    "modalidad": modalidad,
                    "estado": "OK",
                    "size": "x".join(str(v) for v in imagen.GetSize()),
                    "spacing": ";".join(f"{float(v):.8g}" for v in imagen.GetSpacing()),
                    "origin": ";".join(f"{float(v):.8g}" for v in imagen.GetOrigin()),
                    "direction": ";".join(f"{float(v):.8g}" for v in imagen.GetDirection()),
                    "mensaje": "Validación final correcta",
                }
                fila.update(stats)
                filas.append(fila)
            print(f"  ✓ {patient_id}: par final válido")

        except Exception as exc:
            mensaje = f"{type(exc).__name__}: {exc}"
            errores.append(f"{patient_id}: {mensaje}")
            filas.append(
                {
                    "patient_id": patient_id,
                    "modalidad": "PAR",
                    "estado": "ERROR",
                    "mensaje": mensaje,
                }
            )
            print(f"  ✗ {patient_id}: {mensaje}")

    informe = guardar_informe(rutas.final, filas)
    print(f"\nInforme: {informe}")
    if not errores:
        print(
            "VALIDACIÓN FINAL COMPLETADA: todos los pares comparten tamaño y "
            "spacing, mantienen geometría CT/CBCT y están en [0,1]."
        )
        if size_ref is not None and spacing_ref is not None:
            print(f"Tamaño común: {size_ref}")
            print(f"Spacing común: {tuple(float(v) for v in spacing_ref)}")
    else:
        print("Hay errores que deben corregirse antes de usar el dataset:")
        for error in errores:
            print(f"  - {error}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
