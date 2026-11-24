"""Herramienta de diagnóstico: geometría de los NIfTI del paso 1.

No forma parte del pipeline. Imprime tamaño, spacing, origen y dirección del CT
y del CBCT de cada paciente en ``PACIENTES_NIFTI``, tal como los lee SimpleITK.

Para archivos ajustados en SPM (carpetas "XXXX nuevo") use
``inspeccionar_nifti.py``: SimpleITK suele leer el qform y no vería el ajuste
que SPM guarda en el sform.
"""

from __future__ import annotations

import argparse

import numpy as np
import SimpleITK as sitk

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from imagen_utils import validar_geometria_3d


def imprimir_imagen(nombre: str, imagen: sitk.Image) -> None:
    validar_geometria_3d(imagen, nombre)
    dimension = imagen.GetDimension()
    direccion = np.asarray(imagen.GetDirection(), dtype=float).reshape(
        (dimension, dimension)
    )
    print(f"    {nombre}")
    print(f"      tamaño     : {imagen.GetSize()}")
    print(f"      spacing    : {imagen.GetSpacing()}")
    print(f"      origen     : {imagen.GetOrigin()}")
    error_ortogonalidad = float(
        np.max(np.abs(direccion.T @ direccion - np.eye(dimension)))
    )
    print(f"      dirección   : {tuple(float(v) for v in imagen.GetDirection())}")
    print(f"      determinante: {np.linalg.det(direccion):.8f}")
    print(f"      error ortog.: {error_ortogonalidad:.3e}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspecciona orientación CT/CBCT")
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional. Sin él se detectan todos en PACIENTES_NIFTI.",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    rutas = obtener_rutas(args.project_dir)

    try:
        pacientes = detectar_pacientes(rutas.nifti, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not pacientes:
        print(f"ERROR: no hay pacientes con formato XXXX en {rutas.nifti}")
        return 2

    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")
    errores = 0

    for patient_id in pacientes:
        carpeta = rutas.nifti / patient_id
        ruta_ct = carpeta / f"{patient_id}_ct.nii.gz"
        ruta_cbct = carpeta / f"{patient_id}_cbct.nii.gz"
        print(f"\n{'=' * 72}\nPaciente {patient_id}\n{'=' * 72}")

        faltantes = [ruta for ruta in (ruta_ct, ruta_cbct) if not ruta.is_file()]
        if faltantes:
            errores += 1
            print("  ✗ Faltan: " + ", ".join(r.name for r in faltantes))
            continue

        try:
            imprimir_imagen("CT", sitk.ReadImage(str(ruta_ct)))
            imprimir_imagen("CBCT", sitk.ReadImage(str(ruta_cbct)))
        except Exception as exc:
            errores += 1
            print(f"  ✗ {type(exc).__name__}: {exc}")

    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
