"""Herramienta de diagnóstico: qué hay realmente en la cabecera de un NIfTI.

No forma parte del pipeline. Sirve para comprobar, tras el ajuste manual en
SPM, si el ajuste quedó guardado en la cabecera (sform) y cuánto difiere del
qform original. Es la herramienta adecuada para los archivos de las carpetas
"XXXX nuevo", porque lee la cabecera a mano (no necesita SimpleITK).

Uso:
    python inspeccionar_nifti.py archivo1.nii [archivo2.nii.gz ...]

Con exactamente dos archivos, además compara sus posiciones entre sí.
"""
import sys
from pathlib import Path

import numpy as np

from nifti_cabecera import (
    afin_qform_ras,
    afin_sform_ras,
    diferencia_afines,
    leer_cabecera_nifti,
)

CODIGOS = {0: "desconocido", 1: "scanner", 2: "aligned", 3: "talairach", 4: "mni"}


def fmt(a):
    return "\n".join("      " + "  ".join(f"{v:10.4f}" for v in fila) for fila in a)


def main(rutas):
    if not rutas:
        sys.exit("Uso: python inspeccionar_nifti.py archivo.nii [otro.nii.gz ...]")
    for r in rutas:
        ruta = Path(r)
        print("=" * 78)
        print(ruta)
        h = leer_cabecera_nifti(ruta)
        print(f"  tamaño (i,j,k): {h['dim'][1:4]}   pixdim: {tuple(round(v, 4) for v in h['pixdim'][1:4])}")
        print(f"  qform_code = {h['qform_code']} ({CODIGOS.get(h['qform_code'], '?')})   "
              f"sform_code = {h['sform_code']} ({CODIGOS.get(h['sform_code'], '?')})")
        q = afin_qform_ras(h) if h["qform_code"] > 0 else None
        s = afin_sform_ras(h) if h["sform_code"] > 0 else None
        if q is not None:
            print("  QFORM (lo que suele leer SimpleITK/ITK), coordenadas RAS:")
            print(fmt(q))
        if s is not None:
            print("  SFORM (lo que usa SPM), coordenadas RAS:")
            print(fmt(s))
        if q is not None and s is not None:
            dt, ang = diferencia_afines(q, s)
            print(f"\n  Diferencia sform - qform: traslación {dt:.2f} mm | rotación {ang:.2f}°")
            if dt > 0.01 or ang > 0.01:
                print("  >> El sform DIFIERE del qform: SPM guardó aquí un ajuste.")
                print("     SimpleITK puede estar leyendo el qform y 'no ver' ese ajuste.")
            else:
                print("  >> sform y qform son iguales: en este archivo no hay ajuste de SPM")
                print("     guardado en la cabecera (o se guardó en otro archivo).")
    if len(rutas) == 2:
        h1, h2 = (leer_cabecera_nifti(r) for r in rutas)
        a = afin_sform_ras(h1) if h1["sform_code"] > 0 else afin_qform_ras(h1)
        b = afin_sform_ras(h2) if h2["sform_code"] > 0 else afin_qform_ras(h2)
        dt, ang = diferencia_afines(a, b)
        print("=" * 78)
        print("COMPARACIÓN (criterio SPM: sform si existe) entre el 1.º y el 2.º archivo:")
        print(f"  traslación {dt:.2f} mm | rotación {ang:.2f}°")


if __name__ == "__main__":
    main(sys.argv[1:])
