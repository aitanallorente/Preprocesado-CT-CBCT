"""Figura del Apéndice C: falso mínimo del registro automático frente al registro final.

Qué hace
--------
1. Ejecuta el registro AUTOMÁTICO de ``paso2_registro.py`` (búsqueda gruesa + MI)
   para un paciente, pero guardando la salida en una carpeta aparte
   (``PRUEBAS_APENDICE_C``). NO toca ``PACIENTES_REGISTERED`` ni
   ``PACIENTES_REGISTERED_FINAL``.
2. Superpone sobre el CT COMPLETO (``PACIENTES_NIFTI``):
     (A) el CBCT del registro automático;
     (B) el CBCT del registro final (``PACIENTES_REGISTERED_FINAL``), si existe.
3. Guarda dos PNG en ``figuras_apendice_c``:
     XXXX_automatico.png   -> img/apC_XXXX_automatico.png en la memoria
     XXXX_final.png        -> img/apC_XXXX_final.png en la memoria
   Se usa el MISMO corte en las dos imágenes para poder compararlas.

Uso (desde la carpeta del código)
---------------------------------
    python figura_apendice_c.py --paciente 0002
    python figura_apendice_c.py --paciente 0002 --plano sagital
    python figura_apendice_c.py --paciente 0002 --corte 260 --sin-registrar

``--sin-registrar`` reutiliza un registro automático ya hecho en
``PRUEBAS_APENDICE_C`` (para cambiar solo el corte o el plano sin repetirlo).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import obtener_rutas
from paso2_registro import ortonormalizar_direccion, registrar_paciente, reorientar_lps

PADDING_HU = -1024.0


def leer_ct_completo(ruta: Path) -> sitk.Image:
    """CT completo con la misma preparación que el registro (dirección + LPS)."""

    ct = sitk.ReadImage(str(ruta), sitk.sitkFloat32)
    ct, _ = ortonormalizar_direccion(ct)
    return reorientar_lps(ct)


def cbct_en_rejilla_ct(ruta_cbct: Path, ct: sitk.Image) -> np.ndarray:
    """Lleva el CBCT registrado (ya en coordenadas del CT) a la rejilla del CT completo.

    Devuelve un array (z, y, x) con NaN fuera del CBCT, para no pintar el relleno.
    """

    cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
    # Identidad: el CBCT ya está en el espacio físico del CT; solo cambia la rejilla.
    remuestreado = sitk.Resample(
        cbct, ct, sitk.Euler3DTransform(), sitk.sitkLinear, PADDING_HU, sitk.sitkFloat32
    )
    arr = sitk.GetArrayFromImage(remuestreado)
    relleno = np.isclose(arr, PADDING_HU, rtol=0.0, atol=1e-3)
    return np.where(relleno, np.nan, arr)


def ventana(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def corte_por_defecto(cbct_arr: np.ndarray, plano: str) -> int:
    """Corte que pasa por el centro de la zona con CBCT."""

    zz, yy, xx = np.nonzero(np.isfinite(cbct_arr))
    if zz.size == 0:
        raise RuntimeError("El CBCT no cae dentro del CT: no hay nada que dibujar")
    eje = yy if plano == "coronal" else xx
    return int(round((eje.min() + eje.max()) / 2))


def extraer(arr: np.ndarray, plano: str, corte: int) -> np.ndarray:
    """Corte 2D con superior arriba. En LPS, el índice z crece hacia superior."""

    if plano == "coronal":
        return np.flipud(arr[:, corte, :])
    return np.flipud(arr[:, :, corte])


def guardar_superposicion(
    ct_arr: np.ndarray,
    cbct_arr: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    plano: str,
    corte: int,
    titulo: str,
    destino: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sx, sy, sz = spacing_xyz
    aspecto = sz / sx if plano == "coronal" else sz / sy

    ct2d = ventana(extraer(ct_arr, plano, corte), -400.0, 600.0)
    cb2d = extraer(cbct_arr, plano, corte)
    valido = np.isfinite(cb2d)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(ct2d, cmap="gray", aspect=aspecto, vmin=0, vmax=1)
    if valido.any():
        p1, p99 = np.nanpercentile(cb2d, [1, 99])
        cb_n = ventana(np.nan_to_num(cb2d, nan=p1), float(p1), float(p99))
        verde = np.zeros(cb2d.shape + (4,), dtype=float)
        verde[..., 1] = cb_n                     # canal verde = intensidad del CBCT
        verde[..., 3] = np.where(valido, 0.55, 0.0)  # transparente fuera del CBCT
        ax.imshow(verde, aspect=aspecto)
        ax.contour(valido.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    # Sin título dentro de la imagen: en la memoria lo dan los subfloat (A)/(B).
    print(f"  {titulo}: {int(valido.sum())} píxeles con CBCT en el corte")
    ax.axis("off")
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {destino}")


def main() -> int:
    p = argparse.ArgumentParser(description="Figura del Apéndice C (falso mínimo del registro)")
    p.add_argument("--paciente", default="0002")
    p.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    p.add_argument("--plano", choices=("coronal", "sagital"), default="coronal")
    p.add_argument("--corte", type=int, help="Índice del corte (y si coronal, x si sagital)")
    p.add_argument("--sin-registrar", action="store_true",
                   help="Reutiliza el registro automático ya guardado en PRUEBAS_APENDICE_C")
    args = p.parse_args()

    pid = args.paciente
    rutas = obtener_rutas(args.project_dir)
    pruebas = rutas.proyecto / "PRUEBAS_APENDICE_C"
    salida_fig = rutas.proyecto / "figuras_apendice_c"
    ruta_ct = rutas.nifti / pid / f"{pid}_ct.nii.gz"
    ruta_auto = pruebas / pid / f"{pid}_cbct_registered.nii.gz"
    ruta_final = rutas.proyecto / "PACIENTES_REGISTERED_FINAL" / pid / f"{pid}_cbct_registered.nii.gz"

    if not ruta_ct.is_file():
        print(f"ERROR: no existe {ruta_ct}")
        return 2

    # 1) Registro automático en una carpeta aparte (mismos parámetros que el TFG).
    if not args.sin_registrar:
        res = registrar_paciente(
            patient_id=pid,
            nifti_root=rutas.nifti,
            output_root=pruebas,
            tipo_transformacion="rigida",
            inicializacion="robusta",
            interpolador_salida="lineal",
            sampling=0.15,
            seed=42,
            iteraciones=250,
            reset_cbct_origin=False,
            margen_roi_mm=5.0,
            sobrescribir=True,
        )
        if res["estado"] == "ERROR":
            print(f"ERROR en el registro automático: {res['mensaje']}")
            return 1
    if not ruta_auto.is_file():
        print(f"ERROR: no existe {ruta_auto}. Ejecuta sin --sin-registrar.")
        return 2

    # 2) CT completo y CBCT(s) sobre su rejilla.
    ct = leer_ct_completo(ruta_ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    spacing = tuple(float(v) for v in ct.GetSpacing())
    auto_arr = cbct_en_rejilla_ct(ruta_auto, ct)
    final_arr = cbct_en_rejilla_ct(ruta_final, ct) if ruta_final.is_file() else None
    if final_arr is None:
        print(f"  (aviso) no existe {ruta_final}: solo se genera la figura del automático")

    # Mismo corte en las dos figuras: por defecto, el centro del CBCT final
    # (en la vista coronal/sagital el error craneocaudal se ve como un salto vertical).
    referencia = final_arr if final_arr is not None else auto_arr
    corte = args.corte if args.corte is not None else corte_por_defecto(referencia, args.plano)
    limite = ct_arr.shape[1] if args.plano == "coronal" else ct_arr.shape[2]
    if not 0 <= corte < limite:
        print(f"ERROR: corte {corte} fuera de rango (0..{limite - 1})")
        return 2
    print(f"  Plano {args.plano}, corte {corte}")

    guardar_superposicion(ct_arr, auto_arr, spacing, args.plano, corte,
                          f"{pid}: registro automático",
                          salida_fig / f"{pid}_automatico.png")
    if final_arr is not None:
        guardar_superposicion(ct_arr, final_arr, spacing, args.plano, corte,
                              f"{pid}: registro final",
                              salida_fig / f"{pid}_final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
