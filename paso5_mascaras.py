"""Paso 5: máscara de vóxeles válidos del CBCT en la rejilla final.

Por qué hace falta
------------------
En el dataset final (PACIENTES_FINAL) el aire, el fondo del CBCT (-1200 HU), el
relleno del registro (-1024 HU) y el relleno de la rejilla valen todos 0, así que
ya no se pueden separar. Antes de normalizar sí se pueden: en
PACIENTES_REGISTERED_FINAL el relleno del registro vale exactamente -1024 HU y el
fondo fuera del campo de reconstrucción vale -1200 HU (mismo criterio que el
paso 3 para calcular los "vóxeles válidos").

Qué hace, por paciente
----------------------
1. Lee PACIENTES_REGISTERED_FINAL/XXXX/XXXX_cbct_registered.nii.gz (en HU).
2. Marca como válido (1) todo vóxel que NO sea relleno (-1024 exacto) y cuyo
   valor sea mayor que -1200 HU (es decir, que no sea fondo).
3. Opcionalmente erosiona la máscara 1 vóxel para quitar el borde de vóxeles
   interpolados entre datos y relleno/fondo.
4. La remuestrea a la rejilla final con vecino más próximo, usando como
   referencia PACIENTES_FINAL/XXXX/XXXX_cbct_final.nii.gz (el paso 3 no cambia
   la geometría, y el paso 4 usó la transformación identidad, así que la
   correspondencia es exacta). Fuera del volumen se rellena con 0.
5. Guarda PACIENTES_FINAL/XXXX/XXXX_mask_final.nii.gz (uint8, 0/1) y comprueba
   que comparte geometría con el CT y el CBCT finales.

Escribe PACIENTES_FINAL/informe_mascaras.csv (acumulativo) con el porcentaje de
vóxeles válidos y dos controles de coherencia. Con --figura XXXX guarda además
una figura axial y coronal con la máscara superpuesta.

IMPORTANTE: si se vuelve a ejecutar el paso 4 (por ejemplo, al rehacer un
paciente), hay que volver a ejecutar este paso con --sobrescribir.

Uso:
    python paso5_mascaras.py
    python paso5_mascaras.py --figura 0006
    python paso5_mascaras.py --pacientes 0004 --sobrescribir
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from escritura_segura import escribir_grupo_atomico
from imagen_utils import diferencias_geometria, geometrias_coinciden
from informes import guardar_informe_acumulado

PADDING_HU = -1024.0
TOLERANCIA_PADDING = 1e-3     # igual que pasos 2c y 3: solo el -1024 exacto es relleno
FONDO_HU = -1200.0
MARGEN_FONDO_HU = 0.5         # válido si valor > -1199,5 HU
UMBRAL_CERO = 1e-4            # en [0, 1]; por debajo se considera "cero"

# La consola de Windows (cp1252) puede fallar al imprimir ciertos caracteres.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(texto: str) -> None:
    print(texto, flush=True)

CAMPOS_INFORME = [
    "patient_id",
    "estado",
    "erosion_vox",
    "pct_validos_registrado",
    "pct_validos_final",
    "pct_validos_con_cbct_cero",
    "pct_no_validos_con_cbct_no_cero",
    "ruta_salida",
    "mensaje",
]


def mascara_en_registrado(cbct_hu: sitk.Image, erosion_vox: int) -> sitk.Image:
    """Máscara 0/1 de vóxeles con información en la rejilla del par registrado."""

    arr = sitk.GetArrayFromImage(cbct_hu)
    relleno = np.isclose(arr, np.float32(PADDING_HU), rtol=0.0, atol=TOLERANCIA_PADDING)
    fondo = arr <= FONDO_HU + MARGEN_FONDO_HU
    valido = (~relleno) & (~fondo)

    mascara = sitk.GetImageFromArray(valido.astype(np.uint8))
    mascara.CopyInformation(cbct_hu)
    if erosion_vox > 0:
        mascara = sitk.BinaryErode(mascara, [int(erosion_vox)] * 3, sitk.sitkBall, 0, 1)
    return mascara


def a_rejilla_final(mascara: sitk.Image, referencia: sitk.Image) -> sitk.Image:
    """Lleva la máscara a la rejilla final con vecino más próximo (fuera = 0)."""

    return sitk.Resample(
        mascara,
        referencia,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def guardar_figura(pid: str, cbct_final: sitk.Image, mascara: sitk.Image, destino: Path) -> Path:
    """Cortes axial y coronal centrales del CBCT final con la máscara en verde."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = sitk.GetArrayFromImage(cbct_final)          # (z, y, x)
    m = sitk.GetArrayFromImage(mascara).astype(bool)
    sx, sy, sz = cbct_final.GetSpacing()
    zz, yy, _ = np.nonzero(m) if m.any() else (np.array([c.shape[0] // 2]), np.array([c.shape[1] // 2]), None)
    cz, cy = int(np.median(zz)), int(np.median(yy))

    # Ventana aproximada de tejidos blandos en la escala normalizada:
    # x = (HU + 1024) / 4095  ->  [-200, 300] HU
    lo, hi = (-200 + 1024) / 4095, (300 + 1024) / 4095

    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.5))
    for eje, (titulo, img, msk, aspecto) in zip(
        ejes,
        (
            (f"Axial (z={cz})", c[cz], m[cz], sy / sx),
            (f"Coronal (y={cy})", np.flipud(c[:, cy, :]), np.flipud(m[:, cy, :]), sz / sx),
        ),
    ):
        eje.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect=aspecto)
        capa = np.zeros(img.shape + (4,))
        capa[msk] = (0.11, 0.69, 0.48, 0.35)       # verde semitransparente
        eje.imshow(capa, aspect=aspecto)
        eje.contour(msk.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=1.0)
        eje.set_title(titulo)
        eje.axis("off")
    fig.suptitle(f"Paciente {pid}: máscara de vóxeles válidos del CBCT (verde)")
    fig.tight_layout()
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{pid}_mascara_final.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def intentar_figura(pid: str, ruta_cbct_final: Path, ruta_mascara: Path, destino: Path) -> str:
    """Dibuja la figura leyendo los archivos ya guardados. Nunca detiene el proceso."""

    try:
        ruta_fig = guardar_figura(
            pid,
            sitk.ReadImage(str(ruta_cbct_final), sitk.sitkFloat32),
            sitk.ReadImage(str(ruta_mascara), sitk.sitkUInt8),
            destino,
        )
        log(f"  [OK] Figura guardada en: {ruta_fig.resolve()}")
        return ""
    except Exception as exc:
        log(f"  [ERROR] No se pudo crear la figura: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return f"figura no creada ({type(exc).__name__}: {exc})"


def procesar_paciente(
    pid: str,
    registered_final: Path,
    final_root: Path,
    erosion_vox: int,
    sobrescribir: bool,
    figura: bool,
) -> dict[str, object]:
    log(f"\n{'=' * 72}\nPaciente {pid}: máscara de vóxeles válidos\n{'=' * 72}")
    fila: dict[str, object] = {c: "" for c in CAMPOS_INFORME}
    fila.update(patient_id=pid, estado="ERROR", erosion_vox=erosion_vox)

    ruta_cbct_hu = registered_final / pid / f"{pid}_cbct_registered.nii.gz"
    ruta_ct_final = final_root / pid / f"{pid}_ct_final.nii.gz"
    ruta_cbct_final = final_root / pid / f"{pid}_cbct_final.nii.gz"
    ruta_salida = final_root / pid / f"{pid}_mask_final.nii.gz"
    fila["ruta_salida"] = str(ruta_salida)

    faltan = [r for r in (ruta_cbct_hu, ruta_ct_final, ruta_cbct_final) if not r.is_file()]
    if faltan:
        fila["mensaje"] = "Faltan entradas: " + ", ".join(str(r) for r in faltan)
        log(f"  [ERROR] {fila['mensaje']}")
        return fila
    if ruta_salida.exists() and not sobrescribir:
        fila.update(estado="OMITIDO", mensaje="La máscara ya existe; usa --sobrescribir")
        log("  [--] La máscara ya existe; no se regenera.")
        if figura:
            error_fig = intentar_figura(pid, ruta_cbct_final, ruta_salida, final_root / "figuras_mascaras")
            if error_fig:
                fila["mensaje"] += "; " + error_fig
        return fila

    try:
        cbct_hu = sitk.ReadImage(str(ruta_cbct_hu), sitk.sitkFloat32)
        ct_final = sitk.ReadImage(str(ruta_ct_final), sitk.sitkFloat32)
        cbct_final = sitk.ReadImage(str(ruta_cbct_final), sitk.sitkFloat32)

        m_reg = mascara_en_registrado(cbct_hu, erosion_vox)
        fila["pct_validos_registrado"] = round(
            100.0 * float(np.mean(sitk.GetArrayViewFromImage(m_reg))), 2
        )

        m_fin = a_rejilla_final(m_reg, cbct_final)
        if not (geometrias_coinciden(m_fin, cbct_final) and geometrias_coinciden(m_fin, ct_final)):
            raise RuntimeError(
                "La máscara no comparte geometría con el par final: "
                + "; ".join(diferencias_geometria(m_fin, cbct_final))
            )

        m = sitk.GetArrayFromImage(m_fin).astype(bool)
        c = sitk.GetArrayFromImage(cbct_final)
        if not m.any():
            raise RuntimeError("La máscara final está vacía")
        cero = c < UMBRAL_CERO
        fila["pct_validos_final"] = round(100.0 * float(m.mean()), 2)
        # Controles: ambos deberían ser pequeños.
        fila["pct_validos_con_cbct_cero"] = round(100.0 * float((m & cero).sum()) / m.sum(), 2)
        fila["pct_no_validos_con_cbct_no_cero"] = round(
            100.0 * float((~m & ~cero).sum()) / max(int((~m).sum()), 1), 2
        )

        escribir_grupo_atomico(
            [(ruta_salida, lambda t: sitk.WriteImage(m_fin, str(t), True))]
        )
        log(
            f"  [OK] Vóxeles válidos: {fila['pct_validos_final']} % del volumen final "
            f"(controles: {fila['pct_validos_con_cbct_cero']} % / "
            f"{fila['pct_no_validos_con_cbct_no_cero']} %)"
        )

        fila.update(estado="OK", mensaje="Máscara generada")
    except Exception as exc:
        fila.update(estado="ERROR", mensaje=f"{type(exc).__name__}: {exc}")
        log(f"  [ERROR] {fila['mensaje']}")
        log(traceback.format_exc())
        return fila

    # La figura va aparte: si falla, la máscara ya está guardada y se avisa.
    if figura:
        error_fig = intentar_figura(pid, ruta_cbct_final, ruta_salida, final_root / "figuras_mascaras")
        if error_fig:
            fila["mensaje"] += "; " + error_fig
    return fila


def main() -> int:
    p = argparse.ArgumentParser(description="Paso 5: máscara de vóxeles válidos del CBCT")
    p.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    p.add_argument("--pacientes", nargs="+", help="IDs a procesar (sin él: todos los de PACIENTES_FINAL)")
    p.add_argument("--erosion-vox", type=int, default=1, help="Erosión en vóxeles (0 = sin erosión; default: 1)")
    p.add_argument("--figura", nargs="*", default=[], metavar="ID", help="Pacientes de los que guardar figura")
    p.add_argument("--sobrescribir", action="store_true")
    args = p.parse_args()

    if args.erosion_vox < 0:
        log("ERROR: --erosion-vox no puede ser negativo")
        return 2

    rutas = obtener_rutas(args.project_dir)
    registered_final = rutas.proyecto / "PACIENTES_REGISTERED_FINAL"
    try:
        pacientes = detectar_pacientes(rutas.final, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        log(f"ERROR: {exc}")
        return 2
    if not pacientes:
        log(f"ERROR: no hay pacientes en {rutas.final}")
        return 2

    log("Paso 5: máscaras de vóxeles válidos del CBCT")
    log(f"Entrada (HU): {registered_final}")
    log(f"Salida      : {rutas.final}")
    log(f"Pacientes   : {texto_pacientes(pacientes)}")
    if args.figura:
        log(f"Figuras de  : {', '.join(args.figura)} -> {rutas.final / 'figuras_mascaras'}")
    filas = [
        procesar_paciente(
            pid, registered_final, rutas.final, args.erosion_vox,
            args.sobrescribir, pid in set(args.figura),
        )
        for pid in pacientes
    ]
    informe = guardar_informe_acumulado(rutas.final / "informe_mascaras.csv", filas, CAMPOS_INFORME)
    errores = [f for f in filas if f["estado"] == "ERROR"]
    log(f"\nPaso 5 terminado. Informe: {informe}")
    for f in errores:
        log(f"  [ERROR] {f['patient_id']}: {f['mensaje']}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
