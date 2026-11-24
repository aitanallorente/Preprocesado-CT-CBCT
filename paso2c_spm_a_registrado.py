"""Paso 2c: del ajuste manual en SPM al par CT/CBCT registrado definitivo.

Contexto
--------
El paso 2 (automático o por landmarks) deja un registro aproximado. Después,
TODOS los pacientes se ajustan a mano en SPM ("Reorient images") y el resultado
se guarda en una carpeta ``XXXX nuevo`` dentro de PACIENTES_REGISTERED.

Ese ajuste NO remuestrea: SPM escribe la transformación en la cabecera del
NIfTI del CBCT (en el sform; el qform queda con la posición original, y
SimpleITK suele leer el qform, por eso aquí la geometría se lee a mano de la
cabecera: ver nifti_cabecera.py). Para que los pasos 3 y 4 puedan leer un par
CT/CBCT en la misma rejilla, hay que remuestrear el CBCT sobre la rejilla del
CT usando esa cabecera. Eso hace este script.

Entrada (por paciente, carpeta "XXXX nuevo" dentro de PACIENTES_REGISTERED):
    XXXX_cbct_registered.nii   CBCT con la transformación de SPM en la cabecera
    XXXX_ct.nii                CT de referencia (si no existe, XXXX_ct_2.nii)

Salida (PACIENTES_REGISTERED_FINAL/XXXX/):
    XXXX_ct.nii.gz
    XXXX_cbct_registered.nii.gz     mismo tamaño y geometría que el CT

Además:
* Mide cuánto corrigió SPM respecto al registro del paso 2
  (PACIENTES_REGISTERED/XXXX/XXXX_cbct_registered.nii.gz): desplazamiento del
  centro, rotación, y comprobación de que el cambio es rígido (sin escala).
  Los umbrales solo marcan REVISAR en el informe; no bloquean nada.
* Escribe informe_spm_a_registrado.csv (acumulativo: no pierde pacientes).
* Con --figuras guarda en ``figuras_registro`` imágenes de fusión y tablero de
  ajedrez para la memoria.

No hay ninguna lista fija de pacientes: se usan todas las carpetas "XXXX nuevo"
que existan, o las indicadas con --pacientes.

Siguiente paso: ``paso3_normalizacion.py --registered-root
PACIENTES_REGISTERED_FINAL``.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import obtener_rutas, texto_pacientes
from escritura_segura import escribir_grupo_atomico
from imagen_utils import geometrias_coinciden, validar_geometria_3d
from informes import guardar_informe_acumulado
from nifti_cabecera import afin_indice_a_fisico, geometria_spm, metricas_correccion

PADDING_HU = -1024.0
TOLERANCIA_PADDING = 1e-3   # igual que el paso 3: solo el -1024 exacto es relleno
PATRON_CARPETA_NUEVO = re.compile(r"^([0-9]{4})\s+nuevo$", re.IGNORECASE)

# Umbrales solo informativos: marcan REVISAR, no bloquean nada.
UMBRAL_SOLAPAMIENTO_PCT = 70.0
UMBRAL_CORRECCION_MM = 40.0
UMBRAL_CORRECCION_GRADOS = 10.0
TOLERANCIA_ESCALA = 1e-3

CAMPOS_INFORME = [
    "patient_id",
    "estado",
    "ct_origen",
    "cbct_origen",
    "ct_igual_que_registro_previo",
    "roi_final_size",
    "solapamiento_caja_cbct_pct",
    "pct_padding_en_roi",
    "corr_dx_mm",
    "corr_dy_mm",
    "corr_dz_mm",
    "corr_traslacion_centro_mm",
    "corr_rotacion_grados",
    "escala_min",
    "escala_max",
    "error_ortogonalidad",
    "mensaje",
]


# ---------------------------------------------------------------------------
# Detección de carpetas y archivos
# ---------------------------------------------------------------------------

def detectar_carpetas_nuevo(
    registered: Path,
    solicitados: list[str] | None,
) -> dict[str, Path]:
    """Devuelve {id: carpeta} para todas las carpetas 'XXXX nuevo' existentes."""

    if not registered.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {registered}")
    encontradas: dict[str, Path] = {}
    for entrada in sorted(registered.iterdir()):
        coincide = PATRON_CARPETA_NUEVO.match(entrada.name)
        if entrada.is_dir() and coincide:
            encontradas[coincide.group(1)] = entrada

    if solicitados is None:
        return encontradas

    pedidos = list(dict.fromkeys(str(p) for p in solicitados))
    faltan = [p for p in pedidos if p not in encontradas]
    if faltan:
        raise FileNotFoundError(
            f"No hay carpeta 'XXXX nuevo' en {registered} para: " + ", ".join(faltan)
        )
    return {p: encontradas[p] for p in sorted(pedidos)}


def _nifti_en(carpeta: Path) -> list[Path]:
    return sorted(
        p
        for p in carpeta.iterdir()
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
    )


def _exacto(archivos: list[Path], nombres: list[str]) -> Path | None:
    objetivo = {n.lower() for n in nombres}
    for p in archivos:
        if p.name.lower() in objetivo:
            return p
    return None


def elegir_archivos(carpeta: Path, pid: str) -> tuple[Path, Path, list[str]]:
    """Elige el CT y el CBCT de SPM. Devuelve (ruta_ct, ruta_cbct, avisos)."""

    archivos = _nifti_en(carpeta)
    avisos: list[str] = []
    listado = ", ".join(p.name for p in archivos) or "(ninguno)"

    cbct = _exacto(archivos, [f"{pid}_cbct_registered.nii", f"{pid}_cbct_registered.nii.gz"])
    if cbct is None:
        cand = [p for p in archivos if "cbct" in p.name.lower()]
        if len(cand) == 1:
            cbct = cand[0]
            avisos.append(f"CBCT elegido por nombre aproximado: {cbct.name}")
        elif not cand:
            raise FileNotFoundError(f"No encuentro el CBCT en {carpeta}. Hay: {listado}")
        else:
            raise ValueError(f"Varios CBCT posibles en {carpeta}: {listado}")

    ct = _exacto(archivos, [f"{pid}_ct.nii", f"{pid}_ct.nii.gz"])
    if ct is None:
        ct = _exacto(archivos, [f"{pid}_ct_2.nii", f"{pid}_ct_2.nii.gz"])
        if ct is not None:
            avisos.append(f"No hay {pid}_ct.nii; se usa {ct.name}")
    if ct is None:
        cand = [
            p for p in archivos
            if re.search(r"(^|[_\-\s])ct([_\-\s.]|$)", p.name.lower()) and "cbct" not in p.name.lower()
        ]
        if len(cand) == 1:
            ct = cand[0]
            avisos.append(f"CT elegido por nombre aproximado: {ct.name}")
        elif not cand:
            raise FileNotFoundError(f"No encuentro el CT en {carpeta}. Hay: {listado}")
        else:
            raise ValueError(f"Varios CT posibles en {carpeta}: {listado}")

    return ct, cbct, avisos


# ---------------------------------------------------------------------------
# Geometría y magnitud de la corrección de SPM (solo numpy)
# ---------------------------------------------------------------------------

def leer_imagen_spm(ruta: Path) -> tuple[sitk.Image, dict]:
    """Lee la imagen y le impone la geometría de la cabecera según SPM.

    SPM guarda su ajuste en el sform y deja el qform original; SimpleITK suele
    leer el qform, así que sin esto el ajuste manual pasaría desapercibido.
    """

    img = sitk.ReadImage(str(ruta), sitk.sitkFloat32)
    g = geometria_spm(ruta)
    tam_img = tuple(int(v) for v in img.GetSize())
    if tam_img != g["size"]:
        raise ValueError(
            f"{ruta.name}: tamaño de la imagen {tam_img} distinto del de la cabecera {g['size']}"
        )
    g["itk_difiere"] = not (
        np.allclose(img.GetOrigin(), g["origin"], atol=1e-3)
        and np.allclose(img.GetDirection(), g["direction"], atol=1e-4)
    )
    img.SetSpacing(g["spacing"])
    img.SetOrigin(g["origin"])
    img.SetDirection(g["direction"])
    return img, g


def _afin(g: dict) -> np.ndarray:
    return afin_indice_a_fisico(g["size"], g["spacing"], g["origin"], g["direction"])


def cabeceras_iguales(a: dict, b: dict) -> bool:
    return (
        a["size"] == b["size"]
        and np.allclose(a["spacing"], b["spacing"], atol=1e-4)
        and np.allclose(a["origin"], b["origin"], atol=1e-2)
        and np.allclose(a["direction"], b["direction"], atol=1e-4)
    )


# ---------------------------------------------------------------------------
# Huella, recorte y figuras
# ---------------------------------------------------------------------------

def crear_mascara_huella(ct: sitk.Image, cbct: sitk.Image) -> sitk.Image:
    """Caja física del CBCT expresada en la rejilla del CT (1 = dentro)."""

    unos = sitk.Image(cbct.GetSize(), sitk.sitkUInt8)
    unos.CopyInformation(cbct)
    unos = unos + 1
    mascara = sitk.Resample(
        unos, ct, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    return sitk.Cast(mascara > 0, sitk.sitkUInt8)


def estadisticas_huella(mascara: sitk.Image, cbct: sitk.Image):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mascara)
    if not stats.HasLabel(1):
        raise RuntimeError("El CBCT de SPM no solapa con el CT: revisa la cabecera")
    n = float(stats.GetNumberOfPixels(1))
    vox_ct = float(np.prod(np.asarray(mascara.GetSpacing(), dtype=float)))
    vol_cbct = float(
        np.prod(np.asarray(cbct.GetSize(), dtype=float))
        * np.prod(np.asarray(cbct.GetSpacing(), dtype=float))
    )
    pct = max(0.0, min(100.0, 100.0 * n * vox_ct / vol_cbct))
    bbox = tuple(int(v) for v in stats.GetBoundingBox(1))
    return pct, bbox


def roi_con_margen(bbox, referencia: sitk.Image, margen_mm: float):
    size_ref = [int(v) for v in referencia.GetSize()]
    spacing = [float(v) for v in referencia.GetSpacing()]
    inicio, tam = [], []
    for eje in range(3):
        margen_vox = int(math.ceil(max(0.0, margen_mm) / spacing[eje]))
        ini = max(0, bbox[eje] - margen_vox)
        fin = min(size_ref[eje], bbox[eje] + bbox[3 + eje] + margen_vox)
        inicio.append(int(ini))
        tam.append(int(fin - ini))
    if any(v < 2 for v in tam):
        raise RuntimeError(f"ROI degenerada: inicio={inicio}, tamaño={tam}")
    return inicio, tam


def huella_toca_borde(bbox, size) -> bool:
    for eje in range(3):
        if bbox[eje] == 0 or bbox[eje] + bbox[3 + eje] >= int(size[eje]):
            return True
    return False


def es_relleno(arr: np.ndarray) -> np.ndarray:
    """Vóxeles de relleno (exactamente -1024). El aire real del CBCT puede estar
    por debajo de -1024 (p. ej. -1200) y NO es relleno."""

    return np.isclose(arr, np.float32(PADDING_HU), rtol=0.0, atol=TOLERANCIA_PADDING)


def _ventana(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _posiciones_figura(
    valido: np.ndarray,
    shape: tuple[int, int, int],
    n_niveles: int,
    cortes: tuple[int | None, int | None, int | None],
) -> list[tuple[int, int, int]]:
    """Lista de (z, y, x) donde dibujar. Orden (z, y, x) como el array.

    Sin cortes explícitos: n_niveles posiciones repartidas dentro de la zona con
    CBCT (n=1 -> mitad; n=3 -> 25 %, 50 %, 75 %...). Con algún corte explícito,
    ese eje queda fijo y los demás usan el punto medio.
    """

    if valido.any():
        zz, yy, xx = np.nonzero(valido)
        lo = (int(zz.min()), int(yy.min()), int(xx.min()))
        hi = (int(zz.max()), int(yy.max()), int(xx.max()))
    else:
        lo, hi = (0, 0, 0), tuple(s - 1 for s in shape)

    for nombre, valor, n in zip(("axial", "coronal", "sagital"), cortes, shape):
        if valor is not None and not 0 <= valor < n:
            raise ValueError(
                f"corte {nombre}={valor} fuera de rango (0..{n - 1}) en la región final"
            )

    if any(c is not None for c in cortes):
        fracciones = [0.5]
    else:
        fracciones = [(k + 1) / (n_niveles + 1) for k in range(n_niveles)]

    posiciones = []
    for f in fracciones:
        auto = tuple(int(round(lo[i] + f * (hi[i] - lo[i]))) for i in range(3))
        posiciones.append(tuple(auto[i] if cortes[i] is None else int(cortes[i]) for i in range(3)))
    return posiciones


def guardar_figuras(
    pid: str,
    ct_arr: np.ndarray,
    cbct_arr: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    destino: Path,
    n_niveles: int = 1,
    cortes: tuple[int | None, int | None, int | None] = (None, None, None),
) -> list[Path]:
    """Fusión (CT magenta + CBCT verde) y tablero de ajedrez en 3 planos.

    Los arrays tienen orden (z, y, x), como GetArrayFromImage. Se muestra en
    convención radiológica aproximada (izquierda del paciente a la derecha).
    ``cortes`` = (axial, coronal, sagital) como índices de vóxel de la región
    final (axial = índice z, coronal = índice y, sagital = índice x).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destino.mkdir(parents=True, exist_ok=True)

    valido = ~es_relleno(cbct_arr)
    if valido.any():
        p1, p99 = np.percentile(cbct_arr[valido], [1, 99])
    else:
        p1, p99 = float(cbct_arr.min()), float(cbct_arr.max())

    ct_n = _ventana(ct_arr, -200.0, 300.0)
    cbct_n = _ventana(cbct_arr, float(p1), float(p99))
    sx, sy, sz = spacing_xyz

    posiciones = _posiciones_figura(valido, ct_arr.shape, n_niveles, cortes)
    rutas: list[Path] = []
    for k, (cz, cy, cx) in enumerate(posiciones, start=1):
        planos = [
            ("Axial", f"z={cz}", lambda a, cz=cz: a[cz, :, :], sy / sx),
            ("Coronal", f"y={cy}", lambda a, cy=cy: np.flipud(a[:, cy, :]), sz / sx),
            ("Sagital", f"x={cx}", lambda a, cx=cx: np.flipud(a[:, :, cx]), sz / sy),
        ]
        fig, ejes = plt.subplots(2, 3, figsize=(15, 9))
        for col, (nombre, etiqueta, f, aspecto) in enumerate(planos):
            c = f(ct_n)
            b = f(cbct_n)
            ejes[0, col].imshow(np.stack([c, b, c], axis=-1), aspect=aspecto)
            ejes[0, col].set_title(f"{nombre} ({etiqueta}): CT (magenta) + CBCT (verde)")

            alto, ancho = c.shape
            tam = max(8, min(alto, ancho) // 10)
            fi, ci = np.indices((alto, ancho))
            tablero = ((fi // tam + ci // tam) % 2) == 0
            ejes[1, col].imshow(np.where(tablero, c, b), cmap="gray", aspect=aspecto, vmin=0, vmax=1)
            ejes[1, col].set_title(f"{nombre} ({etiqueta}): tablero de ajedrez")
            for fila in (0, 1):
                ejes[fila, col].axis("off")
        fig.suptitle(f"Paciente {pid}: registro final (SPM) | corte z={cz}, y={cy}, x={cx}")
        fig.tight_layout()
        sufijo = "" if len(posiciones) == 1 else f"_nivel{k}"
        ruta = destino / f"{pid}_registro_final{sufijo}.png"
        fig.savefig(ruta, dpi=130)
        plt.close(fig)
        rutas.append(ruta)
    return rutas


# ---------------------------------------------------------------------------
# Proceso por paciente
# ---------------------------------------------------------------------------

def procesar_paciente(
    pid: str,
    carpeta_nuevo: Path,
    registered_dir: Path,
    output_root: Path,
    margen_mm: float,
    sobrescribir: bool,
    sin_escribir: bool,
    figuras: bool,
    fig_niveles: int = 1,
    fig_cortes: tuple[int | None, int | None, int | None] = (None, None, None),
) -> dict[str, object]:
    print(f"\n{'=' * 72}\nPaciente {pid}: SPM -> par registrado final\n{'=' * 72}")

    fila: dict[str, object] = {c: "" for c in CAMPOS_INFORME}
    fila.update(patient_id=pid, estado="ERROR")

    out_dir = output_root / pid
    ruta_ct_out = out_dir / f"{pid}_ct.nii.gz"
    ruta_cbct_out = out_dir / f"{pid}_cbct_registered.nii.gz"

    if ruta_ct_out.exists() and ruta_cbct_out.exists() and not sobrescribir and not sin_escribir:
        fila.update(estado="OMITIDO", mensaje="Salidas existentes; usa --sobrescribir")
        print("  ↷ Salidas existentes; se omite (usa --sobrescribir para regenerar).")
        return fila

    avisos: list[str] = []
    try:
        ruta_ct, ruta_cbct, av = elegir_archivos(carpeta_nuevo, pid)
        avisos += av
        fila["ct_origen"] = ruta_ct.name
        fila["cbct_origen"] = ruta_cbct.name
        print(f"  CT   (SPM): {ruta_ct.name}\n  CBCT (SPM): {ruta_cbct.name}")

        ct, g_ct = leer_imagen_spm(ruta_ct)
        cbct, g_cbct = leer_imagen_spm(ruta_cbct)
        validar_geometria_3d(ct, "CT de SPM")
        validar_geometria_3d(cbct, "CBCT de SPM")
        for etiqueta, g in (("CT", g_ct), ("CBCT", g_cbct)):
            if g["itk_difiere"]:
                print(
                    f"  (info) {etiqueta}: SimpleITK leería otra posición que el {g['fuente']} "
                    f"de la cabecera; se usa el {g['fuente']} (criterio SPM)."
                )
        print(f"  CT   tamaño={ct.GetSize()} spacing={tuple(round(v, 4) for v in ct.GetSpacing())}")
        print(f"  CBCT tamaño={cbct.GetSize()} spacing={tuple(round(v, 4) for v in cbct.GetSpacing())}")

        # --- Comparación con el registro previo (si existe) -----------------
        previo_dir = registered_dir / pid
        ruta_ct_previo = previo_dir / f"{pid}_ct.nii.gz"
        ruta_cbct_previo = previo_dir / f"{pid}_cbct_registered.nii.gz"

        if ruta_ct_previo.is_file():
            igual = cabeceras_iguales(geometria_spm(ruta_ct_previo), g_ct)
            fila["ct_igual_que_registro_previo"] = "SI" if igual else "NO"
            if not igual:
                avisos.append(
                    "la cabecera del CT de SPM difiere de la del registro previo "
                    "(se usa la de SPM como referencia)"
                )
        else:
            fila["ct_igual_que_registro_previo"] = "SIN_REFERENCIA"

        if ruta_cbct_previo.is_file():
            h_prev = geometria_spm(ruta_cbct_previo)
            h_new = g_cbct
            if h_prev["size"] == h_new["size"] and np.allclose(h_prev["spacing"], h_new["spacing"], atol=1e-4):
                m = metricas_correccion(_afin(h_prev), _afin(h_new), h_new["size"])
                fila.update({k: round(v, 4) for k, v in m.items()})
                print(
                    "  Corrección de SPM: "
                    f"centro {m['corr_traslacion_centro_mm']:.1f} mm "
                    f"(dx={m['corr_dx_mm']:.1f}, dy={m['corr_dy_mm']:.1f}, dz={m['corr_dz_mm']:.1f}) | "
                    f"rotación {m['corr_rotacion_grados']:.2f}°"
                )
                if m["corr_traslacion_centro_mm"] < 0.01 and m["corr_rotacion_grados"] < 0.01:
                    avisos.append(
                        "no se detecta NINGUNA corrección respecto al registro previo: "
                        "comprueba que SPM guardó el ajuste en este archivo "
                        "(python inspeccionar_nifti.py <archivo>)"
                    )
                if m["escala_min"] < 1 - TOLERANCIA_ESCALA or m["escala_max"] > 1 + TOLERANCIA_ESCALA or m["error_ortogonalidad"] > TOLERANCIA_ESCALA:
                    avisos.append("el cambio de SPM no es rígido puro (escala/cizalla)")
                if m["corr_traslacion_centro_mm"] > UMBRAL_CORRECCION_MM:
                    avisos.append(f"corrección de traslación grande ({m['corr_traslacion_centro_mm']:.0f} mm)")
                if m["corr_rotacion_grados"] > UMBRAL_CORRECCION_GRADOS:
                    avisos.append(f"corrección de rotación grande ({m['corr_rotacion_grados']:.1f}°)")
            else:
                # Rejillas distintas (p. ej. distinto nº de cortes). Si en el registro
                # previo el CBCT estaba sobre la rejilla del CT, la corrección de SPM
                # es la diferencia entre el CBCT de SPM y el CT de SPM (que no se mueve).
                g_ct_prev = geometria_spm(ruta_ct_previo) if ruta_ct_previo.is_file() else None
                if (
                    g_ct_prev is not None
                    and cabeceras_iguales(h_prev, g_ct_prev)
                    and g_ct["size"] == h_new["size"]
                    and np.allclose(g_ct["spacing"], h_new["spacing"], atol=1e-4)
                ):
                    m = metricas_correccion(_afin(g_ct), _afin(h_new), h_new["size"])
                    fila.update({k: round(v, 4) for k, v in m.items()})
                    avisos.append(
                        "registro previo con otra rejilla; corrección medida respecto al CT de SPM"
                    )
                    print(
                        "  Corrección de SPM (respecto al CT): "
                        f"centro {m['corr_traslacion_centro_mm']:.2f} mm "
                        f"(dx={m['corr_dx_mm']:.2f}, dy={m['corr_dy_mm']:.2f}, dz={m['corr_dz_mm']:.2f}) | "
                        f"rotación {m['corr_rotacion_grados']:.3f}°"
                    )
                    if m["escala_min"] < 1 - TOLERANCIA_ESCALA or m["escala_max"] > 1 + TOLERANCIA_ESCALA or m["error_ortogonalidad"] > TOLERANCIA_ESCALA:
                        avisos.append("el cambio de SPM no es rígido puro (escala/cizalla)")
                    if m["corr_traslacion_centro_mm"] > UMBRAL_CORRECCION_MM:
                        avisos.append(f"corrección de traslación grande ({m['corr_traslacion_centro_mm']:.0f} mm)")
                    if m["corr_rotacion_grados"] > UMBRAL_CORRECCION_GRADOS:
                        avisos.append(f"corrección de rotación grande ({m['corr_rotacion_grados']:.1f}°)")
                else:
                    avisos.append(
                        "el CBCT de SPM y el registro previo no comparten rejilla; "
                        "no se calcula la magnitud de la corrección"
                    )
        else:
            avisos.append("no hay registro previo con el que medir la corrección")

        # --- Remuestreo del CBCT sobre la rejilla del CT -------------------
        cbct_en_ct = sitk.Resample(
            cbct, ct, sitk.Euler3DTransform(), sitk.sitkLinear, PADDING_HU, sitk.sitkFloat32
        )
        huella = crear_mascara_huella(ct, cbct)
        solap_pct, bbox = estadisticas_huella(huella, cbct)
        fila["solapamiento_caja_cbct_pct"] = round(solap_pct, 2)
        print(f"  Caja del CBCT dentro del CT ≈ {solap_pct:.1f}%")
        if solap_pct < UMBRAL_SOLAPAMIENTO_PCT:
            avisos.append(f"solapamiento bajo ({solap_pct:.1f}%)")
        if huella_toca_borde(bbox, ct.GetSize()):
            # Informativo: es lo normal cuando el CT ya viene recortado al CBCT.
            print("  (info) la caja del CBCT llega al borde de la rejilla del CT")

        inicio, tam = roi_con_margen(bbox, ct, margen_mm)
        roi = sitk.RegionOfInterestImageFilter()
        roi.SetIndex(inicio)
        roi.SetSize(tam)
        ct_roi = roi.Execute(ct)
        cbct_roi = roi.Execute(cbct_en_ct)
        if not geometrias_coinciden(ct_roi, cbct_roi):
            raise RuntimeError("CT y CBCT no comparten geometría tras recortar")
        fila["roi_final_size"] = "x".join(str(v) for v in ct_roi.GetSize())

        cbct_arr = sitk.GetArrayFromImage(cbct_roi)
        fila["pct_padding_en_roi"] = round(100.0 * float(np.mean(es_relleno(cbct_arr))), 2)
        print(f"  ROI final CT/CBCT: {ct_roi.GetSize()} | padding en ROI: {fila['pct_padding_en_roi']}%")

        if figuras:
            try:
                rutas_fig = guardar_figuras(
                    pid,
                    sitk.GetArrayFromImage(ct_roi),
                    cbct_arr,
                    tuple(float(v) for v in ct_roi.GetSpacing()),
                    output_root / "figuras_registro",
                    n_niveles=fig_niveles,
                    cortes=fig_cortes,
                )
                for rf in rutas_fig:
                    print(f"  ✓ Figura: {rf}")
            except Exception as exc:  # las figuras nunca deben tumbar el proceso
                avisos.append(f"no se pudo crear la figura: {type(exc).__name__}: {exc}")

        if sin_escribir:
            print("  (--sin-escribir: no se guardan imágenes)")
        else:
            escribir_grupo_atomico(
                [
                    (ruta_ct_out, lambda t: sitk.WriteImage(ct_roi, str(t), True)),
                    (ruta_cbct_out, lambda t: sitk.WriteImage(cbct_roi, str(t), True)),
                ]
            )
            print(f"  ✓ Guardado en {out_dir}")

        if avisos:
            fila.update(estado="REVISAR", mensaje="; ".join(avisos))
            print(f"  ⚠ REVISAR: {fila['mensaje']}")
        else:
            fila.update(estado="OK", mensaje="Conversión completada")
            print("  ✓ Sin avisos.")
    except Exception as exc:
        fila.update(estado="ERROR", mensaje=f"{type(exc).__name__}: {exc}")
        print(f"  ✗ {fila['mensaje']}")
    return fila


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paso 2c: registro manual de SPM -> par CT/CBCT registrado final"
    )
    p.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    p.add_argument("--pacientes", nargs="+", help="IDs a procesar (sin él: todas las carpetas 'XXXX nuevo')")
    p.add_argument("--registered-dir", help="Carpeta con 'XXXX' y 'XXXX nuevo' (default: PACIENTES_REGISTERED)")
    p.add_argument("--output-dir", help="Salida (default: PACIENTES_REGISTERED_FINAL)")
    p.add_argument("--margen-roi-mm", type=float, default=5.0, help="Margen alrededor de la caja del CBCT (default: 5)")
    p.add_argument("--sobrescribir", action="store_true")
    p.add_argument("--sin-escribir", action="store_true", help="Solo comprueba y mide; no guarda imágenes")
    p.add_argument("--figuras", action="store_true", help="Guarda figuras de fusión/tablero para la memoria")
    p.add_argument(
        "--figura-niveles", type=int, default=1,
        help="Nº de niveles repartidos por la zona con CBCT; una imagen por nivel (default: 1)",
    )
    p.add_argument("--corte-axial", type=int, help="Índice z (vóxel) del corte axial en la región final")
    p.add_argument("--corte-coronal", type=int, help="Índice y (vóxel) del corte coronal")
    p.add_argument("--corte-sagital", type=int, help="Índice x (vóxel) del corte sagital")
    return p


def main() -> int:
    args = construir_parser().parse_args()
    if not np.isfinite(args.margen_roi_mm) or args.margen_roi_mm < 0:
        print("ERROR: --margen-roi-mm debe ser finito y no negativo")
        return 2

    if args.figura_niveles < 1 or args.figura_niveles > 20:
        print("ERROR: --figura-niveles debe estar entre 1 y 20")
        return 2
    cortes = (args.corte_axial, args.corte_coronal, args.corte_sagital)
    if any(c is not None and c < 0 for c in cortes):
        print("ERROR: los índices de corte no pueden ser negativos")
        return 2

    rutas = obtener_rutas(args.project_dir)
    registered = Path(args.registered_dir).resolve() if args.registered_dir else rutas.registered
    output_root = Path(args.output_dir).resolve() if args.output_dir else rutas.proyecto / "PACIENTES_REGISTERED_FINAL"

    try:
        carpetas = detectar_carpetas_nuevo(registered, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if not carpetas:
        print(f"ERROR: no hay carpetas 'XXXX nuevo' en {registered}")
        return 2

    print(f"Pacientes con registro SPM: {texto_pacientes(list(carpetas))}")
    print(f"Entrada: {registered}\nSalida : {output_root}")

    filas = [
        procesar_paciente(
            pid, carpeta, registered, output_root,
            args.margen_roi_mm, args.sobrescribir, args.sin_escribir, args.figuras,
            args.figura_niveles, cortes,
        )
        for pid, carpeta in carpetas.items()
    ]

    informe = guardar_informe_acumulado(
        output_root / "informe_spm_a_registrado.csv", filas, CAMPOS_INFORME
    )
    errores = [f for f in filas if f["estado"] == "ERROR"]
    revisar = [f for f in filas if f["estado"] == "REVISAR"]
    print(f"\nPaso 2c terminado. Informe: {informe}")
    for f in revisar:
        print(f"  ⚠ {f['patient_id']}: {f['mensaje']}")
    for f in errores:
        print(f"  ✗ {f['patient_id']}: {f['mensaje']}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
