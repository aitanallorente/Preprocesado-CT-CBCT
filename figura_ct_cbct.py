"""Figura de la memoria (sección 2.2.3): CT y CBCT del mismo paciente, 2 x 2.

    columna izquierda: CT              columna derecha: CBCT registrado
    (a) corte axial                    (b) el mismo corte axial
    (c) corte coronal del CT COMPLETO  (d) el mismo corte coronal: CBCT en su
        + rectángulo discontinuo con       posición y relleno negro hasta la
          la zona que cubre el CBCT        altura del CT

De dónde salen los datos
------------------------
* CT completo: ``PACIENTES_NIFTI/XXXX/XXXX_ct.nii.gz`` (paso 1, sin recortar).
* CBCT registrado: ``PACIENTES_REGISTERED_FINAL/XXXX/XXXX_cbct_registered.nii.gz``
  (paso 2c, en HU, ya con el ajuste de SPM).
* El CBCT se coloca sobre la rejilla del CT completo con la transformación
  identidad: los dos están ya en el mismo espacio físico. Como comprobación, se
  compara el CT de ``PACIENTES_REGISTERED_FINAL`` con el CT completo en la zona
  común; si no coinciden, el script avisa (significaría que SPM movió el CT).

Todos los paneles usan la MISMA ventana (por defecto tejido blando: ancho 400,
nivel 40 HU) y el espaciado real (los ejes están en mm, así que el coronal no
sale deformado). No se dibuja ningún dato del paciente: solo los píxeles.
El relleno (-1024 HU) y las esquinas del FOV circular (-1200 HU) quedan por
debajo de la ventana y se ven negros.

Uso
---
1) Explorar cortes para elegir uno con gas o contraste::

       python figura_ct_cbct.py --paciente 0002 --explorar

   Genera ``figuras_memoria/0002_explorar_axial.png`` y
   ``0002_explorar_coronal.png`` con el índice de cada corte.

2) Figura definitiva con los cortes elegidos (índices del CT completo)::

       python figura_ct_cbct.py --paciente 0002 --corte-axial 85 --corte-coronal 260

3) Para colocar flechas, generar primero con ``--guias`` (ejes en mm), leer
   las coordenadas y pasarlas en un JSON::

       python figura_ct_cbct.py --paciente 0002 --corte-axial 85 --corte-coronal 260 --guias
       python figura_ct_cbct.py --paciente 0002 --corte-axial 85 --corte-coronal 260 --flechas flechas_0002.json

   Formato del JSON (x, y en mm del panel, origen arriba a la izquierda; la
   punta de la flecha está en (x, y) y la cola en (x + dx, y + dy))::

       [
         {"panel": "b", "x": 30, "y": 20, "dx": 25, "dy": 25, "texto": "esquina -1200 HU"},
         {"panel": "b", "x": 200, "y": 150, "dx": -40, "dy": 30, "texto": "sombreado"}
       ]

Salidas (``figuras_memoria/``): la figura compuesta en PNG y PDF, cada panel
por separado (para anotar en otro programa) y un .txt con los cortes usados y
la cobertura craneocaudal del CBCT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PADDING_HU = -1024.0
TOLERANCIA_PADDING = 1e-3
COLOR_MARCAS = "#FFD400"   # amarillo
ANCHO_PAGINA_IN = 6.3      # ~16 cm


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

@dataclass
class Volumenes:
    """Arrays en orden (z, y, x) sobre la rejilla del CT completo, en LPS.

    En LPS: x crece hacia la izquierda del paciente, y hacia posterior y z hacia
    craneal. Así, dibujando la columna 0 a la izquierda se obtiene la convención
    radiológica (izquierda del paciente a la derecha de la imagen) y la fila 0
    del axial es anterior.
    """

    ct: np.ndarray
    cbct: np.ndarray          # -1024 fuera del CBCT
    huella: np.ndarray        # bool: dónde hay CBCT real
    spacing: tuple[float, float, float]   # (sx, sy, sz) en mm


def es_relleno(arr: np.ndarray) -> np.ndarray:
    return np.isclose(arr, np.float32(PADDING_HU), rtol=0.0, atol=TOLERANCIA_PADDING)


def cargar_volumenes(proyecto: Path, pid: str) -> Volumenes:
    import SimpleITK as sitk

    ruta_ct_full = proyecto / "PACIENTES_NIFTI" / pid / f"{pid}_ct.nii.gz"
    carpeta_final = proyecto / "PACIENTES_REGISTERED_FINAL" / pid
    ruta_ct_fin = carpeta_final / f"{pid}_ct.nii.gz"
    ruta_cbct_fin = carpeta_final / f"{pid}_cbct_registered.nii.gz"
    for r in (ruta_ct_full, ruta_ct_fin, ruta_cbct_fin):
        if not r.is_file():
            raise FileNotFoundError(f"No existe: {r}")

    ct_full = sitk.DICOMOrient(sitk.ReadImage(str(ruta_ct_full), sitk.sitkFloat32), "LPS")
    ct_fin = sitk.ReadImage(str(ruta_ct_fin), sitk.sitkFloat32)
    cbct_fin = sitk.ReadImage(str(ruta_cbct_fin), sitk.sitkFloat32)

    identidad = sitk.Euler3DTransform()

    # CBCT sobre la rejilla del CT completo (relleno -1024 fuera)
    cbct_full = sitk.Resample(cbct_fin, ct_full, identidad, sitk.sitkLinear, PADDING_HU, sitk.sitkFloat32)

    # Huella: vóxeles del CBCT que no son relleno, llevados al CT completo
    valido = sitk.GetImageFromArray((~es_relleno(sitk.GetArrayFromImage(cbct_fin))).astype(np.uint8))
    valido.CopyInformation(cbct_fin)
    huella = sitk.Resample(valido, ct_full, identidad, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # Comprobación: el CT del paso 2c debe coincidir con el CT completo
    caja = sitk.Image(ct_fin.GetSize(), sitk.sitkUInt8) + 1
    caja.CopyInformation(ct_fin)
    caja_full = sitk.GetArrayFromImage(
        sitk.Resample(caja, ct_full, identidad, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ).astype(bool)
    if not caja_full.any():
        raise RuntimeError(
            "El par de PACIENTES_REGISTERED_FINAL no cae dentro del CT de PACIENTES_NIFTI: "
            "no están en el mismo espacio físico."
        )
    ct_fin_en_full = sitk.GetArrayFromImage(
        sitk.Resample(ct_fin, ct_full, identidad, sitk.sitkLinear, PADDING_HU, sitk.sitkFloat32)
    )
    ct_full_arr = sitk.GetArrayFromImage(ct_full)
    dif = float(np.median(np.abs(ct_fin_en_full[caja_full] - ct_full_arr[caja_full])))
    print(f"  Comprobación CT final vs CT completo: diferencia mediana {dif:.1f} HU")
    if dif > 5.0:
        print(
            "  ⚠ El CT de PACIENTES_REGISTERED_FINAL no coincide con el CT completo.\n"
            "    Probablemente SPM cambió la cabecera del CT: la posición del CBCT en (c)/(d)\n"
            "    puede no ser correcta. Revisa informe_spm_a_registrado.csv\n"
            "    (columna ct_igual_que_registro_previo)."
        )

    return Volumenes(
        ct=ct_full_arr,
        cbct=sitk.GetArrayFromImage(cbct_full),
        huella=sitk.GetArrayFromImage(huella).astype(bool),
        spacing=tuple(float(v) for v in ct_full.GetSpacing()),
    )


# ---------------------------------------------------------------------------
# Geometría de los cortes
# ---------------------------------------------------------------------------

def caja_huella(huella: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Rangos (min, max) inclusivos de z, y, x de la huella del CBCT."""
    if not huella.any():
        raise RuntimeError("La huella del CBCT está vacía.")
    zz, yy, xx = np.nonzero(huella)
    return (int(zz.min()), int(zz.max())), (int(yy.min()), int(yy.max())), (int(xx.min()), int(xx.max()))


def rango_cuerpo_x(ct: np.ndarray, sx: float, margen_mm: float = 15.0) -> tuple[int, int]:
    """Columnas x con cuerpo en algún punto del CT (para quitar aire lateral)."""
    con_cuerpo = np.nonzero((ct > -500).any(axis=(0, 1)))[0]
    if con_cuerpo.size == 0:
        return 0, ct.shape[2] - 1
    m = int(np.ceil(margen_mm / sx))
    return max(0, int(con_cuerpo.min()) - m), min(ct.shape[2] - 1, int(con_cuerpo.max()) + m)


def ventana(ancho: float, nivel: float) -> tuple[float, float]:
    return nivel - ancho / 2.0, nivel + ancho / 2.0


def corte_axial(v: Volumenes, z: int, margen_mm: float = 5.0):
    """Corte axial recortado a la caja de la huella en x, y. Devuelve
    (ct, cbct, extent) con extent en mm y origen arriba a la izquierda."""
    (_, _), (y0, y1), (x0, x1) = caja_huella(v.huella)
    sx, sy, _ = v.spacing
    my, mx = int(np.ceil(margen_mm / sy)), int(np.ceil(margen_mm / sx))
    y0, y1 = max(0, y0 - my), min(v.ct.shape[1] - 1, y1 + my)
    x0, x1 = max(0, x0 - mx), min(v.ct.shape[2] - 1, x1 + mx)
    ct = v.ct[z, y0:y1 + 1, x0:x1 + 1]
    cb = v.cbct[z, y0:y1 + 1, x0:x1 + 1]
    extent = (0.0, ct.shape[1] * sx, ct.shape[0] * sy, 0.0)
    return ct, cb, extent


def corte_coronal(v: Volumenes, y: int):
    """Corte coronal del CT completo (craneal arriba). Devuelve (ct, cbct,
    extent, rectángulo) con el rectángulo de la huella en mm (x, y, ancho, alto)
    o None si la huella no pasa por ese corte."""
    sx, _, sz = v.spacing
    x0, x1 = rango_cuerpo_x(v.ct, sx)
    ct = np.flipud(v.ct[:, y, x0:x1 + 1])
    cb = np.flipud(v.cbct[:, y, x0:x1 + 1])
    h = np.flipud(v.huella[:, y, x0:x1 + 1])
    extent = (0.0, ct.shape[1] * sx, ct.shape[0] * sz, 0.0)
    rect = None
    if h.any():
        filas, cols = np.nonzero(h)
        rect = (
            cols.min() * sx,
            filas.min() * sz,
            (cols.max() - cols.min() + 1) * sx,
            (filas.max() - filas.min() + 1) * sz,
        )
    return ct, cb, extent, rect


def cobertura_craneocaudal(v: Volumenes) -> tuple[float, float, float]:
    """(extensión CBCT mm, extensión CT mm, porcentaje)."""
    (z0, z1), _, _ = caja_huella(v.huella)
    sz = v.spacing[2]
    cb_mm = (z1 - z0 + 1) * sz
    ct_mm = v.ct.shape[0] * sz
    return cb_mm, ct_mm, 100.0 * cb_mm / ct_mm


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------

def _mostrar(ax, img, extent, lo, hi, guias: bool):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent,
              interpolation="nearest", aspect="equal")
    if guias:
        ax.grid(True, color="cyan", alpha=0.35, linewidth=0.5)
        ax.tick_params(labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def _flechas(ax, flechas: list[dict], panel: str):
    for f in flechas:
        if str(f.get("panel", "")).lower() != panel:
            continue
        x, y = float(f["x"]), float(f["y"])
        dx, dy = float(f.get("dx", 20)), float(f.get("dy", -20))
        ax.annotate(
            f.get("texto", ""),
            xy=(x, y), xytext=(x + dx, y + dy),
            color=COLOR_MARCAS, fontsize=7, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=COLOR_MARCAS, lw=1.4,
                            shrinkA=2, shrinkB=0),
        )


def _rectangulo(ax, rect):
    from matplotlib.patches import Rectangle

    if rect is None:
        return
    ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], fill=False,
                           edgecolor=COLOR_MARCAS, linestyle=(0, (4, 3)), linewidth=1.3))


def _etiqueta(ax, letra: str):
    ax.text(0.02, 0.98, f"({letra})", transform=ax.transAxes, color="white",
            fontsize=9, fontweight="bold", va="top", ha="left")


def dibujar_figura(
    v: Volumenes, z: int, y: int, lo: float, hi: float,
    flechas: list[dict], guias: bool, destino: Path, base: str,
    rotulos: bool = True, dpi: int = 300,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_ct, a_cb, a_ext = corte_axial(v, z)
    c_ct, c_cb, c_ext, rect = corte_coronal(v, y)
    if rect is None:
        print(f"  ⚠ El corte coronal y={y} no atraviesa el CBCT: no hay rectángulo en (c).")

    alto_ax = a_ext[2] / a_ext[1]
    alto_co = c_ext[2] / c_ext[1]
    ancho_col = ANCHO_PAGINA_IN / 2
    fig, ejes = plt.subplots(
        2, 2, figsize=(ANCHO_PAGINA_IN, ancho_col * (alto_ax + alto_co) + 0.35),
        gridspec_kw={"height_ratios": [alto_ax, alto_co], "wspace": 0.03, "hspace": 0.03},
    )
    paneles = {
        "a": (ejes[0, 0], a_ct, a_ext),
        "b": (ejes[0, 1], a_cb, a_ext),
        "c": (ejes[1, 0], c_ct, c_ext),
        "d": (ejes[1, 1], c_cb, c_ext),
    }
    for letra, (ax, img, ext) in paneles.items():
        _mostrar(ax, img, ext, lo, hi, guias)
        _etiqueta(ax, letra)
        _flechas(ax, flechas, letra)
    _rectangulo(ejes[1, 0], rect)
    if rotulos:
        ejes[0, 0].set_title("CT", fontsize=9)
        ejes[0, 1].set_title("CBCT (registrado)", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95 if rotulos else 0.99, bottom=0.01)

    destino.mkdir(parents=True, exist_ok=True)
    sufijo = "_guias" if guias else ""
    rutas = []
    for ext in ("png", "pdf"):
        r = destino / f"{base}{sufijo}.{ext}"
        fig.savefig(r, dpi=dpi, facecolor="white")
        rutas.append(r)
    plt.close(fig)

    # Paneles sueltos (sin rótulos), por si se anotan en otro programa
    if not guias:
        for letra, (_, img, ext) in paneles.items():
            w_in = 3.2
            f2, ax2 = plt.subplots(figsize=(w_in, w_in * ext[2] / ext[1]))
            _mostrar(ax2, img, ext, lo, hi, False)
            _flechas(ax2, flechas, letra)
            if letra == "c":
                _rectangulo(ax2, rect)
            f2.subplots_adjust(0, 0, 1, 1)
            r = destino / f"{base}_panel_{letra}.png"
            f2.savefig(r, dpi=dpi)
            plt.close(f2)
            rutas.append(r)
    return rutas


def dibujar_exploracion(v: Volumenes, lo: float, hi: float, destino: Path, pid: str,
                        n_axial: int = 12, n_coronal: int = 8) -> list[Path]:
    """Hojas de contacto para elegir cortes: pares CT | CBCT con su índice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (z0, z1), (y0, y1), _ = caja_huella(v.huella)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = []

    zs = np.unique(np.linspace(z0, z1, n_axial + 2)[1:-1].round().astype(int))
    ys = np.unique(np.linspace(y0, y1, n_coronal + 2)[1:-1].round().astype(int))

    for nombre, indices, fn in (
        ("axial", zs, lambda i: corte_axial(v, int(i))[:3]),
        ("coronal", ys, lambda i: corte_coronal(v, int(i))[:3]),
    ):
        ncol = 4  # 2 pares por fila
        nfil = int(np.ceil(len(indices) / 2))
        fig, ejes = plt.subplots(nfil, ncol, figsize=(12, 3.2 * nfil), squeeze=False)
        for k, idx in enumerate(indices):
            ct, cb, ext = fn(idx)
            fila, col = divmod(k, 2)
            for j, (img, t) in enumerate(((ct, "CT"), (cb, "CBCT"))):
                ax = ejes[fila, 2 * col + j]
                _mostrar(ax, img, ext, lo, hi, False)
                eje = "z" if nombre == "axial" else "y"
                ax.set_title(f"{t}  {eje}={idx}", fontsize=8)
        for ax in ejes.flat:
            if not ax.images:
                ax.axis("off")
        fig.suptitle(f"Paciente {pid}: cortes {nombre}es (índices del CT completo)")
        fig.tight_layout()
        r = destino / f"{pid}_explorar_{nombre}.png"
        fig.savefig(r, dpi=110)
        plt.close(fig)
        rutas.append(r)
    return rutas


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figura 2x2 CT / CBCT para la memoria (sección 2.2.3)")
    p.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    p.add_argument("--paciente", default="0002")
    p.add_argument("--corte-axial", type=int, help="Índice z en el CT completo (default: centro del CBCT)")
    p.add_argument("--corte-coronal", type=int, help="Índice y en el CT completo (default: centro del CBCT)")
    p.add_argument("--ancho", type=float, default=400.0, help="Ancho de ventana HU (default: 400)")
    p.add_argument("--nivel", type=float, default=40.0, help="Nivel de ventana HU (default: 40)")
    p.add_argument("--explorar", action="store_true", help="Solo hojas de contacto para elegir cortes")
    p.add_argument("--guias", action="store_true", help="Ejes y rejilla en mm para situar flechas")
    p.add_argument("--flechas", help="JSON con las flechas")
    p.add_argument("--sin-rotulos", action="store_true", help="No escribir 'CT' / 'CBCT' encima")
    p.add_argument("--salida", help="Carpeta de salida (default: <proyecto>/figuras_memoria)")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main() -> int:
    args = construir_parser().parse_args()
    from configuracion import obtener_rutas

    rutas = obtener_rutas(args.project_dir)
    destino = Path(args.salida).resolve() if args.salida else rutas.proyecto / "figuras_memoria"
    pid = args.paciente
    lo, hi = ventana(args.ancho, args.nivel)

    print(f"Paciente {pid} | ventana W={args.ancho:g} L={args.nivel:g} -> [{lo:g}, {hi:g}] HU")
    v = cargar_volumenes(rutas.proyecto, pid)
    (z0, z1), (y0, y1), _ = caja_huella(v.huella)
    print(f"  CT completo (z, y, x) = {v.ct.shape}, spacing (x, y, z) = {tuple(round(s, 3) for s in v.spacing)} mm")
    print(f"  CBCT dentro del CT: z {z0}..{z1}, y {y0}..{y1}")
    cb_mm, ct_mm, pct = cobertura_craneocaudal(v)
    print(f"  Cobertura craneocaudal: {cb_mm:.0f} de {ct_mm:.0f} mm = {pct:.1f} %")

    if args.explorar:
        for r in dibujar_exploracion(v, lo, hi, destino, pid):
            print(f"  ✓ {r}")
        return 0

    z = args.corte_axial if args.corte_axial is not None else (z0 + z1) // 2
    y = args.corte_coronal if args.corte_coronal is not None else (y0 + y1) // 2
    if not z0 <= z <= z1:
        print(f"ERROR: el corte axial z={z} está fuera del CBCT ({z0}..{z1})")
        return 2
    if not 0 <= y < v.ct.shape[1]:
        print(f"ERROR: el corte coronal y={y} está fuera del CT (0..{v.ct.shape[1] - 1})")
        return 2

    flechas: list[dict] = []
    if args.flechas:
        flechas = json.loads(Path(args.flechas).read_text(encoding="utf-8"))

    base = f"{pid}_figura_ct_cbct"
    for r in dibujar_figura(v, z, y, lo, hi, flechas, args.guias, destino, base,
                            rotulos=not args.sin_rotulos, dpi=args.dpi):
        print(f"  ✓ {r}")

    resumen = destino / f"{base}.txt"
    resumen.write_text(
        f"paciente={pid}\ncorte_axial_z={z}\ncorte_coronal_y={y}\n"
        f"ventana_ancho={args.ancho:g}\nventana_nivel={args.nivel:g}\n"
        f"spacing_xyz_mm={v.spacing}\n"
        f"cobertura_cbct_mm={cb_mm:.1f}\nextension_ct_mm={ct_mm:.1f}\ncobertura_pct={pct:.1f}\n",
        encoding="utf-8",
    )
    print(f"  ✓ {resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
