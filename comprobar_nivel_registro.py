"""Comprueba si un par CT/CBCT registrado está al mismo nivel craneocaudal.

Pensado para el paciente 0005, cuyos landmarks dan un RMS bajo (2,9 mm) pero
cuyo CBCT parece estar a otro nivel anatómico que el CT. Un error así no lo
detecta el RMS: si en una de las dos imágenes se marca la vértebra de al lado,
los cinco puntos se desplazan juntos y el ajuste sale igual de bueno.

Qué hace: desplaza el CBCT registrado solo en el eje craneocaudal (±rango mm)
y, para cada desplazamiento, mide dentro de la zona donde hay CBCT:
  * Dice del hueso (CT y CBCT umbralizados): sensible al nivel vertebral y a
    la presencia de costillas.
  * Correlación (NCC) de intensidades recortadas a [-1000, 1000] HU.
Si el máximo está en ~0 mm, el nivel es correcto. Si está a 20-40 mm, lo más
probable es que una vértebra se haya tomado por la de al lado.

No modifica ningún archivo: solo imprime una tabla y guarda una figura PNG.

Entrada (en HU, misma rejilla CT/CBCT):
  por defecto  PACIENTES_REGISTERED/XXXX/XXXX_ct.nii.gz y XXXX_cbct_registered.nii.gz
  con --carpeta se puede usar PACIENTES_REGISTERED_FINAL. No sirve para
  PACIENTES_FINAL ni PACIENTES_NORMALIZED, que ya no están en HU.

Uso:
  python comprobar_nivel_registro.py --paciente 0005
  python comprobar_nivel_registro.py --paciente 0006          (control que salió bien)
  python comprobar_nivel_registro.py --paciente 0005 --carpeta PACIENTES_REGISTERED_FINAL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RELLENO_CBCT = -1000.0  # el exterior del CBCT se rellena con -1024


def desplazar_z(volumen: np.ndarray, s: int, relleno: float) -> np.ndarray:
    """Desplaza un volumen (z, y, x) s cortes: salida[k] = volumen[k - s]."""

    salida = np.full_like(volumen, relleno)
    if s > 0:
        salida[s:] = volumen[:-s]
    elif s < 0:
        salida[:s] = volumen[-s:]
    else:
        salida[:] = volumen
    return salida


def barrido_z(
    ct: np.ndarray,
    cbct: np.ndarray,
    spacing_z: float,
    rango_mm: float,
    umbral_hueso: float,
    min_fraccion_solape: float = 0.5,
) -> list[dict[str, float]]:
    """Calcula Dice de hueso y NCC para cada desplazamiento en Z del CBCT."""

    n = int(round(rango_mm / spacing_z))
    valido_0 = cbct > RELLENO_CBCT
    n_valido_0 = int(valido_0.sum())
    if n_valido_0 == 0:
        raise ValueError("El CBCT no tiene vóxeles válidos (todo es relleno)")

    hueso_ct = ct > umbral_hueso
    ct_rec = np.clip(ct, -1000.0, 1000.0)
    resultados: list[dict[str, float]] = []
    for s in range(-n, n + 1):
        cb = desplazar_z(cbct, s, -1024.0)
        valido = cb > RELLENO_CBCT
        n_valido = int(valido.sum())
        if n_valido < min_fraccion_solape * n_valido_0:
            continue
        a = hueso_ct & valido
        b = (cb > umbral_hueso) & valido
        suma = int(a.sum()) + int(b.sum())
        dice = 2.0 * int((a & b).sum()) / suma if suma else float("nan")

        x = ct_rec[valido]
        y = np.clip(cb[valido], -1000.0, 1000.0)
        x = x - x.mean()
        y = y - y.mean()
        den = float(np.sqrt((x * x).sum() * (y * y).sum()))
        ncc = float((x * y).sum() / den) if den > 0 else float("nan")
        resultados.append({"s": s, "mm": s * spacing_z, "dice": dice, "ncc": ncc})
    if not resultados:
        raise ValueError("Ningún desplazamiento deja solape suficiente")
    return resultados


def sentido_indice_z(direccion: tuple[float, ...]) -> float:
    """+1 si aumentar el índice Z va hacia superior (craneal), -1 si hacia inferior.

    SimpleITK trabaja en LPS: la tercera componente física positiva es superior.
    La columna 2 de la matriz de dirección es el eje de índice Z.
    """

    m = np.asarray(direccion, dtype=float).reshape(3, 3)
    eje = m[:, 2]
    if abs(eje[2]) < 0.7:
        raise ValueError(
            "El eje de índice Z no es craneocaudal; reoriente a LPS antes (paso 2 ya lo hace)"
        )
    return 1.0 if eje[2] > 0 else -1.0


def interpretar(resultados: list[dict[str, float]], sentido: float) -> list[str]:
    cero = min(resultados, key=lambda r: abs(r["s"]))
    mejor_d = max(resultados, key=lambda r: r["dice"])
    mejor_n = max(resultados, key=lambda r: r["ncc"])
    lineas = [
        f"Sin desplazar:        Dice hueso = {cero['dice']:.3f} | NCC = {cero['ncc']:.3f}",
        f"Máximo Dice hueso en {mejor_d['mm']:+6.1f} mm (Dice {mejor_d['dice']:.3f})",
        f"Máximo NCC        en {mejor_n['mm']:+6.1f} mm (NCC  {mejor_n['ncc']:.3f})",
    ]
    mm = mejor_d["mm"]
    craneal = mm * sentido > 0
    if abs(mm) < 10.0:
        lineas.append(
            "→ El hueso coincide sin desplazar: el nivel vertebral parece correcto. "
            "Si la anatomía blanda no coincide, la causa probable es la respiración."
        )
    else:
        hacia = "CRANEAL (hacia la cabeza)" if craneal else "CAUDAL (hacia los pies)"
        lineas.append(
            f"→ El hueso coincide mejor moviendo el CBCT {abs(mm):.0f} mm hacia {hacia}."
        )
        if 18.0 <= abs(mm) <= 45.0:
            if craneal:
                lineas.append(
                    "  Encaja con UNA vértebra de diferencia: en el CBCT se marcó la vértebra "
                    "SUPERIOR a la marcada en el CT (o en el CT la inferior). Rehaga los landmarks."
                )
            else:
                lineas.append(
                    "  Encaja con UNA vértebra de diferencia: en el CBCT se marcó la vértebra "
                    "INFERIOR a la marcada en el CT (o en el CT la superior). Rehaga los landmarks."
                )
        if abs(mejor_n["mm"] - mm) > 15.0:
            lineas.append(
                "  Aviso: Dice y NCC no coinciden. Mire la figura antes de decidir."
            )
    return lineas


def _fusion(ct2d: np.ndarray, cb2d: np.ndarray) -> np.ndarray:
    """Fusión verde/magenta con ventana de hueso (-200 a 1000 HU)."""

    def ventana(a: np.ndarray) -> np.ndarray:
        return np.clip((a + 200.0) / 1200.0, 0.0, 1.0)

    c = ventana(ct2d)
    b = np.where(cb2d > RELLENO_CBCT, ventana(cb2d), 0.0)
    return np.stack([c, b, c], axis=-1)


def guardar_figura(
    ruta: Path,
    paciente: str,
    ct: np.ndarray,
    cbct: np.ndarray,
    spacing: tuple[float, float, float],
    resultados: list[dict[str, float]],
    umbral_hueso: float,
    sentido: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mejor = max(resultados, key=lambda r: r["dice"])
    # Corte sagital por el centro de la columna: x mediana del hueso del CT en la
    # mitad posterior de la zona con CBCT.
    valido = cbct > RELLENO_CBCT
    hueso = (ct > umbral_hueso) & valido
    if hueso.any():
        _, _, xs = np.nonzero(hueso)
        x0 = int(np.median(xs))
    else:
        x0 = ct.shape[2] // 2

    sx, sy, sz = spacing
    aspecto = sz / sy
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    mms = [r["mm"] for r in resultados]
    ax[0].plot(mms, [r["dice"] for r in resultados], label="Dice hueso")
    ax[0].plot(mms, [r["ncc"] for r in resultados], label="NCC intensidades")
    ax[0].axvline(0, color="gray", ls="--", lw=1)
    ax[0].axvline(mejor["mm"], color="red", ls=":", lw=1.5)
    signo = "craneal" if sentido > 0 else "caudal"
    ax[0].set_xlabel(f"desplazamiento del CBCT (mm, positivo = hacia {signo})")
    ax[0].set_title("¿Dónde coincide mejor?")
    ax[0].legend()

    for a, s, titulo in (
        (ax[1], 0, "Registro actual (0 mm)"),
        (ax[2], int(mejor["s"]), f"CBCT desplazado {mejor['mm']:+.0f} mm"),
    ):
        cb = desplazar_z(cbct, s, -1024.0)
        img = _fusion(ct[:, :, x0], cb[:, :, x0])  # (z, y)
        a.imshow(img, origin="lower" if sentido > 0 else "upper", aspect=aspecto)
        a.set_title(titulo + "\nmagenta = solo CT, verde = solo CBCT, blanco = ambos")
        a.axis("off")
    fig.suptitle(f"Paciente {paciente}: corte sagital por la columna (ventana de hueso)")
    plt.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paciente", required=True)
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--carpeta",
        default="PACIENTES_REGISTERED",
        help="Carpeta de etapa con los pares en HU (por defecto PACIENTES_REGISTERED)",
    )
    parser.add_argument("--rango-mm", type=float, default=80.0)
    parser.add_argument("--umbral-hueso", type=float, default=200.0)
    args = parser.parse_args()

    import SimpleITK as sitk

    from configuracion import obtener_rutas

    rutas = obtener_rutas(args.project_dir)
    pid = args.paciente
    carpeta = rutas.proyecto / args.carpeta / pid
    ruta_ct = carpeta / f"{pid}_ct.nii.gz"
    ruta_cbct = carpeta / f"{pid}_cbct_registered.nii.gz"
    for r in (ruta_ct, ruta_cbct):
        if not r.is_file():
            print(f"ERROR: no existe {r}")
            return 2

    ct_img = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
    cb_img = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
    if ct_img.GetSize() != cb_img.GetSize():
        print("ERROR: CT y CBCT no comparten rejilla")
        return 2

    sentido = sentido_indice_z(ct_img.GetDirection())
    spacing = tuple(float(v) for v in ct_img.GetSpacing())
    ct = sitk.GetArrayFromImage(ct_img)  # (z, y, x)
    cbct = sitk.GetArrayFromImage(cb_img)

    print(f"Paciente {pid}  |  {carpeta}")
    print(f"Tamaño {ct_img.GetSize()}  spacing {tuple(round(v, 2) for v in spacing)} mm")
    resultados = barrido_z(ct, cbct, spacing[2], args.rango_mm, args.umbral_hueso)

    print("\n  desplaz.(mm)   Dice hueso    NCC")
    paso = max(1, int(round(4.0 / spacing[2])))  # una fila cada ~4 mm
    for r in resultados[::paso]:
        print(f"  {r['mm']:+8.1f}       {r['dice']:.3f}      {r['ncc']:.3f}")
    print()
    for linea in interpretar(resultados, sentido):
        print(linea)

    ruta_png = carpeta / f"{pid}_comprobacion_nivel.png"
    guardar_figura(ruta_png, pid, ct, cbct, spacing, resultados, args.umbral_hueso, sentido)
    print(f"\nFigura: {ruta_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
