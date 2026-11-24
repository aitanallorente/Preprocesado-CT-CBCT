"""Lectura mínima de la cabecera NIfTI-1 (solo numpy; sin SimpleITK).

Por qué existe
--------------
SPM ("Reorient images") guarda el ajuste en el *sform* de la cabecera y deja
intacto el *qform* (posición original). SimpleITK/ITK, cuando ambos existen y
son distintos, suele leer el *qform*: ve la posición ANTERIOR al ajuste. Aquí
se lee la cabecera a mano siguiendo el criterio de SPM (sform si sform_code>0,
si no qform) para obtener la geometría real.

``geometria_spm`` devuelve la geometría en coordenadas LPS (las de
ITK/Slicer); las funciones ``afin_*_ras`` devuelven la matriz en RAS, tal como
está en la cabecera.

Lo usan ``paso2c_spm_a_registrado.py`` e ``inspeccionar_nifti.py``.
"""

from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path

import numpy as np

RAS_A_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _abrir(ruta: Path):
    ruta = Path(ruta)
    if ruta.name.lower().endswith(".gz"):
        return gzip.open(ruta, "rb")
    return open(ruta, "rb")


def leer_cabecera_nifti(ruta: str | Path) -> dict:
    """Campos de la cabecera NIfTI-1 relevantes para la geometría."""

    with _abrir(Path(ruta)) as f:
        raw = f.read(348)
    if len(raw) < 348:
        raise ValueError(f"{Path(ruta).name}: cabecera NIfTI incompleta")
    for end in ("<", ">"):
        if struct.unpack(end + "i", raw[:4])[0] == 348:
            break
    else:
        raise ValueError(f"{Path(ruta).name}: no es NIfTI-1 (sizeof_hdr != 348)")
    if raw[344:347] not in (b"n+1", b"ni1"):
        raise ValueError(f"{Path(ruta).name}: firma NIfTI no válida")

    dim = struct.unpack(end + "8h", raw[40:56])
    pixdim = struct.unpack(end + "8f", raw[76:108])
    qform_code, sform_code = struct.unpack(end + "2h", raw[252:256])
    b, c, d, qx, qy, qz = struct.unpack(end + "6f", raw[256:280])
    srow = struct.unpack(end + "12f", raw[280:328])
    return {
        "dim": dim,
        "pixdim": pixdim,
        "qform_code": int(qform_code),
        "sform_code": int(sform_code),
        "quatern": (b, c, d),
        "qoffset": (qx, qy, qz),
        "srow": srow,
    }


def afin_qform_ras(h: dict) -> np.ndarray:
    b, c, d = (float(v) for v in h["quatern"])
    a2 = 1.0 - (b * b + c * c + d * d)
    if a2 < 1e-7:
        n = math.sqrt(b * b + c * c + d * d)
        b, c, d, a = b / n, c / n, d / n, 0.0
    else:
        a = math.sqrt(a2)
    rot = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
            [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
            [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
        ]
    )
    dx, dy, dz = (abs(float(v)) for v in h["pixdim"][1:4])
    qfac = -1.0 if h["pixdim"][0] < 0 else 1.0
    afin = np.eye(4)
    afin[:3, :3] = rot @ np.diag([dx, dy, dz * qfac])
    afin[:3, 3] = h["qoffset"]
    return afin


def afin_sform_ras(h: dict) -> np.ndarray:
    afin = np.eye(4)
    afin[:3, :] = np.asarray(h["srow"], dtype=float).reshape(3, 4)
    return afin


def afin_spm_ras(h: dict) -> tuple[np.ndarray, str]:
    """Criterio de SPM: sform si sform_code>0; si no, qform si qform_code>0."""

    if h["sform_code"] > 0:
        return afin_sform_ras(h), "sform"
    if h["qform_code"] > 0:
        return afin_qform_ras(h), "qform"
    raise ValueError("la cabecera no tiene sform ni qform válidos")


def geometria_spm(ruta: str | Path) -> dict:
    """Geometría física (LPS) según el criterio de SPM.

    Devuelve size, spacing, origin, direction (lista de 9), fuente y afin.
    Lanza ValueError si la matriz no es una rígida con spacing (cizalla).
    """

    h = leer_cabecera_nifti(ruta)
    afin_ras, fuente = afin_spm_ras(h)
    afin = RAS_A_LPS @ afin_ras
    m = afin[:3, :3]
    spacing = np.linalg.norm(m, axis=0)
    if np.any(spacing <= 0) or not np.all(np.isfinite(afin)):
        raise ValueError(f"{Path(ruta).name}: matriz de cabecera no válida")
    direccion = m / spacing
    error_ort = float(np.max(np.abs(direccion.T @ direccion - np.eye(3))))
    if error_ort > 1e-3:
        raise ValueError(
            f"{Path(ruta).name}: la matriz de la cabecera tiene cizalla "
            f"(error de ortogonalidad {error_ort:.2e}); no es un ajuste rígido"
        )
    return {
        "size": tuple(int(v) for v in h["dim"][1:4]),
        "spacing": tuple(float(v) for v in spacing),
        "origin": tuple(float(v) for v in afin[:3, 3]),
        "direction": tuple(float(v) for v in direccion.ravel()),
        "fuente": fuente,
    }


def afin_indice_a_fisico(size, spacing, origin, direction) -> np.ndarray:
    """Matriz 4x4 que convierte índice (i,j,k) en coordenada física (mm)."""

    d = np.asarray(direction, dtype=float).reshape(3, 3)
    afin = np.eye(4)
    afin[:3, :3] = d @ np.diag(np.asarray(spacing, dtype=float))
    afin[:3, 3] = np.asarray(origin, dtype=float)
    return afin


def diferencia_afines(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(desplazamiento de traslación en mm, ángulo de rotación relativo en grados)."""

    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    ra = a[:3, :3] / np.linalg.norm(a[:3, :3], axis=0)
    rb = b[:3, :3] / np.linalg.norm(b[:3, :3], axis=0)
    rel = ra @ rb.T
    cos_ang = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return dt, math.degrees(math.acos(cos_ang))


def metricas_correccion(afin_previo: np.ndarray, afin_nuevo: np.ndarray, size) -> dict[str, float]:
    """Cuánto movió SPM la imagen: M = A_nuevo · A_previo^-1.

    Solo tiene sentido si ambas cabeceras comparten rejilla de índices.
    """

    m = afin_nuevo @ np.linalg.inv(afin_previo)
    r3 = m[:3, :3]
    sv = np.linalg.svd(r3, compute_uv=False)
    error_ort = float(np.max(np.abs(r3.T @ r3 - np.eye(3))))

    u, _s, vt = np.linalg.svd(r3)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    cos_ang = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    angulo = math.degrees(math.acos(cos_ang))

    centro_idx = np.append((np.asarray(size, dtype=float) - 1.0) / 2.0, 1.0)
    d = (afin_nuevo @ centro_idx)[:3] - (afin_previo @ centro_idx)[:3]

    return {
        "corr_dx_mm": float(d[0]),
        "corr_dy_mm": float(d[1]),
        "corr_dz_mm": float(d[2]),
        "corr_traslacion_centro_mm": float(np.linalg.norm(d)),
        "corr_rotacion_grados": angulo,
        "escala_min": float(sv.min()),
        "escala_max": float(sv.max()),
        "error_ortogonalidad": error_ort,
    }
