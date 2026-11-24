"""Utilidades geométricas compartidas por los pasos que usan SimpleITK.

Validación de la geometría de un volumen 3D y comparación de la rejilla
(tamaño, spacing, origen y dirección) entre CT y CBCT. Las usan los pasos 1, 2,
2c, 3 y 4 y la validación final.
"""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any


def _tupla_float(valores: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in valores)


def describir_geometria(imagen: Any) -> dict[str, tuple]:
    """Extrae tamaño y geometría física de una imagen SimpleITK."""

    return {
        "size": tuple(int(v) for v in imagen.GetSize()),
        "spacing": _tupla_float(imagen.GetSpacing()),
        "origin": _tupla_float(imagen.GetOrigin()),
        "direction": _tupla_float(imagen.GetDirection()),
    }


def validar_geometria_3d(imagen: Any, nombre: str = "imagen") -> None:
    """Valida que una imagen tenga una geometría física 3D utilizable."""

    dimension = int(imagen.GetDimension())
    if dimension != 3:
        raise ValueError(f"{nombre}: se esperaba dimensión 3 y se obtuvo {dimension}")

    geometria = describir_geometria(imagen)
    size = geometria["size"]
    spacing = geometria["spacing"]
    origin = geometria["origin"]
    direction = geometria["direction"]

    if len(size) != 3 or any(v < 1 for v in size):
        raise ValueError(f"{nombre}: tamaño no válido {size}")
    if len(spacing) != 3 or any(not isfinite(v) or v <= 0 for v in spacing):
        raise ValueError(f"{nombre}: spacing no válido {spacing}")
    if len(origin) != 3 or any(not isfinite(v) for v in origin):
        raise ValueError(f"{nombre}: origen no válido {origin}")
    if len(direction) != 9 or any(not isfinite(v) for v in direction):
        raise ValueError(f"{nombre}: matriz de dirección no válida {direction}")

    a, b, c, d, e, f, g, h, i = direction
    determinante = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinante) < 1e-10:
        raise ValueError(f"{nombre}: matriz de dirección singular")


def geometrias_coinciden(
    imagen_a: Any,
    imagen_b: Any,
    tolerancia: float = 1e-5,
) -> bool:
    """Comprueba tamaño, spacing, origen y dirección con tolerancia numérica."""

    ga = describir_geometria(imagen_a)
    gb = describir_geometria(imagen_b)

    if ga["size"] != gb["size"]:
        return False

    for clave in ("spacing", "origin", "direction"):
        if len(ga[clave]) != len(gb[clave]):
            return False
        if not all(
            isclose(a, b, rel_tol=tolerancia, abs_tol=tolerancia)
            for a, b in zip(ga[clave], gb[clave])
        ):
            return False

    return True


def diferencias_geometria(
    imagen_a: Any,
    imagen_b: Any,
    tolerancia: float = 1e-5,
) -> list[str]:
    """Describe las diferencias geométricas encontradas entre dos imágenes."""

    ga = describir_geometria(imagen_a)
    gb = describir_geometria(imagen_b)
    diferencias: list[str] = []

    if ga["size"] != gb["size"]:
        diferencias.append(f"size: {ga['size']} != {gb['size']}")

    for clave in ("spacing", "origin", "direction"):
        iguales = len(ga[clave]) == len(gb[clave]) and all(
            isclose(a, b, rel_tol=tolerancia, abs_tol=tolerancia)
            for a, b in zip(ga[clave], gb[clave])
        )
        if not iguales:
            diferencias.append(f"{clave}: {ga[clave]} != {gb[clave]}")

    return diferencias
