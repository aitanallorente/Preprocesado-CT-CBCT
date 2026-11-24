"""Escritura transaccional de uno o varios archivos de salida.

Los volúmenes médicos son costosos de generar. Esta utilidad evita dejar un
archivo final truncado o un par CT/CBCT mezclado si falla una escritura: primero
se generan todos los temporales y solo después se sustituyen las salidas.

La usan los pasos 1, 2, 2c, 3 y 4 para escribir sus NIfTI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

Escritor = Callable[[Path], None]


def _ruta_auxiliar(destino: Path, etiqueta: str) -> Path:
    sufijos = "".join(destino.suffixes)
    base = destino.name[: -len(sufijos)] if sufijos else destino.name
    return destino.with_name(
        f".{base}.{uuid4().hex}.{etiqueta}{sufijos}"
    )


def _eliminar_si_existe(ruta: Path) -> None:
    try:
        ruta.unlink(missing_ok=True)
    except OSError:
        pass


def escribir_grupo_atomico(
    escrituras: Sequence[tuple[Path, Escritor]],
) -> None:
    """Genera y confirma un conjunto de archivos como una sola operación lógica.

    Cada ``escritor`` recibe una ruta temporal con la misma extensión que el
    destino, requisito importante para que SimpleITK seleccione el formato
    correcto. Si falla cualquier preparación, las salidas anteriores permanecen
    intactas. Durante la confirmación se usan copias de respaldo para poder
    revertir un fallo de sustitución.
    """

    if not escrituras:
        return

    destinos = [Path(destino) for destino, _ in escrituras]
    if len(set(destinos)) != len(destinos):
        raise ValueError("No puede haber destinos repetidos en una escritura atómica")

    temporales: dict[Path, Path] = {}
    respaldos: dict[Path, Path] = {}
    confirmados: list[Path] = []
    confirmacion_completa = False

    try:
        # Fase 1: generar todos los archivos sin tocar las salidas existentes.
        for destino, escritor in escrituras:
            destino = Path(destino)
            destino.parent.mkdir(parents=True, exist_ok=True)
            temporal = _ruta_auxiliar(destino, "tmp")
            temporales[destino] = temporal
            escritor(temporal)
            if not temporal.is_file() or temporal.stat().st_size == 0:
                raise RuntimeError(
                    "El escritor no generó un archivo temporal no vacío: "
                    f"{temporal}"
                )

        # Fase 2: apartar salidas anteriores y confirmar todos los temporales.
        for destino in destinos:
            if destino.exists():
                respaldo = _ruta_auxiliar(destino, "bak")
                os.replace(destino, respaldo)
                respaldos[destino] = respaldo

        for destino in destinos:
            os.replace(temporales[destino], destino)
            confirmados.append(destino)

        confirmacion_completa = True

    except Exception as error_original:
        # Retira cualquier salida nueva ya confirmada.
        for destino in reversed(confirmados):
            _eliminar_si_existe(destino)

        # Restaura las versiones anteriores que se hubieran apartado.
        errores_restauracion: list[str] = []
        for destino, respaldo in respaldos.items():
            if not respaldo.exists():
                continue
            try:
                os.replace(respaldo, destino)
            except OSError as exc:
                errores_restauracion.append(f"{destino}: {exc}")

        if errores_restauracion:
            raise RuntimeError(
                "Falló la escritura y no se pudieron restaurar todas las salidas: "
                + " | ".join(errores_restauracion)
            ) from error_original
        raise

    finally:
        for ruta in temporales.values():
            _eliminar_si_existe(ruta)
        if confirmacion_completa:
            for ruta in respaldos.values():
                _eliminar_si_existe(ruta)
