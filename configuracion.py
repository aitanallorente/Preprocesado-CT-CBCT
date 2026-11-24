"""Configuración común de rutas y detección automática de pacientes.

La estructura esperada es::

    CARPETA DEL PROYECTO/
    ├── Codigo_TFG_preprocesado/      este código (y la carpeta LANDMARKS)
    ├── PACIENTES/                    DICOM originales (entrada del paso 1)
    ├── PACIENTES_NIFTI/              salida del paso 1
    ├── PACIENTES_REGISTERED/         salida del paso 2; aquí se guardan también
    │                                 las carpetas "XXXX nuevo" del ajuste en SPM
    ├── PACIENTES_REGISTERED_FINAL/   salida del paso 2c (entrada del paso 3)
    ├── PACIENTES_NORMALIZED/         salida del paso 3
    └── PACIENTES_FINAL/              salida del paso 4 (dataset para el modelo)

Todos los scripts calculan la raíz del proyecto a partir de su propia ubicación,
por lo que se pueden ejecutar desde cualquier directorio de trabajo. No hay
listas de pacientes en el código: cada paso detecta las carpetas de cuatro
dígitos (0001, 0002...) que existan en su carpeta de entrada.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PATRON_PACIENTE = re.compile(r"^[0-9]{4}$")
VARIABLE_ENTORNO_PROYECTO = "TFG_PROJECT_DIR"


@dataclass(frozen=True)
class RutasProyecto:
    """Rutas normalizadas de todas las etapas del proyecto."""

    proyecto: Path
    codigo: Path
    pacientes: Path
    nifti: Path
    registered: Path
    normalized: Path
    final: Path


def obtener_rutas(proyecto_dir: str | Path | None = None) -> RutasProyecto:
    """Resuelve las rutas del proyecto.

    Prioridad:
    1. Argumento ``proyecto_dir``.
    2. Variable de entorno ``TFG_PROJECT_DIR``.
    3. Carpeta padre de ``Codigo_TFG_preprocesado``.
    """

    codigo = Path(__file__).resolve().parent

    if proyecto_dir is not None:
        proyecto = Path(proyecto_dir).expanduser().resolve()
    elif os.environ.get(VARIABLE_ENTORNO_PROYECTO):
        proyecto = Path(os.environ[VARIABLE_ENTORNO_PROYECTO]).expanduser().resolve()
    else:
        proyecto = codigo.parent

    return RutasProyecto(
        proyecto=proyecto,
        codigo=codigo,
        pacientes=proyecto / "PACIENTES",
        nifti=proyecto / "PACIENTES_NIFTI",
        registered=proyecto / "PACIENTES_REGISTERED",
        normalized=proyecto / "PACIENTES_NORMALIZED",
        final=proyecto / "PACIENTES_FINAL",
    )


def es_id_paciente(nombre: str) -> bool:
    """Indica si ``nombre`` cumple exactamente el formato de cuatro dígitos."""

    return bool(PATRON_PACIENTE.fullmatch(nombre))


def detectar_pacientes(
    carpeta_raiz: str | Path,
    solicitados: Sequence[str] | None = None,
) -> list[str]:
    """Detecta y ordena los pacientes presentes en una carpeta de etapa.

    Si ``solicitados`` es ``None``, devuelve automáticamente todas las carpetas
    cuyo nombre tiene exactamente cuatro dígitos. El filtro opcional sirve para
    ejecutar solo algunos pacientes desde la línea de comandos, sin editar el
    código ni mantener listas manuales dentro de los scripts.
    """

    raiz = Path(carpeta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {raiz}")

    disponibles = sorted(
        entrada.name
        for entrada in raiz.iterdir()
        if entrada.is_dir() and es_id_paciente(entrada.name)
    )

    if solicitados is None:
        return disponibles

    pedidos = list(dict.fromkeys(str(pid) for pid in solicitados))
    invalidos = [pid for pid in pedidos if not es_id_paciente(pid)]
    if invalidos:
        raise ValueError(
            "Identificadores de paciente no válidos (se requieren cuatro dígitos): "
            + ", ".join(invalidos)
        )

    faltantes = [pid for pid in pedidos if pid not in disponibles]
    if faltantes:
        raise FileNotFoundError(
            f"No se encuentran en {raiz}: " + ", ".join(faltantes)
        )

    return sorted(pedidos)


def crear_carpetas(rutas: Iterable[Path]) -> None:
    """Crea una colección de carpetas, incluidos sus directorios padre."""

    for ruta in rutas:
        Path(ruta).mkdir(parents=True, exist_ok=True)


def texto_pacientes(pacientes: Sequence[str]) -> str:
    """Devuelve una representación compacta para mensajes de consola."""

    return ", ".join(pacientes) if pacientes else "ninguno"
