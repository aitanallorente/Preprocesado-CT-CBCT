"""Informes CSV acumulativos, comunes a todos los pasos.

Cada paso escribe un informe con una o varias filas por paciente. Al guardar,
se lee el CSV existente, se sustituyen únicamente las filas de los pacientes
procesados en esta ejecución y se conservan las del resto. Así, ejecutar un
paso solo para algunos pacientes (``--pacientes``) no borra la información de
los demás. No depende de ningún número fijo de pacientes.

Una fila nueva con estado OMITIDO (salidas ya existentes, no regeneradas) nunca
sustituye a una fila previa con resultados reales.

* ``guardar_informes_paso1``: informes del paso 1.
* ``guardar_informe_acumulado``: pasos 2, 2c, 3, 4 y validación.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Iterable

Fila = dict[str, object]


def leer_csv(ruta: Path) -> list[Fila]:
    if not ruta.is_file():
        return []
    with ruta.open("r", newline="", encoding="utf-8-sig") as fichero:
        return [dict(fila) for fila in csv.DictReader(fichero)]


def _escribir_csv(ruta: Path, filas: Iterable[Fila], campos: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("w", newline="", encoding="utf-8-sig") as fichero:
        writer = csv.DictWriter(
            fichero, fieldnames=campos, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(filas)
    os.replace(temporal, ruta)  # escritura atómica


def fusionar_csv(
    ruta: Path,
    filas_nuevas: list[Fila],
    campos: list[str],
    clave_orden: tuple[str, ...] = ("patient_id",),
    conservar_existente: Callable[[Fila, Fila], bool] | None = None,
) -> list[Fila]:
    """Fusiona ``filas_nuevas`` con el CSV existente, por ``patient_id``.

    - Los pacientes que aparecen en ``filas_nuevas`` sustituyen a sus filas
      antiguas (todas las que tengan ese patient_id).
    - Los pacientes que no aparecen se conservan tal cual.
    - ``conservar_existente(fila_vieja, fila_nueva)``: si devuelve True para un
      paciente, se mantienen sus filas antiguas (útil para no pisar datos
      buenos con una fila "OMITIDO" menos informativa).
    Devuelve la lista final ordenada y la escribe en ``ruta``.
    """

    existentes = leer_csv(ruta)
    ids_nuevos = {str(f["patient_id"]) for f in filas_nuevas}

    viejas_por_id: dict[str, list[Fila]] = {}
    for fila in existentes:
        viejas_por_id.setdefault(str(fila.get("patient_id", "")), []).append(fila)

    # Pacientes cuya fila antigua se conserva en lugar de la nueva.
    conservados: set[str] = set()
    if conservar_existente is not None:
        for pid in ids_nuevos:
            viejas = viejas_por_id.get(pid)
            nuevas = [f for f in filas_nuevas if str(f["patient_id"]) == pid]
            if viejas and nuevas and conservar_existente(viejas[0], nuevas[0]):
                conservados.add(pid)

    finales: list[Fila] = [
        f for f in existentes if str(f.get("patient_id", "")) not in ids_nuevos
        or str(f.get("patient_id", "")) in conservados
    ]
    finales += [f for f in filas_nuevas if str(f["patient_id"]) not in conservados]
    finales.sort(key=lambda f: tuple(str(f.get(c, "")) for c in clave_orden))

    _escribir_csv(ruta, finales, campos)
    return finales


# --------------------------------------------------------------------------
# Paso 1: informes de conversión DICOM -> NIfTI
# --------------------------------------------------------------------------

CAMPOS_RESUMEN_PASO1 = [
    "patient_id", "estado", "serie_ct_uid", "serie_ct_descripcion",
    "n_cortes_ct", "serie_cbct_uid", "serie_cbct_descripcion",
    "n_archivos_cbct", "cbct_archivo", "errores_lectura",
    "detalle_errores_lectura", "mensaje",
]

CAMPOS_VOLUMEN_PASO1 = [
    "patient_id", "volumen", "size_x", "size_y", "size_z",
    "spacing_x", "spacing_y", "spacing_z", "origin_x", "origin_y", "origin_z",
]


def _no_pisar_datos_buenos(vieja: Fila, nueva: Fila) -> bool:
    """Una ejecución 'OMITIDO' no trae datos de serie: no sustituye a un 'OK'."""
    return str(nueva.get("estado")) == "OMITIDO" and str(vieja.get("estado")) == "OK"


def guardar_informes_paso1(
    output_root: Path,
    resumenes: list[Fila],
    volumenes: list[Fila],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    resumen_final = fusionar_csv(
        output_root / "informe_conversion.csv",
        resumenes,
        CAMPOS_RESUMEN_PASO1,
        conservar_existente=_no_pisar_datos_buenos,
    )

    ids_conservados = {
        str(f["patient_id"])
        for f in resumenes
        if str(f.get("estado")) == "OMITIDO"
        and any(
            str(v["patient_id"]) == str(f["patient_id"]) and str(v.get("estado")) == "OK"
            for v in resumen_final
        )
    }
    volumenes_a_fusionar = [
        v for v in volumenes if str(v["patient_id"]) not in ids_conservados
    ]
    # Si todos los pacientes de esta ejecución se conservan, no hay nada que sustituir.
    if volumenes_a_fusionar or not ids_conservados:
        vol_final = fusionar_csv(
            output_root / "informe_volumenes.csv",
            volumenes_a_fusionar,
            CAMPOS_VOLUMEN_PASO1,
            clave_orden=("patient_id", "volumen"),
        )
    else:
        vol_final = leer_csv(output_root / "informe_volumenes.csv")

    ruta_txt = output_root / "informe_volumenes.txt"
    with ruta_txt.open("w", encoding="utf-8") as fichero:
        fichero.write("INFORME DE VOLÚMENES — Paso 1: DICOM -> NIfTI\n")
        fichero.write("=" * 88 + "\n")
        for fila in vol_final:
            fichero.write(
                f"{fila['patient_id']} {str(fila['volumen']):<4}  "
                f"size=({fila['size_x']}, {fila['size_y']}, {fila['size_z']})  "
                f"spacing=({float(fila['spacing_x']):.6g}, "
                f"{float(fila['spacing_y']):.6g}, "
                f"{float(fila['spacing_z']):.6g})\n"
            )


# --------------------------------------------------------------------------
# Función genérica para el resto de pasos (paso 2, 3, 4, validación...)
# --------------------------------------------------------------------------

def _no_pisar_con_omitido(vieja: Fila, nueva: Fila) -> bool:
    """Una fila nueva 'OMITIDO' no sustituye a una fila con resultados reales."""
    return str(nueva.get("estado")) == "OMITIDO" and str(vieja.get("estado")) in {
        "OK",
        "REVISAR",
    }


def guardar_informe_acumulado(
    ruta: Path,
    resultados: list[Fila],
    campos: list[str],
) -> Path:
    """Escribe ``ruta`` conservando las filas de los pacientes no procesados."""

    fusionar_csv(
        Path(ruta),
        resultados,
        campos,
        conservar_existente=_no_pisar_con_omitido,
    )
    return Path(ruta)
