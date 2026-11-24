"""Paso 1: conversión automática DICOM -> NIfTI.

Para cada carpeta ``PACIENTES/XXXX/DICOM``:

1. Clasifica todas las cabeceras DICOM con las mismas reglas que el inventario.
2. Agrupa los cortes CT por ``SeriesInstanceUID``.
3. Excluye localizadores y capturas secundarias.
4. Prioriza la mejor serie CT axial y la ordena por posición física.
5. Selecciona la serie CBCT más plausible, tanto multiframe como multiarchivo.
6. Genera ``PACIENTES_NIFTI/XXXX/XXXX_ct.nii.gz`` y
   ``XXXX_cbct.nii.gz``.

El CT se guarda COMPLETO: no se recorta aquí. El recorte a la zona cubierta por
el CBCT se hace al final del paso 2, cuando ya se sabe dónde cae el CBCT.

Informes (acumulativos, ver ``informes.py``) en ``PACIENTES_NIFTI``:
``informe_conversion.csv``, ``informe_volumenes.csv`` e
``informe_volumenes.txt``.

No contiene listas manuales de pacientes; las carpetas de cuatro dígitos se
detectan en cada ejecución.

Siguiente paso: ``paso2_registro.py``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from escritura_segura import escribir_grupo_atomico
from imagen_utils import validar_geometria_3d
from informes import guardar_informes_paso1
from dicom_utils import (
    clasificar_dataset,
    clave_orden_corte,
    entero_tag,
    es_corte_axial,
    identificador_serie,
    normal_plano_corte,
    posicion_escalar_corte,
    puntuacion_cbct,
    texto_tag,
)


TAGS_LECTURA = [
    "Modality",
    "ImageType",
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SeriesInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "NumberOfFrames",
    "Rows",
    "Columns",
]


@dataclass
class EntradaDicom:
    ruta: Path
    dataset: Any
    tipo: str


@dataclass
class SeleccionDicom:
    rutas_ct: list[Path]
    uid_ct: str
    descripcion_ct: str
    rutas_cbct: list[Path]
    uid_cbct: str
    descripcion_cbct: str
    errores_lectura: int
    errores_detalle: list[str]
    conteos_tipo: dict[str, int]
    series_ct: dict[str, int]
    series_cbct: dict[str, int]
    candidatos_cbct: int


def leer_cabecera(filepath: Path) -> Any:
    """Lee metadatos sin cargar Pixel Data, con fallback sin preámbulo."""

    try:
        return pydicom.dcmread(
            str(filepath),
            stop_before_pixels=True,
            force=False,
            specific_tags=TAGS_LECTURA,
        )
    except InvalidDicomError:
        return pydicom.dcmread(
            str(filepath),
            stop_before_pixels=True,
            force=True,
            specific_tags=TAGS_LECTURA,
        )


def _deduplicar_y_ordenar(entradas: list[EntradaDicom]) -> list[EntradaDicom]:
    """Elimina copias del mismo SOPInstanceUID y ordena por posición física."""

    unicas: list[EntradaDicom] = []
    vistos: set[str] = set()

    for entrada in entradas:
        sop_uid = texto_tag(entrada.dataset, "SOPInstanceUID").strip()
        clave = sop_uid or str(entrada.ruta.resolve())
        if clave in vistos:
            continue
        vistos.add(clave)
        unicas.append(entrada)

    return sorted(
        unicas,
        key=lambda entrada: clave_orden_corte(entrada.dataset, entrada.ruta),
    )


def _validar_coherencia_multifichero(
    rutas: list[Path],
    nombre: str,
) -> None:
    """Detecta mezclas de geometrías y posiciones repetidas antes de apilar."""

    if len(rutas) < 2:
        return

    datasets = [leer_cabecera(ruta) for ruta in rutas]
    dimensiones = {
        (
            entero_tag(dataset, "Rows", 0),
            entero_tag(dataset, "Columns", 0),
        )
        for dataset in datasets
        if entero_tag(dataset, "Rows", 0) > 0
        and entero_tag(dataset, "Columns", 0) > 0
    }
    if len(dimensiones) > 1:
        raise ValueError(
            f"{nombre}: la serie mezcla matrices distintas: {sorted(dimensiones)}"
        )

    normales = [normal_plano_corte(dataset) for dataset in datasets]
    normales_validas = [normal for normal in normales if normal is not None]
    if len(normales_validas) >= 2:
        referencia = normales_validas[0]
        for normal in normales_validas[1:]:
            producto = sum(a * b for a, b in zip(referencia, normal))
            if producto < 0.999:
                raise ValueError(
                    f"{nombre}: la serie mezcla orientaciones de corte incompatibles"
                )

    posiciones = [posicion_escalar_corte(dataset) for dataset in datasets]
    if all(posicion is not None for posicion in posiciones):
        ordenadas = sorted(float(posicion) for posicion in posiciones if posicion is not None)
        diferencias = [
            siguiente - actual
            for actual, siguiente in zip(ordenadas, ordenadas[1:])
        ]
        if any(diferencia <= 1e-4 for diferencia in diferencias):
            raise ValueError(
                f"{nombre}: hay cortes duplicados o posiciones físicas repetidas"
            )
        if len(diferencias) >= 2:
            spacing_mediano = float(median(diferencias))
            desviacion_maxima = max(
                abs(diferencia - spacing_mediano) for diferencia in diferencias
            )
            tolerancia = max(0.1, 0.10 * spacing_mediano)
            if desviacion_maxima > tolerancia:
                print(
                    f"    ⚠ {nombre}: separación entre cortes no uniforme "
                    f"(mediana={spacing_mediano:.4g} mm, "
                    f"desviación máxima={desviacion_maxima:.4g} mm)."
                )
    else:
        print(
            f"    ⚠ {nombre}: faltan posiciones físicas en parte de la serie; "
            "se utilizará InstanceNumber/nombre como fallback."
        )


def _axialidad_serie(entradas: list[EntradaDicom]) -> tuple[int, float, str]:
    """Resume la orientación de una serie usando IOP y, si falta, ImageType."""

    valores_geometricos = [es_corte_axial(entrada.dataset) for entrada in entradas]
    conocidos = [valor for valor in valores_geometricos if valor is not None]

    if conocidos:
        proporcion = sum(bool(valor) for valor in conocidos) / len(conocidos)
        fuente = "IOP"
    else:
        marcas = [
            "AXIAL" in texto_tag(entrada.dataset, "ImageType").upper()
            for entrada in entradas
        ]
        proporcion = sum(marcas) / len(marcas) if marcas else 0.0
        fuente = "ImageType" if any(marcas) else "desconocida"

    es_serie_axial = int(proporcion >= 0.80)
    return es_serie_axial, proporcion, fuente


def _calidad_serie_ct(
    entradas: list[EntradaDicom],
) -> tuple[int, int, int, int, int]:
    """Puntúa CT priorizando orientación axial, cortes y resolución."""

    es_axial, proporcion_axial, _ = _axialidad_serie(entradas)
    proporcion_original = sum(
        "ORIGINAL" in texto_tag(entrada.dataset, "ImageType").upper()
        for entrada in entradas
    ) / len(entradas)
    area_maxima = max(
        (
            entero_tag(entrada.dataset, "Rows", 0)
            * entero_tag(entrada.dataset, "Columns", 0)
            for entrada in entradas
        ),
        default=0,
    )
    return (
        es_axial,
        round(proporcion_axial * 1000),
        len(entradas),
        round(proporcion_original * 1000),
        area_maxima,
    )


def _calidad_serie_cbct(entradas: list[EntradaDicom]) -> tuple[int, ...]:
    """Puntúa una serie CBCT, incluida la variante de varios archivos."""

    puntuaciones = [puntuacion_cbct(e.dataset, e.ruta) for e in entradas]
    maximos = tuple(max(p[indice] for p in puntuaciones) for indice in range(4))
    total_frames = sum(max(entero_tag(e.dataset, "NumberOfFrames", 1), 1) for e in entradas)
    tamano_total = 0
    for entrada in entradas:
        try:
            tamano_total += entrada.ruta.stat().st_size
        except OSError:
            pass
    return (*maximos, total_frames, len(entradas), tamano_total)


def clasificar_archivos_dicom(dicom_dir: Path) -> SeleccionDicom:
    """Clasifica, agrupa y selecciona la serie CT principal y el CBCT."""

    archivos = sorted(
        (ruta for ruta in dicom_dir.rglob("*") if ruta.is_file()),
        key=lambda ruta: ruta.as_posix().lower(),
    )
    print(f"    Leyendo metadatos de {len(archivos)} archivos...")

    series_ct: dict[str, list[EntradaDicom]] = defaultdict(list)
    series_cbct: dict[str, list[EntradaDicom]] = defaultdict(list)
    conteos: Counter[str] = Counter()
    errores = 0
    errores_detalle: list[str] = []

    for filepath in archivos:
        try:
            relativa = filepath.relative_to(dicom_dir).as_posix()
        except ValueError:
            relativa = str(filepath)

        try:
            ds = leer_cabecera(filepath)
            if not (getattr(ds, "SOPClassUID", "") or getattr(ds, "Modality", "")):
                errores += 1
                errores_detalle.append(
                    f"{relativa}: sin SOPClassUID ni Modality identificables"
                )
                continue
            tipo = clasificar_dataset(ds, filepath.name)
        except Exception as exc:
            errores += 1
            errores_detalle.append(
                f"{relativa}: {type(exc).__name__}: {exc}"
            )
            continue

        conteos[tipo] += 1
        entrada = EntradaDicom(filepath, ds, tipo)
        uid = identificador_serie(ds, filepath)

        if tipo == "CT_REAL":
            series_ct[uid].append(entrada)
        elif tipo == "CBCT_VOLUME":
            series_cbct[uid].append(entrada)

    series_ct_ordenadas = [
        (uid, _deduplicar_y_ordenar(entradas))
        for uid, entradas in series_ct.items()
    ]
    series_ct_ordenadas.sort(
        key=lambda item: _calidad_serie_ct(item[1]),
        reverse=True,
    )

    print("\n    Series CT válidas encontradas:")
    if not series_ct_ordenadas:
        print("      (ninguna)")
    else:
        for indice, (_, entradas) in enumerate(series_ct_ordenadas, start=1):
            ds0 = entradas[0].dataset
            descripcion = (
                texto_tag(ds0, "SeriesDescription")
                or texto_tag(ds0, "ProtocolName")
            )
            es_axial, proporcion_axial, fuente = _axialidad_serie(entradas)
            orientacion = (
                f"axial {proporcion_axial:.0%} ({fuente})"
                if fuente != "desconocida"
                else "orientación desconocida"
            )
            marcador = "  ← seleccionada" if indice == 1 else ""
            print(
                f"      {len(entradas):>5} cortes | "
                f"serie {texto_tag(ds0, 'SeriesNumber') or '?':>4} | "
                f"{orientacion}{'' if es_axial else ' ⚠'} | "
                f"{descripcion or 'sin descripción'}{marcador}"
            )

    if series_ct_ordenadas:
        uid_ct, entradas_ct = series_ct_ordenadas[0]
        rutas_ct = [entrada.ruta for entrada in entradas_ct]
        descripcion_ct = (
            texto_tag(entradas_ct[0].dataset, "SeriesDescription")
            or texto_tag(entradas_ct[0].dataset, "ProtocolName")
        )
        es_axial_ct, _, fuente_axial_ct = _axialidad_serie(entradas_ct)
        if fuente_axial_ct == "desconocida":
            print(
                "    ⚠ No se pudo verificar la orientación de la serie CT; "
                "revise el NIfTI generado."
            )
        elif not es_axial_ct:
            print(
                "    ⚠ No se encontró una serie predominantemente axial; "
                "se usará la mejor serie CT disponible y debe revisarse."
            )
    else:
        uid_ct, rutas_ct, descripcion_ct = "", [], ""

    series_cbct_ordenadas = [
        (uid, _deduplicar_y_ordenar(entradas))
        for uid, entradas in series_cbct.items()
    ]
    series_cbct_ordenadas.sort(
        key=lambda item: _calidad_serie_cbct(item[1]),
        reverse=True,
    )

    print("\n    Series/candidatos CBCT encontrados:")
    if not series_cbct_ordenadas:
        print("      (ninguno)")
    else:
        for indice, (_, entradas) in enumerate(series_cbct_ordenadas, start=1):
            ds0 = entradas[0].dataset
            descripcion = (
                texto_tag(ds0, "SeriesDescription")
                or texto_tag(ds0, "ProtocolName")
            )
            frames_total = sum(
                max(entero_tag(entrada.dataset, "NumberOfFrames", 1), 1)
                for entrada in entradas
            )
            marcador = "  ← seleccionado" if indice == 1 else ""
            print(
                f"      {len(entradas):>4} archivo(s) | "
                f"{frames_total:>5} frame(s) | "
                f"serie {texto_tag(ds0, 'SeriesNumber') or '?':>4} | "
                f"{descripcion or 'sin descripción'}{marcador}"
            )

    if series_cbct_ordenadas:
        uid_cbct, entradas_cbct = series_cbct_ordenadas[0]
        rutas_cbct = [entrada.ruta for entrada in entradas_cbct]
        descripcion_cbct = (
            texto_tag(entradas_cbct[0].dataset, "SeriesDescription")
            or texto_tag(entradas_cbct[0].dataset, "ProtocolName")
        )
        if len(series_cbct_ordenadas) > 1:
            print(
                f"    ⚠ Hay {len(series_cbct_ordenadas)} series candidatas CBCT; "
                "se ha seleccionado la mejor por SOP Class, descripción, "
                "modalidad, frames y tamaño."
            )
        puntuacion_seleccionada = puntuacion_cbct(
            entradas_cbct[0].dataset,
            entradas_cbct[0].ruta,
        )
        if not puntuacion_seleccionada[0] and not puntuacion_seleccionada[1]:
            print(
                "    ⚠ El CBCT se ha identificado por modalidad/estructura "
                "multiframe, sin SOP Class 3D ni texto CBCT/XperCT; compruebe "
                "visualmente la selección."
            )
    else:
        uid_cbct, rutas_cbct, descripcion_cbct = "", [], ""

    return SeleccionDicom(
        rutas_ct=rutas_ct,
        uid_ct=uid_ct,
        descripcion_ct=descripcion_ct,
        rutas_cbct=rutas_cbct,
        uid_cbct=uid_cbct,
        descripcion_cbct=descripcion_cbct,
        errores_lectura=errores,
        errores_detalle=errores_detalle,
        conteos_tipo=dict(conteos),
        series_ct={uid: len(entradas) for uid, entradas in series_ct_ordenadas},
        series_cbct={uid: len(entradas) for uid, entradas in series_cbct_ordenadas},
        candidatos_cbct=sum(len(entradas) for entradas in series_cbct.values()),
    )


def leer_volumen_ct(rutas_ct: list[Path]) -> sitk.Image:
    """Lee una única serie CT ya filtrada y ordenada."""

    if not rutas_ct:
        raise ValueError("La lista de cortes CT está vacía")

    _validar_coherencia_multifichero(rutas_ct, "CT")
    print(f"    Convirtiendo {len(rutas_ct)} cortes CT a NIfTI...")
    ds_primero = leer_cabecera(rutas_ct[0])
    n_frames = entero_tag(ds_primero, "NumberOfFrames", 1)

    if len(rutas_ct) == 1 and n_frames > 1:
        volumen_ct = sitk.ReadImage(str(rutas_ct[0]))
    else:
        lector = sitk.ImageSeriesReader()
        lector.SetFileNames([str(ruta) for ruta in rutas_ct])
        lector.MetaDataDictionaryArrayUpdateOn()
        lector.LoadPrivateTagsOn()
        volumen_ct = lector.Execute()

    if volumen_ct.GetDimension() != 3:
        raise RuntimeError(
            f"El CT leído tiene {volumen_ct.GetDimension()} dimensiones; se esperaban 3"
        )

    return volumen_ct


def leer_volumen_cbct(rutas_cbct: list[Path]) -> sitk.Image:
    """Lee un CBCT multiframe o una serie CBCT de varios DICOM."""

    if not rutas_cbct:
        raise ValueError("La lista de archivos CBCT está vacía")

    _validar_coherencia_multifichero(rutas_cbct, "CBCT")
    if len(rutas_cbct) == 1:
        print(f"    Convirtiendo CBCT desde {rutas_cbct[0].name}...")
        lector = sitk.ImageFileReader()
        lector.SetFileName(str(rutas_cbct[0]))
        lector.LoadPrivateTagsOn()
        volumen_cbct = lector.Execute()
    else:
        print(f"    Convirtiendo serie CBCT de {len(rutas_cbct)} archivos...")
        lector_serie = sitk.ImageSeriesReader()
        lector_serie.SetFileNames([str(ruta) for ruta in rutas_cbct])
        lector_serie.MetaDataDictionaryArrayUpdateOn()
        lector_serie.LoadPrivateTagsOn()
        volumen_cbct = lector_serie.Execute()

    if volumen_cbct.GetDimension() != 3:
        raise RuntimeError(
            f"El CBCT leído tiene {volumen_cbct.GetDimension()} dimensiones; "
            "se esperaba un volumen 3D. Revise la serie seleccionada."
        )

    return volumen_cbct


def info_volumen(patient_id: str, nombre: str, imagen: sitk.Image) -> dict[str, object]:
    validar_geometria_3d(imagen, nombre)
    size = tuple(int(v) for v in imagen.GetSize())
    spacing = tuple(float(v) for v in imagen.GetSpacing())
    origin = tuple(float(v) for v in imagen.GetOrigin())

    print(f"\n    [{nombre}]")
    print(f"      Dimensiones (X, Y, Z): {size}")
    print(f"      Espaciado (mm):        {tuple(round(v, 5) for v in spacing)}")
    print(f"      Origen (mm):           {tuple(round(v, 3) for v in origin)}")

    return {
        "patient_id": patient_id,
        "volumen": nombre,
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "spacing_x": spacing[0],
        "spacing_y": spacing[1],
        "spacing_z": spacing[2],
        "origin_x": origin[0],
        "origin_y": origin[1],
        "origin_z": origin[2],
    }


def procesar_paciente(
    patient_id: str,
    pacientes_dir: Path,
    output_root: Path,
    sobrescribir: bool,
    min_cortes_ct: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    print(f"\n{'=' * 72}")
    print(f"Procesando paciente: {patient_id}")
    print(f"{'=' * 72}")

    dicom_dir = pacientes_dir / patient_id / "DICOM"
    output_dir = output_root / patient_id
    ruta_ct = output_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct = output_dir / f"{patient_id}_cbct.nii.gz"

    resumen: dict[str, object] = {
        "patient_id": patient_id,
        "estado": "ERROR",
        "serie_ct_uid": "",
        "serie_ct_descripcion": "",
        "n_cortes_ct": 0,
        "serie_cbct_uid": "",
        "serie_cbct_descripcion": "",
        "n_archivos_cbct": 0,
        "cbct_archivo": "",
        "errores_lectura": 0,
        "detalle_errores_lectura": "",
        "mensaje": "",
    }
    infos: list[dict[str, object]] = []

    if not dicom_dir.is_dir():
        resumen["mensaje"] = f"No existe {dicom_dir}"
        print(f"  ✗ {resumen['mensaje']}")
        return resumen, infos

    if ruta_ct.exists() and ruta_cbct.exists() and not sobrescribir:
        print("  ↷ Las dos salidas ya existen; se omite (usa --sobrescribir para regenerar).")
        try:
            infos.append(info_volumen(patient_id, "CT", sitk.ReadImage(str(ruta_ct))))
            infos.append(info_volumen(patient_id, "CBCT", sitk.ReadImage(str(ruta_cbct))))
        except Exception as exc:
            resumen["mensaje"] = f"No se pudieron validar salidas existentes: {exc}"
            return resumen, infos
        resumen.update(estado="OMITIDO", mensaje="Salidas existentes")
        return resumen, infos

    seleccion = clasificar_archivos_dicom(dicom_dir)
    resumen.update(
        serie_ct_uid=seleccion.uid_ct,
        serie_ct_descripcion=seleccion.descripcion_ct,
        n_cortes_ct=len(seleccion.rutas_ct),
        serie_cbct_uid=seleccion.uid_cbct,
        serie_cbct_descripcion=seleccion.descripcion_cbct,
        n_archivos_cbct=len(seleccion.rutas_cbct),
        cbct_archivo=seleccion.rutas_cbct[0].name if seleccion.rutas_cbct else "",
        errores_lectura=seleccion.errores_lectura,
        detalle_errores_lectura=" | ".join(seleccion.errores_detalle),
    )

    print("\n  Clasificación:")
    for tipo, cantidad in sorted(seleccion.conteos_tipo.items()):
        print(f"    {tipo:<14}: {cantidad}")
    if seleccion.errores_lectura:
        print(f"    ERROR_LECTURA : {seleccion.errores_lectura}")
        for detalle in seleccion.errores_detalle[:5]:
            print(f"      - {detalle}")
        if len(seleccion.errores_detalle) > 5:
            print(
                f"      ... y {len(seleccion.errores_detalle) - 5} errores más "
                "en el informe CSV"
            )

    if len(seleccion.rutas_ct) < min_cortes_ct:
        resumen["mensaje"] = (
            f"La mejor serie CT solo tiene {len(seleccion.rutas_ct)} cortes "
            f"(mínimo requerido: {min_cortes_ct})"
        )
        print(f"  ✗ {resumen['mensaje']}")
        return resumen, infos

    if not seleccion.rutas_cbct:
        resumen["mensaje"] = "No se encontró un volumen o una serie CBCT"
        print(f"  ✗ {resumen['mensaje']}")
        return resumen, infos

    try:
        escrituras = []
        if sobrescribir or not ruta_ct.exists():
            imagen_ct = leer_volumen_ct(seleccion.rutas_ct)
            escrituras.append(
                (
                    ruta_ct,
                    lambda temporal, imagen=imagen_ct: sitk.WriteImage(
                        imagen,
                        str(temporal),
                        True,
                    ),
                )
            )
        else:
            print(f"  ↷ CT ya existente: {ruta_ct.name}")
            imagen_ct = sitk.ReadImage(str(ruta_ct))
        infos.append(info_volumen(patient_id, "CT", imagen_ct))

        if sobrescribir or not ruta_cbct.exists():
            imagen_cbct = leer_volumen_cbct(seleccion.rutas_cbct)
            escrituras.append(
                (
                    ruta_cbct,
                    lambda temporal, imagen=imagen_cbct: sitk.WriteImage(
                        imagen,
                        str(temporal),
                        True,
                    ),
                )
            )
        else:
            print(f"  ↷ CBCT ya existente: {ruta_cbct.name}")
            imagen_cbct = sitk.ReadImage(str(ruta_cbct))
        infos.append(info_volumen(patient_id, "CBCT", imagen_cbct))

        escribir_grupo_atomico(escrituras)
        for destino, _ in escrituras:
            print(f"    ✓ Guardado: {destino.name}")
    except Exception as exc:
        resumen["mensaje"] = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ Error durante la conversión: {resumen['mensaje']}")
        return resumen, infos

    resumen.update(estado="OK", mensaje="Conversión completada")
    print(f"\n  ✓ Paciente {patient_id} completado.")
    return resumen, infos


def guardar_informes(
    output_root: Path,
    resumenes: list[dict[str, object]],
    volumenes: list[dict[str, object]],
) -> None:
    # Los informes se acumulan: solo se actualizan las filas de los
    # pacientes procesados en esta ejecución (ver informes.py).
    guardar_informes_paso1(output_root, resumenes, volumenes)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paso 1: DICOM a NIfTI")
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional de IDs. Sin él se detectan todos los pacientes.",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Regenera los NIfTI aunque ya existan.",
    )
    parser.add_argument(
        "--min-cortes-ct",
        type=int,
        default=2,
        help="Número mínimo de cortes para aceptar una serie CT (default: 2).",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    if args.min_cortes_ct < 1:
        print("ERROR: --min-cortes-ct debe ser al menos 1")
        return 2

    rutas = obtener_rutas(args.project_dir)
    try:
        pacientes = detectar_pacientes(rutas.pacientes, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not pacientes:
        print(f"ERROR: no hay pacientes con formato XXXX en {rutas.pacientes}")
        return 2

    rutas.nifti.mkdir(parents=True, exist_ok=True)
    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")
    print(f"Entrada: {rutas.pacientes}")
    print(f"Salida : {rutas.nifti}")

    resumenes: list[dict[str, object]] = []
    volumenes: list[dict[str, object]] = []
    for patient_id in pacientes:
        resumen, infos = procesar_paciente(
            patient_id=patient_id,
            pacientes_dir=rutas.pacientes,
            output_root=rutas.nifti,
            sobrescribir=args.sobrescribir,
            min_cortes_ct=args.min_cortes_ct,
        )
        resumenes.append(resumen)
        volumenes.extend(infos)

    guardar_informes(rutas.nifti, resumenes, volumenes)

    errores = [r for r in resumenes if r["estado"] == "ERROR"]
    print(f"\n{'=' * 72}")
    print(f"Paso 1 terminado: {len(resumenes) - len(errores)}/{len(resumenes)} sin error.")
    print(f"Informes: {rutas.nifti / 'informe_conversion.csv'}")
    if errores:
        print("Pacientes con error:")
        for fila in errores:
            print(f"  - {fila['patient_id']}: {fila['mensaje']}")
    print(f"{'=' * 72}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
