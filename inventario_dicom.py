"""Inventario (paso previo, opcional): qué DICOM hay en cada paciente.

Sirve para auditar los datos antes de convertirlos; el paso 1 no depende de sus
salidas, pero ambos clasifican los archivos con las mismas reglas
(``dicom_utils.py``).

El script detecta automáticamente todos los pacientes con identificador de
cuatro dígitos dentro de ``PACIENTES`` y genera en la raíz del proyecto:

* ``inventario_completo.csv``: una fila por archivo examinado.
* ``resumen_pacientes.csv``: una fila por paciente.
* ``inventario_TFG.xlsx``: las dos tablas anteriores con formato y leyenda.

No se extraen tags directos como PatientName, PatientID o fechas. Se conserva
``SeriesInstanceUID`` únicamente como identificador técnico para agrupar y
auditar las series.

Flujo completo (ver README.md):
    inventario -> paso 1 -> paso 2 -> ajuste manual en SPM -> paso 2c
    -> paso 3 -> paso 4 -> validación
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from dicom_utils import clasificar_dataset, texto_valor


CAMPOS_DICOM = (
    "Modality",
    "ImageType",
    "SOPClassUID",
    "SeriesDescription",
    "ProtocolName",
    "SeriesInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "Rows",
    "Columns",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Manufacturer",
    "ManufacturerModelName",
    "StudyDescription",
    "NumberOfFrames",
)

TIPO_COLORES = {
    "CT_REAL": "D6F0D6",
    "CBCT_VOLUME": "D0E8F5",
    "LOCALIZER": "FFF2CC",
    "SECONDARY": "F5E6CC",
    "DICOMDIR": "ECECEC",
    "OTRO": "F5D0D0",
    "ERROR_LECTURA": "F4CCCC",
    "SIN_ARCHIVOS": "EADCF8",
}

TIPO_LABELS = {
    "CT_REAL": "Imagen CT utilizable",
    "CBCT_VOLUME": "CBCT / XperCT",
    "LOCALIZER": "Localizador / surview",
    "SECONDARY": "Captura secundaria",
    "DICOMDIR": "DICOMDIR (índice)",
    "OTRO": "Otro",
    "ERROR_LECTURA": "Error de lectura",
    "SIN_ARCHIVOS": "Carpeta sin archivos",
}

COLUMNAS_INVENTARIO = [
    "patient_id",
    "filename",
    "tipo",
    "Modality",
    "ImageType",
    "SOPClassUID_Name",
    "SeriesDescription",
    "ProtocolName",
    "SeriesNumber",
    "InstanceNumber",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "NumberOfFrames",
    "Manufacturer",
    "ManufacturerModelName",
    "StudyDescription",
    "SeriesInstanceUID",
    "filepath_relativa",
    "error",
]

COLUMNAS_RESUMEN = [
    "patient_id",
    "CT_REAL",
    "CBCT_VOLUME",
    "LOCALIZER",
    "SECONDARY",
    "DICOMDIR",
    "OTRO",
    "ERROR_LECTURA",
    "SIN_ARCHIVOS",
    "TOTAL",
    "series_CT",
    "series_CBCT",
    "CT_rows_cols",
    "CT_pixel_spacing_mm",
    "CT_slice_thickness_mm",
    "tiene_par_completo",
]


def _leer_dataset(filepath: Path) -> Any:
    """Lee solo la cabecera, con fallback para DICOM sin preámbulo."""

    try:
        return pydicom.dcmread(
            str(filepath),
            stop_before_pixels=True,
            force=False,
            specific_tags=list(CAMPOS_DICOM),
        )
    except InvalidDicomError:
        return pydicom.dcmread(
            str(filepath),
            stop_before_pixels=True,
            force=True,
            specific_tags=list(CAMPOS_DICOM),
        )


def leer_metadatos(filepath: Path, data_dir: Path) -> dict[str, Any]:
    """Extrae únicamente los metadatos técnicos necesarios para el inventario."""

    try:
        ruta_relativa = filepath.relative_to(data_dir).as_posix()
    except ValueError:
        ruta_relativa = str(filepath)

    resultado: dict[str, Any] = {
        "filepath_relativa": ruta_relativa,
        "filename": filepath.name,
        "error": "",
    }

    try:
        ds = _leer_dataset(filepath)
    except Exception as exc:  # pydicom puede lanzar excepciones específicas del codec/VR
        resultado["tipo"] = "ERROR_LECTURA"
        resultado["error"] = f"{type(exc).__name__}: {exc}"
        return resultado

    tiene_identidad_dicom = bool(
        getattr(ds, "SOPClassUID", "") or getattr(ds, "Modality", "")
    )
    if not tiene_identidad_dicom and filepath.name.upper() != "DICOMDIR":
        resultado["tipo"] = "ERROR_LECTURA"
        resultado["error"] = "El archivo no contiene metadatos DICOM identificables"
        return resultado

    for campo in CAMPOS_DICOM:
        resultado[campo] = texto_valor(getattr(ds, campo, ""))

    sop_uid = resultado.get("SOPClassUID", "")
    try:
        resultado["SOPClassUID_Name"] = UID(str(sop_uid)).name if sop_uid else ""
    except Exception:
        resultado["SOPClassUID_Name"] = str(sop_uid)

    resultado["tipo"] = clasificar_dataset(ds, filepath.name)
    return resultado


def archivos_del_paciente(patient_dir: Path) -> list[Path]:
    """Devuelve los ficheros de ``DICOM`` y el ``DICOMDIR`` del paciente."""

    archivos: list[Path] = []
    dicom_dir = patient_dir / "DICOM"

    if dicom_dir.is_dir():
        archivos.extend(f for f in dicom_dir.rglob("*") if f.is_file())
    else:
        print(f"  ⚠ {patient_dir.name}: no existe la carpeta DICOM")

    dicomdir = patient_dir / "DICOMDIR"
    if dicomdir.is_file():
        archivos.append(dicomdir)

    return sorted(set(archivos), key=lambda ruta: ruta.as_posix().lower())


def inventariar_paciente(patient_dir: Path, data_dir: Path) -> list[dict[str, Any]]:
    archivos = archivos_del_paciente(patient_dir)
    print(f"  Paciente {patient_dir.name}: {len(archivos)} archivos examinados")

    if not archivos:
        try:
            relativa = patient_dir.relative_to(data_dir).as_posix()
        except ValueError:
            relativa = str(patient_dir)
        return [
            {
                "patient_id": patient_dir.name,
                "filename": "",
                "tipo": "SIN_ARCHIVOS",
                "filepath_relativa": relativa,
                "error": "No se encontraron archivos en DICOM ni DICOMDIR",
            }
        ]

    registros: list[dict[str, Any]] = []
    for filepath in archivos:
        registro = leer_metadatos(filepath, data_dir)
        registro["patient_id"] = patient_dir.name
        registros.append(registro)
    return registros

def _serie_ct_principal(ct: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las filas de la serie CT que elegiría el paso 1.

    Mismo criterio que ``_calidad_serie_ct`` de paso1_dicom_a_nifti.py: serie
    axial, más cortes, mayor proporción ORIGINAL y mayor matriz. En los empates
    gana la serie cuyo primer archivo va antes por ruta (igual que el paso 1).
    """
    if ct.empty:
        return ct

    candidatas = []
    for uid, g in ct.groupby("SeriesInstanceUID", sort=False):
        tipo_imagen = g["ImageType"].fillna("").astype(str).str.upper()
        filas = pd.to_numeric(g["Rows"], errors="coerce").fillna(0)
        cols = pd.to_numeric(g["Columns"], errors="coerce").fillna(0)
        puntuacion = (
            int((tipo_imagen.str.contains("AXIAL").mean()) >= 0.80),
            round(tipo_imagen.str.contains("AXIAL").mean() * 1000),
            len(g),
            round(tipo_imagen.str.contains("ORIGINAL").mean() * 1000),
            int((filas * cols).max()),
        )
        primera_ruta = g["filepath_relativa"].astype(str).str.lower().min()
        candidatas.append((puntuacion, primera_ruta, uid))

    # Mayor puntuación primero; a igualdad, la de ruta menor.
    candidatas.sort(key=lambda c: c[1])
    candidatas.sort(key=lambda c: c[0], reverse=True)
    return ct[ct["SeriesInstanceUID"] == candidatas[0][2]]

def generar_resumen(df: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []

    for patient_id, grupo in df.groupby("patient_id", sort=True):
        conteos = grupo["tipo"].value_counts().to_dict()
        ct = grupo[grupo["tipo"] == "CT_REAL"]
        cbct = grupo[grupo["tipo"] == "CBCT_VOLUME"]

        serie_ct = _serie_ct_principal(ct)
        primera_ct = serie_ct.iloc[0] if not serie_ct.empty else None
        rows = primera_ct.get("Rows", "") if primera_ct is not None else ""
        cols = primera_ct.get("Columns", "") if primera_ct is not None else ""

        fila = {
            "patient_id": patient_id,
            "CT_REAL": int(conteos.get("CT_REAL", 0)),
            "CBCT_VOLUME": int(conteos.get("CBCT_VOLUME", 0)),
            "LOCALIZER": int(conteos.get("LOCALIZER", 0)),
            "SECONDARY": int(conteos.get("SECONDARY", 0)),
            "DICOMDIR": int(conteos.get("DICOMDIR", 0)),
            "OTRO": int(conteos.get("OTRO", 0)),
            "ERROR_LECTURA": int(conteos.get("ERROR_LECTURA", 0)),
            "SIN_ARCHIVOS": int(conteos.get("SIN_ARCHIVOS", 0)),
            "TOTAL": int((grupo["tipo"] != "SIN_ARCHIVOS").sum()),
            "series_CT": int(ct["SeriesInstanceUID"].replace("", pd.NA).nunique()),
            "series_CBCT": int(cbct["SeriesInstanceUID"].replace("", pd.NA).nunique()),
            "CT_rows_cols": f"{rows}x{cols}" if rows and cols else "",
            "CT_pixel_spacing_mm": (
                primera_ct.get("PixelSpacing", "") if primera_ct is not None else ""
            ),
            "CT_slice_thickness_mm": (
                primera_ct.get("SliceThickness", "") if primera_ct is not None else ""
            ),
            "tiene_par_completo": (
                "✓" if not ct.empty and not cbct.empty else "✗"
            ),
        }
        filas.append(fila)

    return pd.DataFrame(filas, columns=COLUMNAS_RESUMEN)


def _borde(color: str = "CCCCCC") -> Border:
    lado = Side(style="thin", color=color)
    return Border(left=lado, right=lado, top=lado, bottom=lado)


def _titulo(ws, fila: int, n_columnas: int, texto: str, color: str) -> None:
    ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=n_columnas)
    celda = ws.cell(fila, 1, texto)
    celda.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    celda.fill = PatternFill("solid", fgColor=color)
    celda.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[fila].height = 23


def _encabezados(ws, fila: int, columnas: list[str]) -> None:
    for indice, nombre in enumerate(columnas, start=1):
        celda = ws.cell(fila, indice, nombre)
        celda.font = Font(name="Arial", bold=True, size=10, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="2E75B6")
        celda.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        celda.border = _borde("1A5276")
    ws.row_dimensions[fila].height = 31


def _valor_excel(valor: Any) -> Any:
    if pd.isna(valor):
        return ""
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except Exception:
            pass
    return valor


def exportar_excel(df_resumen: pd.DataFrame, df_inventario: pd.DataFrame, ruta: Path) -> None:
    wb = Workbook()

    # Hoja resumen
    ws = wb.active
    ws.title = "Resumen pacientes"
    ws.freeze_panes = "A4"
    columnas = list(df_resumen.columns)
    _titulo(ws, 1, len(columnas), "INVENTARIO TFG — Resumen por paciente", "1F4E79")
    _titulo(ws, 2, len(columnas), f"Pacientes incluidos: {len(df_resumen)}", "2E4057")
    _encabezados(ws, 3, columnas)

    for fila_excel, fila in enumerate(df_resumen.itertuples(index=False, name=None), start=4):
        for col_excel, valor in enumerate(fila, start=1):
            celda = ws.cell(fila_excel, col_excel, _valor_excel(valor))
            celda.font = Font(name="Arial", size=10)
            celda.alignment = Alignment(horizontal="center", vertical="center")
            celda.border = _borde()
            fondo = "FFFFFF" if fila_excel % 2 == 0 else "EBF3FB"
            if columnas[col_excel - 1] == "tiene_par_completo":
                fondo = "C8F0C8" if valor == "✓" else "F5BABA"
                celda.font = Font(name="Arial", size=11, bold=True)
            celda.fill = PatternFill("solid", fgColor=fondo)

    anchos_resumen = {
        "patient_id": 12,
        "CT_REAL": 10,
        "CBCT_VOLUME": 12,
        "LOCALIZER": 12,
        "SECONDARY": 13,
        "DICOMDIR": 10,
        "OTRO": 8,
        "ERROR_LECTURA": 13,
        "SIN_ARCHIVOS": 13,
        "TOTAL": 10,
        "series_CT": 10,
        "series_CBCT": 12,
        "CT_rows_cols": 13,
        "CT_pixel_spacing_mm": 20,
        "CT_slice_thickness_mm": 20,
        "tiene_par_completo": 16,
    }
    for indice, columna in enumerate(columnas, start=1):
        ws.column_dimensions[get_column_letter(indice)].width = anchos_resumen.get(columna, 15)
    ws.auto_filter.ref = f"A3:{get_column_letter(len(columnas))}{3 + len(df_resumen)}"

    # Hoja inventario
    ws2 = wb.create_sheet("Inventario completo")
    ws2.freeze_panes = "A4"
    columnas_inv = list(df_inventario.columns)
    _titulo(ws2, 1, len(columnas_inv), "INVENTARIO TFG — Detalle por archivo", "1F4E79")
    _titulo(ws2, 2, len(columnas_inv), f"Archivos incluidos: {len(df_inventario)}", "2E4057")
    _encabezados(ws2, 3, columnas_inv)

    indice_tipo = columnas_inv.index("tipo") if "tipo" in columnas_inv else None
    for fila_excel, fila in enumerate(df_inventario.itertuples(index=False, name=None), start=4):
        tipo = fila[indice_tipo] if indice_tipo is not None else "OTRO"
        fondo = TIPO_COLORES.get(str(tipo), "FFFFFF")
        for col_excel, valor in enumerate(fila, start=1):
            columna = columnas_inv[col_excel - 1]
            valor_salida = TIPO_LABELS.get(str(valor), valor) if columna == "tipo" else valor
            celda = ws2.cell(fila_excel, col_excel, _valor_excel(valor_salida))
            celda.font = Font(name="Arial", size=9, bold=(columna == "tipo"))
            celda.alignment = Alignment(vertical="center", wrap_text=False)
            celda.border = _borde()
            celda.fill = PatternFill("solid", fgColor=fondo)

    anchos_inv = {
        "patient_id": 10,
        "filename": 18,
        "tipo": 22,
        "Modality": 10,
        "ImageType": 32,
        "SOPClassUID_Name": 34,
        "SeriesDescription": 28,
        "ProtocolName": 24,
        "SeriesNumber": 11,
        "InstanceNumber": 12,
        "Rows": 8,
        "Columns": 10,
        "PixelSpacing": 20,
        "SliceThickness": 16,
        "SpacingBetweenSlices": 20,
        "NumberOfFrames": 14,
        "Manufacturer": 16,
        "ManufacturerModelName": 22,
        "StudyDescription": 25,
        "SeriesInstanceUID": 28,
        "filepath_relativa": 55,
        "error": 45,
    }
    for indice, columna in enumerate(columnas_inv, start=1):
        ws2.column_dimensions[get_column_letter(indice)].width = anchos_inv.get(columna, 15)
    ws2.auto_filter.ref = f"A3:{get_column_letter(len(columnas_inv))}{3 + len(df_inventario)}"

    # Leyenda
    ws3 = wb.create_sheet("Leyenda")
    ws3.column_dimensions["A"].width = 24
    ws3.column_dimensions["B"].width = 76
    _titulo(ws3, 1, 2, "LEYENDA — Clasificación de archivos", "1F4E79")
    _encabezados(ws3, 2, ["Tipo", "Descripción"])

    descripciones = {
        "CT_REAL": (
            "Imágenes con modalidad/SOP Class CT válidas y no localizadoras; "
            "el paso 1 prioriza después la mejor serie axial."
        ),
        "CBCT_VOLUME": "Volumen CBCT/XperCT, normalmente multiframe o de modalidad XA/XRF.",
        "LOCALIZER": "Localizador, scout, surview o topograma; no se mezcla con la serie axial.",
        "SECONDARY": "Captura derivada o secundaria (informes, pantallazos, dose info, etc.).",
        "DICOMDIR": "Índice del estudio; no contiene el volumen de imagen que se entrenará.",
        "OTRO": "DICOM válido no incluido en las categorías anteriores; requiere revisión.",
        "ERROR_LECTURA": "Archivo ilegible o sin metadatos DICOM identificables.",
        "SIN_ARCHIVOS": "Paciente detectado, pero sin archivos en DICOM ni DICOMDIR.",
    }
    for fila, tipo in enumerate(TIPO_COLORES, start=3):
        for columna, valor in enumerate((TIPO_LABELS[tipo], descripciones[tipo]), start=1):
            celda = ws3.cell(fila, columna, valor)
            celda.font = Font(name="Arial", size=10, bold=(columna == 1))
            celda.fill = PatternFill("solid", fgColor=TIPO_COLORES[tipo])
            celda.alignment = Alignment(vertical="center", wrap_text=True)
            celda.border = _borde()
        ws3.row_dimensions[fila].height = 32

    ruta.parent.mkdir(parents=True, exist_ok=True)
    wb.save(ruta)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventario técnico DICOM del TFG CT/CBCT")
    parser.add_argument(
        "--project-dir",
        help="Raíz del proyecto. Por defecto se deduce desde este script.",
    )
    parser.add_argument(
        "--data-dir",
        help="Carpeta PACIENTES alternativa. Tiene prioridad sobre --project-dir.",
    )
    parser.add_argument(
        "--output-dir",
        help="Carpeta de salida. Por defecto se usa la raíz del proyecto.",
    )
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional de IDs (ej. 0001 0002). Sin filtro se detectan todos.",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    rutas = obtener_rutas(args.project_dir)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else rutas.pacientes
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else rutas.proyecto

    try:
        pacientes = detectar_pacientes(data_dir, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not pacientes:
        print(f"ERROR: no hay carpetas de paciente con formato XXXX en {data_dir}")
        return 2

    print(f"\nPacientes detectados en {data_dir}: {texto_pacientes(pacientes)}\n")

    registros: list[dict[str, Any]] = []
    for patient_id in pacientes:
        registros.extend(inventariar_paciente(data_dir / patient_id, data_dir))

    if not registros:
        print("ERROR: no se encontraron archivos para inventariar.")
        return 1

    df = pd.DataFrame(registros)
    for columna in COLUMNAS_INVENTARIO:
        if columna not in df.columns:
            df[columna] = ""

    df_inventario = df[COLUMNAS_INVENTARIO].copy()
    df_inventario = df_inventario.sort_values(
        by=["patient_id", "tipo", "SeriesNumber", "InstanceNumber", "filename"],
        kind="stable",
    ).reset_index(drop=True)
    df_resumen = generar_resumen(df_inventario)

    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_inventario_csv = output_dir / "inventario_completo.csv"
    ruta_resumen_csv = output_dir / "resumen_pacientes.csv"
    ruta_excel = output_dir / "inventario_TFG.xlsx"

    df_inventario.to_csv(ruta_inventario_csv, index=False, encoding="utf-8-sig")
    df_resumen.to_csv(ruta_resumen_csv, index=False, encoding="utf-8-sig")
    exportar_excel(df_resumen, df_inventario, ruta_excel)

    print("\n" + "═" * 80)
    print("RESUMEN POR PACIENTE")
    print("═" * 80)
    print(df_resumen.to_string(index=False))

    print("\nSalidas generadas:")
    print(f"  - {ruta_inventario_csv}")
    print(f"  - {ruta_resumen_csv}")
    print(f"  - {ruta_excel}")

    incompletos = df_resumen[df_resumen["tiene_par_completo"] == "✗"]["patient_id"].tolist()
    if incompletos:
        print(f"\n⚠ Pacientes sin par CT+CBCT completo: {', '.join(incompletos)}")
    else:
        print("\n✓ Todos los pacientes inventariados tienen CT y CBCT.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
