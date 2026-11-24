"""Convierte CSV en .xlsx con formato legible.

Uso:
    python csv_a_excel.py RUTA_CARPETA            todos los CSV de la carpeta (y subcarpetas)
    python csv_a_excel.py ARCHIVO.csv [OTRO.csv]  solo esos CSV
    python csv_a_excel.py                         carpeta actual

Se pueden mezclar archivos y carpetas. Cada CSV genera un .xlsx con el mismo
nombre a su lado; si ya existe, se rehace con el mismo contenido.

Se ignora lo que esté dentro de carpetas llamadas _archivo, .venv, __pycache__,
.git, .pytest_cache o TFG_scratch, y los *.schema.csv (metadatos de Slicer).
"""
import csv
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

COLORES_ESTADO = {"OK": "C6EFCE", "REVISAR": "FFEB9C", "ERROR": "FFC7CE", "OMITIDO": "D9D9D9"}
BORDE = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
CARPETAS_IGNORADAS = {"_archivo", ".venv", "__pycache__", ".git", ".pytest_cache", "TFG_scratch"}


def convertir_valor(texto: str):
    """Números como números; ceros a la izquierda (0001) se conservan como texto."""
    if texto == "":
        return None
    if texto.startswith("0") and len(texto) > 1 and texto.isdigit():
        return texto
    try:
        return int(texto)
    except ValueError:
        try:
            return float(texto)
        except ValueError:
            return texto


def convertir(ruta_csv: Path):
    """Devuelve la ruta del .xlsx creado, o None si el CSV está vacío."""
    with open(ruta_csv, newline="", encoding="utf-8-sig") as f:
        primera = f.readline()
        if not primera.strip():
            return None
        # Detecta ',' o ';' automáticamente; si no puede, asume ','
        try:
            dialecto = csv.Sniffer().sniff(primera, delimiters=",;")
        except csv.Error:
            dialecto = csv.excel
        f.seek(0)
        filas = list(csv.reader(f, dialecto))

    wb = Workbook()
    ws = wb.active
    ws.title = ruta_csv.stem[:31]

    for fila in filas:
        # Une los mensajes que quedaron separados por tabuladores y quita columnas vacías finales
        fila = [c.replace("\t", " | ").replace("\n", " ").strip() for c in fila]
        while fila and fila[-1] == "":
            fila.pop()
        ws.append([convertir_valor(c) for c in fila])

    cabecera = [c.value for c in ws[1]]
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDE
    ws.row_dimensions[1].height = 40

    for fila in ws.iter_rows(min_row=2):
        for c in fila:
            c.border = BORDE
            largo = isinstance(c.value, str) and len(c.value) > 40
            c.alignment = Alignment(
                horizontal="left" if largo else "center", vertical="center", wrap_text=largo
            )
            if cabecera[c.column - 1] == "estado" and c.value in COLORES_ESTADO:
                c.fill = PatternFill("solid", fgColor=COLORES_ESTADO[c.value])
                c.font = Font(bold=True)

    for i, col in enumerate(ws.columns, 1):
        ancho = max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(max(ancho, 8), 50)

    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    destino = ruta_csv.with_suffix(".xlsx")
    wb.save(destino)
    return destino


def recoger_csv(rutas: list[Path]):
    """Genera los CSV a convertir: archivos tal cual y carpetas de forma recursiva."""
    for ruta in rutas:
        if ruta.is_file():
            yield ruta
            continue
        for csv_path in sorted(ruta.rglob("*.csv")):
            if CARPETAS_IGNORADAS & set(csv_path.relative_to(ruta).parts):
                continue
            if csv_path.name.endswith(".schema.csv"):
                continue
            yield csv_path


if __name__ == "__main__":
    rutas = [Path(a) for a in sys.argv[1:]] or [Path(".")]
    for ruta in rutas:
        if not ruta.exists():
            raise SystemExit(f"No existe: {ruta}")
    for csv_path in recoger_csv(rutas):
        destino = convertir(csv_path)
        if destino is None:
            print("– vacío, omitido:", csv_path)
        else:
            print("✓", destino)
