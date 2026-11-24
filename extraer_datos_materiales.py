"""Extrae los datos técnicos que faltan para el Capítulo 5 (Materiales).

No forma parte del pipeline. Genera, en la raíz del proyecto:

* ``protocolo_adquisicion.csv``: parámetros de adquisición del CT y del CBCT
  seleccionados en el paso 1 (una fila por paciente y modalidad), más los días
  transcurridos entre ambos estudios. Solo lee campos técnicos: NO lee nombre,
  identificador, fecha de nacimiento ni ningún otro dato del paciente, y las
  fechas solo se usan para calcular la diferencia en días (no se guardan).
* ``entorno_software_hardware.txt``: versiones de Python y de las librerías,
  sistema operativo, procesador, memoria RAM y GPU.

Uso (desde la carpeta del código, con el entorno virtual activado):

    python extraer_datos_materiales.py

Requiere haber ejecutado antes el paso 1 (lee PACIENTES_NIFTI/informe_conversion.csv).
"""

from __future__ import annotations

import csv
import ctypes
import platform
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

import pydicom

from configuracion import detectar_pacientes, obtener_rutas

# Campos técnicos que se buscan (en el nivel principal o dentro de secuencias).
CAMPOS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "ImageType", "SeriesDescription", "ProtocolName", "BodyPartExamined",
    "PatientPosition", "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime",
    "CTDIvol", "SpiralPitchFactor", "ConvolutionKernel", "FilterType",
    "ReconstructionDiameter", "SliceThickness", "SpacingBetweenSlices",
    "PixelSpacing", "NumberOfFrames", "Rows", "Columns",
    "ContrastBolusAgent", "ContrastBolusRoute", "ContrastBolusVolume",
    "ContrastBolusStartTime", "RescaleSlope", "RescaleIntercept", "RescaleType",
    "DistanceSourceToDetector", "DistanceSourceToPatient",
    "PrimaryPositionerScanArc", "PositionerPrimaryAngleIncrement",
    "DerivationDescription",
]
# Campos de fecha: solo para calcular días entre estudios, nunca se guardan.
FECHAS = ["AcquisitionDate", "ContentDate", "SeriesDate", "StudyDate"]


def buscar(ds, palabra: str) -> str:
    """Primer valor de ``palabra`` en el dataset, incluidas las secuencias."""
    if palabra in ds:
        valor = ds.data_element(palabra).value
        return _texto(valor)
    for elem in ds.iterall():
        if elem.keyword == palabra:
            return _texto(elem.value)
    return ""


def _texto(valor) -> str:
    if isinstance(valor, (list, pydicom.multival.MultiValue)):
        return " / ".join(str(v) for v in valor)
    return str(valor).strip()


def fecha(ds) -> date | None:
    for palabra in FECHAS:
        texto = buscar(ds, palabra)
        if len(texto) >= 8 and texto[:8].isdigit():
            return date(int(texto[:4]), int(texto[4:6]), int(texto[6:8]))
    return None


def leer(ruta: Path):
    return pydicom.dcmread(str(ruta), stop_before_pixels=True, force=True)


def archivo_ct(carpeta: Path, uid_serie: str) -> Path | None:
    """Un archivo (el central) de la serie CT seleccionada en el paso 1."""
    candidatos = []
    for ruta in sorted(p for p in carpeta.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(str(ruta), stop_before_pixels=True, force=True,
                                 specific_tags=["SeriesInstanceUID"])
        except Exception:
            continue
        if str(getattr(ds, "SeriesInstanceUID", "")) == uid_serie:
            candidatos.append(ruta)
    return candidatos[len(candidatos) // 2] if candidatos else None


def protocolo(rutas) -> Path:
    informe = rutas.nifti / "informe_conversion.csv"
    with open(informe, encoding="utf-8-sig", newline="") as f:
        seleccion = {fila["patient_id"]: fila for fila in csv.DictReader(f)}

    filas = []
    for pid in detectar_pacientes(rutas.pacientes):
        if pid not in seleccion:
            print(f"  {pid}: no aparece en informe_conversion.csv, se omite")
            continue
        carpeta = rutas.pacientes / pid / "DICOM"
        ruta_ct = archivo_ct(carpeta, seleccion[pid]["serie_ct_uid"])
        ruta_cbct = next(carpeta.rglob(seleccion[pid]["cbct_archivo"]), None)
        datasets = {"CT": leer(ruta_ct) if ruta_ct else None,
                    "CBCT": leer(ruta_cbct) if ruta_cbct else None}

        dias = ""
        if datasets["CT"] is not None and datasets["CBCT"] is not None:
            f_ct, f_cbct = fecha(datasets["CT"]), fecha(datasets["CBCT"])
            if f_ct and f_cbct:
                dias = str((f_cbct - f_ct).days)

        for modalidad, ds in datasets.items():
            fila = {"patient_id": pid, "volumen": modalidad,
                    "dias_CT_a_CBCT": dias}
            for campo in CAMPOS:
                fila[campo] = buscar(ds, campo) if ds is not None else "NO ENCONTRADO"
            filas.append(fila)
        print(f"  {pid}: hecho")

    salida = rutas.proyecto / "protocolo_adquisicion.csv"
    with open(salida, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["patient_id", "volumen",
                                                 "dias_CT_a_CBCT", *CAMPOS])
        escritor.writeheader()
        escritor.writerows(filas)
    return salida


def ram_gb() -> str:
    try:
        class Estado(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        estado = Estado()
        estado.dwLength = ctypes.sizeof(Estado)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
        return f"{estado.ullTotalPhys / 1024**3:.1f} GB"
    except Exception:
        return "no disponible (consultar en Configuración > Sistema > Información)"


def procesador() -> str:
    try:
        salida = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30)
        return salida.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def gpu() -> str:
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return salida.stdout.strip() or "nvidia-smi sin salida"
    except Exception:
        return "sin GPU NVIDIA o nvidia-smi no disponible"


def entorno(rutas) -> Path:
    lineas = [f"Python: {sys.version.split()[0]}",
              f"Sistema operativo: {platform.system()} {platform.release()} ({platform.version()})",
              f"Procesador: {procesador()}",
              f"Memoria RAM: {ram_gb()}",
              f"GPU: {gpu()}", "", "Librerías:"]
    for paquete in ["numpy", "pydicom", "SimpleITK", "pandas", "openpyxl",
                    "matplotlib", "pytest", "torch", "monai"]:
        try:
            lineas.append(f"  {paquete}: {metadata.version(paquete)}")
        except metadata.PackageNotFoundError:
            lineas.append(f"  {paquete}: no instalado")
    try:
        import torch
        lineas.append(f"  CUDA (PyTorch): {torch.version.cuda}; "
                      f"GPU disponible: {torch.cuda.is_available()}")
    except Exception:
        pass

    salida = rutas.proyecto / "entorno_software_hardware.txt"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return salida


def main() -> None:
    rutas = obtener_rutas()
    print("Leyendo cabeceras DICOM (puede tardar un par de minutos)...")
    print(f"Guardado: {protocolo(rutas)}")
    print(f"Guardado: {entorno(rutas)}")


if __name__ == "__main__":
    main()
