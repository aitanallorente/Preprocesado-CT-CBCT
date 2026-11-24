"""Clasificación DICOM común para el inventario y el paso 1 (DICOM -> NIfTI).

Las reglas están diseñadas para excluir localizadores, surviews y capturas
secundarias antes de seleccionar la serie CT axial principal, y para reconocer
y puntuar las series CBCT candidatas.
"""

from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable

CT_IMAGE_STORAGE_UIDS = {
    "1.2.840.10008.5.1.4.1.1.2",      # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2.1",    # Enhanced CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2.2",    # Legacy Converted Enhanced CT
}

SECONDARY_CAPTURE_UIDS = {
    "1.2.840.10008.5.1.4.1.1.7",
    "1.2.840.10008.5.1.4.1.1.7.1",
    "1.2.840.10008.5.1.4.1.1.7.2",
    "1.2.840.10008.5.1.4.1.1.7.3",
    "1.2.840.10008.5.1.4.1.1.7.4",
}

X_RAY_3D_UIDS = {
    "1.2.840.10008.5.1.4.1.1.13.1.1",  # X-Ray 3D Angiographic Image Storage
    "1.2.840.10008.5.1.4.1.1.13.1.2",  # X-Ray 3D Craniofacial Image Storage
}

PALABRAS_CBCT = ("CBCT", "XPERCT", "XPER CT", "CONE BEAM", "CONE-BEAM")
PALABRAS_LOCALIZADOR = (
    "LOCALIZER",
    "LOCALISER",
    "SCOUT",
    "SURVIEW",
    "TOPOGRAM",
    "SCANOGRAM",
)


def texto_valor(valor: Any, separador: str = " / ") -> str:
    """Convierte valores DICOM simples o múltiples en texto estable."""

    if valor is None:
        return ""
    if isinstance(valor, (str, bytes)):
        return valor.decode(errors="replace") if isinstance(valor, bytes) else valor
    if isinstance(valor, Iterable):
        try:
            return separador.join(str(v) for v in valor)
        except TypeError:
            pass
    return str(valor)


def texto_tag(ds: Any, nombre: str) -> str:
    return texto_valor(getattr(ds, nombre, ""))


def entero_tag(ds: Any, nombre: str, por_defecto: int = 0) -> int:
    try:
        return int(getattr(ds, nombre, por_defecto))
    except (TypeError, ValueError):
        return por_defecto


def _texto_clasificacion(ds: Any) -> str:
    campos = (
        texto_tag(ds, "ImageType"),
        texto_tag(ds, "SeriesDescription"),
        texto_tag(ds, "ProtocolName"),
    )
    return " | ".join(campos).upper()


def es_localizador(ds: Any) -> bool:
    texto = _texto_clasificacion(ds)
    return any(palabra in texto for palabra in PALABRAS_LOCALIZADOR)


def es_cbct(ds: Any) -> bool:
    modalidad = texto_tag(ds, "Modality").upper()
    sop_uid = texto_tag(ds, "SOPClassUID")
    texto = _texto_clasificacion(ds)
    frames = entero_tag(ds, "NumberOfFrames", 1)

    if sop_uid in X_RAY_3D_UIDS:
        return True
    if modalidad in {"XA", "XRF"} and (frames > 1 or any(k in texto for k in PALABRAS_CBCT)):
        return True
    if any(k in texto for k in PALABRAS_CBCT):
        return True

    # Una captura secundaria multiframe no debe convertirse en CBCT solo por
    # tener muchos frames. Las palabras CBCT/XperCT anteriores sí pueden
    # prevalecer cuando el exportador usa una SOP Class secundaria.
    if modalidad == "SC" or sop_uid in SECONDARY_CAPTURE_UIDS:
        return False

    # Fallback conservador para volúmenes multiframe que no se etiquetaron bien.
    # Se excluyen modalidades que suelen representar otros volúmenes 3D.
    if frames > 100 and modalidad not in {"CT", "MR", "PT", "NM", "US", "SC"}:
        return True

    return False


def es_captura_secundaria(ds: Any) -> bool:
    """Detecta capturas secundarias sin descartar reconstrucciones CT válidas.

    ``DERIVED`` no implica por sí solo que una imagen CT sea un pantallazo: es
    frecuente en reconstrucciones axiales o reformateos almacenados con una SOP
    Class CT legítima. La SOP Class y la modalidad tienen prioridad para evitar
    perder series utilizables en futuros pacientes.
    """

    modalidad = texto_tag(ds, "Modality").upper()
    sop_uid = texto_tag(ds, "SOPClassUID")
    image_type = texto_tag(ds, "ImageType").upper()

    if modalidad == "SC" or sop_uid in SECONDARY_CAPTURE_UIDS:
        return True

    if modalidad == "CT" and sop_uid in CT_IMAGE_STORAGE_UIDS:
        return False
    if sop_uid in X_RAY_3D_UIDS:
        return False

    return "SECONDARY" in image_type or "DERIVED" in image_type


def es_ct_real(ds: Any) -> bool:
    modalidad = texto_tag(ds, "Modality").upper()
    sop_uid = texto_tag(ds, "SOPClassUID")

    return (
        modalidad == "CT"
        and sop_uid in CT_IMAGE_STORAGE_UIDS
        and not es_localizador(ds)
        and not es_captura_secundaria(ds)
    )


def clasificar_dataset(ds: Any, filename: str = "") -> str:
    """Clasifica un dataset en una categoría de alto nivel."""

    if Path(filename).name.upper() == "DICOMDIR":
        return "DICOMDIR"
    if es_localizador(ds):
        return "LOCALIZER"
    if es_cbct(ds):
        return "CBCT_VOLUME"
    if es_captura_secundaria(ds):
        return "SECONDARY"
    if es_ct_real(ds):
        return "CT_REAL"
    return "OTRO"


def identificador_serie(ds: Any, filepath: Path) -> str:
    """Obtiene un identificador estable de serie incluso si falta el UID."""

    uid = texto_tag(ds, "SeriesInstanceUID").strip()
    if uid:
        return uid
    numero = texto_tag(ds, "SeriesNumber").strip() or "sin_numero"
    return f"SIN_UID::{filepath.parent.resolve()}::{numero}"


def _lista_float(valor: Any, longitud: int) -> list[float] | None:
    if isinstance(valor, str):
        valor = [parte.strip() for parte in valor.split("\\") if parte.strip()]
    try:
        valores = [float(v) for v in valor]
    except (TypeError, ValueError):
        return None
    if len(valores) < longitud or not all(isfinite(v) for v in valores[:longitud]):
        return None
    return valores


def normal_plano_corte(ds: Any) -> tuple[float, float, float] | None:
    """Calcula la normal unitaria del plano a partir de ImageOrientationPatient."""

    orientacion = _lista_float(getattr(ds, "ImageOrientationPatient", None), 6)
    if orientacion is None:
        return None

    r = orientacion[:3]
    c = orientacion[3:6]
    normal = [
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    ]
    norma = sqrt(sum(v * v for v in normal))
    if norma <= 1e-12:
        return None
    return tuple(v / norma for v in normal)


def es_corte_axial(ds: Any, umbral: float = 0.80) -> bool | None:
    """Indica si la normal del corte es predominantemente craneocaudal.

    Devuelve ``None`` cuando faltan metadatos geométricos. El valor absoluto
    permite reconocer adquisiciones axiales con orden Z ascendente o descendente
    y tolera una inclinación moderada del gantry.
    """

    if not 0.0 < umbral <= 1.0:
        raise ValueError("El umbral axial debe estar en (0, 1]")

    normal = normal_plano_corte(ds)
    if normal is None:
        return None
    return abs(normal[2]) >= umbral


def posicion_escalar_corte(ds: Any) -> float | None:
    """Proyecta ImagePositionPatient sobre la normal del plano del corte."""

    posicion = _lista_float(getattr(ds, "ImagePositionPatient", None), 3)
    normal = normal_plano_corte(ds)
    if posicion is None or normal is None:
        return None
    return sum(p * n for p, n in zip(posicion, normal))


def clave_orden_corte(ds: Any, filepath: Path) -> tuple:
    """Ordena por posición física y usa InstanceNumber como fallback."""

    posicion = posicion_escalar_corte(ds)
    instancia = entero_tag(ds, "InstanceNumber", 0)
    if posicion is not None:
        return (0, posicion, instancia, filepath.name)
    return (1, instancia, filepath.name)


def puntuacion_cbct(ds: Any, filepath: Path) -> tuple[int, int, int, int, int, int]:
    """Puntuación lexicográfica para elegir un candidato CBCT plausible.

    Los indicadores semánticos (SOP Class 3D y palabras CBCT/XperCT) tienen
    prioridad sobre el número de frames, para que una angiografía XA larga no
    gane únicamente por contener más imágenes.
    """

    modalidad = texto_tag(ds, "Modality").upper()
    sop_uid = texto_tag(ds, "SOPClassUID")
    texto = _texto_clasificacion(ds)
    frames = max(entero_tag(ds, "NumberOfFrames", 1), 1)

    try:
        tamano = filepath.stat().st_size
    except OSError:
        tamano = 0

    return (
        int(sop_uid in X_RAY_3D_UIDS),
        int(any(k in texto for k in PALABRAS_CBCT)),
        int(modalidad in {"XA", "XRF"}),
        int(frames > 1),
        min(frames, 1_000_000),
        tamano,
    )
