from pathlib import Path
from types import SimpleNamespace

import pytest

from configuracion import detectar_pacientes, es_id_paciente, obtener_rutas
from dicom_utils import clasificar_dataset, es_corte_axial, posicion_escalar_corte

def ds(**kwargs):
    return SimpleNamespace(**kwargs)

def test_formato_paciente_exacto():
    assert es_id_paciente("0001")
    assert es_id_paciente("9999")
    assert not es_id_paciente("001")
    assert not es_id_paciente("00001")
    assert not es_id_paciente("A001")
    assert not es_id_paciente("٠٠٠١")  # dígitos Unicode, no ASCII

def test_deteccion_automatica_ordenada(tmp_path: Path):
    for nombre in ("0006", "0001", "0010", "123", "abcd"):
        (tmp_path / nombre).mkdir()
    assert detectar_pacientes(tmp_path) == ["0001", "0006", "0010"]
    assert detectar_pacientes(tmp_path, ["0006", "0001"]) == ["0001", "0006"]

def test_deteccion_rechaza_id_invalido(tmp_path: Path):
    (tmp_path / "0001").mkdir()
    with pytest.raises(ValueError):
        detectar_pacientes(tmp_path, ["1"])

def test_clasificacion_localizador():
    dataset = ds(
        Modality="CT",
        SOPClassUID="1.2.840.10008.5.1.4.1.1.2",
        ImageType=["ORIGINAL", "PRIMARY", "LOCALIZER"],
    )
    assert clasificar_dataset(dataset, "IM_0001") == "LOCALIZER"

def test_clasificacion_captura_secundaria():
    dataset = ds(
        Modality="CT",
        SOPClassUID="1.2.840.10008.5.1.4.1.1.7",
        ImageType=["DERIVED", "SECONDARY", "RESULTS"],
    )
    assert clasificar_dataset(dataset, "IM_0003") == "SECONDARY"

def test_clasificacion_ct_axial():
    dataset = ds(
        Modality="CT",
        SOPClassUID="1.2.840.10008.5.1.4.1.1.2",
        ImageType=["ORIGINAL", "PRIMARY", "AXIAL"],
    )
    assert clasificar_dataset(dataset, "IM_0093") == "CT_REAL"


def test_reconstruccion_ct_derivada_no_se_confunde_con_captura_secundaria():
    dataset = ds(
        Modality="CT",
        SOPClassUID="1.2.840.10008.5.1.4.1.1.2",
        ImageType=["DERIVED", "PRIMARY", "AXIAL"],
    )
    assert clasificar_dataset(dataset, "reconstruccion") == "CT_REAL"


def test_clasificacion_cbct_xa_multiframe():
    dataset = ds(
        Modality="XA",
        SOPClassUID="1.2.840.10008.5.1.4.1.1.13.1.1",
        ImageType=["ORIGINAL", "PRIMARY"],
        NumberOfFrames=384,
    )
    assert clasificar_dataset(dataset, "volume") == "CBCT_VOLUME"

def test_orientacion_axial_y_no_axial():
    axial = ds(ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    coronal = ds(ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert es_corte_axial(axial) is True
    assert es_corte_axial(coronal) is False

def test_orientacion_axial_desconocida_sin_iop():
    assert es_corte_axial(ds()) is None


def test_orientacion_no_finita_se_trata_como_desconocida():
    dataset = ds(ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, float("nan"), 0.0])
    assert es_corte_axial(dataset) is None


def test_posicion_escalar_axial():
    dataset = ds(
        ImagePositionPatient=[0.0, 0.0, -10.0],
        ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert posicion_escalar_corte(dataset) == pytest.approx(-10.0)

def test_rutas_por_defecto_apuntan_a_carpetas_nuevas():
    rutas = obtener_rutas()
    assert rutas.pacientes.name == "PACIENTES"
    assert rutas.nifti.name == "PACIENTES_NIFTI"
    assert rutas.registered.name == "PACIENTES_REGISTERED"
    assert rutas.normalized.name == "PACIENTES_NORMALIZED"
    assert rutas.final.name == "PACIENTES_FINAL"
