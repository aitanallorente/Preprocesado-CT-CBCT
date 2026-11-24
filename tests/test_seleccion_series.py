from pathlib import Path
from types import SimpleNamespace

from paso1_dicom_a_nifti import (
    EntradaDicom,
    _calidad_serie_cbct,
    _calidad_serie_ct,
    _validar_coherencia_multifichero,
)
import paso1_dicom_a_nifti
import pytest


def _entrada(ruta: Path, **kwargs) -> EntradaDicom:
    ruta.write_bytes(b"x")
    return EntradaDicom(ruta=ruta, dataset=SimpleNamespace(**kwargs), tipo="PRUEBA")


def test_serie_axial_tiene_prioridad_sobre_reconstruccion_no_axial(tmp_path: Path):
    axial = [
        _entrada(
            tmp_path / f"axial_{i}",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImageType=["ORIGINAL", "PRIMARY", "AXIAL"],
            Rows=512,
            Columns=512,
        )
        for i in range(5)
    ]
    coronal = [
        _entrada(
            tmp_path / f"coronal_{i}",
            ImageOrientationPatient=[1, 0, 0, 0, 0, -1],
            ImageType=["ORIGINAL", "PRIMARY", "CORONAL"],
            Rows=512,
            Columns=512,
        )
        for i in range(20)
    ]

    assert _calidad_serie_ct(axial) > _calidad_serie_ct(coronal)


def test_sop_3d_cbct_tiene_prioridad_sobre_xa_larga_generica(tmp_path: Path):
    cbct_3d = [
        _entrada(
            tmp_path / "cbct_3d",
            SOPClassUID="1.2.840.10008.5.1.4.1.1.13.1.1",
            Modality="XA",
            ImageType=["ORIGINAL", "PRIMARY"],
            SeriesDescription="XperCT",
            NumberOfFrames=384,
        )
    ]
    xa_generica = [
        _entrada(
            tmp_path / "xa_generica",
            SOPClassUID="1.2.840.10008.5.1.4.1.1.12.1",
            Modality="XA",
            ImageType=["ORIGINAL", "PRIMARY"],
            SeriesDescription="Angiography",
            NumberOfFrames=2000,
        )
    ]

    assert _calidad_serie_cbct(cbct_3d) > _calidad_serie_cbct(xa_generica)


def test_coherencia_multifichero_rechaza_posiciones_duplicadas(
    monkeypatch,
    tmp_path: Path,
):
    rutas = [tmp_path / "a", tmp_path / "b"]
    for ruta in rutas:
        ruta.write_bytes(b"x")

    datasets = {
        "a": SimpleNamespace(
            Rows=512,
            Columns=512,
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 10],
        ),
        "b": SimpleNamespace(
            Rows=512,
            Columns=512,
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 10],
        ),
    }
    monkeypatch.setattr(
        paso1_dicom_a_nifti,
        "leer_cabecera",
        lambda ruta: datasets[Path(ruta).name],
    )

    with pytest.raises(ValueError, match="posiciones físicas repetidas"):
        _validar_coherencia_multifichero(rutas, "CT")


def test_coherencia_multifichero_rechaza_matrices_mezcladas(
    monkeypatch,
    tmp_path: Path,
):
    rutas = [tmp_path / "a", tmp_path / "b"]
    for ruta in rutas:
        ruta.write_bytes(b"x")

    datasets = {
        "a": SimpleNamespace(
            Rows=512,
            Columns=512,
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 0],
        ),
        "b": SimpleNamespace(
            Rows=256,
            Columns=256,
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 1],
        ),
    }
    monkeypatch.setattr(
        paso1_dicom_a_nifti,
        "leer_cabecera",
        lambda ruta: datasets[Path(ruta).name],
    )

    with pytest.raises(ValueError, match="matrices distintas"):
        _validar_coherencia_multifichero(rutas, "CT")
