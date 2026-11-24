from pathlib import Path

import pytest

import escritura_segura
from escritura_segura import escribir_grupo_atomico


def _escribir_texto(texto: str):
    def escritor(ruta: Path) -> None:
        ruta.write_text(texto, encoding="utf-8")

    return escritor


def test_escritura_grupal_reemplaza_todas_las_salidas(tmp_path: Path):
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.tfm"
    a.write_text("anterior-a", encoding="utf-8")
    b.write_text("anterior-b", encoding="utf-8")

    escribir_grupo_atomico(
        [
            (a, _escribir_texto("nuevo-a")),
            (b, _escribir_texto("nuevo-b")),
        ]
    )

    assert a.read_text(encoding="utf-8") == "nuevo-a"
    assert b.read_text(encoding="utf-8") == "nuevo-b"
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert not list(tmp_path.glob(".*.bak.*"))


def test_fallo_de_preparacion_conserva_todas_las_salidas_previas(tmp_path: Path):
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii.gz"
    a.write_text("anterior-a", encoding="utf-8")
    b.write_text("anterior-b", encoding="utf-8")

    def fallar(_ruta: Path) -> None:
        raise OSError("fallo simulado")

    with pytest.raises(OSError, match="fallo simulado"):
        escribir_grupo_atomico(
            [
                (a, _escribir_texto("nuevo-a")),
                (b, fallar),
            ]
        )

    assert a.read_text(encoding="utf-8") == "anterior-a"
    assert b.read_text(encoding="utf-8") == "anterior-b"


def test_fallo_durante_confirmacion_restaura_el_grupo(monkeypatch, tmp_path: Path):
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii.gz"
    a.write_text("anterior-a", encoding="utf-8")
    b.write_text("anterior-b", encoding="utf-8")

    replace_real = escritura_segura.os.replace
    llamadas = {"n": 0}

    def replace_con_fallo(origen, destino):
        llamadas["n"] += 1
        if llamadas["n"] == 4:
            raise OSError("fallo de confirmación simulado")
        return replace_real(origen, destino)

    monkeypatch.setattr(escritura_segura.os, "replace", replace_con_fallo)

    with pytest.raises(OSError, match="fallo de confirmación simulado"):
        escribir_grupo_atomico(
            [
                (a, _escribir_texto("nuevo-a")),
                (b, _escribir_texto("nuevo-b")),
            ]
        )

    assert a.read_text(encoding="utf-8") == "anterior-a"
    assert b.read_text(encoding="utf-8") == "anterior-b"
