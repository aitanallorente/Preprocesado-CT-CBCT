from pathlib import Path

import pandas as pd

from inventario_dicom import generar_resumen, inventariar_paciente


def test_paciente_sin_dicom_permanece_en_inventario(tmp_path: Path):
    paciente = tmp_path / "0001"
    paciente.mkdir()

    filas = inventariar_paciente(paciente, tmp_path)

    assert len(filas) == 1
    assert filas[0]["patient_id"] == "0001"
    assert filas[0]["tipo"] == "SIN_ARCHIVOS"

    df = pd.DataFrame(filas)
    for columna in ("SeriesInstanceUID", "Rows", "Columns", "PixelSpacing", "SliceThickness"):
        if columna not in df:
            df[columna] = ""
    resumen = generar_resumen(df).iloc[0]
    assert resumen["TOTAL"] == 0
    assert resumen["SIN_ARCHIVOS"] == 1
