from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import paso2_registro
import paso4_redimensionado
from imagen_utils import validar_geometria_3d


class FakeImage:
    def __init__(
        self,
        size=(10, 11, 12),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ):
        self._size = tuple(int(v) for v in size)
        self._spacing = tuple(float(v) for v in spacing)
        self._origin = tuple(float(v) for v in origin)
        self._direction = tuple(float(v) for v in direction)

    def GetDimension(self):
        return 3

    def GetSize(self):
        return self._size

    def GetSpacing(self):
        return self._spacing

    def GetOrigin(self):
        return self._origin

    def GetDirection(self):
        return self._direction

    def SetSpacing(self, value):
        self._spacing = tuple(float(v) for v in value)

    def SetOrigin(self, value):
        self._origin = tuple(float(v) for v in value)

    def SetDirection(self, value):
        self._direction = tuple(float(v) for v in value)

    def TransformContinuousIndexToPhysicalPoint(self, indice):
        spacing = np.asarray(self._spacing, dtype=float)
        origin = np.asarray(self._origin, dtype=float)
        direction = np.asarray(self._direction, dtype=float).reshape(3, 3)
        return tuple(origin + direction @ (np.asarray(indice, dtype=float) * spacing))


def _centro_fisico(imagen: FakeImage) -> np.ndarray:
    size = np.asarray(imagen.GetSize(), dtype=float)
    spacing = np.asarray(imagen.GetSpacing(), dtype=float)
    origin = np.asarray(imagen.GetOrigin(), dtype=float)
    direction = np.asarray(imagen.GetDirection(), dtype=float).reshape(3, 3)
    return origin + direction @ (((size - 1.0) / 2.0) * spacing)


def test_validar_geometria_3d_acepta_geometria_regular():
    validar_geometria_3d(FakeImage(), "prueba")


def test_validar_geometria_3d_rechaza_spacing_no_finito():
    with pytest.raises(ValueError, match="spacing"):
        validar_geometria_3d(FakeImage(spacing=(1.0, float("nan"), 1.0)))


def test_ortonormalizacion_svd_conserva_rotacion_oblicua():
    angulo = np.deg2rad(17.0)
    rotacion = np.array(
        [
            [np.cos(angulo), -np.sin(angulo), 0.0],
            [np.sin(angulo), np.cos(angulo), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    perturbada = rotacion.copy()
    perturbada[0, 0] += 1e-3
    imagen = FakeImage(direction=tuple(perturbada.ravel()))
    centro_antes = _centro_fisico(imagen)

    corregida, magnitud = paso2_registro.ortonormalizar_direccion(imagen)
    matriz = np.asarray(corregida.GetDirection()).reshape(3, 3)

    assert magnitud > 0
    assert matriz.T @ matriz == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(matriz) == pytest.approx(1.0, abs=1e-10)
    assert abs(matriz[0, 1]) > 0.1  # no se redondeó a orientación cardinal
    assert _centro_fisico(corregida) == pytest.approx(centro_antes, abs=1e-10)


def test_spacing_comun_contiene_el_mayor_campo_de_la_cohorte():
    imagenes = [
        FakeImage(size=(512, 512, 100), spacing=(0.8, 0.8, 2.0)),
        FakeImage(size=(400, 400, 300), spacing=(1.0, 1.0, 1.0)),
    ]
    spacing, fov = paso4_redimensionado.calcular_spacing_comun(
        imagenes, (256, 256, 128), margen_fov_mm=0.0, redondeo_spacing_mm=0.0
    )

    # Mayor extensión por eje: x,y = 511 * 0.8 = 408.8 mm; z = 299 * 1.0 = 299 mm
    assert spacing[0] == pytest.approx(408.8 / 255)
    assert spacing[2] == pytest.approx(299.0 / 127)
    assert fov[0] >= 408.8 - 1e-9
    assert fov[2] >= 299.0 - 1e-9


def test_referencia_centrada_conserva_centro_fisico(monkeypatch):
    imagen = FakeImage(
        size=(512, 384, 100),
        spacing=(0.75, 0.80, 2.0),
        origin=(10.0, -30.0, 4.0),
    )

    fake_sitk = SimpleNamespace(
        sitkFloat32=1,
        Image=lambda size, _pixel_id: FakeImage(size=size),
    )
    monkeypatch.setattr(paso4_redimensionado, "sitk", fake_sitk)

    referencia = paso4_redimensionado.crear_referencia_centrada(
        imagen, (256, 256, 128), (1.6, 1.6, 2.0)
    )

    assert referencia.GetSize() == (256, 256, 128)
    assert referencia.GetSpacing() == pytest.approx((1.6, 1.6, 2.0))
    assert _centro_fisico(referencia) == pytest.approx(_centro_fisico(imagen))


def test_normalizacion_hu_fijo_a_0_1():
    import paso3_normalizacion

    array = np.array([[[-2000.0, -1024.0, 1023.5, 3071.0, 4000.0]]], dtype=np.float32)
    normalizada, stats = paso3_normalizacion.normalizar_array(array)

    # (HU + 1024) / 4095, con recorte previo a [-1024, 3071]
    esperado = np.array([[[0.0, 0.0, 0.5, 1.0, 1.0]]], dtype=np.float32)
    assert normalizada == pytest.approx(esperado)
    assert stats["porcentaje_clipped_bajo"] == pytest.approx(20.0)
    assert stats["porcentaje_clipped_alto"] == pytest.approx(20.0)


def test_normalizacion_excluye_relleno_de_las_estadisticas():
    import paso3_normalizacion

    array = np.array([[[-1024.0, -1024.0, 0.0, 1000.0]]], dtype=np.float32)
    _, stats = paso3_normalizacion.normalizar_array(array, padding_entrada=-1024.0)

    assert stats["voxeles_validos"] == 2
    assert stats["porcentaje_padding"] == pytest.approx(50.0)
