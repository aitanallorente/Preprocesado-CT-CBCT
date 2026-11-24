"""Paso 2: registro rígido CBCT -> CT para abdomen.

Entrada:  ``PACIENTES_NIFTI/XXXX/XXXX_ct.nii.gz`` y ``XXXX_cbct.nii.gz``
Salida:   ``PACIENTES_REGISTERED/XXXX/XXXX_ct.nii.gz``,
          ``XXXX_cbct_registered.nii.gz`` (CBCT remuestreado en la rejilla del
          CT), ``XXXX_transform.tfm`` e ``informe_registro.csv`` (acumulativo).

El CBCT abdominal cubre solo una parte del CT (FOV parcial) y los dos equipos
usan sistemas de coordenadas distintos, por lo que el registro tiene dos modos:

Modo automático (por defecto)
    1. Se ortonormaliza la dirección y se reorientan CT y CBCT a LPS.
    2. Búsqueda gruesa sobre imágenes de bordes: Z -> X -> Y -> rotación (±6°).
    3. Refinamiento multirresolución con Mattes Mutual Information.
    Es el modo usado para los pacientes en los que el resultado es
    anatómicamente correcto.

Modo landmarks (``--landmarks-dir LANDMARKS --pacientes ...``)
    Para los pacientes en los que el modo automático NO localiza bien la
    anatomía (cae en otro nivel craneocaudal). Se marcan en 3D Slicer los mismos
    puntos óseos en CT y CBCT (``XXXX_CT.mrk.json`` / ``XXXX_CBCT.mrk.json``) y
    la transformación rígida se calcula por mínimos cuadrados (Kabsch), sin
    escala y sin Mutual Information, para no estropear la correspondencia
    manual. Se aplica a TODOS los pacientes seleccionados en la llamada, así que
    hay que indicarlos con ``--pacientes`` (``ejecutar_pipeline.py pre-spm`` lo
    hace solo: usa landmarks para quien tenga sus dos archivos en LANDMARKS).

En los dos modos, tras registrar, el par se recorta a la zona donde existe CBCT
(con un margen común) y el exterior del CBCT se rellena con -1024.

Este registro es el PUNTO DE PARTIDA, no el resultado final: después todos los
pacientes se ajustan a mano en SPM y ``paso2c_spm_a_registrado.py`` genera el
par registrado definitivo.

Las opciones ``--inicializacion geometria|momentos``, ``--transformacion
similitud`` y ``--reset-cbct-origin`` son experimentales y no se usan en el
flujo del TFG. Las versiones previas del registro se descartaron (ver memoria, apéndice C).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from configuracion import detectar_pacientes, obtener_rutas, texto_pacientes
from escritura_segura import escribir_grupo_atomico
from informes import guardar_informe_acumulado
from imagen_utils import (
    diferencias_geometria,
    geometrias_coinciden,
    validar_geometria_3d,
)

VERSION_REGISTRO = "robusto_lps_v5"
VERSION_LANDMARKS = "landmarks_rigido_lps_v5"


def ortonormalizar_direccion(
    imagen: sitk.Image,
    tolerancia: float = 1e-6,
) -> tuple[sitk.Image, float]:
    """Proyecta la dirección a la matriz ortogonal más cercana si hace falta.

    No redondea orientaciones oblicuas válidas. Si corrige la matriz, ajusta el
    origen para conservar el centro físico de la imagen.
    """

    if not np.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError("La tolerancia debe ser finita y no negativa")

    validar_geometria_3d(imagen, "imagen para corregir dirección")

    matriz = np.asarray(imagen.GetDirection(), dtype=float).reshape((3, 3))
    if not np.isfinite(matriz).all():
        raise ValueError("La matriz de dirección contiene NaN o infinitos")

    error_ortogonalidad = float(np.max(np.abs(matriz.T @ matriz - np.eye(3))))
    if error_ortogonalidad <= tolerancia:
        return imagen, 0.0

    determinante_original = float(np.linalg.det(matriz))
    if abs(determinante_original) < 1e-10:
        raise ValueError("La matriz de dirección es singular y no se puede corregir")

    u, _, vt = np.linalg.svd(matriz)
    ortogonal = u @ vt
    if np.sign(np.linalg.det(ortogonal)) != np.sign(determinante_original):
        u[:, -1] *= -1.0
        ortogonal = u @ vt

    correccion = float(np.max(np.abs(ortogonal - matriz)))

    size = np.asarray(imagen.GetSize(), dtype=float)
    spacing = np.asarray(imagen.GetSpacing(), dtype=float)
    origin = np.asarray(imagen.GetOrigin(), dtype=float)
    centro_indices_mm = ((size - 1.0) / 2.0) * spacing
    centro_fisico = origin + matriz @ centro_indices_mm
    nuevo_origen = centro_fisico - ortogonal @ centro_indices_mm

    imagen.SetDirection(tuple(float(v) for v in ortogonal.ravel()))
    imagen.SetOrigin(tuple(float(v) for v in nuevo_origen))
    return imagen, correccion


def reorientar_lps(imagen: sitk.Image) -> sitk.Image:
    """Reorienta a LPS preservando las posiciones físicas."""

    validar_geometria_3d(imagen, "imagen antes de LPS")
    orientada = sitk.DICOMOrient(imagen, "LPS")
    validar_geometria_3d(orientada, "imagen LPS")
    return orientada


def centro_fisico(imagen: sitk.Image) -> tuple[float, float, float]:
    indice = [(float(v) - 1.0) / 2.0 for v in imagen.GetSize()]
    punto = imagen.TransformContinuousIndexToPhysicalPoint(indice)
    return tuple(float(v) for v in punto)


def cobertura_fisica(imagen: sitk.Image) -> tuple[float, float, float]:
    return tuple(
        max(0.0, (int(n) - 1) * abs(float(s)))
        for n, s in zip(imagen.GetSize(), imagen.GetSpacing())
    )


def factores_reduccion(imagen: sitk.Image, objetivo_mm: float = 3.0) -> list[int]:
    """Factores de reducción para una rejilla gruesa de ~3 mm."""

    factores: list[int] = []
    for n, spacing in zip(imagen.GetSize(), imagen.GetSpacing()):
        deseado = max(1, int(round(objetivo_mm / float(spacing))))
        # Conserva al menos ~32 muestras por eje cuando sea posible.
        maximo = max(1, int(n) // 32)
        factores.append(max(1, min(deseado, maximo)))
    return factores


def imagen_bordes_gruesa(imagen: sitk.Image) -> sitk.Image:
    """Crea una imagen de bordes suavizada y reducida para la búsqueda gruesa."""

    factores = factores_reduccion(imagen)
    reducida = sitk.Shrink(imagen, factores)

    gradiente = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradiente.SetSigma(2.0)
    bordes = gradiente.Execute(reducida)
    return sitk.RescaleIntensity(bordes, 0.0, 1.0)


def inicializacion_geometrica_rigida(
    ct: sitk.Image,
    cbct: sitk.Image,
) -> sitk.Euler3DTransform:
    transformacion = sitk.CenteredTransformInitializer(
        ct,
        cbct,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    return sitk.Euler3DTransform(transformacion)


def inicializacion_momentos_rigida(
    ct: sitk.Image,
    cbct: sitk.Image,
) -> sitk.Euler3DTransform:
    transformacion = sitk.CenteredTransformInitializer(
        ct,
        cbct,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    return sitk.Euler3DTransform(transformacion)


def _ejecutar_exhaustivo(
    fija: sitk.Image,
    movil: sitk.Image,
    transformacion: sitk.Euler3DTransform,
    numero_pasos: list[int],
    escalas: list[float],
    sampling: float,
    seed: int,
    descripcion: str,
) -> tuple[sitk.Euler3DTransform, float, str]:
    """Ejecuta una búsqueda exhaustiva alrededor de la transformación actual."""

    registro = sitk.ImageRegistrationMethod()
    # En la fase gruesa comparamos estructura (magnitud de bordes), por lo que
    # la correlación resulta útil pese a que CT y CBCT tengan intensidades distintas.
    registro.SetMetricAsCorrelation()
    if sampling < 1.0:
        registro.SetMetricSamplingStrategy(registro.RANDOM)
        registro.SetMetricSamplingPercentage(float(sampling), int(seed))
    else:
        registro.SetMetricSamplingStrategy(registro.NONE)
    registro.SetInterpolator(sitk.sitkLinear)
    registro.SetOptimizerAsExhaustive([int(v) for v in numero_pasos])
    registro.SetOptimizerScales([float(v) for v in escalas])

    tx = sitk.Euler3DTransform(transformacion)
    registro.SetInitialTransform(tx, inPlace=True)
    resultado = registro.Execute(fija, movil)
    tx_resultado = sitk.Euler3DTransform(resultado)

    metrica = float(registro.GetMetricValue())
    condicion = registro.GetOptimizerStopConditionDescription()
    print(
        f"  ✓ {descripcion}: métrica={metrica:.6f} | "
        f"traslación={tuple(round(v, 2) for v in tx_resultado.GetTranslation())}"
    )
    return tx_resultado, metrica, condicion


def recentrar_euler_sin_cambiar_mapeo(
    transformacion: sitk.Euler3DTransform,
    nuevo_centro: tuple[float, float, float],
) -> sitk.Euler3DTransform:
    """Cambia el centro de rotación conservando exactamente el mismo mapeo."""

    r = np.asarray(transformacion.GetMatrix(), dtype=float).reshape(3, 3)
    c = np.asarray(nuevo_centro, dtype=float)
    # Para y = R x + offset, offset es T(0).
    offset = np.asarray(transformacion.TransformPoint((0.0, 0.0, 0.0)), dtype=float)
    traslacion = offset - c + r @ c

    salida = sitk.Euler3DTransform()
    salida.SetComputeZYX(transformacion.GetComputeZYX())
    salida.SetCenter([float(v) for v in c])
    salida.SetMatrix([float(v) for v in r.ravel()])
    salida.SetTranslation([float(v) for v in traslacion])
    return salida


def _delta_transformaciones(
    inicial: sitk.Euler3DTransform,
    final: sitk.Euler3DTransform,
) -> tuple[float, float, float]:
    """Devuelve deriva total de traslación, deriva Z y rotación relativa."""

    ti = np.asarray(inicial.GetTranslation(), dtype=float)
    tf = np.asarray(final.GetTranslation(), dtype=float)
    delta_t = float(np.linalg.norm(tf - ti))
    delta_z = float(abs(tf[2] - ti[2]))

    ri = np.asarray(inicial.GetMatrix(), dtype=float).reshape(3, 3)
    rf = np.asarray(final.GetMatrix(), dtype=float).reshape(3, 3)
    rrel = rf @ ri.T
    cosang = float(np.clip((np.trace(rrel) - 1.0) / 2.0, -1.0, 1.0))
    delta_r = float(math.degrees(math.acos(cosang)))
    return delta_t, delta_z, delta_r


def _resolver_landmarks_dir(valor: str | None) -> Path | None:
    """Resuelve --landmarks-dir sin activar nada de forma implícita.

    Si la ruta es relativa se prueba primero respecto al directorio actual y
    después respecto a la carpeta donde está este script.
    """

    if not valor:
        return None
    candidata = Path(valor).expanduser()
    pruebas = [candidata] if candidata.is_absolute() else [
        Path.cwd() / candidata,
        Path(__file__).resolve().parent / candidata,
    ]
    for prueba in pruebas:
        resuelta = prueba.resolve()
        if resuelta.is_dir():
            return resuelta
    rutas = ", ".join(str(p.resolve()) for p in pruebas)
    raise FileNotFoundError(f"No existe --landmarks-dir. Rutas probadas: {rutas}")


def _leer_markup_lps(ruta: Path) -> dict[int, np.ndarray]:
    """Lee un .mrk.json de 3D Slicer y devuelve puntos físicos en LPS (mm)."""

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de landmarks: {ruta}")
    with ruta.open('r', encoding='utf-8-sig') as fh:
        data = json.load(fh)
    markups = data.get('markups') or []
    if len(markups) != 1:
        raise ValueError(f"{ruta.name}: se esperaba exactamente una lista de Markups")
    markup = markups[0]
    sistema = str(markup.get('coordinateSystem', 'LPS')).upper()
    unidades = str(markup.get('coordinateUnits', 'mm')).lower()
    if unidades not in {'mm', 'millimeter', 'millimeters'}:
        raise ValueError(f"{ruta.name}: unidades no soportadas: {unidades!r}")
    if sistema not in {'LPS', 'RAS'}:
        raise ValueError(f"{ruta.name}: coordinateSystem no soportado: {sistema!r}")

    puntos: dict[int, np.ndarray] = {}
    for cp in markup.get('controlPoints', []):
        if str(cp.get('positionStatus', 'defined')).lower() != 'defined':
            continue
        try:
            pid = int(str(cp.get('id', '')).strip())
        except ValueError:
            continue
        pos = np.asarray(cp.get('position', []), dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"{ruta.name}: posición inválida para P{pid}")
        if sistema == 'RAS':
            pos = np.asarray([-pos[0], -pos[1], pos[2]], dtype=float)
        puntos[pid] = pos
    if not puntos:
        raise ValueError(f"{ruta.name}: no contiene puntos definidos")
    return puntos


def _ajuste_rigido_kabsch(
    puntos_ct: dict[int, np.ndarray],
    puntos_cbct: dict[int, np.ndarray],
    ids: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, float], float, float]:
    """Calcula CT -> CBCT por mínimos cuadrados rígidos, sin escala."""

    if len(ids) < 4:
        raise ValueError("Se necesitan al menos 4 landmarks para un ajuste 3D robusto")
    a = np.vstack([puntos_ct[i] for i in ids]).astype(float)
    b = np.vstack([puntos_cbct[i] for i in ids]).astype(float)
    ca = np.mean(a, axis=0)
    cb = np.mean(b, axis=0)
    h = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    if not np.isfinite(r).all() or abs(float(np.linalg.det(r)) - 1.0) > 1e-5:
        raise RuntimeError("No se pudo obtener una rotación rígida válida")
    offset = cb - r @ ca  # y = R*x + offset, CT -> CBCT
    pred = (r @ a.T).T + offset
    errores_arr = np.linalg.norm(pred - b, axis=1)
    errores = {pid: float(err) for pid, err in zip(ids, errores_arr)}
    rms = float(np.sqrt(np.mean(errores_arr ** 2)))
    max_error = float(np.max(errores_arr))
    return r, offset, errores, rms, max_error


def inicializacion_landmarks_rigida(
    patient_id: str,
    landmarks_dir: Path,
) -> tuple[sitk.Euler3DTransform, dict[str, object]]:
    """Transformación rígida CT -> CBCT a partir de Markups explícitos.

    Protocolo de marcado usado en este TFG (mismos puntos en CT y en CBCT):
      P1 = centro del cuerpo vertebral de referencia
      P2 = centro de la vértebra superior
      P3 = centro de la vértebra inferior
      P4 = apófisis espinosa (se excluye por defecto: fue poco reproducible)
      P5/P6 = apófisis transversas izquierda/derecha

    P1-P3 nunca se eliminan. Si P5 o P6 es claramente discordante, se prueba
    excluir solo uno de ellos y se conserva la solución con menor RMS.
    """

    ruta_ct = landmarks_dir / f"{patient_id}_CT.mrk.json"
    ruta_cbct = landmarks_dir / f"{patient_id}_CBCT.mrk.json"
    puntos_ct = _leer_markup_lps(ruta_ct)
    puntos_cbct = _leer_markup_lps(ruta_cbct)

    comunes = set(puntos_ct) & set(puntos_cbct)
    obligatorios = {1, 2, 3}
    if not obligatorios.issubset(comunes):
        faltan = sorted(obligatorios - comunes)
        raise ValueError(f"Landmarks {patient_id}: faltan P{', P'.join(map(str, faltan))}")

    ids_base = [i for i in (1, 2, 3, 5, 6) if i in comunes]
    if len(ids_base) < 4:
        raise ValueError(
            f"Landmarks {patient_id}: se requieren P1,P2,P3 y al menos uno de P5/P6"
        )

    candidatos: list[tuple[list[int], np.ndarray, np.ndarray, dict[int, float], float, float]] = []
    r, off, errs, rms, mx = _ajuste_rigido_kabsch(puntos_ct, puntos_cbct, ids_base)
    candidatos.append((ids_base, r, off, errs, rms, mx))

    # Solo si el ajuste completo es claramente inconsistente permitimos excluir
    # UNA transversa (P5 o P6). Con un ajuste coherente se conservan todos los
    # puntos; solo se descarta una transversa cuando es un outlier geométrico.
    if rms > 5.0 or mx > 8.0:
        for opcional in (5, 6):
            if opcional in ids_base and len(ids_base) - 1 >= 4:
                ids = [i for i in ids_base if i != opcional]
                rr, oo, ee, rrms, mmx = _ajuste_rigido_kabsch(puntos_ct, puntos_cbct, ids)
                candidatos.append((ids, rr, oo, ee, rrms, mmx))

    # Prioridad: RMS; en empate, más puntos.
    candidatos.sort(key=lambda x: (x[4], -len(x[0])))
    ids_usados, r, offset, errores, rms, max_error = candidatos[0]
    descartados = [i for i in ids_base if i not in ids_usados]

    # Centro en el centroide CT de los puntos usados. SetTranslation se ajusta
    # para conservar exactamente y = R*x + offset.
    c = np.mean(np.vstack([puntos_ct[i] for i in ids_usados]), axis=0)
    traslacion = offset - c + r @ c
    tx = sitk.Euler3DTransform()
    tx.SetCenter([float(v) for v in c])
    tx.SetMatrix([float(v) for v in r.ravel()])
    tx.SetTranslation([float(v) for v in traslacion])

    detalles: dict[str, object] = {
        'ids_usados': ids_usados,
        'ids_descartados': descartados,
        'errores_mm': errores,
        'rms_mm': rms,
        'max_error_mm': max_error,
        'ruta_ct': str(ruta_ct),
        'ruta_cbct': str(ruta_cbct),
    }
    return tx, detalles

def inicializacion_robusta(
    ct: sitk.Image,
    cbct: sitk.Image,
    seed: int,
) -> tuple[sitk.Euler3DTransform, dict[str, object]]:
    """Búsqueda gruesa de posición para CBCT abdominal de FOV parcial.

    Estrategia:
      1) superposición de centros geométricos solo como punto de partida;
      2) búsqueda exhaustiva 1D en superior-inferior;
      3) búsqueda exhaustiva 2D en izquierda-derecha/anterior-posterior;
      4) búsqueda corta de rotaciones alrededor del centro del CBCT ya colocado.
    """

    fija_bordes = imagen_bordes_gruesa(ct)
    movil_bordes = imagen_bordes_gruesa(cbct)

    tx = inicializacion_geometrica_rigida(ct, cbct)
    traslacion_geometria = tuple(float(v) for v in tx.GetTranslation())

    fov_ct = cobertura_fisica(ct)
    fov_cbct = cobertura_fisica(cbct)

    # 1) Localización craneocaudal: el CBCT cubre solo una parte del CT, así
    # que el rango de búsqueda en Z depende de la diferencia de cobertura.
    rango_z = max(60.0, max(0.0, fov_ct[2] - fov_cbct[2]) / 2.0 + 30.0)
    rango_z = min(rango_z, 250.0)
    paso_z = 10.0
    n_z = int(math.ceil(rango_z / paso_z))
    tx, metrica_z, parada_z = _ejecutar_exhaustivo(
        fija_bordes,
        movil_bordes,
        tx,
        numero_pasos=[0, 0, 0, 0, 0, n_z],
        escalas=[1.0, 1.0, 1.0, 1.0, 1.0, paso_z],
        sampling=0.20,
        seed=seed,
        descripcion=f"búsqueda gruesa Z (±{n_z * paso_z:.0f} mm)",
    )

    # 2) Tras resolver la localización craneocaudal, afinamos X e Y por separado.
    # Esto permite cubrir todo el desplazamiento posible de un FOV parcial sin
    # pagar el coste cuadrático de una rejilla XY completa.
    rangos_xy: list[float] = []
    for eje in (0, 1):
        rango = max(30.0, max(0.0, fov_ct[eje] - fov_cbct[eje]) / 2.0 + 15.0)
        rangos_xy.append(min(rango, 90.0))
    paso_xy = 10.0
    n_x = int(math.ceil(rangos_xy[0] / paso_xy))
    n_y = int(math.ceil(rangos_xy[1] / paso_xy))

    tx, metrica_x, parada_x = _ejecutar_exhaustivo(
        fija_bordes,
        movil_bordes,
        tx,
        numero_pasos=[0, 0, 0, n_x, 0, 0],
        escalas=[1.0, 1.0, 1.0, paso_xy, 1.0, 1.0],
        sampling=0.20,
        seed=seed + 1,
        descripcion=f"búsqueda gruesa X (±{n_x * paso_xy:.0f} mm)",
    )

    tx, metrica_y, parada_y = _ejecutar_exhaustivo(
        fija_bordes,
        movil_bordes,
        tx,
        numero_pasos=[0, 0, 0, 0, n_y, 0],
        escalas=[1.0, 1.0, 1.0, 1.0, paso_xy, 1.0],
        sampling=0.20,
        seed=seed + 2,
        descripcion=f"búsqueda gruesa Y (±{n_y * paso_xy:.0f} mm)",
    )

    # 3) Rotar alrededor de la posición actual del centro del CBCT expresada en CT.
    centro_cbct_en_ct = tx.GetInverse().TransformPoint(centro_fisico(cbct))
    tx = recentrar_euler_sin_cambiar_mapeo(
        tx,
        tuple(float(v) for v in centro_cbct_en_ct),
    )

    paso_rot = math.radians(3.0)
    n_rot = 2  # -6, -3, 0, +3, +6 grados por eje
    tx, metrica_rot, parada_rot = _ejecutar_exhaustivo(
        fija_bordes,
        movil_bordes,
        tx,
        numero_pasos=[n_rot, n_rot, n_rot, 0, 0, 0],
        escalas=[paso_rot, paso_rot, paso_rot, 1.0, 1.0, 1.0],
        sampling=0.20,
        seed=seed + 3,
        descripcion="búsqueda gruesa de rotación (±6°)",
    )

    detalles: dict[str, object] = {
        "traslacion_geometria": traslacion_geometria,
        "traslacion_coarse": tuple(float(v) for v in tx.GetTranslation()),
        "rango_busqueda_z_mm": n_z * paso_z,
        "rango_busqueda_x_mm": n_x * paso_xy,
        "rango_busqueda_y_mm": n_y * paso_xy,
        "metrica_coarse_z": metrica_z,
        "metrica_coarse_x": metrica_x,
        "metrica_coarse_y": metrica_y,
        "metrica_coarse_rot": metrica_rot,
        "parada_coarse_z": parada_z,
        "parada_coarse_x": parada_x,
        "parada_coarse_y": parada_y,
        "parada_coarse_rot": parada_rot,
    }
    return tx, detalles


def crear_mascara_huella_cbct(
    ct: sitk.Image,
    cbct: sitk.Image,
    transformacion: sitk.Transform,
) -> sitk.Image:
    """Huella rectangular del CBCT en la rejilla del CT (1=dentro del CBCT)."""

    unos = sitk.Image(cbct.GetSize(), sitk.sitkUInt8)
    unos.CopyInformation(cbct)
    unos = unos + 1

    mascara = sitk.Resample(
        unos,
        ct,
        transformacion,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk.Cast(mascara > 0, sitk.sitkUInt8)


def dilatar_mascara_mm(mascara: sitk.Image, margen_mm: float) -> sitk.Image:
    if margen_mm <= 0:
        return mascara
    radios = [
        max(1, int(math.ceil(margen_mm / float(s))))
        for s in mascara.GetSpacing()
    ]
    filtro = sitk.BinaryDilateImageFilter()
    filtro.SetKernelRadius(radios)
    filtro.SetForegroundValue(1)
    return filtro.Execute(mascara)


def estadisticas_huella(
    mascara: sitk.Image,
    cbct: sitk.Image,
) -> tuple[float, tuple[int, int, int, int, int, int]]:
    """Devuelve porcentaje aproximado del volumen CBCT dentro del CT y bbox."""

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mascara)
    if not stats.HasLabel(1):
        raise RuntimeError("La huella del CBCT no intersecta el CT")

    n_pixeles = float(stats.GetNumberOfPixels(1))
    voxel_ct = float(np.prod(np.asarray(mascara.GetSpacing(), dtype=float)))
    volumen_interseccion = n_pixeles * voxel_ct

    volumen_cbct = float(
        np.prod(np.asarray(cbct.GetSize(), dtype=float))
        * np.prod(np.asarray(cbct.GetSpacing(), dtype=float))
    )
    if volumen_cbct <= 0 or not np.isfinite(volumen_cbct):
        raise RuntimeError("Volumen físico CBCT no válido")

    porcentaje = 100.0 * volumen_interseccion / volumen_cbct
    porcentaje = max(0.0, min(100.0, porcentaje))
    bbox = tuple(int(v) for v in stats.GetBoundingBox(1))
    return porcentaje, bbox


def configurar_registro_fino(
    transformacion_inicial: sitk.Transform,
    sampling: float,
    seed: int,
    iteraciones: int,
    mascara_fija: sitk.Image | None,
    learning_rate: float = 1.0,
    pesos_optimizador: list[float] | None = None,
) -> sitk.ImageRegistrationMethod:
    """Mattes Mutual Information multirresolución para el refinamiento final."""

    if not np.isfinite(sampling) or not 0.0 < sampling <= 1.0:
        raise ValueError("El porcentaje de muestreo debe ser finito y estar en (0, 1]")
    if iteraciones < 1:
        raise ValueError("El número de iteraciones debe ser positivo")

    registro = sitk.ImageRegistrationMethod()
    registro.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registro.SetMetricSamplingStrategy(registro.RANDOM)
    registro.SetMetricSamplingPercentage(float(sampling), int(seed))
    if mascara_fija is not None:
        registro.SetMetricFixedMask(mascara_fija)
    registro.SetInterpolator(sitk.sitkLinear)

    registro.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1e-4,
        numberOfIterations=int(iteraciones),
        gradientMagnitudeTolerance=1e-8,
    )
    registro.SetOptimizerScalesFromPhysicalShift()
    if pesos_optimizador is not None:
        registro.SetOptimizerWeights([float(v) for v in pesos_optimizador])
    registro.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registro.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registro.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registro.SetInitialTransform(transformacion_inicial, inPlace=True)
    return registro


def convertir_a_similitud(
    rigida: sitk.Euler3DTransform,
) -> sitk.Similarity3DTransform:
    """Inicializa una Similarity3DTransform con la misma rígida y escala 1."""

    similitud = sitk.Similarity3DTransform()
    similitud.SetCenter([float(v) for v in rigida.GetCenter()])
    similitud.SetMatrix([float(v) for v in rigida.GetMatrix()])
    similitud.SetTranslation([float(v) for v in rigida.GetTranslation()])
    similitud.SetScale(1.0)
    return similitud


def aplicar_transformacion(
    ct: sitk.Image,
    cbct: sitk.Image,
    transformacion: sitk.Transform,
    interpolador_salida: str,
) -> sitk.Image:
    """Remuestrea el CBCT sobre la rejilla física del CT."""

    if interpolador_salida == "lineal":
        interpolador = sitk.sitkLinear
    elif interpolador_salida == "vecino":
        interpolador = sitk.sitkNearestNeighbor
    else:
        raise ValueError(f"Interpolador no soportado: {interpolador_salida}")

    return sitk.Resample(
        cbct,
        ct,
        transformacion,
        interpolador,
        -1024.0,
        sitk.sitkFloat32,
    )


def bbox_con_margen(
    bbox: tuple[int, int, int, int, int, int],
    referencia: sitk.Image,
    margen_mm: float,
) -> tuple[list[int], list[int]]:
    """Convierte bbox (índice+tamaño) en ROI ampliada y limitada a la imagen."""

    indices = [bbox[0], bbox[1], bbox[2]]
    tamanos = [bbox[3], bbox[4], bbox[5]]
    size_ref = [int(v) for v in referencia.GetSize()]
    spacing = [float(v) for v in referencia.GetSpacing()]

    inicio: list[int] = []
    fin: list[int] = []
    for eje in range(3):
        margen_vox = int(math.ceil(max(0.0, margen_mm) / spacing[eje]))
        ini = max(0, indices[eje] - margen_vox)
        end = min(size_ref[eje], indices[eje] + tamanos[eje] + margen_vox)
        inicio.append(int(ini))
        fin.append(int(end))

    size_roi = [fin[eje] - inicio[eje] for eje in range(3)]
    if any(v < 2 for v in size_roi):
        raise RuntimeError(f"ROI final degenerada: inicio={inicio}, tamaño={size_roi}")
    return inicio, size_roi


def recortar_par_a_huella(
    ct: sitk.Image,
    cbct_registrado: sitk.Image,
    mascara_huella: sitk.Image,
    margen_mm: float,
) -> tuple[sitk.Image, sitk.Image, tuple[int, int, int, int, int, int]]:
    """Recorta CT y CBCT registrado exactamente a la misma ROI física."""

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mascara_huella)
    if not stats.HasLabel(1):
        raise RuntimeError("No se puede recortar: la huella CBCT está vacía")

    bbox = tuple(int(v) for v in stats.GetBoundingBox(1))
    inicio, size_roi = bbox_con_margen(bbox, ct, margen_mm)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(inicio)
    roi.SetSize(size_roi)
    ct_roi = roi.Execute(ct)
    cbct_roi = roi.Execute(cbct_registrado)

    if not geometrias_coinciden(ct_roi, cbct_roi):
        raise RuntimeError(
            "CT y CBCT dejaron de compartir geometría al recortar: "
            + "; ".join(diferencias_geometria(ct_roi, cbct_roi))
        )
    return ct_roi, cbct_roi, bbox


def _texto_vector(valores: object) -> str:
    if isinstance(valores, (tuple, list)):
        return ";".join(f"{float(v):.6g}" for v in valores)
    return str(valores)


def registrar_paciente(
    patient_id: str,
    nifti_root: Path,
    output_root: Path,
    tipo_transformacion: str,
    inicializacion: str,
    interpolador_salida: str,
    sampling: float,
    seed: int,
    iteraciones: int,
    reset_cbct_origin: bool,
    margen_roi_mm: float,
    sobrescribir: bool,
) -> dict[str, object]:
    print(f"\n{'=' * 72}")
    print(f"Registrando paciente: {patient_id}")
    print(f"{'=' * 72}")

    nifti_dir = nifti_root / patient_id
    output_dir = output_root / patient_id

    # Se registra contra el CT COMPLETO del paso 1: recortarlo antes podría
    # eliminar la zona hepática sin saber todavía dónde cae el CBCT.
    ruta_ct = nifti_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct = nifti_dir / f"{patient_id}_cbct.nii.gz"
    ruta_ct_salida = output_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct_salida = output_dir / f"{patient_id}_cbct_registered.nii.gz"
    ruta_transformacion = output_dir / f"{patient_id}_transform.tfm"

    resultado: dict[str, object] = {
        "patient_id": patient_id,
        "estado": "ERROR",
        "version_registro": VERSION_REGISTRO,
        "transformacion": tipo_transformacion,
        "inicializacion": inicializacion,
        "interpolador_salida": interpolador_salida,
        "sampling": sampling,
        "seed": seed,
        "iteraciones_max_por_nivel": iteraciones,
        "iteraciones_ejecutadas": "",
        "metrica_final": "",
        "solapamiento_cbct_pct": "",
        "rango_busqueda_z_mm": "",
        "rango_busqueda_x_mm": "",
        "rango_busqueda_y_mm": "",
        "traslacion_geometria": "",
        "traslacion_coarse": "",
        "traslacion_final": "",
        "angulos_finales_grados": "",
        "correccion_direccion_ct": "",
        "correccion_direccion_cbct": "",
        "condicion_parada": "",
        "roi_final_size": "",
        "delta_refinamiento_mm": "",
        "delta_refinamiento_z_mm": "",
        "delta_rotacion_refinamiento_grados": "",
        "mensaje": "",
    }

    faltantes = [ruta for ruta in (ruta_ct, ruta_cbct) if not ruta.is_file()]
    if faltantes:
        resultado["mensaje"] = "Faltan entradas: " + ", ".join(r.name for r in faltantes)
        print(f"  ✗ {resultado['mensaje']}")
        return resultado

    salidas = (ruta_ct_salida, ruta_cbct_salida, ruta_transformacion)
    if all(ruta.exists() for ruta in salidas) and not sobrescribir:
        try:
            ct_existente = sitk.ReadImage(str(ruta_ct_salida))
            cbct_existente = sitk.ReadImage(str(ruta_cbct_salida))
            validar_geometria_3d(ct_existente, "CT registrado existente")
            validar_geometria_3d(cbct_existente, "CBCT registrado existente")
            if not geometrias_coinciden(ct_existente, cbct_existente):
                raise ValueError(
                    "Las salidas existentes no comparten geometría: "
                    + "; ".join(diferencias_geometria(ct_existente, cbct_existente))
                )
            sitk.ReadTransform(str(ruta_transformacion))
        except Exception as exc:
            resultado["mensaje"] = (
                "Las salidas existentes no son válidas; use --sobrescribir: "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"  ✗ {resultado['mensaje']}")
            return resultado
        resultado.update(
            estado="OMITIDO",
            mensaje=(
                "Salidas existentes validadas; no se regeneran. "
                "Ejecute con --sobrescribir para rehacerlas."
            ),
        )
        print("  ↷ Las salidas ya existen; se omite. Usa --sobrescribir para regenerar.")
        return resultado

    try:
        ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
        cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
        validar_geometria_3d(ct, "CT")
        validar_geometria_3d(cbct, "CBCT")

        print(
            f"  CT original   tamaño={ct.GetSize()} spacing={ct.GetSpacing()} "
            f"origen={ct.GetOrigin()}"
        )
        print(
            f"  CBCT original tamaño={cbct.GetSize()} spacing={cbct.GetSpacing()} "
            f"origen={cbct.GetOrigin()}"
        )

        ct, correccion_ct = ortonormalizar_direccion(ct)
        cbct, correccion_cbct = ortonormalizar_direccion(cbct)
        resultado["correccion_direccion_ct"] = correccion_ct
        resultado["correccion_direccion_cbct"] = correccion_cbct

        # Se llevan ambos volúmenes a orientación LPS: hay CBCT que almacenan
        # superior-inferior en un eje de índice distinto de Z.
        ct = reorientar_lps(ct)
        cbct = reorientar_lps(cbct)

        print(f"  CT   LPS tamaño={ct.GetSize()} spacing={ct.GetSpacing()}")
        print(f"  CBCT LPS tamaño={cbct.GetSize()} spacing={cbct.GetSpacing()}")
        print(
            "  Cobertura física LPS CT/CBCT (mm): "
            f"{tuple(round(v, 1) for v in cobertura_fisica(ct))} / "
            f"{tuple(round(v, 1) for v in cobertura_fisica(cbct))}"
        )

        if reset_cbct_origin:
            cbct.SetOrigin((0.0, 0.0, 0.0))
            print(
                "  ⚠ Origen CBCT forzado a (0,0,0) por petición explícita. "
                "Normalmente no es necesario."
            )

        if inicializacion == "robusta":
            tx_rigida, detalles_coarse = inicializacion_robusta(ct, cbct, seed)
        elif inicializacion == "geometria":
            tx_rigida = inicializacion_geometrica_rigida(ct, cbct)
            detalles_coarse = {
                "traslacion_geometria": tuple(tx_rigida.GetTranslation()),
                "traslacion_coarse": tuple(tx_rigida.GetTranslation()),
            }
            print("  ⚠ Inicialización geométrica simple solicitada explícitamente.")
        elif inicializacion == "momentos":
            tx_rigida = inicializacion_momentos_rigida(ct, cbct)
            detalles_coarse = {
                "traslacion_geometria": "",
                "traslacion_coarse": tuple(tx_rigida.GetTranslation()),
            }
            print("  ⚠ Inicialización por momentos solicitada explícitamente.")
        else:
            raise ValueError(f"Inicialización no soportada: {inicializacion}")

        for clave in (
            "rango_busqueda_z_mm",
            "rango_busqueda_x_mm",
            "rango_busqueda_y_mm",
        ):
            if clave in detalles_coarse:
                resultado[clave] = detalles_coarse[clave]
        resultado["traslacion_geometria"] = _texto_vector(
            detalles_coarse.get("traslacion_geometria", "")
        )
        resultado["traslacion_coarse"] = _texto_vector(
            detalles_coarse.get("traslacion_coarse", "")
        )

        # La máscara fija se centra en la huella estimada del CBCT, pero se
        # dilata para permitir que el refinamiento se mueva varias decenas de mm.
        huella_coarse = crear_mascara_huella_cbct(ct, cbct, tx_rigida)
        mascara_fija = dilatar_mascara_mm(huella_coarse, 35.0)

        if tipo_transformacion == "rigida":
            tx_refinar: sitk.Transform = sitk.Euler3DTransform(tx_rigida)
        elif tipo_transformacion == "similitud":
            tx_refinar = convertir_a_similitud(tx_rigida)
        else:
            raise ValueError(f"Transformación no soportada: {tipo_transformacion}")

        registro = configurar_registro_fino(
            tx_refinar,
            sampling=sampling,
            seed=seed + 10,
            iteraciones=iteraciones,
            mascara_fija=mascara_fija,
        )

        contador = {"n": 0}

        def progreso() -> None:
            contador["n"] += 1
            if contador["n"] % 25 == 0:
                print(
                    f"    refinamiento iteración {contador['n']:>4} | "
                    f"métrica={registro.GetMetricValue():.6f}"
                )

        registro.AddCommand(sitk.sitkIterationEvent, progreso)
        transformacion_final = registro.Execute(ct, cbct)

        metrica = float(registro.GetMetricValue())
        if not np.isfinite(metrica):
            raise RuntimeError(f"La métrica final no es finita: {metrica}")
        condicion = registro.GetOptimizerStopConditionDescription()

        # Cuánto movió el refinamiento MI respecto a la búsqueda gruesa (informe).
        if tipo_transformacion == "rigida":
            tx_candidata = sitk.Euler3DTransform(transformacion_final)
            delta_t, delta_z, delta_r = _delta_transformaciones(tx_rigida, tx_candidata)
            resultado["delta_refinamiento_mm"] = delta_t
            resultado["delta_refinamiento_z_mm"] = delta_z
            resultado["delta_rotacion_refinamiento_grados"] = delta_r
        
        resultado.update(
            metrica_final=metrica,
            iteraciones_ejecutadas=contador["n"],
            condicion_parada=condicion,
        )
        print(f"  ✓ Refinamiento final | métrica={metrica:.6f}")
        print(f"    parada: {condicion}")

        # Convertimos solo para obtener diagnósticos si es una rígida.
        if tipo_transformacion == "rigida":
            tx_diag = sitk.Euler3DTransform(transformacion_final)
            resultado["traslacion_final"] = _texto_vector(tx_diag.GetTranslation())
            resultado["angulos_finales_grados"] = _texto_vector(
                tuple(
                    math.degrees(v)
                    for v in (
                        tx_diag.GetAngleX(),
                        tx_diag.GetAngleY(),
                        tx_diag.GetAngleZ(),
                    )
                )
            )
        else:
            tx_sim = sitk.Similarity3DTransform(transformacion_final)
            resultado["traslacion_final"] = _texto_vector(tx_sim.GetTranslation())
            resultado["angulos_finales_grados"] = "similarity3d"

        cbct_registrado_full = aplicar_transformacion(
            ct,
            cbct,
            transformacion_final,
            interpolador_salida,
        )
        if not geometrias_coinciden(ct, cbct_registrado_full):
            raise RuntimeError(
                "La geometría del CBCT remuestreado no coincide con el CT: "
                + "; ".join(diferencias_geometria(ct, cbct_registrado_full))
            )

        huella_final = crear_mascara_huella_cbct(ct, cbct, transformacion_final)
        solapamiento_pct, _bbox = estadisticas_huella(huella_final, cbct)
        resultado["solapamiento_cbct_pct"] = solapamiento_pct
        print(f"  ✓ CBCT dentro del CT ≈ {solapamiento_pct:.1f}% del volumen rectangular")

        # El par que continúa por el pipeline se limita a la zona donde existe
        # CBCT, con un pequeño margen común (--margen-roi-mm).
        ct_salida, cbct_salida, _ = recortar_par_a_huella(
            ct,
            cbct_registrado_full,
            huella_final,
            margen_mm=margen_roi_mm,
        )
        resultado["roi_final_size"] = "x".join(str(v) for v in ct_salida.GetSize())

        escribir_grupo_atomico(
            [
                (
                    ruta_ct_salida,
                    lambda temporal: sitk.WriteImage(ct_salida, str(temporal), True),
                ),
                (
                    ruta_cbct_salida,
                    lambda temporal: sitk.WriteImage(cbct_salida, str(temporal), True),
                ),
                (
                    ruta_transformacion,
                    lambda temporal: sitk.WriteTransform(
                        transformacion_final,
                        str(temporal),
                    ),
                ),
            ]
        )

        avisos: list[str] = []
        if solapamiento_pct < 70.0:
            avisos.append(f"solapamiento geométrico bajo ({solapamiento_pct:.1f}%)")
        if "Maximum number of iterations" in condicion:
            avisos.append("el optimizador alcanzó el máximo de iteraciones")

        if avisos:
            resultado.update(
                estado="REVISAR",
                mensaje="; ".join(avisos) + ". Revisar en 3D Slicer.",
            )
            print(f"  ⚠ Resultado marcado REVISAR: {resultado['mensaje']}")
        else:
            resultado.update(estado="OK", mensaje="Registro completado")
            print("  ✓ Registro completado sin avisos automáticos.")

        print(f"  ✓ ROI final CT/CBCT: {ct_salida.GetSize()}")
        print(f"  ✓ Salidas guardadas en {output_dir}")

    except Exception as exc:
        resultado["mensaje"] = f"{type(exc).__name__}: {exc}"
        resultado["estado"] = "ERROR"
        print(f"  ✗ {resultado['mensaje']}")
        return resultado

    return resultado



def registrar_paciente_landmarks(
    patient_id: str,
    nifti_root: Path,
    output_root: Path,
    landmarks_dir: Path,
    interpolador_salida: str,
    margen_roi_mm: float,
    sobrescribir: bool,
) -> dict[str, object]:
    """Modo landmarks: transformación rígida a partir de los puntos de Slicer.

    No ejecuta la búsqueda automática ni el refinamiento por Mutual Information.
    Aplica la misma preparación (ortonormalización, LPS) y el mismo recorte
    final que el modo automático, y escribe los mismos archivos de salida.
    """

    print(f"\n{'=' * 72}")
    print(f"Registrando paciente con LANDMARKS: {patient_id}")
    print(f"{'=' * 72}")

    nifti_dir = nifti_root / patient_id
    output_dir = output_root / patient_id
    ruta_ct = nifti_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct = nifti_dir / f"{patient_id}_cbct.nii.gz"
    ruta_ct_salida = output_dir / f"{patient_id}_ct.nii.gz"
    ruta_cbct_salida = output_dir / f"{patient_id}_cbct_registered.nii.gz"
    ruta_transformacion = output_dir / f"{patient_id}_transform.tfm"

    resultado: dict[str, object] = {
        'patient_id': patient_id,
        'estado': 'ERROR',
        'version_registro': VERSION_LANDMARKS,
        'modo_registro': 'LANDMARKS_EXPLICITO',
        'transformacion': 'rigida',
        'inicializacion': 'landmarks',
        'interpolador_salida': interpolador_salida,
        'sampling': '',
        'seed': '',
        'iteraciones_max_por_nivel': 0,
        'iteraciones_ejecutadas': 0,
        'metrica_final': '',
        'solapamiento_cbct_pct': '',
        'rango_busqueda_z_mm': '',
        'rango_busqueda_x_mm': '',
        'rango_busqueda_y_mm': '',
        'traslacion_geometria': '',
        'traslacion_coarse': '',
        'traslacion_final': '',
        'angulos_finales_grados': '',
        'correccion_direccion_ct': '',
        'correccion_direccion_cbct': '',
        'condicion_parada': 'LANDMARKS_ONLY_SIN_MI',
        'roi_final_size': '',
        'delta_refinamiento_mm': '',
        'delta_refinamiento_z_mm': '',
        'delta_rotacion_refinamiento_grados': '',
        'landmarks_rms_mm': '',
        'landmarks_max_error_mm': '',
        'landmarks_usados': '',
        'landmarks_descartados': '',
        'mensaje': '',
    }

    faltantes = [p for p in (ruta_ct, ruta_cbct) if not p.is_file()]
    if faltantes:
        resultado['mensaje'] = 'Faltan entradas: ' + ', '.join(p.name for p in faltantes)
        print(f"  ✗ {resultado['mensaje']}")
        return resultado

    for p in (
        landmarks_dir / f"{patient_id}_CT.mrk.json",
        landmarks_dir / f"{patient_id}_CBCT.mrk.json",
    ):
        if not p.is_file():
            resultado['mensaje'] = f"Falta landmark requerido: {p}"
            print(f"  ✗ {resultado['mensaje']}")
            return resultado

    salidas = (ruta_ct_salida, ruta_cbct_salida, ruta_transformacion)
    if all(p.exists() for p in salidas) and not sobrescribir:
        resultado.update(estado='OMITIDO', mensaje='Salidas existentes; use --sobrescribir')
        print('  ↷ Salidas existentes; se omite. Usa --sobrescribir para regenerar.')
        return resultado

    try:
        ct = sitk.ReadImage(str(ruta_ct), sitk.sitkFloat32)
        cbct = sitk.ReadImage(str(ruta_cbct), sitk.sitkFloat32)
        validar_geometria_3d(ct, 'CT')
        validar_geometria_3d(cbct, 'CBCT')
        ct, correccion_ct = ortonormalizar_direccion(ct)
        cbct, correccion_cbct = ortonormalizar_direccion(cbct)
        resultado['correccion_direccion_ct'] = correccion_ct
        resultado['correccion_direccion_cbct'] = correccion_cbct
        ct = reorientar_lps(ct)
        cbct = reorientar_lps(cbct)

        print(f"  CT   LPS tamaño={ct.GetSize()} spacing={ct.GetSpacing()}")
        print(f"  CBCT LPS tamaño={cbct.GetSize()} spacing={cbct.GetSpacing()}")

        tx, detalles = inicializacion_landmarks_rigida(patient_id, landmarks_dir)
        ids_usados = list(detalles['ids_usados'])
        ids_descartados = list(detalles['ids_descartados'])
        rms = float(detalles['rms_mm'])
        max_error = float(detalles['max_error_mm'])
        errores = dict(detalles['errores_mm'])

        resultado['landmarks_rms_mm'] = rms
        resultado['landmarks_max_error_mm'] = max_error
        resultado['landmarks_usados'] = ';'.join(str(i) for i in ids_usados)
        resultado['landmarks_descartados'] = ';'.join(str(i) for i in ids_descartados)
        resultado['traslacion_coarse'] = _texto_vector(tx.GetTranslation())
        resultado['traslacion_final'] = _texto_vector(tx.GetTranslation())
        resultado['angulos_finales_grados'] = _texto_vector(
            tuple(math.degrees(v) for v in (tx.GetAngleX(), tx.GetAngleY(), tx.GetAngleZ()))
        )

        print('  ✓ Modo LANDMARKS explícito: no se ejecuta búsqueda automática ni MI.')
        print('  ✓ Puntos usados: ' + ', '.join(f'P{i}' for i in ids_usados))
        if ids_descartados:
            print('  ⚠ Puntos descartados por inconsistencia: ' + ', '.join(f'P{i}' for i in ids_descartados))
        print(f"  ✓ Error landmarks: RMS={rms:.2f} mm | máximo={max_error:.2f} mm")
        print('    residuales: ' + ', '.join(f"P{i}={errores[i]:.2f} mm" for i in ids_usados))

        cbct_registrado_full = aplicar_transformacion(ct, cbct, tx, interpolador_salida)
        if not geometrias_coinciden(ct, cbct_registrado_full):
            raise RuntimeError(
                'La geometría del CBCT remuestreado no coincide con el CT: '
                + '; '.join(diferencias_geometria(ct, cbct_registrado_full))
            )

        huella_final = crear_mascara_huella_cbct(ct, cbct, tx)
        solapamiento_pct, _bbox = estadisticas_huella(huella_final, cbct)
        resultado['solapamiento_cbct_pct'] = solapamiento_pct
        print(f"  ✓ CBCT dentro del CT ≈ {solapamiento_pct:.1f}% del volumen rectangular")

        ct_salida, cbct_salida, _ = recortar_par_a_huella(
            ct, cbct_registrado_full, huella_final, margen_mm=margen_roi_mm
        )
        resultado['roi_final_size'] = 'x'.join(str(v) for v in ct_salida.GetSize())

        escribir_grupo_atomico(
            [
                (ruta_ct_salida, lambda temporal: sitk.WriteImage(ct_salida, str(temporal), True)),
                (ruta_cbct_salida, lambda temporal: sitk.WriteImage(cbct_salida, str(temporal), True)),
                (ruta_transformacion, lambda temporal: sitk.WriteTransform(tx, str(temporal))),
            ]
        )

        avisos: list[str] = []
        if rms > 6.0 or max_error > 10.0:
            avisos.append(f"landmarks con error elevado (RMS {rms:.1f} mm, máx {max_error:.1f} mm)")
        if solapamiento_pct < 70.0:
            avisos.append(f"solapamiento geométrico bajo ({solapamiento_pct:.1f}%)")
        if ids_descartados:
            avisos.append('se descartó ' + ','.join(f'P{i}' for i in ids_descartados))
        avisos.append('validación visual obligatoria en 3D Slicer')
        resultado.update(estado='REVISAR', mensaje='; '.join(avisos))
        print(f"  ⚠ Resultado LANDMARKS marcado REVISAR: {resultado['mensaje']}")
        print(f"  ✓ ROI final CT/CBCT: {ct_salida.GetSize()}")
        print(f"  ✓ Salidas guardadas en {output_dir}")
    except Exception as exc:
        resultado['mensaje'] = f"{type(exc).__name__}: {exc}"
        resultado['estado'] = 'ERROR'
        print(f"  ✗ {resultado['mensaje']}")
    return resultado

def guardar_informe(output_root: Path, resultados: list[dict[str, object]]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    ruta = output_root / "informe_registro.csv"
    campos = [
        "patient_id",
        "estado",
        "version_registro",
        "modo_registro",
        "transformacion",
        "inicializacion",
        "interpolador_salida",
        "sampling",
        "seed",
        "iteraciones_max_por_nivel",
        "iteraciones_ejecutadas",
        "metrica_final",
        "solapamiento_cbct_pct",
        "rango_busqueda_z_mm",
        "rango_busqueda_x_mm",
        "rango_busqueda_y_mm",
        "traslacion_geometria",
        "traslacion_coarse",
        "traslacion_final",
        "angulos_finales_grados",
        "correccion_direccion_ct",
        "correccion_direccion_cbct",
        "condicion_parada",
        "roi_final_size",
        "delta_refinamiento_mm",
        "delta_refinamiento_z_mm",
        "delta_rotacion_refinamiento_grados",
        "landmarks_rms_mm",
        "landmarks_max_error_mm",
        "landmarks_usados",
        "landmarks_descartados",
        "mensaje",
    ]
    guardar_informe_acumulado(ruta, resultados, campos)
    return ruta


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paso 2: registro robusto CBCT -> CT abdominal"
    )
    parser.add_argument("--project-dir", help="Ruta de la CARPETA DEL PROYECTO")
    parser.add_argument(
        "--pacientes",
        nargs="+",
        help="Filtro opcional. Sin él se detectan todos en PACIENTES_NIFTI.",
    )
    parser.add_argument(
        "--transformacion",
        choices=("rigida", "similitud"),
        default="rigida",
        help="Rígida recomendada. Similitud solo para experimentos explícitos.",
    )
    parser.add_argument(
        "--inicializacion",
        choices=("robusta", "geometria", "momentos"),
        default="robusta",
        help=(
            "robusta: búsqueda gruesa Z/XY/rotación + refinamiento (recomendada). "
            "geometria/momentos se conservan para comparación experimental."
        ),
    )
    parser.add_argument(
        "--interpolador-salida",
        choices=("lineal", "vecino"),
        default="lineal",
        help="Lineal para intensidades; vecino solo para pruebas específicas.",
    )
    parser.add_argument("--sampling", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iteraciones", type=int, default=250)
    parser.add_argument(
        "--margen-roi-mm",
        type=float,
        default=5.0,
        help="Margen común alrededor de la huella del CBCT tras registrar.",
    )
    parser.add_argument(
        "--reset-cbct-origin",
        action="store_true",
        help=(
            "Fuerza origen (0,0,0). Normalmente NO usar: el registro robusto "
            "tolera sistemas físicos distintos."
        ),
    )
    parser.add_argument(
        "--landmarks-dir",
        help=(
            "Activa EXPLÍCITAMENTE el modo landmarks para los pacientes seleccionados. "
            "La carpeta debe contener XXXX_CT.mrk.json y XXXX_CBCT.mrk.json. "
            "Si se omite, el registro automático se conserva sin cambios."
        ),
    )
    parser.add_argument("--sobrescribir", action="store_true")
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    if not np.isfinite(args.sampling) or not 0.0 < args.sampling <= 1.0:
        print("ERROR: --sampling debe ser finito y estar en (0, 1]")
        return 2
    if not 0 <= args.seed <= 4_294_967_000:
        # Dejamos margen para seed+10 usado en las etapas internas.
        print("ERROR: --seed debe estar entre 0 y 4294967000")
        return 2
    if args.iteraciones < 1:
        print("ERROR: --iteraciones debe ser positivo")
        return 2
    if not np.isfinite(args.margen_roi_mm) or args.margen_roi_mm < 0:
        print("ERROR: --margen-roi-mm debe ser finito y no negativo")
        return 2

    try:
        landmarks_dir = _resolver_landmarks_dir(args.landmarks_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    rutas = obtener_rutas(args.project_dir)
    try:
        pacientes = detectar_pacientes(rutas.nifti, args.pacientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not pacientes:
        print(f"ERROR: no hay pacientes con formato XXXX en {rutas.nifti}")
        return 2

    print(f"Pacientes detectados: {texto_pacientes(pacientes)}")
    print(f"Transformación: {args.transformacion}")
    print(f"Inicialización: {args.inicializacion}")
    print(f"Salida: {rutas.registered}")

    if landmarks_dir is not None:
        print(f"Modo LANDMARKS EXPLÍCITO: {landmarks_dir}")
        print("No se ejecutará la búsqueda automática ni Mutual Information en estos pacientes.")
        resultados = [
            registrar_paciente_landmarks(
                patient_id=pid,
                nifti_root=rutas.nifti,
                output_root=rutas.registered,
                landmarks_dir=landmarks_dir,
                interpolador_salida=args.interpolador_salida,
                margen_roi_mm=args.margen_roi_mm,
                sobrescribir=args.sobrescribir,
            )
            for pid in pacientes
        ]
    else:
        # Modo automático.
        resultados = [
            registrar_paciente(
                patient_id=pid,
                nifti_root=rutas.nifti,
                output_root=rutas.registered,
                tipo_transformacion=args.transformacion,
                inicializacion=args.inicializacion,
                interpolador_salida=args.interpolador_salida,
                sampling=args.sampling,
                seed=args.seed,
                iteraciones=args.iteraciones,
                reset_cbct_origin=args.reset_cbct_origin,
                margen_roi_mm=args.margen_roi_mm,
                sobrescribir=args.sobrescribir,
            )
            for pid in pacientes
        ]
    informe = guardar_informe(rutas.registered, resultados)

    errores = [fila for fila in resultados if fila["estado"] == "ERROR"]
    revisar = [fila for fila in resultados if fila["estado"] == "REVISAR"]
    print(f"\nPaso 2 terminado. Informe: {informe}")
    if revisar:
        print("Pacientes que requieren revisión visual:")
        for fila in revisar:
            print(f"  - {fila['patient_id']}: {fila['mensaje']}")
    if errores:
        print("Pacientes con error:")
        for fila in errores:
            print(f"  - {fila['patient_id']}: {fila['mensaje']}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
