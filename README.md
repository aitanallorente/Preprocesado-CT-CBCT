# Preprocesado CT / CBCT

Código de preprocesado del TFG: convierte los DICOM de cada paciente en pares CT/CBCT registrados, normalizados y con el mismo tamaño, listos para entrenar el modelo de conversión CT ↔ CBCT.

> **Los datos no están incluidos.** Las imágenes de pacientes (DICOM, NIfTI), los puntos marcados en 3D Slicer (`LANDMARKS`) y los informes generados no forman parte de este repositorio por tratarse de datos clínicos. El código espera encontrarlos con la estructura del apartado 2.

## 1. Flujo completo

```text
inventario (opcional)        qué DICOM hay en cada paciente
      │
paso 1   DICOM → NIfTI        PACIENTES            → PACIENTES_NIFTI
      │
paso 2   registro CBCT → CT   PACIENTES_NIFTI      → PACIENTES_REGISTERED
      │        · automático para la mayoría de pacientes
      │        · por landmarks para los que el automático no resuelve
      │
  ── ajuste manual en SPM de TODOS los pacientes ──  → PACIENTES_REGISTERED/"XXXX nuevo"
      │
paso 2c  SPM → par definitivo "XXXX nuevo"         → PACIENTES_REGISTERED_FINAL
      │
paso 3   normalización        PACIENTES_REGISTERED_FINAL → PACIENTES_NORMALIZED
      │
paso 4   redimensionado       PACIENTES_NORMALIZED → PACIENTES_FINAL
      │
paso 5   máscaras de válidos  PACIENTES_REGISTERED_FINAL + PACIENTES_FINAL → PACIENTES_FINAL
      │
validación del dataset final  PACIENTES_FINAL
      │
división entrenamiento/validación   PACIENTES_FINAL\division_entrenamiento_validacion.json
```

La normalización del paso 3 (HU fijos a [0, 1]) y la división 6 a 1 (entrenamiento/validación) son las acordadas con el tutor.

## 2. Estructura de carpetas

```text
CARPETA DEL PROYECTO\
├── Codigo_TFG_preprocesado\      este código
│   ├── LANDMARKS\                puntos de Slicer (XXXX_CT.mrk.json, XXXX_CBCT.mrk.json); no incluida
│   └── tests\
├── PACIENTES\                    DICOM originales:  XXXX\DICOM\...
├── PACIENTES_NIFTI\              salida del paso 1
├── PACIENTES_REGISTERED\         salida del paso 2 + carpetas "XXXX nuevo" de SPM
├── PACIENTES_REGISTERED_FINAL\   salida del paso 2c
├── PACIENTES_NORMALIZED\         salida del paso 3
└── PACIENTES_FINAL\              salida de los pasos 4 y 5 (+ división entrenamiento/validación)
```

No hay listas de pacientes en el código. Cada paso detecta las carpetas de cuatro dígitos (`0001`, `0002`, ...) de su carpeta de entrada. Para añadir un paciente basta con crear `PACIENTES\0007\DICOM\...`.

Opciones comunes a todos los pasos:

- `--project-dir "C:\ruta\CARPETA DEL PROYECTO"`: solo si el código no está dentro de la carpeta del proyecto (también vale la variable de entorno `TFG_PROJECT_DIR`).
- `--pacientes 0001 0004`: procesa solo esos pacientes.
- `--sobrescribir`: regenera salidas que ya existen. Sin esta opción, un paciente ya procesado se omite.

## 3. Instalación

Python 3.10 o superior.

Programas externos (no se instalan con pip), necesarios para los dos pasos manuales:

- **3D Slicer**: marcado de landmarks del paso 2 en los pacientes que lo necesitan.
- **SPM12** (sobre MATLAB): ajuste manual del registro (apartado 5.3).

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para ejecutar los tests: `pip install -r requirements-dev.txt` y después `pytest`.

## 4. Ejecución rápida

El ajuste en SPM es manual, así que el pipeline se lanza en dos tramos:

```bat
python ejecutar_pipeline.py pre-spm
   ... ajuste manual en SPM (apartado 5.3) ...
python ejecutar_pipeline.py post-spm --sobrescribir
```

- `pre-spm` ejecuta inventario, paso 1 y paso 2. Decide solo el modo del paso 2: landmarks para los pacientes que tengan sus dos archivos en `LANDMARKS\`, automático para el resto.
- `post-spm` ejecuta paso 2c, paso 3 (leyendo de `PACIENTES_REGISTERED_FINAL`), paso 4, paso 5, la validación y la división entrenamiento/validación. Aquí `--pacientes` solo filtra el paso 2c: los pasos 3, 4 y 5, la validación y la división usan siempre toda la cohorte.
- Añadiendo `--simular` se muestran los comandos sin ejecutar nada.

## 5. Paso a paso

### 5.0 Inventario (opcional)

```bat
python inventario_dicom.py
```

Clasifica todos los DICOM y genera en la raíz del proyecto `inventario_completo.csv`, `resumen_pacientes.csv` e `inventario_TFG.xlsx`. No extrae datos identificativos del paciente. El paso 1 no depende de él.

### 5.1 Paso 1: DICOM → NIfTI

```bat
python paso1_dicom_a_nifti.py
```

Para cada paciente elige la serie CT axial principal (descartando localizadores y capturas secundarias) y la serie CBCT, y escribe `PACIENTES_NIFTI\XXXX\XXXX_ct.nii.gz` y `XXXX_cbct.nii.gz`. El CT se guarda completo, sin recortar. Informes: `informe_conversion.csv` e `informe_volumenes.csv`.

### 5.2 Paso 2: registro rígido CBCT → CT

El CBCT abdominal cubre solo una parte del CT y cada equipo usa su propio sistema de coordenadas, así que primero hay que localizar el CBCT dentro del CT.

**Modo automático (por defecto):**

```bat
python paso2_registro.py --pacientes 0001 0004 0006
```

1. Reorienta CT y CBCT a LPS.
2. Búsqueda gruesa sobre imágenes de bordes: Z → X → Y → rotación.
3. Refinamiento con Mutual Information (Mattes) multirresolución.

**Modo landmarks**, para los pacientes en los que el automático coloca el CBCT en un nivel anatómico equivocado (ahora mismo 0002, 0003 y 0005):

```bat
python paso2_registro.py --pacientes 0002 0003 0005 --landmarks-dir LANDMARKS --sobrescribir
```

1. En 3D Slicer se marcan los mismos puntos en el CT y en el CBCT y se guardan como `LANDMARKS\XXXX_CT.mrk.json` y `LANDMARKS\XXXX_CBCT.mrk.json`.
2. Protocolo de puntos: P1 centro del cuerpo vertebral de referencia, P2 vértebra superior, P3 vértebra inferior, P5 y P6 apófisis transversas izquierda y derecha. P4 (apófisis espinosa) no se usa por ser poco reproducible.
3. Hacen falta P1, P2, P3 y al menos una transversa. La transformación rígida se calcula por mínimos cuadrados (Kabsch), sin escala. Si el ajuste es claramente incoherente (RMS > 5 mm o error máximo > 8 mm) se prueba a descartar una sola transversa.
4. No se ejecuta Mutual Information, para no estropear la correspondencia manual.
5. El modo se aplica a todos los pacientes de la llamada: hay que indicar siempre `--pacientes`.

En ambos modos la salida es la misma: `PACIENTES_REGISTERED\XXXX\XXXX_ct.nii.gz`, `XXXX_cbct_registered.nii.gz` (CBCT en la rejilla del CT, recortado a la zona con CBCT y relleno con -1024 fuera) y `XXXX_transform.tfm`. El informe `informe_registro.csv` indica el modo usado y marca REVISAR los casos dudosos.

Este registro es solo el punto de partida del ajuste manual.

### 5.3 Ajuste manual en SPM (todos los pacientes)

1. Abrir el par de `PACIENTES_REGISTERED\XXXX` en SPM y ajustar el CBCT sobre el CT con *Reorient images*.
2. Guardar el resultado en `PACIENTES_REGISTERED\XXXX nuevo\` con estos nombres:
   - `XXXX_cbct_registered.nii` (CBCT con el ajuste de SPM en la cabecera)
   - `XXXX_ct.nii` (CT de referencia; también se acepta `XXXX_ct_2.nii`)

SPM no remuestrea la imagen: guarda el ajuste en la cabecera (sform). Para comprobar que quedó guardado:

```bat
python inspeccionar_nifti.py "..\PACIENTES_REGISTERED\0001 nuevo\0001_cbct_registered.nii"
```

### 5.4 Paso 2c: de SPM al par registrado definitivo

```bat
python paso2c_spm_a_registrado.py --figuras
```

Lee la cabecera con el criterio de SPM, remuestrea el CBCT sobre la rejilla del CT y escribe `PACIENTES_REGISTERED_FINAL\XXXX\XXXX_ct.nii.gz` y `XXXX_cbct_registered.nii.gz`. Además mide cuánto corrigió SPM respecto al paso 2 (traslación y rotación) y lo guarda en `informe_spm_a_registrado.csv`. Con `--figuras` genera en `figuras_registro\` imágenes de fusión y tablero de ajedrez para la memoria.

### 5.5 Paso 3: normalización

```bat
python paso3_normalizacion.py --registered-root PACIENTES_REGISTERED_FINAL --sobrescribir
```

**Importante:** sin `--registered-root PACIENTES_REGISTERED_FINAL` el script lee `PACIENTES_REGISTERED`, es decir, el registro anterior al ajuste en SPM.

Criterio (acordado con el tutor), igual para CT y CBCT:

1. Recorte a [-1024, 3071] HU.
2. Escalado min-max a [0, 1] con esos límites fijos: `(HU + 1024) / 4095`. El aire y el relleno -1024 quedan en 0.
3. Al ser límites fijos, no depende del resto de pacientes.
4. La geometría no cambia.

El informe indica el porcentaje de vóxeles recortados. En el CBCT es normal que el aviso «más del 1 % recortado» salte: entre el 22 % y el 26 % de los vóxeles válidos quedan por debajo de -1024 HU, con un mínimo de -1200 HU en todos los pacientes. Es el valor de fondo que el CBCT asigna fuera del campo de reconstrucción circular (esquinas del volumen), no aire dentro del paciente (ver 5.5.1). Si el recorte fuera alto y no estuviera concentrado en ese valor de fondo, los valores no estarían en HU y habría que revisarlo. La versión anterior (CBCT con percentiles globales, salida en [-1, 1]) se descartó (apéndice C de la memoria).

Salida: `PACIENTES_NORMALIZED\XXXX\XXXX_ct_norm.nii.gz`, `XXXX_cbct_norm.nii.gz`, `informe_normalizacion.csv` y `config_normalizacion.json`.

#### 5.5.1 Verificación opcional del fondo del CBCT

```bat
python verificar_fondo_cbct.py 0001
```

Diagnóstico independiente del pipeline (no forma parte de `ejecutar_pipeline.py`). Lee el CBCT registrado del paciente indicado (`PACIENTES_REGISTERED_FINAL`, o `PACIENTES_REGISTERED` si no existe) y, excluyendo el relleno -1024 del registro, muestra por consola el mínimo, el porcentaje por debajo de -1024 HU, el porcentaje exactamente en el mínimo y los valores más frecuentes por debajo de -1024. Sin argumento usa el paciente `0001`.

Genera, junto al script:

- `hist_cbct_XXXX.png`: histograma (escala logarítmica). El fondo aparece como un pico estrecho en -1200 HU, separado del aire real del paciente.
- `mascara_bajo_1024_XXXX.nii.gz`: máscara de los vóxeles por debajo de -1024 HU, con la geometría del CBCT. Superpuesta al CBCT en 3D Slicer (cargada como *Segmentation*) cubre las esquinas del volumen fuera del círculo de reconstrucción y no entra en la anatomía.

Se comprobó en los pacientes 0001 y 0002.

### 5.6 Paso 4: redimensionado

```bat
python paso4_redimensionado.py --sobrescribir
```

Lleva todos los pares a 256 × 256 × 128 con un spacing común, calculado para que quepa el mayor campo físico de la cohorte sin recortar ningún volumen. El exterior se rellena con 0 (el valor normalizado de -1024 HU). Salida: `PACIENTES_FINAL\XXXX\XXXX_ct_final.nii.gz`, `XXXX_cbct_final.nii.gz`, `informe_redimensionado.csv` y `config_redimensionado.json`.

### 5.7 Paso 5: máscara de vóxeles válidos del CBCT

```bat
python paso5_mascaras.py
python paso5_mascaras.py --figura 0006
```

En el dataset final el aire, el fondo del CBCT (-1200 HU), el relleno del registro (-1024 HU) y el relleno de la rejilla valen todos 0, así que ya no se pueden distinguir. Este paso los separa antes de que se pierda la información:

1. Lee el CBCT en HU de `PACIENTES_REGISTERED_FINAL` y marca como válido todo vóxel que no sea relleno (-1024 exacto) ni fondo (-1200).
2. Erosiona la máscara 1 vóxel para quitar el borde interpolado (`--erosion-vox`, 0 para no erosionar).
3. La lleva a la rejilla final con vecino más próximo, usando como referencia el CBCT de `PACIENTES_FINAL`, y comprueba que comparte geometría con el CT y el CBCT finales.

Salida: `PACIENTES_FINAL\XXXX\XXXX_mask_final.nii.gz` (0/1) e `informe_mascaras.csv`, con el porcentaje de vóxeles válidos y dos controles de coherencia que deben salir pequeños. Con `--figura XXXX` guarda en `PACIENTES_FINAL\figuras_mascaras\` cortes axial y coronal con la máscara superpuesta. Es la región en la que se calculan las métricas de la síntesis.

**Importante:** si se repite el paso 4, hay que repetir este paso con `--sobrescribir`.

### 5.8 Validación final

```bat
python validar_dataset_final.py
```

Comprueba que cada par existe, tiene el tamaño fijo, comparte geometría, usa el spacing común de la cohorte y no tiene valores no finitos ni fuera de [0, 1]. Genera `PACIENTES_FINAL\informe_validacion_final.csv`. Aun así conviene abrir algún par en 3D Slicer para confirmar visualmente la alineación.

### 5.9 División entrenamiento / validación

```bat
python dividir_dataset.py
```

Reparte los pacientes 6 a 1 (con 7 pacientes: 6 de entrenamiento y 1 de validación). Con tan pocos pacientes no hay test independiente. Se divide por paciente, nunca por cortes, para que la validación no contenga anatomía vista en el entrenamiento. El paciente de validación se elige al azar con semilla fija (`--semilla 42`) o a mano (`--validacion 0012`). No mueve archivos: escribe `PACIENTES_FINAL\division_entrenamiento_validacion.json`, que es lo que debe leer el código de entrenamiento. Si la división ya existe no se recalcula, salvo con `--sobrescribir`. Al terminar imprime qué pacientes van a cada conjunto.

## 6. Si se añade o se rehace un paciente

- Pasos 1, 2 y 2c: basta con ejecutarlos para ese paciente (`--pacientes`). Los informes son acumulativos y conservan las filas del resto.
- Paso 3: basta con ejecutarlo para ese paciente; los límites son fijos.
- Paso 4: hay que repetirlo para toda la cohorte con `--sobrescribir`, porque el spacing común depende de todos los pacientes.
- Paso 5: después del paso 4, repetirlo para toda la cohorte con `--sobrescribir`.
- División: si cambia la lista de pacientes hay que regenerarla con `python dividir_dataset.py --sobrescribir` (el paciente de validación puede cambiar, así que habría que volver a entrenar).

## 7. Archivos del código

| Archivo | Para qué sirve |
|---|---|
| `ejecutar_pipeline.py` | Lanza el pipeline por tramos (`pre-spm`, `post-spm`) |
| `inventario_dicom.py` | Inventario de los DICOM |
| `paso1_dicom_a_nifti.py` | Paso 1 |
| `paso2_registro.py` | Paso 2 (automático y landmarks) |
| `paso2c_spm_a_registrado.py` | Paso 2c |
| `paso3_normalizacion.py` | Paso 3 |
| `paso4_redimensionado.py` | Paso 4 |
| `paso5_mascaras.py` | Paso 5: máscaras de vóxeles válidos del CBCT |
| `validar_dataset_final.py` | Validación final |
| `dividir_dataset.py` | División entrenamiento/validación (6 a 1) |
| `configuracion.py` | Rutas del proyecto y detección de pacientes |
| `dicom_utils.py` | Clasificación de DICOM (inventario y paso 1) |
| `imagen_utils.py` | Comprobaciones de geometría de los volúmenes |
| `nifti_cabecera.py` | Lectura de la cabecera NIfTI con el criterio de SPM |
| `escritura_segura.py` | Escritura de las salidas sin dejar archivos a medias |
| `informes.py` | Informes CSV acumulativos |
| `inspeccionar_nifti.py` | Diagnóstico: cabecera de un NIfTI (útil tras SPM) |
| `prueba_orientacion_volumenes.py` | Diagnóstico: geometría de los NIfTI del paso 1 |
| `verificar_fondo_cbct.py` | Diagnóstico: comprueba que el valor por debajo de -1024 HU del CBCT es fondo (histograma y máscara) |

Scripts de apoyo a la memoria (no forman parte del pipeline):

| Archivo | Para qué sirve |
|---|---|
| `extraer_datos_materiales.py` | Datos técnicos del capítulo 5: parámetros de adquisición (`protocolo_adquisicion.csv`) y entorno software/hardware |
| `figura_ct_cbct.py` | Figura comparativa CT / CBCT del marco teórico |
| `figura_apendice_c.py` | Figura del apéndice C: registro automático fallido frente al registro final. Repite el registro automático en `PRUEBAS_APENDICE_C` (no toca las carpetas del pipeline) y guarda las figuras en `figuras_apendice_c` |
| `comprobar_nivel_registro.py` | Diagnóstico: comprueba si CT y CBCT registrados están al mismo nivel vertebral, desplazando el CBCT en el eje craneocaudal (Dice de hueso y correlación). No modifica ningún archivo |
| `csv_a_excel.py` | Convierte los informes CSV a Excel con formato |

## 8. Versiones descartadas

Las versiones anteriores del registro y de la normalización que se probaron y descartaron no se incluyen en el repositorio. Se describen en el apéndice C de la memoria.
