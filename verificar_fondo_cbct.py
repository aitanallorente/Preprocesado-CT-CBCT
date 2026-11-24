"""Diagnóstico: comprueba que los valores por debajo de -1024 HU del CBCT son fondo.

No forma parte del pipeline. Ver el apartado 5.5.1 del README.

Uso:
    python verificar_fondo_cbct.py 0002
"""

from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Paciente a comprobar (puedes pasarlo por el terminal: python verificar_fondo_cbct.py 0002)
pid = sys.argv[1] if len(sys.argv) > 1 else "0001"

# La raíz del proyecto es la carpeta padre de la carpeta del código
proyecto = Path(__file__).resolve().parent.parent

# Busca el CBCT en la salida final y, si no está, en la del registro
candidatas = [
    proyecto / "PACIENTES_REGISTERED_FINAL" / pid / f"{pid}_cbct_registered.nii.gz",
    proyecto / "PACIENTES_REGISTERED" / pid / f"{pid}_cbct_registered.nii.gz",
]
ruta = next((r for r in candidatas if r.is_file()), None)
if ruta is None:
    print(f"No se encuentra el CBCT del paciente {pid}. Probé:")
    for r in candidatas:
        print("  ", r)
    print("Carpetas de paciente disponibles en PACIENTES_REGISTERED_FINAL:")
    base = proyecto / "PACIENTES_REGISTERED_FINAL"
    if base.is_dir():
        print("  ", sorted(p.name for p in base.iterdir() if p.is_dir()))
    sys.exit(1)

print("Leyendo:", ruta)
img = sitk.ReadImage(str(ruta))
a = sitk.GetArrayFromImage(img).astype(np.float32)

# Excluye el relleno -1024 que pone el registro (igual que el paso 3)
validos = a[~np.isclose(a, -1024.0, atol=1e-3)]

bajo = validos < -1024
print(f"Mínimo (sin relleno): {validos.min():.1f} HU")
print(f"% de válidos por debajo de -1024: {100 * bajo.mean():.2f}")
print(f"% de válidos exactamente en el mínimo: {100 * np.isclose(validos, validos.min()).mean():.2f}")
valores, cuentas = np.unique(np.round(validos[bajo]), return_counts=True)
orden = np.argsort(cuentas)[::-1][:10]
print("Valores más frecuentes por debajo de -1024 (HU, vóxeles):")
for i in orden:
    print(f"  {valores[i]:.0f}: {cuentas[i]}")

# Histograma (escala log en y)
plt.figure(figsize=(8, 4))
plt.hist(validos, bins=np.arange(-1300, 400, 5), log=True)
plt.axvline(-1024, color="r", label="-1024 HU")
plt.xlabel("HU")
plt.ylabel("vóxeles (log)")
plt.title(f"CBCT {pid}")
plt.legend()
plt.tight_layout()
salida_png = Path(__file__).resolve().parent / f"hist_cbct_{pid}.png"
plt.savefig(salida_png, dpi=150)
print("Histograma guardado en:", salida_png)

# Máscara de los vóxeles < -1024 para verla en Slicer
mascara = sitk.GetImageFromArray((a < -1024).astype(np.uint8))
mascara.CopyInformation(img)
salida_mascara = Path(__file__).resolve().parent / f"mascara_bajo_1024_{pid}.nii.gz"
sitk.WriteImage(mascara, str(salida_mascara))
print("Máscara guardada en:", salida_mascara)