"""
Patchea TP_VC2_LD.ipynb para correr local (Mac MPS / Linux CUDA / CPU) en vez de
Colab + Google Drive, y para que produzca `work/resnet50_cam/test_predictions.csv`
en el schema estandarizado (compatible con comparison.ipynb).

Cambios:
  - Cell 1: imports limpios (sin drive.mount, sin !pip/apt). Imports pytesseract lazy.
  - Cell 3: paths locales (PROJECT_ROOT auto-detectado), OUTPUT_DIR en work/,
            DEVICE con MPS además de CUDA.
  - Cell 11: usa DEVICE global en vez de re-detectar CUDA solo.
  - Cell 15: model checkpoint save/load apunta a OUTPUT_DIR.
  - Cell 19: ejemplo de prueba usa paths locales bien resueltos.
  - Cell 26: además del CSV legacy, escribe `test_predictions.csv` en formato estándar.
  - Cell 28: ejemplo de uso con path local.

Cosas que NO se tocan (claves para que la comparación sea justa):
  - Arquitectura (ResNet50 + CAM head custom)
  - Transformaciones, hiperparámetros (10 epochs, lr=1e-4, Adam, batch 32)
  - Loss y class weights, scheduler
  - Pipeline de inferencia (CAM → bbox → OCR → regex)
"""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).parent / "TP_VC2_LD.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def replace_cell(idx: int, new_source: str):
    nb.cells[idx].source = new_source
    if nb.cells[idx].cell_type == "code":
        nb.cells[idx].outputs = []
        nb.cells[idx].execution_count = None


# -----------------------------------------------------------------------------
# Cell 1 — Setup + imports (sin Colab/Drive, sin !pip)
# -----------------------------------------------------------------------------
replace_cell(1, '''# Setup local — sin Google Drive ni Colab.
#
# Dependencias adicionales requeridas (instalar si faltan):
#   pip install pytesseract                                   # binding Python
#   brew install tesseract tesseract-lang                     # binario (Mac)
#   sudo apt-get install tesseract-ocr tesseract-ocr-spa      # binario (Linux)
#
# Tesseract solo se usa en el pipeline E2E (CAM → bbox → OCR). El
# `test_predictions.csv` para la comparación con otros modelos se genera SIN OCR,
# así que podés correr hasta la celda de evaluación aunque no lo tengas instalado.

import os
import re
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

# Chequear si tesseract está disponible para no romper más adelante
HAS_TESSERACT = shutil.which("tesseract") is not None
if HAS_TESSERACT:
    import pytesseract
    print(f"Tesseract OK: {pytesseract.get_tesseract_version()}")
else:
    print("[INFO] Tesseract no instalado. La parte de OCR fallará, pero el clasificador y la")
    print("       generación de test_predictions.csv funcionan sin él.")

print("Librerias importadas correctamente")
''')

# -----------------------------------------------------------------------------
# Cell 3 — Paths locales + OUTPUT_DIR + DEVICE con MPS
# -----------------------------------------------------------------------------
replace_cell(3, '''# Paths LOCALES (no Colab) — autodetectados desde el CWD.
# Correr el notebook desde la raíz del repo (donde está `dataset_imagenes/`).
PROJECT_ROOT = Path.cwd()
DATA_ROOT    = PROJECT_ROOT / "dataset_imagenes"

TRAIN_IMAGES_PATH = str(DATA_ROOT / "train" / "images")
TRAIN_CSV_PATH    = str(DATA_ROOT / "train" / "train.csv")
TEST_IMAGES_PATH  = str(DATA_ROOT / "test" / "images")
TEST_CSV_PATH     = str(DATA_ROOT / "test" / "test.csv")

# Output dir alineado con el resto de los notebooks (work/<model_name>/)
OUTPUT_DIR = PROJECT_ROOT / "work" / "resnet50_cam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

print("Rutas configuradas:")
print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
print(f"  Train images: {TRAIN_IMAGES_PATH}")
print(f"  Train CSV   : {TRAIN_CSV_PATH}")
print(f"  Test images : {TEST_IMAGES_PATH}")
print(f"  Test CSV    : {TEST_CSV_PATH}")
print(f"  Output dir  : {OUTPUT_DIR}")
print(f"  Device      : {DEVICE}")

# Verificar archivos
for p in (TRAIN_CSV_PATH, TEST_CSV_PATH):
    print(f"  {'OK ' if os.path.exists(p) else 'FAIL'} {p}")
''')

# -----------------------------------------------------------------------------
# Cell 11 — usar DEVICE global (en vez de definir `device` aparte)
# -----------------------------------------------------------------------------
cell11_old = nb.cells[11].source
cell11_new = cell11_old.replace(
    "# Instanciar modelo\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = ResNet50CAM(num_classes=2, pretrained=True)\nmodel = model.to(device)",
    "# Instanciar modelo — usa DEVICE definido en la celda de configuración (cuda > mps > cpu)\ndevice = DEVICE  # alias para no tener que renombrar el resto del código\nmodel = ResNet50CAM(num_classes=2, pretrained=True)\nmodel = model.to(device)",
)
assert cell11_new != cell11_old, "Cell 11 patch no aplicó — ¿cambió el código original?"
replace_cell(11, cell11_new)

# -----------------------------------------------------------------------------
# Cell 15 — checkpoint local (no Drive)
# -----------------------------------------------------------------------------
cell15_old = nb.cells[15].source
cell15_new = cell15_old.replace(
    "torch.save(model.state_dict(), '/content/drive/My Drive/Colab Notebooks/best_resnet50_cam.pth')",
    "torch.save(model.state_dict(), OUTPUT_DIR / 'best_resnet50_cam.pth')",
).replace(
    "model.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/best_resnet50_cam.pth'))",
    "model.load_state_dict(torch.load(OUTPUT_DIR / 'best_resnet50_cam.pth', map_location=DEVICE))",
)
assert cell15_new != cell15_old, "Cell 15 patch no aplicó — ¿cambió el código original?"
replace_cell(15, cell15_new)

# -----------------------------------------------------------------------------
# Cell 19 — la prueba del pipeline ya usa basename() y TEST_IMAGES_PATH,
#           que ahora apuntan a local. La celda funciona tal cual.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cell 26 — agregar export en schema estandarizado para comparison.ipynb
# -----------------------------------------------------------------------------
replace_cell(26, '''# Export de resultados
results_df = pd.DataFrame(test_results)

# Convertir abs paths -> paths relativos como están en el CSV original
# (necesario para que coincidan con los image_path de los otros modelos)
results_df["image_path"] = results_df["image_path"].apply(
    lambda p: "test/images/" + os.path.basename(p)
)

# 1) Schema LEGACY (el que tenía el notebook original) — útil para análisis interno
legacy_csv = OUTPUT_DIR / "test_results.csv"
results_df.to_csv(legacy_csv, index=False)

# 2) Schema ESTANDARIZADO (consumido por comparison.ipynb)
#    image_path, label, score, pred, model_name
standardized = pd.DataFrame({
    "image_path": results_df["image_path"],
    "label":      results_df["true_label"].astype(int),
    "score":      results_df["confidence"].astype(float),
    "pred":       results_df["pred_label"].astype(int),
    "model_name": "resnet50_cam",
})
std_csv = OUTPUT_DIR / "test_predictions.csv"
standardized.to_csv(std_csv, index=False)

print(f"Resultados exportados:")
print(f"  Legacy (interno): {legacy_csv}")
print(f"  Standard (paper): {std_csv}")
print(f"\\nResumen de resultados:")
print(f"  Total imagenes evaluadas: {len(results_df)}")
print(f"  Predicciones correctas  : {results_df['correct'].sum()}")
print(f"  Accuracy: {results_df['correct'].sum()/len(results_df):.3f}")
print(f"  Verdaderos Positivos: {((results_df['true_label']==1) & (results_df['pred_label']==1)).sum()}")
print(f"  Falsos Positivos    : {((results_df['true_label']==0) & (results_df['pred_label']==1)).sum()}")
print(f"  Verdaderos Negativos: {((results_df['true_label']==0) & (results_df['pred_label']==0)).sum()}")
print(f"  Falsos Negativos    : {((results_df['true_label']==1) & (results_df['pred_label']==0)).sum()}")
''')

# -----------------------------------------------------------------------------
# Cell 28 — ejemplo final: usar path local resuelto
# -----------------------------------------------------------------------------
cell28_old = nb.cells[28].source
cell28_new = cell28_old.replace(
    "sample_image = os.path.join(TEST_IMAGES_PATH, test_df.iloc[0]['image_path'])",
    "sample_image = os.path.join(TEST_IMAGES_PATH, os.path.basename(test_df.iloc[0]['image_path']))",
)
assert cell28_new != cell28_old, "Cell 28 patch no aplicó — ¿cambió el código original?"
replace_cell(28, cell28_new)

# -----------------------------------------------------------------------------
# Limpiar las dos celdas vacías al final (29, 30)
# -----------------------------------------------------------------------------
nb.cells = [c for i, c in enumerate(nb.cells) if not (i in (29, 30) and not c.source.strip())]

# -----------------------------------------------------------------------------
# Metadata: kernel del .venv
# -----------------------------------------------------------------------------
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3",
}

nbf.write(nb, NB_PATH)
print(f"OK — TP_VC2_LD.ipynb patcheado ({len(nb.cells)} celdas).")
