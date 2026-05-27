"""
Clasificador de imágenes de prueba para la presentación
========================================================

Corre las imágenes de la carpeta `pruebas presentacion/` por cada modelo cuyo
checkpoint esté disponible (EfficientNet-B0, ResNet-50, ConvNeXT-Tiny) y reporta
la probabilidad de "contacto" + la predicción binaria de cada uno.

USO
---
1. Dejar los checkpoints en la carpeta `checkpoints/` (o en la raíz del proyecto).
   Nombres esperados (los que guardan los notebooks):
       best_efficientnet_b0.pt
       best_resnet50.pt
       best_convnext_tiny.pt

2. Correr:
       python clasificar_pruebas.py

Salida:
  - Tabla por consola (imagen × modelo → prob / pred)
  - CSV en `pruebas presentacion/resultados_pruebas.csv`

Los modelos que NO tengan checkpoint disponible se saltean (se avisa por consola).
El threshold de decisión se lee del checkpoint; si no lo trae, usa 0.67.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)
from PIL import Image

# ============================================================================
# Configuración
# ============================================================================
IMG_DIR = Path("pruebas presentacion")
OUT_CSV = IMG_DIR / "resultados_pruebas.csv"
IMG_SIZE = 384
DEFAULT_THRESHOLD = 0.67

# Carpetas donde buscar cada checkpoint (en orden de prioridad)
SEARCH_DIRS = [Path("checkpoints"), Path("."), Path("work"), Path.home() / "Downloads"]

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


# ============================================================================
# Definición de los 3 modelos
# ============================================================================
def build_efficientnet():
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m, EfficientNet_B0_Weights.DEFAULT


def build_resnet50():
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m, ResNet50_Weights.DEFAULT


def build_convnext():
    m = convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m, ConvNeXt_Tiny_Weights.DEFAULT


MODELS = {
    "EfficientNet-B0": {"builder": build_efficientnet, "ckpt": "best_efficientnet_b0.pt"},
    "ResNet-50":       {"builder": build_resnet50,     "ckpt": "best_resnet50.pt"},
    "ConvNeXT-Tiny":   {"builder": build_convnext,     "ckpt": "best_convnext_tiny.pt"},
}


def find_checkpoint(filename: str):
    """Busca el checkpoint en las carpetas candidatas (búsqueda recursiva en work/)."""
    for d in SEARCH_DIRS:
        # match directo
        p = d / filename
        if p.exists():
            return p
        # búsqueda recursiva (útil para work/<model>/...)
        if d.exists():
            matches = list(d.rglob(filename))
            if matches:
                return matches[0]
    return None


def load_model(builder, ckpt_path):
    """Carga el modelo y devuelve (model, threshold)."""
    model, _weights = builder()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = float(ckpt.get("threshold") or DEFAULT_THRESHOLD)
    else:
        model.load_state_dict(ckpt)
        threshold = DEFAULT_THRESHOLD
    model.to(DEVICE).eval()
    # transform específico del modelo (mean/std de sus weights)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_weights.transforms().mean, _weights.transforms().std),
    ])
    return model, threshold, tf


@torch.no_grad()
def predict(model, tf, image_path):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(model(x)).item()
    return prob


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Carpeta de imágenes: {IMG_DIR}/\n")

    if not IMG_DIR.exists():
        print(f"[ERROR] No existe la carpeta {IMG_DIR}/")
        return

    images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS])
    if not images:
        print(f"[ERROR] No se encontraron imágenes en {IMG_DIR}/")
        return
    print(f"Imágenes encontradas: {len(images)}")

    # Cargar los modelos disponibles
    loaded = {}
    for name, cfg in MODELS.items():
        ckpt_path = find_checkpoint(cfg["ckpt"])
        if ckpt_path is None:
            print(f"  [SKIP] {name}: no se encontró {cfg['ckpt']}")
            continue
        try:
            model, threshold, tf = load_model(cfg["builder"], ckpt_path)
            loaded[name] = (model, threshold, tf)
            print(f"  [OK]   {name}: {ckpt_path}  (threshold {threshold:.2f})")
        except Exception as e:
            print(f"  [FAIL] {name}: error al cargar {ckpt_path} → {e}")

    if not loaded:
        print("\nNo hay ningún modelo disponible. Dejá los .pt en checkpoints/ y reintentá.")
        return

    # Inferencia
    print("\n" + "=" * 90)
    rows = []
    for img_path in images:
        row = {"imagen": img_path.name}
        cells = []
        for name, (model, threshold, tf) in loaded.items():
            prob = predict(model, tf, img_path)
            pred = "CONTACTO" if prob >= threshold else "sin contacto"
            row[f"{name}_prob"] = round(prob, 4)
            row[f"{name}_pred"] = pred
            cells.append(f"{name}: {prob*100:5.1f}% [{pred}]")
        rows.append(row)
        print(f"\n📷 {img_path.name}")
        for c in cells:
            print(f"     {c}")

    print("\n" + "=" * 90)

    # Guardar CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nResultados guardados en: {OUT_CSV}")
    except ImportError:
        import csv
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResultados guardados en: {OUT_CSV}")


if __name__ == "__main__":
    main()
