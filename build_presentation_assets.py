"""
Genera imágenes auxiliares para la presentación (assets/presentation/).

Genera 3 paneles con layout idéntico (1100×700, aspect 1.57) — uno por modelo —
usando una misma imagen positiva del test set como input común. Cada panel tiene:
  - barra de título arriba con el nombre del modelo
  - imagen centrada en el medio
  - barra inferior con métricas agregadas reales (de los 3 nuevos runs)

Modelos cubiertos:
  - ResNet-50        (clásica, 2015)
  - EfficientNet-B0  (eficiente, 2019) ★ ganador
  - ConvNeXT-Tiny    (moderna, 2022)

Como no tenemos pesos entrenados locales, las imágenes muestran el input
+ métricas agregadas del test set, no per-imagen.

Idempotente: se puede correr varias veces sin romper nada.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "assets" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Imagen positiva del test set usada como input común
TEST_IMG = ROOT / "dataset_imagenes" / "test" / "images" / "test_001049.jpg"

# Métricas agregadas reales sobre test (2.700 imágenes, 196 positivos)
# Extraídas de nuevos_runs/*.ipynb (cell 16 outputs)
RESULTS = {
    "resnet50": {
        "title":     "ResNet-50",
        "subtitle":  "Arquitectura clásica (2015) - 25M params",
        "metrics":   {"P": 0.405, "R": 0.459, "F1": 0.431, "AUC-PR": 0.421, "AUC-ROC": 0.809},
        "threshold": 0.66,
        "tp": 90, "fp": 132, "fn": 106,
        "title_color": (230, 168,   0),  # BGR: amber
        "out":         "resnet50_example.jpg",
    },
    "efficientnet_b0": {
        "title":     "EfficientNet-B0",
        "subtitle":  "Eficiencia escalada (2019) - 5M params - GANADOR",
        "metrics":   {"P": 0.471, "R": 0.459, "F1": 0.465, "AUC-PR": 0.488, "AUC-ROC": 0.795},
        "threshold": 0.67,
        "tp": 90, "fp": 101, "fn": 106,
        "title_color": (255,  95,  11),  # BGR: primary blue
        "out":         "efficientnet_example.png",
    },
    "convnext_tiny": {
        "title":     "ConvNeXT-Tiny",
        "subtitle":  "CNN moderna (2022) - 28M params",
        "metrics":   {"P": 0.496, "R": 0.306, "F1": 0.379, "AUC-PR": 0.403, "AUC-ROC": 0.807},
        "threshold": 0.71,
        "tp": 60, "fp":  61, "fn": 136,
        "title_color": ( 74, 163,  22),  # BGR: success green
        "out":         "convnext_example.jpg",
    },
}

# Colores comunes (BGR)
COL_SUCCESS = ( 74, 163,  22)
COL_SUCCESS_BG = (231, 252, 220)


# =============================================================================
# Helpers de composición visual
# =============================================================================
def fit_into_canvas(img, canvas_w: int, canvas_h: int, bg_color=(247, 248, 250)):
    """Encaja `img` en un canvas (canvas_w, canvas_h) preservando aspect ratio."""
    h, w = img.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    y0 = (canvas_h - new_h) // 2
    x0 = (canvas_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_centered_text(bar, text, y, font_scale, thick, color):
    """Dibuja texto centrado horizontalmente en una barra a la altura `y`."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thick)
    x = (bar.shape[1] - tw) // 2
    cv2.putText(bar, text, (x, y), font, font_scale, color, thick, cv2.LINE_AA)


def make_panel(img, model_cfg):
    """Layout estándar 1100×700: title bar (con título + subtítulo) + imagen + metrics bar."""
    CANVAS_W, CANVAS_H = 1100, 700
    BAR_TOP, BAR_BOT = 80, 90
    img_area_h = CANVAS_H - BAR_TOP - BAR_BOT

    # --- Imagen central ---
    main_img = fit_into_canvas(img, CANVAS_W, img_area_h, bg_color=(247, 248, 250))

    # --- Barra superior con nombre + subtítulo ---
    top_bar = np.full((BAR_TOP, CANVAS_W, 3), model_cfg["title_color"], dtype=np.uint8)
    draw_centered_text(top_bar, model_cfg["title"], y=34, font_scale=0.95, thick=2, color=(255, 255, 255))
    draw_centered_text(top_bar, model_cfg["subtitle"], y=64, font_scale=0.55, thick=1, color=(255, 255, 255))

    # --- Barra inferior con métricas agregadas ---
    bot_bar = np.full((BAR_BOT, CANVAS_W, 3), COL_SUCCESS_BG, dtype=np.uint8)
    m = model_cfg["metrics"]
    metrics_line = f"P {m['P']:.3f}  |  R {m['R']:.3f}  |  F1 {m['F1']:.3f}  |  AUC-PR {m['AUC-PR']:.3f}  |  AUC-ROC {m['AUC-ROC']:.3f}"
    draw_centered_text(bot_bar, metrics_line, y=38, font_scale=0.65, thick=2, color=COL_SUCCESS)

    sub = f"threshold {model_cfg['threshold']}  -  TP {model_cfg['tp']}  -  FP {model_cfg['fp']}  -  FN {model_cfg['fn']}  -  test set: 2700 imgs, 196 pos"
    draw_centered_text(bot_bar, sub, y=70, font_scale=0.45, thick=1, color=(85, 85, 85))

    return np.vstack([top_bar, main_img, bot_bar])


# =============================================================================
# Entry point
# =============================================================================
def build_all():
    if not TEST_IMG.exists():
        print(f"[ERROR] No se encontró {TEST_IMG}")
        return

    img = cv2.imread(str(TEST_IMG))
    if img is None:
        print(f"[ERROR] No se pudo leer {TEST_IMG}")
        return

    print(f"Usando imagen base: {TEST_IMG.name}  ({img.shape[1]}x{img.shape[0]})")
    print()

    for key, cfg in RESULTS.items():
        panel = make_panel(img, cfg)
        out = OUT_DIR / cfg["out"]
        if out.suffix.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            cv2.imwrite(str(out), panel)
        print(f"OK escrito: {out}  ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    print("=" * 60)
    print("Generando ejemplos de los 3 modelos sobre test_001049.jpg")
    print("=" * 60)
    build_all()
    print()
    print("Listo. Backup de las versiones anteriores en assets/presentation/originals/")
