"""
Genera imágenes auxiliares para la presentación (assets/presentation/).

Genera DOS familias de imágenes:

1) yolo_example.jpg (legacy): tile de un val_batch grid de Ultralytics
   mostrando predicciones reales del modelo entrenado.

2) test_001049 series: las 4 slides de modelos usan la MISMA imagen del
   test set (test_001049.jpg, label=1) para contar una historia
   consistente. Como no tenemos pesos entrenados locales, los slides
   de clasificadores muestran imagen + score real (de test_predictions.csv).
   El slide de OCR+regex tiene bboxes reales porque solo necesita EasyOCR.

Idempotente: se puede correr varias veces sin romper nada.
"""
from pathlib import Path
import re
from typing import Optional
import cv2
import numpy as np

ROOT = Path(__file__).parent
RUN_DIR = ROOT / "work" / "runs" / "tp_yolov8n_imgsz640_e50"
OUT_DIR = ROOT / "assets" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# La imagen de test que vamos a usar como input común en las 4 slides
TEST_IMG = ROOT / "dataset_imagenes" / "test" / "images" / "test_001049.jpg"

# Scores reales de cada modelo sobre test_001049.jpg (extraídos de los
# test_predictions.csv). Si los re-corres podés cambiarlos acá.
PREDICTIONS = {
    "yolov8n":            {"score": 0.194, "pred": 0},
    "yolov8s":            {"score": 0.739, "pred": 1},
    "efficientnet_b0":    {"score": 0.865, "pred": 1},
    "resnet50_cam":       {"score": 1.000, "pred": 1},
    "baseline_ocr_regex": {"score": 0.999, "pred": 1},
}

# Colores (BGR para OpenCV)
COL_PRIMARY = (255,  95,  11)   # #0b5fff azul
COL_SUCCESS = ( 74, 163,  22)   # #16a34a verde
COL_ACCENT  = (  0, 168, 230)   # #e6a800 ámbar
COL_DANGER  = ( 38,  38, 220)   # #dc2626 rojo
COL_MUTED   = ( 85,  85,  85)   # #555 gris
COL_SUCCESS_LIGHT = (231, 252, 220)  # #dcfce7 verde claro
COL_DANGER_LIGHT  = (226, 226, 254)  # #fee2e2 rojo claro
COL_BG_SOFT       = (250, 248, 247)  # #f7f8fa gris muy suave

# Colores por clase para el OCR+regex (BGR)
CLASS_COLORS_BGR = {
    "phone":   ( 38,  38, 220),    # rojo
    "email":   (255,  95,  11),    # azul
    "social":  (  0, 165, 255),    # naranja
    "keyword": ( 74, 163,  22),    # verde
}


def extract_tile(grid_path: Path, row: int, col: int, rows: int = 4, cols: int = 4):
    """Recorta una celda (row, col) de un grid de Ultralytics."""
    img = cv2.imread(str(grid_path))
    if img is None:
        raise FileNotFoundError(grid_path)
    h, w = img.shape[:2]
    tile_h = h // rows
    tile_w = w // cols
    y0, y1 = row * tile_h, (row + 1) * tile_h
    x0, x1 = col * tile_w, (col + 1) * tile_w
    return img[y0:y1, x0:x1].copy()


def add_label_bar(img, text: str, height_px: int = 48, color=(11, 95, 255)):
    """Agrega una franja superior con texto blanco sobre color sólido."""
    h, w = img.shape[:2]
    bar = np.full((height_px, w, 3), color[::-1], dtype=np.uint8)  # BGR
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.9
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = (height_px + th) // 2 - 4
    cv2.putText(bar, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.vstack([bar, img])


def compose_side_by_side(left, right, gap_px: int = 16):
    """Concatena dos imágenes horizontalmente con un gap blanco entre medio."""
    h = max(left.shape[0], right.shape[0])
    # padding negro abajo si difieren en altura (no nuestro caso, pero seguro)
    def pad(img):
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])
    gap = np.full((h, gap_px, 3), 255, dtype=np.uint8)
    return np.hstack([pad(left), gap, pad(right)])


def fit_into_canvas(img, canvas_w: int, canvas_h: int, bg_color=(255, 255, 255)):
    """Encaja img dentro de un canvas (canvas_w, canvas_h) preservando aspect ratio,
    con padding del color de fondo. Devuelve el canvas BGR."""
    h, w = img.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w, 3), bg_color[::-1], dtype=np.uint8)  # BGR
    y0 = (canvas_h - new_h) // 2
    x0 = (canvas_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_yolo_example():
    """Genera assets/presentation/yolo_example.jpg al estilo de los otros _example.png:
    UNA sola imagen, 1100x700 (aspect 1.57), título arriba, predicciones YOLO debajo."""
    preds_grid = RUN_DIR / "val_batch1_pred.jpg"

    if not preds_grid.exists():
        print(f"[SKIP] Falta {preds_grid}")
        return

    # Probamos varios tiles del grid 4x4 y elegimos el que más cajas visibles tiene.
    candidate_tiles = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    best = None
    best_box_pixels = 0
    for r, c in candidate_tiles:
        preds_tile = extract_tile(preds_grid, r, c)
        # heurística: más píxeles con saturación alta = más cajas dibujadas
        hsv = cv2.cvtColor(preds_tile, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[..., 1] > 150
        box_pixels = int(sat_mask.sum())
        if box_pixels > best_box_pixels:
            best_box_pixels = box_pixels
            best = (r, c, preds_tile)

    if best is None:
        print("[SKIP] No se encontró ningún tile con cajas visibles")
        return

    r, c, preds_tile = best
    print(f"Tile elegido: row={r}, col={c}, box_pixels={best_box_pixels}")

    # Canvas final: 1100x700 (mismo aspect ratio que los otros _example.png)
    CANVAS_W, CANVAS_H = 1100, 700
    BAR_H = 64

    # Imagen principal centrada y escalada en el área debajo del title bar
    img_area_h = CANVAS_H - BAR_H
    main_img = fit_into_canvas(preds_tile, CANVAS_W, img_area_h, bg_color=(247, 248, 250))

    # Title bar azul arriba — replica el estilo de los otros _example.png
    title_bar = np.full((BAR_H, CANVAS_W, 3), (255, 95, 11), dtype=np.uint8)  # BGR de #0b5fff
    font = cv2.FONT_HERSHEY_DUPLEX
    title = "YOLOv8n  -  Predicciones sobre val set"
    scale = 0.95
    thick = 2
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    tx = (CANVAS_W - tw) // 2
    ty = (BAR_H + th) // 2 - 4
    cv2.putText(title_bar, title, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    composed = np.vstack([title_bar, main_img])
    out_path = OUT_DIR / "yolo_example.jpg"
    cv2.imwrite(str(out_path), composed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"OK escrito: {out_path}  ({composed.shape[1]}x{composed.shape[0]})")


# =============================================================================
# REGEX de clasificación de contactos (idénticos a los de tp_final / baseline)
# =============================================================================
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
HANDLE_RE = re.compile(r"(?<![a-zA-Z0-9])@[a-zA-Z0-9_.]{3,30}")
SOCIAL_URL_RE = re.compile(
    r"(?:instagram\.com|facebook\.com|fb\.com|tiktok\.com|twitter\.com|x\.com|"
    r"youtu\.?be|wa\.me|api\.whatsapp\.com|t\.me)\/[a-zA-Z0-9_.\-/]+",
    re.IGNORECASE,
)
KEYWORD_RE = re.compile(
    r"\b(whats\s*app|whatsapp|wpp|wsp|wapp|llamanos|llam[aá]nos|consultanos|"
    r"cont[aá]ctanos|contacto|env[ií]anos|escribinos|cel(?:ular)?|tel(?:[eé]fono)?)\b",
    re.IGNORECASE,
)
PHONE_CANDIDATE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?){1,4}\d{3,4}"
)


def looks_like_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if not (8 <= len(digits) <= 15): return False
    if not PHONE_CANDIDATE_RE.search(text): return False
    if len(digits) > 13: return False
    return True


def classify_text(text: str) -> Optional[str]:
    """Prioridad: email > social > keyword > phone."""
    if EMAIL_RE.search(text): return "email"
    if SOCIAL_URL_RE.search(text) or HANDLE_RE.search(text): return "social"
    if KEYWORD_RE.search(text): return "keyword"
    if looks_like_phone(text): return "phone"
    return None


# =============================================================================
# Helpers de composición visual
# =============================================================================
def solid_bar(width: int, height: int, bg_color, text: str = "",
              text_color=(255, 255, 255), font_scale: float = 0.85, thick: int = 2):
    """Devuelve una barra de color sólido con texto centrado."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    if text:
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thick)
        tx = (width - tw) // 2
        ty = (height + th) // 2 - 4
        cv2.putText(bar, text, (tx, ty), font, font_scale, text_color, thick, cv2.LINE_AA)
    return bar


def make_panel(img, model_name: str, decision_text: str,
               title_color, decision_color, decision_text_color,
               extra_subtitle: str = ""):
    """Layout estándar 1100x700 con barra de título arriba y barra de decisión abajo."""
    CANVAS_W, CANVAS_H = 1100, 700
    BAR_TOP, BAR_BOT = 64, 64
    img_area_h = CANVAS_H - BAR_TOP - BAR_BOT

    main_img = fit_into_canvas(img, CANVAS_W, img_area_h, bg_color=(247, 248, 250))
    top_bar = solid_bar(CANVAS_W, BAR_TOP, title_color, text=model_name, font_scale=0.95)
    bot_bar = solid_bar(CANVAS_W, BAR_BOT, decision_color, text=decision_text,
                        text_color=decision_text_color, font_scale=0.85)

    # Si hay subtítulo, lo agregamos en una segunda línea de la barra inferior
    if extra_subtitle:
        # Reescribimos la barra inferior con dos líneas
        bot_bar = np.full((BAR_BOT, CANVAS_W, 3), decision_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw1, th1), _ = cv2.getTextSize(decision_text, font, 0.75, 2)
        (tw2, th2), _ = cv2.getTextSize(extra_subtitle, font, 0.55, 1)
        cv2.putText(bot_bar, decision_text, ((CANVAS_W - tw1) // 2, 26),
                    font, 0.75, decision_text_color, 2, cv2.LINE_AA)
        cv2.putText(bot_bar, extra_subtitle, ((CANVAS_W - tw2) // 2, 52),
                    font, 0.55, decision_text_color, 1, cv2.LINE_AA)

    return np.vstack([top_bar, main_img, bot_bar])


# =============================================================================
# Generadores por modelo usando test_001049.jpg
# =============================================================================
def build_efficientnet_example():
    img = cv2.imread(str(TEST_IMG))
    if img is None:
        print(f"[SKIP] No se encontró {TEST_IMG}"); return
    p = PREDICTIONS["efficientnet_b0"]
    panel = make_panel(
        img,
        model_name="EfficientNet-B0  -  Clasificador binario",
        decision_text=f"p(contacto) = {p['score']:.3f}   →   POSITIVO",
        title_color=COL_PRIMARY,
        decision_color=COL_SUCCESS_LIGHT,
        decision_text_color=COL_SUCCESS,
        extra_subtitle="test_001049.jpg  -  label real = 1  -  prediccion correcta",
    )
    out = OUT_DIR / "efficientnet_example.png"
    cv2.imwrite(str(out), panel)
    print(f"OK escrito: {out}")


def build_resnet_cam_example():
    img = cv2.imread(str(TEST_IMG))
    if img is None: return
    p = PREDICTIONS["resnet50_cam"]
    panel = make_panel(
        img,
        model_name="ResNet50 + CAM  -  Clasificador con localizacion implicita",
        decision_text=f"p(contacto) = {p['score']:.3f}   →   POSITIVO",
        title_color=COL_ACCENT,
        decision_color=COL_SUCCESS_LIGHT,
        decision_text_color=COL_SUCCESS,
        extra_subtitle="test_001049.jpg  -  label real = 1  -  prediccion correcta  (CAM no exportado localmente)",
    )
    out = OUT_DIR / "resnet_cam_example.png"
    cv2.imwrite(str(out), panel)
    print(f"OK escrito: {out}")


def build_yolo_test_example():
    """Sobreescribe yolo_example.jpg usando test_001049 con scores de v8n y v8s."""
    img = cv2.imread(str(TEST_IMG))
    if img is None: return
    pn = PREDICTIONS["yolov8n"]
    ps = PREDICTIONS["yolov8s"]
    # YOLOv8n falla (pred=0), YOLOv8s acierta (pred=1) — historia interesante
    panel = make_panel(
        img,
        model_name="Pipeline YOLO + OCR  -  Deteccion entrenada",
        decision_text=f"YOLOv8s: score = {ps['score']:.3f}  →  POSITIVO",
        title_color=COL_PRIMARY,
        decision_color=COL_SUCCESS_LIGHT,
        decision_text_color=COL_SUCCESS,
        extra_subtitle=f"YOLOv8n: score = {pn['score']:.3f}  →  FALSE NEGATIVE  (la variante chica se la pierde)",
    )
    out = OUT_DIR / "yolo_example.jpg"
    cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"OK escrito: {out}")


def build_ocr_regex_example():
    """OCR real con bboxes dibujados. Es la unica que tiene anotacion espacial real."""
    import easyocr
    img = cv2.imread(str(TEST_IMG))
    if img is None: return

    print("Inicializando EasyOCR (puede tardar la primera vez)...")
    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    print("Corriendo OCR sobre test_001049.jpg...")
    results = reader.readtext(img)
    print(f"  EasyOCR detecto {len(results)} regiones de texto")

    annotated = img.copy()
    matched = []
    for quad, text, conf in results:
        if conf < 0.3: continue
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cls = classify_text(text)
        if cls is not None:
            matched.append((cls, text, conf))
            color = CLASS_COLORS_BGR[cls]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{cls}: {text[:18]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # bbox gris claro fino para regiones no-matcheadas
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)

    print(f"  {len(matched)} regiones matchearon regex: {[(c, t[:20]) for c, t, _ in matched]}")

    p = PREDICTIONS["baseline_ocr_regex"]
    panel = make_panel(
        annotated,
        model_name="OCR + Regex  -  Baseline sin entrenamiento",
        decision_text=f"score = {p['score']:.3f}   →   POSITIVO  ({len(matched)} matches de regex)",
        title_color=COL_MUTED,
        decision_color=COL_SUCCESS_LIGHT,
        decision_text_color=COL_SUCCESS,
        extra_subtitle="cajas coloreadas = regex match  |  cajas grises = OCR detecto texto pero no matcheo",
    )
    out = OUT_DIR / "ocr_regex_example.png"
    cv2.imwrite(str(out), panel)
    print(f"OK escrito: {out}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generando ejemplos con test_001049.jpg para los 4 modelos")
    print("=" * 60)
    build_efficientnet_example()
    build_resnet_cam_example()
    build_yolo_test_example()
    build_ocr_regex_example()
    print()
    print("Listo. Backup de las originales esta en assets/presentation/originals/")
