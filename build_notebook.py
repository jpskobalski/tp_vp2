"""
Genera tp_final.ipynb — notebook de entrenamiento del detector de datos de contacto.

Pipeline:
  CSV (img-level labels) --> OCR + regex --> pseudo bboxes --> YOLO format
  --> Entrenamiento YOLOv8 --> Evaluación (val mAP + test binary).
"""

from pathlib import Path
import nbformat as nbf


def md(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_code_cell(text)


cells = []

# ============================================================================
# 0. Header
# ============================================================================
cells.append(md(
    "# TP Final — Visión por Computadora II (CEIA)\n"
    "## Detección de datos de contacto en imágenes de e-commerce\n"
    "\n"
    "**Pipeline propuesto:** YOLOv8 detecta regiones sospechosas con datos de contacto "
    "(teléfonos, emails, handles sociales, keywords tipo *whatsapp*) en imágenes de "
    "publicaciones tipo MercadoLibre. Una segunda etapa de OCR extrae y valida el texto.\n"
    "\n"
    "**Este notebook cubre solo la etapa de entrenamiento del detector.**\n"
    "\n"
    "### Decisiones técnicas clave\n"
    "1. **Weak supervision con OCR.** El dataset trae solo labels binarios a nivel imagen "
    "(`0/1`), pero YOLO necesita bounding boxes. Solución: correr OCR sobre los positivos, "
    "filtrar las cajas con regex (teléfonos, emails, etc.) y usar esas pseudo-cajas como "
    "labels para entrenar.\n"
    "2. **YOLOv8 (Ultralytics).** API limpia, exporta a múltiples formatos, mencionado "
    "explícitamente en las pautas del TP.\n"
    "3. **Multi-clase con 4 categorías** (`phone`, `email`, `social`, `keyword`). La "
    "categoría se infiere automáticamente del regex que matcheó, sin costo de anotación.\n"
    "\n"
    "### Evaluación\n"
    "- **Val (split de train):** mAP, precision, recall — métricas estándar de detección.\n"
    "- **Test (test.csv):** evaluación image-level (¿el modelo detectó algo? sí/no vs label). "
    "No hay bboxes ground-truth en test, así que no se puede computar mAP allí.\n"
))

# ============================================================================
# 1. Setup
# ============================================================================
cells.append(md("---\n## 1. Setup\n"))

cells.append(md(
    "### 1.1. Instalación de dependencias\n"
    "Correr una sola vez. Si ya están instaladas, comentar."
))

cells.append(code(
    "# Ejecutar UNA SOLA VEZ — comentar después\n"
    "# !pip install -q ultralytics easyocr opencv-python pandas matplotlib scikit-learn pyyaml tqdm\n"
    "# !pip install -q torch torchvision  # asegurarse de tener CUDA build\n"
))

cells.append(md("### 1.2. Imports y verificación de GPU"))

cells.append(code(
    "from pathlib import Path\n"
    "import json\n"
    "import re\n"
    "import shutil\n"
    "import random\n"
    "from collections import Counter, defaultdict\n"
    "from typing import Optional\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import cv2\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as mpatches\n"
    "import yaml\n"
    "from tqdm.auto import tqdm\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, precision_score, recall_score, f1_score,\n"
    "    confusion_matrix, roc_auc_score, classification_report,\n"
    ")\n"
    "\n"
    "import torch\n"
    "\n"
    "print('Torch:', torch.__version__)\n"
    "print('CUDA disponible:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('GPU:', torch.cuda.get_device_name(0))\n"
    "    print('CUDA version:', torch.version.cuda)\n"
    "\n"
    "SEED = 42\n"
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n"
))

# ============================================================================
# 2. Configuración
# ============================================================================
cells.append(md("---\n## 2. Configuración\n"))

cells.append(code(
    "# Paths del proyecto\n"
    "PROJECT_ROOT   = Path('.').resolve()\n"
    "DATA_ROOT      = PROJECT_ROOT / 'dataset_imagenes'\n"
    "TRAIN_CSV      = DATA_ROOT / 'train' / 'train.csv'\n"
    "TEST_CSV       = DATA_ROOT / 'test'  / 'test.csv'\n"
    "TRAIN_IMG_DIR  = DATA_ROOT / 'train' / 'images'\n"
    "TEST_IMG_DIR   = DATA_ROOT / 'test'  / 'images'\n"
    "\n"
    "# Directorios de trabajo (artefactos generados)\n"
    "WORK_DIR       = PROJECT_ROOT / 'work'\n"
    "OCR_CACHE_DIR  = WORK_DIR / 'ocr_cache'\n"
    "YOLO_DATA_DIR  = WORK_DIR / 'dataset_yolo'  # dataset en formato YOLO\n"
    "RUNS_DIR       = WORK_DIR / 'runs'           # outputs de entrenamiento\n"
    "for d in (WORK_DIR, OCR_CACHE_DIR, YOLO_DATA_DIR, RUNS_DIR):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# Clases (multi-class detection)\n"
    "CLASSES = ['phone', 'email', 'social', 'keyword']\n"
    "CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}\n"
    "\n"
    "# Hiperparámetros de entrenamiento\n"
    "IMG_SIZE     = 640\n"
    "BATCH_SIZE   = 16     # bajar si OOM en GPU\n"
    "EPOCHS       = 50\n"
    "MODEL_SIZE   = 'n'    # 'n' (nano), 's' (small), 'm' (medium) — empezar con n\n"
    "VAL_FRACTION = 0.15   # 15% de train como validación interna\n"
    "\n"
    "# Para iterar rápido durante el desarrollo: limitar cantidad de imágenes procesadas\n"
    "# con OCR. None = procesar todas. Útil setear a 200 al debuggear.\n"
    "MAX_POSITIVES_OCR = None\n"
    "\n"
    "print('PROJECT_ROOT:', PROJECT_ROOT)\n"
    "print('Clases:', CLASSES)\n"
))

# ============================================================================
# 3. EDA — carga del dataset y exploración
# ============================================================================
cells.append(md("---\n## 3. Carga y exploración del dataset\n"))

cells.append(code(
    "df_train = pd.read_csv(TRAIN_CSV)\n"
    "df_test  = pd.read_csv(TEST_CSV)\n"
    "\n"
    "print(f'Train: {len(df_train)} imágenes — positivos: {df_train.label.sum()} ({df_train.label.mean()*100:.1f}%)')\n"
    "print(f'Test : {len(df_test)} imágenes — positivos: {df_test.label.sum()} ({df_test.label.mean()*100:.1f}%)')\n"
    "df_train.head()\n"
))

cells.append(code(
    "# Path absoluto por fila (para evitar errores al cambiar de CWD)\n"
    "df_train['abs_path'] = df_train['image_path'].apply(lambda p: str(DATA_ROOT / p))\n"
    "df_test['abs_path']  = df_test['image_path'].apply(lambda p: str(DATA_ROOT / p))\n"
    "\n"
    "# Sanity check: ¿existen todas las imágenes?\n"
    "missing_train = (~df_train['abs_path'].apply(lambda p: Path(p).exists())).sum()\n"
    "missing_test  = (~df_test['abs_path'].apply(lambda p: Path(p).exists())).sum()\n"
    "print(f'Faltantes en train: {missing_train}  |  Faltantes en test: {missing_test}')\n"
))

cells.append(code(
    "# Visualizar ejemplos positivos y negativos\n"
    "def show_grid(paths, titles=None, ncols=4, figsize=(16, 4)):\n"
    "    nrows = (len(paths) + ncols - 1) // ncols\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))\n"
    "    axes = np.atleast_2d(axes).flatten()\n"
    "    for ax, p, i in zip(axes, paths, range(len(paths))):\n"
    "        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)\n"
    "        ax.imshow(img); ax.axis('off')\n"
    "        if titles is not None:\n"
    "            ax.set_title(titles[i], fontsize=9)\n"
    "    for ax in axes[len(paths):]:\n"
    "        ax.axis('off')\n"
    "    plt.tight_layout(); plt.show()\n"
    "\n"
    "pos_samples = df_train[df_train.label == 1].sample(4, random_state=SEED)['abs_path'].tolist()\n"
    "neg_samples = df_train[df_train.label == 0].sample(4, random_state=SEED)['abs_path'].tolist()\n"
    "\n"
    "print('Positivos (con datos de contacto):')\n"
    "show_grid(pos_samples)\n"
    "print('Negativos (sin datos de contacto):')\n"
    "show_grid(neg_samples)\n"
))

# ============================================================================
# 4. Generación de pseudo-labels con OCR (weak supervision)
# ============================================================================
cells.append(md(
    "---\n## 4. Generación de pseudo-labels con OCR (weak supervision)\n"
    "\n"
    "**Idea:** correr OCR sobre las imágenes positivas → para cada texto detectado, "
    "aplicar regex que matcheen teléfonos, emails, handles sociales o keywords tipo "
    "*whatsapp* → cada match se convierte en una bounding box etiquetada para entrenar YOLO.\n"
    "\n"
    "Pros:\n"
    "- Escala a miles de imágenes sin anotación manual.\n"
    "- Los labels heredan la precisión espacial del OCR (texto bien acotado).\n"
    "\n"
    "Contras:\n"
    "- Los labels tienen ruido: el OCR puede fallar (texto rotado, baja resolución, fuentes "
    "raras), y los regex pueden tener falsos positivos/negativos. Mitigaciones más adelante."
))

cells.append(md("### 4.1. Regex de detección de contactos"))

cells.append(code(
    "# Email: estándar RFC simplificado\n"
    "EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}')\n"
    "\n"
    "# Handle social tipo @usuario (3–30 chars)\n"
    "HANDLE_RE = re.compile(r'(?<![a-zA-Z0-9])@[a-zA-Z0-9_.]{3,30}')\n"
    "\n"
    "# URLs de redes sociales y mensajería\n"
    "SOCIAL_URL_RE = re.compile(\n"
    "    r'(?:instagram\\.com|facebook\\.com|fb\\.com|tiktok\\.com|twitter\\.com|x\\.com|'\n"
    "    r'youtu\\.?be|wa\\.me|api\\.whatsapp\\.com|t\\.me)\\/[a-zA-Z0-9_.\\-/]+',\n"
    "    re.IGNORECASE,\n"
    ")\n"
    "\n"
    "# Keywords típicos de mensajes de contacto en español (Argentina)\n"
    "KEYWORD_RE = re.compile(\n"
    "    r'\\b(whats\\s*app|whatsapp|wpp|wsp|wapp|llamanos|llam[aá]nos|consultanos|'\n"
    "    r'cont[aá]ctanos|contacto|env[ií]anos|escribinos|cel(?:ular)?|tel(?:[eé]fono)?)\\b',\n"
    "    re.IGNORECASE,\n"
    ")\n"
    "\n"
    "# Teléfono: heurística sobre los dígitos. Match preliminar de algo \"con pinta\" de tel,\n"
    "# después validamos contando dígitos.\n"
    "PHONE_CANDIDATE_RE = re.compile(\n"
    "    r'(?:\\+?\\d{1,3}[\\s\\-\\.]?)?(?:\\(?\\d{2,4}\\)?[\\s\\-\\.]?){1,4}\\d{3,4}'\n"
    ")\n"
    "\n"
    "def looks_like_phone(text: str) -> bool:\n"
    "    \"\"\"True si el texto parece ser un teléfono (8-15 dígitos en total).\"\"\"\n"
    "    digits = re.sub(r'\\D', '', text)\n"
    "    if not (8 <= len(digits) <= 15):\n"
    "        return False\n"
    "    # evitar matches que sean solo letras + 1 número (ej. 'PRO5')\n"
    "    if not PHONE_CANDIDATE_RE.search(text):\n"
    "        return False\n"
    "    # filtros anti falsos positivos comunes: códigos de producto, años, EANs muy largos\n"
    "    if len(digits) > 13:  # EAN-13 o más → producto, no teléfono\n"
    "        return False\n"
    "    return True\n"
    "\n"
    "def classify_text(text: str) -> Optional[str]:\n"
    "    \"\"\"Devuelve la clase si el texto matchea algún patrón de contacto, None si no.\n"
    "\n"
    "    Orden de prioridad: email > social > keyword > phone. Esto previene que un email\n"
    "    como 'juan@mail.com' sea clasificado como handle por contener '@'.\n"
    "    \"\"\"\n"
    "    if EMAIL_RE.search(text):\n"
    "        return 'email'\n"
    "    if SOCIAL_URL_RE.search(text) or HANDLE_RE.search(text):\n"
    "        return 'social'\n"
    "    if KEYWORD_RE.search(text):\n"
    "        return 'keyword'\n"
    "    if looks_like_phone(text):\n"
    "        return 'phone'\n"
    "    return None\n"
    "\n"
    "# Smoke tests\n"
    "samples = [\n"
    "    ('Whatsapp 11 5555-1234', 'keyword'),  # keyword aparece primero\n"
    "    ('1144443333', 'phone'),\n"
    "    ('vendedor@gmail.com', 'email'),\n"
    "    ('@mi_tienda.ok', 'social'),\n"
    "    ('https://instagram.com/mi.tienda', 'social'),\n"
    "    ('Producto Original', None),\n"
    "    ('Talle XL', None),\n"
    "]\n"
    "for t, expected in samples:\n"
    "    got = classify_text(t)\n"
    "    flag = 'OK' if got == expected else 'FAIL'\n"
    "    print(f'  [{flag}] {t!r:45s} esperado={expected!s:8s} got={got}')\n"
))

cells.append(md("### 4.2. OCR con EasyOCR"))

cells.append(code(
    "import easyocr\n"
    "\n"
    "# EasyOCR: español + inglés. gpu=True usa CUDA si está disponible.\n"
    "ocr_reader = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())\n"
    "print('EasyOCR listo. GPU:', torch.cuda.is_available())\n"
))

cells.append(code(
    "def quad_to_xyxy(quad):\n"
    "    \"\"\"Convierte un cuadrilátero (4 puntos) a bbox axis-aligned (x1,y1,x2,y2).\"\"\"\n"
    "    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]\n"
    "    return min(xs), min(ys), max(xs), max(ys)\n"
    "\n"
    "def run_ocr_on_image(image_path: str, min_conf: float = 0.3):\n"
    "    \"\"\"Devuelve lista de dicts: {bbox: (x1,y1,x2,y2), text, conf}.\"\"\"\n"
    "    img = cv2.imread(image_path)\n"
    "    if img is None:\n"
    "        return []\n"
    "    h, w = img.shape[:2]\n"
    "    results = ocr_reader.readtext(img)  # [(quad, text, conf), ...]\n"
    "    out = []\n"
    "    for quad, text, conf in results:\n"
    "        if conf < min_conf:\n"
    "            continue\n"
    "        x1, y1, x2, y2 = quad_to_xyxy(quad)\n"
    "        x1 = max(0, min(w - 1, int(round(x1))))\n"
    "        y1 = max(0, min(h - 1, int(round(y1))))\n"
    "        x2 = max(0, min(w - 1, int(round(x2))))\n"
    "        y2 = max(0, min(h - 1, int(round(y2))))\n"
    "        if x2 <= x1 or y2 <= y1:\n"
    "            continue\n"
    "        out.append({\n"
    "            'bbox': [x1, y1, x2, y2],\n"
    "            'text': text,\n"
    "            'conf': float(conf),\n"
    "            'img_w': w,\n"
    "            'img_h': h,\n"
    "        })\n"
    "    return out\n"
    "\n"
    "# Smoke test sobre un positivo\n"
    "sample_pos = df_train[df_train.label == 1].iloc[0]['abs_path']\n"
    "preview = run_ocr_on_image(sample_pos)\n"
    "print(f'OCR detectó {len(preview)} regiones de texto en la imagen de muestra.')\n"
    "for r in preview[:8]:\n"
    "    print(f\"  conf={r['conf']:.2f}  text={r['text']!r}\")\n"
))

cells.append(md(
    "### 4.3. Correr OCR sobre todos los positivos (con caché)\n"
    "\n"
    "Los resultados de OCR se guardan en `work/ocr_cache/train_ocr.json` para no tener "
    "que reprocesar. Si querés forzar el reprocesamiento, borrá ese archivo."
))

cells.append(code(
    "OCR_CACHE_FILE = OCR_CACHE_DIR / 'train_ocr.json'\n"
    "\n"
    "def load_ocr_cache(path: Path) -> dict:\n"
    "    if path.exists():\n"
    "        with open(path) as f:\n"
    "            return json.load(f)\n"
    "    return {}\n"
    "\n"
    "def save_ocr_cache(cache: dict, path: Path):\n"
    "    tmp = path.with_suffix('.tmp.json')\n"
    "    with open(tmp, 'w') as f:\n"
    "        json.dump(cache, f)\n"
    "    tmp.replace(path)\n"
    "\n"
    "ocr_cache = load_ocr_cache(OCR_CACHE_FILE)\n"
    "print(f'Caché existente: {len(ocr_cache)} imágenes')\n"
    "\n"
    "# Solo procesamos positivos (label=1) — los negativos no necesitan bbox labels.\n"
    "positives = df_train[df_train.label == 1].reset_index(drop=True)\n"
    "if MAX_POSITIVES_OCR is not None:\n"
    "    positives = positives.head(MAX_POSITIVES_OCR)\n"
    "print(f'Positivos a procesar: {len(positives)}')\n"
    "\n"
    "to_process = positives[~positives['image_path'].isin(ocr_cache.keys())]\n"
    "print(f'Faltan procesar: {len(to_process)}  (resto ya está en caché)')\n"
    "\n"
    "BATCH_SAVE = 100\n"
    "for i, row in enumerate(tqdm(to_process.itertuples(index=False), total=len(to_process))):\n"
    "    try:\n"
    "        ocr_cache[row.image_path] = run_ocr_on_image(row.abs_path)\n"
    "    except Exception as e:\n"
    "        print(f'Error en {row.image_path}: {e}')\n"
    "        ocr_cache[row.image_path] = []\n"
    "    if (i + 1) % BATCH_SAVE == 0:\n"
    "        save_ocr_cache(ocr_cache, OCR_CACHE_FILE)\n"
    "save_ocr_cache(ocr_cache, OCR_CACHE_FILE)\n"
    "print(f'Caché final: {len(ocr_cache)} imágenes')\n"
))

cells.append(md("### 4.4. Filtrar regiones OCR con regex → pseudo-labels"))

cells.append(code(
    "def build_pseudo_labels(ocr_cache: dict) -> dict:\n"
    "    \"\"\"Aplica los regex a cada región OCR y genera pseudo-bboxes.\n"
    "\n"
    "    Returns dict: image_path -> list of {bbox, class_name, text, conf}\n"
    "    \"\"\"\n"
    "    pseudo = {}\n"
    "    for img_path, regions in ocr_cache.items():\n"
    "        labels = []\n"
    "        for r in regions:\n"
    "            cls = classify_text(r['text'])\n"
    "            if cls is None:\n"
    "                continue\n"
    "            labels.append({\n"
    "                'bbox': r['bbox'],            # x1,y1,x2,y2 en píxeles\n"
    "                'class_name': cls,\n"
    "                'text': r['text'],\n"
    "                'conf': r['conf'],\n"
    "                'img_w': r['img_w'],\n"
    "                'img_h': r['img_h'],\n"
    "            })\n"
    "        pseudo[img_path] = labels\n"
    "    return pseudo\n"
    "\n"
    "pseudo_labels = build_pseudo_labels(ocr_cache)\n"
    "\n"
    "n_with_any = sum(1 for v in pseudo_labels.values() if v)\n"
    "n_total    = len(pseudo_labels)\n"
    "print(f'Imágenes positivas con al menos 1 pseudo-label: {n_with_any}/{n_total} ({n_with_any/max(n_total,1)*100:.1f}%)')\n"
    "\n"
    "# Distribución por clase\n"
    "class_counter = Counter()\n"
    "for v in pseudo_labels.values():\n"
    "    for lab in v:\n"
    "        class_counter[lab['class_name']] += 1\n"
    "print('\\nBoxes por clase:')\n"
    "for c in CLASSES:\n"
    "    print(f'  {c:10s}: {class_counter.get(c, 0)}')\n"
    "print(f'  TOTAL     : {sum(class_counter.values())}')\n"
))

cells.append(md(
    "**Visualizar pseudo-labels** sobre algunas imágenes para inspeccionar la calidad."
))

cells.append(code(
    "CLASS_COLORS = {'phone': 'red', 'email': 'blue', 'social': 'orange', 'keyword': 'green'}\n"
    "\n"
    "def plot_pseudo_labels(img_path_rel: str, labels: list, ax=None):\n"
    "    ax = ax or plt.gca()\n"
    "    img = cv2.cvtColor(cv2.imread(str(DATA_ROOT / img_path_rel)), cv2.COLOR_BGR2RGB)\n"
    "    ax.imshow(img); ax.axis('off')\n"
    "    for lab in labels:\n"
    "        x1, y1, x2, y2 = lab['bbox']\n"
    "        color = CLASS_COLORS.get(lab['class_name'], 'magenta')\n"
    "        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,\n"
    "                                  linewidth=2, edgecolor=color, facecolor='none')\n"
    "        ax.add_patch(rect)\n"
    "        ax.text(x1, max(0, y1 - 4), f\"{lab['class_name']}: {lab['text'][:25]}\",\n"
    "                color='white', fontsize=8,\n"
    "                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor='none'))\n"
    "\n"
    "# Mostrar 6 ejemplos con al menos 1 pseudo-label\n"
    "with_labels = [(k, v) for k, v in pseudo_labels.items() if v]\n"
    "random.seed(SEED)\n"
    "sample = random.sample(with_labels, k=min(6, len(with_labels)))\n"
    "\n"
    "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n"
    "for ax, (path, labs) in zip(axes.flatten(), sample):\n"
    "    plot_pseudo_labels(path, labs, ax=ax)\n"
    "plt.tight_layout(); plt.show()\n"
))

# ============================================================================
# 5. Conversión al formato YOLO
# ============================================================================
cells.append(md(
    "---\n## 5. Conversión al formato YOLO\n"
    "\n"
    "YOLO espera la siguiente estructura:\n"
    "```\n"
    "dataset_yolo/\n"
    "├── images/\n"
    "│   ├── train/    (.jpg)\n"
    "│   └── val/      (.jpg)\n"
    "├── labels/\n"
    "│   ├── train/    (.txt — uno por imagen, vacío si no tiene cajas)\n"
    "│   └── val/\n"
    "└── data.yaml     (apunta a images/train, images/val, lista de clases)\n"
    "```\n"
    "\n"
    "Cada `.txt` contiene una línea por bbox: `class_id cx cy w h` (todo normalizado a [0,1])."
))

cells.append(md(
    "### 5.1. Split train/val\n"
    "\n"
    "Estrategia:\n"
    "- **Solo imágenes positivas con ≥1 pseudo-label** entran al training set "
    "(YOLO no aprende mucho de un positivo sin cajas).\n"
    "- **Incluir negativos como background**: bajan los falsos positivos del detector.\n"
    "- Split estratificado 85/15."
))

cells.append(code(
    "# Positivos que tienen al menos 1 pseudo-label\n"
    "usable_positives = [p for p, labs in pseudo_labels.items() if labs]\n"
    "print(f'Positivos usables (con ≥1 pseudo-label): {len(usable_positives)}')\n"
    "\n"
    "# Negativos: imágenes con label=0. Usamos una cantidad proporcional para no\n"
    "# desbalancear demasiado (mismo orden de magnitud que los positivos).\n"
    "n_negatives_keep = len(usable_positives)  # 1:1\n"
    "negatives_all = df_train[df_train.label == 0]['image_path'].tolist()\n"
    "random.seed(SEED)\n"
    "negatives = random.sample(negatives_all, k=min(n_negatives_keep, len(negatives_all)))\n"
    "print(f'Negativos a incluir (background): {len(negatives)}')\n"
    "\n"
    "# Lista final de imágenes con su tag (pos/neg) para split estratificado\n"
    "all_items = [(p, 'pos') for p in usable_positives] + [(p, 'neg') for p in negatives]\n"
    "paths_arr  = [x[0] for x in all_items]\n"
    "labels_arr = [x[1] for x in all_items]\n"
    "\n"
    "train_paths, val_paths, train_lbls, val_lbls = train_test_split(\n"
    "    paths_arr, labels_arr,\n"
    "    test_size=VAL_FRACTION,\n"
    "    stratify=labels_arr,\n"
    "    random_state=SEED,\n"
    ")\n"
    "print(f'\\nTrain: {len(train_paths)} ({sum(1 for l in train_lbls if l==\"pos\")} pos / {sum(1 for l in train_lbls if l==\"neg\")} neg)')\n"
    "print(f'Val  : {len(val_paths)} ({sum(1 for l in val_lbls if l==\"pos\")} pos / {sum(1 for l in val_lbls if l==\"neg\")} neg)')\n"
))

cells.append(md("### 5.2. Materializar el dataset YOLO (imágenes + labels)"))

cells.append(code(
    "def to_yolo_line(bbox_xyxy, img_w, img_h, class_id):\n"
    "    \"\"\"Convierte (x1,y1,x2,y2) en píxeles → 'cls cx cy w h' normalizado.\"\"\"\n"
    "    x1, y1, x2, y2 = bbox_xyxy\n"
    "    cx = (x1 + x2) / 2 / img_w\n"
    "    cy = (y1 + y2) / 2 / img_h\n"
    "    w  = (x2 - x1) / img_w\n"
    "    h  = (y2 - y1) / img_h\n"
    "    cx = float(np.clip(cx, 0, 1)); cy = float(np.clip(cy, 0, 1))\n"
    "    w  = float(np.clip(w,  0, 1)); h  = float(np.clip(h,  0, 1))\n"
    "    return f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'\n"
    "\n"
    "def materialize_split(split_name: str, paths: list):\n"
    "    img_dst = YOLO_DATA_DIR / 'images' / split_name\n"
    "    lbl_dst = YOLO_DATA_DIR / 'labels' / split_name\n"
    "    # Limpiar antes — evita arrastrar archivos viejos de corridas previas\n"
    "    for d in (img_dst, lbl_dst):\n"
    "        if d.exists():\n"
    "            shutil.rmtree(d)\n"
    "        d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "    for img_rel in tqdm(paths, desc=f'Build {split_name}'):\n"
    "        src_img = DATA_ROOT / img_rel\n"
    "        if not src_img.exists():\n"
    "            continue\n"
    "        # Symlink en vez de copia para no duplicar GB de imágenes\n"
    "        dst_img = img_dst / src_img.name\n"
    "        try:\n"
    "            if dst_img.exists() or dst_img.is_symlink():\n"
    "                dst_img.unlink()\n"
    "            dst_img.symlink_to(src_img.resolve())\n"
    "        except OSError:\n"
    "            shutil.copy2(src_img, dst_img)  # fallback si el FS no permite symlinks\n"
    "\n"
    "        # Archivo de labels (vacío para negativos)\n"
    "        labs = pseudo_labels.get(img_rel, [])\n"
    "        lines = []\n"
    "        for lab in labs:\n"
    "            cid = CLASS_TO_ID[lab['class_name']]\n"
    "            lines.append(to_yolo_line(lab['bbox'], lab['img_w'], lab['img_h'], cid))\n"
    "        (lbl_dst / (src_img.stem + '.txt')).write_text('\\n'.join(lines))\n"
    "\n"
    "materialize_split('train', train_paths)\n"
    "materialize_split('val',   val_paths)\n"
    "\n"
    "# Estadísticas finales\n"
    "for split in ('train', 'val'):\n"
    "    imgs = list((YOLO_DATA_DIR / 'images' / split).glob('*.jpg'))\n"
    "    txts = list((YOLO_DATA_DIR / 'labels' / split).glob('*.txt'))\n"
    "    nonempty = sum(1 for t in txts if t.stat().st_size > 0)\n"
    "    print(f'{split}: {len(imgs)} imgs | {len(txts)} txt files | {nonempty} con cajas')\n"
))

cells.append(md("### 5.3. Crear `data.yaml`"))

cells.append(code(
    "data_yaml_path = YOLO_DATA_DIR / 'data.yaml'\n"
    "data_yaml = {\n"
    "    'path':  str(YOLO_DATA_DIR.resolve()),\n"
    "    'train': 'images/train',\n"
    "    'val':   'images/val',\n"
    "    'nc':    len(CLASSES),\n"
    "    'names': CLASSES,\n"
    "}\n"
    "with open(data_yaml_path, 'w') as f:\n"
    "    yaml.safe_dump(data_yaml, f, sort_keys=False)\n"
    "print('Escrito:', data_yaml_path)\n"
    "print(data_yaml_path.read_text())\n"
))

# ============================================================================
# 6. Entrenamiento YOLOv8
# ============================================================================
cells.append(md(
    "---\n## 6. Entrenamiento YOLOv8\n"
    "\n"
    "Empezamos con `yolov8n` (nano, ~3M parámetros) como baseline: rápido de entrenar, "
    "fácil de iterar. Si la performance es razonable y hay tiempo, escalar a `yolov8s` "
    "o `yolov8m`.\n"
    "\n"
    "Ultralytics maneja internamente:\n"
    "- Mosaic/HSV/flip augmentations.\n"
    "- Anchor-free head.\n"
    "- AdamW/SGD selection automático.\n"
    "- Logging a CSV + plots (P/R curves, confusion matrix)."
))

cells.append(code(
    "from ultralytics import YOLO\n"
    "\n"
    "model = YOLO(f'yolov8{MODEL_SIZE}.pt')  # descarga los pesos pre-entrenados de COCO\n"
    "print('Modelo cargado:', f'yolov8{MODEL_SIZE}')\n"
))

cells.append(code(
    "# Entrenamiento\n"
    "run_name = f'tp_yolov8{MODEL_SIZE}_imgsz{IMG_SIZE}_e{EPOCHS}'\n"
    "\n"
    "train_results = model.train(\n"
    "    data=str(data_yaml_path),\n"
    "    epochs=EPOCHS,\n"
    "    imgsz=IMG_SIZE,\n"
    "    batch=BATCH_SIZE,\n"
    "    project=str(RUNS_DIR),\n"
    "    name=run_name,\n"
    "    exist_ok=True,\n"
    "    device=0 if torch.cuda.is_available() else 'cpu',\n"
    "    patience=10,         # early stopping si no mejora en 10 epochs\n"
    "    seed=SEED,\n"
    "    verbose=True,\n"
    "    # augmentations (defaults razonables, ajustables)\n"
    "    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,\n"
    "    fliplr=0.5,\n"
    "    mosaic=1.0,\n"
    "    # texto en imagen suele ser sensible a rotaciones fuertes — bajamos esto\n"
    "    degrees=0.0,\n"
    "    perspective=0.0,\n"
    ")\n"
    "print('Entrenamiento finalizado.')\n"
    "print('Mejor checkpoint:', model.trainer.best)\n"
))

# ============================================================================
# 7. Evaluación en validación
# ============================================================================
cells.append(md(
    "---\n## 7. Evaluación en validación (mAP)\n"
    "\n"
    "Cargamos el mejor checkpoint y corremos `model.val()` para obtener mAP@0.5, "
    "mAP@0.5:0.95, precision y recall por clase."
))

cells.append(code(
    "best_ckpt = Path(model.trainer.best)\n"
    "print('Cargando best checkpoint:', best_ckpt)\n"
    "model_best = YOLO(str(best_ckpt))\n"
    "\n"
    "val_metrics = model_best.val(\n"
    "    data=str(data_yaml_path),\n"
    "    imgsz=IMG_SIZE,\n"
    "    batch=BATCH_SIZE,\n"
    "    split='val',\n"
    "    device=0 if torch.cuda.is_available() else 'cpu',\n"
    "    verbose=True,\n"
    ")\n"
    "\n"
    "print('\\n=== Métricas globales en VAL ===')\n"
    "print(f'mAP@0.5      : {val_metrics.box.map50:.4f}')\n"
    "print(f'mAP@0.5:0.95 : {val_metrics.box.map:.4f}')\n"
    "print(f'Precision    : {val_metrics.box.mp:.4f}')\n"
    "print(f'Recall       : {val_metrics.box.mr:.4f}')\n"
    "\n"
    "print('\\n=== Por clase ===')\n"
    "for i, c in enumerate(CLASSES):\n"
    "    ap50 = val_metrics.box.ap50[i] if i < len(val_metrics.box.ap50) else float('nan')\n"
    "    print(f'  {c:10s}: AP@0.5 = {ap50:.4f}')\n"
))

cells.append(md(
    "**Plots automáticos** que Ultralytics genera en `runs/<run_name>/`:\n"
    "- `results.png`: curvas de loss y métricas por epoch.\n"
    "- `confusion_matrix.png`: matriz de confusión multi-clase.\n"
    "- `PR_curve.png`, `P_curve.png`, `R_curve.png`, `F1_curve.png`."
))

cells.append(code(
    "from IPython.display import Image, display\n"
    "\n"
    "run_dir = RUNS_DIR / run_name\n"
    "for img_name in ['results.png', 'confusion_matrix.png', 'PR_curve.png']:\n"
    "    p = run_dir / img_name\n"
    "    if p.exists():\n"
    "        print(img_name)\n"
    "        display(Image(filename=str(p)))\n"
    "    else:\n"
    "        print(f'[no encontrado] {p}')\n"
))

# ============================================================================
# 8. Evaluación en TEST (binary, image-level)
# ============================================================================
cells.append(md(
    "---\n## 8. Evaluación en TEST (clasificación binaria image-level)\n"
    "\n"
    "El test set no tiene bounding boxes ground-truth, solo el label binario `0/1`. "
    "Por eso no podemos calcular mAP en test. Lo que sí podemos evaluar es la utilidad "
    "del detector como **alerta a nivel imagen**: si YOLO detecta ≥1 caja con confianza "
    "≥ umbral → predicción positiva.\n"
    "\n"
    "Esta es la métrica que más le importa al caso de uso real (flaggear publicaciones "
    "sospechosas para revisión humana).\n"
    "\n"
    "**Importante:** el test set tiene solo ~3% de positivos (distribución natural), "
    "así que F1 / recall importan mucho más que accuracy."
))

cells.append(code(
    "CONF_THRESH = 0.25  # umbral default de Ultralytics; lo barreremos abajo\n"
    "\n"
    "def predict_test_set(model, df_test, conf_thresh=CONF_THRESH, batch_size=32):\n"
    "    \"\"\"Predice sobre todo el test set. Devuelve y_true, y_pred, scores (max conf por img).\"\"\"\n"
    "    y_true = df_test['label'].values\n"
    "    paths  = df_test['abs_path'].tolist()\n"
    "    scores = np.zeros(len(paths), dtype=np.float32)\n"
    "\n"
    "    for i in tqdm(range(0, len(paths), batch_size), desc='Predict test'):\n"
    "        batch = paths[i : i + batch_size]\n"
    "        results = model.predict(\n"
    "            batch,\n"
    "            conf=0.05,  # bajo, para tener scores\n"
    "            imgsz=IMG_SIZE,\n"
    "            verbose=False,\n"
    "            device=0 if torch.cuda.is_available() else 'cpu',\n"
    "        )\n"
    "        for j, res in enumerate(results):\n"
    "            if res.boxes is None or len(res.boxes) == 0:\n"
    "                scores[i + j] = 0.0\n"
    "            else:\n"
    "                scores[i + j] = float(res.boxes.conf.max().item())\n"
    "\n"
    "    y_pred = (scores >= conf_thresh).astype(int)\n"
    "    return y_true, y_pred, scores\n"
    "\n"
    "y_true, y_pred, scores = predict_test_set(model_best, df_test, conf_thresh=CONF_THRESH)\n"
    "print(f'Predicciones terminadas. {y_pred.sum()} positivos predichos / {len(y_pred)} imágenes.')\n"
))

cells.append(code(
    "# Métricas binarias en test @ umbral default\n"
    "print(f'=== Test @ conf_thresh={CONF_THRESH} ===')\n"
    "print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')\n"
    "print(f'Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}')\n"
    "print(f'Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}')\n"
    "print(f'F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}')\n"
    "try:\n"
    "    print(f'ROC AUC  : {roc_auc_score(y_true, scores):.4f}')\n"
    "except ValueError as e:\n"
    "    print(f'ROC AUC  : no calculable ({e})')\n"
    "\n"
    "print('\\nMatriz de confusión:')\n"
    "cm = confusion_matrix(y_true, y_pred)\n"
    "print(pd.DataFrame(cm,\n"
    "                   index=['actual=0', 'actual=1'],\n"
    "                   columns=['pred=0', 'pred=1']))\n"
    "\n"
    "print('\\nClassification report:')\n"
    "print(classification_report(y_true, y_pred, target_names=['no_contact', 'contact'], zero_division=0))\n"
))

cells.append(md(
    "### 8.1. Barrido de umbral\n"
    "\n"
    "Como el test set está desbalanceado (3% positivos), conviene barrer el umbral "
    "de confianza y elegir el que maximiza F1 (o el que cumple un criterio de "
    "recall/precision deseado)."
))

cells.append(code(
    "thresholds = np.linspace(0.05, 0.9, 18)\n"
    "rows = []\n"
    "for t in thresholds:\n"
    "    yp = (scores >= t).astype(int)\n"
    "    rows.append({\n"
    "        'thresh':    t,\n"
    "        'precision': precision_score(y_true, yp, zero_division=0),\n"
    "        'recall':    recall_score(y_true, yp, zero_division=0),\n"
    "        'f1':        f1_score(y_true, yp, zero_division=0),\n"
    "        'n_pred_pos': int(yp.sum()),\n"
    "    })\n"
    "df_thresh = pd.DataFrame(rows)\n"
    "print(df_thresh.to_string(index=False))\n"
    "\n"
    "best_row = df_thresh.loc[df_thresh['f1'].idxmax()]\n"
    "print(f\"\\nMejor F1 = {best_row['f1']:.4f} en thresh = {best_row['thresh']:.3f}\")\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.plot(df_thresh['thresh'], df_thresh['precision'], label='Precision', marker='o')\n"
    "ax.plot(df_thresh['thresh'], df_thresh['recall'],    label='Recall',    marker='s')\n"
    "ax.plot(df_thresh['thresh'], df_thresh['f1'],        label='F1',        marker='^')\n"
    "ax.axvline(best_row['thresh'], color='gray', linestyle='--', alpha=0.6,\n"
    "           label=f\"best F1 @ {best_row['thresh']:.2f}\")\n"
    "ax.set_xlabel('Umbral de confianza'); ax.set_ylabel('Métrica')\n"
    "ax.set_title('Barrido de umbral en test set (image-level binary)')\n"
    "ax.legend(); ax.grid(alpha=0.3)\n"
    "plt.show()\n"
))

# ============================================================================
# 9. Inferencia visual
# ============================================================================
cells.append(md("---\n## 9. Inferencia visual\n"))

cells.append(code(
    "# Mostrar predicciones sobre algunos positivos del test set\n"
    "test_pos = df_test[df_test.label == 1].sample(\n"
    "    min(6, df_test.label.sum()), random_state=SEED\n"
    ")['abs_path'].tolist()\n"
    "test_neg = df_test[df_test.label == 0].sample(3, random_state=SEED)['abs_path'].tolist()\n"
    "\n"
    "def visualize_predictions(paths, conf=0.25):\n"
    "    results = model_best.predict(paths, conf=conf, imgsz=IMG_SIZE, verbose=False,\n"
    "                                 device=0 if torch.cuda.is_available() else 'cpu')\n"
    "    n = len(results); ncols = 3; nrows = (n + ncols - 1) // ncols\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))\n"
    "    axes = np.atleast_2d(axes).flatten()\n"
    "    for ax, res in zip(axes, results):\n"
    "        img = res.plot()  # devuelve BGR con cajas dibujadas\n"
    "        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis('off')\n"
    "    for ax in axes[n:]:\n"
    "        ax.axis('off')\n"
    "    plt.tight_layout(); plt.show()\n"
    "\n"
    "print('=== Positivos del test set ===')\n"
    "visualize_predictions(test_pos, conf=float(best_row['thresh']))\n"
    "print('=== Negativos del test set (debería no detectar nada) ===')\n"
    "visualize_predictions(test_neg, conf=float(best_row['thresh']))\n"
))

# ============================================================================
# 10. Pipeline end-to-end preview
# ============================================================================
cells.append(md(
    "---\n## 10. Preview: pipeline end-to-end (YOLO → crop → OCR)\n"
    "\n"
    "Mini-demo de cómo se vería el pipeline completo en producción. La etapa de "
    "validación final (regex sobre el texto extraído, dedup, scoring) queda para el "
    "notebook de inferencia."
))

cells.append(code(
    "def end_to_end(image_path: str, conf=0.25):\n"
    "    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)\n"
    "    res = model_best.predict(image_path, conf=conf, imgsz=IMG_SIZE,\n"
    "                             verbose=False,\n"
    "                             device=0 if torch.cuda.is_available() else 'cpu')[0]\n"
    "\n"
    "    detections = []\n"
    "    if res.boxes is not None and len(res.boxes) > 0:\n"
    "        for box in res.boxes:\n"
    "            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]\n"
    "            cls_id = int(box.cls[0].item())\n"
    "            conf_v = float(box.conf[0].item())\n"
    "            crop   = img[y1:y2, x1:x2]\n"
    "            # OCR sobre el crop solamente — esto es lo que ahorra cómputo vs OCR full-image\n"
    "            ocr_out = ocr_reader.readtext(crop) if crop.size else []\n"
    "            text    = ' | '.join(t for _, t, _ in ocr_out)\n"
    "            # Validación adicional con regex sobre el texto extraído\n"
    "            cls_validated = classify_text(text)\n"
    "            detections.append({\n"
    "                'bbox':           (x1, y1, x2, y2),\n"
    "                'class_pred':     CLASSES[cls_id],\n"
    "                'conf':           conf_v,\n"
    "                'ocr_text':       text,\n"
    "                'class_validated': cls_validated,\n"
    "            })\n"
    "    return img, detections\n"
    "\n"
    "# Demo en un positivo de test\n"
    "demo_path = df_test[df_test.label == 1].iloc[0]['abs_path']\n"
    "img, dets = end_to_end(demo_path, conf=float(best_row['thresh']))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 10))\n"
    "ax.imshow(img); ax.axis('off')\n"
    "for d in dets:\n"
    "    x1, y1, x2, y2 = d['bbox']\n"
    "    color = CLASS_COLORS.get(d['class_pred'], 'magenta')\n"
    "    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,\n"
    "                                    linewidth=2, edgecolor=color, facecolor='none'))\n"
    "    label_str = f\"{d['class_pred']} ({d['conf']:.2f}) → {d['ocr_text'][:30]}\"\n"
    "    ax.text(x1, max(0, y1 - 5), label_str, color='white', fontsize=9,\n"
    "            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none'))\n"
    "plt.title('Pipeline end-to-end: YOLO + OCR'); plt.tight_layout(); plt.show()\n"
    "\n"
    "print('\\nDetecciones:')\n"
    "for d in dets:\n"
    "    print(f\"  bbox={d['bbox']}  yolo={d['class_pred']} ({d['conf']:.2f})  \"\n"
    "          f\"ocr={d['ocr_text']!r}  validated={d['class_validated']}\")\n"
))

# ============================================================================
# 11. Cierre
# ============================================================================
cells.append(md(
    "---\n## 11. Próximos pasos\n"
    "\n"
    "- [ ] Anotar manualmente ~100 imágenes del test set para reportar mAP real en test.\n"
    "- [ ] Comparar `yolov8n` vs `yolov8s` para justificar la elección en el paper.\n"
    "- [ ] Ablation sobre el ruido de las pseudo-labels: filtrar regiones con baja conf de OCR.\n"
    "- [ ] Notebook separado para la demo en video (UI con Gradio o Streamlit).\n"
    "- [ ] Exportar el modelo a ONNX/CoreML para deploy.\n"
))

# ============================================================================
# Build and save
# ============================================================================
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3 (.venv)',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.11',
    },
}

OUT = Path(__file__).parent / 'tp_final.ipynb'
nbf.write(nb, OUT)
print(f'OK — {len(cells)} celdas escritas en {OUT}')
