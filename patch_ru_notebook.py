"""
Patchea TP_VC2_RU.ipynb para correr local (Mac MPS / Linux CUDA / CPU) en vez de
Colab + Google Drive. Mantiene SEED, transforms, modelo y métricas idénticos al
notebook original para que sea comparable con tp_final.ipynb.

Cambios:
  - Cell 0: header del notebook (sin referencia a Colab)
  - Cell 2: comentario de config (sin Drive)
  - Cell 3: config — paths locales + DEVICE cuda > mps > cpu + pin_memory dinámico
  - Cell 4: reemplaza el mount de Drive por un check de que el dataset existe local
  - Cell 19: ejemplo final con path local

Cosas que NO se tocan:
  - hiperparámetros (SEED=42, IMG_SIZE=384, BATCH=32, EPOCHS=6, LR=5e-5, etc.)
  - dataset class, transforms, modelo, loss, training loop, evaluación, threshold search
"""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).parent / "TP_VC2_RU.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def replace_cell(idx: int, new_source: str):
    nb.cells[idx].source = new_source
    # Limpiar outputs viejos
    if nb.cells[idx].cell_type == "code":
        nb.cells[idx].outputs = []
        nb.cells[idx].execution_count = None


# -----------------------------------------------------------------------------
# Cell 0 — header / descripción
# -----------------------------------------------------------------------------
replace_cell(0, """# # Clasificador binario de imágenes con EfficientNet-B0
#
# Caso: detectar datos de contacto en imágenes de publicaciones de e-commerce.
#
# **Este notebook entrena un baseline de CLASIFICACIÓN binaria** (imagen tiene/no tiene
# datos de contacto). Se compara contra el detector YOLOv8 entrenado en `tp_final.ipynb`,
# que hace **detección multi-clase + OCR** sobre las mismas imágenes.
#
# Para el paper IEEE las dos métricas comparables son `precision / recall / F1` a
# nivel imagen sobre el mismo `test.csv`.
#
# **Dataset esperado (mismo que tp_final.ipynb):**
#
# ```text
# dataset_imagenes/
# ├── train/
# │   ├── images/
# │   └── train.csv   (image_path, label, image_url)
# └── test/
#     ├── images/
#     └── test.csv
# ```
#
# **CSV esperado:** las columnas mínimas son `image_path` y `label`. El path puede
# ser relativo a la raíz del dataset (`train/images/xxx.jpg`) o al split
# (`images/xxx.jpg`); el código resuelve ambos casos.
""")

# -----------------------------------------------------------------------------
# Cell 2 — config (markdown-style comment)
# -----------------------------------------------------------------------------
replace_cell(2, """# ## 2. Configuración general
#
# Ajustar principalmente:
#
# - `IMG_SIZE`: 384 para la corrida seria; 224 para prueba rápida.
# - `BATCH_SIZE`: si hay error de memoria, bajar a 16.
# - `NUM_EPOCHS`: 6 es el default — subir a 12-20 si la GPU lo permite.
""")

# -----------------------------------------------------------------------------
# Cell 3 — config code: paths locales + DEVICE cuda>mps>cpu + pin_memory dinámico
# -----------------------------------------------------------------------------
replace_cell(3, """SEED = 42
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 6
LR = 5e-5
NUM_WORKERS = 2
VAL_SIZE = 0.15
EARLY_STOPPING_PATIENCE = 2
MIN_PRECISION_FOR_THRESHOLD = 0.70


# --- Paths LOCALES (no Colab) ---------------------------------------------------
# PROJECT_ROOT se autodetecta desde el CWD: corré el notebook desde la raíz del repo
# (donde está `dataset_imagenes/`).
PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / "dataset_imagenes"
OUTPUT_DIR = PROJECT_ROOT / "work" / "efficientnet_b0"


# --- Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU --------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
print("Device:", DEVICE)

# pin_memory solo tiene sentido en CUDA (acelera transferencia CPU->GPU)
PIN_MEMORY = DEVICE.type == "cuda"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


set_seed(SEED)
""")

# -----------------------------------------------------------------------------
# Cell 4 — reemplaza el mount de Drive + descompresión por un check local
# -----------------------------------------------------------------------------
replace_cell(4, """# ## 3. Validar dataset local y crear OUTPUT_DIR
#
# (Reemplaza el mount de Google Drive del notebook original.)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en {DATASET_DIR}. "
        f"Asegurate de correr el notebook desde la raíz del repo (donde vive `dataset_imagenes/`)."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATASET_DIR :", DATASET_DIR)
print("OUTPUT_DIR  :", OUTPUT_DIR)
print()
print("Contenido dataset:")
print(list(DATASET_DIR.iterdir()))
""")

# -----------------------------------------------------------------------------
# Cell 7 — Dataset: hacer pin_memory dinámico y ya está.
#   No cambia el __getitem__ porque la lógica fallback ya soporta ambos formatos
#   de path (relativo al split o relativo a la raíz). Nuestros CSV tienen
#   `train/images/train_000000.jpg` (relativo a la raíz) y el fallback lo resuelve.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cell 8 — DataLoaders: cambiar pin_memory=True → PIN_MEMORY
# -----------------------------------------------------------------------------
cell8_old = nb.cells[8].source
cell8_new = cell8_old.replace("pin_memory=True", "pin_memory=PIN_MEMORY")
assert cell8_new != cell8_old, "Cell 8 patch no aplicó — ¿cambió el código original?"
replace_cell(8, cell8_new)

# -----------------------------------------------------------------------------
# Cell 16 — evaluación en test: agregar columnas `score` y `model_name` para
# que el CSV salga en el formato estándar que consume comparison.ipynb.
# -----------------------------------------------------------------------------
cell16_old = nb.cells[16].source
cell16_new = cell16_old.replace(
    'test_results = test_df.copy()\n'
    'test_results["prob_contact"] = test_p\n'
    'test_results["pred"] = (test_p >= BEST_THRESHOLD).astype(int)\n',
    'test_results = test_df.copy()\n'
    'test_results["prob_contact"] = test_p\n'
    'test_results["score"] = test_p          # alias para schema estandarizado\n'
    'test_results["pred"] = (test_p >= BEST_THRESHOLD).astype(int)\n'
    'test_results["model_name"] = "efficientnet_b0"\n',
)
assert cell16_new != cell16_old, "Cell 16 patch no aplicó — ¿cambió el código original?"
replace_cell(16, cell16_new)

# -----------------------------------------------------------------------------
# Cell 19 — ejemplo final: actualizar path a uno local
# -----------------------------------------------------------------------------
replace_cell(19, """# Ejemplo de inferencia sobre una imagen del test local:
# predict_image(DATASET_DIR / "test" / "images" / "test_000000.jpg")
""")

# -----------------------------------------------------------------------------
# Metadata: kernel actualizado al venv del proyecto
# -----------------------------------------------------------------------------
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3",
}

nbf.write(nb, NB_PATH)
print(f"OK — TP_VC2_RU.ipynb patcheado ({len(nb.cells)} celdas).")
