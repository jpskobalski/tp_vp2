"""
Genera comparison.ipynb — junta los test_predictions.csv de todos los modelos,
calcula métricas comparables y arma la tabla + plots para el paper IEEE.

Espera encontrar en work/<model>/test_predictions.csv con schema:
    image_path, label, score, pred, model_name

Detecta automáticamente:
  - work/yolov8n/test_predictions.csv
  - work/yolov8s/test_predictions.csv (si existe)
  - work/yolov11n/test_predictions.csv (si existe)
  - work/efficientnet_b0/test_predictions.csv
  - work/baseline_ocr_regex/test_predictions.csv
"""

from pathlib import Path
import nbformat as nbf


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


cells = []

# ============================================================================
# Header
# ============================================================================
cells.append(md(
    "# Comparación de modelos — Detección de datos de contacto\n"
    "\n"
    "Este notebook **no entrena nada**. Levanta los `test_predictions.csv` que "
    "produjeron los demás notebooks y arma la comparación final que va al paper.\n"
    "\n"
    "## Runbook — orden sugerido para correr todo\n"
    "\n"
    "1. **YOLOv8n** (modelo principal): correr `tp_final.ipynb` end-to-end con "
    "`MODEL_SIZE = 'n'` (default). Guarda en `work/yolov8n/test_predictions.csv`.\n"
    "2. **YOLOv8s** (variante grande): abrir `tp_final.ipynb` ➜ celda 2 (Config) ➜ "
    "cambiar `MODEL_SIZE = 's'` ➜ *Restart & Run All*. Los caches de OCR y dataset_yolo "
    "se reusan; solo se re-entrena el modelo. Guarda en `work/yolov8s/test_predictions.csv`.\n"
    "3. **EfficientNet-B0** (baseline clasificador): correr `TP_VC2_RU.ipynb` end-to-end. "
    "Guarda en `work/efficientnet_b0/test_predictions.csv`.\n"
    "4. **ResNet50 + CAM** (clasificador con localización via Class Activation Map): "
    "correr `TP_VC2_LD.ipynb` end-to-end. Guarda en `work/resnet50_cam/test_predictions.csv`. "
    "Para el pipeline E2E con OCR (no necesario para la comparación), instalar "
    "`brew install tesseract tesseract-lang`.\n"
    "5. **OCR + regex** (baseline sin detector): correr `baseline_ocr_regex.ipynb`. "
    "Guarda en `work/baseline_ocr_regex/test_predictions.csv`. La parte lenta (OCR sobre "
    "3000 imágenes) cachea a `work/ocr_cache/test_ocr.json`.\n"
    "6. **Este notebook (`comparison.ipynb`)**: ejecutar al final. Levanta los CSVs que "
    "encuentre y arma la tabla.\n"
    "\n"
    "Si un modelo todavía no corrió, lo saltea — podés ir refrescando este notebook "
    "a medida que sumás resultados.\n"
    "\n"
    "## Schema esperado por modelo\n"
    "```\n"
    "image_path, label, score, pred, model_name\n"
    "```\n"
    "\n"
    "## Producto final\n"
    "- Tabla comparativa de Precision / Recall / F1 / AUC-PR / AUC-ROC.\n"
    "- Bar chart de métricas por modelo.\n"
    "- PR curves + ROC curves superpuestas.\n"
    "- Matrices de confusión lado a lado.\n"
    "- Tabla en formato Markdown lista para pegar en el paper.\n"
    "- Análisis cruzado: ¿en qué imágenes coinciden / discrepan los modelos?\n"
))

# ============================================================================
# 1. Setup
# ============================================================================
cells.append(md("## 1. Setup"))

cells.append(code(
    "from pathlib import Path\n"
    "import json\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.metrics import (\n"
    "    precision_score, recall_score, f1_score, accuracy_score,\n"
    "    precision_recall_curve, average_precision_score,\n"
    "    roc_curve, roc_auc_score,\n"
    "    confusion_matrix,\n"
    ")\n"
    "\n"
    "PROJECT_ROOT = Path.cwd()\n"
    "WORK_DIR     = PROJECT_ROOT / 'work'\n"
    "OUT_DIR      = WORK_DIR / 'comparison'\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "print('WORK_DIR:', WORK_DIR)\n"
))

# ============================================================================
# 2. Descubrir modelos y cargar predicciones
# ============================================================================
cells.append(md(
    "## 2. Descubrir modelos disponibles\n"
    "\n"
    "Buscamos cualquier `work/*/test_predictions.csv`. Si un modelo todavía no corrió, "
    "lo salteamos en lugar de fallar."
))

cells.append(code(
    "REQUIRED_COLS = {'image_path', 'label', 'score', 'pred', 'model_name'}\n"
    "\n"
    "def load_all_predictions(work_dir):\n"
    "    preds_by_model = {}\n"
    "    for csv in sorted(work_dir.glob('*/test_predictions.csv')):\n"
    "        df = pd.read_csv(csv)\n"
    "        missing = REQUIRED_COLS - set(df.columns)\n"
    "        if missing:\n"
    "            print(f'[WARN] {csv} no tiene columnas: {missing}. Saltado.')\n"
    "            continue\n"
    "        model = df['model_name'].iloc[0]\n"
    "        preds_by_model[model] = df\n"
    "        print(f'  {model:25s} {len(df):5d} filas  ←  {csv.relative_to(work_dir.parent)}')\n"
    "    return preds_by_model\n"
    "\n"
    "predictions = load_all_predictions(WORK_DIR)\n"
    "print(f'\\nModelos encontrados: {len(predictions)}')\n"
    "assert len(predictions) >= 1, 'No se encontró ningún test_predictions.csv en work/*/'\n"
))

# ============================================================================
# 3. Sanity check: ¿todos evaluados sobre el mismo conjunto?
# ============================================================================
cells.append(md(
    "## 3. Sanity check: mismo test set en todos los modelos\n"
    "\n"
    "Si los modelos se evaluaron sobre subsets distintos, las métricas no son "
    "comparables. Acá verificamos."
))

cells.append(code(
    "sizes = {m: len(df) for m, df in predictions.items()}\n"
    "image_sets = {m: set(df['image_path']) for m, df in predictions.items()}\n"
    "\n"
    "print('Tamaños:')\n"
    "for m, n in sizes.items(): print(f'  {m:25s} {n} imágenes')\n"
    "\n"
    "all_paths = set.intersection(*image_sets.values()) if image_sets else set()\n"
    "print(f'\\nIntersección (imágenes en todos los modelos): {len(all_paths)}')\n"
    "\n"
    "if len(set(sizes.values())) > 1:\n"
    "    print('\\n[WARN] Los modelos no tienen el mismo tamaño de test set.')\n"
    "    print('       Limitamos la comparación a la intersección.')\n"
    "    for m, df in predictions.items():\n"
    "        predictions[m] = df[df['image_path'].isin(all_paths)].copy()\n"
    "        print(f'  {m}: {len(predictions[m])} (filtrado)')\n"
    "else:\n"
    "    print('OK: todos los modelos evaluaron sobre el mismo set.')\n"
))

# ============================================================================
# 4. Métricas por modelo
# ============================================================================
cells.append(md("## 4. Métricas por modelo"))

cells.append(code(
    "def compute_metrics(df):\n"
    "    y_true = df['label'].values\n"
    "    y_pred = df['pred'].values\n"
    "    scores = df['score'].values\n"
    "    metrics = {\n"
    "        'accuracy':  accuracy_score(y_true, y_pred),\n"
    "        'precision': precision_score(y_true, y_pred, zero_division=0),\n"
    "        'recall':    recall_score(y_true, y_pred, zero_division=0),\n"
    "        'f1':        f1_score(y_true, y_pred, zero_division=0),\n"
    "    }\n"
    "    # AUC-PR y AUC-ROC requieren scores continuos y al menos un positivo y un negativo\n"
    "    try:    metrics['auc_pr']  = average_precision_score(y_true, scores)\n"
    "    except: metrics['auc_pr']  = np.nan\n"
    "    try:    metrics['auc_roc'] = roc_auc_score(y_true, scores)\n"
    "    except: metrics['auc_roc'] = np.nan\n"
    "    return metrics\n"
    "\n"
    "rows = []\n"
    "for model, df in predictions.items():\n"
    "    m = compute_metrics(df)\n"
    "    m['model'] = model\n"
    "    m['n_test'] = len(df)\n"
    "    m['n_pos'] = int(df['label'].sum())\n"
    "    rows.append(m)\n"
    "\n"
    "metrics_df = (\n"
    "    pd.DataFrame(rows)\n"
    "      [['model', 'n_test', 'n_pos', 'accuracy', 'precision', 'recall', 'f1', 'auc_pr', 'auc_roc']]\n"
    "      .sort_values('f1', ascending=False)\n"
    "      .reset_index(drop=True)\n"
    ")\n"
    "metrics_df.to_csv(OUT_DIR / 'metrics_summary.csv', index=False)\n"
    "metrics_df.style.format({'accuracy': '{:.4f}', 'precision': '{:.4f}',\n"
    "                         'recall': '{:.4f}', 'f1': '{:.4f}',\n"
    "                         'auc_pr': '{:.4f}', 'auc_roc': '{:.4f}'})\n"
))

# ============================================================================
# 5. Bar chart de F1
# ============================================================================
cells.append(md("## 5. Comparación visual"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
    "\n"
    "# F1 por modelo\n"
    "ax = axes[0]\n"
    "models = metrics_df['model'].tolist()\n"
    "x = np.arange(len(models))\n"
    "width = 0.25\n"
    "ax.bar(x - width, metrics_df['precision'], width, label='Precision')\n"
    "ax.bar(x,         metrics_df['recall'],    width, label='Recall')\n"
    "ax.bar(x + width, metrics_df['f1'],        width, label='F1')\n"
    "ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha='right')\n"
    "ax.set_ylim(0, 1)\n"
    "ax.set_ylabel('Score'); ax.set_title('Precision / Recall / F1 por modelo (test)')\n"
    "ax.legend(); ax.grid(axis='y', alpha=0.3)\n"
    "\n"
    "# AUC-PR / AUC-ROC\n"
    "ax = axes[1]\n"
    "ax.bar(x - width/2, metrics_df['auc_pr'],  width, label='AUC-PR')\n"
    "ax.bar(x + width/2, metrics_df['auc_roc'], width, label='AUC-ROC')\n"
    "ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha='right')\n"
    "ax.set_ylim(0, 1)\n"
    "ax.set_ylabel('AUC'); ax.set_title('AUC-PR / AUC-ROC por modelo (test)')\n"
    "ax.legend(); ax.grid(axis='y', alpha=0.3)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'metrics_barplot.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

# ============================================================================
# 6. PR curves superpuestas
# ============================================================================
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
    "\n"
    "# Precision-Recall curves\n"
    "ax = axes[0]\n"
    "for model, df in predictions.items():\n"
    "    p, r, _ = precision_recall_curve(df['label'], df['score'])\n"
    "    ap = average_precision_score(df['label'], df['score'])\n"
    "    ax.plot(r, p, label=f'{model}  (AP={ap:.3f})', linewidth=2)\n"
    "# Baseline aleatorio: positivo_rate\n"
    "pos_rate = next(iter(predictions.values()))['label'].mean()\n"
    "ax.axhline(pos_rate, color='gray', linestyle='--', alpha=0.5, label=f'random ({pos_rate:.3f})')\n"
    "ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n"
    "ax.set_title('Precision-Recall (test set)')\n"
    "ax.legend(loc='lower left'); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)\n"
    "\n"
    "# ROC curves\n"
    "ax = axes[1]\n"
    "for model, df in predictions.items():\n"
    "    fpr, tpr, _ = roc_curve(df['label'], df['score'])\n"
    "    auc = roc_auc_score(df['label'], df['score'])\n"
    "    ax.plot(fpr, tpr, label=f'{model}  (AUC={auc:.3f})', linewidth=2)\n"
    "ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, label='random')\n"
    "ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n"
    "ax.set_title('ROC (test set)')\n"
    "ax.legend(loc='lower right'); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'pr_roc_curves.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

# ============================================================================
# 7. Matrices de confusión lado a lado
# ============================================================================
cells.append(code(
    "n = len(predictions)\n"
    "ncols = min(n, 3); nrows = (n + ncols - 1) // ncols\n"
    "fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))\n"
    "axes = np.atleast_2d(axes).flatten()\n"
    "\n"
    "for ax, (model, df) in zip(axes, predictions.items()):\n"
    "    cm = confusion_matrix(df['label'], df['pred'])\n"
    "    im = ax.imshow(cm, cmap='Blues')\n"
    "    ax.set_title(model)\n"
    "    ax.set_xticks([0, 1]); ax.set_xticklabels(['pred=0', 'pred=1'])\n"
    "    ax.set_yticks([0, 1]); ax.set_yticklabels(['actual=0', 'actual=1'])\n"
    "    # Anotaciones con cantidad\n"
    "    for (i, j), v in np.ndenumerate(cm):\n"
    "        ax.text(j, i, str(v), ha='center', va='center',\n"
    "                color='white' if v > cm.max() / 2 else 'black', fontsize=12)\n"
    "for ax in axes[n:]:\n"
    "    ax.axis('off')\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

# ============================================================================
# 8. Análisis cruzado: ¿en qué imágenes coinciden / discrepan?
# ============================================================================
cells.append(md(
    "## 6. Análisis cruzado de predicciones\n"
    "\n"
    "¿Cuándo coinciden todos los modelos? ¿Cuándo discrepan? Esto da material para "
    "discutir en el paper qué tipos de errores son específicos de cada enfoque."
))

cells.append(code(
    "# Tabla wide: una fila por imagen, una columna pred_{model} por modelo\n"
    "if len(predictions) >= 2:\n"
    "    wide = None\n"
    "    for model, df in predictions.items():\n"
    "        cols = df[['image_path', 'label', 'pred', 'score']].rename(\n"
    "            columns={'pred': f'pred_{model}', 'score': f'score_{model}'}\n"
    "        )\n"
    "        wide = cols if wide is None else wide.merge(cols, on=['image_path', 'label'])\n"
    "\n"
    "    pred_cols = [c for c in wide.columns if c.startswith('pred_')]\n"
    "    wide['n_agree_positive'] = wide[pred_cols].sum(axis=1)\n"
    "    wide['all_agree'] = (wide['n_agree_positive'] == 0) | (wide['n_agree_positive'] == len(pred_cols))\n"
    "\n"
    "    print(f'Total imágenes: {len(wide)}')\n"
    "    print(f'Modelos en acuerdo (todos 0 o todos 1): {wide[\"all_agree\"].sum()} '\n"
    "          f'({wide[\"all_agree\"].mean()*100:.1f}%)')\n"
    "    print(f'Modelos en desacuerdo: {(~wide[\"all_agree\"]).sum()} '\n"
    "          f'({(~wide[\"all_agree\"]).mean()*100:.1f}%)')\n"
    "\n"
    "    print('\\nDesglose por nivel de acuerdo positivo (cuántos modelos votaron 1):')\n"
    "    print(wide.groupby(['label', 'n_agree_positive']).size().unstack(fill_value=0))\n"
    "\n"
    "    wide.to_csv(OUT_DIR / 'predictions_wide.csv', index=False)\n"
    "else:\n"
    "    print('Solo hay 1 modelo cargado — no hay análisis cruzado posible.')\n"
))

# ============================================================================
# 9. Tabla en Markdown para el paper
# ============================================================================
cells.append(md(
    "## 7. Tabla en Markdown para el paper\n"
    "\n"
    "Se guarda en `work/comparison/metrics_table.md`. Pegar directo en el paper "
    "(IEEE Word) — la mayoría de los editores acepta tablas markdown o las convertís "
    "rápido con un convertidor online."
))

cells.append(code(
    "def df_to_markdown(df, float_fmt='{:.4f}'):\n"
    "    cols = df.columns.tolist()\n"
    "    lines = ['| ' + ' | '.join(cols) + ' |',\n"
    "             '|' + '|'.join(['---'] * len(cols)) + '|']\n"
    "    for _, row in df.iterrows():\n"
    "        vals = []\n"
    "        for c in cols:\n"
    "            v = row[c]\n"
    "            if isinstance(v, float): vals.append(float_fmt.format(v))\n"
    "            else: vals.append(str(v))\n"
    "        lines.append('| ' + ' | '.join(vals) + ' |')\n"
    "    return '\\n'.join(lines)\n"
    "\n"
    "md_table = df_to_markdown(metrics_df)\n"
    "(OUT_DIR / 'metrics_table.md').write_text(md_table)\n"
    "print(md_table)\n"
))

# ============================================================================
# 10. Latencias (si están disponibles)
# ============================================================================
cells.append(md(
    "## 8. Latencia de inferencia (si está disponible)\n"
    "\n"
    "Para el paper conviene reportar también ms/imagen. La baseline OCR+regex "
    "ya guarda `latency_stats.json`; los otros notebooks lo pueden completar después."
))

cells.append(code(
    "latency_rows = []\n"
    "for stats_file in WORK_DIR.glob('*/latency_stats.json'):\n"
    "    with open(stats_file) as f:\n"
    "        latency_rows.append(json.load(f))\n"
    "\n"
    "if latency_rows:\n"
    "    lat_df = pd.DataFrame(latency_rows)\n"
    "    lat_df.to_csv(OUT_DIR / 'latency_summary.csv', index=False)\n"
    "    print(lat_df.to_string(index=False))\n"
    "else:\n"
    "    print('No se encontraron latency_stats.json. Generarlos al correr cada modelo.')\n"
))

# ============================================================================
# Build
# ============================================================================
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3 (.venv)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}

OUT = Path(__file__).parent / 'comparison.ipynb'
nbf.write(nb, OUT)
print(f'OK — {len(cells)} celdas escritas en {OUT}')
