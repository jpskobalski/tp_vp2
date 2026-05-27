
CHECKPOINT_CANDIDATES = [
    Path("work/efficientnet_b0/best_efficientnet_b0.pt"),
    Path("best_efficientnet_b0.pt"),
    Path("nuevos_runs/best_efficientnet_b0.pt"),
]

IMG_SIZE = 384
DEFAULT_THRESHOLD = 0.67


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def find_checkpoint():
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


def build_model():
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def load_trained_model():
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        return None, DEFAULT_THRESHOLD, (
            "⚠️ No se encontró el checkpoint. Colocá `best_efficientnet_b0.pt` en "
            "`work/efficientnet_b0/` o en la raíz del proyecto."
        )

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = build_model()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = float(checkpoint.get("threshold") or DEFAULT_THRESHOLD)
        epoch = checkpoint.get("epoch", "?")
        auc_pr = checkpoint.get("val_auc_pr", None)
        info = f"✅ Modelo cargado desde {ckpt_path} (epoch {epoch}"
        if auc_pr is not None:
            info += f", val AUC-PR {float(auc_pr):.3f}"
        info += f") · threshold {threshold:.2f} · device {DEVICE}"
    else:
        model.load_state_dict(checkpoint)
        threshold = DEFAULT_THRESHOLD
        info = f"✅ Modelo cargado desde {ckpt_path} (state_dict crudo) · threshold {threshold:.2f} · device {DEVICE}"

    model.to(DEVICE)
    model.eval()
    return model, threshold, info


_weights = EfficientNet_B0_Weights.DEFAULT
_imagenet_mean = _weights.transforms().mean
_imagenet_std = _weights.transforms().std

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_imagenet_mean, std=_imagenet_std),
])

MODEL, THRESHOLD, MODEL_INFO = load_trained_model()
print(MODEL_INFO)

def predict(image):
    if image is None:
        return {}, "Subí una imagen para clasificar."

    if MODEL is None:
        return {}, f"### ❌ Modelo no disponible\n{MODEL_INFO}"

    img = image.convert("RGB")
    x = infer_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = MODEL(x)
        prob = torch.sigmoid(logit).item()

    pred = int(prob >= THRESHOLD)

    label_scores = {
        "Con datos de contacto": prob,
        "Sin datos de contacto": 1.0 - prob,
    }

    if pred == 1:
        verdict = (
            f"### 🚩 CONTIENE datos de contacto\n"
            f"**Probabilidad:** {prob*100:.1f}%  \n"
            f"_(umbral de decisión: {THRESHOLD*100:.0f}%)_  \n\n"
            f"Esta publicación debería marcarse para **revisión de moderación**."
        )
    else:
        verdict = (
            f"### ✅ NO contiene datos de contacto\n"
            f"**Probabilidad de contacto:** {prob*100:.1f}%  \n"
            f"_(umbral de decisión: {THRESHOLD*100:.0f}%)_  \n\n"
            f"La publicación pasa el filtro automático."
        )

    return label_scores, verdict

DESCRIPTION = f"""
# Detector de datos de contacto en publicaciones de e-commerce

Subí una imagen de una publicación y el modelo **EfficientNet-B0** predice si
contiene datos de contacto incrustados (teléfonos, redes sociales, etc.).

<small>{MODEL_INFO}</small>
"""

with gr.Blocks(title="Detector de datos de contacto · VC2") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Imagen de la publicación")
            btn = gr.Button("Clasificar", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=2, label="Probabilidades")
            out_verdict = gr.Markdown()

    btn.click(fn=predict, inputs=inp, outputs=[out_label, out_verdict])
    inp.change(fn=predict, inputs=inp, outputs=[out_label, out_verdict])

    gr.Markdown(
        "<small>TP Visión por Computadora 2 · CEIA · "
        "Juan Pablo Skobalski, Luis Díaz, Ronald Uthurralt</small>"
    )


if __name__ == "__main__":
    demo.launch()
