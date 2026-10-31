"""Local Streamlit interface for the trained Sorghum CNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from train_sorghum import SorghumCNN


MODEL_PATH = Path(__file__).parent / "results_224_rust125" / "sorghum_cnn_best.pt"


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = SorghumCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, checkpoint["image_size"], device


def prepare_image(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


st.set_page_config(page_title="Sorghum Disease Classifier", page_icon="🌾", layout="centered")
st.title("Sorghum Disease Classifier")
st.write("Upload a Sorghum leaf or head image to see the model’s top three class predictions.")

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Train the model first, or put sorghum_cnn_best.pt in the results_224_rust125 folder.")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)
        model, classes, image_size, device = load_model(str(MODEL_PATH))
        tensor = prepare_image(image, image_size).to(device)
        with torch.inference_mode():
            scores = torch.softmax(model(tensor), dim=1)[0]
        values, indices = torch.topk(scores, k=min(3, len(classes)))

        st.subheader("Top predictions")
        for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            st.metric(f"{rank}. {classes[index]}", f"{score:.1%}")
        st.caption("Scores are model confidence scores, not calibrated probabilities or a confirmed diagnosis.")
        st.caption(f"Model device: {device}")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Choose a JPEG or PNG image to begin.")
