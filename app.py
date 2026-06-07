"""
Meat Freshness Classifier — Streamlit Web App

Classifies meat images as Fresh or Spoiled using the best trained ResNet50 model.

Two input modes:
  1. Upload Image — any image file (JPEG, PNG, WEBP, BMP, TIFF)
  2. Take a Photo — browser-based camera capture (works on desktop and mobile)

All preprocessing (resize to 224×224, white-background compositing, ImageNet
normalisation) is handled internally.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    1. Push this project to a public GitHub repository.
    2. Go to share.streamlit.io and connect the repo.
    3. Set the main file path to app.py.
    4. Click Deploy — no extra configuration needed.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Register the custom camera component (served from camera_component/index.html).
# declare_component with a local path works both locally and on Streamlit Cloud.
_camera_with_flip = components.declare_component(
    "camera_with_flip",
    path=str(Path(__file__).parent / "camera_component"),
)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Meat Freshness Classifier",
    page_icon="🥩",
    layout="centered",
)

# ── Model loading (cached — only runs once per session) ───────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    from phase3_inference import _find_best_model_name, load_model
    name = _find_best_model_name()
    if name is None:
        st.error(
            "No trained model found. "
            "Run `python phase3_evaluation.py --best` first."
        )
        st.stop()
    return load_model(name), name

# ── Inference ─────────────────────────────────────────────────────────────────

def classify(pil_img: Image.Image) -> dict:
    """Run inference on a PIL Image. Returns a {label: probability} dict."""
    from phase3_inference import predict

    # Composite transparent images (PNG/WEBP with alpha) onto white background
    if pil_img.mode in ("RGBA", "P", "PA", "LA"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        alpha = pil_img.convert("RGBA").split()[3]
        bg.paste(pil_img.convert("RGB"), mask=alpha)
        pil_img = bg
    else:
        pil_img = pil_img.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pil_img.save(tmp_path, "JPEG", quality=95)
        model, _ = get_model()
        results = predict(model, [Path(tmp_path)])
        return results[0]["probabilities"]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ── Result display ────────────────────────────────────────────────────────────

def show_result(probs: dict) -> None:
    fresh_pct   = probs.get("Fresh",   0.0) * 100
    spoiled_pct = probs.get("Spoiled", 0.0) * 100

    if fresh_pct >= spoiled_pct:
        st.success(f"### ✅ FRESH  —  {fresh_pct:.1f}% confidence")
    else:
        st.error(f"### ⚠️ SPOILED  —  {spoiled_pct:.1f}% confidence")

    st.markdown("**Probability breakdown:**")
    st.progress(probs.get("Fresh",   0.0), text=f"Fresh   — {fresh_pct:.1f}%")
    st.progress(probs.get("Spoiled", 0.0), text=f"Spoiled — {spoiled_pct:.1f}%")

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🥩 Meat Freshness Classifier")
st.markdown(
    "Upload a meat image — or take a photo — to classify it as **Fresh** or **Spoiled**."
)

tab_upload, tab_camera = st.tabs(["📂 Upload Image", "📷 Take a Photo"])

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a meat image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)
        with col2:
            with st.spinner("Classifying…"):
                probs = classify(pil_img)
            show_result(probs)

with tab_camera:
    st.markdown(
        "Point the camera at the meat and press **Capture**. "
        "Use **Flip Camera** to switch between front and back."
    )
    data_url = _camera_with_flip(key="camera_flip", default=None)
    if data_url:
        # data_url is "data:image/jpeg;base64,<encoded>" — decode to PIL
        _, encoded = data_url.split(",", 1)
        pil_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Captured image", use_container_width=True)
        with col2:
            with st.spinner("Classifying…"):
                probs = classify(pil_img)
            show_result(probs)

st.markdown("---")
st.caption("Powered by ResNet50 · TensorFlow 2.21 · Deploy free at share.streamlit.io")
