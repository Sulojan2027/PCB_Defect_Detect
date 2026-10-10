"""Streamlit demo: upload a PCB image and see detected defects.

    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from PIL import Image

from src.models import load_model

LOCAL_WEIGHTS = "runs/pcb/yolov8s/weights/best.pt"


@st.cache_resource
def default_weights() -> str:
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return LOCAL_WEIGHTS
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, os.environ.get("HF_MODEL_FILE", "best.pt"))


@st.cache_resource
def get_model(weights: str):
    return load_model(weights)


st.title("PCB Defect Detection")
st.markdown("Upload an image to detect PCB defects")

with st.sidebar:
    weights = st.text_input("Model weights", default_weights())
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.95, 0.7, 0.05)

uploaded_file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png"])

if not Path(weights).is_file():
    st.error(f"Weights not found: `{weights}`. Train a model first or set the path in the sidebar.")
    st.stop()

try:
    model = get_model(weights)
except Exception as e:
    st.error(f"Could not load `{weights}`: {e}")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Pass the PIL image: Ultralytics treats numpy arrays as BGR, PIL images as RGB.
    result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]

    st.image(result.plot(pil=True), caption="Detection result", use_container_width=True)
    st.success(f"Detected {len(result.boxes)} defect(s)")

    with st.expander("Show detection details"):
        boxes = result.boxes
        st.dataframe(pd.DataFrame({
            "class": [model.names[int(c)] for c in boxes.cls],
            "confidence": boxes.conf.cpu().numpy().round(3),
            **dict(zip(["x1", "y1", "x2", "y2"], boxes.xyxy.cpu().numpy().round().astype(int).T)),
        }))
