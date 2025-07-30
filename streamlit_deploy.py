import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Load your trained model
model = YOLO("best.pt")  # <-- Replace with your actual path

st.title("📦 PCB Defect Detection using YOLOv8")
st.markdown("Upload an image to detect PCB defects")

uploaded_file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Draw results
    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result", use_column_width=True)
    st.success("Detection completed!")

    # Optionally, show raw results
    with st.expander("Show Detection Details"):
        st.write(results[0].boxes)
