"""Streamlit app for sugarcane leaf disease prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from src.predict import load_model_and_weights, predict_image
from src.grad_cam import save_gradcam

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/best_model.keras")
WEIGHTS_PATH = Path("weights/best_weights.keras")
CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]


def main() -> None:
    """Render the Streamlit interface for prediction."""
    st.title("Sugarcane Leaf Disease Classifier")
    st.write("Upload an image of a sugarcane leaf and get a disease prediction.")

    if MODEL_PATH.exists():
        model = load_model_and_weights(MODEL_PATH, WEIGHTS_PATH)
    else:
        st.error("Model not found. Please run main.py to train the model first.")
        return

    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        temp_path = Path("predictions/uploaded_image.png")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(image_bytes)

        predicted_label, confidence = predict_image(model, temp_path, CLASS_NAMES)
        st.success(f"Predicted Disease: {predicted_label}")
        st.info(f"Confidence Score: {confidence:.4f}")

        _, _, heatmap_path = save_gradcam(temp_path, model, CLASS_NAMES, Path("predictions"))
        st.image(str(heatmap_path), caption="Grad-CAM Heatmap", use_column_width=True)


if __name__ == "__main__":
    main()
