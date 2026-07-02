"""Make predictions on single sugarcane leaf images."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_model_and_weights(model_path: Path, weights_path: Path) -> tf.keras.Model:
    """Load the saved model and weights from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    if weights_path.exists():
        model.load_weights(weights_path)
        logger.info("Loaded model and weights")
    return model


def preprocess_image(image_path: Path, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Read a single image and prepare it for prediction."""
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, img_size)
    return image.numpy()[None, ...]


def predict_image(model: tf.keras.Model, image_path: Path, class_names: list[str]) -> tuple[str, float]:
    """Predict the disease class for a single image."""
    image = preprocess_image(image_path)
    predictions = model.predict(image, verbose=0)
    confidence = float(np.max(predictions))
    predicted_label = class_names[int(np.argmax(predictions))]
    logger.info("Predicted %s with confidence %.4f", predicted_label, confidence)
    return predicted_label, confidence
