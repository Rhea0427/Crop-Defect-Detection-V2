"""Grad-CAM visualization utilities for model explainability."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for an image and a target class."""
    if last_conv_layer_name is None:
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("Unable to find a convolutional layer for Grad-CAM")
        last_conv_layer_name = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return np.array(heatmap)


def save_gradcam(image_path: Path, model: tf.keras.Model, class_names: list[str], output_dir: Path) -> tuple[str, float, Path]:
    """Save a Grad-CAM heatmap for a single input image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    original_image = tf.image.resize(image, (224, 224)).numpy()
    input_image = np.expand_dims(original_image, axis=0)

    predictions = model.predict(input_image, verbose=0)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    heatmap = make_gradcam_heatmap(model, input_image, class_index)

    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, (224, 224)).numpy()
    heatmap = np.squeeze(heatmap, axis=-1)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    output_path = output_dir / "gradcam_prediction.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    predicted_label = class_names[class_index]
    logger.info("Saved Grad-CAM heatmap to %s", output_path)
    return predicted_label, confidence, output_path
