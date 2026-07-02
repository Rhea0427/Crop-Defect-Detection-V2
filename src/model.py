"""Model construction for sugarcane leaf disease classification."""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

logger = logging.getLogger(__name__)


def build_transfer_model(img_size: tuple[int, int, int] = (224, 224, 3), num_classes: int = 5) -> tf.keras.Model:
    """Build a transfer learning model using EfficientNetB0 as the base."""
    inputs = tf.keras.Input(shape=img_size)
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_transfer")
    logger.info("Built EfficientNetB0 transfer learning model")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> None:
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    logger.info("Compiled model with Adam optimizer and categorical crossentropy")
