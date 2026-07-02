"""Train the transfer learning model for sugarcane leaf disease classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.model import compile_model, build_transfer_model

logger = logging.getLogger(__name__)


def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Train the model and save the best weights and model files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_transfer_model()
    compile_model(model, learning_rate=learning_rate)

    model_path = output_dir / "best_model.keras"
    weights_path = output_dir / "best_weights.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
        ModelCheckpoint(filepath=str(weights_path), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    logger.info("Training completed")
    save_training_plots(history, output_dir.parent / "graphs")
    return model


def save_training_plots(history: tf.keras.callbacks.History, graphs_dir: Path) -> None:
    """Save training accuracy and loss plots to the graphs folder."""
    graphs_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(graphs_dir / "training_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(graphs_dir / "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved training plots")
