"""Model evaluation utilities including metrics and confusion matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset, class_names: list[str], output_dir: Path) -> dict[str, float]:
    """Evaluate the saved model and write metrics and confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)

    weights = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(np.mean([report[label]["precision"] for label in class_names])),
        "macro_recall": float(np.mean([report[label]["recall"] for label in class_names])),
        "macro_f1_score": float(np.mean([report[label]["f1-score"] for label in class_names])),
    }

    save_confusion_matrix(y_true, y_pred, class_names, output_dir.parent / "graphs")
    logger.info("Saved classification report and confusion matrix")
    return weights


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], graphs_dir: Path) -> None:
    """Save confusion matrix figure to graphs folder."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(graphs_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)
