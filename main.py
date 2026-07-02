"""Main pipeline for training and evaluating the sugarcane leaf disease model."""

from __future__ import annotations

import logging
from pathlib import Path

from src.augmentation import apply_augmentation
from src.evaluate import evaluate_model
from src.grad_cam import save_gradcam
from src.predict import load_model_and_weights, predict_image
from src.preprocess import prepare_datasets
from src.train import train_model


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    models_dir = project_dir / "models"
    weights_dir = project_dir / "weights"
    graphs_dir = project_dir / "graphs"
    reports_dir = project_dir / "reports"
    predictions_dir = project_dir / "predictions"

    graphs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting deep learning pipeline")

        train_dataset, val_dataset, test_dataset, class_names = prepare_datasets(dataset_dir)
        train_dataset = apply_augmentation(train_dataset, batch_size=32)

        model_path = models_dir / "best_model.keras"
        weights_path = weights_dir / "best_weights.keras"

        if model_path.exists():
            logger.info("Loading existing model from %s", model_path)
            model = load_model_and_weights(model_path, weights_path)
        else:
            logger.info("Training a new model")
            model = train_model(train_dataset, val_dataset, models_dir)
            model.save(model_path)
            model.save_weights(weights_path)

        logger.info("Evaluating model")
        evaluation_metrics = evaluate_model(model, test_dataset, class_names, reports_dir)
        logger.info("Evaluation metrics: %s", evaluation_metrics)

        sample_image = None
        for ext in ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff", "*.webp"):
            sample_image = next((dataset_dir / class_names[0]).glob(ext), None)
            if sample_image is not None:
                break

        if sample_image is not None and sample_image.exists():
            predicted_label, confidence = predict_image(model, sample_image, class_names)
            logger.info("Sample prediction: %s (%.4f)", predicted_label, confidence)
            save_gradcam(sample_image, model, class_names, predictions_dir)

        logger.info("Pipeline finished successfully")
        print("Success: Deep learning pipeline finished successfully.")
        print(f"Model saved to: {model_path}")
        print(f"Weights saved to: {weights_path}")
        print(f"Reports saved to: {reports_dir}")
        print(f"Graphs saved to: {graphs_dir}")
        print(f"Predictions saved to: {predictions_dir}")
    except Exception as exc:
        logger.exception("Deep learning pipeline failed: %s", exc)
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
