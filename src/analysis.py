"""Core analysis functions for dataset statistics and class weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Compute summary statistics and class balance information."""

    def compute_dataset_statistics(self, summary_df: pd.DataFrame) -> Dict[str, float | int | str]:
        """Calculate descriptive statistics from class counts."""
        if summary_df.empty:
            return {
                "total_classes": 0,
                "total_images": 0,
                "largest_class": "N/A",
                "smallest_class": "N/A",
                "average_images_per_class": 0.0,
                "median": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
            }

        counts = summary_df["image_count"].astype(int)
        largest_row = summary_df.loc[counts.idxmax()]
        smallest_row = summary_df.loc[counts.idxmin()]

        return {
            "total_classes": int(len(summary_df)),
            "total_images": int(counts.sum()),
            "largest_class": f"{largest_row['class_name']} ({int(largest_row['image_count'])})",
            "smallest_class": f"{smallest_row['class_name']} ({int(smallest_row['image_count'])})",
            "average_images_per_class": float(counts.mean()),
            "median": float(counts.median()),
            "variance": float(counts.var()),
            "std_dev": float(counts.std()),
        }

    def compute_class_weights(self, image_records: List[dict]) -> pd.DataFrame:
        """Use sklearn to compute class weights for imbalance handling."""
        if not image_records:
            return pd.DataFrame(columns=["class_name", "class_weight"])

        labels = [record["class_name"] for record in image_records]
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weight_df = pd.DataFrame({"class_name": classes, "class_weight": weights})
        return weight_df.sort_values("class_name").reset_index(drop=True)

    def analyze_image_dimensions(self, image_records: List[dict]) -> pd.DataFrame:
        """Read every image and calculate dimensions."""
        dimensions: List[dict] = []

        for record in image_records:
            image_path = Path(record["image_path"])
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    dimensions.append(
                        {
                            "class_name": record["class_name"],
                            "image_path": str(image_path),
                            "width": int(width),
                            "height": int(height),
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive handling
                logger.warning("Unable to read image %s: %s", image_path, exc)

        dimensions_df = pd.DataFrame(dimensions)
        if dimensions_df.empty:
            return pd.DataFrame(columns=["class_name", "image_path", "width", "height"])

        return dimensions_df
