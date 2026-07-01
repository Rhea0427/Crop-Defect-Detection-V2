"""Utilities for loading and validating image dataset folders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load image class folders from a dataset directory."""

    def __init__(self, dataset_dir: Path | str):
        self.dataset_dir = Path(dataset_dir)
        self.allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def scan_dataset(self) -> Tuple[pd.DataFrame, List[dict]]:
        """Scan class folders and return a summary dataframe and image records."""
        if not self.dataset_dir.exists():
            logger.warning("Dataset directory does not exist: %s", self.dataset_dir)
            return self._empty_result()

        class_dirs = sorted([path for path in self.dataset_dir.iterdir() if path.is_dir()])
        if not class_dirs:
            logger.warning("No class folders found in dataset directory")
            return self._empty_result()

        summary_rows: List[dict] = []
        image_records: List[dict] = []

        for class_dir in class_dirs:
            image_files = sorted(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in self.allowed_extensions
                ]
            )
            image_count = len(image_files)
            summary_rows.append(
                {
                    "class_name": class_dir.name,
                    "image_count": image_count,
                }
            )
            for image_path in image_files:
                image_records.append(
                    {
                        "class_name": class_dir.name,
                        "image_path": str(image_path),
                    }
                )

        if not summary_rows:
            return self._empty_result()

        summary_df = pd.DataFrame(summary_rows).sort_values("class_name").reset_index(drop=True)
        logger.info("Discovered %s classes and %s images", len(summary_df), len(image_records))
        return summary_df, image_records

    def _empty_result(self) -> Tuple[pd.DataFrame, List[dict]]:
        empty_summary = pd.DataFrame(columns=["class_name", "image_count"])
        return empty_summary, []
