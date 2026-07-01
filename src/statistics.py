"""Helpers for printing dataset statistics to the console."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class StatisticsReporter:
    """Print a human-friendly summary of the dataset statistics."""

    def print_summary(
        self,
        summary_df: pd.DataFrame,
        stats: dict,
        dimensions_df: pd.DataFrame,
    ) -> None:
        """Print the dataset overview to the terminal."""
        print("\nDataset Summary")
        print("=" * 50)
        print(f"Total Classes: {stats['total_classes']}")
        print(f"Total Images: {stats['total_images']}")
        print("\nImages Per Class")
        if summary_df.empty:
            print("No data available")
        else:
            print(summary_df.to_string(index=False))

        if not summary_df.empty:
            percentage_df = summary_df.copy()
            percentage_df["percentage"] = percentage_df["image_count"] / percentage_df["image_count"].sum() * 100
            print("\nPercentage Distribution")
            print(percentage_df[["class_name", "percentage"]].to_string(index=False))

        print(f"\nLargest Class: {stats['largest_class']}")
        print(f"Smallest Class: {stats['smallest_class']}")
        print(f"Average Images Per Class: {stats['average_images_per_class']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Variance: {stats['variance']:.2f}")
        print(f"Standard Deviation: {stats['std_dev']:.2f}")

        print("\nImage Dimension Summary")
        if dimensions_df.empty:
            print("No image dimensions available")
        else:
            print(f"Average Width: {dimensions_df['width'].mean():.2f}")
            print(f"Average Height: {dimensions_df['height'].mean():.2f}")
            print(f"Maximum Width: {dimensions_df['width'].max()}")
            print(f"Maximum Height: {dimensions_df['height'].max()}")
            print(f"Minimum Width: {dimensions_df['width'].min()}")
            print(f"Minimum Height: {dimensions_df['height'].min()}")

        logger.info("Displayed dataset summary in the console")
