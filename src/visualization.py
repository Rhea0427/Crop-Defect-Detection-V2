"""Visualization utilities for dataset exploration and reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Create publication-quality charts for the dataset."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_all_graphs(
        self,
        summary_df: pd.DataFrame,
        image_records: List[dict],
        class_weights_df: pd.DataFrame,
        dimensions_df: pd.DataFrame,
    ) -> None:
        """Generate and save all required charts."""
        self._save_bar_chart(summary_df)
        self._save_pie_chart(summary_df)
        self._save_line_chart(summary_df)
        self._save_horizontal_bar_chart(summary_df)
        self._save_boxplot(dimensions_df)
        self._save_histogram(dimensions_df)
        self._save_countplot(image_records)
        self._save_percentage_distribution_chart(summary_df)
        self._save_class_weight_chart(class_weights_df)
        logger.info("Saved all visualization outputs to %s", self.output_dir)

    def _save_bar_chart(self, summary_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if summary_df.empty:
            self._add_empty_message(ax, "Bar Chart", "No class data available")
        else:
            sns.barplot(data=summary_df, x="class_name", y="image_count", palette="viridis", ax=ax)
            ax.set_title("Image Count per Disease Class")
            ax.set_xlabel("Disease Class")
            ax.set_ylabel("Image Count")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        self._save_figure(fig, "bar_chart.png")

    def _save_pie_chart(self, summary_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if summary_df.empty:
            self._add_empty_message(ax, "Pie Chart", "No class data available")
        else:
            ax.pie(
                summary_df["image_count"],
                labels=summary_df["class_name"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white"},
            )
            ax.set_title("Percentage Distribution of Disease Classes")
        self._save_figure(fig, "pie_chart.png")

    def _save_line_chart(self, summary_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if summary_df.empty:
            self._add_empty_message(ax, "Line Chart", "No class data available")
        else:
            ax.plot(summary_df["class_name"], summary_df["image_count"], marker="o", linewidth=2, color="#1f77b4")
            ax.set_title("Disease Classes vs Number of Images")
            ax.set_xlabel("Disease Class")
            ax.set_ylabel("Image Count")
            ax.grid(True, linestyle="--", alpha=0.5)
        self._save_figure(fig, "line_chart.png")

    def _save_horizontal_bar_chart(self, summary_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if summary_df.empty:
            self._add_empty_message(ax, "Horizontal Bar Chart", "No class data available")
        else:
            sorted_df = summary_df.sort_values("image_count", ascending=True)
            sns.barplot(data=sorted_df, x="image_count", y="class_name", palette="mako", ax=ax)
            ax.set_title("Disease Class Distribution (Horizontal)")
            ax.set_xlabel("Image Count")
            ax.set_ylabel("Disease Class")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
        self._save_figure(fig, "horizontal_bar.png")

    def _save_boxplot(self, dimensions_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if dimensions_df.empty:
            self._add_empty_message(ax, "Box Plot", "No image dimensions available")
        else:
            melted = dimensions_df[["width", "height"]].rename(columns={"width": "Width", "height": "Height"})
            melted = melted.melt(var_name="Dimension", value_name="Pixels")
            sns.boxplot(data=melted, x="Dimension", y="Pixels", palette="Set2", ax=ax)
            ax.set_title("Distribution of Image Width and Height")
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Pixels")
        self._save_figure(fig, "boxplot.png")

    def _save_histogram(self, dimensions_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if dimensions_df.empty:
            self._add_empty_message(ax, "Histogram", "No image dimensions available")
        else:
            combined = pd.concat(
                [dimensions_df["width"].rename("Width"), dimensions_df["height"].rename("Height")],
                axis=1,
            )
            combined.plot.hist(bins=15, alpha=0.7, ax=ax, legend=True)
            ax.set_title("Histogram of Image Sizes")
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.5)
        self._save_figure(fig, "histogram.png")

    def _save_countplot(self, image_records: List[dict]) -> None:
        fig, ax = self._create_figure()
        if not image_records:
            self._add_empty_message(ax, "Count Plot", "No image records available")
        else:
            record_df = pd.DataFrame(image_records)
            sns.countplot(data=record_df, x="class_name", order=sorted(record_df["class_name"].unique()), ax=ax)
            ax.set_title("Count Plot of Disease Classes")
            ax.set_xlabel("Disease Class")
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        self._save_figure(fig, "countplot.png")

    def _save_percentage_distribution_chart(self, summary_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if summary_df.empty:
            self._add_empty_message(ax, "Percentage Distribution", "No class data available")
        else:
            percentage_df = summary_df.copy()
            percentage_df["percentage"] = percentage_df["image_count"] / percentage_df["image_count"].sum() * 100
            sns.barplot(data=percentage_df, x="class_name", y="percentage", palette="rocket", ax=ax)
            ax.set_title("Percentage Distribution of Disease Classes")
            ax.set_xlabel("Disease Class")
            ax.set_ylabel("Percentage")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        self._save_figure(fig, "percentage_distribution.png")

    def _save_class_weight_chart(self, class_weights_df: pd.DataFrame) -> None:
        fig, ax = self._create_figure()
        if class_weights_df.empty:
            self._add_empty_message(ax, "Class Weight Chart", "No class weight data available")
        else:
            sns.barplot(data=class_weights_df, x="class_name", y="class_weight", palette="deep", ax=ax)
            ax.set_title("Machine Learning Class Weights")
            ax.set_xlabel("Disease Class")
            ax.set_ylabel("Class Weight")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        self._save_figure(fig, "class_weight_chart.png")

    def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(10, 6))
        return fig, ax

    def _add_empty_message(self, ax: plt.Axes, title: str, message: str) -> None:
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(title)
        ax.set_axis_off()

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
