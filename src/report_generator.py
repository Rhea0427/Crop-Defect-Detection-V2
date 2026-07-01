"""Generate a text report and maintain project documentation files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Create analysis reports and ensure project documentation exists."""

    def __init__(self, project_dir: Path | str):
        self.project_dir = Path(project_dir)
        self.reports_dir = self.project_dir / "reports"
        self.graphs_dir = self.project_dir / "graphs"

    def generate_analysis_report(
        self,
        summary_df: pd.DataFrame,
        stats: dict,
        dimensions_df: pd.DataFrame,
        class_weights_df: pd.DataFrame,
    ) -> None:
        """Write a complete text report with summary statistics and recommendations."""
        lines: List[str] = []
        lines.append("Crop Defect Detection Data Analysis Report")
        lines.append("=" * 45)
        lines.append("")
        lines.append("Purpose of Project")
        lines.append("-" * 20)
        lines.append("This project analyzes a sugarcane leaf disease dataset to explore class distribution, image characteristics, and class imbalance using data science techniques.")
        lines.append("")
        lines.append("Dataset Summary")
        lines.append("-" * 20)
        lines.append(f"Total classes: {stats['total_classes']}")
        lines.append(f"Total images: {stats['total_images']}")
        lines.append(f"Largest class: {stats['largest_class']}")
        lines.append(f"Smallest class: {stats['smallest_class']}")
        lines.append(f"Average images per class: {stats['average_images_per_class']:.2f}")
        lines.append(f"Median: {stats['median']:.2f}")
        lines.append(f"Variance: {stats['variance']:.2f}")
        lines.append(f"Standard Deviation: {stats['std_dev']:.2f}")
        lines.append("")
        lines.append("Disease Distribution")
        lines.append("-" * 20)
        if summary_df.empty:
            lines.append("No class distribution data was found.")
        else:
            for _, row in summary_df.iterrows():
                lines.append(f"- {row['class_name']}: {int(row['image_count'])} images")
        lines.append("")
        lines.append("Image Statistics")
        lines.append("-" * 20)
        if dimensions_df.empty:
            lines.append("No image dimensions were available for analysis.")
        else:
            lines.append(f"Average width: {dimensions_df['width'].mean():.2f} pixels")
            lines.append(f"Average height: {dimensions_df['height'].mean():.2f} pixels")
            lines.append(f"Maximum width: {dimensions_df['width'].max()} pixels")
            lines.append(f"Maximum height: {dimensions_df['height'].max()} pixels")
            lines.append(f"Minimum width: {dimensions_df['width'].min()} pixels")
            lines.append(f"Minimum height: {dimensions_df['height'].min()} pixels")
        lines.append("")
        lines.append("Class Imbalance")
        lines.append("-" * 20)
        if class_weights_df.empty:
            lines.append("No class weight data was generated.")
        else:
            for _, row in class_weights_df.iterrows():
                lines.append(f"- {row['class_name']}: weight {row['class_weight']:.2f}")
        lines.append("")
        lines.append("Recommendations")
        lines.append("-" * 20)
        lines.append("- Keep the analysis pipeline modular so new datasets can be evaluated easily.")
        lines.append("- Use class weights or stratified sampling if the dataset is imbalanced.")
        lines.append("- Validate image preprocessing steps before using the data in any model pipeline.")
        lines.append("")
        lines.append("Conclusion")
        lines.append("-" * 20)
        lines.append("The dataset was successfully scanned, summarized, visualized, and documented. The analysis provides a strong foundation for future machine learning experiments and reporting.")

        report_path = self.reports_dir / "analysis_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        class_weights_path = self.reports_dir / "class_weights.csv"
        if not class_weights_path.exists() and not class_weights_df.empty:
            class_weights_df.to_csv(class_weights_path, index=False)
        elif not class_weights_df.empty:
            class_weights_df.to_csv(class_weights_path, index=False)
        logger.info("Wrote analysis report to %s", report_path)

    def ensure_project_files(self) -> None:
        """Create or refresh core project documentation files."""
        readme_path = self.project_dir / "README.md"
        requirements_path = self.project_dir / "requirements.txt"
        gitignore_path = self.project_dir / ".gitignore"
        license_path = self.project_dir / "LICENSE"

        if not readme_path.exists():
            readme_path.write_text(self._readme_content(), encoding="utf-8")
        if not requirements_path.exists():
            requirements_path.write_text(self._requirements_content(), encoding="utf-8")
        if not gitignore_path.exists():
            gitignore_path.write_text(self._gitignore_content(), encoding="utf-8")
        if not license_path.exists():
            license_path.write_text(self._license_content(), encoding="utf-8")

        logger.info("Ensured project documentation files exist")

    def _readme_content(self) -> str:
        return """# Crop-Defect-Detection-V1

## Project Overview
This project performs exploratory data analysis on a sugarcane leaf disease image dataset. It scans the dataset folders, counts images per class, computes descriptive statistics, generates visualizations, and writes a summary report. No deep learning model is trained.

## Dataset Description
The dataset contains images belonging to five classes:
- Healthy
- Mosaic
- RedRot
- Rust
- Yellow

## Folder Structure
- dataset/: contains the raw image folders
- graphs/: stores generated PNG visualizations
- notebooks/: place for exploratory notebooks
- reports/: stores reports and CSV outputs
- src/: reusable Python modules

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- scikit-learn
- Pillow

## Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- opencv-python
- scikit-learn
- Pillow

## Screenshots Placeholder
Add screenshots of the generated charts here.

## How to Run
```bash
python main.py
```

## Expected Output
The script will generate:
- PNG charts in the graphs folder
- analysis_report.txt in the reports folder
- class_weights.csv in the reports folder

## Future Improvements
- Add image preprocessing and augmentation pipelines
- Integrate a classifier for disease prediction
- Add automated notebook templates for experimentation
"""

    def _requirements_content(self) -> str:
        return """pandas
numpy
matplotlib
seaborn
opencv-python
scikit-learn
Pillow
"""

    def _gitignore_content(self) -> str:
        return """__pycache__/
*.py[cod]
*.png
*.csv
*.txt
.venv/
.ipynb_checkpoints/
"""

    def _license_content(self) -> str:
        return """MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. INFRINGEMENT. SHALL THE SOFTWARE BE
CONSIDERED AS A CONTRACTUAL TORT OR LIABLE FOR THE CLAIMS OF DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF SUCH DAMAGE OR NOT.
"""
