"""Entry point for the crop defect detection analytics pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.analysis import DatasetAnalyzer
from src.data_loader import DatasetLoader
from src.report_generator import ReportGenerator
from src.statistics import StatisticsReporter
from src.visualization import VisualizationGenerator


def configure_logging() -> None:
    """Configure the project logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Run the full dataset analysis pipeline."""
    configure_logging()
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    graphs_dir = project_dir / "graphs"
    reports_dir = project_dir / "reports"

    graphs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting crop defect detection data analysis pipeline")
        logger.info("Loading dataset from %s", dataset_dir)

        loader = DatasetLoader(dataset_dir)
        summary_df, image_records = loader.scan_dataset()

        logger.info("Dataset scan complete. Preparing analysis and visuals")

        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_dataset_statistics(summary_df)
        class_weights_df = analyzer.compute_class_weights(image_records)
        dimensions_df = analyzer.analyze_image_dimensions(image_records)

        class_weights_df.to_csv(reports_dir / "class_weights.csv", index=False)

        statistics_reporter = StatisticsReporter()
        statistics_reporter.print_summary(summary_df, stats, dimensions_df)

        visualizer = VisualizationGenerator(graphs_dir)
        visualizer.generate_all_graphs(summary_df, image_records, class_weights_df, dimensions_df)

        report_generator = ReportGenerator(project_dir)
        report_generator.generate_analysis_report(summary_df, stats, dimensions_df, class_weights_df)
        report_generator.ensure_project_files()

        logger.info("Pipeline completed successfully")
        print("\nSuccess: Analysis pipeline completed successfully.")
        print(f"Reports saved to: {reports_dir}")
        print(f"Graphs saved to: {graphs_dir}")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        print(f"\nError: The analysis pipeline failed: {exc}")


if __name__ == "__main__":
    main()
