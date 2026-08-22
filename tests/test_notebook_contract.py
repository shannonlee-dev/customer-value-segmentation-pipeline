"""Validate the distributable notebook's source-level contract."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import nbformat

from scripts.build_notebook import CHART_STYLE_SOURCE, build_notebook
from scripts.build_precomputed_notebook import build_notebook as build_precomputed_notebook


REQUIRED_MARKERS = (
    "분석 범위: 전체 데이터셋",
)


def inspect_source_notebook(path: Path) -> dict[str, int | str]:
    """Validate that a distributable notebook is clean and full-data scoped."""
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code"):
        raise ValueError("Source notebook must be clean")
    if not all(marker in source for marker in REQUIRED_MARKERS):
        raise ValueError("Source notebook is missing full-data markers")
    if any(token in source for token in ("feature_product_ids", "sampled_product_features", "cohort_size", "stable_customer_ids")):
        raise ValueError("Source notebook contains sampled analysis")
    return {"status": "PASS", "cell_count": len(notebook.cells)}


class NotebookContractTest(unittest.TestCase):
    def test_chart_style_renders_every_label_without_missing_glyph_warnings(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        namespace = {"matplotlib": matplotlib}
        with patch.object(matplotlib.font_manager.fontManager, "ttflist", []):
            exec(CHART_STYLE_SOURCE, namespace)
        self.assertIs(namespace["CHART_TEXT"], namespace["ENGLISH_CHART_TEXT"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure = plt.figure()
            for index, label in enumerate(namespace["CHART_TEXT"].values()):
                figure.text(0, index / 20, label)
            figure.canvas.draw()
            plt.close(figure)

        missing_glyph_warnings = [
            warning for warning in caught if "Glyph" in str(warning.message) and "missing from font" in str(warning.message)
        ]
        self.assertEqual(missing_glyph_warnings, [])

    def test_readme_links_to_the_pipeline_flow_diagram(self) -> None:
        project = Path(__file__).resolve().parents[1]
        readme = (project / "README.md").read_text(encoding="utf-8")

        self.assertIn("assets/pipeline-flow.svg", readme)
        self.assertTrue((project / "assets" / "pipeline-flow.svg").is_file())

    def test_committed_notebook_is_clean_and_full_dataset_scoped(self) -> None:
        project = Path(__file__).resolve().parents[1]
        summary = inspect_source_notebook(project / "notebooks" / "analysis_report.ipynb")

        self.assertEqual(summary["status"], "PASS")

    def test_notebook_generator_produces_a_valid_source_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = inspect_source_notebook(build_notebook(Path(directory) / "analysis_report.ipynb"))

        self.assertEqual(summary["status"], "PASS")

    def test_analysis_notebook_has_a_dedicated_enriched_images_export_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = nbformat.read(build_notebook(Path(directory) / "analysis_report.ipynb"), as_version=4)

        export_cells = [
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and 'output_path = Path("/kaggle/working/product_features.csv")' in cell.source
        ]
        self.assertEqual(len(export_cells), 1)
        self.assertIn("product_features.to_csv(output_path, index=False)", export_cells[0])

    def test_analysis_notebook_marks_iqr_outliers_and_explains_each_chart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = nbformat.read(build_notebook(Path(directory) / "analysis_report.ipynb"), as_version=4)

        chart_cells = [
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and 'plt.stairs(eda["histogram_counts"]' in cell.source
        ]
        self.assertEqual(len(chart_cells), 1)
        chart_code = chart_cells[0]
        self.assertIn("unique_price_outliers", chart_code)
        self.assertIn("showfliers=True", chart_code)
        self.assertIn('flierprops={"marker": "o"', chart_code)
        self.assertEqual(chart_code.count('display(Markdown(f"**차트 인사이트:**'), 6)

    def test_every_notebook_bootstraps_the_project_only_on_kaggle(self) -> None:
        """Catch a missing Kaggle clone fallback or accidental local auto-clone."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebooks = (
                nbformat.read(build_notebook(root / "analysis_report.ipynb"), as_version=4),
                nbformat.read(build_precomputed_notebook(root / "precomputed_report.ipynb"), as_version=4),
            )

        for notebook in notebooks:
            setup = notebook.cells[0].source
            self.assertIn("https://github.com/shannonlee-dev/customer-value-segmentation-pipeline.git", setup)
            self.assertIn('Path("/kaggle/working/customer-value-segmentation-pipeline")', setup)
            self.assertIn("subprocess.run", setup)
            self.assertIn("로컬에서는 자동으로 저장소를 clone하지 않습니다", setup)
            self.assertNotIn('rglob("customer-value-segmentation-pipeline/src/pipeline.py")', setup)

    def test_precomputed_notebook_keeps_the_main_notebook_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = nbformat.read(build_notebook(root / "analysis_report.ipynb"), as_version=4)
            precomputed = nbformat.read(build_precomputed_notebook(root / "precomputed_report.ipynb"), as_version=4)
            code = "\n".join(cell.source for cell in precomputed.cells if cell.cell_type == "code")

        self.assertEqual(len(precomputed.cells), len(main.cells))
        self.assertEqual(
            [cell.source for cell in precomputed.cells if cell.cell_type == "markdown"],
            [cell.source for cell in main.cells if cell.cell_type == "markdown"],
        )
        self.assertIn("aggregates/eda_summary.json", code)
        self.assertNotIn("DataAnalyzer", code)
        self.assertNotIn("src.reporting", code)
        self.assertNotIn("customer-value-segmentation-pipeline/src", code)
        self.assertNotIn('if "product_name_length" not in articles.columns', code)
        self.assertNotIn('articles[["product_id", "product_name_length"]]', code)
        self.assertIn('output_path = Path("/kaggle/working/product_features.csv")', code)
        self.assertIn('product_features_path = PRECOMPUTED_ROOT / "features" / "product_features" / "product_features.csv"', code)
        self.assertIn('print(f"저장 완료: {output_path}")', code)
        self.assertIn("available_fonts", code)
        self.assertIn('plt.title(CHART_TEXT["price_title"])', code)
        self.assertIn("price_image_mean_corr", code)
        self.assertIn("price_image_mean_corr, unique_price_outliers = transaction_feature_correlation", code)
        self.assertIn("transaction_path", code)

    def test_precomputed_notebook_has_a_dedicated_enriched_images_export_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = nbformat.read(
                build_precomputed_notebook(Path(directory) / "precomputed_report.ipynb"),
                as_version=4,
            )

        export_cells = [
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and 'output_path = Path("/kaggle/working/product_features.csv")' in cell.source
        ]
        self.assertEqual(len(export_cells), 1)
        self.assertIn("product_features.head()", export_cells[0])
