"""Validate the distributable notebook's source-level contract."""

import tempfile
import unittest
from pathlib import Path

import nbformat

from scripts.build_notebook import build_notebook
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
    if any(token in source for token in ("feature_product_ids", "sampled_image_features", "cohort_size", "stable_customer_ids")):
        raise ValueError("Source notebook contains sampled analysis")
    return {"status": "PASS", "cell_count": len(notebook.cells)}


class NotebookContractTest(unittest.TestCase):
    def test_committed_notebook_is_clean_and_full_dataset_scoped(self) -> None:
        project = Path(__file__).resolve().parents[1]
        summary = inspect_source_notebook(project / "notebooks" / "analysis_report.ipynb")

        self.assertEqual(summary["status"], "PASS")

    def test_notebook_generator_produces_a_valid_source_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = inspect_source_notebook(build_notebook(Path(directory) / "analysis_report.ipynb"))

        self.assertEqual(summary["status"], "PASS")

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
        self.assertIn('if "product_name_length" not in product_features.columns', code)
        self.assertIn('article_features[["product_id", "product_name_length"]]', code)
        self.assertIn('product_features_path = Path("/kaggle/working/product_images_enriched.csv")', code)
        self.assertIn("available_fonts", code)
        self.assertIn('plt.title(CHART_TEXT["price_title"])', code)
        self.assertIn("price_image_mean_corr", code)
        self.assertIn("transaction_path", code)
