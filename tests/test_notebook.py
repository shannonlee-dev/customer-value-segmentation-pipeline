"""Source contracts for the portable notebook template."""

import tempfile
import unittest
from pathlib import Path

import nbformat

from scripts.build_notebook import build_notebook
from scripts.verify_notebook import inspect_notebook


class NotebookSourceTests(unittest.TestCase):
    def test_generated_notebook_is_clean_and_covers_required_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_notebook(Path(directory) / "analysis_report.ipynb")
            result = inspect_notebook(path, mode="source")
            notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertEqual(result["status"], "PASS")
        for token in ("DataAnalyzer", "Dataset Inventory", ".head()", ".info()", ".describe()", "detect_outliers", "calculate_rfm", "plt.stairs", "ax.bxp", "heatmap", "plt.scatter"):
            self.assertIn(token, source)
        self.assertNotIn("analysis_preview", source)
        self.assertNotIn("PIL", source)
        self.assertNotIn("duckdb", source.lower())

    def test_generated_notebook_reports_full_data_artifact_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_notebook(Path(directory) / "analysis_report.ipynb")
            notebook = nbformat.read(path, as_version=4)

        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("full-data artifact", source)
        self.assertIn("prepare_eda_artifacts", source)
        self.assertIn("format_cache_report", source)


if __name__ == "__main__":
    unittest.main()
