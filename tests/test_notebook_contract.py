"""Validate the distributable notebook's source-level contract."""

import tempfile
import unittest
from pathlib import Path

import nbformat

from scripts.build_notebook import build_notebook


REQUIRED_MARKERS = (
    "분석 범위: 전체 데이터셋",
    "고객 표본 추출: 없음",
    "상품 표본 추출: 없음",
    "이미지 분석 표본 추출: 없음",
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
