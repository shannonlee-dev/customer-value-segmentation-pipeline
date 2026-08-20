"""Lightweight repository rules for the allowed implementation stack."""

import unittest
from pathlib import Path


class RepositoryContractTests(unittest.TestCase):
    def test_pipeline_has_required_public_methods_and_no_forbidden_libraries(self):
        source = Path("src/pipeline.py").read_text(encoding="utf-8")
        for method in ("def load_data", "def handle_missing_values", "def detect_outliers", "def calculate_rfm"):
            self.assertIn(method, source)
        for forbidden in ("PIL", "Pillow", "duckdb", "pyarrow", "parquet"):
            self.assertNotIn(forbidden.lower(), source.lower())

    def test_readme_does_not_claim_real_full_data_results(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("Kaggle", readme)
        self.assertIn("Run All", readme)


if __name__ == "__main__":
    unittest.main()
