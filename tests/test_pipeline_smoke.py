"""Exercise the pipeline against a minimal complete H&M-shaped dataset."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
import nbformat

from scripts.build_notebook import build_notebook
from src.pipeline import DataAnalyzer
from src.runtime import discover_runtime


def write_fixture(raw: Path) -> None:
    image_dir = raw / "images" / "001"
    image_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            ["2020-01-01", "customer-a", "0010000001", 0.10, 1],
            ["2020-01-02", "customer-a", "0010000002", 0.20, 1],
            ["2020-01-03", "customer-b", "0010000001", 0.30, 2],
            ["2020-01-04", "customer-c", "0010000003", 0.50, 2],
        ],
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    ).to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame(
        [["0010000001", "Item one", "Group A"], ["0010000002", "Item two", "Group B"], ["0010000003", "No image", "Group C"]],
        columns=["article_id", "prod_name", "product_group_name"],
    ).to_csv(raw / "articles.csv", index=False)
    pd.DataFrame(
        [["customer-a", 25, "ACTIVE", "Regularly"], ["customer-b", None, "ACTIVE", "None"], ["customer-c", 45, "PRE-CREATE", "Monthly"]],
        columns=["customer_id", "age", "club_member_status", "fashion_news_frequency"],
    ).to_csv(raw / "customers.csv", index=False)
    mpimg.imsave(image_dir / "0010000001.jpg", np.full((10, 8, 3), 0.2))
    mpimg.imsave(image_dir / "0010000002.jpg", np.full((6, 7), 0.7), cmap="gray")


class PipelineSmokeTest(unittest.TestCase):
    def test_extract_image_features_calculates_text_and_pixel_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_fixture(raw)
            context = discover_runtime(
                Path.cwd(),
                {"HM_RAW_DATA_DIR": str(raw), "HM_RUNTIME_DIR": str(root / "runtime")},
            )
            analyzer = DataAnalyzer(context)

            record = analyzer._extract_image_features(raw, "0010000001", "Item one", "images/001/0010000001.jpg")

            self.assertEqual(record["product_name_length"], 8)
            self.assertTrue(np.isfinite(record["image_mean"]))
            self.assertTrue(np.isfinite(record["image_std"]))

    def test_engineer_features_requires_raw_data_when_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            precomputed = root / "precomputed"
            precomputed.mkdir()
            context = discover_runtime(
                Path.cwd(),
                {"HM_PRECOMPUTED_DIR": str(precomputed), "HM_RUNTIME_DIR": str(root / "runtime")},
            )
            analyzer = DataAnalyzer(context)
            analyzer.articles = pd.DataFrame(
                [["0010000001", "Item one", "images/001/0010000001.jpg"]],
                columns=["product_id", "product_name", "image_path"],
            )

            with self.assertRaisesRegex(ValueError, "no raw H&M dataset"):
                analyzer.engineer_features(force=True)

    def test_generated_notebook_uses_product_name_length_from_image_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = build_notebook(Path(directory) / "analysis_report.ipynb")
            rendered = nbformat.read(notebook, as_version=4)
            code = "\n".join(cell.source for cell in rendered.cells if cell.cell_type == "code")

            self.assertIn("product_features = image_features", code)
            self.assertNotIn('articles[["product_id", "product_name_length"]]', code)

    def test_pipeline_generates_all_artifacts_from_a_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_fixture(raw)
            context = discover_runtime(
                Path.cwd(),
                {"HM_RAW_DATA_DIR": str(raw), "HM_RUNTIME_DIR": str(root / "runtime")},
            )
            analyzer = DataAnalyzer(context, chunksize=2)

            summary = analyzer.load_data()
            image_features = analyzer.engineer_features()
            iqr = analyzer.detect_outliers()
            rfm = analyzer.calculate_rfm()
            eda = analyzer.prepare_eda_artifacts()

            self.assertEqual(summary["transaction_rows"], 4)
            self.assertEqual(len(image_features), 3)
            self.assertNotIn("sales_channel_id", pd.read_csv(analyzer.transactions_path, nrows=1).columns)
            self.assertNotIn("fashion_news_frequency", analyzer.customers.columns)
            self.assertNotIn("age_was_missing", analyzer.customers.columns)
            self.assertNotIn("category", analyzer.articles.columns)
            self.assertNotIn("image_status", image_features.columns)
            self.assertNotIn("product_name_length", analyzer.articles.columns)
            self.assertIn("product_name_length", image_features.columns)
            self.assertNotIn("product_name_length", pd.read_csv(analyzer.articles_path, nrows=1).columns)
            self.assertIn("product_name_length", pd.read_csv(analyzer.images_path, nrows=1).columns)
            self.assertEqual(iqr["outlier_count"], 0)
            self.assertEqual(len(rfm), 3)
            self.assertTrue(Path(eda["monthly_summary_path"]).is_file())

    def test_load_data_rebuilds_caches_that_contain_removed_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_fixture(raw)
            context = discover_runtime(
                Path.cwd(),
                {"HM_RAW_DATA_DIR": str(raw), "HM_RUNTIME_DIR": str(root / "runtime")},
            )
            analyzer = DataAnalyzer(context, chunksize=2)
            analyzer.load_data()
            analyzer.engineer_features()

            for path, column in (
                (analyzer.transactions_path, "sales_channel_id"),
                (analyzer.customers_path, "fashion_news_frequency"),
                (analyzer.articles_path, "category"),
                (analyzer.images_path, "image_status"),
            ):
                frame = pd.read_csv(path)
                frame[column] = "obsolete"
                frame.to_csv(path, index=False)

            refreshed = DataAnalyzer(context, chunksize=2)
            refreshed.load_data()
            image_features = refreshed.engineer_features()

            self.assertNotIn("sales_channel_id", pd.read_csv(refreshed.transactions_path, nrows=1).columns)
            self.assertNotIn("fashion_news_frequency", refreshed.customers.columns)
            self.assertNotIn("category", refreshed.articles.columns)
            self.assertNotIn("image_status", image_features.columns)


if __name__ == "__main__":
    unittest.main()
