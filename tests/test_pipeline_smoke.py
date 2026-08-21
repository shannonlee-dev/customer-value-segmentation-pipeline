"""Exercise the pipeline against a minimal complete H&M-shaped dataset."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

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
            self.assertNotIn("product_name_length", analyzer.articles.columns)
            self.assertIn("product_name_length", image_features.columns)
            self.assertNotIn("product_name_length", pd.read_csv(analyzer.articles_path, nrows=1).columns)
            self.assertIn("product_name_length", pd.read_csv(analyzer.images_path, nrows=1).columns)
            self.assertEqual(iqr["outlier_count"], 0)
            self.assertEqual(len(rfm), 3)
            self.assertTrue(Path(eda["monthly_summary_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
