"""Synthetic end-to-end contracts for the portable full-data facade."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from src.pipeline import DataAnalyzer
from src.runtime import discover_runtime


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        images = self.raw / "images" / "001"
        images.mkdir(parents=True)
        pd.DataFrame([
            ["2020-01-01", "c1", "0010000001", 0.10, 1], ["2020-01-01", "c1", "0010000002", 0.20, 1],
            ["2020-01-03", "c1", "0010000001", 0.30, 2], ["2020-01-02", "c2", "0010000002", 0.40, 1],
            ["2020-01-04", "c3", "0010000003", 0.50, 2], ["2020-01-05", "c4", "0010000004", 3.00, 2],
        ], columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]).to_csv(self.raw / "transactions_train.csv", index=False)
        pd.DataFrame([
            ["0010000001", "RGB item", "Group A"], ["0010000002", "Gray item", "Group A"],
            ["0010000003", "RGBA item", "Group B"], ["0010000004", "Missing item", "Group C"],
        ], columns=["article_id", "prod_name", "product_group_name"]).to_csv(self.raw / "articles.csv", index=False)
        pd.DataFrame([
            ["c1", 20, "ACTIVE", "Regularly"], ["c2", None, "ACTIVE", "None"],
            ["c3", 40, "PRE-CREATE", "Monthly"], ["c4", 50, "ACTIVE", "Regularly"],
        ], columns=["customer_id", "age", "club_member_status", "fashion_news_frequency"]).to_csv(self.raw / "customers.csv", index=False)
        mpimg.imsave(images / "0010000001.jpg", np.full((4, 6, 3), 0.1))
        mpimg.imsave(images / "0010000002.jpg", np.full((3, 5), 0.2), cmap="gray")
        mpimg.imsave(images / "0010000003.jpg", np.full((5, 2, 4), 0.3))

    def analyzer(self):
        context = discover_runtime(self.root, {"HM_RAW_DATA_DIR": str(self.raw), "HM_RUNTIME_DIR": str(self.root / "runtime")}, self.root / "missing")
        return DataAnalyzer(context, chunksize=2)

    def test_loads_every_transaction_and_imputes_customer_age(self):
        analyzer = self.analyzer()
        summary = analyzer.load_data()
        self.assertEqual(summary["transaction_rows"], 6)
        self.assertEqual(len(pd.read_csv(analyzer.transactions_path)), 6)
        customers = pd.read_csv(analyzer.customers_path)
        self.assertFalse(customers["age"].isna().any())
        self.assertTrue(customers.loc[customers.customer_id == "c2", "age_was_missing"].item())

    def test_normalized_tables_keep_their_own_grains(self):
        analyzer = self.analyzer()
        analyzer.load_data()
        features = analyzer.engineer_features()
        self.assertEqual(len(features), 4)
        self.assertEqual(set(features["image_status"]), {"ok", "missing"})
        self.assertEqual(len(pd.read_csv(analyzer.transactions_path)), 6)
        self.assertEqual(len(pd.read_csv(analyzer.customers_path)), 4)
        self.assertEqual(len(pd.read_csv(analyzer.articles_path)), 4)
        self.assertFalse(hasattr(analyzer, "analysis_preview"))

    def test_image_features_use_every_pixel_of_each_decoded_image(self):
        image_path = self.raw / "images" / "001" / "0010000001.jpg"
        gradient = np.arange(70 * 70 * 3, dtype=np.uint8).reshape(70, 70, 3)
        mpimg.imsave(image_path, gradient)
        expected = np.asarray(mpimg.imread(image_path))
        analyzer = self.analyzer()
        analyzer.load_data()
        features = analyzer.engineer_features().set_index("product_id")
        self.assertAlmostEqual(features.loc["0010000001", "image_mean"], float(np.mean(expected)), places=10)
        self.assertAlmostEqual(features.loc["0010000001", "image_std"], float(np.std(expected)), places=10)

    def test_image_feature_cache_reuses_existing_csv_without_reopening_images(self):
        analyzer = self.analyzer()
        analyzer.load_data()
        first = analyzer.engineer_features().set_index("product_id")
        (self.raw / "images" / "001" / "0010000001.jpg").unlink()

        cached = analyzer.engineer_features().set_index("product_id")

        self.assertEqual(cached.loc["0010000001", "image_status"], "ok")
        self.assertEqual(cached.loc["0010000001", "image_mean"], first.loc["0010000001", "image_mean"])

    def test_full_iqr_and_partitioned_rfm_keep_unique_purchase_dates(self):
        analyzer = self.analyzer()
        analyzer.load_data()
        iqr = analyzer.detect_outliers()
        rfm = analyzer.calculate_rfm(partition_count=2)
        self.assertGreaterEqual(iqr["outlier_count"], 0)
        self.assertEqual(len(rfm), 4)
        self.assertEqual(rfm.loc[rfm.customer_id == "c1", "frequency"].item(), 2)
        self.assertEqual(rfm["recency"].min(), 1)


if __name__ == "__main__":
    unittest.main()
