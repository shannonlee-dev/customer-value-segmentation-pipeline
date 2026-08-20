"""Contract tests for the multimodal customer-analysis pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from src.pipeline import DataAnalyzer


class DataAnalyzerLoadAndImputationTests(unittest.TestCase):
    """Loading and numeric missing-value behavior."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.raw_dir = Path(self.temporary_directory.name) / "raw"
        self.raw_dir.mkdir()
        self.csv_path = Path(self.temporary_directory.name) / "cohort.csv"
        pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "customer_id": ["c1", "c2", "c3", "c4"],
                "product_id": ["p1", "p2", "p3", "p4"],
                "product_name": ["Top", "Dress", "Jeans", "Coat"],
                "image_path": ["one.jpg", "two.jpg", "three.jpg", "four.jpg"],
                "unit_price": [10.0, 20.0, 30.0, 40.0],
                "age": [20.0, 30.0, 40.0, 50.0],
                "club_member_status": ["ACTIVE", "ACTIVE", "PRE-CREATE", "PRE-CREATE"],
            }
        ).to_csv(self.csv_path, index=False)

    def test_load_data_parses_dates_and_imputes_an_all_missing_group_from_global_median(self):
        """A group with no numeric values falls back to the cohort median."""
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        frame = analyzer.load_data()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["order_date"]))

        frame.loc[:1, "club_member_status"] = "all-missing"
        frame.loc[:1, "age"] = np.nan
        expected_global = frame["age"].median()

        result = analyzer.handle_missing("age", "club_member_status", "median")

        self.assertFalse(result["age"].isna().any())
        self.assertTrue((result.loc[:1, "age"] == expected_global).all())

    def test_load_data_rejects_a_missing_csv(self):
        """Missing input must not be silently treated as an empty cohort."""
        analyzer = DataAnalyzer(self.csv_path.with_name("missing.csv"), image_root=self.raw_dir)

        with self.assertRaises(FileNotFoundError):
            analyzer.load_data()

    def test_handle_missing_rejects_missing_columns(self):
        """A misspelled input column cannot produce a partial result."""
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        analyzer.load_data()

        with self.assertRaises(KeyError):
            analyzer.handle_missing("not_a_column", "club_member_status")

    def test_handle_missing_rejects_nonnumeric_columns(self):
        """Median imputation is only defined for numeric data."""
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        analyzer.load_data()

        with self.assertRaises(TypeError):
            analyzer.handle_missing("club_member_status", "customer_id")

    def test_load_data_rejects_an_empty_csv(self):
        """Analysis requires at least one cohort row."""
        empty_path = Path(self.temporary_directory.name) / "empty.csv"
        pd.DataFrame(columns=["order_date", "unit_price", "age"]).to_csv(empty_path, index=False)
        analyzer = DataAnalyzer(empty_path, image_root=self.raw_dir)

        with self.assertRaises(ValueError):
            analyzer.load_data()

    def test_handle_missing_rejects_an_unsupported_strategy(self):
        """Only supported numeric aggregation strategies are accepted."""
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        analyzer.load_data()

        with self.assertRaises(ValueError):
            analyzer.handle_missing("age", "club_member_status", "mode")


class DataAnalyzerImageFeatureTests(unittest.TestCase):
    """Vectorized, product-level image features."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.raw_dir = Path(self.temporary_directory.name) / "raw"
        self.image_dir = self.raw_dir / "images" / "001"
        self.image_dir.mkdir(parents=True)
        self.csv_path = Path(self.temporary_directory.name) / "cohort.csv"
        self.first_image = self.image_dir / "0010000001.jpg"
        self.second_image = self.image_dir / "0010000002.jpg"
        mpimg.imsave(self.first_image, np.full((4, 4, 3), 0.25, dtype=np.float32))
        mpimg.imsave(self.second_image, np.full((4, 4, 3), 0.75, dtype=np.float32))
        pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "customer_id": ["c1", "c2", "c3"],
                "product_id": ["0010000001", "0010000001", "0010000002"],
                "product_name": ["Top", "Top", "Dress"],
                "image_path": [
                    "images/001/0010000001.jpg",
                    "images/001/0010000001.jpg",
                    "images/001/0010000002.jpg",
                ],
                "unit_price": [10.0, 20.0, 30.0],
                "age": [20.0, 30.0, 40.0],
            }
        ).to_csv(self.csv_path, index=False)

    def _loaded_analyzer(self):
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        analyzer.load_data()
        return analyzer

    def test_engineer_features_reuses_product_features_and_measures_image_statistics(self):
        """A product's decoded RGB pixels produce identical per-row features."""
        analyzer = self._loaded_analyzer()

        result = analyzer.engineer_features(downsample_step=2)
        expected = mpimg.imread(self.first_image)[::2, ::2, :3]
        product_rows = result.loc[result["product_id"] == "0010000001"]

        self.assertAlmostEqual(product_rows["image_mean"].iloc[0], float(expected.mean()))
        self.assertAlmostEqual(product_rows["image_std"].iloc[0], float(expected.std()))
        self.assertEqual(result.loc[0, "product_name_length"], len(result.loc[0, "product_name"]))
        self.assertEqual(product_rows["image_mean"].nunique(), 1)

    def test_engineer_features_stacks_compact_arrays_that_do_not_retain_full_image_memory(self):
        """Downsampled arrays handed to stack must own their compact pixel memory."""
        large_first = np.full((120, 120, 3), 0.25, dtype=np.float32)
        large_second = np.full((120, 120, 3), 0.75, dtype=np.float32)
        mpimg.imsave(self.first_image, large_first)
        mpimg.imsave(self.second_image, large_second)
        analyzer = self._loaded_analyzer()
        received_images = []
        original_stack = np.stack

        def capture_images(images, *arguments, **keywords):
            received_images.extend(images)
            return original_stack(images, *arguments, **keywords)

        with patch("src.pipeline.np.stack", side_effect=capture_images):
            analyzer.engineer_features(downsample_step=10)

        self.assertEqual([image.shape for image in received_images], [(12, 12, 3)] * 2)
        self.assertTrue(all(image.flags["C_CONTIGUOUS"] for image in received_images))
        self.assertTrue(all(image.base is None for image in received_images))

    def test_engineer_features_rejects_a_missing_image(self):
        """Every image reference must resolve under the configured root."""
        analyzer = self._loaded_analyzer()
        analyzer.data.loc[2, "image_path"] = "images/001/missing.jpg"

        with self.assertRaises(FileNotFoundError):
            analyzer.engineer_features()

    def test_engineer_features_rejects_a_nonpositive_downsample_step(self):
        """A zero or negative slice step is invalid."""
        analyzer = self._loaded_analyzer()

        with self.assertRaises(ValueError):
            analyzer.engineer_features(downsample_step=0)

    def test_engineer_features_rejects_inconsistent_downsampled_shapes(self):
        """All product image tensors must be stackable for vectorized statistics."""
        mpimg.imsave(self.second_image, np.full((5, 4, 3), 0.75, dtype=np.float32))
        analyzer = self._loaded_analyzer()

        with self.assertRaisesRegex(ValueError, "same shape"):
            analyzer.engineer_features(downsample_step=1)

    def test_engineer_features_rejects_a_product_with_multiple_paths(self):
        """A product identity must map to one canonical image."""
        alternate = self.image_dir / "alternate.jpg"
        mpimg.imsave(alternate, np.full((4, 4, 3), 0.5, dtype=np.float32))
        analyzer = self._loaded_analyzer()
        analyzer.data.loc[1, "image_path"] = "images/001/alternate.jpg"

        with self.assertRaises(ValueError):
            analyzer.engineer_features()


class DataAnalyzerRfmTests(unittest.TestCase):
    """IQR outlier detection and customer-level RFM segmentation."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.raw_dir = Path(self.temporary_directory.name) / "raw"
        self.raw_dir.mkdir()
        self.csv_path = Path(self.temporary_directory.name) / "transactions.csv"

        latest_dates = pd.date_range("2024-01-02", periods=8, freq="D")[::-1]
        unique_frequencies = [8, 1, 2, 3, 4, 5, 6, 7]
        total_spend = [800.0, 20.0, 40.0, 90.0, 160.0, 250.0, 360.0, 1_000.0]
        rows = []
        for number, (last_date, frequency, spend) in enumerate(
            zip(latest_dates, unique_frequencies, total_spend), start=1
        ):
            for purchase_number in range(frequency):
                rows.append(
                    {
                        "order_date": (
                            pd.Timestamp(last_date) - pd.Timedelta(int(purchase_number), unit="D")
                        ).strftime("%Y-%m-%d"),
                        "customer_id": "c{}".format(number),
                        "unit_price": spend / frequency,
                    }
                )
        rows.append(
            {
                "order_date": "2024-01-08",
                "customer_id": "c2",
                "unit_price": 10.0,
            }
        )
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    def _loaded_analyzer(self):
        analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
        analyzer.load_data(date_columns=("order_date",), numeric_columns=("unit_price",))
        return analyzer

    def test_detect_outliers_returns_only_values_outside_iqr_fences(self):
        """Detected records must fall strictly beyond the computed fences."""
        analyzer = self._loaded_analyzer()
        analyzer.data.loc[len(analyzer.data), "unit_price"] = 10_000.0

        outliers, lower, upper = analyzer.detect_outliers("unit_price", threshold=1.5)

        self.assertGreater(len(outliers), 0)
        self.assertTrue(
            ((outliers["unit_price"] < lower) | (outliers["unit_price"] > upper)).all()
        )

    def test_detect_outliers_rejects_invalid_thresholds_and_nonnumeric_columns(self):
        """IQR is defined only for numeric data and a positive multiplier."""
        analyzer = self._loaded_analyzer()

        with self.assertRaises(ValueError):
            analyzer.detect_outliers("unit_price", threshold=0)
        with self.assertRaises(TypeError):
            analyzer.detect_outliers("customer_id")

    def test_calculate_rfm_scores_and_segments_customers(self):
        """Ranked recency, frequency, and monetary values yield customer segments."""
        analyzer = self._loaded_analyzer()

        rfm = analyzer.calculate_rfm(frequency_mode="unique_dates")

        self.assertTrue(
            {"recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment"}.issubset(
                rfm.columns
            )
        )
        self.assertEqual(rfm.index.name, "customer_id")
        self.assertEqual(rfm["recency"].min(), 1)
        self.assertGreaterEqual(rfm["segment"].nunique(), 4)
        self.assertEqual(rfm.loc["c1", "segment"], "VIP")

    def test_calculate_rfm_scores_a_two_customer_cohort(self):
        """A small valid cohort still receives complete RFM scores."""
        small_path = Path(self.temporary_directory.name) / "two-customers.csv"
        pd.DataFrame(
            {
                "order_date": ["2024-01-09", "2024-01-08"],
                "customer_id": ["recent", "older"],
                "unit_price": [20.0, 10.0],
            }
        ).to_csv(small_path, index=False)
        analyzer = DataAnalyzer(small_path, image_root=self.raw_dir)
        analyzer.load_data(date_columns=("order_date",), numeric_columns=("unit_price",))

        rfm = analyzer.calculate_rfm()

        self.assertEqual(list(rfm.index), ["older", "recent"])
        self.assertFalse(rfm[["r_score", "f_score", "m_score"]].isna().any().any())
        self.assertTrue(rfm[["r_score", "f_score", "m_score"]].apply(pd.api.types.is_integer_dtype).all())

    def test_calculate_rfm_scores_a_one_customer_cohort_without_qcut_edge_failure(self):
        """A lone customer receives the highest complete score rather than a qcut error."""
        small_path = Path(self.temporary_directory.name) / "one-customer.csv"
        pd.DataFrame(
            {
                "order_date": ["2024-01-09"],
                "customer_id": ["only"],
                "unit_price": [20.0],
            }
        ).to_csv(small_path, index=False)
        analyzer = DataAnalyzer(small_path, image_root=self.raw_dir)
        analyzer.load_data(date_columns=("order_date",), numeric_columns=("unit_price",))

        rfm = analyzer.calculate_rfm()

        self.assertEqual(rfm.loc["only", ["r_score", "f_score", "m_score"]].tolist(), [4, 4, 4])
        self.assertEqual(rfm.loc["only", "segment"], "VIP")

    def test_calculate_rfm_distinguishes_unique_dates_from_transaction_rows(self):
        """Two items bought on one date are one visit but two transaction rows."""
        analyzer = self._loaded_analyzer()

        by_date = analyzer.calculate_rfm(frequency_mode="unique_dates")
        by_row = analyzer.calculate_rfm(frequency_mode="rows")

        self.assertEqual(by_date.loc["c2", "frequency"], 1)
        self.assertEqual(by_row.loc["c2", "frequency"], 2)

    def test_calculate_rfm_rejects_an_unknown_frequency_mode(self):
        """Frequency must have an explicit aggregation definition."""
        analyzer = self._loaded_analyzer()

        with self.assertRaises(ValueError):
            analyzer.calculate_rfm(frequency_mode="orders")
