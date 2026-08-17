import os
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hm-rebuild-mpl"))

import matplotlib.image
import numpy as np
import pandas as pd

from scripts.prepare_hm_data import prepare_cohort, stable_customer_ids


class DataPreparationTest(unittest.TestCase):
    def _write_hm_shaped_source(self):
        pd.DataFrame(
            [
                ["2020-01-03", "customer-1", "0100000001", 0.10, 1],
                ["2020-01-01", "customer-2", "0100000002", 0.20, 2],
                ["2020-01-02", "customer-3", "0100000003", 0.30, 1],
                ["2020-01-04", "customer-4", "0100000004", 0.40, 2],
                ["2020-01-05", "customer-1", "0100000001", 0.10, 1],
            ],
            columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        ).to_csv(self.raw_dir / "transactions_train.csv", index=False)
        pd.DataFrame(
            [
                ["0100000001", "Top one", "Garment upper body"],
                ["0100000002", "Top two", "Garment upper body"],
                ["0100000003", "Top three", "Garment upper body"],
                ["0100000004", "Missing image", "Garment upper body"],
            ],
            columns=["article_id", "prod_name", "product_group_name"],
        ).to_csv(self.raw_dir / "articles.csv", index=False)
        pd.DataFrame(
            [
                ["customer-1", 20, "ACTIVE", "Regularly"],
                ["customer-2", None, "PRE-CREATE", "None"],
                ["customer-3", 40, "ACTIVE", "Monthly"],
                ["customer-4", 30, "ACTIVE", "Regularly"],
            ],
            columns=[
                "customer_id",
                "age",
                "club_member_status",
                "fashion_news_frequency",
            ],
        ).to_csv(self.raw_dir / "customers.csv", index=False)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for article_id in ("0100000001", "0100000002", "0100000003"):
            matplotlib.image.imsave(
                self.raw_dir / "images" / "010" / f"{article_id}.jpg", image
            )

    def test_stable_customer_ids_is_order_independent(self):
        ids = ["customer-c", "customer-a", "customer-d", "customer-b"]

        first = stable_customer_ids(ids, cohort_size=3, seed=42)
        second = stable_customer_ids(reversed(ids), cohort_size=3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_stable_customer_ids_rejects_oversized_cohort(self):
        with self.assertRaisesRegex(ValueError, "active customers"):
            stable_customer_ids(["one"], cohort_size=2, seed=42)

    def test_prepare_cohort_keeps_only_rows_with_joined_metadata_and_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.raw_dir = Path(temporary_directory)
            self.output_path = self.raw_dir / "hm_customer_cohort.csv"
            (self.raw_dir / "images" / "010").mkdir(parents=True)
            self._write_hm_shaped_source()

            summary = prepare_cohort(
                self.raw_dir,
                self.output_path,
                cohort_size=4,
                seed=7,
                chunksize=3,
                minimum_rows=4,
            )
            result = pd.read_csv(self.output_path, dtype={"product_id": "string"})
            summary_path_exists = self.output_path.with_suffix(".summary.json").exists()

        self.assertEqual(summary["selected_customers"], 4)
        self.assertEqual(summary["missing_image_rows"], 1)
        self.assertTrue(
            {
                "order_date",
                "customer_id",
                "product_id",
                "product_name",
                "category",
                "unit_price",
                "sales_channel_id",
                "age",
                "club_member_status",
                "fashion_news_frequency",
                "image_path",
            }.issubset(result.columns)
        )
        self.assertTrue(
            result["image_path"].str.match(r"images/\d{3}/\d{10}\.jpg").all()
        )
        self.assertTrue({"quantity", "discount_rate", "order_id"}.isdisjoint(result.columns))
        self.assertEqual(len(result), 4)
        self.assertTrue(result["age"].isna().any())
        self.assertEqual(result["product_id"].tolist(), ["0100000002", "0100000003", "0100000001", "0100000001"])
        self.assertTrue(summary_path_exists)

    def test_prepare_cohort_returns_the_written_output_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.raw_dir = Path(temporary_directory)
            self.output_path = self.raw_dir / "cohort.csv"
            (self.raw_dir / "images" / "010").mkdir(parents=True)
            self._write_hm_shaped_source()

            summary = prepare_cohort(
                self.raw_dir,
                self.output_path,
                cohort_size=4,
                seed=7,
                chunksize=3,
                minimum_rows=4,
            )

            self.assertEqual(summary["output_rows"], len(pd.read_csv(self.output_path)))
            self.assertEqual(summary["output_columns"], 11)
            self.assertTrue(self.output_path.with_suffix(".summary.json").is_file())

    def test_command_line_writes_the_requested_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.raw_dir = Path(temporary_directory)
            self.output_path = self.raw_dir / "requested.csv"
            (self.raw_dir / "images" / "010").mkdir(parents=True)
            self._write_hm_shaped_source()

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_hm_data.py",
                    "--raw-dir",
                    str(self.raw_dir),
                    "--output",
                    str(self.output_path),
                    "--cohort-size",
                    "4",
                    "--minimum-rows",
                    "4",
                ],
                capture_output=True,
                cwd=Path(__file__).parents[1],
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(self.output_path.exists())
