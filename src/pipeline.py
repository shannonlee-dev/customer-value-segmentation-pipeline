"""Portable, full-data H&M analysis implemented with Pandas, NumPy, and Matplotlib."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._pipeline.artifacts import ArtifactStore
from ._pipeline.features import ProductFeatureEngineer
from ._pipeline.loading import DataLoader
from ._pipeline.rfm import RFMEngine
from ._pipeline.contracts import (
    ARTICLE_NORMALIZED_REQUIRED_COLUMNS,
    ARTICLE_RENAMES,
    ARTICLES_CACHE_FILENAME,
    CSV_APPEND_MODE,
    CSV_WRITE_MODE,
    CUSTOMER_AGE_COLUMN,
    CUSTOMER_ID_COLUMN,
    CUSTOMER_MEMBERSHIP_COLUMN,
    CUSTOMER_NORMALIZED_REQUIRED_COLUMNS,
    CUSTOMERS_CACHE_FILENAME,
    DEFAULT_CHUNKSIZE,
    DEFAULT_RFM_AMOUNT_COLUMN,
    DEFAULT_RFM_CUSTOMER_COLUMN,
    DEFAULT_RFM_DATE_COLUMN,
    DEFAULT_RFM_PARTITION_COUNT,
    IMAGE_MEAN_COLUMN,
    ORDER_DATE_COLUMN,
    PRODUCT_FEATURES_CACHE_FILENAME,
    PRODUCT_ID_COLUMN,
    PRODUCT_NAME_LENGTH_COLUMN,
    RAW_ARTICLE_ID_COLUMN,
    RAW_ARTICLE_REQUIRED_COLUMNS,
    RAW_ARTICLES_FILENAME,
    RAW_CUSTOMER_REQUIRED_COLUMNS,
    RAW_CUSTOMERS_FILENAME,
    RAW_DATE_COLUMN,
    RAW_PRODUCT_NAME_COLUMN,
    RAW_PRICE_COLUMN,
    RAW_TRANSACTION_DTYPES,
    RAW_TRANSACTION_REQUIRED_COLUMNS,
    RAW_TRANSACTIONS_FILENAME,
    RFM_FREQUENCY_COLUMN,
    RFM_MONETARY_COLUMN,
    RFM_OUTPUT_FILENAME,
    RFM_SCORE_QUANTILE_COUNT,
    STRING_DTYPE,
    STRICT_PARSING_ERRORS,
    TRANSACTION_NORMALIZED_REQUIRED_COLUMNS,
    TRANSACTION_RENAMES,
    TRANSACTIONS_CACHE_FILENAME,
    UNIT_PRICE_COLUMN,
)
from .reporting import summarize_numeric
from .runtime import RuntimeContext


MEMMAP_WRITE_MODE = "w+"
IQR_OUTPUT_FILENAME_TEMPLATE = "iqr_{column}.json"
EDA_SUMMARY_FILENAME = "eda_summary.json"
MONTHLY_SUMMARY_FILENAME = "monthly_summary.csv"
EDA_HISTOGRAM_BIN_COUNT = 40
# Data-quality and product-feature policies
DEFAULT_MISSING_VALUE_STRATEGY = "median"
SUPPORTED_MISSING_VALUE_STRATEGIES = ("median", "mean")
IMAGE_DIRECTORY = "images"
# IQR policies
DEFAULT_OUTLIER_COLUMN = UNIT_PRICE_COLUMN
DEFAULT_IQR_THRESHOLD = 1.5
IQR_QUANTILES = (0.25, 0.75)
IQR_STATISTIC_KEYS = ("q1", "q3", "lower_fence", "upper_fence", "outlier_count")

def _calculate_iqr_statistics(values: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Calculate IQR fences and an exact outlier count for an in-memory numeric array."""
    q1, q3 = np.quantile(values, IQR_QUANTILES)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return {
        "q1": float(q1),
        "q3": float(q3),
        "lower_fence": float(lower),
        "upper_fence": float(upper),
        "outlier_count": int(np.count_nonzero((values < lower) | (values > upper))),
    }


class DataAnalyzer:
    """Public facade for loading, cleaning, feature engineering, IQR, and RFM."""

    def __init__(self, context: RuntimeContext, chunksize: int = DEFAULT_CHUNKSIZE):
        self.context = context
        self.chunksize = chunksize
        self.runtime_transactions_path = context.processed_root / TRANSACTIONS_CACHE_FILENAME
        self.runtime_customers_path = context.processed_root / CUSTOMERS_CACHE_FILENAME
        self.runtime_articles_path = context.processed_root / ARTICLES_CACHE_FILENAME
        self.runtime_product_features_path = context.feature_root / PRODUCT_FEATURES_CACHE_FILENAME
        self.runtime_rfm_path = context.aggregate_root / RFM_OUTPUT_FILENAME
        self.transactions_path = self.runtime_transactions_path
        self.customers_path = self.runtime_customers_path
        self.articles_path = self.runtime_articles_path
        self.product_features_path = self.runtime_product_features_path
        self.rfm_path = self.runtime_rfm_path
        self.customers: pd.DataFrame | None = None
        self.articles: pd.DataFrame | None = None
        self.artifact_status: dict[str, str] = {}
        self.cache_messages: list[str] = []
        self._artifacts = ArtifactStore(
            context,
            self.artifact_status,
            self.cache_messages,
        )
        self._loader = DataLoader(context, self._artifacts, chunksize)
        self._feature_engineer = ProductFeatureEngineer(context, self._artifacts)
        self._rfm_engine = RFMEngine(context, self._artifacts, chunksize)

    def load_data(self, *, force: bool = False) -> dict[str, int]:
        """Load and normalize full source data without performing imputation."""
        self.customers, self.customers_path = self._loader.load_customers(
            force=force
        )
        self.articles, self.articles_path = self._loader.load_articles(
            force=force
        )
        self.transactions_path, transaction_rows = self._loader.load_transactions(
            self.customers,
            self.articles,
            force=force,
        )
        return {
            "transaction_rows": transaction_rows,
            "customer_rows": len(self.customers),
            "product_rows": len(self.articles),
        }

    def handle_missing_values(
        self,
        column: str,
        group_column: str,
        strategy: str = DEFAULT_MISSING_VALUE_STRATEGY,
    ) -> dict[str, int]:
        """Impute a customer-level numeric attribute using its membership-group statistic."""
        if self.customers is None:
            raise ValueError("Call load_data before handle_missing_values")
        if strategy not in SUPPORTED_MISSING_VALUE_STRATEGIES:
            raise ValueError("strategy must be 'median' or 'mean'")
        missing_before = int(self.customers[column].isna().sum())
        grouped = self.customers.groupby(group_column, dropna=False)[column].transform(strategy)
        fallback = getattr(self.customers[column], strategy)()
        filled = self.customers[column].fillna(grouped).fillna(fallback)
        missing_after = int(filled.isna().sum())
        if missing_after:
            raise ValueError(
                f"could not impute all missing values in {column}: {missing_after} remain"
            )

        self.customers[column] = filled
        self.customers_path = self.runtime_customers_path
        self.customers.loc[:, list(CUSTOMER_NORMALIZED_REQUIRED_COLUMNS)].to_csv(
            self.customers_path,
            index=False,
        )
        self._artifacts.record_status("customers", "COMPUTED", self.customers_path)
        return {
            "missing_before": missing_before,
            "missing_after": missing_after,
        }

    def engineer_features(self, *, force: bool = False) -> pd.DataFrame:
        """Reuse cached product features or calculate them once."""
        reusable = self._feature_engineer.find_reusable(force=force)
        if reusable is not None:
            features, self.product_features_path = reusable
            return features
        if self.articles is None:
            self.articles = pd.read_csv(
                self.articles_path,
                dtype={PRODUCT_ID_COLUMN: STRING_DTYPE},
            )
        features, self.product_features_path = self._feature_engineer.build(
            self.articles,
            force=True,
        )
        return features

    def detect_outliers(
        self,
        column: str = DEFAULT_OUTLIER_COLUMN,
        threshold: float = DEFAULT_IQR_THRESHOLD,
        *,
        force: bool = False,
    ) -> dict[str, float | int]:
        """Calculate exact full-data IQR fences and counts with a disk-backed NumPy array."""
        if not force:
            cached = self._load_matching_iqr_cache(column, threshold)
            if cached is not None:
                return cached
        values_path, row_count = self._materialize_numeric_column(column)
        values = np.memmap(values_path, dtype="float64", mode="r", shape=(row_count,))
        statistics = _calculate_iqr_statistics(values, threshold)
        del values
        return self._save_iqr_result(column, threshold, statistics)

    def calculate_rfm(
        self,
        customer_col: str = DEFAULT_RFM_CUSTOMER_COLUMN,
        date_col: str = DEFAULT_RFM_DATE_COLUMN,
        amount_col: str = DEFAULT_RFM_AMOUNT_COLUMN,
        analysis_date: str | pd.Timestamp | None = None,
        partition_count: int = DEFAULT_RFM_PARTITION_COUNT,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Compute every-customer RFM through hash-partitioned CSV aggregation."""
        rfm, self.rfm_path = self._rfm_engine.calculate(
            self.transactions_path,
            customer_col,
            date_col,
            amount_col,
            analysis_date,
            partition_count,
            force=force,
        )
        return rfm

    def prepare_eda_artifacts(self, *, force: bool = False) -> dict[str, object]:
        """Persist exact full-data chart aggregates so report reuse never rescans transactions."""
        runtime_summary = self.context.aggregate_root / EDA_SUMMARY_FILENAME
        runtime_monthly = self.context.aggregate_root / MONTHLY_SUMMARY_FILENAME
        if not force:
            summary_path = self._artifacts.find_reusable_json(
                "EDA",
                runtime_summary,
                EDA_SUMMARY_FILENAME,
            )
            monthly_path = self._artifacts.find_reusable_csv(
                "monthly EDA", runtime_monthly, MONTHLY_SUMMARY_FILENAME, ("order_month", RFM_MONETARY_COLUMN)
            )
            if summary_path is not None and monthly_path is not None:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                required = {"price_statistics", "histogram_edges", "histogram_counts", "boxplot_before", "boxplot_after", "price_age_correlation", "image_text_correlation"}
                if required.issubset(summary):
                    summary["monthly_summary_path"] = str(monthly_path)
                    self._artifacts.record_status("EDA", "REUSED", summary_path)
                    return summary
                self._artifacts.record_rejection(
                    "EDA",
                    summary_path,
                    "missing required summary fields",
                )

        if self.customers is None:
            raise ValueError("Call load_data before prepare_eda_artifacts")
        if self.customers[CUSTOMER_AGE_COLUMN].isna().any():
            raise ValueError(
                "Call handle_missing_values before prepare_eda_artifacts; customer age still contains missing values"
            )

        values_path, row_count = self._materialize_numeric_column(DEFAULT_OUTLIER_COLUMN)
        values = np.memmap(values_path, dtype="float64", mode="r", shape=(row_count,))
        price_statistics = summarize_numeric(values)
        q1, q3 = price_statistics["q1"], price_statistics["q3"]
        lower, upper = q1 - DEFAULT_IQR_THRESHOLD * (q3 - q1), q3 + DEFAULT_IQR_THRESHOLD * (q3 - q1)
        inliers = values[(values >= lower) & (values <= upper)]

        customers = pd.read_csv(self.customers_path, dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE})[[CUSTOMER_ID_COLUMN, CUSTOMER_AGE_COLUMN]]
        count = sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
        monthly_totals: dict[str, float] = {}
        for chunk in pd.read_csv(
            self.transactions_path,
            usecols=[CUSTOMER_ID_COLUMN, ORDER_DATE_COLUMN, UNIT_PRICE_COLUMN],
            dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE},
            parse_dates=[ORDER_DATE_COLUMN],
            chunksize=self.chunksize,
        ):
            paired = chunk[[CUSTOMER_ID_COLUMN, UNIT_PRICE_COLUMN]].merge(customers, on=CUSTOMER_ID_COLUMN, how="left").dropna()
            x = paired[UNIT_PRICE_COLUMN].to_numpy(dtype=float)
            y = paired[CUSTOMER_AGE_COLUMN].to_numpy(dtype=float)
            count += len(x); sum_x += x.sum(); sum_y += y.sum(); sum_xy += (x * y).sum(); sum_x2 += (x * x).sum(); sum_y2 += (y * y).sum()
            grouped = chunk.groupby(chunk[ORDER_DATE_COLUMN].dt.to_period("M"))[UNIT_PRICE_COLUMN].sum()
            for month, total in grouped.items():
                key = str(month)
                monthly_totals[key] = monthly_totals.get(key, 0.0) + float(total)
        denominator = np.sqrt((count * sum_x2 - sum_x ** 2) * (count * sum_y2 - sum_y ** 2))
        price_age_correlation = 0.0 if denominator == 0 else float((count * sum_xy - sum_x * sum_y) / denominator)

        product_features = self.engineer_features(force=force)
        image_text_correlation = float(product_features[[IMAGE_MEAN_COLUMN, PRODUCT_NAME_LENGTH_COLUMN]].corr().iloc[0, 1])
        histogram_counts, histogram_edges = np.histogram(values, bins=EDA_HISTOGRAM_BIN_COUNT)
        summary = {
            "price_statistics": price_statistics,
            "histogram_counts": histogram_counts.astype(int).tolist(),
            "histogram_edges": histogram_edges.astype(float).tolist(),
            "boxplot_before": self._boxplot_statistics(values, "Before IQR"),
            "boxplot_after": self._boxplot_statistics(inliers, "After IQR"),
            "price_age_correlation": price_age_correlation,
            "image_text_correlation": image_text_correlation,
        }
        monthly = pd.DataFrame(sorted(monthly_totals.items()), columns=["order_month", RFM_MONETARY_COLUMN])
        monthly.to_csv(runtime_monthly, index=False)
        runtime_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del values
        self._artifacts.record_status("monthly EDA", "COMPUTED", runtime_monthly)
        self._artifacts.record_status("EDA", "COMPUTED", runtime_summary)
        summary["monthly_summary_path"] = str(runtime_monthly)
        return summary

    def format_cache_report(self) -> str:
        """Return evaluator-facing full-data artifact reuse status."""
        lines = [f"Runtime mode: {'PRECOMPUTED FULL-DATA ARTIFACTS' if self.context.precomputed_root else self.context.runtime_name.upper()}"]
        if self.context.precomputed_root is not None:
            lines.append(f"Precomputed root: {self.context.precomputed_root}")
        for artifact in ("transactions", "customers", "articles", "product features", "IQR", "RFM"):
            lines.append(f"{artifact}: {self.artifact_status.get(artifact, 'PENDING')}")
        return "\n".join(lines)

    def _require_raw_data_root(self) -> Path:
        if self.context.raw_data_root is None:
            raise ValueError(
                "A required full-data artifact was unavailable or invalid and no raw H&M dataset is attached. "
                "Attach the H&M competition input or set HM_RAW_DATA_DIR to rebuild it."
            )
        return self.context.raw_data_root

    def _materialize_numeric_column(self, column: str) -> tuple[Path, int]:
        """Return a valid disk-backed numeric transaction column, building it only when needed."""
        row_count = self._artifacts.csv_row_count(self.transactions_path)
        values_path = self.context.aggregate_root / f"{column}_values{".dat"}"
        expected_size = row_count * np.dtype("float64").itemsize
        if values_path.is_file() and values_path.stat().st_size == expected_size:
            return values_path, row_count

        values = np.memmap(values_path, dtype="float64", mode=MEMMAP_WRITE_MODE, shape=(row_count,))
        offset = 0
        for chunk in pd.read_csv(self.transactions_path, usecols=[column], chunksize=self.chunksize):
            numeric = pd.to_numeric(chunk[column], errors=STRICT_PARSING_ERRORS).to_numpy(dtype="float64")
            values[offset : offset + len(numeric)] = numeric
            offset += len(numeric)
        values.flush()
        del values
        return values_path, row_count

    def _load_matching_iqr_cache(self, column: str, threshold: float) -> dict[str, float | int] | None:
        iqr_filename = IQR_OUTPUT_FILENAME_TEMPLATE.format(column=column)
        cached = self._artifacts.find_reusable_json(
            "IQR",
            self.context.aggregate_root / iqr_filename,
            iqr_filename,
        )
        if cached is None:
            return None
        result = json.loads(cached.read_text(encoding="utf-8"))
        if result.get("column") == column and result.get("threshold") == threshold:
            return {key: result[key] for key in IQR_STATISTIC_KEYS}
        self._artifacts.record_rejection("IQR", cached, "parameters do not match")
        return None

    def _save_iqr_result(
        self,
        column: str,
        threshold: float,
        statistics: dict[str, float | int],
    ) -> dict[str, float | int]:
        result = {"column": column, "threshold": threshold, **statistics}
        output_path = self.context.aggregate_root / IQR_OUTPUT_FILENAME_TEMPLATE.format(column=column)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._artifacts.record_status("IQR", "COMPUTED", output_path)
        return {key: result[key] for key in IQR_STATISTIC_KEYS}

    @staticmethod
    def _boxplot_statistics(values: np.ndarray, label: str) -> dict[str, float | str]:
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        inliers = values[(values >= q1 - DEFAULT_IQR_THRESHOLD * iqr) & (values <= q3 + DEFAULT_IQR_THRESHOLD * iqr)]
        return {
            "label": label, "q1": float(q1), "med": float(median), "q3": float(q3),
            "whislo": float(np.min(inliers)), "whishi": float(np.max(inliers)),
        }
