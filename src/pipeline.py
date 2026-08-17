"""Multimodal preprocessing helpers for the prepared H&M cohort."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import image as mpimg


DEFAULT_DATE_COLUMNS = ("order_date",)
DEFAULT_NUMERIC_COLUMNS = ("unit_price", "age")
DATA_COLUMN_DTYPES = {"customer_id": "string", "product_id": "string"}
DEFAULT_IMPUTATION_STRATEGY = "median"
SUPPORTED_IMPUTATION_STRATEGIES = ("median", "mean")
DEFAULT_IMAGE_COLUMN = "image_path"
DEFAULT_PRODUCT_COLUMN = "product_id"
DEFAULT_TEXT_COLUMN = "product_name"
DEFAULT_IMAGE_DOWNSAMPLE_STEP = 35
RGB_CHANNEL_COUNT = 3
DEFAULT_IQR_THRESHOLD = 1.5
LOWER_QUARTILE = 0.25
UPPER_QUARTILE = 0.75
DEFAULT_CUSTOMER_COLUMN = "customer_id"
DEFAULT_DATE_COLUMN = "order_date"
DEFAULT_AMOUNT_COLUMN = "unit_price"
DEFAULT_FREQUENCY_MODE = "unique_dates"
SUPPORTED_FREQUENCY_MODES = ("unique_dates", "rows")
RFM_REFERENCE_DAY_OFFSET = 1
RFM_QUANTILE_COUNT = 4
RFM_SCORE_LABELS = (1, 2, 3, 4)
RFM_HIGH_SCORE = 3
RFM_BEST_SCORE = 4
RFM_LOW_SCORE = 2


class DataAnalyzer:
    """Load and progressively enrich a prepared customer cohort."""

    def __init__(self, data_path: Union[str, Path], image_root: Union[str, Path]) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.data = None

    def load_data(
        self,
        date_columns: Sequence[str] = DEFAULT_DATE_COLUMNS,
        numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    ) -> pd.DataFrame:
        """Load cohort data while enforcing its date and numeric types."""
        if not self.data_path.is_file():
            raise FileNotFoundError("Data file does not exist: {}".format(self.data_path))

        frame = pd.read_csv(
            self.data_path,
            dtype=DATA_COLUMN_DTYPES,
        )
        if frame.empty:
            raise ValueError("Data file must contain at least one row")
        self._require_columns(frame, tuple(date_columns) + tuple(numeric_columns))

        for column in date_columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

        self.data = frame
        return self.data

    def handle_missing_values(
        self, column: str, group_col: str, strategy: str = DEFAULT_IMPUTATION_STRATEGY
    ) -> pd.DataFrame:
        """Fill numeric gaps from each group, then the global aggregate."""
        frame = self._loaded_data()
        self._require_columns(frame, (column, group_col))
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError("Column '{}' must be numeric".format(column))
        if strategy not in SUPPORTED_IMPUTATION_STRATEGIES:
            raise ValueError("Unsupported imputation strategy: {}".format(strategy))

        group_values = frame.groupby(group_col, dropna=False)[column].transform(strategy)
        global_value = getattr(frame[column], strategy)()
        if pd.isna(global_value):
            raise ValueError("Column '{}' has no values available for imputation".format(column))
        frame[column] = frame[column].fillna(group_values).fillna(global_value)
        return frame

    def engineer_features(
        self,
        image_col: str = DEFAULT_IMAGE_COLUMN,
        product_col: str = DEFAULT_PRODUCT_COLUMN,
        text_col: str = DEFAULT_TEXT_COLUMN,
        downsample_step: int = DEFAULT_IMAGE_DOWNSAMPLE_STEP,
    ) -> pd.DataFrame:
        """Add product-level RGB statistics and product-name lengths."""
        frame = self._loaded_data()
        self._require_columns(frame, (image_col, product_col, text_col))
        if downsample_step <= 0:
            raise ValueError("downsample_step must be positive")

        unique_images = frame[[product_col, image_col]].drop_duplicates().copy()
        if unique_images[product_col].duplicated().any():
            raise ValueError("Each product must map to exactly one image path")

        loaded = [
            mpimg.imread(self.image_root / relative_path)[
                ::downsample_step, ::downsample_step, :RGB_CHANNEL_COUNT
            ].copy(order="C")
            for relative_path in unique_images[image_col]
        ]
        try:
            image_tensor = np.stack(loaded).astype(np.float32, copy=False)
        except ValueError as error:
            raise ValueError("All downsampled image arrays must have the same shape") from error

        feature_axes = tuple(range(1, image_tensor.ndim))
        unique_images["image_mean"] = image_tensor.mean(axis=feature_axes)
        unique_images["image_std"] = image_tensor.std(axis=feature_axes)
        result = frame.merge(
            unique_images,
            on=[product_col, image_col],
            how="left",
            validate="many_to_one",
        )
        result["product_name_length"] = result[text_col].astype("string").str.len()
        self.data = result
        return self.data

    def detect_outliers(
        self, column: str, threshold: float = DEFAULT_IQR_THRESHOLD
    ) -> Tuple[pd.DataFrame, float, float]:
        """Return numeric records outside the column's IQR fences."""
        frame = self._loaded_data()
        self._require_columns(frame, (column,))
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError("Column '{}' must be numeric".format(column))

        values = frame[column].dropna()
        if values.empty:
            raise ValueError("Column '{}' has no numeric values".format(column))
        lower_quartile = values.quantile(LOWER_QUARTILE)
        upper_quartile = values.quantile(UPPER_QUARTILE)
        interquartile_range = upper_quartile - lower_quartile
        lower_fence = lower_quartile - threshold * interquartile_range
        upper_fence = upper_quartile + threshold * interquartile_range
        outliers = frame.loc[
            (frame[column] < lower_fence) | (frame[column] > upper_fence)
        ].copy()
        return outliers, float(lower_fence), float(upper_fence)

    def calculate_rfm(
        self,
        customer_col: str = DEFAULT_CUSTOMER_COLUMN,
        date_col: str = DEFAULT_DATE_COLUMN,
        amount_col: str = DEFAULT_AMOUNT_COLUMN,
        frequency_mode: str = DEFAULT_FREQUENCY_MODE,
        analysis_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Score customers by recency, frequency, monetary value, and segment."""
        frame = self._loaded_data()
        self._require_columns(frame, (customer_col, date_col, amount_col))
        if frequency_mode not in SUPPORTED_FREQUENCY_MODES:
            raise ValueError("frequency_mode must be 'unique_dates' or 'rows'")
        if not pd.api.types.is_numeric_dtype(frame[amount_col]):
            raise TypeError("Column '{}' must be numeric".format(amount_col))

        working = frame[[customer_col, date_col, amount_col]].copy()
        working[date_col] = pd.to_datetime(working[date_col], errors="raise")
        grouped = working.groupby(customer_col)
        reference_date = (
            pd.Timestamp(analysis_date)
            if analysis_date is not None
            else working[date_col].max() + pd.Timedelta(RFM_REFERENCE_DAY_OFFSET, unit="D")
        )
        rfm = pd.DataFrame(
            {
                "recency": (reference_date - grouped[date_col].max()).dt.days,
                "frequency": (
                    grouped[date_col].nunique()
                    if frequency_mode == "unique_dates"
                    else grouped.size()
                ),
                "monetary": grouped[amount_col].sum(min_count=1),
            }
        )
        if len(rfm) == 1:
            rfm["r_score"] = RFM_BEST_SCORE
            rfm["f_score"] = RFM_BEST_SCORE
            rfm["m_score"] = RFM_BEST_SCORE
        else:
            rfm["r_score"] = pd.qcut(
                rfm["recency"].rank(method="first", ascending=False),
                q=RFM_QUANTILE_COUNT,
                labels=RFM_SCORE_LABELS,
            ).astype(int)
            rfm["f_score"] = pd.qcut(
                rfm["frequency"].rank(method="first"),
                q=RFM_QUANTILE_COUNT,
                labels=RFM_SCORE_LABELS,
            ).astype(int)
            rfm["m_score"] = pd.qcut(
                rfm["monetary"].rank(method="first"),
                q=RFM_QUANTILE_COUNT,
                labels=RFM_SCORE_LABELS,
            ).astype(int)
        rfm["segment"] = np.select(
            [
                (rfm[["r_score", "f_score", "m_score"]] >= RFM_HIGH_SCORE).all(axis=1),
                rfm["f_score"] >= RFM_HIGH_SCORE,
                (rfm["r_score"] == RFM_BEST_SCORE) & (rfm["f_score"] <= RFM_LOW_SCORE),
                rfm["r_score"] <= RFM_LOW_SCORE,
            ],
            ["VIP", "Loyal", "New", "Churned"],
            default="Potential",
        )
        return rfm

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise KeyError("Missing columns: {}".format(", ".join(sorted(missing))))

    def _loaded_data(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Call load_data before performing analysis")
        if self.data.empty:
            raise ValueError("Data must contain at least one row")
        return self.data
