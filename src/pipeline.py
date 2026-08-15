"""Multimodal preprocessing helpers for the prepared H&M cohort."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import image as mpimg


class DataAnalyzer:
    """Load and progressively enrich a prepared customer cohort."""

    def __init__(self, data_path: Union[str, Path], image_root: Union[str, Path]) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.data = None

    def load_data(
        self,
        date_columns: Sequence[str] = ("order_date",),
        numeric_columns: Sequence[str] = ("unit_price", "age"),
    ) -> pd.DataFrame:
        """Load cohort data while enforcing its date and numeric types."""
        if not self.data_path.is_file():
            raise FileNotFoundError("Data file does not exist: {}".format(self.data_path))

        frame = pd.read_csv(
            self.data_path,
            dtype={"customer_id": "string", "product_id": "string"},
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
        self, column: str, group_col: str, strategy: str = "median"
    ) -> pd.DataFrame:
        """Fill numeric gaps from each group, then the global aggregate."""
        frame = self._loaded_data()
        self._require_columns(frame, (column, group_col))
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError("Column '{}' must be numeric".format(column))
        if strategy not in ("median", "mean"):
            raise ValueError("Unsupported imputation strategy: {}".format(strategy))

        group_values = frame.groupby(group_col, dropna=False)[column].transform(strategy)
        global_value = getattr(frame[column], strategy)()
        if pd.isna(global_value):
            raise ValueError("Column '{}' has no values available for imputation".format(column))
        frame[column] = frame[column].fillna(group_values).fillna(global_value)
        return frame

    def engineer_features(
        self,
        image_col: str = "image_path",
        product_col: str = "product_id",
        text_col: str = "product_name",
        downsample_step: int = 35,
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
                ::downsample_step, ::downsample_step, :3
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
        self, column: str, threshold: float = 1.5
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
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        interquartile_range = upper_quartile - lower_quartile
        lower_fence = lower_quartile - threshold * interquartile_range
        upper_fence = upper_quartile + threshold * interquartile_range
        outliers = frame.loc[
            (frame[column] < lower_fence) | (frame[column] > upper_fence)
        ].copy()
        return outliers, float(lower_fence), float(upper_fence)

    def calculate_rfm(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "unit_price",
        frequency_mode: str = "unique_dates",
        analysis_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Score customers by recency, frequency, monetary value, and segment."""
        frame = self._loaded_data()
        self._require_columns(frame, (customer_col, date_col, amount_col))
        if frequency_mode not in ("unique_dates", "rows"):
            raise ValueError("frequency_mode must be 'unique_dates' or 'rows'")
        if not pd.api.types.is_numeric_dtype(frame[amount_col]):
            raise TypeError("Column '{}' must be numeric".format(amount_col))

        working = frame[[customer_col, date_col, amount_col]].copy()
        working[date_col] = pd.to_datetime(working[date_col], errors="raise")
        grouped = working.groupby(customer_col)
        reference_date = (
            pd.Timestamp(analysis_date)
            if analysis_date is not None
            else working[date_col].max() + pd.Timedelta(1, unit="D")
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
            rfm["r_score"] = 4
            rfm["f_score"] = 4
            rfm["m_score"] = 4
        else:
            rfm["r_score"] = pd.qcut(
                rfm["recency"].rank(method="first", ascending=False),
                q=4,
                labels=[1, 2, 3, 4],
            ).astype(int)
            rfm["f_score"] = pd.qcut(
                rfm["frequency"].rank(method="first"),
                q=4,
                labels=[1, 2, 3, 4],
            ).astype(int)
            rfm["m_score"] = pd.qcut(
                rfm["monetary"].rank(method="first"),
                q=4,
                labels=[1, 2, 3, 4],
            ).astype(int)
        rfm["segment"] = np.select(
            [
                (rfm[["r_score", "f_score", "m_score"]] >= 3).all(axis=1),
                rfm["f_score"] >= 3,
                (rfm["r_score"] == 4) & (rfm["f_score"] <= 2),
                rfm["r_score"] <= 2,
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
