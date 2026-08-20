"""Portable, full-data H&M analysis implemented with Pandas, NumPy, and Matplotlib."""

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from .runtime import RuntimeContext


class DataAnalyzer:
    """Public facade for loading, cleaning, feature engineering, IQR, and RFM."""

    def __init__(self, context: RuntimeContext, chunksize: int = 500_000):
        self.context = context
        self.chunksize = chunksize
        self.transactions_path = context.processed_root / "transactions.csv"
        self.customers_path = context.processed_root / "customers.csv"
        self.articles_path = context.processed_root / "articles.csv"
        self.images_path = context.feature_root / "product_images.csv"
        self.customers: pd.DataFrame | None = None
        self.articles: pd.DataFrame | None = None

    def load_data(self) -> dict[str, int]:
        """Cache every source row as normalized CSV files; never sample analysis data."""
        raw = self.context.raw_data_root
        customers = pd.read_csv(raw / "customers.csv", dtype={"customer_id": "string"})
        articles = pd.read_csv(raw / "articles.csv", dtype={"article_id": "string"})
        self._require(customers, ["customer_id", "age", "club_member_status", "fashion_news_frequency"])
        self._require(articles, ["article_id", "prod_name", "product_group_name"])
        self._validate_dimension_keys(customers, "customer_id", "customers.csv")
        self._validate_dimension_keys(articles, "article_id", "articles.csv")
        self.customers = customers.rename(columns={"age": "age_raw"})
        self.articles = articles.rename(columns={"article_id": "product_id", "prod_name": "product_name", "product_group_name": "category"})
        self.articles["product_name_length"] = self.articles["product_name"].astype("string").str.len()
        self.articles["image_path"] = self.articles["product_id"].map(lambda value: f"images/{value[:3]}/{value}.jpg")
        self.handle_missing_values("age_raw", "club_member_status")
        self.customers.to_csv(self.customers_path, index=False)
        self.articles.to_csv(self.articles_path, index=False)

        total_rows, first_chunk = 0, True
        for chunk in pd.read_csv(raw / "transactions_train.csv", usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"], dtype={"customer_id": "string", "article_id": "string"}, chunksize=self.chunksize):
            self._require(chunk, ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
            if chunk[["customer_id", "article_id"]].isna().any().any():
                raise ValueError("transactions_train.csv contains missing identifiers")
            if not chunk["customer_id"].isin(self.customers["customer_id"]).all():
                raise ValueError("transactions_train.csv references an unknown customer")
            if not chunk["article_id"].isin(articles["article_id"]).all():
                raise ValueError("transactions_train.csv references an unknown article")
            frame = chunk.rename(columns={"t_dat": "order_date", "article_id": "product_id", "price": "unit_price"})
            frame["order_date"] = pd.to_datetime(frame["order_date"], errors="raise")
            frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="raise")
            frame.to_csv(self.transactions_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False
            total_rows += len(frame)
        return {"transaction_rows": total_rows, "customer_rows": len(self.customers), "product_rows": len(self.articles)}

    def handle_missing_values(self, column: str, group_column: str, strategy: str = "median") -> dict[str, int]:
        """Impute a customer-level numeric attribute using its membership-group statistic."""
        if self.customers is None:
            raise ValueError("Call load_data before handle_missing_values")
        if strategy not in {"median", "mean"}:
            raise ValueError("strategy must be 'median' or 'mean'")
        missing_before = int(self.customers[column].isna().sum())
        grouped = self.customers.groupby(group_column, dropna=False)[column].transform(strategy)
        fallback = getattr(self.customers[column], strategy)()
        self.customers["age_was_missing"] = self.customers[column].isna()
        self.customers["age"] = self.customers[column].fillna(grouped).fillna(fallback)
        return {"missing_before": missing_before, "missing_after": int(self.customers["age"].isna().sum())}

    def engineer_features(self) -> pd.DataFrame:
        """Read every available product image and calculate full-array NumPy Mean/Std."""
        if self.articles is None:
            self.articles = pd.read_csv(self.articles_path, dtype={"product_id": "string"})
        records: list[dict[str, object]] = []
        for product_id, image_path in self.articles[["product_id", "image_path"]].itertuples(index=False):
            path = self.context.raw_data_root / image_path
            record: dict[str, object] = {"product_id": product_id, "image_path": image_path, "image_status": "ok", "image_mean": np.nan, "image_std": np.nan}
            if not path.is_file():
                record["image_status"] = "missing"
            else:
                try:
                    pixels = np.asarray(mpimg.imread(path))
                    if pixels.ndim == 2:
                        pixels = pixels[..., np.newaxis]
                    pixels = pixels[..., :3]
                    record["image_mean"] = float(np.mean(pixels))
                    record["image_std"] = float(np.std(pixels))
                except (OSError, ValueError, SyntaxError):
                    record["image_status"] = "decode_error"
            records.append(record)
        features = pd.DataFrame.from_records(records)
        features.to_csv(self.images_path, index=False)
        return features

    def detect_outliers(self, column: str = "unit_price", threshold: float = 1.5) -> dict[str, float | int]:
        """Calculate exact full-data IQR fences and counts with a disk-backed NumPy array."""
        row_count = sum(len(chunk) for chunk in pd.read_csv(self.transactions_path, usecols=[column], chunksize=self.chunksize))
        cache_path = self.context.aggregate_root / f"{column}_values.dat"
        values = np.memmap(cache_path, dtype="float64", mode="w+", shape=(row_count,))
        offset = 0
        for chunk in pd.read_csv(self.transactions_path, usecols=[column], chunksize=self.chunksize):
            numeric = pd.to_numeric(chunk[column], errors="raise").to_numpy(dtype="float64")
            values[offset : offset + len(numeric)] = numeric
            offset += len(numeric)
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        count = int(np.count_nonzero((values < lower) | (values > upper)))
        del values
        return {"q1": float(q1), "q3": float(q3), "lower_fence": float(lower), "upper_fence": float(upper), "outlier_count": count}

    def calculate_rfm(self, customer_col: str = "customer_id", date_col: str = "order_date", amount_col: str = "unit_price", analysis_date: str | pd.Timestamp | None = None, partition_count: int = 64) -> pd.DataFrame:
        """Compute every-customer RFM through hash-partitioned CSV aggregation."""
        partition_root = self.context.aggregate_root / "rfm_partitions"
        partition_root.mkdir(exist_ok=True)
        paths = [partition_root / f"part_{index:02d}.csv" for index in range(partition_count)]
        for path in paths:
            if path.exists():
                path.unlink()
        max_date: pd.Timestamp | None = None
        initialized: set[int] = set()
        for chunk in pd.read_csv(self.transactions_path, usecols=[customer_col, date_col, amount_col], dtype={customer_col: "string"}, chunksize=self.chunksize):
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="raise")
            chunk[amount_col] = pd.to_numeric(chunk[amount_col], errors="raise")
            chunk_max = pd.Timestamp(chunk[date_col].max())
            max_date = chunk_max if max_date is None or chunk_max > max_date else max_date
            bucket = pd.util.hash_pandas_object(chunk[customer_col], index=False).to_numpy() % partition_count
            for index in np.unique(bucket):
                subset = chunk.loc[bucket == index]
                subset.to_csv(paths[int(index)], mode="a", header=int(index) not in initialized, index=False)
                initialized.add(int(index))
        if max_date is None:
            raise ValueError("Cannot calculate RFM from an empty transaction cache")
        reference = pd.Timestamp(analysis_date) if analysis_date is not None else pd.Timestamp(max_date.date()) + pd.offsets.Day(1)
        aggregates: list[pd.DataFrame] = []
        for path in paths:
            if path.is_file():
                partition = pd.read_csv(path, dtype={customer_col: "string"})
                partition[date_col] = pd.to_datetime(partition[date_col], errors="raise")
                aggregates.append(partition.groupby(customer_col, as_index=False).agg(recency=(date_col, lambda dates: (reference - dates.max()).days), frequency=(date_col, "nunique"), monetary=(amount_col, "sum")))
        rfm = pd.concat(aggregates, ignore_index=True)
        for value, ascending, score in [("recency", False, "r_score"), ("frequency", True, "f_score"), ("monetary", True, "m_score")]:
            rfm[score] = 4 if len(rfm) == 1 else pd.qcut(rfm[value].rank(method="first", ascending=ascending), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["segment"] = np.select([(rfm[["r_score", "f_score", "m_score"]] >= 3).all(axis=1), rfm["f_score"] >= 3, (rfm["r_score"] == 4) & (rfm["f_score"] <= 2), rfm["r_score"] <= 2], ["VIP", "Loyal", "New", "Churned"], default="Potential")
        rfm.to_csv(self.context.aggregate_root / "rfm.csv", index=False)
        return rfm

    @staticmethod
    def _require(frame: pd.DataFrame, columns: list[str]) -> None:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    @staticmethod
    def _validate_dimension_keys(frame: pd.DataFrame, column: str, filename: str) -> None:
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"{filename} has missing or duplicate {column} values")
