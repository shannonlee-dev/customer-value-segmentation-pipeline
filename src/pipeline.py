"""Portable, full-data H&M analysis implemented with Pandas, NumPy, and Matplotlib."""

import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from .runtime import RuntimeContext


# Runtime and cache defaults
DEFAULT_CHUNKSIZE = 500_000
STRING_DTYPE = "string"
STRICT_PARSING_ERRORS = "raise"
CSV_WRITE_MODE = "w"
CSV_APPEND_MODE = "a"
MEMMAP_WRITE_MODE = "w+"
TRANSACTIONS_CACHE_FILENAME = "transactions.csv"
CUSTOMERS_CACHE_FILENAME = "customers.csv"
ARTICLES_CACHE_FILENAME = "articles.csv"
IMAGE_FEATURES_CACHE_FILENAME = "product_images.csv"
RFM_OUTPUT_FILENAME = "rfm.csv"
RFM_PARTITIONS_DIRECTORY = "rfm_partitions"
RFM_PARTITION_FILENAME = "part_{index:02d}.csv"

# H&M source schema and normalized schema
RAW_TRANSACTIONS_FILENAME = "transactions_train.csv"
RAW_CUSTOMERS_FILENAME = "customers.csv"
RAW_ARTICLES_FILENAME = "articles.csv"
CUSTOMER_ID_COLUMN = "customer_id"
RAW_ARTICLE_ID_COLUMN = "article_id"
PRODUCT_ID_COLUMN = "product_id"
ORDER_DATE_COLUMN = "order_date"
UNIT_PRICE_COLUMN = "unit_price"
SALES_CHANNEL_COLUMN = "sales_channel_id"
RAW_DATE_COLUMN = "t_dat"
RAW_PRICE_COLUMN = "price"
RAW_TRANSACTION_COLUMNS = (RAW_DATE_COLUMN, CUSTOMER_ID_COLUMN, RAW_ARTICLE_ID_COLUMN, RAW_PRICE_COLUMN, SALES_CHANNEL_COLUMN)
RAW_TRANSACTION_DTYPES = {CUSTOMER_ID_COLUMN: STRING_DTYPE, RAW_ARTICLE_ID_COLUMN: STRING_DTYPE}
CUSTOMER_AGE_COLUMN = "age"
AGE_RAW_COLUMN = "age_raw"
AGE_WAS_MISSING_COLUMN = "age_was_missing"
CUSTOMER_MEMBERSHIP_COLUMN = "club_member_status"
FASHION_NEWS_COLUMN = "fashion_news_frequency"
PRODUCT_NAME_COLUMN = "product_name"
PRODUCT_NAME_LENGTH_COLUMN = "product_name_length"
CATEGORY_COLUMN = "category"
RAW_PRODUCT_NAME_COLUMN = "prod_name"
RAW_CATEGORY_COLUMN = "product_group_name"
IMAGE_PATH_COLUMN = "image_path"
IMAGE_STATUS_COLUMN = "image_status"
IMAGE_MEAN_COLUMN = "image_mean"
IMAGE_STD_COLUMN = "image_std"
CUSTOMER_REQUIRED_COLUMNS = (CUSTOMER_ID_COLUMN, CUSTOMER_AGE_COLUMN, CUSTOMER_MEMBERSHIP_COLUMN, FASHION_NEWS_COLUMN)
ARTICLE_REQUIRED_COLUMNS = (RAW_ARTICLE_ID_COLUMN, RAW_PRODUCT_NAME_COLUMN, RAW_CATEGORY_COLUMN)
TRANSACTION_RENAMES = {RAW_DATE_COLUMN: ORDER_DATE_COLUMN, RAW_ARTICLE_ID_COLUMN: PRODUCT_ID_COLUMN, RAW_PRICE_COLUMN: UNIT_PRICE_COLUMN}
ARTICLE_RENAMES = {RAW_ARTICLE_ID_COLUMN: PRODUCT_ID_COLUMN, RAW_PRODUCT_NAME_COLUMN: PRODUCT_NAME_COLUMN, RAW_CATEGORY_COLUMN: CATEGORY_COLUMN}

# Data-quality and image-feature policies
DEFAULT_MISSING_VALUE_STRATEGY = "median"
SUPPORTED_MISSING_VALUE_STRATEGIES = ("median", "mean")
IMAGE_DIRECTORY = "images"
IMAGE_FILE_EXTENSION = ".jpg"
IMAGE_CHANNEL_COUNT = 3
IMAGE_STATUS_OK = "ok"
IMAGE_STATUS_MISSING = "missing"
IMAGE_STATUS_DECODE_ERROR = "decode_error"

# IQR policies
DEFAULT_OUTLIER_COLUMN = UNIT_PRICE_COLUMN
DEFAULT_IQR_THRESHOLD = 1.5
IQR_QUANTILES = (0.25, 0.75)
MEMMAP_DTYPE = "float64"
IQR_CACHE_EXTENSION = ".dat"

# RFM policies
DEFAULT_RFM_CUSTOMER_COLUMN = CUSTOMER_ID_COLUMN
DEFAULT_RFM_DATE_COLUMN = ORDER_DATE_COLUMN
DEFAULT_RFM_AMOUNT_COLUMN = UNIT_PRICE_COLUMN
DEFAULT_RFM_PARTITION_COUNT = 64
RFM_REFERENCE_DAY_OFFSET = 1
RFM_SCORE_QUANTILE_COUNT = 4
RFM_SCORE_LABELS = (1, 2, 3, 4)
RFM_HIGH_SCORE = 3
RFM_BEST_RECENCY_SCORE = 4
RFM_LOW_RECENCY_SCORE = 2
RFM_SEGMENT_COLUMN = "segment"
RFM_RECENCY_COLUMN = "recency"
RFM_FREQUENCY_COLUMN = "frequency"
RFM_MONETARY_COLUMN = "monetary"
RFM_RECENCY_SCORE_COLUMN = "r_score"
RFM_FREQUENCY_SCORE_COLUMN = "f_score"
RFM_MONETARY_SCORE_COLUMN = "m_score"
RFM_SEGMENT_VIP = "VIP"
RFM_SEGMENT_LOYAL = "Loyal"
RFM_SEGMENT_NEW = "New"
RFM_SEGMENT_CHURNED = "Churned"
RFM_SEGMENT_POTENTIAL = "Potential"


class DataAnalyzer:
    """Public facade for loading, cleaning, feature engineering, IQR, and RFM."""

    def __init__(self, context: RuntimeContext, chunksize: int = DEFAULT_CHUNKSIZE):
        self.context = context
        self.chunksize = chunksize
        self.transactions_path = context.processed_root / TRANSACTIONS_CACHE_FILENAME
        self.customers_path = context.processed_root / CUSTOMERS_CACHE_FILENAME
        self.articles_path = context.processed_root / ARTICLES_CACHE_FILENAME
        self.images_path = context.feature_root / IMAGE_FEATURES_CACHE_FILENAME
        self.customers: pd.DataFrame | None = None
        self.articles: pd.DataFrame | None = None

    def load_data(self) -> dict[str, int]:
        """Cache every source row as normalized CSV files; never sample analysis data."""
        raw = self.context.raw_data_root
        customers = pd.read_csv(raw / RAW_CUSTOMERS_FILENAME, dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE})
        articles = pd.read_csv(raw / RAW_ARTICLES_FILENAME, dtype={RAW_ARTICLE_ID_COLUMN: STRING_DTYPE})
        self._require(customers, CUSTOMER_REQUIRED_COLUMNS)
        self._require(articles, ARTICLE_REQUIRED_COLUMNS)
        self._validate_dimension_keys(customers, CUSTOMER_ID_COLUMN, RAW_CUSTOMERS_FILENAME)
        self._validate_dimension_keys(articles, RAW_ARTICLE_ID_COLUMN, RAW_ARTICLES_FILENAME)
        self.customers = customers.rename(columns={CUSTOMER_AGE_COLUMN: AGE_RAW_COLUMN})
        self.articles = articles.rename(columns=ARTICLE_RENAMES)
        self.articles[PRODUCT_NAME_LENGTH_COLUMN] = self.articles[PRODUCT_NAME_COLUMN].astype(STRING_DTYPE).str.len()
        self.articles[IMAGE_PATH_COLUMN] = self.articles[PRODUCT_ID_COLUMN].map(
            lambda value: f"{IMAGE_DIRECTORY}/{value[:3]}/{value}{IMAGE_FILE_EXTENSION}"
        )
        self.handle_missing_values(AGE_RAW_COLUMN, CUSTOMER_MEMBERSHIP_COLUMN)
        self.customers.to_csv(self.customers_path, index=False)
        self.articles.to_csv(self.articles_path, index=False)

        total_rows, first_chunk = 0, True
        for chunk in pd.read_csv(
            raw / RAW_TRANSACTIONS_FILENAME,
            usecols=RAW_TRANSACTION_COLUMNS,
            dtype=RAW_TRANSACTION_DTYPES,
            chunksize=self.chunksize,
        ):
            self._require(chunk, RAW_TRANSACTION_COLUMNS)
            if chunk[[CUSTOMER_ID_COLUMN, RAW_ARTICLE_ID_COLUMN]].isna().any().any():
                raise ValueError("transactions_train.csv contains missing identifiers")
            if not chunk[CUSTOMER_ID_COLUMN].isin(self.customers[CUSTOMER_ID_COLUMN]).all():
                raise ValueError("transactions_train.csv references an unknown customer")
            if not chunk[RAW_ARTICLE_ID_COLUMN].isin(articles[RAW_ARTICLE_ID_COLUMN]).all():
                raise ValueError("transactions_train.csv references an unknown article")
            frame = chunk.rename(columns=TRANSACTION_RENAMES)
            frame[ORDER_DATE_COLUMN] = pd.to_datetime(frame[ORDER_DATE_COLUMN], errors=STRICT_PARSING_ERRORS)
            frame[UNIT_PRICE_COLUMN] = pd.to_numeric(frame[UNIT_PRICE_COLUMN], errors=STRICT_PARSING_ERRORS)
            frame.to_csv(
                self.transactions_path,
                mode=CSV_WRITE_MODE if first_chunk else CSV_APPEND_MODE,
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            total_rows += len(frame)
        return {"transaction_rows": total_rows, "customer_rows": len(self.customers), "product_rows": len(self.articles)}

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
        self.customers[AGE_WAS_MISSING_COLUMN] = self.customers[column].isna()
        self.customers[CUSTOMER_AGE_COLUMN] = self.customers[column].fillna(grouped).fillna(fallback)
        return {
            "missing_before": missing_before,
            "missing_after": int(self.customers[CUSTOMER_AGE_COLUMN].isna().sum()),
        }

    def engineer_features(self) -> pd.DataFrame:
        """Reuse cached image features or calculate full-array NumPy Mean/Std once."""
        if self.images_path.is_file():
            return pd.read_csv(self.images_path, dtype={PRODUCT_ID_COLUMN: STRING_DTYPE})
        if self.articles is None:
            self.articles = pd.read_csv(self.articles_path, dtype={PRODUCT_ID_COLUMN: STRING_DTYPE})
        records: list[dict[str, object]] = []
        for product_id, image_path in self.articles[[PRODUCT_ID_COLUMN, IMAGE_PATH_COLUMN]].itertuples(index=False):
            path = self.context.raw_data_root / image_path
            record: dict[str, object] = {
                PRODUCT_ID_COLUMN: product_id,
                IMAGE_PATH_COLUMN: image_path,
                IMAGE_STATUS_COLUMN: IMAGE_STATUS_OK,
                IMAGE_MEAN_COLUMN: np.nan,
                IMAGE_STD_COLUMN: np.nan,
            }
            if not path.is_file():
                record[IMAGE_STATUS_COLUMN] = IMAGE_STATUS_MISSING
            else:
                try:
                    pixels = np.asarray(mpimg.imread(path))
                    if pixels.ndim == 2:
                        pixels = pixels[..., np.newaxis]
                    pixels = pixels[..., :IMAGE_CHANNEL_COUNT]
                    record[IMAGE_MEAN_COLUMN] = float(np.mean(pixels))
                    record[IMAGE_STD_COLUMN] = float(np.std(pixels))
                except (OSError, ValueError, SyntaxError):
                    record[IMAGE_STATUS_COLUMN] = IMAGE_STATUS_DECODE_ERROR
            records.append(record)
        features = pd.DataFrame.from_records(records)
        features.to_csv(self.images_path, index=False)
        return features

    def detect_outliers(
        self,
        column: str = DEFAULT_OUTLIER_COLUMN,
        threshold: float = DEFAULT_IQR_THRESHOLD,
    ) -> dict[str, float | int]:
        """Calculate exact full-data IQR fences and counts with a disk-backed NumPy array."""
        row_count = sum(len(chunk) for chunk in pd.read_csv(self.transactions_path, usecols=[column], chunksize=self.chunksize))
        cache_path = self.context.aggregate_root / f"{column}_values{IQR_CACHE_EXTENSION}"
        values = np.memmap(cache_path, dtype=MEMMAP_DTYPE, mode=MEMMAP_WRITE_MODE, shape=(row_count,))
        offset = 0
        for chunk in pd.read_csv(self.transactions_path, usecols=[column], chunksize=self.chunksize):
            numeric = pd.to_numeric(chunk[column], errors=STRICT_PARSING_ERRORS).to_numpy(dtype=MEMMAP_DTYPE)
            values[offset : offset + len(numeric)] = numeric
            offset += len(numeric)
        q1, q3 = np.quantile(values, IQR_QUANTILES)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        count = int(np.count_nonzero((values < lower) | (values > upper)))
        del values
        return {"q1": float(q1), "q3": float(q3), "lower_fence": float(lower), "upper_fence": float(upper), "outlier_count": count}

    def calculate_rfm(
        self,
        customer_col: str = DEFAULT_RFM_CUSTOMER_COLUMN,
        date_col: str = DEFAULT_RFM_DATE_COLUMN,
        amount_col: str = DEFAULT_RFM_AMOUNT_COLUMN,
        analysis_date: str | pd.Timestamp | None = None,
        partition_count: int = DEFAULT_RFM_PARTITION_COUNT,
    ) -> pd.DataFrame:
        """Compute every-customer RFM through hash-partitioned CSV aggregation."""
        partition_root = self.context.aggregate_root / RFM_PARTITIONS_DIRECTORY
        partition_root.mkdir(exist_ok=True)
        paths = [
            partition_root / RFM_PARTITION_FILENAME.format(index=index)
            for index in range(partition_count)
        ]
        for path in paths:
            if path.exists():
                path.unlink()
        max_date: pd.Timestamp | None = None
        initialized: set[int] = set()
        for chunk in pd.read_csv(
            self.transactions_path,
            usecols=[customer_col, date_col, amount_col],
            dtype={customer_col: STRING_DTYPE},
            chunksize=self.chunksize,
        ):
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors=STRICT_PARSING_ERRORS)
            chunk[amount_col] = pd.to_numeric(chunk[amount_col], errors=STRICT_PARSING_ERRORS)
            chunk_max = pd.Timestamp(chunk[date_col].max())
            max_date = chunk_max if max_date is None or chunk_max > max_date else max_date
            bucket = pd.util.hash_pandas_object(chunk[customer_col], index=False).to_numpy() % partition_count
            for index in np.unique(bucket):
                subset = chunk.loc[bucket == index]
                subset.to_csv(paths[int(index)], mode="a", header=int(index) not in initialized, index=False)
                initialized.add(int(index))
        if max_date is None:
            raise ValueError("Cannot calculate RFM from an empty transaction cache")
        reference = (
            pd.Timestamp(analysis_date)
            if analysis_date is not None
            else pd.Timestamp(max_date.date()) + pd.offsets.Day(RFM_REFERENCE_DAY_OFFSET)
        )
        aggregates: list[pd.DataFrame] = []
        for path in paths:
            if path.is_file():
                partition = pd.read_csv(path, dtype={customer_col: STRING_DTYPE})
                partition[date_col] = pd.to_datetime(partition[date_col], errors=STRICT_PARSING_ERRORS)
                aggregates.append(
                    partition.groupby(customer_col, as_index=False).agg(
                        **{
                            RFM_RECENCY_COLUMN: (date_col, lambda dates: (reference - dates.max()).days),
                            RFM_FREQUENCY_COLUMN: (date_col, "nunique"),
                            RFM_MONETARY_COLUMN: (amount_col, "sum"),
                        }
                    )
                )
        rfm = pd.concat(aggregates, ignore_index=True)
        score_specs = (
            (RFM_RECENCY_COLUMN, False, RFM_RECENCY_SCORE_COLUMN),
            (RFM_FREQUENCY_COLUMN, True, RFM_FREQUENCY_SCORE_COLUMN),
            (RFM_MONETARY_COLUMN, True, RFM_MONETARY_SCORE_COLUMN),
        )
        for value, ascending, score in score_specs:
            rfm[score] = (
                RFM_SCORE_LABELS[-1]
                if len(rfm) == 1
                else pd.qcut(
                    rfm[value].rank(method="first", ascending=ascending),
                    RFM_SCORE_QUANTILE_COUNT,
                    labels=RFM_SCORE_LABELS,
                ).astype(int)
            )
        score_columns = [RFM_RECENCY_SCORE_COLUMN, RFM_FREQUENCY_SCORE_COLUMN, RFM_MONETARY_SCORE_COLUMN]
        rfm[RFM_SEGMENT_COLUMN] = np.select(
            [
                (rfm[score_columns] >= RFM_HIGH_SCORE).all(axis=1),
                rfm[RFM_FREQUENCY_SCORE_COLUMN] >= RFM_HIGH_SCORE,
                (rfm[RFM_RECENCY_SCORE_COLUMN] == RFM_BEST_RECENCY_SCORE)
                & (rfm[RFM_FREQUENCY_SCORE_COLUMN] <= RFM_LOW_RECENCY_SCORE),
                rfm[RFM_RECENCY_SCORE_COLUMN] <= RFM_LOW_RECENCY_SCORE,
            ],
            [RFM_SEGMENT_VIP, RFM_SEGMENT_LOYAL, RFM_SEGMENT_NEW, RFM_SEGMENT_CHURNED],
            default=RFM_SEGMENT_POTENTIAL,
        )
        rfm.to_csv(self.context.aggregate_root / RFM_OUTPUT_FILENAME, index=False)
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
