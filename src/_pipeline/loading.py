"""Raw-data loading and normalized cache materialization for the pipeline."""

from pathlib import Path

import pandas as pd

from src._pipeline.artifacts import ArtifactStore
from src._pipeline.contracts import (
    ARTICLE_NORMALIZED_REQUIRED_COLUMNS,
    ARTICLE_RENAMES,
    ARTICLES_CACHE_FILENAME,
    CSV_APPEND_MODE,
    CSV_WRITE_MODE,
    CUSTOMER_ID_COLUMN,
    CUSTOMER_NORMALIZED_REQUIRED_COLUMNS,
    CUSTOMERS_CACHE_FILENAME,
    IMAGE_PATH_COLUMN,
    ORDER_DATE_COLUMN,
    PRODUCT_ID_COLUMN,
    RAW_ARTICLE_ID_COLUMN,
    RAW_ARTICLE_REQUIRED_COLUMNS,
    RAW_ARTICLES_FILENAME,
    RAW_CUSTOMER_REQUIRED_COLUMNS,
    RAW_CUSTOMERS_FILENAME,
    RAW_TRANSACTION_DTYPES,
    RAW_TRANSACTION_REQUIRED_COLUMNS,
    RAW_TRANSACTIONS_FILENAME,
    STRICT_PARSING_ERRORS,
    STRING_DTYPE,
    TRANSACTION_NORMALIZED_REQUIRED_COLUMNS,
    TRANSACTION_RENAMES,
    TRANSACTIONS_CACHE_FILENAME,
    UNIT_PRICE_COLUMN,
)
from src.runtime import RuntimeContext


class DataLoader:
    """Load raw H&M data or reuse valid normalized artifact caches."""

    def __init__(
        self,
        context: RuntimeContext,
        artifacts: ArtifactStore,
        chunksize: int,
    ) -> None:
        self.context = context
        self.artifacts = artifacts
        self.chunksize = chunksize
        self.runtime_transactions_path = (
            context.processed_root / TRANSACTIONS_CACHE_FILENAME
        )
        self.runtime_customers_path = context.processed_root / CUSTOMERS_CACHE_FILENAME
        self.runtime_articles_path = context.processed_root / ARTICLES_CACHE_FILENAME

    def load_customers(self, *, force: bool) -> tuple[pd.DataFrame, Path]:
        """Return normalized customers without imputing missing values."""
        source = None if force else self.artifacts.find_reusable_csv(
            "customers",
            self.runtime_customers_path,
            CUSTOMERS_CACHE_FILENAME,
            CUSTOMER_NORMALIZED_REQUIRED_COLUMNS,
            forbidden_columns=("fashion_news_frequency", "age_was_missing"),
        )
        if source is not None:
            customers = pd.read_csv(
                source,
                dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE},
            ).loc[:, list(CUSTOMER_NORMALIZED_REQUIRED_COLUMNS)].copy()
            return customers, source

        raw = self._require_raw_data_root()
        customers = pd.read_csv(
            raw / RAW_CUSTOMERS_FILENAME,
            usecols=RAW_CUSTOMER_REQUIRED_COLUMNS,
            dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE},
        )
        self._require(customers, RAW_CUSTOMER_REQUIRED_COLUMNS)
        self._validate_dimension_keys(
            customers,
            CUSTOMER_ID_COLUMN,
            RAW_CUSTOMERS_FILENAME,
        )
        customers = customers.loc[:, list(CUSTOMER_NORMALIZED_REQUIRED_COLUMNS)].copy()
        customers.to_csv(self.runtime_customers_path, index=False)
        self.artifacts.record_status(
            "customers",
            "COMPUTED",
            self.runtime_customers_path,
        )
        return customers, self.runtime_customers_path

    def load_articles(self, *, force: bool) -> tuple[pd.DataFrame, Path]:
        """Return normalized articles with canonical H&M image paths."""
        source = None if force else self.artifacts.find_reusable_csv(
            "articles",
            self.runtime_articles_path,
            ARTICLES_CACHE_FILENAME,
            ARTICLE_NORMALIZED_REQUIRED_COLUMNS,
            forbidden_columns=("category",),
        )
        if source is not None:
            articles = pd.read_csv(
                source,
                dtype={PRODUCT_ID_COLUMN: STRING_DTYPE},
            ).loc[:, list(ARTICLE_NORMALIZED_REQUIRED_COLUMNS)].copy()
            return articles, source

        raw = self._require_raw_data_root()
        articles = pd.read_csv(
            raw / RAW_ARTICLES_FILENAME,
            usecols=RAW_ARTICLE_REQUIRED_COLUMNS,
            dtype={RAW_ARTICLE_ID_COLUMN: STRING_DTYPE},
        )
        self._require(articles, RAW_ARTICLE_REQUIRED_COLUMNS)
        self._validate_dimension_keys(
            articles,
            RAW_ARTICLE_ID_COLUMN,
            RAW_ARTICLES_FILENAME,
        )
        articles = articles.rename(columns=ARTICLE_RENAMES)
        articles[IMAGE_PATH_COLUMN] = articles[PRODUCT_ID_COLUMN].map(
            lambda value: f"images/{value[:3]}/{value}.jpg"
        )
        articles = articles.loc[:, list(ARTICLE_NORMALIZED_REQUIRED_COLUMNS)].copy()
        articles.to_csv(self.runtime_articles_path, index=False)
        self.artifacts.record_status(
            "articles",
            "COMPUTED",
            self.runtime_articles_path,
        )
        return articles, self.runtime_articles_path

    def load_transactions(
        self,
        customers: pd.DataFrame,
        articles: pd.DataFrame,
        *,
        force: bool,
    ) -> tuple[Path, int]:
        """Return a normalized transaction cache after validating foreign keys."""
        source = None if force else self.artifacts.find_reusable_csv(
            "transactions",
            self.runtime_transactions_path,
            TRANSACTIONS_CACHE_FILENAME,
            TRANSACTION_NORMALIZED_REQUIRED_COLUMNS,
            forbidden_columns=("sales_channel_id",),
        )
        if source is not None:
            return source, self.artifacts.csv_row_count(source)

        raw = self._require_raw_data_root()
        total_rows = 0
        first_chunk = True
        for chunk in pd.read_csv(
            raw / RAW_TRANSACTIONS_FILENAME,
            usecols=RAW_TRANSACTION_REQUIRED_COLUMNS,
            dtype=RAW_TRANSACTION_DTYPES,
            chunksize=self.chunksize,
        ):
            self._require(chunk, RAW_TRANSACTION_REQUIRED_COLUMNS)
            if chunk[[CUSTOMER_ID_COLUMN, RAW_ARTICLE_ID_COLUMN]].isna().any().any():
                raise ValueError("transactions_train.csv contains missing identifiers")
            if not chunk[CUSTOMER_ID_COLUMN].isin(customers[CUSTOMER_ID_COLUMN]).all():
                raise ValueError("transactions_train.csv references an unknown customer")
            if not chunk[RAW_ARTICLE_ID_COLUMN].isin(articles[PRODUCT_ID_COLUMN]).all():
                raise ValueError("transactions_train.csv references an unknown article")

            frame = chunk.rename(columns=TRANSACTION_RENAMES)
            frame[ORDER_DATE_COLUMN] = pd.to_datetime(
                frame[ORDER_DATE_COLUMN],
                errors=STRICT_PARSING_ERRORS,
            )
            frame[UNIT_PRICE_COLUMN] = pd.to_numeric(
                frame[UNIT_PRICE_COLUMN],
                errors=STRICT_PARSING_ERRORS,
            )
            frame.to_csv(
                self.runtime_transactions_path,
                mode=CSV_WRITE_MODE if first_chunk else CSV_APPEND_MODE,
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            total_rows += len(frame)

        self.artifacts.record_status(
            "transactions",
            "COMPUTED",
            self.runtime_transactions_path,
        )
        return self.runtime_transactions_path, total_rows

    def _require_raw_data_root(self) -> Path:
        if self.context.raw_data_root is None:
            raise ValueError(
                "A required full-data artifact was unavailable or invalid and no raw H&M dataset is attached. "
                "Attach the H&M competition input or set HM_RAW_DATA_DIR to rebuild it."
            )
        return self.context.raw_data_root

    @staticmethod
    def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    @staticmethod
    def _validate_dimension_keys(frame: pd.DataFrame, column: str, filename: str) -> None:
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"{filename} has missing or duplicate {column} values")
