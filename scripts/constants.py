"""Shared configuration and schema constants for the project scripts."""

import re
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/analysis_report.ipynb")
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_PATH
DEFAULT_LOG_PATH = Path("artifacts/notebook_execution.log")
CUSTOMER_ID_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")
PRODUCT_ID_PATTERN = re.compile(r"\b0\d{9}\b")
CODE_CELL_TYPE = "code"
RENDERED_OUTPUT_TYPES = ("display_data", "execute_result")
TEXT_MIME_PREFIX = "text/"
CHART_MIME_TYPE = "image/png"
MINIMUM_CHART_COUNT = 6
PASS_STATUS = "PASS"
EVIDENCE_SUMMARY_KEYS = (
    "cell_count",
    "code_cell_count",
    "output_count",
    "chart_count",
    "error_count",
    "unexecuted_cell_count",
    "redaction_status",
    "status",
)

DEFAULT_COHORT_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_CHUNKSIZE = 1_000_000
DEFAULT_MINIMUM_ROWS = 1_000
CSV_ENCODING = "utf-8"
CSV_DATE_FORMAT = "%Y-%m-%d"
STRICT_PARSING_ERRORS = "raise"
CUSTOMER_ID_COLUMN = "customer_id"
ARTICLE_ID_COLUMN = "article_id"
PRODUCT_ID_COLUMN = "product_id"
ORDER_DATE_COLUMN = "order_date"
IMAGE_PATH_COLUMN = "image_path"
STRING_DTYPE = "string"
TRANSACTIONS_FILENAME = "transactions_train.csv"
ARTICLES_FILENAME = "articles.csv"
CUSTOMERS_FILENAME = "customers.csv"
IMAGES_DIRECTORY = "images"
REQUIRED_IMAGE_SHAPE = (1750, 1166, 3)
SHA256_BLOCK_SIZE = 1_024 * 1_024
TRANSACTION_COLUMNS = [
    "t_dat",
    CUSTOMER_ID_COLUMN,
    ARTICLE_ID_COLUMN,
    "price",
    "sales_channel_id",
]
ARTICLE_COLUMNS = [ARTICLE_ID_COLUMN, "prod_name", "product_group_name"]
CUSTOMER_COLUMNS = [
    CUSTOMER_ID_COLUMN,
    "age",
    "club_member_status",
    "fashion_news_frequency",
]
TRANSACTION_RENAMES = {
    "t_dat": ORDER_DATE_COLUMN,
    ARTICLE_ID_COLUMN: PRODUCT_ID_COLUMN,
    "price": "unit_price",
}
ARTICLE_RENAMES = {
    ARTICLE_ID_COLUMN: PRODUCT_ID_COLUMN,
    "prod_name": "product_name",
    "product_group_name": "category",
}
ARTICLE_METADATA_INDICATOR = "_article_metadata"
CUSTOMER_METADATA_INDICATOR = "_customer_metadata"
MERGE_MATCHED_VALUE = "both"
OUTPUT_COLUMNS = [
    ORDER_DATE_COLUMN,
    CUSTOMER_ID_COLUMN,
    PRODUCT_ID_COLUMN,
    "product_name",
    "category",
    "unit_price",
    "sales_channel_id",
    "age",
    "club_member_status",
    "fashion_news_frequency",
    IMAGE_PATH_COLUMN,
]
OUTPUT_SORT_COLUMNS = [ORDER_DATE_COLUMN, CUSTOMER_ID_COLUMN, PRODUCT_ID_COLUMN]
