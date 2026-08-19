"""Prepare a deterministic, image-backed H&M customer cohort."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd
from matplotlib import image as mpimg

if __package__:
    from .constants import (
        ARTICLE_COLUMNS,
        ARTICLE_ID_COLUMN,
        ARTICLE_METADATA_INDICATOR,
        ARTICLE_RENAMES,
        ARTICLES_FILENAME,
        CSV_DATE_FORMAT,
        CSV_ENCODING,
        CUSTOMER_COLUMNS,
        CUSTOMER_ID_COLUMN,
        CUSTOMER_METADATA_INDICATOR,
        CUSTOMERS_FILENAME,
        DEFAULT_CHUNKSIZE,
        DEFAULT_COHORT_SIZE,
        DEFAULT_MINIMUM_ROWS,
        DEFAULT_SEED,
        IMAGES_DIRECTORY,
        IMAGE_PATH_COLUMN,
        MERGE_MATCHED_VALUE,
        ORDER_DATE_COLUMN,
        OUTPUT_COLUMNS,
        OUTPUT_SORT_COLUMNS,
        PRODUCT_ID_COLUMN,
        REQUIRED_IMAGE_SHAPE,
        SHA256_BLOCK_SIZE,
        STRICT_PARSING_ERRORS,
        STRING_DTYPE,
        TRANSACTION_COLUMNS,
        TRANSACTION_RENAMES,
        TRANSACTIONS_FILENAME,
    )
else:
    from constants import (
        ARTICLE_COLUMNS,
        ARTICLE_ID_COLUMN,
        ARTICLE_METADATA_INDICATOR,
        ARTICLE_RENAMES,
        ARTICLES_FILENAME,
        CSV_DATE_FORMAT,
        CSV_ENCODING,
        CUSTOMER_COLUMNS,
        CUSTOMER_ID_COLUMN,
        CUSTOMER_METADATA_INDICATOR,
        CUSTOMERS_FILENAME,
        DEFAULT_CHUNKSIZE,
        DEFAULT_COHORT_SIZE,
        DEFAULT_MINIMUM_ROWS,
        DEFAULT_SEED,
        IMAGES_DIRECTORY,
        IMAGE_PATH_COLUMN,
        MERGE_MATCHED_VALUE,
        ORDER_DATE_COLUMN,
        OUTPUT_COLUMNS,
        OUTPUT_SORT_COLUMNS,
        PRODUCT_ID_COLUMN,
        REQUIRED_IMAGE_SHAPE,
        SHA256_BLOCK_SIZE,
        STRICT_PARSING_ERRORS,
        STRING_DTYPE,
        TRANSACTION_COLUMNS,
        TRANSACTION_RENAMES,
        TRANSACTIONS_FILENAME,
    )


def stable_customer_ids(customer_ids, cohort_size=DEFAULT_COHORT_SIZE, seed=DEFAULT_SEED):
    unique_ids = set(customer_ids)
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    if len(unique_ids) < cohort_size:
        raise ValueError("cohort_size exceeds the number of active customers")
    return heapq.nsmallest(
        cohort_size,
        unique_ids,
        key=lambda customer_id: (
            _customer_digest(customer_id, seed),
            customer_id,
        ),
    )


def prepare_cohort(
    raw_dir,
    output_path,
    cohort_size=DEFAULT_COHORT_SIZE,
    seed=DEFAULT_SEED,
    chunksize=DEFAULT_CHUNKSIZE,
    minimum_rows=DEFAULT_MINIMUM_ROWS,
):
    """Build and write a deterministic, metadata-joined H&M customer cohort."""
    _validate_prepare_options(chunksize, minimum_rows)
    raw_dir, transactions_path, articles_path, customers_path, _ = _resolve_source_paths(
        raw_dir
    )
    articles, customers = _load_metadata_sources(articles_path, customers_path)
    active_customer_ids = _load_active_customer_ids(transactions_path, chunksize)
    selected_customer_ids = set(
        stable_customer_ids(active_customer_ids, cohort_size, seed)
    )
    selected_chunks = _load_selected_transactions(
        transactions_path, selected_customer_ids, chunksize
    )
    cohort = _enrich_transactions(selected_chunks, articles, customers)
    cohort, missing_image_rows, shape_mismatch_rows = _filter_available_images(
        cohort, raw_dir, minimum_rows
    )
    return _write_cohort_and_summary(
        cohort=cohort,
        output_path=output_path,
        active_customer_count=len(active_customer_ids),
        selected_customer_count=len(selected_customer_ids),
        selected_transaction_count=sum(len(chunk) for chunk in selected_chunks),
        missing_image_rows=missing_image_rows,
        shape_mismatch_rows=shape_mismatch_rows,
        seed=seed,
        cohort_size=cohort_size,
        minimum_rows=minimum_rows,
    )


def _validate_prepare_options(chunksize, minimum_rows):
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")


def _resolve_source_paths(raw_dir):
    raw_dir = Path(raw_dir)
    transactions_path = raw_dir / TRANSACTIONS_FILENAME
    articles_path = raw_dir / ARTICLES_FILENAME
    customers_path = raw_dir / CUSTOMERS_FILENAME
    images_dir = raw_dir / IMAGES_DIRECTORY
    for source_path in (transactions_path, articles_path, customers_path, images_dir):
        _require_existing_path(source_path)
    return raw_dir, transactions_path, articles_path, customers_path, images_dir


def _load_metadata_sources(articles_path, customers_path):
    articles = _read_metadata(articles_path, ARTICLE_COLUMNS, ARTICLE_ID_COLUMN)
    customers = _read_metadata(customers_path, CUSTOMER_COLUMNS, CUSTOMER_ID_COLUMN)
    return articles, customers


def _load_active_customer_ids(transactions_path, chunksize):
    active_ids = set()
    for chunk in pd.read_csv(
        transactions_path,
        usecols=[CUSTOMER_ID_COLUMN],
        dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE},
        chunksize=chunksize,
    ):
        _require_columns(chunk, [CUSTOMER_ID_COLUMN], transactions_path.name)
        active_ids.update(chunk[CUSTOMER_ID_COLUMN].dropna().unique())
    if not active_ids:
        raise ValueError("transactions_train.csv must not be empty")
    return active_ids


def _load_selected_transactions(transactions_path, selected_customer_ids, chunksize):
    selected_chunks = []
    for chunk in pd.read_csv(
        transactions_path,
        usecols=TRANSACTION_COLUMNS,
        dtype={CUSTOMER_ID_COLUMN: STRING_DTYPE, ARTICLE_ID_COLUMN: STRING_DTYPE},
        chunksize=chunksize,
    ):
        _require_columns(chunk, TRANSACTION_COLUMNS, transactions_path.name)
        selected_chunks.append(
            chunk.loc[chunk[CUSTOMER_ID_COLUMN].isin(selected_customer_ids)].copy()
        )
    selected_chunks = [chunk for chunk in selected_chunks if not chunk.empty]
    if not selected_chunks:
        raise ValueError("No transactions found for selected customers")
    return selected_chunks


def _enrich_transactions(selected_chunks, articles, customers):
    result = pd.concat(selected_chunks, ignore_index=True)
    result = result.rename(columns=TRANSACTION_RENAMES)
    articles = articles.rename(columns=ARTICLE_RENAMES)
    result = result.merge(
        articles,
        on=PRODUCT_ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=ARTICLE_METADATA_INDICATOR,
    )
    result = result.merge(
        customers,
        on=CUSTOMER_ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=CUSTOMER_METADATA_INDICATOR,
    )
    if (
        (result[ARTICLE_METADATA_INDICATOR] != MERGE_MATCHED_VALUE).any()
        or (result[CUSTOMER_METADATA_INDICATOR] != MERGE_MATCHED_VALUE).any()
    ):
        raise ValueError("Selected transactions are missing article or customer metadata")
    return result.drop(columns=[ARTICLE_METADATA_INDICATOR, CUSTOMER_METADATA_INDICATOR])


def _filter_available_images(cohort, raw_dir, minimum_rows):
    result = cohort.copy()
    result[ORDER_DATE_COLUMN] = pd.to_datetime(
        result[ORDER_DATE_COLUMN], errors=STRICT_PARSING_ERRORS
    )
    result[IMAGE_PATH_COLUMN] = result[PRODUCT_ID_COLUMN].map(_image_path)
    image_exists = result[IMAGE_PATH_COLUMN].map(lambda path: (raw_dir / path).is_file())
    missing_image_rows = int((~image_exists).sum())
    result = result.loc[image_exists].copy()
    image_shapes = {
        path: tuple(mpimg.imread(raw_dir / path).shape)
        for path in result[IMAGE_PATH_COLUMN].drop_duplicates()
    }
    valid_paths = [
        path for path, shape in image_shapes.items() if shape == REQUIRED_IMAGE_SHAPE
    ]
    shape_mismatch_rows = int((~result[IMAGE_PATH_COLUMN].isin(valid_paths)).sum())
    result = result.loc[result[IMAGE_PATH_COLUMN].isin(valid_paths)].copy()
    if len(result) < minimum_rows:
        raise ValueError(
            f"Cohort has {len(result)} supported-image rows; minimum_rows is {minimum_rows}"
        )
    return result, missing_image_rows, shape_mismatch_rows


def _write_cohort_and_summary(
    cohort,
    output_path,
    active_customer_count,
    selected_customer_count,
    selected_transaction_count,
    missing_image_rows,
    shape_mismatch_rows,
    seed,
    cohort_size,
    minimum_rows,
):
    output_path = Path(output_path)
    result = cohort.loc[:, OUTPUT_COLUMNS].sort_values(OUTPUT_SORT_COLUMNS, kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, date_format=CSV_DATE_FORMAT)
    summary = {
        "active_customers": active_customer_count,
        "selected_customers": selected_customer_count,
        "selected_transaction_rows": int(selected_transaction_count),
        "missing_image_rows": missing_image_rows,
        "missing_image_rate": missing_image_rows / selected_transaction_count,
        "shape_mismatch_rows": shape_mismatch_rows,
        "shape_mismatch_rate": shape_mismatch_rows / selected_transaction_count,
        "required_image_shape": REQUIRED_IMAGE_SHAPE,
        "output_rows": len(result),
        "output_columns": len(OUTPUT_COLUMNS),
        "date_range": {
            "start": result[ORDER_DATE_COLUMN].min().strftime(CSV_DATE_FORMAT),
            "end": result[ORDER_DATE_COLUMN].max().strftime(CSV_DATE_FORMAT),
        },
        "seed": seed,
        "cohort_size": cohort_size,
        "minimum_rows": minimum_rows,
        "output_sha256": _sha256(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding=CSV_ENCODING
    )
    return summary


def _customer_digest(customer_id, seed):
    return hashlib.sha256(f"{seed}:{customer_id}".encode(CSV_ENCODING)).digest()


def _require_columns(frame, required, source_name):
    missing = set(required).difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{source_name} is missing required columns: {names}")


def _require_existing_path(path):
    if not path.exists():
        raise FileNotFoundError(f"Required H&M source is missing: {path}")


def _read_metadata(path, columns, key):
    frame = pd.read_csv(path, dtype={key: STRING_DTYPE})
    _require_columns(frame, columns, path.name)
    frame = frame.loc[:, columns].copy()
    if frame.empty:
        raise ValueError(f"{path.name} must not be empty")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate or missing {key} values")
    return frame


def _image_path(article_id):
    article_id = str(article_id)
    return f"{IMAGES_DIRECTORY}/{article_id[:3]}/{article_id}.jpg"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as output_file:
        for block in iter(lambda: output_file.read(SHA256_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort-size", default=DEFAULT_COHORT_SIZE, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--chunksize", default=DEFAULT_CHUNKSIZE, type=int)
    parser.add_argument("--minimum-rows", default=DEFAULT_MINIMUM_ROWS, type=int)
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare_cohort(
                raw_dir=arguments.raw_dir,
                output_path=arguments.output,
                cohort_size=arguments.cohort_size,
                seed=arguments.seed,
                chunksize=arguments.chunksize,
                minimum_rows=arguments.minimum_rows,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
