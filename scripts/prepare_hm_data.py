"""Prepare a deterministic, image-backed H&M customer cohort."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd


TRANSACTION_COLUMNS = [
    "t_dat",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
]
ARTICLE_COLUMNS = ["article_id", "prod_name", "product_group_name"]
CUSTOMER_COLUMNS = [
    "customer_id",
    "age",
    "club_member_status",
    "fashion_news_frequency",
]


def _customer_digest(customer_id, seed):
    return hashlib.sha256(f"{seed}:{customer_id}".encode("utf-8")).digest()


def stable_customer_ids(customer_ids, cohort_size=500, seed=42):
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


def _require_columns(frame, required, source_name):
    missing = set(required).difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{source_name} is missing required columns: {names}")


def _require_existing_path(path):
    if not path.exists():
        raise FileNotFoundError(f"Required H&M source is missing: {path}")


def _read_metadata(path, columns, key):
    frame = pd.read_csv(path, dtype={key: "string"})
    _require_columns(frame, columns, path.name)
    frame = frame.loc[:, columns].copy()
    if frame.empty:
        raise ValueError(f"{path.name} must not be empty")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate or missing {key} values")
    return frame


def _image_path(article_id):
    article_id = str(article_id)
    return f"images/{article_id[:3]}/{article_id}.jpg"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as output_file:
        for block in iter(lambda: output_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cohort(
    raw_dir,
    output_path,
    cohort_size=500,
    seed=42,
    chunksize=1_000_000,
    minimum_rows=1000,
):
    """Build and write a deterministic, metadata-joined H&M customer cohort."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    transactions_path = raw_dir / "transactions_train.csv"
    articles_path = raw_dir / "articles.csv"
    customers_path = raw_dir / "customers.csv"
    images_dir = raw_dir / "images"
    for source_path in (transactions_path, articles_path, customers_path, images_dir):
        _require_existing_path(source_path)

    articles = _read_metadata(articles_path, ARTICLE_COLUMNS, "article_id")
    customers = _read_metadata(customers_path, CUSTOMER_COLUMNS, "customer_id")

    active_ids = set()
    for chunk in pd.read_csv(
        transactions_path,
        usecols=["customer_id"],
        dtype={"customer_id": "string"},
        chunksize=chunksize,
    ):
        _require_columns(chunk, ["customer_id"], transactions_path.name)
        active_ids.update(chunk["customer_id"].dropna().unique())
    if not active_ids:
        raise ValueError("transactions_train.csv must not be empty")

    selected = set(stable_customer_ids(active_ids, cohort_size, seed))
    selected_chunks = []
    for chunk in pd.read_csv(
        transactions_path,
        usecols=TRANSACTION_COLUMNS,
        dtype={"customer_id": "string", "article_id": "string"},
        chunksize=chunksize,
    ):
        _require_columns(chunk, TRANSACTION_COLUMNS, transactions_path.name)
        selected_chunks.append(chunk.loc[chunk["customer_id"].isin(selected)].copy())
    selected_chunks = [chunk for chunk in selected_chunks if not chunk.empty]
    if not selected_chunks:
        raise ValueError("No transactions found for selected customers")

    result = pd.concat(selected_chunks, ignore_index=True)
    result = result.rename(
        columns={
            "t_dat": "order_date",
            "article_id": "product_id",
            "price": "unit_price",
        }
    )
    articles = articles.rename(
        columns={
            "article_id": "product_id",
            "prod_name": "product_name",
            "product_group_name": "category",
        }
    )
    result = result.merge(
        articles,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator="_article_metadata",
    )
    result = result.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator="_customer_metadata",
    )
    if (
        (result["_article_metadata"] != "both").any()
        or (result["_customer_metadata"] != "both").any()
    ):
        raise ValueError("Selected transactions are missing article or customer metadata")
    result = result.drop(columns=["_article_metadata", "_customer_metadata"])

    result["order_date"] = pd.to_datetime(result["order_date"], errors="raise")
    result["image_path"] = result["product_id"].map(_image_path)
    image_exists = result["image_path"].map(lambda path: (raw_dir / path).is_file())
    missing_image_rows = int((~image_exists).sum())
    result = result.loc[image_exists].copy()
    if len(result) < minimum_rows:
        raise ValueError(
            f"Cohort has {len(result)} available-image rows; minimum_rows is {minimum_rows}"
        )

    output_columns = [
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
    ]
    result = result.loc[:, output_columns].sort_values(
        ["order_date", "customer_id", "product_id"], kind="stable"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    summary = {
        "active_customers": len(active_ids),
        "selected_customers": len(selected),
        "selected_transaction_rows": int(sum(len(chunk) for chunk in selected_chunks)),
        "missing_image_rows": missing_image_rows,
        "missing_image_rate": missing_image_rows / sum(len(chunk) for chunk in selected_chunks),
        "output_rows": len(result),
        "output_columns": len(output_columns),
        "date_range": {
            "start": result["order_date"].min().strftime("%Y-%m-%d"),
            "end": result["order_date"].max().strftime("%Y-%m-%d"),
        },
        "seed": seed,
        "cohort_size": cohort_size,
        "minimum_rows": minimum_rows,
        "output_sha256": _sha256(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort-size", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--chunksize", default=1_000_000, type=int)
    parser.add_argument("--minimum-rows", default=1000, type=int)
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
