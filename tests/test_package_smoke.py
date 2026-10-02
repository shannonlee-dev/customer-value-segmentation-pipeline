"""Small installed-package integration checks without competition data."""

import numpy as np
import pandas as pd
import pytest

from customer_value_segmentation.pipeline import DataAnalyzer
from customer_value_segmentation.runtime import discover_runtime


@pytest.fixture
def analyzer(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    pd.DataFrame(
        [["2020-01-01", "a", "001", 0.1, 1], ["2020-01-02", "a", "001", 0.2, 1]],
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    ).to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame(
        [["a", 25, "ACTIVE", "Regularly"]],
        columns=["customer_id", "age", "club_member_status", "fashion_news_frequency"],
    ).to_csv(raw / "customers.csv", index=False)
    pd.DataFrame(
        [["001", "Item", "Group"]], columns=["article_id", "prod_name", "product_group_name"]
    ).to_csv(raw / "articles.csv", index=False)
    context = discover_runtime(
        tmp_path, {"HM_RAW_DATA_DIR": str(raw), "HM_RUNTIME_DIR": str(tmp_path / "runtime")}
    )
    return DataAnalyzer(context, chunksize=1)


@pytest.mark.smoke
def test_installed_pipeline_preserves_rfm_grain(analyzer):
    assert analyzer.load_data()["transaction_rows"] == 2
    rfm = analyzer.calculate_rfm(partition_count=2)
    assert len(rfm) == 1
    assert rfm.loc[0, "frequency"] == 2
    assert rfm.loc[0, "recency"] == 1
    assert np.isclose(rfm.loc[0, "monetary"], 0.3)
