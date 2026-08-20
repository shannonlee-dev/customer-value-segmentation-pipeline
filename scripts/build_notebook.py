"""Build the clean, portable, one-click H&M analysis notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook(path: Path = Path("notebooks/analysis_report.ipynb")) -> Path:
    cells = [
        new_code_cell("""import os
from pathlib import Path

BASE = Path("/kaggle/input/notebooks/classichit/codyssey-a-1-1")

candidates = list(BASE.rglob("src/pipeline.py"))

print("찾은 pipeline.py:", candidates)

if not candidates:
    raise RuntimeError("src/pipeline.py를 찾지 못했습니다.")

PROJECT_ROOT = candidates[0].parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

print("PROJECT_ROOT =", PROJECT_ROOT)"""),
        new_markdown_cell("""# H&M Customer Value Analysis

Open this notebook and select **Run All**. It discovers a writable runtime cache, `HM_PRECOMPUTED_DIR`, the attached Kaggle full-data artifact, Kaggle competition input, `data/raw/h-and-m`, or `HM_RAW_DATA_DIR` automatically. Every metric uses the full available dataset; no customer, transaction, product, or image analysis sampling is used.

Large raw-data preparation and image feature extraction are performed by the same `DataAnalyzer` and may reuse a validated **full-data artifact**. If no artifact is available, that same code processes the complete raw H&M data."""),
        new_code_cell("""import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

def find_project_root():
    candidates = [Path.cwd(), *Path.cwd().parents]
    if os.environ.get("PROJECT_ROOT"):
        candidates.append(Path(os.environ["PROJECT_ROOT"]))
    candidates.append(Path("/kaggle/working/customer-value-segmentation-pipeline"))
    for candidate in candidates:
        if (candidate / "src" / "pipeline.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Project source was not found. Attach or copy the repository, then set PROJECT_ROOT if needed.")

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.pipeline import DataAnalyzer
from src.reporting import build_business_insights, summarize_numeric, summarize_rfm_segments
from src.runtime import discover_runtime

context = discover_runtime(ROOT)
analyzer = DataAnalyzer(context)
started = time.monotonic()
print(f"Runtime: {context.runtime_name}")
print(f"Runtime mode: {context.runtime_mode}")
if context.precomputed_root is not None:
    print(f"Precomputed root: {context.precomputed_root}")
print("Project source: available")
print("H&M source validation: PASS")
print(f"Runtime artifact root: {context.runtime_root}")
print("Analysis scope: FULL DATASET")
print("Customer sampling: NONE")
print("Product sampling: NONE")
print("Image analysis sampling: NONE")"""),
        new_markdown_cell("""## Full-data preparation and multimodal features

Transactions are read in Pandas chunks. `transactions`, `customers`, `articles`, and `image_features` remain at their natural grains. This avoids repeating customer and product attributes across tens of millions of transactions; only columns required by a statistic are joined, in chunks where necessary.

Customer age is imputed once at customer grain. The median represents a typical age while being less sensitive than the mean to a skewed distribution, and `club_member_status` provides a reproducible customer grouping. The global median is a fallback when a group has no known ages. This preserves customers but can reduce variance and exaggerate apparent group differences.

Each image file is read with `matplotlib.image.imread`. The file loop performs I/O only; the image-internal calculation is NumPy vectorized over the complete decoded array with `np.mean` and `np.std`. Images are never stacked into one global tensor."""),
        new_code_cell("""summary = analyzer.load_data()
image_features = analyzer.engineer_features()
iqr = analyzer.detect_outliers()
rfm = analyzer.calculate_rfm()
eda = analyzer.prepare_eda_artifacts()
print(analyzer.format_cache_report())

transaction_schema = pd.read_csv(analyzer.transactions_path, nrows=0)
customers = pd.read_csv(analyzer.customers_path, dtype={"customer_id": "string"})
articles = pd.read_csv(analyzer.articles_path, dtype={"product_id": "string"})
inventory = pd.DataFrame([
    ["transactions", "one transaction", f"{summary['transaction_rows']:,} × {len(transaction_schema.columns)}", "date, numeric, IDs", "RFM and time series"],
    ["customers", "one customer", f"{len(customers):,} × {len(customers.columns)}", "numeric, categories, ID", "age and membership"],
    ["articles", "one product", f"{len(articles):,} × {len(articles.columns)}", "text, categories, ID", "text/category features"],
    ["image_features", "one product image", f"{len(image_features):,} × {len(image_features.columns)}", "numeric, status, ID", "full-image Mean/Std"],
], columns=["Source", "Grain", "Shape", "Primary types", "Role"])
display(Markdown("## Dataset Inventory"))
display(inventory)

transaction_preview = pd.read_csv(analyzer.transactions_path, nrows=5).copy()
transaction_preview["customer_id"] = "<masked>"
transaction_preview["product_id"] = "<masked>"
display(transaction_preview.head())
transaction_preview.info()
display(transaction_preview.describe())
print("Tables remain normalized by transaction, customer, product, and image-product grain.")
print("Image array processing: matplotlib imread + full-array NumPy np.mean/np.std")"""),
        new_markdown_cell("""## IQR, correlation, and visual analysis

IQR uses the full price distribution. It is robust to extreme values because it uses the middle 50%, but a naturally right-skewed fashion-price distribution can still label legitimate premium products as outliers. The before/after boxplot below is therefore a diagnostic, not an instruction to delete data.

Correlation A is full transaction/customer scope: price versus customer age. Correlation B is full product/image scope: image mean versus product-name length. A correlation describes linear association, not causation."""),
        new_code_cell("""price_statistics = eda["price_statistics"]
display(pd.DataFrame([price_statistics], index=["unit_price (full transactions)"]).round(6))
distribution_note = "평균이 중앙값보다 높아 오른쪽 꼬리 가능성을 보여준다" if price_statistics["mean"] > price_statistics["median"] else "평균과 중앙값의 관계상 강한 오른쪽 꼬리 근거는 제한적이다"
display(Markdown(
    f"**전체 거래 가격 기술통계:** 평균은 `{price_statistics['mean']:.6f}`, 중앙값은 `{price_statistics['median']:.6f}`, "
    f"표준편차는 `{price_statistics['std']:.6f}`이며 Q1–Q3는 `{price_statistics['q1']:.6f}–{price_statistics['q3']:.6f}`이다. "
    f"{distribution_note}. 따라서 평균만 보지 않고 중앙값과 사분위 범위를 함께 사용해야 한다."
))
price_age_corr = eda["price_age_correlation"]
image_text_corr = eda["image_text_correlation"]
monthly_value = pd.read_csv(eda["monthly_summary_path"])
product_features = image_features.merge(articles[["product_id", "product_name_length"]], on="product_id", how="left")
display(Markdown(f"**Transaction/customer scope:** price–age correlation is `r = {price_age_corr:+.3f}`. This is a linear-association measure; age alone is unlikely to explain price when the magnitude is close to zero."))
display(Markdown(f"**Product/image scope:** image-mean–name-length correlation is `r = {image_text_corr:+.3f}`. This measures association between simple visual brightness and text length, not product quality or demand."))

plt.figure(figsize=(8, 4))
plt.stairs(eda["histogram_counts"], eda["histogram_edges"])
plt.title("Relative Price Distribution")
plt.xlabel("Relative dataset price value")
plt.ylabel("Transaction count")
plt.show()

plt.figure(figsize=(8, 4))
ax = plt.gca()
ax.bxp([eda["boxplot_before"], eda["boxplot_after"]], showfliers=False)
plt.title("Relative Price Before vs After IQR Filtering")
plt.xlabel("IQR treatment state")
plt.ylabel("Relative dataset price value")
plt.show()

plt.figure(figsize=(8, 4))
rfm["segment"].value_counts().sort_index().plot.bar()
plt.title("RFM Segment Customer Counts")
plt.xlabel("RFM segment")
plt.ylabel("Customer count")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(product_features[["image_mean", "image_std", "product_name_length"]].corr(), annot=True, cmap="Blues")
plt.title("Product Image and Text Feature Correlation")
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.show()

plt.figure(figsize=(8, 4))
plt.scatter(rfm["frequency"], rfm["monetary"], alpha=0.15, s=3, rasterized=True)
plt.title("Customer Purchase Frequency vs Monetary Value")
plt.xlabel("Unique purchase dates")
plt.ylabel("Aggregate relative dataset value")
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(monthly_value["order_month"], monthly_value["monetary"])
plt.title("Monthly Aggregate Relative Dataset Value")
plt.xlabel("Order month")
plt.ylabel("Aggregate relative dataset value")
plt.xticks(rotation=45)
plt.show()"""),
        new_markdown_cell("""## RFM segmentation

The analysis date is a parameter. Its default is the final transaction date plus one day, so the newest purchaser has a positive recency of one day. Frequency is the number of unique purchase dates, preventing multiple product rows bought on the same day from inflating purchase frequency. Monetary is the sum of the relative transaction value.

R, F, and M are each divided into four rank-based quantiles. Quartiles avoid inventing currency-specific business thresholds and create comparable 1–4 scores that transfer to another dataset. Lower Recency receives a higher score; higher Frequency and Monetary receive higher scores. `rank(method="first")` makes quantile assignment deterministic when customers share the same value, although customers next to a boundary should not be treated as fundamentally different.

Rules are applied in priority order: customers with R = 4, F = 4, and M = 4 become **VIP**; customers with R >= 3 and F >= 3 become **Loyal**; very recent low-frequency customers become **New**; customers with R = 1 become **Churned**; the remainder become **Potential**. Business validity is assessed below from observed customer share, Monetary share, and mean R/F/M rather than from labels alone."""),
        new_code_cell("""segment_summary = summarize_rfm_segments(rfm)
display(segment_summary.round(4))
business_insights = build_business_insights(segment_summary)
display(Markdown("## Runtime Business Insights"))
display(Markdown(business_insights))
insight_path = context.artifact_root / "business_insights.md"
insight_path.write_text(business_insights, encoding="utf-8")
print("Copy-ready business insights:", insight_path)
display(Markdown("### Final execution summary"))
print("Processed transactions:", summary["transaction_rows"])
print("Processed customers:", summary["customer_rows"])
print("Processed products:", summary["product_rows"])
print("IQR outlier count:", iqr["outlier_count"])
print("Total execution time (seconds):", round(time.monotonic() - started, 1))"""),
    ]
    notebook = new_notebook(cells=cells)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    build_notebook()
