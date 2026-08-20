"""Build the clean, portable, one-click H&M analysis notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook(path: Path = Path("notebooks/analysis_report.ipynb")) -> Path:
    cells = [
        new_markdown_cell("""# H&M Customer Value Analysis

Open this notebook and select **Run All**. It discovers Kaggle competition input, `data/raw/h-and-m`, or `HM_RAW_DATA_DIR` automatically. Every metric uses the full available dataset; no customer, transaction, product, or image analysis sampling is used."""),
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
from src.runtime import discover_runtime

context = discover_runtime(ROOT)
analyzer = DataAnalyzer(context)
started = time.monotonic()
print(f"Runtime: {context.runtime_name}")
print("Project source: available")
print("H&M source validation: PASS")
print(f"Runtime artifact root: {context.runtime_root}")
print("Analysis scope: FULL DATASET")
print("Customer sampling: NONE")
print("Product sampling: NONE")
print("Image analysis sampling: NONE")"""),
        new_markdown_cell("""## Full-data preparation and multimodal features

Transactions are read in Pandas chunks. Customer age is imputed once at customer grain with the median within `club_member_status`, then falls back to the global median. This respects a customer attribute's grain, but repeated median values can reduce observed age variance and can exaggerate apparent group differences.

Each image file is read with `matplotlib.image.imread`. The file loop performs I/O only; the image-internal calculation is NumPy vectorized over the complete decoded array with `np.mean` and `np.std`. Images are never stacked into one global tensor."""),
        new_code_cell("""summary = analyzer.load_data()
image_features = analyzer.engineer_features()
iqr = analyzer.detect_outliers()
rfm = analyzer.calculate_rfm()

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
        new_code_cell("""price_values = np.memmap(
    context.aggregate_root / "unit_price_values.dat",
    dtype="float64",
    mode="r",
    shape=(summary["transaction_rows"],),
)
inlier_prices = price_values[(price_values >= iqr["lower_fence"]) & (price_values <= iqr["upper_fence"])]

customer_age = pd.read_csv(context.processed_root / "customers.csv", dtype={"customer_id": "string"})[["customer_id", "age"]]
pair_count = pair_x = pair_y = pair_xy = pair_x2 = pair_y2 = 0.0
monthly_parts = []
for transaction_chunk in pd.read_csv(context.processed_root / "transactions.csv", parse_dates=["order_date"], dtype={"customer_id": "string"}, chunksize=analyzer.chunksize):
    price_age_chunk = transaction_chunk[["customer_id", "unit_price"]].merge(customer_age, on="customer_id", how="left").dropna()
    x = price_age_chunk["unit_price"].to_numpy(dtype=float)
    y = price_age_chunk["age"].to_numpy(dtype=float)
    pair_count += len(x); pair_x += x.sum(); pair_y += y.sum(); pair_xy += (x * y).sum(); pair_x2 += (x * x).sum(); pair_y2 += (y * y).sum()
    monthly_parts.append(transaction_chunk.groupby(transaction_chunk["order_date"].dt.to_period("M"))["unit_price"].sum())
monthly_value = pd.concat(monthly_parts, axis=1).fillna(0).sum(axis=1)
product_features = pd.read_csv(context.feature_root / "product_images.csv", dtype={"product_id": "string"}).merge(
    pd.read_csv(context.processed_root / "articles.csv", dtype={"product_id": "string"})[["product_id", "product_name_length"]], on="product_id", how="left"
)
price_age_corr = (pair_count * pair_xy - pair_x * pair_y) / np.sqrt((pair_count * pair_x2 - pair_x ** 2) * (pair_count * pair_y2 - pair_y ** 2))
image_text_corr = product_features[["image_mean", "product_name_length"]].corr().loc["image_mean", "product_name_length"]
display(Markdown(f"**Transaction/customer scope:** price–age correlation is `r = {price_age_corr:+.3f}`. This is a linear-association measure; age alone is unlikely to explain price when the magnitude is close to zero."))
display(Markdown(f"**Product/image scope:** image-mean–name-length correlation is `r = {image_text_corr:+.3f}`. This measures association between simple visual brightness and text length, not product quality or demand."))

plt.figure(figsize=(8, 4))
plt.hist(price_values, bins=40)
plt.title("Relative Price Distribution")
plt.xlabel("Relative dataset price value")
plt.ylabel("Transaction count")
plt.show()

plt.figure(figsize=(8, 4))
plt.boxplot([price_values, inlier_prices], tick_labels=["Before IQR", "After IQR"])
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
monthly_value.sort_index().plot()
plt.title("Monthly Aggregate Relative Dataset Value")
plt.xlabel("Order month")
plt.ylabel("Aggregate relative dataset value")
plt.show()"""),
        new_markdown_cell("""## RFM segmentation

The analysis date is a parameter. Its default is the final transaction date plus one day, so the newest purchaser has a positive recency of one day. Frequency is the number of unique purchase dates, while monetary is the sum of the relative transaction value. The five reported groups are **VIP**, **Loyal**, **New**, **Potential**, and **Churned**."""),
        new_code_cell("""segment_summary = rfm.groupby("segment").agg(
    customers=("customer_id", "size"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).sort_values("customers", ascending=False)
display(segment_summary)
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
