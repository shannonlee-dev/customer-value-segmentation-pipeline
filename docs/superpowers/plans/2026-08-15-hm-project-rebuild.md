# H&M Customer Value Segmentation Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the repository from a clean Git root as a reproducible, license-compliant H&M multimodal EDA and RFM project that passes every mandatory mission and evaluation check.

**Architecture:** A deterministic preparation script selects 500 active H&M customers and joins their complete transaction histories to product, customer, and image metadata without committing source rows. `DataAnalyzer` owns reusable preprocessing, vectorized image features, IQR detection, and RFM segmentation; an executed notebook owns the aggregate report, and focused tests enforce data provenance, privacy, execution, and mission coverage.

**Tech Stack:** Python 3.8+, NumPy, Pandas, Matplotlib, Seaborn, nbformat, nbconvert, Jupyter, Kaggle CLI, standard-library `unittest`

## Global Constraints

- Use the local H&M source under `data/raw/h-and-m/`.
- Select exactly 500 active customers by ascending SHA-256 of `seed + ":" + customer_id`, default seed `42`.
- Retain every available-image transaction for selected customers and record exclusions.
- Produce at least 1,000 rows, 8 columns, and numeric, text/category, date, and image-array modalities.
- Do not invent order IDs, quantity, discounts, missing values, customer behavior, or product attributes.
- Treat H&M `price` as a relative dataset value, not a named real-world currency.
- Never commit raw H&M files, derived row-level files, product images, or Kaggle credentials.
- Use only NumPy, Pandas, Matplotlib, and Seaborn for analysis; do not use OpenCV, Pillow directly, scikit-learn, NLTK, or automated EDA libraries.
- Load JPEGs with Matplotlib, downsample with NumPy slicing, and calculate Mean/Std with stacked-array axis operations.
- Include all six required chart types with titles and axis labels.
- Commit an executed notebook with aggregate/redacted outputs and no error cells.
- Write all commits with author name `shannonlee-dev` and Conventional Commit subjects.
- Preserve the old lineage in `backup/pre-hm-rebuild-20260815` and the old dirty worktree in a named stash.
- Do not push or force-push `origin/main`.

## Target File Map

- `.gitignore`: exclude environments, credentials, raw data, and processed data.
- `requirements.txt`: supported dependency ranges without prohibited libraries.
- `data/README.md`: local layout, acquisition, and redistribution boundary.
- `scripts/prepare_hm_data.py`: validate source files, select customers, join data, and write ignored cohort artifacts.
- `src/pipeline.py`: `DataAnalyzer` and reusable analysis behavior.
- `scripts/build_notebook.py`: deterministic report notebook builder.
- `scripts/verify_notebook.py`: execution and redaction verifier.
- `notebooks/analysis_report.ipynb`: executed aggregate analysis.
- `artifacts/notebook_execution.log`: deterministic execution summary.
- `tests/test_data_preparation.py`: provenance and determinism tests.
- `tests/test_pipeline.py`: preprocessing, image, IQR, and RFM tests.
- `tests/test_notebook.py`: chart, execution, and redaction tests.
- `tests/test_mission_compliance.py`: final repository and README audit.
- `README.md`: workflow, provenance, license, results, insights, and limitations.

---

### Task 1: Preserve the Old Lineage and Initialize the Orphan Worktree

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `data/README.md`
- Create: `docs/superpowers/specs/2026-08-15-hm-project-rebuild-design.md`
- Create: `docs/superpowers/plans/2026-08-15-hm-project-rebuild.md`

**Interfaces:**
- Consumes: approved design at commit `7df4792` and local raw path `/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m`.
- Produces: isolated unborn branch `hm-rebuild` and the first two commits of the new history.

- [ ] **Step 1: Record the recovery point**

Run from the original worktree:

```bash
git status --short --branch
git log --oneline --decorate -n 12
git tag backup/pre-hm-rebuild-20260815 HEAD
git show -s --format='%H %an <%ae> %s' backup/pre-hm-rebuild-20260815
```

Expected: the tag resolves to the approved design commit; the dirty worktree is untouched.

- [ ] **Step 2: Create the orphan worktree**

Use `superpowers:using-git-worktrees` first, then run:

```bash
hm_rebuild_worktree=$(mktemp -d /tmp/hm-project-rebuild.XXXXXX)
git worktree add --orphan -b hm-rebuild "$hm_rebuild_worktree"
git -C "$hm_rebuild_worktree" status --short --branch
```

Expected: `No commits yet on hm-rebuild`.

Use the returned temporary path as the working directory for Tasks 1–6. When a
later shell command needs the variable again, set `hm_rebuild_worktree` to that
exact recorded path before running the command.

- [ ] **Step 3: Create and commit the design root**

Create the approved design and this plan at their exact target paths in the orphan worktree. Verify contents with `cmp`, then commit only those two files:

```bash
cmp docs/superpowers/specs/2026-08-15-hm-project-rebuild-design.md "$hm_rebuild_worktree/docs/superpowers/specs/2026-08-15-hm-project-rebuild-design.md"
cmp docs/superpowers/plans/2026-08-15-hm-project-rebuild.md "$hm_rebuild_worktree/docs/superpowers/plans/2026-08-15-hm-project-rebuild.md"
git -C "$hm_rebuild_worktree" add docs/superpowers/specs/2026-08-15-hm-project-rebuild-design.md docs/superpowers/plans/2026-08-15-hm-project-rebuild.md
git -C "$hm_rebuild_worktree" diff --cached --check
git -C "$hm_rebuild_worktree" -c user.name=shannonlee-dev commit -m "docs(design): define H&M customer analysis"
```

Expected: a parentless root commit containing only the two reviewed documents.

- [ ] **Step 4: Create project boundaries**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints/
.pytest_cache/
.kaggle/
kaggle.json
data/raw/
data/processed/
```

Create `requirements.txt`:

```text
numpy>=1.24,<3
pandas>=2.0,<3
matplotlib>=3.7,<4
seaborn>=0.12,<1
nbformat>=5.9,<6
nbconvert>=7.8,<8
ipykernel>=6.25,<8
kaggle>=1.7,<2
```

Create `data/README.md` with the official data/rules links, Kaggle acceptance and download commands, and this local-only layout:

```text
data/
├── raw/h-and-m/{articles.csv,customers.csv,transactions_train.csv,images/}
└── processed/{hm_customer_cohort.csv,hm_customer_cohort.summary.json}
```

State that neither source nor processed rows may be committed or redistributed.

- [ ] **Step 5: Create the isolated Python environment**

Run inside the orphan worktree:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c 'import numpy, pandas, matplotlib, seaborn, nbformat, nbconvert; print("dependencies: PASS")'
```

Expected: dependency import check prints `dependencies: PASS`; `.venv/` remains ignored.

- [ ] **Step 6: Verify and commit the workspace**

```bash
mkdir -p data/raw/h-and-m data/processed
touch data/raw/h-and-m/should-not-track.csv data/processed/should-not-track.csv
git check-ignore -v data/raw/h-and-m/should-not-track.csv data/processed/should-not-track.csv
rg -n 'opencv|pillow|scikit|nltk|pandas-profiling|sweetviz' requirements.txt
git add .gitignore requirements.txt data/README.md
git diff --cached --check
git -c user.name=shannonlee-dev commit -m "chore(project): initialize H&M analysis workspace"
```

Expected: both test files are ignored, prohibited-library scan has no matches, and the second commit contains only project boundaries.

---

### Task 2: Deterministic H&M Cohort Preparation

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/prepare_hm_data.py`
- Create: `tests/__init__.py`
- Create: `tests/test_data_preparation.py`

**Interfaces:**
- Produces: `stable_customer_ids(customer_ids, cohort_size=500, seed=42) -> list[str]`.
- Produces: `prepare_cohort(raw_dir, output_path, cohort_size=500, seed=42, chunksize=1_000_000, minimum_rows=1000) -> dict`.
- Writes ignored: `hm_customer_cohort.csv` and `hm_customer_cohort.summary.json`.
- Produces relative image paths such as `images/010/0108775015.jpg`.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
class DataPreparationTest(unittest.TestCase):
    def test_stable_customer_ids_is_order_independent(self):
        ids = ["customer-c", "customer-a", "customer-d", "customer-b"]
        first = stable_customer_ids(ids, cohort_size=3, seed=42)
        second = stable_customer_ids(reversed(ids), cohort_size=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_stable_customer_ids_rejects_oversized_cohort(self):
        with self.assertRaisesRegex(ValueError, "active customers"):
            stable_customer_ids(["one"], cohort_size=2, seed=42)
```

- [ ] **Step 2: Run the failing test**

Run: `.venv/bin/python -m unittest tests.test_data_preparation -v`

Expected: import failure for `scripts.prepare_hm_data` or a missing function.

- [ ] **Step 3: Implement stable selection and schema constants**

```python
TRANSACTION_COLUMNS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
ARTICLE_COLUMNS = ["article_id", "prod_name", "product_group_name"]
CUSTOMER_COLUMNS = ["customer_id", "age", "club_member_status", "fashion_news_frequency"]


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
```

Add `_require_columns(frame, required, source_name)` and reject missing files, duplicate article/customer keys, empty inputs, and non-positive chunk sizes.

- [ ] **Step 4: Write failing join and provenance tests**

Build temporary H&M-shaped CSVs with repeated customer purchases, metadata, real missing ages, valid images written by `matplotlib.image.imsave`, and one missing image. Assert:

```python
summary = prepare_cohort(
    self.raw_dir,
    self.output_path,
    cohort_size=4,
    seed=7,
    chunksize=3,
    minimum_rows=4,
)
result = pd.read_csv(self.output_path, dtype={"product_id": "string"})
self.assertEqual(summary["selected_customers"], 4)
self.assertEqual(summary["missing_image_rows"], 1)
self.assertTrue({
    "order_date", "customer_id", "product_id", "product_name", "category",
    "unit_price", "sales_channel_id", "age", "club_member_status",
    "fashion_news_frequency", "image_path",
}.issubset(result.columns))
self.assertTrue(result["image_path"].str.match(r"images/\d{3}/\d{10}\.jpg").all())
self.assertTrue({"quantity", "discount_rate", "order_id"}.isdisjoint(result.columns))
```

- [ ] **Step 5: Implement chunked preparation**

Read the transaction file once for active IDs and once for selected rows:

```python
active_ids = set()
for chunk in pd.read_csv(
    transactions_path,
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
    chunksize=chunksize,
):
    active_ids.update(chunk["customer_id"].dropna().unique())

selected = set(stable_customer_ids(active_ids, cohort_size, seed))
selected_chunks = []
for chunk in pd.read_csv(
    transactions_path,
    usecols=TRANSACTION_COLUMNS,
    dtype={"customer_id": "string", "article_id": "string"},
    chunksize=chunksize,
):
    selected_chunks.append(chunk.loc[chunk["customer_id"].isin(selected)].copy())
```

Concatenate, rename H&M fields to the approved contract, merge metadata with `validate="many_to_one"`, derive relative image paths, check image existence, and exclude missing-image rows. Sort by date/customer/product before writing. The summary contains counts, rates, date range, seed, cohort size, and output SHA-256 but no IDs or row previews.

- [ ] **Step 6: Run focused tests and the real preparation**

```bash
.venv/bin/python -m unittest tests.test_data_preparation -v
HM_RAW_DATA_DIR=/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m
.venv/bin/python scripts/prepare_hm_data.py \
  --raw-dir "$HM_RAW_DATA_DIR" \
  --output data/processed/hm_customer_cohort.csv \
  --cohort-size 500 --seed 42
.venv/bin/python -c 'import json; from pathlib import Path; s=json.loads(Path("data/processed/hm_customer_cohort.summary.json").read_text()); assert s["output_rows"] >= 1000; assert s["output_columns"] >= 8; assert s["selected_customers"] == 500; print(s)'
git check-ignore -v data/processed/hm_customer_cohort.csv data/processed/hm_customer_cohort.summary.json
```

Expected: tests pass; real output has at least 1,000 rows, 8 columns, and exactly 500 customers; both artifacts are ignored.

- [ ] **Step 7: Commit the data feature**

```bash
git add scripts/__init__.py scripts/prepare_hm_data.py tests/__init__.py tests/test_data_preparation.py
git diff --cached --stat
git diff --cached --check
git -c user.name=shannonlee-dev commit -m "feat(data): add deterministic H&M cohort preparation"
git show --check HEAD
```

Expected: implementation and tests are together; no local data artifact is staged.

---

### Task 3: Multimodal Analysis and RFM Pipeline

**Files:**
- Create: `src/__init__.py`
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `DataAnalyzer(data_path, image_root)`.
- Produces: `load_data(date_columns=("order_date",), numeric_columns=("unit_price", "age")) -> pandas.DataFrame`.
- Produces: `handle_missing(column, group_col, strategy="median") -> pandas.DataFrame`.
- Produces: `engineer_features(image_col="image_path", product_col="product_id", text_col="product_name", downsample_step=35) -> pandas.DataFrame`.
- Produces: `detect_outliers(column, threshold=1.5) -> Tuple[pandas.DataFrame, float, float]`.
- Produces: `calculate_rfm(customer_col="customer_id", date_col="order_date", amount_col="unit_price", frequency_mode="unique_dates", analysis_date=None) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing load and imputation tests**

Create a temporary normalized CSV and assert:

```python
analyzer = DataAnalyzer(self.csv_path, image_root=self.raw_dir)
frame = analyzer.load_data()
self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["order_date"]))

frame.loc[:1, "club_member_status"] = "all-missing"
frame.loc[:1, "age"] = np.nan
expected_global = frame["age"].median()
result = analyzer.handle_missing("age", "club_member_status", "median")
self.assertFalse(result["age"].isna().any())
self.assertTrue((result.loc[:1, "age"] == expected_global).all())
```

Also assert `FileNotFoundError`, missing-column `KeyError`, nonnumeric `TypeError`, empty-data `ValueError`, and unsupported-strategy `ValueError`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest tests.test_pipeline -v`

Expected: import failure for `src.pipeline` or missing `DataAnalyzer`.

- [ ] **Step 3: Implement loading and grouped imputation**

Use Python 3.8-compatible annotations from `typing`. Parse dates with `errors="raise"`, convert numeric columns with `errors="raise"`, and implement:

```python
group_values = frame.groupby(group_col, dropna=False)[column].transform(strategy)
global_value = getattr(frame[column], strategy)()
if pd.isna(global_value):
    raise ValueError(f"Column '{column}' has no values available for imputation")
frame[column] = frame[column].fillna(group_values).fillna(global_value)
```

- [ ] **Step 4: Write failing vectorized-image tests**

Write two RGB images with `matplotlib.image.imsave`, reuse one product across rows, and assert exact array statistics:

```python
result = analyzer.engineer_features(downsample_step=2)
expected = mpimg.imread(
    self.raw_dir / "images/001/0010000001.jpg"
)[::2, ::2, :3]
self.assertAlmostEqual(
    result.loc[result["product_id"] == "0010000001", "image_mean"].iloc[0],
    float(expected.mean()),
)
self.assertAlmostEqual(
    result.loc[result["product_id"] == "0010000001", "image_std"].iloc[0],
    float(expected.std()),
)
self.assertEqual(result.loc[0, "product_name_length"], len(result.loc[0, "product_name"]))
```

Add tests rejecting a missing image, non-positive step, inconsistent image shapes, and one product mapped to two paths.

- [ ] **Step 5: Implement unique-image loading and NumPy features**

```python
unique_images = frame[[product_col, image_col]].drop_duplicates()
loaded = [
    mpimg.imread(self.image_root / relative_path)[
        ::downsample_step, ::downsample_step, :3
    ]
    for relative_path in unique_images[image_col]
]
try:
    image_tensor = np.stack(loaded).astype(np.float32, copy=False)
except ValueError as error:
    raise ValueError("All downsampled image arrays must have the same shape") from error

feature_axes = tuple(range(1, image_tensor.ndim))
unique_images["image_mean"] = image_tensor.mean(axis=feature_axes)
unique_images["image_std"] = image_tensor.std(axis=feature_axes)
```

Merge with `validate="many_to_one"` and create `product_name_length` through Pandas string operations. The list comprehension performs file I/O only; feature computation remains vectorized.

- [ ] **Step 6: Write failing IQR and RFM tests**

Use at least eight customers with distinct recency, purchase-date frequency, and spend patterns:

```python
outliers, lower, upper = analyzer.detect_outliers("unit_price", threshold=1.5)
self.assertTrue(
    ((outliers["unit_price"] < lower) | (outliers["unit_price"] > upper)).all()
)

rfm = analyzer.calculate_rfm(frequency_mode="unique_dates")
self.assertTrue({
    "recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment"
}.issubset(rfm.columns))
self.assertEqual(rfm.index.name, "customer_id")
self.assertEqual(rfm["recency"].min(), 1)
self.assertGreaterEqual(rfm["segment"].nunique(), 4)
```

Assert that two items on one date count as one `unique_dates` purchase and two `rows` purchases.

- [ ] **Step 7: Implement IQR and RFM**

IQR uses non-null numeric values and rejects non-positive thresholds. RFM aggregates:

```python
working = frame[[customer_col, date_col, amount_col]].copy()
grouped = working.groupby(customer_col)
rfm = pd.DataFrame({
    "recency": (reference_date - grouped[date_col].max()).dt.days,
    "frequency": (
        grouped[date_col].nunique()
        if frequency_mode == "unique_dates"
        else grouped.size()
    ),
    "monetary": grouped[amount_col].sum(min_count=1),
})
```

Use rank-based `pd.qcut` scores. Apply rules in order: all scores at least 3 → `VIP`; frequency at least 3 → `Loyal`; recency 4 and frequency at most 2 → `New`; recency at most 2 → `Churned`; otherwise → `Potential`.

- [ ] **Step 8: Run tests and commit**

```bash
.venv/bin/python -m unittest tests.test_pipeline -v
.venv/bin/python -m unittest discover -s tests -v
git add src/__init__.py src/pipeline.py tests/test_pipeline.py
git diff --cached --check
git -c user.name=shannonlee-dev commit -m "feat(pipeline): add multimodal preprocessing and RFM"
git show --check HEAD
```

Expected: all available tests pass and the third feature commit contains implementation with its tests.

---

### Task 4: Executed H&M Analysis Notebook

**Files:**
- Create: `scripts/build_notebook.py`
- Create: `scripts/verify_notebook.py`
- Create and execute: `notebooks/analysis_report.ipynb`
- Create: `artifacts/notebook_execution.log`
- Create: `tests/test_notebook.py`

**Interfaces:**
- Consumes: ignored cohort CSV, `HM_RAW_DATA_DIR`, and `DataAnalyzer`.
- Produces: executed notebook with six charts and aggregate/redacted interpretations.
- Produces: `inspect_notebook(path) -> dict` and `format_evidence(path, summary) -> str`.

- [ ] **Step 1: Write failing notebook tests**

Require these source tokens:

```python
required_tokens = [
    "DataAnalyzer", ".head(", ".info(", ".describe(",
    "histplot", "boxplot", "countplot", "heatmap",
    "scatterplot", "lineplot", "calculate_rfm",
]
```

Require markdown tokens `H&M`, `IQR`, `그룹별`, `NumPy`, `RFM`, `표본`, and `price`; integer execution counts for every code cell; zero error outputs; and at least six `image/png` outputs. Scan output text with:

```python
CUSTOMER_ID_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")
PRODUCT_ID_PATTERN = re.compile(r"\b0\d{9}\b")
```

and fail if either matches.

- [ ] **Step 2: Run the failing notebook test**

Run: `.venv/bin/python -m unittest tests.test_notebook -v`

Expected: failure because the rebuilt notebook does not exist.

- [ ] **Step 3: Implement the notebook builder**

Build cells in this exact order:

1. source and non-commercial/redistribution boundary;
2. imports, `HM_RAW_DATA_DIR`, processed path, and analyzer setup;
3. `.head()` structural preview with all values redacted by type;
4. captured `.info()` and aggregate `.describe()`;
5. real age missingness and grouped median imputation;
6. image Mean/Std and product-name length;
7. IQR bounds and before/after boxplot on a copy;
8. descriptive statistics and two numeric correlations;
9. histogram, category bar, correlation heatmap, aggregate scatterplot, and monthly line chart;
10. RFM aggregates and segment-count chart;
11. three numeric findings and limitations.

Use computed `IPython.display.Markdown` prose. Never display product images, complete IDs, raw product names, or raw rows. The preview must call `head()` but replace every value:

```python
raw_head = frame.head()
preview_labels = {
    "order_date": "<date>",
    "customer_id": "<masked-id>",
    "product_id": "<masked-id>",
    "product_name": "<text>",
    "category": "<category>",
    "unit_price": "<numeric>",
    "sales_channel_id": "<numeric>",
    "age": "<numeric-or-missing>",
    "club_member_status": "<category-or-missing>",
    "fashion_news_frequency": "<category-or-missing>",
    "image_path": "<local-image-path>",
}
safe_preview = pd.DataFrame({
    column: [preview_labels[column]] * len(raw_head)
    for column in raw_head.columns
})
display(safe_preview)
```

- [ ] **Step 4: Implement execution/redaction evidence**

`scripts/verify_notebook.py` rejects unexecuted cells, error outputs, fewer than six charts, and ID patterns in textual outputs. Its log includes notebook SHA-256, cell/output/chart/error counts, redaction status, and `status: PASS`. The CLI accepts an optional notebook path and `--no-write` for read-only verification.

- [ ] **Step 5: Build and execute the real notebook**

```bash
HM_RAW_DATA_DIR=/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m
MPLCONFIGDIR=/tmp/hm-matplotlib-cache HM_RAW_DATA_DIR="$HM_RAW_DATA_DIR" \
.venv/bin/python scripts/build_notebook.py
MPLCONFIGDIR=/tmp/hm-matplotlib-cache HM_RAW_DATA_DIR="$HM_RAW_DATA_DIR" \
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/analysis_report.ipynb --ExecutePreprocessor.timeout=600
.venv/bin/python scripts/verify_notebook.py
.venv/bin/python -m unittest tests.test_notebook -v
```

Expected: execution exits `0`, at least six chart outputs are saved, redaction passes, and notebook tests pass.

- [ ] **Step 6: Verify RFM diversity and commit**

Run a read-only local cohort check asserting `rfm["segment"].nunique() >= 4`, then:

```bash
git status --short
git add scripts/build_notebook.py scripts/verify_notebook.py notebooks/analysis_report.ipynb artifacts/notebook_execution.log tests/test_notebook.py
git diff --cached --check
git -c user.name=shannonlee-dev commit -m "feat(analysis): add executed H&M EDA and segmentation report"
git show --check HEAD
```

Expected: no processed CSV or raw image is listed or staged.

---

### Task 5: README Workflow, Provenance, and Business Evidence

**Files:**
- Create: `README.md`
- Create: `tests/test_mission_compliance.py`

**Interfaces:**
- Consumes: real aggregate metrics from the executed notebook and RFM output.
- Produces: public reproduction guide and final static compliance audit.

- [ ] **Step 1: Write failing README and repository-boundary tests**

Require these README terms:

```python
required_terms = [
    "H&M Personalized Fashion Recommendations",
    "데이터를 찾은 방법",
    "선택한 이유",
    "라이선스 및 사용 조건",
    "500",
    "SHA-256",
    "NumPy",
    "IQR",
    "그룹별",
    "RFM",
    "비즈니스 인사이트",
    "추가로 필요한 데이터",
    "반증",
    "A/B",
    "연속 메모리",
    "병목",
    "이탈",
]
```

Use `git ls-files` and fail if a tracked path starts with `data/raw/` or `data/processed/`, ends with `.jpg`, or equals `kaggle.json`. Assert the repository has neither `generate_sample_data.py` nor tracked `ecommerce_transactions.csv`.

- [ ] **Step 2: Run the failing compliance test**

Run: `.venv/bin/python -m unittest tests.test_mission_compliance -v`

Expected: failure because the rebuilt root README does not exist.

- [ ] **Step 3: Extract exact aggregate evidence**

Run this read-only command and use its output verbatim:

```bash
HM_RAW_DATA_DIR=/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m \
.venv/bin/python - <<'PY'
import pandas as pd
from src.pipeline import DataAnalyzer

analyzer = DataAnalyzer(
    "data/processed/hm_customer_cohort.csv",
    image_root="/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m",
)
frame = analyzer.load_data()
missing_age = int(frame["age"].isna().sum())
analyzer.handle_missing("age", "club_member_status", "median")
analyzer.engineer_features()
outliers, lower, upper = analyzer.detect_outliers("unit_price", 1.5)
rfm = analyzer.calculate_rfm()
profiles = rfm.groupby("segment").agg(
    customers=("recency", "size"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
)
profiles["monetary_share"] = (
    rfm.groupby("segment")["monetary"].sum() / rfm["monetary"].sum()
)
print("shape", frame.shape)
print("date_range", frame["order_date"].min(), frame["order_date"].max())
print("missing_age", missing_age)
print("iqr", len(outliers), lower, upper)
print("correlations", frame[[
    "unit_price", "age", "image_mean", "image_std", "product_name_length"
]].corr().round(4))
print(profiles.round(4))
PY
```

- [ ] **Step 4: Write README as a method-first narrative**

Write concise Korean sections in this order:

1. project goal and mission mapping;
2. **데이터를 찾은 방법**: screen Kaggle, AI Hub, UCI Online Retail, Olist, and image-only sources against transaction, customer/date, text/category, and image requirements;
3. **선택한 이유**: one natural `article_id` join across H&M transaction, product, customer, and image data;
4. **라이선스 및 사용 조건**: official links, non-commercial/academic boundary, acceptance requirement, no redistribution, and no CC/MIT claim;
5. exact source mapping and deterministic 500-customer selection;
6. Kaggle download, local tree, cohort preparation, notebook, and test commands;
7. NumPy slicing/stacking, age imputation, IQR, and unique-date RFM rationale;
8. exact aggregate evidence from Step 3;
9. three numbered insights, each with numeric evidence, a target/action/expected direction, and an A/B holdout or additional-data falsification condition;
10. IQR skew limitations, imputation variance shrinkage, sample bias, normalized price, absent order ID, image I/O/memory bottlenecks, and a future 90-day churn target with candidate features.

Do not include any customer ID, product ID, product name, source row, or image.

- [ ] **Step 5: Run documentation and aggregate tests**

```bash
.venv/bin/python -m unittest tests.test_mission_compliance -v
.venv/bin/python -m unittest discover -s tests -v
rg -n "데이터를 찾은 방법|선택한 이유|라이선스 및 사용 조건|IQR|분산|RFM|연속 메모리|A/B|홀드아웃|이탈|반증" README.md
```

Expected: all tests pass and every required method/limitation term appears in context.

- [ ] **Step 6: Commit the documentation feature**

```bash
git add README.md tests/test_mission_compliance.py
git diff --cached --check
git -c user.name=shannonlee-dev commit -m "docs(readme): document provenance, workflow, and insights"
git show --check HEAD
```

---

### Task 6: Aggregate Verification and Local Main Replacement

**Files:**
- Modify no project files unless verification identifies a concrete defect.
- Preserve locally: ignored `data/raw/h-and-m/` and `data/processed/`.

**Interfaces:**
- Consumes: completed six-commit `hm-rebuild` history.
- Produces: local `main` on the verified new lineage, with old lineage and worktree recoverable.

- [ ] **Step 1: Run fresh aggregate verification**

Use `superpowers:verification-before-completion`, then run in the orphan worktree:

```bash
.venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/hm-matplotlib-cache \
HM_RAW_DATA_DIR=/home/shannon/__dev/cody/customer-value-segmentation-pipeline/data/raw/h-and-m \
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/analysis_report.ipynb \
  --output hm-analysis-report-verified.ipynb \
  --output-dir /tmp \
  --ExecutePreprocessor.timeout=600
.venv/bin/python scripts/verify_notebook.py \
  /tmp/hm-analysis-report-verified.ipynb --no-write
```

Expected: all tests pass; notebook exits `0`; verification reports at least six charts, zero errors, and redaction PASS.

- [ ] **Step 2: Audit history and tracked files**

```bash
git log --reverse --format='%h %an <%ae> %s'
git status --short --branch
git ls-files | sort
git ls-files data/raw data/processed '*.jpg' kaggle.json
git fsck --no-dangling
```

Expected: exactly six approved commits, every author name `shannonlee-dev`, a clean worktree, no H&M data tracked, and valid Git objects.

- [ ] **Step 3: Preserve the original dirty worktree**

Return to the original worktree:

```bash
git status --short --branch
git stash push --include-untracked -m "backup/pre-hm-rebuild-20260815-worktree"
git stash list --format='%gd %s' | sed -n '1,5p'
git status --short --branch
```

Expected: original worktree is clean; named stash is visible; ignored raw data remains.

- [ ] **Step 4: Preserve local processed artifacts**

```bash
mkdir -p data/processed
cp "$hm_rebuild_worktree/data/processed/hm_customer_cohort.csv" data/processed/
cp "$hm_rebuild_worktree/data/processed/hm_customer_cohort.summary.json" data/processed/
sha256sum "$hm_rebuild_worktree/data/processed/hm_customer_cohort.csv" \
  data/processed/hm_customer_cohort.csv
```

Expected: both CSV hashes match.

- [ ] **Step 5: Remove the worktree and replace local main**

```bash
git worktree remove "$hm_rebuild_worktree"
git switch hm-rebuild
git branch -M main
git status --short --branch
git log --reverse --format='%h %an <%ae> %s'
```

Expected: original workspace uses the new six-commit `main`; backup tag and stash remain; `origin/main` is unchanged.

- [ ] **Step 6: Verify switched workspace and recovery paths**

```bash
git show -s --format='%H %s' backup/pre-hm-rebuild-20260815
git stash list --format='%gd %s' | sed -n '1,5p'
git rev-parse main
git rev-parse origin/main
git status --short --branch
.venv/bin/python -m unittest discover -s tests -v
```

Expected: backup tag and stash resolve, local and origin `main` differ, the new local worktree is clean, and tests pass. Do not push.
