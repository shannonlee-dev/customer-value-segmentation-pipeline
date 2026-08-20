# Stratified Customer Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run the H&M analysis on a reproducible, 20,000-customer proportional stratified sample that preserves each selected customer's complete transaction history.

**Architecture:** `DataAnalyzer` samples customers by `club_member_status` before streaming the source transactions. The selected IDs filter every transaction chunk, transaction-linked product IDs constrain the article and image tables, and all downstream calculations consume only those cached tables. Constructor options make size and seed explicit while fixtures smaller than the requested sample retain every customer.

**Tech Stack:** Python 3, Pandas, NumPy, Matplotlib, nbformat, unittest.

**Spec:** `docs/superpowers/specs/2026-08-20-stratified-customer-sampling-design.md`

## Global Constraints

- Default sample: exactly 20,000 customers, seed 42, stratified by `club_member_status`, including missing status as a stratum.
- Retain every source transaction for each selected customer; never sample transaction rows.
- Retain only products referenced by retained transactions and analyze only their images.
- Keep dependencies limited to NumPy, Pandas, Matplotlib, and Seaborn for analysis.
- Preserve existing unrelated `.gitignore` and `docs/full-data-portable-notebook-refactor-spec.md` changes outside the feature commit.
- Do not merge to `main`; push only `feat/stratified-customer-sampling` after final verification.

---

### Task 1: Customer sampling contract

**Files:**
- Modify: `tests/test_full_pipeline.py`
- Modify: `src/pipeline.py`

**Interfaces:**
- Produces: `DataAnalyzer(context: RuntimeContext, chunksize: int = DEFAULT_CHUNKSIZE, customer_sample_size: int = 20_000, sampling_seed: int = 42)`.
- Produces: `DataAnalyzer._select_customer_sample(customers: pd.DataFrame) -> pd.DataFrame`.

- [x] **Step 1: Write the failing test**

```python
def test_customer_sampling_is_seeded_proportional_and_exact(self):
    first = self.analyzer(customer_sample_size=5, sampling_seed=7)
    second = self.analyzer(customer_sample_size=5, sampling_seed=7)
    first.load_data(); second.load_data()
    customers = pd.read_csv(first.customers_path)
    self.assertEqual(len(customers), 5)
    self.assertEqual(set(customers["club_member_status"]), {"ACTIVE", "PRE-CREATE"})
    self.assertEqual(
        pd.read_csv(first.customers_path)["customer_id"].tolist(),
        pd.read_csv(second.customers_path)["customer_id"].tolist(),
    )
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_full_pipeline.FullPipelineTests.test_customer_sampling_is_seeded_proportional_and_exact -v`

Expected: FAIL because the existing constructor does not accept sampling parameters and `load_data()` caches all customers.

- [x] **Step 3: Write minimal implementation**

```python
DEFAULT_CUSTOMER_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLING_SEED = 42

def _select_customer_sample(self, customers: pd.DataFrame) -> pd.DataFrame:
    if self.customer_sample_size <= 0:
        raise ValueError("customer_sample_size must be positive")
    if len(customers) <= self.customer_sample_size:
        return customers.copy()
    # Allocate membership-stratum quotas with largest remainder, then sample deterministically.
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_full_pipeline -v`

Expected: PASS, including existing full-pipeline behaviors.

- [x] **Step 5: Commit**

```bash
git add tests/test_full_pipeline.py src/pipeline.py
git commit -m "feat: sample customers proportionally"
```

### Task 2: Transaction-linked article and image scope

**Files:**
- Modify: `tests/test_full_pipeline.py`
- Modify: `src/pipeline.py`

**Interfaces:**
- Consumes: sampled customer IDs produced by `_select_customer_sample`.
- Produces: `load_data()` summary with `source_customer_rows`, `customer_rows`, `transaction_rows`, and `product_rows` for the selected scope.

- [x] **Step 1: Write the failing test**

```python
def test_sampled_customers_keep_all_transactions_and_only_used_products(self):
    analyzer = self.analyzer(customer_sample_size=2, sampling_seed=3)
    summary = analyzer.load_data()
    selected = set(pd.read_csv(analyzer.customers_path)["customer_id"])
    cached = pd.read_csv(analyzer.transactions_path)
    self.assertEqual(set(cached["customer_id"]), selected)
    self.assertEqual(summary["source_customer_rows"], 4)
    self.assertEqual(
        set(pd.read_csv(analyzer.articles_path)["product_id"]),
        set(cached["product_id"]),
    )
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_full_pipeline.FullPipelineTests.test_sampled_customers_keep_all_transactions_and_only_used_products -v`

Expected: FAIL because the current loader writes all transaction chunks and all article records.

- [x] **Step 3: Write minimal implementation**

```python
selected_customer_ids = set(self.customers[CUSTOMER_ID_COLUMN])
selected_product_ids: set[str] = set()
for chunk in pd.read_csv(raw / RAW_TRANSACTIONS_FILENAME, usecols=RAW_TRANSACTION_COLUMNS, dtype=RAW_TRANSACTION_DTYPES, chunksize=self.chunksize):
    selected = chunk.loc[chunk[CUSTOMER_ID_COLUMN].isin(selected_customer_ids)]
    selected_product_ids.update(selected[RAW_ARTICLE_ID_COLUMN].dropna())
    selected.rename(columns=TRANSACTION_RENAMES).to_csv(self.transactions_path, mode=mode, header=first_chunk, index=False)
self.articles = normalized_articles.loc[
    normalized_articles[PRODUCT_ID_COLUMN].isin(selected_product_ids)
].copy()
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_full_pipeline -v`

Expected: PASS; feature engineering and RFM use only sampled cache tables.

- [x] **Step 5: Commit**

```bash
git add tests/test_full_pipeline.py src/pipeline.py
git commit -m "feat: restrict artifacts to sampled customers"
```

### Task 3: Notebook and README sampling contract

**Files:**
- Modify: `tests/test_notebook.py`
- Modify: `tests/test_repository_contract.py`
- Modify: `scripts/build_notebook.py`
- Modify: `README.md`
- Modify: `notebooks/analysis_report.ipynb`

**Interfaces:**
- Consumes: sampling metadata returned by `DataAnalyzer.load_data()`.
- Produces: notebook source that labels the scope and README guidance that explains selection, full-history retention, image scope, seed, and the future stability check.

- [x] **Step 1: Write the failing tests**

```python
def test_generated_notebook_describes_customer_stratified_sampling(self):
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    self.assertIn("PROPORTIONAL STRATIFIED CUSTOMER SAMPLE", source)
    self.assertIn("20,000", source)

def test_readme_documents_customer_level_sampling(self):
    readme = Path("README.md").read_text(encoding="utf-8")
    self.assertIn("고객 단위 비례 층화표본추출", readme)
    self.assertIn("20,000", readme)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_notebook tests.test_repository_contract -v`

Expected: FAIL because the notebook and README currently claim full-dataset processing.

- [x] **Step 3: Write minimal documentation implementation**

```python
print("Analysis scope: PROPORTIONAL STRATIFIED CUSTOMER SAMPLE")
print(f"Sampled customers: {summary['customer_rows']:,} / {summary['source_customer_rows']:,}")
print(f"Customer sampling seed: {analyzer.sampling_seed}")
```

Replace full-data README claims with the approved Korean explanation, mention the default `20,000`, fixed seed, retained full transaction histories, transaction-linked product/image scope, and the future 5,000/10,000/20,000 stability comparison.

- [x] **Step 4: Regenerate and run source tests**

Run: `.venv/bin/python scripts/build_notebook.py && .venv/bin/python -m unittest tests.test_notebook tests.test_repository_contract -v`

Expected: PASS and the tracked notebook matches its generator.

- [x] **Step 5: Commit**

```bash
git add README.md scripts/build_notebook.py notebooks/analysis_report.ipynb tests/test_notebook.py tests/test_repository_contract.py
git commit -m "docs: explain stratified customer sampling"
```

### Task 4: Full verification and feature publication

**Files:**
- Modify: `docs/superpowers/plans/2026-08-20-stratified-customer-sampling.md`

- [x] **Step 1: Run the full test suite**

Run: `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest discover -s tests -v`

Expected: PASS with no test failures.

- [x] **Step 2: Run the synthetic executed-notebook check**

Run: `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/synthetic_e2e.py`

Expected: the notebook executes and `scripts/verify_notebook.py` reports PASS.

- [x] **Step 3: Inspect staged scope and commit the plan**

```bash
git status --short
git diff --check
git add docs/superpowers/plans/2026-08-20-stratified-customer-sampling.md
git commit -m "docs: add sampling implementation plan"
```

- [x] **Step 4: Push the feature branch only**

Run: `git push -u origin feat/stratified-customer-sampling`

Expected: remote branch is created or updated; do not merge to `main` and do not create a pull request.
