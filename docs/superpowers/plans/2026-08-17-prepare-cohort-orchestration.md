# Prepare Cohort Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `prepare_cohort` explicitly orchestrate independently named data-preparation stages without changing its public behavior.

**Architecture:** Keep `prepare_cohort` as the sole public workflow entry point. Extract its source validation, metadata loading, customer selection, transaction loading, enrichment, image filtering, and output writing into private functions with explicit inputs and return values. Preserve the current constants module and reuse existing low-level validation, hashing, image-path, and digest helpers.

**Tech Stack:** Python 3.12, pandas, unittest.

## Global Constraints

- Keep the `prepare_cohort` signature, CLI options, output CSV, summary JSON schema, error behavior, and deterministic cohort selection unchanged.
- Keep support for both `python scripts/prepare_hm_data.py` and `python -m scripts.prepare_hm_data`.
- Do not move unrelated `src` pipeline constants or alter notebook artifacts.

---

### Task 1: Establish the behavior guard

**Files:**
- Modify: `tests/test_data_preparation.py`
- Test: `tests/test_data_preparation.py`

**Interfaces:**
- Consumes: `prepare_cohort(raw_dir, output_path, cohort_size, seed, chunksize, minimum_rows) -> dict`
- Produces: a regression test that verifies the complete cohort output and summary contract through the public API.

- [ ] **Step 1: Add an orchestration contract test**

Add this test beside the existing cohort integration test. It uses the real public API rather than mocks and proves that the refactor continues to coordinate all required stages.

```python
def test_prepare_cohort_returns_the_written_output_summary(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
        self.raw_dir = Path(temporary_directory)
        self.output_path = self.raw_dir / "cohort.csv"
        (self.raw_dir / "images" / "010").mkdir(parents=True)
        self._write_hm_shaped_source()

        summary = prepare_cohort(
            self.raw_dir,
            self.output_path,
            cohort_size=4,
            seed=7,
            chunksize=3,
            minimum_rows=4,
        )

        self.assertEqual(summary["output_rows"], len(pd.read_csv(self.output_path)))
        self.assertEqual(summary["output_columns"], 11)
        self.assertTrue(self.output_path.with_suffix(".summary.json").is_file())
```

- [ ] **Step 2: Run the focused test before refactoring**

Run: `.venv/bin/python -m unittest tests.test_data_preparation.DataPreparationTest.test_prepare_cohort_returns_the_written_output_summary -v`

Expected: PASS. The existing public behavior is the guard for the structural extraction.

- [ ] **Step 3: Commit the regression guard**

```bash
git add tests/test_data_preparation.py
git commit -m "test(data): cover cohort output summary contract"
```

### Task 2: Extract source and selection stages

**Files:**
- Modify: `scripts/prepare_hm_data.py`
- Test: `tests/test_data_preparation.py`

**Interfaces:**
- Consumes: raw directory, `chunksize`, `cohort_size`, and `seed`.
- Produces:
  - `_resolve_source_paths(raw_dir) -> tuple[Path, Path, Path, Path, Path]`
  - `_load_metadata_sources(articles_path, customers_path) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `_select_cohort_customer_ids(transactions_path, cohort_size, seed, chunksize) -> set`
  - `_load_selected_transactions(transactions_path, selected_customer_ids, chunksize) -> list[pd.DataFrame]`

- [ ] **Step 1: Add the private stage functions**

```python
def _resolve_source_paths(raw_dir):
    raw_dir = Path(raw_dir)
    transactions_path = raw_dir / TRANSACTIONS_FILENAME
    articles_path = raw_dir / ARTICLES_FILENAME
    customers_path = raw_dir / CUSTOMERS_FILENAME
    images_dir = raw_dir / IMAGES_DIRECTORY
    for source_path in (transactions_path, articles_path, customers_path, images_dir):
        _require_existing_path(source_path)
    return raw_dir, transactions_path, articles_path, customers_path, images_dir
```

Move the existing metadata reads, active-customer scan, deterministic selection, and selected-chunk loading into the remaining named functions without changing their existing validation or error messages.

- [ ] **Step 2: Run the data-preparation test module**

Run: `.venv/bin/python -m unittest tests.test_data_preparation -v`

Expected: PASS.

- [ ] **Step 3: Commit the extracted input stages**

```bash
git add scripts/prepare_hm_data.py tests/test_data_preparation.py
git commit -m "refactor(data): extract cohort input stages"
```

### Task 3: Extract enrichment and persistence stages

**Files:**
- Modify: `scripts/prepare_hm_data.py`
- Test: `tests/test_data_preparation.py`

**Interfaces:**
- Consumes: selected transaction chunks, metadata data frames, raw directory, output path, and selection metrics.
- Produces:
  - `_enrich_transactions(selected_chunks, articles, customers) -> pd.DataFrame`
  - `_filter_available_images(cohort, raw_dir, minimum_rows) -> tuple[pd.DataFrame, int]`
  - `_write_cohort_and_summary(cohort, output_path, active_customer_count, selected_customer_count, selected_transaction_count, missing_image_rows, seed, cohort_size, minimum_rows) -> dict`

- [ ] **Step 1: Extract enrichment logic**

Move the existing concatenation, renaming, metadata joins, merge validation, and indicator cleanup into `_enrich_transactions`. Preserve `many_to_one` validation and the missing-metadata exception.

- [ ] **Step 2: Extract image filtering and output writing**

Move date coercion, generated image paths, image existence filtering, minimum-row validation, sorting, CSV writing, summary construction, JSON writing, and SHA-256 calculation inputs into the two remaining stage functions. Return the exact current summary dictionary.

- [ ] **Step 3: Reduce `prepare_cohort` to orchestration**

Make the public function read as a linear workflow:

```python
raw_dir, transactions_path, articles_path, customers_path, _ = _resolve_source_paths(raw_dir)
articles, customers = _load_metadata_sources(articles_path, customers_path)
selected_customer_ids = _select_cohort_customer_ids(
    transactions_path, cohort_size, seed, chunksize
)
selected_chunks = _load_selected_transactions(
    transactions_path, selected_customer_ids, chunksize
)
cohort = _enrich_transactions(selected_chunks, articles, customers)
cohort, missing_image_rows = _filter_available_images(cohort, raw_dir, minimum_rows)
return _write_cohort_and_summary(...)
```

- [ ] **Step 4: Run targeted and full validation**

Run:

```bash
.venv/bin/python -m compileall -q scripts tests
.venv/bin/python -m unittest tests.test_data_preparation -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 5: Commit the completed orchestration refactor**

```bash
git add scripts/prepare_hm_data.py tests/test_data_preparation.py
git commit -m "refactor(data): orchestrate cohort preparation stages"
```
