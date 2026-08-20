# Precomputed Kaggle Artifact Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse validated full-data runtime or read-only Kaggle precomputed artifacts so notebook Run All avoids raw transaction, image, and RFM recomputation by default.

**Architecture:** `RuntimeContext` will distinguish writable runtime output from optional read-only precomputed input. `DataAnalyzer` will use a small cache-discovery helper that checks the writable artifact first, then recursively discovers and validates a precomputed artifact, and finally performs the existing full-data computation. EDA summaries will be stored as exact aggregates so charts do not rescan normalized transactions on a reuse path.

**Tech Stack:** Python stdlib, NumPy, Pandas, Matplotlib, Seaborn, unittest, nbformat.

**Spec:** `docs/superpowers/specs/2026-08-20-cache-aware-full-data-run-all-design.md`

## Global Constraints

- Use one `DataAnalyzer` implementation for Kaggle and local execution.
- Reuse only full-data artifacts; do not introduce analysis sampling.
- Use only NumPy, Pandas, Matplotlib, Seaborn, and Python standard library.
- Do not use Pillow, OpenCV, scikit-learn, DuckDB, PyArrow, or Parquet.
- Treat `HM_PRECOMPUTED_DIR` and the default Kaggle precomputed root as read-only.
- Prefer writable runtime artifacts, then `HM_PRECOMPUTED_DIR`, then the default Kaggle precomputed root, then raw full-data computation.
- Preserve existing `DataAnalyzer` public methods and add keyword-only `force=False` defaults.

---

### Task 1: Runtime context and precomputed discovery

**Files:**
- Modify: `src/runtime.py`
- Modify: `tests/test_runtime.py`

**Interfaces:**
- Produces: `RuntimeContext.precomputed_root: Path | None` and `RuntimeContext.runtime_mode: str`.
- Produces: `discover_runtime(..., environ=...)` which resolves raw data independently of optional precomputed input.

- [ ] **Step 1: Write failing runtime discovery tests**

```python
def test_precomputed_environment_takes_precedence_over_default_root(self):
    context = discover_runtime(..., environ={"HM_PRECOMPUTED_DIR": str(explicit)})
    self.assertEqual(context.precomputed_root, explicit)
    self.assertEqual(context.runtime_mode, "precomputed")

def test_precomputed_root_does_not_become_runtime_output(self):
    context = discover_runtime(..., environ={"HM_PRECOMPUTED_DIR": str(read_only_input)})
    self.assertNotEqual(context.runtime_root, read_only_input)
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `python -m unittest tests.test_runtime.RuntimeDiscoveryTests -v`

Expected: FAIL because `RuntimeContext` has no precomputed properties.

- [ ] **Step 3: Implement minimal discovery behavior**

```python
DEFAULT_PRECOMPUTED_ROOT = Path("/kaggle/input/notebooks/classichit/notebook9c33091b06/customer-value-segmentation-pipeline")

precomputed_root = _resolve_precomputed_root(environment, DEFAULT_PRECOMPUTED_ROOT)
runtime_mode = "precomputed" if precomputed_root else runtime_name
```

Keep creation calls limited to `runtime_root` descendants. Do not call
`mkdir`, `unlink`, or any write API for `precomputed_root`.

- [ ] **Step 4: Run the targeted tests to verify success**

Run: `python -m unittest tests.test_runtime.RuntimeDiscoveryTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/runtime.py tests/test_runtime.py
git commit -m "feat(runtime): discover read-only precomputed artifacts"
```

### Task 2: Validated artifact discovery and reuse status

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_full_pipeline.py`

**Interfaces:**
- Produces: `DataAnalyzer.artifact_status: dict[str, str]` with `REUSED` or `COMPUTED` values.
- Produces: cache helpers accepting expected required columns and runtime/precomputed paths.
- Consumes: `RuntimeContext.precomputed_root` from Task 1.

- [ ] **Step 1: Write failing cache-discovery tests**

```python
def test_precomputed_processed_tables_are_reused_without_raw_transaction_read(self):
    analyzer = self.analyzer_with_precomputed_tables()
    (self.raw / "transactions_train.csv").unlink()
    summary = analyzer.load_data()
    self.assertEqual(summary["transaction_rows"], 6)
    self.assertEqual(analyzer.artifact_status["transactions"], "REUSED")

def test_invalid_precomputed_schema_is_reported_and_falls_back(self):
    analyzer = self.analyzer_with_invalid_precomputed_transactions()
    analyzer.load_data()
    self.assertIn("missing required columns", "\n".join(analyzer.cache_messages))
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `python -m unittest tests.test_full_pipeline.FullPipelineTests -v`

Expected: FAIL because precomputed discovery and status reporting do not exist.

- [ ] **Step 3: Implement schema-validated reusable artifacts**

```python
def _find_reusable_csv(self, artifact_name, expected_filename, required_columns):
    for root, source in self._reuse_sources():
        for candidate in root.rglob(expected_filename):
            if self._csv_has_required_columns(candidate, required_columns):
                self._record_status(artifact_name, "REUSED", source, candidate)
                return candidate
            self._record_rejection(artifact_name, candidate, "missing required columns")
    return None
```

Read headers with `pd.read_csv(..., nrows=0)` and reject unreadable, empty, or
schema-mismatched files. Search only the optional precomputed root; check the
known writable runtime path directly before it. `load_data(force=False)` must
copy a reusable external artifact into the writable runtime only when later
pipeline APIs require the stable local path; otherwise it may read directly.

- [ ] **Step 4: Run the targeted tests to verify success**

Run: `python -m unittest tests.test_full_pipeline.FullPipelineTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_full_pipeline.py
git commit -m "feat(cache): reuse validated processed artifacts"
```

### Task 3: Image, IQR, and RFM reuse with force rebuild

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_full_pipeline.py`

**Interfaces:**
- Produces: `engineer_features(*, force=False)`, `detect_outliers(..., force=False)`, and `calculate_rfm(..., force=False)`.
- Produces: JSON IQR summary at `aggregates/iqr_<column>.json`.

- [ ] **Step 1: Write failing artifact-specific tests**

```python
def test_precomputed_image_features_skip_decoder(self):
    analyzer = self.analyzer_with_precomputed_images()
    with mock.patch("src.pipeline.mpimg.imread", side_effect=AssertionError("decoder called")):
        analyzer.engineer_features()
    self.assertEqual(analyzer.artifact_status["image features"], "REUSED")

def test_precomputed_rfm_skips_partition_rebuild(self):
    analyzer = self.analyzer_with_precomputed_rfm()
    with mock.patch.object(Path, "unlink", side_effect=AssertionError("partition rebuild")):
        analyzer.calculate_rfm()
    self.assertEqual(analyzer.artifact_status["RFM"], "REUSED")

def test_force_rebuild_ignores_precomputed_image_features(self):
    analyzer = self.analyzer_with_precomputed_images()
    analyzer.engineer_features(force=True)
    self.assertEqual(analyzer.artifact_status["image features"], "COMPUTED")
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `python -m unittest tests.test_full_pipeline.FullPipelineTests -v`

Expected: FAIL because the methods do not accept `force` and do not inspect precomputed artifacts.

- [ ] **Step 3: Implement per-method reuse and immutable input handling**

```python
if not force:
    reusable = self._find_reusable_csv("image features", IMAGE_FEATURES_CACHE_FILENAME, IMAGE_FEATURE_REQUIRED_COLUMNS)
    if reusable is not None:
        return self._load_csv(reusable, IMAGE_FEATURE_DTYPES)

features.to_csv(self.images_path, index=False)
self._record_status("image features", "COMPUTED", "runtime", self.images_path)
```

Use an IQR JSON result when it includes matching column and threshold. Use
`rfm.csv` only when all RFM columns exist. Rebuild output only under writable
`context.runtime_root`; add a test that captures all precomputed file hashes
before and after a fallback computation to prove input artifacts were not
modified.

- [ ] **Step 4: Run the targeted tests to verify success**

Run: `python -m unittest tests.test_full_pipeline.FullPipelineTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_full_pipeline.py
git commit -m "feat(cache): reuse full-data feature and RFM artifacts"
```

### Task 4: Full-data EDA aggregate artifacts and notebook report

**Files:**
- Modify: `src/pipeline.py`
- Modify: `scripts/build_notebook.py`
- Modify: `tests/test_notebook.py`
- Modify: `tests/test_full_pipeline.py`

**Interfaces:**
- Produces: `DataAnalyzer.prepare_eda_artifacts(*, force=False) -> dict[str, object]`.
- Produces: exact `eda_summary.json` and `monthly_summary.csv` under writable aggregates on computation.

- [ ] **Step 1: Write failing EDA and notebook contract tests**

```python
def test_precomputed_eda_summary_avoids_transaction_scan(self):
    analyzer = self.analyzer_with_precomputed_eda()
    with mock.patch("src.pipeline.pd.read_csv", side_effect=AssertionError("transaction scan")):
        summary = analyzer.prepare_eda_artifacts()
    self.assertEqual(summary["price_age_correlation"], 0.25)

def test_notebook_declares_full_data_artifact_reuse(self):
    source = Path("scripts/build_notebook.py").read_text(encoding="utf-8")
    self.assertIn("full-data artifact", source)
    self.assertIn("artifact_status", source)
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `python -m unittest tests.test_notebook tests.test_full_pipeline -v`

Expected: FAIL because EDA summary artifact preparation and reuse messaging do not exist.

- [ ] **Step 3: Implement exact aggregate reuse and regenerate notebook**

```python
eda = analyzer.prepare_eda_artifacts()
print(analyzer.format_cache_report())

plt.stairs(eda["histogram_counts"], eda["histogram_edges"])
plt.bxp([eda["boxplot_before"], eda["boxplot_after"]])
monthly = pd.read_csv(eda["monthly_summary_path"])
```

Persist full-population histogram bins, boxplot statistics, correlations,
numeric summaries, and monthly totals. The notebook must use only those
artifacts for transaction-level visualizations on a cache hit and retain all
six chart types, titles, labels, RFM, and business insights.

- [ ] **Step 4: Run targeted tests and source notebook verification**

Run: `python -m unittest tests.test_notebook tests.test_full_pipeline -v && python scripts/build_notebook.py && python scripts/verify_notebook.py --source`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py scripts/build_notebook.py notebooks/analysis_report.ipynb tests/test_notebook.py tests/test_full_pipeline.py
git commit -m "feat(notebook): reuse full-data EDA artifacts"
```

### Task 5: Regression verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `scripts/synthetic_e2e.py`

**Interfaces:**
- Consumes: `DataAnalyzer.format_cache_report()` and `prepare_eda_artifacts()` from Tasks 2–4.

- [ ] **Step 1: Write a failing double-run synthetic E2E assertion**

```python
second_run = execute_notebook(...)
self.assertIn("transactions: REUSED", second_run.stdout)
self.assertIn("RFM: REUSED", second_run.stdout)
```

- [ ] **Step 2: Run the synthetic test to verify failure**

Run: `python -m unittest tests.test_notebook -v`

Expected: FAIL until the synthetic runner exercises a cache-hit second notebook run.

- [ ] **Step 3: Document evaluator workflow and cache behavior**

```markdown
1. Attach the H&M competition data and this repository to Kaggle.
2. Attach the prior notebook output, or set `HM_PRECOMPUTED_DIR` to it.
3. Open `notebooks/analysis_report.ipynb` and select Run All.
4. Confirm the printed `REUSED` statuses; use `force=True` only to rebuild.
5. Do not write to `/kaggle/input`; runtime output belongs under `/kaggle/working`.
```

- [ ] **Step 4: Run full verification**

Run: `python -m unittest discover -s tests -v && python scripts/build_notebook.py && python scripts/verify_notebook.py --source && python scripts/synthetic_e2e.py`

Expected: all unit tests, source checks, and two-run synthetic notebook verification PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/synthetic_e2e.py
git commit -m "docs(cache): explain evaluator artifact reuse"
```

## Plan self-review

- Spec coverage: Tasks 1–3 implement the required source priority, schema validation, read-only handling, reuse states, fallback, and `force=True`; Task 4 removes cache-hit transaction scans in the notebook; Task 5 covers E2E verification and the required five-line evaluator instructions.
- Placeholder scan: no incomplete work markers or deferred requirements are present.
- Interface consistency: `RuntimeContext.precomputed_root`, `DataAnalyzer.artifact_status`, `format_cache_report`, and `prepare_eda_artifacts` are introduced before their consumers.
