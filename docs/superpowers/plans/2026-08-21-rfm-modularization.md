# RFM Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `calculate_rfm()` into an orchestrator without changing its API, cache behavior, or RFM output.

**Architecture:** Private instance helpers own runtime I/O; static data-only helpers own aggregation, scoring, and classification. Disk-backed CSV hash partitioning stays unchanged.

**Tech Stack:** Python 3, Pandas, NumPy, unittest.

**Spec:** `docs/superpowers/specs/2026-08-21-rfm-modularization-design.md`

## Global Constraints

- Preserve strict parsing, RFM CSV schema, unique-date frequency, default 64 CSV partitions, tie-preserving percentile scores, and ordered segment rules.
- Do not modify `notebooks/analysis_report.ipynb`.

---

### Task 1: Establish focused helper tests

**Files:**
- Modify: `tests/test_pipeline_smoke.py`

**Interfaces:** Produces regression coverage for partition locality, aggregation, scoring, and segmentation.

- [ ] **Step 1: Write failing tests for the specified behavior**

```python
def test_aggregate_rfm_partition_calculates_metrics(self):
    frame = pd.DataFrame({"customer_id": ["A", "A", "A"], "order_date": ["2020-01-01", "2020-01-01", "2020-01-10"], "unit_price": [10, 20, 30]})
    result = DataAnalyzer._aggregate_rfm_partition(frame, "customer_id", "order_date", "unit_price", pd.Timestamp("2020-01-11"))
    self.assertEqual(result.loc[0, "recency"], 1)
    self.assertEqual(result.loc[0, "frequency"], 2)
    self.assertEqual(result.loc[0, "monetary"], 60)

def test_score_rfm_metric_handles_single_customer_and_ties(self):
    self.assertEqual(DataAnalyzer._score_rfm_metric(pd.Series([10]), ascending=True).iloc[0], 4)
    self.assertListEqual(DataAnalyzer._score_rfm_metric(pd.Series([10, 10, 20, 30]), ascending=True).tolist(), [1, 2, 3, 4])

def test_classify_segment_applies_ordered_rules(self):
    self.assertEqual(DataAnalyzer.classify_segment(4, 4, 4), "VIP")
    self.assertEqual(DataAnalyzer.classify_segment(4, 1, 2), "New")
    self.assertEqual(DataAnalyzer.classify_segment(1, 4, 4), "Churned")
```

- [ ] **Step 2: Run the aggregation test to verify red**

Run: `python -m unittest tests.test_pipeline_smoke.PipelineSmokeTest.test_aggregate_rfm_partition_calculates_metrics`

Expected: FAIL because `_aggregate_rfm_partition` is absent.

- [ ] **Step 3: Commit red tests**

Run: `git add tests/test_pipeline_smoke.py && git commit -m "test: specify modular RFM helpers"`

### Task 2: Extract execution helpers

**Files:**
- Modify: `src/pipeline.py:349-428`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:** Produces `_load_cached_rfm`, `_partition_transactions`, `_resolve_rfm_reference_date`, and `_save_rfm`.

- [ ] **Step 1: Add failing cross-chunk partition locality test**

```python
paths, _ = analyzer._partition_transactions("customer_id", "order_date", "unit_price", 3)
found = [path for path in paths if path.is_file() and "customer-a" in pd.read_csv(path)["customer_id"].tolist()]
self.assertEqual(len(found), 1)
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_pipeline_smoke.PipelineSmokeTest.test_partition_transactions_keeps_each_customer_in_one_partition`

Expected: FAIL because `_partition_transactions` is absent.

- [ ] **Step 3: Extract minimal code**

```python
def _partition_transactions(self, customer_col, date_col, amount_col, partition_count):
    # Move existing directory cleanup, chunk parsing, maximum-date tracking,
    # stable hash bucket assignment, and CSV append/header behavior here.
    # Return (paths, max_date); keep the empty-cache ValueError.

def _resolve_rfm_reference_date(self, analysis_date, max_date):
    return pd.Timestamp(analysis_date) if analysis_date is not None else pd.Timestamp(max_date.date()) + pd.offsets.Day(1)
```

- [ ] **Step 4: Verify green and commit**

Run: `python -m unittest tests.test_pipeline_smoke.PipelineSmokeTest.test_partition_transactions_keeps_each_customer_in_one_partition && git add src/pipeline.py tests/test_pipeline_smoke.py && git commit -m "refactor: extract RFM partition execution"`

Expected: PASS before commit.

### Task 3: Extract analytical and business helpers

**Files:**
- Modify: `src/pipeline.py:349-428`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:** Produces `_aggregate_rfm_partitions`, `_aggregate_rfm_partition`, `_score_rfm_metric`, `_score_rfm`, `classify_segment`, and `_assign_rfm_segments`.

- [ ] **Step 1: Implement the helpers**

```python
@staticmethod
def _aggregate_rfm_partition(frame, customer_col, date_col, amount_col, reference_date):
    return frame.groupby(customer_col, as_index=False).agg(recency=(date_col, lambda dates: (reference_date - dates.max()).days), frequency=(date_col, "nunique"), monetary=(amount_col, "sum"))

@staticmethod
def _score_rfm_metric(values, *, ascending):
    if len(values) == 1:
        return pd.Series(4, index=values.index, dtype="int64")
    ranked = values.rank(method="average", pct=True, ascending=ascending)
    return pd.cut(ranked, bins=[0, 0.25, 0.50, 0.75, 1.0], labels=(1, 2, 3, 4), include_lowest=True).astype(int)
```

- [ ] **Step 2: Implement priority classification**

```python
if (r_score, f_score, m_score) == (4, 4, 4): return "VIP"
if r_score >= 3 and f_score >= 3: return "Loyal"
if r_score == 4 and f_score <= 2: return "New"
if r_score == 1: return "Churned"
return "Potential"
```

- [ ] **Step 3: Verify green and commit**

Run: `python -m unittest tests.test_pipeline_smoke.PipelineSmokeTest.test_aggregate_rfm_partition_calculates_metrics tests.test_pipeline_smoke.PipelineSmokeTest.test_score_rfm_metric_handles_single_customer_and_ties tests.test_pipeline_smoke.PipelineSmokeTest.test_classify_segment_applies_ordered_rules && git add src/pipeline.py tests/test_pipeline_smoke.py && git commit -m "refactor: separate RFM analytics and segmentation"`

Expected: PASS before commit.

### Task 4: Orchestrate and verify compatibility

**Files:**
- Modify: `src/pipeline.py:349-428`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:** Uses extracted helpers to preserve public `calculate_rfm()` behavior.

- [ ] **Step 1: Make the method orchestration only**

```python
cached = self._load_cached_rfm(customer_col, force=force)
if cached is not None: return cached
paths, max_date = self._partition_transactions(customer_col, date_col, amount_col, partition_count)
reference = self._resolve_rfm_reference_date(analysis_date, max_date)
rfm = self._aggregate_rfm_partitions(paths, customer_col, date_col, amount_col, reference)
return self._save_rfm(self._assign_rfm_segments(self._score_rfm(rfm)))
```

- [ ] **Step 2: Run full verification and inspect scope**

Run: `python -m unittest discover -s tests -v && git diff --check && git status --short`

Expected: all tests PASS, no whitespace errors, and the notebook change remains unstaged.

- [ ] **Step 3: Commit intended implementation files only**

Run: `git add src/pipeline.py tests/test_pipeline_smoke.py && git commit -m "refactor: modularize RFM workflow"`
