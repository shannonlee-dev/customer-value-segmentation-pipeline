# Cache-aware full-data Run All design

## Goal

Keep the existing full-data pipeline as the single source of truth while making a
repeat execution of `notebooks/analysis_report.ipynb` reuse validated full-data
artifacts. A cache hit must avoid reading the 30-million-row transaction source
and the product-image files. `force=True` must explicitly rebuild an artifact.

The notebook remains portable across Kaggle, `data/raw/h-and-m`, and an
`HM_RAW_DATA_DIR` location. It must not add a separate local or Kaggle analysis
path and must not use analysis sampling.

## Alternatives considered

1. File-exists-only reuse is simple, but can silently present stale output after
   a source file or feature algorithm changes.
2. Recompute every artifact on every notebook run is always current, but makes
   evaluator-facing Run All impractical.
3. **Selected: versioned manifest validation.** Each artifact is reused only
   when its input fingerprints and producing configuration match a manifest.
   This keeps repeat runs fast while retaining an explicit correctness boundary.

## Artifact contract

`runtime_root` will contain the following reusable, full-data artifacts:

| Artifact | Grain / contents | Producer | Cache dependencies |
| --- | --- | --- | --- |
| `processed/transactions.csv` | transaction | `load_data()` | raw transactions fingerprint, normalization version |
| `processed/customers.csv` | customer | `load_data()` | raw customers fingerprint, imputation configuration/version |
| `processed/articles.csv` | article | `load_data()` | raw articles fingerprint, text-feature configuration/version |
| `features/product_images/product_images.csv` | article image feature | `engineer_features()` | normalized articles fingerprint, image-feature version |
| `aggregates/iqr_unit_price.json` | exact IQR bounds and counts | `detect_outliers()` | normalized transaction artifact fingerprint, column, threshold |
| `aggregates/rfm.csv` | customer RFM and segment | `calculate_rfm()` | normalized transaction fingerprint, RFM parameters/version |
| `aggregates/eda_summary.json` | full-data scalar summaries, correlations, chart statistics | EDA preparation method | transaction, article/image, IQR, and RFM artifact fingerprints |
| `aggregates/monthly_summary.csv` | month-level full-data totals | EDA preparation method | normalized transaction fingerprint |
| `artifacts/cache_manifest.json` | artifact input fingerprints and parameters | cache helper | all entries above |

The cache manifest records the cache schema version, artifact path, file
fingerprint, producer parameters, and input fingerprints. A fingerprint uses a
file's resolved path, size, and nanosecond modification timestamp. It is cheap
to check and avoids re-reading source CSV contents.

## Public API and data flow

Existing public methods retain their role and receive an optional keyword-only
`force: bool = False` argument:

```python
load_data(force=False)
engineer_features(force=False)
detect_outliers(column=..., threshold=..., force=False)
calculate_rfm(..., force=False)
prepare_eda_artifacts(force=False)
```

On a valid hit, each method loads and returns its artifact without processing
the corresponding raw source. On a miss, stale metadata, missing artifact, or
`force=True`, it performs the existing full-data computation and atomically
replaces the artifact and manifest entry.

The notebook calls the same `DataAnalyzer` methods in the usual order, then
loads EDA summaries rather than rescanning transactions. It prints which
artifacts were reused or rebuilt and documents that the artifacts are
full-data outputs of the same pipeline. The six required charts use exact
aggregates and boxplot statistics, not sampled transaction rows.

## EDA cache detail

The EDA preparation pass reads normalized transactions only on a cache miss.
It computes and stores:

- full-distribution histogram bin edges and counts;
- exact before/after-IQR boxplot statistics;
- price/age correlation sufficient statistics and final correlation;
- full monthly totals and transaction counts;
- product-level image/text correlation;
- numerical-statistics values used by the notebook narrative.

The report uses `Axes.bxp` for cached boxplot statistics and histogram counts
for the histogram. Therefore charts remain based on the entire population
without retaining a giant integrated DataFrame or reopening the transaction
CSV on cache hits. RFM charts read `rfm.csv`, which is already a customer-grain
artifact.

## Invalidation and errors

- A raw CSV change invalidates the directly derived normalized artifact and
  downstream dependent artifacts.
- An image-feature implementation-version change invalidates image features
  and EDA summaries that depend on them.
- Changing an IQR column or threshold creates/uses a parameter-specific cache
  entry; changing RFM parameters invalidates its result.
- A malformed or missing manifest entry is a cache miss, never a successful
  hit.
- Cache writes use a temporary sibling path followed by replacement, so an
  interrupted run cannot advertise a partial output as valid.
- `force=True` bypasses otherwise valid entries and rebuilds with the same full
  pipeline. It does not introduce a sampling path.

## Tests and verification

Tests will first demonstrate that cache-aware methods fail expectations before
implementation, then verify that a second invocation succeeds after source
files are made unavailable. Additional tests cover parameter invalidation,
`force=True`, manifest corruption, and EDA artifact reuse. The synthetic
end-to-end notebook run will be performed twice: first for artifact creation,
then again after artifacts exist to confirm a cache-hit Run All completes.

No real H&M results will be fabricated. Full-data cache creation and full-data
verification remain the user's Kaggle execution step.
