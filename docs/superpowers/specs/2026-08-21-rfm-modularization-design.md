# RFM Calculation Modularization Design

## Goal

Refactor `DataAnalyzer.calculate_rfm()` so that it orchestrates the RFM workflow without changing its public signature, output schema, cache format, or business results. The refactoring separates disk-backed execution, analytical aggregation, scoring, segmentation, and artifact persistence into independently testable responsibilities.

## Architecture

`calculate_rfm()` retains the existing parameters and follows this sequence:

1. `_load_cached_rfm()` returns a reusable `rfm.csv` when available.
2. `_partition_transactions()` reads the transaction cache in chunks, validates/parses RFM columns, writes each row to a stable customer-hash partition CSV, and returns partition paths plus the global maximum transaction date.
3. `_resolve_rfm_reference_date()` selects the explicit analysis date or the day after the maximum observed date.
4. `_aggregate_rfm_partitions()` reads partition files and concatenates their customer-level aggregates.
5. `_score_rfm()` assigns ordinal R/F/M scores through `_score_rfm_metric()`.
6. `_assign_rfm_segments()` applies the ordered business rules using `classify_segment()`.
7. `_save_rfm()` persists `rfm.csv`, updates `rfm_path`, and records the artifact status.

The helpers remain methods of `DataAnalyzer` where they need runtime paths, chunk configuration, caching, or artifact status. `_aggregate_rfm_partition()`, `_score_rfm_metric()`, and `classify_segment()` are data-only helpers with no filesystem dependency.

## Data Flow and Compatibility

The partitioning strategy remains CSV-backed and uses `pd.util.hash_pandas_object(customer_id, index=False) % partition_count`. Therefore every occurrence of the same customer identifier is written to the same partition, including occurrences spanning chunks.

The aggregation semantics stay unchanged:

- Recency is `(reference_date - latest_purchase_date).days`.
- Frequency is the count of unique purchase dates.
- Monetary is the sum of amounts.

Scoring uses average percentile ranks and fixed quartile intervals, so equal raw values always receive equal scores. Lower recency gets the higher score; higher frequency and monetary values get higher scores. A one-customer RFM frame receives score 4 for every metric. Segmentation retains its ordered priority: `VIP`, `Loyal`, `New`, `Churned`, then `Potential`.

Existing callers continue to use `calculate_rfm()` unchanged. Cache reuse still returns `rfm.csv` with the requested customer column loaded as Pandas `string` dtype. Empty transaction input still raises the existing `ValueError`.

## Error Handling

Date and amount parsing continues to use strict errors. Empty transaction caches are rejected before a reference date is resolved. Missing partition files are skipped during aggregation, as they represent empty hash buckets; a non-empty input necessarily yields at least one aggregate.

## Testing

Add focused tests alongside the existing smoke tests:

- Same customer identifiers, across multiple chunks, appear in exactly one partition.
- `_aggregate_rfm_partition()` calculates recency, unique-date frequency, and monetary sum from a synthetic frame.
- `_score_rfm_metric()` handles the one-customer policy and preserves intended ordering under ties.
- `classify_segment()` verifies representative rules and their priority.
- The existing end-to-end fixture remains the compatibility check for cache creation and RFM output.

## Scope Boundaries

This change does not alter segment definitions, partition count defaults, input/output formats, or the choice of CSV partition files. Scoring uses four fixed percentile intervals with average ranks for ties. Replacing CSV with Parquet or DuckDB, changing frequency semantics, and changing the number of score intervals are future isolated changes enabled by these boundaries.
