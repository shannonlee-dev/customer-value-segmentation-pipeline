# Full-data portable notebook refactor plan

## Current-state gap analysis

The prior project used a deterministic 500-customer cohort, a small image sample, and an executed notebook with fixed claims. That conflicted with the required full-data Run All workflow. It also did not provide one shared Kaggle/local runtime, customer-partitioned RFM, or a clean notebook source.

## File-by-file implementation plan

1. `src/runtime.py`: detect `HM_RAW_DATA_DIR`, Kaggle competition input, then `data/raw/h-and-m`; create a separate writable runtime root.
2. `src/pipeline.py`: make `DataAnalyzer` the central public interface. Stream all transactions with Pandas; impute customer age by group median; derive text/image features; calculate IQR and partitioned all-customer RFM.
3. `scripts/prepare_hm_data.py`: expose the same facade for local preparation.
4. `scripts/build_notebook.py` and `notebooks/analysis_report.ipynb`: emit one clean notebook whose primary UX is Run All and which shows the required preview, statistics, correlations, and six plot types.
5. `scripts/verify_notebook.py`: validate clean source and executed evidence separately.
6. `tests/`: add runtime, synthetic full-pipeline, notebook-source, and dependency-boundary contracts.
7. `README.md` and `data/README.md`: document local/Kaggle execution and leave real-data business-insight values for the later Kaggle run.

## Constraints applied

- Only standard library plus NumPy, Pandas, Matplotlib, and Seaborn power analysis.
- No Pillow, OpenCV, DuckDB, PyArrow, or Parquet.
- No customer, product, transaction, or image analysis sampling.
- Image file iteration is I/O orchestration only; every decoded image array uses NumPy reductions.
- No fabricated full-dataset result claims.
