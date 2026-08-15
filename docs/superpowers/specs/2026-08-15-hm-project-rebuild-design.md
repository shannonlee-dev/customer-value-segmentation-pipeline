# H&M Customer Value Segmentation Rebuild Design

## Goal

Rebuild the repository from a clean Git root using only analysis results derived
from the H&M Personalized Fashion Recommendations source data. The finished
project must satisfy every mandatory item in `mission.md` and the relevant
checks in `eval.md`: multimodal feature engineering, grouped missing-value
imputation, IQR analysis, six chart types, RFM segmentation, an executed
notebook, and evidence-based business recommendations.

## Source and Distribution Boundary

The source is the H&M Personalized Fashion Recommendations competition dataset:

- `transactions_train.csv` supplies anonymized customer purchases, dates,
  prices, and sales channels;
- `articles.csv` supplies product names, categories, descriptions, and other
  product metadata;
- `customers.csv` supplies age and membership metadata; and
- `images/` supplies product photographs keyed by `article_id`.

The dataset is governed by the Kaggle competition rules, including
non-commercial use and redistribution restrictions. It is not an MIT, CC-BY,
or otherwise open-licensed dataset. Raw files, derived row-level files, and
product images remain local under ignored `data/raw/` and `data/processed/`
paths. The public repository contains acquisition and preparation code, an
executed report with redacted identifiers and aggregate evidence, and no H&M
data files.

## Deterministic Cohort

The preparation script validates the upstream schemas before reading data. It
identifies customers with at least one transaction, orders them by the SHA-256
digest of `seed + customer_id`, and selects the first 500. This produces a
stable sample that is independent of customer value, purchase recency,
frequency, or spend.

All transactions for those customers are retained so RFM is based on repeated
behavior rather than isolated transaction rows. Transactions are joined to
article and customer metadata. Rows without an available product image are
excluded from the integrated analysis table, and the script records both the
count and percentage excluded.

The normalized local table contains at least these columns:

| Output column | H&M source |
| --- | --- |
| `order_date` | `transactions_train.t_dat` |
| `customer_id` | `transactions_train.customer_id` |
| `product_id` | `transactions_train.article_id` |
| `product_name` | `articles.prod_name` |
| `category` | `articles.product_group_name` |
| `unit_price` | `transactions_train.price` |
| `sales_channel_id` | `transactions_train.sales_channel_id` |
| `age` | `customers.age` |
| `club_member_status` | `customers.club_member_status` |
| `fashion_news_frequency` | `customers.fashion_news_frequency` |
| `image_path` | deterministic path from `article_id` |

No order identifier, quantity, discount, product attribute, customer behavior,
or missing value is invented. H&M records one purchased item per transaction
row, so Monetary uses the original `price` directly. Documentation describes
`price` as a relative dataset value rather than a real-world currency amount.

## Components and Data Flow

### Data preparation

`scripts/prepare_hm_data.py` accepts configurable raw and output directories,
cohort size, and seed. It reads the large transaction file in chunks, performs
validated joins, checks image existence, and writes the ignored local cohort
CSV plus a small local preparation summary. It fails on missing files, missing
required columns, duplicate article/customer keys, an empty cohort, or a final
table below the mission's minimum shape.

### Analysis class

`src/pipeline.py` defines `DataAnalyzer` with these public responsibilities:

- `load_data()` validates and restores dates and numeric columns;
- `handle_missing_values()` applies a group mean or median with a global
  fallback;
- `engineer_features()` loads each unique image once, downsamples it with
  NumPy slicing, creates a stacked image tensor, and computes image Mean/Std
  across spatial and channel axes without a feature-computation loop;
- `detect_outliers()` returns IQR outlier rows and configurable bounds; and
- `calculate_rfm()` calculates Recency, unique purchase-date Frequency, and
  Monetary, then assigns quantile scores and segments.

Image I/O may iterate over distinct paths because each JPEG must be decoded
individually. Numerical feature calculation uses `np.stack`, `mean`, and `std`
over the complete tensor. Features are calculated once per product and joined
back to transaction rows.

The default RFM reference date is one day after the last cohort transaction.
Rank-based quartiles handle tied values. Segment rules produce `VIP`, `Loyal`,
`New`, `Churned`, and `Potential` groups, with explicit ordering so every
customer receives one segment.

### Executed analysis report

`scripts/build_notebook.py` builds `notebooks/analysis_report.ipynb`. The
notebook is executed from a clean kernel and committed with outputs. It covers:

1. source, cohort selection, and distribution limitations;
2. masked `head()`, `info()`, and `describe()` inspection;
3. real age missingness and grouped median imputation by membership status;
4. image Mean/Std and product-name length features;
5. price IQR detection and before/after comparison on an analysis copy;
6. descriptive statistics and at least two interpreted correlations;
7. histogram, boxplot, bar chart, heatmap, scatterplot, and line chart;
8. five-segment RFM profiling and evidence-based actions; and
9. sampling, price-unit, missing order-ID, and causal-inference limitations.

The original cohort file is never overwritten by IQR treatment. Every chart
has a title and axis labels. Saved outputs contain no complete customer ID,
product ID, raw transaction row, image, or reconstructable row-level export.

## README Requirements

The root README explains the work as a reproducible sequence rather than only
reporting final numbers. It includes:

- how Kaggle, AI Hub, UCI, and other candidates were screened;
- why H&M was selected over transaction-only and image-only alternatives;
- the competition-rule license boundary and official source links;
- Kaggle acceptance, download, local directory, and preparation commands;
- the exact 500-customer selection and join rules;
- source-to-output column mappings and exclusions;
- NumPy image processing, grouped imputation, IQR, and RFM decisions;
- three insights containing numeric evidence, a target and action, and a
  falsification or additional-data requirement; and
- statistical, sampling, performance, and commercial-use limitations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/README.md
├── src/pipeline.py
├── scripts/prepare_hm_data.py
├── scripts/build_notebook.py
├── scripts/verify_notebook.py
├── notebooks/analysis_report.ipynb
├── artifacts/notebook_execution.log
└── tests/
    ├── test_data_preparation.py
    ├── test_pipeline.py
    ├── test_notebook.py
    └── test_mission_compliance.py
```

Tests use small temporary fixtures solely to isolate program behavior. They do
not serve as analysis data or appear in reported business results.

## Verification

Verification must demonstrate:

- deterministic cohort membership for a fixed seed;
- at least 1,000 joined rows, at least 8 columns, and required data types;
- source keys and images join without invented values;
- grouped imputation, image vectorization, IQR bounds, and RFM behavior;
- at least four non-empty RFM segments in the real executed report;
- all six required chart outputs with no notebook cell errors;
- masked notebook previews and absence of committed H&M row-level data;
- README coverage of provenance, license, method, limitations, and business
  evidence; and
- full test-suite and clean-kernel notebook execution success.

## Git History Rebuild

Before rewriting local `main`, preserve the current commit in a dated backup
tag and preserve uncommitted tracked/untracked work in a named stash. Build the
new history in an isolated temporary repository, using one Conventional Commit
per coherent feature and author name `shannonlee-dev`:

1. `docs(design): define H&M customer analysis`
2. `chore(project): initialize H&M analysis workspace`
3. `feat(data): add deterministic H&M cohort preparation`
4. `feat(pipeline): add multimodal preprocessing and RFM`
5. `feat(analysis): add executed H&M EDA and segmentation report`
6. `docs(readme): document provenance, workflow, and insights`

After aggregate verification, import the isolated branch and replace only the
local `main` reference. Keep the backup tag and stash. Do not force-push or
otherwise change `origin/main` without a separate explicit request.
