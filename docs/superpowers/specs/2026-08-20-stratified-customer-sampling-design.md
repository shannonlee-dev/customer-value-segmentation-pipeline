# Stratified Customer Sampling Design

## Goal

Make the portable H&M analysis run on a reproducible 20,000-customer sample while preserving complete purchase histories for sampled customers, the `club_member_status` population mix, and the exact product/image scope implied by those purchases.

## Scope

The public `DataAnalyzer` facade will read the customer dimension first, create a fixed-seed proportional stratified customer sample, then use that customer ID set to filter transaction chunks. Only selected customers, their complete transaction rows, products referenced by those rows, and the corresponding images will be cached and analyzed.

The sample is an analysis scope, not a row-level preview. It intentionally does not calculate or claim a 5,000/10,000/20,000 stability comparison in the normal pipeline run.

## Sampling Policy

`DataAnalyzer` will default to a sample size of 20,000 customers and random seed 42. Both are constructor parameters so tests and advanced users can reproduce or vary the sample explicitly.

Customers are assigned to strata by `club_member_status`, with missing values treated as their own stratum. Each stratum receives a quota proportional to its full-customer population. Integer quotas use a largest-remainder allocation so they sum exactly to the requested sample size. Sampling within each stratum is random with the fixed seed, and the sampled customer IDs are retained as a set for chunk filtering.

If the requested sample size is at least the available customer count, all customers are retained. A missing membership-status column remains a schema error, as it is required for both sampling and age imputation.

## Data Flow

1. Read and validate full `customers.csv` and `articles.csv` dimensions.
2. Select the customer sample before age imputation; perform the existing customer-grain age imputation on sampled customers only.
3. Stream `transactions_train.csv` in Pandas chunks, validate identifiers against the complete dimensions, and write only rows whose `customer_id` is in the selected sample.
4. Collect the selected transaction `article_id` values and write only those normalized article records.
5. Engineer image features only for the retained article records.
6. Run IQR, correlations, time series, and RFM from the sampled transaction cache. RFM therefore preserves every observed transaction for every selected customer, while excluding customers who have no transactions from RFM as it already does.

The run summary keeps `customer_rows`, `transaction_rows`, and `product_rows` as the analyzed sample counts and adds source/sample metadata suitable for notebook display.

## Interfaces and Compatibility

`DataAnalyzer(context, chunksize=..., customer_sample_size=20_000, sampling_seed=42)` is the new public construction contract. Existing callers that omit the new arguments get the 20,000-customer sample by default. Fixtures smaller than the requested size retain all customers, preserving their existing end-to-end behavior.

The notebook will label the analysis as a customer-level proportional stratified sample, print size and seed, and avoid all previous full-dataset claims. README instructions will describe the sampling unit, inclusion rules, reproducibility settings, and the limitation that population-wide results require a full-data run.

## Error Handling

The pipeline raises a clear `ValueError` if the requested sample size is not positive. It continues to reject missing or duplicate dimension IDs and transaction identifiers that do not resolve to the complete source dimensions. An empty selected transaction cache raises the existing RFM error when RFM is requested.

## Testing

Unit and end-to-end tests will prove the following:

- identical source data plus the same seed produces the same sampled customer IDs;
- per-stratum allocation totals exactly to the requested size and follows proportional population shares;
- the transaction cache contains all and only the source transactions for selected customers;
- articles and image features are limited to product IDs used by those cached transactions;
- small fixtures retain all customers under the default 20,000-customer limit;
- existing IQR and RFM behavior still operates on the sampled caches.

## Documentation

The README will use the approved explanation: customer-level proportional stratification by `club_member_status`, full transaction histories for sampled customers, all referenced products/images, and fixed-seed reproducibility. It will recommend 20,000 customers as the default scope and identify 5,000/10,000/20,000 stability comparison as a future validation step rather than an asserted result.

## Non-Goals

- No transaction-row sampling.
- No imputation-model redesign.
- No new analytical dependencies beyond NumPy, Pandas, Matplotlib, and Seaborn.
- No merge into `main` or pull-request creation.
