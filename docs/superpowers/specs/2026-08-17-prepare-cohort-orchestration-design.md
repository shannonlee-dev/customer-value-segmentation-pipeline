# Prepare Cohort Orchestration Design

## Goal

Refactor `scripts.prepare_hm_data.prepare_cohort` so it coordinates named preparation
stages instead of containing every transformation inline. The public API, command-line
interface, output files, summary schema, validation rules, and deterministic customer
selection must remain unchanged.

## Options considered

1. Keep the current monolithic function and add comments. This improves navigation only
   slightly and leaves the preparation stages hard to test or reuse independently.
2. Extract private, stage-oriented functions and keep `prepare_cohort` as their
   orchestrator. This preserves the API while making the execution order explicit.
3. Introduce a stateful preparation class. This adds lifecycle and mutation complexity
   without a current need for alternative workflows.

Option 2 is selected.

## Architecture

`prepare_cohort` will execute these stages in order:

1. Validate options and resolve required source paths.
2. Read article and customer metadata.
3. Scan active customer identifiers and select the deterministic cohort.
4. Load only transactions for the selected customers.
5. Join and validate metadata, then normalize the output schema.
6. Keep records with available images and enforce the minimum row count.
7. Sort and write the cohort CSV, then write its summary JSON.

Each stage will be a private function with explicit inputs and outputs. Existing low-level
helpers remain private and are reused rather than duplicated.

## Error handling

The refactor retains existing exceptions and validation behavior. Stage functions will
raise the same error types and messages for equivalent invalid inputs; the orchestrator
will not catch or translate them.

## Verification

Existing data-preparation tests must remain green. Add focused tests only if extraction
changes a previously untested stage boundary. Run the complete test suite and confirm the
CLI continues to write the requested output.
