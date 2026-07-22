# Bloomberg Malaysia Bulk Data Admission

## Decision

The eight supplied files are stored in the Quant Lab data lake as immutable raw
sources and normalized candidate research data. They are a material expansion
of the Malaysia price/volume history, but they do not yet replace the frozen PIT
data receipt for the fourteen-strategy replication.

Run ID: `BLOOMBERG-MYS-BULK-2005-202604-V1`

## What was integrated

- Eight source files covering 2005-01-03 through 2026-04-13
- 5,102,436 raw rows
- 5,098,669 rows after deterministic security-date deduplication
- 4,998,498 usable candidate market rows
- 1,625 candidate listing keys
- 560 inactive or disappeared listing candidates
- Identifier history, candidate listing windows, monthly membership candidates,
  a dead-name crosswalk, and candidate return series

All source file sizes and SHA-256 hashes are recorded in the run manifest. Raw
files are retained unchanged. Normalization removed 53,774 blank pre-price
placeholder rows from the market candidate and resolved 7,534 rows that belonged
to duplicated security-date groups. The curated market keys are unique.

## Gate result

The mechanical market-data gates pass: all eight files are complete, dates
parse, security-date keys are unique, inactive candidates exist, and no negative
price or volume records were admitted.

The unchanged frozen PIT protocol still fails closed. Against the existing
Malaysia historical security master, 459 of 646 strategy-relevant inactive
entities have a high-confidence exact ISIN or unique ticker match. Coverage is
71.1%, below the preregistered 95% minimum.

Additional open gates are:

- production adjusted-return construction;
- terminal delisting and other terminal economics reconciliation;
- a frozen and reconstructable historical U2 membership rule;
- complete security-master promotion controls; and
- a denominator that is demonstrably not current-survivor-only.

There are also 612 candidate observations with an absolute one-day return above
100%. These remain candidates and require corporate-action or terminal-event
reconciliation before promotion.

## Lane controls

- Price/volume: candidate-ready after a new protocol freezes this exact data
  receipt.
- Fundamentals and accounting: paused until announcement timestamps and
  restatement handling prove point-in-time availability.
- Analyst estimates: paused until revision timestamps are proven.
- Fourteen-strategy PIT replication: blocked until every frozen institutional
  gate passes.
- Dry run: prohibited.

No strategy attempt was reserved or computed during ingestion.

The user's relationship and permission to ingest these files for internal
research are recorded as provenance. This is not independent verification of
Bloomberg vendor-licence or redistribution rights.

## Recommended next data work

1. Resolve the remaining 187 unmatched strategy-relevant inactive entities.
2. Reconcile the 612 extreme candidate returns against corporate actions and
   terminal events, then build production adjusted returns.
3. Freeze the exact historical U2 eligibility and membership rule.
4. Rerun the same PIT admission gate without modifying its thresholds.
5. Only after a pass, freeze the replacement data receipt and run the fourteen
   preregistered strategies once.

## Evidence locations

- `artifacts/data_lake/vendor/bloomberg_malaysia_bulk/BLOOMBERG-MYS-BULK-2005-202604-V1/manifest.json`
- `artifacts/data_lake/vendor/bloomberg_malaysia_bulk/BLOOMBERG-MYS-BULK-2005-202604-V1/raw_manifest.csv`
- `artifacts/data_lake/vendor/bloomberg_malaysia_bulk/BLOOMBERG-MYS-BULK-2005-202604-V1/quality_profile.csv`
- `artifacts/data_lake/vendor/bloomberg_malaysia_bulk/BLOOMBERG-MYS-BULK-2005-202604-V1/admission_report.json`
- `artifacts/data_lake/vendor/bloomberg_malaysia_bulk/BLOOMBERG-MYS-BULK-2005-202604-V1/frozen_pit_gate_projection.json`
