# Historical Data Lake Backfill

## Backtest Depth Standard

Five years of daily OHLCV is enough for plumbing validation, feature smoke tests, and early strategy triage. It is not enough for promotion-grade alpha research. A production research lake should target:

- 15-20 years of daily prices where the listing has existed that long.
- Explicit bull, bear, rate, FX, commodity, crisis, and post-crisis regimes.
- A current-survivor lake clearly labeled as such.
- A later point-in-time security master with delisted names before institutional claims are made.
- Point-in-time fundamentals, estimates, analyst revisions, and events in separate dated tables keyed to the same security identifiers.

## Batch Method

Use `examples/run_refinitiv_historical_backfill.py` for long Refinitiv jobs. It separates planning from execution:

1. Build a deterministic plan from the validated symbol map.
2. Split by `country`, `exchange`, ticker batch, and date window.
3. Execute one plan row at a time.
4. Write wide prices, wide OHLCV, coverage CSV, long OHLCV Parquet, and a manifest row immediately.
5. Resume from `historical_manifest.csv`; completed rows are skipped.
6. Run `examples/run_data_depth_report.py` after each meaningful batch.

Default active-market Asia plan used in this project:

- Start: `2006-01-01`
- End: `2026-07-06`
- Window size: `5` years
- Ticker batch size: `6`
- Active countries: Malaysia, Singapore, Indonesia, Hong Kong, South Korea, and Japan.
- China and Taiwan are excluded from the quant-project mandate.
- Generate current RIC and plan-row counts from the active symbol map rather than reusing the superseded 2026-07-08 expansion denominator.

The windows are:

- `2006-01-01` to `2010-12-31`
- `2011-01-01` to `2015-12-31`
- `2016-01-01` to `2020-12-31`
- `2021-01-01` to `2025-12-31`
- `2026-01-01` to `2026-07-06`

## Commands

Create the full 20-year plan without touching Refinitiv:

```bash
uv run python examples/run_refinitiv_historical_backfill.py \
  --run-id REFINITIV-ASIA-HISTORICAL-20Y-PLAN-20260707 \
  --symbol-map-csv artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1-20260707/symbol_map.csv \
  --field-catalog-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv \
  --start-date 2006-01-01 \
  --end-date 2026-07-06 \
  --window-years 5 \
  --batch-size 6 \
  --include-countries "Malaysia,Singapore,Indonesia,Hong Kong,South Korea,Japan" \
  --plan-only
```

Run a bounded smoke extraction:

```bash
uv run python examples/run_refinitiv_historical_backfill.py \
  --run-id REFINITIV-ASIA-HISTORICAL-20Y-SMOKE-20260707 \
  --symbol-map-csv artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1-20260707/symbol_map.csv \
  --field-catalog-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv \
  --start-date 2006-01-01 \
  --end-date 2026-07-06 \
  --window-years 5 \
  --batch-size 2 \
  --max-tickers 2 \
  --max-plan-rows 1 \
  --include-countries "Malaysia,Singapore,Indonesia,Hong Kong,South Korea,Japan"
```

Run or resume the full current-survivor historical build:

```bash
uv run python examples/run_refinitiv_historical_backfill.py \
  --run-id REFINITIV-ASIA-HISTORICAL-20Y-L1-20260707 \
  --symbol-map-csv artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1-20260707/symbol_map.csv \
  --field-catalog-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv \
  --start-date 2006-01-01 \
  --end-date 2026-07-06 \
  --window-years 5 \
  --batch-size 6 \
  --include-countries "Malaysia,Singapore,Indonesia,Hong Kong,South Korea,Japan"
```

Run a small chunk of the full job and stop:

```bash
uv run python examples/run_refinitiv_historical_backfill.py \
  --run-id REFINITIV-ASIA-HISTORICAL-20Y-L1-20260707 \
  --symbol-map-csv artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1-20260707/symbol_map.csv \
  --field-catalog-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv \
  --start-date 2006-01-01 \
  --end-date 2026-07-06 \
  --window-years 5 \
  --batch-size 6 \
  --max-plan-rows 10 \
  --include-countries "Malaysia,Singapore,Indonesia,Hong Kong,South Korea,Japan"
```

Create QA after a completed or partial run:

```bash
uv run python examples/run_data_depth_report.py \
  artifacts/data_lake/prices_long/REFINITIV-ASIA-HISTORICAL-20Y-L1-20260707/ohlcv \
  --run-id REFINITIV-ASIA-HISTORICAL-20Y-L1-QA-20260707 \
  --extraction-manifest-path artifacts/historical_backfill/REFINITIV-ASIA-HISTORICAL-20Y-L1-20260707/historical_manifest.csv
```

## Superseded Smoke Evidence

The old-window smoke run completed successfully, but it is retained for engineering audit only and is excluded from all active project universes:

- Run: `REFINITIV-ASIA-HISTORICAL-20Y-SMOKE-20260707`
- Window: `2006-01-01` to `2010-12-31`
- Market: China SSE
- Tickers: `600000.SS`, `600004.SS`
- Long OHLCV rows: `2,359`
- Date range returned: `2006-01-04` to `2010-12-31`
- Failures: `0`
- Volume missing: `0.00%`
- Non-positive close rows: `0`

Do not resume, promote, or combine this China smoke artifact with active-market research.

## Caveats

This L1 historical build is still a current-survivor universe. It is useful for engineering, signal triage, and regime expansion, but it is not final institutional evidence until the security master includes historical constituents, delisted securities, corporate actions, and point-in-time fundamentals/estimates/events.
