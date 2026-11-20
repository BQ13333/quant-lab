# Data Source Roadmap

## Mandate as of 20 November 2026

Benjamin Quek limited active research to Bursa Malaysia, China A-shares (SSE and SZSE), and Hong Kong. Singapore, Indonesia, Japan, Korea, and the United States are frozen. The Beijing Stock Exchange stays out. Fundamentals are official filings only: Bursa, HKEXnews, and CNINFO, subject to access rights. No paid feed. The region table and the tickets below are the 12 July 2026 queue. They are not the active mandate. The scope file is `configs/project_market_scope.yaml`. The decision record is `docs/DECISIONS_2026-11-20.md`.

## Purpose

This roadmap turns the regional data-agent findings into an implementation
queue. The goal is a broad research universe without fooling ourselves with
survivorship bias, missing delistings, or free-data ticker gaps.

## Data Quality Tiers

### Tier 0: Discovery

Current listings, incomplete price coverage, no delistings. Good for finding
symbols and testing adapters. Not good for strategy claims.

### Tier 1: Research Smoke Test

Current listings plus daily OHLCV coverage and explicit failure manifests. Good
for checking whether the engine runs. Still not enough for live capital.

### Tier 2: Serious Historical Research

Point-in-time security master, delistings, corporate actions, adjusted and raw
OHLCV, FX, volume, liquidity, and benchmark mappings.

### Tier 3: Production/Paper-Trading Ready

Tier 2 plus intraday or execution data, slippage model, exchange fee/tax
schedules, broker reconciliation, and monitoring.

## Current Region Status

| Region | Current Status | Main Blocker | Next Move |
| --- | --- | --- | --- |
| Malaysia | Curated `.KL` basket and Malaysia sweep exist | Not full Bursa, no delistings | Add Bursa security-master adapter/vendor mapping |
| Singapore | SGX listing adapter and Yahoo smoke path | Corporate actions and full historical quality | Keep SGX adapter, add adjusted OHLCV vendor path |
| Thailand | Parked outside current three-country universe | Current scope is Malaysia, Singapore, Indonesia | Revisit only if market scope expands |
| Indonesia | IDX public fetch is blocked/challenged | Security master and OHLCV vendor | Add vendor mapping path before backtests |
| Philippines | Removed from active project universe | Not tradable for us | Do not include in default research/backtests |
| Vietnam | Removed from active project universe | Failed strict 20-year L2 coverage gate | Do not include in default research/backtests |
| China | Excluded from project scope | Existing compatibility only | No active extraction or research work |
| Hong Kong | HKEX listing support exists | Historical data/corporate actions | Add paid/vendor or Stooq/Yahoo smoke path |
| Japan | JPX listing support exists | OHLCV/corporate-action integration | Add J-Quants adapter |
| Korea | KRX listing support exists | KRX suffix bug, delisted data, OHLCV | Fix KRX mapping, add KRX data adapter |
| Taiwan | Excluded from project scope | Existing compatibility only | No active extraction or research work |
| US | Nasdaq Trader current listings exist | Point-in-time delisted/corporate-action data | Add CRSP/Sharadar/Polygon-style adapter |
| Global ETFs | ETF proxy research exists | Benchmark metadata and survivorship less severe | Maintain as tradable proxy universe |

## Immediate Data Engineering Queue

1. Fix known mapping defects.
   - KRX KOSPI rows containing `유가` must map to `.KS`. Completed in
     `parse_krx_kind_listed`.
   - KOSDAQ rows must map to `.KQ`. Completed in `parse_krx_kind_listed`.
   - KONEX should be flagged separately until Yahoo mapping is verified.

2. Expand FX support.
   - Add `CNY`, `HKD`, `JPY`, `KRW`, `TWD`. Completed in `fx.py`.
   - Normalize `RMB` to `CNY`. Completed in `fx.py` and `panels.py`.

3. Add regional source registries.
   - `src/quant_lab/data/north_asia_sources.py`
   - Expand `src/quant_lab/data/asean_sources.py`
   - Add a US/global vendor registry.

4. Generalize vendor adapters.
   - Keep Yahoo as smoke-test only.
   - Add provider name, interval, adjusted/raw mode, start/end, retry/backoff,
     and rate-limit metadata.
   - Refinitiv/LSEG adapter scaffold is now available through
     `RefinitivAdapter`; configure credentials outside the repo using
     `docs/REFINITIV_LSEG_SETUP.md`.

5. Generalize batch backfills.
   - Write long/narrow OHLCV Parquet.
   - Partition by provider, interval, adjustment, country, exchange, year, and
     month.
   - Write failure manifests for every batch.

6. Add corporate-action and delisting tables.
   - `corporate_actions.csv`
   - `delistings.csv`
   - `symbol_map.csv`
   - `listing_timeline.csv`

7. Block serious strategy promotion on Tier 0/Tier 1 data.
   - A strategy can run on smoke data.
   - A strategy cannot be called production-grade without Tier 2 data.
   - Every non-price feature must pass the point-in-time contract in
     `docs/POINT_IN_TIME_BACKTESTING.md`.

## Agent Findings Converted To Build Tickets

### ASEAN Tickets

- Bursa:
  - Add `BURSA_LISTED_COMPANIES_URL`.
  - Add `fetch_bursa_listed()`.
  - Add `parse_bursa_listed()`.
  - Add `BursaHistoricalDataAdapter`.

- SGX:
  - Keep the existing SGX securities parser.
  - Add `fetch_sgx_securities()`.
  - Add `parse_sgx_corporate_actions()`.
  - Add `parse_sgx_delistings()`.
  - Add `SgxMarketDataAdapter`.

- SET:
  - Park outside the current three-country universe.
  - Keep parser work dormant unless market scope expands.

- IDX:
  - Add `fetch_idx_stock_list()`.
  - Add `parse_idx_stock_list()`.
  - Add `parse_idx_corporate_actions()`.
  - Add `IdxDataServicesAdapter`.

- PSE:
  - Removed from the active project universe because we cannot trade it.
  - Do not add to default extraction or backtesting queues.

- Vietnam:
  - Removed from the active project universe.
  - Do not add to default extraction or backtesting queues.

### North Asia Tickets

- Add `src/quant_lab/data/north_asia_sources.py`.
- Add `JQuantsJapanAdapter`.
- Do not add or operate China/Taiwan provider routes while those markets remain outside the project mandate.
- Add `TwseOfficialAdapter`.
- Add `TpexOfficialAdapter`.
- Add `KrxDataPortalAdapter`.
- Add `StooqCsvAdapter` for smoke-test fallback where appropriate.

### US And Global Tickets

- Keep Nasdaq Trader current snapshots for discovery only.
- Add strict US common-stock filter separate from ADR/REIT/ordinary-share sleeves.
- Add `SharadarAdapter`, `CRSPFlatFileAdapter`, or `PolygonAdapter` behind the
  vendor adapter contract.
- Introduce stable `sid` keys from PERMNO, FIGI, vendor permanent ID, or an
  internal synthetic ID.
- Keep global ETF proxies separate from local equity universes.

### Maths Engine Tickets

- Add nested model selection around CPCV and final holdout.
- Add a stronger Probability of Backtest Overfitting report.
- Replace full-sample regime thresholds with prior-only rolling regime
  thresholds.
- Add exchange-specific TCA/capacity models.
- Add ML/meta-labeling only after Tier 2 data exists.
- Add event-feature families only through `available_time_utc` joins:
  earnings surprise, analyst revision diffusion, recommendation changes,
  guidance changes, and transcript/call metadata.

## Provider Strategy

### Free/Official First Where Practical

- Nasdaq Trader for current US discovery.
- SGX, PSE, HNX/UPCoM, TWSE, TPEx, JPX, HKEX, SSE, SZSE, KRX official listing
  sources for current security masters.
- Yahoo only for capped smoke tests and tradable ETF proxies.

### Vendor Needed For Serious Research

Markets where a licensed or strongly maintained vendor path is likely required:

- US point-in-time equities and delistings.
- Bursa full history.
- IDX.
- SET only if market scope expands.
- PSE and Vietnam are removed from the active project universe.
- Hong Kong full history.
- Corporate actions across most Asian markets.

Refinitiv/LSEG should become the first licensed provider path for these markets
where entitlements permit it. Yahoo remains a discovery and smoke-test source,
not the final source for capital allocation.

## Storage Contract

Long/narrow OHLCV schema:

```text
date
sid
ticker
local_symbol
exchange
country
currency
open
high
low
close
adj_close
volume
provider
adjustment
run_id
fetched_at_utc
```

Partition path:

```text
artifacts/data_lake/prices/provider=<provider>/interval=<interval>/adjustment=<mode>/country=<country>/exchange=<exchange>/year=<yyyy>/month=<mm>/
```

## Backtest Eligibility Rule

Every backtest artifact must declare:

- Data tier.
- Listing source.
- Price source.
- Adjustment method.
- Delisting coverage.
- Corporate-action coverage.
- Feature source availability timestamp policy.
- Whether the point-in-time feature audit passed.
- Missing-data threshold.
- Survivorship-bias status.
- Promotion eligibility.

If the artifact is Tier 0 or Tier 1, it can be used for research discovery only.
