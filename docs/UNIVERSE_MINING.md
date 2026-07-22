# Universe Mining

Backtests need a broad opportunity set. A hand-picked ticker list can answer
whether a strategy worked on that list, but not whether the strategy survives
across the market.

## Current Sources

### US Listed Securities

`quant_lab.data.universe_mining` mines current US listings from Nasdaq Trader:

- `nasdaqlisted.txt`
- `otherlisted.txt`

The output is normalized into:

- `ticker`
- `yahoo_ticker`
- `name`
- `exchange`
- `country`
- `currency`
- `asset_type`
- `security_type`
- `is_etf`
- `is_stock_like`
- `is_test_issue`
- `source`
- `mined_at_utc`

## Artifacts

`examples/run_universe_mining_us.py` writes:

- `nasdaqlisted.txt`
- `otherlisted.txt`
- `listings.csv`
- `manifest.csv`

With `--download-prices`, it also writes:

- `prices.parquet`
- `prices.ohlcv.parquet`
- `prices.coverage.csv`

The price download defaults to a cap to avoid sending thousands of Yahoo
requests by accident.

By default, the miner keeps stock-like securities only: common stock, ordinary
shares, ADRs, REITs, and otherwise unclassified equity rows. It excludes ETFs,
ETNs, warrants, rights, units, preferreds, debt-like instruments, and funds
unless flags request a broader universe.

## Critical Caveat

This is a current listed-security snapshot. It is not a point-in-time universe
and does not include delisted securities. Production-grade historical research
still needs a vendor source with delisted names, corporate-action history, and
listing-date/de-listing-date availability.

Use mined universes to expand discovery and coverage. Use point-in-time vendor
data before trusting historical performance.

## Asia Sources

`examples/run_universe_mining_asia.py` currently supports:

- Taiwan adapter compatibility is retained for historical artifact reads, but Taiwan is excluded from the active quant-project universe.
- Taipei Exchange open data: TPEx tickers mapped to `.TWO`
- Korea KIND/KRX listed company download: KOSPI mapped to `.KS`, KOSDAQ mapped to `.KQ`
- Hong Kong Exchange securities list: HKEX equities/REITs mapped to `.HK`
- Japan Exchange Group listed issues file: JPX stock rows mapped to `.T`
- Shanghai Stock Exchange query endpoint: SSE A-shares mapped to `.SS`
- Shenzhen Stock Exchange report endpoint: SZSE A-shares mapped to `.SZ`

The Asia miner writes:

- raw exchange source files
- `listings.csv`
- `manifest.csv`

With `--download-prices`, it also writes Yahoo OHLCV Parquet and coverage files
for a capped sample or larger batch.

## ASEAN Gap

The repo has a curated ASEAN seed universe and now has first-pass public
adapters for several ASEAN venues:

- SGX securities v1.1 JSON endpoint, excluding bonds, mapped to `.SI`
- PSE EDGE company-directory AJAX pages mapped to `.PS`
- HNX listed-stock AJAX endpoint mapped to `.VN`
- UPCoM AJAX endpoint from HNX mapped to `.VN`

Run:

```bash
uv run python examples/run_universe_mining_asean.py --sources sgx,pse,hnx,upcom
```

With a capped price smoke test:

```bash
uv run python examples/run_universe_mining_asean.py --sources sgx,pse,hnx,upcom --download-prices --max-tickers 100 --period 1y
```

Remaining ASEAN gaps:

- Bursa Malaysia official list is available as a public PDF, but direct fetches
  are Cloudflare-challenged from this environment and the file is a weak
  code/name seed rather than a proper security master.
- IDX public pages/API endpoints are Cloudflare-protected from this environment.
- SET has an official listed-company Excel export and the repo includes a
  parser, but direct download is blocked by the exchange edge service here.
- HOSE still needs a stable official adapter or a licensed/vendor source.

## ASEAN Price Coverage Caveat

The ASEAN listing adapters are security-master discovery tools. Price coverage
is a separate vendor problem.

Yahoo Finance currently works for many Singapore `.SI` tickers, but the
default ASEAN research universe is now Malaysia, Singapore, and Indonesia only.
Thailand is parked outside the current scope; Philippines and Vietnam are
removed from the active project universe.

For broad ASEAN backtesting, use this priority:

- Use Yahoo for SGX smoke tests and selected liquid Singapore names.
- Attach a licensed or exchange-approved price vendor for Bursa and IDX before
  serious historical conclusions.
- Keep the listing snapshot and price manifests separate so missing price
  coverage does not silently shrink the research universe.

For production, prefer a vendor security master that includes delisted names,
corporate actions, and listing history.
