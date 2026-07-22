# Refinitiv / LSEG Data Setup

Do not paste Refinitiv, LSEG, Workspace, Eikon, or platform credentials into
Codex, Claude, Kimi, GitHub issues, notebooks, or committed files.

The quant lab reads Refinitiv data through the optional LSEG Data Library for
Python package, `lseg-data`. Credentials and app keys must stay in the local
LSEG configuration layer.

## Install

```bash
uv sync --extra refinitiv
```

## Recommended Credential Paths

### Desktop Session

If LSEG Workspace or Eikon is running on the same machine, configure a desktop
session and let the library authenticate through the desktop application:

```python
import lseg.data as ld
ld.open_session(name="desktop.workspace")
```

This is the safest route for local research because the project does not need
your username or password.

### Platform Session

If using LSEG Data Platform directly, create `lseg-data.config.json` outside the
repo, for example:

```text
~/Private/quant_lab_lseg/lseg-data.config.json
```

Then pass that path when opening a session:

```python
import lseg.data as ld
ld.open_session(name="platform.ldp", config_name="~/Private/quant_lab_lseg/lseg-data.config.json")
```

Keep the config file out of this repository and out of Google Drive unless the
folder is encrypted and access-controlled.

## Adapter Usage

Use the adapter directly:

```python
from quant_lab.data.vendor_adapters import RefinitivAdapter

adapter = RefinitivAdapter(
    session_name="desktop.workspace",
    # config_name="/Users/you/Private/quant_lab_lseg/lseg-data.config.json",
)

history = adapter.download(
    tickers=["IBM.N", "7203.T"],
    start="2021-01-01",
    end="2026-07-01",
    fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"],
)
```

Or plug it into the existing data-lake batch backfiller:

```python
from quant_lab.data.data_lake import backfill_prices_in_batches
from quant_lab.data.vendor_adapters import RefinitivAdapter

adapter = RefinitivAdapter(session_name="desktop.workspace")

result = backfill_prices_in_batches(
    listings=listings,
    root="artifacts/data_lake",
    run_id="PRICE-BACKFILL-REFINITIV-TEST",
    period="5y",
    interval="1d",
    batch_size=50,
    provider="refinitiv",
    adjustment="vendor_default",
    data_tier="bronze",
    write_long_ohlcv=True,
    downloader=adapter.download_price_history,
)
```

## Default Equity OHLCV Fields

The adapter maps the common LSEG historical pricing fields to the internal OHLCV
shape:

```text
Open   -> OPEN_PRC
High   -> HIGH_1
Low    -> LOW_1
Close  -> TRDPRC_1
Volume -> ACVOL_UNS
```

Field availability can differ by asset class, venue, and entitlement. Use LSEG
Workspace Data Item Browser or LSEG support to confirm the correct fields for
each market before treating a backtest as production-grade.

## Field And RIC Validation

Before a large Refinitiv extraction, run a controlled field probe:

```bash
uv run python examples/run_refinitiv_field_probe.py \
  --listings artifacts/data_extraction/ALL-SOURCES-PUBLIC-20260705-REFINITIV-QUEUED/combined_listings.csv \
  --run-id REFINITIV-FIELD-PROBE-CORE \
  --sample-per-market 1 \
  --count 60 \
  --progress-every 50
```

The probe writes:

```text
field_probe_results.csv
field_probe_summary.csv
recommended_field_map.csv
field_catalog.csv
ric_validation_manifest.csv
```

Use `ric_validation_manifest.csv` as the gate for live vendor extraction:

```bash
uv run python examples/run_all_source_extraction.py \
  --download-refinitiv \
  --refinitiv-validation-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-CORE/ric_validation_manifest.csv
```

The extraction runner intentionally refuses broad Refinitiv downloads from
inferred RICs unless `--allow-inferred-refinitiv-rics` is passed for a small
smoke test. When `field_catalog.csv` is present next to the validation manifest,
the runner uses it automatically so each market can request the close/open/high/
low/volume fields validated for that market. This matters because Yahoo-style
tickers can differ from Refinitiv RICs. For example, the live desktop probe
validated `MBBM.KL` and `DBSM.SI`, while the Yahoo-style `1155.KL` and `D05.SI`
did not resolve as Refinitiv historical-price RICs.

## Validated Symbol Map Workflow

For a broader research universe, build a validated symbol map before extracting
prices:

```bash
uv run python examples/run_refinitiv_symbol_map.py \
  --listings artifacts/data_extraction/ALL-SOURCES-PUBLIC-20260705-REFINITIV-QUEUED/combined_listings.csv \
  --run-id REFINITIV-SYMBOL-MAP-L1 \
  --sample-per-market 5 \
  --max-symbols 120 \
  --count 5
```

This writes:

```text
symbol_candidates.csv
symbol_validation.csv
symbol_map.csv
manifest.json
```

Then use the validated `symbol_map.csv` directly for a Refinitiv backfill:

```bash
uv run python examples/run_all_source_extraction.py \
  --regions "" \
  --download-refinitiv \
  --refinitiv-symbol-map-csv artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1/symbol_map.csv \
  --refinitiv-field-catalog-csv artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-CORE/field_catalog.csv \
  --max-refinitiv-tickers 120 \
  --batch-size 15 \
  --period 1y \
  --interval 1d
```

The symbol-map path is preferred over raw inferred RICs. It can include curated
seed rows for markets whose public listings do not map cleanly to Refinitiv
RICs, but those rows still must pass live validation before they enter a price
backfill.

## Source Notes

LSEG documentation describes `lseg.data` sessions through `ld.open_session()`,
including named desktop/platform sessions and optional `config_name`. It also
documents `ld.get_history(universe=..., fields=..., start=..., end=...,
interval=...)` for historical pricing.
