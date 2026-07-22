# Point-In-Time Backtesting Standard

The research platform must assume every non-price dataset can leak unless it
proves when the market could know the data.

## Mandatory Rule

No feature may enter a model unless it has:

- `sid`: stable security identifier.
- `event_time_utc`: when the economic event happened, if applicable.
- `release_time_utc`: when the issuer, broker, exchange, or vendor says it was
  released, if available.
- `available_time_utc`: when our system could have known it.
- `source`: vendor or exchange source.
- `source_dataset`: dataset/table name.
- `vendor_timestamp_policy`: explanation of the timestamp semantics.
- `ingested_at_utc`: when our system stored it.

The feature builder uses `available_time_utc`, not fiscal period end, not report
date, not file date, and not the latest restated value.

## Event Types

### Earnings Releases

Store fiscal period data separately from release timing:

```text
sid
fiscal_period_end
event_time_utc
release_time_utc
available_time_utc
eps_actual
revenue_actual
surprise_vs_consensus
source
source_dataset
```

After-market releases are only usable after their `available_time_utc`. If a
strategy makes a decision at the close, an after-close release from that same
date is future information.

### Analyst Estimate Revisions

Store every estimate/recommendation update as an event, not as a current
snapshot:

```text
sid
broker_id
analyst_id
estimate_measure
fiscal_period
old_value
new_value
revision_pct
recommendation_old
recommendation_new
release_time_utc
available_time_utc
source
source_dataset
```

Features can then be built as prior-only windows:

- 1-day/5-day/20-day revision count.
- Revision diffusion: upgrades minus downgrades divided by total revisions.
- Weighted revision magnitude.
- Recommendation-change count.
- Consensus-change momentum.

### Earnings Calls, Transcripts, And Guidance

Calls and transcripts must use the first availability timestamp of the text or
metadata. Final edited transcripts are not automatically point-in-time unless
the vendor supplies historical availability or version timestamps.

### Price And Volume

Price/volume features are only safe when the decision time is explicit. A model
using today's close can trade no earlier than the next executable session unless
intraday execution data is used.

## Data Tiers

### Tier 0: Discovery

Current listings and free-data smoke tests. Never production-grade.

### Tier 1: Event Smoke Test

Event data has `available_time_utc`, but may not have delistings, complete
history, or vendor restatement controls. Useful for proving pipelines.

### Tier 2: Point-In-Time Research

Requirements:

- Point-in-time security master.
- Delistings.
- Corporate actions.
- Raw and adjusted price series.
- Timestamped fundamentals, earnings releases, estimate revisions, and
  recommendation changes.
- Documented vendor timestamp semantics.
- Feature audit output from `assert_no_future_feature_leakage`.

### Tier 3: Production/Paper-Trading Ready

Tier 2 plus execution data, borrow/shortability, fees, taxes, slippage model,
broker reconciliation, and live monitoring.

## Code Contract

Use `quant_lab.data.point_in_time`:

```python
from quant_lab.data.point_in_time import (
    PITFeatureSpec,
    build_point_in_time_event_features,
    point_in_time_asof_join,
)

features = build_point_in_time_event_features(
    decisions=decisions,
    events=events,
    specs=[
        PITFeatureSpec(
            name="revision_count_20d",
            event_type="analyst_revision",
            value_col="revision_pct",
            lookback=pd.Timedelta(days=20),
            agg="count",
        ),
        PITFeatureSpec(
            name="revision_sum_20d",
            event_type="analyst_revision",
            value_col="revision_pct",
            lookback=pd.Timedelta(days=20),
            agg="sum",
        ),
    ],
)
```

The builder only uses events with:

```text
available_time_utc <= decision_time_utc - lag
```

## Promotion Blocks

A strategy cannot be promoted if:

- Any feature lacks `available_time_utc`.
- Fiscal period end is used as a release timestamp.
- Latest/restated fundamentals are used without point-in-time versioning.
- Analyst estimates are current snapshots instead of historical revision
  events.
- After-close events are available to same-close decisions.
- Security membership is current-survivor only.
- Delisted names are absent from the eligible universe.
- Backtest labels overlap the training features without purge/embargo.

## LSEG / Refinitiv Notes

LSEG I/B/E/S is the relevant family for analyst estimates, consensus, actuals,
aggregates, guidance, and recommendation history. StarMine Analyst Revisions
Model is explicitly built around sell-side estimate and recommendation
revisions. The Python `lseg.data.get_history` access layer can retrieve pricing
history and fundamental/reference data history when the account is entitled.

Exact fields and timestamp semantics must be confirmed in Workspace Data Item
Browser or with LSEG support before a dataset is marked Tier 2.
