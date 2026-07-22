from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.corporate_action_reconciliation import (
    build_reconciliation_quality,
    match_events_to_prices,
    write_corporate_action_reconciliation_artifacts,
)


def test_match_events_to_prices_flags_missing_dates_and_matches_next_price():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA.KL", "AAA.KL", "AAA.KL"],
            "country": ["Malaysia", "Malaysia", "Malaysia"],
            "close": [10.0, 11.0, 12.0],
            "adj_close": [10.0, 11.0, 12.0],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "1",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "event_family": "dividend",
                "field_role": "dividend_event",
                "event_date": "2024-01-03",
                "known_at_date": "2024-01-02",
                "Adjustment Factor": 2.0,
                "eligible_for_return_reconstruction": True,
                "alpha_usable": True,
            },
            {
                "event_key": "2",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "event_family": "capital_change",
                "field_role": "split_event",
                "event_date": pd.NA,
                "known_at_date": pd.NA,
                "eligible_for_return_reconstruction": False,
                "alpha_usable": False,
            },
        ]
    )

    matched = match_events_to_prices(events, prices)
    quality = build_reconciliation_quality(matched)

    assert matched.loc[0, "match_status"] == "matched"
    assert matched.loc[0, "raw_return_on_match"] == pytest.approx(0.10)
    assert matched.loc[0, "adjustment_factor"] == pytest.approx(2.0)
    assert matched.loc[1, "match_status"] == "missing_event_date"
    assert int(quality["valid_event_rows"].sum()) == 1
    assert int(quality["matched_event_rows"].sum()) == 1


def test_write_corporate_action_reconciliation_artifacts_is_diagnostic_until_returns_rebuilt(tmp_path):
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA.KL", "AAA.KL"],
            "country": ["Malaysia", "Malaysia"],
            "close": [10.0, 11.0],
            "adj_close": [10.0, 11.0],
        }
    )
    daily_path = tmp_path / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    event_dir = tmp_path / "events"
    event_leaf = event_dir / "events" / "event_family=dividend" / "country=malaysia"
    event_leaf.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_key": "1",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "event_family": "dividend",
                "field_role": "dividend_event",
                "event_date": "2024-01-03",
                "known_at_date": "2024-01-02",
                "eligible_for_return_reconstruction": True,
                "alpha_usable": True,
            }
        ]
    ).to_parquet(event_leaf / "batch.parquet", index=False)

    artifacts = write_corporate_action_reconciliation_artifacts(
        daily_features_path=daily_path,
        corporate_action_event_dir=event_dir,
        artifact_root=tmp_path / "recon",
        run_id="RECON",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.event_match_path.exists()
    assert manifest["status"] == "diagnostic_reconciliation_only"
    assert manifest["production_adjusted_return_ready"] is False
