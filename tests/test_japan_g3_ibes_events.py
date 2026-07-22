from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g3_ibes_events import (
    derive_analyst_revision_events,
    extract_analyst_snapshots_qa,
    extract_consensus_history_qa,
    extract_earnings_actuals_qa,
    write_japan_g3_ibes_events_qa_artifacts,
)


class FakeClient:
    def get_history(self, universe, fields, start, end, interval):
        field = fields[0]
        frame = pd.DataFrame(
            {ric: [1.0, 2.0] for ric in universe},
            index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
        )
        frame.columns.name = field
        return frame

    def get_data(self, universe, fields, parameters):
        if any("RecEstValue" in field for field in fields):
            return pd.DataFrame(
                {
                    "Instrument": [universe[0]] * 3,
                    "Analyst Name": ["A"] * 3,
                    "Broker Name": ["Broker"] * 3,
                    "Calc Date": ["2025-01-02", "2025-01-03", "2025-01-04"],
                    "Date": ["2024-12-01", "2024-12-01", "2025-01-04 12:00:00"],
                    "Standard Rec (1-5) - Broker Estimate": [2, 2, 1],
                }
            )
        if any("TPEstValue" in field for field in fields):
            return pd.DataFrame(
                {
                    "Instrument": [universe[0]] * 3,
                    "Analyst Name": ["A"] * 3,
                    "Broker Name": ["Broker"] * 3,
                    "Calc Date": ["2025-01-02", "2025-01-03", "2025-01-04"],
                    "Date": ["2024-12-01", "2024-12-01", "2025-01-04 12:00:00"],
                    "Target Price - Broker Estimate": [100, 100, 120],
                }
            )
        base = fields[0]
        label = {
            "TR.EPSActValue": "Earnings Per Share - Actual",
            "TR.EPSActSurprise": "Earnings Per Share - Actual Surprise",
        }[base]
        return pd.DataFrame(
            {
                "Instrument": list(universe),
                label: [10.0] * len(universe),
                "Calc Date": ["2025-05-08"] * len(universe),
                "Period End Date": ["2025-03-31"] * len(universe),
                "Financial Period Absolute": ["FY2025"] * len(universe),
            }
        )


def test_revision_transform_drops_repeated_daily_snapshots() -> None:
    snapshots = extract_analyst_snapshots_qa(
        FakeClient(), ["7203.T"], "2025-01-01", "2025-12-31"
    )
    events = derive_analyst_revision_events(snapshots)
    assert events["event_type"].tolist() == [
        "baseline",
        "recommendation_revision",
        "baseline",
        "target_price_revision",
    ]
    assert events["available_time_utc"].notna().all()
    assert not events["pit_timestamp_violation"].any()


def test_consensus_and_events_have_conservative_known_at() -> None:
    consensus = extract_consensus_history_qa(
        FakeClient(), ["7203.T"], "2025-01-01", "2025-12-31"
    )
    events = extract_earnings_actuals_qa(
        FakeClient(), ["7203.T"], "2025-01-01", "2025-12-31", frequencies=["FY"]
    )
    assert set(consensus["field_role"]) == {
        "eps_estimate",
        "revenue_estimate",
        "recommendation",
        "target_price",
    }
    assert set(events["field_role"]) == {"actual_eps", "surprise", "earnings_release"}
    assert consensus["available_time_utc"].notna().all()
    assert events["available_time_utc"].notna().all()


def test_qa_artifact_is_fail_closed(tmp_path: Path) -> None:
    client = FakeClient()
    consensus = extract_consensus_history_qa(client, ["7203.T"], "2025-01-01", "2025-12-31")
    snapshots = extract_analyst_snapshots_qa(client, ["7203.T"], "2025-01-01", "2025-12-31")
    revisions = derive_analyst_revision_events(snapshots)
    events = extract_earnings_actuals_qa(client, ["7203.T"], "2025-01-01", "2025-12-31", ["FY"])
    artifacts = write_japan_g3_ibes_events_qa_artifacts(
        tmp_path,
        "QA",
        consensus,
        snapshots,
        revisions,
        events,
        ["7203.T"],
        "2025-01-01",
        "2025-01-01",
        "2025-12-31",
        "fake",
        rejected_misaligned_event_rows=39,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_partial"
    assert manifest["scorecard_eligible"] is False
    assert manifest["raw_period_end_surprise_history_allowed"] is False
    assert manifest["pit_timestamp_violations"] == 0
    assert manifest["rejected_misaligned_event_rows"] == 39
    assert manifest["artifact_sha256"]["rejected_methodology"]
