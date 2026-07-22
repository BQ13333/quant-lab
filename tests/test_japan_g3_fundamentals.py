from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g3_fundamentals import (
    extract_japan_g3_fundamentals_qa,
    write_japan_g3_fundamentals_qa_artifacts,
)


class FakeClient:
    def get_data(self, universe, fields, parameters):
        base = fields[0]
        label = {
            "TR.Revenue": "Revenue",
            "TR.BankTotalRevenue": "Bank Total Revenue",
            "TR.GrossProfit": "Gross Profit",
            "TR.NetInterestIncome": "Net Interest Income",
            "TR.InterestIncomeBank": "Interest Income",
            "TR.EPSActValue": "EPS",
            "TR.BookValuePerShare": "Book Value Per Share",
        }[base]
        return pd.DataFrame(
            {
                "Instrument": list(universe),
                label: [100.0 + index for index, _ in enumerate(universe)],
                "Calc Date": ["2025-05-08"] * len(universe),
                "Period End Date": ["2025-03-31"] * len(universe),
                "Financial Period Absolute": ["FY2025"] * len(universe),
            }
        )


def test_extracts_calc_dates_and_applies_conservative_known_at() -> None:
    records = extract_japan_g3_fundamentals_qa(
        FakeClient(),
        rics=["7203.T", "6758.T"],
        start_date="2024-01-01",
        end_date="2026-07-06",
        frequencies=["FY"],
    )

    assert set(records["field_role"]) == {
        "revenue",
        "gross_profit",
        "net_interest_income",
        "interest_income",
        "eps",
        "book_value",
    }
    assert len(records) == 12
    assert records["available_time_utc"].notna().all()
    assert not records["pit_timestamp_violation"].any()
    assert records["available_time_utc"].min() > pd.Timestamp("2025-05-08", tz="UTC")


def test_qa_artifact_is_never_scorecard_eligible(tmp_path: Path) -> None:
    records = extract_japan_g3_fundamentals_qa(
        FakeClient(),
        rics=["7203.T"],
        start_date="2024-01-01",
        end_date="2026-07-06",
        frequencies=["FY"],
    )
    artifacts = write_japan_g3_fundamentals_qa_artifacts(
        records,
        artifact_root=tmp_path,
        run_id="JPN-G3-QA",
        requested_rics=["7203.T"],
        start_date="2024-01-01",
        end_date="2026-07-06",
        frequencies=["FY"],
        session_name="fake",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed_partial"
    assert manifest["scorecard_eligible"] is False
    assert manifest["partial"] is True
    assert manifest["pit_timestamp_violations"] == 0
    assert manifest["artifact_sha256"]["fundamentals_qa"]
