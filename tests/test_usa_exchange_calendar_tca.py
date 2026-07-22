from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.usa_exchange_calendar_tca import (
    HALT_COLUMNS,
    BuiltinUSMarketRulesProvider,
    TCAAssumptions,
    build_exchange_calendar,
    build_tca_assumption_grid,
    write_usa_exchange_calendar_tca_artifacts,
)


class AuthoritativeTestProvider:
    def schedule(self, exchange: str, start_date: date, end_date: date):
        schedule = BuiltinUSMarketRulesProvider().schedule(exchange, start_date, end_date)
        return replace(
            schedule,
            calendar_code=f"{exchange}_TEST_AUTH",
            source_provider="authoritative_test_calendar",
            source_version="test",
            source_quality="authoritative_calendar_library",
            authoritative=True,
        )


def test_builtin_calendar_covers_nyse_nasdaq_holidays_early_closes_and_special_closures():
    calendar, schedules = build_exchange_calendar(
        "2012-10-26",
        "2012-11-24",
        provider=BuiltinUSMarketRulesProvider(),
    )

    assert set(calendar["exchange"]) == {"NYSE", "NASDAQ"}
    assert len(calendar) == 60
    assert all(not schedule.authoritative for schedule in schedules)

    nyse = calendar[calendar["exchange"].eq("NYSE")].set_index("date")
    assert bool(nyse.loc["2012-10-29", "is_session"]) is False
    assert nyse.loc["2012-10-29", "closure_type"] == "special_closure"
    assert nyse.loc["2012-10-29", "event_name"] == "hurricane_sandy_market_closure"
    assert bool(nyse.loc["2012-11-22", "is_session"]) is False
    assert nyse.loc["2012-11-22", "event_name"] == "thanksgiving_day"
    assert bool(nyse.loc["2012-11-23", "is_early_close"]) is True
    assert int(nyse.loc["2012-11-23", "scheduled_minutes"]) == 210


def test_builtin_calendar_marks_juneteenth_and_pre_independence_early_close():
    calendar, _ = build_exchange_calendar(
        "2024-06-19",
        "2024-07-04",
        provider=BuiltinUSMarketRulesProvider(),
    )
    nyse = calendar[calendar["exchange"].eq("NYSE")].set_index("date")

    assert nyse.loc["2024-06-19", "event_name"] == "juneteenth"
    assert bool(nyse.loc["2024-06-19", "is_session"]) is False
    assert bool(nyse.loc["2024-07-03", "is_early_close"]) is True
    assert nyse.loc["2024-07-04", "closure_type"] == "exchange_holiday"


def test_tca_grid_is_side_aware_and_validates_participation():
    assumptions = TCAAssumptions(
        commission_bps=0.20,
        exchange_fee_bps=0.30,
        regulatory_sell_fee_bps=0.10,
        half_spread_bps=1.50,
        impact_bps_at_reference_participation=3.0,
        reference_participation_rate=0.05,
        max_participation_rate=0.20,
    )
    grid = build_tca_assumption_grid(assumptions).set_index(["exchange", "side"])

    assert len(grid) == 4
    assert grid.loc[("NYSE", "buy"), "estimated_cost_bps_at_reference"] == pytest.approx(5.0)
    assert grid.loc[("NYSE", "sell"), "estimated_cost_bps_at_reference"] == pytest.approx(5.1)
    assert grid.loc[("NASDAQ", "buy"), "impact_bps_at_max_participation"] == pytest.approx(6.0)
    with pytest.raises(ValueError, match="cannot be below"):
        TCAAssumptions(reference_participation_rate=0.20, max_participation_rate=0.10).validate()


def test_writer_emits_checksummed_incomplete_placeholders_without_official_sources(tmp_path):
    artifacts = write_usa_exchange_calendar_tca_artifacts(
        artifact_root=tmp_path,
        run_id="FX-CALENDAR-TCA-USA-TEST",
        start_date="2024-07-01",
        end_date="2024-07-05",
        calendar_backend="builtin",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    quality = artifacts.evidence_quality.set_index("component")

    assert manifest["status"] == "incomplete"
    assert manifest["production_ready"] is False
    assert manifest["scorecard_eligible"] is False
    assert manifest["does_not_assert_5_of_5"] is True
    assert manifest["halt_suspensions"]["placeholder_only"] is True
    assert manifest["halt_suspensions"]["no_events_claimed"] is False
    assert manifest["quality_fields"]["complete_components"] == 0
    assert set(manifest["quality_fields"]["incomplete_components"]) == {
        "exchange_calendars",
        "halt_suspension_events",
        "tca_assumptions",
    }
    assert quality.loc["exchange_calendars", "status"] == "incomplete_non_authoritative_source"
    assert quality.loc["halt_suspension_events", "status"] == "incomplete_missing_official_source"
    assert quality.loc["tca_assumptions", "status"] == "incomplete_uncalibrated_assumptions"
    assert tuple(artifacts.halt_suspensions.columns) == HALT_COLUMNS
    assert artifacts.halt_suspensions.empty
    for name, path_text in manifest["artifacts"].items():
        digest = hashlib.sha256(Path(path_text).read_bytes()).hexdigest()
        assert manifest["sha256"][name] == digest


def test_writer_requires_all_three_quality_components_for_production_ready(tmp_path):
    halt_path = tmp_path / "official_halts.csv"
    pd.DataFrame(
        [
            {
                "event_id": "nasdaq-20240702-1",
                "exchange": "NASDAQ",
                "symbol": "NA",
                "halt_start_utc": "2024-07-02T14:00:00Z",
                "halt_end_utc": "2024-07-02T14:15:00Z",
                "event_type": "trading_halt",
                "reason_code": "T1",
                "source_name": "official_test_feed",
                "source_record_id": "1",
                "observed_at_utc": "2024-07-02T14:00:01Z",
            }
        ]
    ).to_csv(halt_path, index=False)
    assumptions = TCAAssumptions(
        source_quality="empirically_calibrated_execution_sample",
        source_name="validated_test_execution_sample",
    )

    artifacts = write_usa_exchange_calendar_tca_artifacts(
        artifact_root=tmp_path,
        run_id="COMPLETE-EVIDENCE-TEST",
        start_date="2024-07-01",
        end_date="2024-07-05",
        calendar_provider=AuthoritativeTestProvider(),
        tca_assumptions=assumptions,
        halt_suspension_input=halt_path,
        halt_source_quality="official_exchange_or_regulator",
        halt_source_name="official_test_feed",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["production_ready"] is True
    assert manifest["scorecard_eligible"] is True
    assert manifest["does_not_assert_5_of_5"] is False
    assert manifest["quality_fields"]["complete_components"] == 3
    assert manifest["quality_fields"]["incomplete_components"] == []
    assert artifacts.evidence_quality["production_ready"].all()
    assert artifacts.halt_suspensions.iloc[0]["symbol"] == "NA"


def test_writer_refuses_to_overwrite_an_existing_artifact(tmp_path):
    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "NO-OVERWRITE",
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "calendar_backend": "builtin",
    }
    write_usa_exchange_calendar_tca_artifacts(**kwargs)

    with pytest.raises(FileExistsError, match="not empty"):
        write_usa_exchange_calendar_tca_artifacts(**kwargs)
