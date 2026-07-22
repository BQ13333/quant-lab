from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.data.corporate_actions_pit import (
    build_corporate_action_event_plan,
    execute_corporate_action_event_plan,
    normalize_corporate_action_event_frame,
    refinitiv_field_expression,
    stage_corporate_action_event_plan,
)


class FakeCorporateActionsLSEG:
    def __init__(self):
        self.calls = []

    def get_data(self, universe, fields, parameters=None):
        self.calls.append((tuple(universe), tuple(fields), parameters))
        rows = []
        for ric in universe:
            rows.append(
                {
                    "Instrument": ric,
                    "Dividend Announcement Date": "2022-02-01",
                    "Dividend Ex Date": "2022-03-01",
                    "Dividend Unadjusted Gross": 0.10,
                    "Dividend Currency": "MYR",
                }
            )
        return pd.DataFrame(rows)


class EmptyCorporateActionsLSEG:
    def __init__(self):
        self.calls = 0

    def get_data(self, universe, fields, parameters=None):
        self.calls += 1
        return pd.DataFrame()


class ThrottledCorporateActionsLSEG:
    def __init__(self):
        self.calls = 0

    def get_data(self, universe, fields, parameters=None):
        self.calls += 1
        raise RuntimeError("429 Too many requests")


def _selected_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "Malaysia:AAA",
                "country": "Malaysia",
                "selected_for_u2": True,
                "selected_for_u3": True,
            },
            {
                "ticker": "BBB.KL",
                "refinitiv_ric": "BBB.KL",
                "sid": "Malaysia:BBB",
                "country": "Malaysia",
                "selected_for_u2": True,
                "selected_for_u3": False,
            },
        ]
    )


def test_refinitiv_field_expression_adds_date_parameters():
    expression = refinitiv_field_expression(
        "TR.DivExDate",
        {"SDate": "2021-01-01", "EDate": "2026-07-06", "DateType": "ED"},
    )

    assert expression == "TR.DivExDate(SDate=2021-01-01,EDate=2026-07-06,DateType=ED)"


def test_corporate_action_event_plan_uses_parameterized_grouped_fields(tmp_path):
    plan = build_corporate_action_event_plan(
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-PLAN",
        start_date="2021-01-01",
        end_date="2026-07-06",
        tier="u2",
        batch_size=10,
        event_families=["dividend"],
    )

    assert len(plan) == 1
    assert plan.loc[0, "route"] == "get_data_event"
    assert "TR.DivExDate" in plan.loc[0, "field_expressions"]
    assert "TR.DivAnnouncementDate" in plan.loc[0, "candidate_fields"]
    assert '"DateType": "ED"' in plan.loc[0, "route_parameters"]


def test_normalize_corporate_action_events_marks_alpha_usable_only_with_known_at_date():
    frame = pd.DataFrame(
        [
            {"Instrument": "AAA.KL", "Dividend Announcement Date": "2022-02-01", "Dividend Ex Date": "2022-03-01"},
            {"Instrument": "BBB.KL", "Dividend Announcement Date": pd.NA, "Dividend Ex Date": "2022-03-01"},
        ]
    )
    plan_row = {
        "event_family": "dividend",
        "field_role": "dividend_event",
        "route_parameters": '{"SDate":"2021-01-01"}',
    }

    normalized = normalize_corporate_action_event_frame(
        frame,
        rics=["AAA.KL", "BBB.KL"],
        plan_row=plan_row,
        universe=_selected_universe(),
        source_loaded_at_utc="2026-07-08T00:00:00+00:00",
    )

    assert len(normalized) == 2
    assert normalized["event_key"].nunique() == 2
    assert bool(normalized.loc[normalized["requested_ric"].eq("AAA.KL"), "alpha_usable"].iloc[0]) is True
    assert bool(normalized.loc[normalized["requested_ric"].eq("BBB.KL"), "alpha_usable"].iloc[0]) is False


def test_execute_corporate_action_event_plan_writes_quality(tmp_path):
    plan = build_corporate_action_event_plan(
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-RUN",
        start_date="2021-01-01",
        end_date="2026-07-06",
        tier="u2",
        batch_size=10,
        event_families=["dividend"],
    )

    artifacts = execute_corporate_action_event_plan(
        FakeCorporateActionsLSEG(),
        plan=plan,
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-RUN",
        sleep_seconds=0,
    )

    assert artifacts.manifest_path.exists()
    assert artifacts.quality_path.exists()
    assert artifacts.manifest["status"].eq("completed").all()
    assert artifacts.quality.loc[0, "event_rows"] == 2
    assert artifacts.quality.loc[0, "alpha_usable_rows"] == 2


def test_successful_empty_corporate_action_response_is_terminal_negative_evidence(tmp_path):
    plan = build_corporate_action_event_plan(
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-EMPTY",
        start_date="2021-01-01",
        end_date="2026-07-06",
        tier="u2",
        batch_size=10,
        event_families=["capital_change"],
    )
    vendor = EmptyCorporateActionsLSEG()

    artifacts = execute_corporate_action_event_plan(
        vendor,
        plan=plan,
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-EMPTY",
        sleep_seconds=0,
    )

    run = json.loads(artifacts.run_manifest_path.read_text())
    assert vendor.calls == 1
    assert artifacts.manifest.loc[0, "status"] == "completed_empty"
    assert bool(artifacts.manifest.loc[0, "response_success"]) is True
    assert artifacts.quality.loc[0, "completed_rows"] == 1
    assert artifacts.quality.loc[0, "empty_rows"] == 1
    assert artifacts.quality.loc[0, "min_coverage"] == 1.0
    assert run["completed_rows"] == 1
    assert run["empty_rows"] == 1
    assert run["status"] == "complete"
    assert run["official_source"] is True


def test_corporate_action_rate_limit_halts_after_one_plan_row(tmp_path):
    plan = build_corporate_action_event_plan(
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-429",
        start_date="2021-01-01",
        end_date="2026-07-06",
        tier="u2",
        batch_size=10,
    )
    vendor = ThrottledCorporateActionsLSEG()

    artifacts = execute_corporate_action_event_plan(
        vendor,
        plan=plan,
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="CA-429",
        sleep_seconds=0,
    )

    run = json.loads(artifacts.run_manifest_path.read_text())
    assert vendor.calls == 1
    assert len(artifacts.manifest) == 1
    assert artifacts.manifest.loc[0, "status"] == "failed_retryable_provider"
    assert run["halted_early"] is True
    assert run["provider_halt"]["plan_row_id"] == plan.iloc[0]["plan_row_id"]


def test_corporate_action_plan_is_write_once_and_checksum_guarded(tmp_path):
    path = tmp_path / "event_plan.csv"
    plan = pd.DataFrame({"plan_row_id": ["A"], "ric_list": ["AAA.T"]})
    digest = stage_corporate_action_event_plan(plan, path)
    assert path.with_suffix(".sha256").read_text().startswith(digest)
    assert stage_corporate_action_event_plan(plan.copy(), path) == digest

    changed = pd.DataFrame({"plan_row_id": ["A"], "ric_list": ["BBB.T"]})
    with pytest.raises(ValueError, match="plan drift"):
        stage_corporate_action_event_plan(changed, path)
