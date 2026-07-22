from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.ibes_revisions_pit import (
    RESULT_COLUMNS,
    accepted_ibes_revision_time_map,
    build_ibes_revision_route_probe_queue,
    default_ibes_revision_parameter_grid,
    ibes_revision_timestamp_candidate_fields,
    probe_ibes_revision_timestamp_routes,
    validate_ibes_revision_date_frame,
    write_ibes_revision_route_probe_artifacts,
)


class FakeIBESClient:
    def __init__(self):
        self.calls = []

    def get_data(self, universe, fields, parameters=None):
        self.calls.append((tuple(universe), tuple(fields), dict(parameters or {})))
        field = fields[0]
        rows = []
        for index, ric in enumerate(universe):
            if field == "TR.EPSMean.date":
                rows.extend(
                    [
                        {"Instrument": ric, "Revision Date": "2024-02-01"},
                        {"Instrument": ric, "Revision Date": f"2024-02-0{index + 2}"},
                    ]
                )
            elif field == "TR.EPSMean.CalcDate":
                rows.append({"Instrument": ric, "Calc Date": parameters["SDate"]})
            elif field == "TR.RevenueMean.CalcDate":
                rows.append({"Instrument": ric, "Calc Date": parameters["EDate"]})
            elif field == "TR.RecMean.CalcDate":
                rows.append({"Instrument": ric, "Calc Date": "2026-07-08"})
            elif field == "TR.EPSMean":
                rows.append({"Instrument": ric, "EPS Mean": 1.25})
        return pd.DataFrame(rows)


def _selected_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "MBBM.KL",
                "refinitiv_ric": "MBBM.KL",
                "country": "Malaysia",
                "selected_for_u3": True,
                "adv_60_country_rank": 1,
            },
            {
                "ticker": "DBSM.SI",
                "refinitiv_ric": "DBSM.SI",
                "country": "Singapore",
                "selected_for_u3": True,
                "adv_60_country_rank": 1,
            },
            {
                "ticker": "BBCA.JK",
                "refinitiv_ric": "BBCA.JK",
                "country": "Indonesia",
                "selected_for_u3": True,
                "adv_60_country_rank": 1,
            },
        ]
    )


def test_ibes_revision_queue_builds_fy1_fy2_daily_parameterized_fields():
    fields = ibes_revision_timestamp_candidate_fields()
    queue = build_ibes_revision_route_probe_queue(start_date="2024-01-01", end_date="2026-07-06")

    assert "TR.EPSMean.date" in fields
    assert "TR.AnalystName.CalcDate" in fields
    assert "TR.EstimateDate" not in fields
    assert set(queue["dataset_id"]) == {"ibes_estimates_revisions"}
    assert set(queue["field_role"]) == {"revision_time"}
    assert set(queue["route"]) == {"get_data"}
    assert '{"EDate":"2026-07-06","Frq":"D","Period":"FY1","SDate":"2024-01-01"}' in set(queue["parameter_set"])
    assert '{"EDate":"2026-07-06","Frq":"D","Period":"FY2","SDate":"2024-01-01"}' in set(queue["parameter_set"])


def test_validate_rejects_sdate_edate_run_date_and_accepts_varying_source_dates():
    params = {"SDate": "2024-01-01", "EDate": "2026-07-06", "Frq": "D", "Period": "FY1"}
    base = {
        "dataset_id": "ibes_estimates_revisions",
        "field_role": "revision_time",
        "route": "get_data",
        "field": "TR.EPSMean.date",
        "parameter_set": "{}",
        "country": "MYS",
    }

    accepted = validate_ibes_revision_date_frame(
        pd.DataFrame({"Instrument": ["AAA", "AAA"], "Revision Date": ["2024-02-01", "2024-03-01"]}),
        start_date="2024-01-01",
        end_date="2026-07-06",
        parameters=params,
        run_date="2026-07-08",
        load_date="2026-07-08",
        base_row=base,
    )
    all_sdate = validate_ibes_revision_date_frame(
        pd.DataFrame({"Instrument": ["AAA", "BBB"], "Revision Date": ["2024-01-01", "2024-01-01"]}),
        start_date="2024-01-01",
        end_date="2026-07-06",
        parameters=params,
        run_date="2026-07-08",
        load_date="2026-07-08",
        base_row=base,
    )
    all_run_date = validate_ibes_revision_date_frame(
        pd.DataFrame({"Instrument": ["AAA", "BBB"], "Revision Date": ["2026-07-08", "2026-07-08"]}),
        start_date="2024-01-01",
        end_date="2026-07-06",
        parameters=params,
        run_date="2026-07-08",
        load_date="2026-07-08",
        base_row=base,
    )

    assert accepted["status"] == "accepted"
    assert accepted["unique_date_count"] == 2
    assert all_sdate["rejection_reason"] == "sentinel_date_pattern"
    assert all_run_date["rejection_reason"] == "sentinel_date_pattern"


def test_validate_accepts_pre_window_history_when_in_window_dates_are_source_backed():
    params = {"SDate": "2024-01-01", "EDate": "2026-07-06", "Frq": "D", "Period": "FY1"}
    base = {
        "dataset_id": "ibes_estimates_revisions",
        "field_role": "revision_time",
        "route": "get_data",
        "field": "TR.EPSMean.date",
        "parameter_set": "{}",
        "country": "MYS",
    }

    accepted = validate_ibes_revision_date_frame(
        pd.DataFrame(
            {
                "Instrument": ["AAA", "AAA", "AAA"],
                "Revision Date": ["2023-12-21", "2024-02-01", "2024-03-01"],
            }
        ),
        start_date="2024-01-01",
        end_date="2026-07-06",
        parameters=params,
        run_date="2026-07-08",
        load_date="2026-07-08",
        base_row=base,
    )
    future = validate_ibes_revision_date_frame(
        pd.DataFrame({"Instrument": ["AAA", "AAA"], "Revision Date": ["2024-02-01", "2026-07-07"]}),
        start_date="2024-01-01",
        end_date="2026-07-06",
        parameters=params,
        run_date="2026-07-08",
        load_date="2026-07-08",
        base_row=base,
    )

    assert accepted["status"] == "accepted"
    assert accepted["date_min"] == "2023-12-21"
    assert future["rejection_reason"] == "future_dates_outside_window"


def test_probe_normalizes_results_and_writes_complete_manifest(tmp_path):
    params = default_ibes_revision_parameter_grid("2024-01-01", "2026-07-06")[:1]
    results = probe_ibes_revision_timestamp_routes(
        FakeIBESClient(),
        selected_universe=_selected_universe(),
        start_date="2024-01-01",
        end_date="2026-07-06",
        countries=["MYS", "SGP", "IDN"],
        fields=[
            "TR.EPSMean.date",
            "TR.EPSMean.CalcDate",
            "TR.RevenueMean.CalcDate",
            "TR.RecMean.CalcDate",
            "TR.EPSMean",
        ],
        parameter_grid=params,
        sample_per_country=1,
        run_date="2026-07-08",
        load_date="2026-07-08",
        sleep_seconds=0,
    )

    assert list(results.columns) == RESULT_COLUMNS
    assert set(results["country"]) == {"MYS", "SGP", "IDN"}
    accepted = accepted_ibes_revision_time_map(results, countries=["MYS", "SGP", "IDN"])
    assert set(accepted["country"]) == {"MYS", "SGP", "IDN"}
    assert set(accepted["field"]) == {"TR.EPSMean.date"}
    assert results.loc[results["field"].eq("TR.EPSMean.CalcDate"), "rejection_reason"].eq("sentinel_date_pattern").all()
    assert results.loc[results["field"].eq("TR.EPSMean"), "rejection_reason"].eq("no_source_dates").all()

    artifacts = write_ibes_revision_route_probe_artifacts(
        artifact_root=tmp_path,
        run_id="IBES-REVISION-TEST",
        route_probe_results=results,
        countries=["MYS", "SGP", "IDN"],
        start_date="2024-01-01",
        end_date="2026-07-06",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.route_results_path.exists()
    assert artifacts.accepted_map_path.exists()
    assert manifest["status"] == "complete"
    assert manifest["promotion_target"] == "ibes_estimates_revisions/revision_time"
    assert manifest["accepted_countries"] == ["IDN", "MYS", "SGP"]


def test_probe_supports_japan_without_adv_rank():
    universe = pd.DataFrame(
        {
            "ticker": ["7203"],
            "refinitiv_ric": ["7203.T"],
            "country": ["Japan"],
            "selected_for_u2": [True],
        }
    )
    results = probe_ibes_revision_timestamp_routes(
        FakeIBESClient(),
        selected_universe=universe,
        start_date="2024-01-01",
        end_date="2026-07-06",
        countries=["JPN"],
        fields=["TR.EPSMean.date"],
        parameter_grid=[{"SDate": "2024-01-01", "EDate": "2026-07-06", "Frq": "D", "Period": "FY1"}],
        sample_per_country=1,
        tier="u2",
        run_date="2026-07-08",
        load_date="2026-07-08",
        sleep_seconds=0,
    )

    assert results.loc[0, "country"] == "JPN"
    assert results.loc[0, "status"] == "accepted"


def test_manifest_incomplete_until_all_requested_countries_have_accepted_route(tmp_path):
    results = pd.DataFrame(
        [
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "revision_time",
                "route": "get_data",
                "field": "TR.EPSMean.date",
                "parameter_set": "{}",
                "status": "accepted",
                "country": "MYS",
                "sample_rows": 2,
                "non_null_date_rows": 2,
                "date_min": "2024-02-01",
                "date_max": "2024-03-01",
                "unique_date_count": 2,
                "rejection_reason": "",
            }
        ]
    )

    artifacts = write_ibes_revision_route_probe_artifacts(
        artifact_root=tmp_path,
        run_id="IBES-REVISION-INCOMPLETE",
        route_probe_results=results,
        countries=["MYS", "SGP", "IDN"],
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "incomplete"
    assert manifest["missing_countries"] == ["SGP", "IDN"]
