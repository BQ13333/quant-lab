from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.data.refinitiv_pit_extraction import (
    accepted_refinitiv_pit_field_map,
    accepted_refinitiv_pit_history_field_map,
    build_refinitiv_pit_historical_plan,
    critical_refinitiv_route_probe_queue,
    execute_refinitiv_pit_historical_plan,
    extract_refinitiv_current_snapshot,
    history_field_map_from_route_probe,
    is_special_refinitiv_pit_candidate,
    probe_refinitiv_pit_historical_fields,
    probe_refinitiv_pit_fields,
    probe_refinitiv_pit_route_candidates,
    recover_refinitiv_pit_historical_manifest,
    select_pit_probe_samples,
    stage_refinitiv_pit_historical_plan,
    write_refinitiv_pit_historical_probe_artifacts,
    write_refinitiv_pit_field_probe_artifacts,
    write_refinitiv_pit_route_history_map_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivRetryableProviderError


class FakeLSEG:
    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def get_data(self, universe, fields):
        universe = list(universe)
        fields = list(fields)
        self.calls.append((tuple(universe), tuple(fields)))
        if "TR.BadField" in fields:
            raise ValueError("unknown field")

        data = {"Instrument": universe}
        for field in fields:
            if field == "TR.ISIN":
                data["ISIN"] = [f"ISIN-{ric}" for ric in universe]
            elif field == "TR.SecurityISIN":
                data["Security ISIN"] = [f"SEC-{ric}" for ric in universe]
            elif field == "TR.CompanyMarketCap":
                data["Company Market Cap"] = [1000 + index for index, _ric in enumerate(universe)]
            elif field == "TR.FreeFloatPct":
                data["Free Float (Percent)"] = [65.0 for _ric in universe]
            elif field == "TR.EmptyField":
                data["Empty Field"] = [pd.NA for _ric in universe]
            else:
                data[field] = [pd.NA for _ric in universe]
        return pd.DataFrame(data)

    def get_history(self, universe, fields, start, end, interval):
        universe = list(universe)
        field = list(fields)[0]
        self.calls.append((tuple(universe), (field, interval)))
        if field == "TR.BadHistory":
            raise ValueError("bad history")
        if field == "TR.CurrentOnly":
            return pd.DataFrame({ric: [1.0] for ric in universe}, index=pd.to_datetime(["2026-07-07"]))
        if field in {"TR.CompanyMarketCap", "TR.FreeFloatPct"}:
            dates = pd.date_range(start=start, end=end, freq="ME")
            frame = pd.DataFrame({ric: range(1, len(dates) + 1) for ric in universe}, index=dates)
            frame.columns.name = "Company Market Cap" if field == "TR.CompanyMarketCap" else "Free Float (Percent)"
            return frame
        if field == "TR.DivExDate":
            dates = pd.to_datetime(["2022-03-01", "2023-03-01"])
            frame = pd.DataFrame({ric: [pd.Timestamp("2022-03-01"), pd.Timestamp("2023-03-01")] for ric in universe}, index=dates)
            frame.columns.name = "Dividend Ex Date"
            return frame
        return pd.DataFrame()


class PartialBatchHistoryLSEG(FakeLSEG):
    def get_history(self, universe, fields, start, end, interval):
        universe = list(universe)
        if len(universe) > 1:
            raise ValueError("batch contains unsupported identifier")
        if universe == ["CIMB.KL"]:
            raise ValueError("unsupported identifier")
        return super().get_history(universe, fields, start, end, interval)


class ThrottledHistoryLSEG(FakeLSEG):
    def get_history(self, universe, fields, start, end, interval):
        self.calls.append((tuple(universe), tuple(fields)))
        raise RuntimeError("429 Too many requests")


class RetryableFieldProbeLSEG(FakeLSEG):
    def get_data(self, universe, fields):
        self.calls.append((tuple(universe), tuple(fields)))
        raise RefinitivRetryableProviderError("temporary provider outage")


def _selected_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "MBBM.KL",
                "refinitiv_ric": "MBBM.KL",
                "sid": "MY:MBBM",
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "selected_for_u2": True,
                "selected_for_u3": True,
                "adv_60_country_rank": 1,
            },
            {
                "ticker": "CIMB.KL",
                "refinitiv_ric": "CIMB.KL",
                "sid": "MY:CIMB",
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "selected_for_u2": True,
                "selected_for_u3": True,
                "adv_60_country_rank": 2,
            },
            {
                "ticker": "DBSM.SI",
                "refinitiv_ric": "DBSM.SI",
                "sid": "SG:DBSM",
                "country": "Singapore",
                "exchange": "SGX",
                "selected_for_u2": True,
                "selected_for_u3": True,
                "adv_60_country_rank": 1,
            },
            {
                "ticker": "BBCA.JK",
                "refinitiv_ric": "BBCA.JK",
                "sid": "ID:BBCA",
                "country": "Indonesia",
                "exchange": "IDX",
                "selected_for_u2": True,
                "selected_for_u3": False,
                "adv_60_country_rank": 1,
            },
        ]
    )


def _field_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "security_master_pit",
                "owner_agent": "security",
                "source_family": "refinitiv_reference",
                "field_role": "isin",
                "candidate_field": "TR.BadField",
                "required": True,
            },
            {
                "dataset_id": "security_master_pit",
                "owner_agent": "security",
                "source_family": "refinitiv_reference",
                "field_role": "isin",
                "candidate_field": "TR.ISIN",
                "required": True,
            },
            {
                "dataset_id": "security_master_pit",
                "owner_agent": "security",
                "source_family": "refinitiv_reference",
                "field_role": "listing_status",
                "candidate_field": "TR.EmptyField",
                "required": True,
            },
            {
                "dataset_id": "ibes_estimates_revisions",
                "owner_agent": "estimates",
                "source_family": "ibes_history",
                "field_role": "eps_estimate",
                "candidate_field": "IBES:EPS",
                "required": True,
            },
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "required": True,
            },
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "free_float",
                "candidate_field": "TR.FreeFloatPct",
                "required": True,
            },
            {
                "dataset_id": "corporate_actions",
                "owner_agent": "actions",
                "source_family": "refinitiv_corporate_actions",
                "field_role": "dividend_event",
                "candidate_field": "TR.DivExDate",
                "required": True,
            },
            {
                "dataset_id": "corporate_actions",
                "owner_agent": "actions",
                "source_family": "refinitiv_corporate_actions",
                "field_role": "special_dividend",
                "candidate_field": "IBES:NotAFormula",
                "required": False,
            },
        ]
    )


def test_select_pit_probe_samples_uses_liquid_names_per_country():
    samples = select_pit_probe_samples(_selected_universe(), sample_per_country=1, tier="u3")

    assert set(samples["refinitiv_ric"]) == {"MBBM.KL", "DBSM.SI"}
    assert "BBCA.JK" not in set(samples["refinitiv_ric"])


def test_select_pit_probe_samples_allows_missing_adv_rank():
    universe = pd.DataFrame(
        {
            "country": ["Japan", "Japan"],
            "ticker": ["7203", "6758"],
            "refinitiv_ric": ["7203.T", "6758.T"],
            "selected_for_u2": [True, True],
        }
    )

    samples = select_pit_probe_samples(universe, sample_per_country=1, tier="u2")

    assert samples["refinitiv_ric"].tolist() == ["6758.T"]


def test_special_refinitiv_pit_candidates_are_not_get_data_fields():
    assert is_special_refinitiv_pit_candidate("IBES:EPS")
    assert is_special_refinitiv_pit_candidate("QA:1001")
    assert is_special_refinitiv_pit_candidate("MYR=")
    assert is_special_refinitiv_pit_candidate("exchange_calendar")
    assert not is_special_refinitiv_pit_candidate("TR.ISIN")


def test_critical_refinitiv_route_probe_queue_filters_and_tags_blockers():
    queue = critical_refinitiv_route_probe_queue(
        datasets=["corporate_actions", "ibes_estimates_revisions"],
        field_roles=["split_event", "revision_time"],
    )

    assert not queue.empty
    assert set(queue["dataset_id"]) == {"corporate_actions", "ibes_estimates_revisions"}
    assert set(queue["field_role"]) == {"split_event", "revision_time"}
    assert {"candidate_source", "probe_reason", "validation_method"}.issubset(queue.columns)
    assert "TR.CAAdjustmentFactor" in set(queue["candidate_field"])
    assert "TR.RevisionDate" in set(queue["candidate_field"])
    assert queue["candidate_source"].eq("critical_route_expansion_v1").all()


def test_critical_refinitiv_route_probe_queue_empty_filter_has_schema():
    queue = critical_refinitiv_route_probe_queue(datasets=["does_not_exist"])

    assert queue.empty
    assert {
        "dataset_id",
        "owner_agent",
        "source_family",
        "field_role",
        "candidate_field",
        "required",
        "candidate_source",
    }.issubset(queue.columns)


def test_probe_refinitiv_pit_fields_validates_errors_and_special_routes():
    results = probe_refinitiv_pit_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )

    statuses = dict(zip(results["candidate_field"], results["status"], strict=True))
    assert statuses["TR.BadField"] == "error"
    assert statuses["TR.ISIN"] == "validated"
    assert statuses["TR.EmptyField"] == "field_returned_empty"
    assert statuses["IBES:EPS"] == "needs_special_provider"
    assert statuses["TR.CompanyMarketCap"] == "validated"

    isin_row = results[results["candidate_field"].eq("TR.ISIN")].iloc[0]
    assert isin_row["returned_data_columns"] == "ISIN"
    assert isin_row["non_null_values"] == 2


def test_probe_refinitiv_pit_fields_halts_after_retryable_provider_error():
    vendor = RetryableFieldProbeLSEG()

    results = probe_refinitiv_pit_fields(
        vendor,
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )

    assert len(vendor.calls) == 1
    assert results["candidate_field"].tolist() == ["TR.BadField"]
    assert results["status"].tolist() == ["failed_retryable_provider"]
    assert results.loc[0, "error"].startswith("RefinitivRetryableProviderError:")


def test_accepted_refinitiv_pit_field_map_picks_first_validated_candidate():
    results = probe_refinitiv_pit_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )
    accepted = accepted_refinitiv_pit_field_map(results)

    fields = dict(zip(accepted["field_role"], accepted["candidate_field"], strict=False))
    assert fields["isin"] == "TR.ISIN"
    assert fields["market_cap"] == "TR.CompanyMarketCap"
    assert "listing_status" not in fields


def test_write_refinitiv_pit_field_probe_artifacts(tmp_path):
    results = probe_refinitiv_pit_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )

    artifacts = write_refinitiv_pit_field_probe_artifacts(
        artifact_root=tmp_path,
        run_id="TEST-PROBE",
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        results=results,
        sample_per_country=1,
        tier="u3",
        datasets=None,
        session_name="desktop.workspace",
    )

    assert artifacts.results_path.exists()
    assert artifacts.accepted_field_map_path.exists()
    assert artifacts.summary_path.exists()
    assert len(pd.read_csv(artifacts.accepted_field_map_path)) == 3


def test_extract_refinitiv_current_snapshot_chunks_and_writes(tmp_path):
    results = probe_refinitiv_pit_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )
    accepted = accepted_refinitiv_pit_field_map(results)

    artifacts = extract_refinitiv_current_snapshot(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        accepted_field_map=accepted,
        artifact_root=tmp_path,
        run_id="TEST-SNAPSHOT",
        tier="u2",
        datasets=["security_master_pit", "liquidity_market_cap_free_float"],
        chunk_size=2,
        sleep_seconds=0,
        session_name="desktop.workspace",
    )

    assert artifacts.snapshot_csv_path.exists()
    assert artifacts.snapshot_parquet_path.exists()
    assert artifacts.quality_path.exists()
    assert len(artifacts.snapshot) == 4
    assert set(artifacts.snapshot["requested_ric"]) == {"MBBM.KL", "CIMB.KL", "DBSM.SI", "BBCA.JK"}
    assert "Company Market Cap" in artifacts.snapshot.columns
    assert artifacts.quality["coverage"].min() > 0


def test_probe_refinitiv_pit_historical_fields_classifies_history_routes():
    results = probe_refinitiv_pit_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )
    accepted = accepted_refinitiv_pit_field_map(results)
    extra = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "current_only",
                "candidate_field": "TR.CurrentOnly",
                "returned_data_columns": "Current Only",
            }
        ]
    )
    accepted = pd.concat([accepted, extra], ignore_index=True, sort=False)

    history = probe_refinitiv_pit_historical_fields(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        accepted_field_map=accepted,
        start_date="2022-01-01",
        end_date="2023-12-31",
        interval="1M",
        sample_per_country=1,
        tier="u3",
        datasets=["liquidity_market_cap_free_float"],
        sleep_seconds=0,
    )

    statuses = dict(zip(history["candidate_field"], history["status"], strict=False))
    assert statuses["TR.CompanyMarketCap"] == "historical_validated"
    assert statuses["TR.FreeFloatPct"] == "historical_validated"
    assert statuses["TR.CurrentOnly"] == "current_or_static_only"


def test_probe_refinitiv_pit_historical_fields_halts_after_quota_error():
    accepted = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": role,
                "candidate_field": field,
                "returned_data_columns": role,
            }
            for role, field in (
                ("market_cap", "TR.CompanyMarketCap"),
                ("free_float", "TR.FreeFloatPct"),
            )
        ]
    )
    vendor = ThrottledHistoryLSEG()

    results = probe_refinitiv_pit_historical_fields(
        vendor,
        selected_universe=_selected_universe(),
        accepted_field_map=accepted,
        start_date="2022-01-01",
        end_date="2023-12-31",
        sample_per_country=1,
        tier="u3",
        sleep_seconds=0,
    )

    assert len(vendor.calls) == 1
    assert results["candidate_field"].tolist() == ["TR.CompanyMarketCap"]
    assert results["status"].tolist() == ["failed_retryable_provider"]
    assert results.loc[0, "error"] == "RuntimeError: 429 Too many requests"


def test_historical_probe_artifacts_and_field_map(tmp_path):
    results = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "sample_count": 2,
                "non_null_values": 4,
                "inside_requested_rows": 2,
                "outside_requested_rows": 0,
                "coverage": 1.0,
                "status": "historical_validated",
                "probed_at_utc": "2026-07-08T00:00:00+00:00",
            }
        ]
    )

    artifacts = write_refinitiv_pit_historical_probe_artifacts(
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-PROBE",
        results=results,
        datasets=["liquidity_market_cap_free_float"],
        session_name="desktop.workspace",
    )

    assert artifacts.results_path.exists()
    assert artifacts.history_field_map_path.exists()
    assert len(pd.read_csv(artifacts.history_field_map_path)) == 1


def test_build_and_execute_refinitiv_pit_historical_plan(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "history_status": "historical_validated",
            }
        ]
    )
    plan = build_refinitiv_pit_historical_plan(
        selected_universe=_selected_universe(),
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY",
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        datasets=["liquidity_market_cap_free_float"],
        batch_size=2,
        window_years=1,
        interval="1M",
    )

    assert len(plan) == 3

    artifacts = execute_refinitiv_pit_historical_plan(
        FakeLSEG(),
        plan=plan,
        selected_universe=_selected_universe(),
        artifact_root=tmp_path,
        run_id="TEST-HISTORY",
        sleep_seconds=0,
        session_name="desktop.workspace",
    )

    assert artifacts.manifest_path.exists()
    assert artifacts.quality_path.exists()
    assert artifacts.run_manifest_path.exists()
    assert artifacts.manifest["status"].eq("completed").all()
    assert artifacts.quality["history_rows"].sum() > 0
    assert any(artifacts.output_root.rglob("*.parquet"))


def test_stage_refinitiv_pit_historical_plan_is_write_once(tmp_path):
    path = tmp_path / "historical_plan.csv"
    plan = pd.DataFrame({"plan_row_id": ["row-1"], "status": ["pending"]})

    digest = stage_refinitiv_pit_historical_plan(plan, path)
    assert path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == digest
    assert stage_refinitiv_pit_historical_plan(plan.copy(), path) == digest

    changed = pd.DataFrame({"plan_row_id": ["row-2"], "status": ["pending"]})
    with pytest.raises(ValueError, match="immutable PIT historical plan drift"):
        stage_refinitiv_pit_historical_plan(changed, path)


def test_execute_refinitiv_pit_historical_plan_can_skip_failed_rows_on_resume(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "history_status": "historical_validated",
            },
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "bad_history",
                "candidate_field": "TR.BadHistory",
                "returned_data_columns": "Bad History",
                "interval": "1M",
                "history_status": "historical_validated",
            },
        ]
    )
    selected = _selected_universe().head(1).copy()
    plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RESUME",
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        datasets=["liquidity_market_cap_free_float"],
        batch_size=1,
        window_years=1,
        interval="1M",
    )

    first = execute_refinitiv_pit_historical_plan(
        FakeLSEG(),
        plan=plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RESUME",
        sleep_seconds=0,
    )
    assert set(first.manifest["status"]) == {"completed", "failed"}
    assert len(first.errors) == 1

    fake = FakeLSEG()
    second = execute_refinitiv_pit_historical_plan(
        fake,
        plan=plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RESUME",
        skip_failed_on_resume=True,
        sleep_seconds=0,
    )

    assert fake.calls == []
    assert set(second.manifest["status"]) == {"completed", "failed"}
    assert len(second.errors) == 1


def test_historical_resume_rejects_checkpoint_from_different_tier_or_ric_batch(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "history_status": "historical_validated",
            }
        ]
    )
    selected = _selected_universe().iloc[:2].copy()
    selected["selected_for_u3"] = [True, False]
    run_id = "TEST-HISTORY-RESUME-IDENTITY"
    u3_plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id=run_id,
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u3",
        batch_size=25,
        window_years=1,
        interval="1M",
    )
    execute_refinitiv_pit_historical_plan(
        FakeLSEG(),
        plan=u3_plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id=run_id,
        sleep_seconds=0,
    )

    u2_plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id=run_id,
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        batch_size=25,
        window_years=1,
        interval="1M",
    )
    fake = FakeLSEG()
    with pytest.raises(ValueError, match="immutable PIT historical plan drift"):
        execute_refinitiv_pit_historical_plan(
            fake,
            plan=u2_plan,
            selected_universe=selected,
            artifact_root=tmp_path,
            run_id=run_id,
            sleep_seconds=0,
        )

    assert fake.calls == []


def test_recover_historical_manifest_from_parquet_checkpoints(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "history_status": "historical_validated",
            }
        ]
    )
    selected = _selected_universe().head(1).copy()
    plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RECOVERY",
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        batch_size=1,
        window_years=1,
        interval="1M",
    )
    original = execute_refinitiv_pit_historical_plan(
        FakeLSEG(),
        plan=plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RECOVERY",
        sleep_seconds=0,
    )
    original.manifest_path.write_text("corrupted,manifest\n", encoding="utf-8")

    recovered = recover_refinitiv_pit_historical_manifest(
        plan=plan,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-RECOVERY",
        session_name="test",
    )

    assert recovered.manifest["status"].tolist() == ["completed"]
    assert int(recovered.manifest["history_rows"].sum()) > 0
    assert recovered.quality["failed_rows"].eq(0).all()


def test_historical_batch_falls_back_to_single_rics_and_records_partial_coverage(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": "market_cap",
                "candidate_field": "TR.CompanyMarketCap",
                "returned_data_columns": "Company Market Cap",
                "interval": "1M",
                "history_status": "historical_validated",
            }
        ]
    )
    selected = _selected_universe().iloc[:2].copy()
    plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-PARTIAL",
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        batch_size=2,
        window_years=1,
        interval="1M",
    )

    artifacts = execute_refinitiv_pit_historical_plan(
        PartialBatchHistoryLSEG(),
        plan=plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-PARTIAL",
        sleep_seconds=0,
    )

    assert artifacts.manifest["status"].tolist() == ["completed_partial"]
    assert artifacts.manifest["error_type"].tolist() == ["partial_identifier_failures"]
    assert "CIMB.KL" in artifacts.manifest["error"].iloc[0]
    history = pd.read_parquet(artifacts.manifest["output_path"].iloc[0])
    assert set(history["requested_ric"]) == {"MBBM.KL"}


def test_pit_history_rate_limit_does_not_fan_out_and_halts_plan(tmp_path):
    history_field_map = pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "liquidity",
                "source_family": "refinitiv_reference",
                "field_role": role,
                "candidate_field": field,
                "returned_data_columns": role,
                "interval": "1M",
                "history_status": "historical_validated",
            }
            for role, field in (
                ("market_cap", "TR.CompanyMarketCap"),
                ("free_float", "TR.FreeFloatPct"),
            )
        ]
    )
    selected = _selected_universe().iloc[:2].copy()
    plan = build_refinitiv_pit_historical_plan(
        selected_universe=selected,
        history_field_map=history_field_map,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-429",
        start_date="2022-01-01",
        end_date="2022-12-31",
        tier="u2",
        batch_size=2,
        window_years=1,
        interval="1M",
    )
    vendor = ThrottledHistoryLSEG()

    artifacts = execute_refinitiv_pit_historical_plan(
        vendor,
        plan=plan,
        selected_universe=selected,
        artifact_root=tmp_path,
        run_id="TEST-HISTORY-429",
        sleep_seconds=0,
    )

    run = json.loads(artifacts.run_manifest_path.read_text())
    assert len(vendor.calls) == 1
    assert len(artifacts.manifest) == 1
    assert artifacts.manifest.loc[0, "status"] == "failed_retryable_provider"
    assert run["halted_early"] is True
    assert run["provider_halt"]["plan_row_id"] == plan.iloc[0]["plan_row_id"]


def test_probe_refinitiv_pit_route_candidates_finds_event_history_and_special_routes():
    results = probe_refinitiv_pit_route_candidates(
        FakeLSEG(),
        selected_universe=_selected_universe(),
        field_probe_queue=_field_queue(),
        start_date="2022-01-01",
        end_date="2023-12-31",
        interval="1M",
        sample_per_country=1,
        tier="u3",
        datasets=["corporate_actions"],
        sleep_seconds=0,
    )

    by_field_route = {
        (row.candidate_field, row.route): row.status
        for row in results.itertuples(index=False)
    }
    assert by_field_route[("TR.DivExDate", "get_history")] == "historical_validated"
    assert by_field_route[("IBES:NotAFormula", "get_data")] == "needs_special_provider"


def test_history_field_map_from_route_probe_promotes_best_history_candidate(tmp_path):
    route_results = pd.DataFrame(
        [
            {
                "dataset_id": "fundamentals_pit",
                "owner_agent": "fundamentals",
                "source_family": "worldscope",
                "field_role": "revenue",
                "candidate_field": "TR.Revenue",
                "route": "get_history",
                "status": "historical_validated",
                "required": True,
                "interval": "1M",
                "sample_count": 2,
                "non_null_values": 10,
                "inside_requested_rows": 5,
                "outside_requested_rows": 0,
                "coverage": 0.4,
                "probed_at_utc": "2026-07-08T00:00:00+00:00",
            },
            {
                "dataset_id": "fundamentals_pit",
                "owner_agent": "fundamentals",
                "source_family": "worldscope",
                "field_role": "revenue",
                "candidate_field": "TR.F.TotRevenue",
                "route": "get_history",
                "status": "historical_validated",
                "required": True,
                "interval": "1M",
                "sample_count": 2,
                "non_null_values": 20,
                "inside_requested_rows": 8,
                "outside_requested_rows": 0,
                "coverage": 0.8,
                "probed_at_utc": "2026-07-08T00:00:00+00:00",
            },
            {
                "dataset_id": "fundamentals_pit",
                "owner_agent": "fundamentals",
                "source_family": "worldscope",
                "field_role": "eps",
                "candidate_field": "TR.BadEPS",
                "route": "get_history",
                "status": "get_history_error",
                "required": True,
                "interval": "1M",
                "sample_count": 2,
                "non_null_values": 0,
                "inside_requested_rows": 0,
                "outside_requested_rows": 0,
                "coverage": 0.0,
                "probed_at_utc": "2026-07-08T00:00:00+00:00",
            },
        ]
    )

    field_map = history_field_map_from_route_probe(route_results, datasets=["fundamentals_pit"])

    assert len(field_map) == 1
    assert field_map.loc[0, "candidate_field"] == "TR.F.TotRevenue"
    assert field_map.loc[0, "history_status"] == "accepted_from_route_probe"

    artifacts = write_refinitiv_pit_route_history_map_artifacts(
        artifact_root=tmp_path,
        run_id="TEST-ROUTE-MAP",
        route_probe_results=route_results,
        datasets=["fundamentals_pit"],
    )
    assert artifacts.history_field_map_path.exists()
    assert artifacts.summary_path.exists()
