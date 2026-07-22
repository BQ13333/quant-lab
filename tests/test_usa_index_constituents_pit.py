from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.usa_index_constituents_pit import (
    INDEX_FAMILIES,
    USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    build_refinitiv_index_field_probe_plan,
    build_usa_index_constituent_request_plan,
    default_usa_index_catalog,
    pit_constituent_schema,
    resolve_usa_index_scope_profile,
    usa_index_catalog_for_scope_profile,
    validate_pit_constituent_rows,
    write_usa_index_constituent_plan_artifacts,
)


def _source_row(request: pd.Series, catalog: pd.DataFrame, ordinal: int = 0) -> dict[str, object]:
    target = catalog.set_index("index_id").loc[request["index_id"]]
    as_of = pd.Timestamp(request["request_end_date"])
    return {
        "index_id": request["index_id"],
        "index_family": target["index_family"],
        "index_name": target["index_name"],
        "index_ric": target["candidate_index_ric"],
        "member_ric": f"MEMBER{ordinal}.N",
        "member_perm_id": pd.NA,
        "member_isin": pd.NA,
        "member_ticker": f"MEMBER{ordinal}",
        "member_name": f"Member {ordinal}",
        "membership_action": "snapshot_member",
        "membership_as_of_date": as_of.date().isoformat(),
        "effective_date": pd.NA,
        "weight_percent": 1.0,
        "known_at_utc": f"{as_of.date().isoformat()}T21:00:00Z",
        "known_at_source_field": "TR.IndexConstituentAnnouncementDate",
        "source_provider": "licensed_test_fixture",
        "source_dataset": "pit_constituent_history_fixture",
        "source_record_id": f"record-{ordinal}",
        "source_request_id": request["source_request_id"],
        "source_loaded_at_utc": (as_of + pd.Timedelta(days=1)).isoformat(),
        "raw_payload_sha256": "a" * 64,
    }


def test_catalog_schema_and_refinitiv_route_evidence_are_explicit():
    catalog = default_usa_index_catalog()
    schema = pit_constituent_schema().set_index("column")
    existing_field_catalog = pd.DataFrame(
        [
            {
                "data_family": "price_history",
                "field_role": "Close",
                "field": "TR.IndexConstituentRIC",
                "validation_status": "live_probe_recommended",
            }
        ]
    )
    probes = build_refinitiv_index_field_probe_plan(existing_field_catalog)
    matched = probes[probes["candidate_field"].eq("TR.IndexConstituentRIC")].iloc[0]

    assert set(catalog["index_family"]) == set(INDEX_FAMILIES)
    assert len(catalog[catalog["index_family"].eq("Russell")]) == 3
    assert set(catalog["candidate_ric_status"]) == {
        "missing_constituent_entitlement",
        "validated_current_chain_and_joiner_leaver_route",
    }
    assert schema.loc["known_at_utc", "requirement"] == "required"
    assert schema.loc["known_at_source_field", "requirement"] == "required"
    assert matched["catalog_match_count"] == 1
    assert bool(matched["catalog_evidence_only"])
    assert matched["validation_status"] == "rejected_historical_anchor_modern_members_returned"
    assert bool(matched["live_request_executed"])
    assert not bool(matched["accepted_for_pit"])
    assert int(probes["accepted_for_pit"].sum()) == 5


def test_backtest_no_russell_scope_is_explicit_and_default_catalog_is_unchanged():
    default_catalog = default_usa_index_catalog()
    profile = resolve_usa_index_scope_profile(USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL)
    backtest_catalog = usa_index_catalog_for_scope_profile(profile.name)

    assert default_catalog["index_id"].tolist() == [
        "russell_1000",
        "russell_2000",
        "russell_3000",
        "sp_500",
        "nasdaq_100",
        "dow_jones_industrial_average",
    ]
    assert backtest_catalog["index_id"].tolist() == list(profile.required_index_ids) == [
        "sp_500",
        "nasdaq_100",
        "dow_jones_industrial_average",
    ]
    assert list(profile.excluded_index_ids) == ["russell_1000", "russell_2000", "russell_3000"]
    assert "deliberately excluded" in str(profile.excluded_index_reason)


def test_request_plan_is_annual_deterministic_and_execution_blocked():
    catalog = default_usa_index_catalog().head(1)
    plan = build_usa_index_constituent_request_plan(
        start_date="2023-06-01",
        end_date="2024-03-31",
        index_catalog=catalog,
    )

    assert list(plan["request_start_date"]) == ["2023-06-01", "2024-01-01"]
    assert list(plan["request_end_date"]) == ["2023-12-31", "2024-03-31"]
    assert plan["source_request_id"].is_unique
    assert set(plan["route"]) == {"unresolved_pending_live_probe"}
    assert set(plan["request_status"]) == {"blocked_pending_accepted_pit_route"}
    assert not plan["live_request_allowed"].any()


def test_empty_scaffold_does_not_write_constituent_history_or_claim_scorecard_evidence(tmp_path):
    result = write_usa_index_constituent_plan_artifacts(
        artifact_root=tmp_path / "plans",
        run_id="USA-PIT-INDEX-PLAN-EMPTY",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "incomplete"
    assert manifest["scorecard_eligible"] is False
    assert manifest["scaffold_only"] is True
    assert manifest["live_refinitiv_requests_executed"] is False
    assert manifest["constituent_history_fabricated"] is False
    assert manifest["candidate_fields_accepted"] == 5
    assert manifest["supplied_pit_rows"] == 0
    assert "no actual PIT constituent rows supplied" in manifest["blockers"]
    assert result.pit_rows_path is None
    assert not result.artifact_dir.joinpath("pit_constituent_rows.parquet").exists()
    assert result.index_catalog_path.exists()
    assert result.schema_path.exists()
    assert result.field_probe_plan_path.exists()
    assert result.request_plan_path.exists()


def test_validation_rejects_missing_known_at_and_load_timestamp_surrogate():
    catalog = default_usa_index_catalog()
    request_plan = build_usa_index_constituent_request_plan("2024-01-01", "2024-12-31", catalog.head(1))
    base = _source_row(request_plan.iloc[0], catalog, ordinal=1)
    missing_known_at = dict(base, known_at_utc=pd.NA)
    load_surrogate = dict(
        base,
        source_record_id="record-2",
        known_at_source_field="source_loaded_at_utc",
        known_at_utc=base["source_loaded_at_utc"],
    )

    validation = validate_pit_constituent_rows(
        pd.DataFrame([missing_known_at, load_surrogate]),
        request_plan=request_plan,
        index_catalog=catalog,
    )

    assert validation.usable_rows == 0
    assert not validation.rows["pit_usable"].any()
    issue_codes = set(validation.issues["issue_code"])
    assert "missing_source_provenance" in issue_codes
    assert "load_timestamp_used_as_known_at" in issue_codes


def test_valid_but_partial_source_rows_keep_manifest_incomplete(tmp_path):
    catalog = default_usa_index_catalog()
    request_plan = build_usa_index_constituent_request_plan("2024-01-01", "2024-12-31", catalog)
    one_row = pd.DataFrame([_source_row(request_plan.iloc[0], catalog, ordinal=1)])

    result = write_usa_index_constituent_plan_artifacts(
        artifact_root=tmp_path / "plans",
        run_id="USA-PIT-INDEX-PLAN-PARTIAL",
        start_date="2024-01-01",
        end_date="2024-12-31",
        pit_rows=one_row,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.validation.usable_rows == 1
    assert manifest["status"] == "incomplete"
    assert manifest["scorecard_eligible"] is False
    assert manifest["missing_request_count"] == len(request_plan) - 1
    assert manifest["completion_gates"]["all_request_windows_covered"] is False
    assert result.pit_rows_path is not None and result.pit_rows_path.exists()


def test_manifest_completes_only_with_valid_rows_covering_every_request(tmp_path):
    catalog = default_usa_index_catalog()
    request_plan = build_usa_index_constituent_request_plan("2024-01-01", "2024-12-31", catalog)
    rows = pd.DataFrame(
        [_source_row(request, catalog, ordinal=ordinal) for ordinal, (_, request) in enumerate(request_plan.iterrows())]
    )

    result = write_usa_index_constituent_plan_artifacts(
        artifact_root=tmp_path / "plans",
        run_id="USA-PIT-INDEX-PLAN-COMPLETE",
        start_date="2024-01-01",
        end_date="2024-12-31",
        pit_rows=rows,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.validation.usable_rows == len(request_plan)
    assert result.validation.issues.empty
    assert manifest["status"] == "complete"
    assert manifest["scorecard_eligible"] is True
    assert manifest["rows_with_known_at_utc"] == len(request_plan)
    assert manifest["missing_index_ids"] == []
    assert manifest["missing_request_count"] == 0
    assert all(manifest["completion_gates"].values())
