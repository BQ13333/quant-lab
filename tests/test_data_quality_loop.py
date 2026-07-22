from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_lab.agents.data_quality_loop import (
    InstitutionalDataQualityPaths,
    run_institutional_data_quality_loop,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _minimal_paths(root: Path) -> InstitutionalDataQualityPaths:
    return InstitutionalDataQualityPaths(
        l2_artifact_dir=root / "l2",
        pit_plan_dir=root / "pit_plan",
        field_probe_dir=root / "field_probe",
        route_probe_dir=root / "route_probe",
        route_history_map_dir=root / "route_history_map",
        current_snapshot_dir=root / "current_snapshot",
        liquidity_backfill_dir=root / "liquidity",
        fund_ibes_events_backfill_dir=root / "fund",
        pit_leakage_qa_dir=root / "leakage",
        pit_enhanced_alpha_dir=root / "alpha",
    )


def _seed_minimal_artifacts(root: Path) -> InstitutionalDataQualityPaths:
    paths = _minimal_paths(root)
    _write_json(
        paths.l2_artifact_dir / "manifest.json",
        {
            "date_min": "2006-01-02",
            "date_max": "2026-07-06",
            "raw_rows": 1_086_393,
            "u2_backtest_eligible": 169,
            "u3_liquid_proxy": 95,
        },
    )
    _write_csv(
        paths.l2_artifact_dir / "country_summary.csv",
        [
            {"country": "Malaysia", "u2_backtest_eligible": 62},
            {"country": "Singapore", "u2_backtest_eligible": 66},
            {"country": "Indonesia", "u2_backtest_eligible": 41},
        ],
    )
    _write_csv(
        paths.l2_artifact_dir / "ticker_quality.csv",
        [{"zero_volume_pct": 0.0, "bad_ohlc_rows": 0}],
    )
    _write_csv(
        paths.l2_artifact_dir / "vendor_extraction_matrix.csv",
        [
            {"layer": "corporate_actions", "current_availability": "missing_official_actions_proxy_only"},
        ],
    )
    _write_json(
        paths.pit_plan_dir / "manifest.json",
        {
            "dataset_count": 8,
            "extraction_batches": 1777,
            "selected_names": 169,
            "names_with_refinitiv_ric": 169,
        },
    )
    _write_csv(paths.pit_plan_dir / "selected_universe.csv", [{"ticker": "A", "has_refinitiv_ric": True}])
    _write_csv(paths.pit_plan_dir / "readiness_gates.csv", [{"current_status": "not_started"}])
    _write_csv(
        paths.field_probe_dir / "accepted_field_map.csv",
        [
            {"dataset_id": "security_master_pit", "field_role": "isin"},
            {"dataset_id": "security_master_pit", "field_role": "first_trade_date"},
            {"dataset_id": "liquidity_market_cap_free_float", "field_role": "market_cap"},
            {"dataset_id": "liquidity_market_cap_free_float", "field_role": "free_float"},
            {"dataset_id": "liquidity_market_cap_free_float", "field_role": "shares"},
            {"dataset_id": "liquidity_market_cap_free_float", "field_role": "adv_value"},
            {"dataset_id": "security_master_pit", "field_role": "sedol"},
            {"dataset_id": "security_master_pit", "field_role": "organization_id"},
        ],
    )
    _write_csv(
        paths.route_probe_dir / "route_probe_results.csv",
        [
            {"dataset_id": "corporate_actions", "field_role": "dividend_event", "status": "snapshot_validated"},
            {"dataset_id": "fundamentals_pit", "field_role": "revenue", "status": "historical_validated"},
        ],
    )
    _write_csv(
        paths.route_history_map_dir / "history_field_map.csv",
        [
            {"dataset_id": "fundamentals_pit", "field_role": "revenue"},
            {"dataset_id": "fundamentals_pit", "field_role": "book_value"},
            {"dataset_id": "fundamentals_pit", "field_role": "release_date"},
            {"dataset_id": "ibes_estimates_revisions", "field_role": "eps_estimate"},
            {"dataset_id": "ibes_estimates_revisions", "field_role": "revenue_estimate"},
            {"dataset_id": "ibes_estimates_revisions", "field_role": "recommendation"},
            {"dataset_id": "earnings_events_actuals", "field_role": "earnings_release"},
        ],
    )
    _write_json(
        paths.current_snapshot_dir / "manifest.json",
        {"session_name": "desktop.workspace", "snapshot_rows": 169},
    )
    _write_json(
        paths.liquidity_backfill_dir / "manifest.json",
        {"failed_rows": 0, "history_rows": 120_565},
    )
    _write_csv(
        paths.liquidity_backfill_dir / "historical_quality.csv",
        [{"field_role": "market_cap", "avg_coverage": 0.95}],
    )
    _write_json(
        paths.fund_ibes_events_backfill_dir / "manifest.json",
        {"failed_rows": 0, "history_rows": 30_653, "completed_rows": 70},
    )
    _write_csv(
        paths.fund_ibes_events_backfill_dir / "historical_quality.csv",
        [
            {
                "dataset_id": "earnings_events_actuals",
                "field_role": "earnings_release",
                "country": "Malaysia",
                "last_available": "2026-07-02",
            }
        ],
    )
    _write_json(
        paths.pit_leakage_qa_dir / "manifest.json",
        {
            "status": "pass",
            "total_leakage_violations": 0,
            "decision_entities": 169,
            "decision_rows": 237_952,
        },
    )
    _write_csv(paths.pit_leakage_qa_dir / "leakage_violations.csv", [])
    _write_csv(paths.pit_leakage_qa_dir / "feature_quality.csv", [{"coverage": 0.686}])
    _write_csv(paths.pit_leakage_qa_dir / "leakage_summary.csv", [{"status": "pass"}])
    assert paths.pit_enhanced_alpha_dir is not None
    _write_json(paths.pit_enhanced_alpha_dir / "overfit_audit.json", {"audit_posture": "diagnostic_only"})
    (paths.pit_enhanced_alpha_dir / "run_config.json").write_text('{"holdout_start":"2024-01-02"}', encoding="utf-8")
    return paths


def test_institutional_data_quality_loop_writes_scorecards(tmp_path):
    paths = _seed_minimal_artifacts(tmp_path)

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-TEST",
    )

    assert result.status == "continue_data_quality_loop"
    assert result.accepted is False
    assert result.overall_score < 5.0
    assert result.summary_path.exists()
    assert result.manifest_path.exists()
    assert result.lane_scorecard_path.exists()
    assert result.criteria_scorecard_path.exists()
    assert result.next_queue_path.exists()

    next_queue = pd.read_csv(result.next_queue_path)
    assert "security_master_universe" in set(next_queue["lane_id"])
    assert "survivorship and identifier-history bias not resolved" in set(next_queue["blocker"])


def test_institutional_data_quality_loop_manifest_matches_result(tmp_path):
    paths = _seed_minimal_artifacts(tmp_path)

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-TEST-MANIFEST",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["run_id"] == "DQ-TEST-MANIFEST"
    assert manifest["accepted"] is False
    assert manifest["overall_score"] == result.overall_score
    assert manifest["status"] == result.status
    assert manifest["next_queue_rows"] > 0


def test_current_snapshot_never_passes_non_survivor_denominator(tmp_path):
    paths = _seed_minimal_artifacts(tmp_path)
    _write_csv(
        paths.route_probe_dir / "route_probe_results.csv",
        [
            {
                "dataset_id": "delistings_listing_history",
                "field_role": "delisting_date",
                "route": "get_history",
                "status": "historical_validated",
            },
            {
                "dataset_id": "delistings_listing_history",
                "field_role": "ric_change",
                "route": "get_history",
                "status": "historical_validated",
            },
            {
                "dataset_id": "security_master_pit",
                "field_role": "listing_status",
                "route": "get_history",
                "status": "historical_validated",
            },
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-SNAPSHOT-STILL-SURVIVOR",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["historical_denominator_artifact", "status"] == "fail"
    assert by_id.loc["non_survivor_denominator", "status"] == "fail"


def test_discovery_route_does_not_unlock_denominator_without_artifacts(tmp_path):
    paths = _seed_minimal_artifacts(tmp_path)
    _write_csv(
        paths.route_probe_dir / "route_probe_results.csv",
        [
            {
                "dataset_id": "delistings_listing_history",
                "field_role": "delisting_date",
                "route": "discovery_search_security_master",
                "status": "discovery_validated",
            },
            {
                "dataset_id": "delistings_listing_history",
                "field_role": "ric_change",
                "route": "symbol_conversion",
                "status": "discovery_validated",
            },
            {
                "dataset_id": "security_master_pit",
                "field_role": "listing_status",
                "route": "discovery_search_security_master",
                "status": "discovery_validated",
            },
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-DISCOVERY-ROUTE-ONLY",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["listing_delisting_history", "status"] == "pass"
    assert by_id.loc["historical_denominator_artifact", "status"] == "fail"
    assert by_id.loc["non_survivor_denominator", "status"] == "fail"


def test_historical_security_master_artifacts_unlock_denominator_gate(tmp_path):
    paths = replace(_seed_minimal_artifacts(tmp_path), security_master_history_dir=tmp_path / "security_master_history")
    assert paths.security_master_history_dir is not None
    root = paths.security_master_history_dir
    root.mkdir(parents=True, exist_ok=True)
    for name in ("entities", "identifier_history", "listing_windows", "universe_membership_pit"):
        (root / f"{name}.parquet").write_bytes(b"placeholder")
    _write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "date_min": "2006-01-01",
            "dead_name_rows": 12,
        },
    )
    _write_csv(
        root / "quality/security_master_pit_quality.csv",
        [
            {"country": "Malaysia", "asof_year": 2006, "delisted_count": 4},
            {"country": "Singapore", "asof_year": 2006, "delisted_count": 4},
            {"country": "Indonesia", "asof_year": 2006, "delisted_count": 4},
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-HISTORICAL-SECURITY-MASTER",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["historical_denominator_artifact", "status"] == "pass"
    assert by_id.loc["non_survivor_denominator", "status"] == "pass"


def test_corporate_action_event_backfill_passes_official_action_gate(tmp_path):
    paths = replace(_seed_minimal_artifacts(tmp_path), corporate_actions_event_dir=tmp_path / "corporate_actions_events")
    assert paths.corporate_actions_event_dir is not None
    root = paths.corporate_actions_event_dir
    _write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "failed_rows": 0,
            "event_rows": 100,
        },
    )
    _write_csv(
        root / "event_quality.csv",
        [
            {
                "field_role": field_role,
                "country": country,
                "failed_rows": 0,
                "event_rows": 10,
                "valid_event_date_rows": 10,
            }
            for field_role in ("dividend_event", "split_event")
            for country in ("Malaysia", "Singapore", "Indonesia")
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-CORPACT-EVENTS",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["official_corporate_actions", "status"] == "pass"
    assert by_id.loc["adjusted_raw_integrity", "status"] == "fail"


def test_fx_country_risk_artifact_passes_currency_fx_gate(tmp_path):
    paths = replace(_seed_minimal_artifacts(tmp_path), fx_country_risk_dir=tmp_path / "fx_country_risk")
    assert paths.fx_country_risk_dir is not None
    root = paths.fx_country_risk_dir
    _write_json(
        root / "manifest.json",
        {
            "status": "pass_currency_fx_returns_partial_tca",
            "currency_coverage": 1.0,
            "fx_calendar_coverage": 0.99,
            "usd_return_coverage": 0.98,
        },
    )
    _write_csv(
        root / "currency_reference.csv",
        [
            {"country": "Malaysia", "currency": "MYR", "security_coverage": 1.0},
            {"country": "Singapore", "currency": "SGD", "security_coverage": 1.0},
            {"country": "Indonesia", "currency": "IDR", "security_coverage": 1.0},
        ],
    )
    _write_csv(
        root / "fx_quality.csv",
        [
            {"fx_ticker": "MYR=X", "calendar_coverage": 0.99},
            {"fx_ticker": "SGD=X", "calendar_coverage": 0.99},
            {"fx_ticker": "IDR=X", "calendar_coverage": 0.99},
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-FX-PASS",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["currency_fx_metadata", "status"] == "pass"
    assert "usd_return_coverage=0.980" in by_id.loc["currency_fx_metadata", "evidence"]


def test_diagnostic_reconciliation_is_partial_until_adjusted_returns_are_rebuilt(tmp_path):
    paths = replace(
        _seed_minimal_artifacts(tmp_path),
        corporate_actions_event_dir=tmp_path / "corporate_actions_events",
        corporate_action_reconciliation_dir=tmp_path / "corporate_action_reconciliation",
    )
    assert paths.corporate_actions_event_dir is not None
    event_root = paths.corporate_actions_event_dir
    _write_json(event_root / "manifest.json", {"status": "complete", "failed_rows": 0, "event_rows": 100})
    _write_csv(
        event_root / "event_quality.csv",
        [
            {
                "field_role": field_role,
                "country": country,
                "failed_rows": 0,
                "event_rows": 10,
                "valid_event_date_rows": 10,
            }
            for field_role in ("dividend_event", "split_event")
            for country in ("Malaysia", "Singapore", "Indonesia")
        ],
    )
    assert paths.corporate_action_reconciliation_dir is not None
    recon_root = paths.corporate_action_reconciliation_dir
    _write_json(
        recon_root / "manifest.json",
        {
            "status": "diagnostic_reconciliation_only",
            "valid_event_rows": 100,
            "matched_event_share_min": 1.0,
            "production_adjusted_return_ready": False,
        },
    )
    _write_csv(
        recon_root / "reconciliation_quality.csv",
        [
            {"country": country, "event_family": event_family, "valid_event_rows": 10, "matched_event_share": 1.0}
            for country in ("Malaysia", "Singapore", "Indonesia")
            for event_family in ("dividend", "capital_change")
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-RECON-DIAGNOSTIC",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["adjusted_raw_integrity", "status"] == "partial"
    assert by_id.loc["return_integrity_bias", "status"] == "partial"
    assert "production_adjusted_return_ready=False" in by_id.loc["adjusted_raw_integrity", "evidence"]


def test_production_adjusted_return_reconstruction_passes_return_integrity_gates(tmp_path):
    paths = replace(
        _seed_minimal_artifacts(tmp_path),
        corporate_actions_event_dir=tmp_path / "corporate_actions_events",
        adjusted_return_reconstruction_dir=tmp_path / "adjusted_return_reconstruction",
    )
    assert paths.corporate_actions_event_dir is not None
    event_root = paths.corporate_actions_event_dir
    _write_json(event_root / "manifest.json", {"status": "complete", "failed_rows": 0, "event_rows": 100})
    _write_csv(
        event_root / "event_quality.csv",
        [
            {
                "field_role": field_role,
                "country": country,
                "failed_rows": 0,
                "event_rows": 10,
                "valid_event_date_rows": 10,
            }
            for field_role in ("dividend_event", "split_event")
            for country in ("Malaysia", "Singapore", "Indonesia")
        ],
    )
    assert paths.adjusted_return_reconstruction_dir is not None
    adj_root = paths.adjusted_return_reconstruction_dir
    _write_json(
        adj_root / "manifest.json",
        {
            "status": "pass",
            "production_adjusted_return_ready": True,
            "valid_event_rows": 2303,
            "matched_event_share": 0.999,
            "applied_adjustment_rows": 2112,
            "quarantine_share": 0.0026,
            "max_quarantine_share": 0.005,
        },
    )
    _write_csv(
        adj_root / "reconciliation_quality.csv",
        [
            {"country": country, "event_family": event_family, "valid_event_rows": 10, "matched_event_share": 1.0}
            for country in ("Malaysia", "Singapore", "Indonesia")
            for event_family in ("dividend", "capital_change")
        ],
    )

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-ADJRET-PRODUCTION",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["adjusted_raw_integrity", "status"] == "pass"
    assert by_id.loc["return_integrity_bias", "status"] == "pass"
    assert "adjusted_return_reconstruction_status=pass" in by_id.loc["adjusted_raw_integrity", "evidence"]


def test_ibes_revision_route_probe_artifact_closes_revision_time_gate(tmp_path):
    paths = replace(
        _seed_minimal_artifacts(tmp_path),
        ibes_revision_route_probe_dir=tmp_path / "ibes_revision_probe",
        critical_fields_backfill_dir=tmp_path / "critical_fields",
    )
    _write_csv(
        paths.route_probe_dir / "route_probe_results.csv",
        [
            {
                "dataset_id": "earnings_events_actuals",
                "field_role": "actual_eps",
                "route": "get_history",
                "status": "historical_validated",
            },
            {
                "dataset_id": "earnings_events_actuals",
                "field_role": "surprise",
                "route": "get_history",
                "status": "historical_validated",
            },
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "target_price",
                "route": "get_history",
                "status": "historical_validated",
            },
        ],
    )
    assert paths.critical_fields_backfill_dir is not None
    _write_csv(
        paths.critical_fields_backfill_dir / "historical_quality.csv",
        [
            {"field_role": field_role, "country": country, "history_rows": 10, "failed_rows": 0, "completed_rows": 1}
            for field_role in ("actual_eps", "surprise", "target_price")
            for country in ("Malaysia", "Singapore", "Indonesia")
        ],
    )
    assert paths.ibes_revision_route_probe_dir is not None
    root = paths.ibes_revision_route_probe_dir
    _write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "accepted_countries": ["IDN", "MYS", "SGP"],
            "missing_countries": [],
        },
    )
    _write_csv(
        root / "accepted_revision_time_map.csv",
        [
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "revision_time",
                "route": "get_data",
                "field": "TR.RecLabelEstBrokerName.CalcDate",
                "parameter_set": "{}",
                "status": "accepted",
                "country": country,
                "sample_rows": 100,
                "non_null_date_rows": 100,
                "date_min": "2024-01-02",
                "date_max": "2026-07-06",
                "unique_date_count": 50,
                "rejection_reason": "",
            }
            for country in ("MYS", "SGP", "IDN")
        ],
    )
    _write_csv(root / "route_probe_results.csv", [])

    result = run_institutional_data_quality_loop(
        paths=paths,
        artifact_root=tmp_path / "scorecards",
        run_id="DQ-IBES-REVISION-CLOSED",
    )
    criteria = pd.read_csv(result.criteria_scorecard_path)
    by_id = criteria.set_index("criterion_id")

    assert by_id.loc["actuals_surprises_targets", "status"] == "pass"
    assert "revision_time_artifact=status=complete" in by_id.loc["actuals_surprises_targets", "evidence"]
