from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_lab.agents.japan_data_quality_execution_loop import (
    JapanExecutionLoopConfig,
    _non_overlapping_symbol_map_windows,
    _plan_complete,
    write_japan_execution_loop_artifacts,
)
from quant_lab.agents.japan_data_quality_loop import JapanDataQualityPaths


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _paths(root: Path) -> JapanDataQualityPaths:
    return JapanDataQualityPaths(
        security_master_dir=root / "security_master",
        symbol_map_dir=root / "symbol_map",
        historical_dir=root / "historical",
        depth_dir=root / "depth",
        compaction_dir=root / "compaction",
        universe_reconciliation_dir=root / "universe_reconciliation",
        active_symbol_map_dir=root / "active_symbol_map",
        reference_symbol_map_dir=root / "reference_symbol_map",
        l2_dir=root / "l2",
        pit_plan_dir=root / "pit_plan",
        corporate_actions_dir=root / "corpact",
        adjusted_returns_dir=root / "adjusted",
        liquidity_dir=root / "liquidity",
        fundamentals_ibes_events_dir=root / "fundamentals",
        fx_calendar_tca_dir=root / "fx",
        leakage_qa_dir=root / "leakage",
        commands_log=root / "commands.log",
    )


def _seed_eligible_universe(root: Path) -> JapanDataQualityPaths:
    paths = _paths(root)
    _write_json(
        paths.security_master_dir / "manifest.json",
        {
            "status": "complete",
            "security_master_rows": 2,
            "active_rows": 1,
            "inactive_rows": 1,
            "diagnostics": {"pagination_capped": False},
        },
    )
    paths.security_master_dir.mkdir(parents=True, exist_ok=True)
    for name in ("entities.parquet", "listing_windows.parquet", "identifier_history.parquet", "universe_membership_pit.parquet"):
        (paths.security_master_dir / name).touch()
    _write_json(
        paths.universe_reconciliation_dir / "manifest.json",
        {
            "status": "pass",
            "scorecard_eligible": True,
            "strict_quality_pass": True,
            "historical_pit_ready": True,
            "historical_backtest_eligible": True,
            "counts": {"active_operating_core_rows": 1, "historical_listed_reference_rows": 1},
        },
    )
    map_rows = {
        "active": {
            "country": "Japan",
            "exchange": "JPX",
            "ticker": "7203",
            "refinitiv_ric": "7203.T",
            "currency": "JPY",
            "known_at": NOW.isoformat(),
            "listing_start": "2000-01-01T00:00:00Z",
            "listing_end": "2026-07-02T00:00:00Z",
        },
        "reference": {
            "country": "Japan",
            "exchange": "JPX",
            "ticker": "9999",
            "refinitiv_ric": "9999.T",
            "currency": "JPY",
            "known_at": NOW.isoformat(),
            "listing_start": "2000-01-01T00:00:00Z",
            "listing_end": "2020-01-01T00:00:00Z",
        },
    }
    for role, directory in (
        ("active", paths.active_symbol_map_dir),
        ("reference", paths.reference_symbol_map_dir),
    ):
        _write_json(
            directory / "manifest.json",
            {
                "status": "pass",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "map_role": role,
                "validation_status": "pass",
            },
        )
        _write_csv(directory / "symbol_map.csv", [map_rows[role]])
    _write_csv(paths.historical_dir / "historical_plan.csv", [{"plan_row_id": "old-p1"}])
    _write_csv(
        paths.historical_dir / "historical_manifest.csv",
        [
            {
                "plan_row_id": "old-p1",
                "status": "failed_empty",
                "price_columns": 0,
                "long_ohlcv_rows": 0,
            }
        ],
    )
    _write_json(paths.historical_dir / "historical_run_manifest.json", {"planned_rows": 1, "completed_rows": 0})
    paths.commands_log.write_text("uv run queue generation\n", encoding="utf-8")
    return paths


def _config(root: Path, paths: JapanDataQualityPaths, **overrides) -> JapanExecutionLoopConfig:
    values = {
        "run_id": "JPN-EXEC-TEST",
        "artifact_root": Path("execution"),
        "project_root": root,
        "paths": paths,
        "start_date": "2026-07-01",
        "end_date": "2026-07-02",
        "batch_size": 1,
        "active_max_plan_rows": 2,
        "reference_max_plan_rows": 1,
        "min_security_master_rows": 1,
        "min_active_rows": 1,
        "min_reference_rows": 1,
        "as_of_utc": NOW,
    }
    values.update(overrides)
    return JapanExecutionLoopConfig(**values)


def _seed_g2_vendor_plan(root: Path) -> Path:
    plan_dir = root / "g2_vendor_plan"
    plan_dir.mkdir(parents=True)
    selected = pd.DataFrame(
        {
            "country": ["Japan"],
            "exchange": ["JPX"],
            "ticker": ["7203"],
            "refinitiv_ric": ["7203.T"],
            "currency": ["JPY"],
            "selected_for_u2": [True],
        }
    )
    corporate = pd.DataFrame(
        {
            "plan_row_id": ["dividend-japan-001"],
            "run_id": ["REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1"],
            "event_family": ["dividend"],
            "ric_list": ["7203.T"],
        }
    )
    probe = pd.DataFrame(
        {
            "dataset_id": ["liquidity_market_cap_free_float"] * 3,
            "field_role": ["market_cap", "free_float", "shares_outstanding"],
            "candidate_field": [
                "TR.CompanyMarketCap",
                "TR.FreeFloatPct",
                "TR.CompanySharesOutstanding",
            ],
            "refinitiv_ric": ["7203.T"] * 3,
        }
    )
    files = {
        "selected_universe.csv": selected,
        "corporate_action_plan.csv": corporate,
        "liquidity_probe_queue.csv": probe,
        "field_probe_queue.csv": probe,
        "dataset_requirements.csv": pd.DataFrame({"dataset_id": ["liquidity_market_cap_free_float"]}),
        "extraction_batches.csv": pd.DataFrame({"batch_id": [1]}),
    }
    hashes: dict[str, str] = {}
    for name, frame in files.items():
        frame.to_csv(plan_dir / name, index=False)
        hashes[name] = hashlib.sha256((plan_dir / name).read_bytes()).hexdigest()
    _write_json(
        plan_dir / "manifest.json",
        {
            "artifact_type": "japan_g2_vendor_execution_plan",
            "planning_complete": True,
            "unique_reference_rics": 5752,
            "corporate_action_plan_rows": 462,
            "liquidity_probe_rows": 9,
            "corporate_action_run_id": "REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1",
            "corporate_action_artifact_root": "corpact_runs",
            "input_sha256": {
                str((root / "active_symbol_map/symbol_map.csv").resolve()): hashlib.sha256(
                    (root / "active_symbol_map/symbol_map.csv").read_bytes()
                ).hexdigest(),
                str((root / "reference_symbol_map/symbol_map.csv").resolve()): hashlib.sha256(
                    (root / "reference_symbol_map/symbol_map.csv").read_bytes()
                ).hexdigest(),
            },
            "output_sha256": hashes,
        },
    )
    return plan_dir


def test_absent_jpx_universe_forces_reconciliation_first_and_never_promotes(tmp_path):
    paths = _paths(tmp_path)
    artifacts = write_japan_execution_loop_artifacts(_config(tmp_path, paths))

    queue = pd.read_csv(artifacts.execution_queue_path)
    state = json.loads(artifacts.execution_state_path.read_text(encoding="utf-8"))
    assert queue.iloc[0]["phase"] == "jpx_universe_reconciliation"
    assert queue.iloc[0]["status"] == "blocked_requires_historical_pit_repair"
    assert pd.isna(queue.iloc[0]["command"]) or queue.iloc[0]["command"] == ""
    assert "historical membership/known-at" in queue.iloc[0]["notes"]
    health = queue.loc[queue["phase"].eq("vendor_health_7203T")].iloc[0]
    enrichment = queue.loc[queue["phase"].eq("jpx_listing_master_pit_enrichment")].iloc[0]
    assert health["status"] == "blocked_retry_probe_required"
    assert "except Exception as exc" in health["command"]
    assert "payload.update(error_type=" in health["command"]
    assert enrichment["status"] == "queued_after_vendor_health"
    assert queue["quality_gate"].tolist()[-3:] == ["G4", "G5", "FINAL"]
    assert not queue[["promotes_alpha", "promotes_production", "promotes_ibkr"]].any().any()
    assert state["live_extraction_executed"] is False
    assert state["promotion_policy"] == {
        "alpha": False,
        "production": False,
        "ibkr": False,
        "live_trading": False,
    }
    g4 = queue.loc[queue["phase"].eq("g4_fx_calendar_halts_tca")].iloc[0]
    assert g4["status"] == "queued_after_g3"
    g5 = queue.loc[queue["phase"].eq("g5_leakage_reproducibility")].iloc[0]
    assert g5["status"] == "queued_after_g4"
    assert "examples/run_japan_g5_preflight.py" in g5["command"]


def test_g5_fail_closed_preflight_is_recorded_without_scorecard_credit(tmp_path):
    paths = _paths(tmp_path)
    _write_json(
        paths.leakage_qa_dir / "manifest.json",
        {
            "status": "blocked_upstream",
            "scope": "japan_g5_preflight_not_decision_calendar_audit",
            "scorecard_eligible": False,
            "upstream_components_ready": 2,
            "upstream_components_total": 7,
        },
    )

    artifacts = write_japan_execution_loop_artifacts(_config(tmp_path, paths))
    queue = pd.read_csv(artifacts.execution_queue_path)
    g5 = queue.loc[queue["phase"].eq("g5_leakage_reproducibility")].iloc[0]

    assert g5["status"] == "prepared_blocked_upstream"
    assert pd.isna(g5["command"]) or g5["command"] == ""
    assert "2/7 upstream components" in g5["notes"]
    assert not bool(g5["promotes_alpha"])


def test_g4_partial_lanes_are_explicit_and_never_promote(tmp_path):
    paths = _paths(tmp_path)
    halt = tmp_path / "halt"
    halt_validation = tmp_path / "halt_validation"
    pre2013 = tmp_path / "pre2013"
    return_normal = tmp_path / "return_normal"
    return_validation = tmp_path / "return_validation"
    denominator = tmp_path / "halt_denominator"
    denominator_validation = tmp_path / "halt_denominator_validation"
    _write_json(
        halt / "manifest.json",
        {
            "status": "partial_candidate",
            "scorecard_eligible": False,
            "counts": {"event_rows": 799},
            "coverage": {"month_channels": 480, "complete_month_channels": 0},
            "critical_blockers": ["source_denominator_uncovered"],
        },
    )
    _write_json(
        halt_validation / "manifest.json",
        {
            "status": "pass_partial_component",
            "scorecard_eligible": False,
            "validation_issues": [],
        },
    )
    _write_json(
        pre2013 / "manifest.json",
        {
            "status": "valid_incomplete_public_denominator",
            "immutable": True,
            "archive_denominator_proven": False,
            "completion_claimed": False,
            "scorecard_eligible": False,
            "counts": {"migrated_event_rows": 19},
            "coverage": {"required_months": 86, "months_with_unresolved_denominator": 86},
            "critical_blockers": ["pre2013_denominator_unproven"],
        },
    )
    _write_json(
        return_normal / "manifest.json",
        {
            "status": "partial_reconstruction",
            "immutable": True,
            "full_component_coverage_claimed": False,
            "scorecard_eligible": False,
            "counts": {"activation_cases": 556, "resolved_cases": 201, "unresolved_cases": 355},
            "unresolved_reason_counts": {"missing_ohlcv_without_closing_evidence": 342},
        },
    )
    _write_csv(
        return_normal / "missing_price_extraction_plan.csv",
        [
            {
                "plan_row_id": "missing-1",
                "refinitiv_ric": "7203.T",
                "session_count": 20,
                "execution_enabled": False,
            }
        ],
    )
    _write_json(
        return_validation / "manifest.json",
        {
            "status": "pass_partial_lane",
            "scorecard_eligible": False,
            "validation_issues": [],
        },
    )
    for directory in (denominator, denominator_validation):
        directory.mkdir(parents=True)
        (directory / "payload.csv").write_text("id\n1\n", encoding="utf-8")
        _write_json(
            directory / "artifact_hashes.json",
            {
                "payload.csv": hashlib.sha256(
                    (directory / "payload.csv").read_bytes()
                ).hexdigest()
            },
        )
    _write_json(
        denominator / "manifest.json",
        {
            "status": "partial_recovery_candidate",
            "source_denominator_rows": 366,
            "coverage": {"covered": 194, "uncovered": 172},
            "absence_is_zero": False,
            "scorecard_eligible": False,
            "artifact_hashes_sha256": hashlib.sha256(
                (denominator / "artifact_hashes.json").read_bytes()
            ).hexdigest(),
        },
    )
    _write_json(
        denominator_validation / "manifest.json",
        {
            "status": "pass_partial_component",
            "original_source_rows": 366,
            "fully_reconciled_claims": 194,
            "validation_issue_count": 0,
            "scorecard_eligible": False,
            "artifact_hashes_sha256": hashlib.sha256(
                (denominator_validation / "artifact_hashes.json").read_bytes()
            ).hexdigest(),
        },
    )

    artifacts = write_japan_execution_loop_artifacts(
        _config(
            tmp_path,
            paths,
            g4_halt_history_dir=halt,
            g4_halt_validation_dir=halt_validation,
            g4_pre2013_price_limit_dir=pre2013,
            g4_return_normal_dir=return_normal,
            g4_return_normal_validation_dir=return_validation,
            g4_halt_denominator_recovery_dir=denominator,
            g4_halt_denominator_validation_dir=denominator_validation,
            g4_halt_report_events_validation_dir=tmp_path / "missing_report_validation",
        )
    )
    queue = pd.read_csv(artifacts.execution_queue_path)
    state = json.loads(artifacts.execution_state_path.read_text(encoding="utf-8"))

    halt_row = queue.loc[queue["phase"].eq("g4_halt_history_evidence")].iloc[0]
    pre2013_row = queue.loc[
        queue["phase"].eq("g4_pre2013_exceptional_price_limit_archive")
    ].iloc[0]
    return_row = queue.loc[queue["phase"].eq("g4_post2020_return_to_normal")].iloc[0]
    denominator_row = queue.loc[
        queue["phase"].eq("g4_halt_report_denominator_and_events")
    ].iloc[0]
    assert halt_row["status"] == "validated_partial_completion_required"
    assert "0/480" in halt_row["notes"]
    assert pre2013_row["status"] == "valid_incomplete_denominator"
    assert "86/86" in pre2013_row["notes"]
    assert return_row["status"] == "validated_partial_waiting_for_prices"
    assert denominator_row["status"] == "validated_denominator_waiting_for_event_sets"
    assert "194/366" in denominator_row["notes"]
    assert "201/556" in return_row["notes"]
    assert state["g4_partial_evidence"]["return_to_normal"]["missing_ohlcv_cases"] == 342
    assert state["g4_partial_evidence"]["return_to_normal"]["missing_price_plan_rows"] == 1
    assert state["g4_partial_evidence"]["return_to_normal"][
        "missing_price_plan_execution_disabled"
    ] is True
    assert not queue[queue["phase"].str.startswith("g4_")][
        ["promotes_alpha", "promotes_production", "promotes_ibkr"]
    ].any().any()


def test_fresh_7203T_evidence_stages_bounded_checksummed_active_and_reference_commands(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {"status": "healthy", "symbol": "7203.T", "rows": 2, "observed_at_utc": NOW.isoformat()},
    )
    artifacts = write_japan_execution_loop_artifacts(
        _config(tmp_path, paths, vendor_health_evidence_path=health_path)
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    active = queue.loc[queue["phase"].eq("active_ohlcv")].iloc[0]
    reference = queue.loc[queue["phase"].eq("reference_ohlcv")].iloc[0]
    health = queue.loc[queue["phase"].eq("vendor_health_7203T")].iloc[0]
    assert queue.iloc[0]["status"] == "passed"
    assert health["status"] == "passed"
    enrichment = queue.loc[queue["phase"].eq("jpx_listing_master_pit_enrichment")].iloc[0]
    assert enrichment["status"] == "ready"
    assert "examples/run_japan_g1_enrichment.py" in enrichment["command"]
    assert "--max-rics 25" in enrichment["command"]
    assert "7203.T" in health["command"]
    assert active["status"] == "ready"
    assert reference["status"] == "queued_after_active"
    assert "--max-plan-rows 2" in active["command"]
    assert "--max-plan-rows 1" in reference["command"]
    assert "--existing-plan-only" in active["command"]
    assert "--expected-plan-sha256" in active["command"]
    assert "--require-exclude-plan-row-ids-csv" in active["command"]
    assert "mkdir \"$lock\"" in active["command"]
    assert "shasum -a 256 -c -" in active["command"]
    assert hashlib.sha256(artifacts.active_plan_path.read_bytes()).hexdigest() == active["expected_plan_sha256"]
    assert artifacts.active_plan_path.with_suffix(".sha256").exists()
    assert artifacts.reference_plan_path.with_suffix(".sha256").exists()


def test_stale_vendor_evidence_and_existing_writer_marker_block_live_rows(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {
            "status": "healthy",
            "symbol": "7203.T",
            "rows": 2,
            "observed_at_utc": (NOW - timedelta(hours=2)).isoformat(),
        },
    )
    lock = tmp_path / "writer.lock"
    artifacts = write_japan_execution_loop_artifacts(
        _config(
            tmp_path,
            paths,
            vendor_health_evidence_path=health_path,
            writer_lock_marker=lock,
        )
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    active = queue.loc[queue["phase"].eq("active_ohlcv")].iloc[0]
    health = queue.loc[queue["phase"].eq("vendor_health_7203T")].iloc[0]
    state = json.loads(artifacts.execution_state_path.read_text(encoding="utf-8"))
    assert health["status"] == "blocked_retry_probe_required"
    assert active["status"] == "blocked_by_vendor_health"
    assert state["vendor_health"]["valid"] is False
    assert state["writer_lock"]["marker_exists"] is False

    _write_json(
        health_path,
        {"status": "healthy", "symbol": "7203.T", "rows": 2, "observed_at_utc": NOW.isoformat()},
    )
    lock.mkdir()
    locked = write_japan_execution_loop_artifacts(
        _config(
            tmp_path,
            paths,
            vendor_health_evidence_path=health_path,
            writer_lock_marker=lock,
        )
    )
    locked_queue = pd.read_csv(locked.execution_queue_path)
    locked_active = locked_queue.loc[locked_queue["phase"].eq("active_ohlcv")].iloc[0]
    assert locked_active["status"] == "blocked_by_writer_lock"


def test_completed_current_enrichment_is_not_requeued_and_jquants_plan_is_next(tmp_path):
    paths = _paths(tmp_path)
    enrichment_dir = tmp_path / "current_enrichment"
    _write_json(
        enrichment_dir / "manifest.json",
        {
            "status": "complete_current_observation",
            "current_observation_only": True,
            "historical_pit_ready": False,
            "counts": {
                "success": 6_187,
                "source_unique_rics": 6_187,
                "error": 0,
                "failed_empty": 0,
            },
        },
    )
    artifacts = write_japan_execution_loop_artifacts(
        _config(tmp_path, paths, g1_current_enrichment_dir=enrichment_dir)
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    current = queue.loc[queue["phase"].eq("jpx_listing_master_pit_enrichment")].iloc[0]
    jquants = queue.loc[queue["phase"].eq("jquants_historical_listed_info")].iloc[0]
    assert current["status"] == "passed_current_observation_only"
    assert not bool(current["live_extraction"])
    assert "6187/6187" in current["notes"]
    assert jquants["status"] == "ready_generate_plan_or_bounded_backfill"
    assert "examples/run_jquants_listed_info_backfill.py" in jquants["command"]
    assert "--max-dates 5" in jquants["command"]


def test_jquants_missing_credentials_manifest_blocks_repeat_live_calls(tmp_path):
    paths = _paths(tmp_path)
    jquants_dir = tmp_path / "jquants"
    _write_json(
        jquants_dir / "manifest.json",
        {
            "status": "blocked_credentials",
            "historical_pit_ready": False,
            "scorecard_eligible": False,
        },
    )
    artifacts = write_japan_execution_loop_artifacts(
        _config(tmp_path, paths, jquants_listed_info_dir=jquants_dir)
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    jquants = queue.loc[queue["phase"].eq("jquants_historical_listed_info")].iloc[0]
    assert jquants["status"] == "blocked_by_jquants_credentials"
    assert pd.isna(jquants["command"]) or jquants["command"] == ""
    assert not bool(jquants["live_extraction"])


def test_forged_four_field_jquants_manifest_cannot_pass_queue_phase(tmp_path):
    paths = _paths(tmp_path)
    jquants_dir = tmp_path / "jquants"
    _write_json(
        jquants_dir / "manifest.json",
        {
            "status": "complete",
            "evidence_scope": "full",
            "historical_pit_ready": True,
            "scorecard_eligible": True,
        },
    )
    artifacts = write_japan_execution_loop_artifacts(
        _config(tmp_path, paths, jquants_listed_info_dir=jquants_dir)
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    jquants = queue.loc[queue["phase"].eq("jquants_historical_listed_info")].iloc[0]
    assert jquants["status"] == "ready_generate_plan_or_bounded_backfill"
    assert jquants["status"] != "passed_raw_history_only"


def test_blocked_materializer_is_not_repeated_before_official_history_completes(tmp_path):
    paths = _paths(tmp_path)
    jquants_dir = tmp_path / "jquants"
    materializer_dir = tmp_path / "materializer"
    _write_json(jquants_dir / "manifest.json", {"status": "blocked_credentials"})
    _write_json(
        materializer_dir / "manifest.json",
        {
            "status": "blocked_upstream",
            "critical_blockers": ["jquants_listed_info_not_complete_and_hash_verified"],
        },
    )
    artifacts = write_japan_execution_loop_artifacts(
        _config(
            tmp_path,
            paths,
            jquants_listed_info_dir=jquants_dir,
            g1_materializer_dir=materializer_dir,
        )
    )

    queue = pd.read_csv(artifacts.execution_queue_path)
    materializer = queue.loc[queue["phase"].eq("japan_g1_pit_materializer")].iloc[0]
    validator = queue.loc[queue["phase"].eq("japan_g1_independent_validation")].iloc[0]
    assert materializer["status"] == "queued_after_monthly_membership_qa"
    assert pd.isna(materializer["command"]) or materializer["command"] == ""
    assert validator["status"] == "queued_after_materializer"


def test_non_overlapping_repeated_ric_windows_are_valid_but_overlap_is_not():
    frame = pd.DataFrame(
        [
            {"refinitiv_ric": "1234.T", "listing_start": "2008-01-01", "listing_end": "2010-12-31"},
            {"refinitiv_ric": "1234.T", "listing_start": "2011-01-01", "listing_end": "2020-12-31"},
        ]
    )
    assert _non_overlapping_symbol_map_windows(frame)
    frame.loc[1, "listing_start"] = "2010-12-31"
    assert not _non_overlapping_symbol_map_windows(frame)


def test_partial_ohlcv_rows_do_not_complete_an_immutable_plan():
    partial = pd.DataFrame(
        [{"plan_row_id": "p1", "status": "completed_partial", "price_columns": 5, "long_ohlcv_rows": 20}]
    )
    complete = partial.assign(status="completed")
    assert not _plan_complete(partial, 1)
    assert _plan_complete(complete, 1)


def test_quarantine_merge_is_additive_and_plan_checksum_drift_is_not_overwritten(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    _write_csv(
        paths.historical_dir / "quarantined_plan_rows.csv",
        [{"plan_row_id": "legacy-p0", "reason": "legacy_route_failure"}],
    )
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {"status": "healthy", "symbol": "7203.T", "rows": 1, "observed_at_utc": NOW.isoformat()},
    )
    config = _config(tmp_path, paths, vendor_health_evidence_path=health_path)
    first = write_japan_execution_loop_artifacts(config)
    original_sha = first.active_plan_path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    first.active_plan_path.write_text("tampered\n", encoding="utf-8")

    second = write_japan_execution_loop_artifacts(config)

    queue = pd.read_csv(second.execution_queue_path)
    active = queue.loc[queue["phase"].eq("active_ohlcv")].iloc[0]
    quarantine = pd.read_csv(second.historical_quarantine_path)
    state = json.loads(second.execution_state_path.read_text(encoding="utf-8"))
    assert active["status"] == "blocked_by_plan_checksum"
    assert second.active_plan_path.read_text(encoding="utf-8") == "tampered\n"
    assert second.active_plan_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == original_sha
    assert set(quarantine["plan_row_id"]) == {"legacy-p0", "old-p1"}
    assert state["plans"]["active"]["valid"] is False
    assert not list(paths.historical_dir.glob("*.tmp"))


def test_active_quota_recovery_blocks_normal_active_queue_until_one_row_passes(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {"status": "healthy", "symbol": "7203.T", "rows": 1, "observed_at_utc": NOW.isoformat()},
    )
    config = _config(tmp_path, paths, vendor_health_evidence_path=health_path)
    stage = (
        tmp_path
        / "artifacts/historical_backfill_recovery_stage"
        / config.active_historical_run_id
    )
    stage.mkdir(parents=True)
    _write_csv(stage / "historical_plan.csv", [{"plan_row_id": "retry-1"}])
    _write_csv(stage / "failed_empty_174.csv", [{"plan_row_id": "retry-1"}])

    artifacts = write_japan_execution_loop_artifacts(config)

    queue = pd.read_csv(artifacts.execution_queue_path)
    recovery = queue.loc[queue["phase"].eq("active_ohlcv_quota_recovery")].iloc[0]
    active = queue.loc[queue["phase"].eq("active_ohlcv")].iloc[0]
    assert recovery["status"] == "ready_direct_session_test"
    assert "--max-plan-rows 1" in recovery["command"]
    assert "--include-plan-row-id retry-1" in recovery["command"]
    assert "merge_japan_ohlcv_recovery.py" in recovery["command"]
    assert active["status"] == "queued_after_quota_recovery"


def test_active_recovery_row_replaces_stale_vendor_health_probe(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {
            "status": "healthy",
            "symbol": "7203.T",
            "rows": 1,
            "observed_at_utc": (NOW - timedelta(hours=2)).isoformat(),
        },
    )
    config = _config(tmp_path, paths, vendor_health_evidence_path=health_path)
    stage = (
        tmp_path
        / "artifacts/historical_backfill_recovery_stage"
        / config.active_historical_run_id
    )
    stage.mkdir(parents=True)
    _write_csv(stage / "historical_plan.csv", [{"plan_row_id": "retry-1"}])
    _write_csv(stage / "failed_empty_174.csv", [{"plan_row_id": "retry-1"}])

    artifacts = write_japan_execution_loop_artifacts(config)
    queue = pd.read_csv(artifacts.execution_queue_path)
    health = queue.loc[queue["phase"].eq("vendor_health_7203T")].iloc[0]
    recovery = queue.loc[queue["phase"].eq("active_ohlcv_quota_recovery")].iloc[0]
    active = queue.loc[queue["phase"].eq("active_ohlcv")].iloc[0]

    assert health["status"] == "superseded_by_direct_session_test"
    assert pd.isna(health["command"]) or health["command"] == ""
    assert recovery["status"] == "ready_direct_session_test"
    assert recovery["depends_on"] == "G1"
    assert "--max-plan-rows 1" in recovery["command"]
    assert active["status"] == "queued_after_quota_recovery"
    assert active["depends_on"] == "G1,G2:active_ohlcv_quota_recovery"


def test_g2_vendor_steps_are_serial_and_dependency_gated(tmp_path):
    paths = _seed_eligible_universe(tmp_path)
    plan_dir = _seed_g2_vendor_plan(tmp_path)
    health_path = tmp_path / "health.json"
    _write_json(
        health_path,
        {"status": "healthy", "symbol": "7203.T", "rows": 1, "observed_at_utc": NOW.isoformat()},
    )
    stage = (
        tmp_path
        / "artifacts/historical_backfill_recovery_stage"
        / "REFINITIV-JPN-HISTORICAL-20Y-JPX-ACTIVE-G1V2"
    )
    stage.mkdir(parents=True)
    _write_csv(stage / "historical_plan.csv", [{"plan_row_id": "retry-1"}])
    _write_csv(stage / "failed_empty_174.csv", [{"plan_row_id": "retry-1"}])

    artifacts = write_japan_execution_loop_artifacts(
        _config(
            tmp_path,
            paths,
            vendor_health_evidence_path=health_path,
            g2_vendor_plan_dir=plan_dir,
        )
    )
    queue = pd.read_csv(artifacts.execution_queue_path, keep_default_na=False)
    ordered = queue.loc[
        queue["phase"].isin(
            [
                "active_ohlcv_quota_recovery",
                "active_ohlcv",
                "active_ohlcv_postprocessing",
                "reference_ohlcv",
                "reference_ohlcv_postprocessing",
                "g2_corporate_actions",
                "g2_liquidity_field_probe",
                "g2_liquidity_historical_probe",
                "g2_liquidity_historical_backfill",
            ]
        )
    ]
    assert ordered["phase"].tolist() == [
        "active_ohlcv_quota_recovery",
        "active_ohlcv",
        "active_ohlcv_postprocessing",
        "reference_ohlcv",
        "reference_ohlcv_postprocessing",
        "g2_corporate_actions",
        "g2_liquidity_field_probe",
        "g2_liquidity_historical_probe",
        "g2_liquidity_historical_backfill",
    ]
    assert ordered["status"].tolist() == [
        "ready_direct_session_test",
        "queued_after_quota_recovery",
        "queued_after_active",
        "queued_after_active",
        "queued_after_reference",
        "queued_after_reference_postprocessing",
        "queued_after_corporate_actions",
        "queued_after_liquidity_field_probe",
        "queued_after_liquidity_historical_probe",
    ]
    vendor_rows = ordered.loc[ordered["live_extraction"]]
    assert vendor_rows["command"].str.contains(".japan_ohlcv_writer.lock", regex=False).all()
    assert ordered.loc[
        ordered["phase"].str.endswith("postprocessing"), "live_extraction"
    ].eq(False).all()
    assert "--existing-plan-csv" in ordered.loc[
        ordered["phase"].eq("g2_corporate_actions"), "command"
    ].iloc[0]
