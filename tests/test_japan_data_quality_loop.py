from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from quant_lab.agents.japan_data_quality_loop import JapanDataQualityPaths, run_japan_data_quality_loop


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _full_artifact(**values: object) -> dict:
    return {
        "run_id": "JPN-FULL-INDEPENDENT-AUDIT",
        "status": "pass",
        "scorecard_eligible": True,
        "evidence_scope": "full",
        **values,
    }


def _paths(root: Path) -> JapanDataQualityPaths:
    return JapanDataQualityPaths(
        security_master_dir=root / "security_master",
        symbol_map_dir=root / "legacy_quote_map",
        historical_dir=root / "historical",
        depth_dir=root / "depth",
        compaction_dir=root / "compaction",
        universe_reconciliation_dir=root / "jpx_universe_reconciliation",
        active_symbol_map_dir=root / "active_symbol_map",
        reference_symbol_map_dir=root / "reference_symbol_map",
        pit_plan_dir=root / "pit_plan",
        corporate_actions_dir=root / "corporate_actions",
        adjusted_returns_dir=root / "adjusted_returns",
        g2_admission_dir=root / "g2_admission",
        liquidity_dir=root / "liquidity",
        fundamentals_ibes_events_dir=root / "fund_ibes_events",
        g3_admission_dir=root / "g3_admission",
        fx_calendar_tca_dir=root / "fx_calendar_tca",
        leakage_qa_dir=root / "leakage",
        g1_validation_dir=root / "g1_validation",
        jpx_public_event_reconciliation_dir=root / "jpx_public_event_crosscheck",
        immutable_hashes_path=root / "leakage" / "immutable_hashes.json",
        commands_log=root / "commands.log",
    )


def _seed_complete_evidence(root: Path) -> JapanDataQualityPaths:
    paths = _paths(root)
    _write_json(
        paths.security_master_dir / "manifest.json",
        _full_artifact(
            status="complete",
            security_master_kind="listing_level",
            listing_level_master=True,
            quote_level_master=False,
            listing_rows=1,
            provenance={
                name: {"complete": True, "coverage": 1.0, "missing_rows": 0, "source": "JPX/LSEG"}
                for name in ("currency", "listing_window", "known_at")
            },
            non_overlap_checks={
                name: {"passed": True, "overlap_rows": 0}
                for name in ("listing_windows", "identifier_windows", "currency_windows")
            },
        ),
    )
    for name in (
        "entities.parquet",
        "listing_windows.parquet",
        "identifier_history.parquet",
        "universe_membership_pit.parquet",
    ):
        (paths.security_master_dir / name).touch()
    _write_json(
        paths.universe_reconciliation_dir / "manifest.json",
        _full_artifact(
            artifact_type="jpx_universe_reconciliation",
            independent_audit=True,
            producer_run_id="JPN-G1-PRODUCER",
            validator_run_id="JPN-G1-INDEPENDENT-VALIDATOR",
            source_checksums_verified=True,
            reconciliation_output_sha256_verified=True,
            official_jpx_source=True,
            reconciliation_complete=True,
            jpx_source_rows=2,
            reconciliation_coverage=1.0,
            unmatched_jpx_rows=0,
            unmatched_master_rows=0,
            duplicate_listing_rows=0,
        ),
    )
    map_rows = [
        {
            "refinitiv_ric": "7203.T",
            "listing_start": "1949-05-16T00:00:00Z",
            "listing_end": "2026-07-10T00:00:00Z",
            "currency": "JPY",
            "known_at": "2026-07-10T23:59:00Z",
        }
    ]
    for role, directory in (
        ("active", paths.active_symbol_map_dir),
        ("reference", paths.reference_symbol_map_dir),
    ):
        _write_csv(directory / "symbol_map.csv", map_rows)
        _write_json(
            directory / "manifest.json",
            _full_artifact(
                map_role=role,
                validation_status="pass",
                producer_run_id="JPN-G1-PRODUCER",
                validator_run_id=f"JPN-G1-INDEPENDENT-{role.upper()}-VALIDATOR",
                validator_type="independent_process",
                source_checksums_verified=True,
                symbol_map_sha256=hashlib.sha256(
                    (directory / "symbol_map.csv").read_bytes()
                ).hexdigest(),
                mapping_coverage=1.0,
                unresolved_rows=0,
                invalid_ric_rows=0,
                overlap_conflict_rows=0,
            ),
        )

    _write_json(
        paths.g1_validation_dir / "manifest.json",
        _full_artifact(
            artifact_type="japan_g1_independent_validation",
            independent_audit=True,
            validator_type="independent_process",
            producer_run_id="JPN-G1-PRODUCER",
            validator_run_id="JPN-G1-INDEPENDENT-VALIDATOR",
            source_checksums_verified=True,
            validation_status="pass",
            critical_blocker_count=0,
        ),
    )
    _write_json(
        paths.jpx_public_event_reconciliation_dir / "manifest.json",
        {
            "run_id": "JPX-PUBLIC-CROSSCHECK",
            "status": "pass",
            "artifact_type": "jpx_public_event_refinitiv_cross_validation",
            "evidence_scope": "recent_official_public_archive_cross_validation",
            "counts": {
                "events": 2,
                "mapping_matched": 2,
                "mapping_unmatched_or_ambiguous": 0,
                "date_exact_match": 2,
                "date_missing_or_mismatch": 0,
            },
            "mapping_coverage": 1.0,
            "date_exact_match_coverage": 1.0,
        },
    )

    _write_json(
        paths.historical_dir / "historical_run_manifest.json",
        _full_artifact(
            status="complete",
            planned_rows=1,
            terminal_rows=1,
            failed_rows=0,
            ric_window_coverage=1.0,
            window_coverage_complete=True,
        ),
    )
    _write_csv(
        paths.historical_dir / "historical_plan.csv",
        [
            {
                "plan_row_id": "7203.T__2006_2026",
                "refinitiv_ric": "7203.T",
                "window_start": "2006-07-10",
                "window_end": "2026-07-10",
            }
        ],
    )
    _write_csv(
        paths.historical_dir / "historical_manifest.csv",
        [
            {
                "plan_row_id": "7203.T__2006_2026",
                "status": "completed",
                "price_columns": 5,
                "long_ohlcv_rows": 4_900,
                "structural_reason": "",
            }
        ],
    )
    _write_csv(
        paths.depth_dir / "overall.csv",
        [
            {
                "rows": 4_900,
                "tickers": 1,
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "non_positive_close_rows": 0,
                "duplicate_ohlcv_rows": 0,
                "outside_listing_window_rows": 0,
                "volume_missing_pct": 0.0,
            }
        ],
    )
    _write_json(
        paths.compaction_dir / "manifest.json",
        _full_artifact(
            status="complete",
            source_rows=4_900,
            output_rows=4_900,
            failed_groups=0,
        ),
    )
    _write_json(
        paths.corporate_actions_dir / "manifest.json",
        _full_artifact(
            status="complete",
            planned_rows=1,
            completed_rows=1,
            failed_rows=0,
            empty_rows=0,
            alpha_usable_rows=5,
            known_at_coverage=1.0,
            missing_known_at_rows=0,
            official_source=True,
        ),
    )
    _write_json(
        paths.adjusted_returns_dir / "manifest.json",
        _full_artifact(
            reconciliation_complete=True,
            production_adjusted_return_ready=True,
            reconciled_rows=4_900,
            matched_event_share=1.0,
            unmatched_event_rows=0,
            quarantined_rows=0,
            quarantine_share=0.0,
        ),
    )
    g2_contract_path = paths.g2_admission_dir / "component_contract.csv"
    _write_csv(
        g2_contract_path,
        [
            {"component": "g1", "check_id": "denominator", "passed": True},
            {"component": "ohlcv", "check_id": "active_reference", "passed": True},
            {"component": "components", "check_id": "actions_liquidity", "passed": True},
        ],
    )
    inventory_entries: dict[str, dict] = {}
    for name in ("active", "reference", "components"):
        (paths.g2_admission_dir / f"{name}.parquet").write_bytes(
            f"{name}-strict-source\n".encode()
        )
        inventory_path = paths.g2_admission_dir / f"{name}_inventory.json"
        _write_json(inventory_path, {"files": {f"{name}.parquet": "0" * 64}})
        inventory_entries[name] = {
            "inventory_path": str(inventory_path.resolve()),
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "files": {f"{name}.parquet": "0" * 64},
        }
    g2_source_hashes_path = paths.g2_admission_dir / "source_hashes.json"
    _write_json(
        g2_source_hashes_path,
        {
            "algorithm": "sha256",
            "coverage_complete": True,
            "immutable_source_verification": True,
            "inventories": inventory_entries,
        },
    )
    g2_summary_path = paths.g2_admission_dir / "summary.md"
    g2_summary_path.write_text("strict G2 admission\n", encoding="utf-8")
    _write_json(
        paths.g2_admission_dir / "manifest.json",
        _full_artifact(
            artifact_type="japan_g2_aggregate_admission",
            admission_passed=True,
            g2_complete=True,
            component_scorecard_eligible=True,
            immutable=True,
            atomic_publish=True,
            source_checksums_verified=True,
            window_coverage_complete=True,
            raw_compacted_parity=True,
            official_corporate_actions_reconciled=True,
            pit_liquidity_complete=True,
            critical_blocker_count=0,
            critical_blockers=[],
            check_count=3,
            passed_check_count=3,
            denominators={
                "g1_listing_windows": 1,
                "g1_rics": 1,
                "active_raw_ohlcv_rows": 10,
                "reference_raw_ohlcv_rows": 5,
                "raw_union_rows": 15,
                "corporate_action_rows": 1,
            },
            promotion_policy={
                "g2_scorecard_credit": True,
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            artifact_sha256={
                "component_contract": hashlib.sha256(g2_contract_path.read_bytes()).hexdigest(),
                "source_hashes": hashlib.sha256(g2_source_hashes_path.read_bytes()).hexdigest(),
                "summary": hashlib.sha256(g2_summary_path.read_bytes()).hexdigest(),
            },
            artifacts={
                "component_contract": "component_contract.csv",
                "source_hashes": "source_hashes.json",
                "summary": "summary.md",
            },
        ),
    )
    _write_json(
        paths.liquidity_dir / "manifest.json",
        _full_artifact(
            status="complete",
            planned_rows=1,
            completed_rows=1,
            failed_rows=0,
            empty_rows=0,
            history_rows=4_900,
            known_at_coverage=1.0,
            missing_known_at_rows=0,
            pit_timestamp_violations=0,
        ),
    )
    _write_csv(
        paths.liquidity_dir / "historical_quality.csv",
        [
            {
                "dataset_id": "liquidity_pit",
                "failed_rows": 0,
                "history_rows": 4_900,
                "coverage": 1.0,
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
            }
        ],
    )
    _write_json(
        paths.pit_plan_dir / "manifest.json",
        _full_artifact(
            status="complete",
            selected_names=1,
            selected_listing_windows=1,
            point_in_time_plan=True,
            window_denominator_preserved=True,
            current_survivor=False,
            datasets=sorted(
                {"fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals"}
            ),
        ),
    )
    _write_json(
        paths.fundamentals_ibes_events_dir / "manifest.json",
        _full_artifact(
            status="complete",
            planned_rows=3,
            completed_rows=3,
            failed_rows=0,
            empty_rows=0,
            history_rows=100,
            known_at_coverage=1.0,
            missing_known_at_rows=0,
            pit_timestamp_violations=0,
        ),
    )
    _write_csv(
        paths.fundamentals_ibes_events_dir / "historical_quality.csv",
        [
            {
                "dataset_id": dataset,
                "failed_rows": 0,
                "history_rows": 10,
                "coverage": 1.0,
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
            }
            for dataset in ("fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals")
        ],
    )
    g3_contract_path = paths.g3_admission_dir / "component_contract.csv"
    _write_csv(
        g3_contract_path,
        [
            {"component": "denominator", "check_id": "g1_and_plan", "passed": True},
            {"component": "pit", "check_id": "known_at_and_validators", "passed": True},
        ],
    )
    g3_source_path = paths.pit_plan_dir / "manifest.json"
    g3_source_hashes_path = paths.g3_admission_dir / "source_hashes.json"
    _write_json(
        g3_source_hashes_path,
        {
            "algorithm": "sha256",
            "coverage_complete": True,
            "immutable_source_verification": True,
            "source_count": 1,
            "missing_source_count": 0,
            "files": {
                str(g3_source_path.resolve()): hashlib.sha256(g3_source_path.read_bytes()).hexdigest()
            },
            "missing_files": [],
        },
    )
    g3_summary_path = paths.g3_admission_dir / "summary.md"
    g3_summary_path.write_text("strict G3 admission\n", encoding="utf-8")
    _write_json(
        paths.g3_admission_dir / "manifest.json",
        _full_artifact(
            artifact_type="japan_g3_aggregate_admission",
            admission_passed=True,
            g3_complete=True,
            component_scorecard_eligible=True,
            immutable=True,
            atomic_publish=True,
            source_checksums_verified=True,
            producer_validator_independence_verified=True,
            decision_calendar_eligibility_recomputed=True,
            critical_blocker_count=0,
            critical_blockers=[],
            check_count=2,
            passed_check_count=2,
            denominators={
                "g1_rics": 1,
                "g1_listing_windows": 1,
                "execution_plan_rows": 4,
                "terminal_rows": 4,
                "history_rows": 100,
            },
            datasets=sorted(
                {"fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals"}
            ),
            lanes=["fundamentals", "consensus", "earnings", "analyst_detail"],
            promotion_policy={
                "g3_scorecard_credit": True,
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            artifact_sha256={
                "component_contract": hashlib.sha256(g3_contract_path.read_bytes()).hexdigest(),
                "source_hashes": hashlib.sha256(g3_source_hashes_path.read_bytes()).hexdigest(),
                "summary": hashlib.sha256(g3_summary_path.read_bytes()).hexdigest(),
            },
            artifacts={
                "component_contract": "component_contract.csv",
                "source_hashes": "source_hashes.json",
                "summary": "summary.md",
            },
        ),
    )
    components = [
        "jpy_fx_rates",
        "exchange_calendars",
        "halt_suspension_events",
        "tick_size_bands",
        "price_limits",
        "lot_sizes",
        "tca_assumptions",
    ]
    g4_quality_path = paths.fx_calendar_tca_dir / "evidence_quality.csv"
    _write_csv(
        g4_quality_path,
        [
            {"component": component, "status": "pass", "validated": True, "coverage": 1.0, "source_rows": 1}
            for component in components
        ],
    )
    _write_json(
        paths.fx_calendar_tca_dir / "manifest.json",
        _full_artifact(
            artifact_type="jpx_g4_full_evidence_assembly",
            status="complete",
            evidence_scope="full_hash_verified_g4_2006_07_to_2026_06",
            research_only=True,
            production_ready=False,
            live_execution_ready=False,
            currency="JPY",
            official_jpx_calendar=True,
            tca_calibrated=True,
            complete_components=components,
            partial_components=[],
            critical_blockers=[],
            component_admission={
                "currency_identity": True,
                "exchange_calendar": True,
                "historical_halts_suspensions": True,
                "tick_sizes": True,
                "pit_tick_rules": True,
                "pit_tick_assignment": True,
                "pit_lot_sizes": True,
                "normal_price_limits": True,
                "exceptional_price_limits": True,
                "price_limits": True,
                "liquidity_aware_tca": True,
            },
            sha256={"evidence_quality.csv": hashlib.sha256(g4_quality_path.read_bytes()).hexdigest()},
            promotion_policy={
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
        ),
    )
    leakage_root = paths.leakage_qa_dir
    decision_path = leakage_root / "decision_panel.csv"
    violation_path = leakage_root / "leakage_violations.csv"
    survivorship_path = leakage_root / "survivorship_reconciliation.csv"
    split_path = leakage_root / "split_ledger.csv"
    feature_path = leakage_root / "feature_spec.json"
    config_path = leakage_root / "audit_config.json"
    _write_csv(
        decision_path,
        [
            {
                "decision_id": "d1", "entity": "7203.T", "decision_time_utc": "2020-01-02T00:00:00Z",
                "listing_start": "1949-05-16T00:00:00Z", "listing_end": "2026-07-10T00:00:00Z",
                "membership_valid_at_decision": True, "max_feature_available_time_utc": "2020-01-01T00:00:00Z",
                "label_start_utc": "2020-01-02T00:00:00Z", "label_end_utc": "2020-01-31T00:00:00Z",
            },
            {
                "decision_id": "d2", "entity": "7203.T", "decision_time_utc": "2021-01-04T00:00:00Z",
                "listing_start": "1949-05-16T00:00:00Z", "listing_end": "2026-07-10T00:00:00Z",
                "membership_valid_at_decision": True, "max_feature_available_time_utc": "2021-01-03T00:00:00Z",
                "label_start_utc": "2021-01-04T00:00:00Z", "label_end_utc": "2021-01-29T00:00:00Z",
            },
            {
                "decision_id": "d3", "entity": "7203.T", "decision_time_utc": "2022-01-04T00:00:00Z",
                "listing_start": "1949-05-16T00:00:00Z", "listing_end": "2026-07-10T00:00:00Z",
                "membership_valid_at_decision": True, "max_feature_available_time_utc": "2022-01-03T00:00:00Z",
                "label_start_utc": "2022-01-04T00:00:00Z", "label_end_utc": "2022-01-31T00:00:00Z",
            },
        ],
    )
    violation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["decision_id", "violation_type"]).to_csv(violation_path, index=False)
    _write_csv(survivorship_path, [{"entity": "7203.T", "status": "pass"}])
    _write_csv(
        split_path,
        [
            {"decision_id": "d1", "split": "train"},
            {"decision_id": "d2", "split": "test"},
            {"decision_id": "d3", "split": "holdout"},
        ],
    )
    _write_json(feature_path, {"features": ["fundamentals_pit__book_to_price"]})
    _write_json(
        config_path,
        {
            "purge_days": 5,
            "embargo_days": 5,
            "holdout_start": "2022-01-01",
            "decision_schedule": "daily_close_plus_one_session",
        },
    )
    leakage_artifacts = {
        "decision_panel": str(decision_path.resolve()),
        "leakage_violations": str(violation_path.resolve()),
        "survivorship_reconciliation": str(survivorship_path.resolve()),
        "split_ledger": str(split_path.resolve()),
        "feature_spec": str(feature_path.resolve()),
        "audit_config": str(config_path.resolve()),
    }
    _write_json(
        leakage_root / "manifest.json",
        _full_artifact(
            total_leakage_violations=0,
            future_data_violations=0,
            train_test_overlap_rows=0,
            survivorship_violations=0,
            decision_rows=3,
            decision_entities=1,
            features=["fundamentals_pit__book_to_price"],
            known_at_coverage=1.0,
            artifacts=leakage_artifacts,
            artifact_sha256={
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in leakage_artifacts.items()
            },
        ),
    )
    paths.commands_log.write_text("uv run pytest tests/test_japan_data_quality_loop.py\n", encoding="utf-8")
    _refresh_hashes(paths)
    return paths


def _refresh_hashes(paths: JapanDataQualityPaths) -> None:
    assert paths.immutable_hashes_path is not None
    files: dict[str, str] = {}
    for path in sorted(paths.immutable_hashes_path.parents[1].rglob("*")):
        if path.is_file() and path != paths.immutable_hashes_path:
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(
        paths.immutable_hashes_path,
        {
            "run_id": "JPN-FULL-IMMUTABLE-AUDIT",
            "status": "pass",
            "scorecard_eligible": True,
            "evidence_scope": "full",
            "algorithm": "sha256",
            "immutable": True,
            "coverage_complete": True,
            "files": files,
        },
    )


def _run(paths: JapanDataQualityPaths, root: Path, suffix: str):
    return run_japan_data_quality_loop(
        paths=paths,
        artifact_root=root / "scorecards",
        run_id=f"JPN-DQ-{suffix}",
    )


def test_all_five_independent_gates_pass_as_binary_research_only_score(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)

    result = _run(paths, tmp_path, "PASS")

    assert result.accepted is True
    assert result.overall_score == 5
    assert [lane.score for lane in result.lanes] == [1, 1, 1, 1, 1]
    assert {lane.status for lane in result.lanes} == {"pass"}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["gate_scoring"] == "five_binary_fail_closed_gates"
    assert manifest["research_only"] is True
    assert manifest["production_ready"] is False
    assert manifest["live_execution_ready"] is False
    assert manifest["ibkr_ready"] is False
    lanes = pd.read_csv(result.lane_scorecard_path)
    assert set(lanes["max_score"]) == {1}
    assert "partial" not in set(lanes["status"])


def test_current_quote_level_master_cannot_pass_g1(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    manifest_path = paths.security_master_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "security_master_kind": "quote_level",
            "listing_level_master": False,
            "quote_level_master": True,
        }
    )
    _write_json(manifest_path, manifest)
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, "QUOTE")
    g1 = result.lanes[0]

    assert result.accepted is False
    assert result.overall_score == 4
    assert g1.score == 0
    assert g1.status == "fail"
    assert "current quote-level security master cannot satisfy G1" in g1.critical_blockers


def test_current_survivor_or_incomplete_window_plan_cannot_pass_g3(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    _patch_manifest(
        paths.pit_plan_dir / "manifest.json",
        selected_listing_windows=0,
        window_denominator_preserved=False,
        current_survivor=True,
    )
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, "SURVIVOR-G3")
    g3 = result.lanes[2]

    assert result.accepted is False
    assert result.overall_score == 4
    assert g3.status == "fail"
    assert any("non-survivor" in blocker for blocker in g3.critical_blockers)


@pytest.mark.parametrize(
    ("gate_index", "mutate"),
    [
        (
            0,
            lambda paths: _patch_manifest(
                paths.universe_reconciliation_dir / "manifest.json",
                status="partial",
                scorecard_eligible=True,
            ),
        ),
        (
            1,
            lambda paths: _patch_manifest(
                paths.g2_admission_dir / "manifest.json",
                admission_passed=False,
            ),
        ),
        (
            2,
            lambda paths: _patch_manifest(
                paths.g3_admission_dir / "manifest.json",
                admission_passed=False,
            ),
        ),
        (
            3,
            lambda paths: _remove_fx_lot_sizes(paths),
        ),
    ],
)
def test_partial_stub_or_incomplete_evidence_fails_whole_gate(
    tmp_path: Path,
    gate_index: int,
    mutate: Callable[[JapanDataQualityPaths], None],
) -> None:
    paths = _seed_complete_evidence(tmp_path)
    mutate(paths)
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, f"GATE-{gate_index}")

    assert result.overall_score == 4
    assert result.lanes[gate_index].score == 0
    assert result.lanes[gate_index].status == "fail"
    assert all(lane.score in {0, 1} for lane in result.lanes)


def test_hash_tampering_fails_g5_closed(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    paths.commands_log.write_text("tampered after immutable inventory\n", encoding="utf-8")

    result = _run(paths, tmp_path, "HASH-TAMPER")
    g5 = result.lanes[4]

    assert result.overall_score == 4
    assert g5.score == 0
    assert any("hash mismatch" in blocker for blocker in g5.critical_blockers)


def test_g2_contract_tampering_fails_g2_and_g5_closed(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    contract_path = paths.g2_admission_dir / "component_contract.csv"
    contract = pd.read_csv(contract_path)
    contract.loc[0, "passed"] = False
    contract.to_csv(contract_path, index=False)

    result = _run(paths, tmp_path, "G2-CONTRACT-TAMPER")

    assert result.lanes[1].status == "fail"
    assert any("component contract" in blocker for blocker in result.lanes[1].critical_blockers)
    assert result.lanes[4].status == "fail"
    assert any("hash mismatch" in blocker for blocker in result.lanes[4].critical_blockers)


def test_future_feature_availability_fails_g5_even_with_fresh_hashes(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    decision_path = paths.leakage_qa_dir / "decision_panel.csv"
    decision = pd.read_csv(decision_path)
    decision.loc[0, "max_feature_available_time_utc"] = "2020-01-03T00:00:00Z"
    decision.to_csv(decision_path, index=False)
    leakage_path = paths.leakage_qa_dir / "manifest.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["artifact_sha256"]["decision_panel"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    _write_json(leakage_path, leakage)
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, "FUTURE-AVAILABILITY")

    assert result.lanes[4].status == "fail"
    assert any("availability" in blocker for blocker in result.lanes[4].critical_blockers)


def test_g4_assembled_file_tampering_fails_g4_with_fresh_outer_hashes(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    quality_path = paths.fx_calendar_tca_dir / "evidence_quality.csv"
    quality = pd.read_csv(quality_path)
    quality.loc[quality["component"].eq("price_limits"), "source_rows"] = 2
    quality.to_csv(quality_path, index=False)
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, "G4-ASSEMBLY-TAMPER")

    assert result.lanes[3].status == "fail"
    assert any("artifact hash mismatch" in blocker for blocker in result.lanes[3].critical_blockers)


def test_self_validated_symbol_map_cannot_pass_g1(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    manifest_path = paths.active_symbol_map_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validator_run_id"] = manifest["producer_run_id"]
    _write_json(manifest_path, manifest)
    _refresh_hashes(paths)

    result = _run(paths, tmp_path, "SELF-VALIDATED-MAP")

    assert result.lanes[0].status == "fail"
    assert any(
        "validated JPX active symbol map" in blocker
        for blocker in result.lanes[0].critical_blockers
    )


def test_security_master_parquet_tampering_fails_g5_closed(tmp_path: Path) -> None:
    paths = _seed_complete_evidence(tmp_path)
    (paths.security_master_dir / "listing_windows.parquet").write_bytes(b"tampered")

    result = _run(paths, tmp_path, "PARQUET-TAMPER")

    assert result.lanes[4].status == "fail"
    assert any("hash mismatch" in blocker for blocker in result.lanes[4].critical_blockers)


def _patch_manifest(path: Path, **values: object) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(values)
    _write_json(path, manifest)


def _patch_csv(path: Path, **values: object) -> None:
    frame = pd.read_csv(path)
    for key, value in values.items():
        frame[key] = value
    frame.to_csv(path, index=False)


def _remove_fx_lot_sizes(paths: JapanDataQualityPaths) -> None:
    manifest_path = paths.fx_calendar_tca_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete_components"] = [
        value for value in manifest["complete_components"] if value != "lot_sizes"
    ]
    _write_json(manifest_path, manifest)
    quality_path = paths.fx_calendar_tca_dir / "evidence_quality.csv"
    quality = pd.read_csv(quality_path)
    quality.loc[quality["component"].eq("lot_sizes"), "validated"] = False
    quality.to_csv(quality_path, index=False)
