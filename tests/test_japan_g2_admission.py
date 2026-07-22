from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_lab.data.japan_g2_admission import (
    JapanG2AdmissionInputs,
    JapanG2OHLCVRun,
    _normalize_windows,
    _outside_window_rows,
    _terminal_payload_contract,
    write_japan_g2_admission,
)


def test_complete_packet_is_admitted_and_published_immutably(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)

    result = write_japan_g2_admission(
        packet.inputs,
        artifact_root=tmp_path / "admissions",
        run_id="JPN-G2-COMPLETE",
    )

    assert result.admitted is True
    assert result.manifest["status"] == "pass"
    assert result.manifest["scorecard_eligible"] is True
    assert result.manifest["component_scorecard_eligible"] is True
    assert result.manifest["promotion_allowed"] is True
    assert result.manifest["promotion_policy"]["g2_scorecard_credit"] is True
    assert result.manifest["promotion_policy"]["alpha"] is False
    assert result.checks["passed"].all()
    for name, digest in result.manifest["artifact_sha256"].items():
        filename = {
            "component_contract": "component_contract.csv",
            "source_hashes": "source_hashes.json",
            "summary": "summary.md",
        }[name]
        assert _sha256(result.artifact_dir / filename) == digest

    with pytest.raises(FileExistsError, match="immutable"):
        write_japan_g2_admission(
            packet.inputs,
            artifact_root=tmp_path / "admissions",
            run_id="JPN-G2-COMPLETE",
        )


def test_partial_writer_denominators_fail_closed_without_promotion(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    active_manifest_path = packet.inputs.active_ohlcv.historical_dir / "historical_run_manifest.json"
    active_manifest = json.loads(active_manifest_path.read_text())
    active_manifest["completed_rows"] = 0
    _write_json(active_manifest_path, active_manifest)
    _write_inventory(packet.inputs.active_ohlcv.hash_inventory_path, packet.active_hash_files)

    liquidity_manifest_path = packet.inputs.liquidity_dir / "manifest.json"
    liquidity_manifest = json.loads(liquidity_manifest_path.read_text())
    liquidity_manifest["completed_rows"] = 0
    _write_json(liquidity_manifest_path, liquidity_manifest)
    liquidity_terminal_path = packet.inputs.liquidity_dir / "historical_manifest.csv"
    liquidity_terminal = pd.read_csv(liquidity_terminal_path)
    liquidity_terminal.loc[0, "status"] = "completed_partial"
    liquidity_terminal.to_csv(liquidity_terminal_path, index=False)
    _write_inventory(packet.inputs.component_hash_inventory_path, packet.component_hash_files)

    result = write_japan_g2_admission(
        packet.inputs,
        artifact_root=tmp_path / "admissions",
        run_id="JPN-G2-PARTIAL",
    )

    assert result.admitted is False
    assert result.manifest["status"] == "failed_admission"
    assert result.manifest["evidence_scope"] == "failed_no_admission"
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["component_scorecard_eligible"] is False
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["promotion_policy"]["g2_scorecard_credit"] is False
    assert "ohlcv_active:run_denominator_complete" in result.manifest["critical_blockers"]
    assert "liquidity:full_terminal_liquidity_plan" in result.manifest["critical_blockers"]
    assert result.artifact_dir.is_dir()
    assert not list((tmp_path / "admissions").glob(".JPN-G2-PARTIAL.*"))


def test_hash_tampering_blocks_an_otherwise_complete_packet(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    depth_path = packet.inputs.reference_ohlcv.depth_dir / "overall.csv"
    depth = pd.read_csv(depth_path)
    depth["duplicate_ohlcv_rows"] = 1
    depth.to_csv(depth_path, index=False)

    result = write_japan_g2_admission(
        packet.inputs,
        artifact_root=tmp_path / "admissions",
        run_id="JPN-G2-TAMPERED",
    )

    assert result.admitted is False
    assert result.manifest["promotion_allowed"] is False
    assert "hashes_reference:sha256_inventory_complete_and_verified" in result.manifest[
        "critical_blockers"
    ]
    assert "ohlcv_reference:depth_qa_reconciles_to_raw" in result.manifest["critical_blockers"]


def test_partial_payload_contract_accepts_only_pit_explained_prelisting(tmp_path: Path) -> None:
    coverage_path = tmp_path / "partial.coverage.csv"
    pd.DataFrame({"ticker": ["AAA.T", "NEW.T"], "rows": [2, 0]}).to_csv(coverage_path, index=False)
    terminal = pd.DataFrame(
        {
            "plan_row_id": ["ACTIVE-1"],
            "status": ["completed_partial"],
            "tickers": ["AAA.T,NEW.T"],
            "start_date": ["2020-01-01"],
            "end_date": ["2020-12-31"],
            "long_ohlcv_rows": [2],
            "failures": [1],
            "coverage_path": [str(coverage_path)],
        }
    )
    windows = _normalize_windows(
        pd.DataFrame(
            {
                "refinitiv_ric": ["AAA.T", "NEW.T"],
                "listing_start": ["2020-01-01", "2021-03-01"],
                "listing_end": ["2026-06-30", "2026-06-30"],
                "right_open": [True, True],
                "window_end_known_at": [pd.NA, pd.NA],
                "pit_usable": [True, True],
            }
        )
    )
    required: set[Path] = set()

    accepted, detail = _terminal_payload_contract(terminal, windows, tmp_path, required)

    assert accepted is True
    assert "structural_missing_rics=1" in detail
    assert coverage_path in required

    windows.loc[windows["ric"].eq("NEW.T"), "start"] = pd.Timestamp("2020-06-01")
    accepted, detail = _terminal_payload_contract(terminal, windows, tmp_path, set())
    assert accepted is False
    assert "unexplained zero-history RICs=NEW.T" in detail


def test_outside_window_check_extends_right_open_listing_past_snapshot_cutoff() -> None:
    windows = _normalize_windows(
        pd.DataFrame(
            {
                "refinitiv_ric": ["LIVE.T"],
                "listing_start": ["2020-01-01"],
                "listing_end": ["2026-06-30"],
                "right_open": [True],
                "pit_usable": [True],
            }
        )
    )
    rows = pd.DataFrame({"ric": ["LIVE.T"], "date": [pd.Timestamp("2026-07-06")]})
    assert _outside_window_rows(rows, windows) == 0


def _complete_packet(root: Path) -> SimpleNamespace:
    g1 = root / "g1_security"
    validation = root / "g1_validation"
    active_map = root / "active_map"
    reference_map = root / "reference_map"
    for directory in (g1, validation, active_map, reference_map):
        directory.mkdir(parents=True)

    full_manifest = {"status": "complete", "scorecard_eligible": True, "evidence_scope": "full"}
    _write_json(g1 / "manifest.json", {**full_manifest, "run_id": "G1-SECURITY"})
    _write_json(
        validation / "manifest.json",
        {
            **full_manifest,
            "run_id": "G1-VALIDATION",
            "validation_status": "pass",
            "critical_blocker_count": 0,
            "source_checksums_verified": True,
        },
    )
    windows = pd.DataFrame(
        {
            "refinitiv_ric": ["AAA.T", "BBB.T"],
            "listing_start": ["2020-01-01", "2020-01-01"],
            "listing_end": ["2020-01-02", "2020-01-02"],
        }
    )
    windows.to_parquet(g1 / "listing_windows.parquet", index=False)
    windows.iloc[[0]].to_csv(active_map / "symbol_map.csv", index=False)
    windows.iloc[[1]].to_csv(reference_map / "symbol_map.csv", index=False)
    _write_json(
        active_map / "manifest.json",
        {**full_manifest, "run_id": "G1-ACTIVE", "map_role": "active", "validation_status": "pass"},
    )
    _write_json(
        reference_map / "manifest.json",
        {
            **full_manifest,
            "run_id": "G1-REFERENCE",
            "map_role": "reference",
            "validation_status": "pass",
        },
    )

    active_run, active_files = _ohlcv_run(root, "active", "AAA.T", 100.0)
    reference_run, reference_files = _ohlcv_run(root, "reference", "BBB.T", 200.0)

    corporate_actions = root / "corporate_actions"
    (corporate_actions / "events").mkdir(parents=True)
    pd.DataFrame(
        {
            "plan_row_id": ["CA-1"],
            "dataset_id": ["corporate_actions"],
            "ric_list": ["AAA.T,BBB.T"],
        }
    ).to_csv(corporate_actions / "event_plan.csv", index=False)
    pd.DataFrame(
        {
            "plan_row_id": ["CA-1"],
            "status": ["completed"],
            "event_rows": [1],
            "known_at_rows": [1],
            "valid_event_date_rows": [1],
            "alpha_usable_rows": [1],
            "response_success": [True],
        }
    ).to_csv(corporate_actions / "event_manifest.csv", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["corporate_actions"],
            "plan_rows": [1],
            "completed_rows": [1],
            "failed_rows": [0],
            "empty_rows": [0],
            "event_rows": [1],
            "known_at_rows": [1],
            "valid_event_date_rows": [1],
            "alpha_usable_rows": [1],
            "min_coverage": [1.0],
        }
    ).to_csv(corporate_actions / "event_quality.csv", index=False)
    pd.DataFrame(
        {
            "event_key": ["EVENT-1"],
            "ticker": ["AAA.T"],
            "event_date": ["2020-01-02"],
            "known_at_date": ["2020-01-01"],
            "alpha_usable": [True],
        }
    ).to_parquet(corporate_actions / "events" / "events.parquet", index=False)
    _write_json(
        corporate_actions / "manifest.json",
        {
            "run_id": "CORPACT",
            "status": "complete",
            "official_source": True,
            "planned_rows": 1,
            "completed_rows": 1,
            "failed_rows": 0,
            "empty_rows": 0,
            "event_rows": 1,
        },
    )

    adjusted = root / "adjusted_returns"
    adjusted.mkdir()
    pd.DataFrame({"event_key": ["EVENT-1"]}).to_parquet(
        adjusted / "corporate_action_events_normalized.parquet", index=False
    )
    pd.DataFrame(
        {
            "event_key": ["EVENT-1"],
            "ex_trade_date": ["2020-01-02"],
            "reconciliation_status": ["applied"],
        }
    ).to_parquet(adjusted / "event_price_reconciliation.parquet", index=False)
    pd.DataFrame(
        {
            "ticker": ["AAA.T", "AAA.T", "BBB.T", "BBB.T"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            "total_return_local_net": [None, 0.02, None, 0.01],
        }
    ).to_parquet(adjusted / "daily_adjusted_returns.parquet", index=False)
    pd.DataFrame(
        {
            "event_rows": [1],
            "valid_event_rows": [1],
            "matched_event_rows": [1],
            "quarantined_event_rows": [0],
            "matched_event_share": [1.0],
            "quarantine_share": [0.0],
        }
    ).to_csv(adjusted / "reconciliation_quality.csv", index=False)
    _write_json(
        adjusted / "manifest.json",
        {
            "run_id": "ADJUSTED",
            "status": "pass",
            "production_adjusted_return_ready": True,
            "event_rows": 1,
            "valid_event_rows": 1,
            "matched_event_rows": 1,
            "matched_event_share": 1.0,
            "quarantined_event_rows": 0,
            "quarantine_share": 0.0,
            "ambiguous_factor_rows": 0,
            "daily_adjusted_return_rows": 4,
            "non_null_total_return_rows": 2,
        },
    )

    liquidity = root / "liquidity"
    (liquidity / "history").mkdir(parents=True)
    pd.DataFrame(
        {
            "plan_row_id": ["LIQ-1"],
            "dataset_id": ["liquidity_market_cap_free_float"],
            "ric_list": ["AAA.T,BBB.T"],
            "start_date": ["2020-01-01"],
            "end_date": ["2020-01-02"],
        }
    ).to_csv(liquidity / "historical_plan.csv", index=False)
    pd.DataFrame(
        {
            "plan_row_id": ["LIQ-1"],
            "status": ["completed"],
            "history_rows": [2],
            "outside_requested_rows": [0],
        }
    ).to_csv(liquidity / "historical_manifest.csv", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["liquidity_market_cap_free_float"],
            "plan_rows": [1],
            "completed_rows": [1],
            "failed_rows": [0],
            "empty_rows": [0],
            "history_rows": [2],
            "avg_coverage": [1.0],
            "min_coverage": [1.0],
        }
    ).to_csv(liquidity / "historical_quality.csv", index=False)
    pd.DataFrame(
        {
            "requested_ric": ["AAA.T", "BBB.T"],
            "asof_date": ["2020-01-02", "2020-01-02"],
            "field_role": ["adv_value", "adv_value"],
            "value": [1_000_000.0, 2_000_000.0],
            "plan_row_id": ["LIQ-1", "LIQ-1"],
            "inside_requested_window": [True, True],
        }
    ).to_parquet(liquidity / "history" / "liquidity.parquet", index=False)
    _write_json(
        liquidity / "manifest.json",
        {
            "run_id": "LIQUIDITY",
            "planned_rows": 1,
            "completed_rows": 1,
            "failed_rows": 0,
            "empty_rows": 0,
            "history_rows": 2,
            "output_root": str((liquidity / "history").resolve()),
        },
    )

    component_files = [
        g1 / "manifest.json",
        g1 / "listing_windows.parquet",
        validation / "manifest.json",
        active_map / "manifest.json",
        active_map / "symbol_map.csv",
        reference_map / "manifest.json",
        reference_map / "symbol_map.csv",
        corporate_actions / "manifest.json",
        corporate_actions / "event_plan.csv",
        corporate_actions / "event_manifest.csv",
        corporate_actions / "event_quality.csv",
        corporate_actions / "events" / "events.parquet",
        adjusted / "manifest.json",
        adjusted / "corporate_action_events_normalized.parquet",
        adjusted / "event_price_reconciliation.parquet",
        adjusted / "daily_adjusted_returns.parquet",
        adjusted / "reconciliation_quality.csv",
        liquidity / "manifest.json",
        liquidity / "historical_plan.csv",
        liquidity / "historical_manifest.csv",
        liquidity / "historical_quality.csv",
        liquidity / "history" / "liquidity.parquet",
    ]
    component_inventory = root / "component_hashes.json"
    _write_inventory(component_inventory, component_files)
    inputs = JapanG2AdmissionInputs(
        g1_security_master_dir=g1,
        g1_validation_dir=validation,
        active_symbol_map_dir=active_map,
        reference_symbol_map_dir=reference_map,
        active_ohlcv=active_run,
        reference_ohlcv=reference_run,
        corporate_actions_dir=corporate_actions,
        adjusted_returns_dir=adjusted,
        liquidity_dir=liquidity,
        component_hash_inventory_path=component_inventory,
    )
    return SimpleNamespace(
        inputs=inputs,
        active_hash_files=active_files,
        reference_hash_files=reference_files,
        component_hash_files=component_files,
    )


def _ohlcv_run(root: Path, role: str, ric: str, close: float) -> tuple[JapanG2OHLCVRun, list[Path]]:
    historical = root / f"{role}_historical"
    depth = root / f"{role}_depth"
    compaction = root / f"{role}_compaction"
    raw_root = root / f"{role}_raw"
    compact_root = root / f"{role}_compacted"
    row_equality = root / f"{role}_row_equality"
    for directory in (historical, depth, compaction, raw_root, compact_root, row_equality):
        directory.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "ticker": [ric, ric],
            "date": ["2020-01-01", "2020-01-02"],
            "close": [close, close + 1.0],
            "volume": [1_000.0, 1_100.0],
        }
    )
    raw.to_parquet(raw_root / "part.parquet", index=False)
    raw.to_parquet(compact_root / "part.parquet", index=False)
    plan_id = f"{role.upper()}-1"
    pd.DataFrame(
        {
            "plan_row_id": [plan_id],
            "tickers": [ric],
            "start_date": ["2020-01-01"],
            "end_date": ["2020-01-02"],
        }
    ).to_csv(historical / "historical_plan.csv", index=False)
    pd.DataFrame(
        {
            "plan_row_id": [plan_id],
            "status": ["completed"],
            "long_ohlcv_rows": [2],
            "failures": [0],
        }
    ).to_csv(historical / "historical_manifest.csv", index=False)
    _write_json(
        historical / "historical_run_manifest.json",
        {
            "run_id": f"OHLCV-{role.upper()}",
            "planned_rows": 1,
            "executed_rows": 1,
            "completed_rows": 1,
            "failed_rows": 0,
            "empty_rows": 0,
            "long_ohlcv_root": str(raw_root.resolve()),
        },
    )
    pd.DataFrame(
        {
            "rows": [2],
            "tickers": [1],
            "non_positive_close_rows": [0],
            "volume_missing_pct": [0.0],
        }
    ).to_csv(depth / "overall.csv", index=False)
    _write_json(depth / "data_depth_manifest.json", {"run_id": f"DEPTH-{role.upper()}"})
    _write_json(
        compaction / "manifest.json",
        {
            "run_id": f"COMPACT-{role.upper()}",
            "status": "complete",
            "source_rows": 2,
            "output_rows": 2,
            "failed_groups": 0,
            "output_root": str(compact_root.resolve()),
        },
    )
    _write_json(
        row_equality / "manifest.json",
        {
            "run_id": f"ROW-EQUALITY-{role.upper()}",
            "status": "pass",
            "row_equality_passed": True,
            "source_root": str(raw_root.resolve()),
            "compacted_root": str(compact_root.resolve()),
            "source_rows": 2,
            "compacted_rows": 2,
            "failed_years": [],
        },
    )
    hash_files = [
        historical / "historical_run_manifest.json",
        historical / "historical_plan.csv",
        historical / "historical_manifest.csv",
        depth / "data_depth_manifest.json",
        depth / "overall.csv",
        compaction / "manifest.json",
        row_equality / "manifest.json",
        raw_root / "part.parquet",
        compact_root / "part.parquet",
    ]
    inventory = root / f"{role}_hashes.json"
    _write_inventory(inventory, hash_files)
    return JapanG2OHLCVRun(historical, depth, compaction, inventory, row_equality), hash_files


def _write_inventory(path: Path, files: list[Path]) -> None:
    _write_json(
        path,
        {
            "algorithm": "sha256",
            "immutable": True,
            "coverage_complete": True,
            "files": {str(file.resolve()): _sha256(file) for file in files},
        },
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
