from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_lab.data.japan_g3_admission import (
    EXPECTED_G1_LISTING_WINDOWS,
    EXPECTED_G1_RICS,
    JapanG3AdmissionInputs,
    JapanG3TerminalRun,
    write_japan_g3_admission,
)


def test_complete_packet_is_atomic_immutable_and_scorecard_eligible(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)

    result = write_japan_g3_admission(
        packet.inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-COMPLETE"
    )

    assert result.admitted is True
    assert result.manifest["status"] == "pass"
    assert result.manifest["scorecard_eligible"] is True
    assert result.manifest["producer_validator_independence_verified"] is True
    assert result.manifest["decision_calendar_eligibility_recomputed"] is True
    assert result.manifest["denominators"]["g1_rics"] == EXPECTED_G1_RICS
    assert result.manifest["denominators"]["g1_listing_windows"] == EXPECTED_G1_LISTING_WINDOWS
    assert result.checks["passed"].all()
    analyst_terminal = pd.read_csv(packet.producers["analyst_detail"] / "row_manifest.csv")
    assert "structural_not_applicable" in set(analyst_terminal["status"])
    for key, digest in result.manifest["artifact_sha256"].items():
        assert _sha256(result.artifact_dir / result.manifest["artifacts"][key]) == digest
    assert not list((tmp_path / "admissions").glob(".JPN-G3-COMPLETE.*"))

    with pytest.raises(FileExistsError, match="immutable"):
        write_japan_g3_admission(
            packet.inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-COMPLETE"
        )


def test_partial_qa_producer_can_never_pass(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    manifest_path = packet.producers["fundamentals"] / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "status": "completed_partial",
            "evidence_scope": "qa",
            "scorecard_eligible": False,
            "partial": True,
            "scope": "bounded_qa",
        }
    )
    _write_json(manifest_path, manifest)

    result = write_japan_g3_admission(
        packet.inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-QA"
    )

    assert result.admitted is False
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["promotion_allowed"] is False
    assert "lane_fundamentals:full_non_qa_producer_manifest" in result.manifest["critical_blockers"]


def test_known_at_after_last_decision_fails_recomputed_pit_eligibility(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    payload_path = packet.payloads["earnings"]
    payload = pd.read_parquet(payload_path)
    payload["available_time_utc"] = "2032-01-01T00:00:00Z"
    payload.to_parquet(payload_path, index=False)

    result = write_japan_g3_admission(
        packet.inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-LEAKAGE"
    )

    assert result.admitted is False
    assert "lane_earnings:known_at_decision_eligibility_recomputed" in result.manifest[
        "critical_blockers"
    ]
    assert "pit:all_rows_known_at_or_before_eligible_decision" in result.manifest["critical_blockers"]


def test_ordinary_and_analyst_detail_producers_cannot_share_a_lane(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    runs = list(packet.inputs.terminal_runs)
    analyst_index = next(index for index, run in enumerate(runs) if run.lane == "analyst_detail")
    consensus = next(run for run in runs if run.lane == "consensus")
    runs[analyst_index] = JapanG3TerminalRun(
        lane="analyst_detail",
        producer_dir=consensus.producer_dir,
        validator_dir=consensus.validator_dir,
    )
    inputs = replace(packet.inputs, terminal_runs=tuple(runs))

    result = write_japan_g3_admission(
        inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-MIXED-LANES"
    )

    assert result.admitted is False
    assert "lanes:ordinary_and_analyst_detail_are_separate" in result.manifest["critical_blockers"]
    assert "lane_analyst_detail:full_non_qa_producer_manifest" in result.manifest[
        "critical_blockers"
    ]


def test_declared_upstream_hash_tampering_fails_closed(tmp_path: Path) -> None:
    packet = _complete_packet(tmp_path)
    component = packet.g2_dir / "component_contract.csv"
    component.write_text(component.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    result = write_japan_g3_admission(
        packet.inputs, artifact_root=tmp_path / "admissions", run_id="JPN-G3-TAMPERED"
    )

    assert result.admitted is False
    assert "g2:packet_artifact_hashes_verified" in result.manifest["critical_blockers"]


def test_cli_exposes_all_four_separate_lanes() -> None:
    cli_path = Path(__file__).parents[1] / "examples" / "build_japan_g3_admission.py"
    spec = importlib.util.spec_from_file_location("build_japan_g3_admission", cli_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--run-id",
            "RUN",
            "--g1-validation-dir",
            "g1",
            "--g2-admission-dir",
            "g2",
            "--pit-plan-dir",
            "pit",
            "--execution-plan-dir",
            "execution",
            "--sector-routing-path",
            "sector.csv",
            "--decision-calendar-path",
            "calendar.csv",
            "--fundamentals-producer-dir",
            "fund-prod",
            "--fundamentals-validator-dir",
            "fund-val",
            "--consensus-producer-dir",
            "cons-prod",
            "--consensus-validator-dir",
            "cons-val",
            "--earnings-producer-dir",
            "earn-prod",
            "--earnings-validator-dir",
            "earn-val",
            "--analyst-detail-producer-dir",
            "analyst-prod",
            "--analyst-detail-validator-dir",
            "analyst-val",
        ]
    )

    assert args.analyst_detail_producer_dir == Path("analyst-prod")
    assert args.fundamentals_validator_dir == Path("fund-val")


def _complete_packet(root: Path) -> SimpleNamespace:
    g1_dir = root / "g1"
    g2_dir = root / "g2"
    pit_dir = root / "pit"
    execution_dir = root / "execution"
    for directory in (g1_dir, g2_dir, pit_dir, execution_dir):
        directory.mkdir(parents=True)

    g1_manifest = {
        "run_id": "G1-VALIDATOR",
        "producer_run_id": "G1-PRODUCER",
        "validator_run_id": "G1-VALIDATOR",
        "status": "pass",
        "scorecard_eligible": True,
        "evidence_scope": "full",
        "independent_audit": True,
        "validator_type": "independent_process",
        "source_checksums_verified": True,
        "critical_blocker_count": 0,
    }
    _write_json(g1_dir / "manifest.json", g1_manifest)

    for name, content in (
        ("component_contract.csv", "component,check_id,passed\ng2,complete,True\n"),
        ("source_hashes.json", '{"algorithm":"sha256","coverage_complete":true}\n'),
        ("summary.md", "# Strict G2\n"),
    ):
        (g2_dir / name).write_text(content, encoding="utf-8")
    g2_artifacts = {
        "component_contract": "component_contract.csv",
        "source_hashes": "source_hashes.json",
        "summary": "summary.md",
    }
    g2_manifest = {
        "run_id": "G2-STRICT",
        "artifact_type": "japan_g2_aggregate_admission",
        "status": "pass",
        "admission_passed": True,
        "g2_complete": True,
        "scorecard_eligible": True,
        "evidence_scope": "full",
        "partial": False,
        "immutable": True,
        "atomic_publish": True,
        "source_checksums_verified": True,
        "window_coverage_complete": True,
        "raw_compacted_parity": True,
        "official_corporate_actions_reconciled": True,
        "pit_liquidity_complete": True,
        "critical_blocker_count": 0,
        "check_count": 73,
        "passed_check_count": 73,
        "denominators": {
            "g1_rics": EXPECTED_G1_RICS,
            "g1_listing_windows": EXPECTED_G1_LISTING_WINDOWS,
        },
        "artifacts": g2_artifacts,
        "artifact_sha256": {
            key: _sha256(g2_dir / path) for key, path in g2_artifacts.items()
        },
    }
    _write_json(g2_dir / "manifest.json", g2_manifest)

    rics = [f"J{index:04d}.T" for index in range(EXPECTED_G1_RICS)]
    current = pd.DataFrame(
        {
            "refinitiv_ric": rics,
            "listing_start": "2010-01-01",
            "listing_end": "2030-12-31",
            "known_at": "2009-12-01T00:00:00Z",
            "right_open": False,
            "pit_usable": True,
        }
    )
    historical = pd.DataFrame(
        {
            "refinitiv_ric": rics[: EXPECTED_G1_LISTING_WINDOWS - EXPECTED_G1_RICS],
            "listing_start": "2000-01-01",
            "listing_end": "2009-12-31",
            "known_at": "1999-12-01T00:00:00Z",
            "right_open": False,
            "pit_usable": True,
        }
    )
    selected = pd.concat([current, historical], ignore_index=True)
    selected_path = pit_dir / "selected_universe.csv"
    selected.to_csv(selected_path, index=False)
    pit_manifest = {
        "run_id": "PIT-FULL",
        "status": "complete",
        "scorecard_eligible": True,
        "evidence_scope": "full",
        "point_in_time_plan": True,
        "current_survivor": False,
        "window_denominator_preserved": True,
        "g1_independent_validation_admitted": True,
        "datasets": ["fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals"],
        "selected_names": EXPECTED_G1_RICS,
        "selected_listing_windows": EXPECTED_G1_LISTING_WINDOWS,
        "provenance": {
            "g1_independent_validation": {
                "path": str((g1_dir / "manifest.json").resolve()),
                "sha256": _sha256(g1_dir / "manifest.json"),
                "admitted": True,
            }
        },
        "artifacts": {"selected_universe": str(selected_path.resolve())},
        "artifact_sha256": {"selected_universe": _sha256(selected_path)},
    }
    _write_json(pit_dir / "manifest.json", pit_manifest)

    ric_list = ",".join(rics)
    plan_specs = {
        "fundamentals": [
            {
                "plan_row_id": "fund-all",
                "dataset_id": "fundamentals_pit",
                "route_class": "general",
                "route": "get_data_parameterized",
                "frequency": "FY",
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
                "ric_list": ric_list,
            }
        ],
        "consensus": [
            {
                "plan_row_id": "consensus-all",
                "dataset_id": "ibes_estimates_revisions",
                "route": "get_history",
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
                "ric_list": ric_list,
            }
        ],
        "earnings": [
            {
                "plan_row_id": "earnings-all",
                "dataset_id": "earnings_events_actuals",
                "route": "get_data_parameterized",
                "frequency": "FY",
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
                "ric_list": ric_list,
            }
        ],
        "analyst_detail": [
            {
                "plan_row_id": "analyst-all",
                "dataset_id": "ibes_estimates_revisions",
                "route": "get_data_parameterized_self_keyed",
                "estimate_type": "recommendation",
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
                "ric_list": ric_list,
            },
            {
                "plan_row_id": "analyst-post-listing",
                "dataset_id": "ibes_estimates_revisions",
                "route": "get_data_parameterized_self_keyed",
                "estimate_type": "target_price",
                "start_date": "2031-01-01",
                "end_date": "2035-12-31",
                "ric_list": rics[0],
            },
        ],
    }
    plan_keys = {
        "fundamentals": "fundamentals_plan",
        "consensus": "consensus_plan",
        "earnings": "earnings_plan",
        "analyst_detail": "analyst_detail_plan",
    }
    plan_paths: dict[str, Path] = {}
    plan_frames: dict[str, pd.DataFrame] = {}
    for lane, rows in plan_specs.items():
        frame = pd.DataFrame(rows)
        frame["ric_count"] = frame["ric_list"].str.count(",") + 1
        frame["status"] = (
            "blocked_requires_bulk_or_explicit_long_run"
            if lane == "analyst_detail"
            else "ready_after_vendor_health"
        )
        key = plan_keys[lane]
        path = execution_dir / f"{key}.csv"
        frame.to_csv(path, index=False)
        plan_paths[lane] = path
        plan_frames[lane] = frame

    execution_manifest = {
        "run_id": "G3-EXECUTION-FULL",
        "status": "ready",
        "scorecard_eligible": True,
        "evidence_scope": "full_execution_plan",
        "g1_independent_validation_admitted": True,
        "selected_names": EXPECTED_G1_RICS,
        "total_plan_rows": sum(len(frame) for frame in plan_frames.values()),
        "input_sha256": {
            str((pit_dir / "manifest.json").resolve()): _sha256(pit_dir / "manifest.json"),
            str(selected_path.resolve()): _sha256(selected_path),
        },
        "artifacts": {
            plan_keys[lane]: str(path.resolve()) for lane, path in plan_paths.items()
        },
        "artifact_sha256": {
            plan_keys[lane]: _sha256(path) for lane, path in plan_paths.items()
        },
    }
    for lane, frame in plan_frames.items():
        execution_manifest[f"{plan_keys[lane]}_rows"] = len(frame)
    _write_json(execution_dir / "manifest.json", execution_manifest)

    sector_path = root / "sector_routing.csv"
    pd.DataFrame(
        {"ric": rics, "fundamental_route_class": "general"}
    ).to_csv(sector_path, index=False)
    calendar_path = root / "decision_calendar.csv"
    pd.DataFrame(
        {"ric": rics, "decision_time_utc": "2021-01-04T06:00:00Z"}
    ).to_csv(calendar_path, index=False)

    producers: dict[str, Path] = {}
    payloads: dict[str, Path] = {}
    terminal_runs: list[JapanG3TerminalRun] = []
    for lane in ("fundamentals", "consensus", "earnings", "analyst_detail"):
        producer_dir = root / f"producer-{lane}"
        validator_dir = root / f"validator-{lane}"
        data_dir = producer_dir / "data"
        data_dir.mkdir(parents=True)
        validator_dir.mkdir(parents=True)
        producer_id = f"PRODUCER-{lane.upper()}"
        payload_path = data_dir / f"{lane}.parquet"
        pd.DataFrame(
            {
                "ric": [rics[0]],
                "dataset_id": [
                    "fundamentals_pit"
                    if lane == "fundamentals"
                    else "earnings_events_actuals"
                    if lane == "earnings"
                    else "ibes_estimates_revisions"
                ],
                "observation_date": ["2020-06-30"],
                "available_time_utc": ["2020-07-02T00:00:00Z"],
                "eligible_decision_time_utc": ["2021-01-04T06:00:00Z"],
                "field_role": ["revenue" if lane == "fundamentals" else "actual_eps" if lane == "earnings" else "revision_event"],
                "source_field": ["TR.Revenue" if lane == "fundamentals" else "TR.EPSActValue" if lane == "earnings" else "TR.EPSMean"],
                "pit_timestamp_violation": [False],
            }
        ).to_parquet(payload_path, index=False)
        payloads[lane] = payload_path
        terminal_rows = [
            {
                "plan_row_id": plan_frames[lane].iloc[0]["plan_row_id"],
                "dataset_id": plan_frames[lane].iloc[0]["dataset_id"],
                "status": "completed",
                "output_path": str(payload_path.resolve()),
                "history_rows": 1,
                "structural_reason": "",
            }
        ]
        if lane == "analyst_detail":
            terminal_rows.append(
                {
                    "plan_row_id": "analyst-post-listing",
                    "dataset_id": "ibes_estimates_revisions",
                    "status": "structural_not_applicable",
                    "output_path": "",
                    "history_rows": 0,
                    "structural_reason": "post_delisting_window",
                }
            )
        row_manifest_path = producer_dir / "row_manifest.csv"
        pd.DataFrame(terminal_rows).to_csv(row_manifest_path, index=False)
        producer_manifest = {
            "run_id": producer_id,
            "status": "complete",
            "scorecard_eligible": True,
            "evidence_scope": "full",
            "partial": False,
            "lane": lane,
            "critical_blocker_count": 0,
            "planned_rows": len(plan_frames[lane]),
            "terminal_rows": len(terminal_rows),
            "plan_path": str(plan_paths[lane].resolve()),
            "plan_sha256": _sha256(plan_paths[lane]),
            "artifacts": {"row_manifest": str(row_manifest_path.resolve())},
            "artifact_sha256": {"row_manifest": _sha256(row_manifest_path)},
        }
        producer_manifest_path = producer_dir / "manifest.json"
        _write_json(producer_manifest_path, producer_manifest)
        validator_manifest = {
            "run_id": f"VALIDATOR-{lane.upper()}",
            "validator_run_id": f"VALIDATOR-{lane.upper()}",
            "producer_run_id": producer_id,
            "producer_manifest_sha256": _sha256(producer_manifest_path),
            "status": "pass",
            "scorecard_eligible": True,
            "evidence_scope": "full",
            "independent_audit": True,
            "validator_type": "independent_process",
            "source_checksums_verified": True,
            "critical_blocker_count": 0,
        }
        _write_json(validator_dir / "manifest.json", validator_manifest)
        producers[lane] = producer_dir
        terminal_runs.append(JapanG3TerminalRun(lane, producer_dir, validator_dir))

    inputs = JapanG3AdmissionInputs(
        g1_validation_dir=g1_dir,
        g2_admission_dir=g2_dir,
        pit_plan_dir=pit_dir,
        execution_plan_dir=execution_dir,
        sector_routing_path=sector_path,
        decision_calendar_path=calendar_path,
        terminal_runs=tuple(terminal_runs),
    )
    return SimpleNamespace(
        inputs=inputs,
        g2_dir=g2_dir,
        producers=producers,
        payloads=payloads,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
