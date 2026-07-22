from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.agents.japan_data_quality_loop import (
    JapanDataQualityPaths,
    _g5_row_evidence_ready,
)
import quant_lab.data.japan_g5_decision_calendar_audit as audit_module
from quant_lab.data.japan_g5_decision_calendar_audit import (
    JapanG5AuditConfig,
    JapanG5Input,
    write_japan_g5_decision_calendar_audit,
)


FEATURE = "fundamentals_pit__book_to_price"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _full_manifest(**values: object) -> dict[str, object]:
    return {
        "run_id": "JPN-FULL-UPSTREAM",
        "status": "pass",
        "scorecard_eligible": True,
        "evidence_scope": "full",
        **values,
    }


def _source_manifest(path: Path, key: str, manifest_path: Path, **values: object) -> None:
    _write_json(
        manifest_path,
        _full_manifest(
            artifacts={key: str(path.resolve())},
            artifact_sha256={key: _sha256(path)},
            **values,
        ),
    )


def _seed_required_gate_inputs(gate: Path) -> None:
    gate.mkdir(parents=True, exist_ok=True)
    for name in (
        "entities.parquet",
        "identifier_history.parquet",
        "universe_membership_pit.parquet",
    ):
        (gate / name).write_bytes(b"immutable-gate-input\n")
    for name in (
        "evidence_quality.csv",
        "historical_manifest.csv",
        "historical_plan.csv",
        "historical_quality.csv",
        "overall.csv",
        "symbol_map.csv",
        "component_contract.csv",
        "summary.md",
    ):
        (gate / name).write_text("status\npass\n", encoding="utf-8")
    _write_json(gate / "source_hashes.json", {"status": "pass"})
    _write_json(gate / "historical_run_manifest.json", _full_manifest())


def _fixture(tmp_path: Path) -> tuple[
    JapanDataQualityPaths,
    JapanG5Input,
    JapanG5Input,
    JapanG5Input,
    JapanG5AuditConfig,
]:
    gate = tmp_path / "gate"
    source = tmp_path / "source"
    output = tmp_path / "g5"
    source.mkdir(parents=True)
    _seed_required_gate_inputs(gate)

    decision_path = source / "decision_calendar.csv"
    pd.DataFrame(
        [
            {
                "decision_id": "d-train",
                "entity": "7203.T",
                "decision_time_utc": "2020-01-01T06:00:00Z",
                "label_start_utc": "2020-01-01T06:00:00Z",
                "label_end_utc": "2020-01-02T06:00:00Z",
            },
            {
                "decision_id": "d-test",
                "entity": "7203.T",
                "decision_time_utc": "2020-01-10T06:00:00Z",
                "label_start_utc": "2020-01-10T06:00:00Z",
                "label_end_utc": "2020-01-11T06:00:00Z",
            },
            {
                "decision_id": "d-holdout",
                "entity": "7203.T",
                "decision_time_utc": "2020-01-20T06:00:00Z",
                "label_start_utc": "2020-01-20T06:00:00Z",
                "label_end_utc": "2020-01-21T06:00:00Z",
            },
        ]
    ).to_csv(decision_path, index=False)
    decision_manifest = source / "decision_manifest.json"
    _source_manifest(decision_path, "decision_calendar", decision_manifest)

    listing_path = gate / "listing_windows.parquet"
    pd.DataFrame(
        [
            {
                "stable_listing_id": "JPX:7203:2010-01:1",
                "refinitiv_ric": "7203.T",
                "listing_start": "2010-01-01",
                "listing_end": "2030-01-01",
                "known_at": "2010-01-01T00:00:00Z",
                "window_end_known_at": "2029-12-31T00:00:00Z",
                "right_open": True,
                "pit_usable": True,
            }
        ]
    ).to_parquet(listing_path, index=False)
    listing_manifest = gate / "manifest.json"
    _source_manifest(listing_path, "listing_windows", listing_manifest)

    feature_path = source / "feature_observations.csv"
    pd.DataFrame(
        [
            {
                "source_row_id": "f-1",
                "entity": "7203.T",
                "feature": FEATURE,
                "dataset_id": "fundamentals_pit",
                "value": 0.8,
                "source_asof_time_utc": "2019-12-20T00:00:00Z",
                "source_known_at_utc": "2019-12-21T00:00:00Z",
            }
        ]
    ).to_csv(feature_path, index=False)
    feature_manifest = source / "feature_manifest.json"
    _source_manifest(feature_path, "feature_observations", feature_manifest)

    commands_log = tmp_path / "commands.log"
    commands_log.write_text("run_japan_g5_decision_calendar_audit\n", encoding="utf-8")
    paths = JapanDataQualityPaths(
        security_master_dir=gate,
        historical_dir=gate,
        depth_dir=gate,
        compaction_dir=gate,
        universe_reconciliation_dir=gate,
        active_symbol_map_dir=gate,
        reference_symbol_map_dir=gate,
        g1_validation_dir=gate,
        pit_plan_dir=gate,
        corporate_actions_dir=gate,
        adjusted_returns_dir=gate,
        g2_admission_dir=gate,
        liquidity_dir=gate,
        fundamentals_ibes_events_dir=gate,
        g3_admission_dir=gate,
        fx_calendar_tca_dir=gate,
        leakage_qa_dir=output,
        immutable_hashes_path=output / "immutable_hashes.json",
        commands_log=commands_log,
    )
    config = JapanG5AuditConfig(
        required_features=(FEATURE,),
        purge_days=5,
        embargo_days=5,
        train_end="2020-01-05T00:00:00Z",
        test_start="2020-01-10T00:00:00Z",
        test_end="2020-01-15T00:00:00Z",
        holdout_start="2020-01-20T00:00:00Z",
        decision_schedule="actual_strategy_close_decisions",
    )
    return (
        paths,
        JapanG5Input(decision_path, decision_manifest, "decision_calendar"),
        JapanG5Input(listing_path, listing_manifest, "listing_windows"),
        JapanG5Input(feature_path, feature_manifest, "feature_observations"),
        config,
    )


def _run(tmp_path: Path):
    paths, decisions, listings, features, config = _fixture(tmp_path)
    result = write_japan_g5_decision_calendar_audit(
        paths=paths,
        run_id="JPN-G5-TRUE-CALENDAR-TEST",
        decision_calendar=decisions,
        listing_windows=listings,
        feature_observations=features,
        config=config,
        command="published-g5-command",
    )
    return result, paths, decisions, listings, features, config


def test_full_row_level_audit_satisfies_existing_g5_contract(tmp_path: Path) -> None:
    result, paths, *_ = _run(tmp_path)

    assert result.manifest["status"] == "pass"
    assert result.manifest["scorecard_eligible"] is True
    assert set(result.manifest["artifacts"]) == {
        "decision_panel",
        "leakage_violations",
        "survivorship_reconciliation",
        "split_ledger",
        "feature_spec",
        "audit_config",
    }
    panel = pd.read_parquet(result.decision_panel_path)
    assert {
        "decision_id",
        "entity",
        "decision_time_utc",
        "listing_start",
        "listing_end",
        "membership_valid_at_decision",
        "max_feature_available_time_utc",
        "label_start_utc",
        "label_end_utc",
    }.issubset(panel.columns)
    assert panel["membership_valid_at_decision"].all()
    assert panel["max_feature_available_time_utc"].le(panel["decision_time_utc"]).all()
    assert pd.read_csv(result.split_ledger_path)["split"].tolist() == ["train", "test", "holdout"]
    ready, reason = _g5_row_evidence_ready(result.manifest, paths.leakage_qa_dir)
    assert (ready, reason) == (True, "pass")
    inventory = json.loads(result.immutable_hashes_path.read_text(encoding="utf-8"))
    assert inventory["scorecard_eligible"] is True
    assert inventory["coverage_complete"] is True
    assert paths.commands_log.read_text(encoding="utf-8").endswith("published-g5-command\n")
    assert inventory["files"][str(paths.commands_log.resolve())] == _sha256(paths.commands_log)


def test_listing_and_label_failures_are_recorded_without_dropping_decisions(tmp_path: Path) -> None:
    paths, decisions, listings, features, config = _fixture(tmp_path)
    decision_frame = pd.read_csv(decisions.path)
    decision_frame.loc[0, "label_start_utc"] = "2019-12-31T06:00:00Z"
    decision_frame.to_csv(decisions.path, index=False)
    _source_manifest(decisions.path, decisions.artifact_key, decisions.manifest_path)
    listing_frame = pd.read_parquet(listings.path)
    listing_frame["listing_end"] = "2020-01-15"
    listing_frame.to_parquet(listings.path, index=False)
    _source_manifest(listings.path, listings.artifact_key, listings.manifest_path)

    result = write_japan_g5_decision_calendar_audit(
        paths=paths,
        run_id="JPN-G5-ROW-FAIL",
        decision_calendar=decisions,
        listing_windows=listings,
        feature_observations=features,
        config=config,
    )

    panel = pd.read_parquet(result.decision_panel_path)
    violations = pd.read_csv(result.leakage_violations_path)
    assert len(panel) == 3
    assert result.manifest["scorecard_eligible"] is False
    assert not bool(panel.loc[panel["decision_id"].eq("d-holdout"), "membership_valid_at_decision"].iloc[0])
    assert {"inconsistent_label_interval", "outside_listing_window"}.issubset(
        set(violations["violation_type"])
    )


def test_future_or_missing_feature_evidence_fails_closed(tmp_path: Path) -> None:
    paths, decisions, listings, features, config = _fixture(tmp_path)
    feature_frame = pd.read_csv(features.path)
    feature_frame["source_asof_time_utc"] = "2020-02-01T00:00:00Z"
    feature_frame.to_csv(features.path, index=False)
    _source_manifest(features.path, features.artifact_key, features.manifest_path)

    result = write_japan_g5_decision_calendar_audit(
        paths=paths,
        run_id="JPN-G5-FUTURE-SOURCE",
        decision_calendar=decisions,
        listing_windows=listings,
        feature_observations=features,
        config=config,
    )

    violations = pd.read_csv(result.leakage_violations_path)
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["future_data_violations"] == 1
    assert result.manifest["known_at_coverage"] == 0.0
    assert "source_asof_after_source_known_at" in set(violations["violation_type"])
    assert "missing_feature_at_decision" in set(violations["violation_type"])


def test_partial_g1_g4_admission_cannot_receive_scorecard_credit(tmp_path: Path) -> None:
    paths, decisions, listings, features, config = _fixture(tmp_path)
    _write_json(
        paths.g2_admission_dir / "manifest.json",
        _full_manifest(status="partial", partial=True),
    )

    result = write_japan_g5_decision_calendar_audit(
        paths=paths,
        run_id="JPN-G5-PARTIAL-UPSTREAM",
        decision_calendar=decisions,
        listing_windows=listings,
        feature_observations=features,
        config=config,
    )

    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["total_leakage_violations"] == 0
    assert any(
        blocker == "upstream_admission_not_full_or_hash_verified:g2_strict_admission"
        for blocker in result.manifest["critical_blockers"]
    )
    inventory = json.loads(result.immutable_hashes_path.read_text(encoding="utf-8"))
    assert inventory["scorecard_eligible"] is False


def test_existing_output_directory_is_rejected_without_mutation(tmp_path: Path) -> None:
    paths, decisions, listings, features, config = _fixture(tmp_path)
    assert paths.leakage_qa_dir is not None
    paths.leakage_qa_dir.mkdir()
    sentinel = paths.leakage_qa_dir / "existing-run.txt"
    sentinel.write_text("do not replace\n", encoding="utf-8")
    before_commands = paths.commands_log.read_bytes()

    with pytest.raises(FileExistsError, match="write-once"):
        write_japan_g5_decision_calendar_audit(
            paths=paths,
            run_id="JPN-G5-EXISTING-OUTPUT",
            decision_calendar=decisions,
            listing_windows=listings,
            feature_observations=features,
            config=config,
            command="must-not-be-logged",
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace\n"
    assert list(paths.leakage_qa_dir.iterdir()) == [sentinel]
    assert paths.commands_log.read_bytes() == before_commands


def test_injected_staging_failure_leaves_no_partial_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, decisions, listings, features, config = _fixture(tmp_path)
    assert paths.leakage_qa_dir is not None
    before_commands = paths.commands_log.read_bytes()
    real_atomic_json = audit_module._atomic_json

    def fail_manifest_write(path: Path, payload: dict[str, object]) -> None:
        if path.name == "manifest.json" and path.parent.name.startswith(".g5.tmp-"):
            raise OSError("injected manifest write failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(audit_module, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="injected manifest write failure"):
        write_japan_g5_decision_calendar_audit(
            paths=paths,
            run_id="JPN-G5-INJECTED-FAILURE",
            decision_calendar=decisions,
            listing_windows=listings,
            feature_observations=features,
            config=config,
            command="must-not-be-logged",
        )

    assert not paths.leakage_qa_dir.exists()
    assert not list(tmp_path.glob(".g5.tmp-*"))
    assert paths.commands_log.read_bytes() == before_commands
