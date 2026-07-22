from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_evidence_assembler import assemble_japan_g4_evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path]:
    base = root / "base"
    lot = root / "lot"
    base.mkdir()
    lot.mkdir()
    quality = pd.DataFrame(
        [
            {"component": "jpy_fx_rates", "status": "pass_static_market_currency", "validated": True, "coverage": 1.0, "source_rows": 1, "notes": "JPY"},
            {"component": "lot_sizes", "status": "partial", "validated": False, "coverage": 0.0, "source_rows": 1, "notes": "missing"},
        ]
    )
    quality.to_csv(base / "evidence_quality.csv", index=False)
    (base / "trading_unit_policy.json").write_text("{}", encoding="utf-8")
    raw = base / "raw" / "trading_unit.html"
    raw.parent.mkdir()
    raw.write_text("official JPY trading unit rule", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "component": "trading_unit",
                "http_status": 200,
                "raw_path": str(raw.resolve()),
                "sha256": _sha(raw),
            }
        ]
    ).to_csv(base / "raw_manifest.csv", index=False)
    _write_json(
        base / "manifest.json",
        {
            "run_id": "BASE",
            "status": "incomplete",
            "scorecard_eligible": False,
            "official_jpx_source": True,
            "research_only": True,
            "production_ready": False,
            "live_execution_ready": False,
            "currency": "JPY",
            "official_jpx_calendar": False,
            "tca_calibrated": False,
            "complete_components": ["jpy_fx_rates"],
            "partial_components": ["exchange_calendars", "lot_sizes"],
            "counts": {},
            "critical_blockers": [
                "full 2006-present official JPX session calendar is missing",
                "issuer-level lot-size history before 2018-10-01 is missing",
            ],
            "artifacts": {
                "evidence_quality": str((base / "evidence_quality.csv").resolve()),
                "trading_unit_policy": str((base / "trading_unit_policy.json").resolve()),
                "raw_manifest": str((base / "raw_manifest.csv").resolve()),
            },
            "sha256": {
                "evidence_quality.csv": _sha(base / "evidence_quality.csv"),
                "trading_unit_policy.json": _sha(base / "trading_unit_policy.json"),
                "raw_manifest.csv": _sha(base / "raw_manifest.csv"),
            },
        },
    )
    producer = root / "producer.json"
    producer_output = root / "producer_lot_sizes.csv"
    producer_output.write_text("code,trading_unit\n7203,100\n", encoding="utf-8")
    _write_json(producer, {"output_sha256": {producer_output.name: _sha(producer_output)}})
    history = lot / "validated_monthly_lot_size_history.parquet"
    pd.DataFrame({"code": ["7203"], "trading_unit": [100]}).to_parquet(history, index=False)
    _write_json(
        lot / "manifest.json",
        {
            "run_id": "VALIDATOR",
            "status": "pass_component",
            "component": "lot_sizes",
            "component_scorecard_eligible": True,
            "scorecard_eligible": False,
            "independent_validation": True,
            "official_jpx_source": True,
            "critical_blocker_count": 0,
            "critical_blockers": [],
            "counts": {"history_rows": 1, "months": 1, "venue_specific_variant_rows": 0},
            "producer_manifest": str(producer.resolve()),
            "producer_manifest_sha256": _sha(producer),
            "artifacts": {"validated_monthly_lot_size_history": str(history.resolve())},
            "output_sha256": {history.name: _sha(history)},
            "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        },
    )
    return base, lot


def _calendar_fixture(root: Path) -> Path:
    validation = root / "calendar"
    validation.mkdir()
    output = validation / "validated_exchange_calendar.parquet"
    pd.DataFrame({"session_date": ["2006-07-03"]}).to_parquet(output, index=False)
    producer_dir = root / "calendar_producer"
    producer_dir.mkdir()
    producer_output = producer_dir / "calendar.csv"
    producer_output.write_text("session_date\n2006-07-03\n", encoding="utf-8")
    producer_manifest = producer_dir / "manifest.json"
    _write_json(
        producer_manifest,
        {"output_sha256": {producer_output.name: _sha(producer_output)}},
    )
    _write_json(
        validation / "manifest.json",
        {
            "status": "pass_component",
            "component": "exchange_calendars",
            "component_scorecard_eligible": True,
            "scorecard_eligible": False,
            "independent_validation": True,
            "official_jpx_calendar": True,
            "currency": "JPY",
            "critical_blocker_count": 0,
            "critical_blockers": [],
            "counts": {"sessions": 4891, "months": 240, "source_files": 1},
            "producer_manifest": str(producer_manifest.resolve()),
            "producer_manifest_sha256": _sha(producer_manifest),
            "artifacts": {"validated_exchange_calendar": str(output.resolve())},
            "output_sha256": {output.name: _sha(output)},
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
        },
    )
    return validation


def _full_component_fixture(root: Path, component: str) -> Path:
    validation = root / f"{component}_validation"
    producer = root / f"{component}_producer"
    validation.mkdir()
    producer.mkdir()
    producer_output = producer / "candidate.csv"
    producer_output.write_text("id\n1\n", encoding="utf-8")
    producer_manifest = producer / "manifest.json"
    _write_json(
        producer_manifest,
        {"output_sha256": {producer_output.name: _sha(producer_output)}},
    )
    output = validation / "validated.csv"
    output.write_text("id\n1\n", encoding="utf-8")
    semantic: dict[str, object]
    counts: dict[str, object]
    if component == "halt_suspension_events":
        semantic = {
            "official_jpx_source": True,
            "historical_archive_complete": True,
            "source_denominator_exact": True,
        }
        counts = {"source_rows": 10, "event_rows": 10}
    elif component == "tick_size_rules":
        semantic = {"official_jpx_source": True, "pit_valid_time_complete": True}
        counts = {"rule_rows": 10}
    elif component == "pit_narrow_tick_assignment":
        semantic = {"official_jpx_source": True, "pit_valid_time_complete": True}
        counts = {"assignment_rows": 10}
    elif component == "normal_price_limits":
        semantic = {"official_jpx_source": True, "normal_price_limits_complete": True}
        counts = {"rule_rows": 10}
    elif component == "exceptional_price_limits":
        semantic = {
            "official_jpx_source": True,
            "exceptional_price_limits_complete": True,
            "pre_2013_archive_complete": True,
            "post_2020_return_to_normal_reconciled": True,
        }
        counts = {"event_rows": 10}
    elif component == "tca_assumptions":
        semantic = {
            "currency": "JPY",
            "calibration_type": "japan_liquidity_aware_market_data_calibration",
            "liquidity_aware": True,
            "minimum_sample_requirements_met": True,
        }
        counts = {"calibration_rows": 1000, "entities": 10}
    else:
        raise AssertionError(component)
    _write_json(
        validation / "manifest.json",
        {
            "status": "pass_component",
            "component": component,
            "component_scorecard_eligible": True,
            "scorecard_eligible": False,
            "independent_validation": True,
            "critical_blocker_count": 0,
            "critical_blockers": [],
            "coverage": {
                "start_date": "2006-07-01",
                "end_date": "2026-06-30",
                "complete": True,
            },
            "counts": counts,
            "producer_manifest": str(producer_manifest.resolve()),
            "producer_manifest_sha256": _sha(producer_manifest),
            "artifacts": {"validated": str(output.resolve())},
            "output_sha256": {output.name: _sha(output)},
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            **semantic,
        },
    )
    return validation


def test_assembler_promotes_only_independently_validated_lot_component(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    result = assemble_japan_g4_evidence(
        run_id="ASSEMBLED", base_rules_dir=base, lot_size_validation_dir=lot, artifact_root=tmp_path
    )
    assert result.manifest["status"] == "incomplete"
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["complete_components"] == ["jpy_fx_rates", "lot_sizes"]
    assert result.manifest["critical_blockers"] == [
        "full 2006-present official JPX session calendar is missing",
        "full historical halt/suspension archive is missing",
        "full PIT tick-size rules and issue assignment history are missing",
        "full normal and exceptional price-limit history is missing; exceptional archive before 2013-09 remains unproven",
        "TCA is not calibrated against admitted Japan market data",
    ]
    quality = pd.read_csv(result.artifact_dir / "evidence_quality.csv")
    lot_quality = quality.loc[quality["component"].eq("lot_sizes")].iloc[0]
    assert bool(lot_quality["validated"]) is True
    assert lot_quality["status"] == "pass"
    assert int(lot_quality["source_rows"]) == 1
    assert quality.loc[quality["component"].eq("jpy_fx_rates"), "status"].iloc[0] == "pass"


def test_assembler_rejects_tampered_lot_component(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    (lot / "validated_monthly_lot_size_history.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="lot_size_validation_hash_mismatch"):
        assemble_japan_g4_evidence(
            run_id="ASSEMBLED", base_rules_dir=base, lot_size_validation_dir=lot, artifact_root=tmp_path
        )


def test_assembler_rejects_halt_source_output_parity_failure(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"] = {"halt_rows": 3, "halt_source_rows": 4}
    _write_json(base / "manifest.json", manifest)
    with pytest.raises(ValueError, match="base_rules_halt_source_output_parity_failed:3/4"):
        assemble_japan_g4_evidence(
            run_id="ASSEMBLED", base_rules_dir=base, lot_size_validation_dir=lot, artifact_root=tmp_path
        )


def test_assembler_admits_exactly_seven_full_aliases(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    calendar = _calendar_fixture(tmp_path)
    components = {
        component: _full_component_fixture(tmp_path, component)
        for component in (
            "halt_suspension_events",
            "tick_size_rules",
            "pit_narrow_tick_assignment",
            "normal_price_limits",
            "exceptional_price_limits",
            "tca_assumptions",
        )
    }

    result = assemble_japan_g4_evidence(
        run_id="FULL",
        base_rules_dir=base,
        lot_size_validation_dir=lot,
        calendar_validation_dir=calendar,
        halt_validation_dir=components["halt_suspension_events"],
        tick_rule_validation_dir=components["tick_size_rules"],
        pit_tick_assignment_validation_dir=components["pit_narrow_tick_assignment"],
        normal_price_limit_validation_dir=components["normal_price_limits"],
        exceptional_price_limit_validation_dir=components["exceptional_price_limits"],
        tca_validation_dir=components["tca_assumptions"],
        artifact_root=tmp_path,
    )

    expected = [
        "jpy_fx_rates",
        "exchange_calendars",
        "halt_suspension_events",
        "tick_sizes",
        "price_limits",
        "lot_sizes",
        "tca_assumptions",
    ]
    assert result.manifest["status"] == "complete"
    assert result.manifest["scorecard_eligible"] is True
    assert result.manifest["complete_components"] == expected
    assert result.manifest["partial_components"] == []
    assert result.manifest["critical_blockers"] == []
    quality = pd.read_csv(result.artifact_dir / "evidence_quality.csv")
    admitted = quality.set_index("component").loc[expected]
    assert admitted["status"].eq("pass").all()
    assert admitted["validated"].astype(bool).all()
    assert admitted["coverage"].eq(1.0).all()
    assert admitted["source_rows"].gt(0).all()


def test_assembler_preserves_pre_2013_exceptional_limit_gap(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    exceptional = _full_component_fixture(tmp_path, "exceptional_price_limits")
    manifest_path = exceptional / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pre_2013_archive_complete"] = False
    manifest["coverage"]["start_date"] = "2013-09-01"
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="exceptional_price_limits_coverage_starts_after_2006_07_01",
    ):
        assemble_japan_g4_evidence(
            run_id="GAP",
            base_rules_dir=base,
            lot_size_validation_dir=lot,
            exceptional_price_limit_validation_dir=exceptional,
            artifact_root=tmp_path,
        )


def test_assembler_rejects_empty_validator_hash_inventory(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    manifest_path = lot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"] = {}
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="lot_size_validation_hash_inventory_empty"):
        assemble_japan_g4_evidence(
            run_id="EMPTY-HASHES",
            base_rules_dir=base,
            lot_size_validation_dir=lot,
            artifact_root=tmp_path,
        )


def test_assembler_rejects_artifact_path_substitution(tmp_path: Path) -> None:
    base, lot = _fixture(tmp_path)
    halts = _full_component_fixture(tmp_path, "halt_suspension_events")
    outside = tmp_path / "outside.csv"
    outside.write_text("id\n1\n", encoding="utf-8")
    manifest_path = halts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["validated"] = str(outside.resolve())
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="halt_suspension_events_artifact_path_invalid"):
        assemble_japan_g4_evidence(
            run_id="PATH-SUBSTITUTION",
            base_rules_dir=base,
            lot_size_validation_dir=lot,
            halt_validation_dir=halts,
            artifact_root=tmp_path,
        )
