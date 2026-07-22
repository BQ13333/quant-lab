from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g1_independent_validator import (
    _membership_issues,
    write_japan_g1_independent_validation,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _producer(root: Path, overlap: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    maps = pd.DataFrame(
        [
            {
                "refinitiv_ric": "7203.T",
                "local_code": "7203",
                "listing_start": "2024-01-04",
                "listing_end": "2024-01-05",
                "currency": "JPY",
                "known_at": "2024-01-03T23:00:00+00:00",
                "stable_listing_id": "JPX:7203:2024-01-04:1",
                "pit_usable": True,
                "blocker_reason": "",
            }
        ]
    )
    if overlap:
        maps.loc[len(maps)] = {
            **maps.iloc[0].to_dict(),
            "listing_start": "2024-01-05",
            "listing_end": "2024-01-06",
            "stable_listing_id": "JPX:7203:2024-01-05:2",
        }
    entities = maps[["stable_listing_id", "local_code", "refinitiv_ric"]].copy()
    listing = maps[["stable_listing_id", "refinitiv_ric", "local_code", "listing_start", "listing_end", "known_at", "currency", "pit_usable"]].copy()
    identifiers = pd.DataFrame(
        [
            {
                "stable_listing_id": row.stable_listing_id,
                "identifier_type": "ric",
                "identifier_value": row.refinitiv_ric,
                "valid_from": row.listing_start,
                "valid_to": row.listing_end,
                "known_at": row.known_at,
            }
            for row in maps.itertuples(index=False)
        ]
    )
    membership = listing.rename(columns={"listing_start": "membership_start", "listing_end": "membership_end"})[
        ["stable_listing_id", "refinitiv_ric", "membership_start", "membership_end", "known_at", "pit_usable"]
    ]
    reconciliation = maps.assign(mapping_status="matched")
    files = {
        "entities.parquet": ("entities", entities, "parquet"),
        "listing_windows.parquet": ("listing_windows", listing, "parquet"),
        "identifier_history.parquet": ("identifier_history", identifiers, "parquet"),
        "universe_membership_pit.parquet": ("universe_membership", membership, "parquet"),
        "active_symbol_map_candidate.csv": ("active_candidate", maps.tail(1), "csv"),
        "reference_symbol_map_candidate.csv": ("reference_candidate", maps, "csv"),
        "reconciliation.csv": ("reconciliation", reconciliation, "csv"),
    }
    artifacts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for filename, (alias, frame, kind) in files.items():
        path = root / filename
        if kind == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        artifacts[alias] = str(path.resolve())
        digests[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_manifests: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for name in ("jquants", "enrichment", "valid_time"):
        path = root / f"{name}_manifest.json"
        _json(path, {"name": name, "status": "fixture"})
        input_manifests[name] = str(path.resolve())
        input_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _json(
        root / "manifest.json",
        {
            "artifact_dir": str(root.resolve()),
            "artifact_type": "japan_g1_pit_materializer_producer",
            "status": "complete",
            "producer_run_id": "PRODUCER",
            "independent_audit": False,
            "scorecard_eligible": False,
            "historical_pit_ready": True,
            "critical_blockers": [],
            "artifacts": artifacts,
            "output_sha256": digests,
            "input_manifests": input_manifests,
            "input_manifest_sha256": input_hashes,
        },
    )
    return root


def _run(tmp_path: Path, producer: Path, run_id: str = "VALIDATOR"):
    return write_japan_g1_independent_validation(
        run_id=run_id,
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        security_master_dir=tmp_path / "security",
        reconciliation_dir=tmp_path / "reconciliation",
        active_map_dir=tmp_path / "active",
        reference_map_dir=tmp_path / "reference",
    )


def test_clean_independent_validation_publishes_all_admitted_g1_artifacts(tmp_path: Path):
    result = _run(tmp_path, _producer(tmp_path / "producer"))

    assert result.manifest["status"] == "pass"
    assert result.manifest["scorecard_eligible"] is True
    assert result.manifest["critical_blocker_count"] == 0
    security = json.loads((tmp_path / "security/manifest.json").read_text())
    active = json.loads((tmp_path / "active/manifest.json").read_text())
    reconciliation = json.loads((tmp_path / "reconciliation/manifest.json").read_text())
    assert security["listing_level_master"] is True
    assert active["validator_run_id"] != active["producer_run_id"]
    assert active["symbol_map_sha256"] == hashlib.sha256((tmp_path / "active/symbol_map.csv").read_bytes()).hexdigest()
    active_map = pd.read_csv(tmp_path / "active/symbol_map.csv", dtype=str).fillna("")
    assert set(active_map["country"]) == {"Japan"}
    assert set(active_map["exchange"]) == {"JPX"}
    assert set(active_map["refinitiv_symbol_status"]) == {
        "independently_validated_security_master"
    }
    assert reconciliation["unmatched_jpx_rows"] == 0
    assert reconciliation["strict_quality_pass"] is True
    assert reconciliation["historical_pit_ready"] is True
    assert reconciliation["historical_backtest_eligible"] is True


def test_validator_cannot_be_same_run_as_producer(tmp_path: Path):
    result = _run(tmp_path, _producer(tmp_path / "producer"), run_id="PRODUCER")
    assert result.manifest["status"] == "failed_validation"
    assert any("independence_failure" in value for value in result.manifest["critical_blockers"])
    assert json.loads((tmp_path / "security/manifest.json").read_text())["scorecard_eligible"] is False


def test_hash_tampering_fails_closed_and_publishes_no_pass_manifest(tmp_path: Path):
    producer = _producer(tmp_path / "producer")
    (producer / "reference_symbol_map_candidate.csv").write_text("tampered\n", encoding="utf-8")
    result = _run(tmp_path, producer)
    assert result.manifest["status"] == "failed_validation"
    assert any("producer_hash_mismatch" in value for value in result.manifest["critical_blockers"])


def test_overlapping_repeated_ric_windows_fail_validation(tmp_path: Path):
    result = _run(tmp_path, _producer(tmp_path / "producer", overlap=True))
    assert result.manifest["status"] == "failed_validation"
    assert any("reference_map_overlap" in value for value in result.manifest["critical_blockers"])


def test_lifecycle_exclusion_is_independently_recomputed() -> None:
    frame = pd.DataFrame(
        [
            {
                "observation_month": "2021-03",
                "code": "7366",
                "refinitiv_ric": "7366.T",
                "identifier_mapped": True,
                "lifecycle_excluded": True,
                "lifecycle_exclusion_reason": "official_monthly_observation_outside_exact_listing_window",
                "listing_valid_from": "2021-04-01",
                "listing_valid_to": "2026-06-30",
                "known_at_utc": "2021-04-21T00:00:00+09:00",
                "pit_usable": False,
            }
        ]
    )
    assert _membership_issues(frame) == []
    frame.loc[0, "lifecycle_excluded"] = False
    assert any(check == "membership_lifecycle_recomputation" for check, _ in _membership_issues(frame))
