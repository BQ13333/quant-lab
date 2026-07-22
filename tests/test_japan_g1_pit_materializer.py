from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g1_pit_materializer import (
    conservative_snapshot_known_at,
    write_japan_g1_pit_materializer,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _seed_complete_inputs(root: Path) -> tuple[Path, Path, Path]:
    jq = root / "jq"
    enrich = root / "enrich"
    valid = root / "valid"
    listed_rows = [
        {"effective_date": "2024-01-04", "code": "72030", "market_code": "0111"},
        {"effective_date": "2024-01-04", "code": "67580", "market_code": "0111"},
        {"effective_date": "2024-01-05", "code": "72030", "market_code": "0111"},
    ]
    _csv(jq / "listed_info.csv", listed_rows)
    _csv(jq / "request_plan.csv", [{"request_date": "2024-01-04"}, {"request_date": "2024-01-05"}])
    _csv(jq / "request_manifest.csv", [{"request_date": "2024-01-04", "status": "completed"}, {"request_date": "2024-01-05", "status": "completed"}])
    _json(jq / "quality_metrics.json", {"planned_dates": 2, "completed_dates": 2})
    sha = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (jq / "listed_info.csv", jq / "request_plan.csv", jq / "request_manifest.csv", jq / "quality_metrics.json")
    }
    (jq / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sha.items())), encoding="utf-8"
    )
    _json(
        jq / "manifest.json",
        {
            "artifact_dir": str(jq.resolve()),
            "dataset_id": "jquants_pro_v2_listed_info",
            "status": "complete",
            "quality_metrics": {
                "planned_dates": 2,
                "completed_dates": 2,
                "pending_dates": 0,
                "failed_dates": 0,
                "date_coverage_ratio": 1.0,
                "normalized_rows": 3,
            },
            "artifacts": {"listed_info": str((jq / "listed_info.csv").resolve())},
            "sha256": sha,
            "checksums_file_sha256": hashlib.sha256((jq / "checksums.sha256").read_bytes()).hexdigest(),
        },
    )
    _csv(
        enrich / "normalized_responses.csv",
        [
            {"ric": "7203.T", "price_currency": "JPY", "isin": "JP0000007203", "sedol": "7203JP"},
            {"ric": "6758.T", "price_currency": "JPY", "isin": "JP0000006758", "sedol": "6758JP"},
        ],
    )
    _json(
        enrich / "manifest.json",
        {
            "status": "complete_current_observation",
            "current_observation_only": True,
            "counts": {"success": 2, "source_unique_rics": 2, "error": 0},
            "artifacts": {"normalized_responses": str((enrich / "normalized_responses.csv").resolve())},
        },
    )
    reference = root / "reference.csv"
    _csv(
        reference,
        [
            {"local_code": "7203", "refinitiv_ric": "7203.T", "stable_instrument_id": "isin:JP0000007203"},
            {"local_code": "6758", "refinitiv_ric": "6758.T", "stable_instrument_id": "isin:JP0000006758"},
        ],
    )
    _json(
        valid / "manifest.json",
        {
            "status": "complete_valid_time_baseline_current_observation_only",
            "valid_time_baseline_ready": True,
            "historical_pit_ready": False,
            "inputs": {"reference_map": {"path": str(reference.resolve())}},
        },
    )
    return jq, enrich, valid


def test_conservative_known_at_uses_0800_jst_not_effective_midnight():
    assert conservative_snapshot_known_at("2024-01-05") == "2024-01-04T23:00:00+00:00"


def test_blocked_jquants_input_publishes_no_candidate_outputs(tmp_path: Path):
    jq, enrich, valid = _seed_complete_inputs(tmp_path)
    manifest = json.loads((jq / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "blocked_credentials"
    _json(jq / "manifest.json", manifest)

    result = write_japan_g1_pit_materializer(
        run_id="BLOCKED", jquants_dir=jq, enrichment_dir=enrich, valid_time_dir=valid, artifact_root=tmp_path / "out"
    )

    assert result.manifest["status"] == "blocked_upstream"
    assert not result.entities_path.exists()
    assert result.manifest["scorecard_eligible"] is False


def test_complete_sources_publish_candidate_parquets_without_self_validation(tmp_path: Path):
    jq, enrich, valid = _seed_complete_inputs(tmp_path)

    result = write_japan_g1_pit_materializer(
        run_id="COMPLETE", jquants_dir=jq, enrichment_dir=enrich, valid_time_dir=valid, artifact_root=tmp_path / "out"
    )

    assert result.manifest["status"] == "complete"
    assert result.manifest["historical_pit_ready"] is True
    assert result.manifest["independent_audit"] is False
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["counts"]["listing_windows"] == 2
    assert result.manifest["counts"]["active_candidate_rows"] == 1
    assert result.manifest["counts"]["pit_unusable_windows"] == 0
    assert pd.read_parquet(result.listing_windows_path)["pit_usable"].all()
    assert set(pd.read_csv(result.reference_candidate_path)["refinitiv_ric"]) == {"7203.T", "6758.T"}


def test_ambiguous_local_code_remains_a_blocker(tmp_path: Path):
    jq, enrich, valid = _seed_complete_inputs(tmp_path)
    manifest = json.loads((valid / "manifest.json").read_text(encoding="utf-8"))
    reference = Path(manifest["inputs"]["reference_map"]["path"])
    frame = pd.read_csv(reference)
    frame.loc[len(frame)] = {"local_code": "7203", "refinitiv_ric": "7203a.T", "stable_instrument_id": "other"}
    frame.to_csv(reference, index=False)

    result = write_japan_g1_pit_materializer(
        run_id="AMBIGUOUS", jquants_dir=jq, enrichment_dir=enrich, valid_time_dir=valid, artifact_root=tmp_path / "out"
    )

    assert result.manifest["status"] == "complete_with_blockers"
    assert result.manifest["counts"]["pit_unusable_windows"] == 1
    assert result.manifest["historical_pit_ready"] is False
