from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_monthly_microstructure_validator import (
    write_japan_monthly_microstructure_validation,
)


def _producer(root: Path) -> Path:
    root.mkdir()
    periods = pd.period_range("2006-07", "2026-06", freq="M")
    history = pd.DataFrame({
        "observation_month": periods.astype(str),
        "code": "1001",
        "known_at_utc": [(period + 1).start_time.tz_localize("UTC") + pd.Timedelta(days=20) for period in periods],
        "source_families": "TSE",
        "preferred_source_family": "TSE",
        "trading_unit": 100,
        "source_observations": 1,
        "unit_mapping_status": "direct_official_monthly_pdf",
        "pit_usable": True,
    })
    raw = history[["observation_month", "code", "trading_unit"]].assign(
        parser_route="fixture", source_family="TSE", source_artifact="fixture",
        request_id="fixture", raw_sha256="a" * 64, unit_observed=True,
    )
    files = {
        "monthly_lot_size_history.parquet": history,
        "raw_lot_size_observations.parquet": raw,
    }
    for filename, frame in files.items():
        frame.to_parquet(root / filename, index=False)
    pd.DataFrame([{
        "plan_id": "one", "status": "completed", "expected_rows": 240,
        "parsed_rows": 240, "conflict_codes": 0,
    }]).to_csv(root / "file_quality.csv", index=False)
    pd.DataFrame(columns=history.columns).to_csv(root / "unresolved_lot_sizes.csv", index=False)
    pd.DataFrame(columns=["observation_month", "code", "source_family", "unit_variants"]).to_csv(root / "lot_size_conflicts.csv", index=False)
    pd.DataFrame(columns=["observation_month", "code", "unit_variants"]).to_csv(root / "venue_specific_lot_size_variants.csv", index=False)
    (root / "file_plan.csv").write_text("plan_id\none\n", encoding="utf-8")
    outputs = [
        "monthly_lot_size_history.parquet", "raw_lot_size_observations.parquet",
        "file_quality.csv", "unresolved_lot_sizes.csv", "lot_size_conflicts.csv",
        "venue_specific_lot_size_variants.csv",
    ]
    manifest = {
        "run_id": "PRODUCER",
        "status": "complete_candidate",
        "historical_lot_size_candidate": True,
        "scorecard_eligible": False,
        "independent_validation_required": True,
        "input_plan_sha256": hashlib.sha256((root / "file_plan.csv").read_bytes()).hexdigest(),
        "output_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in outputs},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_independent_monthly_microstructure_validator_passes_clean_source(tmp_path: Path) -> None:
    result = write_japan_monthly_microstructure_validation(
        run_id="VALIDATOR", producer_dir=_producer(tmp_path / "producer"), artifact_root=tmp_path / "out"
    )
    assert result.manifest["status"] == "pass_component"
    assert result.manifest["component_scorecard_eligible"] is True
    assert result.manifest["critical_blocker_count"] == 0


def test_independent_monthly_microstructure_validator_fails_tampering(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    (producer / "unresolved_lot_sizes.csv").write_text("tampered\n", encoding="utf-8")
    result = write_japan_monthly_microstructure_validation(
        run_id="VALIDATOR", producer_dir=producer, artifact_root=tmp_path / "out"
    )
    assert result.manifest["status"] == "failed_validation"
    assert any("producer_output_hash_mismatch" in blocker for blocker in result.manifest["critical_blockers"])
