from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_pit_data_plan import G3_DATASETS, build_japan_g3_pit_data_plan


def _universe(path: Path) -> None:
    pd.DataFrame(
        {
            "refinitiv_ric": ["7203.T", "6758.T"],
            "ticker": ["7203", "6758"],
            "sid": ["JPN:7203.T", "JPN:6758.T"],
            "name": ["Toyota", "Sony"],
            "exchange": ["TYO", "TYO"],
            "currency": ["JPY", "JPY"],
            "stable_listing_id": ["listing-7203", "listing-6758"],
            "stable_instrument_id": ["instrument-7203", "instrument-6758"],
            "listing_start": ["2000-01-01", "2000-01-01"],
            "listing_end": ["2026-12-31", "2026-12-31"],
            "known_at": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "pit_usable": [True, True],
        }
    ).to_csv(path, index=False)


def test_plan_fails_closed_when_g1_is_not_admitted(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe)
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text(
        json.dumps({"status": "failed_validation", "scorecard_eligible": False}),
        encoding="utf-8",
    )

    result = build_japan_g3_pit_data_plan(
        universe_path=universe,
        g1_validation_dir=validation,
        artifact_root=tmp_path / "artifacts",
        run_id="JPN-G3-TEST",
        start_date="2025-01-01",
        end_date="2026-01-01",
        batch_size=1,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    batches = pd.read_csv(result.artifact_dir / "extraction_batches.csv")
    selected = pd.read_csv(result.selected_universe_path)
    assert manifest["status"] == "blocked_upstream"
    assert manifest["scorecard_eligible"] is False
    assert manifest["datasets"] == list(G3_DATASETS)
    assert set(batches["status"]) == {"blocked_upstream_g1"}
    assert not selected["point_in_time_membership_claimed"].any()
    assert manifest["artifact_sha256"]["selected_universe"]


def test_plan_is_admitted_only_by_full_independent_g1(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe)
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "independent_validation": True,
                "source_checksums_verified": True,
                "critical_blocker_count": 0,
            }
        ),
        encoding="utf-8",
    )

    result = build_japan_g3_pit_data_plan(
        universe_path=universe,
        g1_validation_dir=validation,
        artifact_root=tmp_path / "artifacts",
        run_id="JPN-G3-TEST",
        start_date="2025-01-01",
        end_date="2025-12-31",
        batch_size=2,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    batches = pd.read_csv(result.artifact_dir / "extraction_batches.csv")
    assert manifest["status"] == "complete"
    assert manifest["scorecard_eligible"] is True
    assert manifest["window_denominator_preserved"] is True
    assert manifest["selected_listing_windows"] == 2
    assert set(batches["status"]) == {"ready_after_field_validation"}


def test_admitted_g1_cannot_promote_a_current_survivor_plan_without_windows(tmp_path: Path) -> None:
    universe = tmp_path / "current.csv"
    pd.DataFrame({"refinitiv_ric": ["7203.T"], "ticker": ["7203"]}).to_csv(
        universe, index=False
    )
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "independent_audit": True,
                "validator_type": "independent_process",
                "source_checksums_verified": True,
                "critical_blocker_count": 0,
            }
        ),
        encoding="utf-8",
    )
    result = build_japan_g3_pit_data_plan(
        universe_path=universe,
        g1_validation_dir=validation,
        artifact_root=tmp_path / "artifacts",
        run_id="CURRENT-ONLY",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_upstream"
    assert manifest["scorecard_eligible"] is False
    assert manifest["current_survivor"] is True
    assert manifest["window_denominator_preserved"] is False
