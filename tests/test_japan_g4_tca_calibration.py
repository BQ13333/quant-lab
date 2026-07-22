from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_evidence_contracts import full_component_blockers
from quant_lab.data.japan_g4_tca_calibration import (
    CALIBRATION_TYPE,
    calibrate_japan_g4_tca,
    validate_japan_g4_tca_calibration,
)
from quant_lab.data.usa_tca_calibration import FeeInput


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    daily = root / "daily_features.parquet"
    tiers = root / "tiers.csv"
    rows = []
    for ticker in ("7203.T", "6758.T"):
        for day, high in ((2, 101.0), (3, 103.0)):
            rows.append(
                {
                    "date": pd.Timestamp(f"2024-01-{day:02d}"),
                    "ticker": ticker,
                    "high": high,
                    "low": 100.0,
                    "adv_60_local_lag1": 1_000_000.0,
                    "amihud_60_lag1": day * 1e-8,
                    "bad_ohlc_flag": False,
                }
            )
    pd.DataFrame(rows).to_parquet(daily, index=False, row_group_size=1)
    pd.DataFrame(
        {
            "ticker": ["7203.T", "6758.T"],
            "u2_backtest_eligible": [True, True],
        }
    ).to_csv(tiers, index=False)
    admitted = root / "admitted_market_data_manifest.json"
    admitted.write_text(
        json.dumps(
            {
                "status": "complete",
                "currency": "JPY",
                "admitted_for_research": True,
                "liquidity_complete": True,
                "critical_blockers": [],
                "coverage": {
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-03",
                    "complete": True,
                },
                "artifacts": {
                    "daily_features": str(daily.resolve()),
                    "eligible_universe_tiers": str(tiers.resolve()),
                },
                "sha256": {daily.name: _sha(daily), tiers.name: _sha(tiers)},
            }
        ),
        encoding="utf-8",
    )
    return daily, tiers, admitted


def _calibrate(root: Path):
    daily, tiers, admitted = _inputs(root)
    return calibrate_japan_g4_tca(
        daily,
        tiers,
        admitted,
        start_date="2024-01-02",
        end_date="2024-01-03",
        commission=FeeInput(0.1, "broker", "official_schedule"),
        exchange_fee=FeeInput(0.2, "exchange", "official_schedule"),
        regulatory_sell_fee=FeeInput(0.0, "regulator", "official_schedule"),
        artifact_root=root / "calibration",
        run_id="JPN-TCA-CANDIDATE",
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        minimum_calibration_rows=4,
        minimum_entities=2,
        batch_size=1,
    )


def test_tca_producer_and_independent_validator_recompute_liquidity_costs(
    tmp_path: Path,
) -> None:
    candidate = _calibrate(tmp_path)
    assert candidate.manifest["status"] == "complete_candidate"
    assert candidate.manifest["liquidity_aware"] is True
    assumptions = json.loads(
        (candidate.artifact_dir / "tca_assumptions.json").read_text(encoding="utf-8")
    )
    assert assumptions["source_quality"] == CALIBRATION_TYPE
    assert assumptions["impact_bps_at_reference_participation"] == pytest.approx(25.0)

    validation = validate_japan_g4_tca_calibration(
        run_id="JPN-TCA-VALIDATION",
        producer_dir=candidate.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "pass_component"
    assert validation.manifest["component_scorecard_eligible"] is True
    for filename, digest in validation.manifest["output_sha256"].items():
        assert _sha(validation.artifact_dir / filename) == digest


def test_tca_validator_rejects_tampered_calibration_input(tmp_path: Path) -> None:
    candidate = _calibrate(tmp_path)
    daily = tmp_path / "daily_features.parquet"
    daily.write_bytes(b"tampered")

    validation = validate_japan_g4_tca_calibration(
        run_id="JPN-TCA-TAMPER-VALIDATION",
        producer_dir=candidate.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "failed_validation"
    assert "calibration_input_hash_mismatch:daily_features" in validation.manifest[
        "critical_blockers"
    ]


def test_tca_validator_rejects_falsified_producer_counts(tmp_path: Path) -> None:
    candidate = _calibrate(tmp_path)
    manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["calibration_rows"] = 1_000_000
    candidate.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = validate_japan_g4_tca_calibration(
        run_id="JPN-TCA-COUNT-VALIDATION",
        producer_dir=candidate.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "failed_validation"
    assert "calibration_count_reconciliation_failed" in validation.manifest[
        "critical_blockers"
    ]


def test_tca_producer_refuses_unadmitted_market_data(tmp_path: Path) -> None:
    daily, tiers, admitted = _inputs(tmp_path)
    manifest = json.loads(admitted.read_text(encoding="utf-8"))
    manifest["currency"] = "USD"
    admitted.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="admitted_market_data_contract_invalid"):
        calibrate_japan_g4_tca(
            daily,
            tiers,
            admitted,
            start_date="2024-01-02",
            end_date="2024-01-03",
            commission=FeeInput(0.1, "broker", "schedule"),
            exchange_fee=FeeInput(0.2, "exchange", "schedule"),
            regulatory_sell_fee=FeeInput(0.0, "regulator", "schedule"),
            artifact_root=tmp_path / "calibration",
        )


def test_short_tca_validation_cannot_satisfy_full_g4_date_denominator(
    tmp_path: Path,
) -> None:
    candidate = _calibrate(tmp_path)
    validation = validate_japan_g4_tca_calibration(
        run_id="JPN-TCA-SHORT-VALIDATION",
        producer_dir=candidate.artifact_dir,
        artifact_root=tmp_path / "validation",
    )

    blockers = full_component_blockers(
        validation.artifact_dir,
        validation.manifest,
        expected_component="tca_assumptions",
    )
    assert "tca_assumptions_coverage_starts_after_2006_07_01" in blockers
    assert "tca_assumptions_coverage_ends_before_2026_06_30" in blockers
