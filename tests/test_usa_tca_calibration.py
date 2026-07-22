from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_lab.data.usa_exchange_calendar_tca import TCAAssumptions
from quant_lab.data.usa_tca_calibration import (
    CALIBRATION_TYPE,
    FeeInput,
    calibrate_usa_tca,
)


@pytest.fixture
def fee_inputs() -> dict[str, FeeInput]:
    return {
        "commission": FeeInput(0.15, "broker_schedule_test", "explicit_test_schedule"),
        "exchange_fee": FeeInput(0.25, "venue_schedule_test", "explicit_test_schedule"),
        "regulatory_sell_fee": FeeInput(
            0.05, "regulator_schedule_test", "explicit_test_schedule"
        ),
    }


@pytest.fixture
def calibration_inputs(tmp_path: Path) -> tuple[Path, Path]:
    daily_path = tmp_path / "daily_features.parquet"
    tiers_path = tmp_path / "eligible_universe_tiers.csv"
    rows = []
    for ticker, high_scale, eligible in (("AAA", 1.0, True), ("BBB", 10.0, False)):
        for day, high in enumerate((101.0, 102.0, 104.0, 108.0), start=1):
            rows.append(
                {
                    "date": pd.Timestamp(f"2024-01-{day:02d}"),
                    "ticker": ticker,
                    "high": high * high_scale,
                    "low": 100.0 * high_scale,
                    "adv_60_local_lag1": 1_000_000.0,
                    "amihud_60_lag1": day * 1e-8,
                    "bad_ohlc_flag": 0.0,
                }
            )
        if eligible:
            rows.append(
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "ticker": ticker,
                    "high": 200.0,
                    "low": 100.0,
                    "adv_60_local_lag1": 1_000_000.0,
                    "amihud_60_lag1": 1e-4,
                    "bad_ohlc_flag": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(daily_path, index=False, row_group_size=2)
    pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "u2_backtest_eligible": [True, False],
        }
    ).to_csv(tiers_path, index=False)
    return daily_path, tiers_path


def _calibrate(
    tmp_path: Path,
    calibration_inputs: tuple[Path, Path],
    fee_inputs: dict[str, FeeInput],
    **kwargs,
):
    daily_path, tiers_path = calibration_inputs
    return calibrate_usa_tca(
        daily_features_path=daily_path,
        eligible_universe_tiers_path=tiers_path,
        start_date="2024-01-01",
        end_date="2024-01-04",
        commission=fee_inputs["commission"],
        exchange_fee=fee_inputs["exchange_fee"],
        regulatory_sell_fee=fee_inputs["regulatory_sell_fee"],
        artifact_root=tmp_path / "artifacts",
        run_id="USA-TCA-TEST",
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        **kwargs,
    )


def test_calibration_filters_u2_and_dates_and_emits_compatible_scalars(
    tmp_path: Path,
    calibration_inputs: tuple[Path, Path],
    fee_inputs: dict[str, FeeInput],
):
    artifacts = _calibrate(tmp_path, calibration_inputs, fee_inputs, batch_size=2)
    payload = json.loads(artifacts.assumptions_json_path.read_text(encoding="utf-8"))
    assumptions = TCAAssumptions(**payload)
    assumptions.validate()

    expected_spreads = [0.5 * (high - 100.0) / ((high + 100.0) / 2.0) * 10_000 for high in (101, 102, 104, 108)]
    assert assumptions.half_spread_bps == pytest.approx(pd.Series(expected_spreads).median())
    assert assumptions.impact_bps_at_reference_participation == pytest.approx(25.0)
    assert assumptions.source_quality == CALIBRATION_TYPE
    assert artifacts.manifest["sample_count"]["filtered_rows"] == 4
    assert artifacts.manifest["entities"] == {"count": 1, "values": ["AAA"]}
    assert artifacts.manifest["dates"]["observed_start"] == "2024-01-01"
    assert artifacts.manifest["dates"]["observed_end"] == "2024-01-04"
    assert artifacts.manifest["live_trading_ready"] is False
    assert artifacts.manifest["ibkr_ready"] is False
    assert artifacts.evidence["fees"]["commission"]["source_name"] == "broker_schedule_test"
    assert len(pd.read_csv(artifacts.assumptions_csv_path)) == 1


def test_scanner_and_distribution_storage_are_bounded(
    tmp_path: Path,
    fee_inputs: dict[str, FeeInput],
):
    daily_path = tmp_path / "many_rows.parquet"
    tiers_path = tmp_path / "tiers.csv"
    count = 101
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=count, freq="h"),
            "ticker": ["AAA"] * count,
            "high": 101.0 + pd.Series(range(count), dtype=float) / 100.0,
            "low": [100.0] * count,
            "adv_60_local_lag1": [1_000_000.0] * count,
            "amihud_60_lag1": [1e-8] * count,
            "bad_ohlc_flag": [0.0] * count,
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), daily_path, row_group_size=19)
    pd.DataFrame({"ticker": ["AAA"], "u2_backtest_eligible": [True]}).to_csv(
        tiers_path, index=False
    )

    artifacts = calibrate_usa_tca(
        daily_features_path=daily_path,
        eligible_universe_tiers_path=tiers_path,
        start_date="2024-01-01",
        end_date="2024-01-05",
        commission=fee_inputs["commission"],
        exchange_fee=fee_inputs["exchange_fee"],
        regulatory_sell_fee=fee_inputs["regulatory_sell_fee"],
        artifact_root=tmp_path / "bounded",
        run_id="BOUNDED",
        batch_size=7,
        max_calibration_samples=11,
    )

    scanner = artifacts.manifest["scanner"]
    robust = artifacts.evidence["robust_estimation"]
    assert scanner["memory_bounded"] is True
    assert scanner["max_batch_rows_observed"] <= 7
    assert scanner["batches_processed"] > 1
    assert artifacts.manifest["sample_count"]["spread_valid"] == count
    assert robust["spread"]["sample_count_retained"] == 11
    assert robust["impact"]["sample_count_retained"] == 11
    assert robust["spread"]["bounded_sample_approximation"] is True


def test_manifest_checksums_outputs_and_refuses_unintentional_overwrite(
    tmp_path: Path,
    calibration_inputs: tuple[Path, Path],
    fee_inputs: dict[str, FeeInput],
):
    artifacts = _calibrate(tmp_path, calibration_inputs, fee_inputs)

    for name, path_text in artifacts.manifest["artifacts"].items():
        digest = hashlib.sha256(Path(path_text).read_bytes()).hexdigest()
        assert artifacts.manifest["artifact_sha256"][name] == digest
    with pytest.raises(FileExistsError, match="not empty"):
        _calibrate(tmp_path, calibration_inputs, fee_inputs)


def test_fee_inputs_require_value_and_provenance(
    tmp_path: Path,
    calibration_inputs: tuple[Path, Path],
    fee_inputs: dict[str, FeeInput],
):
    daily_path, tiers_path = calibration_inputs
    with pytest.raises(ValueError, match="commission.source_name"):
        calibrate_usa_tca(
            daily_features_path=daily_path,
            eligible_universe_tiers_path=tiers_path,
            start_date="2024-01-01",
            end_date="2024-01-04",
            commission=FeeInput(0.1, "", "schedule"),
            exchange_fee=fee_inputs["exchange_fee"],
            regulatory_sell_fee=fee_inputs["regulatory_sell_fee"],
            artifact_root=tmp_path / "invalid",
        )


def test_numeric_bad_ohlc_flag_is_excluded_from_spread_calibration(
    tmp_path: Path,
    fee_inputs: dict[str, FeeInput],
):
    daily_path = tmp_path / "flagged.parquet"
    tiers_path = tmp_path / "tiers.csv"
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "high": 101.0,
                "low": 100.0,
                "adv_60_local_lag1": 1_000_000.0,
                "amihud_60_lag1": 1e-8,
                "bad_ohlc_flag": 0.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "ticker": "AAA",
                "high": 200.0,
                "low": 100.0,
                "adv_60_local_lag1": 1_000_000.0,
                "amihud_60_lag1": 1e-8,
                "bad_ohlc_flag": 1.0,
            },
        ]
    ).to_parquet(daily_path, index=False)
    pd.DataFrame({"ticker": ["AAA"], "u2_backtest_eligible": [True]}).to_csv(
        tiers_path, index=False
    )

    artifacts = calibrate_usa_tca(
        daily_features_path=daily_path,
        eligible_universe_tiers_path=tiers_path,
        start_date="2024-01-01",
        end_date="2024-01-04",
        commission=fee_inputs["commission"],
        exchange_fee=fee_inputs["exchange_fee"],
        regulatory_sell_fee=fee_inputs["regulatory_sell_fee"],
        artifact_root=tmp_path / "flagged-artifacts",
        run_id="FLAGGED",
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
    )

    assert artifacts.manifest["sample_count"]["spread_valid"] == 1
