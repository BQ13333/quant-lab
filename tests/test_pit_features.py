from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.pit_features import (
    assert_lagged_feature_panel_no_leakage,
    build_lagged_pit_feature_panel,
    read_pit_history_partitions,
    summarize_lagged_feature_panel,
    write_lagged_pit_feature_artifacts,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asof_date": "2024-01-31",
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "MY:AAA",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "dataset_id": "liquidity_market_cap_free_float",
                "field_role": "market_cap",
                "value": 100.0,
                "inside_requested_window": True,
            },
            {
                "asof_date": "2024-02-29",
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "MY:AAA",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "dataset_id": "liquidity_market_cap_free_float",
                "field_role": "market_cap",
                "value": 120.0,
                "inside_requested_window": True,
            },
            {
                "asof_date": "2024-01-31",
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "MY:AAA",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "dataset_id": "fundamentals_pit",
                "field_role": "revenue",
                "value": 9.0,
                "inside_requested_window": True,
            },
        ]
    )


def test_build_lagged_pit_feature_panel_respects_dataset_lags():
    features = build_lagged_pit_feature_panel(
        _history(),
        decision_dates=["2024-02-29", "2024-04-30", "2024-05-31"],
    )

    liquidity_col = "liquidity_market_cap_free_float__market_cap"
    revenue_col = "fundamentals_pit__revenue"

    assert features.loc[features["decision_date"].eq(pd.Timestamp("2024-02-29")), liquidity_col].iloc[0] == 100.0
    assert pd.isna(features.loc[features["decision_date"].eq(pd.Timestamp("2024-02-29")), revenue_col].iloc[0])
    assert features.loc[features["decision_date"].eq(pd.Timestamp("2024-04-30")), revenue_col].iloc[0] == 9.0
    assert features.loc[features["decision_date"].eq(pd.Timestamp("2024-05-31")), liquidity_col].iloc[0] == 120.0

    assert_lagged_feature_panel_no_leakage(features)


def test_lagged_feature_panel_detects_manual_leakage():
    features = build_lagged_pit_feature_panel(
        _history(),
        decision_dates=["2024-02-29"],
    )
    features["bad__source_available_date"] = pd.Timestamp("2024-03-01")

    with pytest.raises(ValueError, match="point-in-time leakage"):
        assert_lagged_feature_panel_no_leakage(features)


def test_write_lagged_pit_feature_artifacts_and_read_partitions(tmp_path):
    source_dir = tmp_path / "history"
    source_dir.mkdir()
    _history().to_parquet(source_dir / "part.parquet", index=False)

    loaded = read_pit_history_partitions(tmp_path)
    artifacts = write_lagged_pit_feature_artifacts(
        loaded,
        artifact_root=tmp_path / "features",
        run_id="TEST-FEATURES",
        decision_dates=["2024-02-29", "2024-04-30"],
    )

    assert artifacts.features_parquet_path.exists()
    assert artifacts.features_csv_path.exists()
    assert artifacts.quality_path.exists()
    quality = summarize_lagged_feature_panel(artifacts.features)
    assert set(quality["feature"]) == {
        "fundamentals_pit__revenue",
        "liquidity_market_cap_free_float__market_cap",
    }
