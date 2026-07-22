from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.data.data_lake import wide_ohlcv_to_long, write_ohlcv_long_partitions
from quant_lab.data.l2_deep import build_l2_deep_extraction


def _ohlcv_frame(tickers: list[str], periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    data = {}
    for idx, ticker in enumerate(tickers):
        base = 10.0 + idx
        close = pd.Series([base + i * 0.1 for i in range(periods)], index=index)
        data[(ticker, "Open")] = close - 0.05
        data[(ticker, "High")] = close + 0.10
        data[(ticker, "Low")] = close - 0.10
        data[(ticker, "Close")] = close
        data[(ticker, "Volume")] = [1000.0 + idx * 100.0 + i for i in range(periods)]
    return pd.DataFrame(data, index=index)


def _listings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "yahoo_ticker": ["AAA.KL", "BBB.SI"],
            "country": ["Malaysia", "Singapore"],
            "exchange": ["Bursa Malaysia", "SGX"],
            "currency": ["MYR", "SGD"],
            "name": ["AAA Bhd", "BBB Ltd"],
        }
    )


def test_build_l2_deep_extraction_writes_core_artifacts(tmp_path):
    long = wide_ohlcv_to_long(
        _ohlcv_frame(["AAA.KL", "BBB.SI"], periods=80),
        listings=_listings(),
        provider="mock",
        run_id="RUN",
    )
    long.loc[(long["ticker"].eq("BBB.SI")) & (long["date"].eq(long["date"].min())), "volume"] = pd.NA
    root = write_ohlcv_long_partitions(long, root=tmp_path / "lake", run_id="RUN")

    result = build_l2_deep_extraction(
        root,
        artifact_root=tmp_path / "artifacts",
        run_id="L2",
        min_rows=60,
        max_missing_ratio=0.05,
        top_liquid_per_country=1,
    )

    assert result.manifest_path.exists()
    assert result.daily_features_path.exists()
    assert result.vendor_matrix_path.exists()
    assert pd.read_csv(result.corporate_action_proxy_path).empty
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["date_min"] == "2024-01-01"
    assert manifest["date_max"] == "2024-04-19"
    assert "eligible_universe_tiers" in manifest["artifacts"]
    bbb = result.universe_tiers[result.universe_tiers["ticker"].eq("BBB.SI")].iloc[0]
    assert bbb["volume_missing_pct"] > 0.0
    assert bbb["zero_volume_pct"] == pytest.approx(0.0)
    features = pd.read_parquet(result.daily_features_path)
    bbb_features = features[features["ticker"].eq("BBB.SI")]
    assert bbb_features["volume_missing_flag"].sum() == pytest.approx(1.0)
    assert bbb_features["zero_volume_flag"].sum() == pytest.approx(0.0)
    assert len(result.security_master) == 2
    assert int(result.universe_tiers["u2_backtest_eligible"].sum()) == 2
    assert int(result.universe_tiers["u3_liquid_proxy"].sum()) == 2


def test_l2_deep_extraction_lags_adv_features(tmp_path):
    long = wide_ohlcv_to_long(
        _ohlcv_frame(["AAA.KL"], periods=80),
        listings=_listings().head(1),
        provider="mock",
        run_id="RUN",
    )
    root = write_ohlcv_long_partitions(long, root=tmp_path / "lake", run_id="RUN")

    result = build_l2_deep_extraction(
        root,
        artifact_root=tmp_path / "artifacts",
        run_id="L2",
        min_rows=60,
        max_missing_ratio=0.05,
    )
    features = pd.read_parquet(result.daily_features_path)
    ticker_features = features[features["ticker"].eq("AAA.KL")].sort_values("date").reset_index(drop=True)
    row_25 = ticker_features.iloc[25]
    prior = ticker_features.iloc[5:25]
    expected = (prior["close"] * prior["volume"]).mean()

    assert row_25["adv_20_local_lag1"] == pytest.approx(expected)
