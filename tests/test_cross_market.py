from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.models.baselines import LowVolatilityModel, MomentumModel
from quant_lab.research.cross_market import (
    build_cross_market_allocator_configs,
    build_cross_market_model_suite,
    load_long_ohlcv_dataset,
    run_cross_market_sweep_from_long_ohlcv,
)


def _long_ohlcv_fixture(tickers: list[str], periods: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(42)
    rows = []
    for idx, ticker in enumerate(tickers):
        country = "Malaysia" if idx % 2 == 0 else "Singapore"
        currency = "MYR" if country == "Malaysia" else "SGD"
        returns = rng.normal(0.0003 + idx * 0.00001, 0.01, size=len(dates))
        close = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
        for date, value in close.items():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "sid": f"{country}:{ticker}",
                    "local_symbol": ticker,
                    "name": ticker,
                    "country": country,
                    "exchange": "TEST",
                    "currency": currency,
                    "open": value,
                    "high": value * 1.01,
                    "low": value * 0.99,
                    "close": value,
                    "adj_close": value,
                    "volume": 1000,
                    "data_tier": "bronze",
                    "provider": "test",
                    "interval": "1d",
                    "adjustment": "raw",
                    "run_id": "TEST",
                    "batch_id": "BATCH",
                    "fetched_at_utc": "2026-07-05T00:00:00+00:00",
                }
            )
    return pd.DataFrame(rows)


def test_load_long_ohlcv_dataset_filters_and_maps_groups(tmp_path):
    path = tmp_path / "ohlcv.parquet"
    _long_ohlcv_fixture(["A", "B"], periods=30).to_parquet(path, index=False)

    dataset = load_long_ohlcv_dataset(path, min_rows=25, max_missing_ratio=0.10)

    assert dataset.prices.shape == (30, 2)
    assert dataset.metadata.shape[0] == 2
    assert dataset.group_map.to_dict() == {"A": "Malaysia", "B": "Singapore"}


def test_cross_market_model_and_allocator_factories():
    models = build_cross_market_model_suite(prefix="test")
    configs = build_cross_market_allocator_configs()

    assert len(models) >= 8
    assert any("trend_lowvol" in model.name for model in models)
    assert any(config["weighting"] == "group_neutral_long_only" for config in configs)
    assert any(config["weighting"] == "group_neutral_dollar_neutral" for config in configs)


def test_run_cross_market_sweep_from_long_ohlcv_smoke(tmp_path):
    path = tmp_path / "ohlcv.parquet"
    _long_ohlcv_fixture([f"T{i}" for i in range(8)], periods=220).to_parquet(path, index=False)

    result = run_cross_market_sweep_from_long_ohlcv(
        long_ohlcv_root=path,
        artifact_root=tmp_path / "cross_market",
        run_id="CROSS-MARKET-TEST",
        convert_to_usd=False,
        min_rows=200,
        train_size=80,
        test_size=20,
        embargo=2,
        cost_bps_grid=(0.0, 15.0),
        models=[
            LowVolatilityModel(name="test_lowvol_21", lookback=21),
            MomentumModel(name="test_mom_21", lookback=21),
        ],
        allocator_configs=[
            {"weighting": "long_only", "top_fraction": 0.25, "rebalance_frequency": "W"},
            {"weighting": "group_neutral_long_only", "top_fraction": 0.50, "rebalance_frequency": "M"},
        ],
        max_asset_weight_limit=1.0,
        min_deflated_sharpe=0.0,
    )

    assert result.artifact_dir.exists()
    assert result.prices.shape[1] == 8
    assert len(result.all_candidates) == 8
    assert result.artifact_dir.joinpath("eligible_prices_local.parquet").exists()
    assert result.artifact_dir.joinpath("all_candidates.csv").exists()
    assert result.artifact_dir.joinpath("sweep_manifest.json").exists()
