from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.models.baselines import LowVolatilityModel, MomentumModel
from quant_lab.research.malaysia_sweep import (
    build_malaysia_allocator_configs,
    build_malaysia_full_model_suite,
    run_malaysia_full_sweep_from_asean_artifact,
)


def test_malaysia_full_model_suite_is_price_only_and_broad():
    models = build_malaysia_full_model_suite(prefix="test")
    names = [model.name for model in models]

    assert len(models) >= 30
    assert any("lowvol" in name for name in names)
    assert any("_mom_" in name for name in names)
    assert any("_rev_" in name for name in names)
    assert any("_vam_" in name for name in names)
    assert any("_tsmom_" in name for name in names)
    assert any("_breakout_" in name for name in names)
    assert any("_volcomp_" in name for name in names)
    assert any("_regime_" in name for name in names)
    assert not any("volume" in name for name in names)


def test_malaysia_allocator_configs_include_daily_weekly_monthly():
    configs = build_malaysia_allocator_configs(top_fractions=(0.2,), rebalance_frequencies=("D", "W", "M"))

    assert {config["rebalance_frequency"] for config in configs} == {"D", "W", "M"}
    assert {config["weighting"] for config in configs} == {"long_only", "dollar_neutral"}


def test_run_malaysia_full_sweep_from_asean_artifact_smoke(tmp_path):
    source = tmp_path / "asean"
    source.mkdir()
    tickers = [f"{1000 + i}.KL" for i in range(6)]
    dates = pd.bdate_range("2020-01-01", periods=220)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    prices = 100 * (1 + returns).cumprod()
    prices["EWM"] = 100 * (1 + returns.mean(axis=1)).cumprod()
    prices.to_parquet(source / "daily_usd_prices.parquet")

    metadata = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "name": ticker,
                "country": "Malaysia",
                "currency": "MYR",
                "exchange": "Bursa Malaysia",
                "tier": "stock",
            }
            for ticker in tickers
        ]
        + [
            {
                "ticker": "EWM",
                "name": "iShares MSCI Malaysia ETF",
                "country": "Malaysia",
                "currency": "USD",
                "exchange": "NYSE Arca",
                "tier": "regional_proxy",
            }
        ]
    )
    metadata.to_csv(source / "universe.csv", index=False)

    result = run_malaysia_full_sweep_from_asean_artifact(
        asean_artifact_dir=source,
        artifact_root=tmp_path / "malaysia",
        vault_root=tmp_path / "vault",
        run_id="MALAYSIA-SWEEP-TEST",
        cost_bps_grid=(0.0, 15.0),
        baseline_cost_bps=15.0,
        train_size=80,
        test_size=20,
        embargo=2,
        min_rows=200,
        models=[
            LowVolatilityModel(name="test_lowvol_21", lookback=21),
            MomentumModel(name="test_mom_21", lookback=21),
        ],
        allocator_configs=[
            {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
            {"weighting": "dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "M"},
        ],
        max_asset_weight_limit=1.0,
        min_deflated_sharpe=0.0,
    )

    assert result.artifact_dir.exists()
    assert len(result.all_candidates) == 8
    assert not result.champions.empty
    assert result.artifact_dir.joinpath("all_candidates.csv").exists()
    assert result.artifact_dir.joinpath("killed_strategies.csv").exists()
    assert result.artifact_dir.joinpath("sweep_manifest.json").exists()
    assert result.note_paths
