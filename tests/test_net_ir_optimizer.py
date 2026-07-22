from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.models.baselines import LowVolatilityModel, MomentumModel
from quant_lab.models.source_strategies import MovingAverageCrossoverModel, build_source_strategy_specs
from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer


def _synthetic_prices(n_days: int = 500, n_assets: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    returns = np.random.default_rng(42).normal(0.0005, 0.015, size=(n_days, n_assets))
    prices = pd.DataFrame(
        np.cumprod(1 + returns, axis=0),
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return prices


def _group_map(n_assets: int = 5) -> pd.Series:
    return pd.Series(
        ["SG", "ID", "MY", "TH", "VN"][:n_assets],
        index=[f"A{i}" for i in range(n_assets)],
    )


def test_optimizer_runs_and_returns_results():
    prices = _synthetic_prices()
    group_map = _group_map()
    benchmark = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)

    models = [
        MomentumModel(name="mom_63", lookback=63),
        LowVolatilityModel(name="lowvol_63", lookback=63),
    ]
    allocator_configs = [
        {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
        {"weighting": "dollar_neutral", "top_fraction": 0.33, "rebalance_frequency": "D"},
    ]

    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=100,
        test_size=30,
    )
    df = opt.optimize(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=(0.0, 15.0),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )

    assert not df.empty
    assert "net_ir" in df.columns
    assert "score" in df.columns
    assert "passed_all" in df.columns
    assert "constraint_max_drawdown" in df.columns
    assert "constraint_turnover" in df.columns
    assert "constraint_deflated_sharpe" in df.columns


def test_optimizer_respects_drawdown_constraint():
    prices = _synthetic_prices()
    group_map = _group_map()
    benchmark = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)

    models = [MomentumModel(name="mom_63", lookback=63)]
    allocator_configs = [
        {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
    ]

    # Very tight drawdown limit should cause violations
    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.01,  # Impossibly tight
        max_turnover=5.0,
        train_size=100,
        test_size=30,
    )
    df = opt.optimize(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=(0.0,),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )

    assert not df.empty
    # All should fail the drawdown constraint
    assert not df["constraint_max_drawdown_passed"].any()


def test_optimizer_score_penalizes_violations():
    prices = _synthetic_prices()
    group_map = _group_map()
    benchmark = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)

    models = [MomentumModel(name="mom_63", lookback=63)]
    allocator_configs = [
        {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
    ]

    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=100,
        test_size=30,
    )
    df = opt.optimize(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=(0.0, 15.0),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )

    assert not df.empty
    # 0 bps should have higher score than 15 bps (same model, lower costs)
    scores_0 = df.loc[df["cost_bps"] == 0.0, "score"].iloc[0]
    scores_15 = df.loc[df["cost_bps"] == 15.0, "score"].iloc[0]
    assert scores_0 >= scores_15


def test_country_cost_grid_scales_net_costs():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]},
        index=dates,
    )
    group_map = pd.Series({"A": "Singapore", "B": "Indonesia"})

    opt = NetIRConstrainedOptimizer()
    cost_0 = opt._compute_costs(weights, 0.0, group_map)
    cost_15 = opt._compute_costs(weights, 15.0, group_map)
    cost_50 = opt._compute_costs(weights, 50.0, group_map)

    assert cost_0.sum() == pytest.approx(0.0)
    assert cost_15.iloc[1] == pytest.approx(0.5 * 0.0015 + 0.5 * 0.0025)
    assert cost_50.iloc[1] == pytest.approx(cost_15.iloc[1] * (50.0 / 15.0))
    assert cost_50.sum() > cost_15.sum() > cost_0.sum()


def test_optimizer_can_return_detailed_candidate_results():
    prices = _synthetic_prices(n_days=220, n_assets=5)

    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=80,
        test_size=20,
        embargo=2,
        max_asset_weight_limit=1.0,
        min_deflated_sharpe=0.0,
    )
    results = opt.evaluate_all_candidates(
        models=[MomentumModel(name="mom_21", lookback=21)],
        allocator_configs=[{"weighting": "long_only", "top_fraction": 0.5, "rebalance_frequency": "W"}],
        cost_bps_grid=(5.0,),
        prices=prices,
        group_map=None,
        benchmark_returns=None,
    )
    frame = opt.results_to_frame(results)

    assert len(results) == 1
    assert not results[0].net_returns.empty
    assert frame.loc[0, "candidate_id"] == results[0].candidate_id


def test_optimizer_preserves_signal_metadata_in_candidate_frame():
    prices = _synthetic_prices(n_days=220, n_assets=5)
    spec = next(spec for spec in build_source_strategy_specs() if spec.strategy_id == "kakushadze_moving_average")
    model = MovingAverageCrossoverModel(name="source_ma_10_40", fast=10, slow=40, source_spec=spec)
    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=80,
        test_size=20,
        embargo=2,
        max_asset_weight_limit=1.0,
        min_deflated_sharpe=0.0,
    )

    frame = opt.optimize(
        models=[model],
        allocator_configs=[{"weighting": "long_only", "top_fraction": 0.5, "rebalance_frequency": "W"}],
        cost_bps_grid=(5.0,),
        prices=prices,
        group_map=None,
        benchmark_returns=None,
    )

    assert frame.loc[0, "meta_source_strategy_id"] == "kakushadze_moving_average"
    assert frame.loc[0, "meta_family"] == "moving_average_crossover"


def test_optimizer_supports_group_neutral_long_only_capped_allocator():
    prices = _synthetic_prices(n_days=220, n_assets=5)
    group_map = _group_map()
    benchmark = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    opt = NetIRConstrainedOptimizer(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=80,
        test_size=20,
        embargo=2,
        target_volatility=0.12,
        max_leverage=1.0,
        max_asset_weight_limit=0.05,
        min_deflated_sharpe=0.0,
    )

    frame = opt.optimize(
        models=[MomentumModel(name="mom_21", lookback=21)],
        allocator_configs=[
            {
                "weighting": "group_neutral_long_only_capped",
                "top_fraction": 1.0,
                "max_weight": 0.05,
                "rebalance_frequency": "W",
            }
        ],
        cost_bps_grid=(5.0,),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )

    assert len(frame) == 1
    assert frame.loc[0, "meta_max_weight"] == pytest.approx(0.05)
    assert frame.loc[0, "constraint_max_asset_weight"] <= 0.05


def test_precomputed_rule_signals_match_walk_forward_rule_signals():
    prices = _synthetic_prices(n_days=220, n_assets=5)
    group_map = _group_map()
    benchmark = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    allocator_configs = [
        {"weighting": "group_neutral_long_only", "top_fraction": 0.5, "rebalance_frequency": "W"},
    ]
    common_kwargs = dict(
        drawdown_limit=-0.30,
        max_turnover=5.0,
        train_size=80,
        test_size=20,
        embargo=2,
        max_asset_weight_limit=1.0,
        min_deflated_sharpe=0.0,
        max_leverage=1.0,
    )
    walk_forward = NetIRConstrainedOptimizer(**common_kwargs).optimize(
        models=[MomentumModel(name="mom_21", lookback=21)],
        allocator_configs=allocator_configs,
        cost_bps_grid=(5.0,),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )
    precomputed = NetIRConstrainedOptimizer(**common_kwargs, precompute_full_signals=True).optimize(
        models=[MomentumModel(name="mom_21", lookback=21)],
        allocator_configs=allocator_configs,
        cost_bps_grid=(5.0,),
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark,
    )

    assert precomputed.loc[0, "net_ir"] == pytest.approx(walk_forward.loc[0, "net_ir"])
    assert precomputed.loc[0, "net_sharpe"] == pytest.approx(walk_forward.loc[0, "net_sharpe"])
    assert precomputed.loc[0, "max_drawdown"] == pytest.approx(walk_forward.loc[0, "max_drawdown"])
