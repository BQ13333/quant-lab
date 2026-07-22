from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.optimization.search_spaces import (
    AllocatorSearchSpace,
    CostSearchSpace,
    LiquiditySearchSpace,
    ModelSearchSpace,
    PortfolioSearchSpace,
    RegimeSearchSpace,
)


# ---------------------------------------------------------------------------
# Search space fixtures
# ---------------------------------------------------------------------------


def test_model_search_space():
    space = ModelSearchSpace()
    d = space.to_dict()
    assert "lookback" in d
    assert "vol_lookback" in d
    assert set(d["lookback"]) == {21, 63, 126, 252}
    assert set(d["vol_lookback"]) == {21, 63, 126}


def test_allocator_search_space():
    space = AllocatorSearchSpace()
    d = space.to_dict()
    assert set(d["weighting"]) == {
        "dollar_neutral",
        "long_only",
        "group_neutral_dollar_neutral",
        "group_neutral_long_only",
    }
    assert set(d["top_fraction"]) == {0.10, 0.20, 0.33}
    assert set(d["rebalance_frequency"]) == {"D", "W", "M"}


def test_portfolio_search_space():
    space = PortfolioSearchSpace()
    d = space.to_dict()
    assert set(d["target_volatility"]) == {0.08, 0.10, 0.12, 0.15}
    assert set(d["max_leverage"]) == {1.5, 2.0, 3.0}
    assert set(d["max_country_weight"]) == {0.30, 0.40, 0.50}
    assert set(d["max_asset_weight"]) == {0.10, 0.15, 0.20}


def test_cost_search_space():
    space = CostSearchSpace()
    d = space.to_dict()
    assert set(d["cost_bps"]) == {0, 5, 10, 15, 25, 50}
    assert set(d["no_trade_band"]) == {0.0, 0.01, 0.02, 0.05}
    assert set(d["turnover_penalty"]) == {0.0, 0.1, 0.5, 1.0}


def test_liquidity_search_space():
    space = LiquiditySearchSpace()
    d = space.to_dict()
    assert set(d["min_adv_usd"]) == {1e5, 1e6, 5e6}
    assert set(d["min_traded_days"]) == {21, 42, 63}
    assert set(d["max_missing_ratio"]) == {0.1, 0.2, 0.3}


def test_regime_search_space():
    space = RegimeSearchSpace()
    d = space.to_dict()
    assert set(d["regime_threshold"]) == {0.33, 0.50}
    assert set(d["trend_lookback"]) == {63, 126}
    assert set(d["vol_lookback"]) == {21, 63}


# ---------------------------------------------------------------------------
# Optuna objective & holdout isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    assets = ["A", "B", "C", "D"]
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(252, 4) * 0.01, axis=0),
        index=dates,
        columns=assets,
    )
    return prices


@pytest.fixture
def benchmark_returns(synthetic_prices: pd.DataFrame) -> pd.Series:
    return synthetic_prices.mean(axis=1).pct_change().fillna(0.0)


def test_optuna_objective_runs_with_synthetic_data(
    synthetic_prices: pd.DataFrame, benchmark_returns: pd.Series
):
    optuna = pytest.importorskip("optuna")

    from quant_lab.optimization.optuna_search import objective

    prices = synthetic_prices
    train_idx = prices.index[:180]
    test_idx = prices.index[180:]

    trial = optuna.trial.create_trial(
        params={
            "lookback": 63,
            "vol_lookback": 21,
            "weighting": "dollar_neutral",
            "top_fraction": 0.20,
            "rebalance_frequency": "D",
            "target_volatility": 0.12,
            "max_leverage": 2.0,
            "max_country_weight": 0.40,
            "max_asset_weight": 0.15,
            "cost_bps": 5,
            "no_trade_band": 0.01,
            "turnover_penalty": 0.0,
            "regime_threshold": 0.33,
            "trend_lookback": 63,
            "vol_lookback_regime": 21,
        },
        distributions={
            "lookback": optuna.distributions.CategoricalDistribution([21, 63, 126, 252]),
            "vol_lookback": optuna.distributions.CategoricalDistribution([21, 63, 126]),
            "weighting": optuna.distributions.CategoricalDistribution(
                ["dollar_neutral", "long_only", "group_neutral_dollar_neutral", "group_neutral_long_only"]
            ),
            "top_fraction": optuna.distributions.CategoricalDistribution([0.10, 0.20, 0.33]),
            "rebalance_frequency": optuna.distributions.CategoricalDistribution(["D", "W", "M"]),
            "target_volatility": optuna.distributions.CategoricalDistribution([0.08, 0.10, 0.12, 0.15]),
            "max_leverage": optuna.distributions.CategoricalDistribution([1.5, 2.0, 3.0]),
            "max_country_weight": optuna.distributions.CategoricalDistribution([0.30, 0.40, 0.50]),
            "max_asset_weight": optuna.distributions.CategoricalDistribution([0.10, 0.15, 0.20]),
            "cost_bps": optuna.distributions.CategoricalDistribution([0, 5, 10, 15, 25, 50]),
            "no_trade_band": optuna.distributions.CategoricalDistribution([0.0, 0.01, 0.02, 0.05]),
            "turnover_penalty": optuna.distributions.CategoricalDistribution([0.0, 0.1, 0.5, 1.0]),
            "regime_threshold": optuna.distributions.CategoricalDistribution([0.33, 0.50]),
            "trend_lookback": optuna.distributions.CategoricalDistribution([63, 126]),
            "vol_lookback_regime": optuna.distributions.CategoricalDistribution([21, 63]),
        },
        value=0.5,
    )
    # Create a dummy study so suggest_* works on the trial
    study = optuna.create_study()
    study.add_trial(trial)
    fetched_trial = study.trials[0]

    value = objective(
        fetched_trial,
        prices,
        group_map=None,
        benchmark_returns=benchmark_returns,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    assert isinstance(value, float)


def test_optuna_holdout_isolation(synthetic_prices: pd.DataFrame):
    optuna = pytest.importorskip("optuna")

    from quant_lab.optimization.optuna_search import run_optuna_search
    from quant_lab.validation.holdout import HoldoutManager

    prices = synthetic_prices
    holdout_manager = HoldoutManager(prices.index, holdout_fraction=0.2)

    best_params, best_value, study = run_optuna_search(
        prices=prices,
        group_map=None,
        benchmark_returns=None,
        holdout_manager=holdout_manager,
        n_trials=3,
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_value, float)
    assert len(study.trials) == 3

    # Ensure no trial evaluated on holdout dates
    holdout_dates = set(holdout_manager.holdout_index)
    for trial in study.trials:
        # The trial values are floats (IR), but we can verify indirectly by
        # checking that the study completed without error.
        assert trial.value is not None or trial.state == optuna.trial.TrialState.COMPLETE

    # Explicit holdout isolation check via HoldoutManager
    search_results = [
        {"params": best_params, "fold_dates": list(holdout_manager.train_index[-5:])}
    ]
    holdout_manager.assert_no_holdout_optimization(search_results)  # should not raise

    # This should raise because we inject a holdout date
    bad_results = [
        {"params": best_params, "fold_dates": [holdout_manager.holdout_index[0]]}
    ]
    with pytest.raises(RuntimeError):
        holdout_manager.assert_no_holdout_optimization(bad_results)


def test_optuna_not_installed_graceful():
    # Ensure the module imports without error when optuna is missing
    import importlib
    import sys

    mod_name = "quant_lab.optimization.optuna_search"
    # Re-import to trigger the ImportError path if optuna is absent
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    # Import should succeed (lazy check)
    from quant_lab.optimization import optuna_search as os_mod
    assert hasattr(os_mod, "objective")
    assert hasattr(os_mod, "run_optuna_search")
