import numpy as np
import pandas as pd
import pytest

from quant_lab.validation.stress import (
    bootstrap_confidence_interval,
    cost_stress,
    country_crisis_slices,
    fx_stress,
    liquidity_stress,
    monthly_stability,
    no_single_country_dominance,
    no_single_month_dominance,
    regime_stability,
)


def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.01, n), index=pd.bdate_range("2024-01-01", periods=n))


def test_cost_stress_monotonically_worse():
    returns = _make_returns()
    weights = pd.DataFrame(np.ones((len(returns), 3)) / 3, index=returns.index, columns=["a", "b", "c"])
    df = cost_stress(returns, weights=weights)
    assert not df.empty
    for i in range(1, len(df)):
        assert df.iloc[i]["net_sharpe"] <= df.iloc[i - 1]["net_sharpe"] + 1e-6


def test_cost_stress_without_weights():
    returns = _make_returns()
    df = cost_stress(returns, weights=None)
    assert not df.empty
    assert df["net_sharpe"].nunique() == 1


def test_liquidity_stress_basic():
    returns = _make_returns()
    weights = pd.DataFrame(np.ones((len(returns), 3)) / 3, index=returns.index, columns=["a", "b", "c"])
    adv = pd.Series({"a": 1e7, "b": 1e7, "c": 1e7})
    df = liquidity_stress(returns, weights, adv)
    assert not df.empty
    assert "capital" in df.columns


def test_fx_stress():
    returns = _make_returns()
    fx = pd.Series(np.random.default_rng(7).normal(0.0001, 0.005, len(returns)), index=returns.index)
    df = fx_stress(returns, fx)
    assert not df.empty
    assert "hedged_sharpe" in df.columns


def test_country_crisis_slices():
    returns = _make_returns()
    weights = pd.DataFrame(np.ones((len(returns), 3)) / 3, index=returns.index, columns=["a", "b", "c"])
    group_map = pd.Series({"a": "USA", "b": "UK", "c": "JPN"})
    result = country_crisis_slices(returns, weights, group_map, crisis_countries=["USA", "UK"])
    assert "USA" in result
    assert "UK" in result


def test_monthly_stability():
    returns = _make_returns(n=252)
    df = monthly_stability(returns)
    assert not df.empty
    assert "best_month" in df.columns
    assert "worst_month" in df.columns


def test_regime_stability():
    returns = _make_returns(n=252)
    regimes = pd.Series(["bull"] * 126 + ["bear"] * 126, index=returns.index)
    df = regime_stability(returns, regimes)
    assert not df.empty
    assert set(df["regime"]) == {"bull", "bear"}


def test_bootstrap_ci_sharpe():
    returns = _make_returns(n=200)
    lower, upper, point = bootstrap_confidence_interval(returns, statistic="sharpe", n_bootstrap=500)
    assert lower <= point <= upper or (np.isnan(lower) and np.isnan(upper))


def test_no_single_country_dominance_passes():
    returns = _make_returns(n=100)
    weights = pd.DataFrame(np.ones((len(returns), 4)) / 4, index=returns.index, columns=["a", "b", "c", "d"])
    group_map = pd.Series({"a": "USA", "b": "UK", "c": "JPN", "d": "DEU"})
    assert no_single_country_dominance(returns, weights, group_map) is True


def test_no_single_month_dominance_passes():
    returns = _make_returns(n=200)
    assert no_single_month_dominance(returns, max_month_contribution=0.3) == True
