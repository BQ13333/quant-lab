from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quant_lab.portfolio import optimizers



@pytest.fixture
def simple_scores() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 1.0, 2.0],
            "B": [2.0, 1.0, 3.0, 2.0, 1.0],
            "C": [3.0, 3.0, 2.0, np.nan, 3.0],
            "D": [np.nan, 4.0, 1.0, 3.0, 4.0],
        },
        index=dates,
    )


@pytest.fixture
def returns_df() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-10-01", periods=150, freq="D")
    assets = ["A", "B", "C", "D"]
    rets = pd.DataFrame(np.random.randn(150, 4) * 0.01, index=dates, columns=assets)
    return rets


@pytest.fixture
def groups() -> pd.Series:
    return pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})


@pytest.fixture
def previous_weights() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "A": [0.25, 0.25, 0.25, 0.25, 0.25],
            "B": [0.25, 0.25, 0.25, 0.25, 0.25],
            "C": [0.25, 0.25, 0.25, 0.25, 0.25],
            "D": [0.25, 0.25, 0.25, 0.25, 0.25],
        },
        index=dates,
    )



def test_equal_weight_sums_to_one(simple_scores):
    w = optimizers.equal_weight(simple_scores)
    row_sums = w.sum(axis=1)
    valid_counts = simple_scores.notna().sum(axis=1)
    assert (row_sums[valid_counts > 0] - 1.0).abs().max() < 1e-10
    assert w.shape == simple_scores.shape
    assert w.isna().sum().sum() == 0



def test_equal_country_weight_sums_to_one(simple_scores, groups):
    w = optimizers.equal_country_weight(simple_scores, groups)
    row_sums = w.sum(axis=1)
    valid_counts = simple_scores.notna().sum(axis=1)
    assert (row_sums[valid_counts > 0] - 1.0).abs().max() < 1e-10
    assert w.isna().sum().sum() == 0


def test_equal_country_weight_balances_groups(simple_scores, groups):
    w = optimizers.equal_country_weight(simple_scores, groups)
    for date in w.index:
        group_x = w.loc[[date], ["A", "B"]].sum(axis=1).iloc[0]
        group_y = w.loc[[date], ["C", "D"]].sum(axis=1).iloc[0]
        if simple_scores.loc[date, ["A", "B"]].notna().any() and simple_scores.loc[date, ["C", "D"]].notna().any():
            assert abs(group_x - group_y) < 1e-10



def test_inverse_volatility_sums_to_one(simple_scores, returns_df):
    w = optimizers.inverse_volatility(simple_scores, returns_df)
    row_sums = w.sum(axis=1)
    valid_counts = simple_scores.notna().sum(axis=1)
    assert (row_sums[valid_counts > 0] - 1.0).abs().max() < 1e-10
    assert w.isna().sum().sum() == 0



def test_country_inverse_volatility_sums_to_one(simple_scores, returns_df, groups):
    w = optimizers.country_inverse_volatility(simple_scores, returns_df, groups)
    row_sums = w.sum(axis=1)
    valid_counts = simple_scores.notna().sum(axis=1)
    assert (row_sums[valid_counts > 0] - 1.0).abs().max() < 1e-10
    assert w.isna().sum().sum() == 0


def test_country_inverse_volatility_balances_groups(simple_scores, returns_df, groups):
    w = optimizers.country_inverse_volatility(simple_scores, returns_df, groups)
    for date in w.index:
        group_x = w.loc[[date], ["A", "B"]].sum(axis=1).iloc[0]
        group_y = w.loc[[date], ["C", "D"]].sum(axis=1).iloc[0]
        if simple_scores.loc[date, ["A", "B"]].notna().any() and simple_scores.loc[date, ["C", "D"]].notna().any():
            assert abs(group_x - group_y) < 1e-10



def test_risk_parity_hrp_sums_to_one(simple_scores, returns_df):
    w = optimizers.risk_parity_hrp(simple_scores, returns_df)
    row_sums = w.sum(axis=1)
    valid = row_sums > 0
    assert (row_sums[valid] - 1.0).abs().max() < 1e-6
    assert w.isna().sum().sum() == 0


def test_risk_parity_hrp_shape(simple_scores, returns_df):
    w = optimizers.risk_parity_hrp(simple_scores, returns_df)
    assert w.shape == simple_scores.shape


def test_risk_parity_hrp_uses_condensed_distance(simple_scores, returns_df):
    sch = pytest.importorskip("scipy.cluster.hierarchy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizers.risk_parity_hrp(simple_scores, returns_df)

    assert not any(isinstance(item.message, sch.ClusterWarning) for item in caught)



def test_max_sharpe_shrinkage_sums_to_one(simple_scores, returns_df):
    w = optimizers.max_sharpe_shrinkage(simple_scores, returns_df)
    row_sums = w.sum(axis=1)
    valid = row_sums > 0
    assert (row_sums[valid] - 1.0).abs().max() < 1e-6
    assert w.isna().sum().sum() == 0


def test_max_sharpe_shrinkage_max_weight(simple_scores, returns_df):
    w = optimizers.max_sharpe_shrinkage(simple_scores, returns_df)
    for date in w.index:
        row = w.loc[date]
        positive = row[row > 0]
        if len(positive) >= 5:
            assert positive.max() <= 0.20 + 1e-9
        else:
            assert positive.max() <= 1.0 / len(positive) + 1e-9



def test_minimum_variance_sums_to_one(simple_scores, returns_df):
    w = optimizers.minimum_variance(simple_scores, returns_df)
    row_sums = w.sum(axis=1)
    valid = row_sums > 0
    assert (row_sums[valid] - 1.0).abs().max() < 1e-6
    assert w.isna().sum().sum() == 0


def test_minimum_variance_max_weight(simple_scores, returns_df):
    w = optimizers.minimum_variance(simple_scores, returns_df)
    for date in w.index:
        row = w.loc[date]
        positive = row[row > 0]
        if len(positive) >= 5:
            assert positive.max() <= 0.20 + 1e-9
        else:
            assert positive.max() <= 1.0 / len(positive) + 1e-9



def test_long_only_top_n_selects_n(simple_scores):
    w = optimizers.long_only_top_n(simple_scores, top_n=2)
    for date in w.index:
        assert w.loc[date].sum() == pytest.approx(1.0)
        assert (w.loc[date] > 0).sum() <= 2


def test_long_only_top_n_with_nans(simple_scores):
    w = optimizers.long_only_top_n(simple_scores, top_n=2)
    assert w.isna().sum().sum() == 0



def test_country_neutral_long_only_groups(simple_scores, groups):
    w = optimizers.country_neutral_long_only(simple_scores, groups, top_fraction=0.5)
    for date in w.index:
        group_x = w.loc[[date], ["A", "B"]].sum(axis=1).iloc[0]
        group_y = w.loc[[date], ["C", "D"]].sum(axis=1).iloc[0]
        if simple_scores.loc[date, ["A", "B"]].notna().any() and simple_scores.loc[date, ["C", "D"]].notna().any():
            assert abs(group_x - group_y) < 1e-10



def test_dollar_neutral_sums_to_zero(simple_scores):
    w = optimizers.dollar_neutral(simple_scores)
    row_sums = w.sum(axis=1)
    valid = simple_scores.notna().sum(axis=1) > 0
    assert (row_sums[valid].abs() < 1e-10).all()



def test_country_neutral_dollar_neutral_groups(simple_scores, groups):
    w = optimizers.country_neutral_dollar_neutral(simple_scores, groups)
    for date in w.index:
        group_x = w.loc[[date], ["A", "B"]].sum(axis=1).iloc[0]
        group_y = w.loc[[date], ["C", "D"]].sum(axis=1).iloc[0]
        if simple_scores.loc[date, ["A", "B"]].notna().any() and simple_scores.loc[date, ["C", "D"]].notna().any():
            assert abs(group_x - group_y) < 1e-10



def test_turnover_constrained_respects_limit(simple_scores, previous_weights):
    w = optimizers.turnover_constrained_optimizer(simple_scores, previous_weights, max_turnover=0.5)
    for date in w.index:
        prev = previous_weights.loc[date]
        curr = w.loc[date]
        turnover = (curr - prev).abs().sum()
        assert turnover <= 0.5 + 1e-9


def test_turnover_constrained_sums_to_one(simple_scores, previous_weights):
    w = optimizers.turnover_constrained_optimizer(simple_scores, previous_weights, max_turnover=0.5)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-9



def test_no_trade_band_respects_band(simple_scores, previous_weights):
    band = 0.05
    w = optimizers.no_trade_band_optimizer(simple_scores, previous_weights, band=band)
    for date in w.index:
        prev = previous_weights.loc[date]
        curr = w.loc[date]
        diff = (curr - prev).abs()
        for asset in w.columns:
            if diff[asset] <= band + 1e-9:
                pass


def test_no_trade_band_sums_to_one(simple_scores, previous_weights):
    w = optimizers.no_trade_band_optimizer(simple_scores, previous_weights, band=0.05)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-9



def test_rebalance_schedule_monthly():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    weights = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=["A", "B", "C"])
    weights = weights.div(weights.sum(axis=1), axis=0)
    w_monthly = optimizers.rebalance_schedule(weights, frequency="M")
    assert w_monthly.shape == weights.shape
    month_ends = weights.resample("ME").last().index
    for me in month_ends:
        if me in weights.index:
            pd.testing.assert_series_equal(w_monthly.loc[me], weights.loc[me])


def test_rebalance_schedule_weekly():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    weights = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=["A", "B", "C"])
    weights = weights.div(weights.sum(axis=1), axis=0)
    w_weekly = optimizers.rebalance_schedule(weights, frequency="W")
    assert w_weekly.shape == weights.shape


def test_rebalance_schedule_daily():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    weights = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=["A", "B", "C"])
    w_daily = optimizers.rebalance_schedule(weights, frequency="D")
    pd.testing.assert_frame_equal(w_daily, weights)
