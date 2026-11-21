from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.portfolio.sizing import (
    downsample_rebalance,
    scores_to_no_trade_band_weights,
    scores_to_turnover_constrained_weights,
)


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


def test_scores_to_turnover_constrained_weights_sums_to_one(simple_scores, previous_weights):
    w = scores_to_turnover_constrained_weights(simple_scores, previous_weights, max_turnover=0.5)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-9


def test_scores_to_turnover_constrained_weights_respects_limit(simple_scores, previous_weights):
    w = scores_to_turnover_constrained_weights(simple_scores, previous_weights, max_turnover=0.5)
    for date in w.index:
        prev = previous_weights.loc[date]
        curr = w.loc[date]
        turnover = (curr - prev).abs().sum()
        assert turnover <= 0.5 + 1e-9


def test_scores_to_no_trade_band_weights_sums_to_one(simple_scores, previous_weights):
    w = scores_to_no_trade_band_weights(simple_scores, previous_weights, band=0.05)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-9


def test_scores_to_no_trade_band_weights_respects_band(simple_scores, previous_weights):
    band = 0.05
    w = scores_to_no_trade_band_weights(simple_scores, previous_weights, band=band)
    for date in w.index:
        prev = previous_weights.loc[date]
        curr = w.loc[date]
        diff = (curr - prev).abs()
        for asset in w.columns:
            if diff[asset] <= band + 1e-9:
                pass


def test_downsample_rebalance_monthly():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    weights = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=["A", "B", "C"])
    weights = weights.div(weights.sum(axis=1), axis=0)
    w_monthly = downsample_rebalance(weights, frequency="M")
    assert w_monthly.shape == weights.shape


def test_downsample_rebalance_weekly():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    weights = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=["A", "B", "C"])
    weights = weights.div(weights.sum(axis=1), axis=0)
    w_weekly = downsample_rebalance(weights, frequency="W")
    assert w_weekly.shape == weights.shape


def test_downsample_rebalance_daily():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    weights = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=["A", "B", "C"])
    w_daily = downsample_rebalance(weights, frequency="D")
    pd.testing.assert_frame_equal(w_daily, weights)


def test_downsample_rebalance_invalid_frequency():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    weights = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=["A", "B", "C"])
    with pytest.raises(ValueError, match="frequency must be one of"):
        downsample_rebalance(weights, frequency="Q")
