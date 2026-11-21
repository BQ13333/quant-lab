from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.portfolio.rebalance import (
    compare_rebalance_frequencies,
    daily_rebalance,
    monthly_rebalance,
    no_trade_band_rebalance,
    rank_hysteresis_rebalance,
    weekly_rebalance,
)


def test_daily_rebalance_identity():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.2], "B": [0.5, 0.7, 0.8]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    result = daily_rebalance(weights)
    pd.testing.assert_frame_equal(result, weights)


def test_weekly_rebalance_on_friday():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.2, 0.1, 0.0, 0.5, 0.4]},
        index=pd.date_range("2024-01-01", periods=7),
    )
    result = weekly_rebalance(weights, day_of_week=4)
    assert result.iloc[0]["A"] == pytest.approx(0.0)
    assert result.iloc[4]["A"] == pytest.approx(0.0)
    assert result.iloc[5]["A"] == pytest.approx(0.0)
    assert result.iloc[6]["A"] == pytest.approx(0.0)


def test_weekly_rebalance_with_exact_match():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.2, 0.1, 0.0]},
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
    )
    result = weekly_rebalance(weights, day_of_week=4)
    assert result.iloc[0]["A"] == pytest.approx(0.0)
    assert result.iloc[4]["A"] == pytest.approx(0.0)


def test_monthly_rebalance_last_day():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-31"]),
    )
    result = monthly_rebalance(weights)
    assert result.iloc[0]["A"] == pytest.approx(0.0)
    assert result.iloc[1]["A"] == pytest.approx(0.0)
    assert result.iloc[2]["A"] == pytest.approx(0.2)


def test_monthly_rebalance_specific_day():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-31"]),
    )
    result = monthly_rebalance(weights, day_of_month=15)
    assert result.iloc[0]["A"] == pytest.approx(0.0)
    assert result.iloc[1]["A"] == pytest.approx(0.3)
    assert result.iloc[2]["A"] == pytest.approx(0.3)


def test_no_trade_band_rebalance_within_band():
    target = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    current = pd.DataFrame(
        {"A": [0.25, 0.25], "B": [0.25, 0.25]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    result = no_trade_band_rebalance(target, current, band=0.3)
    assert result.iloc[0]["A"] == pytest.approx(0.25)
    assert result.iloc[0]["B"] == pytest.approx(0.25)


def test_no_trade_band_rebalance_outside_band():
    target = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    current = pd.DataFrame(
        {"A": [0.25, 0.25], "B": [0.25, 0.25]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    result = no_trade_band_rebalance(target, current, band=0.2)
    assert result.iloc[0]["A"] == pytest.approx(0.5)
    assert result.iloc[0]["B"] == pytest.approx(0.5)


def test_rank_hysteresis_rebalance_no_change():
    current = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    previous = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = rank_hysteresis_rebalance(current, previous, threshold=0.1)
    pd.testing.assert_series_equal(result, previous)


def test_rank_hysteresis_rebalance_large_change():
    current = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    previous = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = rank_hysteresis_rebalance(current, previous, threshold=0.1)
    pd.testing.assert_series_equal(result, current)


def test_rank_hysteresis_rebalance_partial_change():
    current = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    previous = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    current = pd.Series({"A": 2.0, "B": 1.0, "C": 3.0, "D": 4.0})
    result = rank_hysteresis_rebalance(current, previous, threshold=0.1)
    expected = pd.Series({"A": 2.0, "B": 1.0, "C": 3.0, "D": 4.0})
    pd.testing.assert_series_equal(result, expected)


def test_compare_rebalance_frequencies():
    dates = pd.date_range("2024-01-01", periods=20)
    np.random.seed(0)
    weights = pd.DataFrame(
        np.random.dirichlet(np.ones(3), size=20),
        columns=["A", "B", "C"],
        index=dates,
    )
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.01, size=(20, 3)),
        columns=["A", "B", "C"],
        index=dates,
    )
    strategies = {"random": weights}
    result = compare_rebalance_frequencies(strategies, returns)
    assert "strategy" in result.columns
    assert "freq" in result.columns
    assert "total_return" in result.columns
    assert "sharpe" in result.columns
    assert "mean_turnover" in result.columns
    assert len(result) == 3
    assert set(result["freq"].unique()) == {"daily", "weekly", "monthly"}
