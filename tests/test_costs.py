from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.portfolio.costs import (
    adv_participation_cost,
    bps_cost_grid,
    country_cost_schedule,
    linear_transaction_cost,
    slippage_cost,
    spread_proxy_from_ohlcv,
    total_cost,
    turnover_decomposition,
)


def test_country_cost_schedule_known_countries():
    assert country_cost_schedule("Singapore") == 0.0015
    assert country_cost_schedule("Indonesia") == 0.0025
    assert country_cost_schedule("Malaysia") == 0.0020


def test_country_cost_schedule_default():
    assert country_cost_schedule("Unknown") == 0.0015
    assert country_cost_schedule("Thailand") == 0.0015
    assert country_cost_schedule("Vietnam") == 0.0015
    assert country_cost_schedule("Philippines") == 0.0015


def test_country_cost_schedule_etf():
    assert country_cost_schedule("Singapore", tier="etf") == 0.0005
    assert country_cost_schedule("Unknown", tier="etf") == 0.0005


def test_bps_cost_grid():
    grid = bps_cost_grid()
    assert grid == [0.0, 0.0005, 0.001, 0.0015, 0.0025, 0.005, 0.01]


def test_linear_transaction_cost_on_synthetic():
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.0], "B": [0.5, 0.5, 1.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    costs = linear_transaction_cost(weights, 10.0)
    # Day 0: turnover = 0.0 (no prior position), cost = 0.0
    # Day 1: turnover = 0.0, cost = 0.0
    # Day 2: turnover = 1.0, cost = 0.001
    assert costs.iloc[0] == pytest.approx(0.0)
    assert costs.iloc[1] == pytest.approx(0.0)
    assert costs.iloc[2] == pytest.approx(0.001)


def test_adv_participation_cost_no_penalty():
    weights = pd.DataFrame(
        {"A": [0.1, 0.1, 0.1], "B": [0.1, 0.1, 0.1]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    adv = pd.Series({"A": 1e9, "B": 1e9})
    cost = adv_participation_cost(weights, adv, capital=1e8, max_participation=0.1)
    # No weight change, so no penalty
    assert (cost == 0.0).all()


def test_adv_participation_cost_with_penalty():
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    adv = pd.Series({"A": 1e6, "B": 1e6})
    # Day 0: no prior position, trade = 0, penalty = 0
    # Day 1: trade = 0.5 * 1e8 = 5e7 per asset, participation = 50
    # excess = 50/0.1 - 1 = 499, penalty = 499 * 0.005 = 2.495 per asset
    # Day 2: no trade, cost = 0
    cost = adv_participation_cost(weights, adv, capital=1e8, max_participation=0.1)
    assert cost.iloc[0] == pytest.approx(0.0)
    assert cost.iloc[1] == pytest.approx(4.99, rel=1e-3)
    assert cost.iloc[2] == pytest.approx(0.0)


def test_spread_proxy_from_ohlcv():
    high = pd.DataFrame({"A": [10.0, 11.0], "B": [20.0, 21.0]})
    low = pd.DataFrame({"A": [9.0, 10.0], "B": [19.0, 20.0]})
    close = pd.DataFrame({"A": [9.5, 10.5], "B": [19.5, 20.5]})
    spread = spread_proxy_from_ohlcv(high, low, close)
    expected_a = (10.0 - 9.0) / 9.5  # ~0.1053
    expected_b = (20.0 - 19.0) / 19.5  # ~0.0513
    assert spread["A"].iloc[0] == pytest.approx(expected_a, rel=1e-3)
    assert spread["B"].iloc[0] == pytest.approx(expected_b, rel=1e-3)


def test_slippage_cost():
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    spreads = pd.DataFrame(
        {"A": [0.01, 0.01, 0.01], "B": [0.02, 0.02, 0.02]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    cost = slippage_cost(weights, spreads, participation=0.05)
    # Day 0: no prior position, slippage = 0
    # Day 1: |wdiff| = 0.5 for each, slippage = 0.01*0.05*0.5 + 0.02*0.05*0.5 = 0.00075
    # Day 2: 0
    assert cost.iloc[0] == pytest.approx(0.0)
    assert cost.iloc[1] == pytest.approx(0.00075, rel=1e-3)
    assert cost.iloc[2] == pytest.approx(0.0)


def test_total_cost_components():
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    spreads = pd.DataFrame(
        {"A": [0.01, 0.01, 0.01], "B": [0.02, 0.02, 0.02]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    adv = pd.Series({"A": 1e6, "B": 1e6})
    total, components = total_cost(
        weights, cost_bps=0.001, spreads=spreads, adv=adv, capital=1e8
    )
    assert "linear" in components
    assert "adv_penalty" in components
    assert "slippage" in components
    assert len(total) == 3


def test_total_cost_with_country_schedule():
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    country_schedule = {"A": 0.002, "B": 0.003}
    total, components = total_cost(weights, cost_bps=0.001, country_schedule=country_schedule)
    assert "country" in components
    assert "linear" in components
    # Day 0: no trade, cost = 0
    # Day 1: trade = 1.0. A cost = 0.5*0.002 + 0.5*0.003 = 0.0025
    # Day 2: no trade, cost = 0
    assert components["linear"].iloc[0] == pytest.approx(0.0)
    assert components["linear"].iloc[1] == pytest.approx(0.0025)
    assert components["linear"].iloc[2] == pytest.approx(0.0)
    assert total.iloc[1] == pytest.approx(0.0025)


def test_turnover_decomposition():
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5], "C": [0.0, 0.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    metadata = pd.DataFrame(
        {"country": ["Singapore", "Indonesia", "Singapore"]},
        index=["A", "B", "C"],
    )
    result = turnover_decomposition(weights, metadata=metadata)
    assert "total_turnover" in result.columns
    assert "by_month" in result.columns
    assert "by_country" in result.columns
    assert "by_asset" in result.columns
    assert result.loc[result.index[0], "total_turnover"] == pytest.approx(0.0)
    assert result.loc[result.index[1], "total_turnover"] == pytest.approx(1.0)
    assert result.loc[result.index[2], "total_turnover"] == pytest.approx(1.0)


def test_turnover_decomposition_multiindex_model():
    weights = pd.DataFrame(
        {
            ("model1", "A"): [0.0, 0.5, 0.5],
            ("model1", "B"): [0.0, 0.5, 0.5],
            ("model2", "C"): [0.0, 0.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    result = turnover_decomposition(weights)
    assert "by_model" in result.columns
    assert "total_turnover" in result.columns
    assert result.loc[result.index[0], "total_turnover"] == pytest.approx(0.0)
    assert result.loc[result.index[1], "total_turnover"] == pytest.approx(1.0)
    assert result.loc[result.index[2], "total_turnover"] == pytest.approx(1.0)
