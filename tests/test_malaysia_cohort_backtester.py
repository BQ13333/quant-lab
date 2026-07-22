from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.malaysia_pilot import MalaysiaHistoricalBacktester
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort


@pytest.mark.parametrize("strategy", malaysia_atomic_cohort(), ids=lambda item: item.strategy_id)
def test_every_atomic_strategy_score_is_causal_and_finite_when_warmed_up(strategy):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2023-01-02", periods=420)
    columns = [f"{1000 + item:04d}.KL" for item in range(34)]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, size=(420, 34)), index=index, columns=columns)
    prices = 10.0 * (1.0 + returns).cumprod()
    adv = pd.DataFrame(rng.lognormal(15.0, 0.5, size=(420, 34)), index=index, columns=columns)
    vol = returns.rolling(63).std().shift(1)
    baseline = MalaysiaHistoricalBacktester._score(
        strategy.family, prices, adv, vol, dict(strategy.parameters)
    )
    changed = prices.copy()
    changed.iloc[-1] *= 10.0
    perturbed = MalaysiaHistoricalBacktester._score(
        strategy.family, changed, adv, vol, dict(strategy.parameters)
    )
    pd.testing.assert_series_equal(baseline.iloc[-1], perturbed.iloc[-1])
    warmed = baseline.iloc[300:].replace([np.inf, -np.inf], np.nan)
    assert warmed.notna().sum().sum() > 0


def test_calendar_period_rebalance_is_not_dropped_when_month_end_is_weekend():
    index = pd.bdate_range("2026-01-20", "2026-02-05")
    score = pd.DataFrame({"A.KL": 1.0, "B.KL": 0.0}, index=index)
    returns = pd.DataFrame(0.0, index=index, columns=score.columns)
    _, _, weights, _ = MalaysiaHistoricalBacktester._portfolio_returns(
        score, returns, "ME", 0.25
    )
    assert pd.Timestamp("2026-01-31").day_name() == "Saturday"
    assert weights.loc["2026-02-02", "A.KL"] == pytest.approx(1.0)


def test_residual_momentum_survives_periodic_market_wide_missing_prices():
    rng = np.random.default_rng(29)
    index = pd.bdate_range("2022-01-03", periods=420)
    columns = [f"{1000 + item:04d}.KL" for item in range(34)]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, size=(420, 34)), index=index, columns=columns)
    prices = 10.0 * (1.0 + returns).cumprod()
    prices.iloc[::20] = np.nan
    adv = pd.DataFrame(1_000_000.0, index=index, columns=columns)
    vol = returns.rolling(63).std().shift(1)
    score = MalaysiaHistoricalBacktester._score(
        "residual_momentum",
        prices,
        adv,
        vol,
        {"lookback": 126, "beta_lookback": 126},
    )
    assert score.iloc[300:].replace([np.inf, -np.inf], np.nan).notna().sum().sum() > 0
