import numpy as np
import pandas as pd

from quant_lab.validation.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_sharpe_positive_for_positive_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.01])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_is_negative():
    returns = pd.Series([0.1, -0.2, 0.05])
    assert max_drawdown(returns) < 0


def test_probabilistic_sharpe_bounds():
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= psr <= 1.0


def test_deflated_sharpe_bounds():
    returns = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 200))
    dsr = deflated_sharpe_ratio(returns, trial_sharpes=[0.2, 0.4, 0.8])
    assert 0.0 <= dsr <= 1.0
