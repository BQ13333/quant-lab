"""Regression tests for the 15 Nov 2026 verified math / gate bugs (quant-lab part).

Moved from /workspace/quant_fix/patches/tests/test_verified_bugfixes.py (lab v3).
They must not weaken any gate: NaN evidence fails closed; PBO mid-ranks correct
inflated ties without loosening unique-loss paths; t-stats use T, not family size.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_lab.validation.gates import ValidationGate
from quant_lab.validation.metrics import max_drawdown, sortino_ratio
from quant_lab.validation.overfit import (
    multiple_comparison_adjustment,
    probability_of_backtest_overfitting,
)

def test_tstat_uses_T_not_family_size():
    family = [0.8, 0.4, 0.2, 0.1, -0.1]
    # Annualised Sharpes, T=252, periods_per_year=252: t = 0.8 * sqrt(1) = 0.8
    adj = multiple_comparison_adjustment(family, method="bonferroni", n_observations=252, periods_per_year=252)
    from scipy.stats import norm

    t_ann = 0.8 * math.sqrt(252 / 252)
    raw_p = 2 * (1 - norm.cdf(abs(t_ann)))
    assert adj.iloc[0] == pytest.approx(min(raw_p * 5, 1.0), rel=1e-9)

    # Per-period Sharpes, T=252: t = 0.8 * sqrt(251)
    adj_period = multiple_comparison_adjustment(family, method="bonferroni", n_observations=252, periods_per_year=1)
    t_period = 0.8 * math.sqrt(251)
    raw_period = 2 * (1 - norm.cdf(abs(t_period)))
    assert adj_period.iloc[0] == pytest.approx(min(raw_period * 5, 1.0), rel=1e-9)
    # The old bug used t = 0.8 * sqrt(4) = 1.6, Bonferroni p = 0.54799...
    old_wrong = 2 * (1 - norm.cdf(1.6)) * 5
    assert adj.iloc[0] != pytest.approx(old_wrong, rel=1e-6)
    assert adj_period.iloc[0] != pytest.approx(old_wrong, rel=1e-6)


def test_tstat_missing_T_fails_closed():
    with pytest.raises(TypeError):
        multiple_comparison_adjustment([0.8, 0.4], method="bonferroni")
    with pytest.raises(ValueError):
        multiple_comparison_adjustment([0.8, 0.4], method="bonferroni", n_observations=1)


def test_pbo_n_trials_is_not_a_noop():
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 0.01, size=(32, 3))
    base = probability_of_backtest_overfitting(matrix, n_splits=4, strict=True)
    charged = probability_of_backtest_overfitting(matrix, n_trials=30, n_splits=4, strict=True)
    # A larger declared family cannot produce a *lower* PBO than the matrix itself.
    assert 0.0 <= base <= 1.0
    assert 0.0 <= charged <= 1.0
    assert charged >= base
    # Passing a smaller n_trials must not shrink the family (fail-closed).
    ignored = probability_of_backtest_overfitting(matrix, n_trials=1, n_splits=4, strict=True)
    assert ignored == pytest.approx(base)


def test_pbo_midrank_two_way_tie_among_three():
    # Deterministic two-way top tie among 3: two identical alternating series
    # (nonzero vol, so Sharpe is defined) and one strictly worse series.
    # Mid-rank is 0.625 on every CSCV path, so PBO is 0.
    best = np.tile([0.02, 0.00], 20)
    worse = np.tile([-0.02, 0.00], 20)
    matrix = np.column_stack([worse, best, best])
    pbo = probability_of_backtest_overfitting(matrix, n_splits=4, strict=True)
    assert pbo == pytest.approx(0.0)
    # All-identical series: mid-rank is 0.5, still counted as overfit under <= 0.5.
    tied = np.column_stack([best, best, best, best])
    assert probability_of_backtest_overfitting(tied, n_splits=4, strict=True) == pytest.approx(1.0)


def test_max_drawdown_includes_day1_loss():
    assert max_drawdown([-0.2, 0.1]) == pytest.approx(-0.20)
    # After a first-period gain, a 20% drop from the new peak is -0.20.
    assert max_drawdown([0.1, -0.2]) == pytest.approx(-0.20)


def test_sortino_uses_downside_deviation_about_zero():
    returns = [-0.02, -0.02, 0.01, 0.04]
    # Old bug: std of the two identical negatives is 0, so the function returned 0.0.
    expected = 0.0025 / math.sqrt(0.0002) * math.sqrt(252)
    assert expected == pytest.approx(2.806243, rel=1e-6)
    assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-9)
    assert sortino_ratio(returns) != 0.0


def test_evaluate_and_promote_fail_closed_on_nan():
    gate = ValidationGate()
    decision = gate.evaluate(
        sharpe=float("nan"),
        max_drawdown=-0.05,
        turnover=0.1,
        deflated_sharpe_probability=0.99,
    )
    assert decision.passed is False
    assert any("sharpe" in reason and "finite" in reason for reason in decision.reasons)

    row = {
        "sharpe": 1.5,
        "deflated_sharpe": float("nan"),
        "annual_excess_return": 0.10,
        "information_ratio": 0.5,
        "max_drawdown": -0.10,
        "turnover": 0.5,
    }
    promo = gate.promote(row)
    assert promo.passed is False
    assert any("deflated_sharpe" in reason and "finite" in reason for reason in promo.reasons)

    # Finite passing values still pass (gate not tightened on valid evidence).
    ok = gate.evaluate(sharpe=0.6, max_drawdown=-0.15, turnover=2.0, deflated_sharpe_probability=0.85)
    assert ok.passed is True




def test_pbo_invalid_n_trials_fails_closed():
    matrix = np.tile(np.array([0.01, 0.01, 0.01, 0.01]), (8, 1))
    for bad in (0, -1, 0.5, float("nan"), "3"):
        with pytest.raises((ValueError, TypeError)):
            probability_of_backtest_overfitting(matrix, n_trials=bad, n_splits=4, strict=False)


def test_pbo_midrank_near_ties_are_disjoint():
    # Values inside atol of the selected score must not also count as strictly worse.
    from quant_lab.validation.overfit import _mid_relative_rank
    perf = np.array([0.0, 5e-16, 1.0])
    rank = _mid_relative_rank(perf, 5e-16, 3)
    # |0 - 5e-16| < 1e-15, so 0.0 and 5e-16 are a two-way tie; 1.0 is strictly better.
    # mid-rank = 1 + 0 + 0.5 = 1.5; relative = 1.5/4 = 0.375 (below median / overfit).
    # The overlapping formula (worse OR tie) would have incorrectly scored 0.625.
    assert rank == pytest.approx(0.375)
