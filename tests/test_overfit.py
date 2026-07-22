import numpy as np
import pandas as pd
import pytest

from quant_lab.validation.overfit import (
    deflated_sharpe_probability,
    multiple_comparison_adjustment,
    probability_of_backtest_overfitting,
    sharpe_decay_analysis,
)


def test_pbo_range():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0.001, 0.01, size=(100, 5))
    pbo = probability_of_backtest_overfitting(matrix)
    assert 0.0 <= pbo <= 1.0


def test_pbo_with_benchmark():
    rng = np.random.default_rng(2)
    matrix = rng.normal(0.001, 0.01, size=(80, 4))
    benchmark = rng.normal(0.0005, 0.008, size=80)
    pbo = probability_of_backtest_overfitting(matrix, benchmark_returns=benchmark)
    assert 0.0 <= pbo <= 1.0


def test_pbo_short_matrix_returns_zero():
    matrix = np.random.default_rng(3).normal(0, 0.01, size=(3, 2))
    pbo = probability_of_backtest_overfitting(matrix)
    assert pbo == 0.0


def test_pbo_cscv_deterministic_stable_and_regime_flip_fixtures():
    rows = 20
    noise = np.linspace(-0.001, 0.001, rows)
    stable = np.column_stack(
        [np.tile(0.01 + noise, 4), np.tile(0.001 - noise, 4), np.tile(-0.002 + noise, 4)]
    )
    blocks = []
    for block in range(4):
        first = 0.02 if block < 2 else -0.02
        blocks.append(np.column_stack([first + noise, -first - noise, noise * 0.1]))
    regime_flip = np.vstack(blocks)
    assert probability_of_backtest_overfitting(stable, n_splits=4, strict=True) == 0.0
    assert probability_of_backtest_overfitting(regime_flip, n_splits=4, strict=True) == pytest.approx(1 / 3)


def test_pbo_strict_insufficient_evidence_raises():
    with pytest.raises(ValueError, match="at least two rows per split"):
        probability_of_backtest_overfitting(np.ones((6, 2)), n_splits=4, strict=True)


def test_deflated_sharpe_probability_bounds():
    returns = pd.Series(np.random.default_rng(4).normal(0.001, 0.01, 200))
    dsr = deflated_sharpe_probability(returns, trial_sharpes=[0.2, 0.4, 0.8])
    assert 0.0 <= dsr <= 1.0


def test_multiple_comparison_bonferroni():
    sharpes = [0.5, 0.3, 1.2, 0.1, -0.2]
    adj = multiple_comparison_adjustment(sharpes, method="bonferroni")
    assert len(adj) == 5
    assert (adj <= 1.0).all()
    # Bonferroni should be >= raw p-values
    assert (adj >= 0.0).all()


def test_multiple_comparison_holm():
    sharpes = [0.5, 0.3, 1.2, 0.1, -0.2]
    adj = multiple_comparison_adjustment(sharpes, method="holm")
    assert len(adj) == 5
    assert (adj <= 1.0).all()


def test_multiple_comparison_fdr_bh():
    sharpes = [0.5, 0.3, 1.2, 0.1, -0.2]
    adj = multiple_comparison_adjustment(sharpes, method="fdr_bh")
    assert len(adj) == 5
    assert (adj <= 1.0).all()


def test_sharpe_decay_analysis():
    returns = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, 600), index=pd.bdate_range("2022-01-01", periods=600))
    df = sharpe_decay_analysis(returns, windows=(63, 126, 252))
    assert not df.empty
    assert "window" in df.columns
    assert "latest_sharpe" in df.columns
    assert "mean_sharpe" in df.columns


def test_sharpe_decay_analysis_short_series():
    returns = pd.Series(np.random.default_rng(6).normal(0.001, 0.01, 50))
    df = sharpe_decay_analysis(returns, windows=(63, 126))
    assert df.empty
