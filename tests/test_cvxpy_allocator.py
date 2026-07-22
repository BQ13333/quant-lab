from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.optimization.cvxpy_allocator import (
    cvxpy_max_sharpe,
    cvxpy_min_variance,
    cvxpy_risk_parity,
    cvxpy_turnover_constrained,
)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=60)
    assets = ["A", "B", "C", "D", "E"]
    rets = pd.DataFrame(np.random.randn(60, 5) * 0.01, index=dates, columns=assets)
    return rets


@pytest.fixture
def expected_returns(synthetic_returns: pd.DataFrame) -> pd.Series:
    return synthetic_returns.mean()


@pytest.fixture
def cov_matrix(synthetic_returns: pd.DataFrame) -> pd.DataFrame:
    return synthetic_returns.cov()


@pytest.fixture
def prev_weights() -> pd.Series:
    return pd.Series(
        {"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20}
    )


# ---------------------------------------------------------------------------
# cvxpy_max_sharpe
# ---------------------------------------------------------------------------


def test_max_sharpe_weights_sum_to_one(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_max_sharpe(expected_returns, cov_matrix, shrinkage=0.5, max_weight=0.20, long_only=True)
    assert isinstance(w, pd.Series)
    assert w.index.tolist() == expected_returns.index.tolist()
    assert abs(w.sum() - 1.0) < 1e-6


def test_max_sharpe_long_only(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_max_sharpe(expected_returns, cov_matrix, shrinkage=0.5, max_weight=0.20, long_only=True)
    assert (w >= -1e-9).all()


def test_max_sharpe_respects_max_weight(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_max_sharpe(expected_returns, cov_matrix, shrinkage=0.5, max_weight=0.20, long_only=True)
    positive = w[w > 1e-9]
    if len(positive) >= 5:
        assert positive.max() <= 0.20 + 1e-6


# ---------------------------------------------------------------------------
# cvxpy_min_variance
# ---------------------------------------------------------------------------


def test_min_variance_weights_sum_to_one(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_min_variance(expected_returns, cov_matrix, max_weight=0.20, long_only=True)
    assert abs(w.sum() - 1.0) < 1e-6


def test_min_variance_long_only(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_min_variance(expected_returns, cov_matrix, max_weight=0.20, long_only=True)
    assert (w >= -1e-9).all()


def test_min_variance_respects_max_weight(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_min_variance(expected_returns, cov_matrix, max_weight=0.20, long_only=True)
    positive = w[w > 1e-9]
    if len(positive) >= 5:
        assert positive.max() <= 0.20 + 1e-6


# ---------------------------------------------------------------------------
# cvxpy_turnover_constrained
# ---------------------------------------------------------------------------


def test_turnover_constrained_respects_limit(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, prev_weights: pd.Series
):
    w = cvxpy_turnover_constrained(
        expected_returns,
        cov_matrix,
        prev_weights,
        max_turnover=0.5,
        max_weight=0.20,
        long_only=True,
    )
    assert abs(w.sum() - 1.0) < 1e-6
    turnover = (w - prev_weights.reindex(w.index).fillna(0.0)).abs().sum()
    assert turnover <= 0.5 + 1e-6


def test_turnover_constrained_long_only(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, prev_weights: pd.Series
):
    w = cvxpy_turnover_constrained(
        expected_returns,
        cov_matrix,
        prev_weights,
        max_turnover=0.5,
        max_weight=0.20,
        long_only=True,
    )
    assert (w >= -1e-9).all()


# ---------------------------------------------------------------------------
# cvxpy_risk_parity
# ---------------------------------------------------------------------------


def test_risk_parity_weights_sum_to_one(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_risk_parity(expected_returns, cov_matrix, max_weight=0.20)
    assert abs(w.sum() - 1.0) < 1e-6


def test_risk_parity_long_only(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w = cvxpy_risk_parity(expected_returns, cov_matrix, max_weight=0.20)
    assert (w >= -1e-9).all()


# ---------------------------------------------------------------------------
# Fallback determinism
# ---------------------------------------------------------------------------


def test_max_sharpe_deterministic(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w1 = cvxpy_max_sharpe(expected_returns, cov_matrix, shrinkage=0.5, max_weight=0.20, long_only=True)
    w2 = cvxpy_max_sharpe(expected_returns, cov_matrix, shrinkage=0.5, max_weight=0.20, long_only=True)
    pd.testing.assert_series_equal(w1, w2)


def test_min_variance_deterministic(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    w1 = cvxpy_min_variance(expected_returns, cov_matrix, max_weight=0.20, long_only=True)
    w2 = cvxpy_min_variance(expected_returns, cov_matrix, max_weight=0.20, long_only=True)
    pd.testing.assert_series_equal(w1, w2)


# ---------------------------------------------------------------------------
# Missing cvxpy graceful degradation
# ---------------------------------------------------------------------------


def test_module_imports_without_cvxpy():
    import importlib
    import sys

    mod_name = "quant_lab.optimization.cvxpy_allocator"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    from quant_lab.optimization import cvxpy_allocator as ca
    assert hasattr(ca, "cvxpy_max_sharpe")
    assert hasattr(ca, "cvxpy_min_variance")
    assert hasattr(ca, "cvxpy_turnover_constrained")
    assert hasattr(ca, "cvxpy_risk_parity")
