from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def _has_cvxpy() -> bool:
    try:
        import cvxpy as cp  # noqa: F401
        return True
    except Exception:
        return False


def _enforce_max_weight_numpy(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Project long-only weights onto a capped simplex."""

    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    if max_weight <= 0:
        return weights / weights.sum()

    n = len(weights)
    if max_weight * n < 1.0 - 1e-12:
        max_weight = 1.0 / n

    base = weights / weights.sum()
    projected = np.zeros_like(base)
    active = np.ones(n, dtype=bool)
    remaining = 1.0

    for _ in range(n):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break
        active_base = base[active_idx]
        if active_base.sum() <= 0:
            allocation = np.full(len(active_idx), remaining / len(active_idx))
        else:
            allocation = remaining * active_base / active_base.sum()
        over = allocation > max_weight
        if not over.any():
            projected[active_idx] = allocation
            remaining = 0.0
            break
        over_idx = active_idx[over]
        projected[over_idx] = max_weight
        active[over_idx] = False
        remaining -= max_weight * len(over_idx)

    if remaining > 1e-12 and active.any():
        projected[active] += remaining / active.sum()
    projected = np.minimum(projected, max_weight)
    total = projected.sum()
    return projected / total if total > 0 else projected


def _max_sharpe_numpy_fallback(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float,
    long_only: bool,
    assets: list[str],
) -> pd.Series:
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
    if inv_vol.sum() == 0:
        inv_vol = np.ones_like(inv_vol)
    if long_only:
        w = inv_vol * np.clip(mu, a_min=0, a_max=None)
    else:
        w = inv_vol * mu
    w = np.where(w < 0, 0, w) if long_only else w
    if w.sum() == 0:
        w = inv_vol
    w = w / w.sum()
    w = _enforce_max_weight_numpy(w, max_weight)
    return pd.Series(w, index=assets)


def _min_variance_numpy_fallback(
    cov: np.ndarray,
    max_weight: float,
    long_only: bool,
    assets: list[str],
) -> pd.Series:
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
    if inv_vol.sum() == 0:
        inv_vol = np.ones(len(inv_vol))
    w = inv_vol / inv_vol.sum()
    w = _enforce_max_weight_numpy(w, max_weight)
    return pd.Series(w, index=assets)


def _turnover_constrained_numpy_fallback(
    mu: np.ndarray,
    prev: np.ndarray,
    max_turnover: float,
    max_weight: float,
    long_only: bool,
    assets: list[str],
) -> pd.Series:
    # Use max_sharpe heuristic as target
    cov = np.diag(np.ones(len(mu)))
    target = _max_sharpe_numpy_fallback(mu, cov, max_weight, long_only, assets).values
    diff = target - prev
    l1 = np.abs(diff).sum()
    if l1 <= max_turnover:
        result = target
    else:
        scale = max_turnover / l1 if l1 > 0 else 0.0
        result = prev + diff * scale
        result = np.maximum(result, 0.0)
        s = result.sum()
        if s > 0:
            result = result / s
    result = _enforce_max_weight_numpy(result, max_weight)
    return pd.Series(result, index=assets)


def cvxpy_max_sharpe(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    shrinkage: float = 0.5,
    max_weight: float = 0.20,
    long_only: bool = True,
) -> pd.Series:
    """Mean-variance max Sharpe via CVXPY with Ledoit-Wolf shrinkage.

    Falls back to a numpy heuristic if CVXPY is unavailable or the solver fails.
    """
    assets = returns.index.tolist()
    mu = returns.values
    cov = cov_matrix.values

    # Apply shrinkage
    mean_var = np.trace(cov) / cov.shape[0]
    target = np.eye(cov.shape[0]) * mean_var
    cov = (1 - shrinkage) * cov + shrinkage * target

    if _has_cvxpy():
        try:
            import cvxpy as cp

            n = len(mu)
            w = cp.Variable(n)
            constraints: list[Any] = [cp.sum(w) == 1, w >= 0]
            if not long_only:
                constraints = [cp.sum(w) == 1]
            if max_weight < 1.0:
                constraints.append(w <= max_weight)
            risk = cp.quad_form(w, cov)
            ret = mu @ w
            prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(verbose=False)
            if w.value is not None:
                weights = np.array(w.value).flatten()
                return pd.Series(weights, index=assets)
        except Exception:
            pass

    warnings.warn(
        "CVXPY not available or solver failed; falling back to numpy heuristic.",
        stacklevel=2,
    )
    return _max_sharpe_numpy_fallback(mu, cov, max_weight, long_only, assets)


def cvxpy_min_variance(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float = 0.20,
    long_only: bool = True,
) -> pd.Series:
    """Minimum-variance portfolio via CVXPY.

    Falls back to inverse-volatility weights if CVXPY is unavailable or the
    solver fails.
    """
    assets = returns.index.tolist()
    cov = cov_matrix.values

    if _has_cvxpy():
        try:
            import cvxpy as cp

            n = cov.shape[0]
            w = cp.Variable(n)
            constraints: list[Any] = [cp.sum(w) == 1, w >= 0]
            if not long_only:
                constraints = [cp.sum(w) == 1]
            if max_weight < 1.0:
                constraints.append(w <= max_weight)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(verbose=False)
            if w.value is not None:
                weights = np.array(w.value).flatten()
                return pd.Series(weights, index=assets)
        except Exception:
            pass

    warnings.warn(
        "CVXPY not available or solver failed; falling back to numpy inverse-volatility.",
        stacklevel=2,
    )
    return _min_variance_numpy_fallback(cov, max_weight, long_only, assets)


def cvxpy_turnover_constrained(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    max_turnover: float = 0.5,
    max_weight: float = 0.20,
    long_only: bool = True,
) -> pd.Series:
    """Max-Sharpe with an L1 turnover constraint via CVXPY.

    Falls back to a numpy projection onto the turnover ball if CVXPY is
    unavailable or the solver fails.
    """
    assets = returns.index.tolist()
    mu = returns.values
    cov = cov_matrix.values
    prev = prev_weights.reindex(assets).fillna(0.0).values

    if _has_cvxpy():
        try:
            import cvxpy as cp

            n = len(mu)
            w = cp.Variable(n)
            constraints: list[Any] = [
                cp.sum(w) == 1,
                w >= 0,
                cp.norm(w - prev, 1) <= max_turnover,
            ]
            if not long_only:
                constraints = [
                    cp.sum(w) == 1,
                    cp.norm(w - prev, 1) <= max_turnover,
                ]
            if max_weight < 1.0:
                constraints.append(w <= max_weight)
            risk = cp.quad_form(w, cov)
            ret = mu @ w
            prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(verbose=False)
            if w.value is not None:
                weights = np.array(w.value).flatten()
                return pd.Series(weights, index=assets)
        except Exception:
            pass

    warnings.warn(
        "CVXPY not available or solver failed; falling back to numpy turnover projection.",
        stacklevel=2,
    )
    return _turnover_constrained_numpy_fallback(
        mu, prev, max_turnover, max_weight, long_only, assets
    )


def cvxpy_risk_parity(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float = 0.20,
) -> pd.Series:
    """Risk-parity portfolio via CVXPY log-barrier formulation.

    Falls back to inverse-volatility weights if CVXPY is unavailable or the
    solver fails.
    """
    assets = returns.index.tolist()
    cov = cov_matrix.values

    if _has_cvxpy():
        try:
            import cvxpy as cp

            n = cov.shape[0]
            w = cp.Variable(n)
            # Log-barrier risk-parity formulation (Spinu)
            obj = cp.Minimize(
                0.5 * cp.quad_form(w, cov)
                - cp.sum(cp.log(w + 1e-8))
            )
            constraints: list[Any] = [cp.sum(w) == 1, w >= 1e-6]
            if max_weight < 1.0:
                constraints.append(w <= max_weight)
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(verbose=False)
            if w.value is not None:
                weights = np.array(w.value).flatten()
                return pd.Series(weights, index=assets)
        except Exception:
            pass

    warnings.warn(
        "CVXPY not available or solver failed; falling back to inverse-volatility risk parity.",
        stacklevel=2,
    )
    return _min_variance_numpy_fallback(cov, max_weight, long_only=True, assets=assets)
