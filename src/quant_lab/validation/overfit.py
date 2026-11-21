from __future__ import annotations

from typing import Sequence
from itertools import combinations
import math

import numpy as np
import pandas as pd

from quant_lab.validation.metrics import deflated_sharpe_ratio, sharpe_ratio

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore[assignment]


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray | pd.DataFrame,
    benchmark_returns: Sequence[float] | None = None,
    n_trials: int | None = None,
    n_splits: int = 8,
    strict: bool = False,
) -> float:
    """Estimate PBO with CSCV selection ranks.

    Contiguous blocks are split into every half-sample IS/OOS combination. For
    each path, the best in-sample strategy is ranked out of sample; PBO is the
    fraction of paths where that selected strategy falls at or below the OOS
    median. ``strict=True`` raises when evidence is insufficient.
    """

    if isinstance(returns_matrix, pd.DataFrame):
        matrix = returns_matrix.values
    else:
        matrix = np.asarray(returns_matrix)

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        if strict:
            raise ValueError("PBO requires a 2D matrix with at least two strategies")
        return 0.0

    valid_rows = ~np.isnan(matrix).all(axis=1)
    valid_cols = ~np.isnan(matrix).all(axis=0)
    matrix = matrix[valid_rows][:, valid_cols]
    n_obs, n_strategies = matrix.shape
    if n_splits < 4 or n_splits % 2 != 0 or n_obs < n_splits * 2:
        if strict:
            raise ValueError("PBO requires an even n_splits >= 4 and at least two rows per split")
        return 0.0
    if benchmark_returns is not None:
        benchmark = np.asarray(benchmark_returns, dtype=float)
        if len(benchmark) != n_obs:
            raise ValueError("benchmark_returns must align with the returns matrix")
        matrix = matrix - benchmark[:, None]
    blocks = np.array_split(np.arange(n_obs), n_splits)
    overfit_paths = 0
    total_paths = 0
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[index] for index in is_blocks])
        oos_rows = np.concatenate([blocks[index] for index in range(n_splits) if index not in is_set])
        is_perf = np.array([sharpe_ratio(pd.Series(matrix[is_rows, column])) for column in range(n_strategies)])
        oos_perf = np.array([sharpe_ratio(pd.Series(matrix[oos_rows, column])) for column in range(n_strategies)])
        if not np.isfinite(is_perf).all() or not np.isfinite(oos_perf).all():
            if strict:
                raise ValueError("PBO path contains non-finite performance")
            continue
        selected = int(np.argmax(is_perf))
        selected_oos = oos_perf[selected]
        relative_rank = (1.0 + float(np.sum(oos_perf < selected_oos))) / (n_strategies + 1.0)
        overfit_paths += int(relative_rank <= 0.5)
        total_paths += 1
    if total_paths == 0:
        if strict:
            raise ValueError("PBO produced no valid CSCV paths")
        return 0.0
    return float(overfit_paths / total_paths)


def deflated_sharpe_probability(
    returns: pd.Series | Sequence[float],
    trial_sharpes: Sequence[float],
) -> float:
    """Wrapper around existing DSR that returns a probability."""
    return deflated_sharpe_ratio(returns, trial_sharpes)


def multiple_comparison_adjustment(
    trial_sharpes: Sequence[float],
    method: str = "bonferroni",
) -> pd.Series:
    """Return adjusted p-values for a family of Sharpe ratios tested against zero.

    Methods: bonferroni, holm, fdr_bh (Benjamini-Hochberg).
    """

    values = np.asarray([x for x in trial_sharpes if np.isfinite(x)], dtype=float)
    n = len(values)
    if n == 0:
        return pd.Series(dtype=float)

    pvalues = np.zeros(n)
    for i, sr in enumerate(values):
        if sr == 0 or n < 2:
            pvalues[i] = 1.0
            continue
        t_stat = sr * math.sqrt(max(1, n - 1))
        if stats is not None:
            pvalues[i] = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        else:
            pvalues[i] = 2 * (1 - _approx_norm_cdf(abs(t_stat)))

    if method == "bonferroni":
        adjusted = np.minimum(pvalues * n, 1.0)
    elif method == "holm":
        sorted_idx = np.argsort(pvalues)
        adjusted = np.zeros(n)
        prev = 0.0
        for rank, idx in enumerate(sorted_idx, start=1):
            raw = pvalues[idx] * (n - rank + 1)
            adjusted[idx] = max(prev, min(raw, 1.0))
            prev = adjusted[idx]
    elif method == "fdr_bh":
        sorted_idx = np.argsort(pvalues)
        adjusted = np.zeros(n)
        for rank, idx in enumerate(sorted_idx, start=1):
            raw = pvalues[idx] * n / rank
            adjusted[idx] = min(raw, 1.0)
        for i in range(n - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])
    else:
        adjusted = pvalues

    return pd.Series(adjusted, index=[f"trial_{i}" for i in range(n)])


def sharpe_decay_analysis(
    returns: pd.Series | Sequence[float],
    windows: tuple[int, ...] = (63, 126, 252, 504),
) -> pd.DataFrame:
    """Rolling Sharpe over different windows to detect decay."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, float]] = []

    for w in windows:
        if len(series) < w:
            continue
        rolling = series.rolling(window=w)
        roll_mean = rolling.mean()
        roll_std = rolling.std(ddof=1)
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
        rows.append(
            {
                "window": w,
                "latest_sharpe": float(roll_sharpe.iloc[-1]) if not roll_sharpe.empty else 0.0,
                "mean_sharpe": float(roll_sharpe.mean()) if not roll_sharpe.empty else 0.0,
                "min_sharpe": float(roll_sharpe.min()) if not roll_sharpe.empty else 0.0,
                "max_sharpe": float(roll_sharpe.max()) if not roll_sharpe.empty else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _approx_norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation for standard normal CDF."""
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419
    c = 0.39894228

    if x >= 0.0:
        t = 1.0 / (1.0 + p * x)
        return 1.0 - c * math.exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)
    else:
        return 1.0 - _approx_norm_cdf(-x)
