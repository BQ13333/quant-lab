from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import scipy.cluster.hierarchy as sch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cvxpy() -> bool:
    try:
        import cvxpy as cp  # noqa: F401
        return True
    except Exception:
        return False


def _valid_mask(scores: pd.DataFrame) -> pd.DataFrame:
    """Return boolean mask of finite, non-null scores."""
    return scores.notna() & np.isfinite(scores)


def _enforce_max_weight(weights: pd.DataFrame, max_weight: float = 0.20) -> pd.DataFrame:
    """Cap individual weights at max_weight and re-normalize.

    If there are fewer than 5 assets with positive weight, the effective
    maximum is raised to 1 / count so that sum-to-1 remains feasible.
    """
    result = weights.copy()
    for idx in result.index:
        row = result.loc[idx]
        positive = row[row > 0]
        if len(positive) == 0:
            continue
        if len(positive) < 5:
            # With fewer than 5 assets, equal weight is the only way to
            # satisfy both sum-to-1 and a reasonable max-weight constraint.
            result.loc[idx, positive.index] = 1.0 / len(positive)
            continue
        effective_max = max_weight
        capped = row.clip(upper=effective_max)
        row_sum = capped.sum()
        if row_sum > 0:
            capped = capped / row_sum
            # After normalizing, some weights might exceed effective_max again
            # Iteratively cap and re-normalize until convergence
            for _ in range(10):
                over = capped > effective_max
                if not over.any():
                    break
                excess = capped[over] - effective_max
                capped[over] = effective_max
                redistribute = excess.sum() / (~over).sum() if (~over).any() else 0.0
                capped[~over] = capped[~over] + redistribute
                capped = capped.clip(lower=0.0)
            result.loc[idx] = capped
    return result


def _check_groups(groups: pd.Series | pd.DataFrame, columns: list) -> pd.Series:
    """Ensure groups is a Series aligned to columns."""
    if isinstance(groups, pd.DataFrame):
        # Assume single-column DataFrame
        groups = groups.iloc[:, 0]
    aligned = groups.reindex(columns)
    return aligned


def _realized_vol(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Rolling annualized volatility (std dev) per asset."""
    min_periods = max(5, min(20, lookback))
    return returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# 1. Equal weight
# ---------------------------------------------------------------------------

def equal_weight(scores: pd.DataFrame) -> pd.DataFrame:
    """Equal weight among all assets with non-null score on each date."""
    mask = _valid_mask(scores)
    counts = mask.sum(axis=1).replace(0, np.nan)
    weights = mask.div(counts, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# 2. Equal country weight
# ---------------------------------------------------------------------------

def equal_country_weight(scores: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Equal weight per country, then equal within each country."""
    aligned = _check_groups(groups, list(scores.columns))
    valid_groups = aligned.dropna().unique()
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    if len(valid_groups) == 0:
        return weights

    country_weight = 1.0 / len(valid_groups)
    for country in valid_groups:
        cols = aligned[aligned == country].index.tolist()
        sub = scores[cols]
        mask = _valid_mask(sub)
        counts = mask.sum(axis=1).replace(0, np.nan)
        inner = mask.div(counts, axis=0).fillna(0.0) * country_weight
        weights[cols] = inner
    return weights


# ---------------------------------------------------------------------------
# 3. Inverse volatility
# ---------------------------------------------------------------------------

def inverse_volatility(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Weight proportional to 1 / realized volatility."""
    vol = _realized_vol(returns, lookback)
    # Align vol to scores index (vol is lagged, so we use as-of)
    vol_aligned = vol.reindex(scores.index, method="ffill")
    inv_vol = 1.0 / vol_aligned.replace(0, np.nan)
    mask = _valid_mask(scores)
    raw = inv_vol * mask
    gross = raw.sum(axis=1).replace(0, np.nan)
    weights = raw.div(gross, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# 4. Country inverse volatility
# ---------------------------------------------------------------------------

def country_inverse_volatility(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    groups: pd.Series,
    lookback: int = 63,
) -> pd.DataFrame:
    """Inverse vol within each country, equal country gross."""
    aligned = _check_groups(groups, list(scores.columns))
    valid_groups = aligned.dropna().unique()
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    if len(valid_groups) == 0:
        return weights

    vol = _realized_vol(returns, lookback)
    vol_aligned = vol.reindex(scores.index, method="ffill")
    inv_vol = 1.0 / vol_aligned.replace(0, np.nan)
    mask = _valid_mask(scores)
    country_gross = 1.0 / len(valid_groups)

    for country in valid_groups:
        cols = aligned[aligned == country].index.tolist()
        raw = inv_vol[cols] * mask[cols]
        gross = raw.sum(axis=1).replace(0, np.nan)
        inner = raw.div(gross, axis=0).fillna(0.0) * country_gross
        weights[cols] = inner
    return weights


# ---------------------------------------------------------------------------
# 5. Risk parity (HRP)
# ---------------------------------------------------------------------------

def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Recover sorted leaf order from linkage matrix."""
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    num_items = link[-1, 3]
    while max(sort_ix) >= num_items:
        sort_ix = [
            int(sort_ix[ix]) - int(num_items)
            for ix, val in enumerate(sort_ix)
            if val >= num_items
            for sort_ix in [val]
        ]
    return list(map(int, sort_ix))


def _get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
    """Compute variance of a cluster."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    w_ = np.ones(len(cluster_items)) / len(cluster_items)
    return float(np.dot(w_, np.dot(cov_slice, w_)))


def _get_rec_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Recursive bisection for HRP weight allocation."""
    w = np.ones(len(sort_ix))
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [
            i[j:k]
            for i in clusters
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(clusters), 2):
            c_items0 = clusters[i]
            c_items1 = clusters[i + 1]
            c_var0 = _get_cluster_var(cov, c_items0)
            c_var1 = _get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1) if (c_var0 + c_var1) > 0 else 0.5
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w


def risk_parity_hrp(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Hierarchical Risk Parity (HRP) using scipy clustering."""
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        valid_cols = scores.loc[date].dropna().index.tolist()
        if not valid_cols:
            continue
        # Use lookback window up to date
        end_loc = returns.index.get_loc(date)
        if end_loc < 5:
            continue
        start_loc = max(0, end_loc - lookback)
        window = returns.iloc[start_loc:end_loc]
        if window.empty or len(window) < 5:
            continue
        sub_rets = window[valid_cols].dropna(how="any", axis=1)
        if sub_rets.empty or sub_rets.shape[1] < 2:
            continue
        cols = sub_rets.columns.tolist()
        cov = sub_rets.cov().values
        corr = sub_rets.corr().fillna(0.0).clip(-1.0, 1.0).values
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0.0)
        try:
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import squareform

            link = sch.linkage(squareform(dist, checks=False), method="single")
            sort_ix = sch.leaves_list(link)
            w = _get_rec_bisection(cov, sort_ix)
            w_series = pd.Series(w, index=cols)
            # Re-normalize to sum to 1
            w_sum = w_series.sum()
            if w_sum > 0:
                w_series = w_series / w_sum
            weights.loc[date, cols] = w_series.reindex(weights.columns).fillna(0.0)
        except Exception:
            # Fallback to equal weight on this date
            n = len(cols)
            weights.loc[date, cols] = 1.0 / n
    return weights


# ---------------------------------------------------------------------------
# 6. Max Sharpe with shrinkage
# ---------------------------------------------------------------------------

def _max_sharpe_cvxpy(
    mu: np.ndarray, cov: np.ndarray, long_only: bool = True
) -> np.ndarray | None:
    try:
        import cvxpy as cp
    except Exception:
        return None
    n = len(mu)
    w = cp.Variable(n)
    if long_only:
        constraints = [cp.sum(w) == 1, w >= 0]
    else:
        constraints = [cp.sum(w) == 1]
    risk = cp.quad_form(w, cov)
    ret = mu @ w
    # Maximize Sharpe via risk-adjusted return (risk free = 0)
    prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(verbose=False)
        except Exception:
            return None
    if w.value is None:
        return None
    return np.array(w.value).flatten()


def _max_sharpe_numpy(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fallback: equal risk contribution / heuristic max Sharpe."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
    if inv_vol.sum() == 0:
        inv_vol = np.ones_like(inv_vol)
    w = inv_vol * np.clip(mu, a_min=0, a_max=None)
    w = np.where(w < 0, 0, w)
    if w.sum() == 0:
        w = inv_vol
    return w / w.sum()


def max_sharpe_shrinkage(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 63,
    shrinkage: float = 0.5,
) -> pd.DataFrame:
    """Mean-variance with Ledoit-Wolf style shrinkage."""
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        valid_cols = scores.loc[date].dropna().index.tolist()
        if not valid_cols:
            continue
        end_loc = returns.index.get_loc(date)
        if end_loc < 5:
            continue
        start_loc = max(0, end_loc - lookback)
        window = returns.iloc[start_loc:end_loc]
        if window.empty or len(window) < 5:
            continue
        sub_rets = window[valid_cols].dropna(how="any", axis=1)
        if sub_rets.empty or sub_rets.shape[1] < 2:
            continue
        cols = sub_rets.columns.tolist()
        mu = sub_rets.mean().values * 252
        cov = sub_rets.cov().values * 252
        # Shrinkage to identity scaled by mean variance
        mean_var = np.trace(cov) / cov.shape[0]
        target = np.eye(cov.shape[0]) * mean_var
        cov = (1 - shrinkage) * cov + shrinkage * target
        w = _max_sharpe_cvxpy(mu, cov)
        if w is None:
            w = _max_sharpe_numpy(mu, cov)
        w_series = pd.Series(w, index=cols)
        w_sum = w_series.sum()
        if w_sum > 0:
            w_series = w_series / w_sum
        # Enforce max weight via helper
        w_series = _enforce_max_weight(w_series.to_frame().T).iloc[0]
        weights.loc[date, cols] = w_series.reindex(weights.columns).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# 7. Minimum variance
# ---------------------------------------------------------------------------

def _min_var_cvxpy(cov: np.ndarray) -> np.ndarray | None:
    try:
        import cvxpy as cp
    except Exception:
        return None
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(verbose=False)
        except Exception:
            return None
    if w.value is None:
        return None
    return np.array(w.value).flatten()


def _min_var_numpy(cov: np.ndarray) -> np.ndarray:
    """Fallback: inverse volatility weights."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
    if inv_vol.sum() == 0:
        inv_vol = np.ones(len(inv_vol))
    return inv_vol / inv_vol.sum()


def minimum_variance(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Minimum variance portfolio, long-only, sum-to-1."""
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        valid_cols = scores.loc[date].dropna().index.tolist()
        if not valid_cols:
            continue
        end_loc = returns.index.get_loc(date)
        if end_loc < 5:
            continue
        start_loc = max(0, end_loc - lookback)
        window = returns.iloc[start_loc:end_loc]
        if window.empty or len(window) < 5:
            continue
        sub_rets = window[valid_cols].dropna(how="any", axis=1)
        if sub_rets.empty or sub_rets.shape[1] < 2:
            continue
        cols = sub_rets.columns.tolist()
        cov = sub_rets.cov().values
        w = _min_var_cvxpy(cov)
        if w is None:
            w = _min_var_numpy(cov)
        w_series = pd.Series(w, index=cols)
        w_sum = w_series.sum()
        if w_sum > 0:
            w_series = w_series / w_sum
        # Enforce max weight via helper
        w_series = _enforce_max_weight(w_series.to_frame().T).iloc[0]
        weights.loc[date, cols] = w_series.reindex(weights.columns).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# 8. Long-only top N
# ---------------------------------------------------------------------------

def long_only_top_n(scores: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Select top N by score, equal weight among selected."""
    ranks = scores.rank(axis=1, method="first", ascending=False)
    selected = ranks.le(top_n)
    counts = selected.sum(axis=1).replace(0, np.nan)
    weights = selected.div(counts, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# 9. Country neutral long only
# ---------------------------------------------------------------------------

def country_neutral_long_only(
    scores: pd.DataFrame, groups: pd.Series, top_fraction: float = 0.33
) -> pd.DataFrame:
    """Long-only with equal group gross exposure."""
    from quant_lab.portfolio.sizing import scores_to_group_neutral_long_only_weights

    return scores_to_group_neutral_long_only_weights(scores, groups, top_fraction=top_fraction)


# ---------------------------------------------------------------------------
# 10. Dollar neutral
# ---------------------------------------------------------------------------

def dollar_neutral(scores: pd.DataFrame, gross_exposure: float = 1.0) -> pd.DataFrame:
    """Dollar-neutral weights from scores."""
    from quant_lab.portfolio.sizing import scores_to_weights

    return scores_to_weights(scores, gross_exposure=gross_exposure)


# ---------------------------------------------------------------------------
# 11. Country neutral dollar neutral
# ---------------------------------------------------------------------------

def country_neutral_dollar_neutral(
    scores: pd.DataFrame, groups: pd.Series, gross_exposure: float = 1.0
) -> pd.DataFrame:
    """Dollar-neutral with equal group gross."""
    from quant_lab.portfolio.sizing import scores_to_group_neutral_weights

    return scores_to_group_neutral_weights(scores, groups, gross_exposure=gross_exposure)


# ---------------------------------------------------------------------------
# 12. Turnover constrained optimizer
# ---------------------------------------------------------------------------

def turnover_constrained_optimizer(
    scores: pd.DataFrame,
    previous_weights: pd.DataFrame,
    max_turnover: float = 0.5,
    use_cvxpy: bool = True,
) -> pd.DataFrame:
    """Optimize new weights subject to turnover <= max_turnover from previous."""
    if not np.isfinite(max_turnover) or max_turnover < 0:
        raise ValueError("max_turnover must be finite and nonnegative")
    if not np.isfinite(previous_weights.to_numpy()).all():
        raise ValueError("previous holdings must be finite")
    # Start with equal-weight target for simplicity (or could use scores as target)
    target = equal_weight(scores)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for date in scores.index:
        prev = previous_weights.loc[date] if date in previous_weights.index else pd.Series(0.0, index=scores.columns)
        tgt = target.loc[date]
        valid = tgt.notna() & np.isfinite(tgt)
        cols = valid[valid].index.tolist()
        if not cols:
            weights.loc[date] = prev.reindex(scores.columns).fillna(0.0)
            continue

        tgt_vec = tgt[cols].values
        prev_vec = prev.reindex(cols).fillna(0.0).values

        if use_cvxpy and _has_cvxpy():
            try:
                import cvxpy as cp

                n = len(cols)
                w = cp.Variable(n)
                turnover = cp.norm(w - prev_vec, 1)
                constraints = [cp.sum(w) == 1, w >= 0, w <= 0.20, turnover <= max_turnover]
                # Minimize distance to target
                obj = cp.Minimize(cp.sum_squares(w - tgt_vec))
                prob = cp.Problem(obj, constraints)
                try:
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                except Exception:
                    prob.solve(verbose=False)
                if w.value is not None:
                    w_vec = np.array(w.value).flatten()
                else:
                    w_vec = None
            except Exception:
                w_vec = None
        else:
            w_vec = None

        if w_vec is None:
            # Fallback: project target onto turnover ball using greedy clipping
            diff = tgt_vec - prev_vec
            l1 = np.abs(diff).sum()
            if l1 <= max_turnover:
                w_vec = tgt_vec
            else:
                scale = max_turnover / l1 if l1 > 0 else 0.0
                w_vec = prev_vec + diff * scale

        # Validate jointly; clipping and renormalizing would invalidate turnover/cap.
        if (not np.isfinite(w_vec).all() or np.any(w_vec < -1e-8)
                or np.any(w_vec > 0.20 + 1e-8)
                or not np.isclose(w_vec.sum(), 1.0, atol=1e-8, rtol=0)
                or np.abs(w_vec - prev_vec).sum() > max_turnover + 1e-8):
            raise ValueError("infeasible allocation: cap, budget and turnover must hold jointly")
        w_series = pd.Series(w_vec, index=cols).reindex(scores.columns).fillna(0.0)
        weights.loc[date] = w_series
    return weights


# ---------------------------------------------------------------------------
# 13. No-trade band optimizer
# ---------------------------------------------------------------------------

def no_trade_band_optimizer(
    scores: pd.DataFrame,
    previous_weights: pd.DataFrame,
    band: float = 0.05,
) -> pd.DataFrame:
    """Only rebalance if weight change exceeds band; otherwise keep previous."""
    target = equal_weight(scores)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for date in scores.index:
        prev = previous_weights.loc[date] if date in previous_weights.index else pd.Series(0.0, index=scores.columns)
        tgt = target.loc[date].fillna(0.0)
        diff = (tgt - prev).abs()
        # Rebalance only assets where diff > band
        rebalance = diff > band
        # For assets not rebalancing, keep previous; for rebalancing, use target
        w = prev.copy()
        w[rebalance] = tgt[rebalance]
        # Normalize to sum to 1 (long-only assumption)
        w = w.clip(lower=0.0)
        s = w.sum()
        if s > 0:
            w = w / s
        weights.loc[date] = w
    return weights


# ---------------------------------------------------------------------------
# 14. Rebalance schedule
# ---------------------------------------------------------------------------

def rebalance_schedule(weights: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    """Downsample daily weights to rebalance frequency, forward-fill between."""
    if frequency == "D":
        return weights.copy()
    if not isinstance(weights.index, pd.DatetimeIndex):
        weights = weights.copy()
        weights.index = pd.to_datetime(weights.index)
    freq_map = {"W": "W-FRI", "M": "ME", "Q": "QE", "Y": "YE"}
    freq = freq_map.get(frequency, frequency)
    # Resample to end of period, forward-fill, reindex to original
    sampled = weights.resample(freq).last().ffill()
    result = sampled.reindex(weights.index, method="ffill")
    return result
