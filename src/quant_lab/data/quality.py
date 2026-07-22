from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker missingness statistics.

    Returns a DataFrame indexed by ticker with columns:
    total_rows, missing_count, missing_pct, longest_gap.
    """
    if prices.empty:
        return pd.DataFrame(
            columns=["total_rows", "missing_count", "missing_pct", "longest_gap"]
        )

    total_rows = len(prices)
    missing_count = prices.isna().sum()
    missing_pct = missing_count / total_rows

    longest_gap = pd.Series(index=prices.columns, dtype=int)
    for ticker in prices.columns:
        isna = prices[ticker].isna()
        if isna.all():
            longest_gap[ticker] = total_rows
        else:
            groups = (isna != isna.shift()).cumsum()
            gap_lengths = isna.groupby(groups).sum()
            longest_gap[ticker] = int(gap_lengths.max())

    return pd.DataFrame(
        {
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "longest_gap": longest_gap,
        }
    )


def stale_price_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker stale price statistics (price unchanged from previous day).

    Returns a DataFrame indexed by ticker with columns:
    stale_days_count, max_consecutive_stale.
    """
    if prices.empty:
        return pd.DataFrame(columns=["stale_days_count", "max_consecutive_stale"])

    stale = prices == prices.shift(1)
    stale_days_count = stale.sum()

    max_consecutive_stale = pd.Series(index=prices.columns, dtype=int)
    for ticker in prices.columns:
        stale_series = stale[ticker]
        if stale_series.all():
            max_consecutive_stale[ticker] = len(stale_series)
        else:
            groups = (stale_series != stale_series.shift()).cumsum()
            lengths = stale_series.groupby(groups).sum()
            max_consecutive_stale[ticker] = int(lengths.max())

    return pd.DataFrame(
        {
            "stale_days_count": stale_days_count,
            "max_consecutive_stale": max_consecutive_stale,
        }
    )


def zero_volume_report(volumes: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker zero volume statistics.

    Returns a DataFrame indexed by ticker with columns:
    zero_volume_days, zero_volume_pct.
    """
    if volumes.empty:
        return pd.DataFrame(columns=["zero_volume_days", "zero_volume_pct"])

    total_rows = len(volumes)
    zero_volume_days = (volumes == 0).sum()
    zero_volume_pct = zero_volume_days / total_rows

    return pd.DataFrame(
        {
            "zero_volume_days": zero_volume_days,
            "zero_volume_pct": zero_volume_pct,
        }
    )


def outlier_report(returns: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """Per-ticker return outlier statistics.

    Returns a DataFrame indexed by ticker with columns:
    outlier_count, outlier_pct, max_abs_return.
    """
    if returns.empty:
        return pd.DataFrame(columns=["outlier_count", "outlier_pct", "max_abs_return"])

    total_rows = len(returns)
    cleaned = returns.replace([np.inf, -np.inf], np.nan)

    mean = cleaned.mean()
    std = cleaned.std()

    # Avoid division by zero / NaN std
    valid_std = std.replace(0, np.nan).fillna(np.inf)
    outliers = (cleaned - mean).abs() > z_threshold * valid_std

    outlier_count = outliers.sum()
    outlier_pct = outlier_count / total_rows
    max_abs_return = cleaned.abs().max()

    return pd.DataFrame(
        {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "max_abs_return": max_abs_return,
        }
    )


def gap_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker gap report (calendar days between consecutive valid prices).

    Returns a DataFrame indexed by ticker with columns:
    gap_count, max_gap_days.
    """
    if prices.empty:
        return pd.DataFrame(columns=["gap_count", "max_gap_days"])

    gap_count = pd.Series(index=prices.columns, dtype=int)
    max_gap_days = pd.Series(index=prices.columns, dtype=int)

    for ticker in prices.columns:
        valid = prices[ticker].dropna()
        if len(valid) < 2:
            gap_count[ticker] = 0
            max_gap_days[ticker] = 0
            continue

        diffs = valid.index.to_series().diff().dt.days
        gaps = diffs[diffs > 1]
        gap_count[ticker] = int(len(gaps))
        max_gap_days[ticker] = int(gaps.max()) if len(gaps) > 0 else 0

    return pd.DataFrame(
        {
            "gap_count": gap_count,
            "max_gap_days": max_gap_days,
        }
    )


def summary_quality_score(report: pd.DataFrame) -> float:
    """Compute a composite quality score in [0, 1] from any quality report.

    Looks for known penalty columns and averages their complements.
    """
    if report.empty:
        return 0.0

    scores: list[float] = []

    if "missing_pct" in report.columns:
        scores.append(1.0 - report["missing_pct"].mean())

    if "zero_volume_pct" in report.columns:
        scores.append(1.0 - report["zero_volume_pct"].mean())

    if "outlier_pct" in report.columns:
        scores.append(1.0 - report["outlier_pct"].mean())

    if "stale_days_count" in report.columns:
        total_rows = report.get("total_rows", pd.Series([len(report)] * len(report))).iloc[0]
        scores.append(1.0 - (report["stale_days_count"] / total_rows).mean())

    if "max_gap_days" in report.columns:
        total_rows = report.get("total_rows", pd.Series([len(report)] * len(report))).iloc[0]
        scores.append(1.0 - (report["max_gap_days"] / total_rows).mean())

    if not scores:
        return 0.0

    return max(0.0, min(1.0, float(np.mean(scores))))
