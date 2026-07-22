from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.quality import (
    gap_report,
    missingness_report,
    outlier_report,
    stale_price_report,
    summary_quality_score,
    zero_volume_report,
)


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20)
    return pd.DataFrame(
        {
            "A": [100.0] * 20,
            "B": list(range(100, 120)),
            "C": [None] * 5 + list(range(100, 115)),
        },
        index=dates,
    )


def _synthetic_volumes() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20)
    return pd.DataFrame(
        {
            "A": [1_000_000] * 10 + [0] * 10,
            "B": [500_000] * 20,
        },
        index=dates,
    )


def _synthetic_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20)
    np.random.seed(1)
    rets = np.random.randn(20) * 0.01
    rets[5] = 0.50  # outlier
    return pd.DataFrame({"A": rets}, index=dates)


def test_missingness_report():
    prices = _synthetic_prices()
    report = missingness_report(prices)

    assert report.loc["A", "missing_count"] == 0
    assert report.loc["B", "missing_count"] == 0
    assert report.loc["C", "missing_count"] == 5
    assert report.loc["C", "missing_pct"] == 0.25
    assert report.loc["C", "longest_gap"] == 5


def test_stale_price_report():
    prices = _synthetic_prices()
    report = stale_price_report(prices)

    assert report.loc["A", "stale_days_count"] == 19  # all same
    assert report.loc["A", "max_consecutive_stale"] == 19
    assert report.loc["B", "stale_days_count"] == 0


def test_zero_volume_report():
    volumes = _synthetic_volumes()
    report = zero_volume_report(volumes)

    assert report.loc["A", "zero_volume_days"] == 10
    assert report.loc["A", "zero_volume_pct"] == 0.5
    assert report.loc["B", "zero_volume_days"] == 0


def test_outlier_report():
    returns = _synthetic_returns()
    report = outlier_report(returns, z_threshold=4.0)

    assert report.loc["A", "outlier_count"] == 1
    assert report.loc["A", "max_abs_return"] == pytest.approx(0.50, abs=0.01)


def test_gap_report():
    dates = pd.date_range("2020-01-01", periods=20)
    # Create a gap: missing 2020-01-06 to 2020-01-10
    prices = pd.DataFrame(
        {"A": list(range(100, 105)) + [None] * 5 + list(range(110, 120))},
        index=dates,
    )
    report = gap_report(prices)

    assert report.loc["A", "gap_count"] == 1
    # Calendar days between 2020-01-05 and 2020-01-11 = 6
    assert report.loc["A", "max_gap_days"] == 6


def test_gap_report_no_gaps():
    dates = pd.date_range("2020-01-01", periods=20)
    prices = pd.DataFrame({"A": list(range(100, 120))}, index=dates)
    report = gap_report(prices)
    assert report.loc["A", "gap_count"] == 0
    assert report.loc["A", "max_gap_days"] == 0


def test_summary_quality_score_missingness():
    report = missingness_report(_synthetic_prices())
    score = summary_quality_score(report)
    assert 0.0 <= score <= 1.0


def test_summary_quality_score_zero_volume():
    report = zero_volume_report(_synthetic_volumes())
    score = summary_quality_score(report)
    assert 0.0 <= score <= 1.0


def test_summary_quality_score_outliers():
    report = outlier_report(_synthetic_returns())
    score = summary_quality_score(report)
    assert 0.0 <= score <= 1.0


def test_summary_quality_score_empty():
    assert summary_quality_score(pd.DataFrame()) == 0.0


def test_summary_quality_score_combined():
    # Build a combined report
    prices = _synthetic_prices()
    volumes = _synthetic_volumes()
    returns = _synthetic_returns()

    miss = missingness_report(prices)
    stale = stale_price_report(prices)
    zero = zero_volume_report(volumes)
    out = outlier_report(returns)
    gaps = gap_report(prices)

    combined = miss.join(stale, how="outer")
    combined = combined.join(zero, how="outer")
    combined = combined.join(out, how="outer")
    combined = combined.join(gaps, how="outer")
    combined["total_rows"] = 20

    score = summary_quality_score(combined)
    assert 0.0 <= score <= 1.0
