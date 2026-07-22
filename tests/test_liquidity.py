from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.universe_liquidity import (
    country_coverage_diagnostics,
    liquidity_filter,
    rolling_adv,
)


def _synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=100)
    np.random.seed(7)

    prices = pd.DataFrame(
        {
            "A": 100 + np.random.randn(100).cumsum(),
            "B": 50 + np.random.randn(100).cumsum(),
            "C": 0.5 + np.random.randn(100).cumsum() * 0.01,  # cheap stock
        },
        index=dates,
    )

    volumes = pd.DataFrame(
        {
            "A": np.random.randint(100_000, 500_000, size=100),
            "B": np.random.randint(50_000, 200_000, size=100),
            "C": np.random.randint(1_000, 10_000, size=100),
        },
        index=dates,
    )

    metadata = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "country": ["Singapore", "Indonesia", "Malaysia"],
            "currency": ["SGD", "IDR", "MYR"],
            "exchange": ["SGX", "IDX", "Bursa"],
            "tier": ["stock", "stock", "stock"],
        }
    )

    return prices, volumes, metadata


def test_rolling_adv():
    prices, volumes, _ = _synthetic_data()
    adv = rolling_adv(prices, volumes, window=20)

    assert isinstance(adv, pd.DataFrame)
    assert "local_adv" in adv.columns
    assert "usd_adv" in adv.columns
    assert adv.index.tolist() == ["A", "B", "C"]

    # Manual check for ticker A
    expected = (prices["A"] * volumes["A"]).rolling(20).mean().iloc[-1]
    assert adv.loc["A", "local_adv"] == pytest.approx(expected, rel=1e-5)


def test_rolling_adv_with_nans():
    dates = pd.date_range("2020-01-01", periods=30)
    prices = pd.DataFrame({"X": [100.0] * 30}, index=dates)
    prices.iloc[10:15, 0] = np.nan
    volumes = pd.DataFrame({"X": [1000] * 30}, index=dates)

    adv = rolling_adv(prices, volumes, window=20)
    assert adv.loc["X", "local_adv"] == pytest.approx(100.0 * 1000.0, rel=1e-5)


def test_liquidity_filter_mask():
    prices, volumes, metadata = _synthetic_data()
    mask = liquidity_filter(
        prices,
        volumes,
        metadata,
        min_adv_usd=1e6,
        min_traded_days=80,
        min_price=1.0,
        max_missing_ratio=0.2,
        max_stale_days=50,
    )

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.index.tolist() == ["A", "B", "C"]

    # C is cheap (price ~0.5) so it should fail min_price
    assert mask["C"] == False


def test_liquidity_filter_no_lookahead():
    dates = pd.date_range("2020-01-01", periods=50)
    prices = pd.DataFrame({"X": 100.0 + np.random.randn(50).cumsum() * 0.1}, index=dates)
    volumes = pd.DataFrame({"X": [1_000_000] * 50}, index=dates)
    metadata = pd.DataFrame(
        {
            "ticker": ["X"],
            "country": ["US"],
            "currency": ["USD"],
            "exchange": ["NYSE"],
            "tier": ["stock"],
        }
    )

    mask = liquidity_filter(prices, volumes, metadata)
    # With shift(1), the last price used is row 48 (price ~100)
    # The ADV is based on rows 1-48 (shifted), so it should be high
    assert mask["X"] == True


def test_country_coverage_diagnostics():
    prices, _, metadata = _synthetic_data()
    diag = country_coverage_diagnostics(prices, metadata)

    assert set(diag["country"]) == {"Singapore", "Indonesia", "Malaysia"}
    assert diag["ticker_count"].sum() == 3
    assert diag["missing_pct"].max() <= 1.0
    assert diag["missing_pct"].min() >= 0.0


def test_country_coverage_with_gaps():
    dates = pd.date_range("2020-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "A": [None] * 10 + list(range(100, 120)),
            "B": list(range(100, 130)),
        },
        index=dates,
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "country": ["X", "X"],
            "currency": ["USD", "USD"],
            "exchange": ["NYSE", "NYSE"],
            "tier": ["stock", "stock"],
        }
    )

    diag = country_coverage_diagnostics(prices, metadata)
    row = diag[diag["country"] == "X"].iloc[0]
    assert row["ticker_count"] == 2
    assert row["missing_pct"] > 0
