from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.asean_universe import Instrument
from quant_lab.data.panels import build_panels


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=100)
    np.random.seed(0)

    tickers = ["D05.SI", "BBCA.JK"]
    data = {}
    for ticker in tickers:
        close = 100 + np.random.randn(100).cumsum()
        data[(ticker, "Open")] = close - 0.5
        data[(ticker, "High")] = close + 1.0
        data[(ticker, "Low")] = close - 1.0
        data[(ticker, "Close")] = close
        data[(ticker, "Volume")] = np.random.randint(1_000_000, 5_000_000, size=100)

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def instruments() -> list[Instrument]:
    return [
        Instrument("D05.SI", "DBS", "Singapore", "SGD", "SGX", "stock"),
        Instrument("BBCA.JK", "BBCA", "Indonesia", "IDR", "IDX", "stock"),
    ]


@pytest.fixture
def fx_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=100)
    return pd.DataFrame(
        {
            "SGD=X": 1.35 + np.random.randn(100) * 0.01,
            "IDR=X": 15000 + np.random.randn(100) * 100,
        },
        index=dates,
    )


def test_build_panels_keys(synthetic_ohlcv, instruments, fx_prices):
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices)
    expected_keys = {
        "local_prices",
        "usd_prices",
        "local_returns",
        "fx_returns",
        "usd_returns",
        "metadata",
    }
    assert set(panels.keys()) == expected_keys


def test_build_panels_local_prices_shape(synthetic_ohlcv, instruments, fx_prices):
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices)
    lp = panels["local_prices"]
    assert lp.index.equals(synthetic_ohlcv.index)
    assert ("D05.SI", "Close") in lp.columns
    assert ("D05.SI", "Volume") in lp.columns


def test_build_panels_usd_conversion(synthetic_ohlcv, instruments, fx_prices):
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices)
    local_close = panels["local_prices"][("D05.SI", "Close")]
    usd_close = panels["usd_prices"][("D05.SI", "Close")]

    fx = fx_prices["SGD=X"].reindex(local_close.index).ffill(limit=5)
    expected_usd = local_close / fx
    pd.testing.assert_series_equal(usd_close, expected_usd, check_names=False)


def test_build_panels_uses_currency_alias_for_rmb():
    dates = pd.date_range("2026-01-01", periods=3)
    ohlcv = pd.DataFrame(
        {
            ("600000.SS", "Open"): [63.0, 70.0, 77.0],
            ("600000.SS", "High"): [70.0, 77.0, 84.0],
            ("600000.SS", "Low"): [56.0, 63.0, 70.0],
            ("600000.SS", "Close"): [70.0, 77.0, 84.0],
            ("600000.SS", "Volume"): [1_000_000, 1_100_000, 1_200_000],
        },
        index=dates,
    )
    instruments = [Instrument("600000.SS", "SPDB", "China", "RMB", "SSE", "stock")]
    fx_prices = pd.DataFrame({"CNY=X": [7.0, 7.0, 7.0]}, index=dates)

    panels = build_panels(ohlcv, instruments, fx_prices)

    assert panels["usd_prices"][("600000.SS", "Close")].tolist() == [10.0, 11.0, 12.0]
    assert panels["fx_returns"]["600000.SS"].dropna().eq(0.0).all()


def test_build_panels_returns(synthetic_ohlcv, instruments, fx_prices):
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices)
    local_ret = panels["local_returns"]
    usd_ret = panels["usd_returns"]
    fx_ret = panels["fx_returns"]

    assert local_ret.index.equals(synthetic_ohlcv.index)
    assert usd_ret.index.equals(synthetic_ohlcv.index)
    assert fx_ret.index.equals(synthetic_ohlcv.index)

    ticker = "D05.SI"
    expected = local_ret[ticker] - fx_ret[ticker]
    pd.testing.assert_series_equal(usd_ret[ticker], expected, check_names=False)


def test_build_panels_metadata(synthetic_ohlcv, instruments, fx_prices):
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices)
    meta = panels["metadata"]
    assert set(meta["ticker"].tolist()) == {"D05.SI", "BBCA.JK"}
    assert "country" in meta.columns
    assert "currency" in meta.columns
    assert "exchange" in meta.columns
    assert "tier" in meta.columns


def test_build_panels_empty_raises():
    with pytest.raises(ValueError, match="ohlcv_df must not be empty"):
        build_panels(
            pd.DataFrame(),
            [],
            pd.DataFrame(),
        )


def test_build_panels_artifact_dir(synthetic_ohlcv, instruments, fx_prices, tmp_path):
    artifact_dir = tmp_path / "panels"
    panels = build_panels(synthetic_ohlcv, instruments, fx_prices, artifact_dir=artifact_dir)
    for name in panels:
        expected_file = artifact_dir / f"{name}.parquet"
        assert expected_file.exists()
        # Round-trip check (freq may be lost by Parquet)
        if isinstance(panels[name], pd.DataFrame):
            loaded = pd.read_parquet(expected_file)
            pd.testing.assert_frame_equal(
                loaded, panels[name], check_freq=False
            )
