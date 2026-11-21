from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.yfinance_loader import DownloadResult, download_price_history


class _MockTicker:
    def __init__(self, ticker: str, data: dict[str, pd.DataFrame]):
        self.ticker = ticker
        self._data = data

    def history(self, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
        return self._data.get(period, pd.DataFrame())


def _make_synthetic_history(ticker: str, n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n)
    np.random.seed(42)
    base = 100 + np.random.randn(n).cumsum()
    return pd.DataFrame(
        {
            "Open": base - 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=n),
        },
        index=dates,
    )


@pytest.fixture(autouse=True)
def _patch_yfinance(monkeypatch):
    """Monkeypatch yfinance.Ticker so no network calls are made."""
    import yfinance as yf

    def _mock_ticker_factory(ticker: str):
        data = {
            "1mo": _make_synthetic_history(ticker, n=22),
            "5d": _make_synthetic_history(ticker, n=5),
        }
        return _MockTicker(ticker, data)

    monkeypatch.setattr(yf, "Ticker", _mock_ticker_factory)


def test_download_price_history_backward_compatibility():
    """Default ohlcv=False must behave exactly as before."""
    result = download_price_history(["AAPL", "MSFT"], period="1mo")

    assert isinstance(result, DownloadResult)
    assert isinstance(result.prices, pd.DataFrame)
    assert result.ohlcv is None
    assert "AAPL" in result.prices.columns
    assert "MSFT" in result.prices.columns
    assert not result.prices.empty


def test_download_price_history_ohlcv():
    result = download_price_history(["AAPL"], period="1mo", ohlcv=True)

    assert result.ohlcv is not None
    assert isinstance(result.ohlcv, pd.DataFrame)
    assert ("AAPL", "Close") in result.ohlcv.columns
    assert ("AAPL", "Open") in result.ohlcv.columns
    assert ("AAPL", "High") in result.ohlcv.columns
    assert ("AAPL", "Low") in result.ohlcv.columns
    assert ("AAPL", "Volume") in result.ohlcv.columns

    assert "AAPL" in result.prices.columns


def test_download_price_history_empty_ohlcv():
    result = download_price_history([], period="1mo", ohlcv=True)
    assert result.ohlcv is not None
    assert result.ohlcv.empty


def test_download_price_history_failures():
    result = download_price_history(["UNKNOWN"], period="1y", ohlcv=True)
    assert "UNKNOWN" in result.failures
    assert result.ohlcv is not None
    assert result.ohlcv.empty
