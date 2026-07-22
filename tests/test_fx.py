import pandas as pd

from quant_lab.data.asean_universe import Instrument
from quant_lab.data.fx import base_currency_returns, convert_local_prices_to_usd, fx_tickers_for, normalize_currency


def test_base_currency_return_formula_for_fx_depreciation():
    local_return = pd.Series([0.10])
    fx_return = pd.Series([0.05])
    usd_return = base_currency_returns(local_return, fx_return)
    assert usd_return.iloc[0] == ((1.10 / 1.05) - 1.0)


def test_convert_local_prices_to_usd_uses_prior_fx_only():
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"D05.SI": [10.0, 11.0, 12.0]}, index=dates)
    fx = pd.DataFrame({"SGD=X": [1.25, None, 1.50]}, index=dates)
    instruments = [Instrument("D05.SI", "DBS", "Singapore", "SGD", "SGX", "stock")]

    converted = convert_local_prices_to_usd(prices, instruments, fx)

    assert converted.loc[dates[0], "D05.SI"] == 8.0
    assert converted.loc[dates[1], "D05.SI"] == 8.8
    assert converted.loc[dates[2], "D05.SI"] == 8.0


def test_fx_tickers_include_north_asia_and_rmb_alias():
    instruments = [
        Instrument("600000.SS", "SPDB", "China", "RMB", "SSE", "stock"),
        Instrument("005930.KS", "Samsung Electronics", "South Korea", "KRW", "KRX", "stock"),
        Instrument("7203.T", "Toyota", "Japan", "JPY", "JPX", "stock"),
        Instrument("2330.TW", "TSMC", "Taiwan", "TWD", "TWSE", "stock"),
        Instrument("0005.HK", "HSBC", "Hong Kong", "HKD", "HKEX", "stock"),
    ]

    assert normalize_currency("rmb") == "CNY"
    assert fx_tickers_for(instruments) == ["CNY=X", "HKD=X", "JPY=X", "KRW=X", "TWD=X"]


def test_convert_local_prices_to_usd_accepts_rmb_alias():
    dates = pd.date_range("2026-01-01", periods=2)
    prices = pd.DataFrame({"600000.SS": [70.0, 77.0]}, index=dates)
    fx = pd.DataFrame({"CNY=X": [7.0, 7.0]}, index=dates)
    instruments = [Instrument("600000.SS", "SPDB", "China", "RMB", "SSE", "stock")]

    converted = convert_local_prices_to_usd(prices, instruments, fx)

    assert converted["600000.SS"].tolist() == [10.0, 11.0]
