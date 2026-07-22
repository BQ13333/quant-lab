from __future__ import annotations

import pandas as pd

from quant_lab.data.asean_universe import Instrument


FX_TICKERS = {
    "CNY": "CNY=X",
    "HKD": "HKD=X",
    "SGD": "SGD=X",
    "IDR": "IDR=X",
    "JPY": "JPY=X",
    "KRW": "KRW=X",
    "MYR": "MYR=X",
    "THB": "THB=X",
    "PHP": "PHP=X",
    "TWD": "TWD=X",
    "VND": "VND=X",
}

CURRENCY_ALIASES = {
    "RMB": "CNY",
}


def normalize_currency(currency: str) -> str:
    code = str(currency).strip().upper()
    return CURRENCY_ALIASES.get(code, code)


def fx_tickers_for(instruments: list[Instrument]) -> list[str]:
    currencies = sorted(
        {
            normalize_currency(instrument.currency)
            for instrument in instruments
            if normalize_currency(instrument.currency) != "USD"
        }
    )
    return [FX_TICKERS[currency] for currency in currencies if currency in FX_TICKERS]


def convert_local_prices_to_usd(
    local_closes: pd.DataFrame,
    instruments: list[Instrument],
    fx_closes: pd.DataFrame,
    stale_limit_days: int = 5,
) -> pd.DataFrame:
    """Convert local close prices to USD using LC-per-USD FX rates.

    Yahoo Finance FX tickers such as SGD=X are quoted as local currency per
    USD. Therefore USD price = local price / FX. Missing FX is forward-filled
    from prior observations only.
    """

    normalized_fx = fx_closes.sort_index().ffill(limit=stale_limit_days)
    converted_series = {}
    instrument_map = {instrument.ticker: instrument for instrument in instruments}

    for ticker in local_closes.columns:
        instrument = instrument_map.get(ticker)
        if instrument is None:
            continue
        currency = normalize_currency(instrument.currency)
        if currency == "USD":
            converted_series[ticker] = local_closes[ticker]
            continue

        fx_ticker = FX_TICKERS.get(currency)
        if fx_ticker is None or fx_ticker not in normalized_fx:
            converted_series[ticker] = pd.Series(pd.NA, index=local_closes.index)
            continue

        aligned_fx = normalized_fx[fx_ticker].reindex(local_closes.index).ffill(limit=stale_limit_days)
        converted_series[ticker] = local_closes[ticker] / aligned_fx

    return pd.DataFrame(converted_series, index=local_closes.index).astype(float)


def base_currency_returns(local_returns: pd.Series, fx_local_per_usd_returns: pd.Series) -> pd.Series:
    """Convert local asset returns to USD returns from LC-per-USD FX returns."""

    return (1.0 + local_returns) / (1.0 + fx_local_per_usd_returns) - 1.0
