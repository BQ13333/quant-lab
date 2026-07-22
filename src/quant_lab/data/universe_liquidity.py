from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_adv(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling average daily volume (ADV) in local currency and USD.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices (local or USD) indexed by date.
    volumes : pd.DataFrame
        Share volumes indexed by date.
    window : int
        Look-back window for the rolling mean.

    Returns
    -------
    pd.DataFrame
        Columns ``local_adv`` and ``usd_adv`` indexed by ticker.
        Both are computed from the *provided* prices; pass local prices
        for local_adv and USD prices for usd_adv.
    """
    adv = (prices * volumes).rolling(window=window, min_periods=1).mean().iloc[-1]
    return pd.DataFrame({"local_adv": adv, "usd_adv": adv})


def liquidity_filter(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    metadata: pd.DataFrame,
    min_adv_usd: float = 1e6,
    min_traded_days: int = 42,
    min_price: float = 1.0,
    max_missing_ratio: float = 0.2,
    max_stale_days: int = 5,
) -> pd.Series:
    """Apply lagged liquidity filters using only prior data.

    All computations use ``.shift(1)`` so that the filter decision at
    time *t* uses only information available up to *t-1*.

    Returns a boolean Series indexed by ticker.
    """
    # ---- lagged data (no lookahead) ----------------------------------------
    prices_lag = prices.shift(1)
    volumes_lag = volumes.shift(1)

    tickers = prices.columns.tolist()
    meta_by_ticker = (
        metadata.set_index("ticker")
        if "ticker" in metadata.columns
        else metadata
    )

    # ---- ADV (lagged) ------------------------------------------------------
    adv_df = (prices_lag * volumes_lag).rolling(window=63, min_periods=1).mean()
    usd_adv = adv_df.iloc[-1]

    # ---- traded days ---------------------------------------------------------
    traded_days = volumes_lag.notna().sum()

    # ---- missing ratio -------------------------------------------------------
    missing_ratio = prices_lag.isna().mean()

    # ---- stale days ----------------------------------------------------------
    stale = prices_lag == prices_lag.shift(1)
    stale_days = stale.sum()

    # ---- last price -----------------------------------------------------------
    last_price = prices_lag.iloc[-1]

    # ---- apply filters -------------------------------------------------------
    mask = pd.Series(index=tickers, dtype=bool)
    for ticker in tickers:
        adv_val = usd_adv.get(ticker, np.nan)
        if pd.isna(adv_val):
            adv_val = 0.0

        mask[ticker] = (
            adv_val >= min_adv_usd
            and traded_days.get(ticker, 0) >= min_traded_days
            and last_price.get(ticker, 0) >= min_price
            and missing_ratio.get(ticker, 1.0) <= max_missing_ratio
            and stale_days.get(ticker, 999) <= max_stale_days
        )

    return mask


def country_coverage_diagnostics(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Coverage diagnostics by country.

    Returns a DataFrame with columns:
    country, ticker_count, coverage_start, coverage_end, missing_pct.
    """
    meta_by_ticker = (
        metadata.set_index("ticker")
        if "ticker" in metadata.columns
        else metadata
    )
    countries = meta_by_ticker["country"].unique()

    rows: list[dict[str, object]] = []
    for country in countries:
        tickers = meta_by_ticker[meta_by_ticker["country"] == country].index.tolist()
        country_tickers = [t for t in tickers if t in prices.columns]

        if not country_tickers:
            rows.append(
                {
                    "country": country,
                    "ticker_count": 0,
                    "coverage_start": pd.NaT,
                    "coverage_end": pd.NaT,
                    "missing_pct": 1.0,
                }
            )
            continue

        country_prices = prices[country_tickers]
        missing_pct = country_prices.isna().mean().mean()

        first_valid = country_prices.apply(lambda s: s.first_valid_index())
        last_valid = country_prices.apply(lambda s: s.last_valid_index())

        coverage_start = first_valid.min() if not first_valid.empty else pd.NaT
        coverage_end = last_valid.max() if not last_valid.empty else pd.NaT

        rows.append(
            {
                "country": country,
                "ticker_count": len(country_tickers),
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "missing_pct": missing_pct,
            }
        )

    return pd.DataFrame(rows)
