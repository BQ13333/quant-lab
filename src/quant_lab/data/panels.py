from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.asean_universe import Instrument
from quant_lab.data.fx import FX_TICKERS, convert_local_prices_to_usd, normalize_currency


def build_panels(
    ohlcv_df: pd.DataFrame,
    instruments: list[Instrument],
    fx_prices: pd.DataFrame,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build aligned price/return panels from OHLCV data.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        Wide DataFrame with MultiIndex columns (ticker, field) where field is
        one of {Open, High, Low, Close, Volume}.
    instruments : list[Instrument]
        Universe instruments used for FX mapping and metadata.
    fx_prices : pd.DataFrame
        DataFrame of FX close prices indexed by date. Columns are FX tickers
        (e.g. SGD=X).
    artifact_dir : str | Path | None
        If provided, each panel and metadata is saved as a Parquet file
        under this directory.

    Returns
    -------
    dict[str, Any]
        Keys: local_prices, usd_prices, local_returns, fx_returns,
        usd_returns, metadata.
    """
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df must not be empty")

    tickers = sorted({col[0] for col in ohlcv_df.columns})
    fields = ["Open", "High", "Low", "Close", "Volume"]

    # ---- local_prices panel -------------------------------------------------
    local_prices = pd.DataFrame(
        index=ohlcv_df.index,
        columns=pd.MultiIndex.from_product([tickers, fields]),
        dtype=float,
    )
    for ticker in tickers:
        for field in fields:
            if (ticker, field) in ohlcv_df.columns:
                local_prices[(ticker, field)] = ohlcv_df[(ticker, field)].astype(float)

    # Extract local closes for FX conversion and return computation
    local_closes = pd.DataFrame(
        {
            ticker: local_prices[(ticker, "Close")]
            for ticker in tickers
            if (ticker, "Close") in local_prices.columns
        }
    )

    # ---- usd_prices panel ---------------------------------------------------
    usd_closes = convert_local_prices_to_usd(local_closes, instruments, fx_prices)

    usd_prices = local_prices.copy()
    instrument_map = {inst.ticker: inst for inst in instruments}

    for ticker in tickers:
        if ticker not in usd_closes.columns:
            continue
        usd_prices[(ticker, "Close")] = usd_closes[ticker]

        inst = instrument_map.get(ticker)
        currency = normalize_currency(inst.currency) if inst is not None else ""
        if inst is None or currency == "USD":
            continue

        fx_ticker = FX_TICKERS.get(currency)
        if fx_ticker is None or fx_ticker not in fx_prices.columns:
            continue

        aligned_fx = (
            fx_prices[fx_ticker]
            .sort_index()
            .reindex(local_prices.index)
            .ffill(limit=5)
        )

        for field in ["Open", "High", "Low"]:
            if (ticker, field) in local_prices.columns:
                usd_prices[(ticker, field)] = local_prices[(ticker, field)] / aligned_fx
        # Volume is share count; keep as-is

    # ---- returns ------------------------------------------------------------
    local_returns = np.log(local_closes).diff()

    # FX returns: map each ticker to its currency's FX return
    fx_returns = pd.DataFrame(index=local_closes.index)
    for ticker in tickers:
        inst = instrument_map.get(ticker)
        currency = normalize_currency(inst.currency) if inst is not None else ""
        if inst is None or currency == "USD":
            fx_returns[ticker] = 0.0
            continue
        fx_ticker = FX_TICKERS.get(currency)
        if fx_ticker and fx_ticker in fx_prices.columns:
            fx_series = (
                fx_prices[fx_ticker]
                .reindex(local_closes.index)
                .ffill(limit=5)
            )
            fx_returns[ticker] = np.log(fx_series).diff()
        else:
            fx_returns[ticker] = np.nan

    # USD log returns = local log returns - FX log returns
    usd_returns = pd.DataFrame(index=local_returns.index)
    for ticker in tickers:
        if ticker in local_returns.columns and ticker in fx_returns.columns:
            usd_returns[ticker] = local_returns[ticker] - fx_returns[ticker]
        else:
            usd_returns[ticker] = np.nan

    # ---- metadata -----------------------------------------------------------
    metadata = pd.DataFrame(
        [
            {
                "ticker": inst.ticker,
                "country": inst.country,
                "currency": inst.currency,
                "exchange": inst.exchange,
                "tier": inst.tier,
            }
            for inst in instruments
            if inst.ticker in tickers
        ]
    )

    panels = {
        "local_prices": local_prices,
        "usd_prices": usd_prices,
        "local_returns": local_returns,
        "fx_returns": fx_returns,
        "usd_returns": usd_returns,
        "metadata": metadata,
    }

    if artifact_dir is not None:
        dir_path = Path(artifact_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, panel in panels.items():
            if isinstance(panel, pd.DataFrame):
                panel.to_parquet(dir_path / f"{name}.parquet")

    return panels
