from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DownloadResult:
    prices: pd.DataFrame
    coverage: pd.DataFrame
    failures: dict[str, str]
    ohlcv: pd.DataFrame | None = None


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index


def _history_for_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if history.empty:
        return history
    history = history.copy()
    history.index = _normalize_index(history.index)
    if interval.endswith("d") or interval.endswith("wk") or interval.endswith("mo"):
        history.index = history.index.normalize()
    return history


def download_price_history(
    tickers: list[str],
    period: str,
    interval: str = "1d",
    output_path: str | Path | None = None,
    fallback_periods: list[str] | None = None,
    ohlcv: bool = False,
) -> DownloadResult:
    frames: list[pd.DataFrame] = []
    ohlcv_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    for ticker in dict.fromkeys(tickers):
        try:
            periods = [period, *(fallback_periods or [])]
            history = pd.DataFrame()
            period_used = period
            for candidate_period in dict.fromkeys(periods):
                history = _history_for_ticker(ticker, period=candidate_period, interval=interval)
                period_used = candidate_period
                if not history.empty and "Close" in history:
                    break
            if history.empty or "Close" not in history:
                failures[ticker] = "empty history"
                coverage_rows.append(
                    {
                        "ticker": ticker,
                        "rows": 0,
                        "start": pd.NaT,
                        "end": pd.NaT,
                        "interval": interval,
                        "period_used": None,
                    }
                )
                continue

            close = history["Close"].rename(ticker)
            frames.append(close.to_frame())

            if ohlcv:
                fields = ["Open", "High", "Low", "Close", "Volume"]
                available_fields = [f for f in fields if f in history.columns]
                ticker_df = history[available_fields].copy()
                ticker_df.columns = pd.MultiIndex.from_product([[ticker], available_fields])
                ohlcv_frames.append(ticker_df)

            coverage_rows.append(
                {
                    "ticker": ticker,
                    "rows": int(close.dropna().shape[0]),
                    "start": close.dropna().index.min(),
                    "end": close.dropna().index.max(),
                    "interval": interval,
                    "period_used": period_used,
                }
            )
        except Exception as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            coverage_rows.append(
                {
                    "ticker": ticker,
                    "rows": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "interval": interval,
                    "period_used": None,
                }
            )

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    ohlcv_df = pd.concat(ohlcv_frames, axis=1).sort_index() if ohlcv_frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    if not coverage.empty:
        coverage = coverage.sort_values(["rows", "ticker"], ascending=[False, True])

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(path)
        if ohlcv and not ohlcv_df.empty:
            ohlcv_df.to_parquet(path.with_suffix(".ohlcv.parquet"))
        coverage.to_csv(path.with_suffix(".coverage.csv"), index=False)

    return DownloadResult(prices=prices, coverage=coverage, failures=failures, ohlcv=ohlcv_df if ohlcv else None)
