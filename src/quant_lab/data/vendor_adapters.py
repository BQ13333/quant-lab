from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from quant_lab.data.yfinance_loader import DownloadResult, download_price_history


REFINITIV_OHLCV_FIELD_MAP = {
    "Open": "OPEN_PRC",
    "High": "HIGH_1",
    "Low": "LOW_1",
    "Close": "TRDPRC_1",
    "Volume": "ACVOL_UNS",
}


class RefinitivRetryableProviderError(RuntimeError):
    """A transient provider failure that must halt the current extraction run."""


def is_refinitiv_retryable_provider_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in ("429", "too many requests", "rate limit", "rate-limit", "throttl")
    )


_is_refinitiv_retryable_provider_error = is_refinitiv_retryable_provider_error


def _normalize_refinitiv_field_name(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _refinitiv_field_aliases(value: str) -> set[str]:
    aliases = {value}
    if "." in value:
        aliases.add(value.split(".")[-1])
    normalized = {_normalize_refinitiv_field_name(alias) for alias in aliases}
    synonym_groups = (
        {"PRICEOPEN", "OPENPRICE"},
        {"PRICEHIGH", "HIGHPRICE"},
        {"PRICELOW", "LOWPRICE"},
        {"PRICECLOSE", "CLOSEPRICE"},
    )
    for group in synonym_groups:
        if normalized & group:
            normalized.update(group)
    return normalized


class DataVendor(ABC):
    """Abstract base for data vendor adapters."""

    @abstractmethod
    def download(self, tickers: list[str], start: str, end: str, fields: list[str]) -> pd.DataFrame:
        """Download market data for the given tickers and date range."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the vendor API is reachable and healthy."""
        ...


class YahooFinanceAdapter(DataVendor):
    """Adapter wrapping the existing yfinance_loader."""

    def __init__(self, period: str = "max", interval: str = "1d"):
        self.period = period
        self.interval = interval

    def download(self, tickers: list[str], start: str, end: str, fields: list[str]) -> pd.DataFrame:
        ohlcv = any(f in fields for f in ("Open", "High", "Low", "Close", "Volume"))
        result = download_price_history(
            tickers=tickers,
            period=self.period,
            interval=self.interval,
            ohlcv=ohlcv,
        )
        if ohlcv and result.ohlcv is not None:
            return result.ohlcv
        return result.prices

    def health_check(self) -> bool:
        try:
            download_price_history(["AAPL"], period="5d")
            return True
        except Exception:
            return False


class BloombergAdapter(DataVendor):
    """Stub for Bloomberg API integration."""

    def download(self, tickers: list[str], start: str, end: str, fields: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Bloomberg adapter not yet implemented")

    def health_check(self) -> bool:
        raise NotImplementedError("Bloomberg adapter not yet implemented")


class RefinitivAdapter(DataVendor):
    """Refinitiv/LSEG Data Library adapter.

    Credentials are intentionally not accepted by this class. Configure them in
    the local LSEG Data Library config file, through Workspace/Eikon desktop
    session discovery, or with environment variables supported by `lseg-data`.
    """

    def __init__(
        self,
        session_name: str | None = None,
        config_name: str | None = None,
        interval: str = "1D",
        adjustments: list[str] | None = None,
        field_map: dict[str, str] | None = None,
        library: ModuleType | Any | None = None,
        health_ric: str = "IBM.N",
    ):
        self.session_name = session_name
        self.config_name = config_name
        self.interval = interval
        self.adjustments = adjustments
        self.field_map = field_map or REFINITIV_OHLCV_FIELD_MAP
        self._ld = library
        self._session_open = False
        self.health_ric = health_ric
        self.last_health_error = ""

    @property
    def ld(self):
        if self._ld is None:
            try:
                import lseg.data as ld  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RefinitivAdapter requires the optional `lseg-data` package. "
                    "Install it with `uv sync --extra refinitiv`."
                ) from exc
            self._ld = ld
        return self._ld

    def open(self) -> None:
        if self._session_open:
            return
        kwargs: dict[str, str] = {}
        if self.session_name:
            kwargs["name"] = self.session_name
        if self.config_name:
            kwargs["config_name"] = self.config_name
        self.ld.open_session(**kwargs)
        self._session_open = True

    def close(self) -> None:
        if not self._session_open:
            return
        close_session = getattr(self.ld, "close_session", None)
        if close_session is not None:
            close_session()
        self._session_open = False

    def download(self, tickers: list[str], start: str, end: str, fields: list[str]) -> pd.DataFrame:
        """Download raw LSEG history for the given RICs and fields."""

        if not tickers:
            return pd.DataFrame()
        self.open()
        kwargs: dict[str, Any] = {
            "universe": list(dict.fromkeys(tickers)),
            "fields": fields,
            "start": start,
            "end": end,
            "interval": self.interval,
        }
        if self.adjustments:
            kwargs["adjustments"] = self.adjustments
        return self.ld.get_history(**kwargs)

    def _download_with_single_ticker_fallback(self, tickers: list[str], start: str, end: str, fields: list[str]) -> pd.DataFrame:
        try:
            return self.download(tickers=tickers, start=start, end=end, fields=fields)
        except Exception as exc:
            if _is_refinitiv_retryable_provider_error(exc):
                raise RefinitivRetryableProviderError(str(exc)) from exc
            if len(tickers) <= 1:
                raise

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                history = self.download(tickers=[ticker], start=start, end=end, fields=fields)
            except Exception as exc:
                if _is_refinitiv_retryable_provider_error(exc):
                    raise RefinitivRetryableProviderError(str(exc)) from exc
                continue
            if history.empty:
                continue
            frame = history.copy()
            if not isinstance(frame.columns, pd.MultiIndex):
                frame.columns = pd.MultiIndex.from_product([[ticker], [str(col) for col in frame.columns]])
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def download_price_history(
        self,
        tickers: list[str],
        period: str,
        interval: str = "1d",
        output_path: str | Path | None = None,
        fallback_periods: list[str] | None = None,
        ohlcv: bool = False,
        field_map: dict[str, str] | None = None,
    ) -> DownloadResult:
        """Backfill-compatible downloader returning yfinance-shaped frames."""

        if not tickers:
            return DownloadResult(prices=pd.DataFrame(), coverage=pd.DataFrame(), failures={}, ohlcv=pd.DataFrame() if ohlcv else None)

        previous_interval = self.interval
        self.interval = _refinitiv_interval(interval)
        periods = list(dict.fromkeys([period, *(fallback_periods or [])]))
        active_field_map = field_map or self.field_map
        history_field_map = active_field_map if ohlcv else {"Close": active_field_map["Close"]}
        fields = list(history_field_map.values())
        history = pd.DataFrame()
        period_used = period
        for candidate_period in periods:
            start, end = _period_to_start_end(candidate_period)
            history = self._download_with_single_ticker_fallback(tickers=tickers, start=start, end=end, fields=fields)
            period_used = candidate_period
            if not history.empty:
                break
        self.interval = previous_interval

        ohlcv_df = refinitiv_history_to_ohlcv(history, tickers=tickers, field_map=history_field_map)
        if ohlcv_df.empty:
            prices = pd.DataFrame()
        elif "Close" in ohlcv_df.columns.get_level_values(1):
            prices = ohlcv_df.xs("Close", axis=1, level=1)
        else:
            prices = pd.DataFrame()

        coverage, failures = _coverage_from_prices(prices, tickers=tickers, interval=interval, period_used=period_used)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(path)
            if ohlcv and not ohlcv_df.empty:
                ohlcv_df.to_parquet(path.with_suffix(".ohlcv.parquet"))
            coverage.to_csv(path.with_suffix(".coverage.csv"), index=False)

        return DownloadResult(prices=prices, coverage=coverage, failures=failures, ohlcv=ohlcv_df if ohlcv else None)

    def download_price_history_range(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        output_path: str | Path | None = None,
        ohlcv: bool = False,
        field_map: dict[str, str] | None = None,
    ) -> DownloadResult:
        """Download Refinitiv history for an explicit date range."""

        if not tickers:
            return DownloadResult(prices=pd.DataFrame(), coverage=pd.DataFrame(), failures={}, ohlcv=pd.DataFrame() if ohlcv else None)

        previous_interval = self.interval
        self.interval = _refinitiv_interval(interval)
        active_field_map = field_map or self.field_map
        history_field_map = active_field_map if ohlcv else {"Close": active_field_map["Close"]}
        fields = list(history_field_map.values())
        try:
            history = self._download_with_single_ticker_fallback(tickers=tickers, start=start, end=end, fields=fields)
        finally:
            self.interval = previous_interval

        ohlcv_df = refinitiv_history_to_ohlcv(history, tickers=tickers, field_map=history_field_map)
        if ohlcv_df.empty:
            prices = pd.DataFrame()
        elif "Close" in ohlcv_df.columns.get_level_values(1):
            prices = ohlcv_df.xs("Close", axis=1, level=1)
        else:
            prices = pd.DataFrame()

        coverage, failures = _coverage_from_prices(prices, tickers=tickers, interval=interval, period_used=f"{start}:{end}")
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(path)
            if ohlcv and not ohlcv_df.empty:
                ohlcv_df.to_parquet(path.with_suffix(".ohlcv.parquet"))
            coverage.to_csv(path.with_suffix(".coverage.csv"), index=False)

        return DownloadResult(prices=prices, coverage=coverage, failures=failures, ohlcv=ohlcv_df if ohlcv else None)

    def health_check(self) -> bool:
        self.last_health_error = ""
        try:
            self.open()
            data = self.ld.get_history(
                universe=[self.health_ric],
                fields=[self.field_map["Close"]],
                interval=self.interval,
                count=1,
            )
            return not data.empty
        except Exception as exc:
            self.last_health_error = f"{type(exc).__name__}: {exc}"
            return False


def _refinitiv_interval(interval: str) -> str:
    mapping = {
        "1d": "1D",
        "1D": "1D",
        "daily": "daily",
        "1wk": "1W",
        "1mo": "1M",
        "1h": "1h",
        "60m": "60min",
        "30m": "30min",
        "10m": "10min",
        "5m": "5min",
        "1m": "1min",
    }
    return mapping.get(interval, interval)


def _period_to_start_end(period: str) -> tuple[str, str]:
    end = pd.Timestamp.utcnow().normalize()
    normalized = str(period).strip().lower()
    if normalized == "max":
        start = pd.Timestamp("1900-01-01", tz="UTC")
    elif normalized.endswith("y"):
        start = end - pd.DateOffset(years=int(normalized[:-1]))
    elif normalized.endswith("mo"):
        start = end - pd.DateOffset(months=int(normalized[:-2]))
    elif normalized.endswith("d"):
        start = end - pd.DateOffset(days=int(normalized[:-1]))
    else:
        start = end - pd.DateOffset(years=5)
    return start.date().isoformat(), end.date().isoformat()


def _normalize_history_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index.normalize()


def refinitiv_history_to_ohlcv(
    history: pd.DataFrame,
    tickers: list[str],
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert LSEG `get_history` output to `(RIC, Open/High/Low/Close/Volume)` columns."""

    if history.empty:
        return pd.DataFrame()

    field_map = field_map or REFINITIV_OHLCV_FIELD_MAP
    field_lookup = {
        alias: key
        for key, value in field_map.items()
        for alias in _refinitiv_field_aliases(value)
    }
    ticker_set = {str(ticker) for ticker in tickers}
    output: dict[tuple[str, str], pd.Series] = {}

    frame = history.copy()
    frame.index = _normalize_history_index(frame.index)

    if isinstance(frame.columns, pd.MultiIndex):
        for col in frame.columns:
            parts = [str(part) for part in col if pd.notna(part)]
            ticker = next((part for part in parts if part in ticker_set), None)
            field = next(
                (
                    field_lookup[_normalize_refinitiv_field_name(part)]
                    for part in parts
                    if _normalize_refinitiv_field_name(part) in field_lookup
                ),
                None,
            )
            if ticker is None and len(ticker_set) == 1:
                ticker = next(iter(ticker_set))
            if ticker and field:
                output[(ticker, field)] = frame[col]
    else:
        for col in frame.columns:
            col_text = str(col)
            ticker = col_text if col_text in ticker_set else None
            field = field_lookup.get(_normalize_refinitiv_field_name(col_text))
            if ticker is None and field is not None and len(ticker_set) == 1:
                ticker = next(iter(ticker_set))
            if field is None and len(ticker_set) == 1:
                ticker = next(iter(ticker_set))
                field = field_lookup.get(_normalize_refinitiv_field_name(col_text))
            if field is None and len(field_map) == 1 and ticker is not None:
                field = next(iter(field_map))
            if ticker and field:
                output[(ticker, field)] = frame[col]

    if not output:
        return pd.DataFrame()

    result = pd.DataFrame(output, index=frame.index).sort_index()
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    ordered_cols = [
        (ticker, field)
        for ticker in tickers
        for field in ("Open", "High", "Low", "Close", "Volume")
        if (ticker, field) in result.columns
    ]
    return result.loc[:, ordered_cols]


def _coverage_from_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    interval: str,
    period_used: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    failures = {}
    for ticker in dict.fromkeys(tickers):
        series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
        rows.append(
            {
                "ticker": ticker,
                "rows": int(series.shape[0]),
                "start": series.index.min() if not series.empty else pd.NaT,
                "end": series.index.max() if not series.empty else pd.NaT,
                "interval": interval,
                "period_used": period_used if not series.empty else None,
            }
        )
        if series.empty:
            failures[ticker] = "empty history"
    coverage = pd.DataFrame(rows)
    if not coverage.empty:
        coverage = coverage.sort_values(["rows", "ticker"], ascending=[False, True])
    return coverage, failures
