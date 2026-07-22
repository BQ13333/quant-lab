from __future__ import annotations

import os
import re
import shutil
from hashlib import blake2b
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from quant_lab.data.yfinance_loader import DownloadResult, download_price_history


GOOGLE_DRIVE_CANDIDATES = (
    "~/Library/CloudStorage/GoogleDrive-*/My Drive",
    "~/Library/CloudStorage/GoogleDrive*/My Drive",
    "~/Google Drive",
    "~/My Drive",
)

LONG_OHLCV_STRING_COLUMNS = (
    "ticker",
    "sid",
    "local_symbol",
    "name",
    "country",
    "exchange",
    "currency",
    "data_tier",
    "provider",
    "interval",
    "adjustment",
    "run_id",
    "batch_id",
    "fetched_at_utc",
    "isin",
)

LONG_OHLCV_NUMERIC_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class BatchBackfillResult:
    run_id: str
    root: Path
    manifest: pd.DataFrame
    manifest_path: Path
    long_ohlcv_root: Path | None = None


def default_data_lake_root(project_root: str | Path = ".") -> Path:
    """Resolve a storage root without assuming Google Drive is mounted."""

    env_root = os.environ.get("QUANT_LAB_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    for pattern in GOOGLE_DRIVE_CANDIDATES:
        matches = sorted(Path().home().glob(pattern.replace("~/", "")))
        if matches:
            return matches[0] / "QuantLabDataLake"

    return Path(project_root) / "artifacts" / "data_lake"


def _slug(value: object) -> str:
    original = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", original)
    slug = text.strip("_") or "unknown"
    if re.search(r"[^a-z0-9\s_-]", original):
        digest = blake2b(original.encode("utf-8"), digest_size=4).hexdigest()
        slug = f"{slug}_{digest}"
    return slug


def write_universe_partitions(
    listings: pd.DataFrame,
    root: str | Path,
    universe_id: str,
    partition_cols: tuple[str, ...] = ("country", "exchange"),
    sanitize_partitions: bool = True,
    overwrite: bool = False,
) -> Path:
    """Write listings as partitioned Parquet for scan-friendly storage."""

    if listings.empty:
        raise ValueError("listings must not be empty")
    path = Path(root) / "universes" / universe_id / "listings"
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    data = listings.copy()
    physical_partition_cols = []
    for col in partition_cols:
        if col not in data.columns:
            continue
        if sanitize_partitions:
            partition_col = f"{col}_partition"
            data[partition_col] = data[col].map(_slug)
            physical_partition_cols.append(partition_col)
        else:
            physical_partition_cols.append(col)
    if not physical_partition_cols:
        raise ValueError("no partition columns are present in listings")

    data.to_parquet(path, index=False, partition_cols=physical_partition_cols)
    data.head(0).to_csv(path.parent / "schema.csv", index=False)
    pd.DataFrame(
        {
            "logical_column": [col for col in partition_cols if col in listings.columns],
            "physical_partition_column": physical_partition_cols,
        }
    ).to_csv(path.parent / "partitions.csv", index=False)
    return path


def _empty_long_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "sid",
            "local_symbol",
            "name",
            "country",
            "exchange",
            "currency",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "data_tier",
            "provider",
            "interval",
            "adjustment",
            "run_id",
            "batch_id",
            "fetched_at_utc",
        ]
    )


def _ohlcv_field_name(field: object) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(field).strip().lower()).strip("_")
    aliases = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
        "volume": "volume",
    }
    return aliases.get(normalized)


def _normalize_long_ohlcv_schema(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce").dt.tz_localize(None)
    for col in LONG_OHLCV_STRING_COLUMNS:
        if col in normalized.columns:
            normalized[col] = normalized[col].astype("string")
    for col in LONG_OHLCV_NUMERIC_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("float64")
    return normalized


def _listing_metadata(listings: pd.DataFrame, ticker_column: str) -> pd.DataFrame:
    if listings.empty:
        return pd.DataFrame()

    data = listings.copy()
    join_col = ticker_column if ticker_column in data.columns else "ticker"
    if join_col not in data.columns:
        return pd.DataFrame()

    metadata = pd.DataFrame({"_join_ticker": data[join_col].astype("string")})
    if "ticker" in data.columns:
        metadata["local_symbol"] = data["ticker"]
    elif "symbol" in data.columns:
        metadata["local_symbol"] = data["symbol"]
    else:
        metadata["local_symbol"] = pd.NA

    for col in ("sid", "name", "country", "exchange", "currency", "isin"):
        if col in data.columns:
            metadata[col] = data[col]

    if "sid" not in metadata.columns:
        metadata["sid"] = pd.NA

    if "name" not in metadata.columns:
        metadata["name"] = pd.NA

    for col in ("country", "exchange", "currency", "isin"):
        if col not in metadata.columns:
            metadata[col] = pd.NA

    metadata = metadata.dropna(subset=["_join_ticker"]).drop_duplicates("_join_ticker", keep="first")
    return metadata


def wide_ohlcv_to_long(
    ohlcv: pd.DataFrame,
    listings: pd.DataFrame | None = None,
    provider: str = "yahoo",
    interval: str = "1d",
    adjustment: str = "auto_adjusted",
    data_tier: str = "bronze",
    run_id: str | None = None,
    batch_id: str | None = None,
    fetched_at_utc: str | None = None,
    ticker_column: str = "yahoo_ticker",
) -> pd.DataFrame:
    """Convert a wide MultiIndex OHLCV frame into an auditable narrow table."""

    if ohlcv.empty:
        return _empty_long_ohlcv_frame()
    if not isinstance(ohlcv.columns, pd.MultiIndex) or ohlcv.columns.nlevels < 2:
        raise ValueError("ohlcv columns must be a MultiIndex of (ticker, field)")

    fetched_at_utc = fetched_at_utc or datetime.now(timezone.utc).isoformat()
    source = ohlcv.copy()
    source.index = pd.to_datetime(source.index, utc=True, errors="coerce").tz_localize(None)
    source.index.name = "date"

    frames: list[pd.DataFrame] = []
    for ticker in sorted({str(col[0]) for col in source.columns}):
        ticker_cols = [col for col in source.columns if str(col[0]) == ticker]
        frame = source.loc[:, ticker_cols].copy()
        renamed_cols = [_ohlcv_field_name(col[1]) for col in ticker_cols]
        keep = [idx for idx, col in enumerate(renamed_cols) if col is not None]
        if not keep:
            continue
        frame = frame.iloc[:, keep]
        frame.columns = [renamed_cols[idx] for idx in keep]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame = frame.reset_index()
        frame.insert(1, "ticker", ticker)
        frames.append(frame)

    if not frames:
        return _empty_long_ohlcv_frame()

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], utc=True, errors="coerce").dt.tz_localize(None)
    long = long.dropna(subset=["date", "ticker"])

    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col not in long.columns:
            long[col] = pd.NA
        long[col] = pd.to_numeric(long[col], errors="coerce")
    if long["adj_close"].isna().all() and "close" in long.columns:
        long["adj_close"] = long["close"]
    long = long.dropna(subset=["close"])

    if listings is not None and not listings.empty:
        metadata = _listing_metadata(listings, ticker_column=ticker_column)
        if not metadata.empty:
            long = long.merge(metadata, how="left", left_on="ticker", right_on="_join_ticker").drop(columns=["_join_ticker"])

    for col in ("sid", "local_symbol", "name", "country", "exchange", "currency"):
        if col not in long.columns:
            long[col] = pd.NA

    missing_sid = long["sid"].isna() | long["sid"].astype("string").str.len().fillna(0).eq(0)
    if missing_sid.any():
        sid_source = long.loc[missing_sid, ["country", "exchange", "local_symbol", "ticker"]].fillna("")
        long.loc[missing_sid, "sid"] = sid_source.apply(
            lambda row: ":".join([str(row["country"]), str(row["exchange"]), str(row["local_symbol"] or row["ticker"])]).strip(":"),
            axis=1,
        )

    long["data_tier"] = data_tier
    long["provider"] = provider
    long["interval"] = interval
    long["adjustment"] = adjustment
    long["run_id"] = run_id
    long["batch_id"] = batch_id
    long["fetched_at_utc"] = fetched_at_utc

    ordered_cols = list(_empty_long_ohlcv_frame().columns)
    extra_cols = [col for col in long.columns if col not in ordered_cols and col != "isin"]
    if "isin" in long.columns:
        extra_cols.append("isin")
    long = long[ordered_cols + extra_cols].sort_values(["date", "ticker"]).reset_index(drop=True)
    return long


def write_ohlcv_long_partitions(
    ohlcv_long: pd.DataFrame,
    root: str | Path,
    run_id: str,
    partition_cols: tuple[str, ...] = ("provider", "interval", "adjustment", "country", "exchange", "year", "month"),
    overwrite: bool = False,
) -> Path:
    """Append a long OHLCV table into scan-friendly Parquet partitions."""

    if ohlcv_long.empty:
        raise ValueError("ohlcv_long must not be empty")

    path = Path(root) / "prices_long" / run_id / "ohlcv"
    if path.exists() and any(path.iterdir()) and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    data = _normalize_long_ohlcv_schema(ohlcv_long)
    data = data.dropna(subset=["date", "ticker"])
    data["year"] = data["date"].dt.year.astype("Int64")
    data["month"] = data["date"].dt.month.astype("Int64")

    physical_partition_cols: list[str] = []
    for col in partition_cols:
        if col not in data.columns:
            continue
        partition_col = f"{col}_partition"
        if col in {"year", "month"}:
            data[partition_col] = data[col].astype("string").fillna("unknown")
        else:
            data[partition_col] = data[col].map(_slug)
        physical_partition_cols.append(partition_col)

    if not physical_partition_cols:
        raise ValueError("no partition columns are present in ohlcv_long")

    data.to_parquet(path, index=False, partition_cols=physical_partition_cols)
    data.head(0).to_csv(path.parent / "ohlcv_schema.csv", index=False)
    pd.DataFrame(
        {
            "logical_column": [col for col in partition_cols if col in data.columns],
            "physical_partition_column": physical_partition_cols,
        }
    ).to_csv(path.parent / "ohlcv_partitions.csv", index=False)
    return path


def iter_listing_batches(
    listings: pd.DataFrame,
    batch_size: int = 100,
    group_cols: tuple[str, ...] = ("country", "exchange"),
) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    """Yield bounded listing batches grouped by country/exchange."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if listings.empty:
        return

    group_cols = tuple(col for col in group_cols if col in listings.columns)
    if not group_cols:
        groups = [((), listings)]
    else:
        groups = listings.groupby(list(group_cols), dropna=False, sort=True)

    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        meta = {col: value for col, value in zip(group_cols, key)}
        group = group.reset_index(drop=True)
        for batch_no, start in enumerate(range(0, len(group), batch_size), start=1):
            batch = group.iloc[start:start + batch_size].copy()
            yield {**meta, "batch_no": batch_no, "rows": len(batch)}, batch


def backfill_prices_in_batches(
    listings: pd.DataFrame,
    root: str | Path,
    run_id: str | None = None,
    period: str = "5y",
    interval: str = "1d",
    batch_size: int = 100,
    max_tickers: int | None = None,
    group_cols: tuple[str, ...] = ("country", "exchange"),
    downloader: Callable[..., DownloadResult] = download_price_history,
    provider: str = "yahoo",
    adjustment: str = "auto_adjusted",
    data_tier: str = "bronze",
    write_long_ohlcv: bool = False,
    long_partition_cols: tuple[str, ...] = ("provider", "interval", "adjustment", "country", "exchange", "year", "month"),
    download_kwargs_by_batch: Callable[[dict[str, object], pd.DataFrame], dict[str, object]] | None = None,
) -> BatchBackfillResult:
    """Download OHLCV in bounded batches and write wide and optional long Parquet outputs."""

    if listings.empty:
        raise ValueError("listings must not be empty")
    run_id = run_id or datetime.now(timezone.utc).strftime("PRICE-BACKFILL-%Y%m%d-%H%M%S")
    root_path = Path(root)
    batch_root = root_path / "prices" / run_id
    batch_root.mkdir(parents=True, exist_ok=True)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    long_ohlcv_root = root_path / "prices_long" / run_id / "ohlcv" if write_long_ohlcv else None

    remaining = max_tickers
    rows = []
    for meta, batch in iter_listing_batches(listings, batch_size=batch_size, group_cols=group_cols):
        if remaining is not None:
            if remaining <= 0:
                break
            batch = batch.head(remaining)
            remaining -= len(batch)
        if batch.empty:
            continue

        country = _slug(meta.get("country", "unknown"))
        exchange = _slug(meta.get("exchange", "unknown"))
        batch_no = int(meta["batch_no"])
        output_path = batch_root / f"country={country}" / f"exchange={exchange}" / f"batch={batch_no:04d}" / "prices.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ticker_col = "yahoo_ticker" if "yahoo_ticker" in batch.columns else "ticker"
        tickers = batch[ticker_col].dropna().astype(str).tolist()
        extra_download_kwargs = download_kwargs_by_batch(meta, batch) if download_kwargs_by_batch else {}
        result = downloader(
            tickers=tickers,
            period=period,
            interval=interval,
            output_path=output_path,
            fallback_periods=["3y", "1y", "6mo"],
            ohlcv=True,
            **extra_download_kwargs,
        )
        long_ohlcv_path = None
        long_ohlcv_rows = 0
        if write_long_ohlcv and result.ohlcv is not None and not result.ohlcv.empty:
            batch_id = f"{country}-{exchange}-{batch_no:04d}"
            long_ohlcv = wide_ohlcv_to_long(
                result.ohlcv,
                listings=batch,
                provider=provider,
                interval=interval,
                adjustment=adjustment,
                data_tier=data_tier,
                run_id=run_id,
                batch_id=batch_id,
                fetched_at_utc=fetched_at_utc,
            )
            if not long_ohlcv.empty:
                long_ohlcv_path = write_ohlcv_long_partitions(
                    long_ohlcv,
                    root=root_path,
                    run_id=run_id,
                    partition_cols=long_partition_cols,
                )
                long_ohlcv_rows = int(long_ohlcv.shape[0])

        rows.append(
            {
                "run_id": run_id,
                "provider": provider,
                "interval": interval,
                "adjustment": adjustment,
                "data_tier": data_tier,
                "country": meta.get("country"),
                "exchange": meta.get("exchange"),
                "batch_no": batch_no,
                "requested_tickers": len(tickers),
                "price_rows": int(result.prices.shape[0]),
                "price_columns": int(result.prices.shape[1]),
                "failures": len(result.failures),
                "output_path": str(output_path),
                "coverage_path": str(output_path.with_suffix(".coverage.csv")),
                "ohlcv_path": str(output_path.with_suffix(".ohlcv.parquet")),
                "long_ohlcv_path": str(long_ohlcv_path) if long_ohlcv_path else None,
                "long_ohlcv_rows": long_ohlcv_rows,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = batch_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return BatchBackfillResult(
        run_id=run_id,
        root=batch_root,
        manifest=manifest,
        manifest_path=manifest_path,
        long_ohlcv_root=long_ohlcv_root,
    )
