from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataDepthReport:
    artifact_dir: Path
    manifest_path: Path
    overall_path: Path
    country_summary_path: Path
    ticker_summary_path: Path
    markdown_path: Path
    overall: pd.DataFrame
    country_summary: pd.DataFrame
    ticker_summary: pd.DataFrame


def load_long_ohlcv(long_ohlcv_root: str | Path) -> pd.DataFrame:
    """Load a partitioned long OHLCV lake root into a normalized frame."""

    root = Path(long_ohlcv_root)
    if not root.exists():
        raise FileNotFoundError(f"long OHLCV root does not exist: {root}")
    data = pd.read_parquet(root)
    if data.empty:
        raise ValueError(f"long OHLCV root is empty: {root}")
    if "date" not in data.columns or "ticker" not in data.columns:
        raise ValueError("long OHLCV data must contain date and ticker columns")

    normalized = data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce").dt.tz_localize(None)
    normalized = normalized.dropna(subset=["date", "ticker"])
    normalized["ticker"] = normalized["ticker"].astype("string")
    for col in ("country", "exchange", "local_symbol", "name", "currency", "provider", "interval", "adjustment", "run_id"):
        if col not in normalized.columns:
            normalized[col] = pd.NA
        normalized[col] = normalized[col].astype("string")
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col not in normalized.columns:
            normalized[col] = pd.NA
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized.reset_index(drop=True)


def build_data_depth_tables(long_ohlcv: pd.DataFrame, long_ohlcv_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall, country, and ticker coverage/quality tables for long OHLCV data."""

    data = long_ohlcv.copy()
    if data.empty:
        raise ValueError("long_ohlcv must not be empty")
    data["_volume_missing"] = data["volume"].isna()
    data["_zero_volume"] = data["volume"].fillna(0).eq(0)
    data["_bad_close"] = data["close"].isna() | data["close"].le(0)

    ticker_summary = (
        data.groupby("ticker", dropna=False)
        .agg(
            rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
            country=("country", "first"),
            exchange=("exchange", "first"),
            local_symbol=("local_symbol", "first"),
            name=("name", "first"),
            currency=("currency", "first"),
            provider=("provider", "first"),
            interval=("interval", "first"),
            adjustment=("adjustment", "first"),
            volume_missing_pct=("_volume_missing", "mean"),
            zero_volume_pct=("_zero_volume", "mean"),
            non_positive_close_rows=("_bad_close", "sum"),
        )
        .reset_index()
    )
    ticker_summary["calendar_days"] = (ticker_summary["end"] - ticker_summary["start"]).dt.days.add(1).clip(lower=1)
    ticker_summary["calendar_coverage"] = ticker_summary["rows"] / ticker_summary["calendar_days"]
    ticker_summary = ticker_summary.sort_values(["country", "exchange", "ticker"], na_position="last").reset_index(drop=True)

    country_from_tickers = (
        ticker_summary.groupby("country", dropna=False)
        .agg(
            tickers=("ticker", "nunique"),
            exchanges=("exchange", "nunique"),
            rows=("rows", "sum"),
            start=("start", "min"),
            end=("end", "max"),
            min_ticker_rows=("rows", "min"),
            median_ticker_rows=("rows", "median"),
            max_ticker_rows=("rows", "max"),
            median_calendar_coverage=("calendar_coverage", "median"),
        )
        .reset_index()
    )
    country_quality = (
        data.groupby("country", dropna=False)
        .agg(
            volume_missing_pct=("_volume_missing", "mean"),
            zero_volume_pct=("_zero_volume", "mean"),
            non_positive_close_rows=("_bad_close", "sum"),
        )
        .reset_index()
    )
    country_summary = country_from_tickers.merge(country_quality, on="country", how="left")
    country_summary = country_summary.sort_values(["rows", "tickers"], ascending=[False, False]).reset_index(drop=True)

    parquet_files = len(list(Path(long_ohlcv_root).glob("**/*.parquet")))
    overall = pd.DataFrame(
        [
            {
                "long_ohlcv_root": str(long_ohlcv_root),
                "rows": int(data.shape[0]),
                "tickers": int(data["ticker"].nunique(dropna=True)),
                "countries": int(data["country"].nunique(dropna=True)),
                "exchanges": int(data["exchange"].nunique(dropna=True)),
                "start": data["date"].min(),
                "end": data["date"].max(),
                "parquet_files": int(parquet_files),
                "min_ticker_rows": int(ticker_summary["rows"].min()),
                "median_ticker_rows": float(ticker_summary["rows"].median()),
                "max_ticker_rows": int(ticker_summary["rows"].max()),
                "volume_missing_pct": float(data["_volume_missing"].mean()),
                "zero_volume_pct": float(data["_zero_volume"].mean()),
                "non_positive_close_rows": int(data["_bad_close"].sum()),
            }
        ]
    )
    return overall, country_summary, ticker_summary


def write_data_depth_report(
    long_ohlcv_root: str | Path,
    artifact_root: str | Path = "artifacts/data_depth",
    run_id: str | None = None,
    extraction_manifest_path: str | Path | None = None,
) -> DataDepthReport:
    """Write CSV, JSON, and Markdown data-depth diagnostics for a long OHLCV root."""

    run_id = run_id or datetime.now(timezone.utc).strftime("DATA-DEPTH-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_ohlcv(long_ohlcv_root)
    overall, country_summary, ticker_summary = build_data_depth_tables(data, long_ohlcv_root=long_ohlcv_root)

    overall_path = artifact_dir / "overall.csv"
    country_summary_path = artifact_dir / "country_summary.csv"
    ticker_summary_path = artifact_dir / "ticker_summary.csv"
    markdown_path = artifact_dir / "data_depth_summary.md"
    manifest_path = artifact_dir / "data_depth_manifest.json"

    overall.to_csv(overall_path, index=False)
    country_summary.to_csv(country_summary_path, index=False)
    ticker_summary.to_csv(ticker_summary_path, index=False)
    _write_markdown_summary(markdown_path, overall, country_summary, ticker_summary)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "long_ohlcv_root": str(long_ohlcv_root),
        "extraction_manifest_path": str(extraction_manifest_path) if extraction_manifest_path else None,
        "overall_path": str(overall_path),
        "country_summary_path": str(country_summary_path),
        "ticker_summary_path": str(ticker_summary_path),
        "markdown_path": str(markdown_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return DataDepthReport(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        overall_path=overall_path,
        country_summary_path=country_summary_path,
        ticker_summary_path=ticker_summary_path,
        markdown_path=markdown_path,
        overall=overall,
        country_summary=country_summary,
        ticker_summary=ticker_summary,
    )


def _markdown_table(data: pd.DataFrame, max_rows: int = 20) -> str:
    if data.empty:
        return "_No rows._"
    display = data.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].dt.date.astype(str)
    columns = [str(col) for col in display.columns]
    rows = []
    for record in display.astype("object").where(pd.notna(display), "").to_dict("records"):
        rows.append([_markdown_cell(record[col]) for col in display.columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _write_markdown_summary(
    path: Path,
    overall: pd.DataFrame,
    country_summary: pd.DataFrame,
    ticker_summary: pd.DataFrame,
) -> None:
    overall_row = overall.iloc[0].to_dict()
    weakest = ticker_summary.sort_values(["rows", "calendar_coverage"], ascending=[True, True]).head(25)
    lines = [
        f"# Data Depth Summary - {path.parent.name}",
        "",
        "## Overall",
        "",
        f"- Rows: {int(overall_row['rows']):,}",
        f"- Tickers: {int(overall_row['tickers']):,}",
        f"- Countries: {int(overall_row['countries']):,}",
        f"- Exchanges: {int(overall_row['exchanges']):,}",
        f"- Date range: {pd.Timestamp(overall_row['start']).date()} to {pd.Timestamp(overall_row['end']).date()}",
        f"- Median rows per ticker: {float(overall_row['median_ticker_rows']):,.1f}",
        f"- Volume missing: {float(overall_row['volume_missing_pct']):.2%}",
        f"- Zero volume: {float(overall_row['zero_volume_pct']):.2%}",
        f"- Non-positive close rows: {int(overall_row['non_positive_close_rows']):,}",
        "",
        "## Country Summary",
        "",
        _markdown_table(country_summary),
        "",
        "## Weakest Ticker Coverage",
        "",
        _markdown_table(
            weakest[
                [
                    "ticker",
                    "country",
                    "exchange",
                    "rows",
                    "start",
                    "end",
                    "calendar_coverage",
                    "volume_missing_pct",
                    "zero_volume_pct",
                ]
            ]
        ),
        "",
        "## Readiness",
        "",
        "- Ready for first daily OHLCV backtest sweep when each target market has enough rows per ticker for the lookback windows under test.",
        "- Not yet point-in-time fundamentals or analyst-revision ready; those require dated event tables keyed to the same security master.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
