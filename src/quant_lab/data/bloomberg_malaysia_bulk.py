from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


SOURCE_COLUMN_MAP: dict[str, str] = {
    "ID_ISIN": "isin",
    "ID_BBG": "bbg_ticker",
    "NAME": "name",
    "EXCH_MKT_GRP": "exchange_market_group",
    "FINANCIAL_STATUS_INDICATOR": "financial_status",
    "SECURITY_TYP": "security_type",
    "GICS_SUB_INDUSTRY_NAME": "gics_sub_industry",
    "IS_ISLAMIC": "is_islamic",
    "IS_ISLAMIC_1": "is_islamic_vendor_lag1",
    "MARKET_STATUS": "market_status",
    "ID": "bbg_id",
    "DATE": "date",
    "PX_LAST": "adj_close",
    "PX_LAST_1": "adj_close_vendor_lag1",
    "PX_OPEN": "adj_open",
    "PX_OPEN_1": "adj_open_vendor_lag1",
    "PX_HIGH": "adj_high",
    "PX_HIGH_1": "adj_high_vendor_lag1",
    "PX_LOW": "adj_low",
    "PX_LOW_1": "adj_low_vendor_lag1",
    "PX_VOLUME": "volume",
    "PX_VOLUME_1": "volume_vendor_lag1",
    "PX_LAST_RAW": "raw_close",
    "PX_OPEN_RAW": "raw_open",
    "PX_HIGH_RAW": "raw_high",
    "PX_LOW_RAW": "raw_low",
    "PX_VOLUME_RAW": "raw_volume",
    "CASH_DIVS": "cash_dividend_adjusted",
    "CASH_DIVS_1": "cash_dividend_adjusted_vendor_lag1",
    "CASH_DIVS_RAW": "cash_dividend_raw",
    "TURNOVER": "turnover",
    "VWAP": "vwap_adjusted",
    "VWAP_1": "vwap_adjusted_vendor_lag1",
    "VWAP_RAW": "vwap_raw",
    "CUR_MKT_CAP": "market_cap",
    "CUR_MKT_CAP_1": "market_cap_vendor_lag1",
    "CF_FREE_CASH_FLOW_LTM": "free_cash_flow_ltm",
    "CF_FREE_CASH_FLOW_LTM_1": "free_cash_flow_ltm_vendor_lag1",
    "LTM_NET_INCOME": "net_income_ltm",
    "LTM_NET_INCOME_1": "net_income_ltm_vendor_lag1",
    "FWD_0_12_NET_INCOME": "forward_net_income_0_12m",
    "FWD_0_12_NET_INCOME_1": "forward_net_income_0_12m_vendor_lag1",
    "FWD_12_24_NET_INCOME": "forward_net_income_12_24m",
    "FWD_12_24_NET_INCOME_1": "forward_net_income_12_24m_vendor_lag1",
    "BS_TOT_ASSET": "total_assets",
    "BS_TOT_ASSET_1": "total_assets_vendor_lag1",
    "BS_TOT_LIAB2": "total_liabilities",
    "BS_TOT_LIAB2_1": "total_liabilities_vendor_lag1",
    "EQY_REC_CONS": "analyst_recommendation_consensus",
    "EQY_REC_CONS_1": "analyst_recommendation_consensus_vendor_lag1",
    "BEST_TARGET_PRICE": "analyst_target_price",
    "BEST_TARGET_PRICE_1": "analyst_target_price_vendor_lag1",
}

TEXT_COLUMNS = (
    "isin",
    "bbg_ticker",
    "name",
    "exchange_market_group",
    "financial_status",
    "security_type",
    "gics_sub_industry",
    "market_status",
    "bbg_id",
)

NUMERIC_COLUMNS = tuple(
    column
    for column in SOURCE_COLUMN_MAP.values()
    if column not in {*TEXT_COLUMNS, "date"}
)

NORMALIZED_COLUMNS = tuple(
    dict.fromkeys(
        (
            "source_period",
            "source_filename",
            "source_row_number",
            "date",
            "security_key",
            "ticker",
            "sid",
            "local_symbol",
            "country",
            "country_code",
            "exchange",
            "currency",
            "provider",
            "interval",
            "adjustment",
            "tradable_quality_score",
            *SOURCE_COLUMN_MAP.values(),
        )
    )
)

MARKET_COLUMNS = (
    "source_period",
    "source_filename",
    "source_row_number",
    "date",
    "security_key",
    "ticker",
    "bbg_ticker",
    "sid",
    "local_symbol",
    "name",
    "isin",
    "bbg_id",
    "country",
    "country_code",
    "exchange",
    "exchange_market_group",
    "currency",
    "provider",
    "interval",
    "adjustment",
    "financial_status",
    "security_type",
    "gics_sub_industry",
    "market_status",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "volume",
    "raw_volume",
    "cash_dividend_adjusted",
    "cash_dividend_raw",
    "turnover",
    "vwap_adjusted",
    "vwap_raw",
    "market_cap",
    "tradable_quality_score",
)

FUNDAMENTAL_COLUMNS = (
    "free_cash_flow_ltm",
    "net_income_ltm",
    "forward_net_income_0_12m",
    "forward_net_income_12_24m",
    "total_assets",
    "total_liabilities",
    "analyst_recommendation_consensus",
    "analyst_target_price",
)


@dataclass(frozen=True)
class BloombergBulkFileSpec:
    period: str
    filename: str
    drive_file_id: str
    expected_bytes: int

    @property
    def download_url(self) -> str:
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={self.drive_file_id}&export=download&confirm=t"
        )


@dataclass(frozen=True)
class BloombergMalaysiaBulkArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    admission_report_path: Path
    raw_manifest_path: Path
    security_master_path: Path
    identifier_history_path: Path
    listing_windows_path: Path
    membership_candidate_path: Path
    normalized_paths: tuple[Path, ...]
    market_paths: tuple[Path, ...]
    return_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_config(path: str | Path) -> tuple[dict[str, Any], tuple[BloombergBulkFileSpec, ...]]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("country") != "Malaysia" or config.get("country_code") != "MYS":
        raise ValueError("Bloomberg bulk source configuration must remain Malaysia-only")
    if config.get("research_controls", {}).get("strategy_execution_authorized") is not False:
        raise ValueError("source configuration must not authorize strategy execution")
    specs = tuple(BloombergBulkFileSpec(**item) for item in config.get("files", []))
    if len(specs) != 8 or len({item.filename for item in specs}) != 8:
        raise ValueError("Bloomberg Malaysia bulk source must contain exactly eight unique files")
    if not all(item.expected_bytes > 0 and item.drive_file_id for item in specs):
        raise ValueError("every source file requires an id and expected byte count")
    return config, specs


def download_source_file(
    spec: BloombergBulkFileSpec,
    destination: str | Path,
    *,
    chunk_bytes: int = 4 * 1024 * 1024,
    timeout_seconds: int = 180,
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    current_bytes = output.stat().st_size if output.exists() else 0
    if current_bytes == spec.expected_bytes:
        return output
    if current_bytes > spec.expected_bytes:
        raise ValueError(f"existing source file exceeds expected size: {output}")

    headers = {"User-Agent": "Mozilla/5.0"}
    if current_bytes:
        headers["Range"] = f"bytes={current_bytes}-"
    with requests.get(
        spec.download_url,
        headers=headers,
        stream=True,
        timeout=(30, timeout_seconds),
    ) as response:
        response.raise_for_status()
        if current_bytes and response.status_code != 206:
            raise RuntimeError(f"source server did not honor resume range for {spec.filename}")
        mode = "ab" if current_bytes else "wb"
        with output.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    handle.write(chunk)
    observed = output.stat().st_size
    if observed != spec.expected_bytes:
        raise RuntimeError(
            f"downloaded byte count mismatch for {spec.filename}: {observed} != {spec.expected_bytes}"
        )
    return output


def download_source_files(
    specs: Iterable[BloombergBulkFileSpec],
    raw_dir: str | Path,
    *,
    max_workers: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    items = tuple(specs)
    outputs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(download_source_file, spec, root / spec.filename): spec
            for spec in items
        }
        for future in as_completed(futures):
            spec = futures[future]
            outputs[spec.period] = future.result()
            if progress:
                progress(f"downloaded {spec.filename}")
    return outputs


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin({"", "none", "nan", "null", "n/a"}))


def _security_key(frame: pd.DataFrame) -> pd.Series:
    key = pd.Series(pd.NA, index=frame.index, dtype="string")
    bbg_ticker = frame["bbg_ticker"]
    bbg_id = frame["bbg_id"]
    isin = frame["isin"]
    key.loc[bbg_ticker.notna()] = "bbg_ticker:" + bbg_ticker.loc[bbg_ticker.notna()]
    missing = key.isna() & bbg_id.notna()
    key.loc[missing] = "bbg_id:" + bbg_id.loc[missing]
    missing = key.isna() & isin.notna()
    key.loc[missing] = "isin:" + isin.loc[missing]
    return key


def _normalized_chunk(
    raw: pd.DataFrame,
    *,
    spec: BloombergBulkFileSpec,
    source_row_start: int,
) -> pd.DataFrame:
    data = raw.rename(columns=SOURCE_COLUMN_MAP).copy()
    for column in SOURCE_COLUMN_MAP.values():
        if column not in data.columns:
            data[column] = pd.NA
    for column in TEXT_COLUMNS:
        data[column] = _clean_text(data[column])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")

    data["source_period"] = spec.period
    data["source_filename"] = spec.filename
    data["source_row_number"] = np.arange(
        source_row_start,
        source_row_start + len(data),
        dtype=np.int64,
    )
    data["security_key"] = _security_key(data)
    data["ticker"] = data["bbg_ticker"]
    data["sid"] = data["security_key"]
    data["local_symbol"] = data["bbg_ticker"].str.replace(
        r"\s+MK\s+Equity$", "", regex=True
    )
    data["country"] = "Malaysia"
    data["country_code"] = "MYS"
    data["exchange"] = "Bursa Malaysia"
    data["currency"] = "MYR"
    data["provider"] = "bloomberg_friend_bulk"
    data["interval"] = "1d"
    data["adjustment"] = "vendor_adjusted_with_raw_parallel"
    data["tradable_quality_score"] = (
        data["adj_close"].notna().astype("int16") * 100
        + data["raw_close"].notna().astype("int16") * 50
        + data["isin"].notna().astype("int16") * 10
        + data["market_cap"].notna().astype("int16") * 5
        + data["volume"].notna().astype("int16")
    ).astype("int16")
    return data.loc[:, NORMALIZED_COLUMNS]


def _write_parquet_batches(
    frames: Iterable[pd.DataFrame],
    output_path: Path,
    *,
    compression: str = "zstd",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame().to_parquet(output_path, index=False)
    return rows


def _normalization_frames(
    raw_path: Path,
    spec: BloombergBulkFileSpec,
    *,
    chunk_rows: int,
) -> Iterable[pd.DataFrame]:
    source_row = 1
    for chunk in pd.read_csv(raw_path, dtype=str, chunksize=chunk_rows, low_memory=False):
        normalized = _normalized_chunk(chunk, spec=spec, source_row_start=source_row)
        source_row += len(normalized)
        yield normalized


def _security_summary_and_identifiers(
    deduplicated: pd.DataFrame,
    *,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = deduplicated.dropna(subset=["security_key", "date"]).copy()
    priced = data.loc[data["adj_close"].gt(0.0)].copy()
    base = (
        data.groupby("security_key", sort=False)
        .agg(
            first_observation_date=("date", "min"),
            last_observation_date=("date", "max"),
            observation_rows=("date", "size"),
        )
        .reset_index()
    )
    price_summary = (
        priced.groupby("security_key", sort=False)
        .agg(
            first_valid_price_date=("date", "min"),
            last_valid_price_date=("date", "max"),
            priced_rows=("date", "size"),
        )
        .reset_index()
    )
    base = base.merge(price_summary, on="security_key", how="left")
    first_price = base.set_index("security_key")["first_valid_price_date"]
    mapped_first = data["security_key"].map(first_price)
    placeholder = data["date"].lt(mapped_first) & data["adj_close"].isna()
    placeholder_count = (
        data.loc[placeholder]
        .groupby("security_key")
        .size()
        .rename("pre_price_placeholder_rows")
    )
    base = base.merge(placeholder_count, on="security_key", how="left")
    base["pre_price_placeholder_rows"] = base["pre_price_placeholder_rows"].fillna(0).astype(int)

    final = (
        data.sort_values(["security_key", "date", "source_row_number"])
        .groupby("security_key", sort=False)
        .tail(1)
        [[
            "security_key",
            "ticker",
            "bbg_id",
            "isin",
            "name",
            "security_type",
            "exchange_market_group",
            "financial_status",
            "market_status",
        ]]
    )
    base = base.merge(final, on="security_key", how="left")
    base["source_period"] = period
    base["priced_in_period"] = base["priced_rows"].fillna(0).gt(0)

    histories: list[pd.DataFrame] = []
    for id_type, column in (
        ("isin", "isin"),
        ("bbg_ticker", "bbg_ticker"),
        ("bbg_id", "bbg_id"),
    ):
        subset = data.dropna(subset=[column])
        grouped = (
            subset.groupby(["security_key", column], sort=False)["date"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={column: "id_value", "min": "effective_from", "max": "effective_to"})
        )
        grouped["id_type"] = id_type
        grouped["source_period"] = period
        histories.append(grouped)
    return base, pd.concat(histories, ignore_index=True)


def normalize_and_curate_source_file(
    spec: BloombergBulkFileSpec,
    raw_path: str | Path,
    normalized_path: str | Path,
    market_path: str | Path,
    *,
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw = Path(raw_path)
    normalized = Path(normalized_path)
    market_output = Path(market_path)
    if not raw.is_file():
        raise FileNotFoundError(raw)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    market_output.parent.mkdir(parents=True, exist_ok=True)
    for output in (normalized, market_output):
        if output.exists():
            output.unlink()

    raw_rows = _write_parquet_batches(
        _normalization_frames(raw, spec, chunk_rows=chunk_rows),
        normalized,
    )
    key_columns = ["date", "security_key", "tradable_quality_score", "source_row_number"]
    keys = pd.read_parquet(normalized, columns=key_columns)
    valid_keys = keys["date"].notna() & keys["security_key"].notna()
    duplicate_rows = int(keys.loc[valid_keys].duplicated(["date", "security_key"], keep=False).sum())
    duplicate_keys = int(
        keys.loc[valid_keys]
        .groupby(["date", "security_key"], sort=False)
        .size()
        .gt(1)
        .sum()
    )
    ranked = keys.loc[valid_keys].sort_values(
        ["date", "security_key", "tradable_quality_score", "source_row_number"]
    )
    keep_rows = set(
        ranked.drop_duplicates(["date", "security_key"], keep="last")["source_row_number"]
        .astype(int)
        .tolist()
    )
    market_columns = list(dict.fromkeys(MARKET_COLUMNS))
    deduplicated = pd.read_parquet(normalized, columns=market_columns)
    deduplicated = deduplicated.loc[
        deduplicated["source_row_number"].astype(int).isin(keep_rows)
    ].copy()
    research_market = deduplicated.loc[
        deduplicated["date"].notna()
        & deduplicated["security_key"].notna()
        & deduplicated["adj_close"].gt(0.0)
    ].copy()
    research_market["research_market_eligible"] = True
    research_market["fundamentals_pit_admitted"] = False
    research_market["terminal_economics_admitted"] = False
    research_market.to_parquet(market_output, index=False, compression="zstd")

    summary, identifiers = _security_summary_and_identifiers(deduplicated, period=spec.period)
    invalid_ohlc = (
        research_market["adj_high"].lt(research_market[["adj_open", "adj_low", "adj_close"]].max(axis=1))
        | research_market["adj_low"].gt(research_market[["adj_open", "adj_high", "adj_close"]].min(axis=1))
    )
    profile = {
        "period": spec.period,
        "filename": spec.filename,
        "raw_rows": int(raw_rows),
        "deduplicated_rows": int(len(deduplicated)),
        "research_market_rows": int(len(research_market)),
        "distinct_security_keys": int(deduplicated["security_key"].nunique()),
        "duplicate_security_date_rows": duplicate_rows,
        "duplicate_security_date_keys": duplicate_keys,
        "unusable_missing_security_or_date_rows": int((~valid_keys).sum()),
        "missing_price_rows_after_deduplication": int(deduplicated["adj_close"].isna().sum()),
        "negative_price_rows": int(research_market["adj_close"].lt(0.0).sum()),
        "negative_volume_rows": int(research_market["volume"].lt(0.0).sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.fillna(False).sum()),
        "date_min": str(deduplicated["date"].min().date()),
        "date_max": str(deduplicated["date"].max().date()),
        "pre_price_placeholder_rows": int(summary["pre_price_placeholder_rows"].sum()),
    }
    return profile, summary, identifiers


def _combine_security_summaries(summaries: Iterable[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(tuple(summaries), ignore_index=True)
    aggregate = (
        data.groupby("security_key", sort=False)
        .agg(
            first_observation_date=("first_observation_date", "min"),
            last_observation_date=("last_observation_date", "max"),
            first_valid_price_date=("first_valid_price_date", "min"),
            last_valid_price_date=("last_valid_price_date", "max"),
            observation_rows=("observation_rows", "sum"),
            priced_rows=("priced_rows", "sum"),
            pre_price_placeholder_rows=("pre_price_placeholder_rows", "sum"),
        )
        .reset_index()
    )
    final = (
        data.sort_values(["security_key", "last_observation_date", "source_period"])
        .groupby("security_key", sort=False)
        .tail(1)
        [[
            "security_key",
            "ticker",
            "bbg_id",
            "isin",
            "name",
            "security_type",
            "exchange_market_group",
            "financial_status",
            "market_status",
        ]]
    )
    latest_period = (
        data.loc[data["source_period"].eq("2026")]
        .set_index("security_key")["priced_in_period"]
        .astype(bool)
    )
    aggregate = aggregate.merge(final, on="security_key", how="left")
    aggregate["present_with_price_in_2026"] = (
        aggregate["security_key"].map(latest_period).astype("boolean").fillna(False).astype(bool)
    )
    aggregate["active_candidate"] = (
        aggregate["present_with_price_in_2026"]
        & aggregate["market_status"].fillna("").eq("ACTV")
    )
    aggregate["inactive_or_disappeared_candidate"] = ~aggregate["active_candidate"]
    return aggregate.sort_values(["first_valid_price_date", "security_key"]).reset_index(drop=True)


def _combine_identifier_history(histories: Iterable[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(tuple(histories), ignore_index=True)
    combined = (
        data.groupby(["security_key", "id_type", "id_value"], sort=False)
        .agg(effective_from=("effective_from", "min"), effective_to=("effective_to", "max"))
        .reset_index()
    )
    combined["change_type"] = "observed_vendor_identifier"
    combined["change_reason"] = "daily Bloomberg-derived identifier observation"
    combined["source_vendor"] = "bloomberg_friend_bulk"
    combined["evidence_level"] = "user_attested_vendor_export_observation"
    combined["eligible_for_model_denominator"] = False
    return combined.sort_values(["security_key", "id_type", "effective_from"]).reset_index(drop=True)


def _build_listing_windows(security_master: pd.DataFrame) -> pd.DataFrame:
    data = security_master.copy()
    return pd.DataFrame(
        {
            "entity_id": data["security_key"],
            "sid": data["security_key"],
            "ric": pd.Series(pd.NA, index=data.index, dtype="string"),
            "ticker": data["ticker"],
            "country": "Malaysia",
            "country_code": "MYS",
            "exchange": "Bursa Malaysia",
            "exchange_code": data["exchange_market_group"],
            "currency": "MYR",
            "listing_date": data["first_valid_price_date"],
            "first_trade_date": data["first_valid_price_date"],
            "effective_from": data["first_valid_price_date"],
            "effective_to": data["last_valid_price_date"],
            "delisting_date": pd.NaT,
            "last_trade_date": data["last_valid_price_date"],
            "listing_status": np.where(data["active_candidate"], "candidate_active", "candidate_disappeared"),
            "asset_state": data["market_status"],
            "asset_state_name": data["market_status"],
            "delisting_reason": np.where(
                data["active_candidate"],
                pd.NA,
                "unverified_terminal_event_requires_reconciliation",
            ),
            "tradable_flag": data["priced_rows"].fillna(0).gt(0),
            "active_flag": data["active_candidate"],
            "window_source": "bloomberg_daily_first_last_valid_price_candidate",
            "evidence_level": "candidate_window_terminal_economics_unverified",
            "eligible_for_model_denominator": False,
            "source_loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


def _build_monthly_membership_candidate(security_master: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in security_master.itertuples(index=False):
        if pd.isna(item.first_valid_price_date) or pd.isna(item.last_valid_price_date):
            continue
        dates = pd.date_range(item.first_valid_price_date, item.last_valid_price_date, freq="ME")
        if not len(dates):
            dates = pd.DatetimeIndex([pd.Timestamp(item.last_valid_price_date)])
        rows.append(
            pd.DataFrame(
                {
                    "asof_date": dates,
                    "entity_id": item.security_key,
                    "sid": item.security_key,
                    "ric_asof": pd.NA,
                    "ticker_asof": item.ticker,
                    "country": "Malaysia",
                    "country_code": "MYS",
                    "exchange": "Bursa Malaysia",
                    "currency": "MYR",
                    "membership_start": item.first_valid_price_date,
                    "membership_end": item.last_valid_price_date,
                    "is_listed_asof": True,
                    "is_investable_asof": False,
                    "eligible_flag": False,
                    "u2_backtest_eligible": pd.NA,
                    "u2_rule_status": "pending_frozen_point_in_time_definition",
                    "exclusion_reason": "historical_u2_rule_not_yet_frozen",
                    "known_at_utc": pd.NaT,
                    "membership_source": "bloomberg_valid_price_window_candidate",
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_candidate_returns(
    market_paths: Iterable[Path],
    output_dir: Path,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_close: dict[str, float] = {}
    outputs: list[Path] = []
    total_rows = 0
    extreme_rows = 0
    for market_path in market_paths:
        data = pd.read_parquet(
            market_path,
            columns=[
                "source_period",
                "source_row_number",
                "date",
                "security_key",
                "ticker",
                "adj_close",
                "cash_dividend_adjusted",
            ],
        ).sort_values(["security_key", "date", "source_row_number"])
        grouped = data.groupby("security_key", sort=False)
        prior = grouped["adj_close"].shift(1)
        first = ~data.duplicated("security_key", keep="first")
        prior.loc[first] = data.loc[first, "security_key"].map(previous_close)
        dividend = data["cash_dividend_adjusted"].fillna(0.0)
        data["price_return_local_candidate"] = data["adj_close"].div(prior).sub(1.0)
        data["total_return_local_gross_candidate"] = data["adj_close"].add(dividend).div(prior).sub(1.0)
        data["adj_close_total_return_gross_candidate"] = (
            data.groupby("security_key", sort=False)["total_return_local_gross_candidate"]
            .transform(lambda series: series.fillna(0.0).add(1.0).cumprod())
        )
        data["adjustment_status"] = "candidate_vendor_adjusted_terminal_economics_unverified"
        data["production_adjusted_return_ready"] = False
        extreme_rows += int(data["total_return_local_gross_candidate"].abs().gt(1.0).sum())
        total_rows += len(data)
        last = data.groupby("security_key", sort=False)["adj_close"].last()
        previous_close.update({str(key): float(value) for key, value in last.items() if pd.notna(value)})
        period = str(data["source_period"].iloc[0])
        output = output_dir / f"candidate_returns_{period.replace('-', '_')}.parquet"
        data.to_parquet(output, index=False, compression="zstd")
        outputs.append(output)
    return tuple(outputs), {
        "candidate_return_rows": int(total_rows),
        "absolute_return_over_100pct_rows": int(extreme_rows),
        "production_adjusted_return_ready": False,
        "terminal_economics_reconciled": False,
    }


def _normalized_entity_name(value: object) -> str:
    tokens = re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).split()
    removable = {
        "BHD",
        "BERHAD",
        "SDN",
        "HOLDING",
        "HOLDINGS",
        "GROUP",
        "CORP",
        "CORPORATION",
        "MALAYSIA",
    }
    return " ".join(token for token in tokens if token not in removable)


def _value_to_security_keys(rows: pd.DataFrame, value_column: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in rows[["security_key", value_column]].dropna().itertuples(index=False):
        value = str(getattr(row, value_column)).strip().upper()
        if value:
            result.setdefault(value, set()).add(str(row.security_key))
    return result


def _existing_dead_name_coverage(
    identifier_history: pd.DataFrame,
    security_master: pd.DataFrame,
    existing_security_master_root: str | Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if existing_security_master_root is None:
        return {"available": False, "reason": "existing security master not supplied"}, pd.DataFrame()
    root = Path(existing_security_master_root)
    listings_path = root / "listing_windows.parquet"
    identifiers_path = root / "identifier_history.parquet"
    entities_path = root / "entities.parquet"
    if not listings_path.is_file() or not identifiers_path.is_file() or not entities_path.is_file():
        return {"available": False, "reason": "existing security master artifacts missing"}, pd.DataFrame()

    listings = pd.read_parquet(listings_path)
    existing_ids = pd.read_parquet(identifiers_path)
    existing_entities = pd.read_parquet(entities_path)
    inactive = listings.loc[
        listings["country"].eq("Malaysia") & ~listings["active_flag"].astype(bool)
    ].copy()
    inactive_entities = set(inactive["entity_id"].dropna().astype(str))

    source_start = pd.to_datetime(security_master["first_valid_price_date"], errors="coerce").min()
    source_end = pd.to_datetime(security_master["last_valid_price_date"], errors="coerce").max()
    for column in (
        "first_trade_date",
        "listing_date",
        "effective_from",
        "effective_to",
        "delisting_date",
        "last_trade_date",
    ):
        inactive[column] = pd.to_datetime(inactive[column], errors="coerce")
    first_observed = inactive[["first_trade_date", "listing_date", "effective_from"]].min(axis=1)
    last_observed = inactive[["effective_to", "delisting_date", "last_trade_date"]].max(axis=1)
    relevant = inactive.loc[
        (last_observed.isna() | last_observed.ge(source_start))
        & (first_observed.isna() | first_observed.le(source_end))
    ].copy()
    relevant_entities = set(relevant["entity_id"].dropna().astype(str))

    existing_isin = existing_ids.loc[
        existing_ids["entity_id"].astype(str).isin(inactive_entities)
        & existing_ids["id_type"].astype(str).str.casefold().eq("isin")
    ].copy()
    source_identifiers = identifier_history.copy()
    source_identifiers["id_value"] = source_identifiers["id_value"].astype("string").str.strip()
    source_isin_rows = source_identifiers.loc[source_identifiers["id_type"].eq("isin")]
    source_ticker_rows = source_identifiers.loc[source_identifiers["id_type"].eq("bbg_ticker")].copy()
    source_ticker_rows["ticker_value"] = (
        source_ticker_rows["id_value"]
        .str.replace(r"\s+MK\s+Equity$", "", regex=True, case=False)
        .str.upper()
    )
    source_isin_map = _value_to_security_keys(source_isin_rows, "id_value")
    source_ticker_map = _value_to_security_keys(source_ticker_rows, "ticker_value")
    source_name_rows = security_master[["security_key", "name"]].dropna().copy()
    source_name_rows["normalized_name"] = source_name_rows["name"].map(_normalized_entity_name)
    source_name_map = _value_to_security_keys(source_name_rows, "normalized_name")

    relevant_ids = existing_ids.loc[
        existing_ids["entity_id"].astype(str).isin(relevant_entities)
    ].copy()
    entity_identifiers: dict[str, dict[str, set[str]]] = {}
    for row in relevant_ids[["entity_id", "id_type", "id_value"]].dropna().itertuples(index=False):
        entity = str(row.entity_id)
        id_type = str(row.id_type).casefold()
        value = str(row.id_value).strip().upper()
        entity_identifiers.setdefault(entity, {}).setdefault(id_type, set()).add(value)
    for row in relevant[["entity_id", "ticker", "ric"]].dropna(subset=["entity_id"]).itertuples(index=False):
        values = entity_identifiers.setdefault(str(row.entity_id), {})
        if pd.notna(row.ticker):
            values.setdefault("ticker", set()).add(str(row.ticker).strip().upper())
        if pd.notna(row.ric):
            values.setdefault("ric", set()).add(str(row.ric).split(".", 1)[0].strip().upper())

    entity_names: dict[str, set[str]] = {}
    name_columns = [column for column in ("issuer_name", "document_title") if column in existing_entities]
    for row in existing_entities.loc[
        existing_entities["entity_id"].astype(str).isin(relevant_entities),
        ["entity_id", *name_columns],
    ].itertuples(index=False):
        for value in row[1:]:
            normalized = _normalized_entity_name(value)
            if normalized:
                entity_names.setdefault(str(row.entity_id), set()).add(normalized)

    relevant_ric = (
        relevant.sort_values(["entity_id", "last_trade_date"])
        .groupby("entity_id", sort=False)
        .tail(1)
        .set_index("entity_id")["ric"]
    )
    match_rows: list[dict[str, Any]] = []
    for entity in sorted(relevant_entities):
        identifiers = entity_identifiers.get(entity, {})
        candidates: set[str] = set()
        match_type = "unmatched"
        confidence = "none"
        for value in identifiers.get("isin", set()):
            candidates.update(source_isin_map.get(value, set()))
        if len(candidates) == 1:
            match_type, confidence = "isin_exact", "high"
        else:
            candidates = set()
            for value in identifiers.get("ticker", set()) | identifiers.get("ric", set()):
                candidates.update(source_ticker_map.get(value, set()))
            if len(candidates) == 1:
                match_type, confidence = "ticker_exact_unique", "high"
            else:
                candidates = set()
                for value in entity_names.get(entity, set()):
                    candidates.update(source_name_map.get(value, set()))
                if len(candidates) == 1:
                    match_type, confidence = "name_exact_unique", "review"
                elif len(candidates) > 1:
                    match_type, confidence = "ambiguous_exact_name", "none"
        matched_key = next(iter(candidates)) if len(candidates) == 1 else None
        match_rows.append(
            {
                "existing_entity_id": entity,
                "existing_ric": relevant_ric.get(entity),
                "source_security_key": matched_key,
                "match_type": match_type,
                "confidence": confidence,
                "matched": matched_key is not None,
            }
        )
    crosswalk = pd.DataFrame(match_rows)
    high_confidence = int(crosswalk["confidence"].eq("high").sum()) if not crosswalk.empty else 0
    review_matches = int(crosswalk["confidence"].eq("review").sum()) if not crosswalk.empty else 0
    all_exact_matches = int(crosswalk["matched"].sum()) if not crosswalk.empty else 0

    source_isins = set(source_isin_map)
    matched_all_inactive_by_isin = set(
        existing_isin.loc[
            existing_isin["id_value"].astype(str).str.upper().isin(source_isins), "entity_id"
        ].dropna().astype(str)
    )
    denominator = len(inactive_entities)
    mappable_entities = set(existing_isin["entity_id"].dropna().astype(str))
    relevant_isin_entities = set(
        existing_isin.loc[
            existing_isin["entity_id"].astype(str).isin(relevant_entities), "entity_id"
        ].dropna().astype(str)
    )
    report = {
        "available": True,
        "source_date_min": str(pd.Timestamp(source_start).date()),
        "source_date_max": str(pd.Timestamp(source_end).date()),
        "malaysia_inactive_listing_rows": int(len(inactive)),
        "malaysia_inactive_entities": int(denominator),
        "inactive_entities_with_isin": int(len(mappable_entities)),
        "inactive_entities_matched_by_isin": int(len(matched_all_inactive_by_isin)),
        "coverage_against_all_inactive_entities": (
            len(matched_all_inactive_by_isin) / denominator if denominator else 0.0
        ),
        "coverage_against_isin_mappable_inactive_entities": (
            len(matched_all_inactive_by_isin) / len(mappable_entities) if mappable_entities else 0.0
        ),
        "strategy_relevant_inactive_listing_rows": int(len(relevant)),
        "strategy_relevant_inactive_entities": int(len(relevant_entities)),
        "strategy_relevant_inactive_entities_with_isin": int(len(relevant_isin_entities)),
        "strategy_relevant_high_confidence_matches": high_confidence,
        "strategy_relevant_exact_name_review_matches": review_matches,
        "strategy_relevant_all_exact_matches": all_exact_matches,
        "high_confidence_coverage_against_strategy_relevant_entities": (
            high_confidence / len(relevant_entities) if relevant_entities else 0.0
        ),
        "all_exact_match_coverage_against_strategy_relevant_entities": (
            all_exact_matches / len(relevant_entities) if relevant_entities else 0.0
        ),
    }
    return report, crosswalk


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths if path.is_file()}


def _frozen_pit_gate_projection(
    *,
    security_master: pd.DataFrame,
    dead_coverage: Mapping[str, Any],
    institutional_gates: Mapping[str, bool],
) -> dict[str, Any]:
    from quant_lab.research.malaysia_pit_replication import (
        PITCoverageInputs,
        evaluate_admissibility,
    )

    relevant_inactive = int(dead_coverage.get("strategy_relevant_inactive_entities", 0))
    matched = int(dead_coverage.get("strategy_relevant_high_confidence_matches", 0))
    promotion_statuses = {
        gate_id: "pass" if passed else "fail"
        for gate_id, passed in institutional_gates.items()
    }
    inputs = PITCoverageInputs(
        source_hashes_match=True,
        production_adjusted_returns_ready=bool(
            institutional_gates.get("production_adjusted_returns_ready")
        ),
        historical_u2_membership_reconstructable=bool(
            institutional_gates.get("historical_u2_rule_frozen")
        ),
        malaysia_listing_rows=int(len(security_master)),
        malaysia_inactive_rows=relevant_inactive,
        malaysia_current_feature_tickers=int(security_master["active_candidate"].sum()),
        malaysia_current_adjusted_return_tickers=int(security_master["active_candidate"].sum()),
        malaysia_u2_current_tickers=62,
        malaysia_u3_current_tickers=34,
        inactive_signal_tickers_covered=matched,
        inactive_adjusted_return_tickers_covered=matched,
        security_master_promotion_gates_all_pass=all(institutional_gates.values()),
        security_master_promotion_statuses=promotion_statuses,
    )
    evaluation = evaluate_admissibility(
        inputs,
        minimum_dead_name_return_coverage=0.95,
    )
    return {
        "schema_version": 1,
        "projection_only": True,
        "protocol_id": "QFORGE20-MYS-PIT-REPLICATION-14-FROZEN-V1",
        "purpose": "Apply the unchanged frozen admissibility mathematics to the integrated candidate source without replacing frozen data receipts.",
        "coverage_basis": "strategy-relevant inactive Malaysia entities whose listing windows overlap 2005-01-03 through 2026-04-13",
        "inputs": {
            "strategy_relevant_inactive_entities": relevant_inactive,
            "high_confidence_matched_entities": matched,
            "candidate_signal_and_return_coverage": (
                matched / relevant_inactive if relevant_inactive else 0.0
            ),
            "production_adjusted_returns_ready": inputs.production_adjusted_returns_ready,
            "historical_u2_rule_frozen": inputs.historical_u2_membership_reconstructable,
        },
        **evaluation,
        "strategy_trials_reserved": False,
        "strategy_computations_started": False,
        "dry_run_started": False,
    }


def _load_completed_artifacts(artifact_dir: Path, manifest: Mapping[str, Any]) -> BloombergMalaysiaBulkArtifacts:
    artifacts = manifest["artifacts"]
    return BloombergMalaysiaBulkArtifacts(
        run_id=str(manifest["run_id"]),
        artifact_dir=artifact_dir,
        manifest_path=artifact_dir / "manifest.json",
        admission_report_path=Path(artifacts["admission_report"]),
        raw_manifest_path=Path(artifacts["raw_manifest"]),
        security_master_path=Path(artifacts["security_master_candidate"]),
        identifier_history_path=Path(artifacts["identifier_history"]),
        listing_windows_path=Path(artifacts["listing_windows_candidate"]),
        membership_candidate_path=Path(artifacts["membership_candidate"]),
        normalized_paths=tuple(Path(item) for item in artifacts["normalized_daily_observations"]),
        market_paths=tuple(Path(item) for item in artifacts["research_market_candidates"]),
        return_paths=tuple(Path(item) for item in artifacts["candidate_adjusted_returns"]),
        manifest=manifest,
    )


def build_bloomberg_malaysia_bulk_lake(
    source_config_path: str | Path,
    *,
    artifact_root: str | Path,
    run_id: str,
    existing_security_master_root: str | Path | None = None,
    max_download_workers: int = 2,
    chunk_rows: int = 100_000,
    force_rebuild_generated: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BloombergMalaysiaBulkArtifacts:
    config, specs = load_source_config(source_config_path)
    artifact_dir = Path(artifact_root) / run_id
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file() and not force_rebuild_generated:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "integrated_candidate_data_lane":
            return _load_completed_artifacts(artifact_dir, manifest)

    raw_dir = artifact_dir / "raw"
    normalized_dir = artifact_dir / "normalized" / "daily_observations"
    market_dir = artifact_dir / "curated" / "research_market_candidate"
    return_dir = artifact_dir / "curated" / "adjusted_returns_candidate"
    metadata_dir = artifact_dir / "curated" / "metadata"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = download_source_files(
        specs,
        raw_dir,
        max_workers=max_download_workers,
        progress=progress,
    )
    raw_rows = []
    for spec in specs:
        path = raw_paths[spec.period]
        raw_rows.append(
            {
                "period": spec.period,
                "filename": spec.filename,
                "drive_file_id": spec.drive_file_id,
                "expected_bytes": spec.expected_bytes,
                "observed_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "path": str(path.resolve()),
            }
        )
    raw_manifest = pd.DataFrame(raw_rows)
    raw_manifest_path = artifact_dir / "raw_manifest.csv"
    raw_manifest.to_csv(raw_manifest_path, index=False)

    profiles: list[dict[str, Any]] = []
    summaries: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    normalized_paths: list[Path] = []
    market_paths: list[Path] = []
    for spec in specs:
        normalized_path = normalized_dir / f"daily_observations_{spec.period.replace('-', '_')}.parquet"
        market_path = market_dir / f"research_market_{spec.period.replace('-', '_')}.parquet"
        profile, summary, history = normalize_and_curate_source_file(
            spec,
            raw_paths[spec.period],
            normalized_path,
            market_path,
            chunk_rows=chunk_rows,
        )
        profiles.append(profile)
        summaries.append(summary)
        histories.append(history)
        normalized_paths.append(normalized_path)
        market_paths.append(market_path)
        if progress:
            progress(f"normalized and curated {spec.filename}")

    security_master = _combine_security_summaries(summaries)
    identifier_history = _combine_identifier_history(histories)
    listing_windows = _build_listing_windows(security_master)
    membership = _build_monthly_membership_candidate(security_master)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    security_master_path = metadata_dir / "security_master_candidate.parquet"
    identifier_history_path = metadata_dir / "identifier_history.parquet"
    listing_windows_path = metadata_dir / "listing_windows_candidate.parquet"
    membership_path = metadata_dir / "universe_membership_candidate.parquet"
    security_master.to_parquet(security_master_path, index=False)
    identifier_history.to_parquet(identifier_history_path, index=False)
    listing_windows.to_parquet(listing_windows_path, index=False)
    membership.to_parquet(membership_path, index=False)

    return_paths, return_profile = _build_candidate_returns(market_paths, return_dir)
    dead_coverage, dead_crosswalk = _existing_dead_name_coverage(
        identifier_history,
        security_master,
        existing_security_master_root,
    )
    dead_crosswalk_path = metadata_dir / "dead_name_crosswalk_candidate.csv"
    dead_crosswalk.to_csv(dead_crosswalk_path, index=False)
    quality_profile_path = artifact_dir / "quality_profile.csv"
    pd.DataFrame(profiles).to_csv(quality_profile_path, index=False)

    core_gates = {
        "source_attestation_recorded": config["source_attestation"]["status"]
        == "user_attested_permission_to_ingest_for_internal_research",
        "all_eight_source_files_complete": bool(
            len(raw_manifest) == 8
            and raw_manifest["expected_bytes"].eq(raw_manifest["observed_bytes"]).all()
        ),
        "all_dates_parse": sum(item["unusable_missing_security_or_date_rows"] for item in profiles) == 0,
        "curated_security_date_unique": True,
        "no_negative_prices": sum(item["negative_price_rows"] for item in profiles) == 0,
        "no_negative_volume": sum(item["negative_volume_rows"] for item in profiles) == 0,
        "historical_inactive_candidates_present": int(
            security_master["inactive_or_disappeared_candidate"].sum()
        )
        > 0,
    }
    institutional_gates = {
        "historical_u2_rule_frozen": False,
        "terminal_delisting_economics_reconciled": False,
        "production_adjusted_returns_ready": False,
        "fundamentals_point_in_time_proven": False,
        "analyst_estimate_revision_timestamps_proven": False,
        "minimum_95pct_dead_name_coverage": bool(
            dead_coverage.get(
                "high_confidence_coverage_against_strategy_relevant_entities", 0.0
            )
            >= 0.95
        ),
    }
    frozen_gate_projection = _frozen_pit_gate_projection(
        security_master=security_master,
        dead_coverage=dead_coverage,
        institutional_gates=institutional_gates,
    )
    frozen_gate_projection_path = artifact_dir / "frozen_pit_gate_projection.json"
    frozen_gate_projection_path.write_text(
        json.dumps(frozen_gate_projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission_report = {
        "schema_version": 1,
        "run_id": run_id,
        "source_id": config["source_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "market_price_volume_candidate_integrated_institutional_pit_pending",
        "source_attestation": config["source_attestation"],
        "counts": {
            "raw_rows": int(sum(item["raw_rows"] for item in profiles)),
            "deduplicated_rows": int(sum(item["deduplicated_rows"] for item in profiles)),
            "research_market_rows": int(sum(item["research_market_rows"] for item in profiles)),
            "security_keys": int(len(security_master)),
            "inactive_or_disappeared_candidates": int(
                security_master["inactive_or_disappeared_candidate"].sum()
            ),
            "duplicate_security_date_rows": int(
                sum(item["duplicate_security_date_rows"] for item in profiles)
            ),
            "pre_price_placeholder_rows": int(
                sum(item["pre_price_placeholder_rows"] for item in profiles)
            ),
        },
        "core_market_data_gates": core_gates,
        "institutional_pit_gates": institutional_gates,
        "existing_security_master_dead_name_coverage": dead_coverage,
        "candidate_return_profile": return_profile,
        "frozen_pit_gate_projection": {
            "admissible": frozen_gate_projection["admissible"],
            "failed_gate_ids": frozen_gate_projection["failed_gate_ids"],
            "candidate_signal_and_return_coverage": frozen_gate_projection["inputs"][
                "candidate_signal_and_return_coverage"
            ],
        },
        "lane_decisions": {
            "price_volume_signal_research": "candidate_ready_after_protocol_receipt_freeze",
            "fundamentals_and_accounting": "paused",
            "analyst_estimates": "paused",
            "terminal_return_claims": "paused",
            "fourteen_strategy_pit_replication": "blocked_until_all_institutional_pit_gates_pass",
            "dry_run": "prohibited",
        },
    }
    admission_report_path = artifact_dir / "admission_report.json"
    admission_report_path.write_text(
        json.dumps(admission_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_paths = [
        raw_manifest_path,
        quality_profile_path,
        admission_report_path,
        frozen_gate_projection_path,
        security_master_path,
        identifier_history_path,
        listing_windows_path,
        membership_path,
        dead_crosswalk_path,
        *normalized_paths,
        *market_paths,
        *return_paths,
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_id": config["source_id"],
        "status": "integrated_candidate_data_lane",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_source_hashes": {
            row["filename"]: row["sha256"] for row in raw_rows
        },
        "output_sha256": _artifact_hashes(output_paths),
        "admission_status": admission_report["status"],
        "strategy_trials_reserved": 0,
        "strategy_computations_started": 0,
        "dry_run_started": False,
        "artifacts": {
            "raw_manifest": str(raw_manifest_path.resolve()),
            "quality_profile": str(quality_profile_path.resolve()),
            "admission_report": str(admission_report_path.resolve()),
            "frozen_pit_gate_projection": str(frozen_gate_projection_path.resolve()),
            "security_master_candidate": str(security_master_path.resolve()),
            "identifier_history": str(identifier_history_path.resolve()),
            "listing_windows_candidate": str(listing_windows_path.resolve()),
            "membership_candidate": str(membership_path.resolve()),
            "dead_name_crosswalk_candidate": str(dead_crosswalk_path.resolve()),
            "normalized_daily_observations": [str(path.resolve()) for path in normalized_paths],
            "research_market_candidates": [str(path.resolve()) for path in market_paths],
            "candidate_adjusted_returns": [str(path.resolve()) for path in return_paths],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _load_completed_artifacts(artifact_dir, manifest)
