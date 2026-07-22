from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ASEAN_REFINITIV_COUNTRY_CODES: dict[str, str] = {
    "Malaysia": "MY",
    "Singapore": "SG",
    "Indonesia": "ID",
}


REFINITIV_DISCOVERY_FIELDS: tuple[str, ...] = (
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeCountryCode",
    "TR.HQCountryCode",
    "TR.CompanyMarketCap.date",
    "TR.CompanyMarketCap",
    "TR.FreeFloatPct",
    "TR.SharesOutstanding",
    "TR.AvgDailyVolume30D",
    "TR.AvgDailyValTraded30D",
    "TR.PriceClose",
    "TR.Volume",
)


REFINITIV_DISCOVERY_COLUMN_MAP = {
    "Instrument": "refinitiv_ric",
    "Company Common Name": "name",
    "Exchange Name": "exchange_name",
    "Exchange Country Code": "exchange_country_code",
    "Country ISO Code of Headquarters": "hq_country_code",
    "Date": "snapshot_date",
    "Company Market Cap": "market_cap",
    "Free Float (Percent)": "free_float_percent",
    "Outstanding Shares": "shares_outstanding",
    "Average Daily Volume - 30 Days": "adv_30d",
    "Average Daily Value Traded - 30 Days": "adv_value_30d",
    "Price Close": "price_close",
    "Volume": "volume",
}


@dataclass(frozen=True)
class RefinitivDiscoveryArtifact:
    artifact_dir: Path
    raw_universe_path: Path
    ranked_universe_path: Path
    symbol_map_input_path: Path
    country_summary_path: Path
    manifest_path: Path
    raw_universe: pd.DataFrame
    ranked_universe: pd.DataFrame
    symbol_map_input: pd.DataFrame
    country_summary: pd.DataFrame


def refinitiv_equity_screener_expression(country_code: str) -> str:
    """Return an active public primary equity screener for an exchange country."""

    code = country_code.upper().strip()
    return f'U(IN(Equity(active,public,primary))/*UNV:Public*/), IN(TR.ExchangeCountryCode,"{code}")'


def _chunks(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _normalize_screener_fields(frame: pd.DataFrame, country: str, country_code: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.rename(columns={key: value for key, value in REFINITIV_DISCOVERY_COLUMN_MAP.items() if key in frame.columns}).copy()
    if "refinitiv_ric" not in data.columns:
        return pd.DataFrame()
    data["country"] = country
    data["exchange_country_code"] = data.get("exchange_country_code", country_code)
    data["ticker"] = data["refinitiv_ric"].astype(str).str.split(".").str[0]
    data["yahoo_ticker"] = data["refinitiv_ric"]
    data["currency"] = pd.NA
    data["exchange"] = data.get("exchange_name", country)
    data["security_type"] = "Common Stock"
    data["asset_type"] = "Equity"
    data["is_stock_like"] = True
    data["source"] = "refinitiv_discovery_screener"
    data["symbol_map_source"] = "refinitiv_discovery_screener"
    data["source_symbol_id"] = data["country"].astype(str) + "|" + data["refinitiv_ric"].astype(str)
    numeric_cols = [
        "market_cap",
        "free_float_percent",
        "shares_outstanding",
        "adv_30d",
        "adv_value_30d",
        "price_close",
        "volume",
    ]
    for column in numeric_cols:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _rank_refinitiv_discovery_universe(frame: pd.DataFrame, max_per_country: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    for column in ("adv_value_30d", "market_cap", "adv_30d"):
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["l2_rank_score"] = data["adv_value_30d"].fillna(data["market_cap"]).fillna(data["adv_30d"]).fillna(0.0)
    data["l2_rank_method"] = "refinitiv_adv_value_30d"
    data.loc[data["adv_value_30d"].isna() & data["market_cap"].notna(), "l2_rank_method"] = "refinitiv_market_cap"
    data.loc[
        data["adv_value_30d"].isna() & data["market_cap"].isna() & data["adv_30d"].notna(),
        "l2_rank_method",
    ] = "refinitiv_adv_30d"
    data.loc[data[["adv_value_30d", "market_cap", "adv_30d"]].isna().all(axis=1), "l2_rank_method"] = "refinitiv_screener_order"
    data = data.sort_values(
        ["country", "adv_value_30d", "market_cap", "adv_30d", "refinitiv_ric"],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    data["l2_country_rank"] = data.groupby("country", dropna=False).cumcount() + 1
    data["l2_target_candidate"] = data["l2_country_rank"].le(max_per_country)
    return data


def screen_refinitiv_asean_l2_universe(
    ld: Any,
    countries: tuple[str, ...] = tuple(ASEAN_REFINITIV_COUNTRY_CODES),
    max_per_country: int = 100,
    max_screen_results_per_country: int = 1000,
    chunk_size: int = 100,
    sleep_seconds: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use Refinitiv Discovery to build an ASEAN active public primary equity universe."""

    rows: list[pd.DataFrame] = []
    raw_ric_rows: list[dict[str, object]] = []
    screener_cls = getattr(getattr(ld, "discovery"), "Screener")
    for country in countries:
        country_code = ASEAN_REFINITIV_COUNTRY_CODES[country]
        expression = refinitiv_equity_screener_expression(country_code)
        screener = screener_cls(expression)
        rics = list(dict.fromkeys(list(screener)))[:max_screen_results_per_country]
        for rank, ric in enumerate(rics, start=1):
            raw_ric_rows.append(
                {
                    "country": country,
                    "country_code": country_code,
                    "screener_expression": expression,
                    "screener_rank": rank,
                    "refinitiv_ric": ric,
                }
            )
        for chunk in _chunks(rics, chunk_size):
            if not chunk:
                continue
            data = ld.get_data(universe=chunk, fields=list(REFINITIV_DISCOVERY_FIELDS))
            normalized = _normalize_screener_fields(data, country=country, country_code=country_code)
            if not normalized.empty:
                rows.append(normalized)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raw_universe = pd.DataFrame(raw_ric_rows)
    enriched = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not enriched.empty:
        enriched = enriched.drop_duplicates(["country", "refinitiv_ric"], keep="first")
    ranked = _rank_refinitiv_discovery_universe(enriched, max_per_country=max_per_country)
    return raw_universe, ranked


def build_refinitiv_discovery_symbol_map_input(ranked_universe: pd.DataFrame) -> pd.DataFrame:
    if ranked_universe.empty:
        return pd.DataFrame()
    data = ranked_universe[ranked_universe["l2_target_candidate"].fillna(False)].copy()
    keep_cols = [
        "country",
        "exchange",
        "ticker",
        "yahoo_ticker",
        "name",
        "currency",
        "security_type",
        "asset_type",
        "is_stock_like",
        "source",
        "symbol_map_source",
        "refinitiv_ric",
        "market_cap",
        "free_float_percent",
        "adv_30d",
        "adv_value_30d",
        "l2_country_rank",
        "l2_rank_method",
        "l2_rank_score",
    ]
    keep_cols = [column for column in keep_cols if column in data.columns]
    return data[keep_cols].reset_index(drop=True)


def summarize_refinitiv_discovery_universe(ranked_universe: pd.DataFrame, max_per_country: int) -> pd.DataFrame:
    rows = []
    for country in ASEAN_REFINITIV_COUNTRY_CODES:
        country_data = (
            ranked_universe[ranked_universe["country"].astype("string").eq(country)].copy()
            if not ranked_universe.empty and "country" in ranked_universe.columns
            else pd.DataFrame()
        )
        l2 = country_data[country_data.get("l2_target_candidate", pd.Series(False, index=country_data.index)).fillna(False)]
        rows.append(
            {
                "country": country,
                "screened_rows": int(len(country_data)),
                "l2_candidate_rows": int(len(l2)),
                "target_per_country": int(max_per_country),
                "shortfall": max(0, int(max_per_country) - int(len(l2))),
                "median_adv_value_30d": float(l2["adv_value_30d"].median()) if not l2.empty and "adv_value_30d" in l2 else pd.NA,
                "min_adv_value_30d": float(l2["adv_value_30d"].min()) if not l2.empty and "adv_value_30d" in l2 else pd.NA,
                "max_market_cap": float(l2["market_cap"].max()) if not l2.empty and "market_cap" in l2 else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def write_refinitiv_asean_l2_discovery_artifacts(
    ld: Any,
    artifact_root: str | Path = "artifacts/refinitiv_discovery",
    run_id: str | None = None,
    countries: tuple[str, ...] = tuple(ASEAN_REFINITIV_COUNTRY_CODES),
    max_per_country: int = 100,
    max_screen_results_per_country: int = 1000,
    chunk_size: int = 100,
    sleep_seconds: float = 0.05,
) -> RefinitivDiscoveryArtifact:
    run_id = run_id or datetime.now(timezone.utc).strftime("REFINITIV-ASEAN-L2-DISCOVERY-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_universe, ranked_universe = screen_refinitiv_asean_l2_universe(
        ld=ld,
        countries=countries,
        max_per_country=max_per_country,
        max_screen_results_per_country=max_screen_results_per_country,
        chunk_size=chunk_size,
        sleep_seconds=sleep_seconds,
    )
    symbol_map_input = build_refinitiv_discovery_symbol_map_input(ranked_universe)
    country_summary = summarize_refinitiv_discovery_universe(ranked_universe, max_per_country=max_per_country)

    raw_universe_path = artifact_dir / "raw_screener_rics.csv"
    ranked_universe_path = artifact_dir / "ranked_refinitiv_universe.csv"
    symbol_map_input_path = artifact_dir / "refinitiv_symbol_map_input.csv"
    country_summary_path = artifact_dir / "country_summary.csv"
    manifest_path = artifact_dir / "manifest.json"

    raw_universe.to_csv(raw_universe_path, index=False)
    ranked_universe.to_csv(ranked_universe_path, index=False)
    symbol_map_input.to_csv(symbol_map_input_path, index=False)
    country_summary.to_csv(country_summary_path, index=False)
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "countries": list(countries),
        "max_per_country": int(max_per_country),
        "max_screen_results_per_country": int(max_screen_results_per_country),
        "raw_screener_rows": int(len(raw_universe)),
        "ranked_universe_rows": int(len(ranked_universe)),
        "symbol_map_input_rows": int(len(symbol_map_input)),
        "raw_universe_path": str(raw_universe_path),
        "ranked_universe_path": str(ranked_universe_path),
        "symbol_map_input_path": str(symbol_map_input_path),
        "country_summary_path": str(country_summary_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return RefinitivDiscoveryArtifact(
        artifact_dir=artifact_dir,
        raw_universe_path=raw_universe_path,
        ranked_universe_path=ranked_universe_path,
        symbol_map_input_path=symbol_map_input_path,
        country_summary_path=country_summary_path,
        manifest_path=manifest_path,
        raw_universe=raw_universe,
        ranked_universe=ranked_universe,
        symbol_map_input=symbol_map_input,
        country_summary=country_summary,
    )
