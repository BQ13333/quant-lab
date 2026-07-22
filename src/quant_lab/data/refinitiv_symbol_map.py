from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SYMBOL_MAP_CLOSE_FIELDS = ("TRDPRC_1", "TR.CLOSEPRICE")


CURATED_REFINITIV_RIC_ALIASES: dict[tuple[str, str], str] = {
    ("Malaysia", "1155"): "MBBM.KL",
    ("Malaysia", "MAYBANK"): "MBBM.KL",
    ("Singapore", "D05"): "DBSM.SI",
    ("Singapore", "DBS"): "DBSM.SI",
}


@dataclass(frozen=True)
class RefinitivSymbolMapArtifact:
    artifact_dir: Path
    candidate_path: Path
    validation_path: Path
    symbol_map_path: Path
    manifest_path: Path
    candidates: pd.DataFrame
    validations: pd.DataFrame
    symbol_map: pd.DataFrame


def default_refinitiv_symbol_seed_listings() -> pd.DataFrame:
    """Liquid, manually curated seed rows for markets with weak public mapping."""

    return pd.DataFrame(
        [
            {"country": "Malaysia", "exchange": "Bursa Malaysia", "ticker": "MAYBANK", "name": "Maybank", "refinitiv_ric": "MBBM.KL"},
            {"country": "Singapore", "exchange": "SGX", "ticker": "DBS", "name": "DBS Group", "refinitiv_ric": "DBSM.SI"},
            {"country": "Indonesia", "exchange": "IDX", "ticker": "BBRI", "name": "Bank Rakyat Indonesia", "refinitiv_ric": "BBRI.JK"},
            {"country": "Hong Kong", "exchange": "HKEX", "ticker": "00700", "name": "Tencent", "refinitiv_ric": "0700.HK"},
            {"country": "Japan", "exchange": "JPX", "ticker": "7203", "name": "Toyota Motor", "refinitiv_ric": "7203.T"},
            {"country": "South Korea", "exchange": "KRX", "ticker": "005930", "name": "Samsung Electronics", "refinitiv_ric": "005930.KS"},
            {"country": "United States", "exchange": "NYSE", "ticker": "IBM", "name": "IBM", "refinitiv_ric": "IBM.N"},
            {"country": "United States", "exchange": "NASDAQ", "ticker": "MSFT", "name": "Microsoft", "refinitiv_ric": "MSFT.O"},
        ]
    ).assign(symbol_map_source="curated_seed")


def _clean_symbol(value: object, width: int | None = None) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    if width is not None and text.isdigit():
        text = text.zfill(width)
    return text


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _source_symbol_id(row: pd.Series | dict[str, object]) -> str:
    data = row if isinstance(row, dict) else row.to_dict()
    ticker = _clean_text(data.get("ticker"))
    values = [
        _clean_text(data.get("country")),
        _clean_text(data.get("exchange")),
        ticker or _clean_text(data.get("refinitiv_ric")),
    ]
    if not values[-1]:
        values.append(_clean_text(data.get("name")))
    return "|".join(values)


def _us_refinitiv_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper().replace("/", ".").replace("-", ".")
    if "." not in symbol:
        return symbol
    base, suffix = symbol.split(".", 1)
    suffix = re.sub(r"[^A-Z0-9]", "", suffix)
    return f"{base}{suffix[:1].lower()}" if suffix else base


def _candidate_suffix(country: str, exchange: str, ticker: str, yahoo_ticker: str | None) -> list[tuple[str, str]]:
    country_text = country.strip()
    exchange_text = exchange.strip()
    exchange_upper = exchange_text.upper()
    ticker = ticker.strip()
    yahoo_ticker = yahoo_ticker.strip() if yahoo_ticker else None

    if country_text == "United States":
        symbol = _us_refinitiv_symbol(ticker or (yahoo_ticker or ""))
        if not symbol:
            return []
        if "NASDAQ" in exchange_upper:
            return [(f"{symbol}.O", "us_exchange_suffix")]
        if "NYSE AMERICAN" in exchange_upper:
            return [(f"{symbol}.A", "us_exchange_suffix")]
        if "CBOE" in exchange_upper or "BZX" in exchange_upper:
            return [(f"{symbol}.Z", "us_exchange_suffix")]
        return [(f"{symbol}.N", "us_exchange_suffix")]

    if country_text == "Hong Kong":
        code = _clean_symbol(ticker)
        if code and code.isdigit():
            return [(f"{int(code):04d}.HK", "hk_numeric_suffix")]
        return [(yahoo_ticker, "yahoo_suffix")] if yahoo_ticker else []

    if country_text == "Vietnam":
        if "HNX" in exchange_upper:
            return [(f"{ticker}.HN", "vietnam_exchange_suffix"), (f"{ticker}.VN", "yahoo_suffix")]
        if "HOSE" in exchange_upper or "HSX" in exchange_upper:
            return [(f"{ticker}.HM", "vietnam_exchange_suffix"), (f"{ticker}.VN", "yahoo_suffix")]
        if "UPCOM" in exchange_upper:
            return [(f"{ticker}.HN", "vietnam_exchange_suffix"), (f"{ticker}.VN", "yahoo_suffix")]
        return [(f"{ticker}.HM", "vietnam_exchange_suffix"), (f"{ticker}.HN", "vietnam_exchange_suffix")]

    suffix_by_country = {
        "China": None,
        "Indonesia": ".JK",
        "Japan": ".T",
        "Malaysia": ".KL",
        "Philippines": ".PS",
        "Singapore": ".SI",
        "South Korea": None,
        "Taiwan": None,
        "Thailand": ".BK",
    }
    if country_text == "China":
        if "SSE" in exchange_upper:
            return [(f"{_clean_symbol(ticker, width=6)}.SS", "china_exchange_suffix")]
        if "SZSE" in exchange_upper:
            return [(f"{_clean_symbol(ticker, width=6)}.SZ", "china_exchange_suffix")]
    if country_text == "South Korea":
        code = _clean_symbol(ticker, width=6)
        if "KOSDAQ" in exchange_upper or "코스닥" in exchange_text:
            return [(f"{code}.KQ", "krx_exchange_suffix")]
        if "KONEX" in exchange_upper or "코넥스" in exchange_text:
            return []
        return [(f"{code}.KS", "krx_exchange_suffix")]
    if country_text == "Taiwan":
        code = _clean_symbol(ticker, width=4)
        if "TPEX" in exchange_upper or "TWO" in (yahoo_ticker or "").upper():
            return [(f"{code}.TWO", "taiwan_exchange_suffix")]
        return [(f"{code}.TW", "taiwan_exchange_suffix")]

    suffix = suffix_by_country.get(country_text)
    if suffix:
        return [(f"{ticker}{suffix}", "country_suffix")]
    if yahoo_ticker and "." in yahoo_ticker:
        return [(yahoo_ticker, "yahoo_suffix")]
    return []


def candidate_refinitiv_rics(row: pd.Series | dict[str, object]) -> list[dict[str, object]]:
    """Return ordered Refinitiv RIC candidates for one listing row."""

    data = row if isinstance(row, dict) else row.to_dict()
    country = _clean_text(data.get("country"))
    exchange = _clean_text(data.get("exchange"))
    ticker = _clean_symbol(data.get("ticker")) or ""
    yahoo_ticker = _clean_symbol(data.get("yahoo_ticker"))
    supplied_ric = _clean_symbol(data.get("refinitiv_ric"))
    symbol_id = _source_symbol_id(data)
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(ric: str | None, method: str) -> None:
        if not ric or ric in seen:
            return
        seen.add(ric)
        candidates.append(
            {
                "source_symbol_id": symbol_id,
                "country": country,
                "exchange": exchange,
                "ticker": ticker,
                "yahoo_ticker": yahoo_ticker,
                "name": data.get("name"),
                "currency": data.get("currency"),
                "refinitiv_ric": ric,
                "candidate_rank": len(candidates) + 1,
                "candidate_method": method,
                "listing_source": data.get("source"),
                "symbol_map_source": data.get("symbol_map_source", "listing"),
            }
        )

    alias = CURATED_REFINITIV_RIC_ALIASES.get((country, ticker.upper()))
    add(alias, "curated_alias")
    add(supplied_ric, "supplied_ric")
    for ric, method in _candidate_suffix(country, exchange, ticker, yahoo_ticker):
        add(ric, method)
    return candidates


def sample_symbol_map_listings(
    listings: pd.DataFrame,
    sample_per_market: int = 5,
    max_symbols: int | None = None,
    include_curated_seeds: bool = True,
) -> pd.DataFrame:
    frames = []
    if include_curated_seeds:
        frames.append(default_refinitiv_symbol_seed_listings())
    if listings is not None and not listings.empty:
        data = listings.copy()
        group_cols = [col for col in ("country", "exchange") if col in data.columns]
        if group_cols:
            sampled = (
                data.sort_values(group_cols + ["ticker"])
                .groupby(group_cols, dropna=False, group_keys=False)
                .head(sample_per_market)
                .copy()
            )
        else:
            sampled = data.head(sample_per_market).copy()
        sampled["symbol_map_source"] = sampled.get("symbol_map_source", "listing_sample")
        frames.append(sampled)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_source_symbol_id"] = combined.apply(_source_symbol_id, axis=1)
    combined = combined.drop_duplicates("_source_symbol_id", keep="first").drop(columns=["_source_symbol_id"])
    if max_symbols is not None and max_symbols > 0 and len(combined) > max_symbols:
        seed_mask = combined["symbol_map_source"].eq("curated_seed") if "symbol_map_source" in combined.columns else pd.Series(False, index=combined.index)
        seeds = combined[seed_mask].head(max_symbols)
        remaining = combined[~seed_mask].head(max(0, max_symbols - len(seeds)))
        combined = pd.concat([seeds, remaining], ignore_index=True, sort=False)
    return combined.reset_index(drop=True)


def build_refinitiv_symbol_candidates(listings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in listings.iterrows():
        rows.extend(candidate_refinitiv_rics(row))
    return pd.DataFrame(rows)


def _history_stats(history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return {
            "rows": 0,
            "non_null_values": 0,
            "coverage": 0.0,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "last_value": pd.NA,
            "returned_columns": "",
        }
    non_null = history.notna()
    any_row = non_null.any(axis=1)
    values = pd.Series(history.to_numpy().ravel()).dropna()
    return {
        "rows": int(history.shape[0]),
        "non_null_values": int(non_null.sum().sum()),
        "coverage": float(non_null.sum().sum() / max(1, history.shape[0] * history.shape[1])),
        "first_available": history.index[any_row].min() if any_row.any() else pd.NaT,
        "last_available": history.index[any_row].max() if any_row.any() else pd.NaT,
        "last_value": values.iloc[-1] if not values.empty else pd.NA,
        "returned_columns": "|".join(str(col) for col in history.columns),
    }


def validate_refinitiv_symbol_candidates(
    ld: Any,
    candidates: pd.DataFrame,
    close_fields: tuple[str, ...] = DEFAULT_SYMBOL_MAP_CLOSE_FIELDS,
    interval: str = "1D",
    count: int = 5,
    sleep_seconds: float = 0.05,
    progress_every: int = 25,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    total = len(candidates)
    for idx, candidate in enumerate(candidates.itertuples(index=False), start=1):
        data = candidate._asdict()
        best: dict[str, object] | None = None
        errors: list[str] = []
        for field in close_fields:
            try:
                history = ld.get_history(universe=[data["refinitiv_ric"]], fields=[field], interval=interval, count=count)
                stats = _history_stats(history)
                if stats["non_null_values"] > 0:
                    best = {**stats, "validated_close_field": field, "error": ""}
                    break
                errors.append(f"{field}: no data")
            except Exception as exc:  # pragma: no cover - live vendor state
                errors.append(f"{field}: {type(exc).__name__}: {exc}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if best is None:
            best = {
                **_history_stats(pd.DataFrame()),
                "validated_close_field": pd.NA,
                "error": " | ".join(errors),
            }
        status = "validated_price_history" if int(best["non_null_values"]) > 0 else "no_history_for_candidate_ric"
        rows.append({**data, **best, "interval": interval, "count": count, "status": status})
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"Validated {idx}/{total}: {data['refinitiv_ric']} -> {status}", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def select_validated_refinitiv_symbol_map(validations: pd.DataFrame) -> pd.DataFrame:
    if validations.empty:
        return pd.DataFrame()
    data = validations.copy()
    data["is_valid"] = data["status"].eq("validated_price_history")
    data = data.sort_values(
        ["source_symbol_id", "is_valid", "candidate_rank", "coverage", "non_null_values"],
        ascending=[True, False, True, False, False],
    )
    selected = data.drop_duplicates("source_symbol_id", keep="first").copy()
    selected["refinitiv_symbol_status"] = selected["status"]
    selected["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    keep_cols = [
        "source_symbol_id",
        "country",
        "exchange",
        "ticker",
        "yahoo_ticker",
        "name",
        "currency",
        "refinitiv_ric",
        "refinitiv_symbol_status",
        "candidate_rank",
        "candidate_method",
        "symbol_map_source",
        "validated_close_field",
        "rows",
        "non_null_values",
        "coverage",
        "first_available",
        "last_available",
        "interval",
        "count",
        "selected_at_utc",
    ]
    keep_cols = [col for col in keep_cols if col in selected.columns]
    return selected[keep_cols].sort_values(["country", "exchange", "ticker"]).reset_index(drop=True)


def write_refinitiv_symbol_map_artifacts(
    artifact_root: str | Path,
    run_id: str,
    candidates: pd.DataFrame,
    validations: pd.DataFrame,
    session_name: str | None,
    interval: str,
    count: int,
) -> RefinitivSymbolMapArtifact:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    symbol_map = select_validated_refinitiv_symbol_map(validations)
    candidate_path = artifact_dir / "symbol_candidates.csv"
    validation_path = artifact_dir / "symbol_validation.csv"
    symbol_map_path = artifact_dir / "symbol_map.csv"
    manifest_path = artifact_dir / "manifest.json"
    candidates.to_csv(candidate_path, index=False)
    validations.to_csv(validation_path, index=False)
    symbol_map.to_csv(symbol_map_path, index=False)
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "session_name": session_name,
        "interval": interval,
        "count": count,
        "candidate_rows": int(len(candidates)),
        "validation_rows": int(len(validations)),
        "symbol_map_rows": int(len(symbol_map)),
        "validated_symbols": int(symbol_map["refinitiv_symbol_status"].eq("validated_price_history").sum())
        if "refinitiv_symbol_status" in symbol_map
        else 0,
        "unresolved_symbols": int(symbol_map["refinitiv_symbol_status"].ne("validated_price_history").sum())
        if "refinitiv_symbol_status" in symbol_map
        else 0,
        "candidate_path": str(candidate_path),
        "validation_path": str(validation_path),
        "symbol_map_path": str(symbol_map_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivSymbolMapArtifact(
        artifact_dir=artifact_dir,
        candidate_path=candidate_path,
        validation_path=validation_path,
        symbol_map_path=symbol_map_path,
        manifest_path=manifest_path,
        candidates=candidates,
        validations=validations,
        symbol_map=symbol_map,
    )
