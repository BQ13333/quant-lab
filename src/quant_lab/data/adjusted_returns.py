from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_COUNTRY_PRICE_CURRENCY: dict[str, str] = {
    "Indonesia": "IDR",
    "Malaysia": "MYR",
    "Singapore": "SGD",
    "United States": "USD",
}

NORMALIZED_EVENT_COLUMNS = [
    "event_key",
    "ticker",
    "country",
    "event_family",
    "event_type",
    "known_at_date",
    "price_event_date",
    "ex_trade_date",
    "cash_amount",
    "cash_currency",
    "cash_currency_source",
    "adjustment_factor",
    "source_route",
    "alpha_usable",
]

RECONCILIATION_COLUMNS = [
    "event_key",
    "ticker",
    "country",
    "event_family",
    "price_event_date",
    "prev_trade_date",
    "ex_trade_date",
    "prev_close",
    "ex_close",
    "raw_return",
    "price_ratio",
    "factor_reflected_status",
    "cash_yield",
    "applied_cash",
    "applied_factor",
    "reconciliation_status",
    "quarantine_reason",
]

DAILY_ADJUSTED_RETURN_COLUMNS = [
    "date",
    "ticker",
    "close",
    "price_return_local",
    "total_return_local_gross",
    "total_return_local_net",
    "adj_close_total_return_gross",
    "adj_close_total_return_net",
    "adjustment_status",
    "source_event_keys",
]


@dataclass(frozen=True)
class AdjustedReturnReconstructionArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    normalized_events_path: Path
    reconciliation_path: Path
    daily_adjusted_returns_path: Path
    quality_path: Path
    normalized_events: pd.DataFrame
    reconciliation: pd.DataFrame
    daily_adjusted_returns: pd.DataFrame
    quality: pd.DataFrame


def write_adjusted_return_reconstruction_artifacts(
    daily_features_path: str | Path,
    corporate_action_event_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    *,
    fx_rates_path: str | Path | None = None,
    max_match_lag_days: int = 7,
    factor_reflection_tolerance: float = 0.15,
    min_matched_event_share: float = 0.99,
    max_quarantine_share: float = 0.0,
    country_price_currency: Mapping[str, str] | None = None,
) -> AdjustedReturnReconstructionArtifacts:
    prices = pd.read_parquet(daily_features_path)
    prices = prepare_daily_prices(prices, country_price_currency=country_price_currency)
    raw_events = read_corporate_action_events(corporate_action_event_dir)
    normalized_events = normalize_corporate_action_events(
        raw_events,
        prices=prices,
        max_match_lag_days=max_match_lag_days,
    )
    fx_rates = pd.read_parquet(fx_rates_path) if fx_rates_path is not None and Path(fx_rates_path).exists() else None
    reconciliation = build_event_price_reconciliation(
        normalized_events,
        prices,
        fx_rates=fx_rates,
        max_match_lag_days=max_match_lag_days,
        factor_reflection_tolerance=factor_reflection_tolerance,
    )
    daily_adjusted_returns = build_daily_adjusted_returns(prices, reconciliation)
    quality = build_reconciliation_quality(normalized_events, reconciliation)

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    normalized_events_path = artifact_dir / "corporate_action_events_normalized.parquet"
    reconciliation_path = artifact_dir / "event_price_reconciliation.parquet"
    daily_adjusted_returns_path = artifact_dir / "daily_adjusted_returns.parquet"
    quality_path = artifact_dir / "reconciliation_quality.csv"
    manifest_path = artifact_dir / "manifest.json"

    normalized_events.to_parquet(normalized_events_path, index=False)
    reconciliation.to_parquet(reconciliation_path, index=False)
    daily_adjusted_returns.to_parquet(daily_adjusted_returns_path, index=False)
    quality.to_csv(quality_path, index=False)

    manifest = build_reconstruction_manifest(
        run_id=run_id,
        artifact_dir=artifact_dir,
        normalized_events=normalized_events,
        reconciliation=reconciliation,
        daily_adjusted_returns=daily_adjusted_returns,
        quality=quality,
        max_match_lag_days=max_match_lag_days,
        factor_reflection_tolerance=factor_reflection_tolerance,
        min_matched_event_share=min_matched_event_share,
        max_quarantine_share=max_quarantine_share,
        paths={
            "corporate_action_events_normalized": normalized_events_path,
            "event_price_reconciliation": reconciliation_path,
            "daily_adjusted_returns": daily_adjusted_returns_path,
            "reconciliation_quality": quality_path,
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AdjustedReturnReconstructionArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        normalized_events_path=normalized_events_path,
        reconciliation_path=reconciliation_path,
        daily_adjusted_returns_path=daily_adjusted_returns_path,
        quality_path=quality_path,
        normalized_events=normalized_events,
        reconciliation=reconciliation,
        daily_adjusted_returns=daily_adjusted_returns,
        quality=quality,
    )


def read_corporate_action_events(corporate_action_event_dir: str | Path) -> pd.DataFrame:
    root = Path(corporate_action_event_dir)
    if root.is_file():
        return pd.read_parquet(root) if root.suffix == ".parquet" else pd.read_csv(root)
    event_root = root / "events" if (root / "events").exists() else root
    frames = [pd.read_parquet(path) for path in sorted(event_root.rglob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_daily_prices(
    prices: pd.DataFrame,
    *,
    country_price_currency: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    data = prices.copy()
    for column in ("date", "ticker", "country", "currency", "close"):
        if column not in data.columns:
            data[column] = pd.NA
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].astype("string")
    data["country"] = data["country"].astype("string")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["price_currency"] = _clean_text_series(data["currency"])
    currency_map = dict(DEFAULT_COUNTRY_PRICE_CURRENCY)
    if country_price_currency:
        currency_map.update({str(key): str(value) for key, value in country_price_currency.items()})
    missing_currency = data["price_currency"].isna() | data["price_currency"].astype(str).str.strip().eq("")
    data.loc[missing_currency, "price_currency"] = data.loc[missing_currency, "country"].map(currency_map)
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_corporate_action_events(
    events: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
    max_match_lag_days: int = 7,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=NORMALIZED_EVENT_COLUMNS)
    data = events.copy()
    data["event_family"] = _clean_text_series(data.get("event_family", pd.Series(index=data.index, dtype=object))).str.lower()
    data["ticker"] = _coalesce_text(
        data,
        ["ticker", "requested_ric", "Instrument", "refinitiv_ric"],
    )
    data["country"] = _coalesce_text(data, ["country"])

    is_dividend = data["event_family"].eq("dividend")
    is_capital = data["event_family"].eq("capital_change")

    dividend_known = _coalesce_dates(data, ["Dividend Announcement Date", "Div Announcement Date", "TR.DivAnnouncementDate", "known_at_date"])
    dividend_ex = _coalesce_dates(data, ["Dividend Ex Date", "Div Ex Date", "TR.DivExDate"])
    capital_known = _coalesce_dates(
        data,
        ["Capital Change Announcement Date", "CA Announcement Date", "TR.CAAnnouncementDate", "known_at_date"],
    )
    capital_ex = _coalesce_dates(data, ["Capital Change Ex Date", "CA Ex Date", "TR.CAExDate"])
    capital_effective = _coalesce_dates(data, ["Capital Change Effective Date", "CA Effective Date", "TR.CAEffectiveDate"])

    known_at_date = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    known_at_date.loc[is_dividend] = dividend_known.loc[is_dividend]
    known_at_date.loc[is_capital] = capital_known.loc[is_capital]
    known_at_date = known_at_date.fillna(_coalesce_dates(data, ["known_at_date"]))

    price_event_date = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    price_event_date.loc[is_dividend] = dividend_ex.loc[is_dividend]
    price_event_date.loc[is_capital] = capital_ex.loc[is_capital].fillna(capital_effective.loc[is_capital])

    event_type = pd.Series(pd.NA, index=data.index, dtype="object")
    event_type.loc[is_dividend] = _coalesce_text(data, ["Dividend Type", "Div Type", "TR.DivType", "field_role"]).loc[is_dividend]
    event_type.loc[is_capital] = _coalesce_text(
        data,
        ["Capital Change Event Type", "CA Event Type", "TR.CAEventType", "field_role"],
    ).loc[is_capital]
    event_type = _clean_text_series(event_type).fillna(data["event_family"])

    cash_amount = _coalesce_numeric(
        data,
        [
            "Net Dividend Amount",
            "Dividend Net Amount",
            "Net Cash Amount",
            "Gross Dividend Amount",
            "Dividend Amount",
            "Cash Amount",
            "TR.DivUnadjustedGross",
        ],
    )
    cash_amount.loc[~is_dividend] = np.nan
    cash_currency = _coalesce_text(
        data,
        ["Dividend Currency", "Div Currency", "Cash Currency", "TR.DivCurrency", "cash_currency"],
    )
    cash_currency_source = pd.Series(pd.NA, index=data.index, dtype="object")
    cash_currency_source.loc[is_dividend & cash_currency.notna()] = "source_event_field"
    inferred_cash_currency = data["country"].map(DEFAULT_COUNTRY_PRICE_CURRENCY)
    infer_cash_currency = is_dividend & cash_currency.isna() & inferred_cash_currency.notna()
    cash_currency.loc[infer_cash_currency] = inferred_cash_currency.loc[infer_cash_currency]
    cash_currency_source.loc[infer_cash_currency] = "country_price_currency_policy"
    cash_currency.loc[~is_dividend] = pd.NA
    cash_currency_source.loc[~is_dividend] = pd.NA

    adjustment_factor = _coalesce_numeric(
        data,
        [
            "Adjustment Factor",
            "Split Factor",
            "Capital Change Adjustment Factor",
            "TR.CAAdjustmentFactor",
            "TR.SplitFactor",
            "adjustment_factor",
        ],
    )
    adjustment_factor.loc[~is_capital] = np.nan

    normalized = pd.DataFrame(
        {
            "event_key": _coalesce_text(data, ["event_key"]),
            "ticker": data["ticker"],
            "country": data["country"],
            "event_family": data["event_family"],
            "event_type": event_type,
            "known_at_date": known_at_date,
            "price_event_date": price_event_date,
            "ex_trade_date": pd.NaT,
            "cash_amount": cash_amount,
            "cash_currency": cash_currency,
            "cash_currency_source": cash_currency_source,
            "adjustment_factor": adjustment_factor,
            "source_route": _coalesce_text(data, ["source_route"]).fillna("unknown"),
            "alpha_usable": known_at_date.notna() & price_event_date.notna() & known_at_date.le(price_event_date),
        }
    )
    missing_key = normalized["event_key"].isna() | normalized["event_key"].astype(str).str.strip().eq("")
    normalized.loc[missing_key, "event_key"] = [
        _stable_event_key(row)
        for row in normalized.loc[missing_key, ["ticker", "event_family", "known_at_date", "price_event_date", "cash_amount", "adjustment_factor"]]
        .to_dict("records")
    ]

    if prices is not None and not prices.empty:
        normalized["ex_trade_date"] = map_events_to_trading_days(
            normalized,
            prices,
            max_match_lag_days=max_match_lag_days,
        )

    for column in ("known_at_date", "price_event_date", "ex_trade_date"):
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce").dt.normalize()
    return normalized[NORMALIZED_EVENT_COLUMNS].sort_values(["country", "ticker", "price_event_date", "event_key"]).reset_index(drop=True)


def map_events_to_trading_days(events: pd.DataFrame, prices: pd.DataFrame, *, max_match_lag_days: int = 7) -> pd.Series:
    if events.empty:
        return pd.Series(dtype="datetime64[ns]")
    price_groups = _valid_price_groups(prices)
    mapped: list[pd.Timestamp | pd.NaT] = []
    for event in events.itertuples(index=False):
        price_event_date = pd.to_datetime(getattr(event, "price_event_date", pd.NaT), errors="coerce")
        ticker = str(getattr(event, "ticker", ""))
        group = price_groups.get(ticker)
        match = _match_trade_date(group, price_event_date, max_match_lag_days=max_match_lag_days)
        mapped.append(match["ex_trade_date"])
    return pd.Series(mapped, index=events.index, dtype="datetime64[ns]")


def build_event_price_reconciliation(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    fx_rates: pd.DataFrame | None = None,
    max_match_lag_days: int = 7,
    factor_reflection_tolerance: float = 0.15,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=RECONCILIATION_COLUMNS)
    price_groups = _valid_price_groups(prices)
    price_currency = _price_currency_lookup(prices)
    rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        event_dict = event._asdict()
        ticker = _clean_scalar_text(event_dict.get("ticker")) or ""
        price_event_date = pd.to_datetime(event_dict.get("price_event_date"), errors="coerce")
        group = price_groups.get(ticker)
        match = _match_trade_date(group, price_event_date, max_match_lag_days=max_match_lag_days)
        prev_close = match["prev_close"]
        ex_close = match["ex_close"]
        price_ratio = _safe_divide(ex_close, prev_close)
        raw_return = price_ratio - 1.0 if pd.notna(price_ratio) else pd.NA
        event_family = _clean_scalar_text(event_dict.get("event_family")) or ""

        factor_status = "not_applicable"
        applied_cash: float | pd.NA = 0.0
        applied_factor: float | pd.NA = 1.0
        cash_yield: float | pd.NA = pd.NA
        reconciliation_status = "diagnostic_no_adjustment"
        quarantine_reason = ""

        missing_event_payload = (
            pd.isna(price_event_date)
            and pd.isna(event_dict.get("cash_amount"))
            and pd.isna(event_dict.get("adjustment_factor"))
            and pd.isna(event_dict.get("known_at_date"))
        )
        if missing_event_payload:
            reconciliation_status = "diagnostic_no_event"
            quarantine_reason = "missing_price_event_date"
        elif pd.isna(price_event_date):
            reconciliation_status = "quarantined"
            quarantine_reason = "missing_price_event_date"
        elif pd.isna(match["ex_trade_date"]):
            reconciliation_status = "quarantined"
            quarantine_reason = "no_trade_date_within_lag"
        elif pd.isna(prev_close) or pd.isna(ex_close) or float(prev_close) <= 0:
            reconciliation_status = "quarantined"
            quarantine_reason = "missing_price_window"
        elif event_family == "dividend":
            applied_cash, cash_yield, reconciliation_status, quarantine_reason = _reconcile_dividend_cash(
                event_dict=event_dict,
                price_currency=price_currency.get((ticker, match["ex_trade_date"])),
                price_event_date=match["ex_trade_date"],
                prev_close=prev_close,
                fx_rates=fx_rates,
            )
        elif event_family == "capital_change":
            factor_status, applied_factor, reconciliation_status, quarantine_reason = _reconcile_capital_factor(
                adjustment_factor=event_dict.get("adjustment_factor"),
                price_ratio=price_ratio,
                tolerance=factor_reflection_tolerance,
            )

        rows.append(
            {
                "event_key": event_dict.get("event_key"),
                "ticker": ticker,
                "country": event_dict.get("country"),
                "event_family": event_family,
                "price_event_date": price_event_date,
                "prev_trade_date": match["prev_trade_date"],
                "ex_trade_date": match["ex_trade_date"],
                "prev_close": prev_close,
                "ex_close": ex_close,
                "raw_return": raw_return,
                "price_ratio": price_ratio,
                "factor_reflected_status": factor_status,
                "cash_yield": cash_yield,
                "applied_cash": applied_cash,
                "applied_factor": applied_factor,
                "reconciliation_status": reconciliation_status,
                "quarantine_reason": quarantine_reason,
            }
        )
    out = pd.DataFrame(rows, columns=RECONCILIATION_COLUMNS)
    for column in ("prev_trade_date", "ex_trade_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    return out


def build_daily_adjusted_returns(prices: pd.DataFrame, reconciliation: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=DAILY_ADJUSTED_RETURN_COLUMNS)
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].astype("string")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    data["prev_close_valid"] = data.groupby("ticker", dropna=False)["close"].transform(lambda values: values.ffill().shift(1))
    data["price_return_local"] = data["close"] / data["prev_close_valid"] - 1.0
    data.loc[data["close"].isna() | data["prev_close_valid"].isna() | data["prev_close_valid"].eq(0), "price_return_local"] = np.nan

    event_daily = _aggregate_reconciled_events(reconciliation)
    if event_daily.empty:
        data["applied_cash"] = 0.0
        data["applied_factor"] = 1.0
        data["has_event"] = False
        data["has_quarantine"] = False
        data["source_event_keys"] = ""
    else:
        data = data.merge(event_daily, on=["date", "ticker"], how="left")
        data["applied_cash"] = pd.to_numeric(data["applied_cash"], errors="coerce").fillna(0.0)
        data["applied_factor"] = pd.to_numeric(data["applied_factor"], errors="coerce").fillna(1.0)
        data["has_event"] = data["has_event"].eq(True)
        data["has_quarantine"] = data["has_quarantine"].eq(True)
        data["source_event_keys"] = data["source_event_keys"].fillna("")

    adjusted_return = (data["close"] + data["applied_cash"]) / data["prev_close_valid"] / data["applied_factor"] - 1.0
    can_adjust = (
        data["has_event"]
        & data["close"].notna()
        & data["prev_close_valid"].notna()
        & data["prev_close_valid"].ne(0)
        & data["applied_factor"].notna()
        & data["applied_factor"].ne(0)
    )
    data["total_return_local_gross"] = data["price_return_local"]
    data["total_return_local_net"] = data["price_return_local"]
    data.loc[can_adjust, "total_return_local_gross"] = adjusted_return.loc[can_adjust]
    data.loc[can_adjust, "total_return_local_net"] = adjusted_return.loc[can_adjust]

    data["adj_close_total_return_gross"] = _total_return_adjusted_close(data, "total_return_local_gross")
    data["adj_close_total_return_net"] = _total_return_adjusted_close(data, "total_return_local_net")
    data["adjustment_status"] = _daily_adjustment_status(data)

    return data[DAILY_ADJUSTED_RETURN_COLUMNS].reset_index(drop=True)


def build_reconciliation_quality(events: pd.DataFrame, reconciliation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "country",
        "event_family",
        "year",
        "event_rows",
        "valid_event_rows",
        "matched_event_rows",
        "applied_event_rows",
        "diagnostic_event_rows",
        "quarantined_event_rows",
        "currency_quarantine_rows",
        "ambiguous_factor_rows",
        "matched_event_share",
        "applied_event_share",
        "quarantine_share",
        "qa_status",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    merged = events.merge(reconciliation, on="event_key", how="left", suffixes=("_event", "_recon"))
    merged["country"] = _coalesce_text(merged, ["country_event", "country", "country_recon"])
    merged["event_family"] = _coalesce_text(merged, ["event_family_event", "event_family", "event_family_recon"])
    price_event_date = _coalesce_dates(
        merged,
        ["price_event_date_event", "price_event_date", "price_event_date_recon"],
    )
    ex_trade_date = _coalesce_dates(merged, ["ex_trade_date_recon", "ex_trade_date"])
    merged["year"] = price_event_date.dt.year.astype("Int64")
    merged["valid_event"] = price_event_date.notna()
    merged["matched_event"] = merged["valid_event"] & ex_trade_date.notna()
    merged["applied_event"] = merged["reconciliation_status"].astype(str).eq("applied")
    merged["diagnostic_event"] = merged["reconciliation_status"].astype(str).str.startswith("diagnostic")
    merged["quarantined_event"] = merged["reconciliation_status"].astype(str).eq("quarantined")
    merged["currency_quarantine"] = merged["quarantine_reason"].astype(str).isin(
        ["missing_cash_currency", "missing_price_currency", "missing_fx_for_dividend_currency"]
    )
    merged["ambiguous_factor"] = merged["factor_reflected_status"].astype(str).eq("ambiguous_factor_reflection")
    grouped = (
        merged.groupby(["country", "event_family", "year"], dropna=False)
        .agg(
            event_rows=("event_key", "size"),
            valid_event_rows=("valid_event", "sum"),
            matched_event_rows=("matched_event", "sum"),
            applied_event_rows=("applied_event", "sum"),
            diagnostic_event_rows=("diagnostic_event", "sum"),
            quarantined_event_rows=("quarantined_event", "sum"),
            currency_quarantine_rows=("currency_quarantine", "sum"),
            ambiguous_factor_rows=("ambiguous_factor", "sum"),
        )
        .reset_index()
    )
    grouped["matched_event_share"] = grouped["matched_event_rows"] / grouped["valid_event_rows"].replace(0, np.nan)
    grouped["applied_event_share"] = grouped["applied_event_rows"] / grouped["valid_event_rows"].replace(0, np.nan)
    grouped["quarantine_share"] = grouped["quarantined_event_rows"] / grouped["valid_event_rows"].replace(0, np.nan)
    grouped[["matched_event_share", "applied_event_share", "quarantine_share"]] = grouped[
        ["matched_event_share", "applied_event_share", "quarantine_share"]
    ].fillna(0.0)
    grouped["qa_status"] = np.where(
        grouped["quarantined_event_rows"].eq(0) & grouped["ambiguous_factor_rows"].eq(0),
        "pass",
        "diagnostic",
    )
    return grouped[columns].sort_values(["country", "event_family", "year"], na_position="last").reset_index(drop=True)


def build_reconstruction_manifest(
    *,
    run_id: str,
    artifact_dir: Path,
    normalized_events: pd.DataFrame,
    reconciliation: pd.DataFrame,
    daily_adjusted_returns: pd.DataFrame,
    quality: pd.DataFrame,
    max_match_lag_days: int,
    factor_reflection_tolerance: float,
    min_matched_event_share: float,
    max_quarantine_share: float,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    valid_event_rows = int(pd.to_datetime(normalized_events.get("price_event_date", pd.Series(dtype=object)), errors="coerce").notna().sum())
    matched_rows = int(reconciliation.get("ex_trade_date", pd.Series(dtype=object)).notna().sum())
    applied_adjustment = _applied_adjustment_mask(reconciliation)
    applied_adjustment_rows = int(applied_adjustment.sum())
    quarantined_rows = int(reconciliation.get("reconciliation_status", pd.Series(dtype=object)).astype(str).eq("quarantined").sum())
    ambiguous_factor_rows = int(reconciliation.get("factor_reflected_status", pd.Series(dtype=object)).astype(str).eq("ambiguous_factor_reflection").sum())
    returns_non_null = int(
        pd.to_numeric(daily_adjusted_returns.get("total_return_local_net", pd.Series(dtype=float)), errors="coerce").notna().sum()
    )
    matched_event_share = float(matched_rows / valid_event_rows) if valid_event_rows else 0.0
    quarantine_share = float(quarantined_rows / valid_event_rows) if valid_event_rows else 0.0
    ambiguous_factor_share = float(ambiguous_factor_rows / valid_event_rows) if valid_event_rows else 0.0
    rebuilt_returns_exist = bool(len(daily_adjusted_returns) > 0 and returns_non_null > 0 and applied_adjustment_rows > 0)

    diagnostic_reasons: list[str] = []
    if not rebuilt_returns_exist:
        diagnostic_reasons.append("rebuilt adjusted returns do not yet contain any applied cash/factor adjustments")
    if valid_event_rows == 0:
        diagnostic_reasons.append("no valid corporate-action price_event_date rows")
    if matched_event_share < min_matched_event_share:
        diagnostic_reasons.append(f"matched_event_share {matched_event_share:.4f} below threshold {min_matched_event_share:.4f}")
    if quarantine_share > max_quarantine_share:
        diagnostic_reasons.append(f"quarantine_share {quarantine_share:.4f} above threshold {max_quarantine_share:.4f}")
    if ambiguous_factor_share > max_quarantine_share:
        diagnostic_reasons.append(
            f"ambiguous_factor_share {ambiguous_factor_share:.4f} above threshold {max_quarantine_share:.4f}"
        )

    production_ready = rebuilt_returns_exist and not diagnostic_reasons
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "pass" if production_ready else "diagnostic",
        "production_adjusted_return_ready": bool(production_ready),
        "diagnostic_reason": "; ".join(diagnostic_reasons) if diagnostic_reasons else "",
        "event_rows": int(len(normalized_events)),
        "valid_event_rows": valid_event_rows,
        "matched_event_rows": matched_rows,
        "matched_event_share": matched_event_share,
        "applied_adjustment_rows": applied_adjustment_rows,
        "quarantined_event_rows": quarantined_rows,
        "quarantine_share": quarantine_share,
        "ambiguous_factor_rows": ambiguous_factor_rows,
        "ambiguous_factor_share": ambiguous_factor_share,
        "max_quarantine_share": float(max_quarantine_share),
        "daily_adjusted_return_rows": int(len(daily_adjusted_returns)),
        "non_null_total_return_rows": returns_non_null,
        "max_match_lag_days": int(max_match_lag_days),
        "factor_reflection_tolerance": float(factor_reflection_tolerance),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
        "quality_rows": int(len(quality)),
    }


def _reconcile_dividend_cash(
    *,
    event_dict: Mapping[str, Any],
    price_currency: str | None,
    price_event_date: pd.Timestamp | pd.NaT,
    prev_close: float | pd.NA,
    fx_rates: pd.DataFrame | None,
) -> tuple[float | pd.NA, float | pd.NA, str, str]:
    cash_amount = _as_float(event_dict.get("cash_amount"))
    cash_currency = _clean_scalar_text(event_dict.get("cash_currency"))
    price_currency = _clean_scalar_text(price_currency)
    if cash_amount is None or cash_amount <= 0:
        return 0.0, pd.NA, "quarantined", "missing_cash_amount"
    if not cash_currency:
        return 0.0, pd.NA, "quarantined", "missing_cash_currency"
    if not price_currency:
        return 0.0, pd.NA, "quarantined", "missing_price_currency"

    converted_cash = cash_amount
    if cash_currency != price_currency:
        fx_rate = _lookup_fx_rate(
            fx_rates=fx_rates,
            date=price_event_date,
            from_currency=cash_currency,
            to_currency=price_currency,
        )
        if fx_rate is None:
            return 0.0, pd.NA, "quarantined", "missing_fx_for_dividend_currency"
        converted_cash = cash_amount * fx_rate

    cash_yield = _safe_divide(converted_cash, prev_close)
    return float(converted_cash), cash_yield, "applied", ""


def _reconcile_capital_factor(
    *,
    adjustment_factor: Any,
    price_ratio: float | pd.NA,
    tolerance: float,
) -> tuple[str, float | pd.NA, str, str]:
    factor = _as_float(adjustment_factor)
    ratio = _as_float(price_ratio)
    if factor is None or factor <= 0 or math.isclose(factor, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return "unit_or_missing_factor", 1.0, "diagnostic_no_adjustment", ""
    if ratio is None or ratio <= 0:
        return "ambiguous_factor_reflection", 1.0, "quarantined", "missing_price_ratio_for_factor"
    if _relative_close(ratio, factor, tolerance):
        return "raw_close_reflects_factor", float(factor), "applied", ""
    if _relative_close(ratio, 1.0, tolerance):
        return "vendor_close_likely_already_adjusted", 1.0, "diagnostic_no_adjustment", ""
    return "ambiguous_factor_reflection", 1.0, "quarantined", "ambiguous_factor_reflection"


def _aggregate_reconciled_events(reconciliation: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "ticker", "applied_cash", "applied_factor", "has_event", "has_quarantine", "source_event_keys"]
    if reconciliation.empty:
        return pd.DataFrame(columns=columns)
    data = reconciliation.copy()
    if "ticker" not in data.columns:
        data["ticker"] = pd.NA
    data["date"] = pd.to_datetime(data["ex_trade_date"], errors="coerce").dt.normalize()
    data = data[data["date"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["applied_cash"] = pd.to_numeric(data["applied_cash"], errors="coerce").fillna(0.0)
    data["applied_factor"] = pd.to_numeric(data["applied_factor"], errors="coerce").fillna(1.0)
    data["is_quarantined"] = data["reconciliation_status"].astype(str).eq("quarantined")
    grouped = (
        data.groupby(["date", "ticker"], dropna=False)
        .agg(
            applied_cash=("applied_cash", "sum"),
            applied_factor=("applied_factor", _product_non_null),
            has_event=("event_key", lambda values: len(values) > 0),
            has_quarantine=("is_quarantined", "any"),
            source_event_keys=("event_key", lambda values: ";".join(sorted({str(value) for value in values if pd.notna(value)}))),
        )
        .reset_index()
    )
    return grouped[columns]


def _valid_price_groups(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].astype("string")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data[data["date"].notna() & data["ticker"].notna() & data["close"].notna()].copy()
    return {
        str(ticker): group.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for ticker, group in data.groupby("ticker", sort=False, dropna=False)
    }


def _match_trade_date(group: pd.DataFrame | None, price_event_date: pd.Timestamp | pd.NaT, *, max_match_lag_days: int) -> dict[str, Any]:
    if pd.isna(price_event_date) or group is None or group.empty:
        return _empty_trade_match()
    event_date = pd.Timestamp(price_event_date).normalize()
    after = group[group["date"].ge(event_date)].head(1)
    if after.empty:
        return _empty_trade_match()
    ex_trade_date = pd.Timestamp(after["date"].iloc[0]).normalize()
    if int((ex_trade_date - event_date).days) > max_match_lag_days:
        return _empty_trade_match()
    before = group[group["date"].lt(ex_trade_date)].tail(1)
    prev_trade_date = pd.Timestamp(before["date"].iloc[0]).normalize() if not before.empty else pd.NaT
    prev_close = pd.to_numeric(before["close"], errors="coerce").iloc[0] if not before.empty else pd.NA
    ex_close = pd.to_numeric(after["close"], errors="coerce").iloc[0]
    return {
        "prev_trade_date": prev_trade_date,
        "ex_trade_date": ex_trade_date,
        "prev_close": prev_close,
        "ex_close": ex_close,
    }


def _empty_trade_match() -> dict[str, Any]:
    return {
        "prev_trade_date": pd.NaT,
        "ex_trade_date": pd.NaT,
        "prev_close": pd.NA,
        "ex_close": pd.NA,
    }


def _price_currency_lookup(prices: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], str]:
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].astype("string")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    if "price_currency" not in data.columns:
        data["price_currency"] = _clean_text_series(data.get("currency", pd.Series(index=data.index, dtype=object)))
    data = data[data["date"].notna() & data["ticker"].notna() & data["close"].notna()]
    return {
        (str(row.ticker), pd.Timestamp(row.date).normalize()): _clean_scalar_text(row.price_currency)
        for row in data[["ticker", "date", "price_currency"]].itertuples(index=False)
    }


def _lookup_fx_rate(
    *,
    fx_rates: pd.DataFrame | None,
    date: pd.Timestamp | pd.NaT,
    from_currency: str,
    to_currency: str,
) -> float | None:
    if fx_rates is None or fx_rates.empty or pd.isna(date):
        return None
    data = fx_rates.copy()
    if {"date", "from_currency", "to_currency", "rate"}.issubset(data.columns):
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        data["from_currency"] = _clean_text_series(data["from_currency"])
        data["to_currency"] = _clean_text_series(data["to_currency"])
        data["rate"] = pd.to_numeric(data["rate"], errors="coerce")
        eligible = data[
            data["date"].le(pd.Timestamp(date).normalize())
            & data["from_currency"].eq(from_currency)
            & data["to_currency"].eq(to_currency)
            & data["rate"].gt(0)
        ].sort_values("date")
        if not eligible.empty:
            return float(eligible["rate"].iloc[-1])
        inverse = data[
            data["date"].le(pd.Timestamp(date).normalize())
            & data["from_currency"].eq(to_currency)
            & data["to_currency"].eq(from_currency)
            & data["rate"].gt(0)
        ].sort_values("date")
        if not inverse.empty:
            return float(1.0 / inverse["rate"].iloc[-1])
    return None


def _total_return_adjusted_close(data: pd.DataFrame, return_column: str) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby("ticker", sort=False, dropna=False):
        current = np.nan
        for idx, row in group.iterrows():
            close = row["close"]
            if pd.isna(close):
                continue
            ret = row[return_column]
            if pd.isna(current):
                current = float(close)
            elif pd.notna(ret):
                current *= 1.0 + float(ret)
            else:
                current = np.nan
            values.at[idx] = current
    return values


def _daily_adjustment_status(data: pd.DataFrame) -> pd.Series:
    status = pd.Series("raw_price_return", index=data.index, dtype="object")
    status.loc[data["close"].isna()] = "missing_close"
    status.loc[data["close"].notna() & data["prev_close_valid"].isna()] = "first_valid_close"
    event_mask = data["has_event"].fillna(False).astype(bool)
    cash_mask = pd.to_numeric(data["applied_cash"], errors="coerce").fillna(0.0).ne(0.0)
    factor_mask = pd.to_numeric(data["applied_factor"], errors="coerce").fillna(1.0).sub(1.0).abs().gt(1e-12)
    quarantine_mask = data["has_quarantine"].fillna(False).astype(bool)
    status.loc[event_mask & cash_mask & factor_mask] = "cash_factor_adjusted"
    status.loc[event_mask & cash_mask & ~factor_mask] = "cash_adjusted"
    status.loc[event_mask & ~cash_mask & factor_mask] = "factor_adjusted"
    status.loc[event_mask & quarantine_mask & ~(cash_mask | factor_mask)] = "quarantined_event_raw_return"
    status.loc[event_mask & quarantine_mask & (cash_mask | factor_mask)] = "partial_adjusted_with_quarantine"
    status.loc[event_mask & ~quarantine_mask & ~(cash_mask | factor_mask)] = "event_no_adjustment_needed"
    status.loc[data["close"].isna()] = "missing_close"
    status.loc[data["close"].notna() & data["prev_close_valid"].isna()] = "first_valid_close"
    return status


def _applied_adjustment_mask(reconciliation: pd.DataFrame) -> pd.Series:
    if reconciliation.empty:
        return pd.Series(dtype=bool)
    applied_cash = pd.to_numeric(reconciliation.get("applied_cash", pd.Series(index=reconciliation.index, dtype=float)), errors="coerce").fillna(0.0)
    applied_factor = pd.to_numeric(reconciliation.get("applied_factor", pd.Series(index=reconciliation.index, dtype=float)), errors="coerce").fillna(1.0)
    return reconciliation.get("reconciliation_status", pd.Series(index=reconciliation.index, dtype=object)).astype(str).eq("applied") & (
        applied_cash.ne(0.0) | applied_factor.sub(1.0).abs().gt(1e-12)
    )


def _coalesce_dates(data: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    for column in candidates:
        if column not in data.columns:
            continue
        values = pd.to_datetime(data[column], errors="coerce").dt.normalize()
        out = out.fillna(values)
    return out


def _coalesce_text(data: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=data.index, dtype="object")
    for column in candidates:
        if column not in data.columns:
            continue
        values = _clean_text_series(data[column])
        out = out.where(out.notna(), values)
    return out


def _coalesce_numeric(data: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=data.index, dtype=float)
    for column in candidates:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _clean_text_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    out = series.astype("string").str.strip()
    out = out.mask(out.str.lower().isin(["", "nan", "none", "<na>", "nat"]))
    return out


def _clean_scalar_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return None
    return text


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_divide(numerator: Any, denominator: Any) -> float | pd.NA:
    num = _as_float(numerator)
    den = _as_float(denominator)
    if num is None or den is None or den == 0:
        return pd.NA
    return float(num / den)


def _relative_close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(abs(right), 1e-12)


def _product_non_null(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    return float(numeric.prod())


def _stable_event_key(row: Mapping[str, Any]) -> str:
    material = "|".join(str(row.get(key, "")) for key in ["ticker", "event_family", "known_at_date", "price_event_date", "cash_amount", "adjustment_factor"])
    return hashlib.sha1(material.encode("utf-8")).hexdigest()
