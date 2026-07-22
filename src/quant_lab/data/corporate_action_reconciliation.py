from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CorporateActionReconciliationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    event_match_path: Path
    quality_path: Path
    adjustment_policy_path: Path
    event_matches: pd.DataFrame
    quality: pd.DataFrame
    adjustment_policy: pd.DataFrame


def write_corporate_action_reconciliation_artifacts(
    daily_features_path: str | Path,
    corporate_action_event_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    max_match_lag_days: int = 7,
) -> CorporateActionReconciliationArtifacts:
    prices = pd.read_parquet(daily_features_path, columns=["date", "ticker", "country", "close", "adj_close"])
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    events = read_corporate_action_events(corporate_action_event_dir)
    event_matches = match_events_to_prices(events, prices, max_match_lag_days=max_match_lag_days)
    quality = build_reconciliation_quality(event_matches)
    adjustment_policy = build_adjustment_policy(event_matches)

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    event_match_path = artifact_dir / "event_price_matches.parquet"
    quality_path = artifact_dir / "reconciliation_quality.csv"
    adjustment_policy_path = artifact_dir / "adjustment_policy.csv"
    manifest_path = artifact_dir / "manifest.json"

    event_matches.to_parquet(event_match_path, index=False)
    quality.to_csv(quality_path, index=False)
    adjustment_policy.to_csv(adjustment_policy_path, index=False)

    matched_share = _safe_min(quality, "matched_event_share")
    valid_event_rows = int(pd.to_numeric(quality.get("valid_event_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    production_adjusted = bool(adjustment_policy["production_adjusted_return_ready"].all()) if not adjustment_policy.empty else False
    status = "pass" if production_adjusted else "diagnostic_reconciliation_only"
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": status,
        "valid_event_rows": valid_event_rows,
        "matched_event_share_min": float(matched_share),
        "production_adjusted_return_ready": production_adjusted,
        "max_match_lag_days": int(max_match_lag_days),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "event_price_matches": str(event_match_path.resolve()),
            "reconciliation_quality": str(quality_path.resolve()),
            "adjustment_policy": str(adjustment_policy_path.resolve()),
        },
        "warning": "Diagnostic alignment only. Production adjusted-return series is not ready until events are applied and reconciled to returns.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CorporateActionReconciliationArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        event_match_path=event_match_path,
        quality_path=quality_path,
        adjustment_policy_path=adjustment_policy_path,
        event_matches=event_matches,
        quality=quality,
        adjustment_policy=adjustment_policy,
    )


def read_corporate_action_events(corporate_action_event_dir: str | Path) -> pd.DataFrame:
    root = Path(corporate_action_event_dir)
    frames = [pd.read_parquet(path) for path in sorted((root / "events").rglob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def match_events_to_prices(events: pd.DataFrame, prices: pd.DataFrame, max_match_lag_days: int = 7) -> pd.DataFrame:
    columns = [
        "event_key",
        "ticker",
        "country",
        "event_family",
        "field_role",
        "event_date",
        "known_at_date",
        "matched_price_date",
        "match_lag_days",
        "price_before",
        "price_on_or_after",
        "raw_return_on_match",
        "adjustment_factor",
        "eligible_for_return_reconstruction",
        "alpha_usable",
        "match_status",
    ]
    if events.empty or prices.empty:
        return pd.DataFrame(columns=columns)
    events = events.copy()
    prices = prices.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["known_at_date"] = pd.to_datetime(events["known_at_date"], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    price_groups = {ticker: group.dropna(subset=["date"]).sort_values("date") for ticker, group in prices.groupby("ticker", sort=False)}
    rows = []
    for row in events.to_dict("records"):
        ticker = row.get("ticker") or row.get("requested_ric")
        event_date = row.get("event_date")
        group = price_groups.get(ticker)
        match = _match_single_event(group, event_date, max_match_lag_days=max_match_lag_days)
        adjustment_factor = _first_number(row, ["Adjustment Factor", "Split Factor"])
        rows.append(
            {
                "event_key": row.get("event_key"),
                "ticker": ticker,
                "country": row.get("country"),
                "event_family": row.get("event_family"),
                "field_role": row.get("field_role"),
                "event_date": event_date,
                "known_at_date": row.get("known_at_date"),
                "matched_price_date": match["matched_price_date"],
                "match_lag_days": match["match_lag_days"],
                "price_before": match["price_before"],
                "price_on_or_after": match["price_on_or_after"],
                "raw_return_on_match": match["raw_return_on_match"],
                "adjustment_factor": adjustment_factor,
                "eligible_for_return_reconstruction": bool(row.get("eligible_for_return_reconstruction")),
                "alpha_usable": bool(row.get("alpha_usable")),
                "match_status": match["match_status"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_reconciliation_quality(event_matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "country",
        "event_family",
        "event_rows",
        "valid_event_rows",
        "matched_event_rows",
        "alpha_usable_rows",
        "matched_event_share",
        "missing_date_rows",
        "unmatched_rows",
        "non_unit_adjustment_rows",
    ]
    if event_matches.empty:
        return pd.DataFrame(columns=columns)
    data = event_matches.copy()
    data["has_valid_event_date"] = pd.to_datetime(data["event_date"], errors="coerce").notna()
    data["is_matched"] = data["match_status"].astype(str).eq("matched")
    data["non_unit_adjustment"] = pd.to_numeric(data["adjustment_factor"], errors="coerce").fillna(1.0).sub(1.0).abs().gt(1e-9)
    quality = (
        data.groupby(["country", "event_family"], dropna=False)
        .agg(
            event_rows=("event_key", "size"),
            valid_event_rows=("has_valid_event_date", "sum"),
            matched_event_rows=("is_matched", "sum"),
            alpha_usable_rows=("alpha_usable", "sum"),
            missing_date_rows=("match_status", lambda values: int(pd.Series(values).eq("missing_event_date").sum())),
            unmatched_rows=("match_status", lambda values: int(pd.Series(values).eq("unmatched").sum())),
            non_unit_adjustment_rows=("non_unit_adjustment", "sum"),
        )
        .reset_index()
    )
    quality["matched_event_share"] = quality["matched_event_rows"] / quality["valid_event_rows"].replace(0, pd.NA)
    quality["matched_event_share"] = pd.to_numeric(quality["matched_event_share"], errors="coerce").fillna(0.0)
    return quality[columns].sort_values(["country", "event_family"]).reset_index(drop=True)


def build_adjustment_policy(event_matches: pd.DataFrame) -> pd.DataFrame:
    valid_rows = int(pd.to_datetime(event_matches.get("event_date", pd.Series(dtype=object)), errors="coerce").notna().sum()) if not event_matches.empty else 0
    matched_rows = int(event_matches.get("match_status", pd.Series(dtype=object)).astype(str).eq("matched").sum()) if not event_matches.empty else 0
    return pd.DataFrame(
        [
            {
                "policy_id": "event_price_alignment",
                "status": "pass" if valid_rows > 0 and matched_rows == valid_rows else "partial",
                "description": "Official events are aligned to nearest price date within tolerance.",
                "production_adjusted_return_ready": False,
            },
            {
                "policy_id": "adjusted_return_rebuild",
                "status": "not_started",
                "description": "Production adjusted-return series has not been rebuilt from official events.",
                "production_adjusted_return_ready": False,
            },
        ]
    )


def _match_single_event(group: pd.DataFrame | None, event_date: pd.Timestamp, max_match_lag_days: int) -> dict[str, object]:
    if pd.isna(event_date):
        return _empty_match("missing_event_date")
    if group is None or group.empty:
        return _empty_match("unmatched")
    after = group[group["date"].ge(event_date)].head(1)
    before = group[group["date"].lt(event_date)].tail(1)
    if after.empty:
        return _empty_match("unmatched")
    matched_date = after["date"].iloc[0]
    lag_days = int((matched_date - event_date).days)
    if lag_days > max_match_lag_days:
        return _empty_match("unmatched")
    price_after = pd.to_numeric(after["close"], errors="coerce").iloc[0]
    price_before = pd.to_numeric(before["close"], errors="coerce").iloc[0] if not before.empty else pd.NA
    raw_return = (price_after / price_before - 1.0) if pd.notna(price_before) and price_before != 0 else pd.NA
    return {
        "matched_price_date": matched_date,
        "match_lag_days": lag_days,
        "price_before": price_before,
        "price_on_or_after": price_after,
        "raw_return_on_match": raw_return,
        "match_status": "matched",
    }


def _empty_match(status: str) -> dict[str, object]:
    return {
        "matched_price_date": pd.NaT,
        "match_lag_days": pd.NA,
        "price_before": pd.NA,
        "price_on_or_after": pd.NA,
        "raw_return_on_match": pd.NA,
        "match_status": status,
    }


def _first_number(row: dict[str, object], columns: list[str]) -> float | pd.NA:
    for column in columns:
        if column in row:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return pd.NA


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else 0.0
