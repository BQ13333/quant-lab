from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


FUNDAMENTAL_FIELD_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("revenue", "TR.Revenue", 1, "general_revenue"),
    ("revenue", "TR.BankTotalRevenue", 2, "bank_total_revenue"),
    ("gross_profit", "TR.GrossProfit", 1, "general_gross_profit"),
    ("net_interest_income", "TR.NetInterestIncome", 1, "bank_net_interest_income"),
    ("interest_income", "TR.InterestIncomeBank", 1, "bank_interest_income"),
    ("eps", "TR.EPSActValue", 1, "actual_eps"),
    ("book_value", "TR.BookValuePerShare", 1, "book_value_per_share"),
)
FUNDAMENTAL_FIELDS: dict[str, str] = {
    role: source for role, source, priority, _variant in FUNDAMENTAL_FIELD_SPECS if priority == 1
}


@dataclass(frozen=True)
class JapanG3FundamentalsArtifacts:
    run_id: str
    artifact_dir: Path
    records_path: Path
    quality_path: Path
    manifest_path: Path
    records: pd.DataFrame
    quality: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_weekday_utc(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    next_day = dates + pd.offsets.BusinessDay(1)
    localized = next_day.dt.normalize().dt.tz_localize("Asia/Tokyo")
    return localized.dt.tz_convert("UTC")


def normalise_parameterized_metric_response(
    frame: pd.DataFrame,
    field_role: str,
    source_field: str,
    frequency: str,
    loaded_at_utc: str,
    source_priority: int = 1,
    metric_variant: str = "",
    dataset_id: str = "fundamentals_pit",
) -> pd.DataFrame:
    columns = list(frame.columns)
    if frame.empty or len(columns) < 5:
        return pd.DataFrame()

    instrument_col = next((column for column in columns if str(column).strip().lower() == "instrument"), columns[0])
    calc_col = next((column for column in columns if "calc date" in str(column).strip().lower()), None)
    period_end_col = next((column for column in columns if "period end" in str(column).strip().lower()), None)
    fiscal_col = next((column for column in columns if "financial period" in str(column).strip().lower()), None)
    excluded = {instrument_col, calc_col, period_end_col, fiscal_col}
    value_col = next((column for column in columns if column not in excluded), None)
    if calc_col is None or period_end_col is None or fiscal_col is None or value_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "ric": frame[instrument_col].astype("string"),
            "dataset_id": dataset_id,
            "field_role": field_role,
            "source_field": source_field,
            "source_priority": source_priority,
            "metric_variant": metric_variant,
            "frequency": frequency,
            "fiscal_period": frame[fiscal_col].astype("string"),
            "period_end_date": pd.to_datetime(frame[period_end_col], errors="coerce"),
            "calc_date": pd.to_datetime(frame[calc_col], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "source_loaded_at_utc": loaded_at_utc,
        }
    )
    result["available_time_utc"] = next_weekday_utc(result["calc_date"])
    result["known_at_policy"] = "next_weekday_00_jst_pending_official_jpx_calendar"
    result["pit_timestamp_violation"] = (
        result["calc_date"].isna()
        | result["period_end_date"].isna()
        | result["calc_date"].lt(result["period_end_date"])
        | result["available_time_utc"].isna()
    )
    return result.dropna(subset=["ric", "value"]).reset_index(drop=True)


def extract_japan_g3_fundamentals_qa(
    client: Any,
    rics: Sequence[str],
    start_date: str,
    end_date: str,
    frequencies: Sequence[str] = ("FY", "FQ"),
) -> pd.DataFrame:
    """Extract a bounded parameterized fundamentals QA shard with source calc dates."""

    instruments = list(dict.fromkeys(str(ric).strip() for ric in rics if str(ric).strip()))
    if not instruments:
        raise ValueError("at least one RIC is required")
    loaded_at = datetime.now(timezone.utc).isoformat()
    parts: list[pd.DataFrame] = []
    for frequency in frequencies:
        normalized_frequency = str(frequency).upper()
        if normalized_frequency not in {"FY", "FQ", "FI"}:
            raise ValueError(f"unsupported fundamental frequency: {frequency}")
        for field_role, source_field, source_priority, metric_variant in FUNDAMENTAL_FIELD_SPECS:
            fields = [
                source_field,
                f"{source_field}.calcdate",
                f"{source_field}.periodenddate",
                f"{source_field}.fperiod",
            ]
            frame = client.get_data(
                universe=instruments,
                fields=fields,
                parameters={
                    "SDate": start_date,
                    "EDate": end_date,
                    "Frq": normalized_frequency,
                    "Period": f"{normalized_frequency}0",
                },
            )
            part = normalise_parameterized_metric_response(
                frame,
                field_role=field_role,
                source_field=source_field,
                frequency=normalized_frequency,
                loaded_at_utc=loaded_at,
                source_priority=source_priority,
                metric_variant=metric_variant,
            )
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("source_priority")
        .drop_duplicates(["ric", "field_role", "frequency", "fiscal_period", "calc_date"], keep="first")
        .sort_values(["ric", "field_role", "frequency", "period_end_date"])
        .reset_index(drop=True)
    )


def write_japan_g3_fundamentals_qa_artifacts(
    records: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    requested_rics: Sequence[str],
    start_date: str,
    end_date: str,
    frequencies: Sequence[str],
    session_name: str,
) -> JapanG3FundamentalsArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / "fundamentals_qa.parquet"
    records.to_parquet(records_path, index=False)

    roles = {role for role, _field, _priority, _variant in FUNDAMENTAL_FIELD_SPECS}
    quality_rows = []
    for role in sorted(roles):
        rows = records.loc[records["field_role"].eq(role)].copy() if not records.empty else records
        quality_rows.append(
            {
                "dataset_id": "fundamentals_pit",
                "field_role": role,
                "history_rows": int(len(rows)),
                "ric_coverage": float(rows["ric"].nunique() / len(set(requested_rics))) if len(requested_rics) else 0.0,
                "known_at_coverage": float(rows["available_time_utc"].notna().mean()) if not rows.empty else 0.0,
                "pit_timestamp_violations": int(rows["pit_timestamp_violation"].sum()) if not rows.empty else 0,
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality_path = artifact_dir / "quality.csv"
    quality.to_csv(quality_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "completed_partial" if not records.empty else "empty",
        "scope": "bounded_qa",
        "evidence_scope": "qa",
        "scorecard_eligible": False,
        "partial": True,
        "smoke": True,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "dataset_id": "fundamentals_pit",
        "requested_rics": list(requested_rics),
        "requested_ric_count": len(set(requested_rics)),
        "returned_ric_count": int(records["ric"].nunique()) if not records.empty else 0,
        "history_rows": int(len(records)),
        "start_date": start_date,
        "end_date": end_date,
        "frequencies": list(frequencies),
        "field_roles": sorted(roles),
        "known_at_policy": "next_weekday_00_jst_pending_official_jpx_calendar",
        "known_at_coverage": float(records["available_time_utc"].notna().mean()) if not records.empty else 0.0,
        "pit_timestamp_violations": int(records["pit_timestamp_violation"].sum()) if not records.empty else 0,
        "session_name": session_name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            "fundamentals_qa": _sha256(records_path),
            "quality": _sha256(quality_path),
        },
        "artifacts": {
            "fundamentals_qa": str(records_path.resolve()),
            "quality": str(quality_path.resolve()),
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3FundamentalsArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        records_path=records_path,
        quality_path=quality_path,
        manifest_path=manifest_path,
        records=records,
        quality=quality,
    )
