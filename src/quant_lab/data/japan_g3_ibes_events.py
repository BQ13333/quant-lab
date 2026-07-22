from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_lab.data.japan_g3_fundamentals import (
    next_weekday_utc,
    normalise_parameterized_metric_response,
)
from quant_lab.data.refinitiv_pit_extraction import history_frame_to_long


CONSENSUS_FIELDS: dict[str, str] = {
    "eps_estimate": "TR.EPSMean",
    "revenue_estimate": "TR.RevenueMean",
    "recommendation": "TR.RecMean",
    "target_price": "TR.PriceTargetMean",
}
ANALYST_FIELD_SETS: dict[str, tuple[str, ...]] = {
    "recommendation": (
        "TR.RecEstValue.brokername",
        "TR.RecEstValue.analystname",
        "TR.RecEstValue.calcdate",
        "TR.RecEstValue.date",
        "TR.RecEstValue",
    ),
    "target_price": (
        "TR.TPEstValue.brokername",
        "TR.TPEstValue.analystname",
        "TR.TPEstValue.calcdate",
        "TR.TPEstValue.date",
        "TR.TPEstValue",
    ),
}
EVENT_FIELDS: dict[str, str] = {
    "actual_eps": "TR.EPSActValue",
    "surprise": "TR.EPSActSurprise",
}


@dataclass(frozen=True)
class JapanG3IBESEventsArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    quality_path: Path
    consensus: pd.DataFrame
    analyst_snapshots: pd.DataFrame
    revision_events: pd.DataFrame
    earnings_events: pd.DataFrame
    quality: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    source = next((column for column in candidates if column in frame.columns), None)
    return frame[source] if source is not None else pd.Series(pd.NA, index=frame.index)


def extract_consensus_history_qa(
    client: Any,
    rics: Sequence[str],
    start_date: str,
    end_date: str,
    interval: str = "1M",
) -> pd.DataFrame:
    instruments = list(dict.fromkeys(str(ric) for ric in rics))
    universe = pd.DataFrame({"refinitiv_ric": instruments})
    loaded_at = datetime.now(timezone.utc).isoformat()
    parts: list[pd.DataFrame] = []
    for role, field in CONSENSUS_FIELDS.items():
        history = client.get_history(
            universe=instruments,
            fields=[field],
            start=start_date,
            end=end_date,
            interval=interval,
        )
        plan_row = {
            "dataset_id": "ibes_estimates_revisions",
            "field_role": role,
            "candidate_field": field,
            "country": "Japan",
            "tier": "u2",
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "batch_number": 1,
            "window_number": 1,
            "plan_row_id": f"qa-{role}",
        }
        long = history_frame_to_long(history, instruments, plan_row, universe, loaded_at)
        if long.empty:
            continue
        long["observation_date"] = pd.to_datetime(long["asof_date"], errors="coerce")
        long["available_time_utc"] = next_weekday_utc(long["observation_date"])
        long["known_at_policy"] = "next_weekday_00_jst_pending_official_jpx_calendar"
        long["pit_timestamp_violation"] = long["available_time_utc"].isna()
        parts.append(long)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def extract_analyst_snapshots_qa(
    client: Any,
    rics: Sequence[str],
    start_date: str,
    end_date: str,
    period: str = "FY1",
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    loaded_at = datetime.now(timezone.utc).isoformat()
    parameters = {"SDate": start_date, "EDate": end_date, "Frq": "D", "Period": period}
    for estimate_type, fields in ANALYST_FIELD_SETS.items():
        frame = client.get_data(universe=list(rics), fields=list(fields), parameters=parameters)
        if frame is None or frame.empty:
            continue
        value_candidates = (
            ("Standard Rec (1-5) - Broker Estimate",)
            if estimate_type == "recommendation"
            else ("Target Price - Broker Estimate",)
        )
        result = pd.DataFrame(
            {
                "ric": _column(frame, ("Instrument", "RIC")).astype("string"),
                "analyst_name": _column(frame, ("Analyst Name", "Analyst - Full Name")).astype("string"),
                "broker_name": _column(frame, ("Broker Name",)).astype("string"),
                "calc_date": pd.to_datetime(_column(frame, ("Calc Date",)), errors="coerce"),
                "estimate_date": pd.to_datetime(
                    _column(frame, ("Date",)), errors="coerce", format="mixed"
                ),
                "estimate_value": pd.to_numeric(_column(frame, value_candidates), errors="coerce"),
            }
        )
        result = result.dropna(subset=["ric", "calc_date", "estimate_date"]).copy()
        result["estimate_type"] = estimate_type
        result["forecast_period"] = period
        result["source_field"] = fields[-1]
        result["source_loaded_at_utc"] = loaded_at
        result["source_row_keyed"] = True
        parts.append(result)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def derive_analyst_revision_events(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    data = snapshots.copy()
    data["analyst_key"] = data["analyst_name"].fillna("").astype(str).str.strip()
    data["broker_key"] = data["broker_name"].fillna("").astype(str).str.strip()
    data = data.loc[data["analyst_key"].ne("") | data["broker_key"].ne("")].copy()
    group_cols = ["ric", "estimate_type", "analyst_key", "broker_key", "forecast_period"]
    data = (
        data.sort_values(group_cols + ["estimate_date", "calc_date"])
        .drop_duplicates(group_cols + ["estimate_date", "estimate_value"], keep="first")
        .reset_index(drop=True)
    )
    grouped = data.groupby(group_cols, dropna=False, sort=False)
    data["previous_estimate_value"] = grouped["estimate_value"].shift(1)
    prior_seen = grouped.cumcount().gt(0)
    value_change = (
        prior_seen
        & data["estimate_value"].notna()
        & data["previous_estimate_value"].notna()
        & data["estimate_value"].ne(data["previous_estimate_value"])
    )
    data["event_type"] = "no_change"
    data.loc[~prior_seen, "event_type"] = "baseline"
    data.loc[value_change & data["estimate_type"].eq("recommendation"), "event_type"] = "recommendation_revision"
    data.loc[value_change & data["estimate_type"].eq("target_price"), "event_type"] = "target_price_revision"
    data["estimate_change"] = data["estimate_value"] - data["previous_estimate_value"]
    denominator = data["previous_estimate_value"].abs().replace(0, pd.NA)
    data["revision_pct"] = data["estimate_change"] / denominator
    data["available_time_utc"] = next_weekday_utc(data["estimate_date"])
    data["known_at_policy"] = "estimate_date_then_next_weekday_00_jst_pending_official_jpx_calendar"
    data["pit_timestamp_violation"] = (
        data["available_time_utc"].isna()
        | data["estimate_date"].dt.normalize().gt(data["calc_date"].dt.normalize())
    )
    data["dataset_id"] = "ibes_estimates_revisions"
    data["field_role"] = "revision_event"
    return data.loc[data["event_type"].ne("no_change")].reset_index(drop=True)


def extract_earnings_actuals_qa(
    client: Any,
    rics: Sequence[str],
    start_date: str,
    end_date: str,
    frequencies: Sequence[str] = ("FY", "FQ"),
) -> pd.DataFrame:
    loaded_at = datetime.now(timezone.utc).isoformat()
    parts: list[pd.DataFrame] = []
    for frequency in frequencies:
        normalized_frequency = str(frequency).upper()
        for role, field in EVENT_FIELDS.items():
            frame = client.get_data(
                universe=list(rics),
                fields=[field, f"{field}.calcdate", f"{field}.periodenddate", f"{field}.fperiod"],
                parameters={
                    "SDate": start_date,
                    "EDate": end_date,
                    "Frq": normalized_frequency,
                    "Period": f"{normalized_frequency}0",
                },
            )
            part = normalise_parameterized_metric_response(
                frame,
                field_role=role,
                source_field=field,
                frequency=normalized_frequency,
                loaded_at_utc=loaded_at,
                metric_variant="parameterized_earnings_actual",
                dataset_id="earnings_events_actuals",
            )
            if not part.empty:
                parts.append(part)
                if role == "actual_eps":
                    release = part.copy()
                    release["field_role"] = "earnings_release"
                    release["source_field"] = f"{field}.calcdate"
                    release["value"] = 1.0
                    parts.append(release)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def write_japan_g3_ibes_events_qa_artifacts(
    artifact_root: str | Path,
    run_id: str,
    consensus: pd.DataFrame,
    analyst_snapshots: pd.DataFrame,
    revision_events: pd.DataFrame,
    earnings_events: pd.DataFrame,
    requested_rics: Sequence[str],
    consensus_start_date: str,
    revision_start_date: str,
    end_date: str,
    session_name: str,
    rejected_misaligned_event_rows: int = 0,
) -> JapanG3IBESEventsArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "consensus_history": consensus,
        "analyst_snapshots": analyst_snapshots,
        "revision_events": revision_events,
        "earnings_events": earnings_events,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = artifact_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path

    requested_count = len(set(requested_rics))
    quality_rows = []
    for dataset_id, frame in (
        ("ibes_estimates_revisions", pd.concat([consensus, revision_events], ignore_index=True, sort=False)),
        ("earnings_events_actuals", earnings_events),
    ):
        known = frame.get("available_time_utc", pd.Series(dtype="datetime64[ns]")).notna()
        violations = pd.to_numeric(
            frame.get("pit_timestamp_violation", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        ric_col = "ric" if "ric" in frame.columns else "requested_ric"
        quality_rows.append(
            {
                "dataset_id": dataset_id,
                "history_rows": int(len(frame)),
                "requested_rics": requested_count,
                "returned_rics": int(frame[ric_col].nunique()) if not frame.empty and ric_col in frame else 0,
                "coverage": float(frame[ric_col].nunique() / requested_count)
                if requested_count and not frame.empty and ric_col in frame
                else 0.0,
                "known_at_coverage": float(known.mean()) if len(known) else 0.0,
                "pit_timestamp_violations": int(violations.sum()),
                "failed_rows": 0,
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality_path = artifact_dir / "quality.csv"
    quality.to_csv(quality_path, index=False)
    rejected_methodology_path = artifact_dir / "rejected_methodology.json"
    rejected_methodology_path.write_text(
        json.dumps(
            {
                "status": "rejected",
                "method": "mixed_unkeyed_analyst_broker_recommendation_target_fields",
                "reason": "Refinitiv structured field families have different backend row dimensions and cannot be mixed row-wise.",
                "rejected_derived_event_rows": int(rejected_misaligned_event_rows),
                "replacement": "separate self-keyed TR.RecEstValue.* and TR.TPEstValue.* queries",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "completed_partial" if all(not frame.empty for frame in frames.values()) else "incomplete_qa",
        "scope": "bounded_qa",
        "evidence_scope": "qa",
        "scorecard_eligible": False,
        "partial": True,
        "smoke": True,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "requested_rics": requested_count,
        "consensus_rows": int(len(consensus)),
        "analyst_snapshot_rows": int(len(analyst_snapshots)),
        "revision_event_rows": int(len(revision_events)),
        "earnings_event_rows": int(len(earnings_events)),
        "consensus_start_date": consensus_start_date,
        "revision_start_date": revision_start_date,
        "end_date": end_date,
        "known_at_coverage": float(quality["known_at_coverage"].min()) if not quality.empty else 0.0,
        "pit_timestamp_violations": int(quality["pit_timestamp_violations"].sum()) if not quality.empty else 0,
        "raw_period_end_surprise_history_allowed": False,
        "self_keyed_analyst_fields": True,
        "mixed_target_and_recommendation_query_allowed": False,
        "revision_definition": "consecutive broker_or_analyst snapshot value change; first observation is baseline",
        "rejected_misaligned_event_rows": int(rejected_misaligned_event_rows),
        "session_name": session_name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: _sha256(path) for name, path in paths.items()}
        | {
            "quality": _sha256(quality_path),
            "rejected_methodology": _sha256(rejected_methodology_path),
        },
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()}
        | {
            "quality": str(quality_path.resolve()),
            "rejected_methodology": str(rejected_methodology_path.resolve()),
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3IBESEventsArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        quality_path=quality_path,
        consensus=consensus,
        analyst_snapshots=analyst_snapshots,
        revision_events=revision_events,
        earnings_events=earnings_events,
        quality=quality,
    )
