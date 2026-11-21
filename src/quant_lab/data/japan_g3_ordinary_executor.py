from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.japan_g3_fundamentals import (
    next_weekday_utc,
    normalise_parameterized_metric_response,
)
from quant_lab.data.refinitiv_pit_extraction import history_frame_to_long
from quant_lab.data.vendor_adapters import (
    RefinitivRetryableProviderError,
    is_refinitiv_retryable_provider_error,
)


FUNDAMENTAL_ROLE_MAP = {
    "TR.Revenue": "revenue",
    "TR.BankTotalRevenue": "revenue",
    "TR.GrossProfit": "gross_profit",
    "TR.NetInterestIncome": "net_interest_income",
    "TR.InterestIncomeBank": "interest_income",
    "TR.OperatingIncome": "operating_income",
    "TR.EPSActValue": "eps",
    "TR.BookValuePerShare": "book_value",
}
CONSENSUS_ROLE_MAP = {
    "TR.EPSMean": "eps_estimate",
    "TR.RevenueMean": "revenue_estimate",
    "TR.RecMean": "recommendation",
    "TR.PriceTargetMean": "target_price",
}
EARNINGS_ROLE_MAP = {
    "TR.EPSActValue": "actual_eps",
    "TR.EPSActSurprise": "surprise",
}


@dataclass(frozen=True)
class JapanG3OrdinaryExecutionArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    row_manifest_path: Path
    quality_path: Path
    row_manifest: pd.DataFrame
    quality: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _g1_admitted(manifest: dict[str, Any]) -> bool:
    independent = manifest.get("independent_validation") is True or bool(
        manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
    )
    return bool(
        str(manifest.get("status", "")).lower() in {"complete", "pass", "accepted_backtest"}
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
        and independent
        and int(manifest.get("critical_blocker_count", 0) or 0) == 0
        and manifest.get("source_checksums_verified") is True
    )


def _g2_admitted(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest.get("status") == "pass"
        and manifest.get("admission_passed") is True
        and manifest.get("g2_complete") is True
        and manifest.get("scorecard_eligible") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("window_coverage_complete") is True
        and manifest.get("raw_compacted_parity") is True
        and manifest.get("official_corporate_actions_reconciled") is True
        and manifest.get("pit_liquidity_complete") is True
        and int(manifest.get("critical_blocker_count", -1)) == 0
        and int(manifest.get("check_count", -1))
        == int(manifest.get("passed_check_count", -2))
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _row_fields(row: pd.Series) -> list[str]:
    return [field.strip() for field in str(row["source_fields"]).split(",") if field.strip()]


def _parameterized_rows(client: Any, row: pd.Series, lane: str) -> pd.DataFrame:
    rics = [ric for ric in str(row["ric_list"]).split(",") if ric]
    loaded_at = datetime.now(timezone.utc).isoformat()
    frequency = str(row["frequency"])
    parts: list[pd.DataFrame] = []
    role_map = FUNDAMENTAL_ROLE_MAP if lane == "fundamentals" else EARNINGS_ROLE_MAP
    dataset_id = "fundamentals_pit" if lane == "fundamentals" else "earnings_events_actuals"
    for field in _row_fields(row):
        frame = client.get_data(
            universe=rics,
            fields=[field, f"{field}.calcdate", f"{field}.periodenddate", f"{field}.fperiod"],
            parameters={
                "SDate": str(row["start_date"]),
                "EDate": str(row["end_date"]),
                "Frq": frequency,
                "Period": f"{frequency}0",
            },
        )
        part = normalise_parameterized_metric_response(
            frame,
            field_role=role_map[field],
            source_field=field,
            frequency=frequency,
            loaded_at_utc=loaded_at,
            metric_variant=str(row.get("route_class", lane)),
            dataset_id=dataset_id,
        )
        if not part.empty:
            parts.append(part)
            if lane == "earnings" and field == "TR.EPSActValue":
                release = part.copy()
                release["field_role"] = "earnings_release"
                release["source_field"] = "TR.EPSActValue.calcdate"
                release["value"] = 1.0
                parts.append(release)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _consensus_rows(client: Any, row: pd.Series) -> pd.DataFrame:
    rics = [ric for ric in str(row["ric_list"]).split(",") if ric]
    loaded_at = datetime.now(timezone.utc).isoformat()
    universe = pd.DataFrame({"refinitiv_ric": rics})
    parts: list[pd.DataFrame] = []
    for field in _row_fields(row):
        history = client.get_history(
            universe=rics,
            fields=[field],
            start=str(row["start_date"]),
            end=str(row["end_date"]),
            interval=str(row["interval"]),
        )
        plan_row = row.to_dict() | {
            "dataset_id": "ibes_estimates_revisions",
            "field_role": CONSENSUS_ROLE_MAP[field],
            "candidate_field": field,
            "country": "Japan",
            "tier": "u2",
        }
        part = history_frame_to_long(history, rics, plan_row, universe, loaded_at)
        if part.empty:
            continue
        part["observation_date"] = pd.to_datetime(part["asof_date"], errors="coerce")
        part["available_time_utc"] = next_weekday_utc(part["observation_date"])
        part["known_at_policy"] = "next_weekday_00_jst_pending_official_jpx_calendar"
        part["pit_timestamp_violation"] = part["available_time_utc"].isna()
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _quality(row_manifest: pd.DataFrame) -> pd.DataFrame:
    if row_manifest.empty:
        return pd.DataFrame()
    rows = []
    for dataset_id, group in row_manifest.groupby("dataset_id", sort=True):
        completed = group["status"].eq("completed")
        rows.append(
            {
                "dataset_id": dataset_id,
                "plan_rows": len(group),
                "completed_rows": int(completed.sum()),
                "failed_rows": int(group["status"].astype(str).str.startswith("failed").sum()),
                "empty_rows": int(group["status"].eq("empty").sum()),
                "history_rows": int(pd.to_numeric(group["history_rows"], errors="coerce").fillna(0).sum()),
                "known_at_coverage": float(
                    pd.to_numeric(group.loc[completed, "known_at_coverage"], errors="coerce").min()
                )
                if completed.any()
                else 0.0,
                "pit_timestamp_violations": int(
                    pd.to_numeric(group["pit_timestamp_violations"], errors="coerce").fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_japan_g3_ordinary_execution_admission(
    plan_path: str | Path,
    expected_plan_sha256: str,
    g1_validation_dir: str | Path,
    g2_admission_dir: str | Path,
    max_plan_rows: int,
) -> None:
    plan_file = Path(plan_path)
    if not plan_file.exists():
        raise FileNotFoundError(plan_file)
    if _sha256(plan_file) != expected_plan_sha256:
        raise ValueError("Japan G3 plan SHA-256 mismatch")
    if not _g1_admitted(_read_json(Path(g1_validation_dir) / "manifest.json")):
        raise PermissionError("independent Japan G1 admission is required before ordinary G3 extraction")
    if not _g2_admitted(_read_json(Path(g2_admission_dir) / "manifest.json")):
        raise PermissionError("strict Japan G2 aggregate admission is required before ordinary G3 extraction")
    if max_plan_rows <= 0:
        raise ValueError("max_plan_rows must be positive")
    if plan_file.name not in {"fundamentals_plan.csv", "consensus_plan.csv", "earnings_plan.csv"}:
        raise ValueError(f"unsupported ordinary G3 plan: {plan_file.name}")


def execute_japan_g3_ordinary_plan(
    client: Any,
    plan_path: str | Path,
    expected_plan_sha256: str,
    g1_validation_dir: str | Path,
    g2_admission_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    max_plan_rows: int = 1,
    sleep_seconds: float = 0.4,
    resume: bool = True,
) -> JapanG3OrdinaryExecutionArtifacts:
    plan_file = Path(plan_path)
    validate_japan_g3_ordinary_execution_admission(
        plan_file, expected_plan_sha256, g1_validation_dir, g2_admission_dir, max_plan_rows
    )
    actual_hash = _sha256(plan_file)

    lane_by_name = {
        "fundamentals_plan.csv": "fundamentals",
        "consensus_plan.csv": "consensus",
        "earnings_plan.csv": "earnings",
    }
    lane = lane_by_name.get(plan_file.name)
    if lane is None:
        raise ValueError(f"unsupported ordinary G3 plan: {plan_file.name}")
    plan = pd.read_csv(plan_file, low_memory=False)
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_dir / ".writer.lock"
    try:
        os.mkdir(lock_path)
    except FileExistsError as exc:
        raise RuntimeError(f"Japan G3 writer lock exists: {lock_path}") from exc

    row_manifest_path = artifact_dir / "row_manifest.csv"
    quality_path = artifact_dir / "quality.csv"
    manifest_path = artifact_dir / "manifest.json"
    columns = [
        "plan_row_id",
        "dataset_id",
        "status",
        "output_path",
        "history_rows",
        "known_at_coverage",
        "pit_timestamp_violations",
        "started_at_utc",
        "completed_at_utc",
        "error",
    ]
    existing = pd.read_csv(row_manifest_path, low_memory=False) if resume and row_manifest_path.exists() else pd.DataFrame(columns=columns)
    terminal = set(existing.loc[existing["status"].isin(["completed", "empty"]), "plan_row_id"].astype(str))
    pending = plan.loc[~plan["plan_row_id"].astype(str).isin(terminal)].head(max_plan_rows)
    records = existing.to_dict("records")
    try:
        provider_halt: dict[str, str] | None = None
        for _, row in pending.iterrows():
            row_id = str(row["plan_row_id"])
            started = datetime.now(timezone.utc).isoformat()
            output_path = artifact_dir / "data" / lane / f"{row_id}.parquet"
            try:
                data = _consensus_rows(client, row) if lane == "consensus" else _parameterized_rows(client, row, lane)
                status = "empty" if data.empty else "completed"
                if not data.empty:
                    _atomic_parquet(data, output_path)
                known_at = float(data["available_time_utc"].notna().mean()) if not data.empty else 0.0
                violations = int(data["pit_timestamp_violation"].sum()) if not data.empty else 0
                result = {
                    "plan_row_id": row_id,
                    "dataset_id": str(row["dataset_id"]),
                    "status": status,
                    "output_path": str(output_path.resolve()) if not data.empty else "",
                    "history_rows": len(data),
                    "known_at_coverage": known_at,
                    "pit_timestamp_violations": violations,
                    "started_at_utc": started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": "",
                }
            except Exception as exc:
                retryable = isinstance(exc, RefinitivRetryableProviderError) or is_refinitiv_retryable_provider_error(exc)
                result = {
                    "plan_row_id": row_id,
                    "dataset_id": str(row["dataset_id"]),
                    "status": "failed_retryable_provider" if retryable else "failed",
                    "output_path": "",
                    "history_rows": 0,
                    "known_at_coverage": 0.0,
                    "pit_timestamp_violations": 0,
                    "started_at_utc": started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if retryable:
                    provider_halt = {
                        "plan_row_id": row_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            records = [record for record in records if str(record.get("plan_row_id")) != row_id]
            records.append(result)
            checkpoint = pd.DataFrame(records, columns=columns)
            _atomic_csv(checkpoint, row_manifest_path)
            if provider_halt is not None:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if lock_path.exists():
            lock_path.rmdir()

    row_manifest = pd.DataFrame(records, columns=columns)
    quality = _quality(row_manifest)
    _atomic_csv(quality, quality_path)
    full_terminal = set(row_manifest.loc[row_manifest["status"].isin(["completed", "empty"]), "plan_row_id"].astype(str))
    full_complete = set(plan["plan_row_id"].astype(str)).issubset(full_terminal)
    failed_rows = int(row_manifest["status"].astype(str).str.startswith("failed").sum())
    empty_rows = int(row_manifest["status"].eq("empty").sum())
    known_at_coverage = float(quality["known_at_coverage"].min()) if not quality.empty else 0.0
    violations = int(quality["pit_timestamp_violations"].sum()) if not quality.empty else 0
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "complete" if full_complete and failed_rows == 0 and empty_rows == 0 and known_at_coverage == 1.0 and violations == 0 else "completed_partial",
        "evidence_scope": "full" if full_complete else "bounded",
        "scorecard_eligible": bool(full_complete and failed_rows == 0 and empty_rows == 0 and known_at_coverage == 1.0 and violations == 0),
        "partial": not full_complete,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "lane": lane,
        "planned_rows": len(plan),
        "terminal_rows": len(full_terminal),
        "completed_rows": int(row_manifest["status"].eq("completed").sum()),
        "failed_rows": failed_rows,
        "empty_rows": empty_rows,
        "history_rows": int(pd.to_numeric(row_manifest["history_rows"], errors="coerce").fillna(0).sum()),
        "known_at_coverage": known_at_coverage,
        "missing_known_at_rows": 0 if known_at_coverage == 1.0 else -1,
        "pit_timestamp_violations": violations,
        "halted_early": provider_halt is not None,
        "provider_halt": provider_halt,
        "plan_path": str(plan_file.resolve()),
        "plan_sha256": actual_hash,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            "row_manifest": _sha256(row_manifest_path),
            "quality": _sha256(quality_path),
        },
        "artifacts": {
            "row_manifest": str(row_manifest_path.resolve()),
            "quality": str(quality_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3OrdinaryExecutionArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        row_manifest_path=row_manifest_path,
        quality_path=quality_path,
        row_manifest=row_manifest,
        quality=quality,
    )
