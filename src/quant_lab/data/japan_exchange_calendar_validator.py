from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from pypdf import PdfReader

from quant_lab.data.japan_exchange_calendar import END, EXPECTED_MONTHS, EXPECTED_SESSIONS, START


UTC = timezone.utc


@dataclass(frozen=True)
class JapanExchangeCalendarValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_exchange_calendar_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_exchange_calendar_validation",
) -> JapanExchangeCalendarValidationArtifacts:
    producer = Path(producer_dir).resolve()
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    producer_manifest_path = producer / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    blockers: list[str] = []
    if run_id == producer_manifest.get("run_id"):
        blockers.append("independence_failure_same_run_id")
    if not (
        producer_manifest.get("status") == "complete_candidate"
        and producer_manifest.get("scorecard_eligible") is False
        and producer_manifest.get("independent_validation_required") is True
        and producer_manifest.get("official_jpx_calendar_candidate") is True
    ):
        blockers.append("producer_not_complete_candidate")
    plan_path = producer / "file_plan.csv"
    if not plan_path.is_file() or _sha256(plan_path) != producer_manifest.get("input_plan_sha256"):
        blockers.append("producer_plan_hash_mismatch")
    for filename, digest in producer_manifest.get("output_sha256", {}).items():
        path = producer / filename
        if not path.is_file() or _sha256(path) != digest:
            blockers.append(f"producer_output_hash_mismatch:{filename}")

    plan = pd.read_csv(plan_path, dtype=str) if plan_path.is_file() else pd.DataFrame()
    evidence_path = producer / "source_file_evidence.csv"
    evidence = pd.read_csv(evidence_path, dtype=str) if evidence_path.is_file() else pd.DataFrame()
    candidate_path = producer / "session_calendar_candidate.parquet"
    candidate = pd.read_parquet(candidate_path) if candidate_path.is_file() else pd.DataFrame()
    independently_observed = _independent_official_counts(plan, blockers)
    if not evidence.empty and not independently_observed.empty:
        check = evidence.merge(
            independently_observed,
            on=["plan_id", "observation_month", "source_path", "source_sha256"],
            how="outer",
            validate="one_to_one",
        )
        producer_count = pd.to_numeric(check["official_session_count"], errors="coerce")
        validator_count = pd.to_numeric(check["validator_session_count"], errors="coerce")
        if producer_count.isna().any() or not producer_count.eq(validator_count).all():
            blockers.append("independent_source_count_reconciliation_failed")
    else:
        blockers.append("source_evidence_missing")

    expected = _independent_schedule()
    blockers.extend(_schedule_blockers(candidate, expected, independently_observed))
    validated_path = root / "validated_exchange_calendar.parquet"
    if blockers:
        pd.DataFrame(columns=expected.columns).to_parquet(validated_path, index=False)
    else:
        shutil.copy2(candidate_path, validated_path)
    manifest = {
        "schema_version": "1.0-independent",
        "run_id": run_id,
        "validator_run_id": run_id,
        "producer_run_id": producer_manifest.get("run_id"),
        "artifact_type": "japan_exchange_calendar_independent_validation",
        "status": "pass_component" if not blockers else "failed_validation",
        "component": "exchange_calendars",
        "component_scorecard_eligible": not blockers,
        "scorecard_eligible": False,
        "independent_validation": True,
        "official_jpx_calendar": not blockers,
        "currency": "JPY",
        "evidence_scope": "2006-07-01_to_2026-06-30" if not blockers else "failed_no_admission",
        "counts": {
            "source_files": len(plan),
            "months": int(independently_observed["observation_month"].nunique()) if not independently_observed.empty else 0,
            "sessions": len(candidate),
        },
        "critical_blocker_count": len(blockers),
        "critical_blockers": blockers,
        "producer_manifest": str(producer_manifest_path),
        "producer_manifest_sha256": _sha256(producer_manifest_path) if producer_manifest_path.is_file() else "",
        "artifacts": {"validated_exchange_calendar": str(validated_path.resolve())},
        "output_sha256": {validated_path.name: _sha256(validated_path)},
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanExchangeCalendarValidationArtifacts(run_id, root, manifest_path, manifest)


def _independent_official_counts(plan: pd.DataFrame, blockers: list[str]) -> pd.DataFrame:
    rows = []
    for row in plan.to_dict("records"):
        path = Path(row["source_path"])
        if not path.is_file() or _sha256(path) != row["source_sha256"]:
            blockers.append(f"official_source_hash_mismatch:{row.get('plan_id', '')}")
            continue
        text = PdfReader(path).pages[0].extract_text() or ""
        header = re.search(r"立会日数\s*\(Trading\s*Days\)\s*[：:]\s*(\d{1,2})", text, re.I)
        values = []
        if header:
            count = int(header.group(1))
        else:
            prefix = re.compile(rf"{re.escape(row['observation_month'].replace('-', '/'))}\s+[0-9A-Z]{{4}}\s")
            for line in text.splitlines():
                if prefix.match(line):
                    match = re.search(r"\s(\d{1,2})\s*$", line)
                    if match:
                        values.append(int(match.group(1)))
            count = max(values) if values else None
        rows.append({**row, "validator_session_count": count})
    return pd.DataFrame(rows)


def _independent_schedule() -> pd.DataFrame:
    calendar = xcals.get_calendar("XTKS", start="2000-01-01", end="2027-01-01")
    frame = calendar.schedule.loc[START:END].reset_index(names="session_date")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["observation_month"] = frame["session_date"].str[:7]
    frame["currency"] = "JPY"
    frame["timezone"] = "Asia/Tokyo"
    frame["calendar_engine"] = "exchange_calendars.XTKS"
    frame["official_month_count_validated"] = True
    return frame


def _schedule_blockers(
    candidate: pd.DataFrame, expected: pd.DataFrame, observed: pd.DataFrame
) -> list[str]:
    blockers = []
    if len(candidate) != EXPECTED_SESSIONS or len(expected) != EXPECTED_SESSIONS:
        blockers.append("session_denominator_invalid")
    required = set(expected.columns)
    if candidate.empty or not required.issubset(candidate.columns):
        return [*blockers, "calendar_schema_invalid"]
    compare = ["session_date", "open", "break_start", "break_end", "close"]
    left = candidate[compare].copy()
    right = expected[compare].copy()
    for column in compare[1:]:
        left[column] = pd.to_datetime(left[column], utc=True)
        right[column] = pd.to_datetime(right[column], utc=True)
    if not left.equals(right):
        blockers.append("date_or_session_time_reconciliation_failed")
    months = set(candidate["observation_month"].astype(str))
    if months != {str(value) for value in EXPECTED_MONTHS}:
        blockers.append("calendar_month_denominator_invalid")
    if candidate["session_date"].duplicated().any():
        blockers.append("duplicate_session_dates")
    if not observed.empty:
        official = observed.groupby("observation_month")["validator_session_count"].agg(["nunique", "max"])
        generated = candidate.groupby("observation_month").size().rename("generated")
        check = official.join(generated)
        if len(check) != len(EXPECTED_MONTHS) or check["nunique"].ne(1).any() or check["max"].ne(check["generated"]).any():
            blockers.append("official_month_count_validation_failed")
    return blockers


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
