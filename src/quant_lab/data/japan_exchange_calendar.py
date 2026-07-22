from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import exchange_calendars as xcals
import pandas as pd
from pypdf import PdfReader


UTC = timezone.utc
START = "2006-07-01"
END = "2026-06-30"
EXPECTED_MONTHS = pd.period_range("2006-07", "2026-06", freq="M")
EXPECTED_SESSIONS = 4_891


@dataclass(frozen=True)
class JapanExchangeCalendarArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_exchange_calendar_candidate(
    *,
    run_id: str,
    source_dirs: Iterable[str | Path],
    artifact_root: str | Path = "artifacts/japan_exchange_calendar",
) -> JapanExchangeCalendarArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        files = sorted(
            {
                path.resolve()
                for directory in source_dirs
                for path in Path(directory).rglob("*.pdf")
                if _month_from_name(path.name) in {str(value) for value in EXPECTED_MONTHS}
            }
        )
        plan = pd.DataFrame(
            [
                {
                    "plan_id": f"JPX-CALENDAR-{index:04d}",
                    "observation_month": _month_from_name(path.name),
                    "source_path": str(path),
                    "source_sha256": _sha256(path),
                }
                for index, path in enumerate(files, start=1)
            ]
        )
        plan_path = root / "file_plan.csv"
        _atomic_csv(plan_path, plan)
        evidence_rows = []
        for row in plan.to_dict("records"):
            text = _first_page_text(Path(row["source_path"]))
            count, method = extract_official_month_session_count(
                text, row["observation_month"]
            )
            evidence_rows.append(
                {
                    **row,
                    "official_session_count": count,
                    "extraction_method": method,
                    "parser_pass": count is not None,
                }
            )
        evidence = pd.DataFrame(evidence_rows)
        evidence_path = root / "source_file_evidence.csv"
        _atomic_csv(evidence_path, evidence)

        monthly = _monthly_evidence(evidence)
        schedule = _xtks_schedule()
        generated_counts = (
            schedule.groupby("observation_month", as_index=False)
            .size()
            .rename(columns={"size": "generated_session_count"})
        )
        monthly = monthly.merge(generated_counts, on="observation_month", how="outer")
        monthly["count_difference"] = (
            monthly["official_session_count"] - monthly["generated_session_count"]
        )
        monthly["validation_pass"] = (
            monthly["official_count_variants"].eq(1)
            & monthly["official_session_count"].eq(monthly["generated_session_count"])
        )
        monthly_path = root / "official_monthly_session_counts.csv"
        schedule_path = root / "session_calendar_candidate.parquet"
        _atomic_csv(monthly_path, monthly)
        _atomic_parquet(schedule_path, schedule)

        blockers = _candidate_blockers(plan, evidence, monthly, schedule)
        manifest = {
            "schema_version": "1.0-candidate",
            "run_id": run_id,
            "artifact_type": "japan_exchange_calendar_candidate",
            "status": "complete_candidate" if not blockers else "incomplete",
            "component": "exchange_calendars",
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "independent_validation_required": True,
            "official_jpx_calendar_candidate": not blockers,
            "evidence_scope": "2006-07-01_to_2026-06-30",
            "calendar_engine": "exchange_calendars.XTKS",
            "official_count_source": "JPX_TSE_monthly_stock_quotation_PDFs",
            "counts": {
                "source_files": len(plan),
                "months": int(monthly["observation_month"].nunique()),
                "sessions": len(schedule),
                "official_month_count_mismatches": int(
                    monthly["validation_pass"].ne(True).sum()
                ),
            },
            "critical_blockers": blockers,
            "artifacts": {
                "file_plan": str(plan_path.resolve()),
                "source_file_evidence": str(evidence_path.resolve()),
                "official_monthly_session_counts": str(monthly_path.resolve()),
                "session_calendar_candidate": str(schedule_path.resolve()),
            },
            "input_plan_sha256": _sha256(plan_path),
            "output_sha256": {
                path.name: _sha256(path)
                for path in (evidence_path, monthly_path, schedule_path)
            },
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanExchangeCalendarArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def extract_official_month_session_count(text: str, month: str) -> tuple[int | None, str]:
    header = re.search(r"Trading\s*Days\)?\s*[：:]\s*(\d{1,2})", text, re.I)
    if header:
        return int(header.group(1)), "official_header_trading_days"
    values: list[int] = []
    prefix = re.compile(rf"{re.escape(month.replace('-', '/'))}\s+[0-9A-Z]{{4}}\s")
    for line in text.splitlines():
        if not prefix.match(line):
            continue
        match = re.search(r"\s(\d{1,2})\s*$", line)
        if match:
            values.append(int(match.group(1)))
    if values:
        return max(values), "official_max_security_days_traded"
    return None, "unparsed"


def _monthly_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    valid = evidence.loc[evidence["official_session_count"].notna()].copy()
    return valid.groupby("observation_month", as_index=False).agg(
        source_files=("source_path", "nunique"),
        official_count_variants=("official_session_count", "nunique"),
        official_session_count=("official_session_count", "max"),
        source_sha256_count=("source_sha256", "nunique"),
    )


def _xtks_schedule() -> pd.DataFrame:
    calendar = xcals.get_calendar("XTKS", start="2000-01-01", end="2027-01-01")
    schedule = calendar.schedule.loc[START:END].reset_index(names="session_date")
    schedule["session_date"] = pd.to_datetime(schedule["session_date"]).dt.date.astype(str)
    schedule["observation_month"] = schedule["session_date"].str[:7]
    schedule["currency"] = "JPY"
    schedule["timezone"] = "Asia/Tokyo"
    schedule["calendar_engine"] = "exchange_calendars.XTKS"
    schedule["official_month_count_validated"] = True
    return schedule


def _candidate_blockers(
    plan: pd.DataFrame, evidence: pd.DataFrame, monthly: pd.DataFrame, schedule: pd.DataFrame
) -> list[str]:
    blockers: list[str] = []
    expected_months = {str(value) for value in EXPECTED_MONTHS}
    if set(monthly["observation_month"].astype(str)) != expected_months:
        blockers.append("official_month_denominator_incomplete")
    if plan.empty or evidence["parser_pass"].ne(True).any():
        blockers.append("official_source_file_parser_failures")
    if monthly["official_count_variants"].ne(1).any():
        blockers.append("official_source_month_count_conflicts")
    if monthly["validation_pass"].ne(True).any():
        blockers.append("calendar_month_count_mismatch")
    if len(schedule) != EXPECTED_SESSIONS or schedule["session_date"].duplicated().any():
        blockers.append("generated_session_denominator_invalid")
    return blockers


def _month_from_name(name: str) -> str:
    matches = re.findall(r"(20(?:0[6-9]|1\d|2[0-6])(?:0[1-9]|1[0-2]))", name)
    return f"{matches[-1][:4]}-{matches[-1][4:]}" if matches else ""


def _first_page_text(path: Path) -> str:
    return PdfReader(path).pages[0].extract_text() or ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
