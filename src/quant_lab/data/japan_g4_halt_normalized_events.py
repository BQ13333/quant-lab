from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
from pypdf import PdfReader


HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
CHANNEL_HEADINGS = {
    HALT_CHANNEL: "Suspension of Trading",
    DELISTING_CHANNEL: "Delisting",
}
EVENT_TYPES = {
    HALT_CHANNEL: "trading_halt",
    DELISTING_CHANNEL: "delisting_non_trading_interval",
}
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
ANY_DATE = re.compile(r"(?P<year>20\d{2})\s*[./]\s*(?P<month>\d{1,2})\s*[./]\s*(?P<day>\d{1,2})")
SHORT_DATE = re.compile(r"(?P<month>\d{1,2})\s*[./]\s*(?P<day>\d{1,2})")
TIME_TOKEN = r"(?:(?:[01]?\d|2[0-3])\s*[:：;.]\s*[0-5]\d|(?:[01]\d|2[0-3])[0-5]\d)"
TIME_RANGE = re.compile(
    rf"(?P<start>{TIME_TOKEN})\s*[～~]\s*"
    rf"(?:(?P<end_date>20\d{{2}}\s*[./]\s*\d{{1,2}}\s*[./]\s*\d{{1,2}}|\d{{1,2}}\s*[./]\s*\d{{1,2}})\s*)?"
    rf"(?P<end>{TIME_TOKEN})"
)
CODE = re.compile(r"^(?P<code>[0-9A-Z]{4,10})(?:\s+|$)")
EXPLICIT_ZERO = frozenset({"none", "nil", "zero", "no events"})

EventTextExtractor = Callable[[bytes], str]


@dataclass(frozen=True)
class JapanG4NormalizedEventArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    events_path: Path
    manifest: dict[str, Any]


def parse_explicit_report_events(
    text: str,
    *,
    report_month: str,
    channel: str,
    source_url: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    if channel not in CHANNEL_HEADINGS:
        raise ValueError(f"unsupported channel: {channel}")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    section = _explicit_section(lines, CHANNEL_HEADINGS[channel])
    rows: list[dict[str, Any]] = []
    for line in section:
        event = _parse_event_line(line, report_month, channel)
        if event is None:
            continue
        record_payload = {
            "report_month": report_month,
            "channel": channel,
            "source_sha256": source_sha256,
            "source_record_text": line,
        }
        source_record_sha = _sha_bytes(
            json.dumps(record_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        event_id = _sha_bytes(f"{EVENT_TYPES[channel]}|{source_record_sha}".encode("utf-8"))[:24]
        known_at = conservative_report_known_at(report_month)
        rows.append(
            {
                "event_id": event_id,
                "report_month": report_month,
                "channel": channel,
                "event_type": EVENT_TYPES[channel],
                **event,
                "source_url": source_url,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_record_text": line,
                "source_record_sha256": source_record_sha,
                "known_at_utc": known_at,
                "known_at_policy": "official_monthly_statistics_21st_09_jst_following_month",
                "known_at_status": "conservative_official_monthly_publication",
                "historical_event_time_pit_usable": False,
                "stable_listing_id": "",
                "mapping_status": "local_code_only_unmapped" if event["local_code"] else "aggregate_or_derivative_record_without_local_code",
                "admissible_to_shared_halt_history": False,
            }
        )
    return rows


def conservative_report_known_at(report_month: str) -> str:
    following = pd.Period(report_month, freq="M") + 1
    local = datetime.combine(following.start_time.date().replace(day=21), time(9, 0), JST)
    return local.astimezone(UTC).isoformat()


def run_japan_g4_halt_normalized_events(
    *,
    run_id: str,
    denominator_v2_dir: str | Path,
    validation_v2_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_normalized_events",
    text_extractor: EventTextExtractor | None = None,
) -> JapanG4NormalizedEventArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    denominator_root = Path(denominator_v2_dir).resolve()
    validation_root = Path(validation_v2_dir).resolve()
    denominator_manifest = _verify_artifact(denominator_root)
    validation_manifest = _verify_artifact(validation_root)
    if validation_manifest.get("status") != "pass_partial_component":
        raise ValueError("denominator V2 validation must pass as a partial component")
    if int(validation_manifest.get("validation_issue_count", -1)) != 0:
        raise ValueError("denominator V2 validation contains issues")

    root = Path(artifact_root) / run_id
    if root.exists():
        raise FileExistsError(f"write-once normalized event artifact already exists: {root}")
    raw_root = root / "raw" / "lifecycle"
    raw_root.mkdir(parents=True)
    extract = text_extractor or _extract_text

    coverage = pd.read_csv(denominator_root / "month_channel_coverage.csv", dtype=str).fillna("")
    covered = coverage.loc[_bools(coverage["covered"])].copy()
    validation_claims = pd.read_csv(validation_root / "claim_reconciliation.csv", dtype=str).fillna("")
    if len(covered) != 194 or len(validation_claims) != 194:
        raise ValueError("expected exactly 194 V2 covered and independently validated claims")
    if not _bools(validation_claims["claim_reconciled"]).all():
        raise ValueError("all denominator V2 claims must be independently reconciled")
    if set(covered["gap_id"]) != set(validation_claims["gap_id"]):
        raise ValueError("denominator and validation V2 claim sets differ")

    inventory = pd.read_csv(denominator_root / "source_inventory.csv", dtype=str).fillna("")
    valid_sources = inventory.loc[
        inventory["source_kind"].eq("lifecycle_pdf")
        & inventory["validation_status"].eq("exact_month_valid")
    ].set_index("source_key")
    event_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    text_by_month: dict[str, str] = {}
    source_by_month: dict[str, dict[str, str]] = {}
    for month in sorted(set(covered["month"])):
        if month not in valid_sources.index:
            raise ValueError(f"missing exact-month source inventory row: {month}")
        source = valid_sources.loc[month]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"duplicate exact-month source inventory rows: {month}")
        v2_path = (denominator_root / "raw" / "lifecycle" / f"{month}.pdf").resolve()
        observed_hash = _sha_path(v2_path)
        if observed_hash != source["sha256"]:
            raise ValueError(f"V2 raw report hash mismatch: {month}")
        v3_path = raw_root / f"{month}.pdf"
        shutil.copyfile(v2_path, v3_path)
        if _sha_path(v3_path) != observed_hash:
            raise ValueError(f"V3 raw report copy hash mismatch: {month}")
        text_by_month[month] = extract(v3_path.read_bytes())
        source_by_month[month] = {
            "source_url": source["source_url"],
            "source_path": str(v3_path.resolve()),
            "source_sha256": observed_hash,
        }
        report_rows.append(
            {
                "report_month": month,
                "source_url": source["source_url"],
                "v2_source_path": str(v2_path),
                "v3_source_path": str(v3_path.resolve()),
                "sha256": observed_hash,
                "hash_preserved": True,
            }
        )

    claim_rows: list[dict[str, Any]] = []
    for claim in covered.to_dict("records"):
        month = claim["month"]
        source = source_by_month[month]
        parsed = parse_explicit_report_events(
            text_by_month[month],
            report_month=month,
            channel=claim["channel"],
            **source,
        )
        event_rows.extend(parsed)
        expected_count = int(claim["event_rows"])
        claim_rows.append(
            {
                "gap_id": claim["gap_id"],
                "report_month": month,
                "channel": claim["channel"],
                "denominator_v2_event_rows": expected_count,
                "normalized_source_records": len(parsed),
                "count_reconciled": expected_count == len(parsed),
                "count_delta": len(parsed) - expected_count,
                "explicit_nonzero_event_set": len(parsed) > 0,
            }
        )

    events = pd.DataFrame(event_rows)
    if events.empty or events["event_id"].duplicated().any() or events["source_record_sha256"].duplicated().any():
        raise ValueError("normalized event identities must be non-empty and unique")
    events = events.sort_values(["report_month", "channel", "event_date", "local_code", "event_id"]).reset_index(drop=True)
    events_path = root / "normalized_events.parquet"
    events.to_parquet(events_path, index=False)
    pd.DataFrame(report_rows).to_csv(root / "raw_report_inventory.csv", index=False)
    claim_frame = pd.DataFrame(claim_rows)
    claim_frame.to_csv(root / "claim_event_count_reconciliation.csv", index=False)

    completion = events[["event_id", "report_month", "channel", "event_type", "event_date", "local_code", "source_record_sha256", "known_at_utc", "mapping_status", "known_at_status"]].copy()
    completion["mapping_complete"] = False
    completion["timestamp_complete"] = False
    completion["admissible"] = False
    completion["mapping_required_evidence"] = completion["local_code"].map(
        lambda code: "independently admitted point-in-time local-code-to-stable-listing map" if code else "manual classification and identifier for aggregate/derivative report record"
    )
    completion["timestamp_required_evidence"] = "contemporaneous official notice timestamp proving event-time knowledge"
    completion["next_action"] = "resolve mapping and event-time publication evidence without Refinitiv; keep out of shared history"
    completion.to_csv(root / "mapping_timestamp_completion_plan.csv", index=False)

    mismatches = claim_frame.loc[~claim_frame["count_reconciled"]]
    status = "normalized_event_candidate_with_upstream_count_deltas" if len(mismatches) else "normalized_event_candidate"
    hashes = _artifact_hashes(root, exclude={"manifest.json", "artifact_hashes.json"})
    hashes_path = root / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "denominator_v2_dir": str(denominator_root),
        "denominator_v2_manifest_sha256": _sha_path(denominator_root / "manifest.json"),
        "validation_v2_dir": str(validation_root),
        "validation_v2_manifest_sha256": _sha_path(validation_root / "manifest.json"),
        "covered_month_channel_claims": int(len(covered)),
        "unique_raw_reports": int(len(report_rows)),
        "normalized_events": int(len(events)),
        "events_by_channel": {str(key): int(value) for key, value in events.groupby("channel").size().items()},
        "claims_with_upstream_count_delta": int(len(mismatches)),
        "upstream_count_delta_events": int(mismatches["count_delta"].sum()),
        "all_event_sets_nonzero": bool(claim_frame["explicit_nonzero_event_set"].all()),
        "raw_hashes_preserved": bool(pd.DataFrame(report_rows)["hash_preserved"].all()),
        "events_admissible_to_shared_history": 0,
        "absence_is_zero": False,
        "refinitiv_used": False,
        "independent_event_set_validation_status": "not_run",
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "write_once": True,
        "artifact_hashes_sha256": _sha_path(hashes_path),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return JapanG4NormalizedEventArtifacts(run_id, root, manifest_path, events_path, manifest)


def _explicit_section(lines: list[str], heading: str) -> list[str]:
    starts = [index for index, line in enumerate(lines) if line.casefold() == heading.casefold()]
    if len(starts) != 1:
        return []
    start = starts[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        if _is_event_heading(lines[index], heading):
            end = index
            break
    return lines[start:end]


def _is_event_heading(line: str, current: str) -> bool:
    if not line or line.casefold() == current.casefold():
        return False
    return bool(
        re.match(
            r"^(?:New Listings|Delisting|Suspension of Trading|Assignment to|Reassignment|"
            r"Transfer to|Name Change|Change of|Change in|Additional Assignment|"
            r"Exception from|Designation as|Ex-New\b)",
            line,
            flags=re.IGNORECASE,
        )
        or re.match(r"^\d{1,2}(?:-\d+)?\s+.*(?:\(|（)", line)
    )


def _parse_event_line(line: str, report_month: str, channel: str) -> dict[str, Any] | None:
    dates = list(ANY_DATE.finditer(line))
    if not dates:
        return None
    first = dates[0]
    event_date = _date_from_match(first)
    tail = line[first.end() :].strip()
    code_match = CODE.match(tail)
    local_code = code_match.group("code") if code_match else ""
    after_code = tail[code_match.end() :].strip() if code_match else tail
    semantics = _time_semantics(tail, event_date)
    if channel == DELISTING_CHANNEL and not local_code:
        return None
    if channel == HALT_CHANNEL and not local_code and not semantics["has_halt_semantics"]:
        return None
    issue_name = _issue_name(after_code, channel)
    if not issue_name:
        issue_name = after_code
    return {
        "event_date": event_date.isoformat(),
        "event_start_jst": _timestamp(event_date, semantics["start_time"]),
        "interval_end_jst": _timestamp(semantics["end_date"], semantics["end_time"]),
        "event_time_semantics": semantics["label"],
        "local_code": local_code,
        "issue_name": issue_name,
    }


def _time_semantics(line: str, event_date: datetime.date) -> dict[str, Any]:
    normalized = line.replace("：", ":").replace("；", ";")
    match = TIME_RANGE.search(normalized)
    if match:
        start_time = _parse_time(match.group("start"))
        end_time = _parse_time(match.group("end"))
        end_date = _resolve_end_date(match.group("end_date"), event_date) if match.group("end_date") else event_date
        return {"label": "multi_day_interval" if end_date != event_date else "intraday_interval", "start_time": start_time, "end_time": end_time, "end_date": end_date, "has_halt_semantics": True}
    if "終日" in line:
        return {"label": "whole_day_without_inferred_clock", "start_time": None, "end_time": None, "end_date": event_date, "has_halt_semantics": True}
    if "期間売買停止" in line:
        short_dates = list(SHORT_DATE.finditer(line))
        end_date = _resolve_end_date(short_dates[-1].group(0), event_date) if short_dates else event_date
        return {"label": "date_interval_without_clock", "start_time": None, "end_time": None, "end_date": end_date, "has_halt_semantics": True}
    single = re.search(TIME_TOKEN, normalized)
    if single:
        return {"label": "start_time_only", "start_time": _parse_time(single.group(0)), "end_time": None, "end_date": event_date, "has_halt_semantics": True}
    return {"label": "date_only_reported_suspension", "start_time": None, "end_time": None, "end_date": event_date, "has_halt_semantics": False}


def _issue_name(value: str, channel: str) -> str:
    if channel == DELISTING_CHANNEL:
        return value.strip()
    positions = []
    for pattern in [TIME_TOKEN, r"終日", r"期間売買停止", r"当該銘柄"]:
        match = re.search(pattern, value)
        if match:
            positions.append(match.start())
    return value[: min(positions)].strip() if positions else value.strip()


def _parse_time(value: str) -> time:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 3:
        digits = "0" + digits
    return time(int(digits[:-2]), int(digits[-2:]))


def _resolve_end_date(value: str, start: datetime.date) -> datetime.date:
    full = ANY_DATE.search(value)
    if full:
        return _date_from_match(full)
    short = SHORT_DATE.search(value)
    if not short:
        return start
    year = start.year + (1 if int(short.group("month")) < start.month else 0)
    return datetime(year, int(short.group("month")), int(short.group("day"))).date()


def _date_from_match(match: re.Match[str]) -> datetime.date:
    return datetime(int(match.group("year")), int(match.group("month")), int(match.group("day"))).date()


def _timestamp(day: datetime.date | None, clock: time | None) -> str:
    if day is None or clock is None:
        return ""
    return datetime.combine(day, clock, JST).isoformat()


def _verify_artifact(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    hashes_path = root / "artifact_hashes.json"
    if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
        raise ValueError(f"artifact hash inventory mismatch: {root}")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or _sha_path(path) != expected:
            raise ValueError(f"artifact file hash mismatch: {root}/{relative}")
    return manifest


def _bools(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _extract_text(body: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages)


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha_path(path) for path in sorted(root.rglob("*")) if path.is_file() and str(path.relative_to(root)) not in exclude}


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()
