from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
OFFICIAL_HOST = "www.jpx.co.jp"
TARGET_GAP_TYPE = "historical_pit_timestamp_unresolved"
RECOVERED_STATUS = "recovered"
UNRESOLVED_STATUS = "unresolved"
KNOWN_AT_STATUS = "conservative_after_official_decision_publication_date"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

RECOVERY_COLUMNS = (
    "input_ordinal",
    "gap_id",
    "event_id",
    "local_code",
    "issue_name",
    "event_date",
    "recovery_status",
    "known_at_utc",
    "known_at_status",
    "publication_date",
    "decision_date",
    "timestamp_rule",
    "source_url",
    "archive_source_id",
    "archive_source_path",
    "archive_source_sha256",
    "archive_record_index",
    "archive_record_sha256",
    "notice_source_path",
    "notice_source_sha256",
    "candidate_count",
    "evidence_checks",
    "unresolved_reason",
)
SOURCE_COLUMNS = (
    "source_id",
    "source_kind",
    "official_source",
    "source_url",
    "local_path",
    "sha256",
    "bytes",
    "publication_date",
    "event_ids",
)
VALIDATION_COLUMNS = (
    "event_id",
    "recovery_status",
    "archive_record_valid",
    "notice_hash_valid",
    "notice_code_valid",
    "notice_name_valid",
    "notice_date_valid",
    "known_at_valid",
    "passed",
    "issues",
)

Fetcher = Callable[[str, float], bytes]


@dataclass(frozen=True)
class JapanG4HaltTimestampRecoveryArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    recovery_path: Path
    unresolved_path: Path
    validation_manifest_path: Path
    manifest: dict[str, Any]
    validation_manifest: dict[str, Any]


@dataclass(frozen=True)
class _ArchiveRecord:
    source_id: str
    source_path: Path
    source_sha256: str
    record_index: int
    record_sha256: str
    title: str
    subject: str
    source_url: str
    publication_date: date


def produce_japan_g4_halt_timestamp_recovery(
    *,
    run_id: str,
    unresolved_gaps_path: str | Path,
    events_path: str | Path,
    archive_source_manifest_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_timestamp_recovery",
    expected_target_count: int | None = 42,
    fetcher: Fetcher | None = None,
    request_timeout: float = 30.0,
) -> JapanG4HaltTimestampRecoveryArtifacts:
    """Recover date-safe PIT bounds from official JPX delisting decisions only."""
    _validate_run_id(run_id)
    gaps_path = Path(unresolved_gaps_path).resolve()
    event_source_path = Path(events_path).resolve()
    archive_manifest_path = Path(archive_source_manifest_path).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    gaps = _read_csv(gaps_path)
    required_gap_columns = {"gap_id", "gap_type", "event_id", "local_code"}
    if not required_gap_columns.issubset(gaps.columns):
        raise ValueError("unresolved gap input is missing required columns")
    targets = gaps.loc[gaps["gap_type"].eq(TARGET_GAP_TYPE)].copy()
    targets.insert(0, "input_ordinal", range(len(targets)))
    if expected_target_count is not None and len(targets) != expected_target_count:
        raise ValueError(
            f"expected {expected_target_count} {TARGET_GAP_TYPE} rows, found {len(targets)}"
        )
    if targets["event_id"].eq("").any() or targets["event_id"].duplicated().any():
        raise ValueError("target event IDs must be populated and unique")

    events = pd.read_parquet(event_source_path).fillna("")
    required_event_columns = {
        "event_id",
        "event_type",
        "local_code",
        "issue_name",
        "event_date",
    }
    if not required_event_columns.issubset(events.columns):
        raise ValueError("event input is missing required columns")
    event_rows = events.loc[events["event_id"].isin(targets["event_id"])].copy()
    if event_rows["event_id"].duplicated().any() or len(event_rows) != len(targets):
        raise ValueError("target gaps do not map one-to-one to event rows")
    joined = targets.merge(
        event_rows[list(required_event_columns)],
        on=["event_id", "local_code"],
        how="left",
        validate="one_to_one",
    )
    if joined["issue_name"].eq("").any() or not joined["event_type"].eq(
        "delisting_non_trading_interval"
    ).all():
        raise ValueError("target gaps must map to named delisting interval events")

    archive_records, archive_inventory = _load_archive_records(archive_manifest_path)
    decision_records = [
        record for record in archive_records if _is_delisting_decision(record.title)
    ]
    fetch = fetcher or _fetch
    recovery_rows: list[dict[str, Any]] = []
    notice_inventory: list[dict[str, Any]] = []
    for event in joined.sort_values("input_ordinal").to_dict("records"):
        event_day = date.fromisoformat(event["event_date"])
        candidates = [
            record
            for record in decision_records
            if _company_key(record.subject) == _company_key(event["issue_name"])
            and record.publication_date < event_day
        ]
        base = _empty_recovery_row(event, len(candidates))
        if len(candidates) != 1:
            base["unresolved_reason"] = "no_unique_official_delisting_decision_notice"
            recovery_rows.append(base)
            continue

        candidate = candidates[0]
        filename = _notice_filename(event["local_code"], candidate.source_url)
        raw_path = raw_dir / filename
        try:
            payload = fetch(candidate.source_url, request_timeout)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("notice response exceeds byte limit")
            _atomic_bytes(raw_path, payload)
            checks = _notice_checks(
                payload,
                local_code=event["local_code"],
                issue_name=event["issue_name"],
                publication_date=candidate.publication_date,
                source_url=candidate.source_url,
            )
        except Exception as error:
            base.update(_candidate_fields(candidate))
            base["unresolved_reason"] = (
                f"official_notice_fetch_or_parse_failed:{type(error).__name__}:{error}"
            )
            recovery_rows.append(base)
            continue
        base.update(_candidate_fields(candidate))
        base["notice_source_path"] = str(raw_path)
        base["notice_source_sha256"] = _sha256(raw_path)
        base["evidence_checks"] = "|".join(
            key for key, passed in checks.items() if passed
        )
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            base["unresolved_reason"] = "official_notice_evidence_failed:" + "|".join(
                failed
            )
        else:
            known_at = _known_at_after_publication_day(candidate.publication_date)
            base.update(
                {
                    "recovery_status": RECOVERED_STATUS,
                    "known_at_utc": known_at,
                    "known_at_status": KNOWN_AT_STATUS,
                    "publication_date": candidate.publication_date.isoformat(),
                    "decision_date": candidate.publication_date.isoformat(),
                    "timestamp_rule": "first_instant_after_official_publication_date_in_JST",
                    "unresolved_reason": "",
                }
            )
        notice_inventory.append(
            {
                "source_id": f"notice-{event['local_code']}-{candidate.record_sha256[:12]}",
                "source_kind": "jpx_official_delisting_decision_html",
                "official_source": True,
                "source_url": candidate.source_url,
                "local_path": str(raw_path),
                "sha256": _sha256(raw_path),
                "bytes": raw_path.stat().st_size,
                "publication_date": candidate.publication_date.isoformat(),
                "event_ids": event["event_id"],
            }
        )
        recovery_rows.append(base)

    recovery = pd.DataFrame(recovery_rows, columns=RECOVERY_COLUMNS)
    unresolved = recovery.loc[recovery["recovery_status"].eq(UNRESOLVED_STATUS)].copy()
    source_inventory = pd.DataFrame(
        [*archive_inventory, *notice_inventory], columns=SOURCE_COLUMNS
    ).sort_values(["source_kind", "source_id"])
    recovery_path = root / "timestamp_recovery.csv"
    unresolved_path = root / "unresolved_timestamp_gaps.csv"
    source_inventory_path = root / "source_inventory.csv"
    summary_path = root / "summary.md"
    _atomic_csv(recovery_path, recovery)
    _atomic_csv(unresolved_path, unresolved)
    _atomic_csv(source_inventory_path, source_inventory)
    _atomic_text(summary_path, _summary(run_id, recovery))

    outputs = (recovery_path, unresolved_path, source_inventory_path, summary_path)
    recovered_count = int(recovery["recovery_status"].eq(RECOVERED_STATUS).sum())
    unresolved_count = len(recovery) - recovered_count
    manifest = {
        "schema_version": "1.0-isolated-fail-closed",
        "run_id": run_id,
        "component": "japan_g4_halt_pit_timestamp_recovery",
        "status": "recovery_partial" if unresolved_count else "recovery_complete_candidate",
        "official_jpx_source_only": True,
        "refinitiv_calls": False,
        "timestamp_policy": {
            "accepted_notice": "unique official JPX Decision on Delisting notice",
            "known_at_rule": "first instant after the publication date in Asia/Tokyo",
            "current_observation_time_permitted": False,
            "effective_date_permitted_as_known_at": False,
        },
        "counts": {
            "input_rows": len(gaps),
            "target_rows": len(recovery),
            "recovered_rows": recovered_count,
            "unresolved_rows": unresolved_count,
            "archive_json_sources": len(archive_inventory),
            "retained_notice_pages": len(notice_inventory),
        },
        "inputs": {
            "unresolved_gaps": _input_descriptor(gaps_path),
            "events": _input_descriptor(event_source_path),
            "archive_source_manifest": _input_descriptor(archive_manifest_path),
        },
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "independent_validation_required": True,
        "component_scorecard_eligible": False,
        "promotion_eligible": False,
        "promotion_blockers": (
            [f"unresolved_historical_pit_timestamp_rows:{unresolved_count}"]
            if unresolved_count
            else ["isolated_recovery_lane_requires_explicit_shared-lane_admission"]
        ),
        "shared_halt_files_modified": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    validation_manifest_path, validation_manifest = (
        validate_japan_g4_halt_timestamp_recovery(
            producer_dir=root,
            validation_dir=root / "validation",
            expected_target_count=expected_target_count,
        )
    )
    return JapanG4HaltTimestampRecoveryArtifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=manifest_path,
        recovery_path=recovery_path,
        unresolved_path=unresolved_path,
        validation_manifest_path=validation_manifest_path,
        manifest=manifest,
        validation_manifest=validation_manifest,
    )


def validate_japan_g4_halt_timestamp_recovery(
    *,
    producer_dir: str | Path,
    validation_dir: str | Path | None = None,
    expected_target_count: int | None = 42,
) -> tuple[Path, dict[str, Any]]:
    """Independently re-read source bytes and recompute all recovery assertions."""
    producer = Path(producer_dir).resolve()
    destination = (
        Path(validation_dir).resolve()
        if validation_dir is not None
        else producer / "validation"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = producer / "manifest.json"
    manifest = _read_json(manifest_path)
    issues: list[str] = []
    if manifest.get("component") != "japan_g4_halt_pit_timestamp_recovery":
        issues.append("producer_component_invalid")
    if manifest.get("component_scorecard_eligible") is not False or manifest.get(
        "promotion_eligible"
    ) is not False:
        issues.append("fail_closed_promotion_contract_invalid")
    for filename, expected in manifest.get("output_sha256", {}).items():
        path = producer / filename
        if not path.is_file() or _sha256(path) != expected:
            issues.append(f"producer_output_hash_mismatch:{filename}")

    recovery = _read_csv(producer / "timestamp_recovery.csv")
    unresolved = _read_csv(producer / "unresolved_timestamp_gaps.csv")
    inventory = _read_csv(producer / "source_inventory.csv")
    if tuple(recovery.columns) != RECOVERY_COLUMNS:
        issues.append("recovery_schema_invalid")
    if expected_target_count is not None and len(recovery) != expected_target_count:
        issues.append(f"target_count_invalid:{len(recovery)}/{expected_target_count}")
    if recovery["event_id"].duplicated().any():
        issues.append("duplicate_recovery_event_ids")
    expected_unresolved = recovery.loc[
        recovery["recovery_status"].eq(UNRESOLVED_STATUS)
    ].reset_index(drop=True)
    if not unresolved.reset_index(drop=True).equals(expected_unresolved):
        issues.append("unresolved_partition_mismatch")

    inventory_index = {
        row["local_path"]: row for row in inventory.to_dict("records")
    }
    for item in inventory.to_dict("records"):
        path = Path(item["local_path"])
        if (
            not path.is_file()
            or _sha256(path) != item["sha256"]
            or path.stat().st_size != _integer(item["bytes"])
        ):
            issues.append(f"source_inventory_mismatch:{item['source_id']}")

    validation_rows: list[dict[str, Any]] = []
    for row in recovery.to_dict("records"):
        row_issues: list[str] = []
        archive_valid = False
        notice_hash_valid = False
        code_valid = False
        name_valid = False
        date_valid = False
        known_at_valid = False
        if row["recovery_status"] == RECOVERED_STATUS:
            archive_path = Path(row["archive_source_path"])
            try:
                records = _read_json(archive_path)
                record = records[_integer(row["archive_record_index"])]
                record_digest = _json_record_sha256(record)
                published = _record_publication_date(record)
                source_url = _official_url(record.get("url", ""))
                archive_valid = bool(
                    _sha256(archive_path) == row["archive_source_sha256"]
                    and record_digest == row["archive_record_sha256"]
                    and source_url == row["source_url"]
                    and published.isoformat() == row["publication_date"]
                    and _is_delisting_decision(html.unescape(str(record.get("title", ""))))
                    and _company_key(str(record.get("title", "")).split(":", 1)[-1])
                    == _company_key(row["issue_name"])
                )
            except (IndexError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                archive_valid = False
            notice_path = Path(row["notice_source_path"])
            notice_item = inventory_index.get(str(notice_path), {})
            if notice_path.is_file():
                digest = _sha256(notice_path)
                notice_hash_valid = bool(
                    digest == row["notice_source_sha256"]
                    and digest == notice_item.get("sha256")
                )
                try:
                    checks = _notice_checks(
                        notice_path.read_bytes(),
                        local_code=row["local_code"],
                        issue_name=row["issue_name"],
                        publication_date=date.fromisoformat(row["publication_date"]),
                        source_url=row["source_url"],
                    )
                except (OSError, ValueError):
                    checks = {}
                code_valid = bool(checks.get("code"))
                name_valid = bool(checks.get("name"))
                date_valid = bool(checks.get("publication_date"))
            try:
                publication = date.fromisoformat(row["publication_date"])
                known_at_valid = bool(
                    row["decision_date"] == row["publication_date"]
                    and row["known_at_status"] == KNOWN_AT_STATUS
                    and row["timestamp_rule"]
                    == "first_instant_after_official_publication_date_in_JST"
                    and row["known_at_utc"]
                    == _known_at_after_publication_day(publication)
                    and publication < date.fromisoformat(row["event_date"])
                )
            except ValueError:
                known_at_valid = False
            checks_by_name = {
                "archive_record": archive_valid,
                "notice_hash": notice_hash_valid,
                "notice_code": code_valid,
                "notice_name": name_valid,
                "notice_date": date_valid,
                "known_at": known_at_valid,
            }
            row_issues.extend(name for name, passed in checks_by_name.items() if not passed)
        elif row["recovery_status"] == UNRESOLVED_STATUS:
            known_at_valid = bool(
                not row["known_at_utc"]
                and not row["known_at_status"]
                and not row["decision_date"]
                and bool(row["unresolved_reason"])
            )
            if not known_at_valid:
                row_issues.append("unresolved_row_claims_timestamp")
        else:
            row_issues.append("unsupported_recovery_status")
        if row_issues:
            issues.append(f"row_validation_failed:{row['event_id']}:{'|'.join(row_issues)}")
        validation_rows.append(
            {
                "event_id": row["event_id"],
                "recovery_status": row["recovery_status"],
                "archive_record_valid": archive_valid,
                "notice_hash_valid": notice_hash_valid,
                "notice_code_valid": code_valid,
                "notice_name_valid": name_valid,
                "notice_date_valid": date_valid,
                "known_at_valid": known_at_valid,
                "passed": not row_issues,
                "issues": "|".join(row_issues),
            }
        )

    recovered_count = int(recovery["recovery_status"].eq(RECOVERED_STATUS).sum())
    unresolved_count = int(recovery["recovery_status"].eq(UNRESOLVED_STATUS).sum())
    manifest_counts = manifest.get("counts", {})
    if (
        manifest_counts.get("target_rows") != len(recovery)
        or manifest_counts.get("recovered_rows") != recovered_count
        or manifest_counts.get("unresolved_rows") != unresolved_count
        or recovered_count + unresolved_count != len(recovery)
    ):
        issues.append("manifest_count_mismatch")

    details = pd.DataFrame(validation_rows, columns=VALIDATION_COLUMNS)
    details_path = destination / "validation_details.csv"
    _atomic_csv(details_path, details)
    validation_manifest = {
        "schema_version": "1.0-independent",
        "component": "japan_g4_halt_pit_timestamp_recovery_validation",
        "producer_run_id": manifest.get("run_id", ""),
        "producer_manifest_sha256": _sha256(manifest_path),
        "status": "pass" if not issues else "failed_validation",
        "validation_issues": issues,
        "counts": {
            "target_rows": len(recovery),
            "recovered_rows": recovered_count,
            "unresolved_rows": unresolved_count,
            "validated_rows": int(details["passed"].sum()) if len(details) else 0,
        },
        "independent_validation_passed": not issues,
        "component_scorecard_eligible": False,
        "promotion_eligible": False,
        "output_sha256": {details_path.name: _sha256(details_path)},
    }
    validation_manifest_path = destination / "manifest.json"
    _atomic_json(validation_manifest_path, validation_manifest)
    return validation_manifest_path, validation_manifest


def _load_archive_records(
    manifest_path: Path,
) -> tuple[list[_ArchiveRecord], list[dict[str, Any]]]:
    manifest = _read_csv(manifest_path)
    required = {"source_id", "url", "raw_path", "sha256"}
    if not required.issubset(manifest.columns):
        raise ValueError("archive source manifest is missing required columns")
    records: list[_ArchiveRecord] = []
    inventory: list[dict[str, Any]] = []
    for source in manifest.to_dict("records"):
        path = Path(source["raw_path"]).resolve()
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"archive source hash mismatch: {path}")
        source_url = _official_url(source["url"])
        values = _read_json(path)
        if not isinstance(values, list):
            raise ValueError(f"archive source is not a JSON list: {path}")
        inventory.append(
            {
                "source_id": source["source_id"],
                "source_kind": "jpx_official_market_news_archive_json",
                "official_source": True,
                "source_url": source_url,
                "local_path": str(path),
                "sha256": source["sha256"],
                "bytes": path.stat().st_size,
                "publication_date": "",
                "event_ids": "",
            }
        )
        for index, value in enumerate(values):
            if not isinstance(value, dict) or not {
                "title",
                "url",
                "updated_date",
            }.issubset(value):
                continue
            title = html.unescape(str(value["title"]))
            if not _is_delisting_decision(title):
                continue
            records.append(
                _ArchiveRecord(
                    source_id=source["source_id"],
                    source_path=path,
                    source_sha256=source["sha256"],
                    record_index=index,
                    record_sha256=_json_record_sha256(value),
                    title=title,
                    subject=title.split(":", 1)[-1].strip(),
                    source_url=_official_url(str(value["url"])),
                    publication_date=_record_publication_date(value),
                )
            )
    return records, inventory


def _empty_recovery_row(event: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    row = {column: "" for column in RECOVERY_COLUMNS}
    row.update(
        {
            "input_ordinal": event["input_ordinal"],
            "gap_id": event["gap_id"],
            "event_id": event["event_id"],
            "local_code": event["local_code"],
            "issue_name": event["issue_name"],
            "event_date": event["event_date"],
            "recovery_status": UNRESOLVED_STATUS,
            "candidate_count": candidate_count,
        }
    )
    return row


def _candidate_fields(candidate: _ArchiveRecord) -> dict[str, Any]:
    return {
        "publication_date": candidate.publication_date.isoformat(),
        "source_url": candidate.source_url,
        "archive_source_id": candidate.source_id,
        "archive_source_path": str(candidate.source_path),
        "archive_source_sha256": candidate.source_sha256,
        "archive_record_index": candidate.record_index,
        "archive_record_sha256": candidate.record_sha256,
    }


def _notice_checks(
    payload: bytes,
    *,
    local_code: str,
    issue_name: str,
    publication_date: date,
    source_url: str,
) -> dict[str, bool]:
    text = " ".join(BeautifulSoup(payload, "html.parser").stripped_strings)
    url_date = _url_date(source_url)
    date_tokens = {
        publication_date.strftime("%b. %d, %Y"),
        publication_date.strftime("%b %d, %Y"),
        publication_date.strftime("%B %d, %Y"),
    }
    return {
        "official_route": _is_official_notice_url(source_url),
        "code": bool(re.search(rf"(?<!\d){re.escape(local_code)}(?!\d)", text)),
        "name": _company_key(issue_name) in _company_key(text),
        "publication_date": bool(
            url_date == publication_date and any(token in text for token in date_tokens)
        ),
        "decision_heading": "decision on delisting" in html.unescape(text).casefold(),
    }


def _known_at_after_publication_day(publication_date: date) -> str:
    value = datetime.combine(publication_date + timedelta(days=1), time.min, JST)
    return value.astimezone(UTC).isoformat()


def _company_key(value: str) -> str:
    clean = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    clean = clean.replace("＆", "&").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", clean)


def _is_delisting_decision(title: str) -> bool:
    return bool(
        re.match(r"^Decision on Delisting(?:, etc\.)?\s*:", title.strip(), re.I)
    )


def _record_publication_date(record: dict[str, Any]) -> date:
    value = record["updated_date"]
    return date(int(value["year"]), int(value["month"]), int(value["day"]))


def _json_record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _official_url(value: str) -> str:
    url = urllib.parse.urljoin(f"https://{OFFICIAL_HOST}/", str(value))
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != OFFICIAL_HOST:
        raise ValueError(f"non-official JPX URL rejected: {value}")
    return url


def _is_official_notice_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == OFFICIAL_HOST
        and re.match(r"^/english/news/(?:1021|1023)/", parsed.path)
    )


def _url_date(value: str) -> date | None:
    match = re.search(r"/(\d{8})[-_]", urllib.parse.urlparse(value).path)
    return datetime.strptime(match.group(1), "%Y%m%d").date() if match else None


def _notice_filename(local_code: str, source_url: str) -> str:
    stem = Path(urllib.parse.urlparse(source_url).path).stem
    return f"{local_code}-{stem}.html"


def _fetch(url: str, timeout: float) -> bytes:
    if not _is_official_notice_url(url):
        raise ValueError(f"unsupported JPX notice route: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "quant-lab-halt-timestamp-recovery/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    return payload


def _summary(run_id: str, recovery: pd.DataFrame) -> str:
    recovered = int(recovery["recovery_status"].eq(RECOVERED_STATUS).sum())
    unresolved = len(recovery) - recovered
    return (
        f"# Japan G4 Halt PIT Timestamp Recovery: {run_id}\n\n"
        f"- Target historical PIT timestamp gaps: {len(recovery)}\n"
        f"- Recovered from official JPX delisting decisions: {recovered}\n"
        f"- Unresolved: {unresolved}\n"
        "- Known-at rule: first instant after the official publication date in JST.\n"
        "- Current observation time and delisting effective date are never used as known-at.\n"
        "- Scorecard and promotion eligibility remain disabled.\n"
    )


def _input_descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any) -> int:
    return int(str(value))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)
