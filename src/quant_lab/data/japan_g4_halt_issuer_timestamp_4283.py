from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
V16_RUN_ID = "JPN-G4-HALTS-200607-202606-V16"
COMPONENT = "japan_g4_halt_4283_issuer_publication_timestamp_audit"
SCHEMA_VERSION = "1.0-4283-issuer-domain-bounded-immutable"
TARGET_EVENT_ID = "jpx-report-d17a012930a77f7926cbf78e"
TARGET_CODE = "4283"
TARGET_EVENT_DATE = "2015-07-29"
TARGET_SOURCE_SHA256 = "76e65d06d7f5c8f6b0d1f401c80827fdd00532d7d6147602a9d493d2615b90f9"
TARGET_RECORD_SHA256 = "74b789093277bee80e28263acbfe112df23dbc362aef92bdae08eee1f2def4fc"
ALLOWED_ISSUER_HOSTS = ("is-c.panasonic.co.jp",)
KNOWN_AT_RULE = "date_only_conservative_upper_bound_next_midnight_jst"
MAX_SOURCE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class JapanG4IssuerTimestamp4283Artifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    claim_path: Path
    accepted_path: Path
    manifest: dict[str, Any]


CLAIM_COLUMNS = (
    "event_id", "event_type", "local_code", "issue_name", "event_date",
    "event_source_url", "event_source_sha256", "source_record_id",
    "issuer_source_url", "issuer_source_path", "issuer_source_sha256",
    "publication_date", "publication_precision", "known_at_utc", "known_at_rule",
    "status", "evidence_checks", "rejection_reason",
)
INVENTORY_COLUMNS = (
    "source_role", "original_path", "retained_path", "source_url", "expected_sha256",
    "actual_sha256", "bytes", "status", "rejection_reason",
)
FINDING_COLUMNS = ("severity", "check_id", "passed", "evidence", "remediation")


def produce_japan_g4_halt_issuer_timestamp_4283(
    *,
    run_id: str,
    events_path: str | Path,
    event_source_path: str | Path,
    issuer_source_path: str | Path | None = None,
    issuer_source_url: str = "",
    issuer_source_sha256: str = "",
    publication_date: str = "",
    artifact_root: str | Path = "artifacts/japan_g4_halt_issuer_timestamp_4283",
) -> JapanG4IssuerTimestamp4283Artifacts:
    """Audit one retained issuer page for the 4283 delisting, without network access."""
    _validate_run_id(run_id)
    events = Path(events_path).resolve()
    source_manifest_path = events.parent / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    _validate_v16_source(events, source_manifest_path, source_manifest)
    event_row = _target_event(pd.read_parquet(events).fillna(""))

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)

    event_source = Path(event_source_path)
    if event_source.is_symlink():
        raise ValueError("event source cannot be a symlink")
    event_source = event_source.resolve()
    if not event_source.is_file() or _sha256(event_source) != TARGET_SOURCE_SHA256:
        raise ValueError("event source does not match the exact retained JPX PDF")
    event_retained = root / "evidence" / "event" / f"{TARGET_SOURCE_SHA256}.pdf"
    _atomic_copy(event_source, event_retained)

    issuer_path = Path(issuer_source_path) if issuer_source_path else None
    checks, reasons, issuer_retained, issuer_digest, issuer_bytes = _audit_issuer_evidence(
        root=root,
        source_path=issuer_path,
        source_url=issuer_source_url,
        expected_sha256=issuer_source_sha256,
        publication_date=publication_date,
    )
    accepted = not reasons
    known_at = _known_at(publication_date) if accepted else ""

    event_input = pd.DataFrame([event_row.to_dict()])
    claim = pd.DataFrame([{
        "event_id": TARGET_EVENT_ID,
        "event_type": str(event_row["event_type"]),
        "local_code": TARGET_CODE,
        "issue_name": str(event_row["issue_name"]),
        "event_date": TARGET_EVENT_DATE,
        "event_source_url": str(event_row["source_url"]),
        "event_source_sha256": TARGET_SOURCE_SHA256,
        "source_record_id": TARGET_RECORD_SHA256,
        "issuer_source_url": issuer_source_url,
        "issuer_source_path": str(issuer_retained) if issuer_retained else "",
        "issuer_source_sha256": issuer_digest,
        "publication_date": publication_date if accepted else "",
        "publication_precision": "issuer_page_structured_date_only" if accepted else "",
        "known_at_utc": known_at,
        "known_at_rule": KNOWN_AT_RULE if accepted else "",
        "status": "accepted" if accepted else "rejected",
        "evidence_checks": "|".join(name for name, passed in checks.items() if passed),
        "rejection_reason": "|".join(reasons),
    }], columns=CLAIM_COLUMNS)
    accepted_frame = claim.loc[claim["status"].eq("accepted")].copy()
    rejected_frame = claim.loc[claim["status"].eq("rejected")].copy()

    inventory = pd.DataFrame([
        {
            "source_role": "jpx_event_source",
            "original_path": str(event_source),
            "retained_path": str(event_retained),
            "source_url": str(event_row["source_url"]),
            "expected_sha256": TARGET_SOURCE_SHA256,
            "actual_sha256": _sha256(event_retained),
            "bytes": event_retained.stat().st_size,
            "status": "accepted",
            "rejection_reason": "",
        },
        {
            "source_role": "issuer_publication_source",
            "original_path": str(issuer_path) if issuer_path else "",
            "retained_path": str(issuer_retained) if issuer_retained else "",
            "source_url": issuer_source_url,
            "expected_sha256": issuer_source_sha256,
            "actual_sha256": issuer_digest,
            "bytes": issuer_bytes,
            "status": "accepted" if accepted else "rejected",
            "rejection_reason": "|".join(reasons),
        },
    ], columns=INVENTORY_COLUMNS)
    findings = pd.DataFrame([
        {
            "severity": "critical" if not passed else "info",
            "check_id": name,
            "passed": passed,
            "evidence": "satisfied" if passed else name,
            "remediation": "" if passed else _remediation(name),
        }
        for name, passed in checks.items()
    ], columns=FINDING_COLUMNS)

    input_path = root / "input_event_row.csv"
    inventory_path = root / "source_inventory.csv"
    claim_path = root / "timestamp_claim.csv"
    accepted_path = root / "accepted_timestamp_recovery.csv"
    rejected_path = root / "rejected_timestamp_recovery.csv"
    findings_path = root / "audit_findings.csv"
    summary_path = root / "summary.md"
    for path, frame in (
        (input_path, event_input),
        (inventory_path, inventory),
        (claim_path, claim),
        (accepted_path, accepted_frame),
        (rejected_path, rejected_frame),
        (findings_path, findings),
    ):
        _atomic_csv(path, frame)
    _atomic_text(summary_path, _summary(run_id, accepted, reasons, issuer_digest))

    outputs = (
        input_path, inventory_path, claim_path, accepted_path, rejected_path,
        findings_path, summary_path, event_retained,
    ) + ((issuer_retained,) if issuer_retained else ())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "component": COMPONENT,
        "status": "complete" if accepted else "blocked_fail_closed",
        "source_policy": {
            "offline_only": True,
            "vendor_calls": False,
            "issuer_domain_separate_from_jpx_contract": True,
            "allowed_issuer_hosts": list(ALLOWED_ISSUER_HOSTS),
            "redirects_accepted": False,
            "http_headers_used_as_publication_time": False,
            "filesystem_timestamps_used_as_publication_time": False,
            "inferred_publication_time_used": False,
            "structured_publication_metadata_required": True,
            "known_at_rule": KNOWN_AT_RULE,
        },
        "counts": {
            "target_events": 1,
            "accepted_events": int(accepted),
            "rejected_events": int(not accepted),
            "retained_event_sources": 1,
            "retained_issuer_sources": int(issuer_retained is not None),
            "critical_blockers": len(reasons),
        },
        "inputs": {
            "v16_manifest": _descriptor(source_manifest_path),
            "v16_events": _descriptor(events),
            "event_source_original": _descriptor(event_source),
        },
        "outputs": {path.relative_to(root).as_posix(): _descriptor(path) for path in outputs},
        "promotion_eligible": accepted,
        "primary_halt_history_modified": False,
        "independent_validation_required": True,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4IssuerTimestamp4283Artifacts(
        run_id, root, manifest_path, claim_path, accepted_path, manifest
    )


def _validate_v16_source(events: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    if not events.is_file() or not manifest_path.is_file():
        raise ValueError("V16 events and manifest must exist")
    if not (
        manifest.get("run_id") == V16_RUN_ID
        and manifest.get("component") == "halt_suspension_events"
        and manifest.get("artifact_type") == "japan_g4_halt_suspension_history_producer"
    ):
        raise ValueError("events are not the exact V16 halt source")
    if manifest.get("output_sha256", {}).get(events.name) != _sha256(events):
        raise ValueError("V16 event hash mismatch")


def _target_event(events: pd.DataFrame) -> pd.Series:
    required = {
        "event_id", "event_type", "local_code", "issue_name", "event_date",
        "source_url", "source_sha256", "source_record_id",
    }
    if not required.issubset(events.columns):
        raise ValueError("V16 event source is missing required columns")
    rows = events.loc[events["event_id"].eq(TARGET_EVENT_ID)]
    if len(rows) != 1:
        raise ValueError("4283 delisting event denominator is not one-to-one")
    row = rows.iloc[0]
    if not (
        str(row["event_type"]) == "delisting_non_trading_interval"
        and str(row["local_code"]) == TARGET_CODE
        and str(row["event_date"]) == TARGET_EVENT_DATE
        and str(row["source_sha256"]) == TARGET_SOURCE_SHA256
        and str(row["source_record_id"]) == TARGET_RECORD_SHA256
        and _is_jpx_event_url(str(row["source_url"]))
    ):
        raise ValueError("4283 event identity does not match the bounded contract")
    return row


def _audit_issuer_evidence(
    *,
    root: Path,
    source_path: Path | None,
    source_url: str,
    expected_sha256: str,
    publication_date: str,
) -> tuple[dict[str, bool], list[str], Path | None, str, int]:
    checks = {
        "issuer_source_provided": source_path is not None,
        "issuer_source_not_symlink": bool(source_path is not None and not source_path.is_symlink()),
        "issuer_source_regular_file": False,
        "issuer_source_size_bounded": False,
        "issuer_expected_sha256_provided": bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)),
        "issuer_source_sha256_match": False,
        "issuer_url_exact_allowed_domain": _is_allowed_issuer_url(source_url),
        "issuer_source_html": False,
        "structured_publication_date_unique": False,
        "publication_date_matches_input": False,
        "publication_precedes_event": False,
        "issuer_identity_in_source": False,
        "local_code_in_source": False,
        "delisting_event_date_in_source": False,
        "delisting_semantics_in_source": False,
    }
    retained: Path | None = None
    digest = ""
    byte_count = 0
    if source_path is not None and not source_path.is_symlink():
        resolved = source_path.resolve()
        checks["issuer_source_regular_file"] = resolved.is_file()
        if resolved.is_file():
            byte_count = resolved.stat().st_size
            checks["issuer_source_size_bounded"] = 0 < byte_count <= MAX_SOURCE_BYTES
            digest = _sha256(resolved)
            checks["issuer_source_sha256_match"] = digest == expected_sha256
            suffix = resolved.suffix.lower()
            checks["issuer_source_html"] = suffix in {".html", ".htm"}
            retained = root / "quarantine" / "issuer" / f"{digest}{suffix or '.bin'}"
            _atomic_copy(resolved, retained)
            if checks["issuer_source_size_bounded"] and checks["issuer_source_html"]:
                payload = resolved.read_bytes()
                text, structured_dates = _issuer_html_evidence(payload)
                checks["structured_publication_date_unique"] = len(structured_dates) == 1
                checks["publication_date_matches_input"] = (
                    len(structured_dates) == 1 and publication_date in structured_dates
                )
                try:
                    published = date.fromisoformat(publication_date)
                    checks["publication_precedes_event"] = published < date.fromisoformat(TARGET_EVENT_DATE)
                except ValueError:
                    checks["publication_precedes_event"] = False
                canonical = _canonical(text)
                checks["issuer_identity_in_source"] = (
                    "パナソニックインフォメーションシステムズ" in canonical
                    or "panasonicinformationsystemscoltd" in canonical
                )
                checks["local_code_in_source"] = bool(
                    re.search(r"(?<!\d)4283(?!\d)", unicodedata.normalize("NFKC", text))
                )
                checks["delisting_event_date_in_source"] = any(
                    fragment in canonical
                    for fragment in ("20150729", "2015年7月29日", "平成27年7月29日")
                )
                checks["delisting_semantics_in_source"] = (
                    "上場廃止" in canonical or "delisting" in canonical or "delisted" in canonical
                )
    reasons = [name for name, passed in checks.items() if not passed]
    if retained is not None and not reasons:
        accepted_path = root / "evidence" / "issuer" / retained.name
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(retained, accepted_path)
        retained = accepted_path
    return checks, reasons, retained, digest, byte_count


def _issuer_html_evidence(payload: bytes) -> tuple[str, set[str]]:
    soup = BeautifulSoup(payload, "html.parser")
    text = " ".join(soup.stripped_strings)
    values: list[str] = []
    for tag in soup.find_all("meta"):
        key = str(tag.get("property") or tag.get("name") or "").casefold().replace("_", "")
        if key in {
            "article:publishedtime", "datepublished", "publishdate", "publicationdate",
        }:
            values.append(str(tag.get("content") or ""))
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            parsed = json.loads(script.string or script.get_text() or "null")
        except json.JSONDecodeError:
            continue
        values.extend(_jsonld_dates(parsed))
    return text, {normalized for value in values if (normalized := _normalize_date(value))}


def _jsonld_dates(value: Any) -> list[str]:
    if isinstance(value, dict):
        found = [str(value["datePublished"])] if "datePublished" in value else []
        for nested in value.values():
            found.extend(_jsonld_dates(nested))
        return found
    if isinstance(value, list):
        return [item for nested in value for item in _jsonld_dates(nested)]
    return []


def _normalize_date(value: str) -> str:
    clean = unicodedata.normalize("NFKC", str(value)).strip()
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", clean)
    if not match:
        return ""
    try:
        return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    except ValueError:
        return ""


def _is_allowed_issuer_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_ISSUER_HOSTS
        and parsed.port is None
        and parsed.path.startswith("/")
        and parsed.path != "/"
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def _is_jpx_event_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
        and parsed.path.endswith("17_idou1507.pdf")
        and not parsed.query
        and not parsed.fragment
    )


def _known_at(publication_date: str) -> str:
    published = date.fromisoformat(publication_date)
    return datetime.combine(published + timedelta(days=1), time.min, JST).astimezone(UTC).isoformat()


def _canonical(value: str) -> str:
    clean = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in clean if character.isalnum())


def _remediation(check_id: str) -> str:
    if check_id == "issuer_source_provided":
        return "retain the exact issuer page bytes before rerunning this offline contract"
    if check_id.startswith("structured_publication") or check_id.startswith("publication_date"):
        return "obtain issuer bytes with an explicit structured datePublished value"
    if check_id == "publication_precedes_event":
        return "obtain a genuine issuer publication dated before 2015-07-29"
    if check_id.startswith("issuer_source_sha256") or check_id == "issuer_expected_sha256_provided":
        return "supply and independently pin the retained issuer response SHA-256"
    return "supply exact source-backed evidence satisfying this invariant"


def _summary(run_id: str, accepted: bool, reasons: list[str], digest: str) -> str:
    return (
        f"# Japan G4 Code 4283 Issuer Timestamp Audit: {run_id}\n\n"
        f"- Accepted timestamp claims: {int(accepted)}\n"
        f"- Rejected timestamp claims: {int(not accepted)}\n"
        f"- Issuer source SHA-256: {digest or 'not retained'}\n"
        f"- Critical blockers: {len(reasons)}\n"
        f"- Blocker IDs: {'; '.join(reasons) if reasons else 'none'}\n"
        "- Contract is offline-only and separate from the JPX-only timestamp packet.\n"
        "- HTTP headers, filesystem times, filenames, and remembered URLs are not publication evidence.\n"
        "- No primary V16 integration is performed.\n"
    )


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{7,127}", run_id):
        raise ValueError("run_id must be an immutable uppercase artifact identifier")


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)
