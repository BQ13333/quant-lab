from __future__ import annotations

import hashlib
import json
import os
import re
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

from quant_lab.data.japan_g4_halt_issuer_timestamp_4283 import (
    ALLOWED_ISSUER_HOSTS,
    COMPONENT as PRODUCER_COMPONENT,
    KNOWN_AT_RULE,
    SCHEMA_VERSION as PRODUCER_SCHEMA_VERSION,
    TARGET_CODE,
    TARGET_EVENT_DATE,
    TARGET_EVENT_ID,
    TARGET_RECORD_SHA256,
    TARGET_SOURCE_SHA256,
)


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
COMPONENT = "japan_g4_halt_4283_issuer_publication_timestamp_audit_validation"
SCHEMA_VERSION = "1.0-independent-4283-issuer-domain-validation"


@dataclass(frozen=True)
class JapanG4IssuerTimestamp4283ValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    issues_path: Path
    manifest: dict[str, Any]


def validate_japan_g4_halt_issuer_timestamp_4283(
    *,
    run_id: str,
    producer_manifest_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_issuer_timestamp_4283_validation",
) -> JapanG4IssuerTimestamp4283ValidationArtifacts:
    """Independently recompute packet integrity and semantic acceptance for code 4283."""
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{7,127}", run_id):
        raise ValueError("invalid immutable validation run_id")
    producer_manifest_file = Path(producer_manifest_path).resolve()
    producer_root = producer_manifest_file.parent
    producer = json.loads(producer_manifest_file.read_text(encoding="utf-8"))
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)

    issues: list[dict[str, str]] = []

    def issue(check_id: str, detail: str) -> None:
        issues.append({"severity": "critical", "check_id": check_id, "detail": detail})

    if not (
        producer.get("component") == PRODUCER_COMPONENT
        and producer.get("schema_version") == PRODUCER_SCHEMA_VERSION
        and producer.get("source_policy", {}).get("offline_only") is True
        and producer.get("source_policy", {}).get("vendor_calls") is False
        and producer.get("source_policy", {}).get("issuer_domain_separate_from_jpx_contract") is True
        and producer.get("source_policy", {}).get("structured_publication_metadata_required") is True
        and producer.get("source_policy", {}).get("inferred_publication_time_used") is False
    ):
        issue("producer_contract_invalid", "producer manifest policy or schema mismatch")

    outputs = producer.get("outputs", {})
    required_outputs = {
        "input_event_row.csv", "source_inventory.csv", "timestamp_claim.csv",
        "accepted_timestamp_recovery.csv", "rejected_timestamp_recovery.csv",
        "audit_findings.csv", "summary.md", f"evidence/event/{TARGET_SOURCE_SHA256}.pdf",
    }
    if not required_outputs.issubset(outputs):
        issue("required_output_missing", "producer output denominator is incomplete")
    for relative, descriptor in outputs.items():
        path = producer_root / relative
        if path.is_symlink() or not path.is_file():
            issue("output_path_invalid", relative)
            continue
        if descriptor.get("sha256") != _sha256(path) or descriptor.get("bytes") != path.stat().st_size:
            issue("output_hash_mismatch", relative)

    claims = _read_csv(producer_root / "timestamp_claim.csv", issue)
    accepted = _read_csv(producer_root / "accepted_timestamp_recovery.csv", issue)
    rejected = _read_csv(producer_root / "rejected_timestamp_recovery.csv", issue)
    inventory = _read_csv(producer_root / "source_inventory.csv", issue)
    event_rows = _read_csv(producer_root / "input_event_row.csv", issue)

    if len(claims) != 1:
        issue("claim_denominator_invalid", f"expected 1, found {len(claims)}")
    if len(event_rows) != 1:
        issue("event_denominator_invalid", f"expected 1, found {len(event_rows)}")
    if len(inventory) != 2 or set(inventory.get("source_role", [])) != {
        "jpx_event_source", "issuer_publication_source"
    }:
        issue("source_inventory_invalid", "expected exact event and issuer source roles")

    if len(event_rows) == 1:
        event = event_rows.iloc[0]
        expected = {
            "event_id": TARGET_EVENT_ID,
            "event_type": "delisting_non_trading_interval",
            "local_code": TARGET_CODE,
            "event_date": TARGET_EVENT_DATE,
            "source_sha256": TARGET_SOURCE_SHA256,
            "source_record_id": TARGET_RECORD_SHA256,
        }
        for column, value in expected.items():
            if str(event.get(column, "")) != value:
                issue("event_identity_mismatch", f"{column}={event.get(column, '')}")
    event_copy = producer_root / "evidence" / "event" / f"{TARGET_SOURCE_SHA256}.pdf"
    if event_copy.is_file() and _sha256(event_copy) != TARGET_SOURCE_SHA256:
        issue("event_source_hash_mismatch", str(event_copy))

    claim_accepted = len(claims) == 1 and str(claims.iloc[0].get("status", "")) == "accepted"
    if claim_accepted:
        _validate_accepted_claim(producer_root, claims.iloc[0], inventory, issue)
        if len(accepted) != 1 or not rejected.empty:
            issue("accepted_rejected_partition_invalid", "accepted claim is not partitioned one-to-one")
        if producer.get("status") != "complete" or producer.get("promotion_eligible") is not True:
            issue("accepted_manifest_status_invalid", "accepted claim is not promotion eligible")
    else:
        if not accepted.empty or len(rejected) != 1:
            issue("failed_closed_partition_invalid", "rejected packet must have 0 accepted and 1 rejected")
        if len(claims) == 1:
            row = claims.iloc[0]
            if any(str(row.get(column, "")) for column in ("known_at_utc", "known_at_rule", "publication_date")):
                issue("rejected_claim_leaks_timestamp", "rejected claim contains timestamp fields")
            if not str(row.get("rejection_reason", "")):
                issue("rejected_claim_missing_reason", "failed-closed claim has no blocker")
        if producer.get("status") != "blocked_fail_closed" or producer.get("promotion_eligible") is not False:
            issue("failed_closed_manifest_status_invalid", "rejected packet has unsafe status")

    issues_frame = pd.DataFrame(issues, columns=("severity", "check_id", "detail"))
    issues_path = root / "validation_issues.csv"
    _atomic_csv(issues_path, issues_frame)
    summary_path = root / "summary.md"
    validation_passed = issues_frame.empty
    _atomic_text(
        summary_path,
        f"# Independent Code 4283 Issuer Timestamp Validation: {run_id}\n\n"
        f"- Producer packet valid: {validation_passed}\n"
        f"- Producer claim accepted: {claim_accepted}\n"
        f"- Validation issues: {len(issues_frame)}\n"
        f"- Outcome: {'accepted' if claim_accepted and validation_passed else 'validated_fail_closed' if validation_passed else 'invalid'}\n",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "component": COMPONENT,
        "status": "complete" if validation_passed else "failed",
        "producer": {
            "manifest_path": str(producer_manifest_file),
            "manifest_sha256": _sha256(producer_manifest_file),
            "run_id": producer.get("run_id", ""),
        },
        "counts": {
            "validation_issues": len(issues_frame),
            "accepted_claims": int(claim_accepted and validation_passed),
            "rejected_claims": int(not claim_accepted),
        },
        "packet_valid": validation_passed,
        "claim_accepted": claim_accepted and validation_passed,
        "promotion_eligible": claim_accepted and validation_passed,
        "refinitiv_calls": False,
        "outputs": {
            issues_path.name: _descriptor(issues_path),
            summary_path.name: _descriptor(summary_path),
        },
    }
    manifest_path = root / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return JapanG4IssuerTimestamp4283ValidationArtifacts(
        run_id, root, manifest_path, issues_path, manifest
    )


def _validate_accepted_claim(
    producer_root: Path,
    claim: pd.Series,
    inventory: pd.DataFrame,
    issue: Any,
) -> None:
    if not (
        str(claim.get("event_id", "")) == TARGET_EVENT_ID
        and str(claim.get("local_code", "")) == TARGET_CODE
        and str(claim.get("event_date", "")) == TARGET_EVENT_DATE
        and str(claim.get("event_source_sha256", "")) == TARGET_SOURCE_SHA256
        and str(claim.get("source_record_id", "")) == TARGET_RECORD_SHA256
    ):
        issue("accepted_event_binding_invalid", "claim identity mismatch")
    source_path = Path(str(claim.get("issuer_source_path", "")))
    try:
        source_path.relative_to(producer_root)
    except ValueError:
        issue("issuer_source_outside_packet", str(source_path))
        return
    if source_path.is_symlink() or not source_path.is_file():
        issue("issuer_source_missing", str(source_path))
        return
    digest = _sha256(source_path)
    if digest != str(claim.get("issuer_source_sha256", "")):
        issue("issuer_source_hash_mismatch", digest)
    issuer_rows = inventory.loc[inventory.get("source_role", pd.Series(dtype=str)).eq("issuer_publication_source")]
    if len(issuer_rows) != 1 or str(issuer_rows.iloc[0].get("actual_sha256", "")) != digest:
        issue("issuer_inventory_binding_invalid", digest)

    url = str(claim.get("issuer_source_url", ""))
    parsed = urllib.parse.urlparse(url)
    if not (
        parsed.scheme == "https" and parsed.hostname in ALLOWED_ISSUER_HOSTS
        and parsed.port is None and parsed.path not in {"", "/"}
        and not parsed.query and not parsed.fragment and not parsed.username and not parsed.password
    ):
        issue("issuer_domain_invalid", url)

    soup = BeautifulSoup(source_path.read_bytes(), "html.parser")
    visible = " ".join(soup.stripped_strings)
    dates: set[str] = set()
    for tag in soup.select("meta"):
        key = str(tag.get("property") or tag.get("name") or "").lower().replace("_", "")
        if key in {"article:publishedtime", "datepublished", "publishdate", "publicationdate"}:
            normalized = _extract_date(str(tag.get("content") or ""))
            if normalized:
                dates.add(normalized)
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            value = json.loads(script.string or script.get_text() or "null")
        except json.JSONDecodeError:
            continue
        for raw_date in _walk_date_published(value):
            normalized = _extract_date(raw_date)
            if normalized:
                dates.add(normalized)
    publication = str(claim.get("publication_date", ""))
    if dates != {publication}:
        issue("structured_publication_date_invalid", repr(sorted(dates)))
    try:
        published = date.fromisoformat(publication)
    except ValueError:
        issue("publication_date_invalid", publication)
        return
    if published >= date.fromisoformat(TARGET_EVENT_DATE):
        issue("publication_not_pre_event", publication)

    canonical = _canonical(visible)
    if not (
        "パナソニックインフォメーションシステムズ" in canonical
        or "panasonicinformationsystemscoltd" in canonical
    ):
        issue("issuer_identity_missing", "issuer name not in retained bytes")
    if not re.search(r"(?<!\d)4283(?!\d)", unicodedata.normalize("NFKC", visible)):
        issue("local_code_missing", "4283 not in retained bytes")
    if not any(fragment in canonical for fragment in ("20150729", "2015年7月29日", "平成27年7月29日")):
        issue("event_date_missing", "delisting date not in retained bytes")
    if not any(fragment in canonical for fragment in ("上場廃止", "delisting", "delisted")):
        issue("delisting_semantics_missing", "delisting semantics not in retained bytes")
    expected_known_at = datetime.combine(
        published + timedelta(days=1), time.min, JST
    ).astimezone(UTC).isoformat()
    if not (
        str(claim.get("known_at_rule", "")) == KNOWN_AT_RULE
        and str(claim.get("known_at_utc", "")) == expected_known_at
    ):
        issue("known_at_derivation_invalid", str(claim.get("known_at_utc", "")))


def _read_csv(path: Path, issue: Any) -> pd.DataFrame:
    if not path.is_file():
        issue("csv_missing", str(path))
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # pragma: no cover - defensive artifact boundary
        issue("csv_unreadable", f"{path}: {exc}")
        return pd.DataFrame()


def _walk_date_published(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        if "datePublished" in value:
            found.append(str(value["datePublished"]))
        for nested in value.values():
            found.extend(_walk_date_published(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_walk_date_published(nested))
    return found


def _extract_date(value: str) -> str:
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", unicodedata.normalize("NFKC", value))
    if not match:
        return ""
    try:
        return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    except ValueError:
        return ""


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)
