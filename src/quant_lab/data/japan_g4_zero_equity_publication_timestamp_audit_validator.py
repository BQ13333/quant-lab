from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc
PRODUCER_COMPONENT = "japan_g4_zero_equity_publication_timestamp_audit"
VALIDATOR_COMPONENT = "japan_g4_zero_equity_publication_timestamp_audit_independent_validator"
TARGET_MONTH = "2014-11"
TARGET_CHANNEL = "delisting_non_trading"
EXPECTED_PDF_SHA256 = "78c43a0d10e728268f634a5acb11af9322707a80feb49c6933742d81d9c87080"
EXPECTED_PARENT_SHA256 = "21d840b8ef6c83d8600ba530d09a94bc2498319a2906cd1813184f551398af58"


def validate_japan_g4_zero_equity_publication_timestamp_audit(
    *,
    producer_dir: str | Path,
    validation_run_id: str,
    artifact_root: str | Path = "artifacts/japan_g4_zero_equity_publication_timestamp_validation",
) -> tuple[Path, dict[str, Any]]:
    if not validation_run_id.strip():
        raise ValueError("validation_run_id must not be empty")
    producer = Path(producer_dir).resolve()
    manifest = _json(producer / "manifest.json")
    destination = (Path(artifact_root) / validation_run_id).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    issues: list[str] = []
    checks: list[dict[str, Any]] = []

    _check(checks, issues, "producer_component", manifest.get("component") == PRODUCER_COMPONENT)
    _check(
        checks,
        issues,
        "producer_blocked_status",
        manifest.get("status") == "blocked_exact_publication_timestamp_absent",
    )
    _check(checks, issues, "producer_no_promotion", manifest.get("promotion_eligible") is False)
    _check(checks, issues, "producer_no_timestamp", manifest.get("publication_timestamp_accepted") is False)
    _check(checks, issues, "producer_not_pit_complete", manifest.get("pit_complete") is False)

    outputs = manifest.get("outputs", {})
    for relative, descriptor in sorted(outputs.items()):
        path = producer / relative
        _check(
            checks,
            issues,
            f"output_descriptor:{relative}",
            _safe_member(producer, path) and _descriptor_matches(path, descriptor),
        )

    claim = _csv(producer / "claim_identity.csv", issues, "claim")
    rows = _csv(producer / "exact_report_rows.csv", issues, "report_rows")
    sources = _csv(producer / "retained_source_evidence.csv", issues, "sources")
    candidates = _csv(producer / "timestamp_candidates.csv", issues, "candidates")
    accepted = _csv(producer / "accepted_publication_timestamp.csv", issues, "accepted")
    blockers = _csv(producer / "blocker_audit.csv", issues, "blockers")

    claim_ok = bool(
        len(claim) == 1
        and claim.iloc[0].get("month") == TARGET_MONTH
        and claim.iloc[0].get("channel") == TARGET_CHANNEL
        and claim.iloc[0].get("event_rows") == "0"
        and claim.iloc[0].get("exact_report_rows") == "1"
        and claim.iloc[0].get("in_scope_equity_rows") == "0"
        and claim.iloc[0].get("source_sha256") == EXPECTED_PDF_SHA256
        and claim.iloc[0].get("zero_events_inferred", "").lower() == "false"
        and claim.iloc[0].get("pit_complete", "").lower() == "false"
    )
    _check(checks, issues, "exact_claim_identity", claim_ok)
    report_ok = bool(
        len(rows) == 1
        and rows.iloc[0].get("report_month") == TARGET_MONTH
        and rows.iloc[0].get("channel") == TARGET_CHANNEL
        and rows.iloc[0].get("instrument_class") == "straight_bond"
        and rows.iloc[0].get("in_research_scope", "").lower() == "false"
        and rows.iloc[0].get("claim_exact_set_validated", "").lower() == "true"
        and rows.iloc[0].get("source_sha256") == EXPECTED_PDF_SHA256
    )
    _check(checks, issues, "exact_report_row_reconciliation", report_ok)

    source_roles = set(sources.get("source_role", pd.Series(dtype=str)))
    _check(
        checks,
        issues,
        "retained_source_denominator",
        len(sources) == 3
        and source_roles
        == {"official_monthly_report_pdf", "archived_parent_index", "parent_index_cdx"},
    )
    for row in sources.to_dict("records"):
        path = Path(row.get("retained_path", ""))
        _check(
            checks,
            issues,
            f"retained_hash:{row.get('source_role', '')}",
            _safe_member(producer, path)
            and path.is_file()
            and _sha256(path) == row.get("sha256")
            and str(path.stat().st_size) == row.get("bytes"),
        )

    reconstruction = _reconstruct(producer, issues)
    candidate_map = candidates.set_index("candidate_type")["candidate_value"].to_dict() if len(candidates) else {}
    _check(checks, issues, "candidate_denominator", len(candidates) == 6)
    _check(checks, issues, "independent_candidate_values", candidate_map == reconstruction)
    _check(
        checks,
        issues,
        "all_candidates_rejected",
        bool(
            len(candidates) == 6
            and candidates["publication_semantics_proven"].str.lower().eq("false").all()
            and candidates["pit_safe"].str.lower().eq("false").all()
            and candidates["accepted"].str.lower().eq("false").all()
            and candidates["known_at_utc"].eq("").all()
            and candidates["rejection_reason"].ne("").all()
        ),
    )
    _check(checks, issues, "accepted_timestamp_empty", accepted.empty)
    blocker_ok = bool(
        len(blockers) == 1
        and blockers.iloc[0].get("month") == TARGET_MONTH
        and blockers.iloc[0].get("channel") == TARGET_CHANNEL
        and blockers.iloc[0].get("status") == "blocked_exact_publication_timestamp_absent"
        and blockers.iloc[0].get("accepted_candidate_count") == "0"
        and blockers.iloc[0].get("publication_timestamp") == ""
        and blockers.iloc[0].get("known_at_utc") == ""
        and blockers.iloc[0].get("zero_events_inferred", "").lower() == "false"
        and blockers.iloc[0].get("pit_complete", "").lower() == "false"
    )
    _check(checks, issues, "durable_blocker", blocker_ok)

    checks_frame = pd.DataFrame(checks)
    issues_frame = pd.DataFrame(
        ({"issue": issue, "severity": "critical"} for issue in issues),
        columns=("issue", "severity"),
    )
    reconstruction_frame = pd.DataFrame(
        ({"candidate_type": key, "candidate_value": value} for key, value in reconstruction.items())
    )
    checks_path = destination / "validation_checks.csv"
    issues_path = destination / "validation_issues.csv"
    reconstruction_path = destination / "independent_evidence_reconstruction.csv"
    summary_path = destination / "summary.md"
    _atomic_csv(checks_path, checks_frame)
    _atomic_csv(issues_path, issues_frame)
    _atomic_csv(reconstruction_path, reconstruction_frame)
    status = "pass_blocker_confirmed" if not issues else "fail"
    _atomic_text(
        summary_path,
        (
            f"# Independent Zero-Equity Timestamp Audit Validation: {validation_run_id}\n\n"
            f"- Status: {status}\n"
            f"- Checks: {len(checks_frame)}\n"
            f"- Issues: {len(issues_frame)}\n"
            "- Publication timestamp accepted: no\n"
            "- Durable blocker independently confirmed: yes\n"
        ),
    )
    validation_outputs = [checks_path, issues_path, reconstruction_path, summary_path]
    validation_manifest = {
        "schema_version": "1.0-independent-offline-blocker-validation",
        "run_id": validation_run_id,
        "component": VALIDATOR_COMPONENT,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer": _descriptor(producer / "manifest.json"),
        "counts": {
            "checks": len(checks_frame),
            "issues": len(issues_frame),
            "reconstructed_candidates": len(reconstruction_frame),
            "accepted_publication_timestamps": 0,
        },
        "outputs": {
            path.relative_to(destination).as_posix(): _descriptor(path)
            for path in validation_outputs
        },
        "publication_timestamp_accepted": False,
        "blocker_confirmed": not issues,
        "zero_events_inferred": False,
        "pit_complete": False,
        "refinitiv_calls": False,
        "promotion_eligible": False,
    }
    manifest_path = destination / "manifest.json"
    _atomic_json(manifest_path, validation_manifest)
    return manifest_path, validation_manifest


def _reconstruct(producer: Path, issues: list[str]) -> dict[str, str]:
    pdf = producer / "retained/2014-11-monthly-report.pdf"
    html_path = producer / "retained/2014-11-parent-index.html"
    cdx_path = producer / "retained/2014-11-parent-index-cdx.json"
    if not (pdf.is_file() and html_path.is_file() and cdx_path.is_file()):
        issues.append("retained_source_missing")
        return {}
    if _sha256(pdf) != EXPECTED_PDF_SHA256 or _sha256(html_path) != EXPECTED_PARENT_SHA256:
        issues.append("retained_fixed_hash_mismatch")
        return {}
    payload = pdf.read_bytes()
    html = html_path.read_text(encoding="utf-8", errors="replace")
    creation = _extract_pdf_timestamp(payload, b"CreationDate")
    modification = _extract_pdf_timestamp(payload, b"ModDate")
    update_match = re.search(
        r"Update\s*:\s*([A-Z][a-z]{2}\.?\s+\d{1,2},\s+\d{4})",
        html,
        flags=re.IGNORECASE,
    )
    schedule_match = re.search(
        r"monthly statistics report on the (\d{1,2}(?:st|nd|rd|th))",
        html,
        flags=re.IGNORECASE,
    )
    cdx = _json(cdx_path)
    capture = ""
    for row in cdx[1:]:
        if isinstance(row, list) and row and str(row[0]) == "20150310160942":
            capture = datetime.strptime(str(row[0]), "%Y%m%d%H%M%S").replace(tzinfo=UTC).isoformat()
            break
    sources = pd.read_csv(producer / "retained_source_evidence.csv", dtype=str, keep_default_na=False)
    pdf_row = sources.loc[sources["source_role"].eq("official_monthly_report_pdf")]
    last_modified = ""
    if len(pdf_row) == 1:
        source_inventory = _json(producer / "manifest.json").get("inputs", {}).get("source_inventory", {})
        source_inventory_path = Path(str(source_inventory.get("path", "")))
        if _descriptor_matches(source_inventory_path, source_inventory):
            inventory = pd.read_csv(source_inventory_path, dtype=str, keep_default_na=False)
            row = inventory.loc[inventory["report_month"].eq(TARGET_MONTH)]
            if len(row) == 1:
                headers = json.loads(row.iloc[0]["pdf_http_headers"])
                last_modified = str(headers.get("last-modified", ""))
    result = {
        "pdf_creation_metadata": creation,
        "pdf_modification_metadata": modification,
        "parent_index_page_update_date": update_match.group(1) if update_match else "",
        "recurring_publication_schedule": schedule_match.group(1) if schedule_match else "",
        "archive_parent_index_capture": capture,
        "current_http_last_modified": last_modified,
    }
    if any(not value for value in result.values()):
        issues.append("independent_candidate_reconstruction_incomplete")
    return result


def _extract_pdf_timestamp(payload: bytes, key: bytes) -> str:
    expression = rb"/" + key + rb"\s*\(D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})'(\d{2})'\)"
    match = re.search(expression, payload)
    if not match:
        return ""
    parts = [part.decode("ascii") for part in match.groups()]
    year, month, day, hour, minute, second, sign, offset_hour, offset_minute = parts
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}{sign}{offset_hour}:{offset_minute}"


def _check(checks: list[dict[str, Any]], issues: list[str], name: str, passed: bool) -> None:
    checks.append({"check": name, "passed": bool(passed)})
    if not passed:
        issues.append(name)


def _csv(path: Path, issues: list[str], label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        issues.append(f"{label}_unreadable:{type(exc).__name__}")
        return pd.DataFrame()


def _safe_member(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return path.is_file() and not path.is_symlink()
    except (OSError, ValueError):
        return False


def _descriptor_matches(path: Path, descriptor: dict[str, Any]) -> bool:
    try:
        return bool(
            path.is_file()
            and str(path.resolve()) == str(descriptor.get("path", ""))
            and _sha256(path) == descriptor.get("sha256")
            and path.stat().st_size == int(descriptor.get("bytes", -1))
        )
    except (OSError, TypeError, ValueError):
        return False


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
