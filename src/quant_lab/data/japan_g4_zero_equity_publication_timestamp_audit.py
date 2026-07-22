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

import pandas as pd


UTC = timezone.utc
V16_RUN_ID = "JPN-G4-HALTS-200607-202606-V16"
TARGET_MONTH = "2014-11"
TARGET_CHANNEL = "delisting_non_trading"
COMPONENT = "japan_g4_zero_equity_publication_timestamp_audit"
EXPECTED_PDF_SHA256 = "78c43a0d10e728268f634a5acb11af9322707a80feb49c6933742d81d9c87080"
EXPECTED_PARENT_SHA256 = "21d840b8ef6c83d8600ba530d09a94bc2498319a2906cd1813184f551398af58"

ACCEPTED_COLUMNS = (
    "month",
    "channel",
    "publication_timestamp",
    "publication_timestamp_semantics",
    "known_at_utc",
    "known_at_rule",
    "source_sha256",
)
CANDIDATE_COLUMNS = (
    "candidate_id",
    "month",
    "channel",
    "candidate_type",
    "candidate_value",
    "source_role",
    "source_path",
    "source_sha256",
    "publication_semantics_proven",
    "pit_safe",
    "accepted",
    "known_at_utc",
    "rejection_reason",
)


@dataclass(frozen=True)
class JapanG4ZeroEquityPublicationTimestampAuditArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    blocker_path: Path
    manifest: dict[str, Any]


def produce_japan_g4_zero_equity_publication_timestamp_audit(
    *,
    run_id: str,
    v16_root: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_zero_equity_publication_timestamp_audit",
) -> JapanG4ZeroEquityPublicationTimestampAuditArtifacts:
    """Audit retained V16 bytes for an exact 2014-11 report publication timestamp."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    source_root = Path(v16_root).resolve()
    source_manifest_path = source_root / "manifest.json"
    source_manifest = _json(source_manifest_path)
    _validate_v16(source_root, source_manifest)

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)

    plan = _verified_csv(source_root, source_manifest, "acquisition_plan.csv")
    no_event = _verified_csv(source_root, source_manifest, "no_event_evidence.csv")
    report = _verified_csv(source_root, source_manifest, "report_event_admission.csv")
    gaps = _verified_csv(source_root, source_manifest, "unresolved_gaps.csv")
    claim, report_rows = _exact_claim(plan, no_event, report, gaps)

    inventory_path, inventory_row = _locate_source_inventory(
        source_root, source_manifest, claim["source_sha256"]
    )
    inventory_root = inventory_path.parent
    pdf_path = (inventory_root / inventory_row["pdf_path"]).resolve()
    parent_path = (inventory_root / inventory_row["parent_index_path"]).resolve()
    cdx_path = (inventory_root / "raw" / "cdx" / "2014-11_parent_index.json").resolve()
    for path in (pdf_path, parent_path, cdx_path):
        _validate_source_member(source_root, source_manifest, path)

    retained_dir = root / "retained"
    retained_dir.mkdir()
    retained = {
        "official_monthly_report_pdf": retained_dir / "2014-11-monthly-report.pdf",
        "archived_parent_index": retained_dir / "2014-11-parent-index.html",
        "parent_index_cdx": retained_dir / "2014-11-parent-index-cdx.json",
    }
    for source, destination in (
        (pdf_path, retained["official_monthly_report_pdf"]),
        (parent_path, retained["archived_parent_index"]),
        (cdx_path, retained["parent_index_cdx"]),
    ):
        shutil.copyfile(source, destination)

    claim_path = root / "claim_identity.csv"
    report_rows_path = root / "exact_report_rows.csv"
    source_evidence_path = root / "retained_source_evidence.csv"
    candidates_path = root / "timestamp_candidates.csv"
    accepted_path = root / "accepted_publication_timestamp.csv"
    blocker_path = root / "blocker_audit.csv"
    summary_path = root / "summary.md"

    _atomic_csv(claim_path, pd.DataFrame([claim]))
    _atomic_csv(report_rows_path, report_rows)
    source_evidence = _source_evidence(inventory_path, inventory_row, retained)
    _atomic_csv(source_evidence_path, source_evidence)
    candidates = _timestamp_candidates(retained, inventory_row)
    _atomic_csv(candidates_path, candidates)
    _atomic_csv(accepted_path, pd.DataFrame(columns=ACCEPTED_COLUMNS))

    blocker = pd.DataFrame(
        [
            {
                "blocker_id": hashlib.sha256(
                    f"{TARGET_MONTH}|{TARGET_CHANNEL}|exact-publication-timestamp-absent".encode()
                ).hexdigest()[:24],
                "month": TARGET_MONTH,
                "channel": TARGET_CHANNEL,
                "status": "blocked_exact_publication_timestamp_absent",
                "claim_source_sha256": claim["source_sha256"],
                "candidate_count": len(candidates),
                "accepted_candidate_count": 0,
                "publication_timestamp": "",
                "known_at_utc": "",
                "zero_events_inferred": False,
                "pit_complete": False,
                "reason": (
                    "retained bytes prove the exact report and zero in-scope equity rows, "
                    "but contain no timestamp with public-availability semantics"
                ),
                "required_evidence": (
                    "official JPX publication metadata or an archived first-publication "
                    "observation bound to the exact report hash"
                ),
            }
        ]
    )
    _atomic_csv(blocker_path, blocker)
    _atomic_text(summary_path, _summary(run_id, claim, candidates))

    outputs = [
        claim_path,
        report_rows_path,
        source_evidence_path,
        candidates_path,
        accepted_path,
        blocker_path,
        summary_path,
        *retained.values(),
    ]
    manifest = {
        "schema_version": "1.0-offline-zero-equity-publication-audit",
        "run_id": run_id,
        "component": COMPONENT,
        "status": "blocked_exact_publication_timestamp_absent",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {"month": TARGET_MONTH, "channel": TARGET_CHANNEL},
        "policy": {
            "offline_only": True,
            "vendor_calls": False,
            "zero_events_inferred": False,
            "document_creation_time_is_publication_time": False,
            "recurring_schedule_is_instance_timestamp": False,
            "archive_capture_is_exact_publication_time": False,
            "http_last_modified_is_publication_time": False,
            "pit_known_at_requires_public_availability_semantics": True,
        },
        "counts": {
            "exact_month_channels": 1,
            "exact_report_rows": len(report_rows),
            "in_scope_equity_rows": 0,
            "retained_sources": len(retained),
            "timestamp_candidates": len(candidates),
            "accepted_publication_timestamps": 0,
        },
        "inputs": {
            "v16_manifest": _descriptor(source_manifest_path),
            "v16_acquisition_plan": _descriptor(source_root / "acquisition_plan.csv"),
            "v16_no_event_evidence": _descriptor(source_root / "no_event_evidence.csv"),
            "v16_report_event_admission": _descriptor(source_root / "report_event_admission.csv"),
            "v16_unresolved_gaps": _descriptor(source_root / "unresolved_gaps.csv"),
            "source_inventory": _descriptor(inventory_path),
        },
        "outputs": {
            path.relative_to(root).as_posix(): _descriptor(path) for path in outputs
        },
        "publication_timestamp_accepted": False,
        "pit_complete": False,
        "independent_validation_required": True,
        "primary_v16_modified": False,
        "promotion_eligible": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4ZeroEquityPublicationTimestampAuditArtifacts(
        run_id, root, manifest_path, blocker_path, manifest
    )


def _validate_v16(root: Path, manifest: dict[str, Any]) -> None:
    if not root.is_dir() or not (
        manifest.get("run_id") == V16_RUN_ID
        and manifest.get("component") == "halt_suspension_events"
        and manifest.get("artifact_type") == "japan_g4_halt_suspension_history_producer"
    ):
        raise ValueError("input is not the authoritative V16 halt artifact")


def _verified_csv(root: Path, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    path = root / name
    expected = manifest.get("output_sha256", {}).get(name)
    if not expected or _sha256(path) != expected:
        raise ValueError(f"V16 output hash mismatch: {name}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _exact_claim(
    plan: pd.DataFrame,
    no_event: pd.DataFrame,
    report: pd.DataFrame,
    gaps: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    plan_rows = plan.loc[
        plan["month"].eq(TARGET_MONTH) & plan["channel"].eq(TARGET_CHANNEL)
    ]
    evidence_rows = no_event.loc[
        no_event["month"].eq(TARGET_MONTH) & no_event["channel"].eq(TARGET_CHANNEL)
    ]
    report_rows = report.loc[
        report["report_month"].eq(TARGET_MONTH) & report["channel"].eq(TARGET_CHANNEL)
    ].copy()
    gap_rows = gaps.loc[
        gaps["month"].eq(TARGET_MONTH)
        & gaps["scope"].eq(TARGET_CHANNEL)
        & gaps["gap_type"].eq("report_zero_equity_publication_timestamp_unresolved")
    ]
    if not (len(plan_rows) == len(evidence_rows) == len(gap_rows) == 1):
        raise ValueError("V16 zero-equity month-channel identity is not one-to-one")
    if len(report_rows) != 1:
        raise ValueError("V16 exact delisting report row denominator changed")
    if not (
        plan_rows.iloc[0]["event_rows"] == "0"
        and plan_rows.iloc[0]["no_event_evidence"].lower() == "true"
        and report_rows["in_research_scope"].str.lower().eq("false").all()
        and report_rows["claim_exact_set_validated"].str.lower().eq("true").all()
    ):
        raise ValueError("V16 claim is not an exact zero-in-scope-equity report claim")
    hashes = set(report_rows["source_sha256"]) | set(evidence_rows["source_sha256"])
    if hashes != {EXPECTED_PDF_SHA256}:
        raise ValueError("V16 claim source hash changed")
    plan_row, evidence_row, gap_row = (
        plan_rows.iloc[0], evidence_rows.iloc[0], gap_rows.iloc[0]
    )
    claim = {
        "plan_id": plan_row["plan_id"],
        "month": TARGET_MONTH,
        "channel": TARGET_CHANNEL,
        "window_start": plan_row["window_start"],
        "window_end": plan_row["window_end"],
        "event_rows": 0,
        "exact_report_rows": len(report_rows),
        "in_scope_equity_rows": 0,
        "evidence_id": evidence_row["evidence_id"],
        "source_id": evidence_row["source_id"],
        "source_sha256": evidence_row["source_sha256"],
        "unresolved_gap_id": gap_row["gap_id"],
        "zero_events_inferred": False,
        "publication_timestamp_present": False,
        "pit_complete": False,
    }
    return claim, report_rows.reset_index(drop=True)


def _locate_source_inventory(
    root: Path, manifest: dict[str, Any], source_sha256: str
) -> tuple[Path, dict[str, str]]:
    matches: list[tuple[Path, dict[str, str]]] = []
    for path in root.glob("inputs/tse_wayback_lifecycle/packets/*/producer/source_inventory.csv"):
        relative = path.relative_to(root).as_posix()
        if manifest.get("output_sha256", {}).get(relative) != _sha256(path):
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = frame.loc[
            frame["report_month"].eq(TARGET_MONTH)
            & frame["pdf_sha256"].eq(source_sha256)
        ]
        for row in rows.to_dict("records"):
            matches.append((path.resolve(), row))
    if len(matches) != 1:
        raise ValueError("exact retained 2014-11 source inventory row not found")
    return matches[0]


def _validate_source_member(root: Path, manifest: dict[str, Any], path: Path) -> None:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("retained source escapes V16 artifact") from exc
    expected = manifest.get("output_sha256", {}).get(relative)
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ValueError(f"retained source hash mismatch: {relative}")


def _source_evidence(
    inventory_path: Path,
    inventory: dict[str, str],
    retained: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    original_paths = {
        "official_monthly_report_pdf": inventory_path.parent / inventory["pdf_path"],
        "archived_parent_index": inventory_path.parent / inventory["parent_index_path"],
        "parent_index_cdx": inventory_path.parent / "raw/cdx/2014-11_parent_index.json",
    }
    for role, retained_path in retained.items():
        rows.append(
            {
                "month": TARGET_MONTH,
                "channel": TARGET_CHANNEL,
                "source_role": role,
                "original_path": str(original_paths[role].resolve()),
                "retained_path": str(retained_path.resolve()),
                "sha256": _sha256(retained_path),
                "bytes": retained_path.stat().st_size,
                "official_pdf_url": inventory["official_pdf_url"],
                "archived_parent_url": inventory["parent_index_wayback_url"],
                "source_supports_exact_report_identity": role == "official_monthly_report_pdf",
                "source_supports_exact_publication_timestamp": False,
            }
        )
    return pd.DataFrame(rows)


def _timestamp_candidates(
    retained: dict[str, Path], inventory: dict[str, str]
) -> pd.DataFrame:
    pdf = retained["official_monthly_report_pdf"].read_bytes()
    html = retained["archived_parent_index"].read_text(encoding="utf-8", errors="replace")
    cdx = _json(retained["parent_index_cdx"])
    pdf_sha = _sha256(retained["official_monthly_report_pdf"])
    html_sha = _sha256(retained["archived_parent_index"])
    cdx_sha = _sha256(retained["parent_index_cdx"])
    creation = _pdf_date(pdf, b"CreationDate")
    modified = _pdf_date(pdf, b"ModDate")
    update = _html_match(html, r"Update\s*:\s*([A-Z][a-z]{2}\.?\s+\d{1,2},\s+\d{4})")
    schedule = _html_match(html, r"posts the monthly statistics report on the (\d{1,2}(?:st|nd|rd|th))")
    cdx_timestamp = str(inventory["parent_index_cdx_timestamp"])
    if not _cdx_has_timestamp(cdx, cdx_timestamp):
        raise ValueError("retained CDX bytes do not contain selected capture timestamp")
    http_headers = json.loads(inventory["pdf_http_headers"])
    specs = (
        (
            "pdf_creation_metadata",
            creation,
            "official_monthly_report_pdf",
            retained["official_monthly_report_pdf"],
            pdf_sha,
            "PDF CreationDate proves document generation, not public availability",
        ),
        (
            "pdf_modification_metadata",
            modified,
            "official_monthly_report_pdf",
            retained["official_monthly_report_pdf"],
            pdf_sha,
            "PDF ModDate proves document modification, not public availability",
        ),
        (
            "parent_index_page_update_date",
            update,
            "archived_parent_index",
            retained["archived_parent_index"],
            html_sha,
            "page-wide update date is later and is not bound to the 2014-11 PDF publication",
        ),
        (
            "recurring_publication_schedule",
            schedule,
            "archived_parent_index",
            retained["archived_parent_index"],
            html_sha,
            "recurring schedule is not an exact timestamp for this report instance",
        ),
        (
            "archive_parent_index_capture",
            _cdx_iso(cdx_timestamp),
            "parent_index_cdx",
            retained["parent_index_cdx"],
            cdx_sha,
            "archive capture is only a later observation upper bound and does not bind the exact PDF hash",
        ),
        (
            "current_http_last_modified",
            str(http_headers.get("last-modified", "")),
            "official_monthly_report_pdf",
            retained["official_monthly_report_pdf"],
            pdf_sha,
            "2025 object Last-Modified metadata is not the historical publication time",
        ),
    )
    rows = []
    for ordinal, (kind, value, role, path, digest, reason) in enumerate(specs):
        if not value:
            raise ValueError(f"expected retained timestamp candidate missing: {kind}")
        rows.append(
            {
                "candidate_id": hashlib.sha256(
                    f"{TARGET_MONTH}|{TARGET_CHANNEL}|{kind}|{value}|{digest}".encode()
                ).hexdigest()[:24],
                "month": TARGET_MONTH,
                "channel": TARGET_CHANNEL,
                "candidate_type": kind,
                "candidate_value": value,
                "source_role": role,
                "source_path": str(path.resolve()),
                "source_sha256": digest,
                "publication_semantics_proven": False,
                "pit_safe": False,
                "accepted": False,
                "known_at_utc": "",
                "rejection_reason": reason,
                "ordinal": ordinal,
            }
        )
    return pd.DataFrame(rows).sort_values("ordinal").drop(columns="ordinal")


def _pdf_date(payload: bytes, key: bytes) -> str:
    match = re.search(rb"/" + key + rb"\s*\(D:(\d{14})([+-])(\d{2})'(\d{2})'\)", payload)
    if not match:
        return ""
    stamp, sign, hour, minute = (part.decode("ascii") for part in match.groups())
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}{sign}{hour}:{minute}"


def _html_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _cdx_has_timestamp(payload: Any, timestamp: str) -> bool:
    return any(isinstance(row, list) and row and str(row[0]) == timestamp for row in payload[1:])


def _cdx_iso(value: str) -> str:
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC).isoformat()


def _summary(run_id: str, claim: dict[str, Any], candidates: pd.DataFrame) -> str:
    return (
        f"# Japan G4 Zero-Equity Publication Timestamp Audit: {run_id}\n\n"
        f"- Month-channel: `{claim['month']}|{claim['channel']}`\n"
        f"- Exact report rows: {claim['exact_report_rows']}\n"
        "- In-scope common-equity rows: 0 (source-backed, not inferred)\n"
        f"- Timestamp candidates audited: {len(candidates)}\n"
        "- Accepted publication timestamps: 0\n"
        "- PIT status: blocked\n\n"
        "The retained official PDF contains document creation and modification metadata, "
        "but no public-availability timestamp. The retained parent index provides a later "
        "page update, a recurring publication schedule, and an archive observation; none "
        "is an exact publication timestamp bound to the report hash.\n"
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
