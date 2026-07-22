from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader


UTC = timezone.utc
V3_RUN_ID = "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3"
V16_RUN_ID = "JPN-G4-HALTS-200607-202606-V16"
TARGET_6174_EVENT = "jpx-report-331f536854ca55fb9e898f15"
MISSING_REASON = "latest_eligible_source_record_missing"
COMPONENT = "japan_g4_quotation_parser_omission_denominator_audit"


@dataclass(frozen=True)
class JapanG4QuotationParserOmissionAuditArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def produce_japan_g4_quotation_parser_omission_audit(
    *,
    run_id: str,
    mapping_v3_dir: str | Path,
    halt_v16_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_quotation_parser_omission_audit",
) -> JapanG4QuotationParserOmissionAuditArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    v3 = Path(mapping_v3_dir).resolve()
    v16 = Path(halt_v16_dir).resolve()
    _verify_v3(v3)
    _verify_v16(v16)

    decisions = _csv(v3 / "mapping_decisions.csv")
    accepted = _csv(v3 / "accepted_mappings.csv")
    mapping_audit = _csv(v16 / "mapping_audit.csv")
    if not (
        len(decisions) == 48
        and len(accepted) == 23
        and accepted["decision"].eq("accepted").all()
        and accepted["event_id"].is_unique
    ):
        raise ValueError("V3 mapping denominator changed")

    group_a = _population_a(accepted)
    group_b = _population_b(decisions)
    group_c = _population_c(mapping_audit)
    findings = pd.DataFrame(
        [
            {
                "finding_id": "unsupported_eleven_row_denominator",
                "status": "blocked",
                "observed_count": 0,
                "claimed_count": 11,
                "detail": "no sealed artifact enumerates an exact eleven-row parser-omission denominator",
            },
            {
                "finding_id": "accepted_raw_pdf_mapping_population",
                "status": "audited_not_a_repair_denominator",
                "observed_count": len(group_a),
                "claimed_count": "",
                "detail": "23 raw-PDF mappings are already admitted; baseline normalized presence is 1 and absence is 22",
            },
            {
                "finding_id": "verified_missing_source_parser_omissions",
                "status": "zero_verified",
                "observed_count": int(group_b["verified_parser_omission"].sum()),
                "claimed_count": "",
                "detail": "eight selected source PDFs contain zero exact-code text occurrences and their normalized parquets contain zero exact-code rows",
            },
            {
                "finding_id": "code_6174_quotation_packet",
                "status": "blocked_no_packet",
                "observed_count": 0,
                "claimed_count": 1,
                "detail": "code 6174 has no admitted quotation request or PDF assignment",
            },
        ]
    )

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    paths = {
        "population_a_admitted_raw_pdf_mappings.csv": group_a,
        "population_b_source_record_missing.csv": group_b,
        "population_c_code_6174.csv": group_c,
        "audit_findings.csv": findings,
        "accepted_parser_repairs.csv": pd.DataFrame(
            columns=["event_id", "local_code", "source_record_sha256"]
        ),
    }
    for name, frame in paths.items():
        _atomic_csv(root / name, frame)
    _atomic_text(
        root / "summary.md",
        f"# Quotation Parser Omission Denominator Audit: {run_id}\n\n"
        "- Claimed eleven-row denominator: not sealed\n"
        "- Verified parser omissions in eight source-record-missing rows: 0\n"
        "- Accepted parser repairs: 0\n"
        "- Code 6174 quotation packet: absent\n"
        "- Promotion eligible: no\n",
    )
    output_files = [root / name for name in paths] + [root / "summary.md"]
    manifest = {
        "schema_version": "1.0-fail-closed-denominator-audit",
        "run_id": run_id,
        "component": COMPONENT,
        "status": "blocked_unsealed_eleven_row_denominator",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "offline_only": True,
            "vendor_calls": False,
            "parser_repair_admitted": False,
            "ric_inferred": False,
            "timestamp_inferred": False,
            "zero_events_inferred": False,
        },
        "counts": {
            "v3_decisions": len(decisions),
            "already_admitted_raw_pdf_mappings": len(group_a),
            "baseline_normalized_absent_in_admitted_population": int(
                (~group_a["baseline_normalized_present"]).sum()
            ),
            "baseline_normalized_present_in_admitted_population": int(
                group_a["baseline_normalized_present"].sum()
            ),
            "source_record_missing_rows": len(group_b),
            "verified_parser_omissions_in_missing_rows": int(
                group_b["verified_parser_omission"].sum()
            ),
            "code_6174_rows": len(group_c),
            "accepted_parser_repairs": 0,
        },
        "inputs": {
            "v3_manifest": _descriptor(v3 / "manifest.json"),
            "v3_mapping_decisions": _descriptor(v3 / "mapping_decisions.csv"),
            "v3_accepted_mappings": _descriptor(v3 / "accepted_mappings.csv"),
            "v16_manifest": _descriptor(v16 / "manifest.json"),
            "v16_mapping_audit": _descriptor(v16 / "mapping_audit.csv"),
        },
        "outputs": {
            path.name: _descriptor(path) for path in output_files
        },
        "independent_validation_required": True,
        "promotion_eligible": False,
        "primary_v16_modified": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4QuotationParserOmissionAuditArtifacts(
        run_id, root, manifest_path, manifest
    )


def _population_a(accepted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in accepted.to_dict("records"):
        pdf = Path(row["source_pdf_path"]).resolve()
        normalized = Path(row["source_normalized_path"]).resolve()
        _require_hash(pdf, row["source_pdf_sha256"])
        _require_hash(normalized, row["source_normalized_sha256"])
        frame = pd.read_parquet(normalized).fillna("")
        present = int(frame["code"].astype(str).eq(row["local_code"]).sum())
        rows.append(
            {
                "event_id": row["event_id"],
                "event_date": row["event_date"],
                "local_code": row["local_code"],
                "source_request_id": row["source_request_id"],
                "source_pdf_path": str(pdf),
                "source_pdf_sha256": row["source_pdf_sha256"],
                "source_normalized_path": str(normalized),
                "source_normalized_sha256": row["source_normalized_sha256"],
                "baseline_normalized_exact_rows": present,
                "baseline_normalized_present": present > 0,
                "mapping_already_admitted": True,
                "repair_eligible": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["event_date", "event_id"])


def _population_b(decisions: pd.DataFrame) -> pd.DataFrame:
    missing = decisions.loc[decisions["decision_reason"].eq(MISSING_REASON)].copy()
    if len(missing) != 8 or missing["local_code"].nunique() != 8:
        raise ValueError("source-record-missing denominator changed")
    rows: list[dict[str, Any]] = []
    for row in missing.to_dict("records"):
        pdf = Path(row["source_pdf_path"]).resolve()
        normalized = Path(row["source_normalized_path"]).resolve()
        _require_hash(pdf, row["source_pdf_sha256"])
        _require_hash(normalized, row["source_normalized_sha256"])
        plain, layout = _pdf_text(pdf)
        code = row["local_code"]
        plain_hits = _exact_code_count(plain, code)
        layout_hits = _exact_code_count(layout, code)
        frame = pd.read_parquet(normalized).fillna("")
        normalized_hits = int(frame["code"].astype(str).eq(code).sum())
        rows.append(
            {
                "event_id": row["event_id"],
                "event_date": row["event_date"],
                "local_code": code,
                "source_request_id": row["source_request_id"],
                "source_pdf_path": str(pdf),
                "source_pdf_sha256": row["source_pdf_sha256"],
                "source_normalized_path": str(normalized),
                "source_normalized_sha256": row["source_normalized_sha256"],
                "pdf_pages": len(PdfReader(pdf).pages),
                "plain_text_exact_code_hits": plain_hits,
                "layout_text_exact_code_hits": layout_hits,
                "normalized_exact_code_rows": normalized_hits,
                "verified_parser_omission": False,
                "classification": "selected_source_record_absent_not_verified_parser_omission",
                "repair_eligible": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(["event_date", "event_id"])
    if result[
        [
            "plain_text_exact_code_hits",
            "layout_text_exact_code_hits",
            "normalized_exact_code_rows",
        ]
    ].to_numpy().sum() != 0:
        raise ValueError("a source-record-missing row has exact-code evidence")
    return result


def _population_c(mapping_audit: pd.DataFrame) -> pd.DataFrame:
    rows = mapping_audit.loc[mapping_audit["event_id"].eq(TARGET_6174_EVENT)].copy()
    if not (
        len(rows) == 1
        and rows.iloc[0]["local_code"] == "6174"
        and rows.iloc[0]["mapping_status"] == "missing_admitted_mapping"
    ):
        raise ValueError("code 6174 blocker identity changed")
    return pd.DataFrame(
        [
            {
                "event_id": TARGET_6174_EVENT,
                "event_date": rows.iloc[0]["event_date"],
                "local_code": "6174",
                "mapping_status": rows.iloc[0]["mapping_status"],
                "quotation_request_id": "",
                "quotation_pdf_path": "",
                "quotation_pdf_sha256": "",
                "status": "blocked_no_quotation_packet",
                "repair_eligible": False,
            }
        ]
    )


def _verify_v3(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    inventory_path = root / "artifact_hashes.json"
    if not (
        manifest.get("run_id") == V3_RUN_ID
        and manifest.get("status") == "complete_candidate"
        and _sha(inventory_path) == manifest.get("artifact_hashes_sha256")
    ):
        raise ValueError("V3 manifest contract failed")
    inventory = json.loads(inventory_path.read_text())
    for name, digest in inventory.items():
        _require_hash(root / name, digest)


def _verify_v16(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    if not (
        manifest.get("run_id") == V16_RUN_ID
        and manifest.get("component") == "halt_suspension_events"
        and manifest.get("output_sha256", {}).get("mapping_audit.csv")
        == _sha(root / "mapping_audit.csv")
    ):
        raise ValueError("V16 manifest contract failed")


def _pdf_text(path: Path) -> tuple[str, str]:
    reader = PdfReader(path)
    plain = "\n".join(page.extract_text() or "" for page in reader.pages)
    layout = "\n".join(
        page.extract_text(extraction_mode="layout") or "" for page in reader.pages
    )
    return plain, layout


def _exact_code_count(text: str, code: str) -> int:
    return len(re.findall(rf"(?<![0-9A-Z]){re.escape(code)}(?![0-9A-Z])", text.upper()))


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require_hash(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"source hash or path mismatch: {path}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
