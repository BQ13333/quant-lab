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
from pypdf import PdfReader


UTC = timezone.utc
COMPONENT = "japan_g4_quotation_parser_omission_denominator_audit"


def validate_japan_g4_quotation_parser_omission_audit(
    *,
    producer_dir: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts/japan_g4_quotation_parser_omission_audit_validation",
) -> tuple[Path, dict[str, Any]]:
    producer = Path(producer_dir).resolve()
    manifest = json.loads((producer / "manifest.json").read_text())
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    issues: list[str] = []

    if not (
        manifest.get("component") == COMPONENT
        and manifest.get("status") == "blocked_unsealed_eleven_row_denominator"
        and manifest.get("promotion_eligible") is False
        and manifest.get("policy", {}).get("vendor_calls") is False
        and manifest.get("policy", {}).get("parser_repair_admitted") is False
    ):
        issues.append("producer_contract_invalid")
    inputs = manifest.get("inputs", {})
    required_inputs = {
        "v3_manifest",
        "v3_mapping_decisions",
        "v3_accepted_mappings",
        "v16_manifest",
        "v16_mapping_audit",
    }
    if set(inputs) != required_inputs:
        issues.append("input_descriptor_denominator_invalid")
    for name, descriptor in inputs.items():
        path = Path(str(descriptor.get("path", "")))
        if not _external_descriptor_matches(path, descriptor):
            issues.append(f"input_descriptor_invalid:{name}")
    for name, descriptor in manifest.get("outputs", {}).items():
        path = producer / name
        if not _descriptor_matches(path, descriptor, producer):
            issues.append(f"output_descriptor_invalid:{name}")

    group_a = _csv(producer / "population_a_admitted_raw_pdf_mappings.csv", issues)
    group_b = _csv(producer / "population_b_source_record_missing.csv", issues)
    group_c = _csv(producer / "population_c_code_6174.csv", issues)
    repairs = _csv(producer / "accepted_parser_repairs.csv", issues)
    findings = _csv(producer / "audit_findings.csv", issues)

    source_accepted = _descriptor_csv(inputs.get("v3_accepted_mappings", {}), issues)
    source_decisions = _descriptor_csv(inputs.get("v3_mapping_decisions", {}), issues)
    source_mapping_audit = _descriptor_csv(inputs.get("v16_mapping_audit", {}), issues)
    _validate_source_manifests(inputs, issues)

    if not (
        len(group_a) == 23
        and group_a["event_id"].is_unique
        and group_a["mapping_already_admitted"].str.lower().eq("true").all()
        and group_a["repair_eligible"].str.lower().eq("false").all()
        and group_a["baseline_normalized_present"].str.lower().eq("true").sum() == 1
    ):
        issues.append("population_a_denominator_invalid")
    if not (
        len(group_b) == 8
        and group_b["event_id"].is_unique
        and group_b["local_code"].nunique() == 8
        and group_b["verified_parser_omission"].str.lower().eq("false").all()
        and group_b["repair_eligible"].str.lower().eq("false").all()
    ):
        issues.append("population_b_denominator_invalid")
    if not (
        len(group_c) == 1
        and group_c.iloc[0].get("local_code") == "6174"
        and group_c.iloc[0].get("quotation_pdf_path") == ""
        and group_c.iloc[0].get("status") == "blocked_no_quotation_packet"
    ):
        issues.append("population_c_denominator_invalid")
    if not repairs.empty:
        issues.append("parser_repairs_not_empty")
    if not (
        len(findings) == 4
        and "unsupported_eleven_row_denominator" in set(findings["finding_id"])
        and findings.loc[
            findings["finding_id"].eq("verified_missing_source_parser_omissions"),
            "observed_count",
        ].eq("0").all()
    ):
        issues.append("findings_contract_invalid")
    expected_a = set(source_accepted.get("event_id", pd.Series(dtype=str)))
    expected_b = set(
        source_decisions.loc[
            source_decisions.get("decision_reason", pd.Series(dtype=str)).eq(
                "latest_eligible_source_record_missing"
            ),
            "event_id",
        ]
    )
    expected_c = set(
        source_mapping_audit.loc[
            source_mapping_audit.get("event_id", pd.Series(dtype=str)).eq(
                "jpx-report-331f536854ca55fb9e898f15"
            ),
            "event_id",
        ]
    )
    if set(group_a.get("event_id", pd.Series(dtype=str))) != expected_a:
        issues.append("population_a_source_denominator_mismatch")
    if set(group_b.get("event_id", pd.Series(dtype=str))) != expected_b:
        issues.append("population_b_source_denominator_mismatch")
    if set(group_c.get("event_id", pd.Series(dtype=str))) != expected_c:
        issues.append("population_c_source_denominator_mismatch")

    source_a = source_accepted.set_index("event_id").to_dict("index")
    source_b = source_decisions.set_index("event_id").to_dict("index")
    for row in group_a.to_dict("records"):
        source = source_a.get(row["event_id"], {})
        if any(
            row[field] != source.get(field, "")
            for field in (
                "local_code",
                "source_request_id",
                "source_pdf_path",
                "source_pdf_sha256",
                "source_normalized_path",
                "source_normalized_sha256",
            )
        ):
            issues.append(f"population_a_source_binding_mismatch:{row['event_id']}")
    for row in group_b.to_dict("records"):
        source = source_b.get(row["event_id"], {})
        if any(
            row[field] != source.get(field, "")
            for field in (
                "local_code",
                "source_request_id",
                "source_pdf_path",
                "source_pdf_sha256",
                "source_normalized_path",
                "source_normalized_sha256",
            )
        ):
            issues.append(f"population_b_source_binding_mismatch:{row['event_id']}")

    for row in group_a.to_dict("records"):
        normalized = Path(row["source_normalized_path"])
        if not _safe_hashed(normalized, row["source_normalized_sha256"]):
            issues.append(f"population_a_source_invalid:{row['event_id']}")
            continue
        frame = pd.read_parquet(normalized).fillna("")
        count = int(frame["code"].astype(str).eq(row["local_code"]).sum())
        if count != int(row["baseline_normalized_exact_rows"]):
            issues.append(f"population_a_recompute_mismatch:{row['event_id']}")

    for row in group_b.to_dict("records"):
        pdf = Path(row["source_pdf_path"])
        normalized = Path(row["source_normalized_path"])
        if not (
            _safe_hashed(pdf, row["source_pdf_sha256"])
            and _safe_hashed(normalized, row["source_normalized_sha256"])
        ):
            issues.append(f"population_b_source_invalid:{row['event_id']}")
            continue
        reader = PdfReader(pdf)
        plain = "\n".join(page.extract_text() or "" for page in reader.pages)
        layout = "\n".join(
            page.extract_text(extraction_mode="layout") or "" for page in reader.pages
        )
        code = row["local_code"]
        expression = re.compile(
            rf"(?<![0-9A-Z]){re.escape(code)}(?![0-9A-Z])", re.IGNORECASE
        )
        frame = pd.read_parquet(normalized).fillna("")
        observed = (
            len(expression.findall(plain)),
            len(expression.findall(layout)),
            int(frame["code"].astype(str).eq(code).sum()),
        )
        expected = (
            int(row["plain_text_exact_code_hits"]),
            int(row["layout_text_exact_code_hits"]),
            int(row["normalized_exact_code_rows"]),
        )
        if observed != expected or observed != (0, 0, 0):
            issues.append(f"population_b_reparse_mismatch:{row['event_id']}")

    issue_frame = pd.DataFrame({"issue": issues})
    issue_path = root / "validation_issues.csv"
    _atomic_csv(issue_path, issue_frame)
    status = "pass_blocker_confirmed" if not issues else "fail"
    summary_path = root / "summary.md"
    _atomic_text(
        summary_path,
        f"# Independent Quotation Parser Omission Audit Validation: {run_id}\n\n"
        f"- Status: {status}\n- Validation issues: {len(issues)}\n"
        "- Eleven-row repair denominator accepted: no\n- Promotion eligible: no\n",
    )
    result = {
        "schema_version": "1.0-independent-fail-closed-validation",
        "run_id": run_id,
        "component": f"{COMPONENT}_independent_validation",
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer_manifest_sha256": _sha(producer / "manifest.json"),
        "counts": {
            "validation_issues": len(issues),
            "population_a_rows": len(group_a),
            "population_b_rows": len(group_b),
            "population_c_rows": len(group_c),
            "accepted_parser_repairs": len(repairs),
        },
        "blocker_confirmed": not issues,
        "promotion_eligible": False,
        "vendor_calls": False,
        "outputs": {
            issue_path.name: _descriptor(issue_path),
            summary_path.name: _descriptor(summary_path),
        },
    }
    manifest_path = root / "manifest.json"
    _atomic_text(manifest_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return manifest_path, result


def _csv(path: Path, issues: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        issues.append(f"csv_unreadable:{path.name}:{type(exc).__name__}")
        return pd.DataFrame()


def _descriptor_csv(descriptor: dict[str, Any], issues: list[str]) -> pd.DataFrame:
    path = Path(str(descriptor.get("path", "")))
    if not _external_descriptor_matches(path, descriptor):
        return pd.DataFrame()
    return _csv(path, issues)


def _validate_source_manifests(
    inputs: dict[str, dict[str, Any]], issues: list[str]
) -> None:
    try:
        v3_path = Path(inputs["v3_manifest"]["path"])
        v16_path = Path(inputs["v16_manifest"]["path"])
        v3 = json.loads(v3_path.read_text())
        v16 = json.loads(v16_path.read_text())
        if not (
            v3.get("run_id") == "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3"
            and v3.get("status") == "complete_candidate"
            and v3.get("outputs", {}).get("mapping_decisions.csv")
            == inputs["v3_mapping_decisions"]["sha256"]
            and v3.get("outputs", {}).get("accepted_mappings.csv")
            == inputs["v3_accepted_mappings"]["sha256"]
        ):
            issues.append("v3_source_manifest_contract_invalid")
        if not (
            v16.get("run_id") == "JPN-G4-HALTS-200607-202606-V16"
            and v16.get("component") == "halt_suspension_events"
            and v16.get("output_sha256", {}).get("mapping_audit.csv")
            == inputs["v16_mapping_audit"]["sha256"]
        ):
            issues.append("v16_source_manifest_contract_invalid")
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        issues.append("source_manifest_unreadable")


def _descriptor_matches(path: Path, descriptor: dict[str, Any], root: Path) -> bool:
    try:
        return bool(
            path.resolve().is_relative_to(root)
            and not path.is_symlink()
            and path.is_file()
            and _sha(path) == descriptor.get("sha256")
            and path.stat().st_size == int(descriptor.get("bytes", -1))
        )
    except (OSError, TypeError, ValueError):
        return False


def _external_descriptor_matches(path: Path, descriptor: dict[str, Any]) -> bool:
    try:
        return bool(
            path.is_absolute()
            and not path.is_symlink()
            and path.is_file()
            and str(path.resolve()) == str(descriptor.get("path", ""))
            and _sha(path) == descriptor.get("sha256")
            and path.stat().st_size == int(descriptor.get("bytes", -1))
        )
    except (OSError, TypeError, ValueError):
        return False


def _safe_hashed(path: Path, expected: str) -> bool:
    return path.is_absolute() and path.is_file() and not path.is_symlink() and _sha(path) == expected


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
