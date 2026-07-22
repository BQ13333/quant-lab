from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber


HALT_CHANNEL = "halt_suspension_resumption"
IDENTITY_VERSION = "jpx-g4-report-row-v1"
EVENT_VERSION = "jpx-g4-report-event-v1"

_SOURCES = {
    "2012-02": "c86c868a01e18e2afc17b5bf15f408e8c838a3ff2397cf18a81f9cb9581f9720",
    "2013-09": "e7c63b7e0ff65dc6a4ea5ad8a6650c8b6b1306f79454092549c4c42589ba8958",
    "2020-10": "2ff5a87bcc558b574ad38b4ea7c1745f3692b038473812942fb130784bdd4725",
}

_EXPECTED = (
    ("ada193784ba67fd41f728c7098099d7f6812fccd2511de606fcfd1b07e030a2a", "507b4c0a64f374a5de1553e7", "2012-02", 2, "市場第一部 2012.02.02 別紙参照 9:00~12:29", "out_of_instrument_scope_appendix_reference", "", "", "", "別紙参照"),
    ("bb8f1feae007a54e895b8288974f76c29d83f2dca885d7d57bd4fcee26627f93", "df626b8c698ff7dee593f49b", "2012-02", 2, "市場第二部 2012.02.02 別紙参照 9:00~12:29", "out_of_instrument_scope_appendix_reference", "", "", "", "別紙参照"),
    ("4020ded60e62598386caf2bb0cff162ef3146cc1eb8a40788109dc584b7b80c6", "bd3442ebae0389b88a24165b", "2012-02", 2, "マザーズ 2012.02.02 別紙参照 9:00~12:29", "out_of_instrument_scope_appendix_reference", "", "", "", "別紙参照"),
    ("312e13eac18d061acd5b4c0c29bfdd936df8427366413b95047196c397142d06", "eecf102a06628b373a85518a", "2013-09", 4, "2013.09.30 4585 UMNファーマ UMN Pharma Inc. 17:10~17:25", "normalized_wrapped_time_range", "4585", "17:10", "17:25", "UMNファーマ UMN Pharma Inc."),
    ("e065ec4bf5338c23e7a88978d1506d1824a996357adef56bcee658c521da515f", "a43676147845ea79c549f75f", "2020-10", 5, "市場第一部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "All Listed Issues"),
    ("7f6b032500b86c6c9fe21a6cd2334e3944aed821b86cf87fee36b0e56a950c23", "3704202495019130830d2705", "2020-10", 5, "市場第二部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "All Listed Issues"),
    ("9632e9aea346d6a1f3be71f7808bd77a5dce11ea4d7d45b6a5983e7b407767de", "6c2603fbbcfa2be20c62bb4c", "2020-10", 5, "マザーズ 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "All Listed Issues"),
    ("1e6e6be36d3283105355050d055aa2f610d7e41207a00324f05e626e2a8156ea", "bbf5abd3849655ef8ae3cde6", "2020-10", 5, "TOKYO PRO Market 2020.10.01 全銘柄 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "全銘柄"),
    ("31f193cf7ab03e5941aded3b1292d489cdc11202234c72351fa0d0081f6c8ae7", "97891b11479f24eb72938b8a", "2020-10", 5, "JASDAQスタンダード 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "All Listed Issues"),
    ("218f01717eaf43e821f5fa50618973c65066cb0b52c91c216097af8c05898723", "c4d38b30cdae1b7962b1de16", "2020-10", 5, "JASDAQグロース 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", "out_of_instrument_scope_market_aggregate", "全銘柄", "", "", "All Listed Issues"),
)

_UNCHANGED = {
    ("0d9186d176b3143423592d2d944a37d4d0aeed2b3a9ccb1b55a6b59591f0e933", "2009-03", "1872"),
    ("f99c1c447719d8b48c72bb4ded73c3a4bb00a5309190672b41911bad295f6fac", "2009-03", "2652"),
    ("9ec1a725d8151951c3fc31118006121b4b0ca4cb43e1d6666cafe1e48db8ff05", "2009-05", "3064"),
    ("dfd2a1082666f7594a9bc767f6768bd0318a65c845dfd6cc2579256c8e147764", "2009-06", "8685"),
    ("77573ca15b2947706190a7f966c5cadcb38644b8c03e8934ea922675a1fc4099", "2009-06", "8918"),
    ("0e72753d609648f1f4d55d20444c4e85b5b473143963b7914be8a03e13ec8b20", "2009-06", "4350"),
    ("3a7811e2c32df9c56bd7dedf2156b29b8c4b88fd79c32e527e7bb0f77513e3b7", "2009-06", "6730"),
    ("3adf70ee2a80f5f4015fe6ef0486590e059e6a62cb86aff532cc5629a4bfb634", "2009-07", "8685"),
    ("e1c67c1d6907b99445c296c4ea6b1b4e67696c7b5133e7df9d1f608b5559c933", "2009-07", "8918"),
    ("a83b12d3a770b90cf647b455b554caa31872942ef9605295f0f4710d74f0f086", "2011-04", "8710"),
    ("9422ecbd27c1c716ab29eb47cbcc5f98198188c318c30b8773fa328cb014dc91", "2011-05", "8710"),
}


@dataclass(frozen=True)
class HaltNormalizationRepairValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def validate_halt_report_normalization_repair(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_report_normalization_repair_validation",
) -> HaltNormalizationRepairValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    producer = Path(producer_dir).resolve()
    issues: list[dict[str, str]] = []
    producer_manifest = _audit_packet(producer, issues)
    repairs = _read_csv(producer / "normalization_repairs.csv", issues)
    unchanged = _read_csv(producer / "unchanged_date_granular_contract.csv", issues)

    reconciliation = _reconcile_repairs(repairs, issues)
    source_validation, semantic_validation = _independent_source_validation(producer, issues)
    unchanged_validation = _validate_unchanged(unchanged, issues)

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once normalization validation artifact exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        reconciliation.to_csv(staging / "repair_reconciliation.csv", index=False)
        source_validation.to_csv(staging / "independent_source_hash_validation.csv", index=False)
        semantic_validation.to_csv(staging / "independent_semantic_validation.csv", index=False)
        unchanged_validation.to_csv(staging / "unchanged_contract_validation.csv", index=False)
        pd.DataFrame(issues, columns=["issue_code", "record_id", "detail"]).to_csv(
            staging / "validation_issues.csv", index=False
        )
        inventory = _inventory(staging)
        inventory_path = staging / "artifact_hashes.json"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "1.0-independent-pdfplumber-validation",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass_bounded_normalization_repair" if not issues else "failed_validation",
            "producer_run_id": producer_manifest.get("run_id", ""),
            "producer_manifest_sha256": _sha(producer / "manifest.json") if (producer / "manifest.json").is_file() else "",
            "validation_issue_count": len(issues),
            "validation_issue_codes": sorted({issue["issue_code"] for issue in issues}),
            "repair_rows_expected": 10,
            "repair_rows_reconciled": int(reconciliation.get("exact_match", pd.Series(dtype=bool)).sum()),
            "wrapped_time_range_rows": int(reconciliation.get("normalization_disposition", pd.Series(dtype=str)).eq("normalized_wrapped_time_range").sum()),
            "appendix_reference_rows": int(reconciliation.get("normalization_disposition", pd.Series(dtype=str)).eq("out_of_instrument_scope_appendix_reference").sum()),
            "market_aggregate_rows": int(reconciliation.get("normalization_disposition", pd.Series(dtype=str)).eq("out_of_instrument_scope_market_aggregate").sum()),
            "source_hashes_verified": bool(len(source_validation) == 3 and source_validation.get("hash_match", pd.Series(dtype=bool)).all()),
            "source_semantics_verified": bool(len(semantic_validation) == 10 and semantic_validation.get("semantic_match", pd.Series(dtype=bool)).all()),
            "source_record_hashes_recomputed": bool(len(reconciliation) == 10 and reconciliation.get("source_record_hash_match", pd.Series(dtype=bool)).all()),
            "event_ids_recomputed": bool(len(reconciliation) == 10 and reconciliation.get("event_id_match", pd.Series(dtype=bool)).all()),
            "invented_local_codes": 0,
            "evidence_only_rows": 9,
            "instrument_events_created": 1,
            "unchanged_date_granular_rows": int(unchanged_validation.get("exact_match", pd.Series(dtype=bool)).sum()),
            "independent_parser": "pdfplumber sequential page text",
            "refinitiv_used": False,
            "primary_halt_v8_modified": False,
            "scorecard_eligible": False,
            "write_once": True,
            "artifact_hashes_sha256": _sha(inventory_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return HaltNormalizationRepairValidationArtifacts(run_id, root, root / "manifest.json", manifest)


def verify_halt_report_normalization_repair_validation(root: str | Path) -> list[str]:
    path = Path(root).resolve()
    issues: list[dict[str, str]] = []
    manifest = _audit_packet(path, issues)
    if manifest.get("status") != "pass_bounded_normalization_repair":
        _issue(issues, "validation_status_not_pass", "manifest", str(manifest.get("status", "")))
    if int(manifest.get("validation_issue_count", -1)) != 0:
        _issue(issues, "validation_issues_declared", "manifest", str(manifest.get("validation_issue_count", "")))
    return sorted({issue["issue_code"] for issue in issues})


def _reconcile_repairs(repairs: pd.DataFrame, issues: list[dict[str, str]]) -> pd.DataFrame:
    expected = {row[0]: row for row in _EXPECTED}
    if len(repairs) != 10 or set(repairs.get("source_record_sha256", [])) != set(expected):
        _issue(issues, "repair_record_set_mismatch", "repairs", str(len(repairs)))
    rows: list[dict[str, Any]] = []
    for source in repairs.to_dict("records"):
        digest = str(source.get("source_record_sha256", ""))
        spec = expected.get(digest)
        exact = spec is not None
        if spec is not None:
            _, event_id, month, page, _, disposition, raw_code, repaired_start, repaired_end, issue_name = spec
            checks = {
                "report_event_id": event_id,
                "report_month": month,
                "source_sha256": _SOURCES[month],
                "source_page": str(page),
                "normalization_disposition": disposition,
                "raw_code": raw_code,
                "issue_name": issue_name,
                "repaired_halt_start_printed": repaired_start,
                "repaired_halt_end_printed": repaired_end,
                "channel": HALT_CHANNEL,
                "instrument_class": "equity",
                "local_code_invented": "False",
            }
            exact = all(str(source.get(field, "")) == value for field, value in checks.items())
        identity = {
            "version": IDENTITY_VERSION,
            "report_month": str(source.get("report_month", "")),
            "channel": str(source.get("channel", "")),
            "source_sha256": str(source.get("source_sha256", "")),
            "source_page": _integer(source.get("source_page", "")),
            "source_row_ordinal": _integer(source.get("source_row_ordinal", "")),
            "event_date": str(source.get("event_date", "")),
            "raw_code": str(source.get("raw_code", "")),
            "normalized_local_code": str(source.get("normalized_local_code", "")),
            "instrument_class": str(source.get("instrument_class", "")),
            "issue_name": str(source.get("issue_name", "")),
            "halt_start_printed": str(source.get("halt_start_printed", "")),
            "halt_end_printed": str(source.get("halt_end_printed", "")),
        }
        observed_record = _identity_sha(identity)
        expected_event = hashlib.sha256(f"{EVENT_VERSION}|{observed_record}".encode("utf-8")).hexdigest()[:24]
        record_match = observed_record == digest
        event_match = expected_event == str(source.get("report_event_id", ""))
        rows.append(
            {
                "source_record_sha256": digest,
                "normalization_disposition": str(source.get("normalization_disposition", "")),
                "exact_match": exact,
                "source_record_hash_match": record_match,
                "event_id_match": event_match,
            }
        )
        if not (exact and record_match and event_match):
            _issue(issues, "repair_row_reconciliation_failed", digest, f"exact={exact};record={record_match};event={event_match}")
    return pd.DataFrame(rows)


def _independent_source_validation(
    producer: Path, issues: list[dict[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for month, expected_sha in _SOURCES.items():
        path = producer / "raw" / f"{month}.pdf"
        observed = _sha(path) if path.is_file() else ""
        match = observed == expected_sha
        source_rows.append(
            {
                "report_month": month,
                "expected_sha256": expected_sha,
                "observed_sha256": observed,
                "hash_match": match,
            }
        )
        if not match:
            _issue(issues, "independent_source_hash_mismatch", month, observed)
            continue
        try:
            with pdfplumber.open(io.BytesIO(path.read_bytes())) as document:
                pages = [page.extract_text(x_tolerance=2, y_tolerance=2) or "" for page in document.pages]
        except Exception as exc:  # pragma: no cover - defensive boundary
            _issue(issues, "independent_pdf_parse_failed", month, type(exc).__name__)
            continue
        for digest, _, spec_month, page, marker, disposition, *_ in _EXPECTED:
            if spec_month != month:
                continue
            semantic_match = page <= len(pages) and _canonical(marker) in _canonical(pages[page - 1])
            if month == "2012-02":
                semantic_match = semantic_match and any(
                    "売買停止銘柄一覧 (別紙)" in _canonical(value) for value in pages
                )
            semantic_rows.append(
                {
                    "source_record_sha256": digest,
                    "report_month": month,
                    "source_page": page,
                    "normalization_disposition": disposition,
                    "semantic_match": semantic_match,
                }
            )
            if not semantic_match:
                _issue(issues, "independent_source_semantics_mismatch", digest, marker)
    return pd.DataFrame(source_rows), pd.DataFrame(semantic_rows)


def _validate_unchanged(frame: pd.DataFrame, issues: list[dict[str, str]]) -> pd.DataFrame:
    observed = {
        (str(row.get("source_record_sha256", "")), str(row.get("report_month", "")), str(row.get("local_code", "")))
        for row in frame.to_dict("records")
        if str(row.get("normalization_status", "")) == "missing_halt_start"
        and str(row.get("contract_action", "")) == "unchanged"
    }
    exact = observed == _UNCHANGED
    if not exact:
        _issue(issues, "unchanged_date_granular_contract_mismatch", "contract", f"observed={len(observed)}")
    return pd.DataFrame(
        [
            {
                "source_record_sha256": digest,
                "report_month": month,
                "local_code": code,
                "exact_match": (digest, month, code) in observed and exact,
            }
            for digest, month, code in sorted(_UNCHANGED)
        ]
    )


def _audit_packet(root: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        inventory_path = root / "artifact_hashes.json"
        if _sha(inventory_path) != manifest.get("artifact_hashes_sha256"):
            _issue(issues, "packet_inventory_digest_mismatch", "artifact_hashes.json", "")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for relative, expected in inventory.items():
            item = Path(str(relative))
            if item.is_absolute() or ".." in item.parts:
                _issue(issues, "packet_inventory_path_unsafe", str(relative), "")
                continue
            path = root / item
            if not path.is_file() or _sha(path) != str(expected):
                _issue(issues, "packet_file_hash_mismatch", str(relative), "")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        expected = set(inventory) | {"manifest.json", "artifact_hashes.json"}
        if actual != expected:
            _issue(issues, "packet_file_denominator_mismatch", "packet", "")
        return manifest
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "packet_unreadable", "packet", type(exc).__name__)
        return {}


def _read_csv(path: Path, issues: list[dict[str, str]]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except (OSError, ValueError) as exc:
        _issue(issues, "csv_unreadable", path.name, type(exc).__name__)
        return pd.DataFrame()


def _identity_sha(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer(value: Any) -> int:
    try:
        return int(str(value))
    except ValueError:
        return -1


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("～", "~").replace("〜", "~")
    return re.sub(r"\s+", " ", normalized).strip()


def _issue(issues: list[dict[str, str]], code: str, record: str, detail: str) -> None:
    issues.append({"issue_code": code, "record_id": record, "detail": detail})


def _inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
