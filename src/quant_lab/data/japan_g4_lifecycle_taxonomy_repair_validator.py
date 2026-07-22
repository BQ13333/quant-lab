from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

from quant_lab.data.japan_g4_halt_report_events import EVENT_ID_VERSION, SOURCE_RECORD_VERSION
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair import (
    REPAIR_COLUMNS,
    SOURCE_COLUMNS,
    TARGET_CLASS_BY_CODE,
    TARGET_MARKER_BY_CLASS,
    NOTICE_EVENT_ID,
    NOTICE_FOREIGN_SHARE_MARKER,
    NOTICE_URL,
    _canonical,
    _dense_canonical,
    _inventory,
    _sha,
    _verify_inventory,
)
from quant_lab.data.japan_g4_tse_wayback_lifecycle_validator import (
    independently_parse_pdf_pages,
)


VERSION = "1.0-independent-pdfplumber-taxonomy-validation"


@dataclass(frozen=True)
class LifecycleTaxonomyRepairValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def validate_japan_g4_lifecycle_taxonomy_repair(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_lifecycle_taxonomy_repair_validation",
) -> LifecycleTaxonomyRepairValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    producer = Path(producer_dir).resolve()
    issues: list[dict[str, str]] = []
    for code in _verify_inventory(producer):
        _issue(issues, f"producer_{code}", "producer", "")

    producer_manifest = _json(producer / "manifest.json", issues)
    repairs = _csv(producer / "taxonomy_repairs.csv", issues)
    sources = _csv(producer / "source_records.csv", issues)
    lineage = _csv(producer / "source_lineage.csv", issues)
    reconciliation = _reconcile_rows(repairs, sources, issues)
    pdf_validation, independent_events = _validate_official_pdfs(
        producer, sources, lineage, issues
    )
    notice_validation = _validate_official_notice(producer, sources, issues)

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once taxonomy validation artifact exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        reconciliation.to_csv(staging / "repair_reconciliation.csv", index=False)
        pdf_validation.to_csv(staging / "independent_pdf_validation.csv", index=False)
        independent_events.to_csv(staging / "independent_source_records.csv", index=False)
        notice_validation.to_csv(staging / "independent_notice_validation.csv", index=False)
        pd.DataFrame(issues, columns=["issue_code", "record_id", "detail"]).to_csv(
            staging / "validation_issues.csv", index=False
        )
        inventory = _inventory(staging)
        inventory_path = staging / "artifact_hashes.json"
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exact_matches = reconciliation.get("exact_match", pd.Series(dtype=bool))
        manifest = {
            "schema_version": VERSION,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass_bounded_source_backed_taxonomy_repair" if not issues else "failed_validation",
            "producer_run_id": producer_manifest.get("run_id", ""),
            "producer_manifest_sha256": _sha(producer / "manifest.json") if (producer / "manifest.json").is_file() else "",
            "validation_issue_count": len(issues),
            "validation_issue_codes": sorted({issue["issue_code"] for issue in issues}),
            "repair_rows_expected": 9,
            "repair_rows_reconciled": int(exact_matches.sum()),
            "warrant_rows_reconciled": int(reconciliation.get("independent_class", pd.Series(dtype=str)).eq("warrant").sum()),
            "foreign_share_rows_reconciled": int(reconciliation.get("independent_class", pd.Series(dtype=str)).eq("foreign_share").sum()),
            "official_pdf_count_verified": int(pdf_validation.get("exact_match", pd.Series(dtype=bool)).sum()),
            "official_notice_count_verified": int(notice_validation.get("exact_match", pd.Series(dtype=bool)).sum()),
            "source_record_hashes_recomputed": bool(len(reconciliation) == 9 and reconciliation.get("record_hash_match", pd.Series(dtype=bool)).all()),
            "event_ids_recomputed": bool(len(reconciliation) == 9 and reconciliation.get("event_id_match", pd.Series(dtype=bool)).all()),
            "unique_event_ids": int(sources.get("event_id", pd.Series(dtype=str)).nunique()),
            "classification_inferred_rows": 0,
            "mappings_created": 0,
            "timestamps_created": 0,
            "zero_events_created": 0,
            "research_scope_rows_after": 0,
            "independent_parser": "pdfplumber plus lifecycle exact-event reparse",
            "refinitiv_used": False,
            "scorecard_eligible": False,
            "write_once": True,
            "artifact_hashes_sha256": _sha(inventory_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return LifecycleTaxonomyRepairValidationArtifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=root / "manifest.json",
        manifest=manifest,
    )


def verify_japan_g4_lifecycle_taxonomy_repair_validation(root: str | Path) -> list[str]:
    path = Path(root).resolve()
    issues = _verify_inventory(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "pass_bounded_source_backed_taxonomy_repair":
            issues.append("validation_status_not_pass")
        if int(manifest.get("validation_issue_count", -1)) != 0:
            issues.append("validation_issues_declared")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"validation_unreadable:{type(exc).__name__}")
    return sorted(set(issues))


def _reconcile_rows(
    repairs: pd.DataFrame,
    sources: pd.DataFrame,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    if set(repairs.columns) != set(REPAIR_COLUMNS):
        _issue(issues, "repair_schema_mismatch", "repairs", "")
    if set(sources.columns) != set(SOURCE_COLUMNS):
        _issue(issues, "source_schema_mismatch", "sources", "")
    if len(repairs) != 9 or set(repairs.get("raw_code", [])) != set(TARGET_CLASS_BY_CODE):
        _issue(issues, "repair_row_set_mismatch", "repairs", str(len(repairs)))
    if len(sources) != 9 or set(sources.get("raw_code", [])) != set(TARGET_CLASS_BY_CODE):
        _issue(issues, "source_row_set_mismatch", "sources", str(len(sources)))
    if sources.get("event_id", pd.Series(dtype=str)).nunique() != 9:
        _issue(issues, "event_id_not_unique", "sources", "")
    if sources.get("source_record_sha256", pd.Series(dtype=str)).nunique() != 9:
        _issue(issues, "source_record_hash_not_unique", "sources", "")

    by_event = {str(row.get("event_id", "")): row for row in sources.to_dict("records")}
    rows: list[dict[str, Any]] = []
    for repair in repairs.to_dict("records"):
        event_id = str(repair.get("event_id", ""))
        source = by_event.get(event_id, {})
        target_class = TARGET_CLASS_BY_CODE.get(str(repair.get("raw_code", "")), "")
        notice_backed = str(repair.get("raw_code", "")) == "3827"
        marker = NOTICE_FOREIGN_SHARE_MARKER if notice_backed else TARGET_MARKER_BY_CLASS.get(target_class, "")
        independent_class = ""
        text = _canonical(str(source.get("source_record_text", "")))
        if notice_backed and (
            event_id == NOTICE_EVENT_ID
            and str(source.get("taxonomy_evidence_route", "")) == "separate_official_jpx_delisting_notice"
            and str(source.get("taxonomy_evidence_url", "")) == NOTICE_URL
        ):
            independent_class = target_class
        elif marker and marker in text:
            independent_class = target_class
        else:
            _issue(issues, "required_source_marker_missing", event_id, marker)

        identity = {
            "version": SOURCE_RECORD_VERSION,
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
        record_hash = _identity_hash(identity)
        event_hash = hashlib.sha256(
            f"{EVENT_ID_VERSION}|{record_hash}".encode("utf-8")
        ).hexdigest()[:24]
        record_match = record_hash == str(source.get("source_record_sha256", ""))
        event_match = event_hash == event_id
        exact = bool(source) and all(
            str(repair.get(repair_field, "")) == str(source.get(source_field, ""))
            for repair_field, source_field in (
                ("source_record_sha256", "source_record_sha256"),
                ("report_month", "report_month"),
                ("channel", "channel"),
                ("event_date", "event_date"),
                ("raw_code", "raw_code"),
                ("normalized_local_code", "normalized_local_code"),
                ("source_instrument_class", "instrument_class"),
                ("source_sha256", "source_sha256"),
                ("source_page", "source_page"),
                ("source_row_ordinal", "source_row_ordinal"),
                ("source_record_text", "source_record_text"),
                ("taxonomy_evidence_route", "taxonomy_evidence_route"),
                ("taxonomy_evidence_url", "taxonomy_evidence_url"),
                ("taxonomy_evidence_path", "taxonomy_evidence_path"),
                ("taxonomy_evidence_sha256", "taxonomy_evidence_sha256"),
            )
        )
        exact = exact and record_match and event_match and independent_class == target_class
        exact = exact and str(repair.get("reclassified_instrument_class", "")) == target_class
        exact = exact and str(repair.get("required_source_marker", "")) == marker
        exact = exact and str(repair.get("taxonomy_disposition", "")) == f"quarantined_{target_class}"
        for field in (
            "in_research_scope_after",
            "mapping_created",
            "timestamp_created",
            "zero_event_created",
            "classification_inferred",
        ):
            exact = exact and str(repair.get(field, "")).casefold() == "false"
        if not exact:
            _issue(issues, "repair_exact_reconciliation_failed", event_id, "")
        rows.append(
            {
                "event_id": event_id,
                "raw_code": str(repair.get("raw_code", "")),
                "independent_class": independent_class,
                "record_hash_match": record_match,
                "event_id_match": event_match,
                "exact_match": exact,
            }
        )
    return pd.DataFrame(rows)


def _validate_official_pdfs(
    producer: Path,
    sources: pd.DataFrame,
    lineage: pd.DataFrame,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_lineage = {
        (str(row.get("report_month", "")), str(row.get("source_sha256", ""))): row
        for row in lineage.to_dict("records")
    }
    source_groups = sources.groupby(["report_month", "source_sha256"], dropna=False)
    pdf_rows: list[dict[str, Any]] = []
    independent_rows: list[dict[str, Any]] = []
    for (month, source_sha), group in source_groups:
        key = (str(month), str(source_sha))
        line = expected_lineage.get(key, {})
        relative = Path(str(line.get("copied_source_path", "")))
        exact = bool(line)
        if relative.is_absolute() or ".." in relative.parts:
            _issue(issues, "unsafe_copied_pdf_path", str(month), str(relative))
            exact = False
        pdf_path = (producer / relative).resolve()
        if producer not in pdf_path.parents or not pdf_path.is_file():
            _issue(issues, "copied_pdf_missing", str(month), str(relative))
            exact = False
            pages: list[str] = []
        else:
            observed_sha = _sha(pdf_path)
            if observed_sha != str(source_sha):
                _issue(issues, "copied_pdf_hash_mismatch", str(month), observed_sha)
                exact = False
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
            except Exception as exc:  # pdfplumber exposes several backend exception types.
                _issue(issues, "copied_pdf_unreadable", str(month), type(exc).__name__)
                pages = []
                exact = False

        try:
            exemplar = group.iloc[0]
            parsed = (
                independently_parse_pdf_pages(
                    pages,
                    str(month),
                    source_url=str(exemplar.get("source_url", "")),
                    source_path=str(exemplar.get("copied_source_path", "")),
                    source_sha256=str(source_sha),
                )
                if pages
                else {"events": []}
            )
            parsed_by_id = {str(row["event_id"]): row for row in parsed["events"]}
        except (ValueError, KeyError) as exc:
            _issue(issues, "independent_pdf_reparse_failed", str(month), type(exc).__name__)
            parsed_by_id = {}
            exact = False

        for source in group.to_dict("records"):
            event_id = str(source.get("event_id", ""))
            parsed_row = parsed_by_id.get(event_id)
            notice_backed = str(source["raw_code"]) == "3827"
            marker = "" if notice_backed else TARGET_MARKER_BY_CLASS[TARGET_CLASS_BY_CODE[str(source["raw_code"])]]
            page_number = _integer(source.get("source_page", ""))
            page_text = _canonical(pages[page_number - 1]) if 0 < page_number <= len(pages) else ""
            row_exact = (
                parsed_row is not None
                and str(parsed_row.get("source_record_sha256", "")) == str(source.get("source_record_sha256", ""))
                and str(parsed_row.get("source_record_text", "")) == str(source.get("source_record_text", ""))
                and (not marker or marker in page_text)
                and _dense_canonical(str(source.get("source_record_text", ""))) in _dense_canonical(page_text)
            )
            if not row_exact:
                _issue(issues, "independent_source_record_disagreement", event_id, "")
                exact = False
            independent_rows.append(
                {
                    "event_id": event_id,
                    "report_month": str(month),
                    "raw_code": str(source.get("raw_code", "")),
                    "source_record_sha256": str(parsed_row.get("source_record_sha256", "")) if parsed_row else "",
                    "source_record_text": str(parsed_row.get("source_record_text", "")) if parsed_row else "",
                    "required_marker": marker,
                    "exact_match": row_exact,
                }
            )
        pdf_rows.append(
            {
                "report_month": str(month),
                "source_sha256": str(source_sha),
                "copied_source_path": str(relative),
                "target_rows": len(group),
                "exact_match": exact,
            }
        )
    if len(expected_lineage) != len(pdf_rows):
        _issue(issues, "source_lineage_denominator_mismatch", "lineage", str(len(expected_lineage)))
    return pd.DataFrame(pdf_rows), pd.DataFrame(independent_rows)


def _validate_official_notice(
    producer: Path,
    sources: pd.DataFrame,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    selected = sources.loc[sources.get("raw_code", pd.Series(dtype=str)).eq("3827")]
    exact = len(selected) == 1
    if not exact:
        _issue(issues, "notice_event_denominator_mismatch", "3827", str(len(selected)))
        return pd.DataFrame([{"event_id": NOTICE_EVENT_ID, "exact_match": False}])
    source = selected.iloc[0].to_dict()
    relative = Path(str(source.get("taxonomy_evidence_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        _issue(issues, "unsafe_notice_path", NOTICE_EVENT_ID, str(relative))
        exact = False
    notice_path = (producer / relative).resolve()
    if producer not in notice_path.parents or not notice_path.is_file():
        _issue(issues, "official_notice_missing", NOTICE_EVENT_ID, str(relative))
        exact = False
        body = b""
    else:
        body = notice_path.read_bytes()
        observed_sha = hashlib.sha256(body).hexdigest()
        if observed_sha != str(source.get("taxonomy_evidence_sha256", "")):
            _issue(issues, "official_notice_hash_mismatch", NOTICE_EVENT_ID, observed_sha)
            exact = False
    text = _canonical(BeautifulSoup(body, "html.parser").get_text(" ", strip=True))
    semantics = {
        "code": "コード:3827" in text,
        "market_division": NOTICE_FOREIGN_SHARE_MARKER in text,
        "delisting_date": "上場廃止日 2015年3月26日(木)" in text,
        "exact_event_id": str(source.get("event_id", "")) == NOTICE_EVENT_ID,
        "exact_event_date": str(source.get("event_date", "")) == "2015-03-26",
        "exact_code": str(source.get("raw_code", "")) == "3827",
        "exact_name": str(source.get("issue_name", "")).startswith("ジャパンインベスト・グループ・ピー・エル・シー"),
        "official_url": str(source.get("taxonomy_evidence_url", "")) == NOTICE_URL,
    }
    if not all(semantics.values()):
        _issue(issues, "official_notice_semantic_binding_failed", NOTICE_EVENT_ID, json.dumps(semantics, sort_keys=True))
        exact = False
    return pd.DataFrame(
        [
            {
                "event_id": NOTICE_EVENT_ID,
                "raw_code": "3827",
                "event_date": "2015-03-26",
                "source_sha256": hashlib.sha256(body).hexdigest() if body else "",
                **semantics,
                "classification_from_company_name": False,
                "exact_match": exact and all(semantics.values()),
            }
        ]
    )


def _json(path: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "json_unreadable", path.name, type(exc).__name__)
        return {}


def _csv(path: Path, issues: list[dict[str, str]]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except (OSError, ValueError) as exc:
        _issue(issues, "csv_unreadable", path.name, type(exc).__name__)
        return pd.DataFrame()


def _identity_hash(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1


def _issue(issues: list[dict[str, str]], code: str, record_id: str, detail: str) -> None:
    issues.append({"issue_code": code, "record_id": record_id, "detail": detail})
