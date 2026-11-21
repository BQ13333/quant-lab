#!/usr/bin/env python3
"""Retry formula-flagged OCR pages without mutating first-pass artifacts.

The primary GLM-OCR pipeline intentionally retains its original multi-page
responses.  This companion command consumes ``formula_retry_queue.csv``,
renders only flagged source pages, OCRs them as independent images, and writes
every attempt beneath ``formula_retries``.  Successful retries can then be
assembled into a separate page-addressable ``markdown_verified`` corpus.

Live requests require ``ZAI_API_KEY`` or ``Z_AI_API_KEY``.  The key is passed
only to the request helper and is never written to an artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_ocr_pipeline as ocr
import equation_tag_geometry as tag_geometry


LEDGER_FIELDS = [
    "document_id",
    "input_path",
    "page_number",
    "attempt_id",
    "attempt_number",
    "dpi",
    "model",
    "source_pdf_sha256",
    "expected_source_sha256",
    "rendered_image_path",
    "rendered_image_sha256",
    "first_pass_raw_response_path",
    "first_pass_raw_sha256",
    "first_pass_artifact_unchanged",
    "retry_raw_response_path",
    "retry_raw_sha256",
    "retry_markdown_path",
    "retry_markdown_sha256",
    "transport_status",
    "retry_qa_severity",
    "original_qa_reasons",
    "retry_qa_reasons",
    "first_pass_formula_hashes",
    "retry_formula_hashes",
    "resolution_status",
    "request_id",
    "total_tokens",
    "started_at_utc",
    "completed_at_utc",
    "error_type",
    "error",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def split_reasons(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def dedupe_queue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve queue order while merging duplicate page reasons."""
    output: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for source_row in rows:
        document_id = str(source_row.get("document_id") or "").strip()
        page_number = parse_int(source_row.get("page_number"))
        if not document_id or page_number < 1:
            continue
        key = (document_id, page_number)
        if key not in by_key:
            row = dict(source_row)
            row["document_id"] = document_id
            row["page_number"] = page_number
            row["qa_reasons"] = ";".join(dict.fromkeys(split_reasons(row.get("qa_reasons"))))
            by_key[key] = row
            output.append(row)
            continue
        existing = by_key[key]
        reasons = split_reasons(existing.get("qa_reasons")) + split_reasons(source_row.get("qa_reasons"))
        existing["qa_reasons"] = ";".join(dict.fromkeys(reasons))
    return output


def typed_source_risk_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    integer_fields = {
        "page_number",
        "source_chars",
        "source_glyphs",
        "math_font_chars",
        "math_symbols",
        "theorem_headings",
        "math_risk_score",
    }
    output: list[dict[str, Any]] = []
    for source in rows:
        row: dict[str, Any] = dict(source)
        for field in integer_fields:
            row[field] = parse_int(row.get(field))
        row["formula_expected"] = parse_bool(row.get("formula_expected"))
        try:
            row["math_font_ratio"] = float(row.get("math_font_ratio") or 0)
        except (TypeError, ValueError):
            row["math_font_ratio"] = 0.0
        output.append(row)
    return output


def documents_from_plan(rows: list[dict[str, str]]) -> dict[str, ocr.DocumentRecord]:
    documents: dict[str, ocr.DocumentRecord] = {}
    for row in rows:
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            continue
        page_count = parse_int(row.get("page_count"), -1)
        documents[document_id] = ocr.DocumentRecord(
            document_id=document_id,
            input_path=str(row.get("input_path") or ""),
            file_name=str(row.get("file_name") or Path(str(row.get("input_path") or "document.pdf")).name),
            mime_type=str(row.get("mime_type") or "application/pdf"),
            input_sha256=str(row.get("input_sha256") or ""),
            size_bytes=parse_int(row.get("size_bytes")),
            page_count=None if page_count < 0 else page_count,
            duplicate_of=str(row.get("duplicate_of") or "") or None,
            status=str(row.get("status") or "ready"),
            error=str(row.get("error") or "") or None,
        )
    return documents


def merge_ledger(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the same logical attempt, but never discard older attempts."""
    merged = [dict(row) for row in existing]
    indexes = {
        (str(row.get("document_id")), parse_int(row.get("page_number")), str(row.get("attempt_id"))): index
        for index, row in enumerate(merged)
    }
    for row in additions:
        key = (str(row.get("document_id")), parse_int(row.get("page_number")), str(row.get("attempt_id")))
        if key in indexes:
            merged[indexes[key]] = dict(row)
        else:
            indexes[key] = len(merged)
            merged.append(dict(row))
    return merged


def completed_retry_row(
    ledger: list[dict[str, Any]], document_id: str, page_number: int
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in ledger
        if str(row.get("document_id")) == document_id
        and parse_int(row.get("page_number")) == page_number
        and str(row.get("transport_status")) in {"ok", "warning", "cached"}
        and Path(str(row.get("retry_raw_response_path") or "")).exists()
        and Path(str(row.get("retry_markdown_path") or "")).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: parse_int(row.get("attempt_number")))


def next_attempt_number(ledger: list[dict[str, Any]], document_id: str, page_number: int) -> int:
    attempts = [
        parse_int(row.get("attempt_number"))
        for row in ledger
        if str(row.get("document_id")) == document_id
        and parse_int(row.get("page_number")) == page_number
    ]
    return max(attempts, default=0) + 1


def artifact_paths(
    run_dir: Path, document_id: str, page_number: int, attempt_number: int
) -> dict[str, Path]:
    attempt = f"attempt-{attempt_number:02d}"
    page = f"p{page_number:04d}"
    root = run_dir / "formula_retries"
    return {
        "image": root / "rendered" / document_id / page / f"{attempt}.png",
        "raw": root / "raw" / document_id / page / f"{attempt}.json",
        "markdown": root / "markdown" / document_id / page / f"{attempt}.md",
        "error": root / "errors" / document_id / page / f"{attempt}.error.json",
        "result": root / "results" / document_id / page / f"{attempt}.json",
    }


def preflight_failure_row(
    queue_row: dict[str, Any],
    document: ocr.DocumentRecord | None,
    resolution_status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "document_id": str(queue_row.get("document_id") or ""),
        "input_path": document.input_path if document else str(queue_row.get("input_path") or ""),
        "page_number": parse_int(queue_row.get("page_number")),
        "attempt_id": "preflight",
        "attempt_number": 0,
        "dpi": parse_int(queue_row.get("recommended_dpi"), 300),
        "model": "",
        "source_pdf_sha256": "",
        "expected_source_sha256": document.input_sha256 if document else "",
        "transport_status": "blocked",
        "retry_qa_severity": "fail",
        "original_qa_reasons": str(queue_row.get("qa_reasons") or ""),
        "retry_qa_reasons": "",
        "resolution_status": resolution_status,
        "completed_at_utc": ocr.utc_now(),
        "error_type": resolution_status,
        "error": error,
    }


def page_qa_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (str(row.get("document_id") or ""), parse_int(row.get("page_number"))): row
        for row in rows
        if row.get("document_id") and parse_int(row.get("page_number")) > 0
    }


def evaluate_retry(
    *,
    run_dir: Path,
    document: ocr.DocumentRecord,
    queue_row: dict[str, Any],
    process_row: dict[str, Any],
    source_row: dict[str, Any],
    first_qa_row: dict[str, Any],
    dpi: int,
) -> tuple[str, dict[str, Any]]:
    if str(process_row.get("status")) not in {"ok", "warning", "cached"}:
        return "transport_error", {
            "qa_severity": "fail",
            "qa_reasons": str(process_row.get("error") or "OCR transport failed"),
            "formula_hashes": "",
        }
    qa_rows, _ = ocr.build_math_precision_page_qa(
        run_dir,
        [document],
        [process_row],
        [source_row],
        dpi,
    )
    if not qa_rows:
        return "unresolved", {
            "qa_severity": "fail",
            "qa_reasons": "retry_page_qa_missing",
            "formula_hashes": "",
        }
    retry_qa = qa_rows[0]
    severity = str(retry_qa.get("qa_severity") or "fail")
    if severity == "fail":
        return "unresolved", retry_qa
    if severity == "warning":
        return "review_required", retry_qa
    first_hashes = {value for value in str(first_qa_row.get("formula_hashes") or "").split(";") if value}
    retry_hashes = {value for value in str(retry_qa.get("formula_hashes") or "").split(";") if value}
    if first_hashes and retry_hashes and first_hashes != retry_hashes:
        retry_qa = dict(retry_qa)
        existing = split_reasons(retry_qa.get("qa_reasons"))
        retry_qa["qa_reasons"] = ";".join(existing + ["formula_hash_disagreement"])
        return "review_required", retry_qa
    return "resolved", retry_qa


def process_retry_page(
    *,
    run_dir: Path,
    queue_row: dict[str, Any],
    document: ocr.DocumentRecord,
    source_row: dict[str, Any],
    first_qa_row: dict[str, Any],
    attempt_number: int,
    dpi: int,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: float,
    max_image_mb: float,
    render_fn: Callable[[Path, Path, int, int], bool] | None = None,
    process_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    document_id = document.document_id
    page_number = parse_int(queue_row.get("page_number"))
    attempt_id = f"attempt-{attempt_number:02d}"
    paths = artifact_paths(run_dir, document_id, page_number, attempt_number)
    source_pdf = Path(document.input_path)
    first_raw = Path(str(queue_row.get("first_pass_raw_response_path") or ""))
    first_hash_before = ocr.sha256_file(first_raw) if first_raw.exists() else ""
    started_at = ocr.utc_now()
    render_fn = render_fn or ocr.render_page_image
    process_fn = process_fn or ocr.process_chunk

    paths["image"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["image"].exists() and not render_fn(source_pdf, paths["image"], page_number, dpi):
        return {
            **preflight_failure_row(queue_row, document, "transport_error", "Page image rendering failed."),
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "dpi": dpi,
            "model": model,
            "first_pass_raw_response_path": str(first_raw),
            "first_pass_raw_sha256": first_hash_before,
            "started_at_utc": started_at,
        }
    image_size = paths["image"].stat().st_size
    if image_size > int(max_image_mb * 1024 * 1024):
        return {
            **preflight_failure_row(
                queue_row,
                document,
                "transport_error",
                f"Rendered image is {image_size / 1024 / 1024:.2f}MB; provider limit is {max_image_mb:.2f}MB.",
            ),
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "dpi": dpi,
            "model": model,
            "rendered_image_path": str(paths["image"].resolve()),
            "rendered_image_sha256": ocr.sha256_file(paths["image"]),
            "first_pass_raw_response_path": str(first_raw),
            "first_pass_raw_sha256": first_hash_before,
            "started_at_utc": started_at,
        }

    chunk = ocr.ChunkRecord(
        document_id=document_id,
        chunk_id=f"{document_id}__formula-retry-p{page_number:04d}-{attempt_id}",
        input_path=document.input_path,
        chunk_pdf_path=str(paths["image"].resolve()),
        input_kind="image",
        raw_response_path=str(paths["raw"].resolve()),
        text_path=str(paths["markdown"].resolve()),
        start_page=page_number,
        end_page=page_number,
        expected_pages=1,
        size_bytes=image_size,
        status="ready",
    )
    process_row = process_fn(
        chunk,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        run_id=f"{run_dir.name}-formula-retry",
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
        base64_data_uri=True,
        fallback_data_uri=True,
        return_crop_images=True,
        need_layout_visualization=True,
        force=False,
    )
    process_error_path = paths["raw"].with_suffix(".error.json")
    if process_error_path.exists():
        paths["error"].parent.mkdir(parents=True, exist_ok=True)
        if not paths["error"].exists():
            shutil.copy2(process_error_path, paths["error"])
    resolution, retry_qa = evaluate_retry(
        run_dir=run_dir,
        document=document,
        queue_row=queue_row,
        process_row=process_row,
        source_row=source_row,
        first_qa_row=first_qa_row,
        dpi=dpi,
    )
    first_hash_after = ocr.sha256_file(first_raw) if first_raw.exists() else ""
    first_unchanged = bool(first_hash_before) and first_hash_before == first_hash_after
    if not first_unchanged:
        resolution = "artifact_integrity_failure"
    row = {
        "document_id": document_id,
        "input_path": document.input_path,
        "page_number": page_number,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "dpi": dpi,
        "model": model,
        "source_pdf_sha256": document.input_sha256,
        "expected_source_sha256": document.input_sha256,
        "rendered_image_path": str(paths["image"].resolve()),
        "rendered_image_sha256": ocr.sha256_file(paths["image"]),
        "first_pass_raw_response_path": str(first_raw),
        "first_pass_raw_sha256": first_hash_before,
        "first_pass_artifact_unchanged": first_unchanged,
        "retry_raw_response_path": str(paths["raw"].resolve()) if paths["raw"].exists() else "",
        "retry_raw_sha256": ocr.sha256_file(paths["raw"]) if paths["raw"].exists() else "",
        "retry_markdown_path": str(paths["markdown"].resolve()) if paths["markdown"].exists() else "",
        "retry_markdown_sha256": ocr.sha256_file(paths["markdown"]) if paths["markdown"].exists() else "",
        "transport_status": process_row.get("status", "error"),
        "retry_qa_severity": retry_qa.get("qa_severity", "fail"),
        "original_qa_reasons": str(queue_row.get("qa_reasons") or ""),
        "retry_qa_reasons": str(retry_qa.get("qa_reasons") or ""),
        "first_pass_formula_hashes": str(first_qa_row.get("formula_hashes") or ""),
        "retry_formula_hashes": str(retry_qa.get("formula_hashes") or ""),
        "resolution_status": resolution,
        "request_id": process_row.get("request_id", ""),
        "total_tokens": process_row.get("total_tokens", ""),
        "started_at_utc": process_row.get("started_at_utc") or started_at,
        "completed_at_utc": process_row.get("completed_at_utc") or ocr.utc_now(),
        "error_type": process_row.get("error_type", ""),
        "error": process_row.get("error", ""),
    }
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    ocr.write_json(paths["result"], row)
    return row


def equation_tag_repair_order(equation_labels: list[str], formula_count: int) -> list[str]:
    dotted = [label for label in equation_labels if "." in label]
    return next(
        (labels for labels in (equation_labels, dotted) if len(labels) == formula_count),
        [],
    )


def inject_equation_tag(content: str, label: str) -> str:
    inner = label[1:-1] if label.startswith("(") and label.endswith(")") else label
    compact = re.sub(r"\s+", "", content)
    if label in compact or rf"\tag{{{inner}}}" in compact:
        return content
    closing = content.rfind("$$")
    if closing > 0:
        return content[:closing].rstrip() + rf" \tag{{{inner}}}" + "\n" + content[closing:]
    return content.rstrip() + rf" \tag{{{inner}}}"


def layout_page_markdown(
    elements: Any,
    equation_labels: list[str] | None = None,
    geometry_tags_by_element_index: dict[int, str] | None = None,
) -> str:
    if not isinstance(elements, list):
        return ""
    parts: list[str] = []
    sortable = [element for element in elements if isinstance(element, dict)]
    sortable.sort(key=lambda element: parse_int(element.get("index"), 10**9))
    formula_elements = [
        element
        for element in sortable
        if str(element.get("label") or "") == "formula"
        or "formula" in str(element.get("native_label") or "").lower()
    ]
    repair_order = equation_tag_repair_order(equation_labels or [], len(formula_elements))
    formula_index = 0
    for element in sortable:
        content = str(element.get("content") or "").strip()
        if not content:
            continue
        is_formula = (
            str(element.get("label") or "") == "formula"
            or "formula" in str(element.get("native_label") or "").lower()
        )
        if is_formula and geometry_tags_by_element_index is not None:
            try:
                element_index = parse_int(element.get("index"), -1)
                geometry_label = geometry_tags_by_element_index.get(element_index)
            except (TypeError, ValueError):
                geometry_label = None
            if geometry_label:
                repaired, status = tag_geometry.inject_tag_into_display(content, geometry_label)
                if status in {"inserted", "already_present"}:
                    content = repaired
        elif is_formula and repair_order:
            content = inject_equation_tag(content, repair_order[formula_index])
            formula_index += 1
        if str(element.get("label") or "") == "image" and re.match(r"^https?://", content):
            content = f"![OCR image]({content})"
        parts.append(content)
    return "\n\n".join(parts).strip()


def latest_rows_by_page(ledger: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in ledger:
        key = (str(row.get("document_id") or ""), parse_int(row.get("page_number")))
        if not key[0] or key[1] < 1:
            continue
        row_rank = (parse_int(row.get("attempt_number")), str(row.get("completed_at_utc") or ""))
        current_rank = (
            parse_int(latest.get(key, {}).get("attempt_number")),
            str(latest.get(key, {}).get("completed_at_utc") or ""),
        )
        if key not in latest or row_rank >= current_rank:
            latest[key] = row
    return latest


EQUATION_TAG_QA_FIELDS = [
    "document_id",
    "page_number",
    "label",
    "status",
    "source_kind",
    "element_index",
    "source_bbox",
    "formula_bbox",
    "body_sha256_before",
    "body_sha256_after",
    "spatial_cost",
    "vertical_edge_distance",
    "vertical_center_distance",
    "horizontal_gap",
    "label_x_ratio",
    "injected",
    "raw_response_path",
    "raw_response_sha256",
    "source_pdf_path",
    "source_pdf_sha256",
]

EQUATION_TAG_RESIDUAL_STATUSES = {
    "ambiguous",
    "body_hash_mismatch",
    "conflicting_tag",
    "invalid_source_geometry",
    "no_display_delimiter",
    "retry_page_unverified",
    "source_geometry_unavailable",
}


def source_equation_label_words(pdf_page: Any) -> list[dict[str, Any]]:
    """Return all equation-shaped PDF words with source-coordinate boxes.

    Geometry, rather than a fixed page-side threshold, distinguishes true
    equation numbers from prose references and supports two-column papers.
    """
    words: list[dict[str, Any]] = []
    for word in pdf_page.extract_words() or []:
        label = tag_geometry.parenthesized_label(word.get("text"))
        if not tag_geometry.EQUATION_LABEL_RE.fullmatch(label):
            continue
        try:
            bbox = [float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"])]
        except (KeyError, TypeError, ValueError):
            continue
        words.append({"text": label, "bbox": bbox})
    return words


def equation_tag_qa_row(
    *,
    document: ocr.DocumentRecord,
    page_number: int,
    decision: dict[str, Any],
    source_kind: str,
    raw_path: Path | None = None,
    raw_sha256: str = "",
) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "page_number": page_number,
        "label": str(decision.get("label") or ""),
        "status": str(decision.get("status") or ""),
        "source_kind": source_kind,
        "element_index": decision.get("element_index", ""),
        "source_bbox": json.dumps(decision.get("source_bbox") or [], separators=(",", ":")),
        "formula_bbox": json.dumps(decision.get("formula_bbox") or [], separators=(",", ":")),
        "body_sha256_before": str(decision.get("body_sha256_before") or ""),
        "body_sha256_after": str(decision.get("body_sha256_after") or ""),
        "spatial_cost": decision.get("spatial_cost", ""),
        "vertical_edge_distance": decision.get("vertical_edge_distance", ""),
        "vertical_center_distance": decision.get("vertical_center_distance", ""),
        "horizontal_gap": decision.get("horizontal_gap", ""),
        "label_x_ratio": decision.get("label_x_ratio", ""),
        "injected": str(decision.get("status") or "") == "mapped",
        "raw_response_path": str(raw_path.resolve()) if raw_path else "",
        "raw_response_sha256": raw_sha256,
        "source_pdf_path": document.input_path,
        "source_pdf_sha256": document.input_sha256,
    }


def assemble_verified_documents(
    run_dir: Path,
    documents: dict[str, ocr.DocumentRecord],
    queue_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunk_rows = read_csv(run_dir / "ocr_manifest.csv")
    chunks_by_doc: dict[str, list[dict[str, str]]] = {}
    for row in chunk_rows:
        if str(row.get("status")) in {"ok", "warning", "cached"}:
            chunks_by_doc.setdefault(str(row.get("document_id")), []).append(row)
    queued_by_doc: dict[str, set[int]] = {}
    for row in queue_rows:
        queued_by_doc.setdefault(str(row["document_id"]), set()).add(parse_int(row["page_number"]))
    latest = latest_rows_by_page(ledger)
    source_risk_by_page = {
        (str(row.get("document_id") or ""), parse_int(row.get("page_number"))): row
        for row in read_csv(run_dir / "source_page_math_risk.csv")
    }
    tag_qa_rows: list[dict[str, Any]] = []
    assembly_rows: list[dict[str, Any]] = []

    for document_id, document in documents.items():
        if document.status != "ready" or not document.page_count:
            continue
        unresolved = [
            page
            for page in sorted(queued_by_doc.get(document_id, set()))
            if str(latest.get((document_id, page), {}).get("resolution_status")) != "resolved"
        ]
        final_path = run_dir / "markdown_verified" / f"{document_id}.md"
        if unresolved:
            assembly_rows.append(
                {
                    "document_id": document_id,
                    "status": "blocked_unresolved_retry",
                    "verified_markdown_path": "",
                    "page_count": document.page_count,
                    "pages_materialized": 0,
                    "retry_pages_used": 0,
                    "unresolved_pages": ";".join(map(str, unresolved)),
                }
            )
            continue

        page_markdown: dict[int, str] = {}
        pdf_handle: Any = None
        try:
            pdf_handle = ocr.pdfplumber.open(document.input_path)
        except Exception:
            pdf_handle = None
        try:
            for chunk_row in chunks_by_doc.get(document_id, []):
                raw_path = Path(str(chunk_row.get("raw_response_path") or ""))
                if not raw_path.exists():
                    continue
                try:
                    response = json.loads(raw_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                raw_sha256 = ocr.sha256_file(raw_path)
                start_page = parse_int(chunk_row.get("start_page"))
                for offset, elements in enumerate(response.get("layout_details") or []):
                    page_number = start_page + offset
                    geometry_tags: dict[int, str] = {}
                    if page_number not in queued_by_doc.get(document_id, set()):
                        has_formula_geometry = any(
                            isinstance(element, dict)
                            and tag_geometry.is_formula_element(element)
                            and len(element.get("bbox_2d") or []) == 4
                            for element in (elements if isinstance(elements, list) else [])
                        )
                        if pdf_handle is not None and page_number <= len(pdf_handle.pages) and has_formula_geometry:
                            pdf_page = pdf_handle.pages[page_number - 1]
                            plan = tag_geometry.build_geometry_tag_plan(
                                elements if isinstance(elements, list) else [],
                                source_equation_label_words(pdf_page),
                                float(pdf_page.width),
                                float(pdf_page.height),
                            )
                            geometry_tags = dict(plan["tags_by_element_index"])
                            for decision in plan["decisions"]:
                                tag_qa_rows.append(
                                    equation_tag_qa_row(
                                        document=document,
                                        page_number=page_number,
                                        decision=decision,
                                        source_kind="first_pass_geometry",
                                        raw_path=raw_path,
                                        raw_sha256=raw_sha256,
                                    )
                                )
                        elif pdf_handle is None or page_number > len(pdf_handle.pages):
                            source_row = source_risk_by_page.get((document_id, page_number), {})
                            for label in str(source_row.get("display_equation_labels") or "").split(";"):
                                if label and not tag_geometry.is_year_like_label(label):
                                    tag_qa_rows.append(
                                        equation_tag_qa_row(
                                            document=document,
                                            page_number=page_number,
                                            decision={"label": label, "status": "source_geometry_unavailable"},
                                            source_kind="first_pass_geometry",
                                            raw_path=raw_path,
                                            raw_sha256=raw_sha256,
                                        )
                                    )
                    page_markdown[page_number] = layout_page_markdown(
                        elements, geometry_tags_by_element_index=geometry_tags
                    )
        finally:
            if pdf_handle is not None:
                pdf_handle.close()

        for page_number in range(1, document.page_count + 1):
            if page_markdown.get(page_number, "").strip():
                continue
            source_row = source_risk_by_page.get((document_id, page_number), {})
            if parse_int(source_row.get("source_chars")) <= 10 and parse_int(
                source_row.get("math_font_chars")
            ) == 0:
                page_markdown[page_number] = "*[Intentionally blank source page.]*"

        retry_pages_used = 0
        for page_number in queued_by_doc.get(document_id, set()):
            retry_row = latest[(document_id, page_number)]
            retry_path = Path(str(retry_row.get("retry_markdown_path") or ""))
            if retry_path.exists():
                page_markdown[page_number] = retry_path.read_text(encoding="utf-8").strip()
                retry_pages_used += 1
                source_row = source_risk_by_page.get((document_id, page_number), {})
                present_tags = set(tag_geometry.formula_tags(page_markdown[page_number]))
                for label in dict.fromkeys(
                    value for value in str(source_row.get("display_equation_labels") or "").split(";") if value
                ):
                    if tag_geometry.is_year_like_label(label):
                        status = "year_excluded"
                    elif tag_geometry.canonical_label(label) in present_tags:
                        status = "already_present"
                    else:
                        status = "retry_page_unverified"
                    tag_qa_rows.append(
                        equation_tag_qa_row(
                            document=document,
                            page_number=page_number,
                            decision={"label": label, "status": status},
                            source_kind="retry_markdown",
                        )
                    )

        missing = [page for page in range(1, document.page_count + 1) if not page_markdown.get(page, "").strip()]
        if missing:
            assembly_rows.append(
                {
                    "document_id": document_id,
                    "status": "blocked_missing_pages",
                    "verified_markdown_path": "",
                    "page_count": document.page_count,
                    "pages_materialized": document.page_count - len(missing),
                    "retry_pages_used": retry_pages_used,
                    "unresolved_pages": ";".join(map(str, missing)),
                }
            )
            continue

        parts = [
            "---",
            f'title: "{document.file_name.replace(chr(34), chr(39))}"',
            f'document_id: "{document_id}"',
            f'source_path: "{document.input_path.replace(chr(34), chr(39))}"',
            f"sha256: {document.input_sha256}",
            f"page_count: {document.page_count}",
            "ocr_provider: zai",
            "ocr_model: glm-ocr",
            "verification: page-level-formula-retry",
            "---",
            "",
            f"# {document.file_name}",
        ]
        for page_number in range(1, document.page_count + 1):
            if page_number in queued_by_doc.get(document_id, set()):
                retry_model = str(latest.get((document_id, page_number), {}).get("model") or "")
                source = "manual_adjudication" if retry_model == "human-visual-adjudication" else "formula_retry"
            else:
                source = "first_pass_layout"
            parts.extend(
                [
                    "",
                    f"<!-- OCR_PAGE {page_number} source={source} -->",
                    "",
                    page_markdown[page_number],
                ]
            )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        warning_pages = sorted(
            {
                parse_int(row.get("page_number"))
                for row in tag_qa_rows
                if str(row.get("document_id") or "") == document_id
                and str(row.get("status") or "") in EQUATION_TAG_RESIDUAL_STATUSES
                and not (
                    str(latest.get((document_id, parse_int(row.get("page_number"))), {}).get("resolution_status") or "") == "resolved"
                    and str(latest.get((document_id, parse_int(row.get("page_number"))), {}).get("attempt_id") or "").startswith("manual-attempt-")
                )
            }
        )
        assembly_rows.append(
            {
                "document_id": document_id,
                "status": "verified_with_equation_tag_warnings" if warning_pages else "verified",
                "verified_markdown_path": str(final_path.resolve()),
                "page_count": document.page_count,
                "pages_materialized": document.page_count,
                "retry_pages_used": retry_pages_used,
                "unresolved_pages": "",
                "warning_pages": ";".join(map(str, warning_pages)),
            }
        )
    deduped_tag_qa_rows: list[dict[str, Any]] = []
    seen_tag_sources: set[tuple[str, int, str, str, str]] = set()
    for row in tag_qa_rows:
        source_key = (
            str(row.get("document_id") or ""),
            parse_int(row.get("page_number")),
            str(row.get("label") or ""),
            str(row.get("source_kind") or ""),
            str(row.get("source_bbox") or ""),
        )
        if source_key in seen_tag_sources:
            continue
        seen_tag_sources.add(source_key)
        deduped_tag_qa_rows.append(row)
    ocr.write_csv(
        run_dir / "equation_tag_geometry_qa.csv",
        [{field: row.get(field, "") for field in EQUATION_TAG_QA_FIELDS} for row in deduped_tag_qa_rows],
        EQUATION_TAG_QA_FIELDS,
    )
    return assembly_rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry GLM-OCR formula-flagged pages without overwriting first-pass artifacts.")
    parser.add_argument("--run-dir", required=True, help="Existing math-precision OCR run directory.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-new-attempt", action="store_true")
    parser.add_argument(
        "--force-unresolved-only",
        action="store_true",
        help="With --force-new-attempt, retry only pages whose latest ledger status is not resolved.",
    )
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--provider-max-image-mb", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.dpi < 150:
        raise SystemExit("--dpi must be >= 150 for formula retries.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    queue_path = run_dir / "formula_retry_queue.csv"
    queue_rows = dedupe_queue_rows(read_csv(queue_path))
    if not queue_path.exists():
        raise SystemExit(f"Retry queue not found: {queue_path}")
    documents = documents_from_plan(read_csv(run_dir / "ocr_plan.csv"))
    source_rows = typed_source_risk_rows(read_csv(run_dir / "source_page_math_risk.csv"))
    source_by_page = {
        (str(row.get("document_id")), parse_int(row.get("page_number"))): row for row in source_rows
    }
    first_qa_by_page = page_qa_lookup(read_csv(run_dir / "page_formula_qa.csv"))
    ledger_path = run_dir / "formula_retry_results.csv"
    existing_ledger: list[dict[str, Any]] = read_csv(ledger_path)

    manifest_path = run_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    parameters = original_manifest.get("ocr_parameters") or {}
    endpoint = args.endpoint or original_manifest.get("ocr_endpoint") or ocr.DEFAULT_ENDPOINT
    model = args.model or original_manifest.get("ocr_model") or ocr.DEFAULT_MODEL
    timeout_seconds = args.timeout_seconds or parse_int(parameters.get("timeout_seconds"), 1800)
    retries = args.retries or parse_int(parameters.get("retries"), 7)
    max_image_mb = args.provider_max_image_mb or float(parameters.get("provider_max_image_mb") or 10.0)

    additions: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        document_id = str(queue_row["document_id"])
        page_number = parse_int(queue_row["page_number"])
        document = documents.get(document_id)
        if document is None:
            additions.append(preflight_failure_row(queue_row, None, "document_not_in_plan", "Document is absent from ocr_plan.csv."))
            continue
        if document.page_count is None or page_number > document.page_count:
            additions.append(preflight_failure_row(queue_row, document, "invalid_page", "Page number is outside the planned document range."))
            continue
        source_pdf = Path(document.input_path)
        if not source_pdf.exists():
            additions.append(preflight_failure_row(queue_row, document, "source_missing", "Source PDF no longer exists."))
            continue
        actual_sha = ocr.sha256_file(source_pdf)
        if not document.input_sha256 or actual_sha != document.input_sha256:
            row = preflight_failure_row(queue_row, document, "source_changed", "Source PDF SHA-256 differs from ocr_plan.csv.")
            row["source_pdf_sha256"] = actual_sha
            additions.append(row)
            continue
        if args.force_new_attempt and args.force_unresolved_only:
            latest_for_page = latest_rows_by_page(existing_ledger).get((document_id, page_number), {})
            if str(latest_for_page.get("resolution_status") or "") == "resolved":
                continue
        elif not args.force_new_attempt:
            cached = completed_retry_row(existing_ledger, document_id, page_number)
            if cached is not None:
                continue
        jobs.append(
            {
                "run_dir": run_dir,
                "queue_row": queue_row,
                "document": document,
                "source_row": source_by_page.get((document_id, page_number), {}),
                "first_qa_row": first_qa_by_page.get((document_id, page_number), {}),
                "attempt_number": next_attempt_number(existing_ledger + additions, document_id, page_number),
                "dpi": args.dpi,
                "endpoint": endpoint,
                "model": model,
                "timeout_seconds": timeout_seconds,
                "retries": retries,
                "retry_sleep_seconds": args.retry_sleep_seconds,
                "max_image_mb": max_image_mb,
            }
        )

    if jobs:
        api_key = ocr.env_api_key()
        if not api_key:
            raise SystemExit("Missing API key. Export ZAI_API_KEY or Z_AI_API_KEY before live formula retries.")
        for job in jobs:
            job["api_key"] = api_key
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_retry_page, **job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                additions.append(row)
                print(
                    f"[{index}/{len(futures)}] {row['document_id']} page {row['page_number']} -> {row['resolution_status']}",
                    flush=True,
                )

    ledger = merge_ledger(existing_ledger, additions)
    ocr.write_csv(
        ledger_path,
        [{field: row.get(field, "") for field in LEDGER_FIELDS} for row in ledger],
        LEDGER_FIELDS,
    )
    assembly_rows = assemble_verified_documents(run_dir, documents, queue_rows, ledger)
    assembly_fields = [
        "document_id",
        "status",
        "verified_markdown_path",
        "page_count",
        "pages_materialized",
        "retry_pages_used",
        "unresolved_pages",
        "warning_pages",
    ]
    ocr.write_csv(
        run_dir / "book_manifest_verified.csv",
        [{field: row.get(field, "") for field in assembly_fields} for row in assembly_rows],
        assembly_fields,
    )
    counts: dict[str, int] = {}
    for row in latest_rows_by_page(ledger).values():
        status = str(row.get("resolution_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    retry_manifest = {
        "source_run_id": original_manifest.get("run_id") or run_dir.name,
        "generated_at_utc": ocr.utc_now(),
        "model": model,
        "endpoint": endpoint,
        "dpi": args.dpi,
        "workers": args.workers,
        "force_new_attempt": args.force_new_attempt,
        "force_unresolved_only": args.force_unresolved_only,
        "queue_rows": len(queue_rows),
        "new_attempts": len(jobs),
        "resolution_counts": counts,
        "artifacts": {
            "retry_ledger": str(ledger_path.resolve()),
            "retry_root": str((run_dir / "formula_retries").resolve()),
            "verified_book_manifest": str((run_dir / "book_manifest_verified.csv").resolve()),
            "verified_markdown": str((run_dir / "markdown_verified").resolve()),
        },
    }
    ocr.write_json(run_dir / "formula_retry_manifest.json", retry_manifest)
    print(f"Formula retry phase complete: {run_dir}")
    print(f"Retry ledger: {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
