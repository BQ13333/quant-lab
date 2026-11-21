#!/usr/bin/env python3
"""Read-only acceptance audit for a completed math-precision OCR corpus.

The auditor never calls an API and never reads environment variables.  It only
reads the queue, source PDFs, and an existing OCR run directory.  Reports are
written only when ``--output-json`` and/or ``--output-csv`` are explicitly
provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from pypdf import PdfReader


SUCCESS_CHUNK_STATUSES = {"ok", "warning", "cached"}
SUCCESS_RETRY_TRANSPORT_STATUSES = {"ok", "cached"}
MATERIALIZED_BOOK_STATUSES = {"verified", "verified_with_equation_tag_warnings"}
PAGE_MARKER_RE = re.compile(r"<!--\s*OCR_PAGE\s+(\d+)\s+source=([^\s>]+)\s*-->")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
TAG_RE = re.compile(r"\\tag\s*\{\s*([^{}]+?)\s*\}")
DISPLAY_BLOCK_RE = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|"
    r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)
EQUATION_TAG_STATUSES = {
    "already_present", "ambiguous", "body_hash_mismatch", "conflicting_tag",
    "density_excluded", "duplicate_label", "invalid_source_geometry", "mapped", "no_display_delimiter",
    "no_formula_geometry", "no_spatial_match", "retry_page_unverified",
    "numeric_value_excluded", "source_geometry_unavailable", "table_excluded", "year_excluded",
    "zero_excluded",
}
EQUATION_TAG_RESIDUAL_STATUSES = {
    "ambiguous", "body_hash_mismatch", "conflicting_tag",
    "invalid_source_geometry", "no_display_delimiter",
    "retry_page_unverified", "source_geometry_unavailable",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path), strict=False).pages)


def parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def normalized_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def duplicate_values(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def parse_front_matter(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return result
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key.strip()] = value
    return {}


def parse_bool(value: Any) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    return None


def parse_bbox(value: Any) -> list[float] | None:
    try:
        parsed = json.loads(str(value)) if isinstance(value, str) else value
        result = [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(result) != 4 or result[2] < result[0] or result[3] < result[1]:
        return None
    return result


def canonical_tag(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().strip("()[]"))


def formula_body_hash(content: str) -> str:
    normalized = re.sub(r"\s+", "", TAG_RE.sub("", content))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def page_sections(markdown: str) -> dict[int, tuple[str, str]]:
    matches = list(PAGE_MARKER_RE.finditer(markdown))
    return {
        int(match.group(1)): (
            match.group(2),
            markdown[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(markdown)],
        )
        for index, match in enumerate(matches)
    }


def latest_retry_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    latest: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("document_id") or ""), parse_int(row.get("page_number")))
        if not key[0] or key[1] < 1:
            continue
        rank = (parse_int(row.get("attempt_number"), 0), str(row.get("completed_at_utc") or ""))
        current = latest.get(key)
        current_rank = (
            parse_int(current.get("attempt_number"), 0) if current else 0,
            str(current.get("completed_at_utc") or "") if current else "",
        )
        if current is None or rank >= current_rank:
            latest[key] = row
    return latest


def _add_error(errors: list[str], document_errors: dict[str, list[str]], message: str, document_id: str = "") -> None:
    errors.append(message)
    if document_id:
        document_errors[document_id].append(message)


def audit_corpus(
    run_dir: Path,
    queue_manifest: Path,
    workspace_root: Path,
    *,
    expected_documents: int = 279,
    expected_pages: int = 7_128,
    page_counter: Callable[[Path], int] = count_pdf_pages,
) -> dict[str, Any]:
    """Audit a run without modifying it and return a serializable report."""

    run_dir = run_dir.expanduser().resolve()
    queue_manifest = queue_manifest.expanduser().resolve()
    workspace_root = workspace_root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    document_errors: dict[str, list[str]] = defaultdict(list)

    required = [
        queue_manifest,
        run_dir / "ocr_plan.csv",
        run_dir / "ocr_manifest.csv",
        run_dir / "ocr_errors.csv",
        run_dir / "page_formula_qa.csv",
        run_dir / "formula_retry_queue.csv",
        run_dir / "equation_tag_geometry_qa.csv",
        run_dir / "book_manifest_verified.csv",
    ]
    for path in required:
        if not path.exists():
            _add_error(errors, document_errors, f"required artifact missing: {path}")

    queue_rows = read_csv(queue_manifest)
    plan_rows = read_csv(run_dir / "ocr_plan.csv")
    chunk_rows = read_csv(run_dir / "ocr_manifest.csv")
    ocr_error_rows = read_csv(run_dir / "ocr_errors.csv")
    qa_rows = read_csv(run_dir / "page_formula_qa.csv")
    retry_queue_rows = read_csv(run_dir / "formula_retry_queue.csv")
    retry_result_path = run_dir / "formula_retry_results.csv"
    retry_rows = read_csv(retry_result_path)
    tag_qa_rows = read_csv(run_dir / "equation_tag_geometry_qa.csv")
    book_rows = read_csv(run_dir / "book_manifest_verified.csv")

    if len(queue_rows) != expected_documents:
        _add_error(errors, document_errors, f"queue row count {len(queue_rows)} != {expected_documents}")
    if len(plan_rows) != expected_documents:
        _add_error(errors, document_errors, f"plan row count {len(plan_rows)} != {expected_documents}")

    queue_paths = [normalized_path(row.get("pdf_path", ""), workspace_root) for row in queue_rows if row.get("pdf_path")]
    if len(queue_paths) != len(queue_rows):
        _add_error(errors, document_errors, "one or more queue rows have an empty pdf_path")
    duplicate_queue_paths = duplicate_values(str(path) for path in queue_paths)
    if duplicate_queue_paths:
        _add_error(errors, document_errors, f"duplicate queue paths: {len(duplicate_queue_paths)}")

    plan_by_id: dict[str, dict[str, str]] = {}
    plan_by_path: dict[Path, dict[str, str]] = {}
    for row in plan_rows:
        document_id = str(row.get("document_id") or "")
        input_path_value = str(row.get("input_path") or "")
        if not document_id:
            _add_error(errors, document_errors, "plan row has empty document_id")
            continue
        if document_id in plan_by_id:
            _add_error(errors, document_errors, f"duplicate plan document_id: {document_id}", document_id)
        plan_by_id[document_id] = row
        if not input_path_value:
            _add_error(errors, document_errors, f"plan input_path is empty: {document_id}", document_id)
            continue
        path = normalized_path(input_path_value, workspace_root)
        if path in plan_by_path:
            _add_error(errors, document_errors, f"duplicate plan input_path: {path}", document_id)
        plan_by_path[path] = row
        if str(row.get("status") or "") != "ready":
            _add_error(errors, document_errors, f"plan status is not ready: {document_id}", document_id)
        digest = str(row.get("input_sha256") or "")
        if not SHA256_RE.fullmatch(digest):
            _add_error(errors, document_errors, f"invalid plan SHA-256: {document_id}", document_id)
        if parse_int(row.get("page_count")) < 1:
            _add_error(errors, document_errors, f"invalid plan page_count: {document_id}", document_id)

    duplicate_plan_hashes = duplicate_values(str(row.get("input_sha256") or "") for row in plan_rows)
    if duplicate_plan_hashes:
        _add_error(errors, document_errors, f"duplicate canonical plan SHA-256 values: {len(duplicate_plan_hashes)}")

    queue_path_set = set(queue_paths)
    plan_path_set = set(plan_by_path)
    if queue_path_set != plan_path_set:
        _add_error(
            errors,
            document_errors,
            f"queue/plan path set mismatch: queue_only={len(queue_path_set - plan_path_set)}, plan_only={len(plan_path_set - queue_path_set)}",
        )

    queue_by_path = {
        normalized_path(row.get("pdf_path", ""), workspace_root): row
        for row in queue_rows
        if row.get("pdf_path")
    }
    for path, plan in plan_by_path.items():
        document_id = str(plan.get("document_id") or "")
        if not path.exists():
            _add_error(errors, document_errors, f"source PDF missing: {path}", document_id)
            continue
        expected_sha = str(plan.get("input_sha256") or "").lower()
        try:
            actual_sha = sha256_file(path)
        except OSError as exc:
            _add_error(errors, document_errors, f"source PDF unreadable: {path}: {exc}", document_id)
            continue
        if actual_sha != expected_sha:
            _add_error(errors, document_errors, f"source SHA mismatch: {document_id}", document_id)
        queue_sha = str(queue_by_path.get(path, {}).get("sha256") or "").strip().lower()
        if queue_sha:
            if not SHA256_RE.fullmatch(queue_sha):
                _add_error(errors, document_errors, f"invalid nonblank queue SHA-256: {document_id}", document_id)
            elif queue_sha != expected_sha:
                _add_error(errors, document_errors, f"queue/plan SHA mismatch: {document_id}", document_id)
        try:
            actual_pages = page_counter(path)
        except Exception as exc:
            _add_error(errors, document_errors, f"source page-count failure: {document_id}: {exc}", document_id)
        else:
            if actual_pages != parse_int(plan.get("page_count")):
                _add_error(errors, document_errors, f"source page-count mismatch: {document_id}", document_id)

    planned_page_total = sum(max(0, parse_int(row.get("page_count"), 0)) for row in plan_rows)
    if planned_page_total != expected_pages:
        _add_error(errors, document_errors, f"planned page total {planned_page_total} != {expected_pages}")

    chunks_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_page_elements: dict[tuple[str, int], tuple[dict[str, str], list[Any]]] = {}
    for row in chunk_rows:
        document_id = str(row.get("document_id") or "")
        if document_id not in plan_by_id:
            _add_error(errors, document_errors, f"chunk references unknown document: {document_id}")
            continue
        chunks_by_doc[document_id].append(row)
    for document_id, plan in plan_by_id.items():
        page_count = parse_int(plan.get("page_count"))
        intervals: list[tuple[int, int]] = []
        for row in chunks_by_doc.get(document_id, []):
            start = parse_int(row.get("start_page"))
            end = parse_int(row.get("end_page"))
            expected = parse_int(row.get("expected_pages"))
            returned = parse_int(row.get("returned_pages"))
            status = str(row.get("status") or "")
            if status not in SUCCESS_CHUNK_STATUSES:
                _add_error(errors, document_errors, f"unsuccessful chunk {row.get('chunk_id')}: {status}", document_id)
            if start < 1 or end < start or expected != end - start + 1 or returned != expected:
                _add_error(errors, document_errors, f"invalid chunk page counts: {row.get('chunk_id')}", document_id)
                continue
            intervals.append((start, end))
            raw_value = str(row.get("raw_response_path") or "")
            raw_path = normalized_path(raw_value, run_dir) if raw_value else Path()
            if not raw_value or not raw_path.is_file() or raw_path.stat().st_size == 0:
                _add_error(errors, document_errors, f"raw response missing/empty: {row.get('chunk_id')}", document_id)
                continue
            try:
                response = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _add_error(errors, document_errors, f"raw response invalid: {row.get('chunk_id')}: {exc}", document_id)
                continue
            layout = response.get("layout_details")
            if not isinstance(layout, list) or len(layout) != expected:
                _add_error(errors, document_errors, f"raw layout page count mismatch: {row.get('chunk_id')}", document_id)
            else:
                for offset, elements in enumerate(layout):
                    raw_page_elements[(document_id, start + offset)] = (
                        row,
                        elements if isinstance(elements, list) else [],
                    )
        intervals.sort()
        expected_intervals: list[tuple[int, int]] = []
        cursor = 1
        for start, end in intervals:
            if start != cursor:
                _add_error(errors, document_errors, f"chunk coverage gap/overlap before page {start}: {document_id}", document_id)
            cursor = max(cursor, end + 1)
            expected_intervals.append((start, end))
        if not intervals or cursor != page_count + 1:
            _add_error(errors, document_errors, f"chunk coverage does not end at page {page_count}: {document_id}", document_id)

    if ocr_error_rows:
        _add_error(errors, document_errors, f"ocr_errors.csv contains {len(ocr_error_rows)} rows")

    expected_page_keys = {
        (document_id, page)
        for document_id, plan in plan_by_id.items()
        for page in range(1, parse_int(plan.get("page_count")) + 1)
    }
    qa_keys = [(str(row.get("document_id") or ""), parse_int(row.get("page_number"))) for row in qa_rows]
    if len(qa_keys) != len(set(qa_keys)):
        _add_error(errors, document_errors, "page_formula_qa.csv contains duplicate page keys")
    qa_key_set = set(qa_keys)
    if qa_key_set != expected_page_keys:
        _add_error(
            errors,
            document_errors,
            f"formula QA page set mismatch: missing={len(expected_page_keys - qa_key_set)}, extra={len(qa_key_set - expected_page_keys)}",
        )
    fail_keys = {
        (str(row.get("document_id") or ""), parse_int(row.get("page_number")))
        for row in qa_rows
        if str(row.get("qa_severity") or "") == "fail"
    }
    warning_count = sum(1 for row in qa_rows if str(row.get("qa_severity") or "") == "warning")
    retry_keys_list = [
        (str(row.get("document_id") or ""), parse_int(row.get("page_number"))) for row in retry_queue_rows
    ]
    if len(retry_keys_list) != len(set(retry_keys_list)):
        _add_error(errors, document_errors, "formula_retry_queue.csv contains duplicate page keys")
    retry_keys = set(retry_keys_list)
    review_keys = {
        (str(row.get("document_id") or ""), parse_int(row.get("page_number")))
        for row in retry_queue_rows
        if str(row.get("status") or "") == "equation_tag_review"
        or "preidentified_tier1_formula_figure_review" in str(row.get("qa_reasons") or "")
    }
    unexpected_queue_only = retry_keys - fail_keys - review_keys
    if fail_keys - retry_keys or unexpected_queue_only:
        _add_error(
            errors,
            document_errors,
            f"retry queue/failing-QA mismatch: fail_only={len(fail_keys - retry_keys)}, queue_only={len(unexpected_queue_only)}",
        )

    if retry_keys and not retry_result_path.exists():
        _add_error(errors, document_errors, f"required artifact missing: {retry_result_path}")
    latest = latest_retry_rows(retry_rows)
    for key in sorted(retry_keys):
        document_id, page_number = key
        row = latest.get(key)
        if row is None:
            _add_error(errors, document_errors, f"retry result missing: {document_id} page {page_number}", document_id)
            continue
        if str(row.get("resolution_status") or "") != "resolved":
            _add_error(errors, document_errors, f"retry unresolved: {document_id} page {page_number}", document_id)
        if str(row.get("transport_status") or "") not in SUCCESS_RETRY_TRANSPORT_STATUSES:
            _add_error(errors, document_errors, f"retry transport unsuccessful: {document_id} page {page_number}", document_id)
        plan_sha = str(plan_by_id.get(document_id, {}).get("input_sha256") or "").lower()
        source_sha = str(row.get("source_pdf_sha256") or "").lower()
        expected_sha = str(row.get("expected_source_sha256") or "").lower()
        if not plan_sha or source_sha != plan_sha or expected_sha != plan_sha:
            _add_error(errors, document_errors, f"retry source SHA mismatch: {document_id} page {page_number}", document_id)
        for path_field, sha_field in (
            ("retry_raw_response_path", "retry_raw_sha256"),
            ("retry_markdown_path", "retry_markdown_sha256"),
        ):
            value = str(row.get(path_field) or "")
            artifact = normalized_path(value, run_dir) if value else Path()
            if not value or not artifact.is_file() or artifact.stat().st_size == 0:
                _add_error(errors, document_errors, f"retry artifact missing/empty ({path_field}): {document_id} page {page_number}", document_id)
                continue
            recorded = str(row.get(sha_field) or "").lower()
            if not SHA256_RE.fullmatch(recorded) or sha256_file(artifact) != recorded:
                _add_error(errors, document_errors, f"retry artifact SHA mismatch ({path_field}): {document_id} page {page_number}", document_id)

    book_by_id: dict[str, dict[str, str]] = {}
    for row in book_rows:
        document_id = str(row.get("document_id") or "")
        if document_id in book_by_id:
            _add_error(errors, document_errors, f"duplicate verified book row: {document_id}", document_id)
        book_by_id[document_id] = row
    if set(book_by_id) != set(plan_by_id):
        _add_error(
            errors,
            document_errors,
            f"verified book/plan set mismatch: missing={len(set(plan_by_id) - set(book_by_id))}, extra={len(set(book_by_id) - set(plan_by_id))}",
        )

    verified_dir = run_dir / "markdown_verified"
    markdown_paths = sorted(path.resolve() for path in verified_dir.rglob("*.md")) if verified_dir.exists() else []
    expected_names = {f"{document_id}.md" for document_id in plan_by_id}
    actual_names = {path.name for path in markdown_paths}
    if len(markdown_paths) != len(actual_names):
        _add_error(errors, document_errors, "duplicate Markdown basenames found under markdown_verified")
    if actual_names != expected_names:
        _add_error(
            errors,
            document_errors,
            f"Markdown filename set mismatch: missing={len(expected_names - actual_names)}, extra={len(actual_names - expected_names)}",
        )

    verified_sections: dict[tuple[str, int], tuple[str, str]] = {}
    for markdown_path in markdown_paths:
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for page_number, section in page_sections(markdown).items():
            verified_sections[(markdown_path.stem, page_number)] = section

    residual_tag_pages: dict[str, set[int]] = defaultdict(set)
    residual_tag_labels = 0
    mapped_tag_count = 0
    already_present_tag_count = 0
    year_excluded_count = 0
    tag_body_integrity_failures = 0
    tag_row_identities: list[str] = []
    mapped_targets: set[tuple[str, int, int]] = set()
    mapped_labels_by_page: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    manually_resolved_tag_pages = {
        key for key, row in latest.items()
        if str(row.get("resolution_status") or "") == "resolved"
        and str(row.get("attempt_id") or "").startswith("manual-attempt-")
    }
    for row_number, row in enumerate(tag_qa_rows, 2):
        document_id = str(row.get("document_id") or "")
        page_number = parse_int(row.get("page_number"))
        status = str(row.get("status") or "")
        label = canonical_tag(row.get("label"))
        source_bbox = parse_bbox(row.get("source_bbox"))
        identity = "|".join(
            (document_id, str(page_number), label, str(row.get("source_kind") or ""), str(source_bbox or ""))
        )
        tag_row_identities.append(identity)
        if document_id not in plan_by_id or page_number < 1 or (
            document_id in plan_by_id and page_number > parse_int(plan_by_id[document_id].get("page_count"))
        ):
            _add_error(errors, document_errors, f"invalid equation-tag QA page at row {row_number}: {document_id} page {page_number}", document_id)
            continue
        if not label or status not in EQUATION_TAG_STATUSES:
            _add_error(errors, document_errors, f"invalid equation-tag QA label/status: {document_id} page {page_number}", document_id)
        injected = parse_bool(row.get("injected"))
        if injected is None or injected != (status == "mapped"):
            _add_error(errors, document_errors, f"equation-tag injected/status mismatch: {document_id} page {page_number}", document_id)

        plan = plan_by_id[document_id]
        source_value = str(row.get("source_pdf_path") or "")
        source_path = normalized_path(source_value, run_dir) if source_value else Path()
        expected_source_path = normalized_path(str(plan.get("input_path") or ""), workspace_root)
        if not source_value or source_path != expected_source_path:
            _add_error(errors, document_errors, f"equation-tag source path mismatch: {document_id} page {page_number}", document_id)
        if str(row.get("source_pdf_sha256") or "").lower() != str(plan.get("input_sha256") or "").lower():
            _add_error(errors, document_errors, f"equation-tag source SHA mismatch: {document_id} page {page_number}", document_id)

        if status in EQUATION_TAG_RESIDUAL_STATUSES and (document_id, page_number) not in manually_resolved_tag_pages:
            residual_tag_pages[document_id].add(page_number)
            residual_tag_labels += 1
        elif status == "already_present":
            already_present_tag_count += 1
        elif status == "year_excluded":
            year_excluded_count += 1
        if status != "mapped":
            continue

        mapped_tag_count += 1
        mapped_labels_by_page[(document_id, page_number)][label] += 1
        element_index = parse_int(row.get("element_index"))
        formula_bbox = parse_bbox(row.get("formula_bbox"))
        before = str(row.get("body_sha256_before") or "").lower()
        after = str(row.get("body_sha256_after") or "").lower()
        if source_bbox is None or formula_bbox is None or element_index < 0:
            _add_error(errors, document_errors, f"mapped equation tag lacks coordinate/index provenance: {document_id} page {page_number}", document_id)
        target = (document_id, page_number, element_index)
        if target in mapped_targets:
            _add_error(errors, document_errors, f"multiple mapped tags target one formula: {document_id} page {page_number} element {element_index}", document_id)
        mapped_targets.add(target)
        if not SHA256_RE.fullmatch(before) or before != after:
            tag_body_integrity_failures += 1
            _add_error(errors, document_errors, f"mapped equation-tag body hash mismatch: {document_id} page {page_number}", document_id)

        raw_value = str(row.get("raw_response_path") or "")
        raw_path = normalized_path(raw_value, run_dir) if raw_value else Path()
        recorded_raw_sha = str(row.get("raw_response_sha256") or "").lower()
        page_raw = raw_page_elements.get((document_id, page_number))
        expected_raw_path = normalized_path(str(page_raw[0].get("raw_response_path") or ""), run_dir) if page_raw else Path()
        if (
            not raw_value or not raw_path.is_file() or raw_path != expected_raw_path
            or not SHA256_RE.fullmatch(recorded_raw_sha) or sha256_file(raw_path) != recorded_raw_sha
        ):
            _add_error(errors, document_errors, f"mapped equation-tag raw provenance mismatch: {document_id} page {page_number}", document_id)
            continue
        raw_element = next(
            (
                element for element in (page_raw[1] if page_raw else [])
                if isinstance(element, dict) and parse_int(element.get("index")) == element_index
            ),
            None,
        )
        raw_is_formula = bool(
            raw_element
            and (
                str(raw_element.get("label") or "") == "formula"
                or "formula" in str(raw_element.get("native_label") or "").lower()
            )
        )
        if not raw_is_formula or parse_bbox(raw_element.get("bbox_2d")) != formula_bbox:
            _add_error(errors, document_errors, f"mapped equation-tag raw element mismatch: {document_id} page {page_number}", document_id)
        elif formula_body_hash(str(raw_element.get("content") or "")) != before:
            tag_body_integrity_failures += 1
            _add_error(errors, document_errors, f"mapped equation-tag raw body hash mismatch: {document_id} page {page_number}", document_id)

        section = verified_sections.get((document_id, page_number), ("", ""))[1]
        matching_blocks = [
            block.group(0)
            for block in DISPLAY_BLOCK_RE.finditer(section)
            if label in {canonical_tag(value) for value in TAG_RE.findall(block.group(0))}
            and formula_body_hash(block.group(0)) == before
        ]
        if not matching_blocks:
            tag_body_integrity_failures += 1
            _add_error(errors, document_errors, f"mapped equation tag absent/body changed in final Markdown: {document_id} page {page_number}", document_id)

    duplicate_tag_rows = duplicate_values(tag_row_identities)
    if duplicate_tag_rows:
        _add_error(errors, document_errors, f"duplicate equation-tag provenance rows: {len(duplicate_tag_rows)}")
    for key, (source_kind, section) in verified_sections.items():
        if source_kind != "first_pass_layout":
            continue
        document_id, page_number = key
        page_raw = raw_page_elements.get(key)
        raw_content = "\n".join(
            str(element.get("content") or "")
            for element in (page_raw[1] if page_raw else [])
            if isinstance(element, dict)
        )
        raw_tags = Counter(canonical_tag(value) for value in TAG_RE.findall(raw_content))
        final_tags = Counter(canonical_tag(value) for value in TAG_RE.findall(section))
        actual_injections = final_tags - raw_tags
        recorded_injections = mapped_labels_by_page.get(key, Counter())
        if actual_injections != recorded_injections:
            _add_error(
                errors,
                document_errors,
                f"final equation-tag/provenance mismatch: {document_id} page {page_number}",
                document_id,
            )

    per_document: list[dict[str, Any]] = []
    total_page_markers = 0
    for document_id, plan in sorted(plan_by_id.items()):
        page_count = parse_int(plan.get("page_count"))
        book = book_by_id.get(document_id, {})
        book_status = str(book.get("status") or "")
        if book_status not in MATERIALIZED_BOOK_STATUSES:
            _add_error(errors, document_errors, f"book status not materialized: {document_id}", document_id)
        warning_pages = {
            parse_int(value) for value in str(book.get("warning_pages") or "").split(";") if value.strip()
        }
        expected_warning_pages = residual_tag_pages.get(document_id, set())
        if book_status == "verified_with_equation_tag_warnings":
            if not warning_pages or warning_pages != expected_warning_pages:
                _add_error(errors, document_errors, f"verified warning-page set mismatch: {document_id}", document_id)
        elif warning_pages:
            _add_error(errors, document_errors, f"clean verified book has warning pages: {document_id}", document_id)
        elif expected_warning_pages:
            _add_error(errors, document_errors, f"clean verified book omits retained equation-tag warnings: {document_id}", document_id)
        if parse_int(book.get("page_count")) != page_count or parse_int(book.get("pages_materialized")) != page_count:
            _add_error(errors, document_errors, f"verified book page counts mismatch: {document_id}", document_id)
        if str(book.get("unresolved_pages") or "").strip():
            _add_error(errors, document_errors, f"verified book has unresolved pages: {document_id}", document_id)

        markdown_value = str(book.get("verified_markdown_path") or "")
        markdown_path = normalized_path(markdown_value, run_dir) if markdown_value else verified_dir / f"{document_id}.md"
        markdown_sha = ""
        marker_count = 0
        if not markdown_path.is_file() or markdown_path.stat().st_size == 0:
            _add_error(errors, document_errors, f"verified Markdown missing/empty: {document_id}", document_id)
        else:
            markdown = markdown_path.read_text(encoding="utf-8")
            markdown_sha = sha256_file(markdown_path)
            front = parse_front_matter(markdown)
            expected_source = str(normalized_path(str(plan.get("input_path") or ""), workspace_root))
            if front.get("document_id") != document_id:
                _add_error(errors, document_errors, f"Markdown document_id mismatch: {document_id}", document_id)
            if front.get("sha256", "").lower() != str(plan.get("input_sha256") or "").lower():
                _add_error(errors, document_errors, f"Markdown source SHA mismatch: {document_id}", document_id)
            if parse_int(front.get("page_count")) != page_count:
                _add_error(errors, document_errors, f"Markdown page_count mismatch: {document_id}", document_id)
            front_source = str(normalized_path(front.get("source_path", ""), workspace_root)) if front.get("source_path") else ""
            if front_source != expected_source:
                _add_error(errors, document_errors, f"Markdown source_path mismatch: {document_id}", document_id)
            matches = list(PAGE_MARKER_RE.finditer(markdown))
            numbers = [int(match.group(1)) for match in matches]
            marker_count = len(matches)
            total_page_markers += marker_count
            if numbers != list(range(1, page_count + 1)):
                _add_error(errors, document_errors, f"Markdown page markers incomplete/out of order: {document_id}", document_id)
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
                if not markdown[match.end():end].strip():
                    _add_error(errors, document_errors, f"empty Markdown page section: {document_id} page {match.group(1)}", document_id)

        per_document.append(
            {
                "document_id": document_id,
                "source_path": str(normalized_path(str(plan.get("input_path") or ""), workspace_root)),
                "source_sha256": str(plan.get("input_sha256") or ""),
                "page_count": page_count,
                "chunks": len(chunks_by_doc.get(document_id, [])),
                "formula_retry_pages": sum(1 for key in retry_keys if key[0] == document_id),
                "equation_tags_mapped": sum(
                    1 for row in tag_qa_rows
                    if str(row.get("document_id") or "") == document_id and str(row.get("status") or "") == "mapped"
                ),
                "equation_tag_residual_pages": len(residual_tag_pages.get(document_id, set())),
                "verified_markdown_path": str(markdown_path),
                "verified_markdown_sha256": markdown_sha,
                "page_markers": marker_count,
                "status": "pass" if not document_errors.get(document_id) else "fail",
                "errors": " | ".join(document_errors.get(document_id, [])),
            }
        )

    if total_page_markers != expected_pages:
        _add_error(errors, document_errors, f"total Markdown page markers {total_page_markers} != {expected_pages}")

    residual_page_total = sum(len(pages) for pages in residual_tag_pages.values())
    if residual_page_total:
        warnings.append(
            f"equation-tag review lane retained: {residual_tag_labels} candidates on {residual_page_total} pages"
        )

    return {
        "status": "pass" if not errors else "fail",
        "run_dir": str(run_dir),
        "queue_manifest": str(queue_manifest),
        "expected_documents": expected_documents,
        "expected_pages": expected_pages,
        "queue_documents": len(queue_rows),
        "planned_documents": len(plan_rows),
        "planned_pages": planned_page_total,
        "verified_books": sum(1 for row in book_rows if str(row.get("status") or "") in MATERIALIZED_BOOK_STATUSES),
        "verified_clean_books": sum(1 for row in book_rows if str(row.get("status") or "") == "verified"),
        "verified_warning_books": sum(
            1 for row in book_rows if str(row.get("status") or "") == "verified_with_equation_tag_warnings"
        ),
        "verified_markdown_files": len(markdown_paths),
        "verified_page_markers": total_page_markers,
        "formula_fail_pages": len(fail_keys),
        "formula_warning_pages": warning_count,
        "first_pass_formula_warning_pages": warning_count,
        "formula_retry_pages": len(retry_keys),
        "resolved_formula_retry_pages": sum(
            1 for key in retry_keys if str(latest.get(key, {}).get("resolution_status") or "") == "resolved"
        ),
        "equation_tags_mapped": mapped_tag_count,
        "equation_tags_already_present": already_present_tag_count,
        "equation_tag_residual_pages": residual_page_total,
        "equation_tag_residual_labels": residual_tag_labels,
        "year_candidates_excluded": year_excluded_count,
        "tag_body_integrity_failures": tag_body_integrity_failures,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "documents": per_document,
    }


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "document_id", "source_path", "source_sha256", "page_count", "chunks",
        "formula_retry_pages", "equation_tags_mapped", "equation_tag_residual_pages",
        "verified_markdown_path", "verified_markdown_sha256",
        "page_markers", "status", "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Read-only acceptance audit for a verified Z.ai OCR corpus.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--workspace-root", default=str(workspace_default))
    parser.add_argument("--expected-documents", type=int, default=279)
    parser.add_argument("--expected-pages", type=int, default=7_128)
    parser.add_argument("--output-json", help="Optional explicit output path; no JSON is written when omitted.")
    parser.add_argument("--output-csv", help="Optional explicit output path; no CSV is written when omitted.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = audit_corpus(
        Path(args.run_dir),
        Path(args.queue_manifest),
        Path(args.workspace_root),
        expected_documents=args.expected_documents,
        expected_pages=args.expected_pages,
    )
    if args.output_json:
        write_json_report(Path(args.output_json), report)
    if args.output_csv:
        write_csv_report(Path(args.output_csv), report["documents"])
    print(
        f"Corpus audit {report['status'].upper()}: "
        f"{report['verified_books']}/{report['expected_documents']} verified books, "
        f"{report['verified_page_markers']}/{report['expected_pages']} page markers, "
        f"{report['error_count']} errors, {report['warning_count']} warnings."
    )
    if report["errors"]:
        for error in report["errors"][:25]:
            print(f"ERROR: {error}")
        if len(report["errors"]) > 25:
            print(f"ERROR: ... {len(report['errors']) - 25} additional errors")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
