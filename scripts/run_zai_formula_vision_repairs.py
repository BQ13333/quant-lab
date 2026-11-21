#!/usr/bin/env python3
"""Repair unresolved formula OCR pages with GLM-5V-Turbo vision.

This is the second, independent repair stage for pages whose latest row in
``formula_retry_results.csv`` is still ``unresolved`` or ``review_required``.
It uses Z.AI's official chat-completions image input and never mutates a
first-pass or GLM-OCR retry artifact.  Every GLM-5V-Turbo response, parsed
Markdown page, rendered image, error, and result record is written to a new
numbered attempt below ``formula_vision_repairs``.

Live requests require ``ZAI_API_KEY`` or ``Z_AI_API_KEY``.  The secret is used
only in the in-memory Authorization header and is never included in an output
artifact or exception message.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_retries as retry
import run_zai_ocr_pipeline as ocr


DEFAULT_ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5v-turbo"
BEGIN_MARKER = "<BEGIN_MARKDOWN>"
END_MARKER = "<END_MARKDOWN>"
REPAIRABLE_STATUSES = {"unresolved", "review_required", "transport_error"}
TRANSIENT_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class VisionRepairError(RuntimeError):
    """A sanitized transport or response error."""


def redact_secret(value: Any, secret: str) -> str:
    text = str(value or "")
    return text.replace(secret, "[REDACTED]") if secret else text


def redact_secret_value(value: Any, secret: str) -> Any:
    """Recursively remove a secret even if a provider unexpectedly echoes it."""
    if isinstance(value, dict):
        return {key: redact_secret_value(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secret_value(item, secret) for item in value]
    if isinstance(value, tuple):
        return [redact_secret_value(item, secret) for item in value]
    if isinstance(value, str):
        return redact_secret(value, secret)
    return value


def write_text_exclusive(path: Path, value: str) -> None:
    """Create an immutable text artifact; never replace an existing attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_exclusive(path: Path, value: Any) -> None:
    write_text_exclusive(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def vision_artifact_paths(
    run_dir: Path, document_id: str, page_number: int, attempt_number: int
) -> dict[str, Path]:
    page = f"p{page_number:04d}"
    attempt = f"vision-attempt-{attempt_number:02d}"
    root = run_dir / "formula_vision_repairs"
    return {
        "image": root / "rendered" / document_id / page / f"{attempt}.png",
        "raw": root / "raw" / document_id / page / f"{attempt}.json",
        "markdown": root / "markdown" / document_id / page / f"{attempt}.md",
        "error": root / "errors" / document_id / page / f"{attempt}.error.json",
        "result": root / "results" / document_id / page / f"{attempt}.json",
    }


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def parse_page_markdown(content: str) -> str:
    """Extract the page Markdown while tolerating common model wrappers."""
    text = content.strip()
    if BEGIN_MARKER in text and END_MARKER in text:
        start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
        end = text.index(END_MARKER, start)
        text = text[start:end].strip()
    fenced = re.fullmatch(r"```(?:markdown|md)?\s*\n?(.*?)\n?```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return text.replace(BEGIN_MARKER, "").replace(END_MARKER, "").strip()


def extract_display_formulas(markdown: str) -> list[str]:
    pattern = re.compile(
        r"\$\$(.+?)\$\$"
        r"|(?<!\\)\\\[(.+?)(?<!\\)\\\]"
        r"|\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath)\}"
        r"(.+?)\\end\{\3\}",
        re.DOTALL,
    )
    formulas: list[str] = []
    for match in pattern.finditer(markdown):
        formula = next(
            (group for group in (match.group(1), match.group(2), match.group(4)) if group is not None),
            "",
        )
        formulas.append(str(formula or "").strip())
    return formulas


def latex_structure_issues(formula: str) -> list[str]:
    """Check structural LaTeX without matching ``\\right`` in ``\\rightarrow``."""
    issues: list[str] = []
    if not formula.strip():
        return ["empty_formula"]
    without_comments = re.sub(r"(?<!\\)%.*", "", formula)
    unescaped = re.sub(r"\\[{}$]", "", without_comments)
    if unescaped.count("{") != unescaped.count("}"):
        issues.append("unbalanced_braces")

    stack: list[str] = []
    environment_error = False
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", without_comments):
        kind, name = match.groups()
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            environment_error = True
            break
    if stack or environment_error:
        issues.append("unbalanced_environments")

    left_count = len(re.findall(r"\\left(?![A-Za-z])", without_comments))
    right_count = len(re.findall(r"\\right(?![A-Za-z])", without_comments))
    if left_count != right_count:
        issues.append("unbalanced_left_right")
    return issues


def markdown_structure_issues(markdown: str) -> list[str]:
    issues: list[str] = []
    if markdown.count("$$") % 2:
        issues.append("unbalanced_display_dollars")
    if len(re.findall(r"(?<!\\)\\\[", markdown)) != len(re.findall(r"(?<!\\)\\\]", markdown)):
        issues.append("unbalanced_display_brackets")
    for index, formula in enumerate(extract_display_formulas(markdown), 1):
        issues.extend(f"formula_{index}:{issue}" for issue in latex_structure_issues(formula))
    return list(dict.fromkeys(issues))


def reason_equation_labels(*rows: dict[str, Any]) -> list[str]:
    """Use labels explicitly reported missing, not prose references or years."""
    labels: list[str] = []
    for row in rows:
        for field in ("qa_reasons", "original_qa_reasons", "retry_qa_reasons"):
            value = str(row.get(field) or "")
            reason_names = (
                "missing_equation_labels",
                "recoverable_display_equation_labels",
                "unmapped_display_equation_labels",
            )
            pattern = rf"(?:{'|'.join(reason_names)})=([^;]+)"
            for match in re.finditer(pattern, value):
                labels.extend(re.findall(r"\([A-Z]?\.?\d+(?:\.\d+){0,3}[a-z]?\)", match.group(1)))
    return list(dict.fromkeys(labels))


def label_is_present(markdown: str, label: str) -> bool:
    inner = label[1:-1]
    compact = re.sub(r"\s+", "", markdown)
    return label in compact or rf"\tag{{{inner}}}" in compact or rf"\tag{{{label}}}" in compact


def expected_display_formula_count(
    first_qa_row: dict[str, Any], previous_row: dict[str, Any], expected_labels: list[str]
) -> int:
    candidates = [
        retry.parse_int(first_qa_row.get("detected_formula_blocks")),
        len([value for value in str(first_qa_row.get("formula_hashes") or "").split(";") if value]),
        len([value for value in str(previous_row.get("first_pass_formula_hashes") or "").split(";") if value]),
        len([value for value in str(previous_row.get("retry_formula_hashes") or "").split(";") if value]),
        len(expected_labels),
    ]
    previous_markdown = Path(str(previous_row.get("retry_markdown_path") or ""))
    if previous_markdown.exists():
        try:
            candidates.append(len(extract_display_formulas(previous_markdown.read_text(encoding="utf-8"))))
        except OSError:
            pass
    return max(candidates, default=0)


def validate_page_markdown(
    markdown: str, expected_count: int, expected_labels: list[str]
) -> dict[str, Any]:
    formulas = extract_display_formulas(markdown)
    reasons: list[str] = []
    warnings: list[str] = []
    if not markdown.strip():
        reasons.append("empty_vision_markdown")
    minimum_count = 1 if expected_labels else expected_count
    if len(formulas) < minimum_count:
        reasons.append(f"display_formula_count_below_expected={len(formulas)}/{minimum_count}")
    structure = markdown_structure_issues(markdown)
    reasons.extend(structure)
    missing_labels = [label for label in expected_labels if not label_is_present(markdown, label)]
    if missing_labels:
        reasons.append("missing_equation_labels=" + "|".join(missing_labels))
    hashes = [
        hashlib.sha256(ocr.normalized_formula_content(formula).encode("utf-8")).hexdigest()
        for formula in formulas
        if formula.strip()
    ]
    severity = "fail" if reasons else "warning" if warnings else "pass"
    resolution = "unresolved" if reasons else "review_required" if warnings else "resolved"
    return {
        "qa_severity": severity,
        "resolution_status": resolution,
        "qa_reasons": ";".join(reasons + warnings),
        "display_formula_count": len(formulas),
        "expected_display_formula_count": expected_count,
        "formula_hashes": ";".join(hashes),
        "expected_equation_labels": expected_labels,
        "missing_equation_labels": missing_labels,
        "structure_issues": structure,
    }


def build_prompt(expected_count: int, expected_labels: list[str]) -> str:
    labels = ", ".join(expected_labels) if expected_labels else "none pre-identified"
    return f"""Transcribe this single page of a PhD-level mathematics paper into faithful Markdown.

Precision rules:
- Reproduce the entire visible page in reading order. Do not summarize, solve, normalize, or paraphrase.
- Preserve every symbol, operator, accent, delimiter, superscript, subscript, index, limit, integral, matrix, and case distinction.
- Write inline mathematics as $...$ and every standalone/display equation as $$...$$.
- Preserve each visible equation number inside its display as \\tag{{number}}.
- Preserve headings, definitions, propositions, list numbering, citations, and prose.
- Produce structurally valid LaTeX with balanced braces, environments, and \\left/\\right delimiters.
- Do not include commentary, confidence notes, HTML, or Markdown code fences.

QA hints from earlier OCR (verify against the image rather than trusting them):
- Expected display equations: {expected_count}
- Equation labels requiring recovery: {labels}

Return only this envelope:
{BEGIN_MARKER}
<faithful page Markdown>
{END_MARKER}"""


def image_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def call_vision_api(
    *,
    image_path: Path,
    prompt: str,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: float,
    max_tokens: int,
    post_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Call official chat completions with image_url; return only provider JSON."""
    post_fn = post_fn or requests.post
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "thinking": {"type": "enabled"},
        "do_sample": False,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_error = "vision request failed"
    for attempt in range(retries + 1):
        try:
            response = post_fn(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout_seconds,
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            try:
                body = response.json()
            except (TypeError, ValueError, json.JSONDecodeError):
                body = {"response_text": redact_secret(getattr(response, "text", ""), api_key)}
            if 200 <= status_code < 300 and isinstance(body, dict):
                return redact_secret_value(body, api_key)
            last_error = f"HTTP {status_code}: {redact_secret(json.dumps(body, ensure_ascii=False), api_key)}"
            if status_code not in TRANSIENT_HTTP_STATUSES:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {redact_secret(exc, api_key)}"
        if attempt < retries:
            time.sleep(retry_sleep_seconds)
    raise VisionRepairError(last_error)


def copy_or_render_page(
    *,
    source_pdf: Path,
    destination: Path,
    page_number: int,
    dpi: int,
    previous_row: dict[str, Any],
    render_fn: Callable[[Path, Path, int, int], bool] | None = None,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Immutable vision attempt already exists: {destination}")
    previous_image = Path(str(previous_row.get("rendered_image_path") or ""))
    expected_previous_hash = str(previous_row.get("rendered_image_sha256") or "")
    if previous_image.is_file():
        actual_previous_hash = ocr.sha256_file(previous_image)
        if expected_previous_hash and actual_previous_hash != expected_previous_hash:
            raise VisionRepairError("Previous retry image SHA-256 changed; refusing vision repair.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous_image, destination)
        return
    render_fn = render_fn or ocr.render_page_image
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not render_fn(source_pdf, destination, page_number, dpi):
        raise VisionRepairError("Page image rendering failed.")


def compatible_failure_row(
    *,
    document: ocr.DocumentRecord,
    queue_row: dict[str, Any],
    previous_row: dict[str, Any],
    attempt_number: int,
    dpi: int,
    model: str,
    resolution_status: str,
    error_type: str,
    error: str,
    started_at: str,
) -> dict[str, Any]:
    first_raw = Path(str(previous_row.get("first_pass_raw_response_path") or queue_row.get("first_pass_raw_response_path") or ""))
    first_hash = ocr.sha256_file(first_raw) if first_raw.is_file() else ""
    return {
        "document_id": document.document_id,
        "input_path": document.input_path,
        "page_number": retry.parse_int(queue_row.get("page_number")),
        "attempt_id": f"vision-attempt-{attempt_number:02d}",
        "attempt_number": attempt_number,
        "dpi": dpi,
        "model": model,
        "source_pdf_sha256": ocr.sha256_file(Path(document.input_path)) if Path(document.input_path).is_file() else "",
        "expected_source_sha256": document.input_sha256,
        "first_pass_raw_response_path": str(first_raw),
        "first_pass_raw_sha256": first_hash,
        "first_pass_artifact_unchanged": bool(first_hash),
        "transport_status": "blocked" if resolution_status != "transport_error" else "error",
        "retry_qa_severity": "fail",
        "original_qa_reasons": str(previous_row.get("original_qa_reasons") or queue_row.get("qa_reasons") or ""),
        "retry_qa_reasons": error_type,
        "first_pass_formula_hashes": str(previous_row.get("first_pass_formula_hashes") or ""),
        "retry_formula_hashes": "",
        "resolution_status": resolution_status,
        "started_at_utc": started_at,
        "completed_at_utc": ocr.utc_now(),
        "error_type": error_type,
        "error": error,
    }


def process_vision_page(
    *,
    run_dir: Path,
    queue_row: dict[str, Any],
    document: ocr.DocumentRecord,
    previous_row: dict[str, Any],
    first_qa_row: dict[str, Any],
    attempt_number: int,
    dpi: int,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: float,
    max_tokens: int,
    max_image_mb: float,
    post_fn: Callable[..., Any] | None = None,
    render_fn: Callable[[Path, Path, int, int], bool] | None = None,
) -> dict[str, Any]:
    page_number = retry.parse_int(queue_row.get("page_number"))
    attempt_id = f"vision-attempt-{attempt_number:02d}"
    paths = vision_artifact_paths(run_dir, document.document_id, page_number, attempt_number)
    started_at = ocr.utc_now()
    first_raw = Path(str(previous_row.get("first_pass_raw_response_path") or queue_row.get("first_pass_raw_response_path") or ""))
    first_hash_before = ocr.sha256_file(first_raw) if first_raw.is_file() else ""
    previous_retry_raw = Path(str(previous_row.get("retry_raw_response_path") or ""))
    previous_retry_markdown = Path(str(previous_row.get("retry_markdown_path") or ""))
    previous_hashes = {
        "raw": ocr.sha256_file(previous_retry_raw) if previous_retry_raw.is_file() else "",
        "markdown": ocr.sha256_file(previous_retry_markdown) if previous_retry_markdown.is_file() else "",
    }
    expected_labels = reason_equation_labels(first_qa_row, previous_row, queue_row)
    expected_count = expected_display_formula_count(first_qa_row, previous_row, expected_labels)

    try:
        copy_or_render_page(
            source_pdf=Path(document.input_path),
            destination=paths["image"],
            page_number=page_number,
            dpi=dpi,
            previous_row=previous_row,
            render_fn=render_fn,
        )
        if paths["image"].stat().st_size > int(max_image_mb * 1024 * 1024):
            raise VisionRepairError(
                f"Rendered image is {paths['image'].stat().st_size / 1024 / 1024:.2f}MB; limit is {max_image_mb:.2f}MB."
            )
        response = call_vision_api(
            image_path=paths["image"],
            prompt=build_prompt(expected_count, expected_labels),
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
            max_tokens=max_tokens,
            post_fn=post_fn,
        )
        write_json_exclusive(paths["raw"], response)
        markdown = parse_page_markdown(extract_message_content(response))
        if not markdown:
            raise VisionRepairError("GLM-5V-Turbo response did not contain page Markdown.")
        write_text_exclusive(paths["markdown"], markdown.rstrip() + "\n")
        validation = validate_page_markdown(markdown, expected_count, expected_labels)
        first_hash_after = ocr.sha256_file(first_raw) if first_raw.is_file() else ""
        prior_retry_unchanged = (
            (not previous_hashes["raw"] or (previous_retry_raw.is_file() and ocr.sha256_file(previous_retry_raw) == previous_hashes["raw"]))
            and (
                not previous_hashes["markdown"]
                or (previous_retry_markdown.is_file() and ocr.sha256_file(previous_retry_markdown) == previous_hashes["markdown"])
            )
        )
        first_unchanged = bool(first_hash_before) and first_hash_before == first_hash_after
        resolution = str(validation["resolution_status"])
        if not first_unchanged or not prior_retry_unchanged:
            resolution = "artifact_integrity_failure"
            validation["qa_severity"] = "fail"
            validation["qa_reasons"] = ";".join(
                part for part in [str(validation.get("qa_reasons") or ""), "prior_artifact_changed"] if part
            )
        usage = response.get("usage") or {}
        row = {
            "document_id": document.document_id,
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
            "retry_raw_response_path": str(paths["raw"].resolve()),
            "retry_raw_sha256": ocr.sha256_file(paths["raw"]),
            "retry_markdown_path": str(paths["markdown"].resolve()),
            "retry_markdown_sha256": ocr.sha256_file(paths["markdown"]),
            "transport_status": "ok",
            "retry_qa_severity": validation["qa_severity"],
            "original_qa_reasons": str(previous_row.get("original_qa_reasons") or queue_row.get("qa_reasons") or ""),
            "retry_qa_reasons": validation["qa_reasons"],
            "first_pass_formula_hashes": str(previous_row.get("first_pass_formula_hashes") or first_qa_row.get("formula_hashes") or ""),
            "retry_formula_hashes": validation["formula_hashes"],
            "resolution_status": resolution,
            "request_id": str(response.get("request_id") or response.get("id") or ""),
            "total_tokens": usage.get("total_tokens", "") if isinstance(usage, dict) else "",
            "started_at_utc": started_at,
            "completed_at_utc": ocr.utc_now(),
            "error_type": "",
            "error": "",
        }
        write_json_exclusive(
            paths["result"],
            {**row, "validation": validation, "previous_retry_artifacts_unchanged": prior_retry_unchanged},
        )
        return row
    except Exception as exc:
        error = redact_secret(exc, api_key)
        error_type = type(exc).__name__
        if not paths["error"].exists():
            write_json_exclusive(
                paths["error"],
                {
                    "document_id": document.document_id,
                    "page_number": page_number,
                    "attempt_id": attempt_id,
                    "error_type": error_type,
                    "error": error,
                    "completed_at_utc": ocr.utc_now(),
                },
            )
        row = compatible_failure_row(
            document=document,
            queue_row=queue_row,
            previous_row=previous_row,
            attempt_number=attempt_number,
            dpi=dpi,
            model=model,
            resolution_status="transport_error",
            error_type=error_type,
            error=error,
            started_at=started_at,
        )
        if paths["image"].exists():
            row["rendered_image_path"] = str(paths["image"].resolve())
            row["rendered_image_sha256"] = ocr.sha256_file(paths["image"])
        if paths["raw"].exists():
            row["retry_raw_response_path"] = str(paths["raw"].resolve())
            row["retry_raw_sha256"] = ocr.sha256_file(paths["raw"])
        if paths["markdown"].exists():
            row["retry_markdown_path"] = str(paths["markdown"].resolve())
            row["retry_markdown_sha256"] = ocr.sha256_file(paths["markdown"])
        if not paths["result"].exists():
            write_json_exclusive(paths["result"], row)
        return row


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair unresolved formula pages with GLM-5V-Turbo without overwriting earlier OCR artifacts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--provider-max-image-mb", type=float, default=10.0)
    parser.add_argument("--force-new-attempt", action="store_true")
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help="Re-run deterministic QA on the latest vision Markdown without another API request.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.dpi < 150:
        raise SystemExit("--dpi must be >= 150 for formula vision repair.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    ledger_path = run_dir / "formula_retry_results.csv"
    if not ledger_path.exists():
        raise SystemExit(f"Retry ledger not found: {ledger_path}")
    existing_ledger: list[dict[str, Any]] = retry.read_csv(ledger_path)
    latest = retry.latest_rows_by_page(existing_ledger)
    queue_rows = retry.dedupe_queue_rows(retry.read_csv(run_dir / "formula_retry_queue.csv"))
    queue_by_page = {
        (str(row.get("document_id") or ""), retry.parse_int(row.get("page_number"))): row for row in queue_rows
    }
    documents = retry.documents_from_plan(retry.read_csv(run_dir / "ocr_plan.csv"))
    first_qa = retry.page_qa_lookup(retry.read_csv(run_dir / "page_formula_qa.csv"))

    jobs: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    for key, previous_row in latest.items():
        if str(previous_row.get("resolution_status") or "") not in REPAIRABLE_STATUSES:
            continue
        document_id, page_number = key
        document = documents.get(document_id)
        queue_row = queue_by_page.get(key, {"document_id": document_id, "page_number": page_number})
        attempt_number = retry.next_attempt_number(existing_ledger + additions, document_id, page_number)
        if document is None:
            continue
        is_vision_attempt = str(previous_row.get("attempt_id") or "").startswith("vision-attempt-")
        if is_vision_attempt and args.revalidate_existing:
            markdown_path = Path(str(previous_row.get("retry_markdown_path") or ""))
            if markdown_path.exists():
                expected_labels = reason_equation_labels(
                    first_qa.get(key, {}), previous_row, queue_row
                )
                expected_count = expected_display_formula_count(
                    first_qa.get(key, {}), previous_row, expected_labels
                )
                markdown = markdown_path.read_text(encoding="utf-8")
                validation = validate_page_markdown(markdown, expected_count, expected_labels)
                row = dict(previous_row)
                row.update(
                    {
                        "attempt_id": f"vision-revalidation-{attempt_number:02d}",
                        "attempt_number": attempt_number,
                        "transport_status": "cached",
                        "retry_qa_severity": validation["qa_severity"],
                        "retry_qa_reasons": validation["qa_reasons"],
                        "retry_formula_hashes": validation["formula_hashes"],
                        "resolution_status": validation["resolution_status"],
                        "started_at_utc": ocr.utc_now(),
                        "completed_at_utc": ocr.utc_now(),
                        "error_type": "",
                        "error": "",
                    }
                )
                additions.append(row)
                result_path = (
                    run_dir
                    / "formula_vision_repairs"
                    / "revalidation_results"
                    / document_id
                    / f"p{page_number:04d}"
                    / f"vision-revalidation-{attempt_number:02d}.json"
                )
                write_json_exclusive(result_path, {**row, "validation": validation})
                continue
        if is_vision_attempt and not args.force_new_attempt:
            continue
        source_pdf = Path(document.input_path)
        if not source_pdf.exists() or not document.input_sha256 or ocr.sha256_file(source_pdf) != document.input_sha256:
            additions.append(
                compatible_failure_row(
                    document=document,
                    queue_row=queue_row,
                    previous_row=previous_row,
                    attempt_number=attempt_number,
                    dpi=args.dpi,
                    model=args.model,
                    resolution_status="source_changed" if source_pdf.exists() else "source_missing",
                    error_type="source_integrity_failure",
                    error="Source PDF is missing or differs from ocr_plan.csv.",
                    started_at=ocr.utc_now(),
                )
            )
            continue
        jobs.append(
            {
                "run_dir": run_dir,
                "queue_row": queue_row,
                "document": document,
                "previous_row": previous_row,
                "first_qa_row": first_qa.get(key, {}),
                "attempt_number": attempt_number,
                "dpi": args.dpi,
                "endpoint": args.endpoint,
                "model": args.model,
                "timeout_seconds": args.timeout_seconds,
                "retries": args.retries,
                "retry_sleep_seconds": args.retry_sleep_seconds,
                "max_tokens": args.max_tokens,
                "max_image_mb": args.provider_max_image_mb,
            }
        )

    if jobs:
        api_key = ocr.env_api_key()
        if not api_key:
            raise SystemExit("Missing API key. Export ZAI_API_KEY or Z_AI_API_KEY before live vision repairs.")
        for job in jobs:
            job["api_key"] = api_key
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_vision_page, **job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                additions.append(row)
                print(
                    f"[{index}/{len(futures)}] {row['document_id']} page {row['page_number']} -> {row['resolution_status']}",
                    flush=True,
                )

    ledger = retry.merge_ledger(existing_ledger, additions)
    ocr.write_csv(
        ledger_path,
        [{field: row.get(field, "") for field in retry.LEDGER_FIELDS} for row in ledger],
        retry.LEDGER_FIELDS,
    )
    assembly_rows = retry.assemble_verified_documents(run_dir, documents, queue_rows, ledger)
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
    for row in retry.latest_rows_by_page(ledger).values():
        status = str(row.get("resolution_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    ocr.write_json(
        run_dir / "formula_vision_repair_manifest.json",
        {
            "generated_at_utc": ocr.utc_now(),
            "endpoint": args.endpoint,
            "model": args.model,
            "dpi": args.dpi,
            "workers": args.workers,
            "candidate_statuses": sorted(REPAIRABLE_STATUSES),
            "new_attempts": len(jobs),
            "resolution_counts": counts,
            "artifacts": {
                "retry_ledger": str(ledger_path.resolve()),
                "vision_repair_root": str((run_dir / "formula_vision_repairs").resolve()),
                "verified_book_manifest": str((run_dir / "book_manifest_verified.csv").resolve()),
            },
        },
    )
    print(f"Formula vision repair phase complete: {run_dir}")
    print(f"Retry ledger: {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
