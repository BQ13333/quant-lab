#!/usr/bin/env python3
"""Audit the complete canonical 20-agent Markdown library and local assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_vision_repairs as vision  # noqa: E402


PAGE_RE = re.compile(r"<!--\s*OCR_PAGE\s+(\d+)\s+source=[^>]+-->")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
DEFAULT_EXPECTED_CORPUS_ROWS = 447
DEFAULT_EXPECTED_ACTIVE_DOCUMENTS = 446
DEFAULT_EXPECTED_EXCLUDED_DOCUMENTS = 1
DEFAULT_EXPECTED_PAGES = 12_053
DEFAULT_EXPECTED_SECONDARY_ROUTES = 946


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--coverage-csv", required=True)
    parser.add_argument("--expected-corpus-rows", type=int, default=DEFAULT_EXPECTED_CORPUS_ROWS)
    parser.add_argument("--expected-active-documents", type=int, default=DEFAULT_EXPECTED_ACTIVE_DOCUMENTS)
    parser.add_argument("--expected-excluded-documents", type=int, default=DEFAULT_EXPECTED_EXCLUDED_DOCUMENTS)
    parser.add_argument("--expected-pages", type=int, default=DEFAULT_EXPECTED_PAGES)
    parser.add_argument("--expected-secondary-routes", type=int, default=DEFAULT_EXPECTED_SECONDARY_ROUTES)
    args = parser.parse_args()
    library = Path(args.library).expanduser().resolve()
    corpus = rows(library / "indexes/corpus_index.csv")
    active = [row for row in corpus if row["retrieval_status"].startswith("active")]
    excluded = [row for row in corpus if not row["retrieval_status"].startswith("active")]
    errors: list[str] = []
    warnings: list[str] = []
    document_results: list[dict[str, object]] = []
    seen_markdown_hashes: dict[str, str] = {}
    seen_source_hashes: dict[str, str] = {}
    per_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unique_images: set[Path] = set()
    total_images = total_pages = total_display = total_inline = total_chars = 0

    if (
        len(corpus) != args.expected_corpus_rows
        or len(active) != args.expected_active_documents
        or len(excluded) != args.expected_excluded_documents
    ):
        errors.append(
            "corpus cardinality mismatch: "
            f"all={len(corpus)}/{args.expected_corpus_rows} "
            f"active={len(active)}/{args.expected_active_documents} "
            f"excluded={len(excluded)}/{args.expected_excluded_documents}"
        )
    if len({row["document_id"] for row in corpus}) != len(corpus):
        errors.append("duplicate document_id in corpus index")

    for row in active:
        document_id = row["document_id"]
        path = Path(row["markdown_path"]).resolve()
        doc_errors: list[str] = []
        if not path.is_file():
            errors.append(f"missing Markdown: {document_id}")
            continue
        digest = sha256(path)
        if digest != row["markdown_sha256"]:
            doc_errors.append("markdown_hash_mismatch")
        if digest in seen_markdown_hashes:
            doc_errors.append(f"duplicate_markdown_hash:{seen_markdown_hashes[digest]}")
        seen_markdown_hashes[digest] = document_id
        source_hash = row["source_pdf_sha256"]
        if source_hash and source_hash in seen_source_hashes:
            doc_errors.append(f"duplicate_source_hash:{seen_source_hashes[source_hash]}")
        if source_hash:
            seen_source_hashes[source_hash] = document_id
        text = path.read_text(encoding="utf-8")
        page_numbers = [int(value) for value in PAGE_RE.findall(text)]
        expected_pages = int(row["page_count"])
        if len(page_numbers) != expected_pages or page_numbers != list(range(1, expected_pages + 1)):
            doc_errors.append("page_marker_sequence_mismatch")
        issues = vision.markdown_structure_issues(text)
        if issues:
            doc_errors.append("formula_structure:" + "|".join(issues))
        if "\ufffd" in text:
            doc_errors.append("unicode_replacement_character")
        image_count = 0
        for raw_target in IMAGE_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "data:")):
                doc_errors.append("nonlocal_image")
                continue
            image = (path.parent / target).resolve()
            if not image.is_file():
                doc_errors.append(f"broken_image:{target}")
                continue
            try:
                with Image.open(image) as handle:
                    handle.verify()
            except Exception as exc:  # noqa: BLE001
                doc_errors.append(f"invalid_image:{target}:{exc}")
                continue
            image_count += 1
            unique_images.add(image)
        displays = len(DISPLAY_RE.findall(text))
        inlines = len(INLINE_RE.findall(DISPLAY_RE.sub("", text)))
        total_pages += len(page_numbers)
        total_images += image_count
        total_display += displays
        total_inline += inlines
        total_chars += len(text)
        agent = row["primary_specialist"]
        per_agent[agent]["primary_papers"] += 1
        per_agent[agent]["pages"] += len(page_numbers)
        per_agent[agent]["image_references"] += image_count
        per_agent[agent]["display_formulas"] += displays
        per_agent[agent]["inline_math_spans"] += inlines
        if doc_errors:
            errors.extend(f"{document_id}:{error}" for error in doc_errors)
        document_results.append(
            {
                "document_id": document_id,
                "primary_specialist": agent,
                "pages": len(page_numbers),
                "image_references": image_count,
                "display_formulas": displays,
                "inline_math_spans": inlines,
                "characters": len(text),
                "status": "pass" if not doc_errors else "fail",
                "errors": doc_errors,
            }
        )

    roster = rows(library / "indexes/agent_roster.csv")
    if len(roster) != 20:
        errors.append(f"agent roster count {len(roster)} != 20")
    coverage: list[dict[str, object]] = []
    secondary_total = 0
    for row in roster:
        agent = row["agent_id"]
        folder = next(iter((library / "specialists").glob(f"[0-9][0-9]_{agent}")), None)
        if folder is None:
            errors.append(f"specialist folder missing: {agent}")
            continue
        primary = rows(folder / "primary_index.csv")
        secondary = rows(folder / "secondary_index.csv")
        active_primary = [item for item in primary if item["retrieval_status"].startswith("active")]
        active_secondary = [item for item in secondary if item["retrieval_status"].startswith("active")]
        if len(active_primary) != per_agent[agent]["primary_papers"]:
            errors.append(f"primary index count mismatch: {agent}")
        secondary_total += len(active_secondary)
        coverage.append(
            {
                "agent_id": agent,
                "title": row["title"],
                "primary_papers": len(active_primary),
                "secondary_routes": len(active_secondary),
                "pages": per_agent[agent]["pages"],
                "image_references": per_agent[agent]["image_references"],
                "display_formulas": per_agent[agent]["display_formulas"],
                "inline_math_spans": per_agent[agent]["inline_math_spans"],
            }
        )
    if total_pages != args.expected_pages:
        errors.append(f"page-marker total is {total_pages}, expected {args.expected_pages}")
    if secondary_total != args.expected_secondary_routes:
        errors.append(f"secondary route total is {secondary_total}, expected {args.expected_secondary_routes}")

    report = {
        "status": "pass" if not errors else "fail",
        "corpus_index_sha256": sha256(library / "indexes/corpus_index.csv"),
        "corpus_rows": len(corpus),
        "active_documents": len(active),
        "excluded_documents": len(excluded),
        "specialists": len(roster),
        "page_markers": total_pages,
        "characters": total_chars,
        "display_formula_blocks": total_display,
        "inline_math_spans": total_inline,
        "image_references": total_images,
        "unique_local_images": len(unique_images),
        "unique_local_image_bytes": sum(path.stat().st_size for path in unique_images),
        "secondary_routes": secondary_total,
        "expected": {
            "corpus_rows": args.expected_corpus_rows,
            "active_documents": args.expected_active_documents,
            "excluded_documents": args.expected_excluded_documents,
            "page_markers": args.expected_pages,
            "secondary_routes": args.expected_secondary_routes,
        },
        "errors": errors,
        "warnings": warnings,
        "coverage": coverage,
        "documents": document_results,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coverage_path = Path(args.coverage_csv).expanduser().resolve()
    fields = list(coverage[0])
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage)
    print(
        f"Full library audit {report['status'].upper()}: {len(active)} papers, "
        f"{total_pages} pages, {total_display} display formulas, {total_images} image references"
    )
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
