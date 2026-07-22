#!/usr/bin/env python3
"""Join acquisition metadata to verified/localized Markdown artifacts for specialist routing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path


FIELDS = [
    "candidate_id", "lane", "title", "authors", "year", "primary_specialist",
    "secondary_specialists", "tier", "license", "commercial_promotion_status",
    "license_status", "commercial_reuse_status", "warning_pages",
    "pdf_path", "pdf_sha256", "pdf_page_count", "ocr_document_id",
    "verified_markdown_path", "localized_markdown_path", "localized_markdown_sha256",
    "page_markers", "localized_image_references", "formula_repair_pages", "verification_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    candidate_path = Path(args.candidate_manifest).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    plan = read_csv(run_dir / "ocr_plan.csv")
    plan_by_path = {Path(row["input_path"]).resolve(): row for row in plan}
    books = {row["document_id"]: row for row in read_csv(run_dir / "book_manifest_verified.csv")}
    ledger = read_csv(run_dir / "formula_retry_results.csv")
    latest: dict[tuple[str, int], dict[str, str]] = {}
    for row in ledger:
        key = (row["document_id"], int(row["page_number"]))
        if key not in latest or int(row.get("attempt_number") or 0) > int(latest[key].get("attempt_number") or 0):
            latest[key] = row
    repair_counts = Counter(doc for (doc, _), row in latest.items() if row.get("resolution_status") == "resolved")
    rows: list[dict[str, str]] = []
    candidates = [
        candidate for candidate in read_csv(candidate_path)
        if candidate.get("selection_status", "accepted") == "accepted"
    ]
    if len(candidates) != len(plan):
        raise RuntimeError(
            f"Accepted candidate/OCR plan cardinality mismatch: {len(candidates)} != {len(plan)}"
        )
    for candidate in candidates:
        pdf = Path(candidate["pdf_path"])
        if not pdf.is_absolute():
            pdf = (workspace / pdf).resolve()
        plan_row = plan_by_path.get(pdf)
        if not plan_row:
            raise RuntimeError(f"No OCR plan row for {pdf}")
        document_id = plan_row["document_id"]
        book = books.get(document_id)
        if not book or book["status"] not in {"verified", "verified_with_equation_tag_warnings"}:
            raise RuntimeError(f"No verified book for {document_id}")
        localized = run_dir / "markdown_verified_localized" / f"{document_id}.md"
        if not localized.is_file():
            raise RuntimeError(f"Localized Markdown missing: {localized}")
        text = localized.read_text(encoding="utf-8")
        rows.append(
            {
                "candidate_id": candidate["candidate_id"], "lane": candidate["lane"],
                "title": candidate["title"], "authors": candidate["authors"], "year": candidate["year"],
                "primary_specialist": candidate["primary_specialist"],
                "secondary_specialists": candidate["secondary_specialists"], "tier": candidate["tier"],
                "license": candidate["license"],
                "commercial_promotion_status": candidate.get("commercial_reuse_status") or "review_required",
                "license_status": candidate.get("license_status") or "unknown",
                "commercial_reuse_status": candidate.get("commercial_reuse_status") or "review_required",
                "warning_pages": book.get("warning_pages", ""),
                "pdf_path": str(pdf), "pdf_sha256": candidate["pdf_sha256"],
                "pdf_page_count": candidate["pdf_page_count"], "ocr_document_id": document_id,
                "verified_markdown_path": book["verified_markdown_path"],
                "localized_markdown_path": str(localized), "localized_markdown_sha256": sha256(localized),
                "page_markers": str(len(re.findall(r"<!-- OCR_PAGE \d+ ", text))),
                "localized_image_references": str(len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))),
                "formula_repair_pages": str(repair_counts[document_id]),
                "verification_status": "verified_localized_markdown",
            }
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} specialist-routed verified Markdown rows to {output}")


if __name__ == "__main__":
    main()
