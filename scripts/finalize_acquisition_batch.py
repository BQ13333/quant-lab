#!/usr/bin/env python3
"""Validate an acquisition batch and emit the OCR queue and audit report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader


QUEUE_FIELDS = [
    "queue_position", "candidate_id", "lane", "primary_specialist",
    "secondary_specialists", "title", "pdf_path", "pdf_page_count",
    "pdf_sha256", "conversion_status", "ocr_profile", "formula_policy",
    "source_retention_policy",
]

LICENSE_FIELDS = [
    "candidate_id", "title", "license", "landing_url", "pdf_url",
    "license_review_status", "promotion_rule",
]

STOPWORDS = {
    "a", "an", "and", "as", "based", "by", "for", "from", "in", "of",
    "on", "the", "their", "to", "via", "with",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def title_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def normalized_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--corpus-index", default="PhD_Math_Agent_Library/indexes/corpus_index.csv")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    corpus_index = (workspace / args.corpus_index).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) % 6:
        raise SystemExit(f"Expected a nonempty six-per-lane acquisition batch, found {len(rows)} rows")

    with corpus_index.open(newline="", encoding="utf-8-sig") as handle:
        corpus_rows = list(csv.DictReader(handle))
    corpus_hashes = {row["source_pdf_sha256"] for row in corpus_rows if row.get("source_pdf_sha256")}
    corpus_titles = {normalized_title(row["title"]): row["document_id"] for row in corpus_rows}

    lane_counts: Counter[str] = Counter()
    lane_pages: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    peer_reviewed = 0
    cc_by = 0
    total_pages = 0
    total_bytes = 0
    seen_hashes: dict[str, str] = {}
    queue: list[dict[str, str]] = []
    license_review: list[dict[str, str]] = []
    errors: list[str] = []

    for position, row in enumerate(rows, 1):
        cid = row["candidate_id"]
        if row.get("selection_status") != "accepted":
            errors.append(f"{cid}: not accepted")
        if row.get("download_status") != "downloaded_verified_pdf":
            errors.append(f"{cid}: download status is {row.get('download_status')}")
            continue
        path = workspace / row["pdf_path"]
        if not path.is_file():
            errors.append(f"{cid}: PDF missing at {path}")
            continue
        digest = sha256_file(path)
        if digest != row["pdf_sha256"]:
            errors.append(f"{cid}: SHA-256 mismatch")
        if digest in seen_hashes:
            errors.append(f"{cid}: duplicates new candidate {seen_hashes[digest]}")
        seen_hashes[digest] = cid
        if digest in corpus_hashes:
            errors.append(f"{cid}: PDF hash already exists in active corpus")
        title_key = normalized_title(row["title"])
        if title_key in corpus_titles:
            errors.append(f"{cid}: title already exists as {corpus_titles[title_key]}")

        reader = PdfReader(str(path), strict=False)
        pages = len(reader.pages)
        if pages != int(row["pdf_page_count"]):
            errors.append(f"{cid}: page count changed from {row['pdf_page_count']} to {pages}")
        first_text = " ".join((page.extract_text() or "") for page in reader.pages[:2]).lower()
        tokens = title_tokens(row["title"])
        matched = sum(token in first_text for token in tokens)
        coverage = matched / max(len(tokens), 1)
        if coverage < 0.60:
            errors.append(f"{cid}: weak title/PDF match ({coverage:.0%})")

        lane = row["lane"]
        lane_counts[lane] += 1
        lane_pages[lane] += pages
        primary_counts[row["primary_specialist"]] += 1
        total_pages += pages
        total_bytes += path.stat().st_size
        peer_reviewed += "peer-reviewed" in row["peer_review_status"]
        if row["license"].startswith("CC BY"):
            cc_by += 1
        else:
            license_review.append(
                {
                    "candidate_id": cid,
                    "title": row["title"],
                    "license": row["license"],
                    "landing_url": row["landing_url"],
                    "pdf_url": row["pdf_url"],
                    "license_review_status": "pending_reuse_terms_verification",
                    "promotion_rule": "May be converted for internal review; do not promote to a reusable commercial training corpus until terms are confirmed.",
                }
            )
        queue.append(
            {
                "queue_position": str(position),
                "candidate_id": cid,
                "lane": lane,
                "primary_specialist": row["primary_specialist"],
                "secondary_specialists": row["secondary_specialists"],
                "title": row["title"],
                "pdf_path": row["pdf_path"],
                "pdf_page_count": str(pages),
                "pdf_sha256": digest,
                "conversion_status": "pending_300dpi_formula_ocr",
                "ocr_profile": "GLM precision; render 300 DPI; preserve display/inline LaTeX; retain page boundaries",
                "formula_policy": "No omitted formulas; retry low-confidence pages; GLM-5V repair; formula/citation audit before promotion",
                "source_retention_policy": "Retain PDF until Markdown, formula, citation, and page-completeness QA all pass",
            }
        )

    for lane, count in lane_counts.items():
        if count != 6:
            errors.append(f"{lane}: expected 6 papers, found {count}")
    expected_lanes = len(rows) // 6
    if len(lane_counts) != expected_lanes:
        errors.append(f"Expected {expected_lanes} lanes, found {len(lane_counts)}")
    if errors:
        raise SystemExit("Acquisition validation failed:\n- " + "\n- ".join(errors))

    queue_path = manifest.parent / "conversion_queue.csv"
    license_path = manifest.parent / "license_review_queue.csv"
    report_path = manifest.parent / "ACQUISITION_REPORT.md"
    write_csv(queue_path, QUEUE_FIELDS, queue)
    write_csv(license_path, LICENSE_FIELDS, license_review)

    lane_lines = [
        f"| {lane} | {lane_counts[lane]} | {lane_pages[lane]} |"
        for lane in sorted(lane_counts)
    ]
    primary_lines = [
        f"| {specialist} | {primary_counts[specialist]} |"
        for specialist in sorted(primary_counts)
    ]
    report = f"""# Acquisition Report

Status: **ready for precision Markdown conversion**

## Integrity result

- Verified open-access PDFs: **{len(rows)}**
- Total pages: **{total_pages:,}**
- Total PDF size: **{total_bytes / (1024 * 1024):.1f} MiB**
- Existing-corpus title collisions: **0**
- Existing-corpus SHA-256 collisions: **0**
- First-two-page title correspondence checks passed: **{len(rows)}/{len(rows)}**
- Peer-reviewed papers: **{peer_reviewed}**
- Explicit CC BY papers: **{cc_by}**
- Papers requiring reuse-terms verification before commercial-corpus promotion: **{len(license_review)}**

## Lane coverage

| Acquisition lane | PDFs | Pages |
|---|---:|---:|
{chr(10).join(lane_lines)}

## Primary specialist routing

| Primary specialist | Papers |
|---|---:|
{chr(10).join(primary_lines)}

## Conversion gate

Every queued PDF uses the 300-DPI precision profile. Low-confidence formula pages must be retried and sent through GLM-5V repair. PDFs remain in staging until page-completeness, formula, citation, and Markdown audits all pass.

Files:

- `candidate_manifest.csv`: scholarly metadata, provenance, hashes, pages, and specialist mappings.
- `conversion_queue.csv`: production OCR order and precision policy.
- `license_review_queue.csv`: promotion blockers where reuse terms are not explicitly CC BY.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Validated {len(rows)} PDFs / {total_pages} pages; wrote {queue_path.name}, {license_path.name}, and {report_path.name}")


if __name__ == "__main__":
    main()
