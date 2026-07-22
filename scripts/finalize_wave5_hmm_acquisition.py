#!/usr/bin/env python3
"""Validate Wave 5 HMM downloads and create the precision OCR queue.

Unlike earlier fixed six-per-lane waves, Wave 5 is a deliberately uneven depth
basket. Only rows with ``selection_status=accepted`` enter OCR. Watchlist rows
remain in the scholarly manifest and research benchmark but are not downloaded
or promoted without a later explicit admission decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


QUEUE_FIELDS = [
    "queue_position", "candidate_id", "admission_class", "gap_tags", "lane",
    "primary_specialist", "secondary_specialists", "title", "pdf_path",
    "pdf_page_count", "pdf_sha256", "conversion_status", "ocr_profile",
    "formula_policy", "source_retention_policy",
]
LICENSE_FIELDS = [
    "candidate_id", "title", "license", "landing_url", "pdf_url",
    "license_review_status", "promotion_rule",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_title(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def title_tokens(value: str) -> list[str]:
    stop = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "using", "with"}
    return [token for token in normalized_title(value).split() if len(token) >= 4 and token not in stop]


def split_agents(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,|]", value or "") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--corpus-index", default="PhD_Math_Agent_Library/indexes/corpus_index.csv")
    parser.add_argument("--roster", default="PhD_Math_Agent_Library/indexes/agent_roster.csv")
    parser.add_argument("--expected-accepted", type=int, default=18)
    args = parser.parse_args()

    workspace = Path(args.workspace_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    if not manifest.is_absolute():
        manifest = (workspace / manifest).resolve()
    corpus_path = Path(args.corpus_index)
    if not corpus_path.is_absolute():
        corpus_path = (workspace / corpus_path).resolve()
    roster_path = Path(args.roster)
    if not roster_path.is_absolute():
        roster_path = (workspace / roster_path).resolve()

    fields, rows = read_csv(manifest)
    required = {
        "candidate_id", "admission_class", "gap_tags", "lane", "title", "landing_url",
        "pdf_url", "license", "peer_review_status", "primary_specialist",
        "secondary_specialists", "selection_status", "download_status", "pdf_path",
        "pdf_sha256", "pdf_page_count",
    }
    if not required.issubset(fields):
        raise SystemExit(f"Manifest missing fields: {sorted(required - set(fields))}")
    _, corpus = read_csv(corpus_path)
    _, roster = read_csv(roster_path)
    valid_agents = {row["agent_id"] for row in roster}
    corpus_titles = {normalized_title(row.get("title", "")): row.get("document_id", "") for row in corpus}
    corpus_hashes = {row.get("source_pdf_sha256", "") for row in corpus if row.get("source_pdf_sha256")}

    accepted = [row for row in rows if row.get("selection_status") == "accepted"]
    if len(accepted) != args.expected_accepted:
        raise SystemExit(f"Expected {args.expected_accepted} accepted papers, found {len(accepted)}")
    if any(row.get("admission_class") not in {"core", "red_team"} for row in accepted):
        raise SystemExit("Accepted papers must have admission_class core or red_team")

    errors: list[str] = []
    queue: list[dict[str, str]] = []
    license_review: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    lane_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    peer_reviewed = 0
    total_pages = 0
    total_bytes = 0

    for position, row in enumerate(accepted, 1):
        cid = row["candidate_id"]
        primary = row["primary_specialist"]
        secondaries = split_agents(row.get("secondary_specialists", ""))
        if primary not in valid_agents:
            errors.append(f"{cid}: unknown primary specialist {primary}")
        for secondary in secondaries:
            if secondary not in valid_agents:
                errors.append(f"{cid}: unknown secondary specialist {secondary}")
        if row.get("download_status") != "downloaded_verified_pdf":
            errors.append(f"{cid}: download status is {row.get('download_status')}")
            continue
        pdf_path = Path(row["pdf_path"])
        if not pdf_path.is_absolute():
            pdf_path = (workspace / pdf_path).resolve()
        if not pdf_path.is_file():
            errors.append(f"{cid}: missing PDF {pdf_path}")
            continue
        digest = sha256_file(pdf_path)
        if digest != row["pdf_sha256"]:
            errors.append(f"{cid}: SHA-256 mismatch")
        if digest in seen_hashes:
            errors.append(f"{cid}: duplicates candidate {seen_hashes[digest]}")
        seen_hashes[digest] = cid
        if digest in corpus_hashes:
            errors.append(f"{cid}: PDF hash already active as an existing corpus paper")
        title_key = normalized_title(row["title"])
        if title_key in corpus_titles:
            errors.append(f"{cid}: title already active as {corpus_titles[title_key]}")

        reader = PdfReader(str(pdf_path), strict=False)
        pages = len(reader.pages)
        if pages != int(row["pdf_page_count"]):
            errors.append(f"{cid}: page count changed from {row['pdf_page_count']} to {pages}")
        first_text = " ".join((page.extract_text() or "") for page in reader.pages[:2]).lower()
        tokens = title_tokens(row["title"])
        title_coverage = sum(token in first_text for token in tokens) / max(1, len(tokens))
        if title_coverage < 0.55:
            errors.append(f"{cid}: weak title/PDF correspondence ({title_coverage:.0%})")

        lane_counts[row["lane"]] += 1
        class_counts[row["admission_class"]] += 1
        primary_counts[primary] += 1
        total_pages += pages
        total_bytes += pdf_path.stat().st_size
        if row.get("peer_review_status") in {"published_peer_reviewed", "accepted_peer_reviewed"}:
            peer_reviewed += 1
        license_value = row.get("license", "").strip()
        if not license_value.lower().startswith(("cc by", "creative commons attribution")):
            license_review.append(
                {
                    "candidate_id": cid,
                    "title": row["title"],
                    "license": license_value,
                    "landing_url": row["landing_url"],
                    "pdf_url": row["pdf_url"],
                    "license_review_status": "pending_reuse_terms_verification",
                    "promotion_rule": (
                        "May be converted for internal research review; do not use for external distribution "
                        "or commercial model training until source-specific terms are cleared."
                    ),
                }
            )
        queue.append(
            {
                "queue_position": str(position),
                "candidate_id": cid,
                "admission_class": row["admission_class"],
                "gap_tags": row["gap_tags"],
                "lane": row["lane"],
                "primary_specialist": primary,
                "secondary_specialists": row["secondary_specialists"],
                "title": row["title"],
                "pdf_path": str(pdf_path.relative_to(workspace)),
                "pdf_page_count": str(pages),
                "pdf_sha256": digest,
                "conversion_status": "pending_300dpi_formula_ocr",
                "ocr_profile": "GLM-OCR math-precision; 300 DPI; page-addressable; retain layout artifacts",
                "formula_policy": "No omitted formulas; retry flagged pages with GLM-OCR and repair unresolved pages with GLM-5V-Turbo",
                "source_retention_policy": "Retain PDF until page, formula, figure, citation, and localized-Markdown audits pass",
            }
        )

    if errors:
        raise SystemExit("Wave 5 acquisition validation failed:\n- " + "\n- ".join(errors))

    write_csv(manifest.parent / "conversion_queue.csv", QUEUE_FIELDS, queue)
    write_csv(manifest.parent / "license_review_queue.csv", LICENSE_FIELDS, license_review)
    watchlist_count = sum(row.get("selection_status") == "watchlist" for row in rows)
    report = [
        "# Wave 5 HMM and Regime-Inference Acquisition Report",
        "",
        "Status: **ready for precision Markdown conversion**",
        "",
        f"- Accepted PDFs: **{len(accepted)}**",
        f"- Core papers: **{class_counts['core']}**",
        f"- Red-team papers: **{class_counts['red_team']}**",
        f"- Watchlist-only papers: **{watchlist_count}**",
        f"- Total accepted pages: **{total_pages:,}**",
        f"- Total accepted PDF size: **{total_bytes / 1024 / 1024:.1f} MiB**",
        f"- Peer-reviewed accepted papers: **{peer_reviewed}**",
        f"- Reuse-terms review items: **{len(license_review)}**",
        "- Existing-corpus title collisions: **0**",
        "- Existing-corpus PDF-hash collisions: **0**",
        "",
        "## Lane distribution",
        "",
        "| Lane | Papers |",
        "|---|---:|",
    ]
    report.extend(f"| {lane} | {count} |" for lane, count in sorted(lane_counts.items()))
    report.extend(["", "## Primary specialist routing", "", "| Specialist | Papers |", "|---|---:|"])
    report.extend(f"| {agent} | {count} |" for agent, count in sorted(primary_counts.items()))
    report.extend(
        [
            "",
            "Watchlist papers remain outside the OCR queue. Source PDFs must not be deleted until all technical",
            "audits pass and the deletion ledger records their SHA-256 hashes and original paths.",
            "",
        ]
    )
    (manifest.parent / "ACQUISITION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Validated {len(accepted)} PDFs / {total_pages} pages; OCR queue contains {len(queue)} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
