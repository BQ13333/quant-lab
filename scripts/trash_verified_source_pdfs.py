#!/usr/bin/env python3
"""Move only fully verified production source PDFs to a recoverable Trash folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


FIELDS = [
    "document_id", "input_path", "source_root", "file_name", "input_sha256",
    "size_bytes", "page_count", "actual_sha256", "actual_page_count",
    "verified_markdown_path", "verified_markdown_sha256", "book_status",
    "pages_materialized", "preflight_result", "action", "destination_path",
    "executed_at_utc", "post_action_source_exists", "post_action_destination_exists",
    "post_action_destination_sha256", "result", "error",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, str] = {}
    if not lines or lines[0].strip() != "---":
        return result
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def update_queue_status(queue_path: Path, plan_by_path: dict[Path, dict[str, str]]) -> None:
    rows = read_csv(queue_path)
    fields = list(rows[0])
    for row in rows:
        path = (queue_path.parents[1] / row["pdf_path"]).resolve()
        plan = plan_by_path[path]
        row["sha256"] = plan["input_sha256"]
        row["conversion_status"] = "verified_markdown_source_pdf_trashed"
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--library-dir", default="PhD_Math_Agent_Library")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    library = (workspace / args.library_dir).resolve()
    audit = json.loads((run_dir / "verified_corpus_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("verified_books") != 279 or audit.get("verified_page_markers") != 7128:
        raise SystemExit("Production acceptance audit is not a complete 279-document/7,128-page PASS")

    plan = read_csv(run_dir / "ocr_plan.csv")
    books = {row["document_id"]: row for row in read_csv(run_dir / "book_manifest_verified.csv")}
    library_rows = {
        row["document_id"]: row for row in read_csv(library / "indexes/corpus_index.csv")
    }
    if len(plan) != 279 or len(books) != 279 or len(library_rows) != 279:
        raise SystemExit("Plan, verified-book manifest, and categorized library must each contain 279 documents")

    allowed_roots = [
        (workspace / "PhD_Math_Mega_Corpus/papers").resolve(),
        (workspace / "Quant_Math_PhD_Papers").resolve(),
    ]
    plan_by_path = {Path(row["input_path"]).resolve(): row for row in plan}
    if len(plan_by_path) != 279:
        raise SystemExit("Production plan contains duplicate source paths")
    for source in plan_by_path:
        if not any(source.is_relative_to(root) for root in allowed_roots):
            raise SystemExit(f"Source is outside the deletion allowlist: {source}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trash_root = Path.home() / ".Trash" / f"08_Quantitative_Verified_Source_PDFs_{timestamp}"
    if trash_root.exists():
        raise SystemExit(f"Trash destination already exists: {trash_root}")

    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for source, plan_row in sorted(plan_by_path.items(), key=lambda item: item[1]["document_id"]):
        document_id = plan_row["document_id"]
        book = books[document_id]
        markdown = Path(book["verified_markdown_path"]).resolve()
        front = parse_frontmatter(markdown)
        if not source.is_file() or not markdown.is_file():
            raise SystemExit(f"Source or verified Markdown missing before move: {document_id}")
        actual_sha = sha256_file(source)
        actual_pages = len(PdfReader(str(source), strict=False).pages)
        expected_pages = int(plan_row["page_count"])
        expected_sha = plan_row["input_sha256"]
        if actual_sha != expected_sha or actual_pages != expected_pages:
            raise SystemExit(f"Source hash/page mismatch before move: {document_id}")
        if (
            front.get("document_id") != document_id
            or front.get("source_path") != str(source)
            or front.get("sha256") != expected_sha
            or int(front.get("page_count", "-1")) != expected_pages
            or int(book["pages_materialized"]) != expected_pages
            or book["status"] not in {"verified", "verified_with_equation_tag_warnings"}
            or library_rows[document_id]["source_pdf_sha256"] != expected_sha
        ):
            raise SystemExit(f"Verified Markdown/library provenance mismatch: {document_id}")
        relative = source.relative_to(workspace)
        destination = trash_root / relative
        if destination.exists():
            raise SystemExit(f"Trash collision: {destination}")
        size = source.stat().st_size
        total_bytes += size
        rows.append(
            {
                "document_id": document_id,
                "input_path": str(source),
                "source_root": relative.parts[0],
                "file_name": source.name,
                "input_sha256": expected_sha,
                "size_bytes": size,
                "page_count": expected_pages,
                "actual_sha256": actual_sha,
                "actual_page_count": actual_pages,
                "verified_markdown_path": str(markdown),
                "verified_markdown_sha256": sha256_file(markdown),
                "book_status": book["status"],
                "pages_materialized": book["pages_materialized"],
                "preflight_result": "verified",
                "action": "move_to_macos_trash",
                "destination_path": str(destination),
                "result": "pending",
                "error": "",
            }
        )

    manifest = library / "source_pdf_deletion_manifest.csv"
    write_csv(manifest, rows)
    moved: list[tuple[Path, Path]] = []
    try:
        for row in rows:
            source = Path(row["input_path"])
            destination = Path(row["destination_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
            destination_sha = sha256_file(destination)
            if destination_sha != row["input_sha256"]:
                raise RuntimeError(f"Destination hash mismatch: {row['document_id']}")
            row.update(
                {
                    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "post_action_source_exists": source.exists(),
                    "post_action_destination_exists": destination.exists(),
                    "post_action_destination_sha256": destination_sha,
                    "result": "moved_verified",
                }
            )
            write_csv(manifest, rows)
    except Exception as exc:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        for row in rows:
            row["result"] = "rolled_back"
            row["error"] = f"{type(exc).__name__}: {exc}"
        write_csv(manifest, rows)
        raise

    for root in allowed_roots:
        if root.exists():
            for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            try:
                root.rmdir()
            except OSError:
                pass

    update_queue_status(workspace / "PhD_Math_Mega_Corpus/markdown_conversion_queue.csv", plan_by_path)
    receipt = library / "PDF_DELETION_RECEIPT.md"
    receipt.write_text(
        "# Verified Source PDF Removal Receipt\n\n"
        f"- Executed (UTC): {timestamp}\n"
        f"- Verified source PDFs moved from workspace: {len(rows)}\n"
        f"- Verified pages represented in Markdown: {sum(int(row['page_count']) for row in rows)}\n"
        f"- Source bytes moved: {total_bytes:,}\n"
        f"- Recoverable Trash location: `{trash_root}`\n"
        f"- Deletion manifest: `source_pdf_deletion_manifest.csv`\n\n"
        "Only the exact 279 allowlisted production source PDFs were moved. Other PDFs and generated OCR artifacts were not touched. "
        "The original source paths and SHA-256 values remain in the manifests and Markdown front matter for provenance.\n",
        encoding="utf-8",
    )
    print(f"Moved {len(rows)} verified source PDFs ({total_bytes} bytes) to {trash_root}")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
