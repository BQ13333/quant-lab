#!/usr/bin/env python3
"""Delete only Wave 3/4/5 PDFs backed by clean, current verified Markdown audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Deliberately limited to the newly verified sources. Wave 1/2 and all legacy
# or user-owned PDFs are outside this deletion script's authorized scope.
WAVES = (
    ("wave3", "ACQ-WAVE3-DEPTH-20260712", 42),
    ("wave4", "ACQ-WAVE4-SYNTHESIS-20260712", 36),
    ("wave5", "ACQ-WAVE5-HMM-REGIME-20260715", 18),
)
EXPECTED_CORPUS_ROWS = 447
EXPECTED_ACTIVE_DOCUMENTS = 446
EXPECTED_PAGES = 12_053
EXPECTED_SECONDARY_ROUTES = 946
PAGE_RE = re.compile(r"<!--\s*OCR_PAGE\s+(\d+)\s+source=[^>]+-->")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FIELDS = [
    "artifact_kind", "wave", "batch", "candidate_id", "document_id", "pdf_path",
    "pdf_sha256", "pdf_bytes", "source_page_count", "replacement_markdown_path",
    "replacement_markdown_sha256", "replacement_page_markers", "replacement_image_references",
    "verification_status", "preflight_status", "deletion_status", "deleted_at_utc",
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


def check_audits(run_dir: Path) -> None:
    acceptance = json.loads((run_dir / "acceptance_audit.json").read_text(encoding="utf-8"))
    localized = json.loads((run_dir / "localized_acceptance_audit.json").read_text(encoding="utf-8"))
    unexpected_warnings = [
        warning for warning in acceptance.get("warnings", [])
        if not str(warning).startswith("equation-tag review lane retained:")
    ]
    formula_retries_clean = (
        acceptance.get("resolved_formula_retry_pages") == acceptance.get("formula_retry_pages")
        and not acceptance.get("tag_body_integrity_failures")
    )
    if (
        acceptance.get("status") != "pass"
        or acceptance.get("error_count")
        or unexpected_warnings
        or not formula_retries_clean
    ):
        raise RuntimeError(f"Corpus audit is not clean: {run_dir}")
    if localized.get("status") != "pass" or localized.get("errors"):
        raise RuntimeError(f"Localized audit is not clean: {run_dir}")


def require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes authorized root {root}: {path}") from exc


def check_full_library_audit(library: Path, audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("errors") or audit.get("warnings"):
        raise RuntimeError(f"Full library audit is not clean: {audit_path}")
    actual = (
        audit.get("corpus_rows"),
        audit.get("active_documents"),
        audit.get("page_markers"),
        audit.get("secondary_routes"),
    )
    expected = (
        EXPECTED_CORPUS_ROWS,
        EXPECTED_ACTIVE_DOCUMENTS,
        EXPECTED_PAGES,
        EXPECTED_SECONDARY_ROUTES,
    )
    if actual != expected:
        raise RuntimeError(f"Full library audit totals mismatch: actual={actual}, expected={expected}")
    corpus_path = library / "indexes/corpus_index.csv"
    if audit.get("corpus_index_sha256") != sha256_file(corpus_path):
        raise RuntimeError(f"Full library audit is stale for the current corpus index: {audit_path}")
    corpus = read_csv(corpus_path)
    if len(corpus) != EXPECTED_CORPUS_ROWS:
        raise RuntimeError(f"Current corpus index no longer matches the clean audit: {len(corpus)} rows")
    return audit


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--full-library-audit",
        default="PhD_Math_Agent_Library/indexes/FULL_LIBRARY_AUDIT_20260715.json",
        help="Clean post-integration full-library audit required before any source is eligible.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually unlink validated PDFs.")
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    library = workspace / "PhD_Math_Agent_Library"
    ledger_dir = library / "deletion_ledgers"
    ledger_path = ledger_dir / "PDF-DELETION-VERIFIED-WAVES-3-5-20260715.csv"
    summary_path = ledger_dir / "PDF-DELETION-VERIFIED-WAVES-3-5-20260715.json"
    rows: list[dict[str, Any]] = []
    cleanup_roots: set[Path] = set()
    audit_arg = Path(args.full_library_audit).expanduser()
    audit_path = audit_arg.resolve() if audit_arg.is_absolute() else (workspace / audit_arg).resolve()
    check_full_library_audit(library, audit_path)
    corpus_by_id = {row["document_id"]: row for row in read_csv(library / "indexes/corpus_index.csv")}

    for wave_slug, batch_name, expected_documents in WAVES:
        batch = library / "acquisition_staging" / batch_name
        candidate_by_id = {
            row["candidate_id"]: row
            for row in read_csv(batch / "candidate_manifest.csv")
            if row.get("selection_status", "accepted") == "accepted"
        }
        verified_rows = read_csv(batch / "verified_markdown_manifest.csv")
        if len(verified_rows) != expected_documents:
            raise RuntimeError(
                f"Expected {expected_documents} verified rows in {batch_name}, found {len(verified_rows)}"
            )
        if len(candidate_by_id) != expected_documents or {row["candidate_id"] for row in verified_rows} != set(candidate_by_id):
            raise RuntimeError(f"Candidate/verified manifest membership mismatch: {batch_name}")
        run_dirs = {Path(row["localized_markdown_path"]).resolve().parent.parent for row in verified_rows}
        if len(run_dirs) != 1:
            raise RuntimeError(f"Wave maps to multiple OCR runs: {batch_name}")
        run_dir = next(iter(run_dirs))
        check_audits(run_dir)
        authorized_pdf_root = (batch / "pdfs").resolve()
        authorized_chunk_root = (run_dir / "chunks_pdf").resolve()
        authorized_markdown_root = (run_dir / "markdown_verified_localized").resolve()
        cleanup_roots.add(authorized_pdf_root)
        cleanup_roots.add(authorized_chunk_root)

        for item in verified_rows:
            pdf = Path(item["pdf_path"]).resolve()
            markdown = Path(item["localized_markdown_path"]).resolve()
            require_within(pdf, authorized_pdf_root, "Source PDF")
            require_within(markdown, authorized_markdown_root, "Replacement Markdown")
            candidate = candidate_by_id[item["candidate_id"]]
            candidate_pdf = (workspace / candidate["pdf_path"]).resolve() if not Path(candidate["pdf_path"]).is_absolute() else Path(candidate["pdf_path"]).resolve()
            if pdf != candidate_pdf:
                raise RuntimeError(f"Candidate/verified PDF path mismatch: {item['candidate_id']}")
            if (
                item["pdf_sha256"] != candidate["pdf_sha256"]
                or item["pdf_page_count"] != candidate["pdf_page_count"]
            ):
                raise RuntimeError(f"Candidate/verified PDF evidence mismatch: {item['candidate_id']}")
            if item["verification_status"] != "verified_localized_markdown":
                raise RuntimeError(f"Unverified replacement for {item['candidate_id']}")
            if not markdown.is_file() or sha256_file(markdown) != item["localized_markdown_sha256"]:
                raise RuntimeError(f"Replacement Markdown hash mismatch: {markdown}")
            text = markdown.read_text(encoding="utf-8")
            markers = len(PAGE_RE.findall(text))
            images = len(IMAGE_RE.findall(text))
            if markers != int(item["page_markers"]) or markers != int(item["pdf_page_count"]):
                raise RuntimeError(f"Replacement page-count mismatch: {item['candidate_id']}")
            if images != int(item["localized_image_references"]):
                raise RuntimeError(f"Replacement image-count mismatch: {item['candidate_id']}")
            routed_id = f"{wave_slug}-{item['ocr_document_id']}"
            corpus_row = corpus_by_id.get(routed_id)
            if corpus_row is None:
                raise RuntimeError(f"Verified replacement is not integrated into the corpus: {routed_id}")
            routed_markdown = Path(corpus_row["markdown_path"]).resolve()
            if (
                not routed_markdown.is_file()
                or sha256_file(routed_markdown) != corpus_row["markdown_sha256"]
                or corpus_row["source_pdf_sha256"] != item["pdf_sha256"]
                or corpus_row["page_count"] != item["pdf_page_count"]
            ):
                raise RuntimeError(f"Integrated corpus evidence mismatch: {routed_id}")
            if pdf.is_file():
                actual_sha = sha256_file(pdf)
                if actual_sha != item["pdf_sha256"]:
                    raise RuntimeError(f"Source PDF hash mismatch: {pdf}")
                size = pdf.stat().st_size
                preflight = "validated_replacement_and_source_hash"
                status = "validated_pending_delete"
            else:
                actual_sha = item["pdf_sha256"]
                size = 0
                preflight = "replacement_valid_source_already_absent"
                status = "already_absent"
            rows.append(
                {
                    "artifact_kind": "source_pdf", "wave": wave_slug, "batch": batch_name,
                    "candidate_id": item["candidate_id"], "document_id": item["ocr_document_id"],
                    "pdf_path": str(pdf), "pdf_sha256": actual_sha, "pdf_bytes": size,
                    "source_page_count": item["pdf_page_count"],
                    "replacement_markdown_path": str(markdown),
                    "replacement_markdown_sha256": item["localized_markdown_sha256"],
                    "replacement_page_markers": markers, "replacement_image_references": images,
                    "verification_status": item["verification_status"], "preflight_status": preflight,
                    "deletion_status": status, "deleted_at_utc": "",
                }
            )

        for chunk in sorted((run_dir / "chunks_pdf").rglob("*.pdf")):
            require_within(chunk.resolve(), authorized_chunk_root, "OCR transport chunk")
            rows.append(
                {
                    "artifact_kind": "ocr_transport_chunk_pdf", "wave": wave_slug, "batch": batch_name,
                    "candidate_id": "", "document_id": chunk.parent.name, "pdf_path": str(chunk.resolve()),
                    "pdf_sha256": sha256_file(chunk), "pdf_bytes": chunk.stat().st_size,
                    "source_page_count": "", "replacement_markdown_path": str(run_dir / "markdown_verified_localized"),
                    "replacement_markdown_sha256": "", "replacement_page_markers": "",
                    "replacement_image_references": "", "verification_status": "run_audits_pass",
                    "preflight_status": "validated_clean_run_transport_artifact",
                    "deletion_status": "validated_pending_delete", "deleted_at_utc": "",
                }
            )

    # Persist the pre-deletion evidence before any unlink operation.
    write_csv(ledger_path, rows)
    if args.apply:
        deleted_at = datetime.now(timezone.utc).isoformat()
        for row in rows:
            if row["deletion_status"] != "validated_pending_delete":
                continue
            path = Path(row["pdf_path"])
            if not path.is_file():
                raise RuntimeError(f"Validated PDF disappeared before deletion: {path}")
            if sha256_file(path) != row["pdf_sha256"]:
                raise RuntimeError(f"Validated PDF changed before deletion: {path}")
            path.unlink()
            if path.exists():
                raise RuntimeError(f"PDF still exists after deletion: {path}")
            row["deletion_status"] = "deleted"
            row["deleted_at_utc"] = deleted_at
        for root in cleanup_roots:
            prune_empty_dirs(root)
        write_csv(ledger_path, rows)

    source_rows = [row for row in rows if row["artifact_kind"] == "source_pdf"]
    chunk_rows = [row for row in rows if row["artifact_kind"] == "ocr_transport_chunk_pdf"]
    summary = {
        "status": "applied" if args.apply else "validated_dry_run",
        "target_waves": [wave_slug for wave_slug, _, _ in WAVES],
        "wave_source_pdfs": len(source_rows),
        "wave_source_pages": sum(int(row["source_page_count"]) for row in source_rows),
        "wave_source_bytes": sum(int(row["pdf_bytes"]) for row in source_rows),
        "transport_chunk_pdfs": len(chunk_rows),
        "transport_chunk_bytes": sum(int(row["pdf_bytes"]) for row in chunk_rows),
        "deleted_source_pdfs": sum(row["deletion_status"] == "deleted" for row in source_rows),
        "deleted_transport_chunks": sum(row["deletion_status"] == "deleted" for row in chunk_rows),
        "remaining_validated_source_pdfs": sum(Path(row["pdf_path"]).exists() for row in source_rows),
        "remaining_validated_transport_chunks": sum(Path(row["pdf_path"]).exists() for row in chunk_rows),
        "ledger": str(ledger_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
