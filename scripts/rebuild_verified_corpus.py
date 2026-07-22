#!/usr/bin/env python3
"""Rebuild verified books from immutable OCR/retry artifacts without API use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_retries as retry  # noqa: E402
import run_zai_ocr_pipeline as ocr  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    documents = retry.documents_from_plan(retry.read_csv(run_dir / "ocr_plan.csv"))
    queue_rows = retry.dedupe_queue_rows(retry.read_csv(run_dir / "formula_retry_queue.csv"))
    ledger = retry.read_csv(run_dir / "formula_retry_results.csv")
    assembly = retry.assemble_verified_documents(run_dir, documents, queue_rows, ledger)
    fields = [
        "document_id", "status", "verified_markdown_path", "page_count",
        "pages_materialized", "retry_pages_used", "unresolved_pages", "warning_pages",
    ]
    ocr.write_csv(
        run_dir / "book_manifest_verified.csv",
        [{field: row.get(field, "") for field in fields} for row in assembly],
        fields,
    )
    counts: dict[str, int] = {}
    for row in assembly:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(f"Rebuilt {len(assembly)} verified books: {counts}")
    print(f"Geometry tag QA: {run_dir / 'equation_tag_geometry_qa.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
