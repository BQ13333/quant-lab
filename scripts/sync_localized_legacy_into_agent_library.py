#!/usr/bin/env python3
"""Route the localized legacy OCR corpus into the canonical agent library."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import integrate_verified_waves_into_agent_library as integration  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    localized = run_dir / "markdown_verified_localized"
    audit = json.loads((run_dir / "localized_markdown_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("errors"):
        raise RuntimeError("Localized legacy audit is not clean")

    library = workspace / "PhD_Math_Agent_Library"
    corpus_path = library / "indexes/corpus_index.csv"
    corpus_rows = integration.read_csv(corpus_path)
    fields = list(corpus_rows[0])
    localized_ids = {path.stem for path in localized.glob("*.md")}
    active_rows = [
        row for row in corpus_rows
        if row["document_id"] in localized_ids and row["retrieval_status"].startswith("active")
    ]
    if len(active_rows) != 278:
        raise RuntimeError(f"Expected 278 active legacy rows, found {len(active_rows)}")

    updated: dict[str, dict[str, str]] = {}
    manifest_rows: list[dict[str, object]] = []
    total_images = 0
    total_pages = 0
    for row in active_rows:
        document_id = row["document_id"]
        source = localized / f"{document_id}.md"
        target = Path(row["markdown_path"]).resolve()
        if not source.is_file() or not target.parent.is_dir():
            raise RuntimeError(f"Missing source or target package for {document_id}")
        pages, images = integration.route_markdown(source, target, f"legacy-{document_id}")
        if pages != int(row["page_count"]):
            raise RuntimeError(f"Page count changed for {document_id}: {pages} != {row['page_count']}")
        routed = dict(row)
        routed["markdown_sha256"] = integration.sha256_file(target)
        routed["verification_status"] = "verified_localized_markdown"
        updated[document_id] = routed
        total_images += images
        total_pages += pages
        manifest_rows.append(
            {
                "document_id": document_id,
                "primary_specialist": row["primary_specialist"],
                "page_count": pages,
                "image_references": images,
                "source_markdown": str(source),
                "routed_markdown": str(target),
                "routed_markdown_sha256": routed["markdown_sha256"],
            }
        )

    merged = [updated.get(row["document_id"], row) for row in corpus_rows]
    write_csv(corpus_path, fields, merged)
    for index_path in sorted((library / "specialists").glob("[0-9][0-9]_*/primary_index.csv")) + sorted(
        (library / "specialists").glob("[0-9][0-9]_*/secondary_index.csv")
    ):
        rows = integration.read_csv(index_path)
        routed_rows = [updated.get(row["document_id"], row) for row in rows]
        write_csv(index_path, list(rows[0]), routed_rows)

    report = {
        "status": "pass",
        "documents_routed": len(manifest_rows),
        "page_markers": total_pages,
        "image_references": total_images,
        "excluded_legacy_documents": sorted(localized_ids - set(updated)),
        "documents": manifest_rows,
    }
    output = library / "indexes/LEGACY_LOCALIZATION_SYNC.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Routed {len(manifest_rows)} active legacy documents: "
        f"{total_pages} pages, {total_images} localized image references"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
