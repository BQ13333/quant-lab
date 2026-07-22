#!/usr/bin/env python3
"""Publish the unresolved legacy equation-tag candidates as a page-level review queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REVIEW_STATUSES = {"ambiguous", "duplicate_label", "retry_page_unverified", "conflicting_tag"}
PRIORITY = {
    "conflicting_tag": 1,
    "retry_page_unverified": 1,
    "ambiguous": 2,
    "duplicate_label": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    with (run_dir / "equation_tag_geometry_qa.csv").open(newline="", encoding="utf-8-sig") as handle:
        candidates = [row for row in csv.DictReader(handle) if row["status"] in REVIEW_STATUSES]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["document_id"], int(row["page_number"]))].append(row)

    queue: list[dict[str, object]] = []
    for position, ((document_id, page), page_rows) in enumerate(sorted(grouped.items()), 1):
        statuses = sorted({row["status"] for row in page_rows}, key=lambda value: PRIORITY[value])
        labels = [row["label"] for row in page_rows]
        priority = min(PRIORITY[row["status"]] for row in page_rows)
        raw_paths = sorted({row["raw_response_path"] for row in page_rows if row["raw_response_path"]})
        queue.append(
            {
                "queue_position": position,
                "priority": priority,
                "document_id": document_id,
                "page_number": page,
                "candidate_count": len(page_rows),
                "labels": "|".join(labels),
                "statuses": "|".join(statuses),
                "element_indices": "|".join(row["element_index"] for row in page_rows),
                "source_bboxes": "|".join(row["source_bbox"] for row in page_rows),
                "formula_bboxes": "|".join(row["formula_bbox"] for row in page_rows),
                "raw_response_paths": "|".join(raw_paths),
                "localized_markdown_path": str(run_dir / "markdown_verified_localized" / f"{document_id}.md"),
                "review_status": "pending_source_page_adjudication",
                "decision_policy": (
                    "Inject only if the source page visibly assigns the candidate to the matched display; "
                    "otherwise record a false-positive acceptance. Never change an equation body during tag review."
                ),
            }
        )
    fields = list(queue[0])
    output = Path(args.output_csv).expanduser().resolve()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    status_counts = Counter(row["status"] for row in candidates)
    report = {
        "status": "review_queue_published",
        "documents": len({row["document_id"] for row in candidates}),
        "pages": len(queue),
        "candidates": len(candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "priority_page_counts": dict(sorted(Counter(str(row["priority"]) for row in queue).items())),
        "queue_path": str(output),
        "formula_body_mutation_allowed": False,
    }
    summary = Path(args.summary_json).expanduser().resolve()
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Published equation-tag review queue: {len(queue)} pages / {len(candidates)} candidates")


if __name__ == "__main__":
    main()
