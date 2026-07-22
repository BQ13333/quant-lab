#!/usr/bin/env python3
"""Append visually pre-identified high-risk pages to a formula retry queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TIER1 = {
    "wave3": {
        "w3-gen-04-stochastic-interpolants-a-unifying-framework-for-flows-and-diffusions": [5, 11, 19, 36, 64, 65, 68, 71],
        "w3-nop-02-on-universal-approximation-and-error-bounds-for-fourier-neural-operators": [8, 28, 40, 41, 42, 47, 52, 53, 54, 55],
        "w3-gen-03-flow-matching-for-generative-modeling": [4, 5, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28],
        "w3-gen-01-deep-unsupervised-learning-using-nonequilibrium-thermodynamics": [4, 12, 13, 14, 15, 16],
        "w3-gen-02-denoising-diffusion-probabilistic-models": [2, 3, 4, 5, 13, 14, 17, 19, 20, 22, 23, 24, 25],
        "w3-seq-05-corralling-a-band-of-bandit-algorithms": [7, 10, 11, 19, 20, 21, 24, 25, 26],
        "w3-cau-02-estimation-and-inference-of-heterogeneous-treatment-effects-using-random-forests": [21, 29, 34, 35, 42, 46],
        "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi": [4, 12, 13],
    },
    "wave4": {
        "w4-26-data-complexity-estimates-for-operator-learning": [21, 22, 25, 26, 31, 33, 36, 37, 38, 39],
        "w4-30-convergence-of-a-deep-bsde-solver-with-jumps": [9, 10, 14, 15, 16, 17, 22, 24, 25, 26, 30],
        "w4-15-generalized-variational-inference-three-arguments-for-deriving-new-posteriors": [57, 71, 72, 73, 74, 82],
        "w4-33-numerical-solution-of-dynamic-portfolio-optimization-with-transaction-costs": [9, 11, 18, 25, 28, 32, 33, 51, 54],
    },
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wave", choices=sorted(TIER1), required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    queue_path = run_dir / "formula_retry_queue.csv"
    queue = read(queue_path)
    fields = list(queue[0])
    page_qa = {(row["document_id"], int(row["page_number"])): row for row in read(run_dir / "page_formula_qa.csv")}
    plan = {row["document_id"]: row for row in read(run_dir / "ocr_plan.csv")}
    ocr_rows = read(run_dir / "ocr_manifest.csv")
    existing = {(row["document_id"], int(row["page_number"])) for row in queue}
    added: list[dict[str, object]] = []
    for document_id, pages in TIER1[args.wave].items():
        if document_id not in plan:
            raise RuntimeError(f"Tier-1 document not in OCR plan: {document_id}")
        for page in pages:
            key = (document_id, page)
            if key in existing:
                continue
            qa = page_qa[key]
            chunk = next(
                row for row in ocr_rows
                if row["document_id"] == document_id
                and int(row["start_page"]) <= page <= int(row["end_page"])
                and row["status"] == "ok"
            )
            row = {
                "document_id": document_id,
                "input_path": plan[document_id]["input_path"],
                "page_number": page,
                "qa_reasons": "preidentified_tier1_formula_figure_review",
                "recommended_dpi": 300,
                "source_math_risk_score": qa["math_risk_score"],
                "first_pass_formula_blocks": qa["formula_blocks"],
                "first_pass_raw_response_path": chunk["raw_response_path"],
                "status": "pending_single_page_reocr",
            }
            queue.append(row)
            added.append({"document_id": document_id, "page_number": page})
            existing.add(key)
    queue.sort(key=lambda row: (row["document_id"], int(row["page_number"])))
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in queue)
    report = {
        "status": "pass",
        "wave": args.wave,
        "tier1_registry_pages": sum(len(pages) for pages in TIER1[args.wave].values()),
        "already_queued": sum(len(pages) for pages in TIER1[args.wave].values()) - len(added),
        "newly_queued": len(added),
        "final_queue_rows": len(queue),
        "added": added,
    }
    output = run_dir / "tier1_precision_queue_augmentation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
