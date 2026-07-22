#!/usr/bin/env python3
"""Append visually confirmed equation-tag review pages to the formula repair queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_retries as retry  # noqa: E402
import run_zai_ocr_pipeline as ocr  # noqa: E402


REVIEWS = {
    ("w5-09-online-em-algorithm-for-hidden-markov-models", 4): ["(4)", "(5)", "(6)", "(7)"],
    ("w5-11-variational-sequential-monte-carlo", 5): ["(9)"],
    ("p05-time-uniform-nonparametric-nonasymptotic-confidence-sequences", 41): ["(129)", "(130)"],
    ("p06-pac-bayesian-inequalities-for-martingales", 7): ["(28)", "(29)", "(30)"],
    ("ts02-locally-stationary-functional-time-series", 53): ["(3)", "(4)"],
    ("hf05-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process", 4): ["(3b)", "(4b)"],
    ("n01-efficient-error-certification-for-physics-informed-neural-networks", 17): ["(12)", "(13)"],
    ("n05-adaptive-checkpoint-adjoint-method-for-gradient-estimation-in-neural-ode", 5): ["(21)", "(22)", "(25)", "(26)"],
    ("n06-approximation-rates-for-deep-calibration-of-rough-stochastic-volatility-models", 4): ["(1)", "(2)"],
    ("n06-approximation-rates-for-deep-calibration-of-rough-stochastic-volatility-models", 13): ["(2.37)", "(2.38)"],
    ("c05-pareto-front-evolution-for-solving-multi-objective-portfolio-optimization", 5): ["(12)", "(14)", "(15)", "(16)"],
    ("d06-constrained-offline-policy-optimization", 3): ["(1)", "(2)", "(3)", "(4)"],
    ("e04-nonparametric-risk-bounds-for-time-series-forecasting", 34): ["(12)", "(13)"],
    ("f04-jax-lob-a-gpu-accelerated-limit-order-book-simulator-to-unlock-large-scale-reinforcement-lea", 6): ["(6)", "(7)"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    queue_path = run_dir / "formula_retry_queue.csv"
    ledger_path = run_dir / "formula_retry_results.csv"
    queue = retry.read_csv(queue_path)
    ledger = retry.read_csv(ledger_path)
    documents = retry.documents_from_plan(retry.read_csv(run_dir / "ocr_plan.csv"))
    geometry = retry.read_csv(run_dir / "equation_tag_geometry_qa.csv")
    raw_by_page: dict[tuple[str, int], str] = {}
    for row in geometry:
        key = (str(row.get("document_id") or ""), retry.parse_int(row.get("page_number")))
        if row.get("raw_response_path"):
            raw_by_page.setdefault(key, str(row["raw_response_path"]))
    queue_keys = {(str(row.get("document_id")), retry.parse_int(row.get("page_number"))) for row in queue}
    ledger_keys = {(str(row.get("document_id")), retry.parse_int(row.get("page_number"))) for row in ledger}
    added_queue = 0
    added_ledger = 0
    for key, labels in REVIEWS.items():
        document = documents.get(key[0])
        if document is None:
            # The review registry is shared by multiple acquisition waves.
            # Entries from another run are intentionally ignored.
            continue
        raw = raw_by_page.get(key, "")
        if not raw or not Path(raw).is_file():
            raise RuntimeError(f"First-pass raw response missing for {key}")
        reasons = "missing_equation_labels=" + "|".join(labels)
        if key not in queue_keys:
            queue.append(
                {
                    "document_id": key[0], "input_path": document.input_path,
                    "page_number": key[1], "qa_reasons": reasons, "recommended_dpi": 300,
                    "source_math_risk_score": "geometry_review", "first_pass_formula_blocks": "",
                    "first_pass_raw_response_path": raw, "status": "equation_tag_review",
                }
            )
            added_queue += 1
        if key not in ledger_keys:
            ledger.append(
                {
                    "document_id": key[0], "input_path": document.input_path,
                    "page_number": key[1], "attempt_id": "equation-tag-geometry-review",
                    "attempt_number": 0, "dpi": 300, "model": "equation-tag-geometry",
                    "source_pdf_sha256": document.input_sha256,
                    "expected_source_sha256": document.input_sha256,
                    "first_pass_raw_response_path": raw,
                    "first_pass_raw_sha256": ocr.sha256_file(Path(raw)),
                    "first_pass_artifact_unchanged": True,
                    "transport_status": "cached", "retry_qa_severity": "warning",
                    "original_qa_reasons": reasons, "retry_qa_reasons": reasons,
                    "resolution_status": "review_required", "started_at_utc": ocr.utc_now(),
                    "completed_at_utc": ocr.utc_now(), "error_type": "", "error": "",
                }
            )
            added_ledger += 1
    queue_fields = [
        "document_id", "input_path", "page_number", "qa_reasons", "recommended_dpi",
        "source_math_risk_score", "first_pass_formula_blocks", "first_pass_raw_response_path", "status",
    ]
    ocr.write_csv(queue_path, [{field: row.get(field, "") for field in queue_fields} for row in queue], queue_fields)
    ocr.write_csv(ledger_path, [{field: row.get(field, "") for field in retry.LEDGER_FIELDS} for row in ledger], retry.LEDGER_FIELDS)
    print(f"Added {added_queue} equation-tag queue rows and {added_ledger} preflight ledger rows")


if __name__ == "__main__":
    main()
