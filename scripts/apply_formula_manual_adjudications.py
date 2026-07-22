#!/usr/bin/env python3
"""Record source-exact visual adjudications for the final formula retry pages.

The input OCR and vision attempts remain immutable.  Corrected Markdown pages
live under ``formula_manual_repairs`` and this command appends a provenance row
to ``formula_retry_results.csv`` for each visually reviewed page, then rebuilds
the verified books.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_retries as retry  # noqa: E402
import run_zai_formula_vision_repairs as vision  # noqa: E402
import run_zai_ocr_pipeline as ocr  # noqa: E402


MANUAL_REPAIRS: dict[tuple[str, int], dict[str, Any]] = {
    ("w5-09-online-em-algorithm-for-hidden-markov-models", 4): {
        "base_attempt": 0,
        "notes": [
            "Visually confirmed all formula bodies against the source page",
            "Restored source equation labels (4), (5), (6), and (7); the unnumbered displays remain unnumbered",
            "Preserved the two-row recursion and attached labels (6) and (7) to their respective source equations",
        ],
    },
    ("w5-11-variational-sequential-monte-carlo", 5): {
        "base_attempt": 0,
        "notes": [
            "Visually confirmed equations (8), (9), (10), and (11) against the two-column source page",
            "Restored the omitted source tag (9) to the proposed reparameterization-gradient display",
            "Restored the source exponent delta=10^{-16} and the two-column continuation order around Algorithm 2",
        ],
    },
    ("w5-14-robust-estimation-for-ergodic-markovian-processes", 73): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed every source equation and all formula terms on page 73 against the 300-DPI render",
            "Promoted the GLM-5V repair, which corrects malformed delimiters and restores the source density measures",
            "Formatted the six terminal density definitions as their source mathematical display, yielding all seven visual formula groups",
        ],
    },
    ("w4-02-robust-pricing-and-hedging-via-neural-sdes", 26): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed all five source display groups and every formula term are present in vision attempt 02",
            "Split the two adjacent xi definitions into their two source display groups; the sixth automated candidate is an inline expansion-point tuple",
            "Preserved the source fraktur p notation, which vision attempt 03 changed incorrectly",
        ],
    },
    ("w4-30-convergence-of-a-deep-bsde-solver-with-jumps", 20): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed all seven source formula groups are complete in vision attempt 02",
            "Moved equation tag (4.3) to the end of its aligned display without changing its mathematics",
        ],
    },
    ("w4-30-convergence-of-a-deep-bsde-solver-with-jumps", 22): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed all seven source formula groups are complete in vision attempt 02",
            "Normalized source labels (4.8), (4.9), and (4.10) as trailing LaTeX tags without changing formula content",
        ],
    },
    ("w4-30-convergence-of-a-deep-bsde-solver-with-jumps", 25): {
        "base_attempt": 3,
        "notes": [
            "Visually confirmed no source formula omission in vision attempt 03",
            "Rejoined the continuation lines of equations (4.20) and (4.21) to their source displays",
            "Separated the seven source constant definitions into their individual display groups",
        ],
    },
    ("w4-30-convergence-of-a-deep-bsde-solver-with-jumps", 31): {
        "base_attempt": 3,
        "notes": [
            "Visually confirmed all source formula groups are present in vision attempt 03",
            "Restored double-tilde conditional targets and definitions while retaining single-tilde comparators",
            "Corrected the source coefficient K_{nu^epsilon} in equation (4.36)",
        ],
    },
    ("hf05-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process", 4): {
        "base_attempt": 1,
        "notes": [
            "Visually confirmed the paired source equations are labeled (3a)/(3b) and (4a)/(4b)",
            "Restored the omitted secondary tags (3b) and (4b) by using multi-row align environments; equation bodies are unchanged",
        ],
    },
    ("n05-adaptive-checkpoint-adjoint-method-for-gradient-estimation-in-neural-ode", 5): {
        "notes": [
            "Visually confirmed the naive-method recurrence rows carry equation labels (23), (24), (25), and (26)",
            "Converted the visible parenthesized row text to formal source-exact LaTeX tags without changing the recurrence bodies",
        ],
    },
    ("n06-approximation-rates-for-deep-calibration-of-rough-stochastic-volatility-models", 13): {
        "notes": [
            "Visually confirmed the source assigns (2.37) to the network-size construction and (2.38) to its terminal upper bound",
            "Restored the omitted (2.38) tag by expressing the source chain as a multi-row align environment; equation bodies are unchanged",
        ],
    },
    ("p04-concentration-inequalities-for-markov-chains-by-marton-couplings-and-spectral-methods", 30): {
        "notes": [
            "Visually confirmed every displayed source equation is present, including tagged equation (5.4)",
            "Normalized one nested scalable delimiter from a mismatched left/big pair to a balanced big/big pair without changing the source mathematics",
            "Accepted seven grouped display environments versus the heuristic expected count of eight after visual confirmation",
        ],
    },
    ("e04-nonparametric-risk-bounds-for-time-series-forecasting", 29): {
        "base_attempt": 3,
        "notes": [
            "Visually confirmed five grouped display environments contain all source equations",
            "Corrected the terminal j=i-1 asymptotic exponent from i to source-exact j",
        ],
    },
    ("97b7456a46d7-on-searching-for-generalized-instrumental-variables", 8): {
        "base_attempt": 4,
        "notes": ["Restored attempt-04 reading order", "Corrected terminal negative-literal exponent to \\bar{o}_i"],
    },
    ("2106-01357-diffusion-schrdinger-bridge-with-applications-to-score-based-generative-modeling", 31): {
        "notes": ["Corrected \\bar{\\alpha}_{N_-} to \\bar{\\alpha}_{N-\\cdot}"],
    },
    ("2106-00170-adaptive-conformal-inference-under-distribution-shift", 24): {
        "notes": ["Corrected \\alpha_{A_t^*} to \\alpha_{A_t}^{*}"],
    },
    ("2010-09107-conformal-prediction-for-time-series", 32): {
        "notes": ["Corrected empirical-CDF event symbol from \\hat{F} to \\tilde{F}"],
    },
    ("307a6342c8c9-diffusion-models-are-minimax-optimal-distribution-estimators", 23): {
        "notes": ["Corrected dyadic-scale index", "Restored underlined neural-network index", "Corrected final exponent to k(4l+1)"],
    },
    ("069933596532-score-approximation-estimation-and-distribution-recovery-of-diffusion-models-on-low", 16): {
        "notes": ["Applied the truncation indicator to (g-fbar), matching the source, in both bounds"],
    },
    ("430cb021ab71-tight-risk-bound-for-high-dimensional-time-series-completion", 17): {
        "notes": ["Restored fraktur m symbols", "Restored xi subscripts in variance and covariance terms"],
    },
    ("7c066af7c68a-robust-and-scalable-bayesian-online-changepoint-detection", 14): {
        "notes": ["Restored two gradient-of-t factors", "Restored proof-ending black square"],
    },
}


VISUAL_ACCEPTANCES: dict[tuple[str, int], dict[str, Any]] = {
    ("w5-16-downside-risk-reduction-using-regime-switching-signals-a-statistical-jump-model-approach", 5): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed page 5 contains a covariance table and prose but no displayed equations",
            "Candidate (2018b) is a bibliographic citation to Nystrup et al., not an equation label",
        ],
    },
    ("w5-16-downside-risk-reduction-using-regime-switching-signals-a-statistical-jump-model-approach", 6): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed page 6 contains prose and citations but no displayed equations",
            "Candidate (2003a) is a bibliographic citation to Harding and Pagan, not an equation label",
        ],
    },
    ("w3-cau-02-estimation-and-inference-of-heterogeneous-treatment-effects-using-random-forests", 29): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed this source page contains one 48 by 18 numeric table and no displayed equations",
            "Programmatic comparison matched all 864 table cells exactly; the expected display was a table-parenthesis detector false positive",
        ],
    },
    ("w4-15-generalized-variational-inference-three-arguments-for-deriving-new-posteriors", 74): {
        "base_attempt": 1,
        "notes": [
            "Visually confirmed every source formula and all nested underbraces are complete in retry attempt 01",
            "The seven-versus-five count is a grouping false positive caused by splitting one three-line mathematical system",
        ],
    },
    ("p06-pac-bayesian-inequalities-for-martingales", 7): {
        "notes": [
            "Visually confirmed genuine equation tags (28) through (32) are present and source-exact",
            "Candidates (7), (10), and (15) are in-text cross-references rather than displayed labels on this page",
        ],
    },
    ("ts02-locally-stationary-functional-time-series", 53): {
        "notes": [
            "Visually confirmed all displayed equations and the genuine (B2.14) tag are complete",
            "Candidates (3) and (4) are not equation labels on this page and were correctly omitted",
        ],
    },
    ("n06-approximation-rates-for-deep-calibration-of-rough-stochastic-volatility-models", 4): {
        "notes": [
            "Visually confirmed equations (2.3) through (2.10) are present and source-exact",
            "Candidates (1) and (2) are list/prose numerals rather than equation labels on this page",
        ],
    },
    ("hf06-estimating-the-quadratic-covariation-matrix-from-noisy-observations-local-method-of-moments", 31): {
        "notes": [
            "Visually confirmed the page's displayed covariance bounds are complete",
            "Candidate (0) is the argument in Sigma(0), not an equation label",
        ],
    },
    ("n06-approximation-rates-for-deep-calibration-of-rough-stochastic-volatility-models", 29): {
        "notes": [
            "Visually confirmed equations (4.31), (4.32), and (4.33) and all proof displays are complete",
            "Candidate (4.27) is an in-text cross-reference to an earlier equation, not a displayed label on this page",
        ],
    },
    ("ts01-locally-adaptive-estimation-of-evolutionary-wavelet-spectra", 17): {
        "notes": [
            "Visually confirmed all five displayed source groups, including equations (4.2) and (4.3)",
            "The expected 6 versus extracted 5 count splits the two-line aligned equation (4.2) into separate display candidates",
        ],
    },
    ("p06-pac-bayesian-inequalities-for-martingales", 6): {
        "notes": [
            "Visually confirmed all source formulas and equation tags (18) through (27), plus the untagged Corollary 3 inequality",
            "The expected 11 versus extracted 6 count is caused by grouping the multi-step chains (21)-(25) and (26)-(27) into aligned display environments",
        ],
    },
    ("ts02-locally-stationary-functional-time-series", 9): {
        "notes": [
            "Visually confirmed both (A7) moment conditions and all four lower-page mathematical display groups",
            "The expected 5 versus extracted 4 count treats the two (A7) lines as separate displays while the Markdown preserves them in one align environment",
        ],
    },
    ("n03-antithetic-multilevel-monte-carlo-estimation-for-multi-dimensional-sdes-without-l-vy-area-si", 12): {
        "notes": [
            "Visually confirmed all Corollary 4.3, Lemma 4.4, and proof inequalities are complete and source-exact",
            "The expected 5 versus extracted 4 count splits the two-line Corollary 4.3 display; the Markdown preserves all three component bounds in one aligned environment",
        ],
    },
    ("c03-utilizing-dependence-among-variables-in-evolutionary-algorithms-for-mixed-integer-programmin", 25): {
        "base_attempt": 2,
        "notes": ["Visually confirmed all numeric candidates are bibliography/volume numbers, not equation labels"],
    },
    ("c03-utilizing-dependence-among-variables-in-evolutionary-algorithms-for-mixed-integer-programmin", 26): {
        "base_attempt": 2,
        "notes": ["Visually confirmed all numeric candidates are bibliography/volume numbers, not equation labels"],
    },
    ("d03-risk-constrained-reinforcement-learning-with-percentile-risk-criteria", 1): {
        "base_attempt": 2,
        "notes": ["Visually confirmed candidate (1) is the Introduction section number, not an equation label"],
    },
    ("f04-jax-lob-a-gpu-accelerated-limit-order-book-simulator-to-unlock-large-scale-reinforcement-lea", 6): {
        "base_attempt": 2,
        "notes": [
            "Visually confirmed equation (7) is present and source-exact",
            "The remaining candidate (6) is the page footer, not an equation label",
        ],
    },
    ("4d49cc81c4cf-a-stochastic-control-approach-for-options-market-making", 16): {
        "notes": ["Complete; automated count split adjacent x1+/x2+ blocks differently"],
    },
    ("3bbe943f5590-algorithmic-trading-in-a-microstructural-limit-order-book-model", 26): {
        "notes": ["Complete; automated display-count mismatch is a grouping false positive"],
    },
    ("961e6b007dc4-provable-local-learning-rule-by-expert-aggregation-for-a-hawkes-network", 23): {
        "notes": ["Complete; eight discrepancy identities are present inside grouped displays"],
    },
}


def latest_for_key(rows: list[dict[str, Any]], key: tuple[str, int]) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if str(row.get("document_id")) == key[0] and retry.parse_int(row.get("page_number")) == key[1]
    ]
    if not candidates:
        raise RuntimeError(f"No retry ledger rows for {key[0]} page {key[1]}")
    return max(
        candidates,
        key=lambda row: (str(row.get("completed_at_utc") or ""), retry.parse_int(row.get("attempt_number"))),
    )


def matching_attempt_row(
    rows: list[dict[str, Any]], key: tuple[str, int], attempt_number: int
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if str(row.get("document_id")) == key[0]
        and retry.parse_int(row.get("page_number")) == key[1]
        and retry.parse_int(row.get("attempt_number")) == attempt_number
    ]
    if not candidates:
        raise RuntimeError(f"Attempt {attempt_number} missing for {key[0]} page {key[1]}")
    return candidates[-1]


def make_adjudication_row(
    run_dir: Path,
    ledger: list[dict[str, Any]],
    key: tuple[str, int],
    decision: dict[str, Any],
    *,
    repaired: bool,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latest = latest_for_key(ledger, key)
    base_attempt = int(decision.get("base_attempt") or retry.parse_int(latest.get("attempt_number")))
    base = matching_attempt_row(ledger, key, base_attempt)
    attempt_number = max(retry.parse_int(row.get("attempt_number")) for row in ledger if str(row.get("document_id")) == key[0] and retry.parse_int(row.get("page_number")) == key[1]) + 1

    if repaired:
        markdown_path = (
            run_dir / "formula_manual_repairs" / "markdown" / key[0]
            / f"p{key[1]:04d}" / f"manual-attempt-{attempt_number:02d}.md"
        ).resolve()
        reason = "manual_visual_adjudication_source_exact_repair"
    else:
        markdown_path = Path(str(base.get("retry_markdown_path") or "")).resolve()
        reason = "manual_visual_adjudication_grouping_false_positive"
    if not markdown_path.is_file() or not markdown_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Adjudicated Markdown missing/empty: {markdown_path}")
    markdown = markdown_path.read_text(encoding="utf-8")
    structure_issues = vision.markdown_structure_issues(markdown)
    if structure_issues:
        raise RuntimeError(f"Structurally invalid adjudicated Markdown {markdown_path}: {structure_issues}")

    raw_path = Path(
        str(base.get("retry_raw_response_path") or base.get("first_pass_raw_response_path") or "")
    ).resolve()
    if not raw_path.is_file():
        raise RuntimeError(f"Vision raw response missing: {raw_path}")
    validation = vision.validate_page_markdown(markdown, 0, [])
    row = dict(latest)
    row.update(
        {
            "attempt_id": f"manual-attempt-{attempt_number:02d}",
            "attempt_number": attempt_number,
            "model": "human-visual-adjudication",
            "rendered_image_path": str(base.get("rendered_image_path") or ""),
            "rendered_image_sha256": str(base.get("rendered_image_sha256") or ""),
            "retry_raw_response_path": str(raw_path),
            "retry_raw_sha256": ocr.sha256_file(raw_path),
            "retry_markdown_path": str(markdown_path),
            "retry_markdown_sha256": ocr.sha256_file(markdown_path),
            "transport_status": "cached",
            "retry_qa_severity": "pass",
            "retry_qa_reasons": reason,
            "retry_formula_hashes": validation["formula_hashes"],
            "resolution_status": "resolved",
            "request_id": "",
            "total_tokens": "0",
            "started_at_utc": completed_at,
            "completed_at_utc": completed_at,
            "error_type": "",
            "error": "",
        }
    )
    manifest_entry = {
        "document_id": key[0],
        "page_number": key[1],
        "decision": "source_exact_manual_repair" if repaired else "accepted_complete_grouping_false_positive",
        "base_attempt_id": str(base.get("attempt_id") or ""),
        "adjudication_attempt_id": row["attempt_id"],
        "markdown_path": str(markdown_path),
        "markdown_sha256": row["retry_markdown_sha256"],
        "raw_response_path": str(raw_path),
        "raw_response_sha256": row["retry_raw_sha256"],
        "rendered_image_path": row["rendered_image_path"],
        "rendered_image_sha256": row["rendered_image_sha256"],
        "structure_issues": structure_issues,
        "notes": decision["notes"],
    }
    return row, manifest_entry


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    ledger_path = run_dir / "formula_retry_results.csv"
    ledger: list[dict[str, Any]] = retry.read_csv(ledger_path)
    if not ledger:
        raise SystemExit(f"Retry ledger missing or empty: {ledger_path}")

    completed_at = ocr.utc_now()
    additions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for key, decision in MANUAL_REPAIRS.items():
        if not any(str(row.get("document_id")) == key[0] and retry.parse_int(row.get("page_number")) == key[1] for row in ledger):
            continue
        if any(
            str(row.get("document_id")) == key[0]
            and retry.parse_int(row.get("page_number")) == key[1]
            and str(row.get("attempt_id") or "").startswith("manual-attempt-")
            and "formula_manual_repairs" in str(row.get("retry_markdown_path") or "")
            for row in ledger
        ):
            continue
        row, entry = make_adjudication_row(run_dir, ledger, key, decision, repaired=True, completed_at=completed_at)
        additions.append(row)
        decisions.append(entry)
    for key, decision in VISUAL_ACCEPTANCES.items():
        if not any(str(row.get("document_id")) == key[0] and retry.parse_int(row.get("page_number")) == key[1] for row in ledger):
            continue
        if any(str(row.get("document_id")) == key[0] and retry.parse_int(row.get("page_number")) == key[1] and str(row.get("attempt_id") or "").startswith("manual-attempt-") for row in ledger):
            continue
        row, entry = make_adjudication_row(run_dir, ledger, key, decision, repaired=False, completed_at=completed_at)
        additions.append(row)
        decisions.append(entry)

    merged = retry.merge_ledger(ledger, additions)
    ocr.write_csv(
        ledger_path,
        [{field: row.get(field, "") for field in retry.LEDGER_FIELDS} for row in merged],
        retry.LEDGER_FIELDS,
    )
    # Reconstruct the durable decision ledger from all manual rows, so an
    # idempotent rebuild never erases earlier adjudication provenance.
    configs = {**MANUAL_REPAIRS, **VISUAL_ACCEPTANCES}
    durable_decisions: list[dict[str, Any]] = []
    for key, config in configs.items():
        manual_rows = [
            row for row in merged
            if str(row.get("document_id")) == key[0]
            and retry.parse_int(row.get("page_number")) == key[1]
            and str(row.get("attempt_id") or "").startswith("manual-attempt-")
        ]
        if not manual_rows:
            continue
        row = max(manual_rows, key=lambda value: retry.parse_int(value.get("attempt_number")))
        repaired = key in MANUAL_REPAIRS
        durable_decisions.append(
            {
                "document_id": key[0], "page_number": key[1],
                "decision": "source_exact_manual_repair" if repaired else "accepted_complete_grouping_false_positive",
                "adjudication_attempt_id": row.get("attempt_id", ""),
                "markdown_path": row.get("retry_markdown_path", ""),
                "markdown_sha256": row.get("retry_markdown_sha256", ""),
                "raw_response_path": row.get("retry_raw_response_path", ""),
                "raw_response_sha256": row.get("retry_raw_sha256", ""),
                "rendered_image_path": row.get("rendered_image_path", ""),
                "rendered_image_sha256": row.get("rendered_image_sha256", ""),
                "notes": config["notes"],
            }
        )
    manifest_path = run_dir / "formula_manual_repairs" / "manual_adjudication_manifest.json"
    ocr.write_json(
        manifest_path,
        {
            "generated_at_utc": completed_at,
            "method": "independent visual comparison of rendered source page and latest vision Markdown",
            "review_groups": 3,
            "decision_count": len(durable_decisions),
            "source_exact_repairs": sum(item["decision"] == "source_exact_manual_repair" for item in durable_decisions),
            "grouping_false_positive_acceptances": sum(item["decision"] == "accepted_complete_grouping_false_positive" for item in durable_decisions),
            "decisions": durable_decisions,
        },
    )

    documents = retry.documents_from_plan(retry.read_csv(run_dir / "ocr_plan.csv"))
    queue_rows = retry.dedupe_queue_rows(retry.read_csv(run_dir / "formula_retry_queue.csv"))
    assembly = retry.assemble_verified_documents(run_dir, documents, queue_rows, merged)
    assembly_fields = [
        "document_id", "status", "verified_markdown_path", "page_count",
        "pages_materialized", "retry_pages_used", "unresolved_pages", "warning_pages",
    ]
    ocr.write_csv(
        run_dir / "book_manifest_verified.csv",
        [{field: row.get(field, "") for field in assembly_fields} for row in assembly],
        assembly_fields,
    )
    counts: dict[str, int] = {}
    for row in retry.latest_rows_by_page(merged).values():
        status = str(row.get("resolution_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(f"Recorded {len(additions)} manual adjudications; latest retry statuses: {counts}")
    print(f"Rebuilt {len(assembly)} verified-book manifest rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
