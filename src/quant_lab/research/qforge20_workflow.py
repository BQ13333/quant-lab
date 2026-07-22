from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


QFORGE20_NAME = "QForge-20"
QFORGE20_VERSION = "QFORGE20-MALAYSIA-RESEARCH-V1"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"QForge-20 source must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_qforge20_malaysia_index(
    *,
    cohort_manifest_path: str | Path,
    cohort_report_path: str | Path,
    lifetime_counts: Mapping[str, int],
    v4_review_status: str,
    v4_reviewed_strategy_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    manifest_path = Path(cohort_manifest_path)
    report_path = Path(cohort_report_path)
    manifest = _load_object(manifest_path)
    report = _load_object(report_path)
    if not report.get("cohort_closed") or report.get("cohort_version") != manifest.get("cohort_version"):
        raise ValueError("only a closed, matching Malaysia cohort may enter the QForge-20 index")
    allowed_statuses = {"pending_rotated_key", "review_in_progress", "review_complete"}
    if v4_review_status not in allowed_statuses:
        raise ValueError("unsupported QForge-20 V4 review status")
    metrics = report.get("atomic_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("closed Malaysia cohort is missing atomic metrics")
    reviewed = set(v4_reviewed_strategy_ids)
    unknown_reviewed = reviewed - set(metrics)
    if unknown_reviewed:
        raise ValueError(f"unknown QForge-20 reviewed strategy IDs: {sorted(unknown_reviewed)}")
    if v4_review_status == "pending_rotated_key" and reviewed:
        raise ValueError("pending_rotated_key cannot contain completed V4 reviews")
    if v4_review_status == "review_complete" and reviewed != set(metrics):
        raise ValueError("review_complete requires every indexed strategy to have a V4 review")
    strategies = []
    for strategy in manifest.get("strategies", []):
        strategy_id = str(strategy["strategy_id"])
        if strategy_id not in metrics:
            raise ValueError(f"QForge-20 index is missing result metrics for {strategy_id}")
        strategies.append(
            {
                "strategy_id": strategy_id,
                "sequence": int(strategy["sequence"]),
                "family": strategy["family"],
                "hypothesis_id": strategy["hypothesis_id"],
                "economic_claim": strategy["economic_claim"],
                "parameters": strategy["parameters"],
                "source_file": strategy["source_file"],
                "source_section": strategy["source_section"],
                "research_period_result": metrics[strategy_id],
                "deterministic_result_origin": "Local-Ollama-Malaysia atomic cohort",
                "qforge20_v4_review_status": (
                    "review_complete"
                    if strategy_id in reviewed
                    else "pending_rotated_key"
                    if v4_review_status == "pending_rotated_key"
                    else "review_pending"
                ),
            }
        )
    return {
        "schema_version": 1,
        "workflow_name": QFORGE20_NAME,
        "workflow_version": QFORGE20_VERSION,
        "market": "Malaysia",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_cutoff": "2026-07-06",
        "terminal_evidence_available": False,
        "provider_plan": {
            "planning": "local Ollama / qwen3.5:35b-mlx",
            "research": "direct DeepSeek API / deepseek-v4-pro",
            "endpoint": "https://api.deepseek.com",
            "privacy_class": "cloud_sanitized",
            "hosted_fallback": False,
        },
        "provenance": {
            "cohort_manifest_sha256": _sha256(manifest_path),
            "cohort_report_sha256": _sha256(report_path),
            "returns_matrix_sha256": report.get("returns_matrix_hash"),
            "prior_review_workflow": "Local-Ollama-Malaysia",
            "prior_results_are_imported_not_relabelled": True,
        },
        "lifetime_counts": dict(lifetime_counts),
        "strategy_count": len(strategies),
        "strategies": strategies,
        "cohort_selection_diagnostics": {
            "pbo": report.get("pbo"),
            "correlation_matrix": report.get("correlation_matrix"),
            "predeclared_combinations": report.get("predeclared_combination_metrics"),
        },
        "v4_review_status": v4_review_status,
        "v4_reviewed_strategy_count": len(reviewed),
        "promotion_status": "research_only_blocked_from_dry_run",
    }


def persist_qforge20_index(index: Mapping[str, Any], target: str | Path) -> Path:
    path = Path(target)
    content = json.dumps(
        dict(index),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def build_qforge20_v4_review_report(
    *,
    index_path: str | Path,
    ledger_path: str | Path,
    evidence_root: str | Path,
    known_unledgered_call_count: int = 0,
    known_unledgered_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Reconcile completed V4 panels against immutable, redacted evidence.

    Failed malformed calls are deliberately reported as unpriced. The current
    ledger has no trustworthy token receipt for those responses, so treating
    the successful-call subtotal as exact spend would be false precision.
    """

    index_file = Path(index_path)
    ledger_file = Path(ledger_path)
    evidence_dir = Path(evidence_root)
    if known_unledgered_call_count < 0 or known_unledgered_cost_usd < 0.0:
        raise ValueError("known unledgered call reconciliation cannot be negative")
    index = _load_object(index_file)
    if index.get("workflow_name") != QFORGE20_NAME:
        raise ValueError("review report requires a QForge-20 index")
    if index.get("v4_review_status") != "review_complete":
        raise ValueError("review report requires a completed V4 review campaign")
    indexed_strategies = index.get("strategies")
    if not isinstance(indexed_strategies, list) or not indexed_strategies:
        raise ValueError("QForge-20 index contains no strategies")

    connection = sqlite3.connect(ledger_file)
    connection.row_factory = sqlite3.Row
    try:
        event_rows = connection.execute(
            """
            SELECT t.candidate_id, t.family, t.trial_id, e.event_id,
                   e.event_type, e.status, e.failure_class, e.payload_json,
                   e.evidence_sha256
            FROM trials t
            JOIN trial_events e ON e.trial_id = t.trial_id
            WHERE t.campaign = 'qforge20_malaysia_review'
            ORDER BY e.event_id
            """
        ).fetchall()
        evidence_rows = connection.execute(
            "SELECT sha256, relative_path FROM evidence"
        ).fetchall()
    finally:
        connection.close()
    evidence_paths = {str(row["sha256"]): str(row["relative_path"]) for row in evidence_rows}
    events_by_candidate: dict[str, list[sqlite3.Row]] = {}
    for row in event_rows:
        events_by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    verdict_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    failed_call_cost = 0.0
    failed_input_tokens = 0
    failed_output_tokens = 0
    receipted_failure_count = 0
    unreceipted_failure_count = 0
    unpriced_failure_counts: Counter[str] = Counter()
    completed_reviews = 0
    vetoing_reviews = 0
    strategy_reports: list[dict[str, Any]] = []
    for strategy in sorted(indexed_strategies, key=lambda item: int(item["sequence"])):
        strategy_id = str(strategy["strategy_id"])
        candidate_id = f"MYS-C-{strategy_id}"
        rows = events_by_candidate.get(candidate_id, [])
        if not rows:
            raise ValueError(f"review ledger has no trial for {strategy_id}")
        trial_ids = {str(row["trial_id"]) for row in rows}
        if len(trial_ids) != 1:
            raise ValueError(f"review ledger has multiple trials for {strategy_id}")
        completed_rows = [
            row
            for row in rows
            if row["event_type"] == "specialist_review"
            and not row["failure_class"]
            and row["evidence_sha256"]
        ]
        if len(completed_rows) != 3:
            raise ValueError(f"{strategy_id} does not have exactly three completed focused reviews")

        reviews: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        for row in completed_rows:
            evidence_sha = str(row["evidence_sha256"])
            relative_path = evidence_paths.get(evidence_sha)
            if relative_path is None:
                raise ValueError(f"missing evidence registration for {evidence_sha}")
            evidence_path = evidence_dir / relative_path
            if not evidence_path.is_file() or _sha256(evidence_path) != evidence_sha:
                raise ValueError(f"specialist evidence failed SHA-256 verification: {evidence_sha}")
            bundle = _load_object(evidence_path)
            review = bundle.get("review")
            model_call = bundle.get("model_call")
            if not isinstance(review, dict) or not isinstance(model_call, dict):
                raise ValueError("specialist evidence bundle is malformed")
            event_payload = json.loads(str(row["payload_json"]))
            specialist_id = str(review.get("specialist_id", ""))
            verdict = str(review.get("verdict", ""))
            veto = bool(review.get("veto"))
            if (
                specialist_id != event_payload.get("specialist_id")
                or verdict != row["status"]
                or veto != bool(event_payload.get("veto"))
                or model_call.get("provider") != "deepseek"
                or model_call.get("model") != "deepseek-v4-pro"
                or model_call.get("endpoint") != "https://api.deepseek.com"
                or model_call.get("status") != "success"
            ):
                raise ValueError(f"specialist evidence lineage mismatch for {strategy_id}")
            if specialist_id in seen_roles:
                raise ValueError(f"duplicate specialist role in {strategy_id} focused panel")
            seen_roles.add(specialist_id)
            cost = float(model_call.get("cost_usd", 0.0))
            if abs(cost - float(event_payload.get("cost_usd", -1.0))) > 1e-12:
                raise ValueError(f"specialist cost receipt mismatch for {strategy_id}")
            input_tokens = int(model_call.get("input_tokens", 0))
            output_tokens = int(model_call.get("output_tokens", 0))
            total_cost += cost
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            completed_reviews += 1
            vetoing_reviews += int(veto)
            verdict_counts[verdict] += 1
            role_counts[specialist_id] += 1
            reviews.append(
                {
                    "specialist_id": specialist_id,
                    "verdict": verdict,
                    "veto": veto,
                    "confidence": float(review.get("confidence", 0.0)),
                    "claims": list(review.get("claims", [])),
                    "assumptions": list(review.get("assumptions", [])),
                    "counterexamples": list(review.get("counterexamples", [])),
                    "falsification_tests": list(review.get("falsification_tests", [])),
                    "evidence_sha256": evidence_sha,
                    "model_call": {
                        "call_id": str(model_call.get("call_id", "")),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                        "stop_reason": str(model_call.get("stop_reason", "")),
                    },
                }
            )

        failure_rows = [
            row
            for row in rows
            if row["event_type"] == "specialist_review" and row["failure_class"]
        ]
        failures = [str(row["failure_class"]) for row in failure_rows]
        strategy_receipted_failures = 0
        for failure_row in failure_rows:
            evidence_sha = failure_row["evidence_sha256"]
            if not evidence_sha:
                unreceipted_failure_count += 1
                unpriced_failure_counts[str(failure_row["failure_class"])] += 1
                continue
            relative_path = evidence_paths.get(str(evidence_sha))
            if relative_path is None:
                raise ValueError(f"missing failure receipt registration for {evidence_sha}")
            evidence_path = evidence_dir / relative_path
            if not evidence_path.is_file() or _sha256(evidence_path) != str(evidence_sha):
                raise ValueError(f"failure receipt failed SHA-256 verification: {evidence_sha}")
            receipt = _load_object(evidence_path)
            if (
                receipt.get("contract_type") != "model_call"
                or receipt.get("provider") != "deepseek"
                or receipt.get("model") != "deepseek-v4-pro"
                or receipt.get("endpoint") != "https://api.deepseek.com"
                or receipt.get("status") == "success"
            ):
                raise ValueError(f"invalid provider failure receipt for {strategy_id}")
            failed_call_cost += float(receipt.get("cost_usd", 0.0))
            failed_input_tokens += int(receipt.get("input_tokens", 0))
            failed_output_tokens += int(receipt.get("output_tokens", 0))
            receipted_failure_count += 1
            strategy_receipted_failures += 1
        failure_counts.update(failures)
        outcomes = [row for row in rows if row["event_type"] == "outcome"]
        if len(outcomes) != 1:
            raise ValueError(f"{strategy_id} requires exactly one completed outcome")
        outcome = json.loads(str(outcomes[0]["payload_json"]))
        if not bool(outcome.get("review_completed")):
            raise ValueError(f"{strategy_id} outcome is not review-complete")
        dispositions = [row for row in rows if row["event_type"] == "panel_disposition"]
        if not dispositions:
            raise ValueError(f"{strategy_id} has no panel disposition")
        final_disposition = dispositions[-1]
        strategy_reports.append(
            {
                "sequence": int(strategy["sequence"]),
                "strategy_id": strategy_id,
                "family": str(strategy["family"]),
                "trial_id": next(iter(trial_ids)),
                "research_period_result": strategy.get("research_period_result"),
                "panel_status": str(final_disposition["status"]),
                "progression_allowed": bool(outcome.get("progression_allowed")),
                "vetoed": bool(outcome.get("vetoed")),
                "malformed_call_count": len(failures),
                "receipted_failure_count": strategy_receipted_failures,
                "failure_classes": dict(sorted(Counter(failures).items())),
                "reviews": reviews,
            }
        )

    expected_count = int(index.get("strategy_count", 0))
    if len(strategy_reports) != expected_count:
        raise ValueError("review report strategy count does not match the QForge-20 index")
    if any(item["progression_allowed"] for item in strategy_reports):
        raise ValueError("this research-only cohort unexpectedly contains a progressing strategy")
    malformed_count = int(unpriced_failure_counts.get("MALFORMED_MODEL_OUTPUT", 0))
    tracked_total_cost = total_cost + failed_call_cost
    known_cost_lower_bound = tracked_total_cost + known_unledgered_cost_usd
    return {
        "schema_version": 1,
        "workflow_name": QFORGE20_NAME,
        "workflow_version": QFORGE20_VERSION,
        "market": "Malaysia",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "index_sha256": _sha256(index_file),
            "review_ledger_sha256": _sha256(ledger_file),
        },
        "review_scope": {
            "canonical_specialist_registry_size": 20,
            "wave": "focused_post_backtest_research",
            "panel_size_per_strategy": 3,
            "full_finalist_coverage_executed": False,
        },
        "completion": {
            "strategy_count": expected_count,
            "completed_strategy_count": len(strategy_reports),
            "successful_structured_review_count": completed_reviews,
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "vetoing_review_count": vetoing_reviews,
            "vetoed_strategy_count": sum(item["vetoed"] for item in strategy_reports),
            "instantiated_specialist_roles": dict(sorted(role_counts.items())),
        },
        "cost_accounting": {
            "tracked_successful_call_cost_usd": round(total_cost, 12),
            "tracked_successful_input_tokens": total_input_tokens,
            "tracked_successful_output_tokens": total_output_tokens,
            "tracked_failed_call_cost_usd": round(failed_call_cost, 12),
            "tracked_failed_input_tokens": failed_input_tokens,
            "tracked_failed_output_tokens": failed_output_tokens,
            "tracked_total_call_cost_usd": round(tracked_total_cost, 12),
            "known_unledgered_call_count": known_unledgered_call_count,
            "known_unledgered_cost_usd": round(known_unledgered_cost_usd, 12),
            "known_cost_lower_bound_usd": round(known_cost_lower_bound, 12),
            "receipted_failure_count": receipted_failure_count,
            "unreceipted_failure_count": unreceipted_failure_count,
            "unpriced_malformed_call_count": malformed_count,
            "failure_classes": dict(sorted(failure_counts.items())),
            "cost_completeness": (
                "lower_bound_only" if unreceipted_failure_count else "complete"
            ),
            "note": (
                "Malformed-call provider receipts were not persisted by this campaign version; "
                "the known cost including separately reconciled calls is therefore a lower bound, "
                "not total billed spend."
                if unreceipted_failure_count
                else "Every provider call has a persisted or separately reconciled cost receipt."
            ),
        },
        "terminal_evidence_available": bool(index.get("terminal_evidence_available")),
        "dry_run_attempted": False,
        "promotion_status": str(index.get("promotion_status")),
        "conclusion": "no_candidate_promoted",
        "strategies": strategy_reports,
    }


def persist_qforge20_review_report(report: Mapping[str, Any], target: str | Path) -> Path:
    path = Path(target)
    content = json.dumps(
        dict(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path
