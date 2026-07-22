from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from quant_lab.agents.specialist_runtime import MalaysiaReviewDossier, SpecialistReviewRuntime
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort
from quant_lab.research.providers import QForge20ModelRouter, QForge20RoutingPolicy
from quant_lab.research.qforge20_workflow import build_qforge20_malaysia_index, persist_qforge20_index
from quant_lab.research.trial_ledger import TrialLedger


SOURCE_VAULT = Path("artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault")
QFORGE_ROOT = Path("artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _completed_review_ids(ledger: TrialLedger) -> set[str]:
    completed: set[str] = set()
    connection = sqlite3.connect(ledger.db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT t.candidate_id FROM trials t
            JOIN trial_events e ON e.trial_id = t.trial_id
            WHERE t.campaign = 'qforge20_malaysia_review'
              AND e.event_type = 'outcome'
              AND json_extract(e.payload_json, '$.review_completed') = 1
            """
        ).fetchall()
    finally:
        connection.close()
    for row in rows:
        completed.add(str(row["candidate_id"]).removeprefix("MYS-C-"))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded DeepSeek V4-Pro research reviews under QForge-20."
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--batch-start", type=int, default=1)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--source-vault", type=Path, default=SOURCE_VAULT)
    parser.add_argument("--qforge-root", type=Path, default=QFORGE_ROOT)
    parser.add_argument("--max-campaign-cost-usd", type=float, default=0.60)
    args = parser.parse_args()
    if not args.campaign_id.startswith("QFORGE20-MYS-"):
        raise ValueError("campaign-id must begin QFORGE20-MYS-")
    if not 0.0 < args.max_campaign_cost_usd <= 5.0:
        raise ValueError("QForge-20 campaign cost ceiling must be within (0, 5]")
    cohort = malaysia_atomic_cohort()
    if args.batch_start < 1 or args.batch_start + args.batch_size - 1 > len(cohort):
        raise ValueError("requested QForge-20 batch is outside the ten-strategy cohort")
    selected = cohort[args.batch_start - 1 : args.batch_start - 1 + args.batch_size]
    cohort_dir = args.source_vault / "cohorts"
    manifest_path = cohort_dir / "MALAYSIA-ATOMIC-COHORT-V1.json"
    report_path = cohort_dir / "MALAYSIA-ATOMIC-COHORT-V1-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    review_ledger = TrialLedger(
        args.qforge_root / "review_ledger.sqlite",
        args.qforge_root / "review_evidence",
    )
    transitions = []
    campaign_cost = 0.0
    for strategy in selected:
        remaining_cost = args.max_campaign_cost_usd - campaign_cost
        if remaining_cost <= 0.0:
            transitions.append({"strategy_id": strategy.strategy_id, "status": "cost_budget_exhausted"})
            break
        runtime = SpecialistReviewRuntime(
            QForge20ModelRouter(
                policy=QForge20RoutingPolicy(
                    per_call_cost_usd=min(0.10, remaining_cost / 3.0)
                )
            ),
            ledger=review_ledger,
            retry_infrastructure_failures=True,
        )
        reservation = review_ledger.reserve_trial(
            idempotency_key=f"QFORGE20-V4-REVIEW:{strategy.strategy_id}:{report_hash}",
            campaign="qforge20_malaysia_review",
            family=strategy.family,
            hypothesis_id=strategy.hypothesis_id,
            candidate_id=f"MYS-C-{strategy.strategy_id}",
            code_hash=_digest("qforge20_post_backtest_review_v1"),
            data_hash=report_hash,
            config_hash=_digest({"workflow": "QForge-20", "reasoning_effort": "high"}),
            prompt_hash=_digest(asdict(strategy)),
            model_hash=_digest("direct_deepseek_v4_pro_cloud_sanitized_v1"),
        )
        manifest = review_ledger.replay_manifest(reservation.trial_id)
        if reservation.created:
            review_ledger.record_research_step(
                reservation.trial_id,
                event_type="hypothesis_intake",
                status="admitted",
                payload={"hypothesis_id": strategy.hypothesis_id, "workflow_name": "QForge-20"},
            )
        if any(item["event_type"] == "outcome" for item in manifest["events"]):
            transitions.append({"strategy_id": strategy.strategy_id, "status": "already_complete"})
            continue
        metrics = report["atomic_metrics"][strategy.strategy_id]
        dossier = MalaysiaReviewDossier(
            candidate_id=f"MYS-C-{strategy.strategy_id}",
            hypothesis_hash=_digest(strategy.hypothesis_id),
            candidate_hash=strategy.content_hash(),
            problem=(
                f"Adversarially interpret the frozen {strategy.family} Malaysia research result, "
                "identify alternative explanations, and specify falsification tests without tuning."
            ),
            tags=(strategy.family, "selection_bias", "robustness", "transaction_cost"),
            evidence_hashes={
                "cohort_manifest": manifest_hash,
                "cohort_report": report_hash,
                "strategy_spec": strategy.content_hash(),
            },
            research_context={
                "workflow_name": "QForge-20",
                "strategy_id": strategy.strategy_id,
                "family": strategy.family,
                "economic_claim": strategy.economic_claim,
                "frozen_parameters": dict(strategy.parameters),
                "source_file": strategy.source_file,
                "source_section": strategy.source_section,
                "research_period_metrics": metrics,
                "cohort_pbo": report["pbo"],
                "evidence_status": "historical research only; no promotion or parameter tuning",
                "prior_gate_status": "not eligible for institutional audit or dry run",
            },
            review_phase="post_backtest_research",
        )
        disposition = runtime.review_candidate(dossier, trial_id=reservation.trial_id)
        failures = [attempt.failure_code for attempt in disposition.attempts if attempt.status != "success"]
        cost = sum(
            attempt.model_call.cost_usd
            for attempt in disposition.attempts
            if attempt.model_call is not None
        )
        campaign_cost += cost
        if failures:
            review_ledger.record_research_step(
                reservation.trial_id,
                event_type="stop",
                status="deferred",
                payload={"reason_codes": failures, "resumable": True},
            )
            status = "deferred"
        else:
            review_ledger.record_outcome(
                reservation.trial_id,
                status="success",
                payload={
                    "review_completed": True,
                    "workflow_name": "QForge-20",
                    "progression_allowed": disposition.progression_allowed,
                    "vetoed": disposition.vetoed,
                    "disposition_hash": disposition.disposition_hash,
                },
            )
            status = "review_complete"
        transitions.append(
            {
                "strategy_id": strategy.strategy_id,
                "trial_id": reservation.trial_id,
                "status": status,
                "cost_usd": cost,
                "progression_allowed": disposition.progression_allowed,
                "vetoed": disposition.vetoed,
            }
        )
    completed = _completed_review_ids(review_ledger)
    v4_status = "review_complete" if len(completed) == len(cohort) else "review_in_progress"
    source_ledger = TrialLedger(args.source_vault / "trial_ledger.sqlite", args.source_vault / "evidence")
    index = build_qforge20_malaysia_index(
        cohort_manifest_path=manifest_path,
        cohort_report_path=report_path,
        lifetime_counts=source_ledger.lifetime_counts(campaign="malaysia"),
        v4_review_status=v4_status,
        v4_reviewed_strategy_ids=tuple(sorted(completed)),
    )
    persist_qforge20_index(index, args.qforge_root / "index.json")
    print(
        json.dumps(
            {
                "workflow_name": "QForge-20",
                "campaign_id": args.campaign_id,
                "batch_sequences": [item.sequence for item in selected],
                "transitions": transitions,
                "reviewed_strategy_count": len(completed),
                "strategy_count": len(cohort),
                "v4_review_status": v4_status,
                "campaign_cost_usd": campaign_cost,
                "max_campaign_cost_usd": args.max_campaign_cost_usd,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
