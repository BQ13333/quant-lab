from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from quant_lab.agents.phd_math_specialists import load_specialist_panel
from quant_lab.agents.specialist_runtime import (
    SELECTION_BIAS_ROLE,
    MalaysiaReviewDossier,
    PanelDisposition,
    SpecialistReviewRuntime,
)
from quant_lab.research.malaysia_pit_replication import (
    audit_pit_replication_inputs,
    file_sha256,
    load_frozen_protocol,
    object_sha256,
    persist_json,
)
from quant_lab.research.providers import QForge20ModelRouter, QForge20RoutingPolicy
from quant_lab.research.trial_ledger import TrialLedger


DEFAULT_PROTOCOL = Path("configs/qforge20_malaysia_pit_replication_frozen_v1.json")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1"
)


def _stable_data_report_hash(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("generated_at_utc", None)
    return object_sha256(stable)


def _knowledge_pack(repo_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    sources = {
        "qforge_method": repo_root / "docs/QFORGE_20_WORKFLOW.md",
        "atomic_findings": repo_root / "docs/MALAYSIA_ATOMIC_COHORT_V1_FINDINGS.md",
        "next_findings": repo_root / "docs/QFORGE_20_MALAYSIA_NEXT_COHORT_FINDINGS.md",
        "specialist_roster": repo_root / "configs/phd_math_specialists.json",
    }
    configured_run_state = os.environ.get("QUANT_LAB_RUN_STATE")
    if configured_run_state:
        run_state_path = Path(configured_run_state).expanduser().resolve()
        if not run_state_path.is_file():
            raise FileNotFoundError(
                f"QUANT_LAB_RUN_STATE does not name a file: {run_state_path}"
            )
        sources["run_state"] = run_state_path
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    pack = {
        "source_documents": [
            {"source_name": name, "sha256": hashes[name]} for name in sorted(sources)
        ],
        "method_principles": [
            "A model review is advisory; deterministic evidence decides.",
            "One reproducible counterexample can block progression; majority vote cannot prove validity.",
            "Every strategy identity, formula, direction, horizon, cost, and rebalance rule is frozen.",
            "No grid search, sign flip, parameter repair, interim ranking, or post-hoc combination is allowed.",
            "The complete lifetime trial denominator, Holm correction, dependence, drawdown, turnover, capacity, factor, and regime tests remain binding.",
            "Historical observations already used in research are contaminated and cannot establish future performance.",
        ],
        "prior_evidence_principles": [
            "The first ten strategies and the later four are existing failed or blocked research hypotheses, not recommendations.",
            "The replication question is survivorship and corporate-action sensitivity, not a search for a higher Sharpe ratio.",
            "A current-survivor rerun cannot be labeled point-in-time merely because a separate dead-name directory exists.",
            "A valid denominator requires time-varying U2 eligibility plus matching signal inputs and adjusted returns for inactive and delisted names.",
        ],
        "role_instruction": "Use the supplied mandate, proof obligations, kill criteria, and corpus anchors. Separate assumptions, empirical facts, and conjectures. Ask only for data-only or protocol-level repairs; never propose strategy tuning.",
    }
    return pack, hashes


def _data_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scorecard_claim": report["institutional_scorecard_claim"],
        "security_master_claim": report["historical_security_master_claim"],
        "counts": report["data_counts"],
        "membership_diagnostics": {
            "explicit_historical_u2_tier": report["membership_diagnostics"][
                "has_explicit_historical_u2_tier"
            ],
            "repeated_time_snapshots": report["membership_diagnostics"][
                "has_repeated_time_snapshots_by_ric"
            ],
            "reconstructable": report["membership_diagnostics"]["reconstructable"],
        },
        "promotion_statuses": report["security_master_promotion_statuses"],
        "gates": report["gates"],
        "failed_gate_ids": report["failed_gate_ids"],
    }


def _usage_from_manifest(ledger: TrialLedger, trial_id: str) -> dict[str, Any]:
    events = [
        event
        for event in ledger.replay_manifest(trial_id)["events"]
        if event["event_type"] == "specialist_review"
    ]
    total_cost = 0.0
    input_tokens = 0
    output_tokens = 0
    provider_calls = 0
    malformed_or_failed_calls = 0
    for event in events:
        payload = json.loads(event["payload_json"])
        receipt: Mapping[str, Any] = payload
        evidence_sha = event.get("evidence_sha256")
        if evidence_sha:
            evidence_payload = json.loads(
                ledger.verify_evidence(evidence_sha).path.read_text(encoding="utf-8")
            )
            if isinstance(evidence_payload.get("model_call"), Mapping):
                receipt = evidence_payload["model_call"]
            elif isinstance(evidence_payload, Mapping) and evidence_payload.get("call_id"):
                receipt = evidence_payload
        if receipt.get("call_id"):
            provider_calls += 1
            total_cost += float(receipt.get("cost_usd", 0.0) or 0.0)
            input_tokens += int(receipt.get("input_tokens", 0) or 0)
            output_tokens += int(receipt.get("output_tokens", 0) or 0)
        if event["status"] == "failure":
            malformed_or_failed_calls += 1
    return {
        "provider_call_receipts": provider_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": total_cost,
        "failed_call_events": malformed_or_failed_calls,
    }


def _current_reviews(disposition: PanelDisposition) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for attempt in disposition.attempts:
        item: dict[str, Any] = {
            "specialist_id": attempt.specialist_id,
            "debate_role": attempt.debate_role,
            "status": attempt.status,
            "failure_code": attempt.failure_code,
        }
        if attempt.review is not None:
            item.update(
                {
                    "verdict": attempt.review.verdict,
                    "veto": attempt.review.veto,
                    "confidence": attempt.review.confidence,
                    "claims": list(attempt.review.claims),
                    "assumptions": list(attempt.review.assumptions),
                    "counterexamples": list(attempt.review.counterexamples),
                    "falsification_tests": list(attempt.review.falsification_tests),
                    "review_sha256": attempt.review.content_hash(),
                }
            )
        if attempt.model_call is not None:
            item["model_call"] = {
                "provider": attempt.model_call.provider,
                "model": attempt.model_call.model,
                "status": attempt.model_call.status,
                "input_tokens": attempt.model_call.input_tokens,
                "output_tokens": attempt.model_call.output_tokens,
                "cost_usd": attempt.model_call.cost_usd,
                "prompt_sha256": attempt.model_call.prompt_hash,
                "response_sha256": attempt.model_call.response_hash,
                "redactions": list(attempt.model_call.redactions),
            }
        reviews.append(item)
    return reviews


def _build_report(
    *,
    protocol: Mapping[str, Any],
    data_report: Mapping[str, Any],
    disposition: PanelDisposition,
    ledger: TrialLedger,
    trial_id: str,
    knowledge_hashes: Mapping[str, str],
    waves_run: int,
) -> dict[str, Any]:
    reviews = _current_reviews(disposition)
    successful = [item for item in reviews if item.get("verdict")]
    verdict_counts = Counter(str(item["verdict"]) for item in successful)
    usage = _usage_from_manifest(ledger, trial_id)
    role_review_complete = len(successful) == 20 and all(
        item["status"] == "success" for item in reviews
    )
    role_gate = bool(
        role_review_complete
        and disposition.backtest_allowed
        and not disposition.vetoed
    )
    deterministic_gate = bool(data_report["admissible"])
    return {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": object_sha256(protocol),
        "data_admissibility_sha256": _stable_data_report_hash(data_report),
        "phase": "pre_backtest_protocol_and_data_admissibility",
        "route": {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "reasoning_effort": "high",
            "privacy_class": "cloud_sanitized",
            "raw_licensed_data_sent": False,
            "credentials_persisted": False,
        },
        "knowledge_pack_source_hashes": dict(knowledge_hashes),
        "review_trial_id": trial_id,
        "waves_run_this_invocation": waves_run,
        "role_count_required": 20,
        "role_count_completed": len(successful),
        "all_twenty_roles_completed": role_review_complete,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "veto_count": sum(bool(item.get("veto")) for item in successful),
        "role_backtest_gate": role_gate,
        "deterministic_data_gate": deterministic_gate,
        "strategy_reservation_authorized": role_gate and deterministic_gate,
        "failed_data_gate_ids": data_report["failed_gate_ids"],
        "disposition": {
            "backtest_allowed_by_roles": disposition.backtest_allowed,
            "progression_allowed_by_roles": disposition.progression_allowed,
            "vetoed": disposition.vetoed,
            "reasons": list(disposition.reasons),
            "conflicts": list(disposition.conflicts),
            "disposition_sha256": disposition.disposition_hash,
        },
        "model_usage": usage,
        "reviews": reviews,
        "control_decision": {
            "status": (
                "ready_for_atomic_fourteen_strategy_reservation"
                if role_gate and deterministic_gate
                else "stopped_before_strategy_reservation"
            ),
            "strategy_trials_reserved": False,
            "strategy_computations_started": False,
            "paired_performance_evaluation_completed": False,
            "parameter_optimization_performed": False,
            "dry_run_started": False,
            "reason": (
                "all role and deterministic data gates passed"
                if role_gate and deterministic_gate
                else "deterministic data admissibility and completed non-vetoing role coverage are both mandatory"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all twenty sanitized DeepSeek V4-Pro QForge roles on the frozen Malaysia PIT protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-waves", type=int, default=3)
    parser.add_argument("--max-campaign-cost-usd", type=float, default=3.0)
    args = parser.parse_args()
    if not 1 <= args.max_waves <= 5:
        raise ValueError("max-waves must be within [1, 5]")
    if not 0.8 <= args.max_campaign_cost_usd <= 5.0:
        raise ValueError("twenty-role campaign cost ceiling must be within [0.8, 5.0]")

    repo_root = Path.cwd().resolve()
    protocol = load_frozen_protocol(args.protocol)
    data_report = audit_pit_replication_inputs(protocol, repo_root=repo_root)
    persist_json(data_report, args.output_root / "data_admissibility_report.json")
    protocol_hash = object_sha256(protocol)
    data_hash = _stable_data_report_hash(data_report)
    report_path = args.output_root / "twenty_role_review_report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol_sha256") == protocol_hash
            and existing.get("data_admissibility_sha256") == data_hash
            and existing.get("all_twenty_roles_completed") is True
        ):
            print(
                json.dumps(
                    {
                        "protocol_sha256": protocol_hash,
                        "all_twenty_roles_completed": True,
                        "verdict_counts": existing["verdict_counts"],
                        "veto_count": existing["veto_count"],
                        "role_backtest_gate": existing["role_backtest_gate"],
                        "deterministic_data_gate": existing["deterministic_data_gate"],
                        "strategy_reservation_authorized": existing[
                            "strategy_reservation_authorized"
                        ],
                        "model_usage": existing["model_usage"],
                        "report_path": str(report_path),
                        "idempotent_replay": True,
                        "dry_run_started": False,
                    },
                    indent=2,
                )
            )
            return
    knowledge_pack, knowledge_hashes = _knowledge_pack(repo_root)
    panel = load_specialist_panel()
    ordered_roles = (
        panel.by_id(SELECTION_BIAS_ROLE),
        *(item for item in panel.specialists if item.id != SELECTION_BIAS_ROLE),
    )
    if len(ordered_roles) != 20 or len({item.id for item in ordered_roles}) != 20:
        raise RuntimeError("canonical QForge-20 role coverage is not exactly twenty")

    dossier = MalaysiaReviewDossier(
        candidate_id="MYS-C-PIT-REPLICATION-14-PROTOCOL",
        hypothesis_hash=hashlib.sha256(protocol["purpose"].encode()).hexdigest(),
        candidate_hash=protocol_hash,
        problem=(
            "Review the frozen fourteen-strategy Malaysia point-in-time replication protocol and its "
            "data-admissibility findings. Determine whether a valid historical U2 denominator with "
            "inactive and delisted securities is available. Identify concrete leakage, dependence, "
            "multiplicity, numerical, robustness, portfolio, capacity, or execution blockers and the "
            "minimum data-only or protocol-level repair. Do not tune a strategy, change a direction, "
            "search a parameter, invent performance, or authorize a dry run."
        ),
        tags=(
            "selection_bias",
            "point_in_time",
            "survivorship_bias",
            "time_series",
            "high_dimensional",
            "robust_statistics",
            "causal",
            "sequential",
            "learning_theory",
            "optimization",
            "portfolio",
            "transaction_cost",
            "execution",
        ),
        evidence_hashes={
            "protocol": protocol_hash,
            "data_admissibility": data_hash,
            **knowledge_hashes,
        },
        research_context={
            "workflow_name": "QForge-20",
            "protocol_id": protocol["protocol_id"],
            "frozen_scope": {
                "market": "Malaysia",
                "strategy_count": 14,
                "research_cutoff": protocol["research_cutoff"],
                "strategy_identities": [
                    {
                        "sequence": item["sequence"],
                        "strategy_id": item["strategy_id"],
                        "family": item["family"],
                        "spec_sha256": item["spec_sha256"],
                    }
                    for item in protocol["strategies"]
                ],
            },
            "anti_overfit_contract": {
                "parameters_unchanged": True,
                "directions_unchanged": True,
                "costs_unchanged": True,
                "no_parameter_grid": True,
                "no_interim_ranking": True,
                "no_sign_flip": True,
                "no_post_hoc_combination": True,
                "holm_primary_count": 14,
                "lifetime_pbo_required": True,
            },
            "point_in_time_data_findings": _data_summary(data_report),
            "knowledge_pack": knowledge_pack,
            "evaluation_boundary": {
                "role_opinions_are_advisory": True,
                "deterministic_data_gate_is_decisive": True,
                "stop_after_paired_evaluation": True,
                "dry_run_allowed": False,
            },
        },
        review_phase="pre_backtest_design",
    )

    ledger = TrialLedger(
        args.output_root / "twenty_role_review_ledger.sqlite",
        args.output_root / "twenty_role_review_evidence",
    )
    reservation = ledger.reserve_trial(
        idempotency_key=f"QFORGE20-MYS-PIT14-20ROLE-V2:{protocol_hash}:{data_hash}",
        campaign="qforge20_malaysia_pit_protocol_review",
        family="pit_replication_protocol_admissibility",
        hypothesis_id="MYS-H-PIT-REPLICATION-14-PROTOCOL",
        candidate_id="MYS-C-PIT-REPLICATION-14-PROTOCOL",
        code_hash=hashlib.sha256(
            b"qforge20_malaysia_pit_20role_review_v2"
        ).hexdigest(),
        data_hash=data_hash,
        config_hash=protocol_hash,
        prompt_hash=object_sha256(knowledge_pack),
        model_hash=hashlib.sha256(b"direct_deepseek_v4_pro_high_cloud_sanitized").hexdigest(),
    )
    if reservation.created:
        ledger.record_research_step(
            reservation.trial_id,
            event_type="hypothesis_intake",
            status="admitted",
            payload={
                "workflow_name": "QForge-20",
                "protocol_id": protocol["protocol_id"],
                "role_count": 20,
                "strategy_trials_reserved": False,
            },
        )

    waves_run = 0
    disposition: PanelDisposition | None = None
    for _ in range(args.max_waves):
        usage_before = _usage_from_manifest(ledger, reservation.trial_id)
        if usage_before["cost_usd"] >= args.max_campaign_cost_usd:
            break
        runtime = SpecialistReviewRuntime(
            QForge20ModelRouter(
                policy=QForge20RoutingPolicy(
                    per_call_cost_usd=0.05,
                    reasoning_effort="high",
                )
            ),
            ledger=ledger,
            panel=panel,
            retry_infrastructure_failures=True,
        )
        disposition = runtime.review_roles(
            dossier,
            ordered_roles,
            trial_id=reservation.trial_id,
            require_progression_panel=True,
        )
        waves_run += 1
        failures = [item for item in disposition.attempts if item.status != "success"]
        if not failures:
            break
        usage_after = _usage_from_manifest(ledger, reservation.trial_id)
        if usage_after["cost_usd"] >= args.max_campaign_cost_usd:
            break

    if disposition is None:
        raise RuntimeError("twenty-role review did not start within the campaign cost budget")
    failures = [item.failure_code for item in disposition.attempts if item.status != "success"]
    if failures:
        ledger.record_research_step(
            reservation.trial_id,
            event_type="stop",
            status="deferred",
            payload={"reason_codes": failures, "resumable": True},
        )
    elif not any(
        item["event_type"] == "outcome"
        for item in ledger.replay_manifest(reservation.trial_id)["events"]
    ):
        ledger.record_outcome(
            reservation.trial_id,
            status="success",
            payload={
                "review_completed": True,
                "all_twenty_roles": True,
                "role_backtest_gate": disposition.backtest_allowed,
                "deterministic_data_gate": data_report["admissible"],
                "strategy_reservation_authorized": bool(
                    disposition.backtest_allowed and data_report["admissible"]
                ),
                "dry_run_started": False,
            },
        )

    report = _build_report(
        protocol=protocol,
        data_report=data_report,
        disposition=disposition,
        ledger=ledger,
        trial_id=reservation.trial_id,
        knowledge_hashes=knowledge_hashes,
        waves_run=waves_run,
    )
    report_path = persist_json(report, args.output_root / "twenty_role_review_report.json")
    print(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "all_twenty_roles_completed": report["all_twenty_roles_completed"],
                "verdict_counts": report["verdict_counts"],
                "veto_count": report["veto_count"],
                "role_backtest_gate": report["role_backtest_gate"],
                "deterministic_data_gate": report["deterministic_data_gate"],
                "strategy_reservation_authorized": report["strategy_reservation_authorized"],
                "model_usage": report["model_usage"],
                "report_path": str(report_path),
                "dry_run_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
