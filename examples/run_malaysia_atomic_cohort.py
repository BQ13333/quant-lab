from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from quant_lab.agents.specialist_runtime import SpecialistReviewRuntime
from quant_lab.research.bounded_loop import (
    BoundedMalaysiaResearchLoop,
    CampaignBudgets,
    CampaignConfig,
    DeterministicDryRunSimulator,
    MalaysiaUniverseContract,
)
from quant_lab.research.institutional_workflow import (
    DeterministicMalaysiaInstitutionalAuditor,
    HypothesisIntakeRegistry,
    HypothesisTicket,
)
from quant_lab.research.malaysia_pilot import FrozenCohortCandidateGenerator, MalaysiaHistoricalBacktester
from quant_lab.research.malaysia_cohort_analysis import (
    build_malaysia_cohort_report,
    persist_malaysia_cohort_report,
)
from quant_lab.research.malaysia_strategy_registry import (
    COHORT_VERSION,
    malaysia_atomic_cohort,
    malaysia_predeclared_combinations,
)
from quant_lab.research.providers import (
    LocalModelRouter,
    LocalRoutingPolicy,
    ModelCapability,
    OllamaProvider,
)
from quant_lab.research.trial_ledger import TrialLedger


DEFAULT_L2 = Path(
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/"
    "ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _freeze_manifest(vault_dir: Path) -> str:
    cohort = malaysia_atomic_cohort()
    payload = {
        "cohort_version": COHORT_VERSION,
        "market": "Malaysia",
        "strategy_count": len(cohort),
        "ranking_locked_until_complete": True,
        "strategies": [asdict(item) for item in cohort],
        "predeclared_combinations": [asdict(item) for item in malaysia_predeclared_combinations()],
    }
    content = _canonical(payload) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    target = vault_dir / "cohorts" / f"{COHORT_VERSION}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise RuntimeError("frozen Malaysia cohort manifest changed after first registration")
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one sequential batch from the frozen Malaysia atomic cohort.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--batch-start", type=int, default=1, help="One-based first cohort sequence.")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/bounded_malaysia_loop"))
    parser.add_argument("--l2-dir", type=Path, default=DEFAULT_L2)
    parser.add_argument(
        "--review-profile",
        choices=("dual-model", "qwen-only"),
        default="dual-model",
        help=(
            "Use the normal Qwen/DeepSeek panel, or the bounded all-Qwen operational "
            "fallback when DeepSeek times out. Specialist roles remain independent."
        ),
    )
    args = parser.parse_args()

    cohort = malaysia_atomic_cohort()
    if args.batch_start < 1 or args.batch_start + args.batch_size - 1 > len(cohort):
        raise ValueError("requested batch falls outside the frozen ten-strategy cohort")
    selected = cohort[args.batch_start - 1 : args.batch_start - 1 + args.batch_size]
    vault_dir = args.artifact_root / "_malaysia_lifetime_vault"
    cohort_hash = _freeze_manifest(vault_dir)
    ledger = TrialLedger(vault_dir / "trial_ledger.sqlite", vault_dir / "evidence")
    before = build_malaysia_cohort_report(ledger)
    already_completed = set(before["completed_strategy_ids"])
    repeated = [item.strategy_id for item in selected if item.strategy_id in already_completed]
    if repeated:
        raise RuntimeError(f"atomic strategies cannot be rerun: {repeated}")
    registry = HypothesisIntakeRegistry(vault_dir / "trial_ledger.sqlite")
    for strategy in cohort:
        admission = registry.admit(
            HypothesisTicket(
                hypothesis_id=strategy.hypothesis_id,
                economic_claim=strategy.economic_claim,
                family=strategy.family,
                features=("price", "volume", "volatility", "lagged_liquidity"),
                horizon_days=strategy.horizon_days,
                data_requirements=(
                    "point-in-time adjusted prices",
                    "lagged daily returns and volatility",
                    "lagged liquidity and volume quality flags",
                ),
            )
        )
        if not admission.accepted:
            raise RuntimeError(f"cohort hypothesis intake rejected: {admission.reason_code}")

    directives = tuple(
        registry.directive(item.hypothesis_id, directive_id=item.directive_id) for item in selected
    )
    universe = MalaysiaUniverseContract.from_tiers_csv(args.l2_dir / "eligible_universe_tiers.csv")
    policy = LocalRoutingPolicy()
    if args.review_profile == "qwen-only":
        policy = LocalRoutingPolicy(
            models={
                **policy.models,
                ModelCapability.ADVERSARIAL_MATH: policy.models[ModelCapability.ROUTINE_REVIEW],
            }
        )
    router = LocalModelRouter(OllamaProvider(), policy=policy)
    reviewers = SpecialistReviewRuntime(router, ledger=ledger)
    backtester = MalaysiaHistoricalBacktester(
        daily_features_path=args.l2_dir / "daily_features.parquet",
        ledger=ledger,
    )
    config = CampaignConfig(
        campaign_id=args.campaign_id,
        artifact_root=args.artifact_root,
        initial_queue=directives,
        budgets=CampaignBudgets(
            max_hypotheses=args.batch_size,
            max_trials=args.batch_size,
            max_trials_per_family=args.batch_size,
            max_generations=1,
            max_tokens=80_000,
            token_reserve_per_trial=16_000,
            max_consecutive_failures=args.batch_size,
        ),
    )
    replay = BoundedMalaysiaResearchLoop(
        config=config,
        universe=universe,
        ledger=ledger,
        hypothesis_registry=registry,
        candidate_generator=FrozenCohortCandidateGenerator(selected),
        specialist_runtime=reviewers,
        backtester=backtester,
        institutional_auditor=DeterministicMalaysiaInstitutionalAuditor(backtester),
        dry_runner=DeterministicDryRunSimulator(),
    ).run()
    cohort_report = build_malaysia_cohort_report(ledger)
    if cohort_report["cohort_closed"]:
        persist_malaysia_cohort_report(
            ledger,
            vault_dir / "cohorts" / f"{COHORT_VERSION}-report.json",
        )
    sys.stdout.write(
        json.dumps(
            {
                "campaign_id": replay.campaign_id,
                "cohort_version": COHORT_VERSION,
                "cohort_hash": cohort_hash,
                "batch_sequences": [item.sequence for item in selected],
                "transitions": replay.transitions,
                "cohort_completed_strategies": cohort_report["completed_strategy_count"],
                "cohort_total_strategies": len(cohort),
                "cohort_closed": cohort_report["cohort_closed"],
                "ranking_available": cohort_report["ranking_available"],
                "cohort_report": cohort_report,
                "lifetime_trials": ledger.lifetime_counts(campaign="malaysia"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
