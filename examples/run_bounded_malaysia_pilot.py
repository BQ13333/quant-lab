from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_lab.agents.specialist_runtime import SpecialistReviewRuntime
from quant_lab.research.bounded_loop import (
    BoundedMalaysiaResearchLoop,
    CampaignBudgets,
    CampaignConfig,
    DeterministicDryRunSimulator,
    MalaysiaUniverseContract,
)
from quant_lab.research.malaysia_pilot import MalaysiaHistoricalBacktester, OllamaCandidateGenerator
from quant_lab.research.institutional_workflow import (
    DeterministicMalaysiaInstitutionalAuditor,
    HypothesisIntakeRegistry,
    HypothesisTicket,
)
from quant_lab.research.providers import LocalModelRouter, OllamaProvider
from quant_lab.research.research_extractor import ResearchExtractor
from quant_lab.research.trial_ledger import TrialLedger


DEFAULT_L2 = Path(
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/"
    "ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a finite local-Ollama Malaysia research pilot.")
    parser.add_argument("--campaign-id", default="MALAYSIA-OLLAMA-PILOT-001")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/bounded_malaysia_loop"))
    parser.add_argument("--l2-dir", type=Path, default=DEFAULT_L2)
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument("--max-generations", type=int, default=2)
    args = parser.parse_args()

    run_dir = args.artifact_root / args.campaign_id
    vault_dir = args.artifact_root / "_malaysia_lifetime_vault"
    ledger = TrialLedger(vault_dir / "trial_ledger.sqlite", vault_dir / "evidence")
    hypothesis_registry = HypothesisIntakeRegistry(vault_dir / "trial_ledger.sqlite")
    for state_path in sorted(args.artifact_root.glob("MALAYSIA-*/campaign_state.json")):
        source_dir = state_path.parent
        source_db = source_dir / "trial_ledger.sqlite"
        source_evidence = source_dir / "evidence"
        if source_dir == run_dir or not source_db.exists() or not source_evidence.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("active") is None and state.get("trace") and state.get("stop_reasons"):
            ledger.import_completed_ledger(
                source_db,
                source_evidence,
                source_key=source_dir.name,
            )
    legacy = Path("artifacts/malaysia/MALAYSIA-SWEEP-20260705-085432/sweep_manifest.json")
    if legacy.exists():
        ledger.import_legacy_malaysia_sweep(legacy)

    universe = MalaysiaUniverseContract.from_tiers_csv(args.l2_dir / "eligible_universe_tiers.csv")
    router = LocalModelRouter(OllamaProvider())
    generator = OllamaCandidateGenerator(ResearchExtractor(router))
    reviewers = SpecialistReviewRuntime(router, ledger=ledger)
    backtester = MalaysiaHistoricalBacktester(
        daily_features_path=args.l2_dir / "daily_features.parquet",
        ledger=ledger,
    )
    ticket = HypothesisTicket(
        hypothesis_id="MYS-H-MOMENTUM-001",
        economic_claim="Lagged medium-horizon price momentum persists after Malaysia trading costs.",
        family="momentum",
        features=("price", "volume", "volatility", "lagged_liquidity"),
        horizon_days=63,
        data_requirements=("point-in-time adjusted close", "lagged volume and liquidity", "daily returns"),
    )
    admission = hypothesis_registry.admit(ticket)
    if not admission.accepted:
        raise RuntimeError(f"hypothesis intake rejected: {admission.reason_code}")
    config = CampaignConfig(
        campaign_id=args.campaign_id,
        artifact_root=args.artifact_root,
        initial_queue=(
            hypothesis_registry.directive(
                "MYS-H-MOMENTUM-001", directive_id="MYS-MOMENTUM-SEED-001"
            ),
        ),
        budgets=CampaignBudgets(
            max_trials=args.max_trials,
            max_trials_per_family=args.max_trials,
            max_generations=args.max_generations,
            max_tokens=80_000,
            token_reserve_per_trial=16_000,
            max_consecutive_failures=args.max_trials,
        ),
    )
    replay = BoundedMalaysiaResearchLoop(
        config=config,
        universe=universe,
        ledger=ledger,
        hypothesis_registry=hypothesis_registry,
        candidate_generator=generator,
        specialist_runtime=reviewers,
        backtester=backtester,
        institutional_auditor=DeterministicMalaysiaInstitutionalAuditor(backtester),
        dry_runner=DeterministicDryRunSimulator(),
    ).run()
    sys.stdout.write(
        json.dumps(
            {
                "campaign_id": replay.campaign_id,
                "artifact_hash": replay.artifact_hash,
                "iterations": len(replay.transitions),
                "transitions": replay.transitions,
                "lifetime_trials": ledger.lifetime_counts(campaign="malaysia"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
