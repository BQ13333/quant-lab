from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from quant_lab.agents.phd_math_specialists import load_specialist_panel
from quant_lab.agents.specialist_runtime import (
    SELECTION_BIAS_ROLE,
    MalaysiaReviewDossier,
    SpecialistReviewRuntime,
)
from quant_lab.research.next_cohort_protocol import (
    load_next_cohort_protocol,
    next_cohort_advisory_context,
    next_cohort_protocol_hash,
)
from quant_lab.research.providers import (
    LocalModelRouter,
    LocalRoutingPolicy,
    ModelCapability,
    OllamaProvider,
)
from quant_lab.research.trial_ledger import TrialLedger


PROTOCOL_PATH = Path("configs/qforge20_malaysia_next_cohort_prereg_draft01.json")
ADVISORY_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft01_advisory"
)
DEFAULT_MODEL = "qwen3.5:35b-mlx"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _persist_json(payload: object, path: Path) -> Path:
    content = json.dumps(
        payload,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the twenty local PhD-role contexts on Malaysia next-cohort Draft 01."
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--advisory-root", type=Path, default=ADVISORY_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    protocol = load_next_cohort_protocol(args.protocol)
    protocol_hash = next_cohort_protocol_hash(protocol)
    panel = load_specialist_panel()
    ordered_roles = (
        panel.by_id(SELECTION_BIAS_ROLE),
        *(item for item in panel.specialists if item.id != SELECTION_BIAS_ROLE),
    )
    ledger = TrialLedger(
        args.advisory_root / "advisory_ledger.sqlite",
        args.advisory_root / "advisory_evidence",
    )
    advisory_context = next_cohort_advisory_context(protocol)
    advisory_context_hash = _digest(advisory_context)
    reservation = ledger.reserve_trial(
        idempotency_key=(
            f"QFORGE20-NEXT-COHORT-DRAFT01-ADVISORY-BLINDED02:"
            f"{protocol_hash}:{advisory_context_hash}:{args.model}"
        ),
        campaign="qforge20_malaysia_next_cohort_protocol_advisory",
        family="protocol_advisory",
        hypothesis_id="MYS-H-QFORGE20-NEXT-COHORT-PROTOCOL",
        candidate_id="MYS-C-QFORGE20-NEXT-COHORT-PROTOCOL",
        code_hash=_digest("qforge20_next_cohort_protocol_advisory_v1"),
        data_hash=str(protocol["source_hashes"]["daily_features_sha256"]),
        config_hash=protocol_hash,
        prompt_hash=advisory_context_hash,
        model_hash=_digest({"route": "local_only", "model": args.model}),
    )
    if reservation.created:
        ledger.record_research_step(
            reservation.trial_id,
            event_type="hypothesis_intake",
            status="admitted",
            payload={
                "protocol_id": protocol["protocol_id"],
                "protocol_hash": protocol_hash,
                "advisory_only": True,
                "strategy_trial": False,
                "strategy_trials_reserved": False,
                "backtests_started": False,
                "workflow_name": "QForge-20",
            },
        )
    dossier = MalaysiaReviewDossier(
        candidate_id="MYS-C-QFORGE20-NEXT-COHORT-PROTOCOL",
        hypothesis_hash=_digest([item["claim"] for item in protocol["hypotheses"]]),
        candidate_hash=protocol_hash,
        problem=(
            "Independently challenge the exact five-hypothesis QForge-20 Malaysia next-cohort "
            "preregistration before human freeze. Find fatal design defects without proposing a "
            "parameter search, alternative signs, additional strategies, or using strategy results."
        ),
        tags=(
            "selection_bias",
            "time_series",
            "multiple_testing",
            "cointegration",
            "liquidity",
            "calendar",
            "market_microstructure",
            "transaction_cost",
            "execution",
        ),
        evidence_hashes={
            "protocol": protocol_hash,
            "source_inventory": str(
                protocol["source_hashes"]["clustered_strategy_inventory_sha256"]
            ),
            "daily_features": str(protocol["source_hashes"]["daily_features_sha256"]),
            "specialist_registry": str(
                protocol["source_hashes"]["specialist_roster_sha256"]
            ),
        },
        research_context=advisory_context,
        review_phase="pre_backtest_design",
    )
    policy = LocalRoutingPolicy(models={capability: args.model for capability in ModelCapability})
    runtime = SpecialistReviewRuntime(
        LocalModelRouter(OllamaProvider(), policy=policy),
        ledger=ledger,
        panel=panel,
        retry_infrastructure_failures=True,
    )
    disposition = runtime.review_roles(
        dossier,
        ordered_roles,
        trial_id=reservation.trial_id,
        require_progression_panel=False,
    )
    failures = [item for item in disposition.attempts if item.status != "success"]
    manifest = ledger.replay_manifest(reservation.trial_id)
    outcome_exists = any(item["event_type"] == "outcome" for item in manifest["events"])
    if failures:
        ledger.record_research_step(
            reservation.trial_id,
            event_type="stop",
            status="deferred",
            payload={
                "advisory_only": True,
                "resumable": True,
                "failed_roles": [item.specialist_id for item in failures],
                "failure_codes": [item.failure_code for item in failures],
            },
        )
    elif not outcome_exists:
        ledger.record_outcome(
            reservation.trial_id,
            status="success",
            payload={
                "advisory_completed": True,
                "protocol_id": protocol["protocol_id"],
                "protocol_hash": protocol_hash,
                "role_count": 20,
                "strategy_trials_reserved": False,
                "backtests_started": False,
            },
        )
    reviews = [item.review for item in disposition.attempts if item.review is not None]
    report = {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "status": "advisory_complete" if not failures else "advisory_deferred",
        "route": {
            "privacy_class": "local_only",
            "model": args.model,
            "hosted_cost_usd": 0.0,
            "role_contexts_are_not_independent_models": True,
        },
        "trial_id": reservation.trial_id,
        "strategy_trials_reserved": False,
        "backtests_started": False,
        "role_count": 20,
        "completed_role_count": len(reviews),
        "verdict_counts": dict(sorted(Counter(item.verdict for item in reviews).items())),
        "veto_count": sum(bool(item.veto) for item in reviews),
        "failed_roles": [item.specialist_id for item in failures],
        "failure_codes": [item.failure_code for item in failures],
        "model_calls": {
            "input_tokens": sum(
                item.model_call.input_tokens
                for item in disposition.attempts
                if item.model_call is not None and item.model_call.status != "replay"
            ),
            "output_tokens": sum(
                item.model_call.output_tokens
                for item in disposition.attempts
                if item.model_call is not None and item.model_call.status != "replay"
            ),
            "external_cost_usd": 0.0,
        },
        "reviews": [asdict(item) for item in reviews],
        "advisory_disposition": {
            "model_progression_flag": disposition.progression_allowed,
            "model_veto_flag": disposition.vetoed,
            "reasons": list(disposition.reasons),
            "conflicts": list(disposition.conflicts),
            "human_freeze_still_required": True,
            "model_vote_has_no_freeze_authority": True,
        },
    }
    target = _persist_json(report, args.advisory_root / "advisory_report.json")
    print(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_hash": protocol_hash,
                "status": report["status"],
                "completed_role_count": report["completed_role_count"],
                "verdict_counts": report["verdict_counts"],
                "veto_count": report["veto_count"],
                "failed_roles": report["failed_roles"],
                "model_calls": report["model_calls"],
                "strategy_trials_reserved": False,
                "backtests_started": False,
                "report_path": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
