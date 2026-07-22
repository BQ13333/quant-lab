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
from quant_lab.research.providers import (
    LocalModelRouter,
    LocalRoutingPolicy,
    ModelCapability,
    OllamaProvider,
)
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.research.v2_overlay_protocol import (
    load_v2_overlay_protocol,
    v2_overlay_protocol_hash,
    v2_protocol_advisory_context,
)


PROTOCOL_PATH = Path("configs/qforge20_malaysia_v2_overlay_draft.json")
ADVISORY_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_advisory"
)
LOCAL_ADVISORY_MODEL = "qwen3.5:35b-mlx"


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
        description="Run all twenty local mathematical advisory roles on the exact Malaysia V2 draft."
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--advisory-root", type=Path, default=ADVISORY_ROOT)
    args = parser.parse_args()
    protocol = load_v2_overlay_protocol(args.protocol)
    protocol_hash = v2_overlay_protocol_hash(protocol)
    panel = load_specialist_panel()
    ordered_roles = (
        panel.by_id(SELECTION_BIAS_ROLE),
        *(item for item in panel.specialists if item.id != SELECTION_BIAS_ROLE),
    )
    ledger = TrialLedger(
        args.advisory_root / "advisory_ledger.sqlite",
        args.advisory_root / "advisory_evidence",
    )
    reservation = ledger.reserve_trial(
        idempotency_key=f"QFORGE20-V2-PROTOCOL-ADVISORY:{protocol_hash}",
        campaign="qforge20_malaysia_v2_protocol_advisory",
        family="protocol_advisory",
        hypothesis_id="MYS-H-QFORGE20-V2-PROTOCOL",
        candidate_id="MYS-C-QFORGE20-V2-PROTOCOL",
        code_hash=_digest("qforge20_v2_protocol_advisory_v1"),
        data_hash=str(protocol["source_hashes"]["malaysia_v1_report_sha256"]),
        config_hash=protocol_hash,
        prompt_hash=_digest(v2_protocol_advisory_context(protocol)),
        model_hash=_digest({"route": "local_only", "model": LOCAL_ADVISORY_MODEL}),
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
                "workflow_name": "QForge-20",
            },
        )
    dossier = MalaysiaReviewDossier(
        candidate_id="MYS-C-QFORGE20-V2-PROTOCOL",
        hypothesis_hash=_digest(protocol["claim"]),
        candidate_hash=protocol_hash,
        problem=(
            "Independently challenge the exact QForge-20 Malaysia V2 paired overlay protocol before "
            "human freeze. Identify mathematical, causal, numerical, statistical, portfolio, and "
            "execution-design defects without proposing a parameter search or using strategy results."
        ),
        tags=(
            "selection_bias",
            "time_series",
            "robust_statistics",
            "drawdown",
            "optimization",
            "portfolio",
            "transaction_cost",
            "execution",
        ),
        evidence_hashes={
            "protocol": protocol_hash,
            "specialist_registry": str(protocol["source_hashes"]["specialist_registry_sha256"]),
            "v1_index": str(protocol["source_hashes"]["qforge20_v1_index_sha256"]),
            "v1_report": str(protocol["source_hashes"]["malaysia_v1_report_sha256"]),
        },
        research_context=v2_protocol_advisory_context(protocol),
        review_phase="pre_backtest_design",
    )
    policy = LocalRoutingPolicy(
        models={capability: LOCAL_ADVISORY_MODEL for capability in ModelCapability}
    )
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
            "model": LOCAL_ADVISORY_MODEL,
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
