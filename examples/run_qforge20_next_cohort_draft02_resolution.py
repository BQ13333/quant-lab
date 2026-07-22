from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from quant_lab.agents.phd_math_specialists import load_specialist_panel
from quant_lab.agents.specialist_runtime import MalaysiaReviewDossier, SpecialistReviewRuntime
from quant_lab.research.next_cohort_protocol import (
    materialize_next_cohort_protocol,
    materialized_next_cohort_protocol_hash,
)
from quant_lab.research.providers import (
    LocalModelRouter,
    LocalRoutingPolicy,
    ModelCapability,
    OllamaProvider,
)
from quant_lab.research.trial_ledger import TrialLedger


BASE_PROTOCOL = Path("configs/qforge20_malaysia_next_cohort_prereg_draft01.json")
REVISION_MANIFEST = Path("configs/qforge20_malaysia_next_cohort_draft02_revision.json")
RESOLUTION_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft02"
)
LIFETIME_LEDGER = Path(
    "artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault/trial_ledger.sqlite"
)
LOCAL_MODEL = "deepseek-r1:70b"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


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


def _targeted_context(protocol: dict[str, object]) -> dict[str, object]:
    hypotheses = []
    for raw in protocol["hypotheses"]:  # type: ignore[index]
        item = dict(raw)
        hypotheses.append(
            {
                key: item[key]
                for key in (
                    "hypothesis_id",
                    "mathematical_cluster",
                    "claim",
                    "signal_formula",
                    "required_columns",
                    "fixed_parameters",
                    "strategy_specific_gates",
                )
            }
        )
    common = dict(protocol["common_contract"])  # type: ignore[arg-type]
    common.pop("point_in_time_warning", None)
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": materialized_next_cohort_protocol_hash(protocol),
        "active_hypotheses": hypotheses,
        "kiv_pair_contract": protocol["kiv_hypotheses"],
        "validation": protocol["validation"],
        "common_contract": common,
        "approved_resolution_issue_ids": protocol["revision_lineage"][  # type: ignore[index]
            "resolution_issue_ids"
        ],
        "targeted_question": (
            "Do the approved Draft 02 repairs remove the recorded design blockers without adding "
            "a hidden trial, parameter search, unsupported data field, or execution claim? A veto "
            "requires one concrete defect that still exists in this exact draft. Suggestions to "
            "change signs, horizons, thresholds, strategies, or model families are out of scope."
        ),
        "result_information_available": False,
        "strategy_trials_reserved": False,
        "backtests_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize approved Malaysia next-cohort Draft 02 and run three local DeepSeek resolution roles."
    )
    parser.add_argument("--base-protocol", type=Path, default=BASE_PROTOCOL)
    parser.add_argument("--revision-manifest", type=Path, default=REVISION_MANIFEST)
    parser.add_argument("--resolution-root", type=Path, default=RESOLUTION_ROOT)
    parser.add_argument("--model", default=LOCAL_MODEL)
    args = parser.parse_args()

    strategy_ledger_before = _file_sha256(LIFETIME_LEDGER)
    protocol = materialize_next_cohort_protocol(args.base_protocol, args.revision_manifest)
    protocol_hash = materialized_next_cohort_protocol_hash(protocol)
    materialized_path = _persist_json(
        protocol, args.resolution_root / "materialized_protocol.json"
    )
    context = _targeted_context(protocol)
    role_ids = tuple(
        protocol["revision_lineage"]["targeted_resolution"]["role_ids"]  # type: ignore[index]
    )
    if role_ids != (
        "selection_bias_validation_specialist",
        "time_series_dependence_specialist",
        "high_frequency_econometrics_specialist",
    ):
        raise ValueError("Draft 02 targeted resolution roles drifted")
    panel = load_specialist_panel()
    roles = tuple(panel.by_id(role_id) for role_id in role_ids)
    ledger = TrialLedger(
        args.resolution_root / "resolution_ledger.sqlite",
        args.resolution_root / "resolution_evidence",
    )
    reservation = ledger.reserve_trial(
        idempotency_key=f"QFORGE20-NEXT-DRAFT02-RESOLUTION:{protocol_hash}:{args.model}",
        campaign="qforge20_malaysia_next_cohort_draft02_resolution",
        family="protocol_resolution",
        hypothesis_id="MYS-H-QFORGE20-NEXT-DRAFT02-RESOLUTION",
        candidate_id="MYS-C-QFORGE20-NEXT-DRAFT02-RESOLUTION",
        code_hash=_digest("qforge20_next_cohort_draft02_resolution_v1"),
        data_hash=str(protocol["source_hashes"]["daily_features_sha256"]),
        config_hash=protocol_hash,
        prompt_hash=_digest(context),
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
                "role_ids": list(role_ids),
                "model": args.model,
            },
        )
    dossier = MalaysiaReviewDossier(
        candidate_id="MYS-C-QFORGE20-NEXT-DRAFT02-RESOLUTION",
        hypothesis_hash=_digest(context["targeted_question"]),
        candidate_hash=protocol_hash,
        problem=(
            "Resolve only the eight recorded Malaysia next-cohort Draft 01 advisory issues against "
            "the exact approved Draft 02. Verify that pairs remain KIV, four hypotheses form one "
            "sealed Holm family, lag and quality exclusions are explicit, calendar disagreement is "
            "logged, capacity notional is fixed, and opening-auction costs are conservative. "
            "Identify a blocker only if a reproducible defect remains in this exact draft."
        ),
        tags=(
            "selection_bias",
            "time_series",
            "multiple_testing",
            "liquidity",
            "market_microstructure",
            "transaction_cost",
        ),
        evidence_hashes={
            "protocol": protocol_hash,
            "draft01": str(protocol["revision_lineage"]["base_protocol_canonical_sha256"]),
            "advisory_report": str(protocol["revision_lineage"]["advisory_report_sha256"]),
            "revision_manifest": str(protocol["revision_lineage"]["revision_manifest_file_sha256"]),
        },
        research_context=context,
        review_phase="pre_backtest_design",
    )
    runtime = SpecialistReviewRuntime(
        LocalModelRouter(
            OllamaProvider(),
            policy=LocalRoutingPolicy(
                models={capability: args.model for capability in ModelCapability}
            ),
        ),
        ledger=ledger,
        panel=panel,
        retry_infrastructure_failures=False,
    )
    disposition = runtime.review_roles(
        dossier,
        roles,
        trial_id=reservation.trial_id,
        require_progression_panel=True,
    )
    failures = [item for item in disposition.attempts if item.status != "success"]
    manifest = ledger.replay_manifest(reservation.trial_id)
    outcome_exists = any(item["event_type"] == "outcome" for item in manifest["events"])
    if failures:
        ledger.record_research_step(
            reservation.trial_id,
            event_type="stop",
            status="blocked",
            payload={
                "advisory_only": True,
                "failed_roles": [item.specialist_id for item in failures],
                "failure_codes": [item.failure_code for item in failures],
            },
        )
    elif not outcome_exists:
        ledger.record_outcome(
            reservation.trial_id,
            status="success",
            payload={
                "targeted_resolution_reviews_complete": True,
                "protocol_id": protocol["protocol_id"],
                "protocol_hash": protocol_hash,
                "role_count": 3,
                "strategy_trials_reserved": False,
                "backtests_started": False,
            },
        )
    reviews = [item.review for item in disposition.attempts if item.review is not None]
    strategy_ledger_after = _file_sha256(LIFETIME_LEDGER)
    report = {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "status": "resolution_reviews_complete" if not failures else "resolution_reviews_blocked",
        "materialized_protocol_file_sha256": _file_sha256(materialized_path),
        "route": {
            "privacy_class": "local_only",
            "model": args.model,
            "hosted_cost_usd": 0.0,
            "independent_from_draft01_model": args.model != "qwen3.5:35b-mlx",
        },
        "trial_id": reservation.trial_id,
        "role_ids": list(role_ids),
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
        "strategy_trial_safety": {
            "strategy_trials_reserved": False,
            "backtests_started": False,
            "lifetime_ledger_sha256_before": strategy_ledger_before,
            "lifetime_ledger_sha256_after": strategy_ledger_after,
            "lifetime_ledger_unchanged": strategy_ledger_before == strategy_ledger_after,
        },
        "reviews": [asdict(item) for item in reviews],
        "advisory_disposition": {
            "progression_flag": disposition.progression_allowed,
            "veto_flag": disposition.vetoed,
            "reasons": list(disposition.reasons),
            "conflicts": list(disposition.conflicts),
            "model_vote_has_no_freeze_authority": True,
        },
    }
    target = _persist_json(report, args.resolution_root / "resolution_report.json")
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
                "lifetime_ledger_unchanged": report["strategy_trial_safety"][
                    "lifetime_ledger_unchanged"
                ],
                "report_path": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
