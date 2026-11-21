from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.research.bounded_loop import BacktestEvidence, MalaysiaUniverseContract
from quant_lab.research.contracts import CandidateSpec
from quant_lab.research.institutional_workflow import (
    DeterministicMalaysiaInstitutionalAuditor,
    MalaysiaAuditInputs,
)
from quant_lab.research.malaysia_cohort_analysis import _holm_adjust, _one_sided_hac_pvalue
from quant_lab.research.malaysia_next_cohort import (
    ILLIQUIDITY_ID,
    OPENING_GAP_ID,
    PRICE_VOLUME_ID,
    TURN_OF_MONTH_ID,
    MalaysiaNextCohortEngine,
    NextCohortComputation,
    return_series_hash,
)
from quant_lab.research.next_cohort_protocol import (
    materialize_next_cohort_protocol,
    materialized_next_cohort_protocol_hash,
)
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.metrics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio
from quant_lab.validation.overfit import probability_of_backtest_overfitting


L2_ROOT = Path(
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/"
    "ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)
COHORT_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft02"
)
VAULT_ROOT = Path("artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault")
BASE_PROTOCOL = Path("configs/qforge20_malaysia_next_cohort_prereg_draft01.json")
REVISION_MANIFEST = Path("configs/qforge20_malaysia_next_cohort_draft02_revision.json")
EXPECTED_PROTOCOL_HASH = "5f7e02bc669391e22cc712b0827c7c760027dfe5164c10633968b06be909a79d"


class _AuditSource:
    def __init__(self, inputs: dict[str, MalaysiaAuditInputs]) -> None:
        self.inputs = inputs

    def institutional_audit_inputs(self, candidate_id: str) -> MalaysiaAuditInputs:
        return self.inputs[candidate_id]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_json(payload: object, path: Path) -> Path:
    content = _canonical(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def _candidate_id(strategy_id: str) -> str:
    return f"MYS-C-{strategy_id}"


def _has_event(ledger: TrialLedger, trial_id: str, event_type: str) -> bool:
    return any(
        event["event_type"] == event_type
        for event in ledger.replay_manifest(trial_id)["events"]
    )


def _finite_correlation(left: pd.Series, right: pd.Series) -> float:
    value = float(left.corr(right))
    return value if math.isfinite(value) else 1.0


def _gap_metrics(computation: NextCohortComputation) -> dict[str, float]:
    turnover = float(computation.u3_turnover.sum())
    break_even = (
        float(computation.u3_gross_returns.sum()) / turnover * 10_000.0
        if turnover > 0.0
        else 0.0
    )
    annual = computation.u3_base_returns.groupby(computation.u3_base_returns.index.year).sum()
    positive_year_fraction = float((annual > 0.0).mean()) if len(annual) else 0.0
    return {
        "break_even_total_cost_bps_one_way": break_even,
        "positive_year_fraction": positive_year_fraction,
    }


def _evidence_payload(
    computation: NextCohortComputation,
    *,
    protocol_hash: str,
    metrics: dict[str, Any],
    gates: dict[str, bool],
    audit: dict[str, Any],
) -> dict[str, Any]:
    daily_u3 = []
    for date in computation.u3_base_returns.index:
        weights = computation.u3_weights.loc[date]
        daily_u3.append(
            {
                "date": pd.Timestamp(date).isoformat(),
                "gross_return": float(computation.u3_gross_returns.loc[date]),
                "base_cost_return": float(computation.u3_base_returns.loc[date]),
                "stress_cost_return": float(computation.u3_stress_returns.loc[date]),
                "turnover": float(computation.u3_turnover.loc[date]),
                "nonzero_weights": {
                    str(ticker): float(value)
                    for ticker, value in weights.items()
                    if float(value) != 0.0
                },
            }
        )
    return {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "hypothesis_id": computation.hypothesis_id,
        "strategy_id": computation.strategy_id,
        "candidate_id": _candidate_id(computation.strategy_id),
        "family": computation.family,
        "calendar": {
            "observations": len(computation.u3_base_returns),
            "start": computation.u3_base_returns.index.min().isoformat(),
            "end": computation.u3_base_returns.index.max().isoformat(),
            "u3_u2_identical": computation.u3_base_returns.index.equals(
                computation.u2_base_returns.index
            ),
        },
        "series_hashes": {
            "u3_gross": return_series_hash(computation.u3_gross_returns),
            "u3_base": return_series_hash(computation.u3_base_returns),
            "u3_stress": return_series_hash(computation.u3_stress_returns),
            "u3_turnover": return_series_hash(computation.u3_turnover),
            "u2_base": return_series_hash(computation.u2_base_returns),
            "u2_stress": return_series_hash(computation.u2_stress_returns),
        },
        "selection_diagnostics": asdict(computation.selection_diagnostics),
        "lag_and_quality_lineage": computation.lineage_receipt,
        "calendar_receipt": computation.calendar_receipt,
        "metrics": metrics,
        "mandatory_gates": gates,
        "institutional_audit": audit,
        "daily_u3": daily_u3,
        "daily_u2": [
            {
                "date": pd.Timestamp(date).isoformat(),
                "base_cost_return": float(computation.u2_base_returns.loc[date]),
                "stress_cost_return": float(computation.u2_stress_returns.loc[date]),
            }
            for date in computation.u2_base_returns.index
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reserve and run the exact frozen four-strategy QForge-20 Malaysia wave once."
    )
    parser.add_argument("--l2-root", type=Path, default=L2_ROOT)
    parser.add_argument("--cohort-root", type=Path, default=COHORT_ROOT)
    parser.add_argument("--vault-root", type=Path, default=VAULT_ROOT)
    args = parser.parse_args()

    materialized_path = args.cohort_root / "materialized_protocol.json"
    resolution_path = args.cohort_root / "resolution_report.json"
    freeze_path = args.cohort_root / "human_freeze_receipt.json"
    for required in (materialized_path, resolution_path, freeze_path):
        if not required.is_file():
            raise FileNotFoundError(f"required frozen-cohort file is missing: {required}")
    protocol = json.loads(materialized_path.read_text(encoding="utf-8"))
    reconstructed = materialize_next_cohort_protocol(BASE_PROTOCOL, REVISION_MANIFEST)
    if protocol != reconstructed:
        raise RuntimeError("materialized next-cohort protocol differs from deterministic reconstruction")
    protocol_hash = materialized_next_cohort_protocol_hash(protocol)
    if protocol_hash != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("next-cohort protocol differs from the approved canonical hash")
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    if (
        resolution.get("protocol_hash") != protocol_hash
        or resolution.get("completed_role_count") != 3
        or resolution.get("verdict_counts") != {"pass": 3}
        or resolution.get("veto_count") != 0
        or resolution.get("model_calls", {}).get("external_cost_usd") != 0.0
    ):
        raise RuntimeError("targeted local resolution did not authorize progression")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    strategy_ids = [item["strategy_id"] for item in protocol["hypotheses"]]
    if (
        freeze.get("human_freeze_approved") is not True
        or freeze.get("protocol_canonical_sha256") != protocol_hash
        or freeze.get("resolution_report_sha256") != _file_sha256(resolution_path)
        or freeze.get("strategy_ids") != strategy_ids
        or freeze.get("authorized_lifetime_strategy_attempts") != 4
        or freeze.get("dry_run_authorized") is not False
        or freeze.get("live_execution_authorized") is not False
    ):
        raise RuntimeError("human freeze receipt does not authorize this exact four-strategy wave")

    final_report_path = args.cohort_root / "frozen_cohort_report.json"
    tiers_path = args.l2_root / "eligible_universe_tiers.csv"
    features_path = args.l2_root / "daily_features.parquet"
    universe = MalaysiaUniverseContract.from_tiers_csv(tiers_path)
    if universe.content_hash() != protocol["common_contract"]["universe_contract_sha256"]:
        raise RuntimeError("Malaysia universe contract hash drifted")
    data_receipt = {
        "daily_features_file_sha256": _file_sha256(features_path),
        "eligible_universe_tiers_sha256": _file_sha256(tiers_path),
        "universe_contract_sha256": universe.content_hash(),
        "research_cutoff": "2026-07-06",
    }
    data_hash = _digest(data_receipt)
    ledger = TrialLedger(args.vault_root / "trial_ledger.sqlite", args.vault_root / "evidence")
    matrix_before = ledger.candidate_return_matrix(campaign="malaysia")
    counts_before = ledger.lifetime_counts(campaign="malaysia")
    code_hash = _file_sha256(Path("src/quant_lab/research/malaysia_next_cohort.py"))
    freeze_hash = _file_sha256(freeze_path)
    model_hash = _digest({"engine": "deterministic", "model_calls": 0})
    requests = [
        {
            "idempotency_key": f"QFORGE20-MYS-NEXT:{protocol_hash}:{item['strategy_id']}",
            "campaign": "malaysia",
            "family": str(item["mathematical_cluster"]),
            "hypothesis_id": str(item["hypothesis_id"]),
            "candidate_id": _candidate_id(str(item["strategy_id"])),
            "parent_trial_id": None,
            "code_hash": code_hash,
            "data_hash": data_hash,
            "config_hash": protocol_hash,
            "prompt_hash": freeze_hash,
            "model_hash": model_hash,
        }
        for item in protocol["hypotheses"]
    ]
    reservations = ledger.reserve_trials_atomically(requests)
    if any(item.created for item in reservations) and not all(item.created for item in reservations):
        raise RuntimeError("four-strategy reservation was not atomic")
    if not any(item.created for item in reservations) and final_report_path.is_file():
        prior = json.loads(final_report_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "replayed_existing_complete_report",
                    "protocol_hash": protocol_hash,
                    "cohort_status": prior["status"],
                    "report_path": str(final_report_path),
                },
                indent=2,
            )
        )
        return
    counts_after_reservation = ledger.lifetime_counts(campaign="malaysia")
    if counts_after_reservation["reserved"] - counts_before["reserved"] not in {0, 4}:
        raise RuntimeError("Malaysia lifetime denominator did not change by exactly zero or four")
    reservation_by_strategy = {
        strategy_id: reservation for strategy_id, reservation in zip(strategy_ids, reservations)
    }
    for item in protocol["hypotheses"]:
        reservation = reservation_by_strategy[item["strategy_id"]]
        if reservation.created:
            ledger.record_research_step(
                reservation.trial_id,
                event_type="hypothesis_intake",
                status="admitted",
                payload={
                    "protocol_hash": protocol_hash,
                    "freeze_receipt_sha256": freeze_hash,
                    "hypothesis_id": item["hypothesis_id"],
                    "strategy_id": item["strategy_id"],
                    "pre_registered": True,
                    "performance_results_sealed": True,
                    "model_calls": 0,
                },
            )
    reservation_report = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "freeze_receipt_sha256": freeze_hash,
        "reservation_was_atomic": True,
        "authorized_attempt_count": 4,
        "new_reservation_count": sum(item.created for item in reservations),
        "before_lifetime_counts": counts_before,
        "after_lifetime_counts": counts_after_reservation,
        "strategies": [
            {
                "strategy_id": strategy_id,
                "trial_id": reservation_by_strategy[strategy_id].trial_id,
                "created": reservation_by_strategy[strategy_id].created,
            }
            for strategy_id in strategy_ids
        ],
        "backtests_started_at_reservation_receipt": False,
        "performance_results_unsealed": False,
    }
    reservation_report_path = _persist_json(
        reservation_report, args.cohort_root / "four_strategy_reservation_report.json"
    )

    engine = MalaysiaNextCohortEngine(
        daily_features_path=features_path,
        universe=universe,
        protocol=protocol,
    )
    try:
        computations = engine.compute_all()
    except Exception as exc:
        for strategy_id in strategy_ids:
            trial_id = reservation_by_strategy[strategy_id].trial_id
            if not _has_event(ledger, trial_id, "outcome"):
                ledger.record_outcome(
                    trial_id,
                    status="failure",
                    failure_class="ENGINEERING_TERMINAL_FAILURE",
                    payload={
                        "error_type": type(exc).__name__,
                        "performance_results_unsealed": False,
                        "dry_run_completed": False,
                    },
                )
        _persist_json(
            {
                "schema_version": 1,
                "protocol_hash": protocol_hash,
                "status": "engineering_terminal_failure",
                "all_four_attempts_reserved": True,
                "performance_results_unsealed": False,
                "error_type": type(exc).__name__,
            },
            final_report_path,
        )
        raise
    if len(computations) != 4 or [item.strategy_id for item in computations] != strategy_ids:
        raise RuntimeError("complete four-strategy computation did not reach terminal state")

    for computation in computations:
        reservation = reservation_by_strategy[computation.strategy_id]
        if not _has_event(ledger, reservation.trial_id, "candidate_returns"):
            ledger.record_candidate_returns(
                reservation.trial_id,
                candidate_id=_candidate_id(computation.strategy_id),
                family=computation.family,
                returns=computation.u3_base_returns,
            )
    matrix_after = ledger.candidate_return_matrix(campaign="malaysia")
    if matrix_after.shape[1] != matrix_before.shape[1] + 4:
        raise RuntimeError("complete next-cohort wave did not add exactly four active return series")
    lifetime_counts = ledger.lifetime_counts(campaign="malaysia")
    pbo = float(probability_of_backtest_overfitting(matrix_after, n_splits=8, strict=True))
    lifetime_sharpes = [float(sharpe_ratio(matrix_after[column])) for column in matrix_after]
    lifetime_denominator = lifetime_sharpes + [0.0] * max(
        0, lifetime_counts["lifetime_attempts"] - len(lifetime_sharpes)
    )
    raw_pvalues = {
        item.strategy_id: float(
            _one_sided_hac_pvalue(
                item.primary_comparison_series
                if item.primary_comparison_series is not None
                else item.u3_base_returns
            )
        )
        for item in computations
    }
    holm_pvalues = _holm_adjust(raw_pvalues)
    audit_inputs = {
        _candidate_id(item.strategy_id): engine.audit_inputs(item) for item in computations
    }
    auditor = DeterministicMalaysiaInstitutionalAuditor(_AuditSource(audit_inputs))
    momentum_factor = engine.registered_momentum_factor()

    metrics_by_strategy: dict[str, dict[str, Any]] = {}
    audit_by_strategy: dict[str, dict[str, Any]] = {}
    gates_by_strategy: dict[str, dict[str, bool]] = {}
    computation_by_strategy = {item.strategy_id: item for item in computations}
    protocol_by_strategy = {item["strategy_id"]: item for item in protocol["hypotheses"]}
    for computation in computations:
        strategy_id = computation.strategy_id
        candidate_id = _candidate_id(strategy_id)
        net_ir = float(sharpe_ratio(computation.u3_base_returns))
        stress_ir = float(sharpe_ratio(computation.u3_stress_returns))
        u2_ir = float(sharpe_ratio(computation.u2_base_returns))
        drawdown = float(max_drawdown(computation.u3_base_returns))
        annual_turnover = float(computation.u3_turnover.mean() * 252.0)
        maximum_weight = float(computation.u3_weights.abs().max().max())
        dsr = float(
            deflated_sharpe_ratio(
                computation.u3_base_returns,
                lifetime_denominator,
                periods_per_year=252,
            )
        )
        capacity = engine.capacity_metrics(computation)
        gap = _gap_metrics(computation) if strategy_id == OPENING_GAP_ID else None
        momentum_correlation = _finite_correlation(
            computation.u3_base_returns, momentum_factor
        )
        metrics: dict[str, Any] = {
            "raw_information_ratio": float(sharpe_ratio(computation.u3_gross_returns)),
            "net_information_ratio": net_ir,
            "stress_net_information_ratio": stress_ir,
            "u2_net_information_ratio": u2_ir,
            "deflated_sharpe_probability": dsr,
            "complete_lifetime_pbo": pbo,
            "maximum_drawdown": drawdown,
            "annual_turnover": annual_turnover,
            "maximum_asset_weight": maximum_weight,
            "hac_one_sided_pvalue": raw_pvalues[strategy_id],
            "holm_adjusted_one_sided_pvalue": holm_pvalues[strategy_id],
            "registered_momentum_return_correlation": momentum_correlation,
            "capacity": capacity,
            "gap_execution": gap,
            "u3_base_return_hash": return_series_hash(computation.u3_base_returns),
        }
        candidate = CandidateSpec(
            candidate_id=candidate_id,
            hypothesis_id=computation.hypothesis_id,
            family=computation.family,
            implementation_ref=(
                "quant_lab.research.malaysia_next_cohort:MalaysiaNextCohortEngine"
            ),
            parameters={
                "strategy_id": strategy_id,
                "protocol_hash": protocol_hash,
                "fixed_parameters": protocol_by_strategy[strategy_id]["fixed_parameters"],
            },
            evidence=(),
            provenance_hashes={"protocol": protocol_hash, "data": data_hash},
        )
        candidate.validate()
        evidence = BacktestEvidence(
            candidate_id=candidate_id,
            u3_count=len(universe.u3),
            u2_count=len(universe.u2),
            base_cost_bps_one_way=20.0,
            stress_cost_bps_one_way=40.0,
            raw_sharpe=metrics["raw_information_ratio"],
            net_oos_ir=net_ir,
            deflated_sharpe_probability=dsr,
            pbo=pbo,
            max_drawdown=drawdown,
            turnover=annual_turnover,
            stress_net_oos_ir=stress_ir,
            max_asset_weight=maximum_weight,
            u2_net_oos_ir=u2_ir,
            target_weights={
                str(ticker): float(value)
                for ticker, value in computation.u3_weights.iloc[-1].items()
                if float(value) != 0.0
            },
            reference_prices={},
            config_hash=protocol_hash,
        )
        evidence.validate(universe)
        audit = auditor.audit(
            candidate,
            evidence,
            trial_id=reservation_by_strategy[strategy_id].trial_id,
            universe=universe,
            config=None,
        )
        audit.validate(candidate_id)
        audit_payload = audit.to_dict()
        common_gates = {
            "positive_net_information_ratio": net_ir > 0.0,
            "positive_stress_net_information_ratio": stress_ir > 0.0,
            "positive_u2_net_information_ratio": u2_ir > 0.0,
            "deflated_sharpe_probability_at_least_0_95": dsr >= 0.95,
            "complete_lifetime_pbo_at_most_0_20": pbo <= 0.20,
            "maximum_drawdown_at_least_minus_0_20": drawdown >= -0.20,
            "maximum_asset_weight_at_most_0_25": maximum_weight <= 0.25,
            "holm_adjusted_primary_pvalue_at_most_0_05": (
                holm_pvalues[strategy_id] <= 0.05
            ),
            "reference_trade_participation_at_most_0_05_of_lagged_adv": (
                capacity["reference_maximum_trade_participation"] <= 0.05
            ),
            "maximum_positive_pnl_concentration_at_most_0_35": (
                capacity["maximum_positive_pnl_concentration"] <= 0.35
            ),
            "frozen_regime_gate": (
                audit.regime_count >= 4
                and audit.min_regime_observations >= 63
                and audit.positive_regime_fraction >= 4.0 / 6.0
                and audit.worst_regime_ir > -1.0
            ),
            "frozen_factor_residual_gate": (
                audit.factor_residual_ir > 0.0
                and audit.factor_alpha_t_stat > 2.0
                and audit.factor_r_squared <= 0.80
            ),
        }
        strategy_gates: dict[str, bool] = {}
        if strategy_id == PRICE_VOLUME_ID:
            strategy_gates = {
                "positive_factor_residual_ir_after_registered_momentum_exposure": (
                    audit.factor_residual_ir > 0.0
                ),
                "absolute_return_correlation_with_registered_momentum_below_0_90": (
                    abs(momentum_correlation) < 0.90
                ),
            }
        elif strategy_id == ILLIQUIDITY_ID:
            strategy_gates = {
                "annual_turnover_at_most_3_0": annual_turnover <= 3.0,
            }
        elif strategy_id == TURN_OF_MONTH_ID:
            contrast = computation.primary_comparison_series
            strategy_gates = {
                "inside_window_mean_return_greater_than_outside_window": (
                    contrast is not None and float(contrast.mean()) > 0.0
                ),
            }
        elif strategy_id == OPENING_GAP_ID:
            assert gap is not None
            strategy_gates = {
                "positive_net_ir_after_45_bps_one_way_total_cost": net_ir > 0.0,
                "positive_stress_ir_after_65_bps_one_way_total_cost": stress_ir > 0.0,
                "break_even_slippage_bps_one_way_at_least_65": (
                    gap["break_even_total_cost_bps_one_way"] >= 65.0
                ),
                "positive_year_fraction_at_least_0_60": (
                    gap["positive_year_fraction"] >= 0.60
                ),
            }
        metrics_by_strategy[strategy_id] = metrics
        audit_by_strategy[strategy_id] = audit_payload
        gates_by_strategy[strategy_id] = {**common_gates, **strategy_gates}

    evidence_hashes: dict[str, str] = {}
    final_passes: dict[str, bool] = {}
    for strategy_id in strategy_ids:
        computation = computation_by_strategy[strategy_id]
        gates = gates_by_strategy[strategy_id]
        passed = all(gates.values())
        final_passes[strategy_id] = passed
        evidence_payload = _evidence_payload(
            computation,
            protocol_hash=protocol_hash,
            metrics=metrics_by_strategy[strategy_id],
            gates=gates,
            audit=audit_by_strategy[strategy_id],
        )
        evidence_content = _canonical(evidence_payload).encode("utf-8")
        evidence_record = ledger.store_evidence(evidence_content, extension=".json")
        evidence_hashes[strategy_id] = evidence_record.sha256
        trial_id = reservation_by_strategy[strategy_id].trial_id
        if not _has_event(ledger, trial_id, "backtest"):
            ledger.record_research_step(
                trial_id,
                event_type="backtest",
                status="pass" if passed else "failed_gates",
                payload={
                    "strategy_id": strategy_id,
                    "evidence_hash": evidence_record.sha256,
                    "research_gate_pass": passed,
                    "results_unsealed_after_complete_four_strategy_terminal_state": True,
                    "dry_run_completed": False,
                },
                evidence_content=evidence_content,
            )
        if not _has_event(ledger, trial_id, "institutional_audit"):
            audit_content = _canonical(audit_by_strategy[strategy_id]).encode("utf-8")
            ledger.record_research_step(
                trial_id,
                event_type="institutional_audit",
                status=(
                    "pass"
                    if gates["frozen_regime_gate"] and gates["frozen_factor_residual_gate"]
                    else "failed_gates"
                ),
                payload={
                    "strategy_id": strategy_id,
                    "audit_hash": audit_by_strategy[strategy_id]["audit_hash"],
                },
                evidence_content=audit_content,
            )
        if not _has_event(ledger, trial_id, "outcome"):
            failed = [key for key, value in gates.items() if not value]
            ledger.record_outcome(
                trial_id,
                status="success" if passed else "failure",
                failure_class=None if passed else failed[0].upper()[:80],
                payload={
                    "strategy_id": strategy_id,
                    "research_gate_pass": passed,
                    "failed_gate_ids": failed,
                    "dry_run_completed": False,
                    "terminal_holdout_released": False,
                    "research_pass_grants_live_trading": False,
                },
                evidence_sha256=evidence_record.sha256,
            )

    final_counts = ledger.lifetime_counts(campaign="malaysia")
    report = {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "market": "Malaysia",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "status": "frozen_four_strategy_wave_complete_stop_for_evaluation",
        "research_cutoff": "2026-07-06",
        "research_only_current_survivor_evidence": True,
        "human_freeze_receipt_sha256": freeze_hash,
        "targeted_resolution": {
            "roles": resolution["role_ids"],
            "verdict_counts": resolution["verdict_counts"],
            "veto_count": resolution["veto_count"],
            "model": resolution["route"]["model"],
            "external_cost_usd": resolution["model_calls"]["external_cost_usd"],
        },
        "data_receipt": data_receipt,
        "data_hash": data_hash,
        "code_hash": code_hash,
        "reservation_report_file_sha256": _file_sha256(reservation_report_path),
        "lifetime_counts_before_reservation": counts_before,
        "lifetime_counts_after_completion": final_counts,
        "active_return_series_before": matrix_before.shape[1],
        "active_return_series_after": matrix_after.shape[1],
        "complete_lifetime_pbo": pbo,
        "multiplicity_family": {
            "method": "Holm",
            "primary_hypothesis_count": 4,
            "raw_one_sided_hac_pvalues": raw_pvalues,
            "adjusted_one_sided_pvalues": holm_pvalues,
        },
        "strategies": {
            strategy_id: {
                "trial_id": reservation_by_strategy[strategy_id].trial_id,
                "hypothesis_id": computation_by_strategy[strategy_id].hypothesis_id,
                "family": computation_by_strategy[strategy_id].family,
                "metrics": metrics_by_strategy[strategy_id],
                "mandatory_gates": gates_by_strategy[strategy_id],
                "failed_gate_ids": [
                    key for key, value in gates_by_strategy[strategy_id].items() if not value
                ],
                "selection_diagnostics": asdict(
                    computation_by_strategy[strategy_id].selection_diagnostics
                ),
                "lag_and_quality_lineage": computation_by_strategy[
                    strategy_id
                ].lineage_receipt,
                "calendar_receipt": computation_by_strategy[strategy_id].calendar_receipt,
                "institutional_audit": audit_by_strategy[strategy_id],
                "final_research_gate_pass": final_passes[strategy_id],
                "evidence_sha256": evidence_hashes[strategy_id],
            }
            for strategy_id in strategy_ids
        },
        "cohort_decision": {
            "research_gate_pass_count": sum(final_passes.values()),
            "strategy_count": 4,
            "stop_for_evaluation": True,
            "parameter_optimization_performed": False,
            "post_hoc_combination_search_performed": False,
            "failed_strategy_mutation_performed": False,
        },
        "execution_state": {
            "performance_results_unsealed": True,
            "terminal_holdout_released": False,
            "dry_run_started": False,
            "live_execution_authorized": False,
            "model_calls_during_backtest": 0,
            "external_cost_usd_during_backtest": 0.0,
        },
    }
    report_path = _persist_json(report, final_report_path)
    print(
        json.dumps(
            {
                "protocol_hash": protocol_hash,
                "status": report["status"],
                "lifetime_attempts": final_counts["lifetime_attempts"],
                "active_return_series": matrix_after.shape[1],
                "complete_lifetime_pbo": pbo,
                "research_gate_pass_count": sum(final_passes.values()),
                "dry_run_started": False,
                "report_path": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
