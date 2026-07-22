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
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort
from quant_lab.research.malaysia_v2_experiment import (
    MalaysiaV2PairedEngine,
    V2ArmComputation,
    dataframe_hash,
    paired_moving_block_bootstrap,
    return_series_hash,
)
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.research.v2_overlay_protocol import (
    materialize_v2_overlay_protocol,
    v2_overlay_protocol_hash,
)
from quant_lab.validation.metrics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio
from quant_lab.validation.overfit import probability_of_backtest_overfitting


L2_ROOT = Path(
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/"
    "ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)
V2_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02"
)
VAULT_ROOT = Path("artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault")
BASE_PROTOCOL = Path("configs/qforge20_malaysia_v2_overlay_draft.json")
REVISION_MANIFEST = Path("configs/qforge20_malaysia_v2_overlay_draft02_revision.json")
EXPECTED_PROTOCOL_HASH = "9dd12ad0fb963587112989dc5cd31c88f7130a534e1149cb9af954046e5f110c"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _active_parent_series(
    matrix: pd.DataFrame,
    strategy_ids: list[str],
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    series: dict[str, pd.Series] = {}
    trial_ids: dict[str, str] = {}
    for strategy_id in strategy_ids:
        prefix = f"MYS-C-{strategy_id}::"
        columns = [str(column) for column in matrix.columns if str(column).startswith(prefix)]
        if len(columns) != 1:
            raise RuntimeError(f"expected one active V1 parent return series for {strategy_id}")
        column = columns[0]
        series[strategy_id] = matrix[column].copy()
        trial_ids[strategy_id] = column.split("::", 1)[1]
    return series, trial_ids


def _arm_candidate_id(arm_id: str) -> str:
    return f"MYS-C-{arm_id}"


def _arm_evidence_payload(
    arm: V2ArmComputation,
    *,
    protocol_hash: str,
    metrics: dict[str, Any],
    gates: dict[str, bool],
    audit: dict[str, Any],
) -> dict[str, Any]:
    daily = []
    for date in arm.u3_base_returns.index:
        weights = arm.u3_weights.loc[date]
        daily.append(
            {
                "date": pd.Timestamp(date).isoformat(),
                "gross_return": float(arm.u3_gross_returns.loc[date]),
                "base_cost_return": float(arm.u3_base_returns.loc[date]),
                "stress_cost_return": float(arm.u3_stress_returns.loc[date]),
                "turnover": float(arm.u3_turnover.loc[date]),
                "portfolio_exposure_scalar": float(arm.u3_exposure.loc[date]),
                "risky_asset_weights": {
                    str(ticker): float(value)
                    for ticker, value in weights.items()
                    if float(value) != 0.0
                },
            }
        )
    return {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "arm_id": arm.arm_id,
        "experiment_id": arm.experiment_id,
        "strategy_id": arm.strategy_id,
        "family": arm.family,
        "arm": arm.arm,
        "calendar": {
            "observations": len(arm.u3_base_returns),
            "start": arm.u3_base_returns.index.min().isoformat(),
            "end": arm.u3_base_returns.index.max().isoformat(),
            "u3_u2_identical": arm.u3_base_returns.index.equals(arm.u2_base_returns.index),
        },
        "series_hashes": {
            "u3_gross": return_series_hash(arm.u3_gross_returns),
            "u3_base": return_series_hash(arm.u3_base_returns),
            "u3_stress": return_series_hash(arm.u3_stress_returns),
            "u3_turnover": return_series_hash(arm.u3_turnover),
            "u3_exposure": return_series_hash(arm.u3_exposure),
            "u2_base": return_series_hash(arm.u2_base_returns),
            "u2_stress": return_series_hash(arm.u2_stress_returns),
            "u3_weights": dataframe_hash(arm.u3_weights),
        },
        "selection_diagnostics": (
            asdict(arm.selection_diagnostics) if arm.selection_diagnostics is not None else None
        ),
        "metrics": metrics,
        "mandatory_gates": gates,
        "institutional_audit": audit,
        "daily_u3": daily,
        "daily_u2": [
            {
                "date": pd.Timestamp(date).isoformat(),
                "base_cost_return": float(arm.u2_base_returns.loc[date]),
                "stress_cost_return": float(arm.u2_stress_returns.loc[date]),
            }
            for date in arm.u2_base_returns.index
        ],
    }


class _ArmAuditSource:
    def __init__(self, inputs: dict[str, MalaysiaAuditInputs]) -> None:
        self.inputs = inputs

    def institutional_audit_inputs(self, candidate_id: str) -> MalaysiaAuditInputs:
        return self.inputs[candidate_id]


def _has_event(ledger: TrialLedger, trial_id: str, event_type: str) -> bool:
    return any(
        event["event_type"] == event_type for event in ledger.replay_manifest(trial_id)["events"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reserve and run the exact frozen six-arm QForge-20 Malaysia V2 cohort once."
    )
    parser.add_argument("--l2-root", type=Path, default=L2_ROOT)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--vault-root", type=Path, default=VAULT_ROOT)
    args = parser.parse_args()

    materialized_path = args.v2_root / "materialized_protocol.json"
    freeze_receipt_path = args.v2_root / "human_freeze_receipt.json"
    if not materialized_path.is_file() or not freeze_receipt_path.is_file():
        raise FileNotFoundError("Draft 02 materialization and human freeze receipt are required")
    protocol = json.loads(materialized_path.read_text(encoding="utf-8"))
    reconstructed = materialize_v2_overlay_protocol(BASE_PROTOCOL, REVISION_MANIFEST)
    if protocol != reconstructed:
        raise RuntimeError("materialized Draft 02 differs from deterministic reconstruction")
    protocol_hash = v2_overlay_protocol_hash(protocol)
    if protocol_hash != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("Draft 02 canonical hash differs from the human-frozen hash")
    freeze = json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
    if (
        not bool(freeze.get("human_freeze_approved"))
        or freeze.get("protocol_canonical_sha256") != protocol_hash
        or freeze.get("arm_ids") != [item["arm_id"] for item in protocol["arms"]]
    ):
        raise RuntimeError("human freeze receipt does not authorize this exact six-arm cohort")

    tiers_path = args.l2_root / "eligible_universe_tiers.csv"
    features_path = args.l2_root / "daily_features.parquet"
    universe = MalaysiaUniverseContract.from_tiers_csv(tiers_path)
    if universe.source_hash != _file_sha256(tiers_path):
        raise RuntimeError("Malaysia universe content hash drifted")
    data_receipt = {
        "daily_features_file_sha256": _file_sha256(features_path),
        "eligible_universe_tiers_sha256": _file_sha256(tiers_path),
        "universe_contract_sha256": universe.content_hash(),
        "research_cutoff": "2026-07-06",
    }
    data_hash = _digest(data_receipt)
    ledger = TrialLedger(args.vault_root / "trial_ledger.sqlite", args.vault_root / "evidence")
    matrix_before = ledger.candidate_return_matrix(campaign="malaysia")
    strategy_ids = [item["strategy_id"] for item in protocol["parent_strategies"]]
    parent_series, parent_trial_ids = _active_parent_series(matrix_before, strategy_ids)
    registry = {item.strategy_id: item for item in malaysia_atomic_cohort()}
    before_counts = ledger.lifetime_counts(campaign="malaysia")
    module_path = Path("src/quant_lab/research/malaysia_v2_experiment.py")
    code_hash = _file_sha256(module_path)
    freeze_receipt_hash = _file_sha256(freeze_receipt_path)
    model_hash = _digest({"engine": "deterministic", "model_calls": 0})
    requests = []
    for arm in protocol["arms"]:
        strategy = registry[arm["strategy_id"]]
        requests.append(
            {
                "idempotency_key": f"QFORGE20-MYS-V2:{protocol_hash}:{arm['arm_id']}",
                "campaign": "malaysia",
                "family": f"{strategy.family}_v2_{arm['arm']}",
                "hypothesis_id": f"MYS-H-{arm['arm_id']}",
                "candidate_id": _arm_candidate_id(arm["arm_id"]),
                "parent_trial_id": parent_trial_ids[arm["strategy_id"]],
                "code_hash": code_hash,
                "data_hash": data_hash,
                "config_hash": protocol_hash,
                "prompt_hash": freeze_receipt_hash,
                "model_hash": model_hash,
            }
        )
    reservations = ledger.reserve_trials_atomically(requests)
    if any(item.created for item in reservations) and not all(item.created for item in reservations):
        raise RuntimeError("six-arm reservation was not atomic")
    after_reservation_counts = ledger.lifetime_counts(campaign="malaysia")
    if after_reservation_counts["reserved"] - before_counts["reserved"] not in {0, 6}:
        raise RuntimeError("Malaysia lifetime denominator did not change by exactly zero or six")
    reservation_by_arm = {
        arm["arm_id"]: reservation for arm, reservation in zip(protocol["arms"], reservations)
    }
    for arm in protocol["arms"]:
        reservation = reservation_by_arm[arm["arm_id"]]
        if reservation.created:
            ledger.record_research_step(
                reservation.trial_id,
                event_type="hypothesis_intake",
                status="admitted",
                payload={
                    "protocol_hash": protocol_hash,
                    "freeze_receipt_sha256": freeze_receipt_hash,
                    "arm_id": arm["arm_id"],
                    "experiment_id": arm["experiment_id"],
                    "strategy_id": arm["strategy_id"],
                    "arm": arm["arm"],
                    "pre_registered": True,
                    "model_calls": 0,
                },
            )
    reservation_report = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "freeze_receipt_sha256": freeze_receipt_hash,
        "reservation_was_atomic": True,
        "new_reservation_count": sum(item.created for item in reservations),
        "before_lifetime_counts": before_counts,
        "after_lifetime_counts": after_reservation_counts,
        "arms": [
            {
                "arm_id": arm["arm_id"],
                "trial_id": reservation_by_arm[arm["arm_id"]].trial_id,
                "parent_trial_id": reservation_by_arm[arm["arm_id"]].parent_trial_id,
                "created": reservation_by_arm[arm["arm_id"]].created,
            }
            for arm in protocol["arms"]
        ],
        "backtests_started_at_reservation_receipt": False,
    }
    _persist_json(reservation_report, args.v2_root / "six_arm_reservation_report.json")

    engine = MalaysiaV2PairedEngine(
        daily_features_path=features_path,
        universe=universe,
        protocol=protocol,
    )
    controls = engine.compute_arms("control")
    control_replay: dict[str, Any] = {}
    for control in controls:
        expected = parent_series[control.strategy_id]
        exact_values = expected.index.equals(control.u3_base_returns.index) and np.array_equal(
            expected.to_numpy(dtype=float), control.u3_base_returns.to_numpy(dtype=float)
        )
        expected_hash = return_series_hash(expected)
        observed_hash = return_series_hash(control.u3_base_returns)
        control_replay[control.experiment_id] = {
            "strategy_id": control.strategy_id,
            "parent_trial_id": parent_trial_ids[control.strategy_id],
            "expected_return_hash": expected_hash,
            "observed_return_hash": observed_hash,
            "exact_calendar_and_value_match": exact_values,
            "hash_match": expected_hash == observed_hash,
        }
    if not all(item["exact_calendar_and_value_match"] for item in control_replay.values()):
        for arm in protocol["arms"]:
            trial_id = reservation_by_arm[arm["arm_id"]].trial_id
            if not _has_event(ledger, trial_id, "outcome"):
                ledger.record_outcome(
                    trial_id,
                    status="failure",
                    failure_class="CONTROL_RETURN_HASH_MISMATCH",
                    payload={
                        "global_stop": True,
                        "dry_run_completed": False,
                    },
                )
        _persist_json(
            {
                "schema_version": 1,
                "protocol_hash": protocol_hash,
                "status": "global_stop_control_return_hash_mismatch",
                "control_replay": control_replay,
                "challenger_backtests_started": False,
            },
            args.v2_root / "frozen_cohort_report.json",
        )
        raise RuntimeError("global stop: a V2 control return hash differs from frozen V1 evidence")

    challengers = engine.compute_arms("challenger")
    arm_by_id = {arm.arm_id: arm for arm in (*controls, *challengers)}
    ordered_arms = tuple(arm_by_id[item["arm_id"]] for item in protocol["arms"])
    for arm in ordered_arms:
        reservation = reservation_by_arm[arm.arm_id]
        ledger.record_candidate_returns(
            reservation.trial_id,
            candidate_id=_arm_candidate_id(arm.arm_id),
            family=f"{arm.family}_v2_{arm.arm}",
            returns=arm.u3_base_returns,
        )

    matrix_after = ledger.candidate_return_matrix(campaign="malaysia")
    lifetime_counts = ledger.lifetime_counts(campaign="malaysia")
    if matrix_after.shape[1] != matrix_before.shape[1] + 6:
        raise RuntimeError("complete V2 wave did not add exactly six active return series")
    lifetime_sharpes = [float(sharpe_ratio(matrix_after[column])) for column in matrix_after]
    lifetime_denominator = lifetime_sharpes + [
        0.0
    ] * max(0, lifetime_counts["lifetime_attempts"] - len(lifetime_sharpes))
    pbo = float(probability_of_backtest_overfitting(matrix_after, n_splits=8, strict=True))
    raw_pvalues = {
        arm.arm_id: float(_one_sided_hac_pvalue(arm.u3_base_returns)) for arm in ordered_arms
    }
    holm_pvalues = _holm_adjust(raw_pvalues)

    audit_source = _ArmAuditSource(
        {_arm_candidate_id(arm.arm_id): engine.audit_inputs(arm) for arm in ordered_arms}
    )
    auditor = DeterministicMalaysiaInstitutionalAuditor(audit_source)
    arm_metrics: dict[str, dict[str, Any]] = {}
    arm_gates: dict[str, dict[str, bool]] = {}
    arm_audits: dict[str, dict[str, Any]] = {}
    for arm in ordered_arms:
        candidate_id = _arm_candidate_id(arm.arm_id)
        net_ir = float(sharpe_ratio(arm.u3_base_returns))
        drawdown = float(max_drawdown(arm.u3_base_returns))
        annual_turnover = float(arm.u3_turnover.mean() * 252.0)
        stress_ir = float(sharpe_ratio(arm.u3_stress_returns))
        u2_ir = float(sharpe_ratio(arm.u2_base_returns))
        maximum_weight = float(arm.u3_weights.abs().max().max())
        dsr = float(
            deflated_sharpe_ratio(
                arm.u3_base_returns,
                lifetime_denominator,
                periods_per_year=252,
            )
        )
        metrics = {
            "raw_information_ratio": float(sharpe_ratio(arm.u3_gross_returns)),
            "net_information_ratio": net_ir,
            "deflated_sharpe_probability": dsr,
            "complete_lifetime_pbo": pbo,
            "maximum_drawdown": drawdown,
            "annual_turnover": annual_turnover,
            "stress_net_information_ratio": stress_ir,
            "maximum_asset_weight": maximum_weight,
            "u2_net_information_ratio": u2_ir,
            "hac_one_sided_pvalue": raw_pvalues[arm.arm_id],
            "holm_adjusted_one_sided_pvalue": holm_pvalues[arm.arm_id],
            "u3_base_return_hash": return_series_hash(arm.u3_base_returns),
        }
        candidate = CandidateSpec(
            candidate_id=candidate_id,
            hypothesis_id=f"MYS-H-{arm.arm_id}",
            family=arm.family,
            implementation_ref=(
                "quant_lab.research.malaysia_v2_experiment:MalaysiaV2PairedEngine"
            ),
            parameters={"arm_id": arm.arm_id, "protocol_hash": protocol_hash},
            evidence=(),
            provenance_hashes={
                "protocol": protocol_hash,
                "strategy": registry[arm.strategy_id].content_hash(),
            },
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
                for ticker, value in arm.u3_weights.iloc[-1].items()
                if float(value) != 0.0
            },
            reference_prices={},
            config_hash=protocol_hash,
        )
        evidence.validate(universe)
        audit = auditor.audit(
            candidate,
            evidence,
            trial_id=reservation_by_arm[arm.arm_id].trial_id,
            universe=universe,
            config=None,  # type: ignore[arg-type]
        )
        audit.validate(candidate_id)
        audit_payload = audit.to_dict()
        audit_failures = {item.value for item in audit.failure_codes()}
        insufficient = bool(
            arm.selection_diagnostics
            and arm.selection_diagnostics.insufficient_diversification_dates
        )
        gates = {
            "positive_net_information_ratio": net_ir > 0.0,
            "deflated_sharpe_probability_at_least_0_95": dsr >= 0.95,
            "pbo_at_most_0_20": pbo <= 0.20,
            "maximum_drawdown_at_least_minus_0_20": drawdown >= -0.20,
            "annual_turnover_at_most_3_0": annual_turnover <= 3.0,
            "positive_stress_net_information_ratio": stress_ir > 0.0,
            "maximum_asset_weight_at_most_0_25": maximum_weight <= 0.25,
            "positive_u2_net_information_ratio": u2_ir > 0.0,
            "holm_adjusted_one_sided_pvalue_at_most_0_05": (
                holm_pvalues[arm.arm_id] <= 0.05
            ),
            "sufficient_diversification_at_every_evaluated_rebalance": not insufficient,
            "regime_gate": "REGIME_INSTABILITY" not in audit_failures,
            "factor_residual_gate": "FACTOR_RESIDUAL_FAILURE" not in audit_failures,
            "decay_gate": "ALPHA_DECAY_FAILURE" not in audit_failures,
        }
        arm_metrics[arm.arm_id] = metrics
        arm_gates[arm.arm_id] = gates
        arm_audits[arm.arm_id] = audit_payload

    paired_results: dict[str, dict[str, Any]] = {}
    for experiment_id in [item["experiment_id"] for item in protocol["parent_strategies"]]:
        control = next(arm for arm in controls if arm.experiment_id == experiment_id)
        challenger = next(arm for arm in challengers if arm.experiment_id == experiment_id)
        control_metrics = arm_metrics[control.arm_id]
        challenger_metrics = arm_metrics[challenger.arm_id]
        bootstrap = paired_moving_block_bootstrap(
            control.u3_base_returns,
            challenger.u3_base_returns,
        )
        control_turnover = float(control_metrics["annual_turnover"])
        challenger_turnover = float(challenger_metrics["annual_turnover"])
        turnover_ratio = (
            challenger_turnover / control_turnover
            if control_turnover > 0.0
            else (0.0 if challenger_turnover == 0.0 else float("inf"))
        )
        gates = {
            "control_return_hash_match": bool(control_replay[experiment_id]["hash_match"]),
            "net_ir_retention_at_least_0_80": (
                float(challenger_metrics["net_information_ratio"])
                - 0.80 * float(control_metrics["net_information_ratio"])
                >= 0.0
            ),
            "annual_turnover_ratio_at_most_0_75": turnover_ratio <= 0.75,
            "maximum_drawdown_no_worse_than_control": (
                float(challenger_metrics["maximum_drawdown"])
                >= float(control_metrics["maximum_drawdown"])
            ),
            "paired_bootstrap_noninferiority_probability_at_least_0_90": (
                bootstrap.noninferiority_probability >= 0.90
            ),
        }
        paired_results[experiment_id] = {
            "strategy_id": control.strategy_id,
            "control_arm_id": control.arm_id,
            "challenger_arm_id": challenger.arm_id,
            "control_replay": control_replay[experiment_id],
            "net_ir_margin": (
                float(challenger_metrics["net_information_ratio"])
                - 0.80 * float(control_metrics["net_information_ratio"])
            ),
            "annual_turnover_ratio": turnover_ratio,
            "maximum_drawdown_difference": (
                float(challenger_metrics["maximum_drawdown"])
                - float(control_metrics["maximum_drawdown"])
            ),
            "paired_bootstrap": asdict(bootstrap),
            "mandatory_paired_gates": gates,
            "paired_gate_pass": all(gates.values()),
        }

    evidence_by_arm: dict[str, str] = {}
    final_arm_pass: dict[str, bool] = {}
    for arm in ordered_arms:
        gates = dict(arm_gates[arm.arm_id])
        if arm.arm == "control":
            gates["control_return_hash_match"] = bool(
                control_replay[arm.experiment_id]["hash_match"]
            )
        else:
            gates["all_paired_gates"] = bool(
                paired_results[arm.experiment_id]["paired_gate_pass"]
            )
        passed = all(gates.values())
        final_arm_pass[arm.arm_id] = passed
        payload = _arm_evidence_payload(
            arm,
            protocol_hash=protocol_hash,
            metrics=arm_metrics[arm.arm_id],
            gates=gates,
            audit=arm_audits[arm.arm_id],
        )
        content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        evidence_record = ledger.store_evidence(content, extension=".json")
        evidence_by_arm[arm.arm_id] = evidence_record.sha256
        trial_id = reservation_by_arm[arm.arm_id].trial_id
        if not _has_event(ledger, trial_id, "backtest"):
            ledger.record_research_step(
                trial_id,
                event_type="backtest",
                status="pass" if passed else "failed_gates",
                payload={
                    "arm_id": arm.arm_id,
                    "candidate_id": _arm_candidate_id(arm.arm_id),
                    "evidence_hash": evidence_record.sha256,
                    "research_gate_pass": passed,
                    "dry_run_completed": False,
                },
                evidence_content=content,
            )
        if not _has_event(ledger, trial_id, "institutional_audit"):
            audit_content = json.dumps(
                arm_audits[arm.arm_id], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            ledger.record_research_step(
                trial_id,
                event_type="institutional_audit",
                status=(
                    "pass"
                    if all(
                        arm_gates[arm.arm_id][key]
                        for key in ("regime_gate", "factor_residual_gate", "decay_gate")
                    )
                    else "failed_gates"
                ),
                payload={
                    "arm_id": arm.arm_id,
                    "audit_hash": arm_audits[arm.arm_id]["audit_hash"],
                },
                evidence_content=audit_content,
            )
        if not _has_event(ledger, trial_id, "outcome"):
            failed = [key for key, value in gates.items() if not value]
            ledger.record_outcome(
                trial_id,
                status="success" if passed else "failure",
                failure_class=None if passed else failed[0].upper(),
                payload={
                    "arm_id": arm.arm_id,
                    "research_gate_pass": passed,
                    "failed_gate_ids": failed,
                    "dry_run_completed": False,
                    "research_pass_grants_live_trading": False,
                },
                evidence_sha256=evidence_record.sha256,
            )

    challenger_passes = sum(
        final_arm_pass[arm.arm_id] for arm in challengers
    )
    common_overlay_accepted = challenger_passes >= 2
    final_counts = ledger.lifetime_counts(campaign="malaysia")
    report = {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "market": "Malaysia",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "status": "frozen_cohort_complete",
        "research_cutoff": "2026-07-06",
        "research_only_contaminated_evidence": True,
        "human_freeze_receipt_sha256": freeze_receipt_hash,
        "data_receipt": data_receipt,
        "data_hash": data_hash,
        "code_hash": code_hash,
        "reservation_report_file_sha256": _file_sha256(
            args.v2_root / "six_arm_reservation_report.json"
        ),
        "lifetime_counts_before_reservation": before_counts,
        "lifetime_counts_after_completion": final_counts,
        "active_return_series_before": matrix_before.shape[1],
        "active_return_series_after": matrix_after.shape[1],
        "complete_lifetime_pbo": pbo,
        "multiplicity_family": {
            "method": "Holm",
            "arm_count": 6,
            "raw_one_sided_hac_pvalues": raw_pvalues,
            "adjusted_one_sided_pvalues": holm_pvalues,
        },
        "control_replay": control_replay,
        "arms": {
            arm.arm_id: {
                "trial_id": reservation_by_arm[arm.arm_id].trial_id,
                "experiment_id": arm.experiment_id,
                "strategy_id": arm.strategy_id,
                "arm": arm.arm,
                "metrics": arm_metrics[arm.arm_id],
                "mandatory_absolute_and_audit_gates": arm_gates[arm.arm_id],
                "selection_diagnostics": (
                    asdict(arm.selection_diagnostics)
                    if arm.selection_diagnostics is not None
                    else None
                ),
                "institutional_audit": arm_audits[arm.arm_id],
                "final_research_gate_pass": final_arm_pass[arm.arm_id],
                "evidence_sha256": evidence_by_arm[arm.arm_id],
            }
            for arm in ordered_arms
        },
        "paired_experiments": paired_results,
        "cohort_decision": {
            "challenger_pass_count": challenger_passes,
            "challenger_count": 3,
            "acceptance_requires_passes": 2,
            "common_overlay_accepted": common_overlay_accepted,
            "post_hoc_combination_search_performed": False,
            "parameter_optimization_performed": False,
        },
        "execution_state": {
            "dry_run_started": False,
            "live_execution_authorized": False,
            "model_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    report_path = _persist_json(report, args.v2_root / "frozen_cohort_report.json")
    print(
        json.dumps(
            {
                "protocol_hash": protocol_hash,
                "status": report["status"],
                "lifetime_attempts": final_counts["lifetime_attempts"],
                "complete_lifetime_pbo": pbo,
                "challenger_pass_count": challenger_passes,
                "common_overlay_accepted": common_overlay_accepted,
                "dry_run_started": False,
                "report_path": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
