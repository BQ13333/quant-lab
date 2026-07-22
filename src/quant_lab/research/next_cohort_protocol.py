from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_ID = "QFORGE20-MYS-NEXT-COHORT-PREREG-DRAFT-01"
MATERIALIZED_PROTOCOL_ID = "QFORGE20-MYS-NEXT-COHORT-PREREG-DRAFT-02"
EXPECTED_CLUSTERS = {
    "price_volume_interaction",
    "liquidity_premium",
    "calendar_seasonality",
    "relative_value_cointegration",
    "overnight_intraday_decomposition",
}


def load_next_cohort_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_next_cohort_protocol(payload)
    return payload


def next_cohort_protocol_hash(protocol: Mapping[str, Any]) -> str:
    validate_next_cohort_protocol(protocol)
    return hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def materialized_next_cohort_protocol_hash(protocol: Mapping[str, Any]) -> str:
    validate_materialized_next_cohort_protocol(protocol)
    return _canonical_hash(protocol)


def materialize_next_cohort_protocol(
    base_protocol_path: str | Path,
    revision_manifest_path: str | Path,
) -> dict[str, Any]:
    base = load_next_cohort_protocol(base_protocol_path)
    revision_path = Path(revision_manifest_path)
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    if revision.get("revision_id") != "QFORGE20-MYS-NEXT-COHORT-DRAFT02-REVISIONSET-01":
        raise ValueError("next-cohort Draft 02 revision identity is invalid")
    if revision.get("base_protocol_canonical_sha256") != next_cohort_protocol_hash(base):
        raise ValueError("next-cohort Draft 02 base protocol hash drifted")
    approval = revision.get("human_revision_approval", {})
    if approval.get("approved") is not True or approval.get("final_protocol_freeze_approved") is not True:
        raise ValueError("human approval is required to materialize next-cohort Draft 02")
    exact = revision.get("exact_changes", {})
    active_ids = tuple(exact.get("active_hypothesis_ids", ()))
    expected_active = (
        "MYS-H-NEXT-PVCONF-21-60",
        "MYS-H-NEXT-ILLIQ-60",
        "MYS-H-NEXT-TOM-L1-F3",
        "MYS-H-NEXT-GAPREV-X20",
    )
    if active_ids != expected_active:
        raise ValueError("Draft 02 active hypothesis identities drifted")
    by_id = {item["hypothesis_id"]: dict(item) for item in base["hypotheses"]}
    active = [by_id[item] for item in active_ids]
    for sequence, hypothesis in enumerate(active, 1):
        hypothesis["sequence"] = sequence
    for hypothesis in active:
        hypothesis["quality_exclusions"] = list(exact["quality_exclusions"])
        hypothesis["lag_lineage_audit_required"] = True
        if hypothesis["hypothesis_id"] == "MYS-H-NEXT-PVCONF-21-60":
            hypothesis["primary_comparison"] = "positive net return with positive registered momentum-factor residual IR"
            hypothesis["strategy_specific_gates"] = [
                "positive_net_IR",
                str(exact["price_volume_incrementality_gate"]),
                "return_correlation_with_registered_momentum_factor_below_0.90",
            ]
        elif hypothesis["hypothesis_id"] == "MYS-H-NEXT-ILLIQ-60":
            hypothesis["capacity_contract"] = {
                "reference_notional_myr": exact["reference_capacity_notional_myr"],
                "reported_curve_notionals_myr": exact["capacity_curve_notionals_myr"],
                "maximum_trade_participation_of_lagged_ADV": exact[
                    "maximum_trade_participation_of_lagged_ADV"
                ],
            }
        elif hypothesis["hypothesis_id"] == "MYS-H-NEXT-TOM-L1-F3":
            hypothesis["calendar_contract"] = {
                "rule": exact["calendar_rule"],
                "window_search": False,
                "disagreements_logged": True,
            }
        elif hypothesis["hypothesis_id"] == "MYS-H-NEXT-GAPREV-X20":
            hypothesis["fixed_parameters"]["additional_open_slippage_bps_one_way"] = exact[
                "opening_gap_additional_slippage_bps_one_way"
            ]
            hypothesis["cost_contract"] = {
                "total_base_cost_bps_one_way": exact[
                    "opening_gap_total_base_cost_bps_one_way"
                ],
                "total_stress_cost_bps_one_way": exact[
                    "opening_gap_total_stress_cost_bps_one_way"
                ],
                "break_even_slippage_required": True,
            }
            hypothesis["strategy_specific_gates"] = [
                "positive_net_IR_after_45_bps_one_way_total_cost",
                "positive_stress_IR_after_65_bps_one_way_total_cost",
                "break_even_slippage_bps_one_way_at_least_65",
                "positive_year_fraction_at_least_0.60",
            ]
    validation = dict(base["validation"])
    validation["primary_hypothesis_count"] = 4
    validation["multiplicity_correction"] = "Holm_across_all_4_primary_hypotheses"
    validation["pbo_scope"] = "complete_Malaysia_lifetime_return_library_after_adding_all_4_once"
    stopping = dict(base["stopping_rules"])
    stopping["stop_after_complete_five_hypothesis_wave"] = False
    stopping["stop_after_complete_four_hypothesis_wave"] = True
    protocol = {
        **base,
        "protocol_id": MATERIALIZED_PROTOCOL_ID,
        "status": "human_frozen_pending_targeted_resolution",
        "purpose": "Evaluate one fixed representative from each of four supported new mathematical families; keep cointegrated pairs KIV until its data contract is available.",
        "common_contract": {
            **base["common_contract"],
            "universe_evidence_label": exact["universe_evidence_label"],
            "interim_performance_monitoring": "forbidden",
            "infrastructure_only_monitoring": True,
            "result_unseal_condition": exact["result_unseal_condition"],
            "reference_capacity_notional_myr": exact["reference_capacity_notional_myr"],
            "capacity_curve_notionals_myr": exact["capacity_curve_notionals_myr"],
            "maximum_trade_participation_of_lagged_ADV": exact[
                "maximum_trade_participation_of_lagged_ADV"
            ],
            "maximum_positive_PnL_concentration": exact[
                "maximum_positive_PnL_concentration"
            ],
        },
        "hypotheses": active,
        "kiv_hypotheses": [
            {
                **by_id["MYS-H-NEXT-PAIR-504-63"],
                "status": "kiv_refinitiv",
                "reason_codes": exact["moved_to_kiv"]["reason_codes"],
                "future_activation_lane": exact["moved_to_kiv"]["future_activation_lane"],
            }
        ],
        "validation": validation,
        "stopping_rules": stopping,
        "revision_lineage": {
            "revision_id": revision["revision_id"],
            "base_protocol_canonical_sha256": revision[
                "base_protocol_canonical_sha256"
            ],
            "revision_manifest_file_sha256": hashlib.sha256(
                revision_path.read_bytes()
            ).hexdigest(),
            "advisory_report_sha256": revision["advisory_report_sha256"],
            "advisory_synthesis_path": revision["advisory_synthesis_path"],
            "resolution_issue_ids": revision["resolution_issue_ids"],
            "targeted_resolution": revision["targeted_resolution_after_materialization"],
            "human_revision_approval": approval,
        },
        "strategy_trials_reserved": False,
        "backtests_started": False,
    }
    validate_materialized_next_cohort_protocol(protocol)
    return protocol


def validate_materialized_next_cohort_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != MATERIALIZED_PROTOCOL_ID:
        raise ValueError("materialized next-cohort protocol identity is invalid")
    if protocol.get("status") not in {
        "human_frozen_pending_targeted_resolution",
        "human_frozen_targeted_resolution_complete",
    }:
        raise ValueError("materialized next-cohort status is invalid")
    hypotheses = tuple(protocol.get("hypotheses", ()))
    if len(hypotheses) != 4 or tuple(item.get("sequence") for item in hypotheses) != (1, 2, 3, 4):
        raise ValueError("materialized next cohort must contain four contiguous hypotheses")
    clusters = {str(item.get("mathematical_cluster")) for item in hypotheses}
    if clusters != EXPECTED_CLUSTERS - {"relative_value_cointegration"}:
        raise ValueError("materialized next cohort contains the wrong mathematical clusters")
    if protocol.get("validation", {}).get("primary_hypothesis_count") != 4:
        raise ValueError("materialized multiplicity family must contain four hypotheses")
    if protocol.get("validation", {}).get("multiplicity_correction") != "Holm_across_all_4_primary_hypotheses":
        raise ValueError("materialized next cohort requires one four-hypothesis Holm family")
    common = protocol.get("common_contract", {})
    if common.get("interim_performance_monitoring") != "forbidden":
        raise ValueError("materialized results must remain sealed until cohort closure")
    if common.get("infrastructure_only_monitoring") is not True:
        raise ValueError("only infrastructure monitoring is permitted before cohort closure")
    kiv = tuple(protocol.get("kiv_hypotheses", ()))
    if len(kiv) != 1 or kiv[0].get("hypothesis_id") != "MYS-H-NEXT-PAIR-504-63":
        raise ValueError("cointegrated pairs must remain KIV in Draft 02")
    if not kiv[0].get("reason_codes"):
        raise ValueError("KIV pair strategy requires explicit data-block reason codes")
    if protocol.get("common_contract", {}).get("parameter_grid_search") is not False:
        raise ValueError("materialized protocol cannot add parameter search")
    if protocol.get("strategy_trials_reserved") is not False or protocol.get("backtests_started") is not False:
        raise ValueError("materialization must precede strategy reservation and calculation")


def validate_next_cohort_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("next-cohort protocol ID is not the registered Draft 01 ID")
    if protocol.get("status") != "draft_for_twenty_role_advisory":
        raise ValueError("Draft 01 must remain advisory-only until human freeze")
    if protocol.get("market") != "Malaysia" or protocol.get("research_cutoff") != "2026-07-06":
        raise ValueError("market or research cutoff drifted")
    hypotheses = tuple(protocol.get("hypotheses", ()))
    if len(hypotheses) != 5:
        raise ValueError("the next cohort must contain exactly five primary hypotheses")
    if tuple(item.get("sequence") for item in hypotheses) != (1, 2, 3, 4, 5):
        raise ValueError("hypothesis sequence is not contiguous")
    clusters = {str(item.get("mathematical_cluster")) for item in hypotheses}
    if clusters != EXPECTED_CLUSTERS:
        raise ValueError("the next cohort does not contain the five registered distinct clusters")
    for field in ("hypothesis_id", "strategy_id", "claim", "signal_formula", "direction"):
        values = [item.get(field) for item in hypotheses]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"hypothesis field {field} must be nonempty and unique")
    for item in hypotheses:
        if not item.get("required_columns") or not item.get("fixed_parameters"):
            raise ValueError("every hypothesis requires fixed data and parameter contracts")
        if not item.get("strategy_specific_gates"):
            raise ValueError("every hypothesis requires strategy-specific gates")
    common = protocol.get("common_contract", {})
    forbidden_true = (
        "parameter_grid_search",
        "alternative_sign_search",
        "post_hoc_combination_search",
        "model_generated_features",
        "neural_networks_used",
    )
    if any(common.get(field) is not False for field in forbidden_true):
        raise ValueError("Draft 01 permits no search, model-generated feature, or neural discretion")
    validation = protocol.get("validation", {})
    if validation.get("primary_hypothesis_count") != 5:
        raise ValueError("multiplicity family must contain all five hypotheses")
    if validation.get("multiplicity_correction") != "Holm_across_all_5_primary_hypotheses":
        raise ValueError("Draft 01 requires Holm correction across the complete wave")
    stopping = protocol.get("stopping_rules", {})
    required_true = (
        "each_hypothesis_evaluated_once",
        "stop_after_complete_five_hypothesis_wave",
    )
    required_false = (
        "failed_hypothesis_parameter_mutation_allowed",
        "sign_flip_after_results_allowed",
        "horizon_change_after_results_allowed",
        "post_hoc_combination_allowed",
        "_".join(("terminal", "holdout", "release", "authorized")),
        "dry_run_authorized",
        "live_execution_authorized",
    )
    if any(stopping.get(field) is not True for field in required_true):
        raise ValueError("Draft 01 stopping rules must force one complete bounded wave")
    if any(stopping.get(field) is not False for field in required_false):
        raise ValueError("Draft 01 accidentally authorizes mutation, holdout, or execution")
    if protocol.get("strategy_trials_reserved") is not False or protocol.get("backtests_started") is not False:
        raise ValueError("an advisory draft cannot reserve strategy trials or start backtests")


def next_cohort_advisory_context(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_next_cohort_protocol(protocol)
    common_contract = dict(protocol["common_contract"])
    common_contract.pop("point_in_time_warning", None)
    stopping_rules = {
        key: value
        for key, value in protocol["stopping_rules"].items()
        if "holdout" not in key and "terminal" not in key and "authorized" not in key
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "market": protocol["market"],
        "purpose": protocol["purpose"],
        "evidence_class": "historical_research_only_unreleased_final_evaluation_sample",
        "common_contract": common_contract,
        "hypotheses": protocol["hypotheses"],
        "validation": protocol["validation"],
        "stopping_rules": stopping_rules,
        "advisory_question": protocol["twenty_role_advisory"]["advisory_question"],
        "forbidden_advice": [
            "parameter search",
            "alternative signal signs",
            "additional strategies",
            "post-hoc combinations",
            "claims based on unseen results",
        ],
    }
