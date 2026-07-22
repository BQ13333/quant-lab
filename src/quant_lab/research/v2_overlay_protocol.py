from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from quant_lab.agents.phd_math_specialists import load_specialist_panel
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort


V2_DRAFT_PROTOCOL_ID = "QFORGE20-MYS-V2-OVERLAY-DRAFT-01"
V2_DRAFT_STATUS = "draft_pending_twenty_role_review_and_human_freeze"
V2_DRAFT02_PROTOCOL_ID = "QFORGE20-MYS-V2-OVERLAY-DRAFT-02"
V2_DRAFT02_STATUS = "draft_pending_targeted_resolution_and_human_freeze"
V2_DRAFT02_REVISION_ID = "QFORGE20-MYS-V2-DRAFT02-REVISIONSET-01"
V2_PARENT_IDS = (
    "MYS-S-EWMAC-16-64",
    "MYS-S-RESIDMOM-126",
    "MYS-S-LOWVOL-126",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_DRAFT02_EXACT_CHANGES: dict[str, Any] = {
    "common_overlay_version": "MYS-COMMON-RISK-OVERLAY-V2-DRAFT-02",
    "selection_tie_breaker": (
        "ascending_sha256_of_protocol_id_double_colon_ticker_static_across_dates"
    ),
    "minimum_selected_names": 4,
    "insufficient_diversification_action": (
        "set_portfolio_exposure_to_zero_record_INSUFFICIENT_DIVERSIFICATION_"
        "and_fail_challenger_gate"
    ),
    "volatility_estimator": (
        "sample_standard_deviation_of_unit_exposure_pre_scaled_challenger_gross_daily_returns"
    ),
    "exposure_is_single_portfolio_scalar": True,
    "exposure_rule": (
        "At each month-end rebalance, use lagged annualized volatility v from the unit-exposure "
        "pre-scaled challenger gross-return series. Set exposure to 0.0 when minimum observations "
        "are unavailable or v is non-finite or non-positive; set exposure to 1.0 when 0<v<=0.10; "
        "otherwise set exposure to 0.10/v. Apply this one scalar after equal weighting and hold it "
        "constant until the next month-end."
    ),
    "volatility_cap_is_not_realized_volatility_guarantee": True,
    "turnover_definition": (
        "daily_sum_absolute_changes_in_risky_asset_weights_including_changes_from_portfolio_"
        "exposure_scalar_cash_is_residual_and_not_a_separate_turnover_leg"
    ),
    "same_initial_entry_turnover_convention_for_both_arms": True,
    "bootstrap_block_length_sessions": 63,
    "bootstrap_noninferiority_margin_expression": "IR_challenger - 0.80 * IR_control >= 0",
    "bootstrap_nonfinite_draws_count_as_failure": True,
    "parameter_grid_search": False,
    "post_hoc_combination_search": False,
    "new_dataset_or_model_family_added": False,
}
_DRAFT02_TARGETED_ROLES = (
    "selection_bias_validation_specialist",
    "time_series_dependence_specialist",
    "high_frequency_econometrics_specialist",
)
_DRAFT02_RESOLUTION_ISSUES = tuple(f"V2-ADV-{index:03d}" for index in range(1, 9))


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_declared_path(path: str | Path, *, revision_path: Path) -> Path:
    declared = Path(path)
    if declared.is_absolute():
        return declared
    workspace_candidate = revision_path.parent.parent / declared
    if workspace_candidate.exists():
        return workspace_candidate
    return declared


def draft02_exposure_scalar(
    annualized_volatility: float,
    available_observations: int,
) -> float:
    """Executable form of the frozen Draft 02 piecewise portfolio-scalar rule."""

    if not isinstance(available_observations, int) or isinstance(available_observations, bool):
        raise TypeError("available observations must be an integer")
    if available_observations < 0:
        raise ValueError("available observations cannot be negative")
    if available_observations < 42:
        return 0.0
    volatility = float(annualized_volatility)
    if not math.isfinite(volatility) or volatility <= 0.0:
        return 0.0
    if volatility <= 0.10:
        return 1.0
    exposure = 0.10 / volatility
    if not math.isfinite(exposure) or not 0.0 < exposure < 1.0:
        raise ArithmeticError("Draft 02 exposure invariant failed")
    return exposure


def load_v2_overlay_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V2 overlay protocol must be a JSON object")
    validate_v2_overlay_protocol(payload)
    return payload


def load_v2_overlay_revision_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V2 overlay revision manifest must be a JSON object")
    validate_v2_overlay_revision_manifest(payload)
    return payload


def validate_v2_overlay_revision_manifest(revision: Mapping[str, Any]) -> None:
    expected_header = {
        "schema_version": 1,
        "revision_id": V2_DRAFT02_REVISION_ID,
        "materialized_protocol_id": V2_DRAFT02_PROTOCOL_ID,
        "status": "approved_for_materialization_and_targeted_resolution",
        "base_protocol_path": "configs/qforge20_malaysia_v2_overlay_draft.json",
        "advisory_synthesis_path": (
            "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/"
            "v2_protocol_advisory/advisory_synthesis.json"
        ),
        "strategy_trials_reserved": False,
        "backtests_started": False,
    }
    if any(revision.get(key) != value for key, value in expected_header.items()):
        raise ValueError("Draft 02 revision identity or authorization boundary drifted")
    for key in ("base_protocol_canonical_sha256", "advisory_synthesis_file_sha256"):
        if not _SHA256.fullmatch(str(revision.get(key, ""))):
            raise ValueError("Draft 02 revision source hash is invalid")
    approval = revision.get("human_revision_approval")
    if approval != {
        "approved": True,
        "approved_on": "2026-07-13",
        "approval_text": "Approve the Draft 02 revision set.",
        "final_protocol_freeze_approved": False,
    }:
        raise ValueError("Draft 02 revision approval receipt drifted")
    if tuple(revision.get("resolution_issue_ids", ())) != _DRAFT02_RESOLUTION_ISSUES:
        raise ValueError("Draft 02 advisory-resolution lineage is incomplete")
    if revision.get("exact_changes") != _DRAFT02_EXACT_CHANGES:
        raise ValueError("Draft 02 exact revision set drifted")
    if revision.get("targeted_resolution") != {
        "role_ids": list(_DRAFT02_TARGETED_ROLES),
        "maximum_role_calls": 3,
        "privacy_class": "local_only",
        "model": "qwen3.5:35b-mlx",
        "hosted_cost_usd": 0.0,
    }:
        raise ValueError("Draft 02 targeted-resolution boundary drifted")


def materialize_v2_overlay_protocol(
    base_protocol_path: str | Path,
    revision_manifest_path: str | Path,
) -> dict[str, Any]:
    """Reconstruct Draft 02 from immutable Draft 01 plus its approved exact revision set."""

    base = load_v2_overlay_protocol(base_protocol_path)
    revision_path = Path(revision_manifest_path)
    revision = load_v2_overlay_revision_manifest(revision_path)
    base_hash = v2_overlay_protocol_hash(base)
    if base_hash != revision["base_protocol_canonical_sha256"]:
        raise ValueError("Draft 01 canonical hash does not match the approved revision base")
    synthesis_path = _resolve_declared_path(
        revision["advisory_synthesis_path"], revision_path=revision_path
    )
    if not synthesis_path.is_file():
        raise FileNotFoundError("Draft 02 advisory synthesis is unavailable")
    synthesis_file_hash = _file_sha256(synthesis_path)
    if synthesis_file_hash != revision["advisory_synthesis_file_sha256"]:
        raise ValueError("Draft 02 advisory synthesis hash does not match the approved source")
    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    if synthesis.get("protocol_hash") != base_hash:
        raise ValueError("Draft 02 advisory synthesis was not produced from this Draft 01")
    issue_ids = tuple(item.get("issue_id") for item in synthesis.get("resolution_matrix", ()))
    if issue_ids != _DRAFT02_RESOLUTION_ISSUES:
        raise ValueError("Draft 02 advisory synthesis issue set drifted")

    protocol = copy.deepcopy(base)
    protocol["protocol_id"] = V2_DRAFT02_PROTOCOL_ID
    protocol["status"] = V2_DRAFT02_STATUS
    protocol["approval"] = {
        "twenty_role_advisory_complete": True,
        "human_revision_set_approved": True,
        "targeted_resolution_complete": False,
        "human_freeze_approved": False,
        "strategy_trials_reserved": False,
        "backtests_started": False,
    }
    protocol["source_hashes"].update(
        {
            "v2_draft01_canonical_sha256": base_hash,
            "v2_advisory_synthesis_file_sha256": synthesis_file_hash,
            "v2_draft02_revision_manifest_file_sha256": _file_sha256(revision_path),
        }
    )
    overlay = protocol["common_overlay"]
    changes = revision["exact_changes"]
    overlay.update(
        {
            "version": changes["common_overlay_version"],
            "selection_order": (
                "Retain finite-score incumbents at or above the retention quantile, ranked by "
                "descending current score then ascending static SHA-256 of protocol_id::ticker; "
                "truncate at K. Fill remaining slots from non-incumbents at or above the entry "
                "quantile in the same deterministic order. K is ceil(0.20 times the valid-score "
                "universe count). Unfilled slots remain cash."
            ),
            "selection_tie_breaker": changes["selection_tie_breaker"],
            "minimum_selected_names": changes["minimum_selected_names"],
            "insufficient_diversification_action": changes[
                "insufficient_diversification_action"
            ],
            "volatility_estimator": changes["volatility_estimator"],
            "exposure_is_single_portfolio_scalar": changes[
                "exposure_is_single_portfolio_scalar"
            ],
            "exposure_rule": changes["exposure_rule"],
            "volatility_cap_is_not_realized_volatility_guarantee": changes[
                "volatility_cap_is_not_realized_volatility_guarantee"
            ],
            "turnover_definition": changes["turnover_definition"],
            "cash_weight_is_residual_not_turnover_leg": True,
            "same_initial_entry_turnover_convention_for_both_arms": changes[
                "same_initial_entry_turnover_convention_for_both_arms"
            ],
        }
    )
    bootstrap = protocol["mandatory_paired_gates"]["paired_moving_block_bootstrap"]
    bootstrap.update(
        {
            "block_length_sessions": changes["bootstrap_block_length_sessions"],
            "noninferiority_margin_expression": changes[
                "bootstrap_noninferiority_margin_expression"
            ],
            "nonfinite_draws_count_as_failure": changes[
                "bootstrap_nonfinite_draws_count_as_failure"
            ],
        }
    )
    protocol["revision_lineage"] = {
        "revision_id": revision["revision_id"],
        "base_protocol_id": V2_DRAFT_PROTOCOL_ID,
        "base_protocol_canonical_sha256": base_hash,
        "advisory_synthesis_file_sha256": synthesis_file_hash,
        "revision_manifest_canonical_sha256": hashlib.sha256(
            _canonical(revision).encode("utf-8")
        ).hexdigest(),
        "resolution_issue_ids": list(_DRAFT02_RESOLUTION_ISSUES),
        "human_revision_set_approved": True,
        "targeted_resolution_role_ids": list(_DRAFT02_TARGETED_ROLES),
        "targeted_resolution_maximum_role_calls": 3,
        "final_human_freeze_approved": False,
    }
    protocol["stopping_rules"] = [
        *protocol["stopping_rules"],
        (
            "Draft 02 requires exactly the three named local targeted-resolution roles before final "
            "human freeze; no strategy arm may be reserved during that review."
        ),
    ]
    protocol["evidence_requirements"] = [
        *protocol["evidence_requirements"],
        "Content-addressed Draft 02 revision manifest and deterministic materialized protocol.",
        "Three local targeted-resolution reviews and a controller resolution record.",
    ]
    validate_v2_overlay_protocol(protocol)
    return protocol


def v2_overlay_protocol_hash(protocol: Mapping[str, Any]) -> str:
    validate_v2_overlay_protocol(protocol)
    return hashlib.sha256(_canonical(dict(protocol)).encode("utf-8")).hexdigest()


def validate_v2_overlay_protocol(protocol: Mapping[str, Any]) -> None:
    protocol_id = protocol.get("protocol_id")
    if protocol_id not in {V2_DRAFT_PROTOCOL_ID, V2_DRAFT02_PROTOCOL_ID}:
        raise ValueError("V2 protocol identity is invalid")
    if protocol.get("workflow_name") != "QForge-20" or protocol.get("market") != "Malaysia":
        raise ValueError("V2 protocol escaped the QForge-20 Malaysia scope")
    approval = protocol.get("approval")
    if protocol_id == V2_DRAFT_PROTOCOL_ID:
        if protocol.get("status") != V2_DRAFT_STATUS:
            raise ValueError("V2 draft status is invalid")
        if not isinstance(approval, Mapping) or any(bool(value) for value in approval.values()):
            raise ValueError("draft protocol cannot claim review, freeze, reservation, or execution")
    else:
        if protocol.get("status") != V2_DRAFT02_STATUS:
            raise ValueError("Draft 02 status is invalid")
        expected_approval = {
            "twenty_role_advisory_complete": True,
            "human_revision_set_approved": True,
            "targeted_resolution_complete": False,
            "human_freeze_approved": False,
            "strategy_trials_reserved": False,
            "backtests_started": False,
        }
        if approval != expected_approval:
            raise ValueError("Draft 02 approval boundary drifted")
    universe = protocol.get("universe_contract")
    if not isinstance(universe, Mapping) or universe.get("primary") != "MALAYSIA-U3-34":
        raise ValueError("V2 primary universe is invalid")
    if universe.get("robustness") != "MALAYSIA-U2-62" or universe.get("research_cutoff") != "2026-07-06":
        raise ValueError("V2 robustness universe or research cutoff drifted")
    if not bool(universe.get("observations_through_cutoff_are_research_only")):
        raise ValueError("V2 protocol must preserve research-evidence contamination")
    source_hashes = protocol.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("V2 source hashes are required")
    if not all(_SHA256.fullmatch(str(value)) for value in source_hashes.values()):
        raise ValueError("V2 source hashes must be lowercase SHA-256 values")

    registry = {item.strategy_id: item for item in malaysia_atomic_cohort()}
    parents = protocol.get("parent_strategies")
    if not isinstance(parents, list) or tuple(item.get("strategy_id") for item in parents) != V2_PARENT_IDS:
        raise ValueError("V2 parent strategies or ordering drifted")
    for item in parents:
        strategy = registry[str(item["strategy_id"])]
        if item.get("family") != strategy.family or item.get("strategy_spec_sha256") != strategy.content_hash():
            raise ValueError("V2 parent strategy provenance mismatch")

    design = protocol.get("paired_design")
    if not isinstance(design, Mapping):
        raise ValueError("V2 paired design is required")
    expected_design = {
        "experiment_count": 3,
        "evaluated_arm_count": 6,
        "lifetime_denominator_increment_on_reservation": 6,
        "paired_calendar": True,
        "paired_universes": True,
        "paired_cost_models": True,
        "control_replay_must_match_prior_return_hash": True,
        "post_hoc_combination_search": False,
        "parameter_grid_search": False,
    }
    if any(design.get(key) != value for key, value in expected_design.items()):
        raise ValueError("V2 paired design changed a bounded invariant")
    arms = protocol.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise ValueError("V2 requires exactly six pre-registered arms")
    if len({str(item.get("arm_id")) for item in arms}) != 6:
        raise ValueError("V2 arm identities must be unique")
    for parent in parents:
        experiment_id = parent["experiment_id"]
        pair = [item for item in arms if item.get("experiment_id") == experiment_id]
        if {item.get("arm") for item in pair} != {"control", "challenger"}:
            raise ValueError("each V2 experiment requires one control and one challenger")
        if {item.get("strategy_id") for item in pair} != {parent["strategy_id"]}:
            raise ValueError("V2 pair changed its frozen parent strategy")

    overlay = protocol.get("common_overlay")
    if not isinstance(overlay, Mapping):
        raise ValueError("V2 common overlay is required")
    scalar_expectations = {
        "rebalance_frequency": "ME",
        "score_information_lag_sessions": 1,
        "entry_rank_quantile": 0.80,
        "retention_rank_quantile": 0.60,
        "maximum_position_fraction_of_valid_universe": 0.20,
        "position_weighting": "equal_weight_within_selected_names",
        "short_positions_allowed": False,
        "portfolio_volatility_cap_annual": 0.10,
        "volatility_lookback_sessions": 63,
        "volatility_minimum_observations": 42,
        "volatility_information_lag_sessions": 1,
        "annualization_sessions": 252,
        "maximum_gross_exposure": 1.0,
        "cash_return": 0.0,
        "drawdown_timing_rule": None,
        "base_cost_bps_one_way": 20.0,
        "stress_cost_bps_one_way": 40.0,
    }
    if any(overlay.get(key) != value for key, value in scalar_expectations.items()):
        raise ValueError("V2 overlay changed an exact scalar or categorical rule")
    if protocol_id == V2_DRAFT_PROTOCOL_ID:
        expected_overlay_revision = {
            "version": "MYS-COMMON-RISK-OVERLAY-V2-DRAFT-01",
            "volatility_estimator": "sample_standard_deviation_of_control_gross_daily_returns",
            "turnover_definition": "daily_sum_absolute_weight_change_including_cash_exposure_changes",
        }
        if any(overlay.get(key) != value for key, value in expected_overlay_revision.items()):
            raise ValueError("Draft 01 overlay semantics drifted")
    else:
        expected_overlay_revision = {
            "version": _DRAFT02_EXACT_CHANGES["common_overlay_version"],
            "selection_tie_breaker": _DRAFT02_EXACT_CHANGES["selection_tie_breaker"],
            "minimum_selected_names": _DRAFT02_EXACT_CHANGES["minimum_selected_names"],
            "insufficient_diversification_action": _DRAFT02_EXACT_CHANGES[
                "insufficient_diversification_action"
            ],
            "volatility_estimator": _DRAFT02_EXACT_CHANGES["volatility_estimator"],
            "exposure_is_single_portfolio_scalar": True,
            "exposure_rule": _DRAFT02_EXACT_CHANGES["exposure_rule"],
            "volatility_cap_is_not_realized_volatility_guarantee": True,
            "turnover_definition": _DRAFT02_EXACT_CHANGES["turnover_definition"],
            "cash_weight_is_residual_not_turnover_leg": True,
            "same_initial_entry_turnover_convention_for_both_arms": True,
        }
        if any(overlay.get(key) != value for key, value in expected_overlay_revision.items()):
            raise ValueError("Draft 02 overlay semantics drifted")
    if not float(overlay["retention_rank_quantile"]) < float(overlay["entry_rank_quantile"]):
        raise ValueError("V2 retention threshold must be below its entry threshold")
    for key, value in overlay.items():
        if key.endswith(("quantile", "sessions", "annual", "exposure", "return", "one_way")):
            if isinstance(value, (list, tuple, dict)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError("V2 overlay contains a hidden parameter grid or non-finite scalar")

    absolute = protocol.get("mandatory_absolute_gates")
    paired = protocol.get("mandatory_paired_gates")
    if not isinstance(absolute, Mapping) or not isinstance(paired, Mapping):
        raise ValueError("V2 absolute and paired gates are required")
    required_absolute = {
        "deflated_sharpe_probability_minimum": 0.95,
        "pbo_maximum": 0.20,
        "maximum_drawdown_floor": -0.20,
        "annual_turnover_maximum": 3.0,
        "maximum_asset_weight": 0.25,
        "holm_adjusted_one_sided_pvalue_maximum": 0.05,
    }
    if any(absolute.get(key) != value for key, value in required_absolute.items()):
        raise ValueError("V2 absolute gates were relaxed or changed")
    if paired.get("challenger_net_information_ratio_retention_minimum") != 0.80:
        raise ValueError("V2 net-IR non-inferiority rule changed")
    if paired.get("challenger_annual_turnover_ratio_maximum") != 0.75:
        raise ValueError("V2 paired turnover rule changed")
    bootstrap = paired.get("paired_moving_block_bootstrap")
    expected_bootstrap: dict[str, Any] = {
        "block_length_sessions": 21 if protocol_id == V2_DRAFT_PROTOCOL_ID else 63,
        "replications": 10000,
        "random_seed": 20260713,
        "same_block_indices_for_both_arms": True,
        "net_information_ratio_retention_probability_minimum": 0.90,
    }
    if protocol_id == V2_DRAFT02_PROTOCOL_ID:
        expected_bootstrap.update(
            {
                "noninferiority_margin_expression": _DRAFT02_EXACT_CHANGES[
                    "bootstrap_noninferiority_margin_expression"
                ],
                "nonfinite_draws_count_as_failure": True,
            }
        )
    if not isinstance(bootstrap, Mapping) or bootstrap != expected_bootstrap:
        raise ValueError("V2 paired bootstrap configuration drifted")

    decision = protocol.get("cohort_decision")
    if not isinstance(decision, Mapping) or decision.get("common_overlay_acceptance_requires_individual_passes") != 2:
        raise ValueError("V2 cohort decision rule is invalid")
    if decision.get("common_overlay_acceptance_denominator") != 3:
        raise ValueError("V2 cohort acceptance denominator is invalid")
    if bool(decision.get("model_vote_can_override_gate")) or bool(decision.get("research_pass_can_authorize_execution")):
        raise ValueError("V2 improperly delegates evidence or execution authority")
    if len(protocol.get("stopping_rules", [])) < 6 or len(protocol.get("evidence_requirements", [])) < 8:
        raise ValueError("V2 stopping rules or evidence requirements are incomplete")

    advisory = protocol.get("twenty_role_advisory")
    panel = load_specialist_panel()
    if not isinstance(advisory, Mapping) or advisory.get("specialist_count") != 20:
        raise ValueError("V2 advisory must cover all twenty canonical roles")
    if tuple(advisory.get("role_ids", ())) != tuple(item.id for item in panel.specialists):
        raise ValueError("V2 advisory role snapshot differs from the canonical panel")
    if len(advisory.get("common_questions", [])) < 6:
        raise ValueError("V2 advisory questions are incomplete")
    if protocol_id == V2_DRAFT02_PROTOCOL_ID:
        lineage = protocol.get("revision_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("Draft 02 revision lineage is required")
        expected_lineage = {
            "revision_id": V2_DRAFT02_REVISION_ID,
            "base_protocol_id": V2_DRAFT_PROTOCOL_ID,
            "base_protocol_canonical_sha256": protocol["source_hashes"].get(
                "v2_draft01_canonical_sha256"
            ),
            "advisory_synthesis_file_sha256": protocol["source_hashes"].get(
                "v2_advisory_synthesis_file_sha256"
            ),
            "resolution_issue_ids": list(_DRAFT02_RESOLUTION_ISSUES),
            "human_revision_set_approved": True,
            "targeted_resolution_role_ids": list(_DRAFT02_TARGETED_ROLES),
            "targeted_resolution_maximum_role_calls": 3,
            "final_human_freeze_approved": False,
        }
        if any(lineage.get(key) != value for key, value in expected_lineage.items()):
            raise ValueError("Draft 02 revision lineage drifted")
        if not _SHA256.fullmatch(str(lineage.get("revision_manifest_canonical_sha256", ""))):
            raise ValueError("Draft 02 revision manifest hash is invalid")


def v2_protocol_advisory_context(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the allowlisted, research-only context used by local role reviews."""

    validate_v2_overlay_protocol(protocol)
    decision = dict(protocol["cohort_decision"])
    grants_execution = bool(decision.pop("research_pass_can_authorize_execution"))
    decision["research_pass_grants_execution"] = grants_execution
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": v2_overlay_protocol_hash(protocol),
        "status": protocol["status"],
        "research_cutoff": protocol["universe_contract"]["research_cutoff"],
        "parent_strategy_ids": [item["strategy_id"] for item in protocol["parent_strategies"]],
        "claim": protocol["claim"],
        "paired_design": protocol["paired_design"],
        "common_overlay": protocol["common_overlay"],
        "absolute_gates": protocol["mandatory_absolute_gates"],
        "paired_gates": protocol["mandatory_paired_gates"],
        "cohort_decision": decision,
        "stopping_rules": protocol["stopping_rules"],
        "evidence_requirements": protocol["evidence_requirements"],
        "common_advisory_questions": protocol["twenty_role_advisory"]["common_questions"],
    }
