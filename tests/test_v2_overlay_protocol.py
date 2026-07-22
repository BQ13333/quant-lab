from __future__ import annotations

import copy
from pathlib import Path

import pytest

from quant_lab.research.v2_overlay_protocol import (
    V2_DRAFT02_PROTOCOL_ID,
    V2_DRAFT_PROTOCOL_ID,
    V2_PARENT_IDS,
    draft02_exposure_scalar,
    load_v2_overlay_protocol,
    load_v2_overlay_revision_manifest,
    materialize_v2_overlay_protocol,
    v2_overlay_protocol_hash,
    v2_protocol_advisory_context,
    validate_v2_overlay_revision_manifest,
    validate_v2_overlay_protocol,
)
from quant_lab.agents.specialist_runtime import MalaysiaReviewDossier


PROTOCOL = Path("configs/qforge20_malaysia_v2_overlay_draft.json")
REVISION = Path("configs/qforge20_malaysia_v2_overlay_draft02_revision.json")
ADVISORY_SYNTHESIS = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/"
    "v2_protocol_advisory/advisory_synthesis.json"
)
requires_private_advisory = pytest.mark.skipif(
    not ADVISORY_SYNTHESIS.is_file(),
    reason="private advisory-artifact integration test",
)


def test_v2_draft_is_exact_bounded_and_not_falsely_frozen():
    protocol = load_v2_overlay_protocol(PROTOCOL)
    assert protocol["protocol_id"] == V2_DRAFT_PROTOCOL_ID
    assert tuple(item["strategy_id"] for item in protocol["parent_strategies"]) == V2_PARENT_IDS
    assert len(protocol["arms"]) == 6
    assert protocol["paired_design"]["parameter_grid_search"] is False
    assert protocol["paired_design"]["post_hoc_combination_search"] is False
    assert all(value is False for value in protocol["approval"].values())
    assert len(protocol["twenty_role_advisory"]["role_ids"]) == 20
    assert v2_overlay_protocol_hash(protocol) == (
        "05f4454c9cb97b9f8adfcb9485008ed73c1aa5438fc8daaf50cb13f941cd7230"
    )


@requires_private_advisory
def test_v2_draft02_materializes_exact_approved_revision_without_freeze_or_reservation():
    protocol = materialize_v2_overlay_protocol(PROTOCOL, REVISION)
    overlay = protocol["common_overlay"]
    bootstrap = protocol["mandatory_paired_gates"]["paired_moving_block_bootstrap"]
    assert protocol["protocol_id"] == V2_DRAFT02_PROTOCOL_ID
    assert protocol["approval"] == {
        "twenty_role_advisory_complete": True,
        "human_revision_set_approved": True,
        "targeted_resolution_complete": False,
        "human_freeze_approved": False,
        "strategy_trials_reserved": False,
        "backtests_started": False,
    }
    assert overlay["minimum_selected_names"] == 4
    assert overlay["exposure_is_single_portfolio_scalar"] is True
    assert overlay["volatility_estimator"] == (
        "sample_standard_deviation_of_unit_exposure_pre_scaled_challenger_gross_daily_returns"
    )
    assert "protocol_id_double_colon_ticker" in overlay["selection_tie_breaker"]
    assert overlay["cash_weight_is_residual_not_turnover_leg"] is True
    assert bootstrap["block_length_sessions"] == 63
    assert bootstrap["noninferiority_margin_expression"] == (
        "IR_challenger - 0.80 * IR_control >= 0"
    )
    assert bootstrap["nonfinite_draws_count_as_failure"] is True
    assert v2_overlay_protocol_hash(protocol) == (
        "9dd12ad0fb963587112989dc5cd31c88f7130a534e1149cb9af954046e5f110c"
    )


@requires_private_advisory
def test_v2_draft02_advisory_context_remains_research_only():
    context = v2_protocol_advisory_context(materialize_v2_overlay_protocol(PROTOCOL, REVISION))
    assert context["protocol_id"] == V2_DRAFT02_PROTOCOL_ID
    assert "approval" not in context
    assert "arms" not in context
    assert "research_period_result" not in str(context)


def test_v2_draft02_revision_manifest_drift_fails_closed():
    revision = copy.deepcopy(load_v2_overlay_revision_manifest(REVISION))
    revision["targeted_resolution"]["maximum_role_calls"] = 4
    with pytest.raises(ValueError, match="targeted-resolution boundary"):
        validate_v2_overlay_revision_manifest(revision)


@requires_private_advisory
def test_v2_draft02_semantic_drift_fails_closed():
    protocol = materialize_v2_overlay_protocol(PROTOCOL, REVISION)
    protocol["common_overlay"]["minimum_selected_names"] = 3
    with pytest.raises(ValueError, match="overlay semantics"):
        validate_v2_overlay_protocol(protocol)


def test_v2_draft02_piecewise_exposure_is_bounded_and_continuous_at_target():
    below = draft02_exposure_scalar(0.09999999, 63)
    boundary = draft02_exposure_scalar(0.10, 63)
    above = draft02_exposure_scalar(0.10000001, 63)
    assert below == boundary == 1.0
    assert 0.0 < above < 1.0
    assert boundary - above == pytest.approx(9.99999900663795e-08)
    assert draft02_exposure_scalar(0.0, 63) == 0.0
    assert draft02_exposure_scalar(float("nan"), 63) == 0.0
    assert draft02_exposure_scalar(0.20, 41) == 0.0


@pytest.mark.parametrize("available_observations", [-1, 42.0, True])
def test_v2_draft02_piecewise_exposure_rejects_invalid_observation_counts(
    available_observations,
):
    expected = ValueError if available_observations == -1 else TypeError
    with pytest.raises(expected):
        draft02_exposure_scalar(0.10, available_observations)


def test_v2_advisory_context_excludes_approval_claims_and_strategy_results():
    context = v2_protocol_advisory_context(load_v2_overlay_protocol(PROTOCOL))
    assert "approval" not in context
    assert "arms" not in context
    assert "research_period_result" not in str(context)
    assert context["parent_strategy_ids"] == list(V2_PARENT_IDS)
    MalaysiaReviewDossier(
        candidate_id="MYS-C-V2-PROTOCOL-TEST",
        hypothesis_hash="1" * 64,
        candidate_hash=v2_overlay_protocol_hash(load_v2_overlay_protocol(PROTOCOL)),
        problem="Challenge the exact Malaysia V2 protocol before freeze.",
        tags=("selection_bias", "portfolio"),
        evidence_hashes={"protocol": "2" * 64},
        research_context=context,
        review_phase="pre_backtest_design",
    ).validate()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("approval", "human_freeze_approved"), True, "cannot claim"),
        (("paired_design", "evaluated_arm_count"), 7, "bounded invariant"),
        (("common_overlay", "entry_rank_quantile"), [0.75, 0.80], "exact scalar"),
        (("mandatory_absolute_gates", "maximum_drawdown_floor"), -0.30, "relaxed"),
        (("mandatory_paired_gates", "challenger_net_information_ratio_retention_minimum"), 0.50, "non-inferiority"),
        (("cohort_decision", "model_vote_can_override_gate"), True, "delegates"),
    ],
)
def test_v2_protocol_drift_fails_closed(path, value, message):
    protocol = copy.deepcopy(load_v2_overlay_protocol(PROTOCOL))
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_v2_overlay_protocol(protocol)
