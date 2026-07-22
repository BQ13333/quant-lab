from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.bounded_loop import ALLOWED_FEATURES
from quant_lab.research.contracts import CandidateSpec
from quant_lab.research.institutional_workflow import (
    DeterministicMalaysiaInstitutionalAuditor,
    FiniteCampaignSchedule,
    HypothesisIntakeRegistry,
    HypothesisTicket,
    MalaysiaAuditInputs,
)
from quant_lab.validation.firewall import ResearchFailureCode


def ticket(hypothesis_id: str = "MYS-H-MOMENTUM-001", **changes) -> HypothesisTicket:
    base = HypothesisTicket(
        hypothesis_id=hypothesis_id,
        economic_claim="Lagged medium horizon momentum persists after Malaysia trading costs.",
        family="momentum",
        features=ALLOWED_FEATURES,
        horizon_days=63,
        data_requirements=("adjusted prices", "lagged volume", "daily returns"),
    )
    return replace(base, **changes)


def test_hypothesis_intake_is_idempotent_and_rejects_exact_and_semantic_duplicates(tmp_path):
    registry = HypothesisIntakeRegistry(tmp_path / "ledger.sqlite")
    first = registry.admit(ticket())
    repeated = registry.admit(ticket())
    exact = registry.admit(ticket("MYS-H-MOMENTUM-002"))
    semantic = registry.admit(
        ticket(
            "MYS-H-MOMENTUM-003",
            economic_claim="Trading costs after Malaysia momentum persists horizon medium lagged.",
            data_requirements=("daily returns", "adjusted prices", "lagged volume"),
            features=tuple(reversed(ALLOWED_FEATURES)),
        )
    )
    assert first.accepted and first.created
    assert repeated.accepted and not repeated.created and repeated.reason_code == "ALREADY_ADMITTED"
    assert not exact.accepted and exact.reason_code == "EXACT_DUPLICATE"
    assert not semantic.accepted and semantic.reason_code == "SEMANTIC_DUPLICATE"


def test_child_hypothesis_requires_admitted_parent_and_blinded_mutation_codes(tmp_path):
    registry = HypothesisIntakeRegistry(tmp_path / "ledger.sqlite")
    registry.admit(ticket())
    child = ticket(
        "MYS-H-MOMENTUM-CHILD",
        parent_hypothesis_id="MYS-H-MOMENTUM-001",
        mutation_codes=(ResearchFailureCode.COST_STRESS_FAILURE.value,),
    )
    assert registry.admit(child).accepted
    registry.assert_admitted(child.hypothesis_id)
    directive = registry.directive(child.hypothesis_id, directive_id="MYS-CHILD-001")
    assert directive.mutation_codes == (ResearchFailureCode.COST_STRESS_FAILURE.value,)

    with pytest.raises(ValueError, match="parent hypothesis"):
        registry.admit(
            ticket(
                "MYS-H-ORPHAN-001",
                parent_hypothesis_id="MYS-H-MISSING-001",
                mutation_codes=(ResearchFailureCode.PBO_EXCESSIVE.value,),
            )
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"universe": "JAPAN-U3-U2"}, "Malaysia U3/U2"),
        ({"economic_claim": "Use broker orders to exploit a Malaysia return anomaly."}, "forbidden"),
        ({"features": ("news_sentiment",)}, "frozen Malaysia"),
    ],
)
def test_hypothesis_intake_cannot_escape_market_data_or_execution_scope(tmp_path, changes, match):
    registry = HypothesisIntakeRegistry(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match=match):
        registry.admit(ticket(**changes))


class AuditSource:
    def __init__(self, inputs: MalaysiaAuditInputs) -> None:
        self.inputs = inputs

    def institutional_audit_inputs(self, candidate_id: str) -> MalaysiaAuditInputs:
        return self.inputs


def candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="MYS-C-AUDIT-001",
        hypothesis_id="MYS-H-MOMENTUM-001",
        family="momentum",
        implementation_ref="quant_lab.research.malaysia_pilot:MalaysiaHistoricalBacktester",
        parameters={"lookback": 63},
        evidence=(),
        provenance_hashes={"data": "a" * 64},
    )


def audit_inputs(strategy: np.ndarray, factors: pd.DataFrame) -> MalaysiaAuditInputs:
    index = factors.index
    labels = pd.Series(np.repeat(("up_low", "up_high", "down_mid"), len(index) // 3), index=index)
    return MalaysiaAuditInputs(
        strategy_returns=pd.Series(strategy, index=index),
        market_returns=factors["market"],
        factor_returns=factors,
        regime_labels=labels,
    )


def run_audit(inputs: MalaysiaAuditInputs):
    return DeterministicMalaysiaInstitutionalAuditor(AuditSource(inputs)).audit(
        candidate(), None, trial_id="T", universe=None, config=None  # type: ignore[arg-type]
    )


def test_factor_decomposition_rejects_repackaged_beta_and_accepts_residual_alpha():
    rng = np.random.default_rng(17)
    index = pd.bdate_range("2022-01-03", periods=900)
    factors = pd.DataFrame(
        rng.normal(0.0, 0.006, size=(900, 4)),
        index=index,
        columns=("market", "momentum", "low_volatility", "lagged_liquidity"),
    )
    repackaged = run_audit(audit_inputs(factors["market"].to_numpy(), factors))
    genuine = run_audit(audit_inputs(0.0008 + rng.normal(0.0, 0.004, size=900), factors))
    assert ResearchFailureCode.FACTOR_RESIDUAL_FAILURE in repackaged.failure_codes()
    assert ResearchFailureCode.FACTOR_RESIDUAL_FAILURE not in genuine.failure_codes()


def test_regime_concentration_and_recent_alpha_decay_are_separate_failure_gates():
    rng = np.random.default_rng(29)
    index = pd.bdate_range("2022-01-03", periods=900)
    factors = pd.DataFrame(
        rng.normal(0.0, 0.004, size=(900, 4)),
        index=index,
        columns=("market", "momentum", "low_volatility", "lagged_liquidity"),
    )
    concentrated = np.concatenate(
        [np.full(300, 0.002), np.full(300, -0.0002), np.full(300, -0.0002)]
    ) + rng.normal(0.0, 0.001, 900)
    decayed = np.concatenate([np.full(648, 0.001), np.full(252, -0.001)]) + rng.normal(0.0, 0.001, 900)
    concentrated_result = run_audit(audit_inputs(concentrated, factors))
    decayed_result = run_audit(audit_inputs(decayed, factors))
    assert ResearchFailureCode.REGIME_INSTABILITY in concentrated_result.failure_codes()
    assert ResearchFailureCode.ALPHA_DECAY_FAILURE in decayed_result.failure_codes()


def test_finite_schedule_has_no_unbounded_or_overlapping_loop():
    schedule = FiniteCampaignSchedule(
        start_at=datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc),
        campaign_count=4,
        interval_hours=6,
        max_trials_per_campaign=3,
    )
    runs = schedule.planned_runs()
    assert len(runs) == 4
    assert sum(int(item["max_trials"]) for item in runs) == 12
    assert runs[0]["requires_prior_campaign_terminal_state"] is False
    assert all(item["requires_prior_campaign_terminal_state"] is True for item in runs[1:])
    with pytest.raises(ValueError, match="four finite campaigns"):
        replace(schedule, campaign_count=5).planned_runs()
