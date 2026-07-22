from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quant_lab.research.bounded_loop import CampaignConfig, MutationDirective
from quant_lab.research.malaysia_pilot import FrozenCohortCandidateGenerator
from quant_lab.research.malaysia_strategy_registry import (
    COHORT_VERSION,
    malaysia_atomic_cohort,
    malaysia_predeclared_combinations,
    validate_strategy_parameters,
)


def test_atomic_cohort_is_frozen_unique_source_backed_and_contiguous():
    cohort = malaysia_atomic_cohort()
    assert COHORT_VERSION == "MALAYSIA-ATOMIC-COHORT-V1"
    assert len(cohort) == 10
    assert tuple(item.sequence for item in cohort) == tuple(range(1, 11))
    assert len({item.strategy_id for item in cohort}) == 10
    assert len({item.hypothesis_id for item in cohort}) == 10
    assert len({item.content_hash() for item in cohort}) == 10
    root = Path(__file__).resolve().parents[1]
    assert all((root / item.source_file).exists() for item in cohort)
    combinations = malaysia_predeclared_combinations()
    assert len(combinations) == 3
    assert all(item.weighting == "equal_weight" for item in combinations)


def test_frozen_candidate_generator_has_no_model_call_or_parameter_discretion(tmp_path):
    strategy = malaysia_atomic_cohort()[0]
    directive = MutationDirective(strategy.directive_id, strategy.hypothesis_id, strategy.family)
    config = CampaignConfig("MALAYSIA-COHORT-TEST", tmp_path, (directive,))
    generated = FrozenCohortCandidateGenerator((strategy,)).generate(
        directive, trial_id="TRIAL", config=config
    )
    assert generated.model_call is None
    assert generated.external_cost_usd == 0.0
    assert generated.candidate.parameters == strategy.parameters
    assert generated.candidate.provenance_hashes["strategy_spec"] == strategy.content_hash()


def test_family_parameter_contract_rejects_aliases_extra_fields_and_horizon_drift():
    strategy = malaysia_atomic_cohort()[5]
    with pytest.raises(ValueError, match="frequency"):
        validate_strategy_parameters(
            strategy.family,
            {**strategy.parameters, "rebalance_frequency": "weekly"},
            generation=1,
            mutation_codes=(),
        )
    with pytest.raises(ValueError, match="exactly"):
        validate_strategy_parameters(
            strategy.family,
            {**strategy.parameters, "new_knob": 7},
            generation=1,
            mutation_codes=(),
        )
    with pytest.raises(ValueError, match="horizon"):
        drifted = dict(strategy.parameters)
        drifted["lookback"] = 400
        validate_strategy_parameters(
            strategy.family,
            drifted,
            generation=1,
            mutation_codes=(),
        )


def test_content_hash_changes_when_a_frozen_parameter_changes():
    strategy = malaysia_atomic_cohort()[0]
    parameters = dict(strategy.parameters)
    parameters["lookback"] = 63
    changed = replace(strategy, parameters=parameters)
    assert changed.content_hash() != strategy.content_hash()
