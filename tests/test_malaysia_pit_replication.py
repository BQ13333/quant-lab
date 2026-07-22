from __future__ import annotations

import json
from pathlib import Path

from quant_lab.research.malaysia_pit_replication import (
    PITCoverageInputs,
    evaluate_admissibility,
)


ROOT = Path(__file__).resolve().parents[1]


def coverage_inputs(**overrides):
    values = {
        "source_hashes_match": True,
        "production_adjusted_returns_ready": True,
        "historical_u2_membership_reconstructable": True,
        "malaysia_listing_rows": 200,
        "malaysia_inactive_rows": 100,
        "malaysia_current_feature_tickers": 100,
        "malaysia_current_adjusted_return_tickers": 100,
        "malaysia_u2_current_tickers": 62,
        "malaysia_u3_current_tickers": 34,
        "inactive_signal_tickers_covered": 100,
        "inactive_adjusted_return_tickers_covered": 95,
        "security_master_promotion_gates_all_pass": True,
        "security_master_promotion_statuses": {"pit": "pass"},
    }
    values.update(overrides)
    return PITCoverageInputs(**values)


def test_frozen_protocol_contains_exactly_fourteen_unchanged_identities_and_no_dry_run():
    protocol = json.loads(
        (ROOT / "configs/qforge20_malaysia_pit_replication_frozen_v1.json").read_text()
    )
    strategies = protocol["strategies"]
    assert len(strategies) == 14
    assert [item["sequence"] for item in strategies] == list(range(1, 15))
    assert len({item["strategy_id"] for item in strategies}) == 14
    assert all(len(item["spec_sha256"]) == 64 for item in strategies)
    assert protocol["research_contract"]["no_parameter_grid"]
    assert protocol["research_contract"]["no_sign_flip"]
    assert protocol["research_contract"]["no_post_hoc_combination"]
    assert not protocol["stopping_rules"]["dry_run_authorized"]


def test_current_survivor_only_inputs_fail_before_any_strategy_reservation():
    result = evaluate_admissibility(
        coverage_inputs(
            historical_u2_membership_reconstructable=False,
            inactive_signal_tickers_covered=0,
            inactive_adjusted_return_tickers_covered=0,
            security_master_promotion_gates_all_pass=False,
        ),
        minimum_dead_name_return_coverage=0.95,
    )
    assert not result["admissible"]
    assert "historical_u2_membership_reconstructable" in result["failed_gate_ids"]
    assert "minimum_dead_name_return_coverage" in result["failed_gate_ids"]
    assert "no_current_survivor_only_denominator" in result["failed_gate_ids"]


def test_complete_pit_coverage_can_clear_the_data_gate_without_relaxing_thresholds():
    result = evaluate_admissibility(
        coverage_inputs(), minimum_dead_name_return_coverage=0.95
    )
    assert result["admissible"]
    assert result["failed_gate_ids"] == []
    assert result["coverage"]["inactive_adjusted_return_coverage"] == 0.95
