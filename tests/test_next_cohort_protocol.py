from __future__ import annotations

from pathlib import Path

import pytest

from quant_lab.research.next_cohort_protocol import (
    EXPECTED_CLUSTERS,
    load_next_cohort_protocol,
    materialize_next_cohort_protocol,
    materialized_next_cohort_protocol_hash,
    next_cohort_protocol_hash,
    validate_next_cohort_protocol,
)


def _protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/qforge20_malaysia_next_cohort_prereg_draft01.json"


def test_draft01_is_five_distinct_source_backed_hypotheses_with_no_search():
    protocol = load_next_cohort_protocol(_protocol_path())
    hypotheses = protocol["hypotheses"]
    assert {item["mathematical_cluster"] for item in hypotheses} == EXPECTED_CLUSTERS
    assert all(item["source_file"].endswith(".md") for item in hypotheses)
    assert protocol["common_contract"]["parameter_grid_search"] is False
    assert protocol["common_contract"]["post_hoc_combination_search"] is False
    assert protocol["strategy_trials_reserved"] is False
    assert protocol["backtests_started"] is False
    assert len(next_cohort_protocol_hash(protocol)) == 64


def test_protocol_rejects_hidden_sixth_hypothesis_and_sign_search():
    protocol = load_next_cohort_protocol(_protocol_path())
    protocol["hypotheses"].append(dict(protocol["hypotheses"][0], sequence=6))
    with pytest.raises(ValueError, match="exactly five"):
        validate_next_cohort_protocol(protocol)
    protocol = load_next_cohort_protocol(_protocol_path())
    protocol["common_contract"]["alternative_sign_search"] = True
    with pytest.raises(ValueError, match="permits no search"):
        validate_next_cohort_protocol(protocol)


def test_protocol_rejects_dry_run_authority():
    protocol = load_next_cohort_protocol(_protocol_path())
    protocol["stopping_rules"]["dry_run_authorized"] = True
    with pytest.raises(ValueError, match="execution"):
        validate_next_cohort_protocol(protocol)


def test_approved_draft02_materializes_four_hypotheses_and_parks_pairs():
    repo_root = Path(__file__).resolve().parents[1]
    protocol = materialize_next_cohort_protocol(
        _protocol_path(),
        repo_root / "configs/qforge20_malaysia_next_cohort_draft02_revision.json",
    )
    assert len(protocol["hypotheses"]) == 4
    assert [item["sequence"] for item in protocol["hypotheses"]] == [1, 2, 3, 4]
    assert protocol["kiv_hypotheses"][0]["hypothesis_id"] == "MYS-H-NEXT-PAIR-504-63"
    assert protocol["validation"]["multiplicity_correction"] == "Holm_across_all_4_primary_hypotheses"
    assert protocol["common_contract"]["interim_performance_monitoring"] == "forbidden"
    assert len(materialized_next_cohort_protocol_hash(protocol)) == 64
