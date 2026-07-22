from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_lab.research.malaysia_cohort_analysis import (
    build_malaysia_cohort_report,
    persist_malaysia_cohort_report,
)
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort
from quant_lab.research.trial_ledger import TrialLedger


HASHES = {
    "code_hash": "a" * 64,
    "data_hash": "b" * 64,
    "config_hash": "c" * 64,
    "prompt_hash": "d" * 64,
    "model_hash": "e" * 64,
}


def test_cohort_report_withholds_ranking_until_every_atomic_strategy_exists(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    first = malaysia_atomic_cohort()[0]
    trial = ledger.reserve_trial(
        idempotency_key="cohort-one",
        campaign="malaysia",
        family=first.family,
        hypothesis_id=first.hypothesis_id,
        candidate_id=f"MYS-C-{first.strategy_id}",
        **HASHES,
    )
    index = pd.bdate_range("2022-01-03", periods=600)
    ledger.record_candidate_returns(
        trial.trial_id,
        candidate_id=f"MYS-C-{first.strategy_id}",
        family=first.family,
        returns=pd.Series(np.random.default_rng(1).normal(0.0001, 0.01, 600), index=index),
    )
    report = build_malaysia_cohort_report(ledger)
    assert report["completed_strategy_count"] == 1
    assert not report["cohort_closed"] and not report["ranking_available"]
    assert report["atomic_metrics"] is None
    assert report["correlation_matrix"] is None


def test_closed_cohort_emits_multiplicity_correlation_pbo_and_only_predeclared_combinations(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    index = pd.bdate_range("2022-01-03", periods=600)
    rng = np.random.default_rng(8)
    cohort = malaysia_atomic_cohort()
    for number, strategy in enumerate(cohort):
        trial = ledger.reserve_trial(
            idempotency_key=f"cohort-{number}",
            campaign="malaysia",
            family=strategy.family,
            hypothesis_id=strategy.hypothesis_id,
            candidate_id=f"MYS-C-{strategy.strategy_id}",
            **HASHES,
        )
        returns = pd.Series(
            0.00005 * (number + 1) + rng.normal(0.0, 0.01, len(index)),
            index=index,
        )
        ledger.record_candidate_returns(
            trial.trial_id,
            candidate_id=f"MYS-C-{strategy.strategy_id}",
            family=strategy.family,
            returns=returns,
        )
    report = build_malaysia_cohort_report(ledger)
    assert report["cohort_closed"] and report["ranking_available"]
    assert len(report["atomic_metrics"]) == 10
    assert all("holm_adjusted_pvalue" in item for item in report["atomic_metrics"].values())
    assert len(report["correlation_matrix"]) == 10
    assert 0.0 <= report["pbo"] <= 1.0
    assert set(report["predeclared_combination_metrics"]) == {
        "MYS-COMBO-TREND-LOWVOL",
        "MYS-COMBO-MOMENTUM-REVERSION",
        "MYS-COMBO-BREAKOUT-RESIDUAL-LOWVOL",
    }
    target = persist_malaysia_cohort_report(ledger, tmp_path / "cohort-report.json")
    assert json.loads(target.read_text(encoding="utf-8")) == report
