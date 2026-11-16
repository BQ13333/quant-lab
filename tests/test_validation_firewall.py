from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.firewall import (
    ResearchDataFirewall,
    ResearchFailureCode,
    evaluate_research_gate,
)
from quant_lab.validation.terminal_holdout import FrozenCandidateBundle, TerminalHoldoutService


def panels():
    index = pd.bdate_range("2020-01-01", periods=900)
    rng = np.random.default_rng(42)
    features = pd.DataFrame({"lagged_momentum": rng.normal(size=len(index))}, index=index)
    returns = pd.DataFrame(
        {
            "stable": rng.normal(0.001, 0.01, len(index)),
            "weak": rng.normal(0.0, 0.01, len(index)),
            "bad": rng.normal(-0.001, 0.01, len(index)),
        },
        index=index,
    )
    available = pd.DataFrame({"lagged_momentum": index - pd.Timedelta(days=1)}, index=index)
    return features, returns, available


def test_adversarial_holdout_changes_cannot_change_research_rank_or_hash():
    features, returns, available = panels()
    firewall = ResearchDataFirewall(min_observations=504)
    end = features.index[649]
    first = firewall.seal_research_inputs(
        features=features,
        candidate_returns=returns,
        available_at=available, universe_available_at=available,
        research_end=end,
    )
    changed_features = features.copy()
    changed_returns = returns.copy()
    changed_features.loc[changed_features.index > end] = 1e12
    changed_returns.loc[changed_returns.index > end, "bad"] = 100.0
    second = firewall.seal_research_inputs(
        features=changed_features,
        candidate_returns=changed_returns,
        available_at=available, universe_available_at=available,
        research_end=end,
    )
    assert first.content_hash == second.content_hash
    assert firewall.rank_candidates(first) == firewall.rank_candidates(second)


def test_pit_purge_stale_and_sample_checks_fail_closed():
    features, returns, available = panels()
    firewall = ResearchDataFirewall(min_observations=504, max_staleness_days=2)
    end = features.index[649]
    leaking = available.copy()
    leaking.iloc[0, 0] = features.index[0] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match=ResearchFailureCode.POINT_IN_TIME_VIOLATION.value):
        firewall.seal_research_inputs(
            features=features, candidate_returns=returns, available_at=leaking, research_end=end
        )
    with pytest.raises(ValueError, match=ResearchFailureCode.INVALID_PURGE_EMBARGO.value):
        firewall.seal_research_inputs(
            features=features, candidate_returns=returns, research_end=end, purge=300, embargo=300
        )
    with pytest.raises(ValueError, match=ResearchFailureCode.INSUFFICIENT_SAMPLE.value):
        firewall.seal_research_inputs(
            features=features.iloc[:100], candidate_returns=returns.iloc[:100], research_end=features.index[99]
        )
    with pytest.raises(ValueError, match=ResearchFailureCode.STALE_DATA.value):
        firewall.seal_research_inputs(
            features=features.iloc[:600],
            candidate_returns=returns.iloc[:600],
            research_end=features.index[620],
        )


def test_terminal_service_requires_frozen_bundle_and_post_freeze_evidence_only():
    bundle = FrozenCandidateBundle(
        candidate_id="C1",
        candidate_hash="a" * 64,
        code_hash="b" * 64,
        data_hash="c" * 64,
        config_hash="d" * 64,
        research_end="2023-12-29",
        frozen_at="2026-07-12T00:00:00+00:00",
    )
    service = TerminalHoldoutService(min_forward_observations=3)
    contaminated = pd.Series([100.0] * 10, index=pd.bdate_range("2026-06-01", periods=10))
    insufficient = service.evaluate(bundle, forward_returns=contaminated)
    assert insufficient.passed is False
    assert insufficient.reason_codes == ("INSUFFICIENT_FORWARD_EVIDENCE",)
    assert insufficient.observation_count == 0
    future = pd.Series([0.01, 0.02, 0.01], index=pd.bdate_range("2026-07-07", periods=3))
    result = service.evaluate(bundle, forward_returns=pd.concat([contaminated, future]))
    assert result.passed is True
    assert not ({"sharpe", "score", "rank", "mutation"} & {field.name for field in dataclasses.fields(result)})


def test_historical_period_is_explicitly_contaminated_audit_only():
    features, returns, available = panels()
    dataset = ResearchDataFirewall(min_observations=504).seal_research_inputs(
        features=features,
        candidate_returns=returns,
        available_at=available, universe_available_at=available,
        research_end=features.index[649],
    )
    assert dataset.historical_label == "research_or_contaminated_audit_only"


def test_research_gate_uses_lifetime_attempts_and_returns_blinded_codes(tmp_path):
    features, returns, available = panels()
    dataset = ResearchDataFirewall(min_observations=504).seal_research_inputs(
        features=features,
        candidate_returns=returns,
        available_at=available, universe_available_at=available,
        research_end=features.index[649],
    )
    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps({"run_id": "MALAYSIA-SWEEP-LEGACY", "candidate_count": 3330}))
    ledger = TrialLedger(tmp_path / "trials.sqlite", tmp_path / "evidence")
    ledger.import_legacy_malaysia_sweep(manifest)
    result = evaluate_research_gate(
        dataset,
        candidate_id="stable",
        ledger=ledger,
        observed_trial_sharpes=[0.2, 0.3],
    )
    assert result.lifetime_attempts == 3330
    assert all(code in {item.value for item in ResearchFailureCode} for code in result.mutation_feedback())
    assert all(not any(char.isdigit() for char in code) for code in result.mutation_feedback())


def test_research_dependency_graph_cannot_import_terminal_holdout():
    offenders = []
    for path in Path("src/quant_lab/research").glob("*.py"):
        if "terminal_holdout" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []


def test_pit_ablation_optimizes_research_slice_not_full_panel():
    source = Path("src/quant_lab/research/pit_ablation.py").read_text(encoding="utf-8")
    assert "prices = full_prices.loc[full_prices.index < research_end]" in source
    assert "prices=prices" in source
    assert "_add_holdout_metrics" not in source


def test_model_runtime_stays_on_research_side_of_firewall():
    provider_root = Path("src/quant_lab/research/providers")
    assert provider_root.is_dir()
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in provider_root.glob("*.py"))
    assert "quant_lab.execution" not in source
    assert "ibkr" not in source
    assert "openrouter" not in source


def sealed():
    features, returns, available=panels()
    return ResearchDataFirewall().seal_research_inputs(features=features,candidate_returns=returns,available_at=available,universe_available_at=available,research_end=features.index[649])


def test_cross_market_trials_raise_charged_count(tmp_path):
    ledger=TrialLedger(tmp_path/'ledger.sqlite',tmp_path/'evidence')
    for i,market in enumerate(('MY','HK','CN')):
        reservation=ledger.reserve_trial(idempotency_key=market,campaign=market,family=market,hypothesis_id=market,candidate_id=market,code_hash='a'*64,data_hash='b'*64,config_hash='c'*64,prompt_hash='d'*64,model_hash='e'*64)
        ledger.record_outcome(reservation.trial_id,status="success",payload={"sharpe": .2+i*.1})
    snap=ledger.snapshot()
    assert snap.global_count==3 and dict(snap.family_counts)==dict(MY=1,HK=1,CN=1)
    result=evaluate_research_gate(sealed(),candidate_id='stable',ledger=ledger,observed_trial_sharpes=snap.sharpes,snapshot=snap,benchmark_name='CASH')
    assert result.lifetime_attempts==3
    assert ResearchFailureCode.LEDGER_UNAVAILABLE not in result.failure_codes
    changed=dataclasses.replace(snap,dispersion_receipt=(("forged",99.),))
    assert ResearchFailureCode.LEDGER_UNAVAILABLE in evaluate_research_gate(sealed(),candidate_id="stable",ledger=ledger,observed_trial_sharpes=changed.sharpes,snapshot=changed,benchmark_name="CASH").failure_codes


def test_gate_rejects_unbound_trial_sharpes(tmp_path):
    ledger=TrialLedger(tmp_path/'ledger.sqlite',tmp_path/'evidence')
    r=evaluate_research_gate(sealed(),candidate_id='stable',ledger=ledger,observed_trial_sharpes=[.2,.3],snapshot=ledger.snapshot(),benchmark_name='CASH')
    assert not r.passed and ResearchFailureCode.LEDGER_UNAVAILABLE in r.failure_codes
    with pytest.raises(ValueError,match='thresholds'):
        evaluate_research_gate(sealed(),candidate_id='stable',ledger=ledger,observed_trial_sharpes=[],min_deflated_sharpe=float('nan'))


def test_missing_available_at_fails():
    f,r,a=panels()
    for kwargs in ({},{'available_at':a},{'universe_available_at':a}):
        with pytest.raises(ValueError,match='POINT_IN_TIME'):
            ResearchDataFirewall().seal_research_inputs(features=f,candidate_returns=r,research_end=f.index[649],**kwargs)


def test_benchmark_gap_fails(tmp_path):
    ledger=TrialLedger(tmp_path/'ledger.sqlite',tmp_path/'evidence'); data=sealed()
    r=evaluate_research_gate(data,candidate_id='stable',ledger=ledger,observed_trial_sharpes=[],benchmark_name='INDEX',benchmark_returns=pd.Series(0.,index=data.features.index[:-1]))
    assert not r.passed and ResearchFailureCode.NON_POSITIVE_NET_IR in r.failure_codes


def test_cash_benchmark_must_be_explicit(tmp_path):
    ledger=TrialLedger(tmp_path/'ledger.sqlite',tmp_path/'evidence'); data=sealed()
    omitted=evaluate_research_gate(data,candidate_id='stable',ledger=ledger,observed_trial_sharpes=[])
    cash=evaluate_research_gate(data,candidate_id='stable',ledger=ledger,observed_trial_sharpes=[],benchmark_name='CASH')
    assert ResearchFailureCode.NON_POSITIVE_NET_IR in omitted.failure_codes
    assert cash.net_oos_ir > 0
