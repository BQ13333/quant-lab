from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_lab.agents.specialist_runtime import SpecialistReviewRuntime
from quant_lab.research.bounded_loop import (
    ALLOWED_FEATURES,
    BASE_COST_BPS_ONE_WAY,
    STRESS_COST_BPS_ONE_WAY,
    BacktestEvidence,
    BoundedMalaysiaResearchLoop,
    CampaignBudgets,
    CampaignConfig,
    CampaignStopCode,
    CandidateGeneration,
    DeterministicDryRunSimulator,
    InjectedLoopCrash,
    InstitutionalAuditEvidence,
    MalaysiaUniverseContract,
    MutationDirective,
)
from quant_lab.research.contracts import CandidateSpec, ModelCallRecord, SpecialistReviewDraft
from quant_lab.research.institutional_workflow import HypothesisIntakeRegistry, HypothesisTicket
from quant_lab.research.providers import LocalModelResult, LocalModelRouter
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.firewall import ResearchFailureCode


HASH = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_universe(path: Path) -> MalaysiaUniverseContract:
    tickers = [f"{1000 + index:04d}.KL" for index in range(62)]
    pd.DataFrame(
        {
            "ticker": tickers,
            "country": ["Malaysia"] * 62,
            "u2_backtest_eligible": [True] * 62,
            "u3_liquid_proxy": [True] * 34 + [False] * 28,
        }
    ).to_csv(path, index=False)
    return MalaysiaUniverseContract.from_tiers_csv(path)


class PassingReviewProvider:
    def __init__(self, verdict: str = "pass") -> None:
        self.verdict = verdict

    def generate(self, request, *, model: str):
        role = request.payload["specialist_id"]
        review = SpecialistReviewDraft(
            verdict=self.verdict,
            claims=("The candidate declares a testable Malaysia research claim.",),
            assumptions=("Signals are lagged and costs are fixed before evaluation.",),
            counterexamples=(),
            falsification_tests=("Run purged U3 primary and unchanged U2 robustness tests.",),
            veto=False,
            confidence=0.7,
        )
        now = datetime.now(timezone.utc).isoformat()
        record = ModelCallRecord(
            call_id=request.request_id,
            provider="ollama",
            model=model,
            endpoint="http://127.0.0.1:11434",
            started_at=now,
            completed_at=now,
            status="fake",
            prompt_hash=digest(request.system_prompt + request.prompt_payload_json()),
            schema_hash=digest("specialist_review"),
            config_hash=digest(json.dumps(dict(request.config), sort_keys=True)),
            response_hash=digest(review.canonical_json()),
            input_tokens=3,
            output_tokens=5,
            stop_reason="deterministic_fake",
            redactions=("prompt_body", "response_body"),
        )
        return LocalModelResult(review, record, request.fingerprint(model))


class ScriptedCandidateGenerator:
    def __init__(self, ledger: TrialLedger, *, external_cost_usd: float = 0.0) -> None:
        self.ledger = ledger
        self.external_cost_usd = external_cost_usd
        self.directives: list[MutationDirective] = []

    def generate(self, directive: MutationDirective, *, trial_id: str, config: CampaignConfig):
        manifest = self.ledger.replay_manifest(trial_id)
        assert manifest["events"][0]["event_type"] == "reserved"
        assert not any(item["event_type"] == "backtest" for item in manifest["events"])
        self.directives.append(directive)
        candidate = CandidateSpec(
            candidate_id=f"MYS-C-{directive.hypothesis_id}-V{directive.variant}-G{directive.generation}",
            hypothesis_id=directive.hypothesis_id,
            family=directive.family,
            implementation_ref="quant_lab.models.baselines:MomentumModel",
            parameters={
                "lookback": 63 + 21 * (directive.generation - 1),
                "rebalance_frequency": "W-FRI",
                "top_fraction": 0.20,
                "generation": directive.generation,
                "mutation_codes": list(directive.mutation_codes),
                "features": list(ALLOWED_FEATURES),
            },
            evidence=(),
            provenance_hashes={"universe": "b" * 64, "config": "c" * 64},
            parent_candidate_id=directive.parent_candidate_id,
        )
        now = datetime.now(timezone.utc).isoformat()
        record = ModelCallRecord(
            call_id=f"CALL-{directive.directive_id}",
            provider="ollama",
            model="qwen3.5:35b-mlx",
            endpoint="http://127.0.0.1:11434",
            started_at=now,
            completed_at=now,
            status="fake",
            prompt_hash=digest(json.dumps({"mutation_codes": directive.mutation_codes})),
            schema_hash=digest("candidate"),
            config_hash=digest(json.dumps({"features": config.features})),
            response_hash=digest(candidate.canonical_json()),
            input_tokens=4,
            output_tokens=6,
            stop_reason="deterministic_fake",
            redactions=("prompt_body", "response_body"),
        )
        return CandidateGeneration(candidate, record, self.external_cost_usd)


class ScriptedBacktester:
    def __init__(self, ledger: TrialLedger, *, fail_first: bool = True, raw_only: bool = False) -> None:
        self.ledger = ledger
        self.fail_first = fail_first
        self.raw_only = raw_only
        self.calls: list[str] = []

    def evaluate(self, candidate, *, trial_id, universe, config):
        manifest = self.ledger.replay_manifest(trial_id)
        assert manifest["events"][0]["event_type"] == "reserved"
        assert any(item["event_type"] == "candidate_extracted" for item in manifest["events"])
        self.calls.append(candidate.candidate_id)
        generation = int(candidate.parameters["generation"])
        fail = self.fail_first and generation == 1
        if self.raw_only:
            fail = True
        return BacktestEvidence(
            candidate_id=candidate.candidate_id,
            u3_count=34,
            u2_count=62,
            base_cost_bps_one_way=BASE_COST_BPS_ONE_WAY,
            stress_cost_bps_one_way=STRESS_COST_BPS_ONE_WAY,
            raw_sharpe=99.0 if self.raw_only else 1.2,
            net_oos_ir=-0.1 if self.raw_only else 0.8,
            deflated_sharpe_probability=0.2 if self.raw_only else 0.98,
            pbo=0.9 if self.raw_only else 0.10,
            max_drawdown=-0.35 if self.raw_only else -0.10,
            turnover=5.0 if self.raw_only else 1.2,
            stress_net_oos_ir=-0.1 if fail else 0.3,
            max_asset_weight=0.5 if self.raw_only else 0.20,
            u2_net_oos_ir=-0.1 if self.raw_only else 0.4,
            target_weights={universe.u3[0]: 0.2, universe.u3[1]: 0.15},
            reference_prices={universe.u3[0]: 10.0, universe.u3[1]: 20.0},
            config_hash=digest(json.dumps({"candidate": candidate.content_hash(), "cost": 20.0})),
        )


class PassingInstitutionalAuditor:
    def __init__(self, failure_code: ResearchFailureCode | None = None) -> None:
        self.calls: list[str] = []
        self.failure_code = failure_code

    def audit(self, candidate, evidence, *, trial_id, universe, config):
        self.calls.append(candidate.candidate_id)
        return InstitutionalAuditEvidence(
            candidate_id=candidate.candidate_id,
            regime_count=1 if self.failure_code == ResearchFailureCode.REGIME_INSTABILITY else 4,
            min_regime_observations=80,
            positive_regime_fraction=0.75,
            max_positive_pnl_share=0.50,
            worst_regime_ir=-0.20,
            factor_residual_ir=0.0 if self.failure_code == ResearchFailureCode.FACTOR_RESIDUAL_FAILURE else 0.60,
            factor_r_squared=0.30,
            factor_alpha_t_stat=2.10,
            decay_latest_126=-0.10 if self.failure_code == ResearchFailureCode.ALPHA_DECAY_FAILURE else 0.70,
            decay_latest_252=0.80,
            input_hash=digest(candidate.candidate_id + "audit-input"),
            methodology_hash=digest("malaysia-institutional-audit-v1"),
            audit_hash=digest(candidate.candidate_id + "institutional-audit"),
        )


def make_loop(
    tmp_path: Path,
    *,
    budgets: CampaignBudgets | None = None,
    fail_first: bool = True,
    raw_only: bool = False,
    audit_failure: ResearchFailureCode | None = None,
    review_verdict: str = "pass",
    external_cost_usd: float = 0.0,
    monotonic=lambda: 0.0,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    universe = write_universe(tmp_path / "tiers.csv")
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    registry = HypothesisIntakeRegistry(tmp_path / "ledger.sqlite")
    admission = registry.admit(
        HypothesisTicket(
            hypothesis_id="MYS-H-H001",
            economic_claim="Lagged medium horizon momentum persists after Malaysia trading costs.",
            family="momentum",
            features=ALLOWED_FEATURES,
            horizon_days=63,
            data_requirements=("adjusted prices", "lagged volume", "daily returns"),
        )
    )
    assert admission.accepted
    directive = MutationDirective("MYS-SEED-001", "MYS-H-H001", "momentum")
    config = CampaignConfig(
        campaign_id="MALAYSIA-BOUNDED-TEST",
        artifact_root=tmp_path / "artifacts",
        initial_queue=(directive,),
        budgets=budgets or CampaignBudgets(max_trials=3, max_trials_per_family=3, max_generations=2),
    )
    generator = ScriptedCandidateGenerator(ledger, external_cost_usd=external_cost_usd)
    reviewers = SpecialistReviewRuntime(LocalModelRouter(PassingReviewProvider(review_verdict)), ledger=ledger)
    backtester = ScriptedBacktester(ledger, fail_first=fail_first, raw_only=raw_only)
    institutional_auditor = PassingInstitutionalAuditor(audit_failure)
    dry = DeterministicDryRunSimulator()
    loop = BoundedMalaysiaResearchLoop(
        config=config,
        universe=universe,
        ledger=ledger,
        hypothesis_registry=registry,
        candidate_generator=generator,
        specialist_runtime=reviewers,
        backtester=backtester,
        institutional_auditor=institutional_auditor,
        dry_runner=dry,
        monotonic=monotonic,
    )
    return loop, ledger, generator, backtester, dry


def test_two_iterations_consume_blinded_mutation_and_preserve_parent_lineage(tmp_path):
    loop, ledger, generator, backtester, dry = make_loop(tmp_path)
    replay = loop.run()
    assert len(replay.transitions) == 2
    first, second = replay.transitions
    assert first["failure_codes_produced"] == [ResearchFailureCode.COST_STRESS_FAILURE.value]
    assert second["mutation_codes_consumed"] == first["failure_codes_produced"]
    assert second["parent_trial_id"] == first["trial_id"]
    assert second["parent_candidate_id"] == first["candidate_id"]
    assert ledger.lineage(second["trial_id"]) == [first["trial_id"], second["trial_id"]]
    assert generator.directives[0].mutation_codes == ()
    assert generator.directives[1].mutation_codes == (ResearchFailureCode.COST_STRESS_FAILURE.value,)
    assert generator.directives[0].family == generator.directives[1].family
    assert len(backtester.calls) == 2 and dry.calls == 1


def test_every_candidate_is_reserved_before_local_call_and_backtest(tmp_path):
    loop, ledger, _, _, _ = make_loop(tmp_path, fail_first=False)
    replay = loop.run()
    manifest = ledger.replay_manifest(replay.transitions[0]["trial_id"])
    event_types = [item["event_type"] for item in manifest["events"]]
    assert event_types[0] == "reserved"
    assert event_types.index("hypothesis_intake") < event_types.index("model_call")
    assert event_types.index("model_call") < event_types.index("candidate_extracted") < event_types.index("backtest")
    assert event_types.index("backtest") < event_types.index("institutional_audit") < event_types.index("dry_run")


@pytest.mark.parametrize(
    "budgets,expected",
    [
        (CampaignBudgets(max_trials=1, max_trials_per_family=1, max_generations=2), CampaignStopCode.MAX_TRIALS),
        (CampaignBudgets(max_trials=3, max_trials_per_family=3, max_generations=2, max_consecutive_failures=1), CampaignStopCode.CONSECUTIVE_FAILURES),
        (CampaignBudgets(max_trials=3, max_trials_per_family=1, max_generations=2), CampaignStopCode.FAMILY_TRIAL_BUDGET),
        (CampaignBudgets(max_trials=3, max_trials_per_family=3, max_generations=2, max_tokens=8192, token_reserve_per_trial=8192), CampaignStopCode.TOKEN_BUDGET),
    ],
)
def test_trial_family_token_and_failure_budgets_stop_with_persisted_codes(tmp_path, budgets, expected):
    loop, _, _, _, _ = make_loop(tmp_path, budgets=budgets)
    loop.run()
    state = json.loads(loop.state_path.read_text())
    assert expected.value in state["stop_reasons"]
    assert state["usage"]["trials"] <= budgets.max_trials


def test_wall_time_and_external_cost_have_typed_persisted_stops(tmp_path):
    clock_values = iter((0.0, 2.0, 2.0))
    budgets = CampaignBudgets(max_trials=1, max_trials_per_family=1, max_wall_seconds=1.0)
    loop, _, _, _, _ = make_loop(tmp_path / "wall", budgets=budgets, monotonic=lambda: next(clock_values))
    loop.run()
    assert CampaignStopCode.WALL_TIME_BUDGET.value in json.loads(loop.state_path.read_text())["stop_reasons"]

    loop, _, _, _, _ = make_loop(tmp_path / "cost", external_cost_usd=0.01, fail_first=False)
    loop.run()
    assert CampaignStopCode.EXTERNAL_COST_BUDGET.value in json.loads(loop.state_path.read_text())["stop_reasons"]


def test_only_joint_gate_pass_enters_deterministic_dry_run(tmp_path):
    failed_loop, _, _, _, failed_dry = make_loop(tmp_path / "failed", raw_only=True)
    failed = failed_loop.run().transitions[0]
    assert failed["dry_run_completed"] is False
    assert failed_dry.calls == 0
    assert set(failed["failure_codes_produced"]) >= {
        ResearchFailureCode.DEFLATED_SHARPE_FAILURE.value,
        ResearchFailureCode.PBO_EXCESSIVE.value,
        ResearchFailureCode.U2_ROBUSTNESS_FAILURE.value,
    }

    passed_loop, ledger, _, _, passed_dry = make_loop(tmp_path / "passed", fail_first=False)
    passed = passed_loop.run().transitions[0]
    assert passed["dry_run_completed"] is True and passed_dry.calls == 1
    manifest = ledger.replay_manifest(passed["trial_id"])
    event = next(item for item in manifest["events"] if item["event_type"] == "dry_run")
    dry = json.loads(ledger.verify_evidence(event["evidence_sha256"]).path.read_text())
    assert dry["configuration_hash"]
    assert all(set(fill) >= {"status", "slippage_cost", "fee", "rejection_reason"} for fill in dry["fills"])


@pytest.mark.parametrize(
    "failure_code",
    [
        ResearchFailureCode.REGIME_INSTABILITY,
        ResearchFailureCode.FACTOR_RESIDUAL_FAILURE,
        ResearchFailureCode.ALPHA_DECAY_FAILURE,
    ],
)
def test_each_institutional_failure_blocks_dry_run_with_blinded_code(tmp_path, failure_code):
    loop, ledger, _, _, dry = make_loop(
        tmp_path,
        fail_first=False,
        audit_failure=failure_code,
        budgets=CampaignBudgets(max_trials=1, max_trials_per_family=1, max_generations=1),
    )
    transition = loop.run().transitions[0]
    assert transition["failure_codes_produced"] == [failure_code.value]
    assert transition["institutional_audit_completed"] is True
    assert transition["dry_run_completed"] is False and dry.calls == 0
    event_types = [item["event_type"] for item in ledger.replay_manifest(transition["trial_id"])["events"]]
    assert "institutional_audit" in event_types and "dry_run" not in event_types


def test_conditional_panel_generates_backtest_evidence_but_cannot_reach_dry_run(tmp_path):
    loop, ledger, _, backtester, dry = make_loop(
        tmp_path,
        fail_first=False,
        review_verdict="conditional",
        budgets=CampaignBudgets(max_trials=1, max_trials_per_family=1, max_generations=1),
    )
    transition = loop.run().transitions[0]
    assert len(backtester.calls) == 1
    assert transition["failure_codes_produced"] == [ResearchFailureCode.SPECIALIST_REVIEW_FAILURE.value]
    assert transition["institutional_audit_completed"] is False
    assert transition["dry_run_completed"] is False and dry.calls == 0
    event_types = [item["event_type"] for item in ledger.replay_manifest(transition["trial_id"])["events"]]
    assert "backtest" in event_types and "institutional_audit" not in event_types and "dry_run" not in event_types


def test_crash_resume_skips_completed_local_candidate_call(tmp_path):
    loop, _, generator, _, _ = make_loop(tmp_path, fail_first=False)
    with pytest.raises(InjectedLoopCrash, match="candidate_extracted"):
        loop.run(crash_after_stage="candidate_extracted")
    assert len(generator.directives) == 1
    replay = loop.run()
    assert len(generator.directives) == 1
    assert len(replay.transitions) == 1


def test_crash_resume_reuses_persisted_institutional_audit(tmp_path):
    loop, _, _, _, dry = make_loop(tmp_path, fail_first=False)
    auditor = loop.institutional_auditor
    with pytest.raises(InjectedLoopCrash, match="audited"):
        loop.run(crash_after_stage="audited")
    assert len(auditor.calls) == 1
    replay = loop.run()
    assert len(auditor.calls) == 1
    assert replay.transitions[0]["dry_run_completed"] is True and dry.calls == 1


def test_replay_has_identical_transitions_and_artifact_hash(tmp_path):
    loop, _, _, _, _ = make_loop(tmp_path, fail_first=False)
    first = loop.run()
    second = loop.replay()
    assert first.transitions == second.transitions
    assert first.artifact_hash == second.artifact_hash


def test_fixed_universe_costs_features_and_caps_cannot_be_optimized(tmp_path):
    universe = write_universe(tmp_path / "tiers.csv")
    assert len(universe.u3) == 34 and len(universe.u2) == 62
    assert set(universe.u3).issubset(universe.u2)
    with pytest.raises(ValueError, match="costs are predeclared"):
        CampaignConfig(
            "MALAYSIA-BAD-COST",
            tmp_path,
            (MutationDirective("D", "H", "momentum"),),
            base_cost_bps_one_way=10.0,
        ).validate()
    with pytest.raises(ValueError, match="feature set is frozen"):
        CampaignConfig(
            "MALAYSIA-BAD-FEATURE",
            tmp_path,
            (MutationDirective("D", "H", "momentum"),),
            features=("news_sentiment",),
        ).validate()
    with pytest.raises(ValueError, match="bounded Malaysia"):
        CampaignBudgets(max_trials=31).validate()
    with pytest.raises(ValueError, match="traversal-safe"):
        CampaignConfig(
            "MALAYSIA-../../escape",
            tmp_path,
            (MutationDirective("D", "H", "momentum"),),
        ).validate()


def test_mutation_feedback_contains_codes_not_metrics_or_holdout(tmp_path):
    loop, _, generator, _, _ = make_loop(tmp_path)
    replay = loop.run()
    consumed = generator.directives[1].mutation_codes
    assert consumed == tuple(replay.transitions[0]["failure_codes_produced"])
    assert all(code in {item.value for item in ResearchFailureCode} for code in consumed)
    assert "sharpe" not in json.dumps(consumed).lower()
    assert "holdout" not in json.dumps(consumed).lower()


def test_controller_and_simulator_have_no_live_submission_path():
    source = Path("src/quant_lab/research/bounded_loop.py").read_text().lower()
    for forbidden in (
        "quant_lab.execution",
        "ibkr",
        "submit_order",
        "place_order",
        "transmit_orders",
        "ib_insync",
    ):
        assert forbidden not in source


def test_operator_docs_cover_local_setup_replay_holdout_and_recovery():
    architecture = Path("docs/LOCAL_OLLAMA_MALAYSIA_LOOP.md").read_text().lower()
    recovery = Path("docs/MALAYSIA_LOOP_RECOVERY.md").read_text().lower()
    for phrase in (
        "no hosted fallback",
        "u3, exactly 34",
        "u2, exactly 62",
        "raw sharpe",
        "terminal holdout",
        "offline replay walkthrough",
        "no broker",
    ):
        assert phrase in architecture
    for phrase in (
        "normal resume after interruption",
        "malformed local output",
        "ledger contention",
        "evidence corruption",
        "budget stop",
        "safe rollback",
    ):
        assert phrase in recovery
