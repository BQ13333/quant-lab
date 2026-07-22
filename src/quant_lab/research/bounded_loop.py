from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import pandas as pd

from quant_lab.agents.specialist_runtime import MalaysiaReviewDossier, PanelDisposition, SpecialistReviewRuntime
from quant_lab.research.contracts import CandidateSpec, ModelCallRecord
from quant_lab.research.malaysia_strategy_registry import validate_strategy_parameters
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.firewall import ResearchFailureCode


U3_COUNT = 34
U2_COUNT = 62
BASE_COST_BPS_ONE_WAY = 20.0
STRESS_COST_BPS_ONE_WAY = 40.0
ALLOWED_FEATURES = ("price", "volume", "volatility", "lagged_liquidity")
MAX_FIRST_HYPOTHESES = 10
MAX_VARIANTS_PER_HYPOTHESIS = 3
MAX_NEW_TRIALS = 30
MAX_MUTATION_GENERATIONS = 3


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class CampaignStopCode(str, Enum):
    QUEUE_EMPTY = "QUEUE_EMPTY"
    MAX_HYPOTHESES = "MAX_HYPOTHESES"
    MAX_VARIANTS = "MAX_VARIANTS"
    MAX_TRIALS = "MAX_TRIALS"
    MAX_GENERATIONS = "MAX_GENERATIONS"
    FAMILY_TRIAL_BUDGET = "FAMILY_TRIAL_BUDGET"
    TOKEN_BUDGET = "TOKEN_BUDGET"
    EXTERNAL_COST_BUDGET = "EXTERNAL_COST_BUDGET"
    WALL_TIME_BUDGET = "WALL_TIME_BUDGET"
    CONSECUTIVE_FAILURES = "CONSECUTIVE_FAILURES"


class InjectedLoopCrash(RuntimeError):
    pass


@dataclass(frozen=True)
class MalaysiaUniverseContract:
    u3: tuple[str, ...]
    u2: tuple[str, ...]
    source_hash: str

    @classmethod
    def from_tiers_csv(cls, path: str | Path) -> "MalaysiaUniverseContract":
        source = Path(path)
        content = source.read_bytes()
        frame = pd.read_csv(source)
        required = {"ticker", "country", "u2_backtest_eligible", "u3_liquid_proxy"}
        if not required.issubset(frame.columns):
            raise ValueError("Malaysia universe source is missing required tier fields")
        malaysia = frame.loc[frame["country"].eq("Malaysia")].copy()
        u2 = tuple(sorted(malaysia.loc[malaysia["u2_backtest_eligible"].fillna(False).astype(bool), "ticker"].astype(str)))
        u3 = tuple(sorted(malaysia.loc[malaysia["u3_liquid_proxy"].fillna(False).astype(bool), "ticker"].astype(str)))
        result = cls(u3, u2, hashlib.sha256(content).hexdigest())
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.u3) != U3_COUNT or len(self.u2) != U2_COUNT:
            raise ValueError("Malaysia pilot requires exactly 34 U3 and 62 U2 stocks")
        if len(set(self.u3)) != U3_COUNT or len(set(self.u2)) != U2_COUNT:
            raise ValueError("Malaysia universe contains duplicate tickers")
        if not set(self.u3).issubset(self.u2):
            raise ValueError("Malaysia U3 must be a strict subset of U2")
        if any(not ticker.endswith(".KL") for ticker in self.u2):
            raise ValueError("Malaysia pilot admits Bursa .KL securities only")

    def content_hash(self) -> str:
        self.validate()
        return _hash_text(_canonical(asdict(self)))


@dataclass(frozen=True)
class MutationDirective:
    directive_id: str
    hypothesis_id: str
    family: str
    variant: int = 1
    generation: int = 1
    parent_trial_id: str | None = None
    parent_candidate_id: str | None = None
    mutation_codes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not all(value.strip() for value in (self.directive_id, self.hypothesis_id, self.family)):
            raise ValueError("mutation directive identity is required")
        if not 1 <= self.variant <= MAX_VARIANTS_PER_HYPOTHESIS:
            raise ValueError(CampaignStopCode.MAX_VARIANTS.value)
        if not 1 <= self.generation <= MAX_MUTATION_GENERATIONS:
            raise ValueError(CampaignStopCode.MAX_GENERATIONS.value)
        allowed = {item.value for item in ResearchFailureCode}
        if set(self.mutation_codes) - allowed:
            raise ValueError("mutation queue accepts enumerated blinded failure codes only")
        if self.generation > 1 and (not self.parent_trial_id or not self.parent_candidate_id or not self.mutation_codes):
            raise ValueError("child mutation must identify its parent and blinded failure codes")


@dataclass(frozen=True)
class CampaignBudgets:
    max_hypotheses: int = 10
    max_variants_per_hypothesis: int = 3
    max_trials: int = 30
    max_generations: int = 3
    max_trials_per_family: int = 9
    max_tokens: int = 200_000
    token_reserve_per_trial: int = 8_192
    max_external_cost_usd: float = 0.0
    max_wall_seconds: float = 21_600.0
    max_consecutive_failures: int = 5

    def validate(self) -> None:
        bounds = (
            1 <= self.max_hypotheses <= MAX_FIRST_HYPOTHESES,
            1 <= self.max_variants_per_hypothesis <= MAX_VARIANTS_PER_HYPOTHESIS,
            1 <= self.max_trials <= MAX_NEW_TRIALS,
            1 <= self.max_generations <= MAX_MUTATION_GENERATIONS,
            1 <= self.max_trials_per_family <= self.max_trials,
            self.max_tokens > 0,
            0 < self.token_reserve_per_trial <= self.max_tokens,
            self.max_external_cost_usd >= 0.0,
            self.max_wall_seconds > 0.0,
            self.max_consecutive_failures > 0,
        )
        if not all(bounds):
            raise ValueError("campaign budgets exceed the bounded Malaysia pilot limits")


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    artifact_root: Path
    initial_queue: tuple[MutationDirective, ...]
    budgets: CampaignBudgets = field(default_factory=CampaignBudgets)
    base_cost_bps_one_way: float = BASE_COST_BPS_ONE_WAY
    stress_cost_bps_one_way: float = STRESS_COST_BPS_ONE_WAY
    features: tuple[str, ...] = ALLOWED_FEATURES
    workflow_name: str = "Local-Ollama-Malaysia"
    model_routing_id: str = "local_ollama_routing_v1"

    def validate(self) -> None:
        self.budgets.validate()
        if not re.fullmatch(r"MALAYSIA-[A-Z0-9][A-Z0-9_-]{2,80}", self.campaign_id):
            raise ValueError("campaign_id must be a traversal-safe Malaysia identifier")
        if self.base_cost_bps_one_way != BASE_COST_BPS_ONE_WAY or self.stress_cost_bps_one_way != STRESS_COST_BPS_ONE_WAY:
            raise ValueError("Malaysia costs are predeclared at 20 bps base and 40 bps stress")
        if self.features != ALLOWED_FEATURES:
            raise ValueError("Malaysia feature set is frozen to price/volume/volatility/lagged-liquidity")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{2,80}", self.workflow_name):
            raise ValueError("workflow_name is invalid")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", self.model_routing_id):
            raise ValueError("model_routing_id is invalid")
        if not self.initial_queue:
            raise ValueError("initial mutation queue must not be empty")
        if len(self.initial_queue) > self.budgets.max_trials:
            raise ValueError(CampaignStopCode.MAX_TRIALS.value)
        for item in self.initial_queue:
            item.validate()
        hypotheses = {item.hypothesis_id for item in self.initial_queue}
        if len(hypotheses) > self.budgets.max_hypotheses:
            raise ValueError(CampaignStopCode.MAX_HYPOTHESES.value)
        variants: dict[str, set[int]] = {}
        for item in self.initial_queue:
            variants.setdefault(item.hypothesis_id, set()).add(item.variant)
        if any(len(items) > self.budgets.max_variants_per_hypothesis for items in variants.values()):
            raise ValueError(CampaignStopCode.MAX_VARIANTS.value)
        family_counts: dict[str, int] = {}
        for item in self.initial_queue:
            family_counts[item.family] = family_counts.get(item.family, 0) + 1
        if any(count > self.budgets.max_trials_per_family for count in family_counts.values()):
            raise ValueError(CampaignStopCode.FAMILY_TRIAL_BUDGET.value)


@dataclass(frozen=True)
class CandidateGeneration:
    candidate: CandidateSpec
    model_call: ModelCallRecord | None
    external_cost_usd: float = 0.0


class CandidateGenerator(Protocol):
    def generate(self, directive: MutationDirective, *, trial_id: str, config: CampaignConfig) -> CandidateGeneration:
        ...


class HypothesisAdmissionGate(Protocol):
    def assert_admitted(self, hypothesis_id: str) -> None:
        ...


@dataclass(frozen=True)
class BacktestEvidence:
    candidate_id: str
    u3_count: int
    u2_count: int
    base_cost_bps_one_way: float
    stress_cost_bps_one_way: float
    raw_sharpe: float
    net_oos_ir: float
    deflated_sharpe_probability: float
    pbo: float
    max_drawdown: float
    turnover: float
    stress_net_oos_ir: float
    max_asset_weight: float
    u2_net_oos_ir: float
    target_weights: Mapping[str, float]
    reference_prices: Mapping[str, float]
    config_hash: str

    def validate(self, universe: MalaysiaUniverseContract) -> None:
        if self.u3_count != U3_COUNT or self.u2_count != U2_COUNT:
            raise ValueError("backtest must contain fixed Malaysia U3 and U2 evidence")
        if self.base_cost_bps_one_way != BASE_COST_BPS_ONE_WAY or self.stress_cost_bps_one_way != STRESS_COST_BPS_ONE_WAY:
            raise ValueError("backtest cost configuration drifted")
        if not set(self.target_weights).issubset(universe.u3):
            raise ValueError("dry-run targets must come from the Malaysia U3 primary universe")
        values = (
            self.raw_sharpe,
            self.net_oos_ir,
            self.deflated_sharpe_probability,
            self.pbo,
            self.max_drawdown,
            self.turnover,
            self.stress_net_oos_ir,
            self.max_asset_weight,
            self.u2_net_oos_ir,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("backtest metrics must be finite")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_hash):
            raise ValueError("backtest config hash is invalid")

    def failure_codes(self) -> tuple[ResearchFailureCode, ...]:
        failures: list[ResearchFailureCode] = []
        if self.net_oos_ir <= 0.0:
            failures.append(ResearchFailureCode.NON_POSITIVE_NET_IR)
        if self.deflated_sharpe_probability < 0.95:
            failures.append(ResearchFailureCode.DEFLATED_SHARPE_FAILURE)
        if self.pbo > 0.20:
            failures.append(ResearchFailureCode.PBO_EXCESSIVE)
        if self.max_drawdown < -0.20:
            failures.append(ResearchFailureCode.DRAWDOWN_EXCESSIVE)
        if self.turnover > 3.0:
            failures.append(ResearchFailureCode.TURNOVER_EXCESSIVE)
        if self.stress_net_oos_ir <= 0.0:
            failures.append(ResearchFailureCode.COST_STRESS_FAILURE)
        if self.max_asset_weight > 0.25:
            failures.append(ResearchFailureCode.CONCENTRATION_EXCESSIVE)
        if self.u2_net_oos_ir <= 0.0:
            failures.append(ResearchFailureCode.U2_ROBUSTNESS_FAILURE)
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BacktestEngine(Protocol):
    def evaluate(
        self,
        candidate: CandidateSpec,
        *,
        trial_id: str,
        universe: MalaysiaUniverseContract,
        config: CampaignConfig,
    ) -> BacktestEvidence:
        ...


@dataclass(frozen=True)
class InstitutionalAuditEvidence:
    candidate_id: str
    regime_count: int
    min_regime_observations: int
    positive_regime_fraction: float
    max_positive_pnl_share: float
    worst_regime_ir: float
    factor_residual_ir: float
    factor_r_squared: float
    factor_alpha_t_stat: float
    decay_latest_126: float
    decay_latest_252: float
    input_hash: str
    methodology_hash: str
    audit_hash: str

    def validate(self, candidate_id: str) -> None:
        if self.candidate_id != candidate_id:
            raise ValueError("institutional audit candidate identity drifted")
        values = (
            self.positive_regime_fraction,
            self.max_positive_pnl_share,
            self.worst_regime_ir,
            self.factor_residual_ir,
            self.factor_r_squared,
            self.factor_alpha_t_stat,
            self.decay_latest_126,
            self.decay_latest_252,
        )
        if self.regime_count < 0 or self.min_regime_observations < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("institutional audit metrics must be finite and non-negative in count")
        for name, value in {
            "input_hash": self.input_hash,
            "methodology_hash": self.methodology_hash,
            "audit_hash": self.audit_hash,
        }.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"institutional {name} is invalid")

    def failure_codes(self) -> tuple[ResearchFailureCode, ...]:
        failures: list[ResearchFailureCode] = []
        if (
            self.regime_count < 3
            or self.min_regime_observations < 63
            or self.positive_regime_fraction < 0.50
            or self.max_positive_pnl_share > 0.80
            or self.worst_regime_ir < -1.0
        ):
            failures.append(ResearchFailureCode.REGIME_INSTABILITY)
        if self.factor_residual_ir < 0.25 or self.factor_r_squared > 0.80 or self.factor_alpha_t_stat < 1.0:
            failures.append(ResearchFailureCode.FACTOR_RESIDUAL_FAILURE)
        if self.decay_latest_126 <= 0.0 or self.decay_latest_252 <= 0.0:
            failures.append(ResearchFailureCode.ALPHA_DECAY_FAILURE)
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalAuditor(Protocol):
    def audit(
        self,
        candidate: CandidateSpec,
        evidence: BacktestEvidence,
        *,
        trial_id: str,
        universe: MalaysiaUniverseContract,
        config: CampaignConfig,
    ) -> InstitutionalAuditEvidence:
        ...


@dataclass(frozen=True)
class DryRunFill:
    ticker: str
    status: str
    target_weight: float
    quantity: int
    reference_price: float | None
    fill_price: float | None
    slippage_cost: float
    fee: float
    rejection_reason: str | None


@dataclass(frozen=True)
class DryRunEvidence:
    candidate_id: str
    fills: tuple[DryRunFill, ...]
    slippage_bps: float
    fee_bps: float
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicDryRunSimulator:
    """Pure calculation only: no broker adapter, account, connection, or order API exists."""

    def __init__(self, *, equity: float = 1_000_000.0, slippage_bps: float = 5.0, fee_bps: float = 2.0) -> None:
        self.equity = equity
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.calls = 0

    def simulate(self, evidence: BacktestEvidence) -> DryRunEvidence:
        self.calls += 1
        fills: list[DryRunFill] = []
        for ticker in sorted(evidence.target_weights):
            weight = float(evidence.target_weights[ticker])
            raw_price = evidence.reference_prices.get(ticker)
            if raw_price is None or not math.isfinite(float(raw_price)) or float(raw_price) <= 0:
                fills.append(DryRunFill(ticker, "REJECTED", weight, 0, None, None, 0.0, 0.0, "INVALID_REFERENCE_PRICE"))
                continue
            price = float(raw_price)
            quantity = int(abs(weight) * self.equity / price)
            if quantity <= 0:
                fills.append(DryRunFill(ticker, "REJECTED", weight, 0, price, None, 0.0, 0.0, "BELOW_ONE_SHARE"))
                continue
            direction = 1.0 if weight >= 0 else -1.0
            fill_price = price * (1.0 + direction * self.slippage_bps / 10_000.0)
            notional = quantity * price
            fills.append(
                DryRunFill(
                    ticker,
                    "SIMULATED_FILL",
                    weight,
                    quantity,
                    price,
                    fill_price,
                    abs(fill_price - price) * quantity,
                    notional * self.fee_bps / 10_000.0,
                    None,
                )
            )
        config_hash = _hash_text(
            _canonical({"equity": self.equity, "slippage_bps": self.slippage_bps, "fee_bps": self.fee_bps})
        )
        return DryRunEvidence(evidence.candidate_id, tuple(fills), self.slippage_bps, self.fee_bps, config_hash)


@dataclass(frozen=True)
class CampaignReplay:
    campaign_id: str
    transitions: tuple[Mapping[str, Any], ...]
    artifact_hash: str


class BoundedMalaysiaResearchLoop:
    """Finite, restartable Malaysia research -> validation -> simulation controller."""

    def __init__(
        self,
        *,
        config: CampaignConfig,
        universe: MalaysiaUniverseContract,
        ledger: TrialLedger,
        hypothesis_registry: HypothesisAdmissionGate,
        candidate_generator: CandidateGenerator,
        specialist_runtime: SpecialistReviewRuntime,
        backtester: BacktestEngine,
        institutional_auditor: InstitutionalAuditor,
        dry_runner: DeterministicDryRunSimulator,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        universe.validate()
        self.config = config
        self.universe = universe
        self.ledger = ledger
        self.hypothesis_registry = hypothesis_registry
        self.candidate_generator = candidate_generator
        self.specialist_runtime = specialist_runtime
        self.backtester = backtester
        self.institutional_auditor = institutional_auditor
        self.dry_runner = dry_runner
        self.monotonic = monotonic
        self.run_dir = config.artifact_root / config.campaign_id
        self.state_path = self.run_dir / "campaign_state.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for directive in config.initial_queue:
            self.hypothesis_registry.assert_admitted(directive.hypothesis_id)

    def run(self, *, crash_after_stage: str | None = None) -> CampaignReplay:
        state = self._load_state()
        started = self.monotonic()
        while True:
            stop = self._budget_stop(state, started)
            if stop is not None:
                self._append_stop(state, stop)
                self._save_state(state)
                break
            if state["active"] is None:
                if not state["queue"]:
                    self._append_stop(state, CampaignStopCode.QUEUE_EMPTY)
                    self._save_state(state)
                    break
                directive = MutationDirective(**state["queue"][0])
                directive.validate()
                self.hypothesis_registry.assert_admitted(directive.hypothesis_id)
                reservation = self.ledger.reserve_trial(
                    idempotency_key=f"{self.config.campaign_id}:{directive.directive_id}",
                    campaign="malaysia",
                    family=directive.family,
                    hypothesis_id=directive.hypothesis_id,
                    candidate_id=f"PENDING-{directive.directive_id}",
                    parent_trial_id=directive.parent_trial_id,
                    code_hash=_hash_text("bounded_malaysia_loop_v1"),
                    data_hash=self.universe.content_hash(),
                    config_hash=_hash_text(_canonical(self._config_snapshot())),
                    prompt_hash=_hash_text(_canonical(asdict(directive))),
                    model_hash=_hash_text(self.config.model_routing_id),
                )
                if reservation.created:
                    self.ledger.record_research_step(
                        reservation.trial_id,
                        event_type="hypothesis_intake",
                        status="admitted",
                        payload={"hypothesis_id": directive.hypothesis_id},
                    )
                state["active"] = {"directive": asdict(directive), "trial_id": reservation.trial_id, "stage": "reserved"}
                state["usage"]["trials"] += 1
                state["usage"]["families"][directive.family] = state["usage"]["families"].get(directive.family, 0) + 1
                self._save_state(state)
                self._maybe_crash("reserved", crash_after_stage)

            active = state["active"]
            directive = MutationDirective(**active["directive"])
            trial_id = active["trial_id"]
            if active["stage"] == "reserved":
                generated = self.candidate_generator.generate(directive, trial_id=trial_id, config=self.config)
                self._validate_candidate(generated.candidate, directive)
                if generated.model_call is not None:
                    self.ledger.record_model_call(trial_id, generated.model_call)
                    state["usage"]["tokens"] += generated.model_call.input_tokens + generated.model_call.output_tokens
                state["usage"]["external_cost_usd"] += generated.external_cost_usd
                active["candidate"] = generated.candidate.to_dict()
                active["model_call"] = generated.model_call.to_dict() if generated.model_call is not None else None
                active["stage"] = "candidate_extracted"
                content = generated.candidate.canonical_json().encode("utf-8")
                self.ledger.record_research_step(
                    trial_id,
                    event_type="candidate_extracted",
                    status="success",
                    payload={"candidate_id": generated.candidate.candidate_id, "candidate_hash": generated.candidate.content_hash()},
                    evidence_content=content,
                )
                self._save_state(state)
                self._maybe_crash("candidate_extracted", crash_after_stage)

            candidate = CandidateSpec.from_dict(active["candidate"])
            if active["stage"] == "candidate_extracted":
                dossier = MalaysiaReviewDossier(
                    candidate_id=candidate.candidate_id,
                    hypothesis_hash=_hash_text(directive.hypothesis_id),
                    candidate_hash=candidate.content_hash(),
                    problem=f"Audit {candidate.family} for Malaysia U3 primary and U2 robustness under fixed costs.",
                    tags=(candidate.family, "selection_bias", "time_series", "transaction_cost"),
                    evidence_hashes={"universe": self.universe.content_hash(), "candidate": candidate.content_hash()},
                    research_context={
                        "features": list(self.config.features),
                        "base_cost_bps_one_way": self.config.base_cost_bps_one_way,
                        "stress_cost_bps_one_way": self.config.stress_cost_bps_one_way,
                        "u3_count": U3_COUNT,
                        "u2_count": U2_COUNT,
                        "universe_relation": "U3 is the 34-stock primary subset of the 62-stock U2 robustness universe",
                        "candidate_family": candidate.family,
                        "implementation_ref": candidate.implementation_ref,
                        "candidate_parameters": dict(candidate.parameters),
                        "signal_timing": "all price, volatility, and liquidity features are lagged before portfolio formation",
                        "validation_protocol": [
                            "unchanged candidate on U3 primary and U2 robustness",
                            "20 bps base and 40 bps stress costs",
                            "lifetime-denominator deflated Sharpe and PBO",
                            "causal regime, Malaysia factor-residual, and 126/252-day decay audits",
                        ],
                    },
                    parent_candidate_id=directive.parent_candidate_id,
                )
                disposition = self.specialist_runtime.review_candidate(dossier, trial_id=trial_id)
                active["review"] = self._disposition_snapshot(disposition)
                active["stage"] = "reviewed"
                state["usage"]["tokens"] += sum(
                    (item.model_call.input_tokens + item.model_call.output_tokens)
                    for item in disposition.attempts
                    if item.model_call is not None
                )
                state["usage"]["external_cost_usd"] += sum(
                    item.model_call.cost_usd
                    for item in disposition.attempts
                    if item.model_call is not None
                )
                self._save_state(state)
                self._maybe_crash("reviewed", crash_after_stage)

            review = active["review"]
            failures: list[str] = []
            backtest_allowed = bool(review.get("backtest_allowed", review["progression_allowed"]))
            if not backtest_allowed:
                failures.append(
                    ResearchFailureCode.SPECIALIST_COUNTEREXAMPLE_VETO.value
                    if review["vetoed"]
                    else ResearchFailureCode.SPECIALIST_REVIEW_FAILURE.value
                )
            if active["stage"] == "reviewed":
                if not failures:
                    evidence = self.backtester.evaluate(
                        candidate,
                        trial_id=trial_id,
                        universe=self.universe,
                        config=self.config,
                    )
                    evidence.validate(self.universe)
                    active["backtest"] = evidence.to_dict()
                    content = _canonical(evidence.to_dict()).encode("utf-8")
                    self.ledger.record_research_step(
                        trial_id,
                        event_type="backtest",
                        status="success",
                        payload={"candidate_id": candidate.candidate_id, "evidence_hash": hashlib.sha256(content).hexdigest()},
                        evidence_content=content,
                    )
                active["stage"] = "backtested"
                self._save_state(state)
                self._maybe_crash("backtested", crash_after_stage)

            if backtest_allowed and not review["progression_allowed"] and active["stage"] != "reviewed":
                failures.append(ResearchFailureCode.SPECIALIST_REVIEW_FAILURE.value)
            if "backtest" in active:
                evidence = BacktestEvidence(**active["backtest"])
                failures.extend(item.value for item in evidence.failure_codes())
            if active["stage"] == "backtested":
                if not failures and "backtest" in active:
                    audit = self.institutional_auditor.audit(
                        candidate,
                        BacktestEvidence(**active["backtest"]),
                        trial_id=trial_id,
                        universe=self.universe,
                        config=self.config,
                    )
                    audit.validate(candidate.candidate_id)
                    active["institutional_audit"] = audit.to_dict()
                    content = _canonical(audit.to_dict()).encode("utf-8")
                    self.ledger.record_research_step(
                        trial_id,
                        event_type="institutional_audit",
                        status="success" if not audit.failure_codes() else "rejected",
                        payload={
                            "candidate_id": candidate.candidate_id,
                            "audit_hash": audit.audit_hash,
                            "failure_codes": [code.value for code in audit.failure_codes()],
                        },
                        evidence_content=content,
                    )
                active["stage"] = "audited"
                self._save_state(state)
                self._maybe_crash("audited", crash_after_stage)

            if "institutional_audit" in active:
                audit = InstitutionalAuditEvidence(**active["institutional_audit"])
                failures.extend(item.value for item in audit.failure_codes())
            if active["stage"] == "audited":
                if not failures and "backtest" in active and "institutional_audit" in active:
                    dry = self.dry_runner.simulate(BacktestEvidence(**active["backtest"]))
                    active["dry_run"] = dry.to_dict()
                    content = _canonical(dry.to_dict()).encode("utf-8")
                    self.ledger.record_research_step(
                        trial_id,
                        event_type="dry_run",
                        status="simulated",
                        payload={"candidate_id": candidate.candidate_id, "configuration_hash": dry.configuration_hash},
                        evidence_content=content,
                    )
                active["failure_codes"] = list(dict.fromkeys(failures))
                active["stage"] = "evaluated"
                self._save_state(state)
                self._maybe_crash("evaluated", crash_after_stage)

            if active["stage"] == "evaluated":
                failed = bool(active["failure_codes"])
                outcome_evidence = None
                if "backtest" in active:
                    raw = _canonical(active["backtest"]).encode("utf-8")
                    outcome_evidence = self.ledger.store_evidence(raw, extension=".json").sha256
                self.ledger.record_outcome(
                    trial_id,
                    status="rejected" if failed else "success",
                    failure_class="RESEARCH_GATE" if failed else None,
                    payload={"mutation_codes": active["failure_codes"], "dry_run_completed": "dry_run" in active},
                    evidence_sha256=outcome_evidence,
                )
                state["consecutive_failures"] = state["consecutive_failures"] + 1 if failed else 0
                trace = {
                    "iteration": len(state["trace"]) + 1,
                    "directive_id": directive.directive_id,
                    "generation": directive.generation,
                    "trial_id": trial_id,
                    "parent_trial_id": directive.parent_trial_id,
                    "candidate_id": candidate.candidate_id,
                    "parent_candidate_id": directive.parent_candidate_id,
                    "mutation_codes_consumed": list(directive.mutation_codes),
                    "failure_codes_produced": list(active["failure_codes"]),
                    "candidate_hash": candidate.content_hash(),
                    "institutional_audit_completed": "institutional_audit" in active,
                    "dry_run_completed": "dry_run" in active,
                }
                state["trace"].append(trace)
                if failed and directive.generation < self.config.budgets.max_generations:
                    child = self._child_directive(directive, trial_id, candidate, tuple(active["failure_codes"]))
                    child.validate()
                    state["queue"].append(asdict(child))
                    self.ledger.record_research_step(
                        trial_id,
                        event_type="mutation_created",
                        status="queued",
                        payload={
                            "directive_id": child.directive_id,
                            "generation": child.generation,
                            "mutation_codes": list(child.mutation_codes),
                        },
                    )
                state["queue"].pop(0)
                state["active"] = None
                self._save_state(state)
                self._maybe_crash("iteration_complete", crash_after_stage)
        return self.replay()

    def replay(self) -> CampaignReplay:
        state = self._load_state()
        stable = _canonical(state)
        return CampaignReplay(self.config.campaign_id, tuple(state["trace"]), _hash_text(stable))

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("config_hash") != _hash_text(_canonical(self._config_snapshot())):
                raise RuntimeError("campaign resume configuration drift")
            return state
        return {
            "schema_version": 1,
            "campaign_id": self.config.campaign_id,
            "config_hash": _hash_text(_canonical(self._config_snapshot())),
            "universe_hash": self.universe.content_hash(),
            "queue": [asdict(item) for item in self.config.initial_queue],
            "active": None,
            "usage": {"trials": 0, "tokens": 0, "external_cost_usd": 0.0, "families": {}},
            "consecutive_failures": 0,
            "stop_reasons": [],
            "trace": [],
        }

    def _save_state(self, state: Mapping[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(_canonical(state) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "campaign_id": self.config.campaign_id,
            "budgets": asdict(self.config.budgets),
            "base_cost_bps_one_way": self.config.base_cost_bps_one_way,
            "stress_cost_bps_one_way": self.config.stress_cost_bps_one_way,
            "features": self.config.features,
            "workflow_name": self.config.workflow_name,
            "model_routing_id": self.config.model_routing_id,
            "initial_queue": [asdict(item) for item in self.config.initial_queue],
        }

    def _budget_stop(self, state: Mapping[str, Any], started: float) -> CampaignStopCode | None:
        usage = state["usage"]
        budgets = self.config.budgets
        if usage["trials"] >= budgets.max_trials and state["active"] is None:
            return CampaignStopCode.MAX_TRIALS
        if usage["tokens"] + budgets.token_reserve_per_trial > budgets.max_tokens and state["active"] is None:
            return CampaignStopCode.TOKEN_BUDGET
        if usage["external_cost_usd"] > budgets.max_external_cost_usd:
            return CampaignStopCode.EXTERNAL_COST_BUDGET
        if self.monotonic() - started >= budgets.max_wall_seconds:
            return CampaignStopCode.WALL_TIME_BUDGET
        if state["consecutive_failures"] >= budgets.max_consecutive_failures:
            return CampaignStopCode.CONSECUTIVE_FAILURES
        if state["active"] is None and state["queue"]:
            family = state["queue"][0]["family"]
            if usage["families"].get(family, 0) >= budgets.max_trials_per_family:
                return CampaignStopCode.FAMILY_TRIAL_BUDGET
        return None

    @staticmethod
    def _append_stop(state: dict[str, Any], code: CampaignStopCode) -> None:
        if code.value not in state["stop_reasons"]:
            state["stop_reasons"].append(code.value)

    @staticmethod
    def _maybe_crash(stage: str, requested: str | None) -> None:
        if stage == requested:
            raise InjectedLoopCrash(stage)

    @staticmethod
    def _disposition_snapshot(value: PanelDisposition) -> dict[str, Any]:
        return {
            "selected_specialists": list(value.selected_specialists),
            "backtest_allowed": value.backtest_allowed,
            "progression_allowed": value.progression_allowed,
            "vetoed": value.vetoed,
            "reasons": list(value.reasons),
            "conflicts": list(value.conflicts),
            "disposition_hash": value.disposition_hash,
        }

    @staticmethod
    def _validate_candidate(candidate: CandidateSpec, directive: MutationDirective) -> None:
        candidate.validate()
        if candidate.hypothesis_id != directive.hypothesis_id:
            raise ValueError("candidate hypothesis does not match reserved directive")
        if candidate.parent_candidate_id != directive.parent_candidate_id:
            raise ValueError("candidate parent lineage does not match mutation directive")
        validate_strategy_parameters(
            candidate.family,
            candidate.parameters,
            generation=directive.generation,
            mutation_codes=directive.mutation_codes,
        )
        text = candidate.canonical_json().lower()
        if "japan" in text or "jpx" in text or "openrouter" in text or "broker" in text:
            raise ValueError("candidate escaped Malaysia research scope")

    @staticmethod
    def _child_directive(
        parent: MutationDirective,
        trial_id: str,
        candidate: CandidateSpec,
        mutation_codes: tuple[str, ...],
    ) -> MutationDirective:
        identity = _hash_text(_canonical({
            "parent": parent.directive_id,
            "trial": trial_id,
            "candidate": candidate.content_hash(),
            "mutation_codes": mutation_codes,
        }))[:20]
        return MutationDirective(
            directive_id=f"MYS-MUT-{identity}",
            hypothesis_id=parent.hypothesis_id,
            family=parent.family,
            variant=parent.variant,
            generation=parent.generation + 1,
            parent_trial_id=trial_id,
            parent_candidate_id=candidate.candidate_id,
            mutation_codes=mutation_codes,
        )
