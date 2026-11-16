from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.metrics import deflated_sharpe_ratio, sharpe_ratio
from quant_lab.validation.overfit import probability_of_backtest_overfitting


MALAYSIA_CONTAMINATED_THROUGH = pd.Timestamp("2026-07-06")


class ResearchFailureCode(str, Enum):
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    INVALID_PURGE_EMBARGO = "INVALID_PURGE_EMBARGO"
    POINT_IN_TIME_VIOLATION = "POINT_IN_TIME_VIOLATION"
    STALE_DATA = "STALE_DATA"
    LEDGER_UNAVAILABLE = "LEDGER_UNAVAILABLE"
    INSUFFICIENT_CANDIDATES = "INSUFFICIENT_CANDIDATES"
    PBO_EXCESSIVE = "PBO_EXCESSIVE"
    DEFLATED_SHARPE_FAILURE = "DEFLATED_SHARPE_FAILURE"
    NON_POSITIVE_NET_IR = "NON_POSITIVE_NET_IR"
    SPECIALIST_REVIEW_FAILURE = "SPECIALIST_REVIEW_FAILURE"
    SPECIALIST_COUNTEREXAMPLE_VETO = "SPECIALIST_COUNTEREXAMPLE_VETO"
    DRAWDOWN_EXCESSIVE = "DRAWDOWN_EXCESSIVE"
    TURNOVER_EXCESSIVE = "TURNOVER_EXCESSIVE"
    COST_STRESS_FAILURE = "COST_STRESS_FAILURE"
    CONCENTRATION_EXCESSIVE = "CONCENTRATION_EXCESSIVE"
    U2_ROBUSTNESS_FAILURE = "U2_ROBUSTNESS_FAILURE"
    REGIME_INSTABILITY = "REGIME_INSTABILITY"
    FACTOR_RESIDUAL_FAILURE = "FACTOR_RESIDUAL_FAILURE"
    ALPHA_DECAY_FAILURE = "ALPHA_DECAY_FAILURE"


@dataclass(frozen=True)
class ResearchDataset:
    features: pd.DataFrame
    candidate_returns: pd.DataFrame
    research_end: pd.Timestamp
    content_hash: str
    pit_verified: bool = False
    historical_label: str = "research_or_contaminated_audit_only"


@dataclass(frozen=True)
class ResearchGateResult:
    passed: bool
    failure_codes: tuple[ResearchFailureCode, ...]
    net_oos_ir: float
    deflated_sharpe_probability: float
    pbo: float
    lifetime_attempts: int

    def mutation_feedback(self) -> tuple[str, ...]:
        """Return blinded codes only; numerical scores never enter mutation prompts."""

        return tuple(code.value for code in self.failure_codes)


class ResearchDataFirewall:
    """Own the research span so optimization never receives terminal evidence."""

    def __init__(
        self,
        *,
        contaminated_through: str | pd.Timestamp = MALAYSIA_CONTAMINATED_THROUGH,
        min_observations: int = 504,
        max_staleness_days: int = 10,
    ) -> None:
        self.contaminated_through = pd.Timestamp(contaminated_through)
        self.min_observations = min_observations
        self.max_staleness_days = max_staleness_days

    def seal_research_inputs(
        self,
        *,
        features: pd.DataFrame,
        candidate_returns: pd.DataFrame,
        available_at: pd.DataFrame | None = None,
        universe_available_at: pd.DataFrame | None = None,
        research_end: str | pd.Timestamp | None = None,
        purge: int = 5,
        embargo: int = 5,
    ) -> ResearchDataset:
        end = pd.Timestamp(research_end) if research_end is not None else self.contaminated_through
        if purge < 0 or embargo < 0 or purge + embargo >= self.min_observations:
            raise ValueError(ResearchFailureCode.INVALID_PURGE_EMBARGO.value)
        if not isinstance(features.index, pd.DatetimeIndex) or not isinstance(candidate_returns.index, pd.DatetimeIndex):
            raise TypeError("research inputs require DatetimeIndex")
        if not features.index.is_monotonic_increasing or not candidate_returns.index.is_monotonic_increasing:
            raise ValueError("research inputs must be sorted")
        common = features.index.intersection(candidate_returns.index)
        common = common[common <= end]
        if len(common) < self.min_observations:
            raise ValueError(ResearchFailureCode.INSUFFICIENT_SAMPLE.value)
        research_features = features.loc[common].copy(deep=True)
        research_returns = candidate_returns.loc[common].copy(deep=True)
        if not np.isfinite(research_returns.to_numpy(dtype=float)).all():
            raise ValueError("research returns must be finite")
        if available_at is not None:
            aligned = available_at.reindex(index=common, columns=research_features.columns)
            for column in research_features.columns:
                used = research_features[column].notna()
                timestamps = pd.to_datetime(aligned.loc[used, column], errors="coerce")
                decisions = pd.Series(common[used], index=timestamps.index)
                if timestamps.isna().any() or (timestamps > decisions).any():
                    raise ValueError(ResearchFailureCode.POINT_IN_TIME_VIOLATION.value)
        if (end - common.max()).days > self.max_staleness_days:
            raise ValueError(ResearchFailureCode.STALE_DATA.value)
        if available_at is None or universe_available_at is None:
            raise ValueError(ResearchFailureCode.POINT_IN_TIME_VIOLATION.value)
        membership = universe_available_at.reindex(index=common)
        if membership.shape[1] == 0:
            raise ValueError(ResearchFailureCode.POINT_IN_TIME_VIOLATION.value)
        for column in membership:
            times = pd.to_datetime(membership[column], errors='coerce')
            if times.isna().any() or (times > common).any():
                raise ValueError(ResearchFailureCode.POINT_IN_TIME_VIOLATION.value)
        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(research_features, index=True).values.tobytes())
        digest.update(pd.util.hash_pandas_object(research_returns, index=True).values.tobytes())
        digest.update(pd.util.hash_pandas_object(aligned, index=True).values.tobytes())
        digest.update(pd.util.hash_pandas_object(membership, index=True).values.tobytes())
        digest.update(end.isoformat().encode("utf-8"))
        return ResearchDataset(research_features, research_returns, end, digest.hexdigest(), pit_verified=True)

    @staticmethod
    def rank_candidates(dataset: ResearchDataset) -> tuple[str, ...]:
        """Deterministic research-only ranking used for invariance tests and routing."""

        scores = dataset.candidate_returns.mean().sort_values(ascending=False, kind="mergesort")
        return tuple(str(item) for item in scores.index)


def deflated_sharpe_with_lifetime(
    returns: pd.Series,
    *,
    observed_trial_sharpes: Sequence[float],
    lifetime_attempts: int,
    periods_per_year: int = 252,
) -> float:
    if lifetime_attempts < len(observed_trial_sharpes) or lifetime_attempts < 1:
        raise ValueError("lifetime_attempts must cover every observed trial")
    return deflated_sharpe_ratio(returns, observed_trial_sharpes, periods_per_year=periods_per_year, n_trials=lifetime_attempts)


def evaluate_research_gate(
    dataset: ResearchDataset,
    *,
    candidate_id: str,
    ledger: TrialLedger,
    observed_trial_sharpes: Sequence[float],
    benchmark_returns: pd.Series | None = None,
    benchmark_name: str | None = None,
    snapshot=None,
    max_pbo: float = 0.20,
    min_deflated_sharpe: float = 0.95,
) -> ResearchGateResult:
    if not np.isfinite([max_pbo,min_deflated_sharpe]).all() or not 0 <= max_pbo <= 1 or not 0 <= min_deflated_sharpe <= 1:
        raise ValueError("invalid research gate thresholds")
    failures: list[ResearchFailureCode] = []
    if candidate_id not in dataset.candidate_returns:
        raise KeyError(f"candidate not present in research returns: {candidate_id}")
    try:
        current = ledger.snapshot()
        lifetime_attempts = current.global_count
        if snapshot is None or (snapshot.global_count, snapshot.family_counts, snapshot.dispersion_receipt, snapshot.ledger_hash) != (current.global_count, current.family_counts, current.dispersion_receipt, current.ledger_hash) or tuple(observed_trial_sharpes) != snapshot.sharpes:
            failures.append(ResearchFailureCode.LEDGER_UNAVAILABLE)
    except Exception:
        failures.append(ResearchFailureCode.LEDGER_UNAVAILABLE)
        lifetime_attempts = 0
    returns = dataset.candidate_returns[candidate_id]
    if dataset.pit_verified is not True:
        failures.append(ResearchFailureCode.POINT_IN_TIME_VIOLATION)
    if benchmark_name == 'CASH' and benchmark_returns is None:
        benchmark = 0.
    elif benchmark_name and benchmark_name != 'CASH' and benchmark_returns is not None and benchmark_returns.index.is_unique:
        benchmark = benchmark_returns.reindex(returns.index)
        if not np.isfinite(benchmark.to_numpy(dtype=float)).all():
            failures.append(ResearchFailureCode.NON_POSITIVE_NET_IR)
    else:
        benchmark = float('nan')
        failures.append(ResearchFailureCode.NON_POSITIVE_NET_IR)
    active = returns - benchmark
    net_ir = sharpe_ratio(active)
    if dataset.candidate_returns.shape[1] < 2:
        failures.append(ResearchFailureCode.INSUFFICIENT_CANDIDATES)
        pbo = 1.0
    else:
        try:
            pbo = probability_of_backtest_overfitting(dataset.candidate_returns, n_splits=8, strict=True)
        except ValueError:
            failures.append(ResearchFailureCode.INSUFFICIENT_SAMPLE)
            pbo = 1.0
    if lifetime_attempts and ResearchFailureCode.LEDGER_UNAVAILABLE not in failures:
        try:
            dsr = deflated_sharpe_with_lifetime(
                returns, observed_trial_sharpes=observed_trial_sharpes,
                lifetime_attempts=lifetime_attempts)
        except (ValueError, TypeError):
            failures.append(ResearchFailureCode.DEFLATED_SHARPE_FAILURE)
            dsr = 0.0
    else:
        dsr = 0.0
    if pbo > max_pbo:
        failures.append(ResearchFailureCode.PBO_EXCESSIVE)
    if dsr < min_deflated_sharpe:
        failures.append(ResearchFailureCode.DEFLATED_SHARPE_FAILURE)
    if not np.isfinite(net_ir) or net_ir <= 0.0:
        failures.append(ResearchFailureCode.NON_POSITIVE_NET_IR)
    unique = tuple(dict.fromkeys(failures))
    return ResearchGateResult(not unique, unique, float(net_ir), float(dsr), float(pbo), lifetime_attempts)
