from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_lab.research.bounded_loop import (
    BASE_COST_BPS_ONE_WAY,
    STRESS_COST_BPS_ONE_WAY,
    MalaysiaUniverseContract,
)
from quant_lab.research.institutional_workflow import MalaysiaAuditInputs
from quant_lab.research.malaysia_pilot import MalaysiaHistoricalBacktester
from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort
from quant_lab.research.v2_overlay_protocol import (
    V2_DRAFT02_PROTOCOL_ID,
    draft02_exposure_scalar,
    validate_v2_overlay_protocol,
    v2_overlay_protocol_hash,
)


@dataclass(frozen=True)
class V2SelectionDiagnostics:
    rebalance_count: int
    signal_warmup_rebalance_count: int
    evaluated_rebalance_count: int
    insufficient_diversification_dates: tuple[str, ...]
    minimum_valid_score_count: int
    minimum_selected_count: int
    maximum_selected_count: int


@dataclass
class V2ArmComputation:
    arm_id: str
    experiment_id: str
    strategy_id: str
    family: str
    arm: str
    u3_gross_returns: pd.Series
    u3_base_returns: pd.Series
    u3_stress_returns: pd.Series
    u3_weights: pd.DataFrame
    u3_turnover: pd.Series
    u3_exposure: pd.Series
    u2_base_returns: pd.Series
    u2_stress_returns: pd.Series
    selection_diagnostics: V2SelectionDiagnostics | None

    def validate(self, universe: MalaysiaUniverseContract) -> None:
        if self.arm not in {"control", "challenger"}:
            raise ValueError("V2 arm type is invalid")
        if tuple(self.u3_weights.columns) != tuple(universe.u3):
            raise ValueError("V2 U3 weight columns drifted")
        aligned = (
            self.u3_gross_returns.index,
            self.u3_base_returns.index,
            self.u3_stress_returns.index,
            self.u3_weights.index,
            self.u3_turnover.index,
            self.u3_exposure.index,
            self.u2_base_returns.index,
            self.u2_stress_returns.index,
        )
        if any(not aligned[0].equals(index) for index in aligned[1:]):
            raise ValueError("V2 arm calendars are not identical")
        numeric = pd.concat(
            [
                self.u3_gross_returns,
                self.u3_base_returns,
                self.u3_stress_returns,
                self.u3_turnover,
                self.u3_exposure,
                self.u2_base_returns,
                self.u2_stress_returns,
            ],
            axis=1,
        )
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("V2 arm series must be finite")
        if not np.isfinite(self.u3_weights.to_numpy(dtype=float)).all():
            raise ValueError("V2 arm weights must be finite")
        if float(self.u3_weights.abs().sum(axis=1).max()) > 1.0 + 1e-12:
            raise ValueError("V2 arm exceeded maximum gross exposure")
        if (self.u3_exposure < -1e-12).any() or (self.u3_exposure > 1.0 + 1e-12).any():
            raise ValueError("V2 exposure scalar escaped zero-to-one bounds")


@dataclass(frozen=True)
class PairedBootstrapReceipt:
    block_length_sessions: int
    replications: int
    random_seed: int
    successful_margin_draws: int
    finite_margin_draws: int
    nonfinite_draws_counted_as_failures: int
    noninferiority_probability: float


class MalaysiaV2PairedEngine:
    """Exact six-arm computation engine for the frozen Malaysia V2 Draft 02."""

    def __init__(
        self,
        *,
        daily_features_path: str | Path,
        universe: MalaysiaUniverseContract,
        protocol: Mapping[str, Any],
        research_end: str = "2026-07-06",
    ) -> None:
        validate_v2_overlay_protocol(protocol)
        if protocol["protocol_id"] != V2_DRAFT02_PROTOCOL_ID:
            raise ValueError("V2 paired engine requires the exact Draft 02 protocol")
        self.daily_features_path = Path(daily_features_path)
        self.universe = universe
        self.protocol = dict(protocol)
        self.protocol_hash = v2_overlay_protocol_hash(protocol)
        self.research_end = pd.Timestamp(research_end)
        self._loaded: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    def load_matrices(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._loaded is not None:
            return tuple(frame.copy() for frame in self._loaded)  # type: ignore[return-value]
        columns = [
            "date",
            "ticker",
            "country",
            "adj_close",
            "simple_return",
            "adv_60_local_lag1",
            "vol_63_lag1",
        ]
        raw = pd.read_parquet(self.daily_features_path, columns=columns)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.loc[
            raw["country"].eq("Malaysia")
            & raw["ticker"].isin(self.universe.u2)
            & raw["date"].le(self.research_end)
        ].sort_values(["date", "ticker"])
        if raw.empty or raw["date"].nunique() < 504:
            raise ValueError("insufficient Malaysia V2 research history")
        price = raw.pivot(index="date", columns="ticker", values="adj_close").reindex(
            columns=self.universe.u2
        )
        returns = (
            raw.pivot(index="date", columns="ticker", values="simple_return")
            .reindex(columns=self.universe.u2)
            .fillna(0.0)
        )
        adv = raw.pivot(index="date", columns="ticker", values="adv_60_local_lag1").reindex(
            columns=self.universe.u2
        )
        volatility = raw.pivot(index="date", columns="ticker", values="vol_63_lag1").reindex(
            columns=self.universe.u2
        )
        if not price.index.equals(returns.index):
            raise ValueError("Malaysia V2 price and return calendars differ")
        self._loaded = (price, returns, adv, volatility)
        return tuple(frame.copy() for frame in self._loaded)  # type: ignore[return-value]

    def compute_all_arms(self) -> tuple[V2ArmComputation, ...]:
        return (*self.compute_arms("control"), *self.compute_arms("challenger"))

    def compute_arms(self, arm_type: str) -> tuple[V2ArmComputation, ...]:
        if arm_type not in {"control", "challenger"}:
            raise ValueError("V2 engine arm type must be control or challenger")
        price, returns, adv, volatility = self.load_matrices()
        registry = {item.strategy_id: item for item in malaysia_atomic_cohort()}
        scores: dict[str, pd.DataFrame] = {}
        for parent in self.protocol["parent_strategies"]:
            strategy = registry[parent["strategy_id"]]
            scores[strategy.strategy_id] = MalaysiaHistoricalBacktester._score(
                strategy.family,
                price,
                adv,
                volatility,
                dict(strategy.parameters),
            )
        results: list[V2ArmComputation] = []
        for arm in self.protocol["arms"]:
            if arm["arm"] != arm_type:
                continue
            strategy = registry[arm["strategy_id"]]
            score = scores[strategy.strategy_id]
            if arm["arm"] == "control":
                result = self._control_arm(arm, strategy.family, score, returns)
            else:
                result = self._challenger_arm(arm, strategy.family, score, returns)
            result.validate(self.universe)
            results.append(result)
        return tuple(results)

    def audit_inputs(self, arm: V2ArmComputation) -> MalaysiaAuditInputs:
        price, returns, adv, volatility = self.load_matrices()
        u3_returns = returns.reindex(columns=self.universe.u3)
        u3_prices = price.reindex(columns=self.universe.u3)
        u3_adv = adv.reindex(columns=self.universe.u3)
        u3_volatility = volatility.reindex(columns=self.universe.u3)
        market_returns = u3_returns.mean(axis=1).fillna(0.0)
        factors = pd.DataFrame(
            {
                "market": market_returns,
                "momentum": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    u3_prices.pct_change(126, fill_method=None).shift(1), u3_returns
                ),
                "low_volatility": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    -u3_volatility.shift(1), u3_returns
                ),
                "lagged_liquidity": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    np.log1p(u3_adv.shift(1)), u3_returns
                ),
            }
        ).fillna(0.0)
        return MalaysiaAuditInputs(
            strategy_returns=arm.u3_base_returns.copy(),
            market_returns=market_returns,
            factor_returns=factors,
            regime_labels=MalaysiaHistoricalBacktester._causal_regime_labels(market_returns),
        )

    def _control_arm(
        self,
        arm: Mapping[str, Any],
        family: str,
        score: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> V2ArmComputation:
        u3_score = score.reindex(columns=self.universe.u3)
        u3_returns = returns.reindex(columns=self.universe.u3)
        u3_base, u3_stress, u3_weights, turnover = MalaysiaHistoricalBacktester._portfolio_returns(
            u3_score, u3_returns, "ME", 0.20
        )
        u2_base, u2_stress, _, _ = MalaysiaHistoricalBacktester._portfolio_returns(
            score, returns, "ME", 0.20
        )
        gross = (u3_weights * u3_returns.reindex_like(u3_weights).fillna(0.0)).sum(axis=1)
        exposure = u3_weights.abs().sum(axis=1)
        return V2ArmComputation(
            arm_id=str(arm["arm_id"]),
            experiment_id=str(arm["experiment_id"]),
            strategy_id=str(arm["strategy_id"]),
            family=family,
            arm="control",
            u3_gross_returns=gross,
            u3_base_returns=u3_base,
            u3_stress_returns=u3_stress,
            u3_weights=u3_weights,
            u3_turnover=turnover,
            u3_exposure=exposure,
            u2_base_returns=u2_base,
            u2_stress_returns=u2_stress,
            selection_diagnostics=None,
        )

    def _challenger_arm(
        self,
        arm: Mapping[str, Any],
        family: str,
        score: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> V2ArmComputation:
        u3 = self._challenger_portfolio(
            score.reindex(columns=self.universe.u3),
            returns.reindex(columns=self.universe.u3),
        )
        u2 = self._challenger_portfolio(score, returns)
        return V2ArmComputation(
            arm_id=str(arm["arm_id"]),
            experiment_id=str(arm["experiment_id"]),
            strategy_id=str(arm["strategy_id"]),
            family=family,
            arm="challenger",
            u3_gross_returns=u3[0],
            u3_base_returns=u3[1],
            u3_stress_returns=u3[2],
            u3_weights=u3[3],
            u3_turnover=u3[4],
            u3_exposure=u3[5],
            u2_base_returns=u2[1],
            u2_stress_returns=u2[2],
            selection_diagnostics=u3[6],
        )

    def _challenger_portfolio(
        self,
        score: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        V2SelectionDiagnostics,
    ]:
        sampled_scores = score.resample("ME").last()
        unit_sampled, selected_counts, diagnostics = hysteresis_unit_weights(
            sampled_scores,
            protocol_id=str(self.protocol["protocol_id"]),
        )
        unit_weights = _expand_month_end_values(unit_sampled, score.index).fillna(0.0)
        aligned_returns = returns.reindex_like(unit_weights).fillna(0.0)
        unit_gross = (unit_weights * aligned_returns).sum(axis=1)
        unit_available = unit_weights.abs().sum(axis=1).gt(0.0)
        risk_input = unit_gross.where(unit_available)
        rolling_count = risk_input.rolling(63, min_periods=1).count().shift(1)
        annualized_volatility = risk_input.rolling(63, min_periods=42).std(ddof=1).shift(1) * math.sqrt(252)
        sampled_volatility = annualized_volatility.resample("ME").last()
        sampled_count = rolling_count.resample("ME").last().fillna(0).astype(int)
        sampled_exposure = pd.Series(0.0, index=unit_sampled.index, dtype=float)
        for date in sampled_exposure.index:
            selected_count = int(selected_counts.get(date, 0))
            if selected_count < 4:
                sampled_exposure.loc[date] = 0.0
                continue
            sampled_exposure.loc[date] = draft02_exposure_scalar(
                float(sampled_volatility.get(date, float("nan"))),
                int(sampled_count.get(date, 0)),
            )
        exposure = _expand_month_end_values(sampled_exposure, score.index).fillna(0.0)
        weights = unit_weights.mul(exposure, axis=0)
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        gross = (weights * aligned_returns).sum(axis=1)
        base = gross - turnover * BASE_COST_BPS_ONE_WAY / 10_000.0
        stress = gross - turnover * STRESS_COST_BPS_ONE_WAY / 10_000.0
        return gross, base, stress, weights, turnover, exposure, diagnostics


def hysteresis_unit_weights(
    sampled_scores: pd.DataFrame,
    *,
    protocol_id: str,
) -> tuple[pd.DataFrame, pd.Series, V2SelectionDiagnostics]:
    """Apply exact Draft 02 rank hysteresis at already-sampled month ends."""

    if protocol_id != V2_DRAFT02_PROTOCOL_ID:
        raise ValueError("hysteresis tie ordering requires the Draft 02 protocol identity")
    output = pd.DataFrame(0.0, index=sampled_scores.index, columns=sampled_scores.columns)
    selected_counts = pd.Series(0, index=sampled_scores.index, dtype=int)
    previous: set[str] = set()
    warmup = 0
    valid_counts: list[int] = []
    evaluated_selected_counts: list[int] = []
    insufficient: list[str] = []

    def order(names: set[str] | list[str], row: pd.Series) -> list[str]:
        return sorted(
            names,
            key=lambda ticker: (
                -float(row[ticker]),
                hashlib.sha256(f"{protocol_id}::{ticker}".encode("utf-8")).hexdigest(),
            ),
        )

    for date, raw_row in sampled_scores.iterrows():
        row = raw_row.replace([np.inf, -np.inf], np.nan).dropna()
        valid_count = len(row)
        if valid_count == 0:
            warmup += 1
            previous = set()
            continue
        valid_counts.append(valid_count)
        ordered_all = order(list(row.index), row)
        ranks = pd.Series(
            {
                ticker: (valid_count - position) / valid_count
                for position, ticker in enumerate(ordered_all)
            }
        )
        capacity = int(math.ceil(0.20 * valid_count))
        retained = {ticker for ticker in previous if ticker in ranks and ranks[ticker] >= 0.60}
        chosen = order(retained, row)[:capacity]
        entrants = [
            ticker
            for ticker in row.index
            if ticker not in previous and ranks[ticker] >= 0.80
        ]
        if len(chosen) < capacity:
            chosen.extend(order(entrants, row)[: capacity - len(chosen)])
        previous = set(chosen)
        selected_counts.loc[date] = len(chosen)
        evaluated_selected_counts.append(len(chosen))
        if len(chosen) < 4:
            insufficient.append(pd.Timestamp(date).isoformat())
        elif chosen:
            output.loc[date, chosen] = 1.0 / len(chosen)
    diagnostics = V2SelectionDiagnostics(
        rebalance_count=len(sampled_scores),
        signal_warmup_rebalance_count=warmup,
        evaluated_rebalance_count=len(valid_counts),
        insufficient_diversification_dates=tuple(insufficient),
        minimum_valid_score_count=min(valid_counts) if valid_counts else 0,
        minimum_selected_count=min(evaluated_selected_counts) if evaluated_selected_counts else 0,
        maximum_selected_count=max(evaluated_selected_counts) if evaluated_selected_counts else 0,
    )
    return output, selected_counts, diagnostics


def paired_moving_block_bootstrap(
    control: pd.Series,
    challenger: pd.Series,
    *,
    block_length_sessions: int = 63,
    replications: int = 10_000,
    random_seed: int = 20260713,
) -> PairedBootstrapReceipt:
    if block_length_sessions != 63 or replications != 10_000 or random_seed != 20260713:
        raise ValueError("V2 paired bootstrap parameters are frozen")
    aligned = pd.concat(
        [control.rename("control"), challenger.rename("challenger")], axis=1, join="inner"
    )
    values = aligned.to_numpy(dtype=float)
    if len(values) < block_length_sessions or not np.isfinite(values).all():
        raise ValueError("V2 paired bootstrap requires finite aligned returns and one full block")
    n = len(values)
    blocks_per_draw = int(math.ceil(n / block_length_sessions))
    offsets = np.arange(block_length_sessions)
    maximum_start = n - block_length_sessions + 1
    rng = np.random.default_rng(random_seed)
    successful = 0
    finite = 0
    batch_size = 100
    for start in range(0, replications, batch_size):
        size = min(batch_size, replications - start)
        starts = rng.integers(0, maximum_start, size=(size, blocks_per_draw))
        indices = (starts[:, :, None] + offsets[None, None, :]).reshape(size, -1)[:, :n]
        sampled = values[indices]
        means = sampled.mean(axis=1)
        standard_deviations = sampled.std(axis=1, ddof=1)
        control_ir = means[:, 0] / standard_deviations[:, 0] * math.sqrt(252)
        challenger_ir = means[:, 1] / standard_deviations[:, 1] * math.sqrt(252)
        margins = challenger_ir - 0.80 * control_ir
        valid = np.isfinite(margins)
        finite += int(valid.sum())
        successful += int((valid & (margins >= 0.0)).sum())
    return PairedBootstrapReceipt(
        block_length_sessions=block_length_sessions,
        replications=replications,
        random_seed=random_seed,
        successful_margin_draws=successful,
        finite_margin_draws=finite,
        nonfinite_draws_counted_as_failures=replications - finite,
        noninferiority_probability=successful / replications,
    )


def return_series_hash(series: pd.Series) -> str:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("V2 return-series hash requires a DatetimeIndex")
    values = series.astype(float).sort_index()
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("V2 return-series hash requires finite values")
    payload = [[timestamp.isoformat(), float(value)] for timestamp, value in values.items()]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dataframe_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update("|".join(str(item) for item in frame.columns).encode("utf-8"))
    return digest.hexdigest()


def _expand_month_end_values(sampled: pd.DataFrame | pd.Series, index: pd.DatetimeIndex):
    expanded_index = index.union(sampled.index)
    return sampled.reindex(expanded_index).sort_index().ffill().reindex(index)
