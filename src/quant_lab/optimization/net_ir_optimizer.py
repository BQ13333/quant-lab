from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant_lab.models.base import AlphaModel
from quant_lab.portfolio.costs import (
    bps_cost_grid,
    country_cost_schedule,
    linear_transaction_cost,
    total_cost,
)
from quant_lab.portfolio.rebalance import monthly_rebalance, weekly_rebalance
from quant_lab.portfolio.sizing import (
    scores_to_group_neutral_long_only_capped_weights,
    scores_to_group_neutral_long_only_weights,
    scores_to_group_neutral_weights,
    scores_to_long_only_capped_weights,
    scores_to_long_only_weights,
    scores_to_weights,
    volatility_target,
)
from quant_lab.validation.gates import GateDecision, ValidationGate
from quant_lab.validation.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)
from quant_lab.validation.splits import walk_forward_splits
from quant_lab.validation.stress import cost_stress


BASE_COUNTRY_COST_BPS = 15.0


@dataclass(frozen=True)
class ConstraintResult:
    """Result of a single constraint evaluation."""
    name: str
    passed: bool
    value: float
    limit: float
    violation: float


@dataclass(frozen=True)
class OptimizedCandidateResult:
    """Full result for a candidate model + allocator config."""
    model_name: str
    allocator_name: str
    cost_bps: float
    net_ir: float
    net_sharpe: float
    gross_sharpe: float
    deflated_sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    turnover: float
    annual_excess_return: float
    tracking_error: float
    information_ratio: float
    constraints: tuple[ConstraintResult, ...]
    net_returns: pd.Series = field(repr=False)
    weights: pd.DataFrame = field(repr=False)
    cost_returns: pd.Series = field(repr=False)
    gross_returns: pd.Series = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return all(c.passed for c in self.constraints)

    @property
    def constraint_violation(self) -> float:
        return sum(c.violation for c in self.constraints)

    @property
    def score(self) -> float:
        """Composite score: net IR minus penalty for constraint violations."""
        return self.net_ir - 10.0 * self.constraint_violation

    @property
    def candidate_id(self) -> str:
        return f"{self.model_name}|{self.allocator_name}|cost={self.cost_bps:g}"

    def to_dict(self) -> dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "model_name": self.model_name,
            "allocator_name": self.allocator_name,
            "cost_bps": self.cost_bps,
            "net_ir": self.net_ir,
            "net_sharpe": self.net_sharpe,
            "gross_sharpe": self.gross_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "annual_excess_return": self.annual_excess_return,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "passed_all": self.passed_all,
            "constraint_violation": self.constraint_violation,
            "score": self.score,
            **{f"constraint_{c.name}": c.value for c in self.constraints},
            **{f"constraint_{c.name}_passed": c.passed for c in self.constraints},
        }
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"meta_{key}"] = value
            elif isinstance(value, tuple):
                row[f"meta_{key}"] = ",".join(str(item) for item in value)
            else:
                row[f"meta_{key}"] = str(value)
        return row


@dataclass
class NetIRConstrainedOptimizer:
    """Maximize net out-of-sample information ratio subject to constraints.

    Constraints:
    - max_drawdown >= drawdown_limit (e.g., -20%)
    - turnover <= max_turnover (e.g., 3.0 annual)
    - max_country_weight <= max_country_weight_limit (e.g., 40%)
    - cost_stress: net_sharpe at 2x cost_bps > 0
    - deflated_sharpe >= min_deflated_sharpe (e.g., 0.80)
    """

    drawdown_limit: float = -0.20
    max_turnover: float = 3.0
    max_country_weight_limit: float = 0.40
    max_asset_weight_limit: float = 0.20
    min_deflated_sharpe: float = 0.80
    target_volatility: float = 0.12
    periods_per_year: int = 252
    train_size: int = 504
    test_size: int = 63
    embargo: int = 5
    gross_exposure: float = 1.0
    max_leverage: float = 3.0
    precompute_full_signals: bool = False

    def _weights_from_scores(
        self,
        scores: pd.DataFrame,
        weighting: str,
        group_map: pd.Series | None,
        top_fraction: float,
        max_weight: float | None = None,
    ) -> pd.DataFrame:
        if weighting == "dollar_neutral":
            return scores_to_weights(scores, gross_exposure=self.gross_exposure)
        if weighting == "long_only":
            return scores_to_long_only_weights(scores, gross_exposure=self.gross_exposure, top_fraction=top_fraction)
        if weighting == "long_only_capped":
            return scores_to_long_only_capped_weights(
                scores,
                gross_exposure=self.gross_exposure,
                top_fraction=top_fraction,
                max_weight=max_weight or self.max_asset_weight_limit,
            )
        if weighting == "group_neutral_dollar_neutral":
            if group_map is None:
                raise ValueError("group_map required")
            return scores_to_group_neutral_weights(scores, group_map, gross_exposure=self.gross_exposure)
        if weighting == "group_neutral_long_only":
            if group_map is None:
                raise ValueError("group_map required")
            return scores_to_group_neutral_long_only_weights(
                scores, group_map, gross_exposure=self.gross_exposure, top_fraction=top_fraction
            )
        if weighting == "group_neutral_long_only_capped":
            if group_map is None:
                raise ValueError("group_map required")
            return scores_to_group_neutral_long_only_capped_weights(
                scores,
                group_map,
                gross_exposure=self.gross_exposure,
                top_fraction=top_fraction,
                max_weight=max_weight or self.max_asset_weight_limit,
            )
        raise ValueError(f"unknown weighting: {weighting}")

    def _apply_rebalance(
        self, weights: pd.DataFrame, frequency: str
    ) -> pd.DataFrame:
        if frequency == "W":
            return weekly_rebalance(weights)
        if frequency == "M":
            return monthly_rebalance(weights)
        return weights

    def _compute_costs(
        self,
        weights: pd.DataFrame,
        cost_bps: float,
        group_map: pd.Series | None,
    ) -> pd.Series:
        if group_map is not None:
            if cost_bps <= 0:
                return pd.Series(0.0, index=weights.index)

            scale = cost_bps / BASE_COUNTRY_COST_BPS
            country_schedule = {}
            for asset in weights.columns:
                country = group_map.get(asset)
                if country:
                    country_schedule[asset] = country_cost_schedule(country) * scale
            per_asset_turnover = weights.diff().abs().fillna(0.0)
            cost_vector = (
                pd.Series(country_schedule, dtype=float)
                .reindex(weights.columns)
                .fillna(cost_bps / 10000.0)
            )
            return per_asset_turnover.mul(cost_vector, axis=1).sum(axis=1)
        return linear_transaction_cost(weights, cost_bps)

    def _evaluate_constraints(
        self,
        net_returns: pd.Series,
        gross_returns: pd.Series,
        weights: pd.DataFrame,
        group_map: pd.Series | None,
        cost_bps: float,
        trial_sharpes: list[float],
    ) -> tuple[ConstraintResult, ...]:
        constraints: list[ConstraintResult] = []
        pp = self.periods_per_year

        dd = max_drawdown(net_returns)
        constraints.append(
            ConstraintResult(
                name="max_drawdown",
                passed=dd >= self.drawdown_limit,
                value=dd,
                limit=self.drawdown_limit,
                violation=max(0.0, self.drawdown_limit - dd),
            )
        )

        turnover = weights.diff().abs().sum(axis=1).mean()
        constraints.append(
            ConstraintResult(
                name="turnover",
                passed=turnover <= self.max_turnover,
                value=turnover,
                limit=self.max_turnover,
                violation=max(0.0, turnover - self.max_turnover),
            )
        )

        if group_map is not None and not group_map.empty:
            country_weights = self._country_max_weight(weights, group_map)
            max_cw = country_weights.max() if not country_weights.empty else 0.0
            constraints.append(
                ConstraintResult(
                    name="max_country_weight",
                    passed=max_cw <= self.max_country_weight_limit,
                    value=max_cw,
                    limit=self.max_country_weight_limit,
                    violation=max(0.0, max_cw - self.max_country_weight_limit),
                )
            )
        else:
            constraints.append(
                ConstraintResult(
                    name="max_country_weight",
                    passed=True,
                    value=0.0,
                    limit=self.max_country_weight_limit,
                    violation=0.0,
                )
            )

        max_aw = weights.abs().max().max() if not weights.empty else 0.0
        constraints.append(
            ConstraintResult(
                name="max_asset_weight",
                passed=max_aw <= self.max_asset_weight_limit,
                value=max_aw,
                limit=self.max_asset_weight_limit,
                violation=max(0.0, max_aw - self.max_asset_weight_limit),
            )
        )

        cs = cost_stress(gross_returns, weights, cost_grid=(0, cost_bps, cost_bps * 2))
        net_sharpe_2x = 0.0
        if not cs.empty and len(cs) >= 3:
            net_sharpe_2x = float(cs.iloc[2]["net_sharpe"])
        constraints.append(
            ConstraintResult(
                name="cost_stress_2x",
                passed=net_sharpe_2x > 0.0,
                value=net_sharpe_2x,
                limit=0.0,
                violation=max(0.0, -net_sharpe_2x),
            )
        )

        dsr = deflated_sharpe_ratio(net_returns, trial_sharpes=trial_sharpes, periods_per_year=pp)
        constraints.append(
            ConstraintResult(
                name="deflated_sharpe",
                passed=dsr >= self.min_deflated_sharpe,
                value=dsr,
                limit=self.min_deflated_sharpe,
                violation=max(0.0, self.min_deflated_sharpe - dsr),
            )
        )

        return tuple(constraints)

    def _country_max_weight(self, weights: pd.DataFrame, group_map: pd.Series) -> pd.Series:
        aligned = group_map.reindex(weights.columns)
        country_weights = pd.Series(0.0, index=weights.index)
        for country in aligned.dropna().unique():
            cols = aligned[aligned == country].index
            common = [c for c in cols if c in weights.columns]
            if common:
                country_weights = pd.concat([country_weights, weights[common].sum(axis=1)], axis=1).sum(axis=1)
        cw = pd.DataFrame(index=weights.index)
        for country in aligned.dropna().unique():
            cols = aligned[aligned == country].index
            common = [c for c in cols if c in weights.columns]
            if common:
                cw[country] = weights[common].sum(axis=1).abs()
        if cw.empty:
            return pd.Series(dtype=float)
        return cw.max(axis=1)

    def _allocator_name(
        self,
        weighting: str,
        top_fraction: float,
        rebalance_frequency: str,
        max_weight: float | None,
    ) -> str:
        cap_suffix = ""
        if max_weight is not None:
            cap_suffix = f"_cap{max_weight:.3f}".replace(".", "p")
        return f"{weighting}_top{int(top_fraction*100)}{cap_suffix}_{rebalance_frequency}"

    def _build_candidate_result(
        self,
        model_name: str,
        model_metadata: dict[str, Any],
        weighting: str,
        top_fraction: float,
        rebalance_frequency: str,
        max_weight: float | None,
        cost_bps: float,
        net_returns: pd.Series,
        weights_all: pd.DataFrame,
        cost_returns: pd.Series,
        gross_returns: pd.Series,
        group_map: pd.Series | None,
        benchmark_returns: pd.Series | None,
        trial_sharpes: list[float],
        splits_count: int,
    ) -> OptimizedCandidateResult:
        pp = self.periods_per_year
        net_sharpe = sharpe_ratio(net_returns, pp)
        gross_sharpe = sharpe_ratio(gross_returns, pp)
        annual_return = net_returns.mean() * pp
        annual_vol = net_returns.std(ddof=1) * (pp**0.5)
        dd = max_drawdown(net_returns)
        turnover = weights_all.diff().abs().sum(axis=1).mean()

        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned = pd.concat(
                [net_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
                axis=1,
                join="inner",
            ).dropna()
            if not aligned.empty:
                active = aligned["strategy"] - aligned["benchmark"]
                tracking_error = active.std(ddof=1) * (pp**0.5)
                annual_excess = active.mean() * pp
                ir = 0.0 if tracking_error == 0 or pd.isna(tracking_error) else annual_excess / tracking_error
            else:
                tracking_error = 0.0
                annual_excess = 0.0
                ir = 0.0
        else:
            tracking_error = 0.0
            annual_excess = 0.0
            ir = 0.0

        dsr = deflated_sharpe_ratio(net_returns, trial_sharpes=trial_sharpes, periods_per_year=pp)
        constraints = self._evaluate_constraints(
            net_returns, gross_returns, weights_all, group_map, cost_bps, trial_sharpes
        )

        return OptimizedCandidateResult(
            model_name=model_name,
            allocator_name=self._allocator_name(weighting, top_fraction, rebalance_frequency, max_weight),
            cost_bps=cost_bps,
            net_ir=ir,
            net_sharpe=net_sharpe,
            gross_sharpe=gross_sharpe,
            deflated_sharpe=dsr,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            max_drawdown=dd,
            turnover=turnover,
            annual_excess_return=annual_excess,
            tracking_error=tracking_error,
            information_ratio=ir,
            constraints=constraints,
            net_returns=net_returns,
            weights=weights_all,
            cost_returns=cost_returns,
            gross_returns=gross_returns,
            metadata={
                **model_metadata,
                "weighting": weighting,
                "top_fraction": top_fraction,
                "max_weight": max_weight,
                "rebalance_frequency": rebalance_frequency,
                "splits": splits_count,
            },
        )

    def evaluate_candidate(
        self,
        model: AlphaModel,
        weighting: str,
        top_fraction: float,
        rebalance_frequency: str,
        cost_bps: float,
        prices: pd.DataFrame,
        group_map: pd.Series | None,
        benchmark_returns: pd.Series | None,
        trial_sharpes: list[float],
        max_weight: float | None = None,
    ) -> OptimizedCandidateResult:
        """Evaluate a single model + allocator configuration."""
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        net_returns_parts: list[pd.Series] = []
        weights_parts: list[pd.DataFrame] = []
        gross_returns_parts: list[pd.Series] = []
        model_metadata: dict[str, Any] = {}

        splits = list(
            walk_forward_splits(
                prices.index,
                train_size=self.train_size,
                test_size=self.test_size,
                embargo=self.embargo,
            )
        )
        for split in splits:
            model.fit(prices, split.train_start, split.train_end)
            signals = model.predict(prices, split.test_start, split.test_end)
            signals.validate_against(prices)
            if not model_metadata:
                model_metadata = dict(signals.metadata)

            weights = self._weights_from_scores(signals.scores, weighting, group_map, top_fraction, max_weight=max_weight)
            weights = self._apply_rebalance(weights, rebalance_frequency)
            weights = volatility_target(
                weights=weights,
                asset_returns=asset_returns.reindex(weights.index),
                target_volatility=self.target_volatility,
                periods_per_year=self.periods_per_year,
                max_leverage=self.max_leverage,
            )
            if max_weight is not None and "capped" in weighting:
                weights = weights.clip(lower=0.0, upper=max_weight)

            gross = (weights.shift(1).fillna(0.0) * asset_returns.reindex(weights.index)).sum(axis=1)
            costs = self._compute_costs(weights, cost_bps, group_map)
            net = gross - costs

            net_returns_parts.append(net)
            weights_parts.append(weights)
            gross_returns_parts.append(gross)

        net_returns = pd.concat(net_returns_parts).sort_index()
        weights_all = pd.concat(weights_parts).sort_index()
        gross_returns = pd.concat(gross_returns_parts).sort_index()
        cost_returns = gross_returns - net_returns

        return self._build_candidate_result(
            model_name=model.name,
            model_metadata=model_metadata,
            weighting=weighting,
            top_fraction=top_fraction,
            rebalance_frequency=rebalance_frequency,
            max_weight=max_weight,
            cost_bps=cost_bps,
            net_returns=net_returns,
            weights_all=weights_all,
            cost_returns=cost_returns,
            gross_returns=gross_returns,
            group_map=group_map,
            benchmark_returns=benchmark_returns,
            trial_sharpes=trial_sharpes,
            splits_count=len(splits),
        )

    def _evaluate_model_grid_cached(
        self,
        model: AlphaModel,
        allocator_configs: list[dict[str, Any]],
        cost_bps_grid: tuple[float, ...],
        prices: pd.DataFrame,
        group_map: pd.Series | None,
        benchmark_returns: pd.Series | None,
    ) -> list[OptimizedCandidateResult]:
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        model_metadata: dict[str, Any] = {}
        split_scores: list[pd.DataFrame] = []
        splits = list(
            walk_forward_splits(
                prices.index,
                train_size=self.train_size,
                test_size=self.test_size,
                embargo=self.embargo,
            )
        )

        if self.precompute_full_signals:
            if splits:
                model.fit(prices, splits[0].train_start, splits[0].train_end)
            signals = model.predict(prices, 0, len(prices))
            signals.validate_against(prices)
            model_metadata = dict(signals.metadata)
            split_scores = [signals.scores.iloc[split.test_start:split.test_end] for split in splits]
        else:
            for split in splits:
                model.fit(prices, split.train_start, split.train_end)
                signals = model.predict(prices, split.test_start, split.test_end)
                signals.validate_against(prices)
                if not model_metadata:
                    model_metadata = dict(signals.metadata)
                split_scores.append(signals.scores)

        results: list[OptimizedCandidateResult] = []
        for config in allocator_configs:
            weighting = config["weighting"]
            top_fraction = config["top_fraction"]
            rebalance_frequency = config["rebalance_frequency"]
            max_weight = config.get("max_weight")

            try:
                weights_parts: list[pd.DataFrame] = []
                gross_returns_parts: list[pd.Series] = []
                for scores in split_scores:
                    weights = self._weights_from_scores(
                        scores,
                        weighting,
                        group_map,
                        top_fraction,
                        max_weight=max_weight,
                    )
                    weights = self._apply_rebalance(weights, rebalance_frequency)
                    weights = volatility_target(
                        weights=weights,
                        asset_returns=asset_returns.reindex(weights.index),
                        target_volatility=self.target_volatility,
                        periods_per_year=self.periods_per_year,
                        max_leverage=self.max_leverage,
                    )
                    if max_weight is not None and "capped" in weighting:
                        weights = weights.clip(lower=0.0, upper=max_weight)
                    gross = (weights.shift(1).fillna(0.0) * asset_returns.reindex(weights.index)).sum(axis=1)
                    weights_parts.append(weights)
                    gross_returns_parts.append(gross)

                weights_all = pd.concat(weights_parts).sort_index()
                gross_returns = pd.concat(gross_returns_parts).sort_index()
                for cost_bps in cost_bps_grid:
                    cost_parts = [
                        self._compute_costs(weights, cost_bps, group_map)
                        for weights in weights_parts
                    ]
                    cost_returns = pd.concat(cost_parts).sort_index()
                    net_returns = gross_returns - cost_returns
                    results.append(
                        self._build_candidate_result(
                            model_name=model.name,
                            model_metadata=model_metadata,
                            weighting=weighting,
                            top_fraction=top_fraction,
                            rebalance_frequency=rebalance_frequency,
                            max_weight=max_weight,
                            cost_bps=cost_bps,
                            net_returns=net_returns,
                            weights_all=weights_all,
                            cost_returns=cost_returns,
                            gross_returns=gross_returns,
                            group_map=group_map,
                            benchmark_returns=benchmark_returns,
                            trial_sharpes=[],
                            splits_count=len(splits),
                        )
                    )
            except Exception:
                continue

        return results

    def evaluate_all_candidates(
        self,
        models: list[AlphaModel],
        allocator_configs: list[dict[str, Any]],
        cost_bps_grid: tuple[float, ...],
        prices: pd.DataFrame,
        group_map: pd.Series | None,
        benchmark_returns: pd.Series | None,
    ) -> list[OptimizedCandidateResult]:
        """Evaluate all candidates and return ranked detailed results.

        Each allocator_config is a dict with keys:
        - weighting: str
        - top_fraction: float
        - max_weight: float | None
        - rebalance_frequency: str
        """
        results: list[OptimizedCandidateResult] = []
        trial_sharpes: list[float] = []

        for model in models:
            try:
                model_results = self._evaluate_model_grid_cached(
                    model=model,
                    allocator_configs=allocator_configs,
                    cost_bps_grid=cost_bps_grid,
                    prices=prices,
                    group_map=group_map,
                    benchmark_returns=benchmark_returns,
                )
            except Exception:
                continue
            trial_sharpes.extend(candidate.net_sharpe for candidate in model_results)
            results.extend(model_results)

        final_results: list[OptimizedCandidateResult] = []
        for result in results:
            constraints = self._evaluate_constraints(
                result.net_returns,
                result.gross_returns,
                result.weights,
                group_map,
                result.cost_bps,
                trial_sharpes,
            )
            final_results.append(
                OptimizedCandidateResult(
                    model_name=result.model_name,
                    allocator_name=result.allocator_name,
                    cost_bps=result.cost_bps,
                    net_ir=result.net_ir,
                    net_sharpe=result.net_sharpe,
                    gross_sharpe=result.gross_sharpe,
                    deflated_sharpe=deflated_sharpe_ratio(
                        result.net_returns, trial_sharpes=trial_sharpes, periods_per_year=self.periods_per_year
                    ),
                    annual_return=result.annual_return,
                    annual_volatility=result.annual_volatility,
                    max_drawdown=result.max_drawdown,
                    turnover=result.turnover,
                    annual_excess_return=result.annual_excess_return,
                    tracking_error=result.tracking_error,
                    information_ratio=result.information_ratio,
                    constraints=constraints,
                    net_returns=result.net_returns,
                    weights=result.weights,
                    cost_returns=result.cost_returns,
                    gross_returns=result.gross_returns,
                    metadata=result.metadata,
                )
            )

        final_results.sort(key=lambda r: r.score, reverse=True)
        return final_results

    def results_to_frame(self, results: list[OptimizedCandidateResult]) -> pd.DataFrame:
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    def optimize(
        self,
        models: list[AlphaModel],
        allocator_configs: list[dict[str, Any]],
        cost_bps_grid: tuple[float, ...],
        prices: pd.DataFrame,
        group_map: pd.Series | None,
        benchmark_returns: pd.Series | None,
    ) -> pd.DataFrame:
        """Evaluate all candidates and return a ranked DataFrame."""

        return self.results_to_frame(
            self.evaluate_all_candidates(
                models=models,
                allocator_configs=allocator_configs,
                cost_bps_grid=cost_bps_grid,
                prices=prices,
                group_map=group_map,
                benchmark_returns=benchmark_returns,
            )
        )
