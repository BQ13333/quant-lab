from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quant_lab.models.base import AlphaModel
from quant_lab.portfolio.costs import linear_transaction_cost
from quant_lab.portfolio.sizing import (
    scores_to_group_neutral_long_only_weights,
    scores_to_group_neutral_weights,
    scores_to_long_only_weights,
    scores_to_weights,
    volatility_target,
)
from quant_lab.validation.metrics import (
    calmar_ratio,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quant_lab.validation.splits import walk_forward_splits


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    annual_return: float
    annual_volatility: float
    turnover: float
    deflated_sharpe: float
    annual_excess_return: float
    tracking_error: float
    information_ratio: float
    splits: int


@dataclass
class ModelTournament:
    models: Iterable[AlphaModel]
    target_volatility: float = 0.12
    cost_bps: float = 5.0
    periods_per_year: int = 252
    weighting: str = "dollar_neutral"
    group_map: pd.Series | None = None
    top_fraction: float = 0.33

    def __post_init__(self) -> None:
        self.models = list(self.models)

    def run(
        self,
        prices: pd.DataFrame,
        train_size: int,
        test_size: int,
        embargo: int = 0,
        benchmark_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        results: list[CandidateResult] = []
        trial_sharpes: list[float] = []

        for model in self.models:
            net_returns_parts: list[pd.Series] = []
            weights_parts: list[pd.DataFrame] = []

            splits = list(
                walk_forward_splits(
                    prices.index,
                    train_size=train_size,
                    test_size=test_size,
                    embargo=embargo,
                )
            )
            for split in splits:
                model.fit(prices, split.train_start, split.train_end)
                signals = model.predict(prices, split.test_start, split.test_end)
                signals.validate_against(prices)

                weights = self._weights_from_scores(signals.scores)
                weights = volatility_target(
                    weights=weights,
                    asset_returns=asset_returns.reindex(weights.index),
                    target_volatility=self.target_volatility,
                    periods_per_year=self.periods_per_year,
                )
                gross_returns = (weights.shift(1).fillna(0.0) * asset_returns.reindex(weights.index)).sum(axis=1)
                costs = linear_transaction_cost(weights, self.cost_bps)
                net_returns_parts.append(gross_returns - costs)
                weights_parts.append(weights)

            if not net_returns_parts:
                continue

            net_returns = pd.concat(net_returns_parts).sort_index()
            weights_all = pd.concat(weights_parts).sort_index()
            sr = sharpe_ratio(net_returns, self.periods_per_year)
            trial_sharpes.append(sr)
            annual_return = net_returns.mean() * self.periods_per_year
            annual_volatility = net_returns.std(ddof=1) * (self.periods_per_year ** 0.5)
            turnover = weights_all.diff().abs().sum(axis=1).mean()

            results.append(
                CandidateResult(
                    model_name=model.name,
                    sharpe=sr,
                    sortino=sortino_ratio(net_returns, self.periods_per_year),
                    calmar=calmar_ratio(net_returns, self.periods_per_year),
                    max_drawdown=max_drawdown(net_returns),
                    annual_return=annual_return,
                    annual_volatility=annual_volatility,
                    turnover=turnover,
                    deflated_sharpe=0.0,
                    annual_excess_return=0.0,
                    tracking_error=0.0,
                    information_ratio=0.0,
                    splits=len(splits),
                )
            )

        rows = []
        for result, model_returns in zip(results, self._rerun_returns(prices, train_size, test_size, embargo)):
            active = self._active_metrics(model_returns, benchmark_returns)
            reason = ""
            try:
                dsr = deflated_sharpe_ratio(
                    model_returns, trial_sharpes=trial_sharpes,
                    periods_per_year=self.periods_per_year,
                )
            except ValueError as exc:
                dsr = float("nan")
                reason = f"invalid deflated Sharpe evidence: {exc}"
            rows.append(
                {
                    **result.__dict__,
                    "deflated_sharpe": dsr,
                    "evidence_valid": not reason,
                    "status": "rejected" if reason else "evaluated",
                    "failure_reasons": reason,
                    **active,
                }
            )

        leaderboard = pd.DataFrame(rows)
        if leaderboard.empty:
            return leaderboard
        return leaderboard.sort_values(["deflated_sharpe", "sharpe"], ascending=False).reset_index(drop=True)

    def _rerun_returns(
        self,
        prices: pd.DataFrame,
        train_size: int,
        test_size: int,
        embargo: int,
    ) -> Iterable[pd.Series]:
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        for model in self.models:
            net_returns_parts: list[pd.Series] = []
            for split in walk_forward_splits(prices.index, train_size, test_size, embargo=embargo):
                model.fit(prices, split.train_start, split.train_end)
                signals = model.predict(prices, split.test_start, split.test_end)
                weights = self._weights_from_scores(signals.scores)
                weights = volatility_target(weights, asset_returns.reindex(weights.index), self.target_volatility)
                gross_returns = (weights.shift(1).fillna(0.0) * asset_returns.reindex(weights.index)).sum(axis=1)
                net_returns_parts.append(gross_returns - linear_transaction_cost(weights, self.cost_bps))
            yield pd.concat(net_returns_parts).sort_index() if net_returns_parts else pd.Series(dtype=float)

    def _weights_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        if self.weighting == "dollar_neutral":
            return scores_to_weights(scores)
        if self.weighting == "long_only":
            return scores_to_long_only_weights(scores, top_fraction=self.top_fraction)
        if self.weighting == "group_neutral_dollar_neutral":
            if self.group_map is None:
                raise ValueError("group_map is required for group-neutral weighting")
            return scores_to_group_neutral_weights(scores, self.group_map)
        if self.weighting == "group_neutral_long_only":
            if self.group_map is None:
                raise ValueError("group_map is required for group-neutral weighting")
            return scores_to_group_neutral_long_only_weights(scores, self.group_map, top_fraction=self.top_fraction)
        raise ValueError(f"unknown weighting: {self.weighting}")

    def _active_metrics(self, returns: pd.Series, benchmark_returns: pd.Series | None) -> dict[str, float]:
        if benchmark_returns is None or returns.empty:
            return {"annual_excess_return": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}

        aligned = pd.concat(
            [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            return {"annual_excess_return": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}

        active = aligned["strategy"] - aligned["benchmark"]
        tracking_error = active.std(ddof=1) * (self.periods_per_year**0.5)
        annual_excess_return = active.mean() * self.periods_per_year
        information_ratio = 0.0 if tracking_error == 0 or pd.isna(tracking_error) else annual_excess_return / tracking_error
        return {
            "annual_excess_return": float(annual_excess_return),
            "tracking_error": float(tracking_error),
            "information_ratio": float(information_ratio),
        }
