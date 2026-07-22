from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.research.bounded_loop import (
    ALLOWED_FEATURES,
    BASE_COST_BPS_ONE_WAY,
    STRESS_COST_BPS_ONE_WAY,
    BacktestEvidence,
    CampaignConfig,
    CandidateGeneration,
    MalaysiaUniverseContract,
    MutationDirective,
)
from quant_lab.research.contracts import CandidateSpec
from quant_lab.research.institutional_workflow import MalaysiaAuditInputs
from quant_lab.research.malaysia_strategy_registry import (
    FrozenMalaysiaStrategy,
    validate_strategy_parameters,
)
from quant_lab.research.providers import ModelCapability
from quant_lab.research.research_extractor import ResearchExtractor
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.metrics import deflated_sharpe_ratio, sharpe_ratio
from quant_lab.validation.overfit import probability_of_backtest_overfitting


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class OllamaCandidateGenerator:
    """Generate one schema-valid candidate locally; no hosted fallback or ledger write occurs here."""

    def __init__(self, extractor: ResearchExtractor) -> None:
        if extractor.ledger is not None:
            raise ValueError("controller owns model-call persistence; extractor ledger must be unset")
        self.extractor = extractor

    def generate(
        self,
        directive: MutationDirective,
        *,
        trial_id: str,
        config: CampaignConfig,
    ) -> CandidateGeneration:
        provenance_hashes = {
            "directive": _digest(json.dumps(directive.__dict__, sort_keys=True, default=str)),
            "campaign": _digest(config.campaign_id),
        }
        payload = {
            "task": "Produce one bounded implementation candidate, not a performance claim.",
            "required_identity": {
                "candidate_id": f"MYS-C-{directive.directive_id}",
                "hypothesis_id": directive.hypothesis_id,
                "parent_candidate_id": directive.parent_candidate_id,
                "family": directive.family,
            },
            "market": "Malaysia",
            "exchange": "Bursa Malaysia",
            "universe_policy": {"primary": "U3", "primary_count": 34, "robustness": "U2", "robustness_count": 62},
            "fixed_costs_bps_one_way": {"base": BASE_COST_BPS_ONE_WAY, "stress": STRESS_COST_BPS_ONE_WAY},
            "allowed_features": list(ALLOWED_FEATURES),
            "mutation_codes": list(directive.mutation_codes),
            "implementation_allowlist": ["momentum", "mean_reversion", "low_volatility", "volatility_adjusted_momentum"],
            "parameter_contract": {
                "lookback": {"type": "integer", "allowed": [21, 42, 63, 84, 126, 252]},
                "rebalance_frequency": {"type": "string", "allowed": ["W-FRI", "ME"]},
                "top_fraction": {"type": "number", "allowed": [0.10, 0.20, 0.25]},
                "generation": {"exact": directive.generation},
                "mutation_codes": {"exact": list(directive.mutation_codes)},
                "features": {"exact": list(ALLOWED_FEATURES)},
            },
            "required_exact_fields": {
                "contract_type": "candidate",
                "schema_version": "1.0",
                "evidence": [],
                "provenance_hashes": provenance_hashes,
            },
            "constraints": [
                "Use the required identity exactly.",
                "Use required_exact_fields exactly; evidence must be the empty JSON array [].",
                "implementation_ref must be quant_lab.research.malaysia_pilot:MalaysiaHistoricalBacktester.",
                "parameters may contain lookback, rebalance_frequency, top_fraction, generation, mutation_codes, and features only.",
                "Every supplied parameter must follow parameter_contract exactly; never write monthly, weekly, or aliases.",
                "Do not include another market, alternative universe, holdout result, broker, or order instruction.",
            ],
        }
        receipt = self.extractor.extract(
            request_id=f"MYS-CANDIDATE-{_digest(directive.directive_id)[:20]}",
            capability=ModelCapability.HYPOTHESIS_EXTRACTION,
            expected_contract=CandidateSpec,
            payload=payload,
            max_tokens=1200,
            timeout_seconds=300,
        )
        candidate = receipt.result.contract
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("local candidate extraction returned the wrong contract")
        return CandidateGeneration(candidate, receipt.result.call_record, external_cost_usd=0.0)


class FrozenCohortCandidateGenerator:
    """Build the predeclared candidate exactly; Ollama reviews but never tunes it."""

    def __init__(self, strategies: tuple[FrozenMalaysiaStrategy, ...]) -> None:
        self.strategies = {item.directive_id: item for item in strategies}

    def generate(
        self,
        directive: MutationDirective,
        *,
        trial_id: str,
        config: CampaignConfig,
    ) -> CandidateGeneration:
        try:
            strategy = self.strategies[directive.directive_id]
        except KeyError as exc:
            raise KeyError(f"directive is not in the frozen cohort batch: {directive.directive_id}") from exc
        strategy.validate()
        if strategy.hypothesis_id != directive.hypothesis_id or strategy.family != directive.family:
            raise ValueError("frozen strategy identity differs from its reserved directive")
        parameters = dict(strategy.parameters)
        parameters["generation"] = directive.generation
        parameters["mutation_codes"] = list(directive.mutation_codes)
        candidate = CandidateSpec(
            candidate_id=f"MYS-C-{strategy.strategy_id}",
            hypothesis_id=strategy.hypothesis_id,
            family=strategy.family,
            implementation_ref="quant_lab.research.malaysia_pilot:MalaysiaHistoricalBacktester",
            parameters=parameters,
            evidence=(),
            provenance_hashes={
                "strategy_spec": strategy.content_hash(),
                "source": _digest(f"{strategy.source_file}::{strategy.source_section}"),
                "cohort": _digest("MALAYSIA-ATOMIC-COHORT-V1"),
            },
            parent_candidate_id=directive.parent_candidate_id,
        )
        candidate.validate()
        return CandidateGeneration(candidate, None, external_cost_usd=0.0)


class MalaysiaHistoricalBacktester:
    """One fixed-parameter candidate evaluated on U3 primary and unchanged U2 robustness."""

    def __init__(
        self,
        *,
        daily_features_path: str | Path,
        ledger: TrialLedger,
        research_end: str = "2026-07-06",
    ) -> None:
        self.daily_features_path = Path(daily_features_path)
        self.ledger = ledger
        self.research_end = pd.Timestamp(research_end)
        self._candidate_returns: dict[str, pd.Series] = {}
        self._audit_inputs: dict[str, MalaysiaAuditInputs] = {}
        self._raw_cache: pd.DataFrame | None = None

    def evaluate(
        self,
        candidate: CandidateSpec,
        *,
        trial_id: str,
        universe: MalaysiaUniverseContract,
        config: CampaignConfig,
    ) -> BacktestEvidence:
        params = dict(candidate.parameters)
        validate_strategy_parameters(
            candidate.family,
            params,
            generation=int(params["generation"]),
            mutation_codes=tuple(params["mutation_codes"]),
        )
        lookback = int(params.get("lookback", params.get("slow", 63)))
        frequency = str(params.get("rebalance_frequency", "W-FRI"))
        top_fraction = float(params.get("top_fraction", 0.20))
        columns = [
            "date",
            "ticker",
            "country",
            "adj_close",
            "simple_return",
            "adv_60_local_lag1",
            "vol_63_lag1",
        ]
        if self._raw_cache is None:
            loaded = pd.read_parquet(self.daily_features_path, columns=columns)
            loaded["date"] = pd.to_datetime(loaded["date"])
            self._raw_cache = loaded.loc[
                loaded["country"].eq("Malaysia")
                & loaded["ticker"].isin(universe.u2)
                & loaded["date"].le(self.research_end)
            ].sort_values(["date", "ticker"])
        raw = self._raw_cache
        if raw.empty or raw["date"].nunique() < 504:
            raise ValueError("insufficient Malaysia research history")
        price = raw.pivot(index="date", columns="ticker", values="adj_close").reindex(columns=universe.u2)
        returns = raw.pivot(index="date", columns="ticker", values="simple_return").reindex(columns=universe.u2).fillna(0.0)
        adv = raw.pivot(index="date", columns="ticker", values="adv_60_local_lag1").reindex(columns=universe.u2)
        vol = raw.pivot(index="date", columns="ticker", values="vol_63_lag1").reindex(columns=universe.u2)
        score = self._score(candidate.family, price, adv, vol, params)
        finite_score = score.replace([np.inf, -np.inf], np.nan)
        if int(finite_score.notna().sum().sum()) == 0:
            raise ValueError("candidate generated no finite signal observations")
        u3_net, u3_stress, u3_weights, turnover = self._portfolio_returns(
            score.reindex(columns=universe.u3),
            returns.reindex(columns=universe.u3),
            frequency,
            top_fraction,
        )
        u2_net, _, _, _ = self._portfolio_returns(score, returns, frequency, top_fraction)
        self._candidate_returns[candidate.candidate_id] = u3_net
        self.ledger.record_candidate_returns(
            trial_id,
            candidate_id=candidate.candidate_id,
            family=candidate.family,
            returns=u3_net,
        )
        u3_returns = returns.reindex(columns=universe.u3)
        u3_prices = price.reindex(columns=universe.u3)
        u3_adv = adv.reindex(columns=universe.u3)
        u3_vol = vol.reindex(columns=universe.u3)
        market_returns = u3_returns.mean(axis=1).fillna(0.0)
        factor_returns = pd.DataFrame(
            {
                "market": market_returns,
                "momentum": self._cross_sectional_factor_return(
                    u3_prices.pct_change(126, fill_method=None).shift(1), u3_returns
                ),
                "low_volatility": self._cross_sectional_factor_return(-u3_vol.shift(1), u3_returns),
                "lagged_liquidity": self._cross_sectional_factor_return(np.log1p(u3_adv.shift(1)), u3_returns),
            }
        ).fillna(0.0)
        self._audit_inputs[candidate.candidate_id] = MalaysiaAuditInputs(
            strategy_returns=u3_net.copy(),
            market_returns=market_returns,
            factor_returns=factor_returns,
            regime_labels=self._causal_regime_labels(market_returns),
        )
        matrix = self.ledger.candidate_return_matrix(campaign="malaysia")
        pbo = (
            probability_of_backtest_overfitting(matrix, n_splits=8, strict=True)
            if matrix.shape[1] >= 2 and len(matrix) >= 16
            else 1.0
        )
        lifetime = max(1, self.ledger.lifetime_counts(campaign="malaysia")["lifetime_attempts"])
        raw_sharpes = [sharpe_ratio(matrix[column]) for column in matrix]
        denominator = raw_sharpes + [0.0] * max(0, lifetime - len(raw_sharpes))
        dsr = deflated_sharpe_ratio(u3_net, denominator, periods_per_year=252)
        cumulative = (1.0 + u3_net.fillna(0.0)).cumprod()
        max_drawdown = float((cumulative / cumulative.cummax() - 1.0).min())
        last_weights = u3_weights.iloc[-1].dropna() if not u3_weights.empty else pd.Series(dtype=float)
        latest_prices = price.reindex(columns=universe.u3).ffill().iloc[-1]
        snapshot = {
            "candidate_hash": candidate.content_hash(),
            "data_hash": universe.source_hash,
            "research_end": self.research_end.isoformat(),
            "lookback": lookback,
            "frequency": frequency,
            "top_fraction": top_fraction,
            "family": candidate.family,
            "parameters": params,
            "base_cost": BASE_COST_BPS_ONE_WAY,
            "stress_cost": STRESS_COST_BPS_ONE_WAY,
        }
        return BacktestEvidence(
            candidate_id=candidate.candidate_id,
            u3_count=len(universe.u3),
            u2_count=len(universe.u2),
            base_cost_bps_one_way=BASE_COST_BPS_ONE_WAY,
            stress_cost_bps_one_way=STRESS_COST_BPS_ONE_WAY,
            raw_sharpe=float(sharpe_ratio(u3_net + turnover * BASE_COST_BPS_ONE_WAY / 10_000.0)),
            net_oos_ir=float(sharpe_ratio(u3_net)),
            deflated_sharpe_probability=float(dsr),
            pbo=float(pbo),
            max_drawdown=max_drawdown,
            turnover=float(turnover.mean() * 252.0),
            stress_net_oos_ir=float(sharpe_ratio(u3_stress)),
            max_asset_weight=float(last_weights.abs().max()) if not last_weights.empty else 1.0,
            u2_net_oos_ir=float(sharpe_ratio(u2_net)),
            target_weights={str(key): float(value) for key, value in last_weights.items() if value != 0.0},
            reference_prices={str(key): float(value) for key, value in latest_prices.items() if pd.notna(value)},
            config_hash=_digest(json.dumps(snapshot, sort_keys=True)),
        )

    def institutional_audit_inputs(self, candidate_id: str) -> MalaysiaAuditInputs:
        try:
            inputs = self._audit_inputs[candidate_id]
        except KeyError as exc:
            raise KeyError(f"candidate has no completed Malaysia backtest: {candidate_id}") from exc
        return MalaysiaAuditInputs(
            strategy_returns=inputs.strategy_returns.copy(),
            market_returns=inputs.market_returns.copy(),
            factor_returns=inputs.factor_returns.copy(),
            regime_labels=inputs.regime_labels.copy(),
        )

    @staticmethod
    def _cross_sectional_factor_return(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        ranks = signal.rank(axis=1, pct=True, method="average")
        long = ranks.ge(0.80).astype(float)
        short = ranks.le(0.20).astype(float)
        long = long.div(long.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        short = short.div(short.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        return ((long - short) * returns.reindex_like(signal).fillna(0.0)).sum(axis=1)

    @staticmethod
    def _causal_regime_labels(market_returns: pd.Series) -> pd.Series:
        trend = market_returns.rolling(126, min_periods=126).sum().shift(1)
        volatility = market_returns.rolling(63, min_periods=63).std().shift(1) * np.sqrt(252)
        low = volatility.expanding(min_periods=252).quantile(0.33).shift(1)
        high = volatility.expanding(min_periods=252).quantile(0.67).shift(1)
        labels = pd.Series(index=market_returns.index, dtype="object", name="regime")
        valid = trend.notna() & volatility.notna() & low.notna() & high.notna()
        trend_label = pd.Series(np.where(trend >= 0.0, "up", "down"), index=trend.index)
        vol_label = pd.Series(
            np.where(volatility <= low, "low", np.where(volatility >= high, "high", "mid")),
            index=volatility.index,
        )
        labels.loc[valid] = (trend_label + "_" + vol_label).loc[valid]
        return labels

    @staticmethod
    def _score(
        family: str,
        price: pd.DataFrame,
        adv: pd.DataFrame,
        vol: pd.DataFrame,
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        lookback = int(parameters.get("lookback", parameters.get("slow", 63)))
        volatility_lookback = int(parameters.get("volatility_lookback", 63))
        asset_returns = price.pct_change(fill_method=None)
        momentum = price.pct_change(lookback, fill_method=None).shift(1)
        realized_vol = asset_returns.rolling(
            volatility_lookback, min_periods=max(5, volatility_lookback // 3)
        ).std().shift(1)
        if family == "momentum":
            return momentum
        if family == "mean_reversion":
            return -momentum
        if family == "low_volatility":
            return -realized_vol
        if family == "volatility_adjusted_momentum":
            return momentum / realized_vol.replace(0.0, np.nan)
        if family == "time_series_momentum":
            return np.sign(momentum) / realized_vol.replace(0.0, np.nan)
        if family == "ewmac":
            fast = int(parameters["fast"])
            slow = int(parameters["slow"])
            fast_ema = price.ewm(span=fast, min_periods=max(5, fast // 3), adjust=False).mean().shift(1)
            slow_ema = price.ewm(span=slow, min_periods=max(5, slow // 3), adjust=False).mean().shift(1)
            return fast_ema.sub(slow_ema).div(price.shift(1).replace(0.0, np.nan)).div(
                realized_vol.replace(0.0, np.nan)
            )
        if family in {"breakout", "channel_position"}:
            lagged = price.shift(1)
            high = price.shift(2).rolling(lookback, min_periods=max(5, lookback // 3)).max()
            low = price.shift(2).rolling(lookback, min_periods=max(5, lookback // 3)).min()
            if family == "breakout":
                return pd.DataFrame(
                    np.where(lagged > high, 1.0, np.where(lagged < low, -1.0, 0.0)),
                    index=price.index,
                    columns=price.columns,
                )
            return lagged.sub(low).div(high.sub(low).replace(0.0, np.nan)).mul(2.0).sub(1.0)
        if family == "residual_momentum":
            beta_lookback = int(parameters["beta_lookback"])
            market = asset_returns.mean(axis=1)
            minimum = max(20, beta_lookback // 2)
            covariance = asset_returns.rolling(beta_lookback, min_periods=minimum).cov(market)
            market_variance = market.rolling(beta_lookback, min_periods=minimum).var()
            beta = covariance.div(market_variance.replace(0.0, np.nan), axis=0).shift(1)
            residual = asset_returns.sub(beta.mul(market, axis=0))
            return residual.rolling(lookback, min_periods=max(20, lookback // 2)).sum().shift(1)
        if family == "bollinger_reversion":
            lagged = price.shift(1)
            mean = lagged.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
            std = lagged.rolling(lookback, min_periods=max(5, lookback // 3)).std()
            zscore = lagged.sub(mean).div(std.replace(0.0, np.nan))
            threshold = float(parameters["entry_z"])
            return -zscore.where(zscore.abs() >= threshold, 0.0)
        if family == "lagged_liquidity":
            return np.log1p(adv.shift(1))
        raise ValueError("candidate family is outside the Malaysia price/liquidity allowlist")

    @staticmethod
    def _portfolio_returns(
        score: pd.DataFrame,
        returns: pd.DataFrame,
        frequency: str,
        top_fraction: float,
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
        ranks = score.rank(axis=1, pct=True, method="average")
        raw_weights = ranks.ge(1.0 - top_fraction).astype(float)
        raw_weights = raw_weights.div(raw_weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        sampled = raw_weights.resample(frequency).last()
        expanded_index = raw_weights.index.union(sampled.index)
        weights = sampled.reindex(expanded_index).sort_index().ffill().reindex(raw_weights.index).fillna(0.0)
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        gross = (weights * returns.reindex_like(weights).fillna(0.0)).sum(axis=1)
        base = gross - turnover * BASE_COST_BPS_ONE_WAY / 10_000.0
        stress = gross - turnover * STRESS_COST_BPS_ONE_WAY / 10_000.0
        return base, stress, weights, turnover
