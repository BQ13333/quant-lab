from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


FROZEN_FEATURES = ("price", "volume", "volatility", "lagged_liquidity")
COHORT_VERSION = "MALAYSIA-ATOMIC-COHORT-V1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FrozenMalaysiaStrategy:
    sequence: int
    strategy_id: str
    directive_id: str
    hypothesis_id: str
    family: str
    economic_claim: str
    source_file: str
    source_section: str
    horizon_days: int
    parameters: Mapping[str, Any]

    def validate(self) -> None:
        if not 1 <= self.sequence <= 10:
            raise ValueError("atomic cohort sequence must be within 1..10")
        if not re.fullmatch(r"MYS-S-[A-Z0-9_-]{3,80}", self.strategy_id):
            raise ValueError("strategy_id must be a canonical Malaysia strategy identifier")
        if not re.fullmatch(r"MYS-H-COHORT-[A-Z0-9_-]{3,80}", self.hypothesis_id):
            raise ValueError("cohort hypothesis_id is invalid")
        if len(self.economic_claim) < 20 or not self.source_file.endswith(".md") or not self.source_section:
            raise ValueError("strategy requires a falsifiable claim and Markdown provenance")
        validate_strategy_parameters(self.family, self.parameters, generation=1, mutation_codes=())

    def content_hash(self) -> str:
        self.validate()
        return _digest(
            {
                "cohort_version": COHORT_VERSION,
                "sequence": self.sequence,
                "strategy_id": self.strategy_id,
                "directive_id": self.directive_id,
                "hypothesis_id": self.hypothesis_id,
                "family": self.family,
                "economic_claim": self.economic_claim,
                "source_file": self.source_file,
                "source_section": self.source_section,
                "horizon_days": self.horizon_days,
                "parameters": dict(self.parameters),
            }
        )


@dataclass(frozen=True)
class PredeclaredCombination:
    combination_id: str
    component_strategy_ids: tuple[str, ...]
    weighting: str = "equal_weight"

    def validate(self) -> None:
        if not re.fullmatch(r"MYS-COMBO-[A-Z0-9_-]{3,80}", self.combination_id):
            raise ValueError("combination_id is invalid")
        if not 2 <= len(self.component_strategy_ids) <= 3 or len(set(self.component_strategy_ids)) != len(
            self.component_strategy_ids
        ):
            raise ValueError("combinations require two or three distinct atomic strategies")
        if self.weighting != "equal_weight":
            raise ValueError("the first cohort permits equal-weight combinations only")


def _parameters(**values: Any) -> dict[str, Any]:
    return {
        **values,
        "generation": 1,
        "mutation_codes": [],
        "features": list(FROZEN_FEATURES),
    }


def malaysia_atomic_cohort() -> tuple[FrozenMalaysiaStrategy, ...]:
    provenance = "docs/STRATEGY_PROVENANCE.md"
    kakushadze = provenance
    kaufman = provenance
    carver = provenance
    cohort = (
        FrozenMalaysiaStrategy(
            1, "MYS-S-LOWVOL-126", "MYS-COHORT-01-LOWVOL-126", "MYS-H-COHORT-LOWVOL-126",
            "low_volatility", "Lower realized-volatility Bursa stocks outperform after costs with smaller drawdowns.",
            kakushadze, "3.4 Low-Volatility Anomaly", 126,
            _parameters(lookback=126, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            2, "MYS-S-MOMENTUM-126", "MYS-COHORT-02-MOMENTUM-126", "MYS-H-COHORT-MOMENTUM-126",
            "momentum", "Slow cross-sectional price momentum persists in Malaysia after conservative trading costs.",
            kakushadze, "3.1 Price-Momentum", 126,
            _parameters(lookback=126, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            3, "MYS-S-VAM-126-63", "MYS-COHORT-03-VAM-126-63", "MYS-H-COHORT-VAM-126-63",
            "volatility_adjusted_momentum", "Volatility-scaled slow momentum improves stability relative to raw momentum in Malaysia.",
            kakushadze, "3.1 Price-Momentum", 126,
            _parameters(lookback=126, volatility_lookback=63, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            4, "MYS-S-TSMOM-126", "MYS-COHORT-04-TSMOM-126", "MYS-H-COHORT-TSMOM-126",
            "time_series_momentum", "Volatility-scaled time-series momentum identifies persistent directional trends in Bursa stocks.",
            carver, "EWMAC rule and volatility targeting framework", 126,
            _parameters(lookback=126, volatility_lookback=63, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            5, "MYS-S-EWMAC-16-64", "MYS-COHORT-05-EWMAC-16-64", "MYS-H-COHORT-EWMAC-16-64",
            "ewmac", "A slow volatility-scaled exponential moving-average crossover captures durable Malaysia trends.",
            carver, "EWMAC rule and volatility targeting framework", 64,
            _parameters(fast=16, slow=64, volatility_lookback=64, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            6, "MYS-S-BREAKOUT-55", "MYS-COHORT-06-BREAKOUT-55", "MYS-H-COHORT-BREAKOUT-55",
            "breakout", "Fifty-five-day price breakouts contain persistent directional information after Malaysia costs.",
            kaufman, "Trend, moving average, breakout and channel systems", 55,
            _parameters(lookback=55, rebalance_frequency="W-FRI", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            7, "MYS-S-CHANNEL-55", "MYS-COHORT-07-CHANNEL-55", "MYS-H-COHORT-CHANNEL-55",
            "channel_position", "Position within a lagged 55-day price channel predicts subsequent Bursa returns.",
            kakushadze, "3.15 Channel", 55,
            _parameters(lookback=55, rebalance_frequency="W-FRI", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            8, "MYS-S-RESIDMOM-126", "MYS-COHORT-08-RESIDMOM-126", "MYS-H-COHORT-RESIDMOM-126",
            "residual_momentum", "Market-residual momentum contains incremental Malaysia stock-selection information.",
            kakushadze, "3.7 Residual Momentum", 126,
            _parameters(lookback=126, beta_lookback=126, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            9, "MYS-S-MEANREV-63", "MYS-COHORT-09-MEANREV-63", "MYS-H-COHORT-MEANREV-63",
            "mean_reversion", "Medium-horizon cross-sectional losers mean-revert after Malaysia trading costs.",
            kakushadze, "3.9-3.10 Mean-Reversion", 63,
            _parameters(lookback=63, rebalance_frequency="ME", top_fraction=0.20),
        ),
        FrozenMalaysiaStrategy(
            10, "MYS-S-BOLLREV-63", "MYS-COHORT-10-BOLLREV-63", "MYS-H-COHORT-BOLLREV-63",
            "bollinger_reversion", "Large lagged deviations from a 63-day mean revert in Bursa stocks after costs.",
            kaufman, "Mean reversion and volatility systems", 63,
            _parameters(lookback=63, entry_z=1.0, rebalance_frequency="W-FRI", top_fraction=0.20),
        ),
    )
    if tuple(item.sequence for item in cohort) != tuple(range(1, 11)):
        raise RuntimeError("atomic cohort sequence is not contiguous")
    for item in cohort:
        item.validate()
    if len({item.content_hash() for item in cohort}) != len(cohort):
        raise RuntimeError("atomic cohort contains a duplicate specification")
    return cohort


def malaysia_predeclared_combinations() -> tuple[PredeclaredCombination, ...]:
    combinations = (
        PredeclaredCombination(
            "MYS-COMBO-TREND-LOWVOL",
            ("MYS-S-EWMAC-16-64", "MYS-S-LOWVOL-126"),
        ),
        PredeclaredCombination(
            "MYS-COMBO-MOMENTUM-REVERSION",
            ("MYS-S-MOMENTUM-126", "MYS-S-MEANREV-63"),
        ),
        PredeclaredCombination(
            "MYS-COMBO-BREAKOUT-RESIDUAL-LOWVOL",
            ("MYS-S-BREAKOUT-55", "MYS-S-RESIDMOM-126", "MYS-S-LOWVOL-126"),
        ),
    )
    atomic_ids = {item.strategy_id for item in malaysia_atomic_cohort()}
    for item in combinations:
        item.validate()
        if not set(item.component_strategy_ids).issubset(atomic_ids):
            raise RuntimeError("combination refers to a strategy outside the frozen atomic cohort")
    return combinations


def strategy_by_directive(directive_id: str) -> FrozenMalaysiaStrategy:
    for item in malaysia_atomic_cohort():
        if item.directive_id == directive_id:
            return item
    raise KeyError(f"unknown frozen Malaysia directive: {directive_id}")


def validate_strategy_parameters(
    family: str,
    parameters: Mapping[str, Any],
    *,
    generation: int,
    mutation_codes: tuple[str, ...],
) -> None:
    common = {"rebalance_frequency", "top_fraction", "generation", "mutation_codes", "features"}
    family_fields = {
        "momentum": {"lookback"},
        "mean_reversion": {"lookback"},
        "low_volatility": {"lookback"},
        "volatility_adjusted_momentum": {"lookback", "volatility_lookback"},
        "time_series_momentum": {"lookback", "volatility_lookback"},
        "ewmac": {"fast", "slow", "volatility_lookback"},
        "breakout": {"lookback"},
        "channel_position": {"lookback"},
        "residual_momentum": {"lookback", "beta_lookback"},
        "bollinger_reversion": {"lookback", "entry_z"},
        "lagged_liquidity": {"lookback"},
    }
    if family not in family_fields:
        raise ValueError("candidate family is outside the frozen Malaysia strategy registry")
    if set(parameters) != common | family_fields[family]:
        raise ValueError("candidate parameters do not exactly match the frozen family contract")
    if parameters["rebalance_frequency"] not in {"W-FRI", "ME"}:
        raise ValueError("candidate rebalance frequency is outside the frozen grid")
    if float(parameters["top_fraction"]) not in {0.10, 0.20, 0.25}:
        raise ValueError("candidate top fraction is outside the frozen grid")
    if int(parameters["generation"]) != generation:
        raise ValueError("candidate generation does not match its directive")
    if tuple(parameters["mutation_codes"]) != mutation_codes:
        raise ValueError("candidate mutation codes do not match its directive")
    if tuple(parameters["features"]) != FROZEN_FEATURES:
        raise ValueError("candidate features drifted from the frozen Malaysia contract")
    for key in ("lookback", "volatility_lookback", "beta_lookback", "fast", "slow"):
        if key in parameters and not 5 <= int(parameters[key]) <= 252:
            raise ValueError(f"candidate {key} is outside the frozen horizon bounds")
    if family == "ewmac" and int(parameters["fast"]) >= int(parameters["slow"]):
        raise ValueError("EWMAC fast horizon must be smaller than slow horizon")
    if "entry_z" in parameters and float(parameters["entry_z"]) not in {1.0, 1.5, 2.0}:
        raise ValueError("Bollinger entry threshold is outside the frozen grid")
