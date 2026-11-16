from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from quant_lab.validation.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)
from quant_lab.validation.stress import (
    cost_stress,
    no_single_country_dominance,
    no_single_month_dominance,
)



def _finite(value: object, name: str, reasons: list[str]) -> float | None:
    """Parse a gate input; non-finite evidence fails closed."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        reasons.append(f"{name} is not a finite number")
        return None
    if not math.isfinite(number):
        reasons.append(f"{name} is not a finite number")
        return None
    return number


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ValidationGate:
    min_sharpe: float = 0.5
    max_drawdown_limit: float = -0.2
    max_turnover: float = 3.0
    min_deflated_sharpe_probability: float = 0.8

    def evaluate(
        self,
        sharpe: float,
        max_drawdown: float,
        turnover: float,
        deflated_sharpe_probability: float,
    ) -> GateDecision:
        reasons: list[str] = []
        sharpe_v = _finite(sharpe, "sharpe", reasons)
        if sharpe_v is not None and sharpe_v < self.min_sharpe:
            reasons.append(f"sharpe {sharpe_v:.2f} below {self.min_sharpe:.2f}")
        dd_v = _finite(max_drawdown, "max_drawdown", reasons)
        if dd_v is not None and dd_v < self.max_drawdown_limit:
            reasons.append(f"max drawdown {dd_v:.2%} below limit {self.max_drawdown_limit:.2%}")
        to_v = _finite(turnover, "turnover", reasons)
        if to_v is not None and to_v > self.max_turnover:
            reasons.append(f"turnover {to_v:.2f} above {self.max_turnover:.2f}")
        dsr_v = _finite(deflated_sharpe_probability, "deflated_sharpe_probability", reasons)
        if dsr_v is not None and dsr_v < self.min_deflated_sharpe_probability:
            reasons.append(
                "deflated Sharpe probability "
                f"{dsr_v:.2f} below {self.min_deflated_sharpe_probability:.2f}"
            )
        return GateDecision(passed=not reasons, reasons=tuple(reasons))

    def promote(
        self,
        leaderboard_row: dict,
        weights: pd.DataFrame | None = None,
        group_map: pd.Series | None = None,
    ) -> GateDecision:
        """Promotion gate with strict criteria for live deployment.

        Expected keys in *leaderboard_row*:
            - sharpe (float)
            - deflated_sharpe (float)
            - annual_excess_return (float)
            - information_ratio (float)
            - max_drawdown (float)
            - turnover (float)
            - cost_bps (float, optional)
            - returns (pd.Series, optional)
        """
        reasons: list[str] = []
        row = leaderboard_row
        if row.get('cost_label') == 'LAB_SIMULATED_COST':
            reasons.append('LAB_SIMULATED_COST cannot approve trading')
        from quant_lab.models.signal_factory import require_admission_markets
        try:
            require_admission_markets(row.get('markets', ('MY',)))
        except ValueError:
            reasons.append('out-of-scope admission market')

        # Net Sharpe thresholds — non-finite evidence fails closed
        net_sharpe = _finite(row.get("sharpe", 0.0), "sharpe", reasons)
        if net_sharpe is not None:
            if net_sharpe < 0.8:
                reasons.append(f"net sharpe {net_sharpe:.2f} below minimum 0.80")
            elif net_sharpe < 1.25:
                # Not fatal, but noted
                pass

        # Deflated Sharpe probability
        dsr = _finite(row.get("deflated_sharpe", 0.0), "deflated_sharpe", reasons)
        if dsr is not None:
            if dsr < 0.80:
                reasons.append(f"deflated Sharpe probability {dsr:.2f} below minimum 0.80")
            elif dsr < 0.95:
                pass

        # Excess return vs equal-weight
        excess = _finite(row.get("annual_excess_return", 0.0), "annual_excess_return", reasons)
        if excess is not None and excess <= 0.0:
            reasons.append(f"annual excess return {excess:.2%} not positive")

        # Information ratio
        ir = _finite(row.get("information_ratio", 0.0), "information_ratio", reasons)
        if ir is not None and ir <= 0.0:
            reasons.append(f"information ratio {ir:.2f} not positive")

        # Cost stress: survives 2x assumed cost
        cost_bps_v = _finite(row.get("cost_bps", 5.0), "cost_bps", reasons)
        cost_bps = 5.0 if cost_bps_v is None else cost_bps_v
        returns = row.get("returns")
        if returns is not None and isinstance(returns, pd.Series) and not returns.empty:
            cs = cost_stress(returns, weights=weights, cost_grid=(0, cost_bps, cost_bps * 2))
            if not cs.empty and len(cs) >= 3:
                net_sharpe_2x = cs.iloc[2]["net_sharpe"]
                net_sharpe_2x_v = _finite(net_sharpe_2x, "net_sharpe_at_2x_cost", reasons)
                if net_sharpe_2x_v is not None and net_sharpe_2x_v <= 0.0:
                    reasons.append(f"net Sharpe at 2x cost ({cost_bps * 2:.0f} bps) {net_sharpe_2x_v:.2f} not positive")
        else:
            # Cannot verify cost stress without returns
            pass

        # Max drawdown within mandate
        dd = _finite(row.get("max_drawdown", 0.0), "max_drawdown", reasons)
        if dd is not None and dd < self.max_drawdown_limit:
            reasons.append(f"max drawdown {dd:.2%} below mandate {self.max_drawdown_limit:.2%}")

        # Turnover
        to = _finite(row.get("turnover", 0.0), "turnover", reasons)
        if to is not None and to > self.max_turnover:
            reasons.append(f"turnover {to:.2f} above limit {self.max_turnover:.2f}")

        # Country / month dominance checks
        if returns is not None and isinstance(returns, pd.Series) and not returns.empty:
            if weights is not None and group_map is not None and not group_map.empty:
                if not no_single_country_dominance(returns, weights, group_map, max_country_contribution=0.5):
                    reasons.append("single country contribution exceeds 50%")
            if not no_single_month_dominance(returns, max_month_contribution=0.3):
                reasons.append("single month contribution exceeds 30%")

        return GateDecision(passed=not reasons, reasons=tuple(reasons))
