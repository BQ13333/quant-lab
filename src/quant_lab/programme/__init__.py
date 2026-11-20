"""Programme mandate: scope, source limits, and the cost and lot gaps.

This package records Benjamin Quek's 20 November 2026 decisions. It does not
place orders, and it does not reimplement the unmerged market-rules engine.
"""

from quant_lab.programme.cost_coverage import historical_cost_gate_met, load_cost_coverage
from quant_lab.programme.hk_lots import HkLotUnavailable, lot_size, lot_table_gate_met
from quant_lab.programme.scope import MarketScopeError, load_market_scope

__all__ = [
    "HkLotUnavailable",
    "MarketScopeError",
    "historical_cost_gate_met",
    "load_cost_coverage",
    "load_market_scope",
    "lot_size",
    "lot_table_gate_met",
]
