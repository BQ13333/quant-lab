"""Executable quantities, separate from monetary exposure approval.

Inventory is instrument-local. settled_for_sale excludes today's CN purchases;
open sell orders must already be represented in reserved. Checking reserves stock.
"""
from dataclasses import dataclass
import math

@dataclass
class Inventory:
    owned: float
    settled_for_sale: float
    reserved: float = 0.

    def validate(self):
        if not all(math.isfinite(x) for x in (self.owned, self.settled_for_sale, self.reserved)) or not 0 <= self.reserved <= self.settled_for_sale <= self.owned:
            raise ValueError('invalid inventory')

@dataclass(frozen=True)
class Order:
    instrument: str
    market: str
    side: str
    qty: float

@dataclass(frozen=True)
class ExecutableCheck:
    approved_qty: float
    reasons: tuple[str, ...]


def reserve(inventory, order):
    inventory.validate()
    if order.side != 'SELL' or not math.isfinite(order.qty) or not 0 <= order.qty <= inventory.settled_for_sale - inventory.reserved:
        raise ValueError('cannot reserve unavailable stock')
    inventory.reserved += order.qty


def check_executable(order, inventory, market_state, rules):
    from dataclasses import replace
    inventory.validate()
    if order.market not in {'MY', 'CN', 'HK'} or order.side not in {'BUY', 'SELL'} or not order.instrument or not math.isfinite(order.qty) or order.qty <= 0:
        raise ValueError('invalid order')
    if market_state.get('suspended') is not False or rules.get('verified') is not True:
        return ExecutableCheck(0., ('suspended or unknown executable rules',))
    if order.side == 'BUY':
        return ExecutableCheck(order.qty, ())
    qty = min(order.qty, inventory.settled_for_sale - inventory.reserved)
    reserve(inventory, replace(order, qty=qty))
    return ExecutableCheck(qty, ('sell clipped to free settled inventory',) if qty < order.qty else ())
