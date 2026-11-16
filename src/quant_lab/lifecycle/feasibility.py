"""Planning-currency capital floors and equal child-order construction."""
from dataclasses import dataclass, replace
import math

@dataclass(frozen=True)
class Limits:
    ticket: float = 5000.
    order_cap_pct: float = .02
    pilot_share_cap: float = .20
    pilot_abs_cap: float = 75000.
    names: int = 20
    pilot_names: int = 6
    pilot_ticket_mult: float = 1.25
    pilot_reserve: float = 1.10

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()) or self.order_cap_pct > 1 or self.pilot_share_cap > 1 or any(type(v) is not int for v in (self.names, self.pilot_names)) or self.pilot_ticket_mult < 1 or self.pilot_reserve < 1:
            raise ValueError('invalid limits')

@dataclass(frozen=True)
class FloorReport:
    minimum_nav: float
    binding_constraint: str
    status: str = 'FEASIBLE'
    reason: str = ''

    def require_nav(self, nav):
        if not math.isfinite(nav) or self.status != 'FEASIBLE' or nav < self.minimum_nav:
            raise ValueError('INFEASIBLE: book below capital floor')


def pilot_funding(ticket_myr, limits=None):
    limits = limits or Limits(ticket=ticket_myr)
    if not math.isfinite(ticket_myr) or ticket_myr <= 0:
        raise ValueError('invalid ticket')
    return limits.pilot_names * limits.pilot_ticket_mult * ticket_myr * limits.pilot_reserve


def child_orders(position, nav, limits):
    if not all(math.isfinite(x) and x > 0 for x in (position, nav)):
        raise ValueError('INFEASIBLE: invalid position/NAV')
    count = math.ceil(position/(nav*limits.order_cap_pct)-1e-12)
    child = position/count
    if child < limits.ticket-1e-8:
        raise ValueError('INFEASIBLE: child below minimum ticket')
    return (child,)*count


def min_nav(weights, tickets_myr, limits=Limits()):
    if len(weights) != len(tickets_myr) or not weights or any(not math.isfinite(x) or x <= 0 for x in (*weights, *tickets_myr)) or not math.isclose(sum(weights), 1.):
        raise ValueError('invalid sleeve construction')
    constraints = {}
    for i, (w,ticket) in enumerate(zip(weights,tickets_myr)):
        funding = pilot_funding(ticket, limits)
        if funding > limits.pilot_abs_cap:
            return FloorReport(math.inf, 'pilot_abs_cap', 'INFEASIBLE', f'sleeve {i}: pilot exceeds absolute cap')
        children = math.ceil(w/limits.names/limits.order_cap_pct-1e-12)
        constraints[f'sleeve {i}: pilot_share_cap'] = funding/limits.pilot_share_cap/w
        constraints[f'sleeve {i}: pilot_order_cap'] = limits.pilot_ticket_mult*ticket/limits.order_cap_pct
        constraints[f'sleeve {i}: child_min_ticket'] = ticket*children*limits.names/w
    binding = max(constraints, key=constraints.get)
    return FloorReport(constraints[binding], binding)
