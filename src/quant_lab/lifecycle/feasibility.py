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
    label: str = 'NECESSARY_ONLY'

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

@dataclass(frozen=True)
class Instrument:
    price: float
    lot_size: int
    min_ticket: float
    adv: float
    fx: float

@dataclass(frozen=True)
class StageConstruction:
    stage: float
    names: tuple[str, ...]
    target_values: tuple[float, ...]  # NAV currency, at this stage

@dataclass(frozen=True)
class StageLimits:
    participation: float
    weight_tolerance: float
    cost_rate: float
    order_cap_pct: float = .02

@dataclass(frozen=True)
class StageReport:
    stage: float
    status: str
    reasons: tuple[str, ...]
    quantities: tuple[tuple[str, int], ...]
    children: tuple[tuple[str, tuple[int, ...]], ...]
    total_cost: float
    execution_limits: tuple = ()  # name, min shares, max daily shares, lot shares

    @property
    def passed(self):
        return self.status == 'FEASIBLE'


def stage_feasibility(construction, instruments, nav, cash, reserves, limits):
    """Check lots, daily children, aggregate participation, costs and cash jointly.

    ADV and tickets are in instrument currency. Weight tolerance is relative to
    each target value; cash, reserves and target values are in NAV currency.
    """
    c = construction
    scalars = (nav, cash, reserves, limits.participation, limits.weight_tolerance,
               limits.cost_rate, limits.order_cap_pct, c.stage)
    if any(not math.isfinite(x) for x in scalars) or nav <= 0 or min(cash, reserves, limits.cost_rate, limits.weight_tolerance) < 0 or not 0 < limits.participation <= 1 or not 0 < limits.order_cap_pct <= .02 or not 0 < c.stage <= 1 or not c.names or len(set(c.names)) != len(c.names) or len(c.names) != len(c.target_values):
        raise ValueError('invalid stage inputs')
    reasons, quantities, children = [], [], []
    total = 0.
    execution_limits = []
    for name, target in zip(c.names, c.target_values):
        if name not in instruments:
            reasons.append(f'{name}: missing instrument'); continue
        i = instruments[name]
        if any(not math.isfinite(x) or x <= 0 for x in (target, i.price, i.lot_size, i.min_ticket, i.adv, i.fx)) or type(i.lot_size) is not int:
            raise ValueError('invalid instrument/target')
        lot = i.price * i.lot_size * i.fx
        lots = math.floor(target / lot + 1e-12)
        value = lots * lot
        quantities.append((name, lots * i.lot_size))
        # Both constraints apply to the sum of children on each session.
        max_lots = math.floor(min(nav * limits.order_cap_pct, i.adv * i.fx * limits.participation) / lot)
        min_lots = math.ceil(i.min_ticket * i.fx / lot - 1e-12)
        execution_limits.append((name, min_lots*i.lot_size, max_lots*i.lot_size, i.lot_size))
        if not lots or max_lots < min_lots:
            reasons.append(f'{name}: ticket/participation/order cap'); continue
        n = math.ceil(lots / max_lots)
        base, extra = divmod(lots, n)
        chunks = tuple((base + (j < extra)) * i.lot_size for j in range(n))
        if base < min_lots:
            reasons.append(f'{name}: child below ticket')
        if abs(value - target) / target > limits.weight_tolerance + 1e-12:
            reasons.append(f'{name}: lot rounding exceeds weight tolerance')
        children.append((name, chunks))
        total += value * (1 + limits.cost_rate)
    if total + reserves > cash + 1e-8:
        reasons.append('cash: costs plus reserves exceed cash')
    return StageReport(c.stage, 'INFEASIBLE' if reasons else 'FEASIBLE', tuple(reasons), tuple(quantities), tuple(children), total, tuple(execution_limits))


def first_feasible_stage(plan, instruments, nav, cash, reserves, limits):
    reports = tuple(stage_feasibility(c, instruments, nav, cash, reserves, limits) for c in plan)
    passing = [r.stage for r in reports if r.passed]
    if not passing:
        raise ValueError('INFEASIBLE: no executable stage')
    return min(passing)
