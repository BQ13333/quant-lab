"""Independent sizing limits and post-trade monetary exposure checks."""
from typing import Mapping
import math
from .decay import finite
from .states import Permission


def compose_weight(lifecycle_cap: float, ramp_cap: float, ops_cap: float,
                   permission: Permission) -> float:
    finite(lifecycle_cap, ramp_cap, ops_cap)
    if not isinstance(permission, Permission) or any(not 0 <= c <= 1 for c in (lifecycle_cap, ramp_cap, ops_cap)):
        raise ValueError('invalid cap or permission')
    return min(lifecycle_cap, ramp_cap, ops_cap) if permission == Permission.OK else 0.


def check_exposure_cap(holdings_value_by_bucket: Mapping[str, float],
                       proposed_trades_value_by_bucket: Mapping[str, float], nav: float,
                       cap: float = .45, bucket: str = 'CNY', *,
                       lot_values: Mapping[str, float] | None = None) -> dict[str, float]:
    """Thin legacy bucket adapter to the instrument-level exposure guard."""
    def position(key, value):
        return Position(key, 'MY', value, (lot_values or {}).get(key, 1.),
                        'CNY_ECONOMIC' if key == bucket else 'NON_CNY')
    checked = check_exposure(
        [position(k,v) for k,v in holdings_value_by_bucket.items()], [],
        [position(k,v) for k,v in proposed_trades_value_by_bucket.items()], nav,
        caps={'CNY_ECONOMIC': cap, 'FOREIGN': 1.})
    return {p.instrument: p.value_nav_ccy for p in checked.approved_trades}


final_weight = compose_weight

from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Position:
    instrument: str
    market: str
    value_nav_ccy: float
    lot_value: float
    classification: str | None

@dataclass(frozen=True)
class ExposureCheck:
    approved_trades: tuple[Position, ...]
    breach: bool
    required_sell_value: dict[str, float]
    reasons: tuple[str, ...]


def check_exposure(holdings, open_orders, proposed, nav, caps=None, min_ticket=0.):
    """Reserve buys without crediting unfilled sells; overlapping caps both bind.

    Sells pass the exposure check only; this is not order approval. Proposed sells do not fund buys until filled.
    required_sell_value reports outstanding breach by bucket (do not sum overlaps).
    """
    caps = {'CNY_ECONOMIC': .45, 'FOREIGN': .60} if caps is None else dict(caps)
    finite(nav, min_ticket, list(caps.values()))
    if nav <= 0 or min_ticket < 0 or set(caps) != {'CNY_ECONOMIC', 'FOREIGN'} or any(not 0 <= x <= 1 for x in caps.values()):
        raise ValueError('invalid exposure limits')
    holdings, open_orders, proposed = tuple(holdings), tuple(open_orders), tuple(proposed)
    identities = {}
    for p in holdings + open_orders + proposed:
        finite(p.value_nav_ccy, p.lot_value)
        if not p.instrument or p.market not in ('MY', 'HK', 'CN') or p.lot_value <= 0:
            raise ValueError('invalid position')
        identity = (p.market, p.classification, p.lot_value)
        if p.instrument in identities and identities[p.instrument] != identity:
            raise ValueError('conflicting instrument classification')
        identities[p.instrument] = identity
    if any(p.value_nav_ccy < 0 for p in holdings):
        raise ValueError('negative holding')
    def buckets(p):
        return [b for b in caps if (b == 'FOREIGN' and p.market != 'MY') or
                (b == 'CNY_ECONOMIC' and (p.market == 'CN' or p.classification != 'NON_CNY'))]
    exposure = {b: sum(max(0., p.value_nav_ccy) for p in holdings + open_orders if b in buckets(p)) for b in caps}
    required = {b: max(0., exposure[b]-nav*caps[b]) for b in caps}
    reasons = [f'{b}: exposure breach' for b,v in required.items() if v > 1e-9]
    approved = []
    for p in proposed:
        value = p.value_nav_ccy
        if value > 0:
            for b in buckets(p):
                value = min(value, max(0., nav*caps[b]-exposure[b]))
            value = math.floor(value/p.lot_value)*p.lot_value
            if value < min_ticket:
                value = 0.
            if value != p.value_nav_ccy:
                reasons.append(f'{p.instrument}: buy clipped or dropped')
            for b in buckets(p):
                exposure[b] += value
        approved.append(replace(p, value_nav_ccy=value))
    return ExposureCheck(tuple(approved), any(v > 1e-9 for v in required.values()), required, tuple(reasons))
