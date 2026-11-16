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
    """Long-only bucket values in one NAV currency; round trades toward zero.

    lot_values gives monetary lot increments (default one currency unit). Buys
    are clipped to available headroom. Existing drift above the cap requires a
    sufficient proposed sell: otherwise no feasible scaled trade exists and we
    raise ValueError. No unrequested liquidation is manufactured.
    """
    finite(nav, cap, list(holdings_value_by_bucket.values()), list(proposed_trades_value_by_bucket.values()))
    if nav <= 0 or not 0 <= cap <= 1 or any(v < 0 for v in holdings_value_by_bucket.values()):
        raise ValueError('invalid holdings, NAV or cap')
    result = {}
    for key, value in proposed_trades_value_by_bucket.items():
        lot = (lot_values or {}).get(key, 1.)
        finite(lot)
        if lot <= 0:
            raise ValueError('lot must be positive')
        holding = holdings_value_by_bucket.get(key, 0.)
        if value > 0 and key == bucket:
            value = min(value, max(0., cap*nav-holding))
        value = max(value, -holding)
        result[key] = math.trunc(value/lot)*lot
    if holdings_value_by_bucket.get(bucket, 0.) + result.get(bucket, 0.) > cap*nav + 1e-9:
        raise ValueError('drifted holdings require more risk-reducing sells')
    return result
