"""Parent-order shortfall, with explicit price basis and fill provenance."""
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass(frozen=True)
class ParentOrder:
    side: int
    qty: float
    decision_px: float
    arrival_px: float
    horizon_sessions: tuple
    end_px: float

@dataclass(frozen=True)
class Fill:
    qty: float
    px: float
    ts: datetime
    source: str

@dataclass(frozen=True)
class Shortfall:
    delay: float
    execution: float
    opportunity: float
    fees: float
    notional: float
    ref: str
    source: str

    @property
    def ex_fee(self): return self.delay + self.execution + self.opportunity
    @property
    def total(self): return self.ex_fee + self.fees
    @property
    def bps(self): return self.total / self.notional * 10000

    def require_measured(self):
        if self.source != 'LIVE':
            raise ValueError('simulated fills are not slippage evidence')
        return self


def implementation_shortfall(order, fills, fees, ref='DECISION'):
    fills=tuple(fills)
    if order.side not in (-1,1) or ref not in {'DECISION','ARRIVAL'} or not order.horizon_sessions or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x <= 0 for x in (order.qty,order.decision_px,order.arrival_px,order.end_px)) or not math.isfinite(fees) or fees < 0:
        raise ValueError('invalid order/prices/fees/basis')
    for f in fills:
        if f.source not in {'SIMULATED','LIVE'} or not isinstance(f.ts,datetime) or not all(math.isfinite(x) and x > 0 for x in (f.qty,f.px)) or f.ts.date() not in {s.date() if isinstance(s,datetime) else s for s in order.horizon_sessions}:
            raise ValueError('invalid fill or outside horizon')
    filled=sum(f.qty for f in fills)
    if filled > order.qty:
        raise ValueError('overfilled parent')
    reference=order.decision_px if ref == 'DECISION' else order.arrival_px
    delay=order.side*filled*(order.arrival_px-reference)
    execution=sum(order.side*f.qty*(f.px-order.arrival_px) for f in fills)
    opportunity=order.side*(order.qty-filled)*(order.end_px-reference)
    return Shortfall(delay,execution,opportunity,fees,order.qty*reference,ref,
                     'LIVE' if fills and all(f.source == 'LIVE' for f in fills) else 'SIMULATED')


def day_means(orders):
    """(day, Shortfall) pairs: notional weighted within day, equal across days."""
    groups={}
    for day,s in orders:
        if type(s) is not Shortfall: raise ValueError('shortfall required')
        total,notional=groups.get(day,(0.,0.))
        groups[day]=(total+s.total,notional+s.notional)
    return {day:total/notional for day,(total,notional) in groups.items()}

class OrderActivityLog:
    def __init__(self): self.events=[]
    def record(self, market, ts):
        if market not in {'MY','CN','HK'} or not isinstance(ts,datetime) or ts.tzinfo is None:
            raise ValueError('market and timezone-aware order time required')
        self.events.append((market,ts))
    def daily_count(self, market, day):
        from zoneinfo import ZoneInfo
        zone=ZoneInfo('Asia/Shanghai')  # all three markets UTC+8
        return sum(m==market and ts.astimezone(zone).date()==day for m,ts in self.events)
    def order_rate(self, market, start, end):
        seconds=(end-start).total_seconds()
        if seconds <= 0: raise ValueError('positive interval required')
        return sum(m==market and start <= ts < end for m,ts in self.events)/seconds
