"""Composite hypothesis card with explicit turnover units."""
from dataclasses import dataclass
from typing import Mapping
import math

@dataclass(frozen=True)
class HoldingMeasurement:
    method: str
    weighting: str
    censoring: str
    median_months: float | None
    censored_share: float

    def validate(self):
        if (self.method, self.weighting, self.censoring) != ('FIFO_LOTS', 'ENTRY_VALUE', 'RIGHT_KM') or not math.isfinite(self.censored_share) or not 0 <= self.censored_share <= 1 or (self.median_months is not None and (not math.isfinite(self.median_months) or self.median_months < 0)):
            raise ValueError('holding measurement requires conventions and valid values')


def measure_holding(lots):
    """Entry-value weighted KM median from FIFO lot durations, including open lots.

    Each tuple is (duration_months, entry_value, closed). Split partial exits
    into their FIFO lot pieces before passing them here. None means median
    not reached; open lots remain in the risk set until their censor time.
    """
    if not lots or any(not math.isfinite(t) or t < 0 or not math.isfinite(w) or w <= 0 or type(e) is not bool for t,w,e in lots):
        raise ValueError('invalid FIFO lot observations')
    total = sum(w for t,w,e in lots)
    survival, median = 1., None
    for time in sorted({t for t,w,e in lots}):
        risk = sum(w for t,w,e in lots if t >= time)
        deaths = sum(w for t,w,e in lots if t == time and e)
        survival *= 1 - deaths / risk
        if survival <= .5 and median is None:
            median = time
    return HoldingMeasurement('FIFO_LOTS', 'ENTRY_VALUE', 'RIGHT_KM', median,
                              sum(w for t,w,e in lots if not e) / total)

@dataclass(frozen=True)
class StrategyCard:
    rebalance_interval_months: float
    tau_per_rebalance: float
    annual_turnover: float
    composite_spec: Mapping
    evidence_grade: str
    measured_holding: HoldingMeasurement | None = None
    fill_basis: str = 'DECISION'

    @property
    def turnover_implied_holding_months(self):
        return 12 / self.annual_turnover

    def validate(self):
        if self.fill_basis not in {'DECISION', 'ARRIVAL'}:
            raise ValueError('invalid fill basis')
        if self.measured_holding is not None:
            self.measured_holding.validate()
        values = (self.rebalance_interval_months,self.tau_per_rebalance,self.annual_turnover)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('invalid turnover/holding')
        if not math.isclose(self.annual_turnover,self.tau_per_rebalance*12/self.rebalance_interval_months,rel_tol=.01):
            raise ValueError('inconsistent holding and turnover')
        required = {'signals','weights','tilts','quotas','stage_constructions'}
        if not isinstance(self.composite_spec, Mapping) or not required <= self.composite_spec.keys() or any(self.composite_spec[k] is None for k in required) or not self.composite_spec['signals'] or not self.composite_spec['weights']:
            raise ValueError('full composite specification required')
        if self.evidence_grade != 'COMPOSITE' or self.composite_spec.get('evidence_scope', 'COMPOSITE') != 'COMPOSITE':
            raise ValueError('component prior is not sleeve evidence')
