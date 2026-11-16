"""Composite hypothesis card with explicit turnover units."""
from dataclasses import dataclass
from typing import Mapping
import math

@dataclass(frozen=True)
class StrategyCard:
    rebalance_interval_months: float
    tau_per_rebalance: float
    annual_turnover: float
    avg_holding_months: float
    composite_spec: Mapping
    evidence_grade: str

    def validate(self):
        values = (self.rebalance_interval_months,self.tau_per_rebalance,self.annual_turnover,self.avg_holding_months)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('invalid turnover/holding')
        if not math.isclose(self.annual_turnover,self.tau_per_rebalance*12/self.rebalance_interval_months,rel_tol=.01) or not math.isclose(self.avg_holding_months,12/self.annual_turnover,rel_tol=.01):
            raise ValueError('inconsistent holding and turnover')
        required = {'signals','weights','tilts','quotas','stage_constructions'}
        if not isinstance(self.composite_spec, Mapping) or not required <= self.composite_spec.keys() or any(self.composite_spec[k] is None for k in required) or not self.composite_spec['signals'] or not self.composite_spec['weights']:
            raise ValueError('full composite specification required')
        if self.evidence_grade != 'COMPOSITE' or self.composite_spec.get('evidence_scope', 'COMPOSITE') != 'COMPOSITE':
            raise ValueError('component prior is not sleeve evidence')
