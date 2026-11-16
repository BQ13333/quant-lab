"""Synthetic net active-return experiment, never an allocator adoption verdict.

Means, covariance and birth intensity use annual units. Monthly returns have
covariance annual_cov/12. Births are research proposals, never live admissions.
Ramping requires the same evidence and human authorization as the book.
"""
from dataclasses import dataclass
import numpy as np
from .allocator import Sleeve, default_allocate, book_weights
from .states import Permission
from .ramp import advance_ramp

@dataclass(frozen=True)
class ReturnSimulation:
    gross: np.ndarray
    net: np.ndarray
    costs: np.ndarray
    live_counts: tuple[int, ...]
    births: tuple[int, ...]
    annual_net_active_return: float
    net_active_ir: float


def simulate_returns(annual_means, annual_cov, states, *, months=120, annual_birth_rate=0., cost_rate=.001,
                     discretionary_turnover=.25, seed=0, state_updates=None, ramp_events=None, monitor_events=None):
    means=np.asarray(annual_means,float); cov=np.asarray(annual_cov,float)
    n=len(states)
    if means.shape!=(n,) or cov.shape!=(n,n) or n==0 or not np.isfinite(means).all() or not np.isfinite(cov).all() or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min() < -1e-12 or np.any(np.diag(cov)<=0) or type(months) is not int or months<2 or not np.isfinite([annual_birth_rate,cost_rate,discretionary_turnover]).all() or min(annual_birth_rate,cost_rate,discretionary_turnover)<0:
        raise ValueError('invalid annual return simulator inputs')
    rng=np.random.default_rng(seed)
    returns=rng.multivariate_normal(means/12,cov/12,size=months)
    births=tuple(int(x) for x in rng.poisson(annual_birth_rate/12,months))
    states=list(states); previous=np.zeros(n); gross=[]; costs=[]; live=[]
    for month,row in enumerate(returns):
        if state_updates and month in state_updates:
            proposed=state_updates[month]
            if len(proposed)!=n: raise ValueError('state dimension mismatch')
            # Only risk cuts through this hook; increases use signed ramp events.
            if any(new.weight>old.weight or new.ramp_cap>old.ramp_cap for old,new in zip(states,proposed)):
                raise ValueError('unsigned risk increase')
            states=list(proposed)
        for index,plan,evidence in (ramp_events or {}).get(month,()):
            states[index]=advance_ramp(states[index],plan,evidence)
        for index,inputs,calendar,policy in (monitor_events or {}).get(month,()):
            from .monitor import evaluate
            states[index]=evaluate(states[index],inputs,calendar,policy).state
        sleeves=[Sleeve.from_monitor(s,permission=Permission.OK) for s in states]
        weights=book_weights(default_allocate(np.diag(cov),previous,sleeves,discretionary_turnover))
        # Reserve cash for turnover charges before earning this month's return.
        if weights.sum()+np.abs(weights-previous).sum()*cost_rate > 1:
            low,high=0.,1.
            for _ in range(60):
                scale=(low+high)/2
                if (weights*scale).sum()+np.abs(weights*scale-previous).sum()*cost_rate <= 1:
                    low=scale
                else: high=scale
            weights=weights*low
        gross.append(float(weights@row))
        costs.append(float(np.abs(weights-previous).sum()*cost_rate))
        live.append(sum(s.bounds()[1]>0 for s in sleeves))
        # Drift the actual holdings before next month's turnover computation.
        denominator=1+float(weights@row)-costs[-1]
        if denominator<=0 or (row < -1).any(): raise ValueError('synthetic insolvent return path')
        previous=weights*(1+row)/denominator
        if previous.sum()>1+1e-9:
            # Costs must be funded from available cash; never silently borrow.
            raise ValueError('insufficient cash for simulated costs')
    gross=np.asarray(gross); costs=np.asarray(costs); net=gross-costs
    sd=float(net.std(ddof=1))
    return ReturnSimulation(gross,net,costs,tuple(live),births,float(net.mean()*12),float(net.mean()/sd*np.sqrt(12)) if sd else 0.)
