"""Day-level measured cost bounds; no imputation for unobserved buckets."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import t

NOT_EVALUABLE = 'NOT_EVALUABLE'

@dataclass(frozen=True)
class Bound:
    value: float
    days: int
    source: str = 'LIVE'
    margin: float = 0.
    def __post_init__(self):
        if not np.isfinite([self.value,self.margin]).all() or self.margin < 0 or type(self.days) is not int or self.days < 0 or self.source != 'LIVE':
            raise ValueError('invalid measured bound')

@dataclass(frozen=True)
class StressAssumption:
    value: float
    label: str = 'STRESS_ASSUMPTION'


def unit_impact(sigma_d, q, adv):
    if not np.isfinite([sigma_d,q,adv]).all() or min(sigma_d,q) < 0 or adv <= 0:
        raise ValueError('invalid impact inputs')
    return sigma_d*np.sqrt(q/adv)


def mean_unit_impact(impacts, notionals):
    x,w=np.asarray(impacts,float),np.asarray(notionals,float)
    if x.ndim != 1 or not len(x) or x.shape != w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (x<0).any() or (w<=0).any():
        raise ValueError('invalid impacts/weights')
    order=np.argsort(x)
    p95=x[order][np.searchsorted(np.cumsum(w[order])/w.sum(),.95)]
    return dict(mean=float(np.average(x,weights=w)),p95=float(p95),max=float(x.max()))


def robust_upper_bound(day_means):
    x=np.asarray(day_means,float)
    if x.ndim != 1 or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('at least three finite day means required')
    n=len(x); z=x-x.mean(); variance=np.var(x,ddof=1)
    gamma0=float(z@z/n)
    long_run=gamma0+2*sum((1-lag/3)*float(z[lag:]@z[:-lag]/n) for lag in (1,2))
    inflation=max(1.,long_run/gamma0) if gamma0 else 1.
    margin=upper_margin(np.sqrt(variance),n,inflation)
    return Bound(float(x.mean()+margin),n,margin=margin)


def next_stage_bound(observed, next_impacts, next_notionals, obs_impacts, obs_notionals, kappa_hi):
    if type(observed) is not Bound or observed.days < 10 or not np.isfinite(kappa_hi) or kappa_hi < 0:
        raise ValueError('measured cost evidence required')
    delta=mean_unit_impact(next_impacts,next_notionals)['mean']-mean_unit_impact(obs_impacts,obs_notionals)['mean']
    return Bound(observed.value+kappa_hi*max(0.,delta),observed.days,margin=observed.margin)


def transfer_bound(bucket_bounds, target_weights, min_days=10):
    if type(min_days) is not int or min_days < 10 or not target_weights or not np.isfinite(list(target_weights.values())).all() or any(w<0 for w in target_weights.values()) or not np.isclose(sum(target_weights.values()),1):
        raise ValueError('invalid registered buckets/minimum')
    selected=[]
    for bucket,w in target_weights.items():
        if w == 0: continue
        b=bucket_bounds.get(bucket)
        if type(b) is not Bound or b.days < min_days: return NOT_EVALUABLE
        selected.append((w,b))
    return Bound(sum(w*b.value for w,b in selected),min(b.days for w,b in selected))


def stress_scenario(model_cost, multiplier):
    if not np.isfinite([model_cost,multiplier]).all() or model_cost < 0 or multiplier < 1:
        raise ValueError('invalid stress assumption')
    return StressAssumption(model_cost*multiplier)


def cost_test(bound, limit, min_days=10):
    return type(bound) is Bound and type(min_days) is int and min_days >= 10 and bound.days >= min_days and np.isfinite(limit) and bound.value <= limit


def upper_margin(day_sd, days, inflation):
    """One-sided t margin with conservative floored effective degrees of freedom."""
    if not np.isfinite([day_sd,days,inflation]).all() or day_sd<0 or type(days) is not int or days<3 or inflation<1:
        raise ValueError('invalid uncertainty inputs')
    df=int(np.floor(days/inflation))-1
    if df<1: raise ValueError('insufficient effective days')
    return float(t.ppf(.95,df)*day_sd*np.sqrt(inflation/days))
