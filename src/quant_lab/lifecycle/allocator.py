"""Discounted Gaussian Thompson sampling with cash and hard feasible bounds."""
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.optimize import linprog
from .caps import compose_weight
from .decay import finite
from .states import Lifecycle, Permission

@dataclass(frozen=True)
class Sleeve:
    permission: Permission = Permission.OK
    lifecycle: Lifecycle = Lifecycle.KEEP
    floor: float = 0.
    lifecycle_cap: float = 1.
    ramp_cap: float = 1.
    ops_cap: float = 1.
    _monitor_bound: bool = field(default=False, repr=False)

    @classmethod
    def from_monitor(cls, state, ops_cap=1., permission=Permission.BLOCK, *, floor=0.):
        # The monitor's actual permitted weight is the binding lifecycle ceiling.
        return cls(permission=permission, lifecycle=state.lifecycle,
                   lifecycle_cap=state.weight, ramp_cap=state.ramp_cap, ops_cap=ops_cap, floor=floor, _monitor_bound=True)

    def bounds(self) -> tuple[float, float]:
        if not self._monitor_bound:
            raise ValueError("sleeve must be built from monitor state")
        finite(self.floor)
        if not isinstance(self.lifecycle, Lifecycle) or not 0 <= self.floor <= 1:
            raise ValueError('invalid sleeve')
        cap = compose_weight(self.lifecycle_cap, self.ramp_cap, self.ops_cap, self.permission)
        if self.lifecycle in (Lifecycle.PAUSE, Lifecycle.RETIRED, Lifecycle.CANDIDATE, Lifecycle.VALIDATED, Lifecycle.PAPER):
            cap = 0.
        floor = self.floor if cap else 0.
        if floor > cap:
            raise ValueError('floor exceeds cap')
        return floor, cap


def constrained_weights(scores: Sequence[float], previous: Sequence[float], sleeves: Sequence[Sleeve],
                        max_turnover: float) -> np.ndarray:
    """Maximise linear score; L1 turnover includes sleeve trades, excludes cash.

    Mandatory safety cuts precede the discretionary turnover budget.
    """
    finite(scores, previous, max_turnover)
    n = len(sleeves)
    if len(scores) != n or len(previous) != n or n == 0 or max_turnover < 0:
        raise ValueError('invalid allocation dimensions/budget')
    prev = np.asarray(previous, dtype=float)
    if np.any(prev < 0) or prev.sum() > 1+1e-9:
        raise ValueError('invalid prior allocation')
    bounds = [s.bounds() for s in sleeves]
    prev = np.minimum(prev, [b[1] for b in bounds])
    # w-p <= u; p-w <= u; sum(w)<=1; sum(u)<=turnover.
    eye = np.eye(n)
    a = np.vstack([np.c_[eye, -eye], np.c_[-eye, -eye], np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]])
    b = np.r_[prev, -prev, 1., max_turnover]
    result = linprog(np.r_[-np.asarray(scores), np.full(n, 1e-10)], A_ub=a, b_ub=b,
                     bounds=bounds+[(0., None)]*n, method='highs')
    if not result.success:
        raise ValueError('floors, safety caps and turnover have no feasible allocation')
    weights = np.clip(result.x[:n], [b[0] for b in bounds], [b[1] for b in bounds])
    if weights.sum() > 1+1e-8 or np.abs(weights-prev).sum() > max_turnover+1e-8:
        raise ValueError("solver returned an infeasible allocation")
    return weights


class DiscountedThompson:
    """Discount likelihood sufficient statistics; retain a fixed Normal prior."""
    def __init__(self, n: int, *, gamma: float = .95, sigma: float = .1,
                 prior_mean: float = 0., prior_sd: float = .1, seed: int = 0):
        finite(gamma, sigma, prior_mean, prior_sd)
        if type(n) is not int or n < 1 or not 0 < gamma <= 1 or min(sigma, prior_sd) <= 0:
            raise ValueError('invalid bandit parameters')
        self.gamma, self.sigma = gamma, sigma
        self.prior_mean, self.prior_sd = prior_mean, prior_sd
        self.count = np.zeros(n); self.total = np.zeros(n)
        self.rng = np.random.default_rng(seed)

    def update(self, rewards: Sequence[float | None]) -> None:
        """Full-information shadow feedback; None discounts without imputing."""
        if len(rewards) != len(self.count):
            raise ValueError('reward dimension mismatch')
        for x in rewards:
            if x is not None:
                finite(x)
        self.count *= self.gamma; self.total *= self.gamma
        for i, x in enumerate(rewards):
            if x is not None:
                self.count[i] += 1; self.total[i] += x

    def allocate(self, previous: Sequence[float], sleeves: Sequence[Sleeve],
                 max_turnover: float = .25) -> np.ndarray:
        precision = 1/self.prior_sd**2+self.count/self.sigma**2
        mean = (self.prior_mean/self.prior_sd**2+self.total/self.sigma**2)/precision
        scores = self.rng.normal(mean, 1/np.sqrt(precision))
        return ShadowWeights(tuple(constrained_weights(scores, previous, sleeves, max_turnover)), 'UNREGISTERED_SYNTHETIC')


def inverse_variance(variances: Sequence[float], sleeves: Sequence[Sleeve]) -> np.ndarray:
    """Bounded inverse-variance baseline, with residual cash when caps bind."""
    finite(variances)
    v = np.asarray(variances, dtype=float)
    if len(v) != len(sleeves) or np.any(v <= 0):
        raise ValueError('positive aligned variances required')
    bounds = np.array([s.bounds() for s in sleeves])
    if bounds[:, 0].sum() > 1:
        raise ValueError('infeasible floors')
    low, high = 0., float(v.max()*2)
    for _ in range(80):
        mid = (low+high)/2
        if np.clip(mid/v, bounds[:, 0], bounds[:, 1]).sum() > 1:
            high = mid
        else:
            low = mid
    return np.clip(low/v, bounds[:, 0], bounds[:, 1])


def default_allocate(variances, previous, sleeves, discretionary_turnover):
    """Mandatory cap cuts first; L1 budget applies only to discretionary moves."""
    finite(previous, discretionary_turnover)
    prev = np.asarray(previous, dtype=float)
    if prev.shape != (len(sleeves),) or np.any(prev < 0) or prev.sum() > 1 + 1e-9 or discretionary_turnover < 0:
        raise ValueError('invalid prior allocation/budget')
    caps = np.array([s.bounds()[1] for s in sleeves])
    safe = np.minimum(prev, caps)
    target = inverse_variance(variances, sleeves)
    distance = np.abs(target - safe).sum()
    return safe + (target - safe) * (min(1., discretionary_turnover / distance) if distance else 0.)

@dataclass(frozen=True)
class ShadowWeights:
    weights: tuple[float, ...]
    trial_id: str
    label: str = 'EXPERIMENTAL_SHADOW_ONLY'

class ExperimentalArm:
    def __init__(self, sampler, ledger):
        self.sampler, self.ledger = sampler, ledger
        self.log = []

    def allocate(self, previous, sleeves, *, reservation, max_turnover=.25):
        trial = self.ledger.reserve_trial(**reservation)
        weights = self.sampler.allocate(previous, sleeves, max_turnover)
        result = ShadowWeights(tuple(float(x) for x in weights.weights), trial.trial_id)
        self.ledger.record_outcome(trial.trial_id, status='success', payload={'kind': result.label, 'weights': list(result.weights)})
        self.log.append(result)
        return result


def book_weights(weights):
    if isinstance(weights, ShadowWeights):
        raise ValueError('shadow weights cannot set book weights')
    result = np.asarray(weights, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0).any() or result.sum() > 1 + 1e-9:
        raise ValueError('invalid book weights')
    return result
