import numpy as np
import pytest
from quant_lab.lifecycle.allocator import *


def test_constraints_seed_and_infeasibility():
    sleeves = [Sleeve(ramp_cap=.3, floor=.1), Sleeve(lifecycle=Lifecycle.PAUSE),
               Sleeve(lifecycle=Lifecycle.RETIRED), Sleeve(permission=Permission.BLOCK), Sleeve(ops_cap=.4)]
    a = DiscountedThompson(5, prior_mean=.2, seed=12)
    b = DiscountedThompson(5, prior_mean=.2, seed=12)
    p = np.zeros(5)
    for _ in range(10):
        w = a.allocate(p, sleeves, .2)
        assert np.array_equal(w, b.allocate(p, sleeves, .2))
        assert .1 <= w[0] <= .3 and w[4] <= .4
        assert np.all(w[1:4] == 0) and abs(w-p).sum() <= .2+1e-8
        a.update([.1]*5); b.update([.1]*5); p = w
    with pytest.raises(ValueError, match='feasible'):
        constrained_weights([1], [1], [Sleeve(lifecycle=Lifecycle.PAUSE)], .2)


def test_bandit_beats_baseline_on_decaying_alpha():
    # Two independent seeds, full shadow feedback; decisions precede rewards.
    advantages = []
    for seed in (8, 19):
        rng = np.random.default_rng(seed)
        bandit = DiscountedThompson(3, gamma=.85, sigma=.04, prior_sd=.1, seed=seed)
        sleeves = [Sleeve()]*3
        base = inverse_variance([.04**2]*3, sleeves)
        p = np.zeros(3); earned = []; benchmark = []
        for t in range(180):
            means = np.array([.15*np.exp(-t/25), .15*np.exp(-max(0,t-60)/25) if t >= 60 else -.03,
                              .15 if t >= 120 else -.03])
            p = bandit.allocate(p, sleeves, .4)
            rewards = means+rng.normal(0,.04,3)
            earned.append(p@rewards); benchmark.append(base@rewards)
            bandit.update(rewards)
        advantages.append(np.mean(earned)-np.mean(benchmark))
    assert min(advantages) > .025


def test_baseline_bounds_and_nan_update_atomic():
    weights = inverse_variance([1,4], [Sleeve(ops_cap=.6), Sleeve()])
    assert weights == pytest.approx([.6,.4])
    bandit = DiscountedThompson(2)
    with pytest.raises(ValueError):
        bandit.update([1,np.nan])
    assert np.all(bandit.count == 0)
