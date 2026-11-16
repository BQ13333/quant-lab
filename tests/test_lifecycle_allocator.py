import numpy as np
import pytest
from quant_lab.lifecycle.allocator import *


def sleeve(*, lifecycle=Lifecycle.KEEP, ramp_cap=1., ops_cap=1., permission=Permission.OK, floor=0.):
    from quant_lab.lifecycle.monitor import MonitorState
    return Sleeve.from_monitor(MonitorState(lifecycle=lifecycle,weight=1.,ramp_cap=ramp_cap),ops_cap,permission,floor=floor)


def test_constraints_seed_and_infeasibility():
    sleeves = [sleeve(ramp_cap=.3, floor=.1), sleeve(lifecycle=Lifecycle.PAUSE),
               sleeve(lifecycle=Lifecycle.RETIRED), sleeve(permission=Permission.BLOCK), sleeve(ops_cap=.4)]
    a = DiscountedThompson(5, prior_mean=.2, seed=12)
    b = DiscountedThompson(5, prior_mean=.2, seed=12)
    p = np.zeros(5)
    for _ in range(10):
        w = np.asarray(a.allocate(p, sleeves, .2).weights)
        assert np.array_equal(w, b.allocate(p, sleeves, .2).weights)
        assert .1 <= w[0] <= .3 and w[4] <= .4
        assert np.all(w[1:4] == 0) and abs(w-p).sum() <= .2+1e-8
        a.update([.1]*5); b.update([.1]*5); p = w
    assert constrained_weights([1], [1], [sleeve(lifecycle=Lifecycle.PAUSE)], .2) == [0]


def test_bandit_beats_baseline_on_decaying_alpha_smoke_synthetic():
    """Synthetic smoke test, not evidence for adoption."""
    # Two independent seeds, full shadow feedback; decisions precede rewards.
    advantages = []
    for seed in (8, 19):
        rng = np.random.default_rng(seed)
        bandit = DiscountedThompson(3, gamma=.85, sigma=.04, prior_sd=.1, seed=seed)
        sleeves = [sleeve()]*3
        base = inverse_variance([.04**2]*3, sleeves)
        p = np.zeros(3); earned = []; benchmark = []
        for t in range(180):
            means = np.array([.15*np.exp(-t/25), .15*np.exp(-max(0,t-60)/25) if t >= 60 else -.03,
                              .15 if t >= 120 else -.03])
            p = np.asarray(bandit.allocate(p, sleeves, .4).weights)
            rewards = means+rng.normal(0,.04,3)
            earned.append(p@rewards); benchmark.append(base@rewards)
            bandit.update(rewards)
        advantages.append(np.mean(earned)-np.mean(benchmark))
    assert min(advantages) > .025


def test_baseline_bounds_and_nan_update_atomic():
    weights = inverse_variance([1,4], [sleeve(ops_cap=.6), sleeve()])
    assert weights == pytest.approx([.6,.4])
    bandit = DiscountedThompson(2)
    with pytest.raises(ValueError):
        bandit.update([1,np.nan])
    assert np.all(bandit.count == 0)


def test_sleeve_built_from_monitor_state():
    from quant_lab.lifecycle.monitor import MonitorState
    state = MonitorState(weight=.2,ramp_cap=.25)
    sleeve = Sleeve.from_monitor(state,ops_cap=.15,permission=Permission.OK)
    assert sleeve.bounds() == (0.,.15)
    assert Sleeve.from_monitor(state).bounds() == (0.,0.)


def test_mandatory_cuts_never_blocked_by_turnover_budget():
    assert default_allocate([1],[1],[sleeve(lifecycle=Lifecycle.PAUSE)],0) == [0]

def test_default_is_deterministic_and_seed_free():
    assert np.array_equal(default_allocate([1,2],[.4,.6],[sleeve()]*2,.1),default_allocate([1,2],[.4,.6],[sleeve()]*2,.1))

def test_default_respects_discretionary_turnover():
    w=default_allocate([1,3],[0,0],[sleeve()]*2,.1)
    assert w.sum() == pytest.approx(.1)

def test_thompson_output_cannot_set_book_weights():
    with pytest.raises(ValueError): book_weights(DiscountedThompson(1).allocate([0],[sleeve()]))

def test_shadow_arm_is_logged_and_charged(tmp_path):
    from quant_lab.research.trial_ledger import TrialLedger
    ledger=TrialLedger(tmp_path/'l.db',tmp_path/'e')
    arm=ExperimentalArm(DiscountedThompson(1),ledger)
    reservation=dict(idempotency_key='shadow',campaign='lab',family='shadow',hypothesis_id='h',candidate_id='c',**{k:'a'*64 for k in ('code_hash','data_hash','config_hash','prompt_hash','model_hash')})
    output=arm.allocate([0],[sleeve()],reservation=reservation)
    assert arm.log == [output] and ledger.snapshot().charged_count == 1
