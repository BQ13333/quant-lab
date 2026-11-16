"""Canonical executable port of the lifecycle expectations in decay_example.py."""
from dataclasses import replace
import math
import pytest
from quant_lab.lifecycle.monitor import *
from quant_lab.lifecycle.states import *

LIVE = [.05,-.08,.02,-.11,.01,-.06,.04,-.13,-.02,-.09,.03,-.10]
IMPACT = 2*.60*.023*math.sqrt(31250/300e6)
BREAKEVEN = (.60*.0030+IMPACT)/(.10*.27996/.2)
POLICY = Policy(breakeven=BREAKEVEN)


def run(xs, *, policy=POLICY, state=None, costs=None, crowds=None, dd=None, capacity=None, broken=None, halts=None, retire_at=None):
    state = state or MonitorState(weight=1., ramp_cap=1.)
    rows = []
    for t, x in enumerate(xs):
        cost = Value(costs[t]) if costs is not None and costs[t] is not None else NotApplicable()
        crowd = Value(crowds[t]) if crowds is not None and crowds[t] is not None else NotApplicable()
        inp = monthly_inputs(x if isinstance(x, NoObservationExpected) else Value(x), cost=cost, crowd=crowd,
            drawdown=dd[t] if dd else 0., capacity=capacity[t] if capacity else 10.,
            mechanism_broken=broken[t] if broken else False, halt=halts[t] if halts else False)
        schedule = monthly_schedule(ic=Expectation.NO_OBSERVATION_EXPECTED if isinstance(x, NoObservationExpected) else Expectation.DUE,
                                    cost_due=isinstance(cost, Value), crowd_due=isinstance(crowd, Value))
        if retire_at == t+1:
            state = replace(state, lifecycle=Lifecycle.RETIRED)
        result = evaluate(state, inp, schedule, policy)
        state = result.state
        rows.append(result)
    return rows


def test_canonical_example():
    assert IMPACT*1200 == pytest.approx(.338, abs=.002)
    assert BREAKEVEN == pytest.approx(.01487, abs=2e-5)
    r = run(LIVE)
    for i, ratio, pnet, life, weight in [(5,.795,.481,Lifecycle.KEEP,.795), (7,.710,.427,Lifecycle.REDUCE,.5), (11,.553,.328,Lifecycle.PAUSE,0.)]:
        assert r[i].posterior.mean/POLICY.mu_p == pytest.approx(ratio, abs=.002)
        assert r[i].posterior.p_net == pytest.approx(pnet, abs=.003)
        assert r[i].state.lifecycle == life
        assert r[i].state.weight == pytest.approx(weight, abs=.002)
    assert r[7].state.cusum == pytest.approx(3.31)
    assert r[11].state.cusum == pytest.approx(5.23)
    assert r[11].posterior.p_positive == pytest.approx(.815, abs=.003)
    assert r[11].proposals == ()
    serial = run(LIVE, policy=replace(POLICY, psi=.74))
    assert next(r.state.calendar for r in serial if r.state.lifecycle == Lifecycle.REDUCE) == 10
    assert serial[-1].posterior.p_net == pytest.approx(.393, abs=.005)
    assert serial[-1].state.lifecycle == Lifecycle.PAUSE


def test_fail_and_two_clocks():
    r = run([.05, float('nan'), float('nan'), .02])
    assert r[1].permission == Permission.BLOCK and r[1].target is None
    assert r[1].state.cusum == r[0].state.cusum
    assert r[2].state.lifecycle == r[3].state.lifecycle == Lifecycle.PAUSE
    assert len(r[3].state.observations) == 2 and r[3].state.calendar == 4
    r = run(LIVE[:7]+[float('nan')]+LIVE[7:])
    assert r[8].state.lifecycle == Lifecycle.REDUCE and len(r[8].state.observations) == 8
    assert r[11].state.lifecycle == Lifecycle.REDUCE and r[12].state.lifecycle == Lifecycle.PAUSE
    assert run([.05,float('nan'),.05,float('nan'),.05,.05,float('nan')])[-1].state.lifecycle == Lifecycle.PAUSE


def test_due_optional_sentinel_blocks_and_holds():
    state = MonitorState(cusum=2.)
    for cost in (NotApplicable(), Missing(), Value(float('nan'))):
        r = evaluate(state, monthly_inputs(Value(.05), cost=cost), monthly_schedule(cost_due=True))
        assert r.permission == Permission.BLOCK and r.target is None
        assert r.state.cusum == 2. and not r.state.observations


@pytest.mark.parametrize('kwargs', [dict(dd=[0,float('nan')]), dict(costs=[None,float('nan')]),
    dict(broken=[False,'yes']), dict(policy=replace(POLICY, psi=1.3)),
    dict(state=MonitorState(cusum=float('nan'))), dict(state=MonitorState(weight=float('nan')))])
def test_corrupt_parameters_and_inputs(kwargs):
    assert run([.05,.05], **kwargs)[-1].permission == Permission.BLOCK


def test_operational_triggers_and_hysteresis():
    assert run([.05]*3, dd=[0,.15,.22])[-1].state.lifecycle == Lifecycle.PAUSE
    assert run([.05]*3, costs=[None,1.2,1.6])[-1].state.lifecycle == Lifecycle.PAUSE
    for kwargs in [dict(capacity=[10,.5,10]), dict(broken=[False,True,False])]:
        r = run([.05]*3, **kwargs)
        assert r[1].state.lifecycle == Lifecycle.PAUSE and 'RETIRE' in r[1].proposals
        assert r[2].state.lifecycle == Lifecycle.PAUSE
    r = run([.05]*8, crowds=[None,None,2.5,2.5,1.,1.,1.,1.])
    assert r[5].state.lifecycle == Lifecycle.REDUCE
    assert r[6].state.lifecycle == Lifecycle.KEEP and r[6].target == .75 and r[7].target == 1.
    assert all(x.state.lifecycle == Lifecycle.REDUCE for x in run([.05]*8, crowds=[None,None,2.5,1.8,1.8,1.8,1.8,1.8])[2:])
    r = run([.05]*8, crowds=[None,None,2.5,1.,float('nan'),1.,1.,1.])
    assert r[6].state.lifecycle == Lifecycle.REDUCE and r[7].state.lifecycle == Lifecycle.KEEP


def test_halt_and_scheduled_gaps():
    r = run([.05]*3, halts=[False,True,False])
    assert r[1].permission == Permission.HALT and r[1].target is None
    assert len(r[2].state.observations) == 3
    r = run([.05,NoObservationExpected(),.05,NoObservationExpected()])
    assert len(r[-1].state.observations) == 2 and r[-1].permission == Permission.OK
    assert run([.05,NoObservationExpected()], dd=[0,.15])[-1].state.lifecycle == Lifecycle.REDUCE
    r = run([.05]*4, retire_at=3)
    assert all(x.state.lifecycle == Lifecycle.RETIRED and x.permission == Permission.BLOCK and x.state.weight == 0 for x in r[2:])


def test_reentry_requires_approval_and_resets_ramp():
    r = run([.05]*9, costs=[None,1.2,1.6]+[1.]*6)
    assert r[7].proposals == () and r[8].proposals == ('REENTRY',)
    assert r[8].target == 0
    result = evaluate(r[-1].state, monthly_inputs(Value(.05)), monthly_schedule(), POLICY, approve_reentry=True)
    assert result.state.lifecycle == Lifecycle.RAMP and result.target == POLICY.first_ramp
    assert result.state.ramp_cap == POLICY.first_ramp
    early = run([.05]*5, state=MonitorState(weight=.1, ramp_cap=.1))
    assert all(x.target <= .1 for x in early)


def test_retirement_decision_absorbing():
    state = MonitorState(lifecycle=Lifecycle.PAUSE, paused=11, weight=0.)
    r = evaluate(state, monthly_inputs(Value(-.1)), monthly_schedule(), approve_retirement=True)
    assert r.state.lifecycle == Lifecycle.RETIRED
    assert evaluate(r.state, {}, monthly_schedule()).permission == Permission.BLOCK


@pytest.mark.parametrize('bad', [Value(True), Value([.1,.2]), Value('0.1')])
def test_non_scalar_ic_blocks(bad):
    r = evaluate(MonitorState(), monthly_inputs(bad), monthly_schedule())
    assert r.permission == Permission.BLOCK and not r.state.observations


def test_old_failures_expire_and_gap_holds_shadow():
    r = run([float('nan')]+[.05]*12+[float('nan'),.05,float('nan')])
    assert r[-1].state.lifecycle != Lifecycle.PAUSE
    state = MonitorState(lifecycle=Lifecycle.PAUSE, shadow=3, weight=0.)
    gap = evaluate(state, monthly_inputs(NoObservationExpected()), monthly_schedule(ic=Expectation.NO_OBSERVATION_EXPECTED))
    assert gap.state.shadow == 3 and gap.state.paused == 1
    bad = evaluate(gap.state, monthly_inputs(Missing()), monthly_schedule())
    assert bad.state.shadow == 0
