"""Canonical decay example: PYTHONPATH=src python examples/decay_example.py."""
import math
from dataclasses import replace
from quant_lab.lifecycle import Policy, MonitorState, Lifecycle, Value, evaluate
from quant_lab.lifecycle.states import ScheduleCalendar
from quant_lab.lifecycle.monitor import monthly_inputs

class Approx:
    def __init__(self, value, abs=1e-8): self.value, self.tol = value, abs
    def __eq__(self, actual): return math.isclose(actual,self.value,abs_tol=self.tol)
def approx(value, abs=1e-8): return Approx(value,abs)

LIVE = [.05,-.08,.02,-.11,.01,-.06,.04,-.13,-.02,-.09,.03,-.10]
IMPACT = 2*.60*.023*math.sqrt(31250/300e6)
BREAKEVEN = (.60*.0030+IMPACT)/(.10*.27996/.2)
POLICY = Policy(breakeven=BREAKEVEN)


def run(xs, *, policy=POLICY):
    state = MonitorState(weight=1., ramp_cap=1.)
    calendar = ScheduleCalendar()
    rows = []
    for x in xs:
        result = evaluate(state, monthly_inputs(Value(x)), calendar, policy)
        state = result.state
        rows.append(result)
    return rows

def main():
    assert IMPACT*1200 == approx(.338, abs=.002)
    assert BREAKEVEN == approx(.01487, abs=2e-5)
    r = run(LIVE)
    for i, ratio, pnet, life, weight in [(5,.795,.481,Lifecycle.KEEP,.795), (7,.710,.427,Lifecycle.REDUCE,.5), (11,.553,.328,Lifecycle.PAUSE,0.)]:
        assert r[i].posterior.mean/POLICY.mu_p == approx(ratio, abs=.002)
        assert r[i].posterior.p_net == approx(pnet, abs=.003)
        assert r[i].state.lifecycle == life
        assert r[i].state.weight == approx(weight, abs=.002)
    assert r[7].state.cusum == approx(3.31)
    assert r[11].state.cusum == approx(5.23)
    assert r[11].posterior.p_positive == approx(.815, abs=.003)
    assert r[11].proposals == ()
    serial = run(LIVE, policy=replace(POLICY, psi=.74))
    assert next(r.state.calendar for r in serial if r.state.lifecycle == Lifecycle.REDUCE) == 10
    assert serial[-1].posterior.p_net == approx(.393, abs=.005)
    assert serial[-1].state.lifecycle == Lifecycle.PAUSE


if __name__ == "__main__":
    main()
    print("canonical decay assertions passed")
