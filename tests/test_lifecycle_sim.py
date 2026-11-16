import numpy as np
from quant_lab.lifecycle.sim import simulate


def test_renewal_persists_and_stop_degrades():
    live = simulate(months=100, birth_rate=.4, seed=22)
    stopped = simulate(months=100, birth_rate=.4, stop_births=25, seed=22)
    late = live.path.iloc[-25:]
    old = stopped.path.iloc[-25:]
    assert late.expected_ic.mean() > 2*old.expected_ic.mean()
    assert late.portfolio_ir.mean() > 2*old.portfolio_ir.mean()
    assert late.portfolio_ir.mean() > 1.
    assert np.isfinite(live.path.to_numpy()).all()
    assert (live.path.invested <= 1+1e-8).all()
    assert np.array_equal(live.strategies.to_numpy(), stopped.strategies.to_numpy())
    assert np.all(np.exp(-np.log(2)*100/live.strategies.half_life) < .1)
