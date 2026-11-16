from dataclasses import replace
import numpy as np
import pytest
from quant_lab.lifecycle.sim_returns import *
from quant_lab.lifecycle.monitor import MonitorState
from quant_lab.lifecycle.states import Lifecycle

def run(cost=.001,states=None):
    return simulate_returns([.03,.02],[[.01,.003],[.003,.02]],states or [MonitorState(weight=.4,ramp_cap=.4)]*2,months=24,cost_rate=cost,annual_birth_rate=12)

def test_units_are_annual_and_net():
    r=run()
    assert r.annual_net_active_return == pytest.approx(r.net.mean()*12)
    assert r.net_active_ir == pytest.approx(r.net.mean()/r.net.std(ddof=1)*np.sqrt(12))
    assert sum(r.births)<60  # rate 12/year, not 12/month

def test_paused_sleeves_not_counted_live():
    r=run(states=[MonitorState(lifecycle=Lifecycle.PAUSE),MonitorState(weight=.4,ramp_cap=.4)])
    assert set(r.live_counts)=={1}

def test_costs_reduce_pnl():
    r=run(); assert r.net.sum()<r.gross.sum() and r.costs.sum()>0
