import pytest
from quant_lab.lifecycle.feasibility import *

def test_pilot_funding_41250_68062_74250():
    assert [pilot_funding(t) for t in (5000,8250,9000)] == pytest.approx([41250,68062.5,74250])

def test_three_sleeve_floor_is_1_485m():
    assert min_nav([.4,.35,.25],[5000,8250,9000]).minimum_nav == pytest.approx(1485000)

def test_two_sleeve_floor_is_756_250():
    assert min_nav([.55,.45],[5000,8250]).minimum_nav == pytest.approx(756250)

def test_bursa_only_200k_infeasible():
    with pytest.raises(ValueError,match='INFEASIBLE'): min_nav([1.],[5000]).require_nav(200000)
    with pytest.raises(ValueError): child_orders(10000,200000,Limits())

def test_bursa_only_floor_312_500_with_three_children():
    report=min_nav([1.],[5000]); assert report.minimum_nav == 312500
    assert len(child_orders(312500/20,312500,Limits())) == 3

def test_child_orders_respect_ticket_and_2pct():
    for weights,tickets in [([1.],[5000]),([.55,.45],[5000,8250]),([.4,.35,.25],[5000,8250,9000])]:
        nav=min_nav(weights,tickets).minimum_nav
        for w,t in zip(weights,tickets):
            children=child_orders(w*nav/20,nav,Limits(ticket=t))
            assert sum(children)==pytest.approx(w*nav/20)
            assert all(t-1e-8 <= c <= .02*nav+1e-8 for c in children)

L=StageLimits(.1,.05,.01)
def stages(weight):
    return tuple(StageConstruction(s,tuple(map(str,range(20))),(1500000*weight*s/20,)*20) for s in (.1,.25,.5,1.))
def instruments(ticket,fx=1,price=1):
    return {str(n):Instrument(price,100,ticket,1e8,fx) for n in range(20)}
def test_20_names_at_10pct_infeasible_at_1_5m():
    for w,t,f in ((.4,5000,1),(.35,15000,.55),(.25,15000,.6)):
        assert not stage_feasibility(stages(w)[0],instruments(t,f),1500000,1500000,0,L).passed

def test_first_feasible_stage_25_50_50():
    assert [first_feasible_stage(stages(w),instruments(t,f),1500000,1500000,0,L) for w,t,f in ((.4,5000,1),(.35,15000,.55),(.25,15000,.6))] == [.25,.5,.5]

def test_lot_rounding_can_break_scalar_pass():
    c=StageConstruction(.5,('a',),(15625*.6,))
    assert not stage_feasibility(c,{'a':Instrument(80,100,15000,1e8,.6)},1500000,100000,0,L).passed

def test_scalar_floor_is_labelled_necessary_only():
    assert min_nav([1.],[5000]).label == 'NECESSARY_ONLY'

def test_cash_reserve_counted():
    c=StageConstruction(1.,('a',),(5000,))
    assert not stage_feasibility(c,{'a':Instrument(1,100,5000,1e8,1)},1e6,5050,1,L).passed

def test_ramp_plan_first_stage_must_match():
    from quant_lab.lifecycle.ramp import RampPlan, require_feasible_plan
    reports=tuple(stage_feasibility(c,instruments(5000),1500000,1500000,0,L) for c in stages(.4))
    with pytest.raises(ValueError): require_feasible_plan(RampPlan(stage_reports=reports),.25)
    require_feasible_plan(RampPlan(first_feasible=.25,stage_reports=reports),.25)
