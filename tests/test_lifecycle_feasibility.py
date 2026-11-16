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
