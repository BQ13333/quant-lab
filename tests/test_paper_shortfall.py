from datetime import datetime, timezone, timedelta
import pytest
from quant_lab.paper.shortfall import *
from quant_lab.paper.pnl import *
T=datetime(2026,1,1,tzinfo=timezone.utc)
def example(side=1,ref='DECISION'):
    o=ParentOrder(side,100,10,11,(T.date(),),12)
    fills=[Fill(60,11.5,T,'SIMULATED')]
    return o,fills,implementation_shortfall(o,fills,2,ref)
def test_components_sum_to_total():
    s=example()[2]
    assert s.total == 172 and s.total == s.delay+s.execution+s.opportunity+s.fees

def test_buy_and_sell_signs():
    assert example(-1)[2].ex_fee == -example(1)[2].ex_fee

def test_unfilled_order_carries_opportunity_cost():
    o,_,_=example()
    assert implementation_shortfall(o,[],0).opportunity == 200

@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('ref',['DECISION','ARRIVAL'])
def test_intended_minus_is_total_equals_actual_fill_pnl(side,ref):
    o,f,s=example(side,ref)
    gross=side*o.qty*(o.end_px-(o.decision_px if ref=='DECISION' else o.arrival_px))
    actual=ActualFillBook(sum(side*x.qty*(o.end_px-x.px) for x in f)).net(2)
    assert IntendedBook(ref).net(gross,s)==actual

def test_actual_fill_book_rejects_shortfall_subtraction():
    with pytest.raises(TypeError): ActualFillBook(30).net(fees=2,shortfall=example()[2])

def test_fill_basis_mismatch_raises():
    with pytest.raises(ValueError): IntendedBook('ARRIVAL').net(200,example()[2])

def test_day_weighting_convention():
    from dataclasses import replace
    s=example()[2]
    means=day_means([(1,s),(1,replace(s,notional=2000)),(2,s)])
    assert means[1]==pytest.approx(344/3000) and means[2]==pytest.approx(172/1000)

def test_simulated_fills_rejected_as_evidence():
    with pytest.raises(ValueError): example()[2].require_measured()

def test_paper_engine_logs_order_rate_and_daily_count():
    log=OrderActivityLog(); log.record('CN',T); log.record('CN',T+timedelta(seconds=1))
    assert log.daily_count('CN',T.date())==2
    assert log.order_rate('CN',T,T+timedelta(seconds=2))==1
