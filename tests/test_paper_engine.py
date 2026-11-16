from datetime import datetime,timezone,timedelta
from quant_lab.paper.engine import PaperEngine
from quant_lab.paper.reconciliation import reconcile
from quant_lab.lifecycle.inventory import Inventory,Order
from quant_lab.lifecycle.caps import Position
from quant_lab.lifecycle.feasibility import *

def engine():
    report=stage_feasibility(StageConstruction(.1,('a',),(1100,)),{'a':Instrument(10,1,10,1e7,1)},1e6,1e6,0,StageLimits(.1,.05,.001))
    return PaperEngine({'a':Inventory(100,100)},report)

def submit(e,side,qty,ts):
    sign=1 if side=='BUY' else -1
    return e.submit(Order('a','CN',side,qty),Position('a','CN',sign*qty*10,10,'CNY_ECONOMIC'),holdings=[Position('a','CN',e.inventory['a'].owned*10,10,'CNY_ECONOMIC')],open_orders=[],nav=1e6,market_state={'suspended':False},rules={'verified':True},ts=ts)

def test_paper_engine_calls_both_checks():
    e=engine(); t=datetime(2026,1,1,tzinfo=timezone.utc)
    _,ex,executable=submit(e,'SELL',150,t)
    assert ex.approved_trades[0].value_nav_ccy==-1500 and executable.approved_qty==100
    assert submit(e,'SELL',1,t)[0] is None

def test_60_session_book_reconciles():
    e=engine(); t=datetime(2026,1,1,tzinfo=timezone.utc)
    for day in range(60):
        e.settle_next_session(); ts=t+timedelta(days=day)
        order_id,_,_=submit(e,'BUY' if day%2==0 else 'SELL',10,ts)
        e.fill(order_id,6,10,ts); e.cancel(order_id)
        assert reconcile(e)=={'a':0}
    assert all(f.source=='SIMULATED' for _,f in e.fills)
