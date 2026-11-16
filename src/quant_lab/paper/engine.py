"""Paper execution: stage feasibility -> exposure -> executable reservation."""
from dataclasses import replace
from .shortfall import Fill, OrderActivityLog
from quant_lab.lifecycle.inventory import check_executable
from quant_lab.lifecycle.caps import check_exposure
from quant_lab.lifecycle.feasibility import StageReport

class PaperEngine:
    def __init__(self, inventory, stage_report):
        if type(stage_report) is not StageReport or not stage_report.passed:
            raise ValueError('passing current StageReport required')
        self.inventory=inventory
        self.stage_report=stage_report
        self.activity=OrderActivityLog()
        self.open={}
        self.fills=[]
        self.initial={k:v.owned for k,v in inventory.items()}
        self._next_id=0
        self.daily_quantity={}

    def submit(self, order, position, *, holdings, open_orders, nav, market_state, rules, ts):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        if not isinstance(ts,datetime) or ts.tzinfo is None:
            raise ValueError('timezone-aware order time required')
        key=(order.instrument, ts.astimezone(ZoneInfo('Asia/Shanghai')).date())
        if order.instrument != position.instrument or order.market != position.market or (position.value_nav_ccy > 0) != (order.side == 'BUY') or position.value_nav_ccy == 0:
            raise ValueError('exposure/executable identity mismatch')
        if order.side=='BUY' and order.instrument not in dict(self.stage_report.quantities):
            raise ValueError('instrument outside stage construction')
        # Include internally reserved buys; never credit unfilled sells.
        pending=[p for o,p,remaining in self.open.values() if o.side=='BUY']
        exposure=check_exposure(holdings,[*open_orders,*pending],[position],nav)
        fraction=abs(exposure.approved_trades[0].value_nav_ccy/position.value_nav_ccy)
        if fraction == 0: return None, exposure, None
        clipped=replace(order,qty=order.qty*fraction)
        if order.side=='BUY':
            limits={name:(minimum,maximum,lot) for name,minimum,maximum,lot in self.stage_report.execution_limits}
            if order.instrument not in limits: raise ValueError('missing execution limits')
            minimum,maximum,lot=limits[order.instrument]
            pending_qty=sum(remaining for o,p,remaining in self.open.values() if o.instrument==order.instrument and o.side=='BUY')
            capacity=dict(self.stage_report.quantities)[order.instrument]-self.inventory[order.instrument].owned-pending_qty
            if clipped.qty < minimum or clipped.qty % lot or clipped.qty > capacity or clipped.qty+self.daily_quantity.get(key,0)>maximum:
                raise ValueError('order violates stage lot/ticket/target/daily participation')
        inv=self.inventory[order.instrument]
        executable=check_executable(clipped,inv,market_state,rules)
        if executable.approved_qty == 0: return None,exposure,executable
        self.activity.record(order.market,ts)
        self.daily_quantity[key]=self.daily_quantity.get(key,0)+executable.approved_qty
        self._next_id+=1
        accepted=replace(order,qty=executable.approved_qty)
        self.open[self._next_id]=(accepted,replace(position,value_nav_ccy=position.value_nav_ccy*accepted.qty/order.qty),accepted.qty)
        return self._next_id,exposure,executable

    def fill(self, order_id, qty, px, ts):
        import math
        o,p,remaining=self.open[order_id]
        if not all(math.isfinite(v) and v>0 for v in (qty,px)) or qty>remaining:
            raise ValueError('invalid fill')
        inv=self.inventory[o.instrument]
        if o.side=='SELL':
            inv.owned-=qty; inv.settled_for_sale-=qty; inv.reserved-=qty
        else:
            inv.owned+=qty
            if o.market!='CN': inv.settled_for_sale+=qty
        inv.validate()
        fill=Fill(qty,px,ts,'SIMULATED')
        self.fills.append((o,fill))
        if qty==remaining: del self.open[order_id]
        else: self.open[order_id]=(o,replace(p,value_nav_ccy=p.value_nav_ccy*(remaining-qty)/remaining),remaining-qty)
        return fill

    def cancel(self, order_id):
        o,p,remaining=self.open.pop(order_id)
        if o.side=='SELL': self.inventory[o.instrument].reserved-=remaining

    def settle_next_session(self):
        for inv in self.inventory.values(): inv.settled_for_sale=inv.owned
