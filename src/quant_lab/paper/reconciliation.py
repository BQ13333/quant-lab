"""Reconcile filled shares to held shares; intended remainder is outstanding."""
def reconcile(engine):
    expected=dict(engine.initial)
    for order,fill in engine.fills:
        expected[order.instrument]=expected.get(order.instrument,0)+(fill.qty if order.side=='BUY' else -fill.qty)
    differences={name:inv.owned-expected.get(name,0) for name,inv in engine.inventory.items()}
    if any(abs(x)>1e-9 for x in differences.values()):
        raise ValueError('share reconciliation failed')
    return differences
