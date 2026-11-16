import pytest
from quant_lab.lifecycle.caps import *


def test_composition():
    assert compose_weight(1,.1,.8,Permission.OK) == .1
    for permission in (Permission.BLOCK, Permission.HALT):
        assert compose_weight(1,1,1,permission) == 0
    with pytest.raises(ValueError):
        compose_weight(float('nan'),1,1,Permission.OK)


def test_actual_drift_and_lots():
    trades = check_exposure_cap({'CNY': 440, 'MYR': 100}, {'CNY': 50,'MYR': 20}, 1000, lot_values={'CNY': 6})
    assert trades == {'CNY':6, 'MYR':20}
    assert 440+trades['CNY'] <= 450
    assert check_exposure_cap({'CNY':470}, {'CNY':20}, 1000)['CNY'] == 0
    assert check_exposure_cap({'CNY':470}, {'CNY':-30}, 1000)['CNY'] == -30
    assert check_exposure_cap({'CNY':470}, {'CNY':-20}, 1000, lot_values={'CNY':12})['CNY'] == -20


def position(value, classification='CNY_ECONOMIC', market='HK', lot=6):
    return Position('x',market,value,lot,classification)

def test_partial_sell_allowed_when_over_cap():
    r=check_exposure([position(470)],[],[position(-6)],1000)
    assert r.approved_trades[0].value_nav_ccy == -6 and r.breach

def test_drift_breach_reports_required_sell():
    r=check_exposure([position(470)],[],[],1000)
    assert r.required_sell_value['CNY_ECONOMIC'] == 20

def test_buy_clipped_in_lots_or_dropped_below_ticket():
    assert check_exposure([position(440)],[],[position(50)],1000,min_ticket=5).approved_trades[0].value_nav_ccy == 6
    assert check_exposure([position(440)],[],[position(50)],1000,min_ticket=7).approved_trades[0].value_nav_ccy == 0

def test_unknown_classification_counts_as_cny():
    assert check_exposure([position(470,None)],[],[],1000).breach

def test_open_orders_count():
    assert check_exposure([position(420)],[position(24)],[position(50)],1000).approved_trades[0].value_nav_ccy == 6
    assert check_exposure([position(450)],[position(-60)],[position(50)],1000).approved_trades[0].value_nav_ccy == 0

def test_foreign_cap_60():
    r=check_exposure([position(594,'NON_CNY')],[],[position(50,'NON_CNY')],1000)
    assert r.approved_trades[0].value_nav_ccy == 6
