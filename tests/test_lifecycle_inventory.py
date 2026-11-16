import pytest
from quant_lab.lifecycle.inventory import *
from quant_lab.lifecycle.caps import Position, check_exposure

def sell(qty=80): return Order('a', 'CN', 'SELL', qty)
def check(i, qty=80, suspended=False): return check_executable(sell(qty), i, {'suspended': suspended}, {'verified': True})
def test_two_sells_cannot_exceed_holding():
    i=Inventory(100,100)
    assert check(i).approved_qty + check(i).approved_qty == 100

def test_sell_reserves_inventory():
    i=Inventory(100,100,10)
    assert check(i).approved_qty == 80
    assert i.reserved == 90

def test_same_day_a_share_buy_not_sellable():
    assert check(Inventory(100,0)).approved_qty == 0

def test_suspended_instrument_not_executable():
    assert check(Inventory(100,100),suspended=True).approved_qty == 0

def test_exposure_approval_is_not_order_approval():
    p=Position('a','CN',-100,1,'CNY_ECONOMIC')
    assert check_exposure([],[],[p],1000).approved_trades[0] == p
    assert check(Inventory(0,0)).approved_qty == 0
