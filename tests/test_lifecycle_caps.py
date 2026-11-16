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
    with pytest.raises(ValueError):
        check_exposure_cap({'CNY':470}, {'CNY':20}, 1000)
    assert check_exposure_cap({'CNY':470}, {'CNY':-30}, 1000)['CNY'] == -30
    with pytest.raises(ValueError):
        check_exposure_cap({'CNY':470}, {'CNY':-20}, 1000, lot_values={'CNY':12})
