import math
import pytest
from quant_lab.lifecycle.renewal import *


def test_renewal_numeric():
    assert live_strategies(2,3) == 6
    assert survival(3,3) == .5
    assert family_survival(3, {'technical':3}) == {'technical':.5}
    assert portfolio_ir(4,.5,0) == 1
    assert portfolio_ir(4,.5,1) == .5
    rate = required_birth_rate(1,.5,.1,3)
    assert portfolio_ir(live_strategies(rate,3), .5,.1) == pytest.approx(1)
    assert ideas_needed(2,.1) == pytest.approx(20)
    assert ideas_needed(2,.1,100,.1) > 20
    assert expected_maximum(100) == pytest.approx(2.53, abs=.02)
    assert worked_example()['live'] == pytest.approx(6/math.log(2))
    with pytest.raises(ValueError):
        required_birth_rate(2,.5,.1,3)
