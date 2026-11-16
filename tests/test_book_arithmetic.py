import pytest
from quant_lab.portfolio.book_risk import *

def test_active_vol_5_15_ir_0_29():
    vol=book_active_risk([.4,.35,.25],[.07]*3,.3)
    assert vol == pytest.approx(.05151,abs=.00001)
    assert .015/vol == pytest.approx(.291,abs=.001)

def test_uncorrelated_4_11_ir_0_365():
    vol=book_active_risk([.4,.35,.25],[.07]*3,0.)
    assert vol == pytest.approx(.04111,abs=.00001)
    assert .015/vol == pytest.approx(.365,abs=.001)

def test_hk_constrained_range():
    assert net_ir_range(1.8,4.05,.6075,.07,7) == pytest.approx((.16,.48),abs=.002)

def test_non_psd_correlation_rejected():
    with pytest.raises(ValueError): book_active_risk([.4,.35,.25],[.07]*3,-.9)
