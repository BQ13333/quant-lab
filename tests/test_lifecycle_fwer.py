import pytest
from quant_lab.lifecycle.fwer import *


def test_independent_validation_interval():
    report = calibrate(Policy(), list(range(10)), list(range(100,120)), months=18, family_size=2)
    assert report.validation.replicates == 20
    lo, hi = report.validation.interval_95
    assert 0 <= lo <= report.validation.rate <= hi <= 1
    assert hi > report.validation.rate
    assert not set(report.calibration_seeds) & set(report.validation_seeds)
    assert binomial_rate(0,20).interval_95[1] == pytest.approx(.1684, abs=.0001)
    with pytest.raises(ValueError):
        calibrate(Policy(), [1,2], [2,3])


def test_false_retire_is_family_event_and_reproducible():
    policy = replace(Policy(), retire_positive=.99, retire_months=1)
    one = false_retire_rate(policy, [11,12,13], months=18, family_size=1)
    assert one.events > 0
    assert one == false_retire_rate(policy, [11,12,13], months=18, family_size=1)
