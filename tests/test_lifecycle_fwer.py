import pytest
from quant_lab.lifecycle.fwer import *
from quant_lab.lifecycle.fwer import calibrate as _calibrate
def calibrate(*args, **kwargs):
    return _calibrate(*args, seed_registry=SeedRegistry(), **kwargs)


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


def test_pass_requires_validation_upper_bound():
    r=calibrate(Policy(),[1],[2,3],months=1,family_size=1,min_validation=1)
    assert r.target_met_on_calibration and not r.passed
    assert replace(r,validation=binomial_rate(0,2000)).passed

def test_min_replicates_enforced():
    r=calibrate(Policy(),[1],[2],months=1,family_size=1)
    assert not replace(r,validation=binomial_rate(0,100)).passed

def test_zero_events_needs_72():
    assert binomial_rate(0,71).interval_95[1] > .05
    assert binomial_rate(0,72).interval_95[1] <= .05

def test_not_calibrated_is_explicit():
    r=calibrate(replace(Policy(),retire_positive=.999),[1,2],[3],months=15,family_size=2,multipliers=[1.])
    assert not r.target_met_on_calibration and r.status == 'NOT_CALIBRATED'
    assert '15-month' in r.report() and 'family of 2' in r.report()

def test_block_bootstrap_null_supported():
    null=BlockBootstrap((-.1,0.,.03,.07,.12,.02),block=2)
    a=false_event_rate(Policy(),[1,2],months=12,family_size=2,null=null)
    assert a==false_event_rate(Policy(),[1,2],months=12,family_size=2,null=null)

def test_false_pause_rate_reported():
    r=calibrate(Policy(),[1],[2,3],months=12,family_size=1)
    assert r.false_pause.replicates == 2 and 'False-pause rate' in r.report()


def test_family_and_per_sleeve_pause_rates_have_separate_denominators():
    a,b=pause_rates(Policy(),[1,2],months=2,family_size=3)
    assert a.replicates == 2 and b.replicates == 6

def test_report_states_null_scope():
    r=calibrate(Policy(),[1],[2],months=1,family_size=1)
    assert 'IC-only null' in r.report() and 'operational triggers' in r.report()

def test_validation_seeds_cannot_be_reused_after_retune():
    registry=SeedRegistry()
    registry.consume([1,2],Policy())
    with pytest.raises(ValueError): registry.consume([2,3],replace(Policy(),pause_h=9))


def test_seed_registry_survives_restart(tmp_path):
    path=tmp_path/'seeds.json'
    SeedRegistry(path).consume([1],Policy())
    with pytest.raises(ValueError): SeedRegistry(path).consume([1],replace(Policy(),pause_h=9))


def test_validation_minimum_and_five_percent_cannot_be_loosened():
    r=calibrate(Policy(),[1],[2],months=1,family_size=1)
    assert not replace(r,min_validation=1,validation=binomial_rate(0,100)).passed
    assert not replace(r,target=.9,validation=binomial_rate(81,2000)).passed
    assert replace(r,validation=binomial_rate(80,2000)).passed
