import numpy as np
import pytest
from quant_lab.paper.ramp_cost import *

def test_ramp_bound_uses_mean_not_median():
    obs=[unit_impact(.02,4950,adv) for adv in [3e5]*2+[3e6]*18]
    nxt=[x*np.sqrt(2) for x in obs]
    b=next_stage_bound(Bound(.001,30),nxt,[1]*20,obs,[1]*20,2)
    assert b.value-.001 == pytest.approx(.000818542,rel=1e-5)

def test_weighting_matches_shortfall_convention():
    assert mean_unit_impact([1,3],[3,1])['mean']==1.5

def test_margin_0_142_at_30_days():
    margin=upper_margin(.005*np.sqrt(1.3/2),30,1.266667)
    assert margin == pytest.approx(.00142222,abs=.000001)
    # Raw day means use the explicit NW lag-2 estimator, not a fixed SE.
    b=robust_upper_bound(np.linspace(0,.01,30))
    assert b.margin>0 and b.value==pytest.approx(.005+b.margin)

def test_transfer_is_bucket_weighted():
    assert transfer_bound({'a':Bound(.01,10),'b':Bound(.02,20)},{'a':.25,'b':.75}).value==pytest.approx(.0175)

def test_thin_bucket_is_not_evaluable():
    assert transfer_bound({'a':Bound(.01,9)},{'a':1})==NOT_EVALUABLE

def test_all_buckets_empty_is_not_evaluable():
    assert transfer_bound({},{'a':1})==NOT_EVALUABLE

def test_stress_scenario_cannot_pass_cost_test():
    s=stress_scenario(.01,2)
    assert not cost_test(s,1) and transfer_bound({'a':s},{'a':1})==NOT_EVALUABLE

def test_negative_autocorrelation_never_shrinks():
    from scipy.stats import t
    x=np.array([0.,1.]*15)
    assert robust_upper_bound(x).margin >= t.ppf(.95,29)*x.std(ddof=1)/np.sqrt(30)-1e-12


def test_stress_assumption_rejected_by_ramp_evidence(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_lifecycle_ramp import RampPlan, EVIDENCE
    from quant_lab.lifecycle.ramp import advance_ramp, ramp_evidence
    from quant_lab.lifecycle.monitor import MonitorState
    from lifecycle_approval_helpers import authorize
    state=MonitorState(); plan=RampPlan()
    evidence=replace(EVIDENCE,cost_test_passed=stress_scenario(.01,2))
    auth=authorize(state,ramp_evidence(plan,evidence),tmp_path,monkeypatch)
    with pytest.raises(ValueError): advance_ramp(state,plan,replace(evidence,authorization=auth))
