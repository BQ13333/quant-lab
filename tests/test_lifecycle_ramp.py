from dataclasses import replace
import numpy as np
import pytest
from quant_lab.lifecycle.ramp import *
from quant_lab.lifecycle.monitor import *
from quant_lab.lifecycle.states import *

EVIDENCE = StageEvidence(8,2,12,5,True,'APR-001')

def test_target_never_exceeds_ramp_cap_property():
    rng=np.random.default_rng(4)
    for cap in (.1,.25,.5,1.):
        state=MonitorState(ramp_cap=cap)
        for x in rng.normal(.02,.1,40):
            r=evaluate(state,monthly_inputs(Value(float(x))),ScheduleCalendar())
            state=r.state
            assert r.target is None or r.target <= cap

def test_advance_requires_evidence_single_step(tmp_path, monkeypatch):
    from lifecycle_approval_helpers import authorize
    state=MonitorState(); plan=RampPlan()
    authorization=authorize(state,ramp_evidence(plan,EVIDENCE),tmp_path,monkeypatch)
    evidence=replace(EVIDENCE,authorization=authorization)
    advanced=advance_ramp(state,plan,evidence)
    assert advanced.ramp_cap == .25
    with pytest.raises(ValueError,match="approval"): advance_ramp(advanced,RampPlan(),evidence)
    for i,bad in enumerate((replace(EVIDENCE,weeks=7),replace(EVIDENCE,cost_test_passed=False),replace(EVIDENCE,fill_days=4))):
        bad=replace(bad,approval_id=f'APR-BAD-{i}')
        signed=authorize(state,ramp_evidence(plan,bad),tmp_path,monkeypatch,bad.approval_id)
        with pytest.raises(ValueError): advance_ramp(state,plan,replace(bad,authorization=signed))

def test_no_advance_in_reduce_or_pause(tmp_path, monkeypatch):
    from lifecycle_approval_helpers import authorize
    for life in (Lifecycle.REDUCE,Lifecycle.PAUSE):
        state=MonitorState(lifecycle=life); plan=RampPlan()
        evidence=replace(EVIDENCE,approval_id='APR-'+life.value)
        signed=authorize(state,ramp_evidence(plan,evidence),tmp_path,monkeypatch,evidence.approval_id)
        with pytest.raises(ValueError,match='lifecycle'): advance_ramp(state,plan,replace(evidence,authorization=signed))


def test_reentry_resets_to_first_feasible_stage(tmp_path, monkeypatch):
    state=MonitorState(lifecycle=Lifecycle.PAUSE,shadow=6,observations=(.1,)*20,weight=0,ramp_cap=1.)
    from lifecycle_approval_helpers import authorize
    from quant_lab.lifecycle.approval import monitor_evidence
    inputs=monthly_inputs(Value(.1)); calendar=ScheduleCalendar(); plan=RampPlan(first_feasible=.25)
    authorization=authorize(state,monitor_evidence(inputs,calendar,Policy(),True,False,plan),tmp_path,monkeypatch)
    r=evaluate(state,inputs,calendar,approve_reentry=True,approval_id='APR-001',ramp_plan=plan,authorization=authorization)
    assert r.state.lifecycle == Lifecycle.RAMP and r.state.ramp_cap == .25
    assert evaluate(state,monthly_inputs(Value(.1)),ScheduleCalendar(),approve_reentry=True,approval_id='model:x').permission == Permission.BLOCK


def test_model_or_unsigned_record_cannot_approve(tmp_path, monkeypatch):
    from lifecycle_approval_helpers import authorize
    from quant_lab.lifecycle.approval import Authorization
    state=MonitorState(); plan=RampPlan()
    with pytest.raises(ValueError,match='signed human'):
        advance_ramp(state,plan,replace(EVIDENCE,approval_id='human:invented'))
    auth=authorize(state,ramp_evidence(plan,EVIDENCE),tmp_path,monkeypatch)
    for record in (replace(auth.record,approved_by='model:astra'),replace(auth.record,signature='AA==')):
        with pytest.raises(ValueError):
            advance_ramp(state,plan,replace(EVIDENCE,authorization=Authorization(record,auth.trust_store)))
