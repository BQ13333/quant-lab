from dataclasses import replace
import pytest
from quant_lab.lifecycle.events import *
from quant_lab.lifecycle.monitor import MonitorState,monthly_inputs
from quant_lab.lifecycle.states import Value,ScheduleCalendar,Permission

def test_replay_reproduces_state(tmp_path):
    store=EventStore(tmp_path/'events.jsonl'); state=MonitorState()
    for _ in range(4): state=store.evaluate(state,monthly_inputs(Value(.03)),ScheduleCalendar()).state
    assert replay(EventStore(store.path).events)==state

def test_duplicate_month_rejected(tmp_path):
    store=EventStore(tmp_path/'e'); state=MonitorState()
    store.evaluate(state,monthly_inputs(Value(.03)),ScheduleCalendar())
    with pytest.raises(ValueError): store.evaluate(state,monthly_inputs(Value(.03)),ScheduleCalendar())

def test_tampered_chain_detected(tmp_path):
    store=EventStore(tmp_path/'e'); store.evaluate(MonitorState(),monthly_inputs(Value(.03)),ScheduleCalendar())
    with pytest.raises(ValueError): replay([replace(store.events[0],evidence='tampered')])

def test_model_cannot_approve(tmp_path):
    store=EventStore(tmp_path/'e')
    r=store.evaluate(MonitorState(),monthly_inputs(Value(.03)),ScheduleCalendar(),approve_retirement=True,approval_id='model:astra')
    assert r.permission==Permission.BLOCK
    assert replay(store.events)==r.state


def test_direct_unsigned_retirement_append_rejected(tmp_path):
    from quant_lab.lifecycle.monitor import Evaluation
    from quant_lab.lifecycle.states import Lifecycle
    store=EventStore(tmp_path/'e'); state=MonitorState()
    forged=Evaluation(replace(state,calendar=1,lifecycle=Lifecycle.RETIRED),Permission.BLOCK,None)
    with pytest.raises(ValueError,match='signed human'): store.append(state,forged,{'approval_id':'model'})

def test_runner_persists_blocked_and_successful_results(tmp_path):
    from quant_lab.lifecycle.states import Missing
    runner=LifecycleRunner(EventStore(tmp_path/'e'),MonitorState())
    runner.evaluate(monthly_inputs(Value(.03)),ScheduleCalendar())
    runner.evaluate(monthly_inputs(Missing()),ScheduleCalendar())
    assert len(runner.store.events)==2 and replay(runner.store.events)==runner.state


def test_signed_human_retirement_is_persisted(tmp_path,monkeypatch):
    from quant_lab.lifecycle.approval import monitor_evidence
    from quant_lab.lifecycle.monitor import Policy
    from quant_lab.lifecycle.states import Lifecycle
    from lifecycle_approval_helpers import authorize
    state=MonitorState(); calendar=ScheduleCalendar(); policy=Policy()
    inputs=monthly_inputs(Value(.03),mechanism_broken=True)
    proof=monitor_evidence(inputs,calendar,policy,False,True,None)
    auth=authorize(state,proof,tmp_path,monkeypatch)
    store=EventStore(tmp_path/'e')
    result=store.evaluate(state,inputs,calendar,policy,approve_retirement=True,approval_id='APR-001',authorization=auth)
    assert replay(store.events).lifecycle==Lifecycle.RETIRED
