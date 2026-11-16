"""Append-only lifecycle journal; hash chain and deterministic replay."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from .states import to_json, from_json

@dataclass(frozen=True)
class TransitionEvent:
    month: int
    before: str
    after: str
    evidence: str
    previous_hash: str
    event_hash: str

    def digest(self):
        body=asdict(self); body.pop('event_hash')
        return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def replay(events):
    previous='0'*64; state=None; month=None
    for e in events:
        if e.previous_hash!=previous or e.event_hash!=e.digest() or (month is not None and e.month!=month+1):
            raise ValueError('tampered chain or duplicate month')
        before=from_json(e.before); after=from_json(e.after)
        if state is not None and before!=state or after.calendar!=e.month or before.calendar+1!=e.month:
            raise ValueError('discontinuous state chain')
        previous=e.event_hash; state=after; month=e.month
    return state

class EventStore:
    def __init__(self,path):
        self.path=Path(path)
        self.events=tuple(TransitionEvent(**json.loads(line)) for line in self.path.read_text().splitlines()) if self.path.exists() else ()
        replay(self.events)

    def append(self,before,result,evidence,*,authorization=None,approval_evidence=None):
        from dataclasses import replace
        from .states import Lifecycle
        from .approval import require_human_approval
        needs_human=(result.state.lifecycle==Lifecycle.RETIRED and before.lifecycle!=Lifecycle.RETIRED) or (before.lifecycle==Lifecycle.PAUSE and result.state.lifecycle==Lifecycle.RAMP)
        if needs_human:
            require_human_approval(evidence.get('approval_id'),authorization,before,approval_evidence)
        e=TransitionEvent(result.state.calendar,to_json(before),to_json(result.state),to_json(evidence),self.events[-1].event_hash if self.events else '0'*64,'')
        e=replace(e,event_hash=e.digest())
        replay((*self.events,e))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a+') as f:
            import fcntl
            import os
            fcntl.flock(f,fcntl.LOCK_EX)
            f.seek(0)
            current=tuple(TransitionEvent(**json.loads(line)) for line in f.read().splitlines())
            if current!=self.events: raise ValueError('journal changed externally')
            f.write(json.dumps(asdict(e),sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())
        self.events=(*self.events,e)

    def evaluate(self,state,inputs,calendar,policy=None,**kwargs):
        from .monitor import evaluate, Policy
        return evaluate(state,inputs,calendar,policy or Policy(),event_store=self,**kwargs)

class LifecycleRunner:
    """Single-sleeve operational entrypoint; evaluations are always journalled."""
    def __init__(self,store,initial_state):
        self.store=store
        self.state=replay(store.events) or initial_state

    def evaluate(self,inputs,calendar,policy=None,*,ramp=None,**kwargs):
        from .monitor import _evaluate, Policy
        from .ramp import advance_ramp
        before=self.state
        effective=advance_ramp(before,*ramp) if ramp is not None else before
        result=_evaluate(effective,inputs,calendar,policy or Policy(),**kwargs)
        evidence=dict(inputs=dict(inputs),calendar=calendar,policy=policy or Policy(),
                      permission=result.permission,target=result.target,proposals=result.proposals,
                      approval_id=kwargs.get('approval_id'))
        if ramp is not None:
            from .approval import ramp_evidence
            evidence['ramp']=ramp_evidence(*ramp)
            evidence['ramp_approval_id']=ramp[1].approval_id
        from .approval import monitor_evidence
        auth=kwargs.get('authorization')
        if auth is not None: evidence['human_record']=asdict(auth.record)
        if ramp is not None: evidence['ramp_human_record']=asdict(ramp[1].authorization.record)
        approval=monitor_evidence(inputs,calendar,policy or Policy(),kwargs.get('approve_reentry',False),kwargs.get('approve_retirement',False),kwargs.get('ramp_plan'))
        self.store.append(before,result,evidence,authorization=auth,approval_evidence=approval)
        self.state=result.state
        return result
