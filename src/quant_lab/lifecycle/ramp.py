"""Evidence-gated registered ramp stages. Approval IDs refer to human records."""
from dataclasses import dataclass, replace
import math
from .states import Lifecycle


from .approval import Authorization, require_human_approval, ramp_evidence


@dataclass(frozen=True)
class RampPlan:
    stages: tuple[float, ...] = (.1, .25, .5, 1.)
    first_feasible: float = .1
    required_rebalances: int = 2
    stage_reports: tuple = ()

    def __post_init__(self):
        if not self.stages or any(not math.isfinite(x) or not 0 < x <= 1 for x in self.stages) or tuple(sorted(set(self.stages))) != self.stages or self.first_feasible not in self.stages or type(self.required_rebalances) is not int or self.required_rebalances < 1:
            raise ValueError('invalid ramp plan')


@dataclass(frozen=True)
class StageEvidence:
    weeks: int
    rebalances: int
    fills: int
    fill_days: int
    cost_test_passed: bool
    approval_id: str
    authorization: Authorization | None = None


def require_feasible_plan(plan, stage):
    from .feasibility import StageReport
    reports = plan.stage_reports
    if not reports or any(type(r) is not StageReport for r in reports) or tuple(r.stage for r in reports) != plan.stages:
        raise ValueError('complete stage feasibility reports required')
    passing = [r.stage for r in reports if r.passed]
    if not passing or plan.first_feasible != min(passing) or stage not in passing:
        raise ValueError('ramp plan first stage must match feasible construction')


def advance_ramp(state, plan, evidence):
    require_human_approval(evidence.approval_id, evidence.authorization, state, ramp_evidence(plan,evidence))
    if evidence.approval_id in state.ramp_approvals:
        raise ValueError('stage evidence already consumed')
    if state.lifecycle not in (Lifecycle.RAMP, Lifecycle.KEEP):
        raise ValueError('lifecycle cannot advance')
    for name, minimum in [('weeks', 8), ('rebalances', plan.required_rebalances), ('fills', 12), ('fill_days', 5)]:
        if type(getattr(evidence, name)) is not int or getattr(evidence, name) < minimum:
            raise ValueError('incomplete stage evidence')
    if evidence.cost_test_passed is not True or evidence.fill_days > evidence.fills:
        raise ValueError('invalid cost/fill evidence')
    if state.ramp_cap not in plan.stages or state.ramp_cap < plan.first_feasible:
        raise ValueError('unregistered stage')
    index = plan.stages.index(state.ramp_cap)
    if index + 1 == len(plan.stages):
        raise ValueError('already at last stage')
    require_feasible_plan(plan, state.ramp_cap)
    require_feasible_plan(plan, plan.stages[index+1])
    return replace(state, ramp_cap=plan.stages[index+1], ramp_approvals=state.ramp_approvals+(evidence.approval_id,))


def start_ramp(state, plan, calibration, approval_id, authorization):
    """PAPER exit requires calibrated policy, feasible construction and a person."""
    from .fwer import Calibration
    if state.lifecycle != Lifecycle.PAPER or type(calibration) is not Calibration or not calibration.passed:
        raise ValueError('no sleeve leaves PAPER while NOT_CALIBRATED')
    require_feasible_plan(plan, plan.first_feasible)
    evidence=dict(action='start_ramp',plan=plan,calibration=calibration)
    require_human_approval(approval_id,authorization,state,evidence)
    return replace(state,lifecycle=Lifecycle.RAMP,ramp_cap=plan.first_feasible,
                   weight=plan.first_feasible,ramp_approvals=state.ramp_approvals+(approval_id,))
