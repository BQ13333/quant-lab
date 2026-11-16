"""Pure monthly state transition. Decisions are explicit, never auto-approved."""
from dataclasses import dataclass, fields, replace
from typing import Mapping
from numbers import Real
from .states import (Lifecycle, Permission, Observed, Value, Schedule, Expectation,
                     NotApplicable, validate_inputs, ScheduleCalendar)
from .decay import finite, normal_posterior, cusum, Posterior
from .caps import compose_weight

@dataclass(frozen=True)
class Policy:
    mu_p: float = .018
    tau: float = .012
    sigma: float = .10
    breakeven: float = .01487
    psi: float = 1.
    k: float = .15
    te: float = .07
    floor: float = 1.
    reduce_h: float = 3.
    pause_h: float = 5.
    reduce_ratio: float = .75
    recover_ratio: float = .90
    recover_h: float = 1.5
    pause_net: float = .50
    hard_pause_net: float = .25
    retire_positive: float = .20
    reentry_net: float = .60
    reduce_n: int = 6
    pause_n: int = 12
    min_effective: float = 3.
    recovery_n: int = 3
    shadow_n: int = 6
    retire_months: int = 12
    gap_consecutive: int = 2
    gap_window: int = 12
    gap_count: int = 3
    step: float = .25
    reduce_cap: float = .5
    first_ramp: float = .1
    crowd_reduce: float = 2.
    crowd_clear: float = 1.5
    cost_pause: float = 1.5
    dd_reduce: float = 2.
    dd_pause: float = 3.
    dd_clear: float = 1.5

    def validate(self) -> None:
        finite(*[getattr(self, f.name) for f in fields(self)])
        if any(getattr(self, f.name) <= 0 for f in fields(self) if f.name != 'breakeven'):
            raise ValueError('policy thresholds must be positive')
        for name in ('psi', 'pause_net', 'hard_pause_net', 'retire_positive', 'reentry_net', 'step', 'reduce_cap', 'first_ramp'):
            if getattr(self, name) > 1:
                raise ValueError('probability/cap exceeds one')
        for name in ('reduce_n', 'pause_n', 'recovery_n', 'shadow_n', 'retire_months', 'gap_consecutive', 'gap_window', 'gap_count'):
            if type(getattr(self, name)) is not int:
                raise ValueError('counts must be integers')

@dataclass(frozen=True)
class MonitorState:
    lifecycle: Lifecycle = Lifecycle.KEEP
    schedule_hash: str = ""
    ramp_approvals: tuple[str, ...] = ()
    calendar: int = 0
    observations: tuple[float, ...] = ()
    cusum: float = 0.
    weight: float = .1
    ramp_cap: float = .1
    failures: tuple[int, ...] = ()
    consecutive_failures: int = 0
    recovery: int = 0
    shadow: int = 0
    paused: int = 0

@dataclass(frozen=True)
class Evaluation:
    state: MonitorState
    permission: Permission
    target: float | None
    posterior: Posterior | None = None
    proposals: tuple[str, ...] = ()


def monthly_inputs(ic: Observed, *, drawdown: float = 0., cost: Observed = NotApplicable(),
                   crowd: Observed = NotApplicable(), capacity: float = 10.,
                   mechanism_broken: bool = False, halt: bool = False) -> dict[str, Observed]:
    """Cost is the registered two-month realised/model ratio, already aggregated."""
    return dict(ic=ic, drawdown=Value(drawdown), cost=cost, crowd=crowd,
                capacity=Value(capacity), mechanism_broken=Value(mechanism_broken), halt=Value(halt))


def monthly_schedule(*, ic: Expectation = Expectation.DUE, cost_due: bool = False,
                     crowd_due: bool = False) -> Schedule:
    return Schedule(tuple((k, e) for k, e in dict(ic=ic, drawdown=Expectation.DUE,
        cost=Expectation.DUE if cost_due else Expectation.NOT_APPLICABLE,
        crowd=Expectation.DUE if crowd_due else Expectation.NOT_APPLICABLE,
        capacity=Expectation.DUE, mechanism_broken=Expectation.DUE, halt=Expectation.DUE).items()))


def evaluate(state: MonitorState, inputs: Mapping[str, Observed], calendar: ScheduleCalendar,
             policy: Policy = Policy(), *, ops_cap: float = 1.,
             approve_reentry: bool = False, approve_retirement: bool = False, approval_id: str | None = None, ramp_plan=None, authorization=None) -> Evaluation:
    """One calendar step; hard safety caps override the soft weight step limit.

    Ramp progress is carried by state and can only be advanced externally after
    G8 evidence. Re-entry resets it to the first registered stage.
    """
    approval_state = state
    t = state.calendar + 1
    halt = inputs.get('halt') == Value(True)
    if state.lifecycle == Lifecycle.RETIRED:
        return Evaluation(replace(state, calendar=t, weight=0.), Permission.BLOCK, None)
    try:
        if not isinstance(calendar, ScheduleCalendar):
            raise ValueError('registered calendar required')
        calendar_hash = calendar.sha256()
        if state.schedule_hash and state.schedule_hash != calendar_hash:
            raise ValueError('calendar mismatch')
        if not state.schedule_hash and state.calendar:
            raise ValueError('unregistered carried state')
        state = replace(state, schedule_hash=calendar_hash)
        schedule = calendar.schedule_for(t)
        if approve_reentry or approve_retirement:
            from .approval import require_human_approval, monitor_evidence
            require_human_approval(approval_id, authorization, approval_state, monitor_evidence(inputs,calendar,policy,approve_reentry,approve_retirement,ramp_plan))
        policy.validate()
        finite(state.cusum, state.weight, state.ramp_cap, state.observations, ops_cap)
        if state.cusum < 0 or not 0 <= state.weight <= 1 or not 0 <= state.ramp_cap <= 1 or not 0 <= ops_cap <= 1:
            raise ValueError('invalid carried state')
        if not isinstance(state.lifecycle, Lifecycle):
            raise ValueError('invalid lifecycle')
        for name in ('calendar', 'consecutive_failures', 'recovery', 'shadow', 'paused'):
            if type(getattr(state, name)) is not int or getattr(state, name) < 0:
                raise ValueError('invalid counter')
        if set(inputs) != set(monthly_inputs(Value(0.))) or validate_inputs(schedule, inputs) != Permission.OK:
            raise ValueError('invalid inputs')
        for name in ('ic', 'drawdown', 'cost', 'crowd', 'capacity'):
            item = inputs[name]
            if isinstance(item, Value) and (not isinstance(item.value, Real) or isinstance(item.value, bool)):
                raise ValueError('metrics must be real scalars')
        if any(type(f) is not int or not 1 <= f <= state.calendar for f in state.failures):
            raise ValueError('invalid failure history')
        for name in ('mechanism_broken', 'halt'):
            if not isinstance(inputs[name], Value) or type(inputs[name].value) is not bool:
                raise ValueError('flags must be bool')
        for name in ('drawdown', 'capacity'):
            if not isinstance(inputs[name], Value) or isinstance(inputs[name].value, bool):
                raise ValueError('required numeric metric')
        d, q = inputs['drawdown'].value, inputs['capacity'].value
        if d < 0 or q <= 0:
            raise ValueError('invalid drawdown/capacity')
        obs = isinstance(inputs['ic'], Value)
        xs = state.observations + ((inputs['ic'].value,) if obs else ())
        posterior = normal_posterior(xs, policy.mu_p, policy.tau, policy.sigma, policy.breakeven, psi=policy.psi)
        s, _ = cusum(state.cusum, inputs['ic'].value if obs else None, policy.mu_p, policy.sigma, policy.k)
        ratio = posterior.mean/policy.mu_p
        finite(ratio)
    except (ValueError, TypeError, OverflowError):
        failures = tuple(f for f in state.failures if f > t-policy.gap_window) + (t,)
        pause = state.consecutive_failures+1 >= policy.gap_consecutive or len(failures) >= policy.gap_count
        life = Lifecycle.PAUSE if pause else state.lifecycle
        failed = replace(state, calendar=t, failures=failures, consecutive_failures=state.consecutive_failures+1,
                         recovery=0, shadow=0, lifecycle=life, weight=0. if pause else state.weight,
                         paused=state.paused+int(life == Lifecycle.PAUSE))
        return Evaluation(failed, Permission.HALT if halt else Permission.BLOCK, None)
    n = len(xs)
    c = inputs['cost'].value if isinstance(inputs['cost'], Value) else None
    z = inputs['crowd'].value if isinstance(inputs['crowd'], Value) else None
    broken = inputs['mechanism_broken'].value
    stat = obs and posterior.n_eff >= policy.min_effective
    pause = (stat and n >= policy.pause_n and ((s > policy.pause_h and posterior.p_net < policy.pause_net) or posterior.p_net < policy.hard_pause_net)) or d > policy.dd_pause*policy.te or (c is not None and c > policy.cost_pause) or broken or q < policy.floor
    reduce = (stat and n >= policy.reduce_n and s > policy.reduce_h and ratio < policy.reduce_ratio) or (z is not None and z > policy.crowd_reduce) or d > policy.dd_reduce*policy.te
    life = state.lifecycle
    if life != Lifecycle.PAUSE:
        life = Lifecycle.PAUSE if pause else Lifecycle.REDUCE if reduce else life
    rec = state.recovery
    if life == state.lifecycle == Lifecycle.REDUCE:
        if obs:
            clear = ratio >= policy.recover_ratio and s < policy.recover_h and not reduce and not pause and (z is None or z <= policy.crowd_clear) and d <= policy.dd_clear*policy.te
            rec = rec+1 if clear else 0
            if rec >= policy.recovery_n:
                life, rec = Lifecycle.KEEP, 0
    else:
        rec = 0
    paused = state.paused+1 if life == Lifecycle.PAUSE else 0
    shadow = state.shadow if life == Lifecycle.PAUSE else 0
    if life == Lifecycle.PAUSE and obs:
        shadow = shadow+1 if posterior.p_net >= policy.reentry_net and not pause else 0
    proposals = []
    if life == Lifecycle.PAUSE and shadow >= policy.shadow_n:
        proposals.append('REENTRY')
    if (paused >= policy.retire_months and shadow < policy.shadow_n) or (stat and n >= policy.pause_n and posterior.p_positive < policy.retire_positive) or broken or q < policy.floor:
        proposals.append('RETIRE')
    ramp = state.ramp_cap
    if approve_reentry and 'REENTRY' in proposals and 'RETIRE' not in proposals:
        life, ramp, paused, shadow = Lifecycle.RAMP, (ramp_plan.first_feasible if ramp_plan else policy.first_ramp), 0, 0
    if approve_retirement and 'RETIRE' in proposals:
        life = Lifecycle.RETIRED
    cap = 0. if life in (Lifecycle.PAUSE, Lifecycle.RETIRED, Lifecycle.CANDIDATE, Lifecycle.VALIDATED, Lifecycle.PAPER) else policy.reduce_cap if life == Lifecycle.REDUCE else 1.
    goal = min(1., max(0., ratio)) if obs and n >= policy.reduce_n else 1. if obs else state.weight
    weight = min(max(state.weight-policy.step, min(state.weight+policy.step, goal)), compose_weight(cap, ramp, ops_cap, Permission.OK))
    perm = Permission.HALT if halt else Permission.BLOCK if life == Lifecycle.RETIRED else Permission.OK
    new = replace(state, lifecycle=life, calendar=t, observations=xs, cusum=s, weight=weight, ramp_cap=ramp,
                  consecutive_failures=0, recovery=rec, shadow=shadow, paused=paused)
    return Evaluation(new, perm, weight if perm == Permission.OK else None, posterior, tuple(proposals))
