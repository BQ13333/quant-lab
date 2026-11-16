# Alpha lifecycle lab

All times are monthly unless an API explicitly says otherwise. This package
uses NumPy/SciPy/pandas and the existing Ed25519 approval verifier, makes no external calls, and submits no orders.

`states` separates a value, a scheduled non-applicable metric, a scheduled
no-observation period, and a missing input. Callers register an immutable `ScheduleCalendar` before
seeing the data. The monitor derives schedules by month and binds its calendar hash
in state; a changed calendar blocks evaluation. Values on non-due fields also block. `validate_inputs` blocks missing, non-finite, extra or omitted
inputs. `to_json`/`from_json` preserve versioned observation, enum, calendar, policy and monitor-state tags; unknown tags or versions raise.

`monitor.evaluate` is an immutable monthly transition. Default fresh sleeves
carry a 10% ramp cap. A mature sleeve must explicitly carry its approved cap;
calendar age alone never advances G8. A FAIL emits no target and holds the IC
history and CUSUM. Two consecutive FAILs or three in twelve months pause the
sleeve. HALT suppresses targets while valid statistics continue. Retirements and
re-entries are proposals until the caller supplies a signed `Authorization`. The
existing externally anchored human trust store verifies signatures bound to both
the carried state and action evidence; a boolean or claimed human ID is insufficient. Re-entry resets the
carried cap to the first registered stage (10% by default; register 25% where
that is the first feasible construction). Safety caps may cut weight faster
than the soft monthly 25% step limit.

The single executable lifecycle worked example is `examples/decay_example.py`:
run `PYTHONPATH=src python examples/decay_example.py` from the repository root.
It imports the actual monitor and asserts the canonical numbers. Companion tests
cover failure, gap, recovery, operational and re-entry scenarios. The source
example's cost input is interpreted as the registered **two-month** cost ratio,
already aggregated upstream. The monitor does not reinterpret a single-month
ratio as two independent consecutive breaches.

`decay` implements Normal-Normal mean inference, capped AR(1) effective sample
size, CUSUM and exact untruncated BOCPD recursion with known observation variance.
BOCPD run length zero is a boundary after the datum, so with constant hazard its
probability alone is constant. Detection uses the sum of recent run lengths.
The horizon half-life fit is a log-linear diagnostic for positive marginal
IC/rates; it must not be fed cumulative-horizon IC. Its slope standard error and
the rolling-time slope standard error assume iid residuals. Neither is a
bootstrap certification of half-life. Page–Hinkley is optional and omitted.

`caps.check_exposure` accepts instrument-level current holdings, open orders and
signed proposed trade values in NAV currency. Unknown economic classifications
count as CNY-economic. CNY-economic (45%) and foreign (60%) caps both bind.
Open buys consume headroom; unfilled sells never create headroom for new buys.
Sells pass unchanged, buys round down in instrument lots and drop below the
minimum ticket. Existing breaches return required sell values per overlapping
bucket rather than raising. `check_exposure_cap` is a legacy bucket adapter.

`ramp.advance_ramp` requires complete stage evidence and a signed human approval,
advances exactly one registered stage in KEEP/RAMP, and prevents approval reuse.
`Sleeve.from_monitor` carries the monitor weight, ramp ceiling, permission and
operational ceiling into allocation; manually constructed sleeves cannot allocate.
`feasibility` computes pilot/child-order floors and offers `FloorReport.require_nav`
for admission. Paper-engine wiring is the plan's P1 integration task.

`allocator` uses discounted full-information shadow observations, a fixed Normal
prior, and seeded Thompson draws. Floors/caps, cash, and L1 sleeve turnover are
explicit. When liquidation requirements and turnover are incompatible it raises
an infeasibility error; a caller must explicitly choose a liquidation budget.
The inverse-variance baseline uses the same bounds. Synthetic superiority is a
test of adaptation in the specified scenario, not a universal superiority claim.

`renewal` implements Little's law, exponential family survival, equicorrelation
IR, the inverse birth-rate calculation, and a Gaussian discovery-funnel haircut.
The funnel haircut is a planning assumption. `discovery_probability` reuses
`validation.overfit.deflated_sharpe_probability`; existing CSCV/PBO remains in
`validation.overfit`. Supply the complete family from
`research.trial_ledger.TrialLedger.lifetime_counts`, including inherited and
failed attempts; do not count only accepted sleeves. Evidence persistence stays
in that ledger. Existing `regimes.VolTrendRegimeDetector` remains a diagnostic;
no unregistered regime sizing rule is introduced here.

`fwer` counts any retirement proposal across a simulated family within a stated
finite horizon. It searches a fixed policy grid on calibration seeds and runs
one independent validation seed set, returning a two-sided 95% Clopper–Pearson
interval. Pass requires the validation upper bound at or below the target and at least
`min_validation` replicates (default 2,000). A failed grid search or insufficient
validation remains `NOT_CALIBRATED`. False-pause rates are reported separately.
Nulls include Gaussian AR(1) and centered circular block bootstrap histories;
register the corresponding `Policy.psi`. This is an empirical finite-horizon
calibration, not a guarantee beyond the registered null and horizon.

`sim` pre-generates Poisson births, prior IC, random half-lives and observation
noise for paired birth-stop experiments. Decisions precede rewards. Retirements
remain proposals without automatic approval, including in this experiment. Stage caps stay
at the first stage because the simulation has no execution evidence to approve
G8 advances. The output includes realised portfolio IC and conditional expected
annualised ICIR under independent observation noise, not realised return IR.
