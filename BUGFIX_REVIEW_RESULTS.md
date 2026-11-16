# Review bug fixes — 16 November 2026

Branch: lab-v3-bugfix. No remote operations. No thresholds relaxed.

| Finding | Source files under src/quant_lab | Regression in tests/test_review_regressions.py | Result |
| --- | --- | --- | --- |
| F01 | validation/metrics.py, validation/firewall.py | more charged trials never improve DSR; nonfinite trials raise | 4 failed before, 4 passed after |
| F03 | research/trial_ledger.py | identical retry succeeds; each of ten identity fields rejects drift | 10 failed before, 10 passed after |
| F04 | validation/metrics.py | short, constant, NaN and infinite return samples raise | 4 failed before, 4 passed after |
| F12 | models/ensembles.py, contracts.py | empty/missing or nonfinite coefficients; nonfinite scores | 4 failed before, 4 passed after |
| F14 | portfolio/costs.py | missing/nonpositive/nonfinite ADV; unsupported country | 7 failed before, 7 passed after |
| F15 | portfolio/costs.py | opening cost from cash and explicit starting holdings | 1 failed before, 1 passed after |
| F11 | regimes.py | price-unit rescaling invariance; calibration warmup UNKNOWN | 2 failed before, 2 passed after |
| F16 | portfolio/optimizers.py | annual covariance convention; infeasible cap/turnover | 2 failed before, 2 passed after |

## Validation

Same selected suite, including new regressions:

| Run | Passed | Failed | Skipped | Collection/setup errors |
| --- | ---: | ---: | ---: | ---: |
| Before | 168 | 34 | 1 | 0 |
| After | 202 | 0 | 1 | 0 |

The original source and original existing tests were restored temporarily for the baseline, then the fixes were restored. All 34 new regression cases failed on original source. Existing tests were updated with comments where they required unknown-market default fees, free initial trades, or an impossible four-asset fully invested allocation under a 20% cap.

Command (with PYTHONDONTWRITEBYTECODE=1 and TMPDIR inside this worktree):

```sh
/workspace/quant_fix/.venv/bin/python -m pytest -o addopts="" -p no:cacheprovider tests/test_review_regressions.py tests/test_verified_bugfixes.py tests/test_overfit.py tests/test_metrics.py tests/test_gates.py tests/test_cpcv.py tests/test_costs.py tests/test_ensembles.py tests/test_diagnostics.py tests/test_trial_ledger.py tests/test_validation_firewall.py tests/test_optimizers.py tests/test_research_contracts.py tests/test_model_zoo.py tests/test_stress.py --basetemp=.bugfix-results/tmp-final
```

## Implementation limits

DSR uses observed, positive finite dispersion independently of the charged count; it does not invent missing trial observations. This does not implement the separate global-ledger evidence-binding finding F02.

Cost APIs preserve positional arguments and add keyword-only starting_holdings, defaulting to cash. Unsupported markets raise; this does not invent CN/HK fee schedules.

The turnover optimizer validates cap, full-investment budget and turnover jointly, including its NumPy fallback. A fallback proposal that cannot satisfy all constraints raises rather than being clipped and renormalized. This conservative fallback is not a general feasibility solver and can reject a book that a more capable solver could allocate. No requested finding was skipped.

## Commit blocked by filesystem permissions

The requested git add and git -c user.name="Codex Astra (box)" -c user.email="box@local" commit were attempted. Both failed because Git could not create index.lock in the read-only parent repository worktree metadata. No commit was created. Source and tests remain modified in this authorized worktree. No remote was accessed.
