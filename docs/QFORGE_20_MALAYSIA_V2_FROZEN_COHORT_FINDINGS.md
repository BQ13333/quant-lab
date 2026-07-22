# QForge-20 Malaysia V2 — Frozen Cohort Findings

## Decision

Reject the V2 common overlay and close the cohort. Zero of the three
challengers passed every pre-registered absolute, paired, and institutional
gate. No dry run or live execution is permitted.

This is a useful negative result, not a failed workflow. The controls reproduced
their immutable V1 return hashes exactly, all six arms entered the lifetime
denominator before their results were calculated, and the frozen gates stopped
partial improvements from being promoted as investable strategies.

## Cohort receipt

- Frozen protocol hash:
  `9dd12ad0fb963587112989dc5cd31c88f7130a534e1149cb9af954046e5f110c`
- Lifetime attempts: 3,347 before reservation; 3,353 after reservation.
- Active return series: 10 before; 16 after.
- Complete-lifetime PBO: 5.71%, below the 20% ceiling.
- Six arm outcomes: six terminal research failures.
- Verified evidence objects: 18.
- SQLite integrity: OK.
- Backtest model calls and external cost: zero.
- Dry-run events: zero.

## What the overlay actually achieved

| Strategy | IR retention | Turnover reduction | Drawdown change | Bootstrap probability | Paired gates |
|---|---:|---:|---:|---:|---:|
| EWMAC 16/64 | 88.6% | 54.4% | +11.33 pp | 91.41% | Pass |
| Residual momentum 126 | 91.3% | 56.1% | +15.70 pp | 93.55% | Pass |
| Low volatility 126 | 94.3% | 67.2% | -5.95 pp | 74.47% | Fail |

The overlay therefore has a credible relative effect on turnover. For EWMAC
and residual momentum it also improved drawdown and passed the fixed 63-session
paired bootstrap while retaining more than 80% of net IR.

That is not enough to accept it. The protocol required each successful
challenger to pass every absolute and paired gate—not merely improve on a weak
control.

## Why each challenger failed

### EWMAC 16/64

- Net IR fell from 1.483 to 1.314 but retained 88.6% of control.
- Annual turnover fell from 9.72 to 4.43, a large improvement that still
  breached the absolute 3.0 ceiling.
- Maximum drawdown improved from -37.35% to -26.02%, but still breached -20%.
- Latest 126- and 252-session decay IRs were negative.
- Worst regime IR was below -1.0.

### Residual momentum 126

- Net IR fell from 1.313 to 1.198 and retained 91.3% of control.
- Annual turnover fell from 6.42 to 2.82 and passed the absolute turnover gate.
- Maximum drawdown improved from -41.96% to -26.26%, but still breached -20%.
- Latest 126- and 252-session decay IRs remained negative.

This was the strongest challenger, but it still did not clear the complete
protocol.

### Low volatility 126

- Net IR retained 94.3% and annual turnover fell from 5.06 to 1.66.
- Maximum drawdown worsened from -22.22% to -28.17%.
- Paired bootstrap non-inferiority was only 74.47%, below 90%.
- Deflated Sharpe probability remained below 95%.
- Factor-residual and regime gates failed.

Low volatility is direct evidence against declaring this a common overlay for
mathematically distinct strategy families.

## Findings that survived the gate

1. Rank hysteresis plus the portfolio-level risk scalar substantially lowers
   turnover across all three families.
2. The same mechanism can improve relative drawdown for trend and residual
   momentum without destroying their research-period IR.
3. Relative improvement does not solve the absolute drawdown problem.
4. Recent decay is a decisive concern for EWMAC and residual momentum.
5. A common overlay assumption is too strong: low volatility responded
   differently and materially worse on drawdown.

## Anti-overfitting disposition

Do not loosen the -20% drawdown ceiling, change the volatility target, alter the
rank thresholds, substitute another estimator, or rerun the failed arms. Those
changes would be selected after observing contaminated research-period results.

Any future V3 must be separately named, independently motivated, pre-registered
before calculation, and counted as new lifetime attempts. V2 itself is closed.

The recommended next step returns to the previously agreed strategy-inventory
sequence: cluster the remaining Markdown strategies by mathematical equivalence
and data requirements, remove duplicates and unsupported ideas, then
pre-register approximately five to eight genuinely distinct representatives as
a new cohort. That expansion requires a separate approval.

## Evidence

- Human freeze receipt:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/human_freeze_receipt.json`
- Atomic six-arm reservation receipt:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/six_arm_reservation_report.json`
- Complete machine report:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/frozen_cohort_report.json`
- Controller synthesis:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/frozen_cohort_synthesis.json`
- Lifetime ledger:
  `artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault/trial_ledger.sqlite`
