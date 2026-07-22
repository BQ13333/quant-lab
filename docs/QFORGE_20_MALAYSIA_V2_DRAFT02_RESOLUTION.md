# QForge-20 Malaysia V2 — Draft 02 Resolution

## Outcome

Draft 02 has been deterministically reconstructed from immutable Draft 01 plus
the user-approved seven-revision manifest. Its canonical protocol hash is:

`9dd12ad0fb963587112989dc5cd31c88f7130a534e1149cb9af954046e5f110c`

The exact three-role local resolution review is complete. Selection-bias and
time-series dependence passed the design. High-frequency econometrics returned
a conditional veto. The controller rejects that veto because its two stated
counterexamples fail direct arithmetic; the model flag is preserved in full in
the evidence rather than silently removed.

Draft 02 is ready for **final human freeze approval**. This means the design is
specified well enough to test. It is not evidence of alpha, Sharpe improvement,
or execution readiness.

## Review receipt

- Roles: selection-bias validation, time-series dependence, and high-frequency
  econometrics.
- Model route: local `qwen3.5:35b-mlx` through Ollama.
- Calls: exactly three; no retry and no follow-up call.
- Result: two pass, one conditional, one raw veto.
- Hosted cost: USD 0.
- Strategy trials reserved: none.
- Backtests started: none.
- Malaysia lifetime strategy ledger: unchanged before and after review.

The roles are three isolated contexts from the same local model. They are not
three independent mathematical minds, and their outputs do not vote.

## Why the veto is not a blocker

The veto claimed that the piecewise exposure scalar is discontinuous at 10%
volatility and can create a massive turnover spike. It is continuous:

| Annualized volatility | Exposure |
|---:|---:|
| 0.09999999 | 1.00000000 |
| 0.10000000 | 1.00000000 |
| 0.10000001 | 0.99999990000001 |

The boundary change is about `1e-7`. At `v = 0.105263...`, exposure is `0.95`;
that five-percent reduction represents real portfolio de-risking and therefore
real risky-weight turnover.

The second counterexample claimed a sample standard deviation can become
negative through numerical precision and that a transition to cash creates no
trade. Standard deviation is non-negative by construction. Non-finite and zero
values fail closed. When a previously invested portfolio goes to zero exposure,
the risky assets must be sold, so counting that turnover is correct.

Other concerns were retained as implementation checks where appropriate:
the frozen universe ticker field and universe hash must agree, timing must use
the one-session lag, and the exactly-four-name boundary must be tested. Requests
to tune the observation count, shift the lookback, substitute EWMA, or add
microstructure models were rejected because they would create a new search or
expand beyond daily OHLCV.

## What is now frozen in the candidate hash

- One month-end rank-hysteresis overlay applied to EWMAC 16/64, residual
  momentum 126, and low volatility 126.
- Static SHA-256 tie ordering from protocol ID and frozen ticker.
- At least four selected names or zero exposure plus a recorded challenger
  failure.
- One portfolio exposure scalar estimated from the lagged, unit-exposure,
  pre-scaled challenger basket.
- Risky-asset turnover with cash excluded as a second leg, under the same
  initial-entry convention for both arms.
- Fixed 63-session paired moving-block bootstrap, 10,000 replications, seed
  20260713, and the margin `IR_B - 0.80*IR_A >= 0`.
- All original absolute and paired gates, including DSR, PBO, U2, Holm,
  drawdown, turnover, concentration, costs, regime, factor, and decay checks.
- No parameter grid, no post-hoc combination, and no research pass that can
  authorize execution.

## Approval boundary

The revision-set approval authorized materialization and the three local
resolution calls. It did not authorize strategy-trial reservation or a
backtest. The next approval must name the canonical Draft 02 hash. Only then may
the six pre-registered arm identities enter the Malaysia lifetime denominator.

## Evidence

- Revision manifest:
  `configs/qforge20_malaysia_v2_overlay_draft02_revision.json`
- Materialized protocol:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/materialized_protocol.json`
- Raw three-role report:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/resolution_report.json`
- Separate resolution ledger:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/resolution_ledger.sqlite`
- Deterministic controller resolution:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_draft02/controller_resolution.json`
