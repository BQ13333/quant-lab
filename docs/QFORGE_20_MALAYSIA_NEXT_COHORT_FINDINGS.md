# QForge-20 Malaysia Next-Cohort Findings

## Decision

The frozen four-strategy wave is complete and stopped for evaluation. Zero of four strategies passed every preregistered institutional gate. No dry run, terminal-holdout release, parameter optimization, combination search, or live execution was performed.

This is a valid research outcome rather than an engineering failure. The four attempts were atomically reserved before return calculation, all four reached terminal computation state before results were unsealed, and every result entered the lifetime ledger.

## Frozen identity

- Protocol: `QFORGE20-MYS-NEXT-COHORT-PREREG-DRAFT-02`
- Canonical protocol SHA-256: `5f7e02bc669391e22cc712b0827c7c760027dfe5164c10633968b06be909a79d`
- Human-approved revision: `QFORGE20-MYS-NEXT-COHORT-DRAFT02-REVISIONSET-01`
- Research cutoff: 2026-07-06
- Market: Malaysia
- Evidence class: current-survivor historical research only; non-promotable
- Local targeted resolution: three DeepSeek roles passed, zero vetoes, USD 0 external cost

## Cohort result

| Strategy family | Net IR | Stress IR | U2 IR | Holm p-value | DSR | Max drawdown | Formal result |
|---|---:|---:|---:|---:|---:|---:|---|
| Price-volume confirmation | 0.704 | 0.215 | 0.921 | 0.00309 | 0.9488 | -47.1% | Failed |
| Lagged illiquidity premium | 1.058 | 0.983 | 1.154 | 0.00000232 | 0.9995 | -37.8% | Failed |
| Turn-of-month | -0.529 | -1.796 | -0.662 | 0.314 | 0.00004 | -53.1% | Failed |
| Opening-gap reversal | -3.631 | -7.244 | -3.439 | 1.000 | 0.0000 | approximately -100% | Failed |

The complete Malaysia lifetime PBO was 0.0429 after adding the four return series. This passes the global 0.20 selection-bias threshold, but it does not override a strategy's failed risk, capacity, factor, or strategy-specific gates.

## Interpretation

### Price-volume confirmation

The observed net, stress, breadth, and multiplicity-corrected results are positive. It nevertheless failed the frozen DSR threshold by a small margin, the 20% drawdown limit, historical capacity, worst-regime stability, and factor-residual requirements. Its factor-residual IR was -0.274 and its worst-regime IR was -2.135. It is not independent enough from registered factors and is not deployable in its present form.

### Lagged illiquidity premium

This was the strongest research finding. It passed positive net IR, stress IR, U2 breadth, DSR, Holm significance, asset-PnL concentration, and the frozen regime test. It failed four decisive gates: maximum drawdown (-37.8%), the 5% historical lagged-ADV participation cap at the MYR 10 million reference notional, factor residual evidence (IR -0.054; alpha t-stat -0.271), and annual turnover (8.65 versus the frozen maximum of 3.0).

The result is evidence that the registered illiquidity exposure is present in this dataset. It is not evidence of a new independent, institutionally executable alpha.

### Turn-of-month

The preregistered inside-versus-outside comparison was not significant after Holm correction, and net, stress, and U2 results were negative. This version should be rejected without window mutation or calendar-window search.

### Opening-gap reversal

The gross intraday effect did not survive the frozen 45 bps one-way base execution cost. Break-even total cost was only 26.39 bps one way, below the required 65 bps, and only 9.52% of calendar years were positive after base costs. This version should be rejected without changing the sign, gap threshold, holding period, or cost model on the same evidence.

## Institutional disposition

- Retain price-volume and illiquidity as scientifically interesting failed hypotheses in the immutable vault.
- Reject the registered turn-of-month and opening-gap versions.
- Do not relax gates, tune parameters, search combinations, or rerun these four identities.
- Do not start execution dry runs.
- Keep the pairs family KIV until point-in-time industry classification and historical borrow availability/cost arrive.
- A future wave must be a newly preregistered cohort. The defensible inputs for that design are point-in-time vendor data, an ex-ante capacity/turnover mechanism, and a genuinely unreleased evaluation sample—not modifications chosen to repair these observed failures.

## Vault receipts

- Frozen cohort report: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft02/frozen_cohort_report.json`
- Atomic reservation report: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft02/four_strategy_reservation_report.json`
- Human freeze receipt: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/next_cohort_prereg_draft02/human_freeze_receipt.json`
- Frozen cohort report SHA-256: `a0f2ebeb323b6dafcd31f02ee41b1431a7b36bd28e43dc5cfe517510384bf256`
- Atomic reservation report SHA-256: `bb01c8ae691e118c6891360dc71a0c5d9866454165afee2f3343d2c233b13053`

## Verification

- Four new attempts were reserved atomically.
- Malaysia lifetime attempts increased from 3,353 to 3,357.
- Active Malaysia return series increased from 16 to 20.
- An idempotency replay created zero additional attempts.
- The complete repository test suite passed after the run.
