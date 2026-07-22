# QForge-20 Malaysia V4 Focused Review — Findings

## Decision

The first live QForge-20 focused wave is complete. DeepSeek V4-Pro produced 30
accepted structured specialist reviews for the ten frozen Malaysia strategies.
No strategy is authorized for a child trial, institutional audit, terminal
evaluation, or execution dry run.

This decision is deterministic. The V4 panels can identify risks and propose
falsification tests, but they cannot relax the failed drawdown, turnover,
multiple-testing, cost-stress, or evidence-separation gates already recorded by
the Malaysia atomic cohort.

## Scope

- Workflow: `QForge-20 / QFORGE20-MALAYSIA-RESEARCH-V1`
- Review phase: focused post-backtest research interpretation
- Frozen strategies: 10
- Reviews: 3 isolated role contexts per strategy, 30 total
- Verdicts: 28 conditional, 2 fail
- Vetoing reviews: 3 across 2 vetoed strategies
- Role contexts instantiated: selection bias (10), microstructure/execution
  (10), measure/probability/martingale (9), and time-series dependence (1)
- Full twenty-role finalist wave: not executed because no strategy qualified as
  a finalist
- Terminal evidence: unavailable; all observations through 2026-07-06 are
  contaminated research history
- Dry run: not attempted

The twenty specialists are a registry of mandates, not twenty independent
models or twenty human employees. Staged routing deliberately spends three
reviews on a weak research candidate and reserves all-mandate coverage for a
genuine finalist.

## Panel outcome by strategy

| Strategy | Research net Sharpe | Max drawdown | V4 panel | Progression |
|---|---:|---:|---|---|
| EWMAC 16/64 | 1.483 | -37.4% | 3 conditional | blocked |
| Residual momentum 126 | 1.313 | -42.0% | 3 conditional | blocked |
| Momentum 126 | 1.303 | -42.8% | 3 conditional | blocked |
| Volatility-adjusted momentum 126/63 | 1.258 | -45.6% | 3 conditional | blocked |
| Time-series momentum 126 | 0.796 | -28.0% | 3 conditional, 1 veto | blocked |
| Bollinger reversion 63 | 0.572 | -42.4% | 3 conditional | blocked |
| Low volatility 126 | 0.561 | -22.2% | 3 conditional | blocked |
| Channel 55 | 0.494 | -61.9% | 2 fail/veto, 1 conditional | blocked |
| Mean reversion 63 | 0.415 | -43.0% | 3 conditional | blocked |
| Breakout 55 | 0.237 | -59.7% | 3 conditional | blocked |

`Net Sharpe` is the deterministic 20-bps-cost research-period statistic from
the frozen cohort. It is not terminal OOS Sharpe. A conditional V4 verdict is a
request for more evidence, not a pass.

## What the V4 wave added

The useful part of the review is the convergence on five falsification classes:

1. **Execution realism.** Measure turnover, effective spread, participation,
   market impact, delayed fills, shortability, and capacity with Bursa-specific
   point-in-time inputs. The existing 20-bps base and 40-bps stress are useful
   gates but not a substitute for instrument-level TCA.
2. **Causal data integrity.** Re-audit lagging, corporate actions, delistings,
   universe membership, liquidity filters, and signal-to-fill timing.
3. **Selection-aware statistics.** Use block or stationary bootstrap, deflated
   Sharpe/reality-check methods, the full lifetime denominator, and purged or
   embargoed temporal splits. The cohort PBO of 0.0143 covers the ten active
   return series; it does not erase the 3,330 imported legacy attempts.
4. **Economic distinctness.** Test factor exposures, liquidity/size dependence,
   regime concentration, and whether performance survives in liquid names.
5. **Future evidence.** Preserve the frozen candidates and collect genuinely
   post-2026-07-06 observations. Earlier calendar periods suggested by model
   prose are already research-contaminated and cannot be relabeled as holdout.

Channel 55 is the clearest rejection. Its adjusted p-value exceeds 5%, maximum
drawdown is about 62%, cost stress fails, and two independent role contexts
issued fail/veto verdicts. Time-series momentum is also vetoed because the
selection-bias reviewer challenged whether its cohort PBO fully represents the
lifetime search. That challenge is a demand for reconciliation, not proof that
the reported cohort PBO was computed incorrectly.

## AI suggestions that are not authorized

Several reviews proposed useful categories but unsafe implementations. The
controller must reject or reinterpret them as follows:

- Dates such as 2019-2025 are not new OOS data; they predate the research
  cutoff and remain research-only robustness periods.
- Varying lookbacks, thresholds, rebalance days, universes, or costs cannot be
  used to select a better version. A small predeclared neighborhood may diagnose
  fragility, but every variation counts as a trial and cannot replace the frozen
  candidate.
- Cross-market tests are separate market hypotheses. Singapore or other ASEAN
  evidence cannot promote a Malaysia candidate without its own reservation and
  scope.
- A model's confidence, consensus, theorem language, or veto is not proof.
  Deterministic evidence remains decisive.
- Tick/Hawkes execution work is not required for every rejected strategy. It is
  justified only after cheaper causal, statistical, drawdown, and turnover gates
  leave a small candidate set.

## Cost and evidence accounting

Thirty successful V4 calls have immutable, redacted token and cost receipts:

- successful input tokens: 34,727
- successful output tokens: 74,204
- tracked successful-call cost: USD 0.070608301
- separately reconciled pre-campaign schema smoke call: USD 0.001210660
- known spend lower bound: USD 0.071818961
- malformed structured responses: 17

The malformed responses were deferred and retried; none was treated as research
evidence. This campaign version did not persist their provider usage receipts,
so USD 0.071818961 is a lower bound, not the exact amount billed. The report
keeps the unledgered schema smoke call separate and states this limitation
instead of inventing a total.

No raw prompt body, raw response envelope, reasoning content, environment
secret, or API key is stored in the evidence bundles. The validated structured
review is retained as research evidence; each accepted review is linked to a
redacted call receipt and verified by SHA-256.

## Next bounded research iteration

The V4 wave does not justify searching more strategy parameters. It reinforces
the existing V2 recommendation: freeze one common risk-and-execution overlay
before inspecting its results, then apply it once to three distinct
representatives—EWMAC 16/64, residual momentum 126, and low volatility 126.

The overlay target is lower turnover and drawdown, not higher in-sample Sharpe:
monthly rebalance, fixed rank hysteresis, ex-ante volatility cap, unchanged U3
and U2 universes, unchanged 20/40-bps costs, and unchanged gates. Each material
variation must be a separately reserved child trial. No post-hoc combination
search is authorized.

## Durable evidence

- Strategy/result index:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/index.json`
- Append-only review ledger:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/review_ledger.sqlite`
- Verified machine report:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v4_review_report.json`
- Redacted content-addressed evidence:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/review_evidence/`
