# Malaysia Atomic Cohort V1 — Research Findings

## Decision

The frozen ten-strategy Malaysia cohort is complete. None of the strategies is eligible for institutional audit or execution dry-run. Every candidate breached at least the drawdown and/or turnover policy, and every panel left at least one unresolved design condition.

These are research-period results, not pristine out-of-sample results. The history through 2026-07-06 has been used for research and audit and is therefore contaminated for promotion. A terminal holdout can begin only with observations strictly after that date.

## Scope and provenance

- Cohort: `MALAYSIA-ATOMIC-COHORT-V1`
- Manifest hash: `ff1dbf697c037fac51919d2c296297fac6ed1c897dee34a85f50d2430e721915`
- Active atomic strategies: 10
- Universe: Bursa Malaysia U3 primary (34 stocks), unchanged U2 robustness universe (62 stocks)
- Costs: 20 bps one way; 40 bps one-way stress
- Historical research range: 2006-01-02 through 2026-07-06
- Lifetime denominator after operational retries and the corrected implementation: 3,347 attempts, including 3,330 legacy attempts
- Frozen machine report: `artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault/cohorts/MALAYSIA-ATOMIC-COHORT-V1-report.json`
- QForge-20 strategy/result index: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/index.json`
- QForge-20 V4 review report: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v4_review_report.json`
- QForge-20 V4 human diagnosis: `docs/QFORGE_20_MALAYSIA_V4_FINDINGS.md`

## Atomic results

`Net Sharpe` below is the annualized Sharpe of the 20 bps net research-period return series. It must not be described as terminal OOS Sharpe.

| Strategy | Net Sharpe | Max drawdown | Annual turnover | U2 net IR | Holm p-value | Principal unresolved gates |
|---|---:|---:|---:|---:|---:|---|
| EWMAC 16/64 | 1.483 | -37.4% | 9.72 | 1.425 | <0.000001 | specialist condition, PBO at time of trial, drawdown, turnover |
| Residual momentum 126 | 1.313 | -42.0% | 6.42 | 1.476 | <0.000001 | specialist condition, drawdown, turnover |
| Momentum 126 | 1.303 | -42.8% | 7.08 | 1.376 | <0.000001 | specialist condition, drawdown, turnover |
| Volatility-adjusted momentum 126/63 | 1.258 | -45.6% | 7.87 | 1.389 | <0.000001 | specialist condition, PBO at time of trial, drawdown, turnover |
| Time-series momentum 126 | 0.796 | -28.0% | 7.88 | 0.868 | 0.0012 | specialist condition, drawdown, turnover |
| Bollinger reversion 63 | 0.572 | -42.4% | 47.38 | 0.457 | 0.0200 | specialist condition, drawdown, turnover |
| Low volatility 126 | 0.561 | -22.2% | 5.06 | 0.508 | 0.0276 | specialist condition, PBO at time of trial, drawdown, turnover |
| Channel 55 | 0.494 | -61.9% | 44.38 | 0.466 | 0.0525 | specialist condition, deflated Sharpe, drawdown, turnover, cost stress |
| Mean reversion 63 | 0.415 | -43.0% | 10.66 | 0.268 | 0.0656 | specialist condition, deflated Sharpe, drawdown, turnover |
| Breakout 55 | 0.237 | -59.7% | 64.76 | 0.122 | 0.1461 | specialist condition, deflated Sharpe, drawdown, turnover, cost stress |

The final cohort-level PBO is 0.0143. Earlier per-trial PBO receipts were necessarily order-dependent because the return matrix was incomplete; the frozen cohort-level value supersedes those provisional diagnostics for selection analysis. It is still not evidence of terminal out-of-sample performance.

## Cross-strategy findings

The apparent leaders are one crowded trend/momentum cluster, not four independent alphas. Pairwise correlations include momentum versus volatility-adjusted momentum at 0.888, momentum versus residual momentum at 0.815, residual momentum versus EWMAC at 0.747, and volatility-adjusted momentum versus EWMAC at 0.725.

The reversal and breakout families did not compensate for the leaders' implementation risk. Breakout, channel and Bollinger reversion generated especially high turnover. Low volatility produced the smallest drawdown, but still breached the fixed -20% policy.

Seven raw strategies retain a positive-mean result after the ten-way Holm correction: EWMAC, residual momentum, momentum, volatility-adjusted momentum, time-series momentum, Bollinger reversion and low volatility. This controls only the ten tests in this frozen cohort. The 3,330 legacy attempts remain relevant to broader selection-bias and deflated-Sharpe interpretation.

## Predeclared combinations

| Combination | Net Sharpe | Max drawdown | Incremental IR versus best component | Decision |
|---|---:|---:|---:|---|
| EWMAC + low volatility | 1.318 | -28.5% | -1.342 | reject |
| Momentum + mean reversion | 1.041 | -41.1% | -0.786 | reject |
| Breakout + residual momentum + low volatility | 0.895 | -39.1% | -1.066 | reject |

All three combinations are inferior to their best component on the predeclared incremental-IR test. No post-hoc combination search is authorized from this cohort.

## Audit corrections retained in the vault

The first residual-momentum run created no finite score observations because its market-variance rolling window required an uninterrupted history. Its zero return series was invalidated by correction event 154, not deleted. The fixed formula uses the same minimum-observation rule for covariance and market variance, and the unchanged strategy was rerun under trial `b477dd0b-9e99-48bf-b930-996121f76703`.

The first Bollinger review was blocked when DeepSeek timed out. The original failure remains recorded. A separate local-Qwen operational retry completed under trial `1b29baac-38e5-44df-882d-f934dc8280f5`; no strategy parameter changed.

## Next bounded iteration

Do not tune lookbacks or search combinations against these results. Freeze one common risk-and-execution overlay before seeing its results, then apply it to a small, source-backed follow-up cohort. The justified targets are turnover and drawdown, not a higher backtest Sharpe.

A defensible V2 would test the three distinct representatives below, once each, with the same overlay and no grid search:

1. EWMAC 16/64 as the trend representative.
2. Residual momentum 126 as the cross-sectional momentum representative.
3. Low volatility 126 as the defensive representative.

The overlay should be declared in advance: monthly rebalance, fixed entry/exit rank hysteresis, ex-ante portfolio volatility cap, unchanged U3/U2 universes, unchanged 20/40 bps costs, and the existing drawdown/turnover thresholds. Promotion remains blocked until all three finish, multiplicity and correlations are recomputed, every specialist condition is machine-resolved, and genuinely post-2026-07-06 observations become available for terminal holdout.
