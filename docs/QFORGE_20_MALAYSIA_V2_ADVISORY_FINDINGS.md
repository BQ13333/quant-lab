# QForge-20 Malaysia V2 — Twenty-Role Advisory Findings

## Outcome

All twenty canonical mathematical roles reviewed the same exact V2 Draft 01 in
isolated local-Qwen contexts. Every role returned a valid conditional review;
the high-frequency econometrics role issued the only veto. No strategy trial
was reserved, no backtest ran, no cloud credential was used, and hosted cost was
zero.

The controller decision is **revise before freeze**. This is not a model vote.
The twenty contexts are correlated outputs from one local model, and several
repeated the same incorrect reading of the exposure equation.

## What the panel got wrong

Many reviews claimed that `min(1, 10% / volatility)` could produce realized
gross exposure above one. It cannot: `min` caps the single portfolio-level
scalar at one. Several reviews also treated the scalar as if it were calculated
independently for each asset, which the draft did not specify.

Those claims are not accepted as mathematical counterexamples. Nevertheless,
direct division near zero and the phrase “portfolio volatility cap” were easy
to misread. Draft 02 should express the identical cap as a piecewise branch and
state explicitly that it is one portfolio scalar, not per-asset leverage or a
guarantee of future realized volatility.

## Accepted advisory improvements

1. **Exposure arithmetic:** use a piecewise branch—zero for invalid/non-positive
   volatility, one for positive volatility at or below 10%, otherwise
   `10% / volatility`. This avoids division in the low-volatility branch.
2. **Risk-series alignment:** estimate risk from the lagged unit-exposure,
   pre-scaled challenger basket, not the V1 control basket, because hysteresis
   can change holdings.
3. **Neutral ties:** replace alphabetic ticker ordering with a static SHA-256
   order derived from protocol ID and ticker.
4. **Diversification floor:** require at least four selected names. Fewer names
   produce zero exposure plus a recorded failure, protecting the 25% asset cap.
5. **Turnover accounting:** sum changes in risky-asset weights, including
   exposure-scalar changes. Cash is the residual and is not double-counted.
6. **Dependence:** replace the 21-session block with one fixed 63-session block.
   Do not run a block-length search.
7. **Non-inferiority arithmetic:** bootstrap the margin
   `IR_B - 0.80*IR_A >= 0` rather than an unstable IR ratio. Non-finite draws
   count as failures.

## Recommendations not accepted

The panel also requested tick-level/Hawkes analysis, neural and rough-path
models, Bayesian latent volatility, graph covariance work, and sensitivity
grids over lookbacks and thresholds. Those requests either exceed the daily
OHLCV evidence or create new searches. They are recorded as out-of-scope
advisory recommendations and will not enter V2.

## Rules that remain unchanged

- Challenger net IR retention: at least 80% of control.
- Challenger turnover: at most 75% of control and at most 3.0 annually.
- Maximum drawdown: no worse than control and no worse than -20%.
- Existing DSR, PBO, cost-stress, concentration, U2, Holm, regime, factor, and
  decay gates remain mandatory.
- The common overlay is accepted only if at least two of three challengers pass
  every individual gate.
- No model verdict can authorize a strategy, and no research-period pass can
  authorize execution.

## Recommended next step

After the user approves the seven exact revisions, create immutable Draft 02
and route it once to only three resolution roles: high-frequency econometrics,
selection-bias validation, and time-series dependence. The targeted review is
capped at three local calls. It must not become a loop that asks the same model
until it agrees.

If the revisions resolve the concrete objections, present the final Draft 02
hash for human freeze approval. Only after that second approval may the six
strategy-arm identities enter the lifetime denominator and the three paired
experiments begin.

## Evidence

- Reviewed protocol: `configs/qforge20_malaysia_v2_overlay_draft.json`
- Full advisory report:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_advisory/advisory_report.json`
- Advisory ledger:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_advisory/advisory_ledger.sqlite`
- Controller synthesis:
  `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v2_protocol_advisory/advisory_synthesis.json`
