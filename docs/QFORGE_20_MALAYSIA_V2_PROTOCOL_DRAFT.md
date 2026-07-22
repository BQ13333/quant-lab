# QForge-20 Malaysia V2 Overlay Protocol — Draft for Review

## Current authority state

This is an exact protocol candidate, not a frozen experiment. No V2 strategy
trial has been reserved and no V2 backtest has started. The protocol must first
receive all-twenty-role advisory review, deterministic resolution of actionable
objections, and explicit human freeze approval.

Machine-readable source of truth:
`configs/qforge20_malaysia_v2_overlay_draft.json`.

## Research question

Can one common, causal risk overlay reduce turnover and drawdown for three
distinct frozen Malaysia signals while retaining at least 80% of each paired
control's research-period net information ratio?

The three parents are EWMAC 16/64, residual momentum 126, and low volatility
126. Their signal parameters, U3/U2 universes, research cutoff, and 20/40-bps
cost assumptions remain unchanged.

## Three pairs, six counted arms

Each experiment contains:

- A: exact V1 control replay without the overlay;
- B: the identical signal plus the exact common overlay.

All six arms will enter the lifetime denominator when—and only when—the
protocol is frozen and reserved. A control return-hash mismatch stops the whole
cohort before challenger results are inspected.

## Exact common overlay candidate

- Rebalance: month-end.
- Information lag: one session for scores and volatility.
- Entry: rank at or above the 80th percentile.
- Retention: an incumbent remains eligible while at or above the 60th
  percentile.
- Maximum names: ceiling of 20% of the valid-score universe.
- Deterministic priority: eligible incumbents first; then new entrants; within
  each group, descending score and ascending ticker.
- Weighting: equal weight among selected names; unfilled capacity remains cash.
- Shorting: none.
- Portfolio volatility cap: 10% annualized, estimated from the lagged 63-session
  control gross-return series with at least 42 observations.
- Exposure: `min(1, 10% / lagged annualized volatility)` at month-end, held until
  the next month-end; zero when the estimate is unavailable or invalid.
- Gross exposure maximum: 1.0; cash return: zero.
- Turnover: all absolute asset-weight changes, including exposure changes.
- Costs: unchanged 20 bps one-way base and 40 bps one-way stress.
- Drawdown timing rule: none. The protocol does not add a threshold selected
  from observed drawdown paths.

There is no overlay grid, strategy-specific overlay, combination search, or
within-V2 mutation.

## Gates

Every challenger must pass the existing absolute requirements: positive net
IR, DSR at least 0.95, PBO at most 0.20, maximum drawdown no worse than -20%,
annual turnover at most 3.0, positive 40-bps stress IR, maximum asset weight at
most 25%, positive U2 IR, and a Holm-adjusted one-sided p-value at most 0.05.

It must also:

- retain at least 80% of its paired control net IR;
- use no more than 75% of control annual turnover;
- have maximum drawdown no worse than control;
- achieve at least 90% paired moving-block-bootstrap probability of the 80% IR
  retention condition, using 21-session blocks, 10,000 replications, common
  paired block indices, and seed 20260713.

An individual passes only if every absolute and paired gate passes. The common
overlay is accepted as a reusable research mechanism only if at least two of
three challengers pass. Even that outcome remains research-only and cannot
authorize execution.

## Twenty-role advisory question

All twenty canonical mathematical roles receive the same blinded protocol plus
their own mandate, proof obligations, and kill criteria. They must assess
causal measurability, mathematical definition, numerical stability, paired-test
validity, hidden leverage/turnover, reproducible counterexamples, deterministic
resolutions, and which requests exceed daily-OHLCV evidence.

There is no vote. One reproducible counterexample blocks freeze until resolved.
Out-of-scope sophistication is recorded as an abstention rather than used to
expand the experiment after seeing results.

## Approval boundary

After the advisory synthesis, the user must explicitly approve the final
content hash. Only then may the six arm identities be reserved and the three
paired experiments begin.

## Advisory status — Draft 01

The all-twenty local advisory completed with twenty conditional reviews and one
veto. The controller did not accept the repeated claim that the capped
portfolio exposure scalar could exceed one, but it accepted seven clarification
or design revisions before freeze. Draft 01 therefore remains reviewed evidence
and must not be overwritten.

The exact resolution matrix is documented in
`docs/QFORGE_20_MALAYSIA_V2_ADVISORY_FINDINGS.md`. No strategy trial or backtest
may begin from Draft 01. The next permitted action, after user approval of the
revision set, is to create Draft 02 and run a single three-role local resolution
panel.
