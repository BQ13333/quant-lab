# QForge-20 Malaysia PIT Replication — Preflight Findings

## Decision

The exact fourteen-strategy Malaysia replication protocol is frozen, but the
strategy wave did not start. The available data cannot yet support the stated
point-in-time denominator. No strategy trial was reserved, no return series was
calculated, no parameter was changed, no paired performance result was created,
and no dry run was started.

This is a data-admissibility stop, not a negative result for the fourteen
strategies. Re-running them on the same current-survivor universe would not
answer the approved survivorship-bias question.

## Frozen scope

- Protocol: `QFORGE20-MYS-PIT-REPLICATION-14-FROZEN-V1`
- Protocol SHA-256: `72a88829e97b9aa4043025301e8a3c4e3df62b9ad8017ed63195fcb162d78a35`
- Market: Malaysia / Bursa Malaysia only
- Strategies: the ten frozen atomic identities plus the four frozen next-cohort identities
- Optimization: forbidden
- Sign, direction, formula, lookback, rebalance, cost, and selection changes: forbidden
- Multiplicity if the wave later becomes admissible: Holm across all fourteen
- Selection-bias accounting if the wave later becomes admissible: complete Malaysia lifetime PBO after all fourteen are added once
- Stopping rule: stop after the paired evaluation; no dry run

## Deterministic data finding

The source receipts and production adjusted-return artifact passed their hash
and integrity checks. The blocking issue is cross-artifact coverage:

| Evidence | Observed |
|---|---:|
| Malaysia historical listing-window rows | 5,433 |
| Malaysia inactive or delisted rows | 1,537 |
| Current Malaysia feature tickers | 100 |
| Current Malaysia adjusted-return tickers | 100 |
| Current U2 / U3 tickers | 62 / 34 |
| Inactive or delisted tickers with strategy signal inputs | 0 |
| Inactive or delisted tickers with adjusted returns | 0 |

The membership artifact also lacks an explicit time-varying historical U2 tier
and repeated membership snapshots that could reproduce the U2 eligibility set
at each historical decision date. Its own promotion file contains two failed
and two partial gates, including `no_current_survivor_only_universe=fail` and
`has_delisted_names_or_policy=fail`.

Six predeclared admissibility gates therefore failed:

1. historical U2 membership reconstructable;
2. inactive/delisted names have signal inputs;
3. inactive/delisted names have adjusted returns;
4. minimum 95% dead-name adjusted-return coverage;
5. all security-master promotion gates pass; and
6. the denominator is not current-survivor-only.

The earlier `5/5` scorecard established other useful data-lake properties, but
it did not establish a strategy-ready dead-name price and feature denominator.
The present strategy-specific preflight supersedes that narrower claim.

## Twenty-role DeepSeek V4-Pro review

All twenty canonical QForge roles reviewed the same frozen, sanitized dossier
in isolated role contexts. Each received its mandate, routing tags, proof
obligations, kill criteria, corpus anchors, the fourteen specification hashes,
the anti-overfit contract, aggregate data counts, and provenance hashes for the
relevant Markdown knowledge pack.

- Validated role reviews: 20/20
- Verdicts: 20 fail
- Concrete vetoes: 20
- API receipts: 26
- Malformed structured responses retried: 6
- Input tokens: 88,959
- Output tokens: 53,236
- Recorded API cost: USD 0.076398789
- Raw licensed rows sent: none
- Credential stored in evidence: none

The common valid conclusion was that future survival status would enter the
historical universe implicitly if the current 62-name U2 set were used. The
minimum repair is data-only: time-varying historical U2 eligibility, identifier
lineage, strategy inputs, and adjusted returns including terminal delisting
economics for the historical inactive population. The same frozen formulas can
then be evaluated once.

The twenty contexts are not twenty independent foundation models, so their
unanimity is not proof. The deterministic zero-coverage finding is the proof of
the present blocker. A few model statements were also excluded from the
institutional synthesis: U2 is an internal research tier rather than a Bursa
index, and one role's numerical claim about a specific Sharpe reduction was not
supported by a calculation.

## Paired evaluation status

The reference side contains the fourteen existing current-survivor research
results. The PIT replication side is absent because the denominator failed
before strategy computation. A paired performance delta cannot be calculated
honestly yet.

The Malaysia lifetime research ledger remains unchanged at 3,357 attempts and
20 active return series. The DeepSeek protocol-review ledger is separate and
does not consume strategy attempts. An idempotent replay created no additional
API review events.

## Required data-only completion package

Before this frozen wave can resume, the vendor/data lane must provide:

- historical U2 eligibility at every strategy decision date, derived without future information;
- resolved identifier lineage across RIC/ticker changes, mergers, deactivations, and relistings;
- OHLCV and every frozen signal input for historically eligible inactive/delisted names while listed;
- adjusted total returns through the final investable observation, including verified delisting economics;
- at least 95% coverage of the strategy-relevant inactive/dead denominator; and
- a rerun of the security-master promotion gates with every required gate passing.

Once those items exist, rerun the same frozen preflight. Only if it passes may
exactly fourteen attempts be reserved atomically and the paired wave run once.

## Audit artifacts

- Frozen protocol: `configs/qforge20_malaysia_pit_replication_frozen_v1.json`
- Data-admissibility report: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1/data_admissibility_report.json`
- Control-gate receipt: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1/control_gate_receipt.json`
- Twenty-role report: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1/twenty_role_review_report.json`
- Redacted review ledger: `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1/twenty_role_review_ledger.sqlite`
