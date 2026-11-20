# Quant house kickoff

20 November 2026, Asia/Kuala_Lumpur. Product owner: Benjamin Quek.

This is a lab. It finds a strategy, checks it, watches it decay, and replaces it. It does not send orders. Brokers, including IBKR, stay unwired. Markets for this programme are Bursa Malaysia, mainland China, and Hong Kong. No crypto, no US markets, no high-frequency trading. Fundamentals and technicals are both in scope, and only where the data can actually be known on the day a decision would have been made. Point-in-time dates, delistings, and costs are mandatory.

## Decisions recorded 20 November 2026, 18:26 MYT

Benjamin Quek closed the five questions in section 4. The record is [docs/DECISIONS_2026-11-20.md](DECISIONS_2026-11-20.md). The scope file is Malaysia, China A-shares (SSE and SZSE), and Hong Kong. Singapore, Indonesia, Japan, Korea, and the United States are frozen. BSE stays out. Ziplime 2.12.0 is allowed only as a lowest-priority research sandbox in a separate environment, file exchange only, and its licence is still unclear (GPL-3.0, a commercial-use restriction, and a README that says the project is moving to AGPL). It is not a dependency. `lime-trader-sdk` stays unwired. Both pull request #1s stay unmerged. The private pull request's base is `outbox/box`, not `main`. The US archive outside git is the copy of record. Japan stays in-tree and dormant. Fundamentals are official public filings only: Bursa, HKEXnews, and CNINFO, subject to access rights. No paid feeds.

The target architecture is the copied blueprint:

- [docs/cloud/EXEC_SUMMARY.md](cloud/EXEC_SUMMARY.md)
- [docs/cloud/CLOUD_PRODUCTION_BLUEPRINT.md](cloud/CLOUD_PRODUCTION_BLUEPRINT.md)
- [docs/cloud/architecture.png](cloud/architecture.png)

Still open: Ben must name the human root-account holder. Phase 2 of that blueprint, which is provisioning, needs his separate GO. No cloud accounts and no spend. Historical cost coverage and the per-instrument Hong Kong lot table are inventoried in [docs/HISTORICAL_COST_AND_LOTS.md](HISTORICAL_COST_AND_LOTS.md). Both gates are unmet, which is why spend stays held.

Two repositories:

- Public research framework: [BQ13333/quant-lab](https://github.com/BQ13333/quant-lab), default branch `main` at `5535260` ("publish evidence-first Quant Lab research framework"). MIT.
- Private data and archive side: [BQ13333/08-quantitative](https://github.com/BQ13333/08-quantitative). This kickoff read it. It did not change it.

Both already have an open pull request #1 from the 16 November lab-v3 work. This kickoff builds on those pull requests. It does not merge them and does not re-implement them.

## 1. Current state

### quant-lab `main` (landed)

`main` is the evidence-first research framework: point-in-time admission, a signal factory, validation (walk-forward, CPCV/PBO, deflated Sharpe, a firewall), portfolio costs, a Malaysia research loop, and a large Japan data-quality stack. The execution package can build paper intents and contains an IBKR adapter that refuses live ports unless live mode is explicitly armed. This programme does not extend that adapter.

The published market file `configs/project_market_scope.yaml` (effective 12 July 2026) is older than the November direction. It still lists Malaysia, Singapore, Indonesia, Hong Kong, Japan, Korea, and the United States as active, and it lists China as excluded. `docs/DATA_SOURCE_ROADMAP.md` and `docs/HISTORICAL_DATA_LAKE_BACKFILL.md` say the same thing. There is no `.github` workflow in the tree. The README records 1,130 passing tests on the public snapshot, with 78 integration tests skipped unless private data is present. That suite was not re-run for this kickoff.

Ziplime is not a dependency. `pyproject.toml` pins `pyarrow>=24` and `yfinance>=1.5.1`.

### quant-lab pull request #1 (open, not merged)

[PR #1](https://github.com/BQ13333/quant-lab/pull/1), "Lab v3 Eternal Alpha". Branch `lab-v3-eternal-alpha`, head `f13073b1d1344beb9d7d9b0deedb44dd7dff84ef`, base `main`. Opened 16 November 2026, 19:29 MYT. Still open on 20 November. GitHub reports it mergeable. 13 commits, 116 files, +4,785 / −9,489. Not merged. This branch does not contain that diff.

What already landed on the pull request, and must not be rebuilt:

- Validation fixes from 15 November: t-stat uses T, PBO trial count and mid-rank ties, drawdown on day 1, Sortino downside deviation, NaN gates fail closed.
- The `quant_lab.lifecycle` package: states, Bayesian plus CUSUM plus BOCPD decay, a monthly monitor with two clocks, cap composition, a post-trade CNY exposure cap, renewal maths, a discounted-Thompson allocator that cannot become book weights, empirical false-pause calibration, and a factory simulation. The worked example is `examples/decay_example.py`.
- Review fixes (F01, F03, F04, F11, F12, F14, F15, F16) and the caller updates those fixes required.
- Approval, simulated ramp, feasibility, strategy cards, book risk, trial snapshots, a stricter research firewall, and a CPCV embargo.
- The later C-items except a blocking Holm gate. Holm is recorded here and implemented in the private repo. See the decisions already accepted below.
- `quant_lab.markets.rules`: fail-closed, source-dated costs for MY, CN, and HK only. Northbound China charges that are not verified are not invented. `LabCostContract` refuses any stage past paper.
- `quant_lab.paper`: a simulated engine. It is not a broker.
- Admission guards that reject US and JP at the signal factory and the research gate (`tests/test_scope.py`).
- US-only index, universe, data-quality, and NASDAQ halt modules removed from the branch. They were moved to a local archive outside git, not deleted. Shared helpers `usa_tca_calibration` and `usa_exchange_calendar_tca` stay, because Japan TCA still imports them. A skipped test records why the US tests are gone.

The pull request body reports the suite after the archive move: 1 failed, 1,320 passed, 78 skipped. The failure is the pre-existing Japan test `tests/test_jpx_monthly_quotations.py::test_runner_is_bounded_resumable_and_partitioned`. It was left unfixed on purpose.

Still open on that pull request:

- Ben has not merged it.
- The Japan quotations test is still red.
- `configs/project_market_scope.yaml` was not updated. The new market-rules code only accepts MY, CN, and HK, while the yaml still activates the United States and still excludes China. Japan code is still in the tree.
- No ziplime work. That input is from 20 November.

### 08-quantitative `main`

Private. The data lake and the alpha archive live here, not in the public framework. Top-level lanes this kickoff confirmed on `main`:

- `MY_Alpha_Lab/` — Malaysia research waves, receipts, and the consolidation note.
- `china_lake/` — China lane. `china_lake/STATE.md` (5 August 2026, research-admitted) records an HK vertical certified first and an A-share lane after it, a point-in-time master, a returns rebuild, a guard layer (bands, inferred ST, suspensions, limit locks, Stock Connect), and a delist rule for terminal suspensions. Known gaps already written down there: official ST flags are inferred, Connect sell-only states cannot be told apart, BSE is absent, and the US ADR layer was deferred. Do not rebuild this lake.
- `CN_Alpha_Lab/`, `Quant_Lib/`, and `signal_factory/` (the canonical lab and campaign S0).
- `CHINA_LAB_RULES.md` rule 8 (18 August 2026): each lane is measured in its own currency. A-shares stay CNY. Hong Kong stays HKD. No FX conversion inside research. Do not revive the old pooled `data_cn` lane.

Hong Kong is a vertical inside the China lane, not a separate top-level folder on `main`. Quant-lab `main` also has HKEX listing discovery (Yahoo `.HK`) and SSE discovery (Yahoo `.SS`). Those public adapters are smoke paths. They are not the private lake.

### 08-quantitative pull request #1 (open, not merged)

[PR #1](https://github.com/BQ13333/08-quantitative/pull/1), "Lab v3 Eternal Alpha (re-applied on full repo)". Branch `lab-v3-eternal-alpha`, head `30816df3a9a3830328437b27b6e9e2113889069b`. Base is `outbox/box` at `a5b153ca`, not `main`. Opened 16 November 2026, 19:31 MYT. Still open. GitHub reports it mergeable. 3 commits, 14 files, +705 / −23. The pull request does not report a test count. This kickoff did not re-run that suite and did not clone the private tree.

What landed:

1. `eb58f83` — unused imports, distinct census invalid keys, scanner mid-line and JSON secret findings, CPCV PBO mid-rank ties, plus regression tests.
2. `df4ad0b` — Astra findings F08, F18, and F19 fail closed, plus regression tests.
3. `30816df` — Holm gate contract v2 in `signal_factory/canonical_lab/campaign_s0/src/campaign_s0/holm_gate_v2.py`. Report-only. It does not set `blocked` and it does not replace v1 `evaluate_gate1`.

The pull request body still says Holm was unbuilt. That sentence is stale. The third commit is on the branch. Do not build a second Holm gate in quant-lab.

The diff does not touch `china_lake/` or `MY_Alpha_Lab/`.

### Decisions already accepted on 16 November

Recorded in `quant_lab.lifecycle.registered_decisions` on quant-lab PR #1. Do not reopen them in this kickoff:

- C17 is Holm step-down, report-only, not blocking, and not Benjamini–Yekutieli. The code is in the private repo.
- Working charter numbers: MYR floors 1,485,000 / 756,250 / 312,500, active vol 0.0515, information ratio 0.29, Bursa ticket commission 0.42% with a MYR 12 floor.
- False pause: at most 20% per sleeve over 60 months. The family rate is report-only.
- Live-history costs enter only as PRIMARY or CAPTURED. VENDOR_PIT is a later option and is not admissible yet.
- No spend that week.

### What not to redo

- Lifecycle, decay maths, ramp, caps, firewall, strategy cards, paper engine, and the MY/CN/HK cost rules on quant-lab PR #1.
- The US-module archive on that same pull request. The modules are still on `main` until that pull request merges.
- Holm v2, the census and scanner fixes, and F08/F18/F19 in the private pull request.
- The China lake admission, the PIT master, and the Malaysia wave archive.
- A second backtester inside quant-lab. The lab keeps its own validation. An external engine is a second opinion behind a file boundary.

## 2. Workstreams

| Workstream | Already in place | Do not redo | Next, after the decisions below |
| --- | --- | --- | --- |
| Data lake and point-in-time | Private China lane admitted August 2026, with an HK vertical and an A-share lane. Malaysia waves in `MY_Alpha_Lab`. Public repo has PIT contracts, a Malaysia pilot that is explicit about missing delistings, and HKEX/SSE discovery. | Re-extracting `china_lake`. Treating Yahoo `.KL` as a full Bursa history. | A coverage sheet for names already on disk: listing date, delist date, corporate actions present or explicitly absent. No new paid feed. |
| Research and signals | Signal factory and Malaysia strategy registry on `main`. Strategy cards on PR #1. | A new model zoo. | Keep fundamental and technical ideas on data the lake already has. New China research waits on the scope-file decision. |
| Validation and decay | Firewall, CPCV/PBO, deflated Sharpe on `main`. Decay monitor, false-pause calibration, and Holm-as-report on the two pull requests. | Re-deriving the decay update or a blocking Holm gate. | After PR #1 merges, run the existing decay example and leave the Japan quotations failure as a separate fix. |
| Backtest engines, including the ziplime pilot | The lab's own research backtests and, on PR #1, a simulated paper engine. | Vendoring ziplime, importing it, or pointing it at a broker. | Task 1, started here: a file contract. A separate-environment smoke is task 3 and waits on the licence decision. |
| Risk and portfolio | Cost schedules and optimizers on `main`. Book risk, exposure caps, and lot feasibility on PR #1. China and Hong Kong stay in local currency. | A live risk service, or an FX-pooled China book. | Use the PR #1 cost rules once they are on `main`. This kickoff does not copy them. |
| Infra and CI | `pytest` via `pyproject.toml`. No GitHub Actions workflow in quant-lab. | A deploy pipeline, a broker connection, or a lockfile churned to add ziplime. | Add CI only after the two PR #1s are merged or explicitly deferred, so the workflow tests one tree. |

## 3. Milestones

Dates are Asia/Kuala_Lumpur. The window is four weeks from today.

| Date | Milestone |
| --- | --- |
| Fri 20 Nov 2026 | Kickoff written. Task 1 started on a branch off `main`: the external-engine file contract. Neither PR #1 is merged. Ziplime is not installed. |
| Mon 24 Nov 2026 | Ben answers the open decisions, especially the scope file, the ziplime research-only posture, and whether the two PR #1s should be merged as they stand. |
| Fri 27 Nov 2026 | PR #1 review notes are either accepted or explicitly deferred. Nobody rebuilds lifecycle or Holm. The Japan quotations failure stays a named exception unless Ben asks for that fix. |
| Fri 4 Dec 2026 | Task 2, if the scope decision allows it: a point-in-time coverage sheet for Bursa and HKEX names already in the lake. China rows stay listed as blocked until that decision says otherwise. No new vendor pull. |
| Fri 11 Dec 2026 | Task 3, only if the licence decision is yes: one Bursa daily name through the file contract in a separate environment. If the decision is no, write that ziplime will not be used and stop. |
| Fri 18 Dec 2026 | The lab's own validation still has to accept or reject any external result. One paper sleeve can be discussed against the PR #1 decay monitor. No orders. |

## 4. Open decisions

Answered 20 November 2026, 18:26 MYT. See the block at the top of this file and [docs/DECISIONS_2026-11-20.md](DECISIONS_2026-11-20.md). The questions are kept so this kickoff still shows what was open that morning. They do not block the file contract, which admits no orders and changes no lake.

1. **Scope file.** `configs/project_market_scope.yaml` and the public data-source docs still exclude China and still activate Singapore, Indonesia, Japan, Korea, and the United States. The 20 November direction, the private China lane, and the unmerged market-rules code are Malaysia, China, and Hong Kong. Should the public scope file be edited to that mandate, and should Singapore, Indonesia, Japan, Korea, and US extraction freeze? Inside China, the private lane already treats A-shares and Hong Kong as separate currency verticals and records BSE as absent. Confirm that BSE stays out.
2. **Ziplime posture.** May a research-only pilot run in a separate environment, exchanging files with quant-lab, under the licence flag in the next section? Until this is a yes, nobody installs it. Live trading and `lime-trader-sdk` are out either way.
3. **The two pull requests.** Quant-lab PR #1 targets `main` and still has the Japan quotations failure. The private PR #1 targets `outbox/box`, not `main`. Confirm both bases, and whether the Japan failure blocks the quant-lab merge.
4. **Japan code and the US archive.** PR #1 removes US modules from git and leaves Japan in the tree, including the shared US fee helpers Japan still imports. The removed copies sit in a local archive outside git. Confirm that this archive is the copy of record, and whether Japan should later be archived the same way or left dormant.
5. **Fundamentals without a paid feed.** Which public filings are in bounds for Bursa, HKEX, and SSE/SZSE. The 16 November no-spend decision still stands. This kickoff does not add a data vendor.

Charter numbers, the 20% false-pause cap, PRIMARY/CAPTURED costs, and Holm-as-report are already accepted. They are not questions.

## 5. The first three tasks

### Task 1 — started

External backtest file contract in this repo.

`src/quant_lab/engines/file_exchange.py` writes a request directory (`request.json`, `bars.csv`) and reads a `result.json`. It admits only MY, CN, and HK numeric symbols (`.KL`, `.SS`, `.SZ`, `.HK`), daily sessions, and a fill on the next session the lab names. It rejects look-ahead bars, same-bar fills, a fill that skips that next session, a missing or non-positive cost, live-trading flags, broker fields, broker-like engine names, and a missing research-only licence line. It does not import ziplime, does not add a dependency, and does not place an order. Board-lot tables stay in PR #1's market rules and are not copied here.

Tests: `tests/test_external_engine_file_exchange.py`.

### Task 2 — next pull request

Point-in-time coverage sheet for identities the lake already has. Decision 1 now allows China A-share rows. Official filings only.

For Bursa, HKEX, and China A-share names already on disk, record listing date, delist date, and whether a corporate-action row exists. A name marked inactive with no delist date fails closed. BSE stays out. Fundamentals on the sheet, if any, are Bursa, HKEXnews, or CNINFO only. No new download, no paid source, no change to `china_lake`. Reviewable as a schema plus a reader and a fixture, in a later pull request.

### Task 3 — allowed, lowest priority, not started in the scope pull request

One-name Bursa smoke of the optional engine, in a separate environment.

Decision 2 is a conditional yes. Run `ziplime==2.12.0` outside this repo's environment, feed it one daily `.KL` request written by task 1, and bring `result.json` back. The lab admits that file only if task 1's reader accepts it. Do not install ziplime into quant-lab, do not import `lime-trader-sdk`, and do not enable same-bar fills. The licence is still unclear. If the separate environment cannot be built without spend, secrets, or accepting that licence, stop and write down the blocker. Do not do it ahead of the scope record or the coverage sheet.

## Ziplime licence flag

Research only. Do not vendor it. Do not add it to `pyproject.toml`. Do not install it into the quant-lab environment. Do not wire its execution engine or `lime-trader-sdk`.

What was checked on 20 November 2026, without installing the package:

- PyPI `ziplime==2.12.0` has an empty licence field. It requires Python `>=3.12,<4`, `pyarrow>=19,<20`, `yfinance>=0.2.65,<0.3`, and `polars==1.44.2`. Those pins conflict with quant-lab's `pyarrow>=24` and `yfinance>=1.5.1`. That is why the environments stay separate and talk only through files.
- The `LICENSE` file on `Limex-com/ziplime` `master` is the GNU GPL-3.0 text.
- `END_USER_LICENSE_AGREEMENT` says the grant is "consistent with" GPL-3.0 and then restricts commercial use and redistribution. Those extra restrictions are hard to reconcile with GPL-3.0. This kickoff is not a legal opinion. It is a product control: research only, no commercial deployment, no live trading.
- The README badge and licence section say AGPL-3.0, and they still contain a note describing the licence-file swap as a change to make. Releases up to 1.19.x are described there as remaining GPL-3.0. 2.12.0 is newer than that, and the file on `master` is still the GPL text. Treat the licence as unclear.
- A code search of the default branch for Bursa, XKLS, KLSE, and Kuala Lumpur returned no hits. The 20 November brief says Yahoo `.KL` and a Bursa calendar. Ziplime depends on `exchange-calendars`, so a calendar could be passed through, but that was not verified here. Task 3 is the verification. It does not start until decision 2.
- The package also depends on `lime-trader-sdk`. That SDK is out of scope permanently for this programme.

Upstream: <https://github.com/Limex-com/ziplime>.

## How this pull request sits next to PR #1

This branch starts from `main`, not from `lab-v3-eternal-alpha`, so the diff is only the kickoff and the file contract. Merge quant-lab PR #1 on its own review. The two diffs do not edit the same files. After both are on `main`, the file contract is the boundary in front of any external engine, and the lifecycle package remains the decay monitor. Neither one places an order.
