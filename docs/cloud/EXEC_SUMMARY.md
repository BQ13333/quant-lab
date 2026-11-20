# Quant House cloud lab — executive summary

20 November 2026 · Asia/Kuala_Lumpur (MYT) · Benjamin Quek.

Reconciled after Astra's draft, Fable's red-team and one Astra fix pass. Source aliases and full arithmetic are in the [full blueprint](CLOUD_PRODUCTION_BLUEPRINT.md), §11. Architecture: [architecture.png](architecture.png).

**Recommendation [A]:** Keep Parquet + DuckDB + systemd + Healthchecks + public GitHub Actions and both existing repositories. Phase 1 runs on the existing box at **USD0 incremental**, within current entitlements. After feasibility and GO, try **Oracle A1 Always Free** first (home-region/out-of-host-capacity risk), then **Hetzner CX23** (4 GB/40 GB, FI/DE, stock/reachability risk). **Lightsail 8 GB only after written reachability, stock or RAM failures on both alternatives**, plus workload replay and approval. Use **R2 primary + an independent Backblaze B2 full retained copy**. [S prices: F/G and executor additions; design A]

**MY/CN/HK, lab only [S: B].** No live/broker/order-routing code, crypto, US, HFT, paid feeds or paid model APIs. Subscription drafting/review stays separate from deterministic runtime jobs.

| Modeled monthly services [A] | Low: A1 | Base: CX23 | High: Lightsail 8 GB |
|---|---:|---:|---:|
| Compute USD | 0.00 | 7.09 | 44.00 |
| R2 + B2 USD | 1.98 | 5.27 | 21.73 |
| R2 operations USD | 0.00 | 0.00 | 4.86 |
| **Subtotal USD** | **1.98** | **12.36** | **70.59** |
| **Subtotal RM** | **8.14** | **50.83** | **290.30** |

[S] FX: **4.1125**, [BNM KL reference rate](https://financialmarkets.bnm.gov.my/data-download-exchange-rates), 20 Nov 2026. [A] Versus 4.10, the RM increases are 0.02/0.15/0.88. Footnote: F's separate BNM 17:00 mid is 4.1138 (buy 4.1120/sell 4.1155), not used. [A] Retained footprint is 100/250/1,000 GB **per provider**, still planning headroom: the ~60 GB packed inventory includes possible US overlap and is not a deduplicated in-scope measurement. Tax/VAT, B2 transactions, excess transfer, card spread and existing subscriptions remain [U], excluded; these are not all-in spending caps. Full arithmetic/source links are in §3.

**Spend stays held [A].** Require Phase 1 exit, at least one market's historical cost intervals and usable PIT price/universe evidence, a dated per-instrument HK lot table, a named human root holder/deploy path, and a successful probe from the candidate's actual egress IP. If that probe requires paid provisioning, hold until a no-charge test arrangement exists. Recommend Ben as root/key custodian with human-only manifest deployment; agents cannot obtain root or rewrite trust anchors. There is no USD65/month approval request.

**Four phases [A]:** (1) establish scope, source rights, inventory, historical feasibility and trust ownership on the box; (2) after the gates and GO, provision and checksum-restore; (3) prove shadow operation, independent B2 recovery, writer fencing and calendar-edge replay; (4) admit a qualifying simulated sleeve. No qualifying sleeve is acceptable.

**Corrected lifecycle [A]:** Allow existing simulated RAMP/KEEP/REDUCE/PAUSE states while a constant-stage adapter always passes `'PAPER'` to `LabCostContract`. Required integration tests prove allocation, reductions/pauses, cost checks and replay; absent the adapter, admission blocks. Cut the ten-recipes/week quota; use registered family-wide trial budgets and existing DSR/selection gates. **One monthly cycle does not establish monitor power.**

**Controls [A]:** Public hosted CI USD0; GitHub Free private included compute USD12 ≈ 2,000 Linux minutes, account-wide [S: F/G]. Set USD0 Actions spending limit; no public self-hosted runner, fork-PR secrets or lake artifacts. Use 16 Healthchecks jobs; alert route/responder is a decision. One publisher holds a local lock; manual host shutdown, credential revocation and epoch advance fence takeover. Reboots require operator credential unlock; unattended cold recovery is not promised.

**Five kickoff decisions [A]:** YES to MY/CN/HK scope correction; conditional, lowest-priority isolated Ziplime trial; HOLD PR #1 merges pending Japan-test resolution and private-base GO; preserve US archives/Japan dormant; official filings without paid feeds, subject to rights. Unknown `china_lake` provenance and uncertified reconstructed vintages cannot support admission. Fable's substantive findings are accepted; no required fix rejected.

Full blueprint: [CLOUD_PRODUCTION_BLUEPRINT.md](CLOUD_PRODUCTION_BLUEPRINT.md). This document does not authorize accounts, spend, deployment, repo edits, pushes or merges. [A; scope S: user instruction]
