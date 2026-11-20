# Decisions, 20 November 2026, 18:26 MYT

Product owner: Benjamin Quek. Times are Asia/Kuala_Lumpur.

These decisions close the five questions in [PROJECT_KICKOFF.md](PROJECT_KICKOFF.md). They do not merge a pull request, open a cloud account, or approve spend.

The target architecture is the copied blueprint, not a second design:

- [docs/cloud/EXEC_SUMMARY.md](cloud/EXEC_SUMMARY.md)
- [docs/cloud/CLOUD_PRODUCTION_BLUEPRINT.md](cloud/CLOUD_PRODUCTION_BLUEPRINT.md)
- [docs/cloud/architecture.png](cloud/architecture.png)

## 1. Scope

The public scope file is now Malaysia, China A-shares, and Hong Kong only.

| Status | Markets |
| --- | --- |
| Active | Bursa Malaysia. SSE and SZSE A-shares. HKEX. |
| Frozen | Singapore, Indonesia, Japan, Korea, and the United States. Code may stay in the tree. New extraction, admission, and research do not use them. |
| Out | Beijing Stock Exchange. Taiwan, Thailand, the Philippines, and Vietnam stay outside the mandate. |

`configs/project_market_scope.yaml` is effective 20 November 2026. `quant_lab.programme.scope` reads it and fails closed. Dormant parsers and old fixtures stay so existing tests still describe the code that is already in the tree. They are not an active queue. The 12 July region tables in `docs/DATA_SOURCE_ROADMAP.md` and `docs/HISTORICAL_DATA_LAKE_BACKFILL.md` are historical. The banners on those files say so.

## 2. Ziplime

[Ziplime](https://github.com/Limex-com/ziplime) `2.12.0` is allowed only as a research-only sandbox. It stays in a separate environment. It exchanges files with this repository through the contract already on pull request #2. It is not a dependency of quant-lab. It is the lowest priority of the three tasks.

The licence is still unclear. The upstream `LICENSE` file is GPL-3.0. The end-user agreement adds a commercial-use restriction. The README says the project is moving to AGPL. PyPI's licence field is empty. This note is not a legal opinion and it is not permission for commercial or fund use.

`lime-trader-sdk` stays unwired. Nothing in this programme places a live order. Installing ziplime into the quant-lab environment, importing it from `quant_lab`, or adding it to `pyproject.toml` is still forbidden.

## 3. Both pull requests #1 stay unmerged

| Pull request | Base branch | Head | State |
| --- | --- | --- | --- |
| [quant-lab #1](https://github.com/BQ13333/quant-lab/pull/1) | `main` | `f13073b1d1344beb9d7d9b0deedb44dd7dff84ef` | Open. Not merged. |
| [08-quantitative #1](https://github.com/BQ13333/08-quantitative/pull/1) | `outbox/box` at `a5b153ca614c45a468407745f91572342927163c` | `30816df3a9a3830328437b27b6e9e2113889069b` | Open. Not merged. |

Confirmed from the GitHub pull request objects on 20 November 2026. The private repository's default branch is `main`. Pull request #1 does not target `main`. It targets `outbox/box`. This work does not retarget it.

Merges wait for Ben's explicit GO.

### Japan quotations test

Test id: `tests/test_jpx_monthly_quotations.py::test_runner_is_bounded_resumable_and_partitioned`

The file blob `945cd235dd054f59c34c7b165422d23532c9d936` is the same on `main` and on quant-lab pull request #1. Japan stays in-tree and dormant. The fix below does not add Japan to `active_markets`, and this branch does not push onto `lab-v3-eternal-alpha`.

Re-run on 20 November 2026 with the locked stack, pandas 2.3.3 and numpy 2.4.6 from `uv.lock`: the five tests in `tests/test_jpx_monthly_quotations.py` passed. The 16 November receipt on pull request #1 (1 failed, 1,320 passed, 78 skipped, this test the failure) was not reproduced on that stack.

The resume path does fail when `quality_pass` is a pandas string column. `_invalidate_low_denominator_combined` assigned a boolean into that column. CSV reloads on pandas 3 use that string dtype for `dtype=str`, and an explicit string dtype fails on pandas 2.3.3 as well. This branch stores a text flag when the column is not boolean. The new test `test_combined_file_invalidation_accepts_a_string_quality_flag` locks that. There is no skip marker and no named exception. The test stays in the suite. Pull request #1's head was not edited, so a checkout of that head without this commit still has the old assignment.

## 4. US archive and Japan

The US archive outside git is the copy of record. Japan stays in-tree and dormant.

Recorded locations, from the cloud blueprint's sources, not re-checked on this machine:

- US code archive: `/workspace/quant_house/_archive_not_in_use/`
- US data archive on the Mac: `/Users/benjyq/Downloads/Investment_Banking_MD/_archive_not_in_use/`, reported at 151 GB

This pull request does not copy, delete, or move those trees. It also does not repeat pull request #1's removal of US modules from git. Until that pull request merges, those modules are still on `main`. The decision is about which copy is the record, not a second deletion. Shared helpers that Japan still imports stay, because Japan code stays in the tree. Frozen means no new US or Japan extraction.

## 5. Fundamentals

Official public filings only: Bursa company announcements, HKEXnews, and CNINFO. Subject to access rights. No paid feeds. Detail is in [SOURCE_RIGHTS.md](SOURCE_RIGHTS.md).

## Costs, lots, and root custody

Historical cost intervals and the Hong Kong lot table are reconciled in [HISTORICAL_COST_AND_LOTS.md](HISTORICAL_COST_AND_LOTS.md). Both gates are `not_met`. That blocks cloud spend. It does not block this documentation pull request.

## Still open

- Ben must name the human root-account holder. The blueprint recommends Benjamin Quek and says that recommendation is not acceptance. No name is recorded here as accepted.
- Phase 2 of the blueprint (provisioning) needs Ben's separate GO.
- No cloud account was created. No money was spent.

Charter numbers, the 20% false-pause cap, PRIMARY or CAPTURED costs, and Holm-as-report were already accepted on 16 November. They are not reopened.
