# Historical costs and Hong Kong board lots

20 November 2026, Asia/Kuala_Lumpur. This is a reconciliation of records already in the repositories. It does not certify a fee, and it does not invent a lot.

The cost engine and `src/quant_lab/markets/rules_v1.json` are on unmerged [quant-lab PR #1](https://github.com/BQ13333/quant-lab/pull/1) at `f13073b1d1344beb9d7d9b0deedb44dd7dff84ef`. This branch does not copy that engine. The inventory in [`configs/historical_cost_coverage.json`](../configs/historical_cost_coverage.json) starts from that file and replaces capture dates with official effective dates where a public page states one.

What the inventory contains after the 20 November pass:

- Malaysia stamp duty is still 13 July 2023 through 12 July 2028, from P.U.(A) 208/2023. Clearing at 0.03% capped at RM1,000, the normal board lot of 100, and the 0.5 sen tick below RM1 are on the Bursa pages read that day. Those pages do not state a start date, so `effective_from` is null. The old 16 November 2026 date was a capture date and has been removed.
- Hong Kong stamp duty is dated from the Inland Revenue Department rate sheet and the 2023 HKEX circular: 0.1% from 1 September 2001 through 31 July 2021, 0.13% from 1 August 2021 through 16 November 2023, and 0.1% from 17 November 2023. The trading fee of 0.00565% starts 1 January 2023 (SEHK circular 086/22). The SFC levy of 0.0027% starts 1 November 2014. The AFRC levy of 0.00015% starts 1 January 2022. Settlement still has no start date. The market-wide board lot stays null.
- China stamp duty of 0.05% still starts 28 August 2023. The SSE handling fee is 0.00341% of consideration, both sides, from that same day; SZSE states the same change. ChinaClear's chronology puts the stock transfer fee at 0.02 per mille to each side from 1 August 2022. The SZSE regulatory fee of 0.02 per mille is on the January 2026 table and is undated. The ST band is still secondary. The main-board lot of 100 and the sell delay still cite a Stock Connect booklet.
- The working Bursa commission of 0.42% with a MYR 12 minimum is the 16 November charter. It is not a broker quote.

`historical_cost_gate_met()` is false. The gate value is `partial`. Dated intervals are not written backwards into years the source does not cover. A null component is not filled with zero.

## Hong Kong lots

[`configs/hk_board_lots.csv`](../configs/hk_board_lots.csv) is the per-instrument table. `0700.HK` is 100 shares from 17 August 2009. HKSCC circular ce_136_2009 says the lot changed from 200 shares to 100 that day. The 9 June 2023 HKEXnews announcement for the RMB counter still states a board lot of 100 for 00700. The known time on the row is 3 August 2009, the date of the HKEX news release of the same change. The circular does not date the start of the old 200-share lot, so that earlier interval is not invented.

`0005.HK` is on the coverage sheet and has no sourced lot. [`configs/hk_board_lot_status.json`](../configs/hk_board_lot_status.json) records the gate as `partial` and the market-wide default as null. `lot_size` returns 100 for `0700.HK` on and after 17 August 2009 and raises for `0005.HK`, for a date before 17 August 2009, and for any other sid. It does not substitute a default. No spend follows from that.
