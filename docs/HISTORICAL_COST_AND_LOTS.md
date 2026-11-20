# Historical costs and Hong Kong board lots

20 November 2026, Asia/Kuala_Lumpur. This is a reconciliation of records already in the repositories. It does not certify a fee, and it does not invent a lot.

The cost engine and `src/quant_lab/markets/rules_v1.json` are on unmerged [quant-lab PR #1](https://github.com/BQ13333/quant-lab/pull/1) at `f13073b1d1344beb9d7d9b0deedb44dd7dff84ef`. This branch does not copy that engine. The inventory in [`configs/historical_cost_coverage.json`](../configs/historical_cost_coverage.json) quotes the intervals from that file so a reviewer can see the gaps without opening the other pull request.

What that file actually contains:

- Malaysia stamp duty has an interval, 13 July 2023 through 12 July 2028, from the PUA 208/2023 notice. Clearing, the board lot of 100, and the tick below 1 ringgit start on 16 November 2026 only.
- Hong Kong stamp duty of 0.1% starts 17 November 2023, from the Inland Revenue Department notice. Trading, SFC, AFRC, and settlement components start on 16 November 2026 only. `board_lot` is null, source kind `NONE`.
- China stamp duty of 0.05% starts 28 August 2023, from the Ministry of Finance notice. Handling, management, and transfer rates are null. The ST band is marked secondary. The main-board lot of 100 and the one-session sell delay cite an HKEX Stock Connect booklet, which does not establish a direct SSE or SZSE rule.
- The working Bursa commission of 0.42% with a MYR 12 minimum is the 16 November charter. It is not a new broker quote.

`historical_cost_gate_met()` is false. No market on this inventory has a complete source-dated cost stack over a multi-year backtest. Current rates are not written backwards into earlier years. A null component is not filled with zero.

## Hong Kong lots

[`configs/hk_board_lots.csv`](../configs/hk_board_lots.csv) is the per-instrument table. Its columns are the stable id, exchange, instrument code, effective interval, known time, lot size, source URL, and source kind. It has a header and no instrument rows. [`configs/hk_board_lot_status.json`](../configs/hk_board_lot_status.json) records the gate as `not_met` and the market-wide default as null.

`lot_size(sid, session)` raises `HkLotUnavailable` when that row is missing. It will not substitute 100, or any other default. A search of this repository and of the private `08-quantitative` code index did not turn up a dated per-instrument HK lot table to copy. Phase 1 of the cloud blueprint asked for that table before spend. The table is started, and the gate is open because the rows are not sourced. No spend follows from that.
