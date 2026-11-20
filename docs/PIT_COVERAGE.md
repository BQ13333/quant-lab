# Point-in-time coverage

20 November 2026, Asia/Kuala_Lumpur. This is the coverage sheet for names already committed in quant-lab. It does not download a filing and it does not copy the private lake.

The reader is `quant_lab.data.pit_coverage`. The sheet is [`configs/pit_coverage_on_disk.csv`](../configs/pit_coverage_on_disk.csv).

What is on disk here:

- Thirty Bursa stocks in `src/quant_lab/data/asean_universe.py`. That module says it is a current-survivor starter list, not a point-in-time constituent database.
- `0700.HK` in `src/quant_lab/data/refinitiv_symbol_map.py`. The file is a vendor symbol map. It is not an HKEXnews filing.
- `600000.SS` and `600004.SS` in `docs/HISTORICAL_DATA_LAKE_BACKFILL.md`, inside a China smoke note that says not to promote that artifact.

Each of those rows is blocked. The listing date is blank, and corporate actions are not marked present or explicitly absent. An inactive name with no delist date would also be blocked. No row claims a Bursa, HKEXnews, or CNINFO filing, because none of those documents is in this tree.

The private `08-quantitative` lake is not this checkout. `china_lake/STATE.md` on that repository's `main` describes a large admitted lake, including an HK vertical and an A-share lane. Those tables were not copied into this public pull request, and this sheet does not invent their listing dates. BSE stays out.

A row can be admitted only when the market is Bursa, SSE, SZSE, or HKEX, the listing date is a real date, an inactive name has a delist date, corporate actions are `present` or `explicitly_absent`, and any fundamental source is `bursa_announcements`, `hkexnews`, or `cninfo` with a pointer. `absent` means no fundamental is claimed. Yahoo, Refinitiv, and other paid or scrape feeds are refused as fundamentals.
