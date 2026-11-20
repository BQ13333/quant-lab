# Ziplime Bursa smoke

20 November 2026, Asia/Kuala_Lumpur. Research only. Lowest priority. The separate environment was built. A backtest result was not.

`ziplime==2.12.0` installed from PyPI into a directory outside this repository. Nothing was added to `pyproject.toml`. No API key, no credentials file, and no cloud account. The install pulled `lime-trader-sdk==0.4.1` because ziplime depends on it. That SDK was not imported. `get_exchange("lime-trader-sdk")` was not called. It requires `LIME_SDK_CREDENTIALS_FILE`, and that variable was not set.

Importing `ziplime` did not load a `lime_trader` module. The only registered research data provider is Yahoo, and its registry entry says it needs no credentials. Yahoo was not called. The file contract is the only data path this smoke uses.

`exchange_calendars` in that environment has `XKLS`. 2, 3, 4 and 5 January 2024 are sessions on that calendar. That is the Bursa calendar check. It is not a backtest.

The lab wrote one request for `1155.KL` under `tests/fixtures/ziplime_bursa_1155/`. The bar is a synthetic print, not a downloaded price. The decision session is 2024-01-02 and the next session is 2024-01-03. The one-way cost is 42 basis points, the 16 November charter rate, not a broker quote. There is no `result.json`.

Why the engine did not run:

- The shipped research connector reads Yahoo over the network. That is not the file contract.
- The custom-bundle ingest resolves symbols with MIC `XNGS`, not `XKLS`.
- `run_algorithm` defaults `same_bar_execution` to true. The file contract rejects a same-bar fill.
- Completing a live or broker path needs the Lime SDK credentials. Those were not created.

The licence is still unclear. The upstream `LICENSE` file is GPL-3.0. The end-user agreement adds a commercial-use restriction. The README says the project is moving to AGPL. This smoke is not a legal clearance and it is not a commercial deployment.

No order was sent.
