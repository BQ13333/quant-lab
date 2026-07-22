# Execution Safety

This project is still a research and paper-trading system. Live trading is not
enabled by default.

## Execution Stages

1. Research backtest
   - Run model tournaments.
   - Save artifacts, costs, validation metrics, and Obsidian notes.
   - Do not generate broker orders.

2. Paper order generation
   - Convert approved target weights into `OrderIntent` objects.
   - Enforce per-order and batch risk limits.
   - Submit through `PaperBroker` or IBKR paper mode only.

3. IBKR paper dry-run
   - Use `IBKRConfig(trading_mode="paper", port=7497, transmit_orders=False)`.
   - The adapter validates orders but does not transmit when disconnected.
   - With IBKR TWS/Gateway connected and `transmit_orders=True`, orders are
     paper-account orders.

4. Live trading
   - Blocked unless `trading_mode="live"` and `allow_live_trading=True`.
   - Should remain disabled until corrected cost accounting, promotion gates,
     paper fills, slippage, drawdown limits, and manual review all pass.

## Guardrails

- `ExecutionRiskLimits` caps order quantity, order notional, batch notional, and
  allowed tickers.
- `targets_to_order_intents` skips tiny trades and rounds to whole shares.
- `IBKRBroker` rejects known live ports when configured for paper mode.
- `IBKRBroker` rejects live mode unless explicitly authorized.
- `transmit_orders=False` by default, even for paper mode.

## Current Status

The research engine can create candidate strategies and target weights. The
execution layer can create and validate paper orders. A production OMS is not
yet implemented: no durable broker state reconciliation, kill switch service,
latency/TCA monitor, position drift repair, or compliance approval queue exists.
