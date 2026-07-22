from __future__ import annotations

import pandas as pd

from quant_lab.execution import (
    ExecutionRiskLimits,
    IBKRBroker,
    IBKRConfig,
    PaperTradingLedger,
    PaperBroker,
    targets_to_order_intents,
)


def main() -> None:
    prices = pd.Series({"SPY": 550.0, "EFA": 85.0, "EEM": 45.0})
    current_positions = pd.Series({"SPY": 5.0, "EFA": 0.0, "EEM": 0.0})
    target_weights = pd.Series({"SPY": 0.20, "EFA": 0.15, "EEM": 0.10})

    limits = ExecutionRiskLimits(
        max_order_notional=25_000.0,
        max_batch_notional=50_000.0,
        allowed_tickers=frozenset(prices.index),
    )
    intents = targets_to_order_intents(
        target_weights=target_weights,
        current_positions=current_positions,
        equity=100_000.0,
        prices=prices,
        strategy_id="global_lite_demo",
        limits=limits,
    )

    paper = PaperBroker(limits=limits)
    fills = paper.submit_orders(intents, prices)
    ledger = PaperTradingLedger("artifacts/paper_trading/demo")
    ledger.record_intents("paper-demo", intents)
    ledger.record_results("paper-demo", fills)
    print("--- Local paper fills ---")
    for fill in fills:
        print(f"{fill.broker_order_id}: {fill.intent.side} {fill.intent.quantity:.0f} {fill.intent.ticker} @ {fill.fill_price}")

    ibkr = IBKRBroker(
        config=IBKRConfig(
            trading_mode="paper",
            port=7497,
            transmit_orders=False,
        ),
        limits=limits,
    )
    print()
    print("--- IBKR dry-run validation ---")
    for intent in intents:
        result = ibkr.submit_order(intent, reference_price=float(prices[intent.ticker]))
        print(f"{intent.side} {intent.quantity:.0f} {intent.ticker}: {result.status} - {result.message}")
    print()
    print("Ledger:", ledger.ledger_path)


if __name__ == "__main__":
    main()
