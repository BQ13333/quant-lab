from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.execution import (
    ExecutionRiskLimits,
    IBKRBroker,
    IBKRConfig,
    LiveTradingBlocked,
    OrderIntent,
    PaperTradingLedger,
    PaperBroker,
    TradeSafetyError,
    targets_to_order_intents,
)


def test_targets_to_order_intents_and_paper_fills():
    prices = pd.Series({"SPY": 500.0, "EFA": 80.0})
    target_weights = pd.Series({"SPY": 0.20, "EFA": 0.10})
    current_positions = pd.Series({"SPY": 10.0, "EFA": 0.0})
    limits = ExecutionRiskLimits(max_order_notional=25_000.0, max_batch_notional=50_000.0)

    intents = targets_to_order_intents(
        target_weights=target_weights,
        current_positions=current_positions,
        equity=100_000.0,
        prices=prices,
        limits=limits,
    )

    assert [intent.ticker for intent in intents] == ["SPY", "EFA"]
    assert intents[0].side == "BUY"
    assert intents[0].quantity == 30

    broker = PaperBroker(limits=limits)
    fills = broker.submit_orders(intents, prices)

    assert len(fills) == 2
    assert all(fill.status == "FILLED" for fill in fills)
    assert fills[0].broker_order_id == "PAPER-000001"


def test_order_notional_limit_blocks_large_order():
    broker = PaperBroker(limits=ExecutionRiskLimits(max_order_notional=1_000.0))
    intent = OrderIntent(ticker="SPY", side="BUY", quantity=10)

    with pytest.raises(TradeSafetyError):
        broker.submit_order(intent, reference_price=500.0)


def test_ibkr_blocks_live_mode_by_default():
    with pytest.raises(LiveTradingBlocked):
        IBKRBroker(config=IBKRConfig(trading_mode="live", port=7496))


def test_ibkr_blocks_paper_mode_on_live_port():
    with pytest.raises(LiveTradingBlocked):
        IBKRBroker(config=IBKRConfig(trading_mode="paper", port=7496))


def test_ibkr_disconnected_submit_is_dry_run():
    broker = IBKRBroker(config=IBKRConfig(trading_mode="paper", port=7497, transmit_orders=False))
    result = broker.submit_order(OrderIntent(ticker="SPY", side="BUY", quantity=1), reference_price=500.0)

    assert result.status == "DRY_RUN"
    assert "not connected" in result.message


def test_paper_trading_ledger_records_orders_and_fills(tmp_path):
    broker = PaperBroker()
    intent = OrderIntent(ticker="SPY", side="BUY", quantity=2, metadata={"target_weight": 0.1})
    result = broker.submit_order(intent, reference_price=500.0)
    ledger = PaperTradingLedger(tmp_path)

    orders = ledger.record_intents("RUN-1", [intent])
    fills = ledger.record_results("RUN-1", [result])

    assert ledger.orders_path.exists()
    assert ledger.fills_path.exists()
    assert orders.loc[0, "metadata_json"] == '{"target_weight": 0.1}'
    assert fills.loc[0, "status"] == "FILLED"
    assert ledger.load_orders().shape[0] == 1
    assert ledger.load_fills().shape[0] == 1
