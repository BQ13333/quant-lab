from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_lab.execution.broker import (
    ExecutionRiskLimits,
    OrderIntent,
    OrderResult,
    TradeSafetyError,
    validate_order_batch,
    validate_order_intent,
)


TradingMode = Literal["paper", "live"]


class LiveTradingBlocked(TradeSafetyError):
    """Raised when a live-trading configuration is not explicitly authorized."""


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 71
    trading_mode: TradingMode = "paper"
    account: str | None = None
    transmit_orders: bool = False
    allow_live_trading: bool = False


class IBKRBroker:
    """Thin Interactive Brokers adapter with paper-first guardrails.

    This class intentionally keeps `ib_insync` optional. Import and connection
    happen lazily so research/backtest environments do not need broker
    dependencies installed.
    """

    PAPER_PORTS = {7497, 4002}
    LIVE_PORTS = {7496, 4001}

    def __init__(
        self,
        config: IBKRConfig | None = None,
        limits: ExecutionRiskLimits | None = None,
    ) -> None:
        self.config = config or IBKRConfig()
        self.limits = limits or ExecutionRiskLimits()
        self._ib = None
        self.validate_config()

    def validate_config(self) -> None:
        if self.config.trading_mode == "live" and not self.config.allow_live_trading:
            raise LiveTradingBlocked("live trading requires allow_live_trading=True")
        if self.config.trading_mode == "paper" and self.config.port in self.LIVE_PORTS:
            raise LiveTradingBlocked("paper mode cannot use a known live IBKR port")
        if self.config.trading_mode == "live" and self.config.port in self.PAPER_PORTS:
            raise LiveTradingBlocked("live mode cannot use a known paper IBKR port")

    def connect(self) -> None:
        self.validate_config()
        try:
            from ib_insync import IB
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("Install the execution extra to use IBKR: pip install 'quant-lab[execution]'") from exc

        ib = IB()
        ib.connect(self.config.host, self.config.port, clientId=self.config.client_id)
        self._ib = ib

    @property
    def is_connected(self) -> bool:
        return bool(self._ib is not None and self._ib.isConnected())

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():  # pragma: no cover - broker dependent
            self._ib.disconnect()

    def submit_order(self, intent: OrderIntent, reference_price: float | None = None) -> OrderResult:
        self.validate_config()
        validate_order_intent(intent, reference_price, self.limits)

        if not self.is_connected:
            return OrderResult(
                intent=intent,
                status="DRY_RUN",
                filled_quantity=0.0,
                fill_price=None,
                message="IBKR not connected; order validated but not transmitted",
            )

        from ib_insync import LimitOrder, MarketOrder, Stock

        contract = Stock(intent.ticker, "SMART", "USD")
        if intent.order_type == "LMT":
            order = LimitOrder(intent.side, intent.quantity, intent.limit_price)
        elif intent.order_type == "MKT":
            order = MarketOrder(intent.side, intent.quantity)
        else:
            raise TradeSafetyError(f"unsupported order type: {intent.order_type}")

        order.account = intent.account or self.config.account or ""
        order.transmit = self.config.transmit_orders

        trade = self._ib.placeOrder(contract, order)
        status = "ACCEPTED" if self.config.transmit_orders else "DRY_RUN"
        return OrderResult(
            intent=intent,
            status=status,
            broker_order_id=str(getattr(order, "orderId", "")) or None,
            filled_quantity=0.0,
            fill_price=None,
            message=f"IBKR order {'transmitted' if self.config.transmit_orders else 'created with transmit=False'}",
        )

    def submit_orders(self, intents: list[OrderIntent], reference_prices: pd.Series) -> list[OrderResult]:
        validate_order_batch(intents, reference_prices, self.limits)
        return [
            self.submit_order(intent, float(reference_prices.get(intent.ticker, float("nan"))))
            for intent in intents
        ]
