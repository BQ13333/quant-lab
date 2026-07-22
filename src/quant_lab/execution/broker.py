from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd


OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT"]
OrderStatus = Literal["ACCEPTED", "FILLED", "REJECTED", "DRY_RUN"]


class TradeSafetyError(ValueError):
    """Raised when an order violates execution risk controls."""


@dataclass(frozen=True)
class ExecutionRiskLimits:
    max_order_quantity: float = 100_000.0
    max_order_notional: float = 100_000.0
    max_batch_notional: float = 500_000.0
    min_trade_notional: float = 50.0
    allowed_tickers: frozenset[str] | None = None


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = "MKT"
    limit_price: float | None = None
    strategy_id: str = "quant_lab"
    account: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "BUY" else -self.quantity


@dataclass(frozen=True)
class OrderResult:
    intent: OrderIntent
    status: OrderStatus
    broker_order_id: str | None = None
    filled_quantity: float = 0.0
    fill_price: float | None = None
    message: str = ""
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def validate_order_intent(
    intent: OrderIntent,
    reference_price: float | None,
    limits: ExecutionRiskLimits,
) -> None:
    """Validate one order before it can reach a broker adapter."""

    if not intent.ticker:
        raise TradeSafetyError("ticker is required")
    if intent.side not in ("BUY", "SELL"):
        raise TradeSafetyError(f"unsupported side: {intent.side}")
    if intent.quantity <= 0:
        raise TradeSafetyError("quantity must be positive")
    if intent.quantity > limits.max_order_quantity:
        raise TradeSafetyError(
            f"quantity {intent.quantity:.2f} exceeds max_order_quantity {limits.max_order_quantity:.2f}"
        )
    if intent.order_type == "LMT" and (intent.limit_price is None or intent.limit_price <= 0):
        raise TradeSafetyError("limit orders require a positive limit_price")
    if intent.order_type not in ("MKT", "LMT"):
        raise TradeSafetyError(f"unsupported order type: {intent.order_type}")
    if limits.allowed_tickers is not None and intent.ticker not in limits.allowed_tickers:
        raise TradeSafetyError(f"{intent.ticker} is not in allowed_tickers")

    price = intent.limit_price if intent.limit_price is not None else reference_price
    if price is not None:
        if price <= 0:
            raise TradeSafetyError("reference price must be positive")
        notional = abs(intent.quantity * price)
        if notional > limits.max_order_notional:
            raise TradeSafetyError(
                f"order notional {notional:.2f} exceeds max_order_notional {limits.max_order_notional:.2f}"
            )


def validate_order_batch(
    intents: list[OrderIntent],
    reference_prices: pd.Series,
    limits: ExecutionRiskLimits,
) -> None:
    """Validate a batch of orders and aggregate notional exposure."""

    total_notional = 0.0
    for intent in intents:
        price = float(reference_prices.get(intent.ticker, float("nan")))
        if pd.isna(price):
            price = intent.limit_price if intent.limit_price is not None else None
        validate_order_intent(intent, price, limits)
        if price is not None:
            total_notional += abs(intent.quantity * price)

    if total_notional > limits.max_batch_notional:
        raise TradeSafetyError(
            f"batch notional {total_notional:.2f} exceeds max_batch_notional {limits.max_batch_notional:.2f}"
        )


def targets_to_order_intents(
    target_weights: pd.Series,
    current_positions: pd.Series,
    equity: float,
    prices: pd.Series,
    strategy_id: str = "quant_lab",
    account: str | None = None,
    limits: ExecutionRiskLimits | None = None,
) -> list[OrderIntent]:
    """Convert target weights into share-level order intents.

    Quant research produces target weights, not broker orders. This function is
    the explicit bridge: it compares desired shares to current shares, rounds
    down to whole shares, and skips trades below `min_trade_notional`.
    """

    if equity <= 0:
        raise TradeSafetyError("equity must be positive")

    limits = limits or ExecutionRiskLimits()
    aligned_prices = prices.reindex(target_weights.index).astype(float)
    current = current_positions.reindex(target_weights.index).fillna(0.0).astype(float)

    intents: list[OrderIntent] = []
    for ticker, weight in target_weights.fillna(0.0).items():
        price = aligned_prices.get(ticker)
        if pd.isna(price) or price <= 0:
            continue

        target_shares = int((float(weight) * equity) / float(price))
        delta = target_shares - int(current.get(ticker, 0.0))
        if delta == 0:
            continue

        notional = abs(delta * float(price))
        if notional < limits.min_trade_notional:
            continue

        intent = OrderIntent(
            ticker=str(ticker),
            side="BUY" if delta > 0 else "SELL",
            quantity=float(abs(delta)),
            order_type="MKT",
            strategy_id=strategy_id,
            account=account,
            metadata={"target_weight": float(weight), "reference_price": float(price)},
        )
        validate_order_intent(intent, float(price), limits)
        intents.append(intent)

    validate_order_batch(intents, aligned_prices, limits)
    return intents


class PaperBroker:
    """In-memory broker used for dry-runs and paper-trading smoke tests."""

    def __init__(self, limits: ExecutionRiskLimits | None = None) -> None:
        self.limits = limits or ExecutionRiskLimits()
        self.orders: list[OrderResult] = []
        self._next_id = 1

    def submit_order(self, intent: OrderIntent, reference_price: float | None = None) -> OrderResult:
        validate_order_intent(intent, reference_price, self.limits)
        fill_price = intent.limit_price if intent.limit_price is not None else reference_price
        order_id = f"PAPER-{self._next_id:06d}"
        self._next_id += 1
        result = OrderResult(
            intent=intent,
            status="FILLED",
            broker_order_id=order_id,
            filled_quantity=intent.quantity,
            fill_price=fill_price,
            message="paper fill",
        )
        self.orders.append(result)
        return result

    def submit_orders(self, intents: list[OrderIntent], reference_prices: pd.Series) -> list[OrderResult]:
        validate_order_batch(intents, reference_prices, self.limits)
        return [
            self.submit_order(intent, float(reference_prices.get(intent.ticker, float("nan"))))
            for intent in intents
        ]
