from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_lab.execution.broker import OrderIntent, OrderResult


@dataclass
class PaperTradingLedger:
    """Durable CSV ledger for paper orders and fills."""

    ledger_dir: str | Path

    def __post_init__(self) -> None:
        self.ledger_path = Path(self.ledger_dir)
        self.ledger_path.mkdir(parents=True, exist_ok=True)
        self.orders_path = self.ledger_path / "orders.csv"
        self.fills_path = self.ledger_path / "fills.csv"

    def record_intents(self, run_id: str, intents: list[OrderIntent]) -> pd.DataFrame:
        rows = []
        for i, intent in enumerate(intents, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "sequence": i,
                    "ticker": intent.ticker,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "limit_price": intent.limit_price,
                    "strategy_id": intent.strategy_id,
                    "account": intent.account,
                    "metadata_json": json.dumps(intent.metadata, sort_keys=True),
                }
            )
        frame = pd.DataFrame(rows)
        self._append(self.orders_path, frame)
        return frame

    def record_results(self, run_id: str, results: list[OrderResult]) -> pd.DataFrame:
        rows = []
        for i, result in enumerate(results, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "sequence": i,
                    "broker_order_id": result.broker_order_id,
                    "status": result.status,
                    "ticker": result.intent.ticker,
                    "side": result.intent.side,
                    "quantity": result.intent.quantity,
                    "filled_quantity": result.filled_quantity,
                    "fill_price": result.fill_price,
                    "message": result.message,
                    "timestamp_utc": result.timestamp_utc.isoformat(),
                    "strategy_id": result.intent.strategy_id,
                }
            )
        frame = pd.DataFrame(rows)
        self._append(self.fills_path, frame)
        return frame

    def load_orders(self) -> pd.DataFrame:
        return pd.read_csv(self.orders_path) if self.orders_path.exists() else pd.DataFrame()

    def load_fills(self) -> pd.DataFrame:
        return pd.read_csv(self.fills_path) if self.fills_path.exists() else pd.DataFrame()

    @staticmethod
    def _append(path: Path, frame: pd.DataFrame) -> None:
        if frame.empty:
            if not path.exists():
                frame.to_csv(path, index=False)
            return
        if path.exists():
            existing = pd.read_csv(path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_csv(path, index=False)
