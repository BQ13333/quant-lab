"""Per-instrument Hong Kong board lots.

A null market-wide lot is not a substitute. Until a row names one instrument,
an effective interval, and a source, lookup fails closed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


class HkLotUnavailable(LookupError):
    """No sourced board lot for this instrument and session."""


_COLUMNS = (
    "sid",
    "exchange",
    "instrument_code",
    "effective_from",
    "effective_until",
    "known_at",
    "lot_size",
    "source_url",
    "source_kind",
)


def lot_table_gate_met(status_path: Path | None = None) -> bool:
    payload = json.loads(_status_path(status_path).read_text(encoding="utf-8"))
    return payload.get("gate") == "met" and payload.get("market_wide_default") is None and int(payload.get("per_instrument_rows", -1)) > 0


def lot_size(sid: str, session: str, *, table: Path | None = None) -> int:
    """Return one sourced lot. Never invents a market-wide default."""

    if not sid or not session:
        raise HkLotUnavailable("sid and session are required")
    matches: list[int] = []
    for row in _rows(table):
        if row["sid"] != sid:
            continue
        start = row["effective_from"]
        end = row["effective_until"]
        if not start or session < start or (end and session > end):
            continue
        if row["source_kind"] not in {"PRIMARY", "CAPTURED"} or not row["source_url"]:
            raise HkLotUnavailable(f"{sid} has a lot row without a primary or captured source")
        try:
            size = int(row["lot_size"])
        except ValueError as exc:
            raise HkLotUnavailable(f"{sid} lot size is not an integer") from exc
        if size <= 0:
            raise HkLotUnavailable(f"{sid} lot size is not positive")
        matches.append(size)
    if len(matches) != 1:
        raise HkLotUnavailable(
            f"no single sourced Hong Kong board lot for {sid} on {session}"
        )
    return matches[0]


def _rows(table: Path | None) -> list[dict[str, str]]:
    path = table or _default_table()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HkLotUnavailable(f"could not read {path}") from exc
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != _COLUMNS:
        raise HkLotUnavailable("Hong Kong lot table columns do not match the contract")
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row or any(not isinstance(value, str) for value in row.values()):
            raise HkLotUnavailable("Hong Kong lot table has a ragged row")
        rows.append({key: row[key].strip() for key in _COLUMNS})
    return rows


def _default_table() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "hk_board_lots.csv"


def _status_path(path: Path | None) -> Path:
    return path or Path(__file__).resolve().parents[3] / "configs" / "hk_board_lot_status.json"
