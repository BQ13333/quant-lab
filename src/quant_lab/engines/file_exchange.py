"""File contract for an optional external backtester.

The lab writes a request directory and later reads a result file. Nothing in
this module imports an external engine, opens a socket, or emits an order.
Ziplime, if used at all, stays in another environment. Its licence is unclear
(GPL-3.0 text, a commercial-use restriction in the end-user agreement, and a
README that describes a move to AGPL-3.0), so every bundle carries a
research-only acknowledgement.

Fills must land on the next session the lab names. Bars after the decision
session are rejected. A fill without a positive applied cost is rejected.
Markets other than Malaysia, China, and Hong Kong are rejected. Daily bars
only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

CONTRACT_VERSION = "ql-external-engine-1"
PURPOSE = "research_only"
FILL_POLICY = "next_bar"
BAR_INTERVAL = "1d"
LICENCE_ACKNOWLEDGEMENT = (
    "RESEARCH_ONLY. Ziplime licence is unclear: the repository LICENSE file is "
    "GPL-3.0, the end-user agreement adds a commercial-use restriction, and the "
    "README describes a move to AGPL-3.0. Not for live trading, not a quant-lab "
    "dependency, separate environment, file exchange only."
)

_ALLOWED_MARKETS = frozenset({"MY", "CN", "HK"})
_SUFFIXES = {"MY": (".KL",), "CN": (".SS", ".SZ"), "HK": (".HK",)}
_SESSION = re.compile(r"\d{4}-\d{2}-\d{2}")
_ENGINE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_BLOCKED_ENGINE_TOKENS = ("broker", "ibkr", "live", "lime-trader", "lime_trader", "order")
_BAR_FIELDS = ("symbol", "session", "open", "high", "low", "close", "volume")
_REQUEST_FIELDS = frozenset(
    {
        "bars_sha256",
        "contract_version",
        "costs_bps_one_way",
        "decision_session",
        "external_engine_name",
        "fill_policy",
        "interval",
        "licence_acknowledgement",
        "live_trading",
        "market",
        "next_session",
        "purpose",
        "request_sha256",
        "symbols",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "contract_version",
        "external_engine_name",
        "fills",
        "licence_acknowledgement",
        "live_trading",
        "lookahead_detected",
        "purpose",
        "request_sha256",
    }
)
_FILL_FIELDS = frozenset(
    {
        "cost_applied",
        "cost_bps",
        "decision_session",
        "fill_price",
        "fill_session",
        "quantity",
        "side",
        "symbol",
    }
)


class FileExchangeError(ValueError):
    """The request or result cannot be admitted as research evidence."""


@dataclass(frozen=True)
class DailyBar:
    symbol: str
    session: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class ResearchFill:
    symbol: str
    decision_session: str
    fill_session: str
    side: str
    quantity: float
    fill_price: float
    cost_bps: float
    cost_applied: bool


@dataclass(frozen=True)
class ExchangeRequest:
    market: str
    decision_session: str
    next_session: str
    costs_bps_one_way: float
    symbols: tuple[str, ...]
    bars: tuple[DailyBar, ...]
    external_engine_name: str
    directory: Path
    bars_sha256: str
    request_sha256: str


@dataclass(frozen=True)
class ExchangeResult:
    external_engine_name: str
    request_sha256: str
    fills: tuple[ResearchFill, ...]


def write_request(
    directory: Path | str,
    *,
    market: str,
    decision_session: str,
    next_session: str,
    costs_bps_one_way: float,
    bars: tuple[DailyBar, ...] | list[DailyBar],
    external_engine_name: str = "unspecified",
) -> ExchangeRequest:
    """Write ``request.json`` and ``bars.csv`` for one decision session."""

    market = _market(market)
    decision = _session(decision_session, field="decision_session")
    nxt = _session(next_session, field="next_session")
    if nxt <= decision:
        raise FileExchangeError("next session must be after the decision session")
    cost = _cost_bps(costs_bps_one_way, field="costs_bps_one_way")
    engine = _engine_name(external_engine_name)
    clean_bars = _validate_bars(market, tuple(bars), decision_session=decision, next_session=nxt)
    symbols = tuple(sorted({bar.symbol for bar in clean_bars}))
    target = Path(directory)
    target.mkdir(parents=False, exist_ok=False)
    csv_bytes = _render_bars(clean_bars)
    bars_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    payload = _request_payload(
        market=market,
        decision_session=decision,
        next_session=nxt,
        costs_bps_one_way=cost,
        symbols=symbols,
        external_engine_name=engine,
        bars_sha256=bars_sha256,
    )
    request_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    payload["request_sha256"] = request_sha256
    (target / "bars.csv").write_bytes(csv_bytes)
    (target / "request.json").write_bytes(_canonical(payload))
    return ExchangeRequest(
        market=market,
        decision_session=decision,
        next_session=nxt,
        costs_bps_one_way=cost,
        symbols=symbols,
        bars=clean_bars,
        external_engine_name=engine,
        directory=target,
        bars_sha256=bars_sha256,
        request_sha256=request_sha256,
    )


def load_request(directory: Path | str) -> ExchangeRequest:
    """Reload a request directory and reject a hash mismatch."""

    target = Path(directory)
    payload = _read_json(target / "request.json")
    _exact_fields(payload, _REQUEST_FIELDS, what="request")
    _common_header(payload)
    if payload["fill_policy"] != FILL_POLICY or payload["interval"] != BAR_INTERVAL:
        raise FileExchangeError("request is not a next-bar daily research bundle")
    market = _market(payload["market"])
    decision = _session(payload["decision_session"], field="decision_session")
    nxt = _session(payload["next_session"], field="next_session")
    cost = _cost_bps(payload["costs_bps_one_way"], field="costs_bps_one_way")
    engine = _engine_name(payload["external_engine_name"])
    symbols = payload["symbols"]
    if not isinstance(symbols, list) or not symbols or any(not isinstance(item, str) for item in symbols):
        raise FileExchangeError("request symbols are invalid")
    if list(symbols) != sorted(set(symbols)):
        raise FileExchangeError("request symbols must be unique and sorted")
    for symbol in symbols:
        _symbol(market, symbol)
    try:
        csv_bytes = (target / "bars.csv").read_bytes()
    except OSError as exc:
        raise FileExchangeError("could not read bars.csv") from exc
    bars_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    if payload["bars_sha256"] != bars_sha256:
        raise FileExchangeError("bars.csv does not match the request hash")
    stored = payload["request_sha256"]
    check = dict(payload)
    check.pop("request_sha256")
    if stored != hashlib.sha256(_canonical(check)).hexdigest():
        raise FileExchangeError("request.json hash does not match its contents")
    bars = _parse_bars(csv_bytes)
    clean = _validate_bars(market, bars, decision_session=decision, next_session=nxt)
    if tuple(sorted({bar.symbol for bar in clean})) != tuple(symbols):
        raise FileExchangeError("bars.csv symbols do not match the request")
    return ExchangeRequest(
        market=market,
        decision_session=decision,
        next_session=nxt,
        costs_bps_one_way=cost,
        symbols=tuple(symbols),
        bars=clean,
        external_engine_name=engine,
        directory=target,
        bars_sha256=bars_sha256,
        request_sha256=stored,
    )


def read_result(directory: Path | str, request: ExchangeRequest) -> ExchangeResult:
    """Admit a result file only when it matches the request and the research rules."""

    payload = _read_json(Path(directory) / "result.json")
    _exact_fields(payload, _RESULT_FIELDS, what="result")
    _common_header(payload)
    if payload["lookahead_detected"] is not False:
        raise FileExchangeError("result reports look-ahead and is rejected")
    if payload["request_sha256"] != request.request_sha256:
        raise FileExchangeError("result is not bound to this request")
    engine = _engine_name(payload["external_engine_name"])
    raw_fills = payload["fills"]
    if not isinstance(raw_fills, list):
        raise FileExchangeError("fills must be a list")
    fills = tuple(_fill(item, request) for item in raw_fills)
    return ExchangeResult(external_engine_name=engine, request_sha256=request.request_sha256, fills=fills)


def _request_payload(
    *,
    market: str,
    decision_session: str,
    next_session: str,
    costs_bps_one_way: float,
    symbols: tuple[str, ...],
    external_engine_name: str,
    bars_sha256: str,
) -> dict[str, object]:
    return {
        "bars_sha256": bars_sha256,
        "contract_version": CONTRACT_VERSION,
        "costs_bps_one_way": costs_bps_one_way,
        "decision_session": decision_session,
        "external_engine_name": external_engine_name,
        "fill_policy": FILL_POLICY,
        "interval": BAR_INTERVAL,
        "licence_acknowledgement": LICENCE_ACKNOWLEDGEMENT,
        "live_trading": False,
        "market": market,
        "next_session": next_session,
        "purpose": PURPOSE,
        "symbols": list(symbols),
    }


def _common_header(payload: dict[str, object]) -> None:
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise FileExchangeError("unknown contract version")
    if payload.get("purpose") != PURPOSE:
        raise FileExchangeError("bundle is not marked research_only")
    if payload.get("licence_acknowledgement") != LICENCE_ACKNOWLEDGEMENT:
        raise FileExchangeError("research-only licence acknowledgement is missing or altered")
    if payload.get("live_trading") is not False:
        raise FileExchangeError("live trading is refused")


def _fill(item: object, request: ExchangeRequest) -> ResearchFill:
    if not isinstance(item, dict):
        raise FileExchangeError("fill must be an object")
    _exact_fields(item, _FILL_FIELDS, what="fill")
    symbol = item["symbol"]
    if not isinstance(symbol, str) or symbol not in request.symbols:
        raise FileExchangeError("fill symbol is outside the request")
    _symbol(request.market, symbol)
    decision = _session(item["decision_session"], field="decision_session")
    fill_session = _session(item["fill_session"], field="fill_session")
    if decision != request.decision_session or fill_session != request.next_session:
        raise FileExchangeError("fill is not the next bar after the decision")
    side = item["side"]
    if side not in {"BUY", "SELL"}:
        raise FileExchangeError("fill side must be BUY or SELL")
    quantity = _whole_shares(item["quantity"])
    price = _positive(item["fill_price"], field="fill_price")
    cost_bps = _cost_bps(item["cost_bps"], field="cost_bps")
    if item["cost_applied"] is not True:
        raise FileExchangeError("fill dropped its cost")
    return ResearchFill(
        symbol=symbol,
        decision_session=decision,
        fill_session=fill_session,
        side=side,
        quantity=quantity,
        fill_price=price,
        cost_bps=cost_bps,
        cost_applied=True,
    )


def _validate_bars(
    market: str,
    bars: tuple[DailyBar, ...],
    *,
    decision_session: str,
    next_session: str,
) -> tuple[DailyBar, ...]:
    if not bars:
        raise FileExchangeError("request needs at least one daily bar")
    seen: set[tuple[str, str]] = set()
    decision_symbols: set[str] = set()
    clean: list[DailyBar] = []
    for bar in bars:
        if not isinstance(bar, DailyBar):
            raise FileExchangeError("bars must be DailyBar values")
        _symbol(market, bar.symbol)
        session = _session(bar.session, field="session")
        if session > decision_session or session == next_session:
            raise FileExchangeError("bar is after the decision session")
        key = (bar.symbol, session)
        if key in seen:
            raise FileExchangeError("duplicate bar for a symbol and session")
        seen.add(key)
        open_ = _finite(bar.open)
        high = _finite(bar.high)
        low = _finite(bar.low)
        close = _finite(bar.close)
        volume = _finite(bar.volume)
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise FileExchangeError("bar prices must be positive and volume cannot be negative")
        if high < max(open_, close, low) or low > min(open_, close):
            raise FileExchangeError("bar high/low does not contain the open and close")
        if session == decision_session:
            decision_symbols.add(bar.symbol)
        clean.append(DailyBar(bar.symbol, session, open_, high, low, close, volume))
    symbols = {bar.symbol for bar in clean}
    if decision_symbols != symbols:
        raise FileExchangeError("every symbol needs a bar on the decision session")
    return tuple(sorted(clean, key=lambda bar: (bar.symbol, bar.session)))


def _parse_bars(csv_bytes: bytes) -> tuple[DailyBar, ...]:
    try:
        text = csv_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise FileExchangeError("bars.csv is not utf-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(_BAR_FIELDS):
        raise FileExchangeError("bars.csv columns are not the daily research schema")
    bars: list[DailyBar] = []
    for row in reader:
        bars.append(
            DailyBar(
                symbol=str(row["symbol"]),
                session=str(row["session"]),
                open=_float_cell(row["open"]),
                high=_float_cell(row["high"]),
                low=_float_cell(row["low"]),
                close=_float_cell(row["close"]),
                volume=_float_cell(row["volume"]),
            )
        )
    return tuple(bars)


def _render_bars(bars: tuple[DailyBar, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_BAR_FIELDS, lineterminator="\n")
    writer.writeheader()
    for bar in bars:
        writer.writerow(
            {
                "symbol": bar.symbol,
                "session": bar.session,
                "open": _num(bar.open),
                "high": _num(bar.high),
                "low": _num(bar.low),
                "close": _num(bar.close),
                "volume": _num(bar.volume),
            }
        )
    return buffer.getvalue().encode("utf-8")


def _market(value: object) -> str:
    if value not in _ALLOWED_MARKETS:
        raise FileExchangeError("market must be MY, CN, or HK")
    return str(value)


def _symbol(market: str, symbol: object) -> str:
    if not isinstance(symbol, str):
        raise FileExchangeError("symbol must be a string")
    upper = symbol.upper()
    if any(token in upper for token in ("BTC", "ETH", "USDT", "CRYPTO")):
        raise FileExchangeError("crypto symbols are out of scope")
    suffixes = _SUFFIXES[market]
    if not symbol.endswith(suffixes):
        raise FileExchangeError("symbol suffix does not match the market")
    stem = symbol.split(".", 1)[0]
    if not stem.isdigit() or not 1 <= len(stem) <= 6:
        raise FileExchangeError("symbol must be an exchange numeric code")
    return symbol


def _session(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SESSION.fullmatch(value) is None:
        raise FileExchangeError(f"{field} must be a daily session YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise FileExchangeError(f"{field} is not a real calendar date") from exc
    return value


def _engine_name(value: object) -> str:
    if not isinstance(value, str) or _ENGINE_NAME.fullmatch(value) is None:
        raise FileExchangeError("external engine name is not a research label")
    if any(token in value.lower() for token in _BLOCKED_ENGINE_TOKENS):
        raise FileExchangeError("external engine name looks like a broker or live route")
    return value


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FileExchangeError("expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise FileExchangeError("expected a finite number")
    return number


def _positive(value: object, *, field: str) -> float:
    number = _finite(value)
    if number <= 0:
        raise FileExchangeError(f"{field} must be positive")
    return number


def _cost_bps(value: object, *, field: str) -> float:
    number = _positive(value, field=field)
    if number > 10_000:
        raise FileExchangeError(f"{field} is not a plausible one-way cost in bps")
    return number


def _whole_shares(value: object) -> float:
    number = _finite(value)
    if number <= 0 or number != math.floor(number):
        raise FileExchangeError("quantity must be a positive whole number of shares")
    return number


def _float_cell(value: object) -> float:
    if not isinstance(value, str):
        raise FileExchangeError("bar cell is not text")
    try:
        number = float(value)
    except ValueError as exc:
        raise FileExchangeError("bar cell is not a number") from exc
    if not math.isfinite(number):
        raise FileExchangeError("bar cell is not finite")
    return number


def _num(value: float) -> str:
    text = format(value, ".10f").rstrip("0").rstrip(".")
    return text or "0"


def _exact_fields(payload: dict[str, object], allowed: frozenset[str], *, what: str) -> None:
    extra = set(payload) - allowed
    missing = allowed - set(payload)
    if extra or missing:
        raise FileExchangeError(f"{what} fields do not match the research schema")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExchangeError(f"could not read {path.name}") from exc
    if not isinstance(payload, dict):
        raise FileExchangeError(f"{path.name} must be an object")
    return payload


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
