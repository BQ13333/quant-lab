"""Load the project market mandate and refuse anything outside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class MarketScopeError(ValueError):
    """The market is not in the active Bursa, China A-share, and Hong Kong mandate."""


@dataclass(frozen=True)
class MarketEntry:
    country: str
    code: str
    exchanges: tuple[str, ...]
    reason: str = ""
    currency: str = ""
    boards: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketScope:
    version: int
    effective_date: str
    decided_by: str
    active: tuple[MarketEntry, ...]
    frozen: tuple[MarketEntry, ...]
    excluded: tuple[MarketEntry, ...]
    policy: dict[str, object]
    path: Path

    def require_active(self, country: str, exchange: str) -> MarketEntry:
        """Admit one exchange, or raise. Frozen and excluded markets never pass."""

        if self._match(self.frozen, country, exchange) is not None or self._country(
            self.frozen, country
        ):
            raise MarketScopeError(f"{country} / {exchange} is frozen")
        if self._match(self.excluded, country, exchange) is not None:
            raise MarketScopeError(f"{country} / {exchange} is excluded")
        found = self._match(self.active, country, exchange)
        if found is None:
            raise MarketScopeError(f"{country} / {exchange} is not an active market")
        return found

    @staticmethod
    def _country(entries: tuple[MarketEntry, ...], country: str) -> bool:
        return any(entry.country == country for entry in entries)

    @staticmethod
    def _match(
        entries: tuple[MarketEntry, ...], country: str, exchange: str
    ) -> MarketEntry | None:
        for entry in entries:
            if entry.country == country and exchange in entry.exchanges:
                return entry
        return None


def load_market_scope(path: Path | None = None) -> MarketScope:
    target = path or _default_scope_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise MarketScopeError(f"could not read {target}") from exc
    raw = _parse_scope_yaml(text)
    return MarketScope(
        version=_require_int(raw, "version"),
        effective_date=_require_str(raw, "effective_date"),
        decided_by=_require_str(raw, "decided_by"),
        active=_entries(raw, "active_markets"),
        frozen=_entries(raw, "frozen_markets"),
        excluded=_entries(raw, "excluded_markets"),
        policy=_require_map(raw, "policy"),
        path=target,
    )


def _default_scope_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "project_market_scope.yaml"


def _entries(raw: dict[str, object], key: str) -> tuple[MarketEntry, ...]:
    rows = raw.get(key)
    if not isinstance(rows, list) or not rows:
        raise MarketScopeError(f"{key} must be a non-empty list")
    entries: list[MarketEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            raise MarketScopeError(f"{key} rows must be maps")
        exchanges = row.get("exchanges")
        exchanges_ok = (
            isinstance(exchanges, list)
            and bool(exchanges)
            and all(isinstance(item, str) and item for item in exchanges)
        )
        if not exchanges_ok:
            raise MarketScopeError(f"{key} exchanges must be a non-empty list of names")
        boards = row.get("boards", [])
        if not isinstance(boards, list) or any(not isinstance(item, str) for item in boards):
            raise MarketScopeError(f"{key} boards must be a list of names")
        country = row.get("country")
        code = row.get("code")
        if not isinstance(country, str) or not country or not isinstance(code, str) or not code:
            raise MarketScopeError(f"{key} needs a country and a code")
        reason = row.get("reason", "")
        currency = row.get("currency", "")
        if not isinstance(reason, str) or not isinstance(currency, str):
            raise MarketScopeError(f"{key} reason and currency must be text")
        entries.append(
            MarketEntry(
                country=country,
                code=code,
                exchanges=tuple(exchanges),
                reason=reason,
                currency=currency,
                boards=tuple(boards),
            )
        )
    return tuple(entries)


def _require_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise MarketScopeError(f"{key} must be text")
    return value


def _require_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise MarketScopeError(f"{key} must be an integer")
    return value


def _require_map(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict) or not value:
        raise MarketScopeError(f"{key} must be a map")
    return value


def _parse_scope_yaml(text: str) -> dict[str, object]:
    """Parse the narrow shape of configs/project_market_scope.yaml.

    Root keys, one map, and lists of flat maps. Anything else fails closed.
    """

    root: dict[str, object] = {}
    pending: str | None = None
    current_list: list[dict[str, object]] | None = None
    current_item: dict[str, object] | None = None
    current_map: dict[str, object] | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("\t") or "\t" in raw_line:
            raise MarketScopeError(f"tabs are not allowed in the scope file, line {lineno}")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0:
            key, value = _split_kv(line, lineno)
            pending = None
            current_list = None
            current_item = None
            current_map = None
            if value == "":
                pending = key
            else:
                root[key] = _scalar(value, lineno)
            continue
        if pending is not None:
            if line.startswith("- "):
                current_list = []
                root[pending] = current_list
            else:
                current_map = {}
                root[pending] = current_map
            pending = None
        if line.startswith("- "):
            if current_list is None:
                raise MarketScopeError(f"list item outside a list, line {lineno}")
            body = line[2:].strip()
            key, value = _split_kv(body, lineno)
            if value == "":
                raise MarketScopeError(f"nested blocks are not supported, line {lineno}")
            current_item = {key: _scalar(value, lineno)}
            current_list.append(current_item)
            continue
        key, value = _split_kv(line, lineno)
        if value == "":
            raise MarketScopeError(f"nested blocks are not supported, line {lineno}")
        parsed = _scalar(value, lineno)
        if current_item is not None and current_map is None:
            current_item[key] = parsed
        elif current_map is not None and current_item is None:
            current_map[key] = parsed
        else:
            raise MarketScopeError(f"could not place line {lineno}")
    if pending is not None:
        raise MarketScopeError(f"scope key {pending} has no body")
    return root


def _split_kv(line: str, lineno: int) -> tuple[str, str]:
    if ":" not in line:
        raise MarketScopeError(f"expected key: value, line {lineno}")
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key or " " in key:
        raise MarketScopeError(f"invalid key on line {lineno}")
    return key, value


def _scalar(value: str, lineno: int) -> object:
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    if value.startswith("["):
        if not value.endswith("]"):
            raise MarketScopeError(f"unclosed list on line {lineno}")
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip() for part in inner.split(",")]
    if value.startswith(("'", '"')):
        raise MarketScopeError(f"quoted values are not used in the scope file, line {lineno}")
    if value.isdigit():
        return int(value)
    return value
