from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
SOURCES = {
    "calendar": "https://www.jpx.co.jp/english/corporate/about-jpx/calendar/",
    "trading_unit": "https://www.jpx.co.jp/english/equities/trading/domestic/03.html",
    "price_limits": "https://www.jpx.co.jp/english/equities/trading/domestic/06.html",
    "tick_sizes": "https://www.jpx.co.jp/english/equities/trading/domestic/07.html",
    "halts": "https://www.jpx.co.jp/english/markets/equities/suspended/index.html",
}


@dataclass(frozen=True)
class JPXMarketRulesArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    quality_path: Path
    manifest: dict[str, Any]


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


def run_jpx_market_rules_public(
    *,
    run_id: str = "FX-CALENDAR-TCA-JPN-20260711",
    artifact_root: str | Path = "artifacts/fx_calendar_tca",
    fetcher: Fetcher | None = None,
) -> JPXMarketRulesArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    fetch = fetcher or _fetch
    root = Path(artifact_root) / run_id
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    observed = datetime.now(UTC).isoformat()
    raw_meta: list[dict[str, Any]] = []
    bodies: dict[str, bytes] = {}
    for name, url in SOURCES.items():
        status, body, headers = fetch(url)
        path = raw_dir / f"{name}.html"
        if status == 200:
            _atomic_write_bytes(path, body)
            bodies[name] = body
        raw_meta.append({
            "component": name,
            "url": url,
            "http_status": status,
            "observed_at_utc": observed,
            "raw_path": str(path.resolve()) if path.is_file() else "",
            "sha256": _sha256(path) if path.is_file() else "",
            "content_type": headers.get("Content-Type", headers.get("content-type", "")),
        })
    raw_manifest = pd.DataFrame(raw_meta)
    _atomic_write_csv(root / "raw_manifest.csv", raw_manifest)

    holidays = parse_market_holidays(bodies.get("calendar", b""))
    price_limits = parse_price_limits(bodies.get("price_limits", b""))
    tick_sizes = parse_tick_sizes(bodies.get("tick_sizes", b""))
    halts = parse_halts(bodies.get("halts", b""))
    halt_source_rows = _halt_source_row_count(bodies.get("halts", b""))
    halt_parity_passed = len(halts) == halt_source_rows
    lot_policy = parse_trading_unit_policy(bodies.get("trading_unit", b""))
    _atomic_write_csv(root / "exchange_holidays.csv", holidays)
    _atomic_write_csv(root / "price_limit_bands.csv", price_limits)
    _atomic_write_csv(root / "tick_size_bands.csv", tick_sizes)
    _atomic_write_csv(root / "halt_suspension_events.csv", halts)
    _atomic_write_json(root / "trading_unit_policy.json", lot_policy)
    _atomic_write_csv(
        root / "tca_assumptions.csv",
        pd.DataFrame(columns=["effective_date", "cost_component", "value_bps", "calibration_source"]),
    )

    quality = pd.DataFrame(
        [
            _quality("jpy_fx_rates", "pass_static_market_currency", True, 1.0, 1, "JPX cash-equity rules are denominated in JPY; dynamic FX series remains outside this static rule artifact"),
            _quality("exchange_calendars", "incomplete_current_years_only", False, 0.0, len(holidays), "official 2026-2027 holidays retained; full 2006-present session archive not sourced"),
            _quality(
                "halt_suspension_events",
                "incomplete_recent_archive_only_parity_passed" if halt_parity_passed else "failed_source_output_parity",
                False,
                0.0,
                len(halts),
                f"official current/recent halt page retained with {len(halts)}/{halt_source_rows} source rows normalized; full historical halt archive absent",
            ),
            _quality("price_limits", "incomplete_current_rule_only", False, 0.0, len(price_limits), "official current bands retained; historical rule validity windows absent"),
            _quality("lot_sizes", "incomplete_post_2018_policy_only", False, 0.0, int(bool(lot_policy)), "100-share standardization from 2018-10-01 retained; issuer-level prior history absent"),
            _quality("tca_assumptions", "missing_market_data_calibration", False, 0.0, 0, "calibration awaits admitted Japan OHLCV, liquidity, and halt history"),
        ]
    )
    quality_path = root / "evidence_quality.csv"
    _atomic_write_csv(quality_path, quality)
    manifest_path = root / "manifest.json"
    manifest = {
        "run_id": run_id,
        "artifact_type": "jpx_public_market_rules_partial",
        "status": "incomplete",
        "scorecard_eligible": False,
        "evidence_scope": "official_current_rules_and_recent_halts_partial",
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "currency": "JPY",
        "official_jpx_source": True,
        "official_jpx_calendar": False,
        "tca_calibrated": False,
        "complete_components": ["jpy_fx_rates"],
        "partial_components": ["exchange_calendars", "halt_suspension_events", "price_limits", "lot_sizes"],
        "missing_components": ["tca_assumptions"],
        "counts": {
            "holiday_rows": len(holidays),
            "price_limit_bands": len(price_limits),
            "tick_size_bands": len(tick_sizes),
            "halt_rows": len(halts),
            "halt_source_rows": halt_source_rows,
            "lot_size": lot_policy.get("standard_lot_size"),
        },
        "critical_blockers": [
            *([] if halt_parity_passed else [f"current halt source-to-output parity failed: {len(halts)}/{halt_source_rows}"]),
            "full 2006-present official JPX session calendar is missing",
            "full historical halt/suspension archive is missing",
            "historical price-limit and tick-size validity windows are missing",
            "issuer-level lot-size history before 2018-10-01 is missing",
            "TCA is not calibrated against admitted Japan market data",
        ],
        "artifacts": {
            "raw_manifest": str((root / "raw_manifest.csv").resolve()),
            "exchange_holidays": str((root / "exchange_holidays.csv").resolve()),
            "price_limit_bands": str((root / "price_limit_bands.csv").resolve()),
            "tick_size_bands": str((root / "tick_size_bands.csv").resolve()),
            "halt_suspension_events": str((root / "halt_suspension_events.csv").resolve()),
            "trading_unit_policy": str((root / "trading_unit_policy.json").resolve()),
            "tca_assumptions": str((root / "tca_assumptions.csv").resolve()),
            "evidence_quality": str(quality_path.resolve()),
        },
        "sha256": {
            path.name: _sha256(path)
            for path in root.iterdir()
            if path.is_file() and path.name != "manifest.json"
        },
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": observed,
    }
    _atomic_write_json(manifest_path, manifest)
    return JPXMarketRulesArtifacts(run_id, root, manifest_path, quality_path, manifest)


def parse_market_holidays(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    for year, table in zip((2026, 2027), soup.find_all("table")):
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.stripped_strings) for cell in tr.find_all("td")]
            if len(cells) < 2:
                continue
            match = re.match(r"([A-Z][a-z]{2})\.\s+(\d{1,2})", cells[0])
            if not match:
                continue
            event_date = datetime.strptime(f"{match.group(1)} {match.group(2)} {year}", "%b %d %Y").date()
            rows.append({"date": event_date.isoformat(), "year": str(year), "holiday_name": cells[1], "source": SOURCES["calendar"]})
    return pd.DataFrame(rows, columns=["date", "year", "holiday_name", "source"])


def parse_trading_unit_policy(html: bytes | str) -> dict[str, Any]:
    text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
    lot = 100 if re.search(r"100\s+shares", text, re.I) else None
    standardized = "2018-10-01" if re.search(r"Oct\.\s*1,\s*2018", text, re.I) else None
    return {
        "standard_lot_size": lot,
        "standardization_effective_date": standardized,
        "source": SOURCES["trading_unit"],
        "historical_issuer_level_complete": False,
    }


def parse_price_limits(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    table = tables[-1] if tables else None
    rows: list[dict[str, str]] = []
    if table:
        for tr in table.find_all("tr")[1:]:
            cells = [" ".join(cell.stripped_strings) for cell in tr.find_all("td")]
            if len(cells) >= 3:
                rows.append({"bound_type": cells[0], "base_price_upper_yen": cells[1], "daily_limit_yen": cells[2], "source": SOURCES["price_limits"]})
            elif len(cells) == 2:
                rows.append({"bound_type": cells[0], "base_price_upper_yen": "", "daily_limit_yen": cells[1], "source": SOURCES["price_limits"]})
    return pd.DataFrame(rows, columns=["bound_type", "base_price_upper_yen", "daily_limit_yen", "source"])


def parse_tick_sizes(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    categories = ("topix500", "etf_unit_one", "other_issues")
    rows: list[dict[str, str]] = []
    for category, table in zip(categories, soup.find_all("table")):
        for tr in table.find_all("tr")[1:]:
            cells = [" ".join(cell.stripped_strings) for cell in tr.find_all("td")]
            if len(cells) >= 3:
                rows.append({"category": category, "bound_type": cells[0], "price_upper_yen": cells[1], "tick_size_yen": cells[2], "source": SOURCES["tick_sizes"]})
            elif len(cells) == 2:
                rows.append({"category": category, "bound_type": cells[0], "price_upper_yen": "", "tick_size_yen": cells[1], "source": SOURCES["tick_sizes"]})
    return pd.DataFrame(rows, columns=["category", "bound_type", "price_upper_yen", "tick_size_yen", "source"])


def parse_halts(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    for scope, table in zip(("current", "archive"), soup.find_all("table")):
        for cells in _expanded_table_rows(table):
            if len(cells) < 7:
                continue
            event_date = _english_date(cells[2])
            if not event_date or not cells[1].strip():
                continue
            rows.append({
                "issue_name": cells[0], "code": cells[1].strip().upper(), "date": event_date,
                "suspension_start": cells[3], "suspension_end": cells[4], "restart_time": cells[5],
                "reason": cells[6], "scope": scope, "source": SOURCES["halts"],
            })
    return pd.DataFrame(rows, columns=["issue_name", "code", "date", "suspension_start", "suspension_end", "restart_time", "reason", "scope", "source"])


def _halt_source_row_count(html: bytes | str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    return sum(
        1
        for table in soup.find_all("table")[:2]
        for cells in _expanded_table_rows(table)
        if len(cells) >= 7 and cells[1].strip()
    )


def _expanded_table_rows(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    for tr in table.find_all("tr"):
        slots: dict[int, str] = {}
        for column, (remaining, value) in list(pending.items()):
            slots[column] = value
            if remaining == 1:
                del pending[column]
            else:
                pending[column] = (remaining - 1, value)
        column = 0
        for cell in tr.find_all("td"):
            while column in slots:
                column += 1
            value = " ".join(cell.stripped_strings)
            colspan = max(1, int(cell.get("colspan", 1)))
            rowspan = max(1, int(cell.get("rowspan", 1)))
            for offset in range(colspan):
                target = column + offset
                slots[target] = value
                if rowspan > 1:
                    pending[target] = (rowspan - 1, value)
            column += colspan
        if slots:
            rows.append([slots.get(index, "") for index in range(max(slots) + 1)])
    return rows


def _quality(component: str, status: str, validated: bool, coverage: float, source_rows: int, notes: str) -> dict[str, Any]:
    return {"component": component, "status": status, "validated": validated, "coverage": coverage, "source_rows": source_rows, "notes": notes}


def _english_date(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip())
    for pattern in ("%b. %d, %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(normalized, pattern).date().isoformat()
        except ValueError:
            continue
    return ""


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-jpx-market-rules/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(64 * 1024 * 1024 + 1), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
