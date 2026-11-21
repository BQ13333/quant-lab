from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quant_lab.data.usa_exchange_calendar_tca import HALT_COLUMNS


BASE_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts&haltdate={haltdate}"
NDAQ_NAMESPACE = "http://www.nasdaqtrader.com/"
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
SUCCESS_STATUSES = frozenset({"completed", "completed_empty"})
MARKET_CODE_TO_EXCHANGE = {
    "Q": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "N": "NYSE",
    "NYSE": "NYSE",
    "A": "NYSE_AMERICAN",
    "C": "NYSE_NATIONAL",
    "P": "NYSE_ARCA",
    "V": "IEX",
    "Z": "CBOE_BZX",
}

REQUEST_MANIFEST_COLUMNS = (
    "request_date",
    "source_url",
    "status",
    "attempts",
    "observed_at_utc",
    "completed_at_utc",
    "declared_items",
    "parsed_items",
    "normalized_events",
    "unsupported_market_events",
    "duplicate_source_items",
    "invalid_resumption_events",
    "market_codes",
    "reason_codes",
    "raw_path",
    "raw_sha256",
    "error_type",
    "error",
)

COVERAGE_COLUMNS = (
    "request_date",
    "source_url",
    "status",
    "covered",
    "declared_items",
    "parsed_items",
    "normalized_events",
    "unsupported_market_events",
    "duplicate_source_items",
    "invalid_resumption_events",
    "observed_at_utc",
    "raw_path",
    "raw_sha256",
    "error",
)

Fetcher = Callable[[str, float], bytes]
Sleeper = Callable[[float], None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ParsedTradeHaltFeed:
    events: pd.DataFrame
    declared_items: int
    parsed_items: int
    unsupported_market_events: int
    duplicate_source_items: int
    invalid_resumption_events: int
    market_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class NasdaqTradeHaltArtifacts:
    run_id: str
    artifact_dir: Path
    request_plan_path: Path
    request_manifest_path: Path
    raw_dir: Path
    halt_events_path: Path
    coverage_path: Path
    manifest_path: Path
    summary_path: Path
    request_plan: pd.DataFrame
    request_manifest: pd.DataFrame
    halt_events: pd.DataFrame
    coverage: pd.DataFrame
    manifest: dict[str, object]


@dataclass(frozen=True)
class _PendingFetch:
    payload: bytes | None
    parsed: ParsedTradeHaltFeed | None
    attempts: int
    observed_at_utc: str
    error: Exception | None


def build_business_date_plan(start_date: str | date, end_date: str | date) -> pd.DataFrame:
    """Build an inclusive Monday-Friday request plan for the historical daily route."""
    start = _parse_iso_date(start_date, "start_date")
    end = _parse_iso_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")

    rows: list[dict[str, str]] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            rows.append(
                {
                    "request_date": current.isoformat(),
                    "source_url": source_url_for_date(current),
                }
            )
        current += timedelta(days=1)
    if not rows:
        raise ValueError("date range contains no business dates")
    return pd.DataFrame(rows, columns=["request_date", "source_url"])


def source_url_for_date(day: str | date) -> str:
    parsed = _parse_iso_date(day, "day")
    return BASE_URL.format(haltdate=parsed.strftime("%m%d%Y"))


def fetch_nasdaq_trade_halt_xml(url: str, timeout_seconds: float) -> bytes:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/rss+xml, application/xml, text/xml",
            "User-Agent": "quant-lab-nasdaq-trade-halts/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response exceeds {MAX_RESPONSE_BYTES} byte safety limit")
    return payload


def parse_trade_halt_xml(
    payload: bytes,
    *,
    requested_date: str | date,
    source_url: str,
    observed_at_utc: str | datetime,
) -> ParsedTradeHaltFeed:
    """Parse and normalize one official Nasdaq Trader historical RSS response."""
    requested = _parse_iso_date(requested_date, "requested_date")
    observed_at = _as_utc_iso(observed_at_utc, "observed_at_utc")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"response is not valid XML: {exc}") from exc
    if _local_name(root.tag).lower() != "rss":
        raise ValueError(f"response root must be rss, got {_local_name(root.tag)!r}")
    channel = next((child for child in root if _local_name(child.tag) == "channel"), None)
    if channel is None:
        raise ValueError("RSS response is missing channel")

    num_items_text = _child_text(channel, "numItems", namespace=NDAQ_NAMESPACE)
    if not num_items_text:
        raise ValueError("RSS response is missing ndaq:numItems")
    try:
        declared_items = int(num_items_text)
    except ValueError as exc:
        raise ValueError(f"ndaq:numItems is not an integer: {num_items_text!r}") from exc
    if declared_items < 0:
        raise ValueError("ndaq:numItems cannot be negative")

    items = [child for child in channel if _local_name(child.tag) == "item"]
    if len(items) != declared_items:
        raise ValueError(
            f"RSS item count mismatch: ndaq:numItems={declared_items}, parsed_items={len(items)}"
        )

    rows: list[dict[str, object]] = []
    market_codes: set[str] = set()
    reason_codes: set[str] = set()
    unsupported_market_events = 0
    duplicate_source_items = 0
    invalid_resumption_events = 0
    seen_source_record_ids: set[str] = set()
    for item in items:
        symbol = _required_ndaq_text(item, "IssueSymbol")
        market_code = _required_ndaq_text(item, "Mkt").upper()
        reason_code = _child_text(item, "ReasonCode", namespace=NDAQ_NAMESPACE)
        halt_date_text = _required_ndaq_text(item, "HaltDate")
        halt_time_text = _required_ndaq_text(item, "HaltTime")
        resumption_date_text = _child_text(item, "ResumptionDate", namespace=NDAQ_NAMESPACE)
        quote_time_text = _child_text(item, "ResumptionQuoteTime", namespace=NDAQ_NAMESPACE)
        trade_time_text = _child_text(item, "ResumptionTradeTime", namespace=NDAQ_NAMESPACE)

        halt_date = _parse_nasdaq_date(halt_date_text, "HaltDate")
        if halt_date != requested:
            raise ValueError(
                f"date coverage mismatch: requested {requested.isoformat()}, item has {halt_date.isoformat()}"
            )
        normalized_halt_time = _normalize_time_text(halt_time_text, "HaltTime")
        normalized_trade_time = (
            _normalize_time_text(trade_time_text, "ResumptionTradeTime")
            if trade_time_text
            else ""
        )
        normalized_quote_time = (
            _normalize_time_text(quote_time_text, "ResumptionQuoteTime")
            if quote_time_text
            else ""
        )
        source_record_id = "|".join(
            (
                halt_date.isoformat(),
                normalized_halt_time,
                symbol,
                f"mkt={market_code}",
                f"reason={reason_code}",
                f"resume_date={resumption_date_text}",
                f"quote_time={normalized_quote_time}",
                f"trade_time={normalized_trade_time}",
            )
        )
        if source_record_id in seen_source_record_ids:
            duplicate_source_items += 1
            continue
        seen_source_record_ids.add(source_record_id)

        halt_start = _eastern_datetime_to_utc(halt_date, halt_time_text, "HaltTime")
        halt_end: datetime | None = None
        if trade_time_text:
            if not resumption_date_text:
                raise ValueError("ResumptionTradeTime is present without ResumptionDate")
            resumption_date = _parse_nasdaq_date(resumption_date_text, "ResumptionDate")
            candidate_end = _eastern_datetime_to_utc(
                resumption_date, trade_time_text, "ResumptionTradeTime"
            )
            if candidate_end < halt_start:
                invalid_resumption_events += 1
            else:
                halt_end = candidate_end

        event_id = "nasdaq-trader-" + hashlib.sha256(
            source_record_id.encode("utf-8")
        ).hexdigest()[:24]

        market_codes.add(market_code)
        if reason_code:
            reason_codes.add(reason_code)
        exchange = MARKET_CODE_TO_EXCHANGE.get(market_code)
        if exchange is None:
            unsupported_market_events += 1
            continue
        rows.append(
            {
                "event_id": event_id,
                "exchange": exchange,
                "symbol": symbol,
                "halt_start_utc": halt_start.isoformat(timespec="microseconds"),
                "halt_end_utc": halt_end.isoformat(timespec="microseconds") if halt_end else "",
                "event_type": "trade_halt",
                "reason_code": reason_code,
                "source_name": source_url,
                "source_record_id": source_record_id,
                "observed_at_utc": observed_at,
            }
        )

    events = pd.DataFrame(rows, columns=HALT_COLUMNS)
    validate_halt_events(events)
    return ParsedTradeHaltFeed(
        events=events,
        declared_items=declared_items,
        parsed_items=len(items),
        unsupported_market_events=unsupported_market_events,
        duplicate_source_items=duplicate_source_items,
        invalid_resumption_events=invalid_resumption_events,
        market_codes=tuple(sorted(market_codes)),
        reason_codes=tuple(sorted(reason_codes)),
    )


def validate_halt_events(events: pd.DataFrame) -> None:
    if tuple(events.columns) != tuple(HALT_COLUMNS):
        raise ValueError(f"halt event columns must exactly equal HALT_COLUMNS: {HALT_COLUMNS}")
    if events.empty:
        return
    required_text = (
        "event_id",
        "exchange",
        "symbol",
        "event_type",
        "source_name",
        "source_record_id",
        "observed_at_utc",
    )
    for column in required_text:
        if (events[column].isna() | events[column].astype(str).str.strip().eq("")).any():
            raise ValueError(f"halt events contain blank {column}")
    for column in ("event_id", "source_record_id"):
        if events[column].astype(str).duplicated().any():
            raise ValueError(f"halt events contain duplicate {column} values")
    if not events["exchange"].isin(set(MARKET_CODE_TO_EXCHANGE.values())).all():
        raise ValueError("halt events contain unsupported downstream exchange values")
    starts = pd.to_datetime(events["halt_start_utc"], errors="coerce", utc=True)
    observed = pd.to_datetime(events["observed_at_utc"], errors="coerce", utc=True)
    if starts.isna().any():
        raise ValueError("halt events contain invalid halt_start_utc values")
    if observed.isna().any():
        raise ValueError("halt events contain invalid observed_at_utc values")
    raw_ends = events["halt_end_utc"]
    ends = pd.to_datetime(raw_ends, errors="coerce", utc=True)
    invalid_ends = raw_ends.notna() & raw_ends.astype(str).str.strip().ne("") & ends.isna()
    if invalid_ends.any():
        raise ValueError("halt events contain invalid halt_end_utc values")
    ended = ends.notna()
    if (ends[ended] < starts[ended]).any():
        raise ValueError("halt events contain halt_end_utc before halt_start_utc")


def _fetch_pending_trade_halt(
    plan_row: dict[str, object],
    *,
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
    fetcher: Fetcher,
    sleeper: Sleeper,
    clock: Clock,
) -> _PendingFetch:
    request_date = str(plan_row["request_date"])
    source_url = str(plan_row["source_url"])
    last_payload: bytes | None = None
    last_observed_at = ""
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            payload = fetcher(source_url, timeout_seconds)
            if not isinstance(payload, bytes):
                raise TypeError("fetcher must return bytes")
            last_payload = payload
            last_observed_at = _as_utc_iso(clock(), "clock result")
            parsed = parse_trade_halt_xml(
                payload,
                requested_date=request_date,
                source_url=source_url,
                observed_at_utc=last_observed_at,
            )
            return _PendingFetch(payload, parsed, attempt, last_observed_at, None)
        except Exception as exc:
            last_error = exc
            if attempt <= retries and sleep_seconds > 0:
                sleeper(sleep_seconds)
    assert last_error is not None
    return _PendingFetch(
        last_payload,
        None,
        retries + 1,
        last_observed_at,
        last_error,
    )


def extract_nasdaq_trade_halts(
    *,
    artifact_root: str | Path,
    run_id: str,
    start_date: str | date,
    end_date: str | date,
    max_dates: int,
    timeout_seconds: float = 20.0,
    retries: int = 2,
    sleep_seconds: float = 60.0,
    workers: int = 1,
    resume: bool = True,
    fetcher: Fetcher = fetch_nasdaq_trade_halt_xml,
    sleeper: Sleeper = time.sleep,
    clock: Clock | None = None,
) -> NasdaqTradeHaltArtifacts:
    """Run a bounded, checkpointed historical trade-halt extraction."""
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    if max_dates <= 0:
        raise ValueError("max_dates must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")

    now = clock or (lambda: datetime.now(UTC))
    plan = build_business_date_plan(start_date, end_date)
    artifact_dir = Path(artifact_root) / run_id
    raw_dir = artifact_dir / "raw"
    plan_path = artifact_dir / "request_plan.csv"
    request_manifest_path = artifact_dir / "request_manifest.csv"
    events_path = artifact_dir / "halt_suspension_events.csv"
    coverage_path = artifact_dir / "coverage.csv"
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.md"

    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not resume:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    _run_lock = _acquire_run_lock(artifact_dir)
    if resume and plan_path.exists():
        _validate_existing_plan(plan_path, plan)
    _atomic_write_csv(plan, plan_path)

    request_rows = _load_request_manifest(request_manifest_path) if resume else []
    request_rows = _mark_stale_successes(request_rows)
    _validate_manifest_scope(plan, request_rows)
    successful_dates = {
        str(row["request_date"])
        for row in request_rows
        if row.get("status") in SUCCESS_STATUSES
    }
    pending = plan.loc[~plan["request_date"].isin(successful_dates)].head(max_dates)

    pending_records = pending.to_dict("records")
    attempted_this_run = len(pending_records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(pending_records), workers):
            batch = pending_records[batch_start : batch_start + workers]
            fetched = list(
                executor.map(
                    lambda row: _fetch_pending_trade_halt(
                        row,
                        timeout_seconds=timeout_seconds,
                        retries=retries,
                        sleep_seconds=sleep_seconds,
                        fetcher=fetcher,
                        sleeper=sleeper,
                        clock=now,
                    ),
                    batch,
                )
            )
            for plan_row, result in zip(batch, fetched, strict=True):
                request_date = str(plan_row["request_date"])
                source_url = str(plan_row["source_url"])
                raw_path = raw_dir / f"{request_date}.xml"
                previous = next(
                    (
                        row
                        for row in request_rows
                        if str(row.get("request_date")) == request_date
                    ),
                    None,
                )
                previous_attempts = _as_int(previous.get("attempts") if previous else 0)
                if result.payload is not None:
                    _atomic_write_bytes(raw_path, result.payload)
                if result.parsed is not None:
                    parsed = result.parsed
                    raw_sha256 = _sha256(raw_path)
                    completed_at = _as_utc_iso(now(), "clock result")
                    manifest_row: dict[str, object] = {
                        "request_date": request_date,
                        "source_url": source_url,
                        "status": (
                            "completed_empty" if parsed.declared_items == 0 else "completed"
                        ),
                        "attempts": previous_attempts + result.attempts,
                        "observed_at_utc": result.observed_at_utc,
                        "completed_at_utc": completed_at,
                        "declared_items": parsed.declared_items,
                        "parsed_items": parsed.parsed_items,
                        "normalized_events": len(parsed.events),
                        "unsupported_market_events": parsed.unsupported_market_events,
                        "duplicate_source_items": parsed.duplicate_source_items,
                        "invalid_resumption_events": parsed.invalid_resumption_events,
                        "market_codes": "|".join(parsed.market_codes),
                        "reason_codes": "|".join(parsed.reason_codes),
                        "raw_path": str(raw_path.resolve()),
                        "raw_sha256": raw_sha256,
                        "error_type": "",
                        "error": "",
                    }
                else:
                    assert result.error is not None
                    manifest_row = {
                        "request_date": request_date,
                        "source_url": source_url,
                        "status": "failed",
                        "attempts": previous_attempts + result.attempts,
                        "observed_at_utc": result.observed_at_utc,
                        "completed_at_utc": _as_utc_iso(now(), "clock result"),
                        "declared_items": 0,
                        "parsed_items": 0,
                        "normalized_events": 0,
                        "unsupported_market_events": 0,
                        "duplicate_source_items": 0,
                        "invalid_resumption_events": 0,
                        "market_codes": "",
                        "reason_codes": "",
                        "raw_path": str(raw_path.resolve()) if raw_path.is_file() else "",
                        "raw_sha256": _sha256(raw_path) if raw_path.is_file() else "",
                        "error_type": type(result.error).__name__,
                        "error": str(result.error),
                    }
                request_rows = _replace_request_row(request_rows, manifest_row)
            _atomic_write_csv(_request_manifest_frame(request_rows), request_manifest_path)
            if batch_start + workers < len(pending_records) and sleep_seconds > 0:
                sleeper(sleep_seconds)

    request_rows = _reconcile_success_manifest_from_raw(plan, request_rows)
    request_manifest = _request_manifest_frame(request_rows)
    _validate_manifest_scope(plan, request_rows)
    halt_events = _events_from_verified_raw(plan, request_rows)
    validate_halt_events(halt_events)
    coverage = _build_coverage(plan, request_rows)
    coverage_validation = validate_date_coverage(plan, coverage)

    _atomic_write_csv(request_manifest, request_manifest_path)
    _atomic_write_csv(halt_events, events_path)
    _atomic_write_csv(coverage, coverage_path)

    completed_dates = int(coverage["covered"].sum())
    failed_dates = int(coverage["status"].eq("failed").sum())
    pending_dates = int(len(coverage) - completed_dates)
    status = "complete" if completed_dates == len(plan) else "incomplete"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "scope": "official_nasdaq_trader_historical_trade_halts",
        "status": status,
        "artifact_dir": str(artifact_dir.resolve()),
        "source_route": BASE_URL,
        "source_timezone": "America/New_York",
        "start_date": _parse_iso_date(start_date, "start_date").isoformat(),
        "end_date": _parse_iso_date(end_date, "end_date").isoformat(),
        "planned_business_dates": int(len(plan)),
        "attempted_dates_this_run": attempted_this_run,
        "max_dates_this_run": max_dates,
        "covered_dates": completed_dates,
        "failed_dates": failed_dates,
        "pending_dates": pending_dates,
        "completed_empty_dates": int(coverage["status"].eq("completed_empty").sum()),
        "date_coverage_ratio": completed_dates / len(plan),
        "raw_item_rows": int(
            pd.to_numeric(coverage["parsed_items"], errors="coerce").fillna(0).sum()
        ),
        "normalized_event_rows": int(len(halt_events)),
        "unsupported_market_rows": int(
            pd.to_numeric(coverage["unsupported_market_events"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "duplicate_source_items": int(
            pd.to_numeric(coverage["duplicate_source_items"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "invalid_resumption_events": int(
            pd.to_numeric(coverage["invalid_resumption_events"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "event_columns": list(HALT_COLUMNS),
        "uniqueness_validated": True,
        "date_coverage_validated": coverage_validation,
        "empty_day_policy": "zero-item responses create coverage rows only, never event rows",
        "request_policy": {
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "sleep_seconds": sleep_seconds,
            "workers": workers,
            "max_response_bytes": MAX_RESPONSE_BYTES,
            "single_writer_lock": True,
            "resume": resume,
        },
        "generated_at_utc": _as_utc_iso(now(), "clock result"),
        "artifacts": {
            "request_plan": str(plan_path.resolve()),
            "request_manifest": str(request_manifest_path.resolve()),
            "raw_directory": str(raw_dir.resolve()),
            "halt_suspension_events": str(events_path.resolve()),
            "coverage": str(coverage_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    summary = _summary_markdown(manifest)
    _atomic_write_text(summary_path, summary)
    manifest["sha256"] = {
        "request_plan": _sha256(plan_path),
        "request_manifest": _sha256(request_manifest_path),
        "halt_suspension_events": _sha256(events_path),
        "coverage": _sha256(coverage_path),
        "summary": _sha256(summary_path),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return NasdaqTradeHaltArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        request_plan_path=plan_path,
        request_manifest_path=request_manifest_path,
        raw_dir=raw_dir,
        halt_events_path=events_path,
        coverage_path=coverage_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        request_plan=plan,
        request_manifest=request_manifest,
        halt_events=halt_events,
        coverage=coverage,
        manifest=manifest,
    )


def validate_date_coverage(plan: pd.DataFrame, coverage: pd.DataFrame) -> bool:
    planned_dates = plan["request_date"].astype(str)
    coverage_dates = coverage["request_date"].astype(str)
    if planned_dates.duplicated().any():
        raise ValueError("request plan contains duplicate dates")
    if coverage_dates.duplicated().any():
        raise ValueError("coverage contains duplicate dates")
    if planned_dates.tolist() != coverage_dates.tolist():
        raise ValueError("coverage dates do not exactly match request plan dates")
    successful = coverage["status"].isin(SUCCESS_STATUSES)
    if not coverage.loc[successful, "covered"].all():
        raise ValueError("successful request dates must be marked covered")
    if coverage.loc[~successful, "covered"].any():
        raise ValueError("pending or failed request dates cannot be marked covered")
    empty = coverage["status"].eq("completed_empty")
    parsed_items = pd.to_numeric(coverage["parsed_items"], errors="coerce").fillna(0)
    normalized_events = pd.to_numeric(
        coverage["normalized_events"], errors="coerce"
    ).fillna(0)
    if (parsed_items[empty] != 0).any() or (normalized_events[empty] != 0).any():
        raise ValueError("completed_empty dates cannot contain source items or normalized events")
    return True


def _events_from_verified_raw(
    plan: pd.DataFrame, request_rows: list[dict[str, object]]
) -> pd.DataFrame:
    rows_by_date = {str(row["request_date"]): row for row in request_rows}
    frames: list[pd.DataFrame] = []
    for plan_row in plan.to_dict("records"):
        request_date = str(plan_row["request_date"])
        row = rows_by_date.get(request_date)
        if not row or row.get("status") not in SUCCESS_STATUSES:
            continue
        raw_path = Path(str(row["raw_path"]))
        if not raw_path.is_file() or _sha256(raw_path) != str(row["raw_sha256"]):
            raise ValueError(f"raw response verification failed for {request_date}")
        parsed = parse_trade_halt_xml(
            raw_path.read_bytes(),
            requested_date=request_date,
            source_url=str(row["source_url"]),
            observed_at_utc=str(row["observed_at_utc"]),
        )
        expected = (
            _as_int(row.get("declared_items")),
            _as_int(row.get("parsed_items")),
            _as_int(row.get("normalized_events")),
            _as_int(row.get("duplicate_source_items")),
            _as_int(row.get("invalid_resumption_events")),
        )
        actual = (
            parsed.declared_items,
            parsed.parsed_items,
            len(parsed.events),
            parsed.duplicate_source_items,
            parsed.invalid_resumption_events,
        )
        if actual != expected:
            raise ValueError(f"manifest/raw count mismatch for {request_date}: {expected} != {actual}")
        if not parsed.events.empty:
            frames.append(parsed.events)
    if not frames:
        return pd.DataFrame(columns=HALT_COLUMNS)
    result = pd.concat(frames, ignore_index=True).loc[:, HALT_COLUMNS]
    return result.sort_values(["halt_start_utc", "exchange", "symbol"]).reset_index(drop=True)


def _reconcile_success_manifest_from_raw(
    plan: pd.DataFrame, request_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    plan_urls = dict(zip(plan["request_date"].astype(str), plan["source_url"].astype(str)))
    reconciled: list[dict[str, object]] = []
    for row in request_rows:
        if row.get("status") not in SUCCESS_STATUSES:
            reconciled.append(row)
            continue
        request_date = str(row["request_date"])
        raw_path = Path(str(row["raw_path"]))
        if not raw_path.is_file() or _sha256(raw_path) != str(row["raw_sha256"]):
            raise ValueError(f"raw response verification failed for {request_date}")
        parsed = parse_trade_halt_xml(
            raw_path.read_bytes(),
            requested_date=request_date,
            source_url=plan_urls[request_date],
            observed_at_utc=str(row["observed_at_utc"]),
        )
        updated = dict(row)
        updated.update(
            {
                "declared_items": parsed.declared_items,
                "parsed_items": parsed.parsed_items,
                "normalized_events": len(parsed.events),
                "unsupported_market_events": parsed.unsupported_market_events,
                "duplicate_source_items": parsed.duplicate_source_items,
                "invalid_resumption_events": parsed.invalid_resumption_events,
                "market_codes": "|".join(parsed.market_codes),
                "reason_codes": "|".join(parsed.reason_codes),
            }
        )
        reconciled.append(updated)
    return reconciled


def _build_coverage(
    plan: pd.DataFrame, request_rows: list[dict[str, object]]
) -> pd.DataFrame:
    rows_by_date = {str(row["request_date"]): row for row in request_rows}
    coverage_rows: list[dict[str, object]] = []
    for plan_row in plan.to_dict("records"):
        request_date = str(plan_row["request_date"])
        request = rows_by_date.get(request_date, {})
        status = str(request.get("status") or "pending")
        coverage_rows.append(
            {
                "request_date": request_date,
                "source_url": str(plan_row["source_url"]),
                "status": status,
                "covered": status in SUCCESS_STATUSES,
                "declared_items": _as_int(request.get("declared_items")),
                "parsed_items": _as_int(request.get("parsed_items")),
                "normalized_events": _as_int(request.get("normalized_events")),
                "unsupported_market_events": _as_int(
                    request.get("unsupported_market_events")
                ),
                "duplicate_source_items": _as_int(request.get("duplicate_source_items")),
                "invalid_resumption_events": _as_int(
                    request.get("invalid_resumption_events")
                ),
                "observed_at_utc": str(request.get("observed_at_utc") or ""),
                "raw_path": str(request.get("raw_path") or ""),
                "raw_sha256": str(request.get("raw_sha256") or ""),
                "error": str(request.get("error") or ""),
            }
        )
    return pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)


def _mark_stale_successes(
    request_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    for row in request_rows:
        if row.get("status") not in SUCCESS_STATUSES:
            continue
        raw_path = Path(str(row.get("raw_path") or ""))
        expected_sha = str(row.get("raw_sha256") or "")
        if raw_path.is_file() and expected_sha and _sha256(raw_path) == expected_sha:
            continue
        row["status"] = "stale_raw"
        row["error_type"] = "RawVerificationError"
        row["error"] = "raw response is missing or its SHA256 does not match"
    return request_rows


def _validate_manifest_scope(
    plan: pd.DataFrame, request_rows: list[dict[str, object]]
) -> None:
    planned = set(plan["request_date"].astype(str))
    seen: set[str] = set()
    for row in request_rows:
        request_date = str(row.get("request_date") or "")
        if request_date not in planned:
            raise ValueError(f"request manifest date is outside current plan: {request_date}")
        if request_date in seen:
            raise ValueError(f"request manifest contains duplicate date: {request_date}")
        seen.add(request_date)
        expected_url = source_url_for_date(request_date)
        if str(row.get("source_url")) != expected_url:
            raise ValueError(f"request manifest URL does not match plan for {request_date}")


def _validate_existing_plan(path: Path, expected: pd.DataFrame) -> None:
    existing = pd.read_csv(path, keep_default_na=False)
    required = ["request_date", "source_url"]
    if any(column not in existing.columns for column in required):
        raise ValueError("existing request plan has an invalid schema")
    actual_records = existing.loc[:, required].astype(str).to_dict("records")
    expected_records = expected.loc[:, required].astype(str).to_dict("records")
    if actual_records != expected_records:
        raise ValueError("existing request plan does not match the requested date range")


def _load_request_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    for column in ("duplicate_source_items", "invalid_resumption_events"):
        if column not in frame.columns:
            frame[column] = 0
    missing = [column for column in REQUEST_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"request manifest is missing columns: {missing}")
    return frame.loc[:, REQUEST_MANIFEST_COLUMNS].to_dict("records")


def _request_manifest_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=REQUEST_MANIFEST_COLUMNS)
    return (
        pd.DataFrame(rows, columns=REQUEST_MANIFEST_COLUMNS)
        .sort_values("request_date")
        .reset_index(drop=True)
    )


def _replace_request_row(
    rows: list[dict[str, object]], new_row: dict[str, object]
) -> list[dict[str, object]]:
    request_date = str(new_row["request_date"])
    result = [row for row in rows if str(row.get("request_date")) != request_date]
    result.append(new_row)
    return result


def _summary_markdown(manifest: dict[str, object]) -> str:
    return "\n".join(
        (
            f"# {manifest['run_id']}",
            "",
            "## Nasdaq Trader Historical Trade Halts",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Planned business dates: `{manifest['planned_business_dates']}`",
            f"- Covered dates: `{manifest['covered_dates']}`",
            f"- Failed dates: `{manifest['failed_dates']}`",
            f"- Pending dates: `{manifest['pending_dates']}`",
            f"- Source rows: `{manifest['raw_item_rows']}`",
            f"- Normalized Nasdaq/NYSE events: `{manifest['normalized_event_rows']}`",
            f"- Unsupported listing-market rows retained in raw evidence: `{manifest['unsupported_market_rows']}`",
            f"- Exact duplicate source items removed: `{manifest['duplicate_source_items']}`",
            f"- Invalid source resumption timestamps left blank: `{manifest['invalid_resumption_events']}`",
            "",
            "A successful zero-item response proves request-date coverage but does not create an event row.",
            "Original market and reason codes remain in each source record ID; request URLs and raw XML are retained for audit.",
            "",
        )
    )


def _child_text(element: ET.Element, local_name: str, *, namespace: str) -> str:
    child = element.find(f"{{{namespace}}}{local_name}")
    return "" if child is None or child.text is None else child.text.strip()


def _required_ndaq_text(element: ET.Element, local_name: str) -> str:
    value = _child_text(element, local_name, namespace=NDAQ_NAMESPACE)
    if not value:
        raise ValueError(f"RSS item is missing ndaq:{local_name}")
    return value


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_iso_date(value: str | date, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD): {value!r}") from exc


def _parse_nasdaq_date(value: str, field: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%m/%d/%Y").date()
    except ValueError as exc:
        raise ValueError(f"{field} must use MM/DD/YYYY: {value!r}") from exc


def _normalize_time_text(value: str, field: str) -> str:
    cleaned = re.sub(r"\s+", "", value)
    try:
        return datetime.strptime(cleaned, "%H:%M:%S.%f").time().isoformat()
    except ValueError:
        try:
            return datetime.strptime(cleaned, "%H:%M:%S").time().isoformat()
        except ValueError as exc:
            raise ValueError(f"{field} must use HH:MM:SS[.ffffff]: {value!r}") from exc


def _eastern_datetime_to_utc(day: date, time_text: str, field: str) -> datetime:
    normalized = _normalize_time_text(time_text, field)
    local = datetime.fromisoformat(f"{day.isoformat()}T{normalized}").replace(tzinfo=NEW_YORK)
    return local.astimezone(UTC)


def _as_utc_iso(value: str | datetime, field: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field} is not a valid timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{field} must be a datetime or ISO timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _acquire_run_lock(artifact_dir: Path):
    lock_path = artifact_dir / ".extract.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another extractor owns the run lock: {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
