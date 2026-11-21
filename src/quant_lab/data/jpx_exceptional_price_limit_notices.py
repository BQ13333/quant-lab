from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"
RESEARCH_START = "2013-09"
RESEARCH_END = "2026-06"
EXPECTED_ARCHIVE_SOURCES = 154
EXPECTED_NOTICE_PAGES = 1_509
EXPECTED_EVENT_ROWS = 1_650
EVENT_TYPES = ("preliminary", "activate", "continue", "cancel", "return", "correction")
EXPECTED_NOTICE_TYPE_COUNTS = {
    "preliminary": 477,
    "activate": 534,
    "continue": 57,
    "cancel": 319,
    "return": 121,
    "correction": 1,
}
PRE_2013_BLOCKER = "official issue-specific price-limit archive before 2013-09 is not yet proven complete"
POST_2020_BLOCKER = (
    "post-2020 zero-volume removals remain unresolved: return-to-normal dates require admitted "
    "daily price/calendar evidence because JPX discontinued removal notices"
)
REQUIRED_COMPLETION_BLOCKERS = (PRE_2013_BLOCKER, POST_2020_BLOCKER)
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
ACTIVATION_TYPES = {"activate", "continue", "correction"}
Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


class NoticeContractError(ValueError):
    """Raised when retained sources do not satisfy the researched denominator contract."""


class NoticeParseError(ValueError):
    """Raised when an official notice cannot be mapped to one row per security."""


class ImmutableArtifactError(RuntimeError):
    """Raised when a retained or finalized artifact has changed in place."""


@dataclass(frozen=True)
class JPXExceptionalPriceLimitArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(value))
    value = value.replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def classify_notice_title(title: str) -> str:
    """Map an in-scope JPX stock notice title to the canonical six-event taxonomy."""
    value = normalize_text(title)
    if re.search(r"\b(?:ETF|ETN)\b|subscription warrant|securities option", value, re.I):
        return ""

    if re.search(r"[\[【](?:revised?|correction)[\]】]", value, re.I) and re.match(
        r"^Daily Price Limits to be Broadened\s*:", value, re.I
    ):
        return "correction"

    patterns = (
        (
            "preliminary",
            r"^(?:Expansion of daily price limits|Broadening of Daily Price Limits)\s*"
            r"\(Preliminary Notice\)\s*[:\-]",
        ),
        (
            "cancel",
            r"^(?:Handling of daily price limits\s*\(Daily price limits will not be expanded\)|"
            r"Announced Broadening of Daily Price Limits to be Cancelled)\s*[:\-]",
        ),
        (
            "activate",
            r"^(?:Expansion of daily price limits\s*\(Daily price limits? will be expanded\)|"
            r"Daily Price Limits to be Broadened)\s*[:\-]",
        ),
        (
            "continue",
            r"^(?:Continued expansion of daily price limits\s*"
            r"\(Daily price limits will continue to be expanded\)|Continue Broadened Daily Price Limits|"
            r"Daily Price Limits Continue to be Broadened)\s*[:\-]",
        ),
        (
            "return",
            r"^(?:Removal of expanded daily price limits\s*\(Return to normal daily price limits\)|"
            r"Daily Price Limits to be Returned to Normal)\s*[:\-]",
        ),
    )
    for event_type, pattern in patterns:
        if re.search(pattern, value, re.I):
            return event_type
    return ""


def build_notice_plan(
    archive_manifest_path: str | Path,
    *,
    expected_archive_sources: int = EXPECTED_ARCHIVE_SOURCES,
    expected_notice_pages: int = EXPECTED_NOTICE_PAGES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify retained monthly JSON and deterministically enumerate the notice denominator."""
    manifest_path = Path(archive_manifest_path).resolve()
    archives = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"source_id", "url", "http_status", "raw_path", "sha256", "bytes"}
    if not required.issubset(archives.columns):
        raise NoticeContractError(f"archive manifest missing columns: {sorted(required - set(archives.columns))}")
    if len(archives) != expected_archive_sources:
        raise NoticeContractError(
            f"archive source denominator mismatch: expected {expected_archive_sources}, observed {len(archives)}"
        )
    if archives["source_id"].duplicated().any() or archives["url"].duplicated().any():
        raise NoticeContractError("archive source ids and URLs must be unique")

    enforce_researched_sources = expected_archive_sources == EXPECTED_ARCHIVE_SOURCES
    if enforce_researched_sources and set(archives["url"]) != _researched_archive_urls():
        missing = sorted(_researched_archive_urls() - set(archives["url"]))
        extra = sorted(set(archives["url"]) - _researched_archive_urls())
        raise NoticeContractError(f"researched archive URL set mismatch: missing={missing}, extra={extra}")

    inventory_rows: list[dict[str, Any]] = []
    notices: dict[str, dict[str, Any]] = {}
    for source in archives.to_dict("records"):
        raw_path = _resolve_input_path(source["raw_path"], manifest_path.parent)
        if source["http_status"] != "200":
            raise NoticeContractError(f"archive source HTTP status is not 200: {source['source_id']}")
        if not raw_path.is_file():
            raise NoticeContractError(f"archive source is missing: {source['source_id']}")
        observed_hash = sha256_file(raw_path)
        if observed_hash != source["sha256"]:
            raise ImmutableArtifactError(f"archive source hash mismatch: {source['source_id']}")
        observed_bytes = raw_path.stat().st_size
        if int(source["bytes"]) != observed_bytes:
            raise ImmutableArtifactError(f"archive source size mismatch: {source['source_id']}")
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NoticeContractError(f"archive JSON is invalid: {source['source_id']}") from exc

        selected = 0
        for item in _walk_news_items(payload):
            event_type = classify_notice_title(str(item.get("title", "")))
            publication = _publication_date(item)
            target = str(item.get("url", ""))
            if not event_type or publication is None or not target:
                continue
            if not (date(2013, 9, 1) <= publication <= date(2026, 6, 30)):
                continue
            url = urljoin(JPX_ROOT, target)
            if not re.fullmatch(
                r"https://www\.jpx\.co\.jp/english/news/(?:detail/detail_[0-9]+|1030/[A-Za-z0-9_-]+)\.html",
                url,
            ):
                raise NoticeContractError(f"selected notice URL is invalid: {url}")
            title = normalize_text(str(item["title"]))
            current = notices.setdefault(
                url,
                {
                    "publication_date": publication.isoformat(),
                    "event_type": event_type,
                    "title": title,
                    "url": url,
                    "archive_source_ids": [],
                },
            )
            if (
                current["publication_date"] != publication.isoformat()
                or current["event_type"] != event_type
                or current["title"] != title
            ):
                raise NoticeContractError(f"conflicting duplicate notice metadata: {url}")
            current["archive_source_ids"].append(source["source_id"])
            selected += 1
        inventory_rows.append(
            {
                "source_id": source["source_id"],
                "url": source["url"],
                "raw_path": str(raw_path),
                "sha256": observed_hash,
                "bytes": observed_bytes,
                "selected_notice_references": selected,
            }
        )

    records = []
    for value in notices.values():
        records.append(
            {**value, "archive_source_ids": "|".join(sorted(set(value["archive_source_ids"])))}
        )
    records.sort(key=lambda row: (row["publication_date"], row["url"]))
    for index, row in enumerate(records, 1):
        row["notice_id"] = f"JPX-EPL-{index:05d}"
    columns = ["notice_id", "publication_date", "event_type", "title", "url", "archive_source_ids"]
    plan = pd.DataFrame(records, columns=columns)
    if len(plan) != expected_notice_pages:
        raise NoticeContractError(
            f"notice page denominator mismatch: expected {expected_notice_pages}, observed {len(plan)}"
        )
    if expected_notice_pages == EXPECTED_NOTICE_PAGES:
        observed_counts = plan["event_type"].value_counts().to_dict()
        if observed_counts != EXPECTED_NOTICE_TYPE_COUNTS:
            raise NoticeContractError(
                f"researched notice type counts mismatch: expected {EXPECTED_NOTICE_TYPE_COUNTS}, "
                f"observed={observed_counts}"
            )
        if plan["publication_date"].min() != "2013-09-02" or plan["publication_date"].max() != "2026-06-30":
            raise NoticeContractError("researched notice publication endpoints mismatch")
    return plan, pd.DataFrame(inventory_rows)


def read_area_lines(body: bytes | str) -> list[str]:
    """Extract NFKC-normalized, BR-delimited body prose from the official #readArea."""
    soup = BeautifulSoup(body, "html.parser")
    area = soup.select_one("#readArea")
    if area is None:
        raise NoticeParseError("notice is missing #readArea")
    paragraphs = area.select("p.component-text, p.txt")
    if not paragraphs:
        paragraphs = [
            node
            for node in area.find_all("p")
            if node.find_parent(class_=re.compile(r"(?:inquiry|contact)", re.I)) is None
        ]
    lines: list[str] = []
    for paragraph in paragraphs:
        fragment = BeautifulSoup(str(paragraph), "html.parser")
        for br in fragment.find_all("br"):
            br.replace_with("\n")
        for raw_line in fragment.get_text(" ", strip=False).splitlines():
            value = normalize_text(raw_line)
            if value:
                lines.append(value)
    if not lines:
        raise NoticeParseError("#readArea contains no BR-delimited notice prose")
    return lines


def parse_notice_page(
    body: bytes | str,
    *,
    event_type: str,
    archive_publication_date: str,
    source_url: str,
    title: str,
    source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Parse one event row per unique four-character security code."""
    if event_type not in EVENT_TYPES:
        raise NoticeParseError(f"unsupported event type: {event_type}")
    archive_date = date.fromisoformat(archive_publication_date)
    lines = read_area_lines(body)
    text = "\n".join(lines)
    code_matches = list(
        re.finditer(r"(?:\(\s*)?Code\s*:\s*([0-9A-Z]{4})(?:\s*\))?", text, re.I)
    )
    if not code_matches:
        raise NoticeParseError("notice prose contains no four-character security code")
    codes = [match.group(1).upper() for match in code_matches]
    if len(codes) != len(set(codes)):
        raise NoticeParseError(f"notice repeats a security code: {codes}")

    url_date = _url_publication_date(source_url)
    page_date = _page_publication_date(body)
    publication = url_date or page_date or archive_date
    date_consistent = all(
        candidate == publication for candidate in (url_date, page_date, archive_date) if candidate is not None
    )
    effective = _effective_date(text, event_type, publication)
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(code_matches):
        line_start = text.rfind("\n", 0, match.start()) + 1
        next_start = (
            text.rfind("\n", 0, code_matches[index + 1].start()) + 1
            if index + 1 < len(code_matches)
            else len(text)
        )
        segment = text[line_start:next_start]
        code = match.group(1).upper()
        side = _extract_side(segment)
        range_match = re.search(
            r"\b(Upper|Lower) daily price limit\s*:\s*([0-9,]+)\s*yen\s*"
            r"\((?:Upper|Lower) limit will remain\s*([0-9,]+)\s*yen\s*as per normal\)",
            segment,
            re.I,
        )
        if range_match:
            side = range_match.group(1).lower()
            expanded_range = _integer(range_match.group(2))
            normal_range = _integer(range_match.group(3))
        else:
            expanded_range = None
            normal_range = None
        band_match = re.search(
            r"Base price\s*:\s*([0-9,]+)\s*yen\s*;\s*"
            r"Upper daily price limit\s*:\s*([0-9,]+)\s*yen\s*;\s*"
            r"Lower daily price limit\s*:\s*([0-9,]+)\s*yen",
            segment,
            re.I,
        )
        base_price = _integer(band_match.group(1)) if band_match else None
        upper_price = _integer(band_match.group(2)) if band_match else None
        lower_price = _integer(band_match.group(3)) if band_match else None
        observed_multiplier = (
            float(Decimal(expanded_range) / Decimal(normal_range))
            if expanded_range is not None and normal_range
            else None
        )
        rows.append(
            {
                "archive_publication_date": archive_date.isoformat(),
                "publication_date": publication.isoformat(),
                "effective_date": effective.isoformat() if effective else "",
                "event_type": event_type,
                "security_code": code,
                "security_name": _security_name(segment, code),
                "side": side,
                "base_price_yen": base_price,
                "expanded_range_yen": expanded_range,
                "normal_range_yen": normal_range,
                "upper_limit_price_yen": upper_price,
                "lower_limit_price_yen": lower_price,
                "observed_multiplier": observed_multiplier,
                "publication_date_consistent": date_consistent,
                "source_url": source_url,
                "source_sha256": source_sha256,
                "title": normalize_text(title),
            }
        )
    return rows


def validate_event(event: dict[str, Any], sessions: Sequence[str] | set[str]) -> dict[str, Any]:
    """Apply fail-closed code, numeric, tick-rounding, and effective-session checks."""
    session_list = sorted(set(str(value) for value in sessions))
    session_set = set(session_list)
    issues: list[str] = []
    code = str(event.get("security_code", ""))
    if not re.fullmatch(r"[0-9A-Z]{4}", code):
        issues.append("security_code_invalid")
    try:
        publication = date.fromisoformat(str(event.get("publication_date", "")))
    except ValueError:
        publication = None
        issues.append("publication_date_invalid")
    try:
        effective = date.fromisoformat(str(event.get("effective_date", "")))
    except ValueError:
        effective = None
        issues.append("effective_date_missing_or_invalid")
    if event.get("publication_date_consistent") is not True:
        issues.append("publication_date_sources_disagree")
    if publication and publication.isoformat() not in session_set:
        issues.append("publication_date_not_session")
    if effective and effective.isoformat() not in session_set:
        issues.append("effective_date_not_session")
    event_type = str(event.get("event_type", ""))
    expected_effective = _expected_effective_session(publication, event_type, session_list)
    if effective and expected_effective != effective:
        issues.append("effective_session_not_expected")

    expected_multiplier: float | None = None
    numeric_valid = True
    upper_tick_valid = True
    lower_tick_valid = True
    if event_type in ACTIVATION_TYPES:
        numeric_fields = (
            "base_price_yen",
            "expanded_range_yen",
            "normal_range_yen",
            "upper_limit_price_yen",
            "lower_limit_price_yen",
        )
        values = {field: _positive_decimal(event.get(field)) for field in numeric_fields}
        if any(value is None for value in values.values()) or event.get("side") not in {"upper", "lower"}:
            numeric_valid = False
            issues.append("activation_numeric_fields_missing_or_invalid")
        else:
            expected_multiplier = 2.0 if publication and publication < date(2020, 8, 3) else 4.0
            observed = _decimal_or_none(event.get("observed_multiplier"))
            if observed is None or observed != Decimal(str(expected_multiplier)):
                numeric_valid = False
                issues.append("range_multiplier_invalid")
            base = values["base_price_yen"]
            expanded = values["expanded_range_yen"]
            normal = values["normal_range_yen"]
            upper = values["upper_limit_price_yen"]
            lower = values["lower_limit_price_yen"]
            assert all(value is not None for value in (base, expanded, normal, upper, lower))
            upper_target = base + (expanded if event["side"] == "upper" else normal)
            lower_target = max(Decimal(1), base - (expanded if event["side"] == "lower" else normal))
            upper_tick_valid = _valid_tick_rounding(upper_target, upper, "upper")
            lower_tick_valid = _valid_tick_rounding(lower_target, lower, "lower")
            if not upper_tick_valid:
                issues.append("upper_limit_tick_rounding_invalid")
            if not lower_tick_valid:
                issues.append("lower_limit_tick_rounding_invalid")
            if not (upper > base > lower):
                numeric_valid = False
                issues.append("limit_price_order_invalid")
            numeric_valid = numeric_valid and upper_tick_valid and lower_tick_valid

    return {
        "publication_date": event.get("publication_date", ""),
        "event_type": event_type,
        "security_code": code,
        "source_url": event.get("source_url", ""),
        "expected_multiplier": expected_multiplier,
        "observed_multiplier": event.get("observed_multiplier"),
        "numeric_valid": numeric_valid,
        "upper_tick_valid": upper_tick_valid,
        "lower_tick_valid": lower_tick_valid,
        "effective_session_valid": not any("session" in issue for issue in issues),
        "validation_pass": not issues,
        "validation_issues": "|".join(issues),
    }


def execute_notice_fetch(
    plan: pd.DataFrame,
    *,
    state_path: str | Path,
    raw_dir: str | Path,
    max_requests: int | None = None,
    sleep_seconds: float = 0.02,
    offline: bool = False,
    fetcher: Fetcher | None = None,
) -> tuple[pd.DataFrame, int]:
    """Resume the fetch queue while treating every successful page blob as immutable."""
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be non-negative")
    state_file = Path(state_path).resolve()
    raw_root = Path(raw_dir).resolve()
    previous = _read_previous_state(state_file)
    state_rows: list[dict[str, Any]] = []
    requests = 0
    fetch = fetcher or _fetch
    for item in plan.to_dict("records"):
        prior = previous.get(item["notice_id"], {})
        retained = _verified_prior_page(prior, item)
        if retained is not None:
            state_rows.append({**item, **retained})
            continue
        if offline or (max_requests is not None and requests >= max_requests):
            state_rows.append(_pending_state(item, "offline" if offline else "bounded_run_limit"))
            continue
        raw_path = raw_root / f"{item['notice_id']}_{hashlib.sha256(item['url'].encode()).hexdigest()[:12]}.html"
        try:
            status, body, headers = fetch(item["url"])
            requests += 1
            if status == 200 and body:
                read_area_lines(body)
                _write_blob_once(raw_path, body)
                state_rows.append(
                    {
                        **item,
                        "http_status": status,
                        "terminal_status": "success",
                        "content_type": headers.get("Content-Type", headers.get("content-type", "")),
                        "raw_path": str(raw_path),
                        "sha256": sha256_file(raw_path),
                        "bytes": raw_path.stat().st_size,
                        "error": "",
                    }
                )
            else:
                state_rows.append(_failed_state(item, status, f"http_status={status}"))
        except ImmutableArtifactError:
            raise
        except Exception as exc:
            state_rows.append(_failed_state(item, 0, f"{type(exc).__name__}:{exc}"[:1000]))
        _atomic_csv(state_file, pd.DataFrame(state_rows))
        if sleep_seconds and requests:
            time.sleep(sleep_seconds)
    state = pd.DataFrame(state_rows)
    _atomic_csv(state_file, state)
    return state, requests


def parse_archived_notice_pages(
    plan: pd.DataFrame,
    state: pd.DataFrame,
    sessions: Sequence[str] | set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically parse hash-verified archived pages without network access."""
    required_plan = {"notice_id", "publication_date", "event_type", "title", "url"}
    required_state = required_plan | {"terminal_status", "raw_path", "sha256"}
    if not required_plan.issubset(plan.columns) or not required_state.issubset(state.columns):
        raise NoticeContractError("plan or page state schema is invalid for offline parsing")
    if plan["notice_id"].duplicated().any() or state["notice_id"].duplicated().any():
        raise NoticeContractError("plan and page state notice ids must be unique")
    planned = {
        (str(row["notice_id"]), str(row["url"])) for row in plan.to_dict("records")
    }
    observed = {
        (str(row["notice_id"]), str(row["url"])) for row in state.to_dict("records")
    }
    if planned != observed:
        raise NoticeContractError("page state does not exactly reconcile to the notice plan")

    events: list[dict[str, Any]] = []
    quarantine: list[dict[str, str]] = []
    for item in state.to_dict("records"):
        if item["terminal_status"] != "success":
            continue
        raw_path = Path(str(item["raw_path"]))
        if not raw_path.is_file() or sha256_file(raw_path) != str(item["sha256"]):
            raise ImmutableArtifactError(f"retained notice hash mismatch: {item['notice_id']}")
        try:
            parsed = parse_notice_page(
                raw_path.read_bytes(),
                event_type=str(item["event_type"]),
                archive_publication_date=str(item["publication_date"]),
                source_url=str(item["url"]),
                title=str(item["title"]),
                source_sha256=str(item["sha256"]),
            )
            for row in parsed:
                row["notice_id"] = item["notice_id"]
            events.extend(parsed)
        except (OSError, NoticeParseError, ValueError) as exc:
            quarantine.append(
                {
                    "notice_id": str(item["notice_id"]),
                    "url": str(item["url"]),
                    "raw_path": str(raw_path),
                    "sha256": str(item["sha256"]),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
    event_columns = [
        "notice_id",
        "archive_publication_date",
        "publication_date",
        "effective_date",
        "event_type",
        "security_code",
        "security_name",
        "side",
        "base_price_yen",
        "expanded_range_yen",
        "normal_range_yen",
        "upper_limit_price_yen",
        "lower_limit_price_yen",
        "observed_multiplier",
        "publication_date_consistent",
        "source_url",
        "source_sha256",
        "title",
    ]
    event_frame = pd.DataFrame(events, columns=event_columns)
    if not event_frame.empty:
        if event_frame.duplicated(["notice_id", "security_code"]).any():
            raise NoticeParseError("event output contains duplicate notice/security keys")
        event_frame = event_frame.sort_values(
            ["publication_date", "notice_id", "security_code"]
        ).reset_index(drop=True)
    quality_columns = [
        "publication_date",
        "event_type",
        "security_code",
        "source_url",
        "expected_multiplier",
        "observed_multiplier",
        "numeric_valid",
        "upper_tick_valid",
        "lower_tick_valid",
        "effective_session_valid",
        "validation_pass",
        "validation_issues",
    ]
    quality = pd.DataFrame(
        [validate_event(row, sessions) for row in event_frame.to_dict("records")],
        columns=quality_columns,
    )
    quarantine_frame = pd.DataFrame(
        quarantine, columns=["notice_id", "url", "raw_path", "sha256", "error"]
    )
    return event_frame, quality, quarantine_frame


def parse_archived_notice_corpus(
    *,
    archive_manifest_path: str | Path,
    page_state_path: str | Path,
    session_calendar_path: str | Path,
    expected_archive_sources: int = EXPECTED_ARCHIVE_SOURCES,
    expected_notice_pages: int = EXPECTED_NOTICE_PAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Offline API over retained monthly JSON and page HTML, with exact reconciliation."""
    plan, _ = build_notice_plan(
        archive_manifest_path,
        expected_archive_sources=expected_archive_sources,
        expected_notice_pages=expected_notice_pages,
    )
    state = pd.read_csv(page_state_path, dtype=str, keep_default_na=False)
    sessions = _load_sessions(Path(session_calendar_path).resolve())
    return parse_archived_notice_pages(plan, state, sessions)


def produce_jpx_exceptional_price_limits(
    *,
    run_id: str,
    archive_manifest_path: str | Path,
    session_calendar_path: str | Path,
    artifact_root: str | Path = "artifacts/jpx_exceptional_price_limit_notices",
    expected_archive_sources: int = EXPECTED_ARCHIVE_SOURCES,
    expected_notice_pages: int = EXPECTED_NOTICE_PAGES,
    expected_event_rows: int = EXPECTED_EVENT_ROWS,
    max_requests: int | None = None,
    sleep_seconds: float = 0.02,
    offline: bool = False,
    fetcher: Fetcher | None = None,
) -> JPXExceptionalPriceLimitArtifacts:
    """Produce a resumable and hash-verified official JPX English notice bundle."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be non-negative")
    archive_manifest = Path(archive_manifest_path).resolve()
    session_calendar = Path(session_calendar_path).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc

    try:
        config = {
            "schema_version": "1.0",
            "run_id": run_id,
            "archive_manifest": str(archive_manifest),
            "archive_manifest_sha256": sha256_file(archive_manifest),
            "session_calendar": str(session_calendar),
            "session_calendar_sha256": sha256_file(session_calendar),
            "expected_archive_sources": expected_archive_sources,
            "expected_notice_pages": expected_notice_pages,
            "expected_event_rows": expected_event_rows,
        }
        config_path = root / "producer_config.json"
        _verify_or_create_config(config_path, config)
        finalized = _load_finalized(root, config)
        if finalized is not None:
            return JPXExceptionalPriceLimitArtifacts(run_id, root, root / "manifest.json", finalized)

        plan, inventory = build_notice_plan(
            archive_manifest,
            expected_archive_sources=expected_archive_sources,
            expected_notice_pages=expected_notice_pages,
        )
        sessions = _load_sessions(session_calendar)
        plan_path = root / "notice_plan.csv"
        inventory_path = root / "archive_inventory.csv"
        state_path = root / "page_state.csv"
        _atomic_csv(plan_path, plan)
        _atomic_csv(inventory_path, inventory)

        state, requests = execute_notice_fetch(
            plan,
            state_path=state_path,
            raw_dir=root / "raw" / "notice_pages",
            max_requests=max_requests,
            sleep_seconds=sleep_seconds,
            offline=offline,
            fetcher=fetcher,
        )
        event_frame, quality, quarantine = parse_archived_notice_pages(plan, state, sessions)
        events_path = root / "events.csv"
        quality_path = root / "event_quality.csv"
        errors_path = root / "parse_quarantine.csv"
        _atomic_csv(events_path, event_frame)
        _atomic_csv(quality_path, quality)
        _atomic_csv(errors_path, quarantine)

        success_pages = int(state["terminal_status"].eq("success").sum())
        pending_pages = int(state["terminal_status"].eq("pending").sum())
        failed_pages = int(state["terminal_status"].eq("retryable_failure").sum())
        denominator_complete = success_pages == expected_notice_pages and not pending_pages and not failed_pages
        parsed_denominator_complete = denominator_complete and quarantine.empty and len(event_frame) == expected_event_rows
        validation_failures = int(quality["validation_pass"].ne(True).sum()) if not quality.empty else 0
        producer_blockers: list[str] = []
        if failed_pages:
            producer_blockers.append(f"notice_fetch_failures={failed_pages}")
        if not quarantine.empty:
            producer_blockers.append(f"notice_parse_failures={len(quarantine)}")
        if denominator_complete and len(event_frame) != expected_event_rows:
            producer_blockers.append(
                f"event_row_denominator_mismatch={len(event_frame)}/{expected_event_rows}"
            )
        if validation_failures:
            producer_blockers.append(f"strict_event_validation_failures={validation_failures}")

        output_paths = (plan_path, inventory_path, state_path, events_path, quality_path, errors_path)
        immutable = parsed_denominator_complete
        status = (
            "complete_denominator_partial_component"
            if immutable
            else "in_progress"
            if pending_pages
            else "incomplete_denominator"
        )
        manifest = {
            "schema_version": "1.0-immutable-resumable-partial",
            "run_id": run_id,
            "artifact_type": "jpx_english_exceptional_price_limit_notice_events",
            "status": status,
            "component": "temporary_price_limit_events",
            "official_jpx_source": True,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "bounded_public_window_scorecard_credit": False,
            "pre_2013_complete": False,
            "post_2020_removal_complete": False,
            "immutable": immutable,
            "resumable": True,
            "single_writer_lock": True,
            "atomic_writes": True,
            "network_accessed": bool(requests),
            "research_scope": f"{RESEARCH_START}_through_{RESEARCH_END}",
            "expected_months": expected_archive_sources,
            "expected_notice_pages": expected_notice_pages,
            "denominator_reconciliation": {
                "expected_months": expected_archive_sources,
                "observed_month_sources": len(inventory),
                "month_source_denominator_exact": len(inventory) == expected_archive_sources,
                "expected_notice_pages": expected_notice_pages,
                "planned_notice_pages": len(plan),
                "completed_notice_pages": success_pages,
                "notice_plan_denominator_exact": len(plan) == expected_notice_pages,
                "completed_page_denominator_exact": success_pages == expected_notice_pages,
                "page_state_exact_plan": {
                    (str(row["notice_id"]), str(row["url"])) for row in state.to_dict("records")
                }
                == {
                    (str(row["notice_id"]), str(row["url"])) for row in plan.to_dict("records")
                },
            },
            "counts": {
                "archive_sources": len(inventory),
                "planned_notice_pages": len(plan),
                "completed_notice_pages": success_pages,
                "pending_notice_pages": pending_pages,
                "failed_notice_pages": failed_pages,
                "parsed_event_rows": len(event_frame),
                "page_parse_failures": len(quarantine),
                "strict_event_validation_failures": validation_failures,
            },
            "event_type_counts": event_frame["event_type"].value_counts().sort_index().to_dict()
            if not event_frame.empty
            else {},
            "producer_blockers": producer_blockers,
            "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
            "critical_blockers": [*producer_blockers, *REQUIRED_COMPLETION_BLOCKERS],
            "archive_manifest": str(archive_manifest),
            "archive_manifest_sha256": config["archive_manifest_sha256"],
            "session_calendar": str(session_calendar),
            "session_calendar_sha256": config["session_calendar_sha256"],
            "producer_config_sha256": sha256_file(config_path),
            "artifacts": {path.stem: str(path) for path in output_paths},
            "per_page_sha256": {
                str(row["notice_id"]): str(row["sha256"])
                for row in state.loc[state["terminal_status"].eq("success")].to_dict("records")
            },
            "output_sha256": {path.name: sha256_file(path) for path in output_paths},
            "promotion_policy": dict(PROMOTION_POLICY),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JPXExceptionalPriceLimitArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce hash-verified JPX English exceptional price-limit notice events."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--artifact-root", default="artifacts/jpx_exceptional_price_limit_notices")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--expected-archive-sources", type=int, default=EXPECTED_ARCHIVE_SOURCES)
    parser.add_argument("--expected-notice-pages", type=int, default=EXPECTED_NOTICE_PAGES)
    parser.add_argument("--expected-event-rows", type=int, default=EXPECTED_EVENT_ROWS)
    return parser


def main(argv: Sequence[str] | None = None, *, fetcher: Fetcher | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    result = produce_jpx_exceptional_price_limits(
        run_id=args.run_id,
        archive_manifest_path=args.archive_manifest,
        session_calendar_path=args.session_calendar,
        artifact_root=args.artifact_root,
        expected_archive_sources=args.expected_archive_sources,
        expected_notice_pages=args.expected_notice_pages,
        expected_event_rows=args.expected_event_rows,
        max_requests=args.max_requests,
        sleep_seconds=args.sleep_seconds,
        offline=args.offline,
        fetcher=fetcher,
    )
    print(json.dumps({"manifest": str(result.manifest_path), "status": result.manifest["status"]}, sort_keys=True))
    return 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _researched_archive_urls() -> set[str]:
    months = pd.period_range("2013-09", "2024-08", freq="M")
    explicit = {f"{JPX_ROOT}/english/news/news_ym_{period.strftime('%Y%m')}.json" for period in months}
    rolling = {f"{JPX_ROOT}/english/news/news_ym_{index:02d}.json" for index in range(1, 23)}
    return explicit | rolling


def _walk_news_items(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"title", "url", "updated_date"}.issubset(value):
            yield value
        for child in value.values():
            yield from _walk_news_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_news_items(child)


def _publication_date(item: dict[str, Any]) -> date | None:
    value = item.get("updated_date")
    if not isinstance(value, dict):
        return None
    try:
        return date(int(value["year"]), int(value["month"]), int(value["day"]))
    except (KeyError, TypeError, ValueError):
        return None


def _url_publication_date(source_url: str) -> date | None:
    match = re.search(r"/([0-9]{8})-[0-9]+\.html(?:\?.*)?$", source_url)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _page_publication_date(body: bytes | str) -> date | None:
    soup = BeautifulSoup(body, "html.parser")
    node = soup.select_one("#readArea .date-news")
    if node is None:
        return None
    value = normalize_text(node.get_text(" ", strip=True))
    match = re.search(r"([A-Za-z]{3})\.?\s+(\d{1,2}),\s+(\d{4})", value)
    if not match:
        return None
    try:
        month = datetime.strptime(match.group(1), "%b").month
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def _effective_date(text: str, event_type: str, publication: date) -> date | None:
    if event_type == "cancel":
        return publication
    patterns = (
        r"next business day\s*\(\s*([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\s*"
        r"(?:\)|(?=as follows\b))",
        r"return the daily price limits to normal on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        r"continue to (?:broaden|expand)[^.]{0,120}?\s+on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        r"continue to be (?:broadened|expanded)\s+on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        r"on or after\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        try:
            month = datetime.strptime(match.group(1)[:3], "%b").month
            year = int(match.group(3)) if match.group(3) else publication.year + int(month < publication.month - 6)
            return date(year, month, int(match.group(2)))
        except (TypeError, ValueError):
            continue
    return None


def _extract_side(segment: str) -> str:
    patterns = (
        r"(?:broaden|expand)(?:ed|ing)?\s+(?:only\s+)?(?:its\s+|the\s+)?(upper|lower)\b",
        r"\b(upper|lower) daily price limit\b[^.]{0,100}\b(?:broaden|expand)(?:ed|ing)?\b",
        r"\b(upper|lower) daily price limit\b",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, re.I)
        if match:
            return match.group(1).lower()
    return ""


def _security_name(segment: str, code: str) -> str:
    match = re.search(rf"([^()\n]{{1,180}}?)\s*\(\s*Code\s*:\s*{re.escape(code)}\s*\)", segment, re.I)
    if not match:
        return ""
    value = match.group(1).strip(" :-")
    value = re.sub(
        r"^(?:The (?:upper|lower) daily price limit for|The trading conditions? of|Today, trades in|Trades in)\s+",
        "",
        value,
        flags=re.I,
    )
    return value.strip(" :-")


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _positive_decimal(value: Any) -> Decimal | None:
    parsed = _decimal_or_none(value)
    return parsed if parsed is not None and parsed > 0 else None


def _valid_tick_rounding(target: Decimal, actual: Decimal, direction: str) -> bool:
    if actual <= 0:
        return False
    delta = actual - target if direction == "upper" else target - actual
    if delta < 0:
        return False
    ticks = tuple(
        Decimal(value)
        for value in ("0.1", "0.5", "1", "5", "10", "50", "100", "500", "1000", "5000", "10000", "50000", "100000")
    )
    return any(delta < tick and actual % tick == 0 for tick in ticks)


def _expected_effective_session(
    publication: date | None, event_type: str, sessions: Sequence[str]
) -> date | None:
    if publication is None:
        return None
    if event_type == "cancel":
        return publication
    publication_text = publication.isoformat()
    for value in sessions:
        if value > publication_text:
            return date.fromisoformat(value)
    return None


def _load_sessions(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["session_date"])
    else:
        frame = pd.read_csv(path, dtype=str)
    if "session_date" not in frame.columns:
        raise NoticeContractError("session calendar must contain session_date")
    sessions = sorted(frame["session_date"].dropna().astype(str).unique())
    if not sessions or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in sessions):
        raise NoticeContractError("session calendar contains no valid ISO session dates")
    return sessions


def _verify_or_create_config(path: Path, config: dict[str, Any]) -> None:
    if path.is_file():
        observed = _read_json(path)
        if observed != config:
            raise ImmutableArtifactError("producer configuration changed for an existing run_id")
    else:
        _atomic_json(path, config)


def _load_finalized(root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("immutable") is not True:
        return None
    if manifest.get("archive_manifest_sha256") != config["archive_manifest_sha256"]:
        raise ImmutableArtifactError("finalized archive manifest hash changed")
    if manifest.get("session_calendar_sha256") != config["session_calendar_sha256"]:
        raise ImmutableArtifactError("finalized session calendar hash changed")
    for filename, expected_hash in manifest.get("output_sha256", {}).items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ImmutableArtifactError(f"finalized output hash mismatch: {filename}")
    state = pd.read_csv(root / "page_state.csv", dtype=str, keep_default_na=False)
    for row in state.loc[state["terminal_status"].eq("success")].to_dict("records"):
        raw_path = Path(row["raw_path"])
        if not raw_path.is_file() or sha256_file(raw_path) != row["sha256"]:
            raise ImmutableArtifactError(f"retained notice hash mismatch: {row['notice_id']}")
    return manifest


def _read_previous_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "notice_id" not in frame.columns or frame["notice_id"].duplicated().any():
        raise ImmutableArtifactError("existing page state schema or keys are invalid")
    return frame.set_index("notice_id").to_dict("index")


def _verified_prior_page(prior: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    if not prior or prior.get("terminal_status") != "success":
        return None
    if prior.get("url") != item["url"]:
        raise ImmutableArtifactError(f"retained notice URL changed: {item['notice_id']}")
    raw_path = Path(str(prior.get("raw_path", "")))
    if not raw_path.is_file() or sha256_file(raw_path) != prior.get("sha256"):
        raise ImmutableArtifactError(f"retained notice hash mismatch: {item['notice_id']}")
    return {key: value for key, value in prior.items() if key not in item}


def _pending_state(item: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **item,
        "http_status": 0,
        "terminal_status": "pending",
        "content_type": "",
        "raw_path": "",
        "sha256": "",
        "bytes": 0,
        "error": error,
    }


def _failed_state(item: dict[str, Any], status: int, error: str) -> dict[str, Any]:
    return {
        **item,
        "http_status": status,
        "terminal_status": "retryable_failure",
        "content_type": "",
        "raw_path": "",
        "sha256": "",
        "bytes": 0,
        "error": error,
    }


def _write_blob_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise ImmutableArtifactError(f"refusing to overwrite retained blob: {path}")
        return
    _atomic_bytes(path, payload)


def _resolve_input_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "quant-lab-jpx-exceptional-price-limit-notices/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(16 * 1024 * 1024 + 1)
            if len(body) > 16 * 1024 * 1024:
                raise NoticeContractError(f"notice response exceeds 16 MiB: {url}")
            return int(response.status), body, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(16 * 1024 * 1024 + 1), dict(exc.headers.items())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImmutableArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ImmutableArtifactError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
