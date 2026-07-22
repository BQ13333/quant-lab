from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"


@dataclass(frozen=True)
class JapanPriceLimitEventArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


def classify_notice_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(title))
    value = re.sub(r"\s+", " ", value.strip())
    if re.search(r"\b(?:ETF|ETN)\b|subscription warrant|securities option", value, re.I):
        return ""
    patterns = (
        ("PRELIMINARY", r"^(?:Expansion of daily price limits|Broadening of Daily Price Limits)\s*\(Preliminary Notice\)\s*[:\-]"),
        ("CANCELLED", r"^(?:Handling of daily price limits\s*\(Daily price limits will not be expanded\)|Announced Broadening of Daily Price Limits to be Cancelled)\s*[:\-]"),
        ("BROADENING", r"^(?:Expansion of daily price limits\s*\(Daily price limits? will be expanded\)|Daily Price Limits to be Broadened)\s*[:\-]"),
        ("CONTINUATION", r"^(?:Continued expansion of daily price limits\s*\(Daily price limits will continue to be expanded\)|Continue Broadened Daily Price Limits|Daily Price Limits Continue to be Broadened)\s*[:\-]"),
        ("RETURN_NORMAL", r"^(?:Removal of expanded daily price limits\s*\(Return to normal daily price limits\)|Daily Price Limits to be Returned to Normal)\s*[:\-]"),
    )
    for event_type, pattern in patterns:
        if re.search(pattern, value, re.I):
            return event_type
    return ""


def build_notice_plan(archive_manifest_path: str | Path, cutoff_date: str = "2026-06-30") -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(archive_manifest_path)
    cutoff = date.fromisoformat(cutoff_date)
    issues: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for source in manifest.to_dict("records"):
        raw = Path(str(source["raw_path"]))
        if int(source["http_status"]) != 200 or not raw.is_file() or _sha256(raw) != source["sha256"]:
            issues.append(f"archive_source_invalid:{source['source_id']}")
            continue
        payload = json.loads(raw.read_text(encoding="utf-8"))
        for item in _walk_news_items(payload):
            event_type = classify_notice_title(str(item.get("title", "")))
            published = _publication_date(item)
            target = str(item.get("url", ""))
            if not event_type or not published or published > cutoff or not target:
                continue
            url = urljoin(JPX_ROOT, target)
            if not re.fullmatch(
                r"https://www\.jpx\.co\.jp/english/news/(?:detail/detail_[0-9]+|1030/[A-Za-z0-9_-]+)\.html",
                url,
            ):
                issues.append(f"selected_notice_url_invalid:{url}")
                continue
            record = rows.setdefault(
                url,
                {
                    "publication_date": published.isoformat(),
                    "event_type": event_type,
                    "title": str(item["title"]).strip(),
                    "url": url,
                    "archive_source_ids": [],
                },
            )
            record["archive_source_ids"].append(str(source["source_id"]))
    plan = pd.DataFrame(
        [
            {**value, "archive_source_ids": "|".join(sorted(set(value["archive_source_ids"])))}
            for value in rows.values()
        ]
    )
    if not plan.empty:
        plan = plan.sort_values(["publication_date", "url"]).reset_index(drop=True)
        plan.insert(0, "plan_row_id", [f"JPL-{index:05d}" for index in range(1, len(plan) + 1)])
    return plan, issues


def write_japan_price_limit_event_history(
    *,
    run_id: str,
    archive_manifest_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_price_limit_event_history",
    cutoff_date: str = "2026-06-30",
    max_pages: int = 0,
    sleep_seconds: float = 0.02,
    fetcher: Fetcher | None = None,
) -> JapanPriceLimitEventArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if max_pages < 0 or sleep_seconds < 0:
        raise ValueError("max_pages and sleep_seconds must be non-negative")
    archive_manifest = Path(archive_manifest_path).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        plan, archive_issues = build_notice_plan(archive_manifest, cutoff_date)
        plan_path = root / "notice_plan.csv"
        _atomic_csv(plan_path, plan)
        raw_dir = root / "raw/notice_pages"
        raw_dir.mkdir(parents=True, exist_ok=True)
        existing_state_path = root / "page_state.csv"
        existing = pd.read_csv(existing_state_path).set_index("plan_row_id").to_dict("index") if existing_state_path.is_file() else {}
        fetch = fetcher or _fetch
        state_rows: list[dict[str, Any]] = []
        new_requests = 0
        for item in plan.to_dict("records"):
            prior = existing.get(item["plan_row_id"], {})
            prior_path = Path(str(prior.get("raw_path", "")))
            if (
                prior.get("terminal_status") == "success"
                and prior.get("url") == item["url"]
                and prior_path.is_file()
                and _sha256(prior_path) == prior.get("sha256")
            ):
                state_rows.append({**item, **prior})
                continue
            raw_path = raw_dir / f"{item['plan_row_id']}_{hashlib.sha256(item['url'].encode()).hexdigest()[:12]}.html"
            if max_pages and new_requests >= max_pages:
                state_rows.append({**item, "http_status": 0, "terminal_status": "pending", "raw_path": "", "sha256": "", "bytes": 0, "error": "bounded_run_limit"})
                continue
            try:
                status, body, headers = fetch(item["url"])
                new_requests += 1
                if status == 200 and body:
                    _atomic_bytes(raw_path, body)
                    terminal = "success"
                    error = ""
                else:
                    terminal = "retryable_failure"
                    error = f"http_status={status}"
            except Exception as exc:
                status, headers, terminal, error = 0, {}, "retryable_failure", f"{type(exc).__name__}:{exc}"[:1000]
            state_rows.append(
                {
                    **item,
                    "http_status": status,
                    "terminal_status": terminal,
                    "content_type": headers.get("Content-Type", headers.get("content-type", "")),
                    "raw_path": str(raw_path) if raw_path.is_file() else "",
                    "sha256": _sha256(raw_path) if raw_path.is_file() else "",
                    "bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
                    "error": error,
                }
            )
            if new_requests and new_requests % 25 == 0:
                _atomic_csv(existing_state_path, pd.DataFrame(state_rows))
            if sleep_seconds and new_requests:
                time.sleep(sleep_seconds)
        state = pd.DataFrame(state_rows)
        _atomic_csv(existing_state_path, state)

        event_rows: list[dict[str, Any]] = []
        parse_failures: list[str] = []
        for item in state.loc[state["terminal_status"].eq("success")].to_dict("records"):
            parsed = parse_price_limit_notice(
                Path(item["raw_path"]).read_bytes(),
                event_type=item["event_type"],
                publication_date=item["publication_date"],
                source_url=item["url"],
                title=item["title"],
            )
            if not parsed:
                parse_failures.append(item["plan_row_id"])
            event_rows.extend(parsed)
        events = pd.DataFrame(event_rows)
        if not events.empty:
            events = events.drop_duplicates(["publication_date", "event_type", "code", "source_url"]).sort_values(["publication_date", "code", "event_type"]).reset_index(drop=True)
        events_path = root / "price_limit_events.csv"
        _atomic_csv(events_path, events)
        quality = event_quality(events)
        quality_path = root / "event_quality.csv"
        _atomic_csv(quality_path, quality)

        completed = int(state["terminal_status"].eq("success").sum())
        terminal_failures = int(state["terminal_status"].eq("retryable_failure").sum())
        pending = len(state) - completed - terminal_failures
        regime_mismatches = int(quality["regime_multiplier_valid"].eq(False).sum()) if not quality.empty else 0
        producer_blockers = list(archive_issues)
        if terminal_failures:
            producer_blockers.append(f"page_fetch_failures={terminal_failures}")
        if completed == len(plan) and parse_failures:
            producer_blockers.append(f"page_parse_failures={len(parse_failures)}")
        if regime_mismatches:
            producer_blockers.append(f"regime_multiplier_mismatches={regime_mismatches}")
        remaining = [
            "official issue-specific price-limit archive before 2013-09 is not yet proven complete",
            "post-2020 return-to-normal dates require admitted daily price/calendar evidence because JPX discontinued removal notices",
        ]
        complete_crawl = completed == len(plan) and terminal_failures == 0
        status = "partial_candidate" if complete_crawl and not producer_blockers else "in_progress" if pending else "failed_candidate"
        manifest = {
            "schema_version": "1.0-resumable-partial",
            "run_id": run_id,
            "artifact_type": "japan_exceptional_price_limit_event_history_candidate",
            "status": status,
            "component": "temporary_price_limit_events",
            "official_jpx_source": True,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "independent_validation_required": True,
            "cutoff_date": cutoff_date,
            "counts": {
                "archive_sources": 154,
                "planned_notice_pages": len(plan),
                "completed_notice_pages": completed,
                "pending_notice_pages": pending,
                "failed_notice_pages": terminal_failures,
                "parsed_event_rows": len(events),
                "page_parse_failures": len(parse_failures),
                "regime_multiplier_mismatches": regime_mismatches,
            },
            "event_type_counts": events["event_type"].value_counts().sort_index().to_dict() if not events.empty else {},
            "producer_blockers": producer_blockers,
            "remaining_completion_blockers": remaining,
            "critical_blockers": [*producer_blockers, *remaining],
            "parse_failure_plan_rows": parse_failures,
            "archive_manifest": str(archive_manifest),
            "archive_manifest_sha256": _sha256(archive_manifest),
            "artifacts": {"notice_plan": str(plan_path), "page_state": str(existing_state_path), "price_limit_events": str(events_path), "event_quality": str(quality_path)},
            "output_sha256": {path.name: _sha256(path) for path in (plan_path, existing_state_path, events_path, quality_path)},
            "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanPriceLimitEventArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def parse_price_limit_notice(body: bytes, *, event_type: str, publication_date: str, source_url: str, title: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    paragraphs = [" ".join(node.stripped_strings) for node in soup.find_all("p")]
    text = " ".join(paragraphs) or soup.get_text(" ", strip=True)
    code_matches = list(re.finditer(r"\(Code\s*:\s*([0-9A-Z]{4})\)", text, re.I))
    if not code_matches:
        code_matches = list(re.finditer(r"\bCode\s*[:：]?\s*([0-9A-Z]{4})\b", text, re.I))
    canonical_publication_date = _notice_url_date(source_url) or publication_date
    effective = _effective_event_date(text, event_type, canonical_publication_date)
    rows = []
    for index, code_match in enumerate(code_matches):
        code = code_match.group(1).upper()
        segment_end = code_matches[index + 1].start() if index + 1 < len(code_matches) else len(text)
        segment = text[code_match.end() : segment_end]
        issue_name = _issue_name_before_code(text, code_match.start())
        limits = re.search(
            r"\b(Upper|Lower) daily price limit\s*:\s*([\d,]+)\s*yen\s*"
            r"\((?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen\s*as per normal\)",
            segment,
            re.I,
        )
        if limits:
            side = limits.group(1).upper()
            expanded = int(limits.group(2).replace(",", ""))
            normal = int(limits.group(3).replace(",", ""))
        else:
            side_match = re.search(
                r"(?:broaden(?:ed|ing)?|expand(?:ed|ing)?|removal of)\s+(?:only\s+)?(?:its\s+)?(?:the\s+)?(upper|lower)",
                segment,
                re.I,
            )
            side = side_match.group(1).upper() if side_match else ""
            expanded = _first_int(
                segment,
                rf"{side.title() if side else '(?:Upper|Lower)'} daily price limit\s*:\s*([\d,]+)\s*yen\s*\(",
            )
            normal = _first_int(
                segment, r"(?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen"
            )
        base = _first_int(segment, r"Base price\s*:\s*([\d,]+)\s*yen")
        multiplier = (
            (expanded / normal)
            if expanded and normal
            else 2.0
            if "twice the range" in segment.lower()
            else 4.0
            if ("four times" in segment.lower() or "quadruple" in segment.lower())
            else None
        )
        rows.append(
            {
                "publication_date": canonical_publication_date,
                "archive_publication_date": publication_date,
                "effective_date": effective,
                "event_type": event_type,
                "code": code.upper(),
                "issue_name": issue_name,
                "side": side,
                "base_price_yen": base,
                "expanded_range_yen": expanded,
                "normal_range_yen": normal,
                "observed_multiplier": multiplier,
                "source_url": source_url,
                "title": title,
            }
        )
    return rows


def _notice_url_date(source_url: str) -> str:
    match = re.search(r"/([0-9]{8})-[0-9]+\.html(?:\?.*)?$", source_url)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return ""


def _issue_name_before_code(text: str, code_start: int) -> str:
    prefix = text[max(0, code_start - 180) : code_start]
    boundary = max(
        prefix.rfind(". "),
        prefix.rfind("; "),
        prefix.rfind(") "),
    )
    candidate = prefix[boundary + 2 :] if boundary >= 0 else prefix
    candidate = re.sub(
        r"^(?:trades in|the .*? for|the trading conditions of)\s+",
        "",
        candidate,
        flags=re.I,
    )
    return candidate.strip(" :-")


def event_quality(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["publication_date", "event_type", "code", "expected_multiplier", "observed_multiplier", "regime_multiplier_valid"])
    rows = []
    for row in events.to_dict("records"):
        expected = 2.0 if row["publication_date"] < "2020-08-03" else 4.0
        observed = row.get("observed_multiplier")
        applies = row["event_type"] in {"BROADENING", "CONTINUATION"} and pd.notna(observed)
        valid = bool(abs(float(observed) - expected) < 1e-9) if applies else True
        rows.append({"publication_date": row["publication_date"], "event_type": row["event_type"], "code": row["code"], "expected_multiplier": expected if applies else None, "observed_multiplier": observed, "regime_multiplier_valid": valid})
    return pd.DataFrame(rows)


def _effective_event_date(text: str, event_type: str, publication_date: str) -> str:
    published = date.fromisoformat(publication_date)
    patterns = [
        r"next business day\s*\(([A-Za-z]+)\s*(\d{1,2})\)",
        r"return the daily price limits to normal on\s+([A-Za-z]+)\s*(\d{1,2})",
        r"continue to broaden[^.]*? on\s+([A-Za-z]+)\s*(\d{1,2})",
        r"on\s+([A-Za-z]+)\s*(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        try:
            month = datetime.strptime(match.group(1)[:3], "%b").month
            year = published.year + int(month < published.month - 6)
            return date(year, month, int(match.group(2))).isoformat()
        except ValueError:
            continue
    return publication_date


def _first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.I)
    return int(match.group(1).replace(",", "")) if match else None


def _walk_news_items(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "title" in value and "url" in value and "updated_date" in value:
            rows.append(value)
        for child in value.values():
            rows.extend(_walk_news_items(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_news_items(child))
    return rows


def _publication_date(item: dict[str, Any]) -> date | None:
    value = item.get("updated_date", {})
    try:
        return date(int(value["year"]), int(value["month"]), int(value["day"]))
    except (KeyError, TypeError, ValueError):
        return None


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-jpx-price-limit-events/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(16 * 1024 * 1024 + 1), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
