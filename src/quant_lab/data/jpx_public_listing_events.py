from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
BASE = "https://www.jpx.co.jp"
SOURCE_PAGES = {
    "delisting": f"{BASE}/english/listing/stocks/delisted/index.html",
    "new_listing": f"{BASE}/english/listing/stocks/new/index.html",
}
USER_AGENT = "quant-lab-jpx-public-archive/1.0"


@dataclass(frozen=True)
class JPXPublicListingArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    plan_path: Path
    page_status_path: Path
    delisting_path: Path
    new_listing_path: Path
    manifest: dict[str, Any]


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


def discover_archive_pages(html: bytes | str, current_url: str, event_type: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    selected = soup.select_one("select.backnumber")
    options = selected.find_all("option") if selected else []
    if not options:
        options = [soup.new_tag("option", value=urllib.parse.urlparse(current_url).path)]
        options[0].string = str(datetime.now(UTC).year)
    for option in options:
        raw = str(option.get("value", "")).strip()
        if not raw:
            continue
        label = " ".join(option.stripped_strings)
        year_match = re.search(r"(20\d{2})", label)
        rows.append({
            "event_type": event_type,
            "archive_year": year_match.group(1) if year_match else "",
            "url": urllib.parse.urljoin(current_url, raw),
        })
    return pd.DataFrame(rows, columns=["event_type", "archive_year", "url"]).drop_duplicates("url")


def parse_delisting_html(html: bytes | str, source_url: str, archive_year: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame(columns=_delisting_columns())
    for table in tables:
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.stripped_strings) for cell in tr.find_all("td")]
            if len(cells) < 5:
                continue
            event_date = _parse_english_date(cells[0])
            code = cells[2].strip().upper()
            if not event_date or not code:
                continue
            rows.append({
                "event_date": event_date,
                "issue_name": cells[1],
                "code": code,
                "market_segment": cells[3],
                "reason": cells[4],
                "archive_year": archive_year,
                "source_url": source_url,
                "known_at_utc": "",
                "known_at_status": "missing_exact_decision_or_publication_timestamp",
            })
    return pd.DataFrame(rows, columns=_delisting_columns())


def parse_new_listing_html(html: bytes | str, source_url: str, archive_year: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows: list[dict[str, str]] = []
    if not table:
        return pd.DataFrame(columns=_new_listing_columns())
    trs = table.find_all("tr")
    index = 0
    while index < len(trs):
        first = [" ".join(cell.stripped_strings) for cell in trs[index].find_all("td")]
        if len(first) >= 6:
            listing_date = _parse_english_date(first[0])
            approval_date = _parse_english_date(first[1])
            code = first[3].strip().upper()
            if listing_date and code:
                rows.append({
                    "listing_date": listing_date,
                    "approval_date": approval_date,
                    "issue_name": first[2],
                    "code": code,
                    "market_segment": first[4],
                    "trading_unit": "",
                    "archive_year": archive_year,
                    "source_url": source_url,
                    "known_at_utc": "",
                    "known_at_status": "approval_date_has_no_intraday_publication_timestamp",
                })
                index += 1
                continue
        if len(first) < 8:
            index += 1
            continue
        second = [" ".join(cell.stripped_strings) for cell in trs[index + 1].find_all("td")] if index + 1 < len(trs) else []
        dates = re.findall(r"[A-Z][a-z]{2}\.\s+\d{1,2},\s+20\d{2}", first[0])
        listing_date = _parse_english_date(dates[0]) if dates else ""
        approval_date = _parse_english_date(dates[1]) if len(dates) > 1 else ""
        if listing_date and first[2].strip():
            rows.append({
                "listing_date": listing_date,
                "approval_date": approval_date,
                "issue_name": first[1],
                "code": first[2].strip().upper(),
                "market_segment": second[0] if second else "",
                "trading_unit": first[7],
                "archive_year": archive_year,
                "source_url": source_url,
                "known_at_utc": "",
                "known_at_status": "approval_date_has_no_intraday_publication_timestamp",
            })
        index += 2
    return pd.DataFrame(rows, columns=_new_listing_columns())


def run_jpx_public_listing_events(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/jpx_public_listing_events",
    max_pages: int = 20,
    pacing_seconds: float = 0.25,
    retries: int = 3,
    fetcher: Fetcher | None = None,
) -> JPXPublicListingArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    if max_pages <= 0 or retries < 0 or pacing_seconds < 0:
        raise ValueError("bounds must be non-negative and max_pages positive")
    fetch = fetcher or _fetch
    root = Path(artifact_root) / run_id
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan_path = root / "request_plan.csv"
    status_path = root / "page_status.csv"
    delisting_path = root / "delisting_events.csv"
    new_listing_path = root / "new_listing_events.csv"
    manifest_path = root / "manifest.json"

    if plan_path.is_file():
        plan = pd.read_csv(plan_path, dtype=str).fillna("")
    else:
        discovered: list[pd.DataFrame] = []
        for event_type, url in SOURCE_PAGES.items():
            status, body, _ = fetch(url)
            if status != 200:
                raise RuntimeError(f"archive discovery failed for {event_type}: HTTP {status}")
            discovered.append(discover_archive_pages(body, url, event_type))
        plan = pd.concat(discovered, ignore_index=True).drop_duplicates("url").sort_values(
            ["event_type", "archive_year"], ascending=[True, False]
        )
        plan.insert(0, "page_id", [f"page-{index:03d}" for index in range(1, len(plan) + 1)])
        _atomic_write_csv(plan_path, plan)
    status_frame = pd.read_csv(status_path, dtype=str).fillna("") if status_path.is_file() else pd.DataFrame(
        columns=["page_id", "status", "http_status", "attempts", "observed_at_utc", "raw_path", "sha256", "error"]
    )
    complete_ids = set(status_frame.loc[status_frame["status"].eq("completed"), "page_id"])
    selected = plan.loc[~plan["page_id"].isin(complete_ids)].head(max_pages)
    status_rows = status_frame.to_dict("records")
    for item in selected.to_dict("records"):
        response_status = 0
        body = b""
        error = ""
        attempts = 0
        for attempt in range(1, retries + 2):
            attempts = attempt
            try:
                response_status, body, _ = fetch(item["url"])
                if response_status == 200:
                    break
                if response_status != 429 and not 500 <= response_status <= 599:
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 30))
        observed = datetime.now(UTC).isoformat()
        raw_path = raw_dir / f"{item['page_id']}.html"
        digest = ""
        terminal = "failed"
        if response_status == 200:
            _atomic_write_bytes(raw_path, body)
            digest = _sha256(raw_path)
            terminal = "completed"
        status_rows = [row for row in status_rows if row.get("page_id") != item["page_id"]]
        status_rows.append({
            "page_id": item["page_id"],
            "status": terminal,
            "http_status": response_status,
            "attempts": attempts,
            "observed_at_utc": observed,
            "raw_path": str(raw_path.resolve()) if raw_path.is_file() else "",
            "sha256": digest,
            "error": error,
        })
        _atomic_write_csv(status_path, pd.DataFrame(status_rows))
        if pacing_seconds:
            time.sleep(pacing_seconds)

    status_frame = pd.read_csv(status_path, dtype=str).fillna("")
    completed = status_frame.loc[status_frame["status"].eq("completed")]
    delisting_frames: list[pd.DataFrame] = []
    new_frames: list[pd.DataFrame] = []
    for item in plan.merge(completed, on="page_id", how="inner").to_dict("records"):
        raw_path = Path(item["raw_path"])
        if not raw_path.is_file() or _sha256(raw_path) != item["sha256"]:
            raise ValueError(f"raw archive hash mismatch: {item['page_id']}")
        if item["event_type"] == "delisting":
            delisting_frames.append(parse_delisting_html(raw_path.read_bytes(), item["url"], item["archive_year"]))
        else:
            new_frames.append(parse_new_listing_html(raw_path.read_bytes(), item["url"], item["archive_year"]))
    delistings = _concat(delisting_frames, _delisting_columns()).drop_duplicates(
        ["event_date", "code", "source_url"]
    )
    new_listings = _concat(new_frames, _new_listing_columns()).drop_duplicates(
        ["listing_date", "code", "source_url"]
    )
    _atomic_write_csv(delisting_path, delistings)
    _atomic_write_csv(new_listing_path, new_listings)

    planned = len(plan)
    completed_count = int(status_frame["status"].eq("completed").sum())
    years = {
        kind: sorted({int(value) for value in plan.loc[plan["event_type"].eq(kind), "archive_year"] if str(value).isdigit()})
        for kind in SOURCE_PAGES
    }
    coverage_complete = completed_count == planned and planned > 0
    historical_pit_ready = bool(
        coverage_complete
        and years.get("delisting")
        and years.get("new_listing")
        and min(years["delisting"]) <= 2008
        and min(years["new_listing"]) <= 2008
        and delistings["known_at_utc"].astype(str).str.strip().ne("").all()
        and new_listings["known_at_utc"].astype(str).str.strip().ne("").all()
    )
    manifest = {
        "run_id": run_id,
        "artifact_type": "jpx_public_listing_event_cross_validation",
        "status": "complete_limited_coverage" if coverage_complete else "incomplete",
        "official_jpx_source": True,
        "scorecard_eligible": False,
        "evidence_scope": "independent_public_archive_cross_check",
        "historical_pit_ready": historical_pit_ready,
        "known_at_coverage": 0.0,
        "coverage": {
            "planned_pages": planned,
            "completed_pages": completed_count,
            "delisting_years": years.get("delisting", []),
            "new_listing_years": years.get("new_listing", []),
            "delisting_events": len(delistings),
            "new_listing_events": len(new_listings),
            "required_start_year": 2008,
            "public_archive_closes_full_gap": historical_pit_ready,
        },
        "critical_limitations": [
            "discoverable public archives do not reach the required 2008 start year",
            "event dates are not exact source publication timestamps",
            "this source independently validates only the recent portion of J-Quants history",
        ],
        "posting_policy": {
            "delisting": "JPX states stock information is posted by the next business day after the delisting decision, but the table does not expose the decision/publication timestamp for every row",
            "new_listing": "approval date is retained separately and is not treated as an intraday known-at timestamp",
        },
        "artifacts": {
            "request_plan": str(plan_path.resolve()),
            "page_status": str(status_path.resolve()),
            "delisting_events": str(delisting_path.resolve()),
            "new_listing_events": str(new_listing_path.resolve()),
            "raw_directory": str(raw_dir.resolve()),
        },
        "sha256": {
            "request_plan.csv": _sha256(plan_path),
            "page_status.csv": _sha256(status_path),
            "delisting_events.csv": _sha256(delisting_path),
            "new_listing_events.csv": _sha256(new_listing_path),
        },
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(manifest_path, manifest)
    return JPXPublicListingArtifacts(run_id, root, manifest_path, plan_path, status_path, delisting_path, new_listing_path, manifest)


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(64 * 1024 * 1024 + 1), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _parse_english_date(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.strip())
    for pattern in ("%b. %d, %Y", "%b %d, %Y", "%b. %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(clean, pattern).date().isoformat()
        except ValueError:
            continue
    return ""


def _delisting_columns() -> list[str]:
    return ["event_date", "issue_name", "code", "market_segment", "reason", "archive_year", "source_url", "known_at_utc", "known_at_status"]


def _new_listing_columns() -> list[str]:
    return ["listing_date", "approval_date", "issue_name", "code", "market_segment", "trading_unit", "archive_year", "source_url", "known_at_utc", "known_at_status"]


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).reindex(columns=columns) if frames else pd.DataFrame(columns=columns)


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
