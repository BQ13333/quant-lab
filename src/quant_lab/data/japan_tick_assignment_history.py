from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"
FIXED_START = date(2013, 9, 1)
FIXED_END = date(2024, 8, 1)
ROLLING_SOURCE_COUNT = 22
CONCORDIA_URL = "https://www.jpx.co.jp/english/news/1044/20160303-01.html"
CUTOFF_MASTER_URL = "https://www.jpx.co.jp/automation/english/markets/indices/topix/files/topixweight_e.csv"
CORPORATE_EXCEPTION_SOURCES = {
    "kyoto_financial_group": "https://www.jpx.co.jp/english/news/6030/uorii50000000zub-att/data_e.pdf",
    "delisting_ledger_2025": "https://www.jpx.co.jp/english/listing/stocks/delisted/archives-01.html",
    "sony_financial_spin": "https://www.jpx.co.jp/english/news/6030/sjcobq000002555l-att/deta_e.pdf",
    "blue_zones_transfer": "https://www.jpx.co.jp/english/news/6030/sjcobq00000246yi-att/data_e.pdf",
    "topix_classification_2025": "https://www.jpx.co.jp/english/news/6030/sjcobq0000025n0g-att/e_mei2_12_size.pdf",
    "sony_financial_old_delisting": "https://www.jpx.co.jp/english/news/1023/20200720-11.html",
    "sony_financial_new_listing": "https://www.jpx.co.jp/english/listing/stocks/new/sjcobq000002509n-att/09SonyFinancialGroup-OutlineEN.pdf",
}
CORPORATE_EXCEPTION_EVENTS = [
    ("2023-10-02", "REMOVE", "8369", "The Bank of Kyoto,Ltd.", "kyoto_financial_group"),
    ("2023-10-02", "ADD", "5844", "Kyoto Financial Group,Inc.", "kyoto_financial_group"),
    ("2025-06-27", "REMOVE", "8905", "AEON Mall Co.,Ltd.", "delisting_ledger_2025"),
    ("2025-09-29", "ADD", "8729", "Sony Financial Group Inc.", "sony_financial_spin"),
    ("2025-10-01", "REMOVE", "8279", "YAOKO CO.,LTD.", "blue_zones_transfer"),
    ("2025-10-01", "ADD", "417A", "BLUE ZONES HOLDINGS CO.,LTD.", "blue_zones_transfer"),
    ("2025-11-27", "REMOVE", "3141", "WELCIA HOLDINGS CO.,LTD.", "delisting_ledger_2025"),
]


@dataclass(frozen=True)
class JapanTickAssignmentArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


def archive_urls() -> list[str]:
    months = []
    cursor = FIXED_START
    while cursor <= FIXED_END:
        months.append(
            f"{JPX_ROOT}/english/news/news_ym_{cursor:%Y%m}.json"
        )
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    months.extend(
        f"{JPX_ROOT}/english/news/news_ym_{index:02d}.json"
        for index in range(1, ROLLING_SOURCE_COUNT + 1)
    )
    return months


def _master_snapshot_date(master: pd.DataFrame) -> date | None:
    if master.empty or "Date" not in master.columns:
        return None
    values = master["Date"].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    return parsed.max().date() if parsed.notna().any() else None


def write_japan_tick_assignment_history(
    *,
    run_id: str,
    checkpoint_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_tick_assignment_history",
    cutoff_date: str = "2026-06-30",
    fetcher: Fetcher | None = None,
) -> JapanTickAssignmentArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    cutoff = date.fromisoformat(cutoff_date)
    checkpoint = Path(checkpoint_path).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        fetch = fetcher or _fetch
        archive_dir = root / "raw/archive_json"
        notice_dir = root / "raw/notice_pages"
        corporate_dir = root / "raw/corporate_exceptions"
        archive_dir.mkdir(parents=True, exist_ok=True)
        notice_dir.mkdir(parents=True, exist_ok=True)
        corporate_dir.mkdir(parents=True, exist_ok=True)

        archive_rows: list[dict[str, Any]] = []
        discovered: dict[str, dict[str, Any]] = {}
        for url in archive_urls():
            source_id = Path(urlparse(url).path).stem
            raw_path = archive_dir / f"{source_id}.json"
            status, body, headers = fetch(url)
            if status == 200 and body:
                _atomic_bytes(raw_path, body)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    for item in _walk_news_items(payload):
                        title = str(item.get("title", "")).strip()
                        if not re.match(r"^Handling of Tick Sizes from\b", title, re.I):
                            continue
                        publication_date = _publication_date(item)
                        if not publication_date or publication_date > cutoff:
                            continue
                        page_url = urljoin(JPX_ROOT, str(item.get("url", "")))
                        if not page_url.startswith(f"{JPX_ROOT}/english/news/"):
                            continue
                        record = discovered.setdefault(
                            page_url,
                            {
                                "publication_date": publication_date.isoformat(),
                                "title": title,
                                "page_url": page_url,
                                "discovery_sources": [],
                            },
                        )
                        record["discovery_sources"].append(source_id)
            archive_rows.append(
                _source_row(source_id, url, status, headers, raw_path)
            )

        notice_rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        parse_failures: list[str] = []
        for page_url, item in sorted(discovered.items(), key=lambda value: (value[1]["publication_date"], value[0])):
            page_id = Path(urlparse(page_url).path).stem
            raw_path = notice_dir / f"{item['publication_date']}_{page_id}.html"
            status, body, headers = fetch(page_url)
            if status == 200 and body:
                _atomic_bytes(raw_path, body)
            parsed = _parse_notice_page(body, item) if status == 200 and body else []
            if not parsed:
                parse_failures.append(page_url)
            events.extend(parsed)
            notice_rows.append(
                {
                    **_source_row(page_id, page_url, status, headers, raw_path),
                    "publication_date": item["publication_date"],
                    "effective_date": _effective_date(item["title"], item["publication_date"]),
                    "title": item["title"],
                    "discovery_sources": "|".join(sorted(set(item["discovery_sources"]))),
                    "event_rows": len(parsed),
                }
            )

        concordia_path = notice_dir / "2016-03-03_20160303-01.html"
        status, body, headers = fetch(CONCORDIA_URL)
        if status == 200 and body:
            _atomic_bytes(concordia_path, body)
            text = BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
            if all(anchor in text for anchor in ("8332", "7186", "April 1, 2016")):
                events.append(
                    {
                        "publication_date": "2016-03-03",
                        "effective_date": "2016-04-01",
                        "assignment_scope": "TOPIX100",
                        "action": "REMOVE",
                        "code": "8332",
                        "issue_name": "The Bank of Yokohama,Ltd.",
                        "source_url": CONCORDIA_URL,
                        "source_type": "corporate_action_supplement",
                    }
                )
            else:
                parse_failures.append(CONCORDIA_URL)
        notice_rows.append(
            {
                **_source_row("concordia_20160303", CONCORDIA_URL, status, headers, concordia_path),
                "publication_date": "2016-03-03",
                "effective_date": "2016-04-01",
                "title": "TOPIX treatment for Concordia Financial Group",
                "discovery_sources": "corporate_action_supplement",
                "event_rows": int(any(row["source_url"] == CONCORDIA_URL for row in events)),
            }
        )

        corporate_rows: list[dict[str, Any]] = []
        for source_id, url in CORPORATE_EXCEPTION_SOURCES.items():
            suffix = Path(urlparse(url).path).suffix or ".html"
            raw_path = corporate_dir / f"{source_id}{suffix}"
            status, body, headers = fetch(url)
            if status == 200 and body:
                _atomic_bytes(raw_path, body)
            corporate_rows.append(_source_row(source_id, url, status, headers, raw_path))
        source_urls = CORPORATE_EXCEPTION_SOURCES
        for effective_date, action, code, issue_name, source_id in CORPORATE_EXCEPTION_EVENTS:
            events.append(
                {
                    "publication_date": effective_date,
                    "effective_date": effective_date,
                    "assignment_scope": "TOPIX500",
                    "action": action,
                    "code": code,
                    "issue_name": issue_name,
                    "source_url": source_urls[source_id],
                    "source_type": "corporate_action_exception",
                }
            )

        master_path = corporate_dir / "topixweight_e.csv"
        status, body, headers = fetch(CUTOFF_MASTER_URL)
        if status == 200 and body:
            _atomic_bytes(master_path, body)
        master = pd.read_csv(master_path, encoding="cp932", dtype={"Code": str}) if master_path.is_file() else pd.DataFrame()
        master_snapshot_date = _master_snapshot_date(master)
        corporate_rows.append(
            _source_row(
                f"topix_component_master_{master_snapshot_date.strftime('%Y%m%d') if master_snapshot_date else 'unknown'}",
                CUTOFF_MASTER_URL,
                status,
                headers,
                master_path,
            )
        )
        eligible_series = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}
        master_codes = set(
            master.loc[master.get("New Index Series Code", pd.Series(dtype=str)).isin(eligible_series), "Code"].dropna().astype(str)
        ) if not master.empty else set()

        archives = pd.DataFrame(archive_rows)
        notices = pd.DataFrame(notice_rows)
        corporate = pd.DataFrame(corporate_rows)
        event_columns = [
            "publication_date", "effective_date", "assignment_scope", "action", "code",
            "issue_name", "source_url", "source_type",
        ]
        event_frame = pd.DataFrame(events, columns=event_columns).drop_duplicates()
        event_frame = event_frame.sort_values(["effective_date", "code", "action", "source_url"]).reset_index(drop=True)
        checkpoint_frame = pd.read_csv(checkpoint, dtype={"code": str})
        reconciliation, final_state = _replay_and_reconcile(event_frame, checkpoint_frame, cutoff)
        master_reconciliation = pd.DataFrame(
            [
                {
                    "master_date": master_snapshot_date.isoformat() if master_snapshot_date else "",
                    "master_names": len(master_codes),
                    "replayed_names": len(final_state),
                    "missing_names": len(master_codes - final_state),
                    "unexpected_names": len(final_state - master_codes),
                    "exact_set_match": master_codes == final_state,
                    "missing_codes": "|".join(sorted(master_codes - final_state)),
                    "unexpected_codes": "|".join(sorted(final_state - master_codes)),
                }
            ]
        )

        outputs = {
            "archive_source_manifest.csv": archives,
            "notice_source_manifest.csv": notices,
            "corporate_source_manifest.csv": corporate,
            "assignment_events.csv": event_frame,
            "checkpoint_reconciliation.csv": reconciliation,
            "cutoff_master_reconciliation.csv": master_reconciliation,
            "cutoff_assignment_state.csv": pd.DataFrame({"code": sorted(final_state)}),
        }
        for filename, frame in outputs.items():
            _atomic_csv(root / filename, frame)

        source_failures = int(archives["http_status"].ne(200).sum() + notices["http_status"].ne(200).sum() + corporate["http_status"].ne(200).sum())
        checkpoint_failures = int(reconciliation["exact_set_match"].ne(True).sum()) if not reconciliation.empty else 1
        duplicate_events = int(event_frame.duplicated(["effective_date", "code", "action"]).sum())
        producer_blockers = []
        if source_failures:
            producer_blockers.append(f"official_source_failures={source_failures}")
        if parse_failures:
            producer_blockers.append(f"notice_parse_failures={len(parse_failures)}")
        if checkpoint_failures:
            producer_blockers.append(f"checkpoint_set_mismatches={checkpoint_failures}")
        if duplicate_events:
            producer_blockers.append(f"duplicate_effective_code_actions={duplicate_events}")
        if not bool(master_reconciliation.iloc[0]["exact_set_match"]):
            producer_blockers.append("cutoff_master_set_mismatch")
        remaining = []
        if master_snapshot_date is None or master_snapshot_date < cutoff:
            remaining.append(
                "official June-2026 end-of-month Master of Index is not yet published and retained as the final cutoff checkpoint"
            )
        manifest = {
            "schema_version": "1.0-source-backed-partial",
            "run_id": run_id,
            "artifact_type": "japan_tick_assignment_notice_history_candidate",
            "status": "partial_candidate" if not producer_blockers else "failed_candidate",
            "component": "pit_narrow_tick_assignment",
            "cutoff_date": cutoff.isoformat(),
            "official_jpx_source": True,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "independent_validation_required": True,
            "counts": {
                "archive_sources": len(archives),
                "unique_notice_pages": len(discovered),
                "notice_sources_with_supplement": len(notices),
                "corporate_exception_sources": len(corporate),
                "source_failures": source_failures,
                "notice_parse_failures": len(parse_failures),
                "assignment_events": len(event_frame),
                "corporate_exception_events": len(CORPORATE_EXCEPTION_EVENTS) + 1,
                "checkpoint_rows": len(reconciliation),
                "checkpoint_exact_matches": int(reconciliation["exact_set_match"].eq(True).sum()),
                "cutoff_assignment_names": len(final_state),
                "cutoff_master_names": len(master_codes),
                "cutoff_master_date": master_snapshot_date.isoformat() if master_snapshot_date else "",
                "cutoff_master_exact_match": bool(master_reconciliation.iloc[0]["exact_set_match"]),
            },
            "producer_blockers": producer_blockers,
            "remaining_completion_blockers": remaining,
            "critical_blockers": [*producer_blockers, *remaining],
            "parse_failure_urls": parse_failures,
            "checkpoint_source": str(checkpoint),
            "checkpoint_source_sha256": _sha256(checkpoint),
            "artifacts": {Path(name).stem: str((root / name).resolve()) for name in outputs},
            "output_sha256": {name: _sha256(root / name) for name in outputs},
            "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanTickAssignmentArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


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


def _effective_date(title: str, publication_date: str) -> str:
    match = re.search(r"from\s+([A-Za-z]+)\s*(\d{1,2})\s+onward", title, re.I)
    if not match:
        return ""
    published = date.fromisoformat(publication_date)
    month = datetime.strptime(match.group(1)[:3], "%b").month
    year = published.year + int(month < published.month - 6)
    return date(year, month, int(match.group(2))).isoformat()


def _parse_notice_page(body: bytes, item: dict[str, Any]) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    effective = _effective_date(item["title"], item["publication_date"])
    if not effective:
        return []
    page_text = soup.get_text(" ", strip=True)
    scope = "TOPIX500" if "TOPIX500" in page_text or "TOPIX 500" in page_text else "TOPIX100"
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        context_node = table.find_previous(["p", "h3", "h4"])
        context = " ".join(context_node.stripped_strings).lower() if context_node else ""
        if "no longer" in context:
            action = "REMOVE"
        elif "newly" in context or "will be subject" in context:
            action = "ADD"
        else:
            continue
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.stripped_strings) for cell in tr.find_all(["th", "td"])]
            code = cells[0].strip().upper()
            if len(cells) < 2 or not re.fullmatch(r"(?=.*\d)[0-9A-Z]{4}", code):
                continue
            rows.append(
                {
                    "publication_date": item["publication_date"],
                    "effective_date": effective,
                    "assignment_scope": scope,
                    "action": action,
                    "code": code,
                    "issue_name": cells[1].strip(),
                    "source_url": item["page_url"],
                    "source_type": "tick_assignment_notice",
                }
            )
    return rows


def _replay_and_reconcile(events: pd.DataFrame, checkpoints: pd.DataFrame, cutoff: date) -> tuple[pd.DataFrame, set[str]]:
    checkpoints = checkpoints.copy()
    checkpoints["effective_date"] = checkpoints["effective_date"].astype(str)
    topix = checkpoints.loc[
        checkpoints["effective_date"].eq("2014-01-14") & checkpoints["index_scope"].eq("TOPIX100"), "code"
    ].astype(str)
    state = set(topix)
    rows = []
    event_rows = events.to_dict("records")
    checkpoint_dates = sorted(
        value for value in checkpoints["effective_date"].unique() if value <= cutoff.isoformat()
    )
    for checkpoint_date in checkpoint_dates:
        if checkpoint_date == "2023-05-26":
            continue
        checkpoint_basis = "official_full_topix100_snapshot"
        if checkpoint_date == "2023-06-05":
            mid = set(
                checkpoints.loc[
                    checkpoints["effective_date"].eq(checkpoint_date)
                    & checkpoints["index_scope"].eq("TOPIX_MID400"), "code"
                ].astype(str)
            )
            composite_expected = set(state) | mid
            state = set(composite_expected)
            checkpoint_basis = "replayed_topix100_plus_official_mid400_seed"
        for event in event_rows:
            if event.get("_applied") or event["effective_date"] > checkpoint_date:
                continue
            if event["effective_date"] < "2023-06-05" and event["assignment_scope"] != "TOPIX100":
                continue
            if event["action"] == "ADD":
                state.add(event["code"])
            else:
                state.discard(event["code"])
            event["_applied"] = True
        scope = "TOPIX500" if checkpoint_date >= "2023-06-05" else "TOPIX100"
        if scope == "TOPIX500":
            expected = composite_expected
        else:
            expected = set(
                checkpoints.loc[
                    checkpoints["effective_date"].eq(checkpoint_date)
                    & checkpoints["index_scope"].eq("TOPIX100"), "code"
                ].astype(str)
            )
        if not expected:
            continue
        rows.append(
            {
                "effective_date": checkpoint_date,
                "assignment_scope": scope,
                "checkpoint_basis": checkpoint_basis,
                "expected_names": len(expected),
                "replayed_names": len(state),
                "missing_names": len(expected - state),
                "unexpected_names": len(state - expected),
                "exact_set_match": state == expected,
                "missing_codes": "|".join(sorted(expected - state)),
                "unexpected_codes": "|".join(sorted(state - expected)),
            }
        )
    for event in event_rows:
        if event.get("_applied") or event["effective_date"] > cutoff.isoformat():
            continue
        if event["action"] == "ADD":
            state.add(event["code"])
        else:
            state.discard(event["code"])
    return pd.DataFrame(rows), state


def _source_row(source_id: str, url: str, status: int, headers: dict[str, str], path: Path) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "url": url,
        "http_status": status,
        "content_type": headers.get("Content-Type", headers.get("content-type", "")),
        "raw_path": str(path.resolve()) if path.is_file() else "",
        "sha256": _sha256(path) if path.is_file() else "",
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-jpx-tick-assignment/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(32 * 1024 * 1024 + 1), dict(response.headers.items())
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
