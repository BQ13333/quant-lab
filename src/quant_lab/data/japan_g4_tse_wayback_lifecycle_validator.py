from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pdfplumber


VERSION = "2.1-independent-jpx-copied-file-pdfplumber"
TSE_HOST = "www.tse.or.jp"
JPX_HOST = "www.jpx.co.jp"
WAYBACK_ROOT = "https://web.archive.org/web"
CDX_FIELDS = ("timestamp", "original", "statuscode", "digest", "length", "mimetype")
HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
CHANNEL_HEADINGS = {
    DELISTING_CHANNEL: "Delisting",
    HALT_CHANNEL: "Suspension of Trading",
}
DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*[./]\s*(?P<month>\d{1,2})\s*[./]\s*(?P<day>\d{1,2})\s+(?P<tail>.+)$"
)
CODE_PATTERN = re.compile(r"^(?P<code>[0-9][0-9A-Z]{3,9})(?:\s+|$)")
CLOCK_EXPRESSION = r"(?:[01]?\d|2[0-3])\s*[:：;.]\s*[0-5]\d"
TIME_PATTERN = re.compile(CLOCK_EXPRESSION)
TIME_INTERVAL_PATTERN = re.compile(
    rf"(?P<start>{CLOCK_EXPRESSION})\s*[～~〜]\s*"
    rf"(?:(?P<end_date>20\d{{2}}\s*[./]\s*\d{{1,2}}\s*[./]\s*\d{{1,2}})\s*)?"
    rf"(?P<end>{CLOCK_EXPRESSION})"
)
NEXT_HEADING = re.compile(
    r"^(?:New Listings|Delisting|Suspension of Trading|Assignment to|Reassignment|"
    r"Transfer to|Name Change|Change of|Change in|Additional Assignment|"
    r"Exception from|Designation as|Lifted Assignment|Ex-New\b)",
    flags=re.IGNORECASE,
)
NUMBERED_REPORT = re.compile(r"^\d{1,2}(?:-\d+)?\s+.*(?:\(|（)")

SOURCE_RECORD_VERSION = "jpx-g4-report-row-v1"
EVENT_ID_VERSION = "jpx-g4-report-event-v1"
EVENT_TYPES = {
    DELISTING_CHANNEL: "delisting_report_row",
    HALT_CHANNEL: "trading_halt_report_row",
}
EVENT_COLUMNS = [
    "event_id",
    "report_month",
    "channel",
    "event_type",
    "event_date",
    "raw_code",
    "normalized_local_code",
    "instrument_class",
    "issue_name",
    "halt_start_printed",
    "halt_end_printed",
    "halt_start_jst",
    "halt_end_jst",
    "source_url",
    "source_path",
    "source_sha256",
    "source_page",
    "source_row_ordinal",
    "source_record_text",
    "source_record_sha256",
    "known_at",
    "ric",
    "mapping_status",
    "timestamp_status",
    "admissible_to_shared_halt_history",
]
MAPPING_PENDING = "pending_independent_source_backed_mapping"
TIMESTAMP_PENDING = "pending_independent_source_backed_known_at"
COPIED_FILE_ROUTE = "official_jpx_copied_file_with_parent_index_wayback_lineage"
LEGACY_WAYBACK_ROUTE = "legacy_tse_page_and_pdf_wayback"
COPIED_FILE_ROUTE_ENABLED = True
COPIED_PRESERVATION_TRANSPORT = (
    "Official JPX live PDF plus Internet Archive parent-index lineage"
)
DIRECT_ARCHIVED_PDF_LINEAGE = "archived_parent_index_directly_links_official_pdf"
WEAKER_ARCHIVED_MONTH_LINEAGE = "archived_parent_index_links_month_page_only"

_TSE_PARENT_INDEX = "http://www.tse.or.jp:80/english/market/data/geppo/index.html"
_TSE_PARENT_CAPTURE = "20150310160942"
_JPX_2015_PARENT_INDEX = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
    "00-archives-10.html"
)
_JPX_2015_PARENT_CAPTURE = "20250503084309"


def _validator_route(
    official_pdf_url: str,
    parent_index_url: str,
    parent_capture_timestamp: str,
    parent_link_url: str,
) -> dict[str, str]:
    return {
        "official_pdf_url": official_pdf_url,
        "parent_index_url": parent_index_url,
        "parent_capture_timestamp": parent_capture_timestamp,
        "parent_link_url": parent_link_url,
        "archived_parent_lineage_strength": (
            DIRECT_ARCHIVED_PDF_LINEAGE
            if parent_link_url == official_pdf_url
            else WEAKER_ARCHIVED_MONTH_LINEAGE
        ),
    }


INDEPENDENT_COPIED_FILE_REGISTRY: dict[str, dict[str, str]] = {
    "2014-09": _validator_route(
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje60000026ao9-att/17_idou1409.pdf",
        _TSE_PARENT_INDEX,
        _TSE_PARENT_CAPTURE,
        "http://www.tse.or.jp/english/market/data/geppo/201409.html",
    ),
    "2014-10": _validator_route(
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje60000027r55-att/17_idou1410.pdf",
        _TSE_PARENT_INDEX,
        _TSE_PARENT_CAPTURE,
        "http://www.tse.or.jp/english/market/data/geppo/201410.html",
    ),
    "2014-11": _validator_route(
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje600000294jl-att/17_idou1411.pdf",
        _TSE_PARENT_INDEX,
        _TSE_PARENT_CAPTURE,
        "http://www.tse.or.jp/english/market/data/geppo/201411.html",
    ),
    "2014-12": _validator_route(
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje6000002aeyz-att/17_idou1412.pdf",
        _TSE_PARENT_INDEX,
        _TSE_PARENT_CAPTURE,
        "http://www.tse.or.jp/english/market/data/geppo/201412.html",
    ),
    "2015-01": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj0000004mgs-att/17_idou1501.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj0000004mgs-att/17_idou1501.pdf",
    ),
    "2015-02": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000fnkm-att/17_idou1502.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000fnkm-att/17_idou1502.pdf",
    ),
    "2015-03": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000idmr-att/17_idou1503.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000idmr-att/17_idou1503.pdf",
    ),
    "2015-04": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000kmok-att/17_idou1504.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000kmok-att/17_idou1504.pdf",
    ),
    "2015-05": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000m7co-att/17_idou1505.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000m7co-att/17_idou1505.pdf",
    ),
    "2015-06": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000nd3i-att/17_idou1506.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000nd3i-att/17_idou1506.pdf",
    ),
    "2015-07": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000oz0h-att/17_idou1507.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000oz0h-att/17_idou1507.pdf",
    ),
    "2015-08": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000qc9y-att/17_idou1508.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000qc9y-att/17_idou1508.pdf",
    ),
    "2015-09": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000rgwu-att/17_idou1509.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000rgwu-att/17_idou1509.pdf",
    ),
    "2015-10": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000t1f1-att/17_idou1510.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000t1f1-att/17_idou1510.pdf",
    ),
    "2015-11": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000u9pb-att/17_idou1511.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000u9pb-att/17_idou1511.pdf",
    ),
    "2015-12": _validator_route(
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000vj5c-att/17_idou1512.pdf",
        _JPX_2015_PARENT_INDEX,
        _JPX_2015_PARENT_CAPTURE,
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000vj5c-att/17_idou1512.pdf",
    ),
}

PageExtractor = Callable[[bytes], list[str]]


@dataclass(frozen=True)
class WaybackLifecycleValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        for key, value in attrs:
            if key.casefold() == "href" and value is not None:
                self.hrefs.append(value.strip())


def independently_parse_pdf_pages(
    pages: list[str],
    month: str,
    *,
    source_url: str = "",
    source_path: str = "",
    source_sha256: str = "",
) -> dict[str, Any]:
    """Reparse exact event sets without importing producer parser code."""
    period = pd.Period(month, freq="M")
    located = [
        (page_number, _canonical(line))
        for page_number, page in enumerate(pages, start=1)
        for line in page.splitlines()
        if _canonical(line)
    ]
    title_months: set[str] = set()
    title_pattern = re.compile(
        r"Changes\s+in\s+Listed\s+Companies\s+and\s+Issues.*?"
        r"\((?P<name>[A-Za-z]{3,9})\.?\s*(?P<year>20\d{2})\)",
        flags=re.IGNORECASE,
    )
    for _, line in located:
        match = title_pattern.search(line)
        if not match:
            continue
        try:
            title_months.add(
                pd.to_datetime(f"{match.group('name')} {match.group('year')}").strftime("%Y-%m")
            )
        except ValueError:
            continue

    sections: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for channel, heading in CHANNEL_HEADINGS.items():
        starts = [index for index, (_, line) in enumerate(located) if line.casefold() == heading.casefold()]
        channel_events: list[dict[str, Any]] = []
        if len(starts) == 1:
            current_class = ""
            for page_number, line in located[starts[0] + 1 :]:
                if _section_boundary(line):
                    break
                match = DATE_PATTERN.search(line)
                if not match:
                    continue
                event_day = date(
                    int(match.group("year")), int(match.group("month")), int(match.group("day"))
                )
                if event_day.year != period.year or event_day.month != period.month:
                    raise ValueError(
                        f"out-of-month row in {month} {channel}: {event_day.isoformat()}"
                    )
                prefix = line[: match.start()].strip()
                current_class = _independent_instrument_class(prefix, current_class)
                if not current_class:
                    raise ValueError(f"instrument class unavailable in {month} {channel}: {line}")
                tail = match.group("tail").strip()
                code_match = CODE_PATTERN.match(tail)
                if code_match:
                    raw_code = code_match.group("code")
                    body = tail[code_match.end() :].strip()
                elif tail.startswith("全銘柄"):
                    raw_code = "全銘柄"
                    body = tail[len("全銘柄") :].strip()
                else:
                    raw_code = ""
                    body = tail
                timing = (
                    _independent_halt_timing(body, event_day)
                    if channel == HALT_CHANNEL
                    else _blank_timing()
                )
                issue_name = _independent_issue_name(body, channel, timing["range_start"])
                if not issue_name and raw_code:
                    issue_name = raw_code
                if not issue_name:
                    raise ValueError(f"issue name unavailable in {month} {channel}: {line}")
                ordinal = len(channel_events) + 1
                normalized_code = (
                    unicodedata.normalize("NFKC", raw_code).strip().upper()
                    if CODE_PATTERN.fullmatch(raw_code)
                    else ""
                )
                identity = {
                    "version": SOURCE_RECORD_VERSION,
                    "report_month": month,
                    "channel": channel,
                    "source_sha256": source_sha256,
                    "source_page": page_number,
                    "source_row_ordinal": ordinal,
                    "event_date": event_day.isoformat(),
                    "raw_code": raw_code,
                    "normalized_local_code": normalized_code,
                    "instrument_class": current_class,
                    "issue_name": issue_name,
                    "halt_start_printed": timing["start_printed"],
                    "halt_end_printed": timing["end_printed"],
                }
                record_sha = _record_sha256(identity)
                channel_events.append(
                    {
                        "event_id": hashlib.sha256(
                            f"{EVENT_ID_VERSION}|{record_sha}".encode("utf-8")
                        ).hexdigest()[:24],
                        "report_month": month,
                        "channel": channel,
                        "event_type": EVENT_TYPES[channel],
                        "event_date": event_day.isoformat(),
                        "raw_code": raw_code,
                        "normalized_local_code": normalized_code,
                        "instrument_class": current_class,
                        "issue_name": issue_name,
                        "halt_start_printed": timing["start_printed"],
                        "halt_end_printed": timing["end_printed"],
                        "halt_start_jst": timing["start_jst"],
                        "halt_end_jst": timing["end_jst"],
                        "source_url": source_url,
                        "source_path": source_path,
                        "source_sha256": source_sha256,
                        "source_page": page_number,
                        "source_row_ordinal": ordinal,
                        "source_record_text": line,
                        "source_record_sha256": record_sha,
                        "known_at": "",
                        "ric": "",
                        "mapping_status": MAPPING_PENDING,
                        "timestamp_status": TIMESTAMP_PENDING,
                        "admissible_to_shared_halt_history": False,
                    }
                )
        sections[channel] = {
            "heading_count": len(starts),
            "explicit": len(starts) == 1,
            "recognized": len(starts) == 1,
            "event_count": len(channel_events),
            "zero_inferred": False,
        }
        events.extend(channel_events)
    return {
        "title_months": sorted(title_months),
        "exact_month": title_months == {month},
        "sections": sections,
        "required_sections_valid": all(
            section["heading_count"] == 1 and section["recognized"]
            for section in sections.values()
        ),
        "events": events,
    }


def validate_japan_g4_tse_wayback_lifecycle(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_tse_wayback_lifecycle_validation",
    page_extractor: PageExtractor | None = None,
) -> WaybackLifecycleValidationArtifacts:
    _safe_name(run_id)
    producer_input = Path(producer_dir)
    producer = producer_input.resolve()
    root = Path(artifact_root) / run_id
    if root.exists():
        raise FileExistsError(f"write-once validation artifact already exists: {root}")
    root.mkdir(parents=True)
    issues: list[dict[str, str]] = []
    hash_rows = _validate_packet_hashes(producer, issues)
    manifest = _read_json(producer / "manifest.json", issues, "producer_manifest")
    selected = [str(value) for value in manifest.get("selected_months", [])]
    inventory = _read_csv(producer / "source_inventory.csv", issues, "source_inventory")
    captures = _read_csv(producer / "cdx_capture_inventory.csv", issues, "cdx_capture_inventory")
    transport = _read_csv(producer / "transport_inventory.csv", issues, "transport_inventory")
    claims = _read_csv(producer / "exact_nonempty_claims.csv", issues, "exact_nonempty_claims")
    events = _read_csv(producer / "exact_events.csv", issues, "exact_events")
    parquet_events = _read_parquet(
        producer / "exact_events.parquet", issues, "exact_events_parquet"
    )
    completion = _read_csv(producer / "completion_plan.csv", issues, "completion_plan")
    failures = _read_csv(
        producer / "failures_quarantine.csv", issues, "failures_quarantine"
    )
    acquisition_plan = _read_csv(
        producer / "acquisition_plan.csv", issues, "acquisition_plan"
    )
    checkpoint = _read_json(
        producer / "acquisition_checkpoint.json", issues, "acquisition_checkpoint"
    )
    extract_pages = page_extractor or _extract_pages_pdfplumber

    if _path_has_symlink_component(producer_input):
        _issue(issues, "producer_root_symlink_rejected", "producer", str(producer_input))
    _validate_global_contracts(
        manifest,
        selected,
        inventory,
        captures,
        transport,
        claims,
        events,
        parquet_events,
        completion,
        failures,
        acquisition_plan,
        checkpoint,
        issues,
    )
    month_rows: list[dict[str, Any]] = []
    independent_events: list[dict[str, Any]] = []
    for month in selected:
        month_inventory = inventory.loc[inventory.get("report_month", pd.Series(dtype=str)).eq(month)]
        if len(month_inventory) != 1:
            _issue(issues, "source_inventory_month_cardinality", month, str(len(month_inventory)))
            month_rows.append(_failed_month_row(month, "source_inventory_missing_or_duplicate"))
            continue
        item = month_inventory.iloc[0].to_dict()
        row, parsed = _validate_month(
            producer,
            month,
            item,
            captures,
            transport,
            extract_pages,
            issues,
        )
        month_rows.append(row)
        if parsed:
            independent_events.extend(parsed["events"])

    reconciliation = _reconcile_event_sets(
        producer,
        selected,
        inventory,
        claims,
        events,
        independent_events,
        month_rows,
        issues,
    )
    _write_validation_outputs(
        root,
        hash_rows,
        month_rows,
        reconciliation,
        independent_events,
        issues,
    )

    acquired = sum(bool(row.get("capture_valid")) for row in month_rows)
    fully_reconciled = sum(bool(row.get("validated")) for row in reconciliation)
    status = "failed_validation" if issues else "pass_bounded_exact_nonempty_evidence"
    hashes = _artifact_hashes(root, exclude={"manifest.json", "artifact_hashes.json"})
    hashes_path = root / "artifact_hashes.json"
    _atomic_text(hashes_path, json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    validation_manifest = {
        "artifact_type": "independent_tse_wayback_lifecycle_validation",
        "version": VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "producer_dir": str(producer),
        "producer_manifest_sha256": _sha_path(producer / "manifest.json") if (producer / "manifest.json").is_file() else "",
        "selected_month_count": len(selected),
        "capture_valid_month_count": acquired,
        "independent_exact_event_count": len(independent_events),
        "reconciled_nonempty_claim_count": fully_reconciled,
        "producer_nonempty_claim_count": len(claims),
        "validation_issue_count": len(issues),
        "validation_issue_codes": sorted({item["issue_code"] for item in issues}),
        "parser_path": "pdfplumber plus independent section/event parser",
        "html_parser_path": "stdlib html.parser",
        "route_modes": sorted(
            set(inventory.get("route_mode", pd.Series(dtype=str)).astype(str))
        ),
        "archived_parent_lineage_strengths": sorted(
            {
                str(value)
                for value in inventory.get(
                    "archived_parent_lineage_strength", pd.Series(dtype=str)
                )
                if str(value)
            }
        ),
        "live_official_pdf_response_independently_validated": bool(
            not inventory.empty
            and inventory.get("route_mode", pd.Series(dtype=str))
            .eq(COPIED_FILE_ROUTE)
            .all()
            and acquired == len(selected)
        ),
        "absence_is_zero": False,
        "missing_capture_fails_closed": True,
        "parser_disagreement_fails_closed": True,
        "refinitiv_used": False,
        "full_required_month_channel_completion": False,
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "write_once": True,
        "artifact_hashes_sha256": _sha_path(hashes_path),
    }
    manifest_path = root / "manifest.json"
    _atomic_text(manifest_path, json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n")
    return WaybackLifecycleValidationArtifacts(run_id, root, manifest_path, validation_manifest)


def verify_wayback_lifecycle_validation(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir).resolve()
    issues: list[str] = []
    try:
        manifest_path = _contained_nonsymlink_path(root, "manifest.json")
        hashes_path = _contained_nonsymlink_path(root, "artifact_hashes.json")
        if manifest_path is None or hashes_path is None:
            return ["validation_control_path_symlink_or_escape"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("validation_hash_inventory_mismatch")
        declared = json.loads(hashes_path.read_text(encoding="utf-8"))
        for relative, expected in declared.items():
            if _unsafe(relative):
                issues.append(f"unsafe_validation_path:{relative}")
                continue
            path = _contained_nonsymlink_path(root, relative)
            if path is None or not path.is_file() or _sha_path(path) != expected:
                issues.append(f"validation_hash_mismatch:{relative}")
    except Exception as exc:
        issues.append(f"validation_parse_error:{type(exc).__name__}")
    return issues


def _validate_month(
    producer: Path,
    month: str,
    item: dict[str, Any],
    captures: pd.DataFrame,
    transport: pd.DataFrame,
    extract_pages: PageExtractor,
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if str(item.get("route_mode", "")) == COPIED_FILE_ROUTE:
        return _validate_copied_month(
            producer,
            month,
            item,
            captures,
            transport,
            extract_pages,
            issues,
        )
    page_path = _safe_source_path(producer, item.get("html_path", ""), f"raw/pages/{month}.html", issues, month)
    pdf_path = _safe_source_path(producer, item.get("pdf_path", ""), f"raw/pdfs/{month}.pdf", issues, month)
    if page_path is None or pdf_path is None or not page_path.is_file() or not pdf_path.is_file():
        _issue(issues, "missing_capture_bytes", month, "HTML or PDF raw bytes unavailable")
        return _failed_month_row(month, "missing_capture_bytes"), None

    page_bytes = page_path.read_bytes()
    pdf_bytes = pdf_path.read_bytes()
    page_hash_match = _sha_bytes(page_bytes) == str(item.get("html_sha256", ""))
    pdf_hash_match = _sha_bytes(pdf_bytes) == str(item.get("pdf_sha256", ""))
    byte_lengths_match = (
        len(page_bytes) == _int(item.get("html_byte_length"))
        and len(pdf_bytes) == _int(item.get("pdf_byte_length"))
    )
    pdf_magic = pdf_bytes.startswith(b"%PDF-") and str(item.get("pdf_magic", "")).startswith("%PDF-")
    for passed, code, detail in (
        (page_hash_match, "html_byte_hash_mismatch", str(item.get("html_sha256", ""))),
        (pdf_hash_match, "pdf_byte_hash_mismatch", str(item.get("pdf_sha256", ""))),
        (byte_lengths_match, "retained_byte_length_mismatch", f"html={len(page_bytes)},pdf={len(pdf_bytes)}"),
        (pdf_magic, "pdf_magic_invalid", pdf_bytes[:8].hex()),
    ):
        if not passed:
            _issue(issues, code, month, detail)

    route_ok = _monthly_page_contract(str(item.get("official_monthly_page_url", "")), month)
    html_original_ok = _monthly_page_contract(str(item.get("html_cdx_original", "")), month)
    pdf_original_ok = _lifecycle_pdf_contract(str(item.get("official_pdf_url", "")), month)
    pdf_cdx_original_ok = _same_original(
        str(item.get("official_pdf_url", "")), str(item.get("pdf_cdx_original", ""))
    )
    html_wayback_ok = str(item.get("html_wayback_url", "")) == _raw_wayback(
        str(item.get("html_cdx_timestamp", "")), str(item.get("html_cdx_original", ""))
    )
    pdf_wayback_ok = str(item.get("pdf_wayback_url", "")) == _raw_wayback(
        str(item.get("pdf_cdx_timestamp", "")), str(item.get("pdf_cdx_original", ""))
    )
    linked_pdf = _independent_exact_pdf_link(page_bytes, str(item.get("html_cdx_original", "")), month)
    linked_pdf_ok = linked_pdf == str(item.get("official_pdf_url", ""))
    cdx_ok = _validate_cdx_contract(producer, month, item, captures, issues)
    transport_ok = _validate_transport_contract(
        producer,
        month,
        item,
        transport,
        {"html": page_bytes, "pdf": pdf_bytes},
        issues,
    )
    for passed, code, detail in (
        (route_ok, "official_html_route_invalid", str(item.get("official_monthly_page_url", ""))),
        (html_original_ok, "html_cdx_original_contract_invalid", str(item.get("html_cdx_original", ""))),
        (pdf_original_ok, "official_pdf_url_contract_invalid", str(item.get("official_pdf_url", ""))),
        (pdf_cdx_original_ok, "pdf_cdx_original_mismatch", str(item.get("pdf_cdx_original", ""))),
        (html_wayback_ok, "html_wayback_url_contract_invalid", str(item.get("html_wayback_url", ""))),
        (pdf_wayback_ok, "pdf_wayback_url_contract_invalid", str(item.get("pdf_wayback_url", ""))),
        (linked_pdf_ok, "archived_html_exact_link_mismatch", linked_pdf),
    ):
        if not passed:
            _issue(issues, code, month, detail)

    parsed: dict[str, Any] | None = None
    parse_error = ""
    try:
        parsed = independently_parse_pdf_pages(
            extract_pages(pdf_bytes),
            month,
            source_url=str(item.get("official_pdf_url", "")),
            source_path=f"raw/pdfs/{month}.pdf",
            source_sha256=str(item.get("pdf_sha256", "")),
        )
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
        _issue(issues, "independent_pdf_parse_failed", month, parse_error)
    exact_month = bool(parsed and parsed.get("exact_month"))
    if not exact_month:
        _issue(issues, "independent_exact_month_failed", month, str((parsed or {}).get("title_months", [])))
    sections_valid = bool(parsed and parsed.get("required_sections_valid"))
    if not sections_valid:
        detail = json.dumps((parsed or {}).get("sections", {}), sort_keys=True)
        _issue(issues, "required_section_cardinality_invalid", month, detail)
    capture_valid = all(
        [
            page_hash_match,
            pdf_hash_match,
            byte_lengths_match,
            pdf_magic,
            route_ok,
            html_original_ok,
            pdf_original_ok,
            pdf_cdx_original_ok,
            html_wayback_ok,
            pdf_wayback_ok,
            linked_pdf_ok,
            cdx_ok,
            transport_ok,
            exact_month,
            sections_valid,
        ]
    )
    return (
        {
            "report_month": month,
            "capture_valid": capture_valid,
            "html_sha256_match": page_hash_match,
            "pdf_sha256_match": pdf_hash_match,
            "byte_lengths_match": byte_lengths_match,
            "pdf_magic_valid": pdf_magic,
            "cdx_contract_valid": cdx_ok,
            "url_month_contract_valid": route_ok and html_original_ok and pdf_original_ok,
            "archived_html_exact_link_valid": linked_pdf_ok,
            "transport_contract_valid": transport_ok,
            "independent_exact_month": exact_month,
            "required_sections_valid": sections_valid,
            "independent_delisting_events": len([row for row in (parsed or {}).get("events", []) if row["channel"] == DELISTING_CHANNEL]),
            "independent_halt_events": len([row for row in (parsed or {}).get("events", []) if row["channel"] == HALT_CHANNEL]),
            "parse_error": parse_error,
        },
        parsed,
    )


def _validate_copied_month(
    producer: Path,
    month: str,
    item: dict[str, Any],
    captures: pd.DataFrame,
    transport: pd.DataFrame,
    extract_pages: PageExtractor,
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    route = INDEPENDENT_COPIED_FILE_REGISTRY.get(month)
    if not COPIED_FILE_ROUTE_ENABLED or route is None:
        _issue(
            issues,
            "copied_file_route_not_independently_admitted",
            month,
            "month is outside the enabled independently audited registry",
        )
        return _failed_month_row(month, "copied_file_route_not_admitted"), None

    parent_path = _safe_source_path(
        producer,
        item.get("parent_index_path", ""),
        f"raw/pages/{month}.html",
        issues,
        month,
    )
    pdf_path = _safe_source_path(
        producer,
        item.get("pdf_path", ""),
        f"raw/pdfs/{month}.pdf",
        issues,
        month,
    )
    if (
        parent_path is None
        or pdf_path is None
        or not parent_path.is_file()
        or not pdf_path.is_file()
    ):
        _issue(issues, "missing_capture_bytes", month, "parent index or live PDF bytes unavailable")
        return _failed_month_row(month, "missing_capture_bytes"), None

    parent_response = parent_path.read_bytes()
    try:
        parent_html = _independent_decode_entity(parent_response)
    except ValueError as exc:
        _issue(issues, "parent_index_decode_failed", month, str(exc))
        parent_html = b""
    pdf_bytes = pdf_path.read_bytes()
    parent_response_hash_match = (
        _sha_bytes(parent_response) == str(item.get("parent_index_response_sha256", ""))
    )
    parent_decoded_hash_match = (
        bool(parent_html)
        and _sha_bytes(parent_html) == str(item.get("parent_index_decoded_sha256", ""))
    )
    pdf_hash_match = _sha_bytes(pdf_bytes) == str(item.get("pdf_sha256", ""))
    byte_lengths_match = (
        len(parent_response) == _int(item.get("parent_index_response_byte_length"))
        and len(parent_html) == _int(item.get("parent_index_decoded_byte_length"))
        and len(pdf_bytes) == _int(item.get("pdf_byte_length"))
    )
    pdf_magic = pdf_bytes.startswith(b"%PDF-") and str(item.get("pdf_magic", "")).startswith("%PDF-")

    registry_ok = all(
        str(item.get(key, "")) == route[key]
        for key in (
            "official_pdf_url",
            "parent_index_url",
            "parent_link_url",
            "archived_parent_lineage_strength",
        )
    )
    parent_original_ok = _same_archive_original(
        str(item.get("parent_index_cdx_original", "")), route["parent_index_url"]
    )
    parent_timestamp_ok = (
        str(item.get("parent_index_cdx_timestamp", ""))
        == route["parent_capture_timestamp"]
    )
    parent_wayback_ok = str(item.get("parent_index_wayback_url", "")) == _raw_wayback(
        str(item.get("parent_index_cdx_timestamp", "")),
        str(item.get("parent_index_cdx_original", "")),
    )
    pdf_url_ok = (
        str(item.get("official_pdf_url", "")) == route["official_pdf_url"]
        and _official_jpx_pdf_contract(route["official_pdf_url"], month)
        and str(item.get("pdf_final_url", "")) == route["official_pdf_url"]
    )
    parent_link_ok = _independent_parent_link_present(
        parent_html,
        str(item.get("parent_index_cdx_original", "")),
        route["parent_link_url"],
    )
    lineage_strength_ok = (
        str(item.get("archived_parent_lineage_strength", ""))
        == route["archived_parent_lineage_strength"]
        and _semantic_scalar(item.get("archived_parent_direct_pdf_link", ""))
        == (
            "true"
            if route["archived_parent_lineage_strength"] == DIRECT_ARCHIVED_PDF_LINEAGE
            else "false"
        )
    )
    cdx_ok = _validate_copied_cdx_contract(
        producer, month, item, captures, issues
    )
    transport_ok = _validate_copied_transport_contract(
        producer,
        month,
        item,
        transport,
        parent_response,
        pdf_bytes,
        issues,
    )
    for passed, code, detail in (
        (parent_response_hash_match, "parent_index_response_hash_mismatch", "raw response"),
        (parent_decoded_hash_match, "parent_index_decoded_hash_mismatch", "decoded HTML"),
        (pdf_hash_match, "pdf_byte_hash_mismatch", str(item.get("pdf_sha256", ""))),
        (byte_lengths_match, "retained_byte_length_mismatch", "parent response/decoded/PDF"),
        (pdf_magic, "pdf_magic_invalid", pdf_bytes[:8].hex()),
        (registry_ok, "independent_copied_registry_disagreement", str(item.get("official_pdf_url", ""))),
        (parent_original_ok, "parent_index_cdx_original_contract_invalid", str(item.get("parent_index_cdx_original", ""))),
        (parent_timestamp_ok, "parent_index_capture_timestamp_invalid", str(item.get("parent_index_cdx_timestamp", ""))),
        (parent_wayback_ok, "parent_index_wayback_url_contract_invalid", str(item.get("parent_index_wayback_url", ""))),
        (pdf_url_ok, "official_pdf_url_contract_invalid", str(item.get("official_pdf_url", ""))),
        (parent_link_ok, "archived_parent_exact_link_mismatch", route["parent_link_url"]),
        (lineage_strength_ok, "archived_parent_lineage_strength_invalid", str(item.get("archived_parent_lineage_strength", ""))),
    ):
        if not passed:
            _issue(issues, code, month, detail)

    parsed: dict[str, Any] | None = None
    parse_error = ""
    try:
        parsed = independently_parse_pdf_pages(
            extract_pages(pdf_bytes),
            month,
            source_url=route["official_pdf_url"],
            source_path=f"raw/pdfs/{month}.pdf",
            source_sha256=str(item.get("pdf_sha256", "")),
        )
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
        _issue(issues, "independent_pdf_parse_failed", month, parse_error)
    exact_month = bool(parsed and parsed.get("exact_month"))
    sections_valid = bool(parsed and parsed.get("required_sections_valid"))
    sections_nonempty = bool(
        parsed
        and all(
            _int(section.get("event_count")) > 0
            for section in parsed.get("sections", {}).values()
        )
        and set(parsed.get("sections", {})) == set(CHANNEL_HEADINGS)
    )
    if not exact_month:
        _issue(issues, "independent_exact_month_failed", month, str((parsed or {}).get("title_months", [])))
    if not sections_valid:
        _issue(issues, "required_section_cardinality_invalid", month, json.dumps((parsed or {}).get("sections", {}), sort_keys=True))
    if not sections_nonempty:
        _issue(issues, "required_section_nonempty_invalid", month, "no zero claim is inferred")

    capture_valid = all(
        (
            parent_response_hash_match,
            parent_decoded_hash_match,
            pdf_hash_match,
            byte_lengths_match,
            pdf_magic,
            registry_ok,
            parent_original_ok,
            parent_timestamp_ok,
            parent_wayback_ok,
            pdf_url_ok,
            parent_link_ok,
            lineage_strength_ok,
            cdx_ok,
            transport_ok,
            exact_month,
            sections_valid,
            sections_nonempty,
        )
    )
    return (
        {
            "report_month": month,
            "route_mode": COPIED_FILE_ROUTE,
            "archived_parent_lineage_strength": route["archived_parent_lineage_strength"],
            "capture_valid": capture_valid,
            "html_sha256_match": parent_response_hash_match and parent_decoded_hash_match,
            "pdf_sha256_match": pdf_hash_match,
            "byte_lengths_match": byte_lengths_match,
            "pdf_magic_valid": pdf_magic,
            "cdx_contract_valid": cdx_ok,
            "url_month_contract_valid": registry_ok and parent_original_ok and pdf_url_ok,
            "archived_html_exact_link_valid": parent_link_ok,
            "transport_contract_valid": transport_ok,
            "independent_exact_month": exact_month,
            "required_sections_valid": sections_valid and sections_nonempty,
            "independent_delisting_events": len([row for row in (parsed or {}).get("events", []) if row["channel"] == DELISTING_CHANNEL]),
            "independent_halt_events": len([row for row in (parsed or {}).get("events", []) if row["channel"] == HALT_CHANNEL]),
            "parse_error": parse_error,
        },
        parsed,
    )


def _validate_cdx_contract(
    producer: Path,
    month: str,
    item: dict[str, Any],
    captures: pd.DataFrame,
    issues: list[dict[str, str]],
) -> bool:
    valid = True
    for kind in ("html", "pdf"):
        relative = f"raw/cdx/{month}_{kind}.json"
        path = _contained_nonsymlink_path(producer, relative)
        if path is None or not path.is_file():
            _issue(issues, "cdx_raw_bytes_missing", month, kind)
            valid = False
            continue
        try:
            rows = _parse_cdx_independently(path.read_bytes())
        except Exception as exc:
            _issue(issues, "cdx_raw_parse_failed", month, f"{kind}:{type(exc).__name__}")
            valid = False
            continue
        expected = {
            field: str(item.get(f"{kind}_cdx_{field}", ""))
            for field in CDX_FIELDS
        }
        if expected not in rows:
            _issue(issues, "selected_cdx_capture_not_in_raw_response", month, kind)
            valid = False
        csv_rows = captures.loc[
            captures.get("report_month", pd.Series(dtype=str)).eq(month)
            & captures.get("resource_kind", pd.Series(dtype=str)).eq(kind)
        ]
        raw_counter = Counter(tuple(row[field] for field in CDX_FIELDS) for row in rows)
        csv_counter = Counter(
            tuple(str(row.get(field, "")) for field in CDX_FIELDS)
            for row in csv_rows.to_dict("records")
        )
        if raw_counter != csv_counter:
            _issue(issues, "cdx_csv_raw_disagreement", month, kind)
            valid = False
    return valid


def _validate_copied_cdx_contract(
    producer: Path,
    month: str,
    item: dict[str, Any],
    captures: pd.DataFrame,
    issues: list[dict[str, str]],
) -> bool:
    relative = f"raw/cdx/{month}_parent_index.json"
    path = _contained_nonsymlink_path(producer, relative)
    if path is None or not path.is_file():
        _issue(issues, "cdx_raw_bytes_missing", month, "parent_index")
        return False
    try:
        rows = _parse_cdx_independently(path.read_bytes())
    except Exception as exc:
        _issue(issues, "cdx_raw_parse_failed", month, f"parent_index:{type(exc).__name__}")
        return False
    expected = {
        field: str(item.get(f"parent_index_cdx_{field}", ""))
        for field in CDX_FIELDS
    }
    valid = expected in rows
    if not valid:
        _issue(issues, "selected_cdx_capture_not_in_raw_response", month, "parent_index")
    csv_rows = captures.loc[
        captures.get("report_month", pd.Series(dtype=str)).eq(month)
        & captures.get("resource_kind", pd.Series(dtype=str)).eq("parent_index")
    ]
    raw_counter = Counter(tuple(row[field] for field in CDX_FIELDS) for row in rows)
    csv_counter = Counter(
        tuple(str(row.get(field, "")) for field in CDX_FIELDS)
        for row in csv_rows.to_dict("records")
    )
    if raw_counter != csv_counter:
        _issue(issues, "cdx_csv_raw_disagreement", month, "parent_index")
        valid = False
    return valid


def _validate_copied_transport_contract(
    producer: Path,
    month: str,
    item: dict[str, Any],
    transport: pd.DataFrame,
    parent_response: bytes,
    pdf_bytes: bytes,
    issues: list[dict[str, str]],
) -> bool:
    rows = transport.loc[transport.get("report_month", pd.Series(dtype=str)).eq(month)]
    kinds = Counter(rows.get("request_kind", pd.Series(dtype=str)).tolist())
    expected_kinds = {
        "parent_index_cdx",
        "parent_index_capture",
        "official_pdf_live",
    }
    if set(kinds) != expected_kinds or any(kinds[kind] != 1 for kind in expected_kinds):
        _issue(issues, "transport_request_set_invalid", month, json.dumps(kinds, sort_keys=True))
        return False

    valid = True
    contracts = {
        "parent_index_cdx": {
            "path": f"raw/cdx/{month}_parent_index.json",
            "url": "cdx",
            "body": None,
        },
        "parent_index_capture": {
            "path": f"raw/pages/{month}.html",
            "url": str(item.get("parent_index_wayback_url", "")),
            "body": parent_response,
        },
        "official_pdf_live": {
            "path": f"raw/pdfs/{month}.pdf",
            "url": str(item.get("official_pdf_url", "")),
            "body": pdf_bytes,
        },
    }
    for row in rows.to_dict("records"):
        request_kind = str(row.get("request_kind", ""))
        contract = contracts[request_kind]
        raw_path = str(row.get("raw_path", ""))
        path = _contained_nonsymlink_path(producer, raw_path)
        body = path.read_bytes() if path is not None and path.is_file() else b""
        row_valid = (
            raw_path == contract["path"]
            and path is not None
            and path.is_file()
            and _int(row.get("http_status")) == 200
            and _int(row.get("byte_length")) == len(body)
            and str(row.get("sha256", "")) == _sha_bytes(body)
        )
        if contract["url"] == "cdx":
            row_valid = row_valid and _archive_cdx_request_url_contract(
                str(row.get("request_url", "")),
                str(item.get("parent_index_cdx_original", "")),
            ) and str(row.get("final_url", "")) == str(row.get("request_url", ""))
        else:
            row_valid = row_valid and (
                str(row.get("request_url", "")) == contract["url"]
                and str(row.get("final_url", "")) == contract["url"]
                and body == contract["body"]
            )
            if request_kind == "parent_index_capture":
                row_valid = row_valid and (
                    str(row.get("http_headers", ""))
                    == str(item.get("parent_index_http_headers", ""))
                    and _retained_header_contract(
                        row,
                        str(item.get("parent_index_cdx_timestamp", "")),
                        str(item.get("parent_index_cdx_mimetype", "")),
                        len(body),
                    )
                )
            else:
                row_valid = row_valid and (
                    str(row.get("http_headers", ""))
                    == str(item.get("pdf_http_headers", ""))
                    and _official_live_pdf_header_contract(row, len(body))
                )
        row_valid = row_valid and _attempts_contract(row.get("attempts"), len(body))
        if not row_valid:
            _issue(issues, "transport_row_binding_invalid", f"{month}:{request_kind}", raw_path)
            valid = False

    selected_valid = (
        str(item.get("parent_index_cdx_statuscode", "")) == "200"
        and str(item.get("parent_index_cdx_timestamp", ""))
        == INDEPENDENT_COPIED_FILE_REGISTRY[month]["parent_capture_timestamp"]
        and str(item.get("parent_index_cdx_mimetype", "")) == "text/html"
        and _int(item.get("parent_index_cdx_length")) > 0
        and str(item.get("parent_index_cdx_digest", ""))
        == _cdx_digest(parent_response)
    )
    if not selected_valid:
        _issue(issues, "selected_cdx_retained_bytes_binding_invalid", f"{month}:parent_index", "")
        valid = False
    return valid


def _validate_transport_contract(
    producer: Path,
    month: str,
    item: dict[str, Any],
    transport: pd.DataFrame,
    retained: dict[str, bytes],
    issues: list[dict[str, str]],
) -> bool:
    rows = transport.loc[transport.get("report_month", pd.Series(dtype=str)).eq(month)]
    kinds = Counter(rows.get("request_kind", pd.Series(dtype=str)).tolist())
    expected = {"html_cdx", "html_capture", "pdf_cdx", "pdf_capture"}
    valid = set(kinds) == expected and all(kinds[kind] == 1 for kind in expected)
    if not valid:
        _issue(issues, "transport_request_set_invalid", month, json.dumps(kinds, sort_keys=True))
        return False
    valid = True
    expected = {
        "html_cdx": {
            "path": f"raw/cdx/{month}_html.json",
            "url": "cdx",
            "kind": "html",
        },
        "html_capture": {
            "path": f"raw/pages/{month}.html",
            "url": str(item.get("html_wayback_url", "")),
            "kind": "html",
        },
        "pdf_cdx": {
            "path": f"raw/cdx/{month}_pdf.json",
            "url": "cdx",
            "kind": "pdf",
        },
        "pdf_capture": {
            "path": f"raw/pdfs/{month}.pdf",
            "url": str(item.get("pdf_wayback_url", "")),
            "kind": "pdf",
        },
    }
    for row in rows.to_dict("records"):
        request_kind = str(row.get("request_kind", ""))
        contract = expected[request_kind]
        raw_path = str(row.get("raw_path", ""))
        path = _contained_nonsymlink_path(producer, raw_path)
        body = path.read_bytes() if path is not None and path.is_file() else b""
        row_valid = (
            raw_path == contract["path"]
            and path is not None
            and path.is_file()
            and _int(row.get("http_status")) == 200
            and _int(row.get("byte_length")) == len(body)
            and str(row.get("sha256", "")) == _sha_bytes(body)
        )
        if contract["url"] == "cdx":
            row_valid = row_valid and _cdx_request_url_contract(
                str(row.get("request_url", "")),
                str(item.get(f"{contract['kind']}_cdx_original", "")),
            ) and str(row.get("final_url", "")) == str(row.get("request_url", ""))
        else:
            row_valid = row_valid and (
                str(row.get("request_url", "")) == contract["url"]
                and str(row.get("final_url", "")) == contract["url"]
                and body == retained[contract["kind"]]
                and str(row.get("http_headers", ""))
                == str(item.get(f"{contract['kind']}_http_headers", ""))
            )
            row_valid = row_valid and _retained_header_contract(
                row,
                str(item.get(f"{contract['kind']}_cdx_timestamp", "")),
                str(item.get(f"{contract['kind']}_cdx_mimetype", "")),
                len(body),
            )
        row_valid = row_valid and _attempts_contract(row.get("attempts"), len(body))
        if not row_valid:
            _issue(issues, "transport_row_binding_invalid", f"{month}:{request_kind}", raw_path)
            valid = False

    for kind, body in retained.items():
        selected_valid = (
            str(item.get(f"{kind}_cdx_statuscode", "")) == "200"
            and bool(re.fullmatch(r"\d{14}", str(item.get(f"{kind}_cdx_timestamp", ""))))
            and str(item.get(f"{kind}_cdx_mimetype", ""))
            == ("text/html" if kind == "html" else "application/pdf")
            and _int(item.get(f"{kind}_cdx_length")) > 0
            and str(item.get(f"{kind}_cdx_digest", "")) == _cdx_digest(body)
        )
        if not selected_valid:
            _issue(issues, "selected_cdx_retained_bytes_binding_invalid", f"{month}:{kind}", "")
            valid = False
    return valid


def _cdx_request_url_contract(url: str, original: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "web.archive.org"
        and parsed.path == "/cdx/search/cdx"
        and len(query.get("url", [])) == 1
        and _same_original(query["url"][0], original)
        and {key: value for key, value in query.items() if key != "url"} == {
            "output": ["json"],
            "fl": [",".join(CDX_FIELDS)],
            "filter": ["statuscode:200"],
            "collapse": ["digest"],
        }
    )


def _archive_cdx_request_url_contract(url: str, original: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "web.archive.org"
        and parsed.path == "/cdx/search/cdx"
        and len(query.get("url", [])) == 1
        and _same_archive_original(query["url"][0], original)
        and {key: value for key, value in query.items() if key != "url"}
        == {
            "output": ["json"],
            "fl": [",".join(CDX_FIELDS)],
            "filter": ["statuscode:200"],
            "collapse": ["digest"],
        }
    )


def _attempts_contract(raw: object, body_length: int) -> bool:
    try:
        attempts = json.loads(str(raw))
    except (TypeError, ValueError):
        return False
    if not isinstance(attempts, list) or not attempts:
        return False
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or _int(attempt.get("attempt")) != index:
            return False
    terminal = attempts[-1]
    return (
        _int(terminal.get("status")) == 200
        and _int(terminal.get("body_length")) == body_length
        and str(terminal.get("error", "")) == ""
    )


def _retained_header_contract(
    row: dict[str, Any], timestamp: str, mimetype: str, body_length: int
) -> bool:
    try:
        headers = json.loads(str(row.get("http_headers", "")))
    except (TypeError, ValueError):
        return False
    if not isinstance(headers, dict):
        return False
    lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
    content_type = lowered.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type and content_type != mimetype.casefold():
        return False
    if "content-length" in lowered and _int(lowered["content-length"]) != body_length:
        return False
    if "memento-datetime" in lowered:
        try:
            observed = datetime.strptime(
                lowered["memento-datetime"], "%a, %d %b %Y %H:%M:%S GMT"
            ).strftime("%Y%m%d%H%M%S")
        except ValueError:
            return False
        if observed != timestamp:
            return False
    return True


def _official_live_pdf_header_contract(row: dict[str, Any], body_length: int) -> bool:
    try:
        headers = json.loads(str(row.get("http_headers", "")))
    except (TypeError, ValueError):
        return False
    if not isinstance(headers, dict):
        return False
    lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
    content_type = lowered.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/pdf":
        return False
    return "content-length" not in lowered or _int(lowered["content-length"]) == body_length


def _cdx_digest(body: bytes) -> str:
    return base64.b32encode(hashlib.sha1(body).digest()).decode("ascii").rstrip("=")


def _reconcile_event_sets(
    producer: Path,
    selected: list[str],
    inventory: pd.DataFrame,
    claims: pd.DataFrame,
    events: pd.DataFrame,
    independent: list[dict[str, Any]],
    month_rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    producer_keys = _event_counter(events.to_dict("records"))
    independent_keys = _event_counter(independent)
    capture_valid_by_month = {
        str(row.get("report_month", "")): bool(row.get("capture_valid")) for row in month_rows
    }
    if producer_keys != independent_keys:
        _issue(issues, "global_exact_event_set_disagreement", "exact_events", _counter_delta(producer_keys, independent_keys))
    for month in selected:
        inv = inventory.loc[inventory.get("report_month", pd.Series(dtype=str)).eq(month)]
        source_sha = str(inv.iloc[0].get("pdf_sha256", "")) if len(inv) == 1 else ""
        source_url = str(inv.iloc[0].get("official_pdf_url", "")) if len(inv) == 1 else ""
        for channel in CHANNEL_HEADINGS:
            producer_channel = Counter(
                key for key, count in producer_keys.items()
                for _ in range(count)
                if key[1] == month and key[2] == channel
            )
            independent_channel = Counter(
                key for key, count in independent_keys.items()
                for _ in range(count)
                if key[1] == month and key[2] == channel
            )
            channel_claims = claims.loc[
                claims.get("report_month", pd.Series(dtype=str)).eq(month)
                & claims.get("channel", pd.Series(dtype=str)).eq(channel)
            ]
            expected_claims = 1 if sum(independent_channel.values()) > 0 else 0
            claim_cardinality = len(channel_claims) == expected_claims
            event_set_match = producer_channel == independent_channel
            claim_count_match = True
            source_contract = True
            if expected_claims:
                claim = channel_claims.iloc[0].to_dict() if len(channel_claims) == 1 else {}
                claim_count_match = _int(claim.get("event_count")) == sum(independent_channel.values())
                inventory_row = inv.iloc[0]
                if inventory_row.get("route_mode") == COPIED_FILE_ROUTE:
                    expected_timestamp = inventory_row.get("parent_index_cdx_timestamp")
                    expected_wayback = inventory_row.get("parent_index_wayback_url")
                else:
                    expected_timestamp = inventory_row.get("pdf_cdx_timestamp")
                    expected_wayback = inventory_row.get("pdf_wayback_url")
                source_contract = (
                    claim.get("source_sha256") == source_sha
                    and claim.get("official_source_url") == source_url
                    and str(claim.get("zero_event_asserted", "")).casefold() == "false"
                    and claim.get("claim_kind") == "exact_nonempty_official_report_event_set"
                    and claim.get("validation_status") == "pending_independent_validation"
                    and claim.get("archive_timestamp") == expected_timestamp
                    and claim.get("wayback_url") == expected_wayback
                )
            capture_valid = capture_valid_by_month.get(month, False)
            passed = (
                capture_valid
                and claim_cardinality
                and event_set_match
                and claim_count_match
                and source_contract
            )
            rows.append(
                {
                    "report_month": month,
                    "channel": channel,
                    "producer_event_count": sum(producer_channel.values()),
                    "independent_event_count": sum(independent_channel.values()),
                    "claim_count": len(channel_claims),
                    "expected_nonempty_claim_count": expected_claims,
                    "event_set_match": event_set_match,
                    "claim_cardinality_match": claim_cardinality,
                    "claim_event_count_match": claim_count_match,
                    "claim_source_contract_match": source_contract,
                    "capture_valid": capture_valid,
                    "zero_inferred": False,
                    "validated": passed,
                }
            )
            for condition, code in (
                (event_set_match, "channel_exact_event_set_disagreement"),
                (claim_cardinality, "nonempty_claim_cardinality_disagreement"),
                (claim_count_match, "claim_event_count_disagreement"),
                (source_contract, "claim_source_contract_disagreement"),
                (capture_valid, "claim_capture_not_fully_validated"),
            ):
                if not condition:
                    _issue(issues, code, f"{month}:{channel}", "fail closed")

    for item in events.to_dict("records"):
        month = str(item.get("report_month", ""))
        expected_path = f"raw/pdfs/{month}.pdf"
        inv = inventory.loc[inventory.get("report_month", pd.Series(dtype=str)).eq(month)]
        valid_source = len(inv) == 1 and (
            item.get("source_path") == expected_path
            and item.get("source_sha256") == inv.iloc[0].get("pdf_sha256")
            and item.get("source_url") == inv.iloc[0].get("official_pdf_url")
        )
        if not valid_source:
            _issue(issues, "event_source_lineage_invalid", str(item.get("event_id", "")), month)
    return rows


def _event_counter(rows: list[dict[str, Any]]) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(_semantic_scalar(row.get(field, "")) for field in EVENT_COLUMNS)
        for row in rows
    )


def _validate_global_contracts(
    manifest: dict[str, Any],
    selected: list[str],
    inventory: pd.DataFrame,
    captures: pd.DataFrame,
    transport: pd.DataFrame,
    claims: pd.DataFrame,
    events: pd.DataFrame,
    parquet_events: pd.DataFrame,
    completion: pd.DataFrame,
    failures: pd.DataFrame,
    acquisition_plan: pd.DataFrame,
    checkpoint: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    route_modes = set(
        inventory.get("route_mode", pd.Series(dtype=str)).astype(str)
    )
    copied_only = route_modes == {COPIED_FILE_ROUTE}
    expected_preservation = (
        COPIED_PRESERVATION_TRANSPORT
        if copied_only
        else "Internet Archive Wayback Machine"
    )
    if manifest.get("preservation_transport") != expected_preservation:
        _issue(issues, "preservation_transport_contract_invalid", "manifest", str(manifest.get("preservation_transport")))
    if manifest.get("original_source") != "Tokyo Stock Exchange / JPX official monthly PDF bytes":
        _issue(issues, "original_source_contract_invalid", "manifest", str(manifest.get("original_source")))
    if manifest.get("absence_is_zero") is not False or int(manifest.get("zero_event_claims_published", -1)) != 0:
        _issue(issues, "inferred_zero_manifest_contract_invalid", "manifest", "absence must never be zero")
    if len(selected) != int(manifest.get("selected_month_count", -1)) or len(set(selected)) != len(selected):
        _issue(issues, "selected_month_manifest_contract_invalid", "manifest", str(selected))
    if not selected:
        _issue(issues, "selected_month_set_empty", "manifest", "institutional admission requires evidence")
    if copied_only:
        copied_contract = (
            set(selected).issubset(INDEPENDENT_COPIED_FILE_REGISTRY)
            and manifest.get("route_modes") == [COPIED_FILE_ROUTE]
            and _int(manifest.get("official_copied_file_registry_month_count"))
            == len(selected)
            and _int(manifest.get("registry_entries_guessed")) == 0
            and manifest.get("live_official_pdf_response_preserved") is True
            and manifest.get("source_contract")
            == (
                "explicit official JPX copied-file registry -> exact parent-index CDX/Wayback "
                "lineage -> live official JPX PDF response metadata and bytes"
            )
        )
        if not copied_contract:
            _issue(issues, "copied_manifest_route_contract_invalid", "manifest", "")
    try:
        invalid_months = [month for month in selected if str(pd.Period(month, freq="M")) != month]
    except ValueError:
        invalid_months = selected
    if invalid_months:
        _issue(issues, "selected_month_format_invalid", "manifest", "|".join(invalid_months))

    _validate_csv_parquet_events(events, parquet_events, issues)
    _validate_event_controls_and_identity(events, issues)
    _validate_month_membership(
        selected,
        {
            "source_inventory": inventory,
            "cdx_capture_inventory": captures,
            "transport_inventory": transport,
            "exact_nonempty_claims": claims,
            "exact_events": events,
            "failures_quarantine": failures,
        },
        issues,
    )
    if not claims.empty:
        zeros = claims.get("zero_event_asserted", pd.Series(dtype=str)).astype(str).str.casefold()
        counts = pd.to_numeric(claims.get("event_count", pd.Series(dtype=str)), errors="coerce")
        if not zeros.eq("false").all() or not counts.gt(0).all():
            _issue(issues, "producer_published_zero_or_empty_claim", "exact_nonempty_claims", "fail closed")
    if not completion.empty:
        inferred = completion.get("zero_inferred", pd.Series(dtype=str)).astype(str).str.casefold()
        if not inferred.eq("false").all():
            _issue(issues, "completion_plan_infers_zero", "completion_plan", "fail closed")
    if not failures.empty:
        quarantined = failures.get("quarantined", pd.Series(dtype=str)).map(_semantic_scalar)
        inferred = failures.get("zero_inferred", pd.Series(dtype=str)).map(_semantic_scalar)
        if not quarantined.eq("true").all() or not inferred.eq("false").all():
            _issue(issues, "quarantine_contract_invalid", "failures_quarantine", "")
    _validate_manifest_counts(
        manifest, selected, inventory, claims, events, failures, issues
    )
    _validate_completion_reconciliation(
        selected, claims, inventory, completion, acquisition_plan, issues
    )
    _validate_checkpoint_reconciliation(
        manifest,
        selected,
        checkpoint,
        inventory,
        captures,
        transport,
        claims,
        events,
        failures,
        issues,
    )


def _validate_csv_parquet_events(
    csv_events: pd.DataFrame,
    parquet_events: pd.DataFrame,
    issues: list[dict[str, str]],
) -> None:
    if list(csv_events.columns) != EVENT_COLUMNS or list(parquet_events.columns) != EVENT_COLUMNS:
        _issue(
            issues,
            "event_file_schema_mismatch",
            "exact_events",
            f"csv={list(csv_events.columns)} parquet={list(parquet_events.columns)}",
        )
        return
    if _record_counter(csv_events.to_dict("records")) != _record_counter(
        parquet_events.to_dict("records")
    ):
        _issue(issues, "exact_events_csv_parquet_semantic_mismatch", "exact_events", "")


def _validate_event_controls_and_identity(
    events: pd.DataFrame, issues: list[dict[str, str]]
) -> None:
    required = set(EVENT_COLUMNS)
    missing = required - set(events.columns)
    if missing:
        _issue(issues, "event_columns_missing", "exact_events", "|".join(sorted(missing)))
        return
    if events["event_id"].duplicated().any():
        _issue(issues, "event_id_not_unique", "exact_events", "")
    if events["source_record_sha256"].duplicated().any():
        _issue(issues, "source_record_sha256_not_unique", "exact_events", "")
    controls = {
        "known_at": "",
        "ric": "",
        "mapping_status": MAPPING_PENDING,
        "timestamp_status": TIMESTAMP_PENDING,
        "admissible_to_shared_halt_history": "false",
    }
    for field, expected in controls.items():
        observed = events[field].map(_semantic_scalar)
        if not observed.eq(expected).all():
            _issue(issues, "producer_event_control_invalid", field, expected)

    for row in events.to_dict("records"):
        scope = str(row.get("event_id", ""))
        try:
            source_page = int(str(row["source_page"]))
            ordinal = int(str(row["source_row_ordinal"]))
        except (KeyError, ValueError):
            _issue(issues, "event_source_position_invalid", scope, "")
            continue
        identity = {
            "version": SOURCE_RECORD_VERSION,
            "report_month": str(row["report_month"]),
            "channel": str(row["channel"]),
            "source_sha256": str(row["source_sha256"]),
            "source_page": source_page,
            "source_row_ordinal": ordinal,
            "event_date": str(row["event_date"]),
            "raw_code": str(row["raw_code"]),
            "normalized_local_code": str(row["normalized_local_code"]),
            "instrument_class": str(row["instrument_class"]),
            "issue_name": str(row["issue_name"]),
            "halt_start_printed": str(row["halt_start_printed"]),
            "halt_end_printed": str(row["halt_end_printed"]),
        }
        observed_sha = _record_sha256(identity)
        expected_id = hashlib.sha256(
            f"{EVENT_ID_VERSION}|{observed_sha}".encode("utf-8")
        ).hexdigest()[:24]
        if str(row["source_record_sha256"]) != observed_sha:
            _issue(issues, "source_record_sha256_recomputation_failed", scope, observed_sha)
        if str(row["event_id"]) != expected_id:
            _issue(issues, "event_id_recomputation_failed", scope, expected_id)


def _validate_month_membership(
    selected: list[str],
    tables: dict[str, pd.DataFrame],
    issues: list[dict[str, str]],
) -> None:
    allowed = set(selected)
    for name, frame in tables.items():
        if frame.empty:
            continue
        if "report_month" not in frame.columns:
            _issue(issues, "report_month_column_missing", name, "")
            continue
        outside = sorted(set(frame["report_month"].astype(str)) - allowed)
        if outside:
            _issue(issues, "out_of_selected_month_rows", name, "|".join(outside))


def _validate_manifest_counts(
    manifest: dict[str, Any],
    selected: list[str],
    inventory: pd.DataFrame,
    claims: pd.DataFrame,
    events: pd.DataFrame,
    failures: pd.DataFrame,
    issues: list[dict[str, str]],
) -> None:
    acquired = sorted(set(inventory.get("report_month", pd.Series(dtype=str)).astype(str)))
    expected = {
        "selected_month_count": len(selected),
        "acquired_exact_pdf_month_count": len(acquired),
        "exact_nonempty_claim_count": len(claims),
        "exact_event_count": len(events),
        "failure_count": len(failures),
    }
    for field, value in expected.items():
        if _int(manifest.get(field)) != value:
            _issue(issues, "manifest_count_reconciliation_failed", field, f"{manifest.get(field)}!={value}")
    if [str(value) for value in manifest.get("acquired_exact_pdf_months", [])] != acquired:
        _issue(issues, "manifest_acquired_months_reconciliation_failed", "manifest", str(acquired))
    expected_status = (
        "bounded_acquisition_complete"
        if acquired == sorted(selected) and failures.empty
        else "bounded_acquisition_with_failures"
    )
    if manifest.get("status") != expected_status:
        _issue(issues, "manifest_status_reconciliation_failed", "manifest", expected_status)
    if manifest.get("resumable_checkpoint") != "acquisition_checkpoint.json":
        _issue(issues, "manifest_checkpoint_path_invalid", "manifest", str(manifest.get("resumable_checkpoint")))
    if manifest.get("refinitiv_used") is not False:
        _issue(issues, "manifest_refinitiv_contract_invalid", "manifest", "")


def _validate_completion_reconciliation(
    selected: list[str],
    claims: pd.DataFrame,
    inventory: pd.DataFrame,
    completion: pd.DataFrame,
    acquisition_plan: pd.DataFrame,
    issues: list[dict[str, str]],
) -> None:
    if acquisition_plan.empty or completion.empty:
        _issue(issues, "completion_or_acquisition_plan_missing", "packet", "")
        return
    required_plan = {"report_month", "selected_in_run"}
    required_completion = {
        "report_month", "channel", "completion_status", "exact_nonempty_event_count", "zero_inferred"
    }
    if not required_plan.issubset(acquisition_plan.columns) or not required_completion.issubset(completion.columns):
        _issue(issues, "completion_or_acquisition_plan_schema_invalid", "packet", "")
        return
    plan_months = acquisition_plan["report_month"].astype(str).tolist()
    if len(plan_months) != len(set(plan_months)):
        _issue(issues, "acquisition_plan_month_duplicate", "acquisition_plan", "")
    selected_from_plan = sorted(
        acquisition_plan.loc[
            acquisition_plan["selected_in_run"].map(_semantic_scalar).eq("true"), "report_month"
        ].astype(str)
    )
    if selected_from_plan != sorted(selected):
        _issue(issues, "acquisition_plan_selected_months_mismatch", "acquisition_plan", "")
    expected_keys = {(month, channel) for month in plan_months for channel in CHANNEL_HEADINGS}
    observed_keys = list(zip(completion["report_month"], completion["channel"]))
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
        _issue(issues, "completion_plan_month_channel_set_mismatch", "completion_plan", "")
        return
    acquired = set(inventory.get("report_month", pd.Series(dtype=str)).astype(str))
    claim_counts = {
        (str(row["report_month"]), str(row["channel"])): _int(row["event_count"])
        for row in claims.to_dict("records")
    }
    selected_set = set(selected)
    for row in completion.to_dict("records"):
        key = (str(row["report_month"]), str(row["channel"]))
        if key in claim_counts:
            expected_status = "exact_nonempty_claim_pending_independent_validation"
            expected_count = claim_counts[key]
        elif key[0] in acquired:
            expected_status = "unresolved_no_nonempty_claim"
            expected_count = 0
        elif key[0] in selected_set:
            expected_status = "capture_or_parse_failed"
            expected_count = 0
        else:
            expected_status = "pending_not_in_bounded_run"
            expected_count = 0
        if (
            str(row["completion_status"]) != expected_status
            or _int(row["exact_nonempty_event_count"]) != expected_count
            or _semantic_scalar(row["zero_inferred"]) != "false"
        ):
            _issue(issues, "completion_plan_row_reconciliation_failed", f"{key[0]}:{key[1]}", "")


def _validate_checkpoint_reconciliation(
    manifest: dict[str, Any],
    selected: list[str],
    checkpoint: dict[str, Any],
    inventory: pd.DataFrame,
    captures: pd.DataFrame,
    transport: pd.DataFrame,
    claims: pd.DataFrame,
    events: pd.DataFrame,
    failures: pd.DataFrame,
    issues: list[dict[str, str]],
) -> None:
    if checkpoint.get("run_id") != manifest.get("run_id") or checkpoint.get("selected_months") != selected:
        _issue(issues, "checkpoint_run_month_contract_invalid", "acquisition_checkpoint", "")
    months = checkpoint.get("months")
    if not isinstance(months, dict) or set(months) != set(selected):
        _issue(issues, "checkpoint_month_set_mismatch", "acquisition_checkpoint", "")
        return
    table_map = {
        "inventory": inventory,
        "captures": captures,
        "transport": transport,
        "claims": claims,
        "events": events,
        "failures": failures,
    }
    for month in selected:
        state = months.get(month)
        if not isinstance(state, dict):
            _issue(issues, "checkpoint_month_state_invalid", month, "")
            continue
        if state.get("phase") != "complete" or state.get("terminal") is not True:
            _issue(issues, "checkpoint_month_not_terminal_complete", month, "")
        for key, frame in table_map.items():
            packet_rows = frame.loc[
                frame.get("report_month", pd.Series(dtype=str)).eq(month)
            ].to_dict("records")
            checkpoint_value = state.get(key, [] if key != "inventory" else {})
            checkpoint_rows = (
                [checkpoint_value] if key == "inventory" and checkpoint_value else checkpoint_value
            )
            if not isinstance(checkpoint_rows, list) or _record_counter(packet_rows) != _record_counter(checkpoint_rows):
                _issue(issues, "checkpoint_table_reconciliation_failed", f"{month}:{key}", "")


def _record_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        json.dumps(
            {str(key): _semantic_scalar(value) for key, value in sorted(row.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    )


def _semantic_scalar(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    if text.casefold() in {"true", "false"}:
        return text.casefold()
    return text


def _validate_packet_hashes(producer: Path, issues: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_path = _contained_nonsymlink_path(producer, "manifest.json")
    hashes_path = _contained_nonsymlink_path(producer, "artifact_hashes.json")
    if manifest_path is None or hashes_path is None:
        _issue(issues, "producer_control_path_symlink_or_escape", "packet", "")
        return rows
    manifest = _read_json(manifest_path, issues, "producer_manifest_hash")
    expected_inventory = str(manifest.get("artifact_hashes_sha256", ""))
    observed_inventory = _sha_path(hashes_path) if hashes_path.is_file() else ""
    inventory_match = bool(expected_inventory) and expected_inventory == observed_inventory
    rows.append({"path": "artifact_hashes.json", "expected_sha256": expected_inventory, "observed_sha256": observed_inventory, "hash_match": inventory_match})
    if not inventory_match:
        _issue(issues, "producer_hash_inventory_mismatch", "artifact_hashes.json", observed_inventory)
    declared = _read_json(hashes_path, issues, "producer_hash_inventory")
    for relative, expected in sorted(declared.items()):
        if _unsafe(relative):
            rows.append({"path": relative, "expected_sha256": expected, "observed_sha256": "", "hash_match": False})
            _issue(issues, "unsafe_producer_artifact_path", relative, "absolute or traversal path rejected")
            continue
        path = _contained_nonsymlink_path(producer, relative)
        if path is None:
            rows.append({"path": relative, "expected_sha256": expected, "observed_sha256": "", "hash_match": False})
            _issue(issues, "producer_artifact_symlink_or_escape", relative, "")
            continue
        observed = _sha_path(path) if path.is_file() else ""
        match = bool(expected) and expected == observed
        rows.append({"path": relative, "expected_sha256": expected, "observed_sha256": observed, "hash_match": match})
        if not match:
            _issue(issues, "producer_artifact_hash_mismatch", relative, observed)
    actual: set[str] = set()
    for path in producer.rglob("*"):
        relative = str(path.relative_to(producer))
        if path.is_symlink():
            _issue(issues, "producer_packet_symlink_rejected", relative, "")
            continue
        if path.is_file():
            actual.add(relative)
    expected_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
    for relative in sorted(actual - expected_files):
        _issue(issues, "undeclared_producer_file", relative, "not in immutable hash inventory")
    return rows


def _independent_exact_pdf_link(html: bytes, page_url: str, month: str) -> str:
    collector = _HrefCollector()
    try:
        collector.feed(html.decode("utf-8", errors="replace"))
    except Exception:
        return ""
    found = {
        urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, parsed.path, "", ""))
        for href in collector.hrefs
        for parsed in [urllib.parse.urlsplit(urllib.parse.urljoin(page_url, href))]
        if _lifecycle_pdf_contract(urllib.parse.urlunsplit(parsed), month)
    }
    return next(iter(found)) if len(found) == 1 else ""


def _independent_parent_link_present(
    html: bytes, parent_url: str, expected_url: str
) -> bool:
    collector = _HrefCollector()
    try:
        collector.feed(html.decode("utf-8", errors="replace"))
    except Exception:
        return False
    expected = _normalized_http_url(expected_url)
    found = {
        _normalized_http_url(urllib.parse.urljoin(parent_url, href))
        for href in collector.hrefs
    }
    return bool(expected) and expected in found


def _independent_decode_entity(body: bytes) -> bytes:
    if not body.startswith(b"\x1f\x8b"):
        return body
    try:
        return gzip.decompress(body)
    except (OSError, EOFError) as exc:
        raise ValueError("invalid gzip parent-index response") from exc


def _normalized_http_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )


def _parse_cdx_independently(body: bytes) -> list[dict[str, str]]:
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, list) or not value or value[0] != list(CDX_FIELDS):
        raise ValueError("CDX schema invalid")
    rows = []
    for raw in value[1:]:
        if not isinstance(raw, list) or len(raw) != len(CDX_FIELDS):
            raise ValueError("CDX row invalid")
        rows.append(dict(zip(CDX_FIELDS, (str(item) for item in raw))))
    return rows


def _monthly_page_contract(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    period = pd.Period(month, freq="M")
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == TSE_HOST
        and parsed.path == f"/english/market/data/geppo/{period.strftime('%Y%m')}.html"
        and not parsed.query
        and not parsed.fragment
    )


def _lifecycle_pdf_contract(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    yymm = pd.Period(month, freq="M").strftime("%y%m")
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname == TSE_HOST
        and re.fullmatch(
            rf"/english/market/data/geppo/(?:[^/?#]+/)*[^/?#]+-att/\d+_idou{yymm}\.pdf",
            parsed.path,
            flags=re.IGNORECASE,
        )
        and not parsed.query
        and not parsed.fragment
    )


def _official_jpx_pdf_contract(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    yymm = pd.Period(month, freq="M").strftime("%y%m")
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == JPX_HOST
        and re.fullmatch(
            rf"/(?:files/tse/english/market/data/geppo|"
            rf"english/markets/statistics-equities/monthly)/"
            rf"[^/?#]+-att/17_idou{yymm}\.pdf",
            parsed.path,
            flags=re.IGNORECASE,
        )
        and not parsed.query
        and not parsed.fragment
    )


def _same_original(left: str, right: str) -> bool:
    a = urllib.parse.urlsplit(left)
    b = urllib.parse.urlsplit(right)
    return (
        a.scheme in {"http", "https"}
        and b.scheme in {"http", "https"}
        and a.hostname == b.hostname == TSE_HOST
        and a.path == b.path
        and not a.query
        and not b.query
        and not a.fragment
        and not b.fragment
    )


def _same_archive_original(left: str, right: str) -> bool:
    a = urllib.parse.urlsplit(left)
    b = urllib.parse.urlsplit(right)
    return (
        a.scheme in {"http", "https"}
        and b.scheme in {"http", "https"}
        and a.hostname == b.hostname
        and a.port in {None, 80, 443}
        and b.port in {None, 80, 443}
        and a.path == b.path
        and not a.query
        and not b.query
        and not a.fragment
        and not b.fragment
    )


def _raw_wayback(timestamp: str, original: str) -> str:
    if not re.fullmatch(r"\d{14}", timestamp) or not original:
        return ""
    return f"{WAYBACK_ROOT}/{timestamp}id_/{original}"


def _section_boundary(line: str) -> bool:
    return bool(NEXT_HEADING.match(line) or NUMBERED_REPORT.match(line))


def _independent_issue_name(body: str, channel: str, range_start: int | None) -> str:
    end = len(body) if range_start is None else range_start
    if channel == HALT_CHANNEL:
        for marker in ("当該銘柄", "終日", "期間売買停止", "売買停止"):
            position = body.find(marker)
            if position >= 0:
                end = min(end, position)
    return _canonical(body[:end])


def _independent_instrument_class(label: str, prior: str) -> str:
    if not label:
        return prior
    normalized = _canonical(label).replace("＜", "<").replace("＞", ">")
    if normalized.casefold() in {"<外国株>", "<foreign stocks>"}:
        return prior or "equity"
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = re.sub(r"\s*([()])\s*", r"\1", normalized)
    normalized = re.sub(r"\s*<外国株>\s*", "", normalized).strip().casefold()
    groups = {
        "equity": {
            "市場第一部", "1st section", "市場第二部", "2nd section", "マザーズ",
            "mothers", "jasdaq standard", "jasdaq growth", "jasdaqスタンダード",
            "jasdaqグロース", "prime", "standard", "growth", "プライム",
            "スタンダード", "グロース", "tokyo pro market",
        },
        "etf": {"etf", "etfs", "etf(jdr)", "投信受益証券"},
        "etn": {"etn(jdr)", "etns(jdr)"},
        "reit": {"不動産投信(reit)", "reits"},
        "investment_certificate": {"投資証券", "investment certificate"},
        "straight_bond": {"普通債", "straight bonds"},
        "convertible_bond": {"転換社債型新株予約権付社債", "convertible bonds"},
        "equity_option": {"有価証券オプション", "individual options"},
        "index_future": {"指数先物"},
        "index_option": {
            "指数先物オプション", "index options", "topixオプション", "topix options"
        },
        "jgb_future": {"国債先物", "jgb futures"},
        "jgb_future_option": {"国債先物オプション", "options on jgb futures"},
        "all_derivatives": {"全派生商品", "the derivatives market"},
    }
    for instrument_class, labels in groups.items():
        if normalized in labels:
            return instrument_class
    raise ValueError(f"unrecognized independent instrument-class prefix: {label}")


def _independent_halt_timing(body: str, event_day: date) -> dict[str, Any]:
    jst = timezone(timedelta(hours=9))
    interval = TIME_INTERVAL_PATTERN.search(body)
    if interval:
        start_clock = _parse_clock(interval.group("start"))
        end_clock = _parse_clock(interval.group("end"))
        end_day = _parse_printed_date(interval.group("end_date")) if interval.group("end_date") else event_day
        start_dt = datetime.combine(event_day, start_clock).replace(tzinfo=jst)
        end_dt = datetime.combine(end_day, end_clock).replace(tzinfo=jst)
        printed_end_date = end_day.isoformat() if interval.group("end_date") else ""
        return {
            "start_printed": start_clock.strftime("%H:%M"),
            "end_printed": f"{printed_end_date} {end_clock.strftime('%H:%M')}".strip(),
            "start_jst": start_dt.isoformat(),
            "end_jst": end_dt.isoformat(),
            "range_start": interval.start(),
        }
    singles = list(TIME_PATTERN.finditer(body))
    if singles:
        start_clock = _parse_clock(singles[-1].group(0))
        start_dt = datetime.combine(event_day, start_clock).replace(tzinfo=jst)
        return {
            "start_printed": start_clock.strftime("%H:%M"),
            "end_printed": "",
            "start_jst": start_dt.isoformat(),
            "end_jst": "",
            "range_start": singles[-1].start(),
        }
    return _blank_timing()


def _blank_timing() -> dict[str, Any]:
    return {
        "start_printed": "",
        "end_printed": "",
        "start_jst": "",
        "end_jst": "",
        "range_start": None,
    }


def _parse_clock(value: str) -> time:
    hour, minute = [int(item) for item in re.findall(r"\d+", value)[:2]]
    return time(hour, minute)


def _parse_printed_date(value: str) -> date:
    year, month, day = [int(item) for item in re.findall(r"\d+", value)[:3]]
    return date(year, month, day)


def _record_sha256(identity: dict[str, Any]) -> str:
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: str) -> str:
    result = unicodedata.normalize("NFKC", str(value))
    result = " ".join(result.split())
    result = re.sub(r"\s+([),.;])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    result = re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff>])(?=[A-Za-z])", " ", result)
    result = re.sub(r"(?<=[A-Za-z.)])(?=\d{1,2}:\d{2})", " ", result)
    result = re.sub(r"\bFun d\b", "Fund", result)
    return result.strip()


def _safe_source_path(
    root: Path,
    recorded: object,
    expected: str,
    issues: list[dict[str, str]],
    scope: str,
) -> Path | None:
    value = str(recorded)
    if _unsafe(value) or value != expected:
        _issue(issues, "unsafe_or_noncanonical_source_path", scope, value)
        return None
    path = _contained_nonsymlink_path(root, value)
    if path is None:
        _issue(issues, "source_path_symlink_or_escape", scope, value)
        return None
    return path


def _unsafe(value: object) -> bool:
    key = str(value)
    path = Path(key)
    return not key or path.is_absolute() or ".." in path.parts


def _contained_nonsymlink_path(root: Path, relative: object) -> Path | None:
    value = str(relative)
    if _unsafe(value):
        return None
    root_resolved = root.resolve()
    candidate = root / value
    current = root
    for part in Path(value).parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    return resolved


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _failed_month_row(month: str, error: str) -> dict[str, Any]:
    return {
        "report_month": month,
        "capture_valid": False,
        "html_sha256_match": False,
        "pdf_sha256_match": False,
        "byte_lengths_match": False,
        "pdf_magic_valid": False,
        "cdx_contract_valid": False,
        "url_month_contract_valid": False,
        "archived_html_exact_link_valid": False,
        "transport_contract_valid": False,
        "independent_exact_month": False,
        "required_sections_valid": False,
        "independent_delisting_events": 0,
        "independent_halt_events": 0,
        "parse_error": error,
    }


def _write_validation_outputs(
    root: Path,
    hash_rows: list[dict[str, Any]],
    month_rows: list[dict[str, Any]],
    reconciliation: list[dict[str, Any]],
    independent_events: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    pd.DataFrame(hash_rows).to_csv(root / "producer_hash_validation.csv", index=False)
    pd.DataFrame(month_rows).to_csv(root / "month_validation.csv", index=False)
    pd.DataFrame(reconciliation).to_csv(root / "event_set_reconciliation.csv", index=False)
    pd.DataFrame(independent_events).to_csv(root / "independent_exact_events.csv", index=False)
    pd.DataFrame(issues, columns=["issue_code", "scope", "detail"]).to_csv(
        root / "validation_issues.csv", index=False
    )


def _counter_delta(left: Counter[Any], right: Counter[Any]) -> str:
    return json.dumps(
        {
            "producer_only": sum((left - right).values()),
            "independent_only": sum((right - left).values()),
        },
        sort_keys=True,
    )


def _read_json(path: Path, issues: list[dict[str, str]], scope: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        _issue(issues, "json_read_failed", scope, type(exc).__name__)
        return {}


def _read_csv(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        _issue(issues, "csv_read_failed", scope, type(exc).__name__)
        return pd.DataFrame()


def _read_parquet(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        _issue(issues, "parquet_read_failed", scope, type(exc).__name__)
        return pd.DataFrame()


def _extract_pages_pdfplumber(body: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(body)) as document:
        return [page.extract_text() or "" for page in document.pages]


def _int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1


def _safe_name(value: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _issue(issues: list[dict[str, str]], code: str, scope: str, detail: str) -> None:
    issues.append({"issue_code": code, "scope": scope, "detail": detail})


def _atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
