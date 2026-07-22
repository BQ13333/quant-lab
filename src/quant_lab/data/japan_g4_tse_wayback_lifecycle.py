from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from quant_lab.data import japan_g4_halt_report_events as report_event_parser
from quant_lab.data.japan_g4_halt_report_events import (
    DELISTING_CHANNEL,
    EVENT_COLUMNS,
    HALT_CHANNEL,
)


VERSION = "1.1-jpx-copied-file-lifecycle"
USER_AGENT = "quant-lab-japan-g4-tse-wayback-lifecycle/1.0"
CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WAYBACK_ROOT = "https://web.archive.org/web"
TSE_HOST = "www.tse.or.jp"
JPX_HOST = "www.jpx.co.jp"
START_MONTH = "2006-07"
END_MONTH = "2015-12"
CHANNELS = (DELISTING_CHANNEL, HALT_CHANNEL)
CDX_FIELDS = ("timestamp", "original", "statuscode", "digest", "length", "mimetype")
RETAINED_INSTRUMENT_CLASSES = {
    "全派生商品": "all_derivatives",
    "the derivatives market": "all_derivatives",
    "国債先物": "jgb_future",
    "jgb futures": "jgb_future",
    "国債先物オプション": "jgb_future_option",
    "options on jgb futures": "jgb_future_option",
}
UNKNOWN_INSTRUMENT_CLASS = "quarantined_unknown_instrument"
MISSING_INSTRUMENT_CLASS = "quarantined_missing_instrument"
COPIED_FILE_ROUTE = "official_jpx_copied_file_with_parent_index_wayback_lineage"
LEGACY_WAYBACK_ROUTE = "legacy_tse_page_and_pdf_wayback"
COPIED_FILE_ROUTE_ENABLED = True
DIRECT_ARCHIVED_PDF_LINEAGE = "archived_parent_index_directly_links_official_pdf"
WEAKER_ARCHIVED_MONTH_LINEAGE = "archived_parent_index_links_month_page_only"

_TSE_PARENT_INDEX = "http://www.tse.or.jp:80/english/market/data/geppo/index.html"
_TSE_PARENT_CAPTURE = "20150310160942"
_JPX_2015_PARENT_INDEX = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
    "00-archives-10.html"
)
_JPX_2015_PARENT_CAPTURE = "20250503084309"


def _copied_route(
    month: str,
    official_pdf_url: str,
    *,
    parent_index_url: str,
    parent_capture_timestamp: str,
    parent_link_url: str,
) -> dict[str, str]:
    lineage_strength = (
        DIRECT_ARCHIVED_PDF_LINEAGE
        if parent_link_url == official_pdf_url
        else WEAKER_ARCHIVED_MONTH_LINEAGE
    )
    return {
        "report_month": month,
        "route_mode": COPIED_FILE_ROUTE,
        "official_pdf_url": official_pdf_url,
        "parent_index_url": parent_index_url,
        "parent_capture_timestamp": parent_capture_timestamp,
        "parent_link_url": parent_link_url,
        "archived_parent_lineage_strength": lineage_strength,
    }


OFFICIAL_JPX_COPIED_FILE_REGISTRY: dict[str, dict[str, str]] = {
    "2014-09": _copied_route(
        "2014-09",
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje60000026ao9-att/17_idou1409.pdf",
        parent_index_url=_TSE_PARENT_INDEX,
        parent_capture_timestamp=_TSE_PARENT_CAPTURE,
        parent_link_url="http://www.tse.or.jp/english/market/data/geppo/201409.html",
    ),
    "2014-10": _copied_route(
        "2014-10",
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje60000027r55-att/17_idou1410.pdf",
        parent_index_url=_TSE_PARENT_INDEX,
        parent_capture_timestamp=_TSE_PARENT_CAPTURE,
        parent_link_url="http://www.tse.or.jp/english/market/data/geppo/201410.html",
    ),
    "2014-11": _copied_route(
        "2014-11",
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje600000294jl-att/17_idou1411.pdf",
        parent_index_url=_TSE_PARENT_INDEX,
        parent_capture_timestamp=_TSE_PARENT_CAPTURE,
        parent_link_url="http://www.tse.or.jp/english/market/data/geppo/201411.html",
    ),
    "2014-12": _copied_route(
        "2014-12",
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje6000002aeyz-att/17_idou1412.pdf",
        parent_index_url=_TSE_PARENT_INDEX,
        parent_capture_timestamp=_TSE_PARENT_CAPTURE,
        parent_link_url="http://www.tse.or.jp/english/market/data/geppo/201412.html",
    ),
    "2015-01": _copied_route(
        "2015-01",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj0000004mgs-att/17_idou1501.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj0000004mgs-att/17_idou1501.pdf",
    ),
    "2015-02": _copied_route(
        "2015-02",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000fnkm-att/17_idou1502.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000fnkm-att/17_idou1502.pdf",
    ),
    "2015-03": _copied_route(
        "2015-03",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000idmr-att/17_idou1503.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000idmr-att/17_idou1503.pdf",
    ),
    "2015-04": _copied_route(
        "2015-04",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000kmok-att/17_idou1504.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000kmok-att/17_idou1504.pdf",
    ),
    "2015-05": _copied_route(
        "2015-05",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000m7co-att/17_idou1505.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000m7co-att/17_idou1505.pdf",
    ),
    "2015-06": _copied_route(
        "2015-06",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000nd3i-att/17_idou1506.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000nd3i-att/17_idou1506.pdf",
    ),
    "2015-07": _copied_route(
        "2015-07",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000oz0h-att/17_idou1507.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000oz0h-att/17_idou1507.pdf",
    ),
    "2015-08": _copied_route(
        "2015-08",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000qc9y-att/17_idou1508.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000qc9y-att/17_idou1508.pdf",
    ),
    "2015-09": _copied_route(
        "2015-09",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000rgwu-att/17_idou1509.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000rgwu-att/17_idou1509.pdf",
    ),
    "2015-10": _copied_route(
        "2015-10",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000t1f1-att/17_idou1510.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000t1f1-att/17_idou1510.pdf",
    ),
    "2015-11": _copied_route(
        "2015-11",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000u9pb-att/17_idou1511.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000u9pb-att/17_idou1511.pdf",
    ),
    "2015-12": _copied_route(
        "2015-12",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000vj5c-att/17_idou1512.pdf",
        parent_index_url=_JPX_2015_PARENT_INDEX,
        parent_capture_timestamp=_JPX_2015_PARENT_CAPTURE,
        parent_link_url="https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000vj5c-att/17_idou1512.pdf",
    ),
}

Transport = Callable[[str], "FetchResponse"]
PageExtractor = Callable[[bytes], list[str]]


@dataclass(frozen=True)
class FetchResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]
    final_url: str


@dataclass(frozen=True)
class WaybackLifecycleArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


class AcquisitionError(RuntimeError):
    pass


class SequentialFetcher:
    """One rate-limited request stream with bounded retries."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        sleep_seconds: float = 1.0,
        retries: int = 2,
        timeout_seconds: float = 45.0,
    ) -> None:
        if sleep_seconds < 0 or retries < 0 or timeout_seconds <= 0:
            raise ValueError("invalid fetch pacing or retry bounds")
        self.sleep_seconds = float(sleep_seconds)
        self.retries = int(retries)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or self._urlopen
        self._last_request_at: float | None = None

    def get(self, url: str) -> tuple[FetchResponse, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            self._pace()
            try:
                response = self.transport(url)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": int(response.status),
                        "error": "",
                        "body_length": len(response.body),
                    }
                )
                if response.status not in {408, 425, 429, 500, 502, 503, 504}:
                    return response, attempts
            except Exception as exc:  # Network errors are evidence, not zero observations.
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": 0,
                        "error": f"{type(exc).__name__}:{exc}",
                        "body_length": 0,
                    }
                )
            if attempt <= self.retries:
                time.sleep(self.sleep_seconds * attempt)
        if last_error is not None and all(item["status"] == 0 for item in attempts):
            raise AcquisitionError(f"request failed after bounded retries: {url}") from last_error
        return response, attempts  # type: ignore[possibly-undefined]

    def _pace(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self.sleep_seconds - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _urlopen(self, url: str) -> FetchResponse:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return FetchResponse(
                    int(response.status),
                    response.read(),
                    dict(response.headers.items()),
                    response.geturl(),
                )
        except urllib.error.HTTPError as exc:
            return FetchResponse(
                int(exc.code),
                exc.read(),
                dict(exc.headers.items()),
                exc.geturl(),
            )


def official_monthly_page_url(month: str) -> str:
    period = _period(month)
    return f"http://{TSE_HOST}/english/market/data/geppo/{period.strftime('%Y%m')}.html"


def build_acquisition_plan(months: Iterable[str] | None = None) -> pd.DataFrame:
    requested = {_month(value) for value in months} if months is not None else None
    rows: list[dict[str, Any]] = []
    for period in pd.period_range(START_MONTH, END_MONTH, freq="M"):
        month = str(period)
        if requested is not None and month not in requested:
            continue
        rows.append(
            {
                "report_month": month,
                "priority": _priority(month),
                "priority_reason": _priority_reason(month),
                "official_monthly_page_url": official_monthly_page_url(month),
                "acquisition_route": (
                    COPIED_FILE_ROUTE
                    if COPIED_FILE_ROUTE_ENABLED
                    and month in OFFICIAL_JPX_COPIED_FILE_REGISTRY
                    else LEGACY_WAYBACK_ROUTE
                ),
                "registered_official_pdf_url": OFFICIAL_JPX_COPIED_FILE_REGISTRY.get(
                    month, {}
                ).get("official_pdf_url", ""),
            }
        )
    if requested is not None and len(rows) != len(requested):
        available = {row["report_month"] for row in rows}
        raise ValueError(f"months outside bounded scope: {sorted(requested - available)}")
    return pd.DataFrame(rows).sort_values(["priority", "report_month"]).reset_index(drop=True)


def cdx_query_url(original_url: str) -> str:
    query = urllib.parse.urlencode(
        [
            ("url", original_url),
            ("output", "json"),
            ("fl", ",".join(CDX_FIELDS)),
            ("filter", "statuscode:200"),
            ("collapse", "digest"),
        ]
    )
    return f"{CDX_ENDPOINT}?{query}"


def wayback_raw_url(timestamp: str, original_url: str) -> str:
    if not re.fullmatch(r"\d{14}", timestamp):
        raise ValueError("invalid Wayback timestamp")
    _require_http_url(original_url)
    return f"{WAYBACK_ROOT}/{timestamp}id_/{original_url}"


def parse_cdx_payload(body: bytes) -> list[dict[str, str]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise AcquisitionError(f"invalid CDX JSON: {type(exc).__name__}") from exc
    if not isinstance(payload, list) or not payload:
        return []
    if payload[0] != list(CDX_FIELDS):
        raise AcquisitionError("CDX field contract mismatch")
    rows: list[dict[str, str]] = []
    for raw in payload[1:]:
        if not isinstance(raw, list) or len(raw) != len(CDX_FIELDS):
            raise AcquisitionError("CDX row shape mismatch")
        row = dict(zip(CDX_FIELDS, (str(value) for value in raw)))
        if not re.fullmatch(r"\d{14}", row["timestamp"]):
            raise AcquisitionError("CDX timestamp invalid")
        rows.append(row)
    return rows


def exact_lifecycle_pdf_url(html: bytes, page_url: str, month: str) -> str:
    period = _period(month)
    yymm = period.strftime("%y%m")
    pattern = re.compile(
        rf"^/english/market/data/geppo/(?:[^/?#]+/)*[^/?#]+-att/"
        rf"\d+_idou{yymm}\.pdf$",
        flags=re.IGNORECASE,
    )
    found: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        candidate = urllib.parse.urljoin(page_url, str(link["href"]).strip())
        parsed = urllib.parse.urlsplit(candidate)
        if parsed.hostname == TSE_HOST and pattern.fullmatch(parsed.path):
            found.add(urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, parsed.path, "", "")))
    if len(found) != 1:
        raise AcquisitionError(
            f"expected one exact official lifecycle PDF link for {month}, found {len(found)}"
        )
    return next(iter(found))


def run_japan_g4_tse_wayback_lifecycle(
    *,
    run_id: str,
    months: Iterable[str],
    artifact_root: str | Path = "artifacts/japan_g4_tse_wayback_lifecycle",
    max_months: int = 12,
    sleep_seconds: float = 1.0,
    retries: int = 2,
    timeout_seconds: float = 45.0,
    transport: Transport | None = None,
    page_extractor: PageExtractor | None = None,
    resume: bool = True,
) -> WaybackLifecycleArtifacts:
    _safe_name(run_id)
    if max_months <= 0:
        raise ValueError("max_months must be positive")
    selected_plan = build_acquisition_plan(months)
    if len(selected_plan) > max_months:
        raise ValueError(f"bounded run permits at most {max_months} months")

    root = Path(artifact_root) / run_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"finalized write-once artifact already exists: {root}")
    if root.exists() and not resume:
        raise FileExistsError(f"resumable artifact directory already exists: {root}")
    for relative in ("raw/cdx", "raw/pages", "raw/pdfs", "raw/http"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    checkpoint_path = root / "acquisition_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, run_id, selected_plan)
    fetcher = SequentialFetcher(
        transport=transport,
        sleep_seconds=sleep_seconds,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    extract_pages = page_extractor or _extract_pages

    for plan_row in selected_plan.to_dict("records"):
        month = plan_row["report_month"]
        if checkpoint["months"].get(month, {}).get("terminal"):
            continue
        state = {
            "terminal": False,
            "phase": "started",
            "inventory": {},
            "captures": [],
            "transport": [],
            "claims": [],
            "events": [],
            "failures": [],
        }
        checkpoint["months"][month] = state
        _write_checkpoint(checkpoint_path, checkpoint)
        try:
            if (
                COPIED_FILE_ROUTE_ENABLED
                and month in OFFICIAL_JPX_COPIED_FILE_REGISTRY
            ):
                _acquire_copied_file_month(
                    root=root,
                    month=month,
                    state=state,
                    fetcher=fetcher,
                    extract_pages=extract_pages,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                )
            else:
                _acquire_month(
                    root=root,
                    month=month,
                    state=state,
                    fetcher=fetcher,
                    extract_pages=extract_pages,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                )
        except (AcquisitionError, ValueError) as exc:
            state["failures"].append(
                {
                    "report_month": month,
                    "stage": state.get("phase", "unknown"),
                    "failure_code": type(exc).__name__,
                    "detail": str(exc),
                    "quarantined": True,
                    "zero_inferred": False,
                }
            )
        state["terminal"] = True
        state["phase"] = "complete" if state["inventory"].get("pdf_sha256") else "failed"
        _write_checkpoint(checkpoint_path, checkpoint)

    manifest = _finalize_artifact(root, run_id, selected_plan, checkpoint)
    return WaybackLifecycleArtifacts(run_id, root, manifest_path, manifest)


def verify_wayback_lifecycle_artifact(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    issues: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        hashes_path = root / "artifact_hashes.json"
        if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("artifact_hash_inventory_mismatch")
        declared = json.loads(hashes_path.read_text(encoding="utf-8"))
        for relative, expected in declared.items():
            if _unsafe_relative(relative):
                issues.append(f"unsafe_artifact_path:{relative}")
                continue
            path = root / relative
            if not path.is_file() or _sha_path(path) != expected:
                issues.append(f"artifact_hash_mismatch:{relative}")
    except Exception as exc:
        issues.append(f"artifact_parse_error:{type(exc).__name__}")
    return issues


def _acquire_copied_file_month(
    *,
    root: Path,
    month: str,
    state: dict[str, Any],
    fetcher: SequentialFetcher,
    extract_pages: PageExtractor,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    route = OFFICIAL_JPX_COPIED_FILE_REGISTRY[month]
    parent_original = route["parent_index_url"]
    pdf_original = route["official_pdf_url"]
    state["inventory"].update(
        {
            "report_month": month,
            "route_mode": COPIED_FILE_ROUTE,
            "official_monthly_page_url": route["parent_link_url"],
            "preservation_transport": (
                "official JPX live PDF plus Internet Archive parent-index lineage"
            ),
            "original_source_publisher": "Tokyo Stock Exchange / JPX",
            "registry_entry_guessed": False,
            "zero_inferred": False,
            "parent_index_url": parent_original,
            "parent_link_url": route["parent_link_url"],
            "archived_parent_lineage_strength": route[
                "archived_parent_lineage_strength"
            ],
            "archived_parent_direct_pdf_link": (
                route["archived_parent_lineage_strength"]
                == DIRECT_ARCHIVED_PDF_LINEAGE
            ),
            "official_pdf_url": pdf_original,
        }
    )
    if not _official_jpx_copied_pdf_contract(pdf_original, month):
        raise AcquisitionError("registered official JPX PDF URL/month contract invalid")

    state["phase"] = "parent_index_cdx"
    parent_rows, parent_cdx_meta = _fetch_cdx(
        root, month, "parent_index", parent_original, fetcher
    )
    state["transport"].append(parent_cdx_meta)
    state["captures"].extend(_capture_rows(month, "parent_index", parent_rows))
    parent_capture = _select_registered_capture(
        parent_rows,
        parent_original,
        route["parent_capture_timestamp"],
    )
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "parent_index_capture"
    parent_wayback = wayback_raw_url(parent_capture["timestamp"], parent_capture["original"])
    parent_response, parent_attempts = fetcher.get(parent_wayback)
    parent_path = root / "raw" / "pages" / f"{month}.html"
    _write_once(parent_path, parent_response.body)
    state["transport"].append(
        _transport_row(
            month,
            "parent_index_capture",
            parent_wayback,
            parent_response,
            parent_attempts,
            parent_path,
            root,
        )
    )
    if parent_response.status != 200 or not parent_response.body:
        raise AcquisitionError(
            f"archived parent-index fetch status/bytes invalid: {parent_response.status}"
        )
    if parent_response.final_url != parent_wayback:
        raise AcquisitionError("archived parent-index response redirected away from lineage URL")
    parent_html = _decoded_entity_body(parent_response.body)
    if not _parent_index_proves_route(parent_html, parent_capture["original"], route):
        raise AcquisitionError("registered copied-file route lacks exact parent-index proof")
    state["inventory"].update(
        {
            "parent_index_cdx_timestamp": parent_capture["timestamp"],
            "parent_index_cdx_original": parent_capture["original"],
            "parent_index_cdx_statuscode": parent_capture["statuscode"],
            "parent_index_cdx_digest": parent_capture["digest"],
            "parent_index_cdx_length": parent_capture["length"],
            "parent_index_cdx_mimetype": parent_capture["mimetype"],
            "parent_index_wayback_url": parent_wayback,
            "parent_index_http_status": parent_response.status,
            "parent_index_http_headers": _headers_json(parent_response.headers),
            "parent_index_path": str(parent_path.relative_to(root)),
            "parent_index_response_sha256": _sha_bytes(parent_response.body),
            "parent_index_response_byte_length": len(parent_response.body),
            "parent_index_decoded_sha256": _sha_bytes(parent_html),
            "parent_index_decoded_byte_length": len(parent_html),
            "parent_index_exact_link_proved": True,
        }
    )
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "official_pdf_live"
    pdf_response, pdf_attempts = fetcher.get(pdf_original)
    pdf_path = root / "raw" / "pdfs" / f"{month}.pdf"
    _write_once(pdf_path, pdf_response.body)
    state["transport"].append(
        _transport_row(
            month,
            "official_pdf_live",
            pdf_original,
            pdf_response,
            pdf_attempts,
            pdf_path,
            root,
        )
    )
    if pdf_response.status != 200:
        raise AcquisitionError(f"official JPX PDF fetch status invalid: {pdf_response.status}")
    if pdf_response.final_url != pdf_original:
        raise AcquisitionError("official JPX PDF redirected away from the registered URL")
    if not pdf_response.body.startswith(b"%PDF-"):
        raise AcquisitionError("official JPX PDF magic invalid")
    content_type = _header_value(pdf_response.headers, "content-type").split(";", 1)[0]
    if content_type.casefold() != "application/pdf":
        raise AcquisitionError(f"official JPX PDF content type invalid: {content_type}")
    pdf_sha = _sha_bytes(pdf_response.body)
    state["inventory"].update(
        {
            "pdf_final_url": pdf_response.final_url,
            "pdf_http_status": pdf_response.status,
            "pdf_http_headers": _headers_json(pdf_response.headers),
            "pdf_path": str(pdf_path.relative_to(root)),
            "pdf_magic": pdf_response.body[:5].decode("ascii", errors="replace"),
            "pdf_sha256": pdf_sha,
            "pdf_byte_length": len(pdf_response.body),
            "pdf_content_type": content_type,
        }
    )
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "producer_parse"
    pages = extract_pages(pdf_response.body)
    if not _exact_report_month(pages, month):
        raise AcquisitionError("retained PDF does not prove the exact report month")
    _publish_event_claims(
        state,
        pages,
        month=month,
        pdf_original=pdf_original,
        pdf_path=pdf_path,
        root=root,
        pdf_sha=pdf_sha,
        archive_timestamp=parent_capture["timestamp"],
        lineage_url=parent_wayback,
        require_all_nonempty=True,
    )


def _acquire_month(
    *,
    root: Path,
    month: str,
    state: dict[str, Any],
    fetcher: SequentialFetcher,
    extract_pages: PageExtractor,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    page_original = official_monthly_page_url(month)
    state["inventory"].update(
        {
            "report_month": month,
            "route_mode": LEGACY_WAYBACK_ROUTE,
            "official_monthly_page_url": page_original,
            "preservation_transport": "Internet Archive Wayback Machine",
            "original_source_publisher": "Tokyo Stock Exchange / JPX",
            "zero_inferred": False,
        }
    )

    state["phase"] = "html_cdx"
    html_rows, html_cdx_meta = _fetch_cdx(root, month, "html", page_original, fetcher)
    state["transport"].append(html_cdx_meta)
    state["captures"].extend(_capture_rows(month, "html", html_rows))
    html_capture = _select_capture(html_rows, page_original, "html", month)
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "html_capture"
    page_wayback = wayback_raw_url(html_capture["timestamp"], html_capture["original"])
    page_response, page_attempts = fetcher.get(page_wayback)
    page_path = root / "raw" / "pages" / f"{month}.html"
    _write_once(page_path, page_response.body)
    page_meta = _transport_row(month, "html_capture", page_wayback, page_response, page_attempts, page_path, root)
    state["transport"].append(page_meta)
    if page_response.status != 200 or not page_response.body:
        raise AcquisitionError(f"archived HTML fetch status/bytes invalid: {page_response.status}")
    if not _same_monthly_page(html_capture["original"], month):
        raise AcquisitionError("HTML CDX original does not match exact monthly route")
    pdf_original = exact_lifecycle_pdf_url(page_response.body, html_capture["original"], month)
    state["inventory"].update(
        {
            "html_cdx_timestamp": html_capture["timestamp"],
            "html_cdx_original": html_capture["original"],
            "html_cdx_statuscode": html_capture["statuscode"],
            "html_cdx_digest": html_capture["digest"],
            "html_cdx_length": html_capture["length"],
            "html_cdx_mimetype": html_capture["mimetype"],
            "html_wayback_url": page_wayback,
            "html_http_status": page_response.status,
            "html_http_headers": _headers_json(page_response.headers),
            "html_path": str(page_path.relative_to(root)),
            "html_sha256": _sha_bytes(page_response.body),
            "html_byte_length": len(page_response.body),
            "official_pdf_url": pdf_original,
        }
    )
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "pdf_cdx"
    pdf_rows, pdf_cdx_meta = _fetch_cdx(root, month, "pdf", pdf_original, fetcher)
    state["transport"].append(pdf_cdx_meta)
    state["captures"].extend(_capture_rows(month, "pdf", pdf_rows))
    pdf_capture = _select_capture(pdf_rows, pdf_original, "pdf", month)
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "pdf_capture"
    pdf_wayback = wayback_raw_url(pdf_capture["timestamp"], pdf_capture["original"])
    pdf_response, pdf_attempts = fetcher.get(pdf_wayback)
    pdf_path = root / "raw" / "pdfs" / f"{month}.pdf"
    _write_once(pdf_path, pdf_response.body)
    pdf_meta = _transport_row(month, "pdf_capture", pdf_wayback, pdf_response, pdf_attempts, pdf_path, root)
    state["transport"].append(pdf_meta)
    if pdf_response.status != 200:
        raise AcquisitionError(f"archived PDF fetch status invalid: {pdf_response.status}")
    if not pdf_response.body.startswith(b"%PDF-"):
        raise AcquisitionError("archived PDF magic invalid")
    pdf_sha = _sha_bytes(pdf_response.body)
    state["inventory"].update(
        {
            "pdf_cdx_timestamp": pdf_capture["timestamp"],
            "pdf_cdx_original": pdf_capture["original"],
            "pdf_cdx_statuscode": pdf_capture["statuscode"],
            "pdf_cdx_digest": pdf_capture["digest"],
            "pdf_cdx_length": pdf_capture["length"],
            "pdf_cdx_mimetype": pdf_capture["mimetype"],
            "pdf_wayback_url": pdf_wayback,
            "pdf_http_status": pdf_response.status,
            "pdf_http_headers": _headers_json(pdf_response.headers),
            "pdf_path": str(pdf_path.relative_to(root)),
            "pdf_magic": pdf_response.body[:5].decode("ascii", errors="replace"),
            "pdf_sha256": pdf_sha,
            "pdf_byte_length": len(pdf_response.body),
        }
    )
    _checkpoint_phase(checkpoint_path, checkpoint)

    state["phase"] = "producer_parse"
    pages = extract_pages(pdf_response.body)
    if not _exact_report_month(pages, month):
        raise AcquisitionError("retained PDF does not prove the exact report month")
    _publish_event_claims(
        state,
        pages,
        month=month,
        pdf_original=pdf_original,
        pdf_path=pdf_path,
        root=root,
        pdf_sha=pdf_sha,
        archive_timestamp=pdf_capture["timestamp"],
        lineage_url=pdf_wayback,
    )


def _publish_event_claims(
    state: dict[str, Any],
    pages: list[str],
    *,
    month: str,
    pdf_original: str,
    pdf_path: Path,
    root: Path,
    pdf_sha: str,
    archive_timestamp: str,
    lineage_url: str,
    require_all_nonempty: bool = False,
) -> None:
    parsed_by_channel: dict[str, list[dict[str, Any]]] = {}
    for channel in CHANNELS:
        try:
            rows = _parse_retained_report_events(
                pages,
                report_month=month,
                channel=channel,
                source_url=pdf_original,
                source_path=str(pdf_path.relative_to(root)),
                source_sha256=pdf_sha,
            )
            parsed_by_channel[channel] = rows
        except ValueError as exc:
            state["failures"].append(
                {
                    "report_month": month,
                    "stage": f"producer_parse:{channel}",
                    "failure_code": "channel_parse_rejected",
                    "detail": str(exc),
                    "quarantined": True,
                    "zero_inferred": False,
                }
            )
            continue
        if not rows:
            state["failures"].append(
                {
                    "report_month": month,
                    "stage": f"producer_parse:{channel}",
                    "failure_code": "no_nonempty_exact_event_claim",
                    "detail": "empty section is retained as unresolved; zero is not inferred",
                    "quarantined": True,
                    "zero_inferred": False,
                }
            )
            continue
    if require_all_nonempty and not all(
        parsed_by_channel.get(channel) for channel in CHANNELS
    ):
        raise AcquisitionError(
            "copied-file route requires one explicit nonempty section for every channel"
        )
    for channel in CHANNELS:
        rows = parsed_by_channel.get(channel, [])
        if not rows:
            continue
        state["events"].extend(rows)
        state["claims"].append(
            {
                "report_month": month,
                "channel": channel,
                "claim_kind": "exact_nonempty_official_report_event_set",
                "event_count": len(rows),
                "zero_event_asserted": False,
                "official_source_url": pdf_original,
                "source_sha256": pdf_sha,
                "archive_timestamp": archive_timestamp,
                "wayback_url": lineage_url,
                "validation_status": "pending_independent_validation",
            }
        )


def _fetch_cdx(
    root: Path,
    month: str,
    kind: str,
    original_url: str,
    fetcher: SequentialFetcher,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    query_url = cdx_query_url(original_url)
    response, attempts = fetcher.get(query_url)
    path = root / "raw" / "cdx" / f"{month}_{kind}.json"
    _write_once(path, response.body)
    meta = _transport_row(month, f"{kind}_cdx", query_url, response, attempts, path, root)
    if response.status != 200:
        raise AcquisitionError(f"CDX {kind} query status invalid: {response.status}")
    rows = parse_cdx_payload(response.body)
    if not rows:
        raise AcquisitionError(f"CDX has no 200 {kind} capture")
    return rows, meta


def _parse_retained_report_events(
    pages: list[str],
    *,
    report_month: str,
    channel: str,
    source_url: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    return _parse_with_retained_instrument_taxonomy(
        pages,
        report_month=report_month,
        channel=channel,
        source_url=source_url,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _parse_with_retained_instrument_taxonomy(
    pages: list[str],
    *,
    report_month: str,
    channel: str,
    source_url: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Parse retained TSE rows without allowing one unknown class to drop a channel."""
    located = [
        (page_number, report_event_parser._canonical_text(line))
        for page_number, page in enumerate(pages, start=1)
        for line in page.splitlines()
        if report_event_parser._canonical_text(line)
    ]
    heading = report_event_parser.CHANNEL_HEADINGS[channel].casefold()
    starts = [index for index, (_, line) in enumerate(located) if line.casefold() == heading]
    if len(starts) != 1:
        raise ValueError(f"expected one explicit {channel} section in {report_month}, got {len(starts)}")

    rows: list[dict[str, Any]] = []
    current_class = ""
    current_class_quarantine = ""
    for page_number, line in located[starts[0] + 1 :]:
        if report_event_parser._section_boundary(line):
            break
        match = report_event_parser.ROW_PATTERN.search(line)
        if not match:
            continue
        prefix = line[: match.start()].strip()
        current_class, current_class_quarantine = _retained_instrument_class(
            prefix,
            current_class,
            current_class_quarantine,
        )
        event_date = report_event_parser._parse_date(match.group("date"))
        tail = match.group("tail").strip()
        code_match = report_event_parser.CODE_PREFIX.match(tail)
        if code_match:
            raw_code = code_match.group("code")
            body = tail[code_match.end() :].strip()
        elif tail.startswith("全銘柄"):
            raw_code = "全銘柄"
            body = tail[len("全銘柄") :].strip()
        else:
            raw_code = ""
            body = tail
        times = (
            report_event_parser._halt_times(body, event_date)
            if channel == HALT_CHANNEL
            else report_event_parser._blank_times()
        )
        issue_name = report_event_parser._issue_name(body, channel, times["range_start"])
        if not issue_name and raw_code:
            issue_name = raw_code
        if not issue_name:
            issue_name = report_event_parser._canonical_text(body) or line
            issue_quarantine = "quarantined_issue_name_unavailable"
        else:
            issue_quarantine = ""
        mapping_status = (
            current_class_quarantine
            or issue_quarantine
            or "pending_independent_source_backed_mapping"
        )
        timestamp_status = (
            "quarantined_negative_same_day_halt_interval"
            if _negative_same_day_halt_interval(times)
            else "pending_independent_source_backed_known_at"
        )
        ordinal = len(rows) + 1
        normalized_code = (
            unicodedata.normalize("NFKC", raw_code).strip().upper()
            if report_event_parser.CODE_PREFIX.fullmatch(raw_code)
            else ""
        )
        identity = {
            "version": report_event_parser.SOURCE_RECORD_VERSION,
            "report_month": report_month,
            "channel": channel,
            "source_sha256": source_sha256,
            "source_page": page_number,
            "source_row_ordinal": ordinal,
            "event_date": event_date.isoformat(),
            "raw_code": raw_code,
            "normalized_local_code": normalized_code,
            "instrument_class": current_class,
            "issue_name": issue_name,
            "halt_start_printed": times["start_printed"],
            "halt_end_printed": times["end_printed"],
        }
        record_sha = report_event_parser.source_record_sha256(identity)
        rows.append(
            {
                "event_id": hashlib.sha256(
                    f"{report_event_parser.EVENT_ID_VERSION}|{record_sha}".encode("utf-8")
                ).hexdigest()[:24],
                "report_month": report_month,
                "channel": channel,
                "event_type": report_event_parser.EVENT_TYPES[channel],
                "event_date": event_date.isoformat(),
                "raw_code": raw_code,
                "normalized_local_code": normalized_code,
                "instrument_class": current_class,
                "issue_name": issue_name,
                "halt_start_printed": times["start_printed"],
                "halt_end_printed": times["end_printed"],
                "halt_start_jst": times["start_jst"],
                "halt_end_jst": times["end_jst"],
                "source_url": source_url,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_page": page_number,
                "source_row_ordinal": ordinal,
                "source_record_text": line,
                "source_record_sha256": record_sha,
                "known_at": "",
                "ric": "",
                "mapping_status": mapping_status,
                "timestamp_status": timestamp_status,
                "admissible_to_shared_halt_history": False,
            }
        )
    return rows


def _retained_instrument_class(
    prefix: str,
    current_class: str,
    current_quarantine: str,
) -> tuple[str, str]:
    if not prefix:
        if current_class:
            return current_class, current_quarantine
        return MISSING_INSTRUMENT_CLASS, "quarantined_missing_instrument_class"
    retained_class = RETAINED_INSTRUMENT_CLASSES.get(_class_key(prefix))
    if retained_class:
        return retained_class, ""
    try:
        return report_event_parser._classify_instrument(prefix, current_class), ""
    except ValueError as exc:
        if "unrecognized instrument-class prefix:" not in str(exc):
            raise
        return UNKNOWN_INSTRUMENT_CLASS, "quarantined_unrecognized_instrument_class"


def _negative_same_day_halt_interval(times: Mapping[str, Any]) -> bool:
    start_value = str(times.get("start_jst", ""))
    end_value = str(times.get("end_jst", ""))
    if not start_value or not end_value:
        return False
    start = datetime.fromisoformat(start_value)
    end = datetime.fromisoformat(end_value)
    return start.date() == end.date() and end < start


def _class_key(value: str) -> str:
    result = report_event_parser._canonical_text(value).replace("（", "(").replace("）", ")")
    result = re.sub(r"\s*([()])\s*", r"\1", result)
    return result.casefold()


def _select_capture(
    rows: list[dict[str, str]], original_url: str, kind: str, month: str
) -> dict[str, str]:
    matching = []
    for row in rows:
        if row["statuscode"] != "200" or not _same_original(row["original"], original_url):
            continue
        if kind == "html" and not _same_monthly_page(row["original"], month):
            continue
        if kind == "pdf" and not _same_lifecycle_pdf(row["original"], month):
            continue
        matching.append(row)
    if not matching:
        raise AcquisitionError(f"no CDX capture satisfies exact {kind} original/month contract")
    return sorted(matching, key=lambda row: row["timestamp"])[0]


def _select_registered_capture(
    rows: list[dict[str, str]], original_url: str, timestamp: str
) -> dict[str, str]:
    matching = [
        row
        for row in rows
        if row["timestamp"] == timestamp
        and row["statuscode"] == "200"
        and _same_archive_original(row["original"], original_url)
        and row["mimetype"] == "text/html"
    ]
    if len(matching) != 1:
        raise AcquisitionError(
            "registered parent-index capture is absent or ambiguous: "
            f"{timestamp} ({len(matching)} matches)"
        )
    return matching[0]


def _parent_index_proves_route(
    html: bytes, page_url: str, route: Mapping[str, str]
) -> bool:
    expected = route["parent_link_url"]
    found: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        candidate = urllib.parse.urljoin(page_url, str(link["href"]).strip())
        parsed = urllib.parse.urlsplit(candidate)
        found.add(
            urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, "", "")
            )
        )
    return expected in found


def _decoded_entity_body(body: bytes) -> bytes:
    if body.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(body)
        except (OSError, EOFError) as exc:
            raise AcquisitionError("archived parent-index gzip body is invalid") from exc
    return body


def _capture_rows(month: str, kind: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"report_month": month, "resource_kind": kind, **row}
        for row in rows
    ]


def _transport_row(
    month: str,
    kind: str,
    request_url: str,
    response: FetchResponse,
    attempts: list[dict[str, Any]],
    raw_path: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        "report_month": month,
        "request_kind": kind,
        "request_url": request_url,
        "final_url": response.final_url,
        "http_status": int(response.status),
        "http_headers": _headers_json(response.headers),
        "attempts": json.dumps(attempts, sort_keys=True, separators=(",", ":")),
        "raw_path": str(raw_path.relative_to(root)),
        "byte_length": len(response.body),
        "sha256": _sha_bytes(response.body),
    }


def _finalize_artifact(
    root: Path,
    run_id: str,
    selected_plan: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    full_plan = build_acquisition_plan()
    selected = set(selected_plan["report_month"])
    plan = full_plan.copy()
    plan["selected_in_run"] = plan["report_month"].isin(selected)
    plan.to_csv(root / "acquisition_plan.csv", index=False)

    states = list(checkpoint["months"].values())
    inventory_rows = [state["inventory"] for state in states if state.get("inventory")]
    capture_rows = [row for state in states for row in state.get("captures", [])]
    transport_rows = [row for state in states for row in state.get("transport", [])]
    claim_rows = [row for state in states for row in state.get("claims", [])]
    event_rows = [row for state in states for row in state.get("events", [])]
    failure_rows = [row for state in states for row in state.get("failures", [])]

    inventory = pd.DataFrame(inventory_rows)
    captures = pd.DataFrame(capture_rows)
    transport = pd.DataFrame(transport_rows)
    claims = pd.DataFrame(claim_rows)
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    failures = pd.DataFrame(
        failure_rows,
        columns=["report_month", "stage", "failure_code", "detail", "quarantined", "zero_inferred"],
    )
    inventory.to_csv(root / "source_inventory.csv", index=False)
    captures.to_csv(root / "cdx_capture_inventory.csv", index=False)
    transport.to_csv(root / "transport_inventory.csv", index=False)
    claims.to_csv(root / "exact_nonempty_claims.csv", index=False)
    events.to_csv(root / "exact_events.csv", index=False)
    events.to_parquet(root / "exact_events.parquet", index=False)
    failures.to_csv(root / "failures_quarantine.csv", index=False)

    acquired = {
        row.get("report_month")
        for row in inventory_rows
        if row.get("pdf_sha256") and int(row.get("pdf_http_status", 0)) == 200
    }
    claim_index = {
        (row["report_month"], row["channel"]): int(row["event_count"])
        for row in claim_rows
    }
    completion_rows: list[dict[str, Any]] = []
    for row in full_plan.to_dict("records"):
        for channel in CHANNELS:
            key = (row["report_month"], channel)
            if key in claim_index:
                status = "exact_nonempty_claim_pending_independent_validation"
                action = "run independent byte/CDX/parser validation"
            elif row["report_month"] in acquired:
                status = "unresolved_no_nonempty_claim"
                action = "retain unresolved; do not infer zero from an empty or absent section"
            elif row["report_month"] in selected:
                status = "capture_or_parse_failed"
                action = "inspect quarantine and retry in a new bounded run"
            else:
                status = "pending_not_in_bounded_run"
                action = "resume in priority order with a bounded immutable chunk"
            completion_rows.append(
                {
                    "report_month": row["report_month"],
                    "channel": channel,
                    "priority": row["priority"],
                    "completion_status": status,
                    "exact_nonempty_event_count": claim_index.get(key, 0),
                    "zero_inferred": False,
                    "next_action": action,
                }
            )
    completion = pd.DataFrame(completion_rows)
    completion.to_csv(root / "completion_plan.csv", index=False)

    pending = full_plan.loc[~full_plan["report_month"].isin(acquired), "report_month"].tolist()
    next_resume = pending[0] if pending else ""
    route_modes = sorted(
        {
            str(row.get("route_mode", LEGACY_WAYBACK_ROUTE))
            for row in inventory_rows
            if row.get("report_month") in selected
        }
    )
    copied_only = route_modes == [COPIED_FILE_ROUTE]
    hashes = _artifact_hashes(root, exclude={"manifest.json", "artifact_hashes.json"})
    hashes_path = root / "artifact_hashes.json"
    _atomic_text(hashes_path, json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    manifest = {
        "artifact_type": "bounded_tse_wayback_lifecycle_source_acquisition",
        "version": VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_acquisition_complete" if len(acquired) == len(selected) else "bounded_acquisition_with_failures",
        "scope": {"start_month": START_MONTH, "end_month": END_MONTH},
        "selected_months": sorted(selected),
        "selected_month_count": len(selected),
        "acquired_exact_pdf_months": sorted(acquired),
        "acquired_exact_pdf_month_count": len(acquired),
        "exact_nonempty_claim_count": len(claim_rows),
        "exact_event_count": len(event_rows),
        "failure_count": len(failure_rows),
        "next_resume_point": next_resume,
        "route_modes": route_modes,
        "official_copied_file_registry_month_count": sum(
            month in OFFICIAL_JPX_COPIED_FILE_REGISTRY for month in selected
        ),
        "registry_entries_guessed": 0,
        "preservation_transport": (
            "Official JPX live PDF plus Internet Archive parent-index lineage"
            if copied_only
            else "Internet Archive Wayback Machine"
        ),
        "original_source": "Tokyo Stock Exchange / JPX official monthly PDF bytes",
        "source_contract": (
            "explicit official JPX copied-file registry -> exact parent-index CDX/Wayback "
            "lineage -> live official JPX PDF response metadata and bytes"
            if copied_only
            else "CDX exact monthly HTML -> id_ archived HTML -> exact idouYYMM PDF link "
            "-> PDF CDX -> id_ raw PDF bytes"
        ),
        "live_official_pdf_response_preserved": copied_only,
        "absence_is_zero": False,
        "zero_event_claims_published": 0,
        "refinitiv_used": False,
        "independent_validation_status": "not_run",
        "full_required_month_channel_completion": False,
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "fetch_policy": {"sequential": True, "bounded_retries": True},
        "resumable_checkpoint": "acquisition_checkpoint.json",
        "write_once_after_manifest": True,
        "artifact_hashes_sha256": _sha_path(hashes_path),
    }
    _atomic_text(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _load_checkpoint(path: Path, run_id: str, plan: pd.DataFrame) -> dict[str, Any]:
    plan_months = list(plan["report_month"])
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("run_id") != run_id or checkpoint.get("selected_months") != plan_months:
            raise ValueError("checkpoint run/month contract mismatch")
        return checkpoint
    checkpoint = {
        "run_id": run_id,
        "selected_months": plan_months,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "months": {},
    }
    _write_checkpoint(path, checkpoint)
    return checkpoint


def _checkpoint_phase(path: Path, checkpoint: dict[str, Any]) -> None:
    _write_checkpoint(path, checkpoint)


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_text(path, json.dumps(checkpoint, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _exact_report_month(pages: list[str], month: str) -> bool:
    period = _period(month)
    month_name = period.start_time.strftime("%b")
    title = "\n".join(pages)
    matches = re.findall(
        r"Changes\s+in\s+Listed\s+Companies\s+and\s+Issues.*?"
        r"\((?P<name>[A-Za-z]{3,9})\.?\s*(?P<year>20\d{2})\)",
        title,
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = {(name[:3].casefold(), year) for name, year in matches}
    return normalized == {(month_name.casefold(), str(period.year))}


def _extract_pages(body: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages]


def _same_monthly_page(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == TSE_HOST
        and parsed.path == f"/english/market/data/geppo/{_period(month).strftime('%Y%m')}.html"
        and not parsed.query
        and not parsed.fragment
    )


def _same_lifecycle_pdf(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    yymm = _period(month).strftime("%y%m")
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


def _official_jpx_copied_pdf_contract(url: str, month: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    yymm = _period(month).strftime("%y%m")
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


def _priority(month: str) -> int:
    year = _period(month).year
    if 2012 <= year <= 2015:
        return 1
    if 2006 <= year <= 2008:
        return 2
    if 2010 <= year <= 2011:
        return 3
    return 4


def _priority_reason(month: str) -> str:
    return {
        1: "first_priority_2012_2015",
        2: "second_priority_2006_2008",
        3: "third_priority_2010_2011_gaps",
        4: "remaining_bounded_scope",
    }[_priority(month)]


def _period(month: str) -> pd.Period:
    period = pd.Period(str(month), freq="M")
    if str(period) != str(month):
        raise ValueError(f"month must be exact YYYY-MM: {month}")
    return period


def _month(value: str) -> str:
    return str(_period(value))


def _safe_name(value: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _require_http_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be absolute HTTP(S)")


def _headers_json(headers: Mapping[str, str]) -> str:
    return json.dumps(
        {str(key).casefold(): str(value) for key, value in headers.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def _header_value(headers: Mapping[str, str], name: str) -> str:
    expected = name.casefold()
    return next(
        (str(value).strip() for key, value in headers.items() if str(key).casefold() == expected),
        "",
    )


def _write_once(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise AcquisitionError(f"write-once raw byte mismatch: {path.name}")
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(body)
    os.replace(temp, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _unsafe_relative(value: object) -> bool:
    key = str(value)
    path = Path(key)
    return not key or path.is_absolute() or ".." in path.parts


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
