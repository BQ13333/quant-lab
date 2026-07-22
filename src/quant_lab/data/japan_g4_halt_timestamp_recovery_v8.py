from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
OFFICIAL_HOST = "www.jpx.co.jp"
V8_RUN_ID = "JPN-G4-HALTS-200607-202606-V8"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PUBLICATION_PRECISION = "official_page_date_only"
KNOWN_AT_RULE = "date_only_conservative_upper_bound_next_midnight_jst"


@dataclass(frozen=True)
class HttpResponse:
    payload: bytes
    status: int
    final_url: str
    headers: Mapping[str, str]
    acquired_at_utc: str


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    local_code: str
    official_issue_name: str
    event_date: str
    required_fragments: tuple[str, ...]


@dataclass(frozen=True)
class PageSpec:
    page_id: str
    url: str
    publication_date: str
    heading_fragment: str
    events: tuple[EventSpec, ...]
    support_url: str = ""


@dataclass(frozen=True)
class JapanG4HaltTimestampRecoveryV8Artifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    claims_path: Path
    accepted_path: Path
    rejected_path: Path
    manifest: dict[str, Any]


PAGE_SPECS = (
    PageSpec(
        "detail_696",
        "https://www.jpx.co.jp/english/news/detail/detail_696.html",
        "2013-09-13",
        "Decision on Delisting, etc. -Simplex Holdings,Inc.-",
        (
            EventSpec(
                "jpx-report-d129a47fac42d1338b0cdf10",
                "4340",
                "Simplex Holdings,Inc.",
                "2013-10-15",
                (
                    "simplexholdingsincstockcode4340",
                    "delistingdateoct152013",
                ),
            ),
        ),
    ),
    PageSpec(
        "detail_933",
        "https://www.jpx.co.jp/english/news/detail/detail_933.html",
        "2014-01-21",
        "Decision on Delisting, etc. -TOTAL MEDICAL SERVICE CO.,LTD.",
        (
            EventSpec(
                "jpx-report-e05f17d2e939486e13fde78d",
                "3163",
                "TOTAL MEDICAL SERVICE CO.,LTD.",
                "2014-02-25",
                (
                    "totalmedicalservicecoltdstockcode3163",
                    "delistingdatefeb252014",
                ),
            ),
        ),
    ),
    PageSpec(
        "detail_1022",
        "https://www.jpx.co.jp/english/news/detail/detail_1022.html",
        "2014-03-05",
        "Calculation of TOPIX in relation to shares of Maruha Nichiro Corporation.",
        (
            EventSpec(
                "jpx-report-5211314f9df91d3614fbf85c",
                "1334",
                "Maruha Nichiro Holdings,Inc.",
                "2014-03-27",
                (
                    "maruhanichiroholdingsinc1334",
                    "sharestobedelistedonmarch27",
                ),
            ),
        ),
    ),
    PageSpec(
        "detail_1207",
        "https://www.jpx.co.jp/english/news/detail/detail_1207.html",
        "2014-07-24",
        "Decision on Delisting, etc. -Minerva Holdings CO.,LTD.-",
        (
            EventSpec(
                "jpx-report-c41cf5f4ef754f5adfede406",
                "3090",
                "Minerva Holdings CO.,LTD.",
                "2014-08-25",
                (
                    "minervaholdingscoltdstockcode3090",
                    "delistingdateaug252014",
                ),
            ),
        ),
    ),
    PageSpec(
        "20160401-11",
        "https://www.jpx.co.jp/english/news/1021/20160401-11.html",
        "2016-04-01",
        "Decision on Delisting, etc.: MISAWA HABITA&CO.,LTD.",
        (
            EventSpec(
                "jpx-report-bd31eeba13db47900ee0cf58",
                "1427",
                "MISAWA HABITA&CO.,LTD.",
                "2016-04-15",
                (
                    "misawahabitaandcoltdstockcode1427",
                    "delistingdateapr152016",
                ),
            ),
        ),
    ),
    PageSpec(
        "20160630-02",
        "https://www.jpx.co.jp/english/news/1021/20160630-02.html",
        "2016-06-30",
        "Decision on Delisting, etc.: MAG NET HOLDINGS, INC.",
        (
            EventSpec(
                "jpx-report-40ff9f970778ba0e64673c62",
                "8073",
                "MAG NET HOLDINGS,INC.",
                "2016-08-01",
                (
                    "magnetholdingsincstockcode8073",
                    "delistingdateaug012016",
                ),
            ),
        ),
    ),
    PageSpec(
        "20180905-02",
        "https://www.jpx.co.jp/english/news/1044/20180905-02.html",
        "2018-09-05",
        "Calculation of TOPIX in relation to shares of Daishi Hokuetsu Financial Group,Inc.",
        (
            EventSpec(
                "jpx-delist-f21453f0f4aff06cbceedf0d",
                "8324",
                "The Daishi Bank,Ltd.",
                "2018-09-26",
                ("thedaishibankltd8324", "tobedelistedonseptember262018"),
            ),
            EventSpec(
                "jpx-delist-8f88eba45e321e35ff26a1c7",
                "8325",
                "The Hokuetsu Bank,Ltd.",
                "2018-09-26",
                ("thehokuetsubankltd8325", "tobedelistedonseptember262018"),
            ),
        ),
        "https://www.jpx.co.jp/english/news/1044/b5b4pj000002ai5a-att/data_e.pdf",
    ),
    PageSpec(
        "20190205-01",
        "https://www.jpx.co.jp/english/news/1044/20190205-01.html",
        "2019-02-05",
        "Calculation of TOPIX in relation to shares of HITO-Communications Holdings,Inc",
        (
            EventSpec(
                "jpx-delist-3197ccfb203a833a61027944",
                "3654",
                "HITO-Communications,Inc.",
                "2019-02-26",
                ("hitocommunicationsinc3654", "tobedelistedonfebruary262019"),
            ),
        ),
        "https://www.jpx.co.jp/english/news/1044/b5b4pj000002on5e-att/hp01_e.pdf",
    ),
    PageSpec(
        "20260603-01",
        "https://www.jpx.co.jp/english/news/6030/20260603-01.html",
        "2026-06-03",
        "Calculation of TOPIX in relation to shares of giftee Group,Inc.",
        (
            EventSpec(
                "jpx-delist-f81fc020ff0855d311c690e3",
                "4449",
                "giftee Inc.",
                "2026-06-29",
                ("gifteeinc4449", "tobedelistedonjune292026"),
            ),
        ),
        "https://www.jpx.co.jp/english/news/6030/vk0khi000001mqgh-att/data_e.pdf",
    ),
)

CLAIM_COLUMNS = (
    "ordinal",
    "event_id",
    "local_code",
    "v8_issue_name",
    "official_issue_name",
    "event_date",
    "publication_date",
    "publication_time_jst",
    "publication_precision",
    "known_at_utc",
    "known_at_rule",
    "page_id",
    "page_url",
    "page_source_path",
    "page_sha256",
    "support_url",
    "support_source_path",
    "support_sha256",
    "status",
    "evidence_checks",
    "rejection_reason",
)
SOURCE_COLUMNS = (
    "source_id",
    "page_id",
    "source_role",
    "event_ids",
    "requested_url",
    "final_url",
    "http_status",
    "content_type",
    "content_length_header",
    "response_date",
    "response_last_modified",
    "response_etag",
    "acquired_at_utc",
    "raw_path",
    "sha256",
    "bytes",
    "metadata_path",
    "metadata_sha256",
    "accepted",
    "quarantine_reason",
)
QUARANTINE_ROW_COLUMNS = (
    "event_id",
    "local_code",
    "event_date",
    "page_id",
    "reason",
)

Fetcher = Callable[[str, float], HttpResponse]


def produce_japan_g4_halt_timestamp_recovery_v8(
    *,
    run_id: str,
    events_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_timestamp_recovery",
    fetcher: Fetcher | None = None,
    request_timeout: float = 30.0,
) -> JapanG4HaltTimestampRecoveryV8Artifacts:
    """Recover the sealed 10-event V8 publication-date denominator from JPX bytes."""
    _validate_run_id(run_id)
    event_source = Path(events_path).resolve()
    producer_manifest_path = event_source.parent / "manifest.json"
    producer_manifest = _read_json(producer_manifest_path)
    _validate_v8_input(event_source, producer_manifest_path, producer_manifest)

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw_dir = root / "raw"
    quarantine_raw_dir = root / "quarantine" / "raw"
    metadata_dir = raw_dir / "metadata"
    quarantine_metadata_dir = root / "quarantine" / "metadata"

    v8_events = pd.read_parquet(event_source).fillna("")
    target_rows = _target_v8_rows(v8_events)
    input_rows_path = root / "input_event_rows.csv"
    _atomic_csv(input_rows_path, target_rows)

    fetch = fetcher or _fetch
    source_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, str]] = []
    ordinal = 0
    for page in PAGE_SPECS:
        page_response = fetch(page.url, request_timeout)
        support_response = fetch(page.support_url, request_timeout) if page.support_url else None
        bundle_checks, evidence_text = _bundle_checks(page, page_response, support_response)
        bundle_failures = [name for name, passed in bundle_checks.items() if not passed]
        bundle_accepted = not bundle_failures
        bundle_reason = "|".join(bundle_failures)

        page_source = _retain_response(
            root=root,
            accepted=bundle_accepted,
            raw_dir=raw_dir,
            quarantine_raw_dir=quarantine_raw_dir,
            metadata_dir=metadata_dir,
            quarantine_metadata_dir=quarantine_metadata_dir,
            source_id=f"{page.page_id}-page",
            page_id=page.page_id,
            role="page_html",
            event_ids="|".join(event.event_id for event in page.events),
            requested_url=page.url,
            response=page_response,
            extension=".html",
            quarantine_reason=bundle_reason,
        )
        source_rows.append(page_source)
        support_source: dict[str, Any] | None = None
        if support_response is not None:
            support_source = _retain_response(
                root=root,
                accepted=bundle_accepted,
                raw_dir=raw_dir,
                quarantine_raw_dir=quarantine_raw_dir,
                metadata_dir=metadata_dir,
                quarantine_metadata_dir=quarantine_metadata_dir,
                source_id=f"{page.page_id}-support",
                page_id=page.page_id,
                role="linked_official_pdf",
                event_ids="|".join(event.event_id for event in page.events),
                requested_url=page.support_url,
                response=support_response,
                extension=".pdf",
                quarantine_reason=bundle_reason,
            )
            source_rows.append(support_source)

        for event in page.events:
            input_match, v8_row, input_reason = _match_v8_event(target_rows, event)
            event_checks = {
                "source_bundle": bundle_accepted,
                "v8_event_row": input_match,
                "official_name": _canonical(event.official_issue_name) in evidence_text,
                "official_code": event.local_code in evidence_text,
                "official_event_date": all(
                    fragment in evidence_text for fragment in event.required_fragments
                ),
                "known_at_not_after_event": (
                    _known_at_utc(page.publication_date).date()
                    <= date.fromisoformat(event.event_date)
                ),
            }
            failures = [name for name, passed in event_checks.items() if not passed]
            if input_reason:
                failures.append(input_reason)
            accepted = not failures
            if not accepted:
                quarantine_rows.append(
                    {
                        "event_id": event.event_id,
                        "local_code": event.local_code,
                        "event_date": event.event_date,
                        "page_id": page.page_id,
                        "reason": "|".join(dict.fromkeys(failures)),
                    }
                )
            claim_rows.append(
                {
                    "ordinal": ordinal,
                    "event_id": event.event_id,
                    "local_code": event.local_code,
                    "v8_issue_name": v8_row.get("issue_name", ""),
                    "official_issue_name": event.official_issue_name,
                    "event_date": event.event_date,
                    "publication_date": page.publication_date,
                    "publication_time_jst": "",
                    "publication_precision": PUBLICATION_PRECISION,
                    "known_at_utc": _known_at_utc(page.publication_date).isoformat() if accepted else "",
                    "known_at_rule": KNOWN_AT_RULE if accepted else "",
                    "page_id": page.page_id,
                    "page_url": page.url,
                    "page_source_path": page_source["raw_path"],
                    "page_sha256": page_source["sha256"],
                    "support_url": page.support_url,
                    "support_source_path": support_source["raw_path"] if support_source else "",
                    "support_sha256": support_source["sha256"] if support_source else "",
                    "status": "accepted" if accepted else "rejected",
                    "evidence_checks": "|".join(
                        name for name, passed in event_checks.items() if passed
                    ),
                    "rejection_reason": "|".join(dict.fromkeys(failures)),
                }
            )
            ordinal += 1

    claims = pd.DataFrame(claim_rows, columns=CLAIM_COLUMNS)
    accepted = claims.loc[claims["status"].eq("accepted")].copy()
    rejected = claims.loc[claims["status"].eq("rejected")].copy()
    sources = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
    quarantined = pd.DataFrame(quarantine_rows, columns=QUARANTINE_ROW_COLUMNS)

    claims_path = root / "timestamp_recovery.csv"
    accepted_path = root / "accepted_timestamp_recovery.csv"
    rejected_path = root / "rejected_timestamp_recovery.csv"
    source_manifest_path = root / "official_source_manifest.csv"
    quarantine_rows_path = root / "quarantine" / "rows.csv"
    summary_path = root / "summary.md"
    _atomic_csv(claims_path, claims)
    _atomic_csv(accepted_path, accepted)
    _atomic_csv(rejected_path, rejected)
    _atomic_csv(source_manifest_path, sources)
    _atomic_csv(quarantine_rows_path, quarantined)
    _atomic_text(summary_path, _summary(run_id, claims, sources))

    outputs = (
        input_rows_path,
        claims_path,
        accepted_path,
        rejected_path,
        source_manifest_path,
        quarantine_rows_path,
        summary_path,
    )
    manifest = {
        "schema_version": "1.0-v8-bounded-immutable",
        "run_id": run_id,
        "component": "japan_g4_halt_v8_official_publication_timestamp_recovery",
        "status": "complete" if len(rejected) == 0 else "partial_fail_closed",
        "source_policy": {
            "official_jpx_only": True,
            "exact_page_count": len(PAGE_SPECS),
            "http_response_date_used_as_publication_time": False,
            "http_last_modified_used_as_publication_time": False,
            "archive_capture_time_used": False,
            "report_month_used": False,
            "same_code_prior_observation_used": False,
            "inferred_publication_time_used": False,
            "publication_time_claim": "not stated in retained official page bytes",
            "known_at_rule": KNOWN_AT_RULE,
        },
        "counts": {
            "target_events": len(claims),
            "exact_pages": len(PAGE_SPECS),
            "linked_support_documents": sum(bool(page.support_url) for page in PAGE_SPECS),
            "retained_source_responses": len(sources),
            "accepted_events": len(accepted),
            "rejected_events": len(rejected),
            "accepted_source_responses": int(sources["accepted"].eq(True).sum()),
            "quarantined_source_responses": int(sources["accepted"].eq(False).sum()),
            "publication_time_stated_events": 0,
            "publication_date_only_events": len(claims),
        },
        "inputs": {
            "v8_manifest": _descriptor(producer_manifest_path),
            "v8_events": _descriptor(event_source),
        },
        "outputs": {path.relative_to(root).as_posix(): _descriptor(path) for path in outputs},
        "raw_source_sha256": {
            row["source_id"]: row["sha256"] for row in sources.to_dict("records")
        },
        "response_metadata_sha256": {
            row["source_id"]: row["metadata_sha256"]
            for row in sources.to_dict("records")
        },
        "refinitiv_calls": False,
        "primary_halt_history_modified": False,
        "independent_validation_required": True,
        "promotion_eligible": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4HaltTimestampRecoveryV8Artifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=manifest_path,
        claims_path=claims_path,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        manifest=manifest,
    )


def _validate_v8_input(
    events_path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> None:
    if not events_path.is_file() or not manifest_path.is_file():
        raise ValueError("V8 halt events and manifest must exist")
    if not (
        manifest.get("run_id") == V8_RUN_ID
        and manifest.get("component") == "halt_suspension_events"
        and manifest.get("artifact_type") == "japan_g4_halt_suspension_history_producer"
    ):
        raise ValueError("events are not the bounded V8 halt source")
    expected = manifest.get("output_sha256", {}).get(events_path.name)
    if not expected or _sha256(events_path) != expected:
        raise ValueError("V8 halt event hash mismatch")


def _target_v8_rows(events: pd.DataFrame) -> pd.DataFrame:
    expected = {event.event_id: event for page in PAGE_SPECS for event in page.events}
    required = {"event_id", "event_type", "local_code", "issue_name", "event_date"}
    if not required.issubset(events.columns):
        raise ValueError("V8 event input is missing required columns")
    rows = events.loc[events["event_id"].isin(expected)].copy()
    if len(rows) != len(expected) or rows["event_id"].duplicated().any():
        raise ValueError("V8 target event denominator is not one-to-one")
    rows = rows[list(required)].sort_values("event_id").reset_index(drop=True)
    return rows


def _match_v8_event(
    target_rows: pd.DataFrame, event: EventSpec
) -> tuple[bool, dict[str, str], str]:
    rows = target_rows.loc[target_rows["event_id"].eq(event.event_id)]
    if len(rows) != 1:
        return False, {}, "v8_event_id_not_unique"
    row = {key: str(value) for key, value in rows.iloc[0].to_dict().items()}
    checks = {
        "event_type": row["event_type"] == "delisting_non_trading_interval",
        "local_code": row["local_code"] == event.local_code,
        "event_date": row["event_date"] == event.event_date,
        "issue_name": _canonical(event.official_issue_name) in _canonical(row["issue_name"]),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, row, "v8_" + "_".join(failures) if failures else ""


def _bundle_checks(
    spec: PageSpec,
    page_response: HttpResponse,
    support_response: HttpResponse | None,
) -> tuple[dict[str, bool], str]:
    page_text = _html_text(page_response.payload)
    canonical_page = _canonical(page_text)
    checks = {
        "page_status_200": page_response.status == 200,
        "page_exact_final_url": page_response.final_url == spec.url,
        "page_content_type_html": _content_type(page_response) in {
            "text/html",
            "application/xhtml+xml",
        },
        "page_publication_date": _publication_date_visible(
            page_text, spec.publication_date
        ),
        "page_heading": _canonical(spec.heading_fragment) in canonical_page,
    }
    evidence_text = canonical_page
    if spec.support_url:
        linked_urls = {
            urllib.parse.urljoin(spec.url, anchor.get("href", ""))
            for anchor in BeautifulSoup(page_response.payload, "html.parser").find_all(
                "a", href=True
            )
        }
        checks["support_exact_link_in_page"] = spec.support_url in linked_urls
        checks["support_response_present"] = support_response is not None
        if support_response is not None:
            checks.update(
                {
                    "support_status_200": support_response.status == 200,
                    "support_exact_final_url": support_response.final_url == spec.support_url,
                    "support_content_type_pdf": _content_type(support_response)
                    == "application/pdf",
                }
            )
            try:
                evidence_text = _canonical(_pdf_text(support_response.payload))
                checks["support_pdf_parseable"] = bool(evidence_text)
            except Exception:
                evidence_text = ""
                checks["support_pdf_parseable"] = False
    for event in spec.events:
        checks[f"event_{event.local_code}_evidence"] = all(
            fragment in evidence_text for fragment in event.required_fragments
        )
    return checks, evidence_text


def _retain_response(
    *,
    root: Path,
    accepted: bool,
    raw_dir: Path,
    quarantine_raw_dir: Path,
    metadata_dir: Path,
    quarantine_metadata_dir: Path,
    source_id: str,
    page_id: str,
    role: str,
    event_ids: str,
    requested_url: str,
    response: HttpResponse,
    extension: str,
    quarantine_reason: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(response.payload).hexdigest()
    filename = f"{source_id}-{digest[:16]}{extension}"
    byte_dir = raw_dir if accepted else quarantine_raw_dir
    meta_dir = metadata_dir if accepted else quarantine_metadata_dir
    raw_path = byte_dir / filename
    metadata_path = meta_dir / f"{source_id}-{digest[:16]}.json"
    _atomic_bytes(raw_path, response.payload)
    headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    metadata = {
        "source_id": source_id,
        "requested_url": requested_url,
        "final_url": response.final_url,
        "http_status": response.status,
        "headers": dict(sorted(headers.items())),
        "acquired_at_utc": response.acquired_at_utc,
        "payload_sha256": digest,
        "payload_bytes": len(response.payload),
        "accepted": accepted,
        "quarantine_reason": quarantine_reason,
        "publication_timestamp_input": False,
    }
    _atomic_json(metadata_path, metadata)
    return {
        "source_id": source_id,
        "page_id": page_id,
        "source_role": role,
        "event_ids": event_ids,
        "requested_url": requested_url,
        "final_url": response.final_url,
        "http_status": response.status,
        "content_type": headers.get("content-type", ""),
        "content_length_header": headers.get("content-length", ""),
        "response_date": headers.get("date", ""),
        "response_last_modified": headers.get("last-modified", ""),
        "response_etag": headers.get("etag", ""),
        "acquired_at_utc": response.acquired_at_utc,
        "raw_path": str(raw_path),
        "sha256": digest,
        "bytes": len(response.payload),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "accepted": accepted,
        "quarantine_reason": quarantine_reason,
    }


def _publication_date_visible(text: str, value: str) -> bool:
    publication = date.fromisoformat(value)
    candidates = {
        publication.strftime("%b. %d, %Y"),
        publication.strftime("%b %d, %Y"),
        publication.strftime("%B %d, %Y"),
    }
    return any(candidate in text for candidate in candidates)


def _known_at_utc(publication_date: str) -> datetime:
    publication = date.fromisoformat(publication_date)
    value = datetime.combine(publication + timedelta(days=1), time.min, JST)
    return value.astimezone(UTC)


def _html_text(payload: bytes) -> str:
    return " ".join(BeautifulSoup(payload, "html.parser").stripped_strings)


def _pdf_text(payload: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)


def _canonical(value: str) -> str:
    clean = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    clean = clean.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", clean)


def _content_type(response: HttpResponse) -> str:
    headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _fetch(url: str, timeout: float) -> HttpResponse:
    _require_official_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quant-lab-japan-g4-v8-publication-recovery/1.0",
            "Accept": "text/html,application/xhtml+xml,application/pdf",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        result = HttpResponse(
            payload=payload,
            status=int(response.status),
            final_url=response.geturl(),
            headers=dict(response.headers.items()),
            acquired_at_utc=datetime.now(UTC).isoformat(),
        )
    if len(result.payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"JPX response exceeds byte limit: {url}")
    return result


def _require_official_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if not (
        parsed.scheme == "https"
        and parsed.hostname == OFFICIAL_HOST
        and parsed.path.startswith("/english/news/")
        and not parsed.query
        and not parsed.fragment
    ):
        raise ValueError(f"non-official or unsupported JPX URL: {url}")


def _summary(run_id: str, claims: pd.DataFrame, sources: pd.DataFrame) -> str:
    accepted = int(claims["status"].eq("accepted").sum())
    rejected = len(claims) - accepted
    quarantined = int(sources["accepted"].eq(False).sum())
    return (
        f"# Japan G4 Halt V8 Official Publication Timestamp Recovery: {run_id}\n\n"
        f"- Exact target events: {len(claims)}\n"
        f"- Exact JPX pages: {len(PAGE_SPECS)}\n"
        f"- Accepted event claims: {accepted}\n"
        f"- Rejected event claims: {rejected}\n"
        f"- Retained source responses: {len(sources)}\n"
        f"- Quarantined source responses: {quarantined}\n"
        "- Publication time: not stated in the retained official page bytes.\n"
        "- Known-at: first instant after the official publication date in JST; this is a conservative date-only upper bound, not an asserted publication time.\n"
        "- HTTP response Date, Last-Modified, acquisition time, archive capture time, report month, and same-code observations are not timestamp evidence.\n"
        "- Refinitiv calls: none.\n"
    )


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)
