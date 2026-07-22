from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
OFFICIAL_HOST = "www.jpx.co.jp"
V14_RUN_ID = "JPN-G4-HALTS-200607-202606-V14"
COMPONENT = "japan_g4_halt_v14_six_official_publication_timestamp_recovery"
PUBLICATION_PRECISION = "official_page_date_only"
KNOWN_AT_RULE = "date_only_conservative_upper_bound_next_midnight_jst"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class HttpResponse:
    payload: bytes
    status: int
    final_url: str
    headers: Mapping[str, str]
    acquired_at_utc: str


@dataclass(frozen=True)
class EventSpec:
    page_id: str
    event_id: str
    local_code: str
    issue_name: str
    event_date: str
    source_record_id: str
    event_source_url: str
    event_source_sha256: str
    notice_url: str
    publication_date: str
    required_fragments: tuple[str, ...]


@dataclass(frozen=True)
class JapanG4OfficialPublicationTimestampV14Artifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    claims_path: Path
    accepted_path: Path
    manifest: dict[str, Any]


EVENT_SPECS = (
    EventSpec(
        "detail_1789",
        "jpx-report-729b287a25a2fa47697f890e",
        "6784",
        "プラネックスホールディング PLANEX HOLDING INC.",
        "2014-12-12",
        "baf898d9a50e814b83579dd280f8109de5d3aa16e292855d62e18cc33747478f",
        "https://www.jpx.co.jp/files/tse/english/market/data/geppo/"
        "b7gje6000002aeyz-att/17_idou1412.pdf",
        "34ed4610f56de4e8275f861d98fba09052820d69750c329452e5f86c4936ccdc",
        "https://www.jpx.co.jp/news/detail/detail_1789.html",
        "2014-11-19",
        ("2014/11/19", "プラネックスホールディング株式会社", "コード:6784", "平成26年12月12日"),
    ),
    EventSpec(
        "detail_1616",
        "jpx-report-a0c03f3de0132d297e33660f",
        "2712",
        "スターバックス コーヒー ジャパン Starbucks Coffee Japan,Ltd.",
        "2015-03-23",
        "7e46e2356741d8e6d4be402f62691cceafd5731131e52a6b062b6846ddeef20c",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000idmr-att/17_idou1503.pdf",
        "df7b071480630fb57c0b621aeebb1e8abaf3efbc1cf483100e299390e8b30131",
        "https://www.jpx.co.jp/english/news/detail/detail_1616.html",
        "2015-02-20",
        ("Feb. 20, 2015", "Starbucks Coffee Japan, Ltd. stock", "Code: 2712", "Delisting Date Mar. 23, 2015"),
    ),
    EventSpec(
        "20150430-12",
        "jpx-report-c0565af152c4387ecd7c2457",
        "9963",
        "江守グループホールディングス EMORI GROUP HOLDINGS CO.,LTD.",
        "2015-05-31",
        "afb4ed11cead13eb69d2d5e4a51956fd04bcb14b4fe9cd3340b5cfa98e630c84",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000m7co-att/17_idou1505.pdf",
        "559a1136d5096c20b67cbfb57a1ac0430e9969cbc402700219de150d3555871d",
        "https://www.jpx.co.jp/news/1021/20150430-12.html",
        "2015-04-30",
        ("2015/04/30", "江守グループホールディングス株式会社", "コード:9963", "2015年5月31日"),
    ),
    EventSpec(
        "20150630-15",
        "jpx-report-9ad0ce8ccb9c6a57f7a8ce39",
        "7708",
        "石山 Gateway Holdings Ishiyama Gateway Holdings Inc.",
        "2015-08-01",
        "c2fd6aca2fca864c12a3bcaf03c34c6f42376bfe74289a6e72b9b8abee0cf81e",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000qc9y-att/17_idou1508.pdf",
        "16ffb220908d6e06a40f2eee41f95e0a9a6dc59ef73731d75f6352480efa93ab",
        "https://www.jpx.co.jp/news/1021/20150630-15.html",
        "2015-06-30",
        ("2015/06/30", "石山Gateway Holdings株式会社", "コード:7708", "2015年8月1日"),
    ),
    EventSpec(
        "20150811-14",
        "jpx-report-bcc8fea64fb029baf0251360",
        "3587",
        "グローバルアジアホールディングス Global Asia Holdings Inc.",
        "2015-09-12",
        "cf0ee6a81465427c4f11070c83af7fd8adad720de7614e9138657892329fb123",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000rgwu-att/17_idou1509.pdf",
        "a380cdf4d6900fb9f276305c16a217274cdf431ad90ee514b245f39be00d8edb",
        "https://www.jpx.co.jp/news/1021/20150811-14.html",
        "2015-08-11",
        ("2015/08/11", "グローバルアジアホールディングス株式会社", "コード:3587", "2015年9月12日"),
    ),
    EventSpec(
        "20151030-13",
        "jpx-report-570db7455dcb723f56c9febf",
        "4798",
        "エル・シー・エーホールディングス LCA Holdings Corporation",
        "2015-12-01",
        "cb50a33bb04b362af9190fb2725503c396881d737433a066a702d259843bc523",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
        "b5b4pj000000vj5c-att/17_idou1512.pdf",
        "a8ebffef8353a15264afbe49ddb7a8afdd34c9799cac6ca52dcefea221723b3d",
        "https://www.jpx.co.jp/news/1021/20151030-13.html",
        "2015-10-30",
        ("2015/10/30", "株式会社エル・シー・エーホールディングス", "コード:4798", "2015年12月1日"),
    ),
)

CLAIM_COLUMNS = (
    "ordinal", "event_id", "event_type", "local_code", "v14_issue_name",
    "event_date", "event_source_id", "source_record_id", "event_source_url",
    "event_source_sha256", "publication_date", "publication_time_jst",
    "publication_precision", "known_at_utc", "known_at_rule", "page_id",
    "source_url", "source_path", "source_sha256", "status", "evidence_checks",
    "rejection_reason",
)
SOURCE_COLUMNS = (
    "source_id", "page_id", "event_id", "requested_url", "final_url", "http_status",
    "content_type", "content_length_header", "response_date", "response_last_modified",
    "response_etag", "acquired_at_utc", "raw_path", "sha256", "bytes",
    "metadata_path", "metadata_sha256", "accepted", "quarantine_reason",
)
Fetcher = Callable[[str, float], HttpResponse]


def produce_japan_g4_halt_official_publication_timestamp_v14(
    *,
    run_id: str,
    events_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_timestamp_recovery",
    fetcher: Fetcher | None = None,
    request_timeout: float = 30.0,
) -> JapanG4OfficialPublicationTimestampV14Artifacts:
    """Build a fail-closed official-JPX timestamp packet for six exact V14 events."""
    _validate_run_id(run_id)
    event_source = Path(events_path).resolve()
    source_manifest_path = event_source.parent / "manifest.json"
    source_manifest = _json(source_manifest_path)
    _validate_v14_input(event_source, source_manifest_path, source_manifest)

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    events = pd.read_parquet(event_source).fillna("")
    target_rows = _target_rows(events)
    input_rows_path = root / "input_event_rows.csv"
    _atomic_csv(input_rows_path, target_rows)

    fetch = fetcher or _fetch
    claims: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    quarantine: list[dict[str, str]] = []
    for ordinal, spec in enumerate(EVENT_SPECS):
        response = fetch(spec.notice_url, request_timeout)
        text = " ".join(BeautifulSoup(response.payload, "html.parser").stripped_strings)
        canonical_text = _canonical(text)
        row = target_rows.loc[target_rows["event_id"].eq(spec.event_id)].iloc[0]
        row_checks = _event_row_checks(row, spec)
        source_checks = {
            "status_200": response.status == 200,
            "exact_final_url": response.final_url == spec.notice_url,
            "official_final_domain": _is_exact_official_url(response.final_url),
            "content_type_html": _content_type(response) in {"text/html", "application/xhtml+xml"},
            "nonempty_html": len(response.payload) > 0 and bool(canonical_text),
            "publication_date": _canonical(spec.required_fragments[0]) in canonical_text,
            "company": _canonical(spec.required_fragments[1]) in canonical_text,
            "code": _canonical(spec.required_fragments[2]) in canonical_text,
            "delisting_date": _canonical(spec.required_fragments[3]) in canonical_text,
        }
        failures = [name for name, passed in {**row_checks, **source_checks}.items() if not passed]
        accepted = not failures
        source = _retain_response(root, spec, response, accepted, "|".join(failures))
        sources.append(source)
        if not accepted:
            quarantine.append({
                "event_id": spec.event_id,
                "local_code": spec.local_code,
                "page_id": spec.page_id,
                "reason": "|".join(failures),
            })
        claims.append({
            "ordinal": ordinal,
            "event_id": spec.event_id,
            "event_type": str(row["event_type"]),
            "local_code": spec.local_code,
            "v14_issue_name": str(row["issue_name"]),
            "event_date": spec.event_date,
            "event_source_id": str(row["source_id"]),
            "source_record_id": spec.source_record_id,
            "event_source_url": spec.event_source_url,
            "event_source_sha256": spec.event_source_sha256,
            "publication_date": spec.publication_date,
            "publication_time_jst": "",
            "publication_precision": PUBLICATION_PRECISION,
            "known_at_utc": _known_at(spec.publication_date) if accepted else "",
            "known_at_rule": KNOWN_AT_RULE if accepted else "",
            "page_id": spec.page_id,
            "source_url": spec.notice_url,
            "source_path": source["raw_path"],
            "source_sha256": source["sha256"],
            "status": "accepted" if accepted else "rejected",
            "evidence_checks": "|".join(
                name for name, passed in {**row_checks, **source_checks}.items() if passed
            ),
            "rejection_reason": "|".join(failures),
        })

    claims_frame = pd.DataFrame(claims, columns=CLAIM_COLUMNS)
    accepted_frame = claims_frame.loc[claims_frame["status"].eq("accepted")].copy()
    rejected_frame = claims_frame.loc[claims_frame["status"].eq("rejected")].copy()
    sources_frame = pd.DataFrame(sources, columns=SOURCE_COLUMNS)
    quarantine_frame = pd.DataFrame(quarantine, columns=("event_id", "local_code", "page_id", "reason"))
    claims_path = root / "timestamp_recovery.csv"
    accepted_path = root / "accepted_timestamp_recovery.csv"
    rejected_path = root / "rejected_timestamp_recovery.csv"
    official_sources_path = root / "official_source_manifest.csv"
    quarantine_path = root / "quarantine" / "rows.csv"
    summary_path = root / "summary.md"
    for path, frame in (
        (claims_path, claims_frame), (accepted_path, accepted_frame),
        (rejected_path, rejected_frame), (official_sources_path, sources_frame),
        (quarantine_path, quarantine_frame),
    ):
        _atomic_csv(path, frame)
    _atomic_text(summary_path, _summary(run_id, claims_frame, sources_frame))

    outputs = (input_rows_path, claims_path, accepted_path, rejected_path,
               official_sources_path, quarantine_path, summary_path)
    manifest = {
        "schema_version": "1.0-v14-six-bounded-immutable",
        "run_id": run_id,
        "component": COMPONENT,
        "status": "complete" if rejected_frame.empty else "partial_fail_closed",
        "source_policy": {
            "official_jpx_only": True,
            "exact_event_denominator": 6,
            "exact_page_denominator": 6,
            "redirects_accepted": False,
            "issuer_domains_accepted": False,
            "http_headers_used_as_publication_time": False,
            "inferred_publication_time_used": False,
            "known_at_rule": KNOWN_AT_RULE,
            "excluded_codes": ["4283"],
            "zero_equity_channel_included": False,
        },
        "counts": {
            "target_events": len(claims_frame),
            "exact_pages": len(EVENT_SPECS),
            "retained_source_responses": len(sources_frame),
            "accepted_events": len(accepted_frame),
            "rejected_events": len(rejected_frame),
            "accepted_source_responses": int(sources_frame["accepted"].eq(True).sum()),
            "quarantined_source_responses": int(sources_frame["accepted"].eq(False).sum()),
            "publication_date_only_events": len(claims_frame),
        },
        "inputs": {
            "v14_manifest": _descriptor(source_manifest_path),
            "v14_events": _descriptor(event_source),
        },
        "outputs": {path.relative_to(root).as_posix(): _descriptor(path) for path in outputs},
        "raw_source_sha256": {row["source_id"]: row["sha256"] for row in sources},
        "response_metadata_sha256": {
            row["source_id"]: row["metadata_sha256"] for row in sources
        },
        "refinitiv_calls": False,
        "primary_halt_history_modified": False,
        "independent_validation_required": True,
        "promotion_eligible": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4OfficialPublicationTimestampV14Artifacts(
        run_id, root, manifest_path, claims_path, accepted_path, manifest
    )


def _validate_v14_input(events_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    if not events_path.is_file() or not manifest_path.is_file():
        raise ValueError("V14 halt events and manifest must exist")
    if not (
        manifest.get("run_id") == V14_RUN_ID
        and manifest.get("component") == "halt_suspension_events"
        and manifest.get("artifact_type") == "japan_g4_halt_suspension_history_producer"
    ):
        raise ValueError("events are not the exact bounded V14 halt source")
    expected = manifest.get("output_sha256", {}).get(events_path.name)
    if not expected or expected != _sha256(events_path):
        raise ValueError("V14 halt event hash mismatch")


def _target_rows(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "event_type", "local_code", "issue_name", "event_date", "source_id",
        "source_url", "source_sha256", "source_record_id",
    }
    if not required.issubset(events.columns):
        raise ValueError("V14 event input is missing required columns")
    expected = {spec.event_id for spec in EVENT_SPECS}
    rows = events.loc[events["event_id"].isin(expected), sorted(required)].copy()
    if len(rows) != len(expected) or rows["event_id"].duplicated().any():
        raise ValueError("V14 six-event denominator is not one-to-one")
    return rows.sort_values("event_id").reset_index(drop=True)


def _event_row_checks(row: pd.Series, spec: EventSpec) -> dict[str, bool]:
    return {
        "event_id": str(row["event_id"]) == spec.event_id,
        "event_type": str(row["event_type"]) == "delisting_non_trading_interval",
        "local_code": str(row["local_code"]) == spec.local_code,
        "issue_name": all(
            _canonical(fragment) in _canonical(str(row["issue_name"]))
            for fragment in spec.issue_name.split(" ") if fragment
        ),
        "event_date": str(row["event_date"]) == spec.event_date,
        "source_record_id": str(row["source_record_id"]) == spec.source_record_id,
        "event_source_url": str(row["source_url"]) == spec.event_source_url,
        "event_source_sha256": str(row["source_sha256"]) == spec.event_source_sha256,
    }


def _retain_response(
    root: Path, spec: EventSpec, response: HttpResponse, accepted: bool, reason: str
) -> dict[str, Any]:
    digest = hashlib.sha256(response.payload).hexdigest()
    base = root / ("raw" if accepted else "quarantine/raw")
    metadata_base = root / ("raw/metadata" if accepted else "quarantine/metadata")
    raw_path = base / f"{spec.page_id}-{digest[:16]}.html"
    metadata_path = metadata_base / f"{spec.page_id}-{digest[:16]}.json"
    _atomic_bytes(raw_path, response.payload)
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    metadata = {
        "source_id": f"{spec.page_id}-page",
        "requested_url": spec.notice_url,
        "final_url": response.final_url,
        "http_status": response.status,
        "headers": dict(sorted(headers.items())),
        "acquired_at_utc": response.acquired_at_utc,
        "payload_sha256": digest,
        "payload_bytes": len(response.payload),
        "accepted": accepted,
        "quarantine_reason": reason,
        "publication_timestamp_input": False,
    }
    _atomic_json(metadata_path, metadata)
    return {
        "source_id": f"{spec.page_id}-page",
        "page_id": spec.page_id,
        "event_id": spec.event_id,
        "requested_url": spec.notice_url,
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
        "quarantine_reason": reason,
    }


def _fetch(url: str, timeout: float) -> HttpResponse:
    if not _is_exact_official_url(url):
        raise ValueError(f"non-official or unsupported JPX URL: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quant-lab-japan-g4-v14-publication-evidence/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        result = HttpResponse(
            payload, int(response.status), response.geturl(),
            dict(response.headers.items()), datetime.now(UTC).isoformat(),
        )
    if len(result.payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"JPX response exceeds byte limit: {url}")
    return result


def _is_exact_official_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == OFFICIAL_HOST
        and parsed.port is None
        and parsed.path.startswith(("/news/", "/english/news/"))
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def _known_at(publication_date: str) -> str:
    publication = date.fromisoformat(publication_date)
    return datetime.combine(publication + timedelta(days=1), time.min, JST).astimezone(UTC).isoformat()


def _canonical(value: str) -> str:
    clean = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return "".join(character for character in clean if character.isalnum())


def _content_type(response: HttpResponse) -> str:
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _summary(run_id: str, claims: pd.DataFrame, sources: pd.DataFrame) -> str:
    return (
        f"# Japan G4 V14 Six-Event Official Publication Timestamp Packet: {run_id}\n\n"
        f"- Exact target events: {len(claims)}\n"
        f"- Accepted claims: {int(claims['status'].eq('accepted').sum())}\n"
        f"- Rejected claims: {int(claims['status'].eq('rejected').sum())}\n"
        f"- Retained official JPX responses: {len(sources)}\n"
        "- Known-at rule: first next midnight JST after the official page date.\n"
        "- HTTP headers and acquisition timestamps are not publication evidence.\n"
        "- Code 4283 and the 2014-11 zero-equity channel are excluded.\n"
        "- Refinitiv calls: none.\n"
    )


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _json(path: Path) -> Any:
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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)
