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
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
PRODUCER_COMPONENT = "japan_g4_halt_v8_official_publication_timestamp_recovery"
V8_RUN_ID = "JPN-G4-HALTS-200607-202606-V8"
PUBLICATION_PRECISION = "official_page_date_only"
KNOWN_AT_RULE = "date_only_conservative_upper_bound_next_midnight_jst"

PAGE_EXPECTATIONS = {
    "detail_696": {
        "url": "https://www.jpx.co.jp/english/news/detail/detail_696.html",
        "publication_date": "2013-09-13",
        "heading": "Decision on Delisting, etc. -Simplex Holdings,Inc.-",
        "support_url": "",
    },
    "detail_933": {
        "url": "https://www.jpx.co.jp/english/news/detail/detail_933.html",
        "publication_date": "2014-01-21",
        "heading": "Decision on Delisting, etc. -TOTAL MEDICAL SERVICE CO.,LTD.",
        "support_url": "",
    },
    "detail_1022": {
        "url": "https://www.jpx.co.jp/english/news/detail/detail_1022.html",
        "publication_date": "2014-03-05",
        "heading": "Calculation of TOPIX in relation to shares of Maruha Nichiro Corporation.",
        "support_url": "",
    },
    "detail_1207": {
        "url": "https://www.jpx.co.jp/english/news/detail/detail_1207.html",
        "publication_date": "2014-07-24",
        "heading": "Decision on Delisting, etc. -Minerva Holdings CO.,LTD.-",
        "support_url": "",
    },
    "20160401-11": {
        "url": "https://www.jpx.co.jp/english/news/1021/20160401-11.html",
        "publication_date": "2016-04-01",
        "heading": "Decision on Delisting, etc.: MISAWA HABITA&CO.,LTD.",
        "support_url": "",
    },
    "20160630-02": {
        "url": "https://www.jpx.co.jp/english/news/1021/20160630-02.html",
        "publication_date": "2016-06-30",
        "heading": "Decision on Delisting, etc.: MAG NET HOLDINGS, INC.",
        "support_url": "",
    },
    "20180905-02": {
        "url": "https://www.jpx.co.jp/english/news/1044/20180905-02.html",
        "publication_date": "2018-09-05",
        "heading": "Calculation of TOPIX in relation to shares of Daishi Hokuetsu Financial Group,Inc.",
        "support_url": "https://www.jpx.co.jp/english/news/1044/b5b4pj000002ai5a-att/data_e.pdf",
    },
    "20190205-01": {
        "url": "https://www.jpx.co.jp/english/news/1044/20190205-01.html",
        "publication_date": "2019-02-05",
        "heading": "Calculation of TOPIX in relation to shares of HITO-Communications Holdings,Inc",
        "support_url": "https://www.jpx.co.jp/english/news/1044/b5b4pj000002on5e-att/hp01_e.pdf",
    },
    "20260603-01": {
        "url": "https://www.jpx.co.jp/english/news/6030/20260603-01.html",
        "publication_date": "2026-06-03",
        "heading": "Calculation of TOPIX in relation to shares of giftee Group,Inc.",
        "support_url": "https://www.jpx.co.jp/english/news/6030/vk0khi000001mqgh-att/data_e.pdf",
    },
}

EVENT_EXPECTATIONS = {
    "jpx-report-d129a47fac42d1338b0cdf10": (
        "detail_696",
        "4340",
        "Simplex Holdings,Inc.",
        "2013-10-15",
        ("simplexholdingsincstockcode4340", "delistingdateoct152013"),
    ),
    "jpx-report-e05f17d2e939486e13fde78d": (
        "detail_933",
        "3163",
        "TOTAL MEDICAL SERVICE CO.,LTD.",
        "2014-02-25",
        ("totalmedicalservicecoltdstockcode3163", "delistingdatefeb252014"),
    ),
    "jpx-report-5211314f9df91d3614fbf85c": (
        "detail_1022",
        "1334",
        "Maruha Nichiro Holdings,Inc.",
        "2014-03-27",
        ("maruhanichiroholdingsinc1334", "sharestobedelistedonmarch27"),
    ),
    "jpx-report-c41cf5f4ef754f5adfede406": (
        "detail_1207",
        "3090",
        "Minerva Holdings CO.,LTD.",
        "2014-08-25",
        ("minervaholdingscoltdstockcode3090", "delistingdateaug252014"),
    ),
    "jpx-report-bd31eeba13db47900ee0cf58": (
        "20160401-11",
        "1427",
        "MISAWA HABITA&CO.,LTD.",
        "2016-04-15",
        ("misawahabitaandcoltdstockcode1427", "delistingdateapr152016"),
    ),
    "jpx-report-40ff9f970778ba0e64673c62": (
        "20160630-02",
        "8073",
        "MAG NET HOLDINGS,INC.",
        "2016-08-01",
        ("magnetholdingsincstockcode8073", "delistingdateaug012016"),
    ),
    "jpx-delist-f21453f0f4aff06cbceedf0d": (
        "20180905-02",
        "8324",
        "The Daishi Bank,Ltd.",
        "2018-09-26",
        ("thedaishibankltd8324", "tobedelistedonseptember262018"),
    ),
    "jpx-delist-8f88eba45e321e35ff26a1c7": (
        "20180905-02",
        "8325",
        "The Hokuetsu Bank,Ltd.",
        "2018-09-26",
        ("thehokuetsubankltd8325", "tobedelistedonseptember262018"),
    ),
    "jpx-delist-3197ccfb203a833a61027944": (
        "20190205-01",
        "3654",
        "HITO-Communications,Inc.",
        "2019-02-26",
        ("hitocommunicationsinc3654", "tobedelistedonfebruary262019"),
    ),
    "jpx-delist-f81fc020ff0855d311c690e3": (
        "20260603-01",
        "4449",
        "giftee Inc.",
        "2026-06-29",
        ("gifteeinc4449", "tobedelistedonjune292026"),
    ),
}

CLAIM_VALIDATION_COLUMNS = (
    "event_id",
    "page_id",
    "producer_status",
    "v8_event_valid",
    "source_hashes_valid",
    "response_metadata_valid",
    "page_claims_valid",
    "event_claims_valid",
    "publication_precision_valid",
    "known_at_valid",
    "partition_valid",
    "passed",
    "issues",
)
SOURCE_VALIDATION_COLUMNS = (
    "source_id",
    "page_id",
    "source_role",
    "hash_valid",
    "metadata_valid",
    "route_valid",
    "content_type_valid",
    "location_valid",
    "passed",
    "issues",
)


def validate_japan_g4_halt_timestamp_recovery_v8(
    *,
    producer_dir: str | Path,
    validation_run_id: str,
    artifact_root: str | Path = "artifacts/japan_g4_halt_timestamp_recovery_validation",
) -> tuple[Path, dict[str, Any]]:
    """Independently reconstruct all 10 claims from retained V8 and JPX bytes."""
    _validate_run_id(validation_run_id)
    producer = Path(producer_dir).resolve()
    destination = (Path(artifact_root) / validation_run_id).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = producer / "manifest.json"
    manifest = _json(manifest_path)
    issues: list[str] = []

    if manifest.get("component") != PRODUCER_COMPONENT:
        issues.append("producer_component_invalid")
    if manifest.get("refinitiv_calls") is not False:
        issues.append("refinitiv_contract_invalid")
    policy = manifest.get("source_policy", {})
    prohibited = (
        "http_response_date_used_as_publication_time",
        "http_last_modified_used_as_publication_time",
        "archive_capture_time_used",
        "report_month_used",
        "same_code_prior_observation_used",
        "inferred_publication_time_used",
    )
    if any(policy.get(key) is not False for key in prohibited):
        issues.append("prohibited_timestamp_source_policy")
    for relative, descriptor in manifest.get("outputs", {}).items():
        path = producer / relative
        if not _descriptor_matches(path, descriptor):
            issues.append(f"producer_output_hash_mismatch:{relative}")

    claims = _csv(producer / "timestamp_recovery.csv")
    accepted = _csv(producer / "accepted_timestamp_recovery.csv")
    rejected = _csv(producer / "rejected_timestamp_recovery.csv")
    sources = _csv(producer / "official_source_manifest.csv")
    quarantine_rows = _csv(producer / "quarantine" / "rows.csv")
    if set(claims["event_id"]) != set(EVENT_EXPECTATIONS) or len(claims) != 10:
        issues.append("claim_denominator_invalid")
    if claims["event_id"].duplicated().any():
        issues.append("duplicate_claim_event_id")
    if set(PAGE_EXPECTATIONS) != set(sources["page_id"]):
        issues.append("page_denominator_invalid")
    expected_source_count = len(PAGE_EXPECTATIONS) + sum(
        bool(page["support_url"]) for page in PAGE_EXPECTATIONS.values()
    )
    if len(sources) != expected_source_count or sources["source_id"].duplicated().any():
        issues.append("source_denominator_invalid")

    input_events_descriptor = manifest.get("inputs", {}).get("v8_events", {})
    input_manifest_descriptor = manifest.get("inputs", {}).get("v8_manifest", {})
    input_events_path = Path(str(input_events_descriptor.get("path", "")))
    input_manifest_path = Path(str(input_manifest_descriptor.get("path", "")))
    v8_valid = _descriptor_matches(input_events_path, input_events_descriptor)
    v8_manifest_valid = _descriptor_matches(input_manifest_path, input_manifest_descriptor)
    if not v8_valid or not v8_manifest_valid:
        issues.append("v8_input_hash_invalid")
        v8_events = pd.DataFrame()
    else:
        v8_manifest = _json(input_manifest_path)
        if not (
            v8_manifest.get("run_id") == V8_RUN_ID
            and v8_manifest.get("output_sha256", {}).get(input_events_path.name)
            == _sha256(input_events_path)
        ):
            issues.append("v8_manifest_lineage_invalid")
        v8_events = pd.read_parquet(input_events_path).fillna("")

    source_validation, source_state = _validate_sources(producer, sources)
    if not source_validation["passed"].all():
        issues.extend(
            f"source_validation_failed:{row.source_id}:{row.issues}"
            for row in source_validation.loc[~source_validation["passed"]].itertuples()
        )

    accepted_ids = set(accepted["event_id"])
    rejected_ids = set(rejected["event_id"])
    quarantined_ids = set(quarantine_rows["event_id"])
    claim_validation_rows: list[dict[str, Any]] = []
    for event_id, expectation in EVENT_EXPECTATIONS.items():
        page_id, code, official_name, event_date, fragments = expectation
        row_matches = claims.loc[claims["event_id"].eq(event_id)]
        row_issues: list[str] = []
        row = row_matches.iloc[0].to_dict() if len(row_matches) == 1 else {}
        state = source_state.get(page_id, {})
        page_valid = bool(state.get("page_claims_valid"))
        event_content = str(state.get("event_content", ""))
        event_claims_valid = bool(
            all(fragment in event_content for fragment in fragments)
            and code in event_content
            and _canonical(official_name) in event_content
        )
        source_hashes_valid = bool(state.get("source_hashes_valid"))
        response_metadata_valid = bool(state.get("response_metadata_valid"))
        v8_event_valid, v8_issue_name = _validate_v8_event(
            v8_events, event_id, code, official_name, event_date
        )
        precision_valid = bool(
            row.get("publication_date") == PAGE_EXPECTATIONS[page_id]["publication_date"]
            and row.get("publication_time_jst") == ""
            and row.get("publication_precision") == PUBLICATION_PRECISION
        )
        independently_acceptable = bool(
            v8_event_valid
            and source_hashes_valid
            and response_metadata_valid
            and page_valid
            and event_claims_valid
            and precision_valid
        )
        producer_status = row.get("status", "")
        if independently_acceptable:
            known_at_valid = bool(
                producer_status == "accepted"
                and row.get("known_at_rule") == KNOWN_AT_RULE
                and row.get("known_at_utc")
                == _known_at(PAGE_EXPECTATIONS[page_id]["publication_date"])
                and datetime.fromisoformat(row.get("known_at_utc", ""))
                <= datetime.combine(date.fromisoformat(event_date), time.max, JST).astimezone(UTC)
            )
            partition_valid = event_id in accepted_ids and event_id not in rejected_ids
        else:
            known_at_valid = bool(
                producer_status == "rejected"
                and not row.get("known_at_utc")
                and not row.get("known_at_rule")
            )
            partition_valid = bool(
                event_id in rejected_ids
                and event_id not in accepted_ids
                and event_id in quarantined_ids
            )
        field_valid = bool(
            row.get("page_id") == page_id
            and row.get("local_code") == code
            and row.get("official_issue_name") == official_name
            and row.get("event_date") == event_date
            and row.get("v8_issue_name") == v8_issue_name
            and row.get("page_url") == PAGE_EXPECTATIONS[page_id]["url"]
            and row.get("support_url") == PAGE_EXPECTATIONS[page_id]["support_url"]
        )
        classification_valid = producer_status == (
            "accepted" if independently_acceptable else "rejected"
        )
        checks = {
            "claim_fields": field_valid,
            "source_hashes": source_hashes_valid,
            "response_metadata": response_metadata_valid,
            "publication_precision": precision_valid,
            "classification": classification_valid,
            "known_at": known_at_valid,
            "partition": partition_valid,
        }
        if producer_status == "accepted":
            checks.update(
                {
                    "v8_event": v8_event_valid,
                    "page_claims": page_valid,
                    "event_claims": event_claims_valid,
                }
            )
        row_issues.extend(name for name, passed in checks.items() if not passed)
        if row_issues:
            issues.append(f"claim_validation_failed:{event_id}:{'|'.join(row_issues)}")
        claim_validation_rows.append(
            {
                "event_id": event_id,
                "page_id": page_id,
                "producer_status": producer_status,
                "v8_event_valid": v8_event_valid,
                "source_hashes_valid": source_hashes_valid,
                "response_metadata_valid": response_metadata_valid,
                "page_claims_valid": page_valid,
                "event_claims_valid": event_claims_valid,
                "publication_precision_valid": precision_valid,
                "known_at_valid": known_at_valid,
                "partition_valid": partition_valid,
                "passed": not row_issues,
                "issues": "|".join(row_issues),
            }
        )

    details = pd.DataFrame(claim_validation_rows, columns=CLAIM_VALIDATION_COLUMNS)
    producer_accepted = int(claims["status"].eq("accepted").sum())
    producer_rejected = int(claims["status"].eq("rejected").sum())
    counts = manifest.get("counts", {})
    if not (
        counts.get("target_events") == len(claims)
        and counts.get("accepted_events") == producer_accepted
        and counts.get("rejected_events") == producer_rejected
        and producer_accepted + producer_rejected == len(claims)
    ):
        issues.append("producer_count_mismatch")

    claims_path = destination / "claim_validation.csv"
    sources_path = destination / "source_validation.csv"
    issues_path = destination / "validation_issues.csv"
    _atomic_csv(claims_path, details)
    _atomic_csv(sources_path, source_validation)
    _atomic_csv(issues_path, pd.DataFrame({"issue": issues}))
    outputs = (claims_path, sources_path, issues_path)
    validation_manifest = {
        "schema_version": "1.0-independent-retained-byte-validation",
        "run_id": validation_run_id,
        "component": "japan_g4_halt_v8_official_publication_timestamp_recovery_validation",
        "producer_run_id": manifest.get("run_id", ""),
        "producer_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "status": "pass" if not issues else "failed_validation",
        "counts": {
            "target_events": len(details),
            "producer_accepted_events": producer_accepted,
            "producer_rejected_events": producer_rejected,
            "independently_validated_events": int(details["passed"].sum()),
            "source_responses": len(source_validation),
            "validated_source_responses": int(source_validation["passed"].sum()),
            "validation_issues": len(issues),
        },
        "validation_issues": issues,
        "independent_implementation": True,
        "retained_bytes_reparsed": True,
        "refinitiv_calls": False,
        "promotion_eligible": False,
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        },
    }
    validation_manifest_path = destination / "manifest.json"
    _atomic_json(validation_manifest_path, validation_manifest)
    return validation_manifest_path, validation_manifest


def _validate_sources(
    producer: Path, sources: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    source_index = {row["source_id"]: row for row in sources.to_dict("records")}
    for page_id, expected in PAGE_EXPECTATIONS.items():
        page_row = source_index.get(f"{page_id}-page", {})
        support_row = source_index.get(f"{page_id}-support", {}) if expected["support_url"] else None
        page_result = _validate_source_row(producer, page_row, expected["url"], "text/html")
        rows.append(page_result)
        support_result = None
        if support_row is not None:
            support_result = _validate_source_row(
                producer, support_row, expected["support_url"], "application/pdf"
            )
            rows.append(support_result)

        try:
            page_payload = Path(page_row["raw_path"]).read_bytes()
            page_text = " ".join(BeautifulSoup(page_payload, "html.parser").stripped_strings)
            page_claims_valid = bool(
                _publication_visible(page_text, expected["publication_date"])
                and _canonical(expected["heading"]) in _canonical(page_text)
            )
            event_content = _canonical(page_text)
            if expected["support_url"]:
                links = {
                    urllib.parse.urljoin(expected["url"], anchor.get("href", ""))
                    for anchor in BeautifulSoup(page_payload, "html.parser").find_all(
                        "a", href=True
                    )
                }
                page_claims_valid = bool(
                    page_claims_valid and expected["support_url"] in links
                )
                support_payload = Path(support_row["raw_path"]).read_bytes()
                event_content = _canonical(
                    "\n".join(
                        page.extract_text() or ""
                        for page in PdfReader(BytesIO(support_payload)).pages
                    )
                )
        except Exception:
            page_claims_valid = False
            event_content = ""
        source_results = [page_result] + ([support_result] if support_result else [])
        source_hashes_valid = all(bool(result["hash_valid"]) for result in source_results)
        metadata_valid = all(bool(result["metadata_valid"]) for result in source_results)
        content_type_valid = all(
            bool(result["content_type_valid"]) for result in source_results
        )
        bundle_content_valid = page_claims_valid and all(
            all(fragment in event_content for fragment in expectation[4])
            for expectation in EVENT_EXPECTATIONS.values()
            if expectation[0] == page_id
        )
        expected_accepted = bool(
            source_hashes_valid and metadata_valid and content_type_valid and bundle_content_valid
        )
        manifest_acceptance_valid = all(
            str(source_index[result["source_id"]].get("accepted", ""))
            == str(expected_accepted)
            for result in source_results
        )
        if not manifest_acceptance_valid:
            for result in source_results:
                result["passed"] = False
                result["issues"] = "|".join(
                    filter(None, [result["issues"], "source_acceptance_classification"])
                )
        state[page_id] = {
            "page_claims_valid": page_claims_valid,
            "event_content": event_content,
            "source_hashes_valid": source_hashes_valid,
            "response_metadata_valid": metadata_valid,
        }
    return pd.DataFrame(rows, columns=SOURCE_VALIDATION_COLUMNS), state


def _validate_source_row(
    producer: Path, row: dict[str, str], expected_url: str, expected_content_type: str
) -> dict[str, Any]:
    row_issues: list[str] = []
    raw_path = Path(row.get("raw_path", ""))
    metadata_path = Path(row.get("metadata_path", ""))
    hash_valid = bool(
        raw_path.is_file()
        and _sha256(raw_path) == row.get("sha256")
        and raw_path.stat().st_size == _integer(row.get("bytes", "-1"))
    )
    metadata_valid = False
    if metadata_path.is_file() and _sha256(metadata_path) == row.get("metadata_sha256"):
        try:
            metadata = _json(metadata_path)
            metadata_valid = bool(
                metadata.get("source_id") == row.get("source_id")
                and metadata.get("requested_url") == expected_url
                and metadata.get("final_url") == row.get("final_url")
                and str(metadata.get("http_status")) == row.get("http_status")
                and metadata.get("payload_sha256") == row.get("sha256")
                and metadata.get("payload_bytes") == _integer(row.get("bytes", "-1"))
                and metadata.get("publication_timestamp_input") is False
                and str(metadata.get("accepted")) == row.get("accepted")
            )
        except (ValueError, OSError, json.JSONDecodeError):
            metadata_valid = False
    parsed = urllib.parse.urlparse(expected_url)
    route_valid = bool(
        row.get("requested_url") == expected_url
        and row.get("final_url") == expected_url
        and row.get("http_status") == "200"
        and parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
    )
    content_type_valid = (
        row.get("content_type", "").split(";", 1)[0].strip().lower()
        == expected_content_type
    )
    accepted = row.get("accepted") == "True"
    expected_parent = producer / ("raw" if accepted else Path("quarantine") / "raw")
    try:
        location_valid = raw_path.resolve().is_relative_to(expected_parent.resolve())
    except (OSError, ValueError):
        location_valid = False
    checks = {
        "hash": hash_valid,
        "metadata": metadata_valid,
        "route": route_valid,
        "content_type": content_type_valid,
        "location": location_valid,
    }
    row_issues.extend(name for name, passed in checks.items() if not passed)
    return {
        "source_id": row.get("source_id", ""),
        "page_id": row.get("page_id", ""),
        "source_role": row.get("source_role", ""),
        "hash_valid": hash_valid,
        "metadata_valid": metadata_valid,
        "route_valid": route_valid,
        "content_type_valid": content_type_valid,
        "location_valid": location_valid,
        "passed": not row_issues,
        "issues": "|".join(row_issues),
    }


def _validate_v8_event(
    events: pd.DataFrame,
    event_id: str,
    local_code: str,
    official_name: str,
    event_date: str,
) -> tuple[bool, str]:
    if events.empty or "event_id" not in events:
        return False, ""
    matches = events.loc[events["event_id"].eq(event_id)]
    if len(matches) != 1:
        return False, ""
    row = matches.iloc[0]
    valid = bool(
        str(row.get("event_type", "")) == "delisting_non_trading_interval"
        and str(row.get("local_code", "")) == local_code
        and str(row.get("event_date", "")) == event_date
        and _canonical(official_name) in _canonical(str(row.get("issue_name", "")))
    )
    return valid, str(row.get("issue_name", ""))


def _publication_visible(text: str, value: str) -> bool:
    publication = date.fromisoformat(value)
    return any(
        candidate in text
        for candidate in {
            publication.strftime("%b. %d, %Y"),
            publication.strftime("%b %d, %Y"),
            publication.strftime("%B %d, %Y"),
        }
    )


def _known_at(publication_date: str) -> str:
    publication = date.fromisoformat(publication_date)
    local = datetime.combine(publication + timedelta(days=1), time.min, JST)
    return local.astimezone(UTC).isoformat()


def _canonical(value: str) -> str:
    clean = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    clean = clean.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", clean)


def _descriptor_matches(path: Path, descriptor: dict[str, Any]) -> bool:
    return bool(
        path.is_file()
        and str(path) == str(descriptor.get("path", ""))
        and _sha256(path) == descriptor.get("sha256")
        and path.stat().st_size == descriptor.get("bytes")
    )


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("validation_run_id must be a path-safe name")


def _integer(value: Any) -> int:
    try:
        return int(str(value))
    except ValueError:
        return -1


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
