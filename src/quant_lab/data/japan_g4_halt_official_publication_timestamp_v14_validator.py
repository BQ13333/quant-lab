from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import tempfile
import unicodedata
import urllib.parse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
OFFICIAL_HOST = "www.jpx.co.jp"
V14_RUN_ID = "JPN-G4-HALTS-200607-202606-V14"
PRODUCER_COMPONENT = "japan_g4_halt_v14_six_official_publication_timestamp_recovery"
VALIDATOR_COMPONENT = (
    "japan_g4_halt_v14_six_official_publication_timestamp_recovery_validation"
)
PUBLICATION_PRECISION = "official_page_date_only"
KNOWN_AT_RULE = "date_only_conservative_upper_bound_next_midnight_jst"


EXPECTATIONS = {
    "jpx-report-729b287a25a2fa47697f890e": {
        "page_id": "detail_1789", "local_code": "6784",
        "issue_name": "プラネックスホールディング PLANEX HOLDING INC.",
        "event_date": "2014-12-12",
        "source_record_id": "baf898d9a50e814b83579dd280f8109de5d3aa16e292855d62e18cc33747478f",
        "event_source_url": "https://www.jpx.co.jp/files/tse/english/market/data/geppo/b7gje6000002aeyz-att/17_idou1412.pdf",
        "event_source_sha256": "34ed4610f56de4e8275f861d98fba09052820d69750c329452e5f86c4936ccdc",
        "notice_url": "https://www.jpx.co.jp/news/detail/detail_1789.html",
        "publication_date": "2014-11-19",
        "fragments": ("2014/11/19", "プラネックスホールディング株式会社", "コード:6784", "平成26年12月12日"),
    },
    "jpx-report-a0c03f3de0132d297e33660f": {
        "page_id": "detail_1616", "local_code": "2712",
        "issue_name": "スターバックス コーヒー ジャパン Starbucks Coffee Japan,Ltd.",
        "event_date": "2015-03-23",
        "source_record_id": "7e46e2356741d8e6d4be402f62691cceafd5731131e52a6b062b6846ddeef20c",
        "event_source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/b5b4pj000000idmr-att/17_idou1503.pdf",
        "event_source_sha256": "df7b071480630fb57c0b621aeebb1e8abaf3efbc1cf483100e299390e8b30131",
        "notice_url": "https://www.jpx.co.jp/english/news/detail/detail_1616.html",
        "publication_date": "2015-02-20",
        "fragments": ("Feb. 20, 2015", "Starbucks Coffee Japan, Ltd. stock", "Code: 2712", "Delisting Date Mar. 23, 2015"),
    },
    "jpx-report-c0565af152c4387ecd7c2457": {
        "page_id": "20150430-12", "local_code": "9963",
        "issue_name": "江守グループホールディングス EMORI GROUP HOLDINGS CO.,LTD.",
        "event_date": "2015-05-31",
        "source_record_id": "afb4ed11cead13eb69d2d5e4a51956fd04bcb14b4fe9cd3340b5cfa98e630c84",
        "event_source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/b5b4pj000000m7co-att/17_idou1505.pdf",
        "event_source_sha256": "559a1136d5096c20b67cbfb57a1ac0430e9969cbc402700219de150d3555871d",
        "notice_url": "https://www.jpx.co.jp/news/1021/20150430-12.html",
        "publication_date": "2015-04-30",
        "fragments": ("2015/04/30", "江守グループホールディングス株式会社", "コード:9963", "2015年5月31日"),
    },
    "jpx-report-9ad0ce8ccb9c6a57f7a8ce39": {
        "page_id": "20150630-15", "local_code": "7708",
        "issue_name": "石山 Gateway Holdings Ishiyama Gateway Holdings Inc.",
        "event_date": "2015-08-01",
        "source_record_id": "c2fd6aca2fca864c12a3bcaf03c34c6f42376bfe74289a6e72b9b8abee0cf81e",
        "event_source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/b5b4pj000000qc9y-att/17_idou1508.pdf",
        "event_source_sha256": "16ffb220908d6e06a40f2eee41f95e0a9a6dc59ef73731d75f6352480efa93ab",
        "notice_url": "https://www.jpx.co.jp/news/1021/20150630-15.html",
        "publication_date": "2015-06-30",
        "fragments": ("2015/06/30", "石山Gateway Holdings株式会社", "コード:7708", "2015年8月1日"),
    },
    "jpx-report-bcc8fea64fb029baf0251360": {
        "page_id": "20150811-14", "local_code": "3587",
        "issue_name": "グローバルアジアホールディングス Global Asia Holdings Inc.",
        "event_date": "2015-09-12",
        "source_record_id": "cf0ee6a81465427c4f11070c83af7fd8adad720de7614e9138657892329fb123",
        "event_source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/b5b4pj000000rgwu-att/17_idou1509.pdf",
        "event_source_sha256": "a380cdf4d6900fb9f276305c16a217274cdf431ad90ee514b245f39be00d8edb",
        "notice_url": "https://www.jpx.co.jp/news/1021/20150811-14.html",
        "publication_date": "2015-08-11",
        "fragments": ("2015/08/11", "グローバルアジアホールディングス株式会社", "コード:3587", "2015年9月12日"),
    },
    "jpx-report-570db7455dcb723f56c9febf": {
        "page_id": "20151030-13", "local_code": "4798",
        "issue_name": "エル・シー・エーホールディングス LCA Holdings Corporation",
        "event_date": "2015-12-01",
        "source_record_id": "cb50a33bb04b362af9190fb2725503c396881d737433a066a702d259843bc523",
        "event_source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/b5b4pj000000vj5c-att/17_idou1512.pdf",
        "event_source_sha256": "a8ebffef8353a15264afbe49ddb7a8afdd34c9799cac6ca52dcefea221723b3d",
        "notice_url": "https://www.jpx.co.jp/news/1021/20151030-13.html",
        "publication_date": "2015-10-30",
        "fragments": ("2015/10/30", "株式会社エル・シー・エーホールディングス", "コード:4798", "2015年12月1日"),
    },
}
PRODUCER_OUTPUTS = {
    "input_event_rows.csv", "timestamp_recovery.csv", "accepted_timestamp_recovery.csv",
    "rejected_timestamp_recovery.csv", "official_source_manifest.csv",
    "quarantine/rows.csv", "summary.md",
}


def validate_japan_g4_halt_official_publication_timestamp_v14(
    *,
    producer_dir: str | Path,
    validation_run_id: str,
    artifact_root: str | Path = "artifacts/japan_g4_halt_timestamp_recovery_validation",
) -> tuple[Path, dict[str, Any]]:
    """Independently rehash and reparse the six retained official JPX notices."""
    _validate_run_id(validation_run_id)
    producer = Path(producer_dir).resolve()
    destination = (Path(artifact_root) / validation_run_id).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    issues: list[str] = []

    manifest_path = producer / "manifest.json"
    manifest = _json(manifest_path)
    if not (
        manifest.get("component") == PRODUCER_COMPONENT
        and manifest.get("schema_version") == "1.0-v14-six-bounded-immutable"
        and manifest.get("source_policy", {}).get("official_jpx_only") is True
        and manifest.get("source_policy", {}).get("redirects_accepted") is False
        and manifest.get("source_policy", {}).get("excluded_codes") == ["4283"]
        and manifest.get("source_policy", {}).get("zero_equity_channel_included") is False
    ):
        issues.append("producer_manifest_contract_invalid")

    outputs = manifest.get("outputs", {})
    if set(outputs) != PRODUCER_OUTPUTS:
        issues.append("producer_output_denominator_invalid")
    for relative, descriptor in outputs.items():
        if not _safe_descriptor_matches(producer, relative, descriptor):
            issues.append(f"producer_output_invalid:{relative}")

    claims = _csv(producer / "timestamp_recovery.csv")
    accepted = _csv(producer / "accepted_timestamp_recovery.csv")
    rejected = _csv(producer / "rejected_timestamp_recovery.csv")
    sources = _csv(producer / "official_source_manifest.csv")
    quarantine = _csv(producer / "quarantine" / "rows.csv")
    expected_ids = set(EXPECTATIONS)
    if set(claims.get("event_id", [])) != expected_ids or len(claims) != 6:
        issues.append("claim_denominator_invalid")
    if claims.get("event_id", pd.Series(dtype=str)).duplicated().any():
        issues.append("duplicate_claim_event_id")
    required_claim_columns = {
        "event_id", "event_type", "local_code", "v14_issue_name", "event_date",
        "source_record_id", "event_source_url", "event_source_sha256",
        "publication_date", "publication_time_jst", "publication_precision",
        "known_at_utc", "known_at_rule", "page_id", "source_url", "source_sha256",
        "status", "rejection_reason",
    }
    if not required_claim_columns.issubset(claims.columns):
        issues.append("claim_schema_invalid")
    if "4283" in set(claims.get("local_code", [])):
        issues.append("contract_blocked_code_present")
    if len(sources) != 6 or sources.get("source_id", pd.Series(dtype=str)).duplicated().any():
        issues.append("source_denominator_invalid")

    v14_events, input_issues = _validate_v14_inputs(manifest)
    issues.extend(input_issues)
    source_details, source_state = _validate_sources(producer, sources)
    for row in source_details.loc[~source_details["passed"]].itertuples():
        issues.append(f"source_validation_failed:{row.source_id}:{row.issues}")

    accepted_ids = set(accepted.get("event_id", []))
    rejected_ids = set(rejected.get("event_id", []))
    quarantine_ids = set(quarantine.get("event_id", []))
    claim_details: list[dict[str, Any]] = []
    for event_id, expected in EXPECTATIONS.items():
        matches = (
            claims.loc[claims["event_id"].eq(event_id)]
            if "event_id" in claims.columns
            else pd.DataFrame()
        )
        row = matches.iloc[0].to_dict() if len(matches) == 1 else {}
        event_matches = v14_events.loc[v14_events.get("event_id", pd.Series(dtype=str)).eq(event_id)]
        event_row = event_matches.iloc[0].to_dict() if len(event_matches) == 1 else {}
        state = source_state.get(event_id, {})
        event_valid = _event_matches(event_row, expected)
        source_valid = bool(state.get("passed"))
        independently_acceptable = event_valid and source_valid
        expected_status = "accepted" if independently_acceptable else "rejected"
        row_issues: list[str] = []
        field_checks = {
            "event_type": row.get("event_type") == "delisting_non_trading_interval",
            "local_code": row.get("local_code") == expected["local_code"],
            "v14_issue_name": row.get("v14_issue_name") == expected["issue_name"],
            "event_date": row.get("event_date") == expected["event_date"],
            "source_record_id": row.get("source_record_id") == expected["source_record_id"],
            "event_source_url": row.get("event_source_url") == expected["event_source_url"],
            "event_source_sha256": row.get("event_source_sha256") == expected["event_source_sha256"],
            "publication_date": row.get("publication_date") == expected["publication_date"],
            "publication_time_blank": row.get("publication_time_jst") == "",
            "publication_precision": row.get("publication_precision") == PUBLICATION_PRECISION,
            "page_id": row.get("page_id") == expected["page_id"],
            "source_url": row.get("source_url") == expected["notice_url"],
            "source_sha256": row.get("source_sha256") == state.get("sha256", ""),
            "status": row.get("status") == expected_status,
        }
        if independently_acceptable:
            field_checks.update({
                "known_at": row.get("known_at_utc") == _known_at(expected["publication_date"]),
                "known_at_rule": row.get("known_at_rule") == KNOWN_AT_RULE,
                "partition": event_id in accepted_ids and event_id not in rejected_ids,
                "no_rejection_reason": row.get("rejection_reason") == "",
            })
        else:
            field_checks.update({
                "known_at_blank": row.get("known_at_utc") == "",
                "known_at_rule_blank": row.get("known_at_rule") == "",
                "partition": event_id in rejected_ids and event_id in quarantine_ids,
            })
        row_issues.extend(name for name, passed in field_checks.items() if not passed)
        if row_issues:
            issues.append(f"claim_validation_failed:{event_id}:{'|'.join(row_issues)}")
        claim_details.append({
            "event_id": event_id,
            "local_code": expected["local_code"],
            "v14_event_valid": event_valid,
            "retained_source_valid": source_valid,
            "producer_status": row.get("status", ""),
            "known_at_valid": field_checks.get("known_at", field_checks.get("known_at_blank", False)),
            "passed": not row_issues,
            "issues": "|".join(row_issues),
        })

    claim_frame = pd.DataFrame(claim_details)
    claim_status = claims.get("status", pd.Series(dtype=str))
    counts = manifest.get("counts", {})
    if not (
        counts.get("target_events") == 6
        and counts.get("exact_pages") == 6
        and counts.get("retained_source_responses") == 6
        and counts.get("accepted_events") == int(claim_status.eq("accepted").sum())
        and counts.get("rejected_events") == int(claim_status.eq("rejected").sum())
    ):
        issues.append("producer_count_mismatch")

    claims_path = destination / "claim_validation.csv"
    sources_path = destination / "source_validation.csv"
    issues_path = destination / "validation_issues.csv"
    _atomic_csv(claims_path, claim_frame)
    _atomic_csv(sources_path, source_details)
    _atomic_csv(issues_path, pd.DataFrame({"issue": issues}))
    validation_outputs = (claims_path, sources_path, issues_path)
    validation_manifest = {
        "schema_version": "1.0-independent-v14-six-retained-byte-validation",
        "run_id": validation_run_id,
        "component": VALIDATOR_COMPONENT,
        "producer_run_id": manifest.get("run_id", ""),
        "producer_manifest": _descriptor(manifest_path),
        "status": "pass" if not issues else "failed_validation",
        "counts": {
            "target_events": len(claim_frame),
            "producer_accepted_events": int(claim_status.eq("accepted").sum()),
            "producer_rejected_events": int(claim_status.eq("rejected").sum()),
            "independently_validated_events": int(claim_frame["passed"].sum()),
            "source_responses": len(source_details),
            "validated_source_responses": int(source_details["passed"].sum()),
            "validation_issues": len(issues),
        },
        "validation_issues": issues,
        "independent_implementation": True,
        "retained_bytes_reparsed": True,
        "event_lineage_revalidated": True,
        "refinitiv_calls": False,
        "promotion_eligible": False,
        "outputs": {path.name: _descriptor(path) for path in validation_outputs},
    }
    manifest_output = destination / "manifest.json"
    _atomic_json(manifest_output, validation_manifest)
    return manifest_output, validation_manifest


def _validate_v14_inputs(manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    descriptors = manifest.get("inputs", {})
    events_descriptor = descriptors.get("v14_events", {})
    manifest_descriptor = descriptors.get("v14_manifest", {})
    events_path = Path(str(events_descriptor.get("path", "")))
    source_manifest_path = Path(str(manifest_descriptor.get("path", "")))
    if not _descriptor_matches(events_path, events_descriptor):
        issues.append("v14_event_input_hash_invalid")
        return pd.DataFrame(), issues
    if not _descriptor_matches(source_manifest_path, manifest_descriptor):
        issues.append("v14_manifest_input_hash_invalid")
        return pd.DataFrame(), issues
    source_manifest = _json(source_manifest_path)
    if not (
        source_manifest.get("run_id") == V14_RUN_ID
        and source_manifest.get("component") == "halt_suspension_events"
        and source_manifest.get("artifact_type") == "japan_g4_halt_suspension_history_producer"
        and source_manifest.get("output_sha256", {}).get(events_path.name) == _sha256(events_path)
    ):
        issues.append("v14_manifest_lineage_invalid")
        return pd.DataFrame(), issues
    events = pd.read_parquet(events_path).fillna("")
    target = events.loc[events["event_id"].isin(EXPECTATIONS)].copy()
    if len(target) != 6 or target["event_id"].duplicated().any():
        issues.append("v14_event_denominator_invalid")
    return target, issues


def _validate_sources(
    producer: Path, sources: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    for event_id, expected in EXPECTATIONS.items():
        matches = sources.loc[sources.get("event_id", pd.Series(dtype=str)).eq(event_id)]
        row = matches.iloc[0].to_dict() if len(matches) == 1 else {}
        row_issues: list[str] = []
        raw_path = Path(str(row.get("raw_path", "")))
        metadata_path = Path(str(row.get("metadata_path", "")))
        raw_valid = _safe_member_file(producer, raw_path)
        metadata_valid = _safe_member_file(producer, metadata_path)
        payload = raw_path.read_bytes() if raw_valid else b""
        raw_sha = hashlib.sha256(payload).hexdigest() if raw_valid else ""
        if not raw_valid or raw_sha != row.get("sha256") or str(len(payload)) != row.get("bytes"):
            row_issues.append("raw_hash_or_size_invalid")
        metadata = _json(metadata_path) if metadata_valid else {}
        if not metadata_valid or _sha256(metadata_path) != row.get("metadata_sha256"):
            row_issues.append("metadata_hash_invalid")
        if not (
            metadata.get("source_id") == f"{expected['page_id']}-page"
            and metadata.get("requested_url") == expected["notice_url"]
            and metadata.get("final_url") == expected["notice_url"]
            and metadata.get("http_status") == 200
            and metadata.get("payload_sha256") == raw_sha
            and metadata.get("payload_bytes") == len(payload)
            and metadata.get("publication_timestamp_input") is False
        ):
            row_issues.append("metadata_contract_invalid")
        content_type = str(row.get("content_type", "")).split(";", 1)[0].strip().lower()
        if not (
            row.get("source_id") == f"{expected['page_id']}-page"
            and row.get("page_id") == expected["page_id"]
            and row.get("requested_url") == expected["notice_url"]
            and row.get("final_url") == expected["notice_url"]
            and _is_exact_official_url(str(row.get("final_url", "")))
            and row.get("http_status") == "200"
            and content_type in {"text/html", "application/xhtml+xml"}
        ):
            row_issues.append("response_contract_invalid")
        canonical_text = _canonical(
            " ".join(BeautifulSoup(payload, "html.parser").stripped_strings)
        ) if payload else ""
        if not all(_canonical(fragment) in canonical_text for fragment in expected["fragments"]):
            row_issues.append("official_content_binding_invalid")
        passed = not row_issues
        state[event_id] = {"passed": passed, "sha256": raw_sha}
        details.append({
            "source_id": f"{expected['page_id']}-page",
            "event_id": event_id,
            "local_code": expected["local_code"],
            "raw_sha256": raw_sha,
            "raw_bytes": len(payload),
            "metadata_valid": metadata_valid and "metadata_contract_invalid" not in row_issues,
            "content_binding_valid": "official_content_binding_invalid" not in row_issues,
            "passed": passed,
            "issues": "|".join(row_issues),
        })
    return pd.DataFrame(details), state


def _event_matches(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    issue = _canonical(str(row.get("issue_name", "")))
    expected_issue_tokens = [
        _canonical(token) for token in str(expected["issue_name"]).split(" ") if token
    ]
    return bool(
        row.get("event_id") in EXPECTATIONS
        and str(row.get("event_type")) == "delisting_non_trading_interval"
        and str(row.get("local_code")) == expected["local_code"]
        and all(token in issue for token in expected_issue_tokens)
        and str(row.get("event_date")) == expected["event_date"]
        and str(row.get("source_record_id")) == expected["source_record_id"]
        and str(row.get("source_url")) == expected["event_source_url"]
        and str(row.get("source_sha256")) == expected["event_source_sha256"]
    )


def _safe_descriptor_matches(root: Path, relative: str, descriptor: dict[str, Any]) -> bool:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return False
    path = root / relative
    return _safe_member_file(root, path) and _descriptor_matches(path, descriptor)


def _safe_member_file(root: Path, path: Path) -> bool:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        return resolved.is_file() and not path.is_symlink() and resolved.is_relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _descriptor_matches(path: Path, descriptor: dict[str, Any]) -> bool:
    try:
        return bool(
            path.is_file()
            and str(path.resolve()) == str(Path(str(descriptor.get("path", ""))).resolve())
            and _sha256(path) == descriptor.get("sha256")
            and path.stat().st_size == descriptor.get("bytes")
        )
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _is_exact_official_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return bool(
        parsed.scheme == "https" and parsed.hostname == OFFICIAL_HOST and parsed.port is None
        and parsed.path.startswith(("/news/", "/english/news/"))
        and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
    )


def _known_at(publication_date: str) -> str:
    publication = date.fromisoformat(publication_date)
    return datetime.combine(publication + timedelta(days=1), time.min, JST).astimezone(UTC).isoformat()


def _canonical(value: str) -> str:
    clean = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return "".join(character for character in clean if character.isalnum())


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


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("validation_run_id must be a path-safe name")


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
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)
