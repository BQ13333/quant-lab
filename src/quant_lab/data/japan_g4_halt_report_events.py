from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pypdf import PdfReader


HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
EXPECTED_INPUT_ROWS = 366
EXPECTED_COVERED_CLAIMS = 194
EXPECTED_REPORTS = 155
SOURCE_RECORD_VERSION = "jpx-g4-report-row-v1"
EVENT_ID_VERSION = "jpx-g4-report-event-v1"

CHANNEL_HEADINGS = {
    HALT_CHANNEL: "Suspension of Trading",
    DELISTING_CHANNEL: "Delisting",
}
EVENT_TYPES = {
    HALT_CHANNEL: "trading_halt_report_row",
    DELISTING_CHANNEL: "delisting_report_row",
}

DATE_TOKEN = r"20\d{2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{1,2}"
TIME_TOKEN = r"(?:[01]?\d|2[0-3])\s*[:：;.]\s*[0-5]\d"
ROW_PATTERN = re.compile(
    rf"(?P<date>{DATE_TOKEN})\s+(?P<tail>.+)$"
)
CODE_PREFIX = re.compile(r"^(?P<code>[0-9][0-9A-Z]{3,9})(?:\s+|$)")
TIME_RANGE = re.compile(
    rf"(?P<start>{TIME_TOKEN})\s*[～~〜]\s*"
    rf"(?:(?P<end_date>{DATE_TOKEN})\s*)?(?P<end>{TIME_TOKEN})"
)
NUMBERED_REPORT = re.compile(r"^\d{1,2}(?:-\d+)?\s+.*(?:\(|（)")
NEXT_EVENT_HEADING = re.compile(
    r"^(?:New Listings|Delisting|Suspension of Trading|Assignment to|Reassignment|"
    r"Transfer to|Name Change|Change of|Change in|Additional Assignment|"
    r"Exception from|Designation as|Ex-New\b)",
    flags=re.IGNORECASE,
)

PageExtractor = Callable[[bytes], list[str]]

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


@dataclass(frozen=True)
class JapanG4HaltReportEventArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    events_path: Path
    manifest: dict[str, Any]


def parse_report_event_rows(
    pages: list[str],
    *,
    report_month: str,
    channel: str,
    source_url: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Extract explicit table rows from one admitted monthly report section."""
    if channel not in CHANNEL_HEADINGS:
        raise ValueError(f"unsupported channel: {channel}")
    pd.Period(report_month, freq="M")
    located = [
        (page_number, line_number, _canonical_text(line))
        for page_number, page in enumerate(pages, start=1)
        for line_number, line in enumerate(page.splitlines(), start=1)
        if _canonical_text(line)
    ]
    heading = CHANNEL_HEADINGS[channel].casefold()
    starts = [index for index, (_, _, line) in enumerate(located) if line.casefold() == heading]
    if len(starts) != 1:
        raise ValueError(f"expected one explicit {channel} section in {report_month}, got {len(starts)}")

    rows: list[dict[str, Any]] = []
    current_class = ""
    for page_number, _, line in located[starts[0] + 1 :]:
        if _section_boundary(line):
            break
        match = ROW_PATTERN.search(line)
        if not match:
            continue
        prefix = line[: match.start()].strip()
        current_class = _classify_instrument(prefix, current_class)
        if not current_class:
            raise ValueError(f"instrument class unavailable in {report_month} {channel}: {line}")
        event_date = _parse_date(match.group("date"))
        tail = match.group("tail").strip()
        code_match = CODE_PREFIX.match(tail)
        if code_match:
            raw_code = code_match.group("code")
            body = tail[code_match.end() :].strip()
        elif tail.startswith("全銘柄"):
            raw_code = "全銘柄"
            body = tail[len("全銘柄") :].strip()
        else:
            raw_code = ""
            body = tail
        times = _halt_times(body, event_date) if channel == HALT_CHANNEL else _blank_times()
        issue_name = _issue_name(body, channel, times["range_start"])
        if not issue_name and raw_code:
            issue_name = raw_code
        if not issue_name:
            raise ValueError(f"issue name unavailable in {report_month} {channel}: {line}")
        ordinal = len(rows) + 1
        normalized_code = (
            unicodedata.normalize("NFKC", raw_code).strip().upper()
            if CODE_PREFIX.fullmatch(raw_code)
            else ""
        )
        identity = {
            "version": SOURCE_RECORD_VERSION,
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
        record_sha = source_record_sha256(identity)
        rows.append(
            {
                "event_id": hashlib.sha256(
                    f"{EVENT_ID_VERSION}|{record_sha}".encode("utf-8")
                ).hexdigest()[:24],
                "report_month": report_month,
                "channel": channel,
                "event_type": EVENT_TYPES[channel],
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
                "mapping_status": "pending_independent_source_backed_mapping",
                "timestamp_status": "pending_independent_source_backed_known_at",
                "admissible_to_shared_halt_history": False,
            }
        )
    return rows


def source_record_sha256(identity: dict[str, Any]) -> str:
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_japan_g4_halt_report_events(
    *,
    run_id: str,
    denominator_dir: str | Path,
    validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_report_events",
    page_extractor: PageExtractor | None = None,
) -> JapanG4HaltReportEventArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    denominator = Path(denominator_dir).resolve()
    validation = Path(validation_dir).resolve()
    denominator_manifest, denominator_hash_rows = _verify_packet(denominator, "denominator")
    validation_manifest, validation_hash_rows = _verify_packet(validation, "validation")
    denominator_manifest_sha = _sha_path(denominator / "manifest.json")
    validation_manifest_sha = _sha_path(validation / "manifest.json")
    if validation_manifest.get("status") != "pass_partial_component":
        raise ValueError("independent denominator validation did not pass")
    if int(validation_manifest.get("validation_issue_count", -1)) != 0:
        raise ValueError("independent denominator validation contains issues")
    if validation_manifest.get("producer_manifest_sha256") != denominator_manifest_sha:
        raise ValueError("denominator-validation lineage does not match denominator producer")

    denominator_rows = pd.read_csv(denominator / "input_denominator.csv", dtype=str).fillna("")
    coverage = pd.read_csv(denominator / "month_channel_coverage.csv", dtype=str).fillna("")
    claims = pd.read_csv(validation / "claim_reconciliation.csv", dtype=str).fillna("")
    if len(denominator_rows) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"expected {EXPECTED_INPUT_ROWS} denominator rows, got {len(denominator_rows)}")
    if denominator_rows["gap_id"].duplicated().any():
        raise ValueError("denominator gap ids are not unique")
    covered = coverage.loc[_bools(coverage["covered"])].copy()
    if len(covered) != EXPECTED_COVERED_CLAIMS or len(claims) != EXPECTED_COVERED_CLAIMS:
        raise ValueError("expected exactly 194 covered and independently reconciled claims")
    if not _bools(claims["claim_reconciled"]).all():
        raise ValueError("validation packet contains an unreconciled claim")
    if set(covered["gap_id"]) != set(claims["gap_id"]):
        raise ValueError("covered and independently validated claim sets differ")
    months = sorted(set(covered["month"]))
    if len(months) != EXPECTED_REPORTS:
        raise ValueError(f"expected {EXPECTED_REPORTS} admitted reports, got {len(months)}")

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once report-event artifact already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir()
    extract_pages = page_extractor or _extract_pages

    try:
        raw_root = staging / "raw" / "lifecycle"
        input_root = staging / "inputs"
        raw_root.mkdir(parents=True)
        input_root.mkdir()
        shutil.copyfile(denominator / "input_denominator.csv", input_root / "input_denominator.csv")
        shutil.copyfile(validation / "claim_reconciliation.csv", input_root / "claim_reconciliation.csv")

        inventory = pd.read_csv(denominator / "source_inventory.csv", dtype=str).fillna("")
        admitted = inventory.loc[
            inventory["source_kind"].eq("lifecycle_pdf")
            & inventory["validation_status"].eq("exact_month_valid")
        ].set_index("source_key")
        report_rows: list[dict[str, Any]] = []
        pages_by_month: dict[str, list[str]] = {}
        source_by_month: dict[str, dict[str, str]] = {}
        final_raw_root = root / "raw" / "lifecycle"
        for month in months:
            if month not in admitted.index:
                raise ValueError(f"missing admitted source inventory row: {month}")
            source = admitted.loc[month]
            if isinstance(source, pd.DataFrame):
                raise ValueError(f"duplicate admitted source inventory rows: {month}")
            upstream_path = denominator / "raw" / "lifecycle" / f"{month}.pdf"
            observed = _sha_path(upstream_path)
            if observed != source["sha256"]:
                raise ValueError(f"upstream PDF hash mismatch: {month}")
            staged_path = raw_root / f"{month}.pdf"
            shutil.copyfile(upstream_path, staged_path)
            if _sha_path(staged_path) != observed:
                raise ValueError(f"copied PDF hash mismatch: {month}")
            pages_by_month[month] = extract_pages(staged_path.read_bytes())
            final_path = final_raw_root / staged_path.name
            source_by_month[month] = {
                "source_url": str(source["source_url"]),
                "source_path": str(final_path),
                "source_sha256": observed,
            }
            report_rows.append(
                {
                    "report_month": month,
                    "source_url": str(source["source_url"]),
                    "upstream_source_path": str(upstream_path.resolve()),
                    "source_path": str(final_path),
                    "source_sha256": observed,
                    "hash_preserved": True,
                }
            )

        events: list[dict[str, Any]] = []
        claim_rows: list[dict[str, Any]] = []
        for claim in covered.sort_values(["month", "channel"]).to_dict("records"):
            parsed = parse_report_event_rows(
                pages_by_month[claim["month"]],
                report_month=claim["month"],
                channel=claim["channel"],
                **source_by_month[claim["month"]],
            )
            if not parsed:
                raise ValueError(
                    f"covered claim has no explicit source rows; zero inference rejected: {claim['gap_id']}"
                )
            events.extend(parsed)
            input_occurrences = int(claim["event_rows"])
            claim_rows.append(
                {
                    "gap_id": claim["gap_id"],
                    "report_month": claim["month"],
                    "channel": claim["channel"],
                    "input_date_occurrences": input_occurrences,
                    "explicit_report_rows": len(parsed),
                    "row_minus_occurrence_delta": len(parsed) - input_occurrences,
                    "explicit_nonzero_row_set": True,
                    "zero_inferred": False,
                    "source_sha256": source_by_month[claim["month"]]["source_sha256"],
                }
            )

        frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
        if frame.empty:
            raise ValueError("no report events were extracted")
        if frame["event_id"].duplicated().any() or frame["source_record_sha256"].duplicated().any():
            raise ValueError("event and source-record identities must be unique")
        if frame["known_at"].ne("").any() or frame["ric"].ne("").any():
            raise ValueError("known_at and RIC must remain unassigned")
        frame = frame.sort_values(
            ["report_month", "channel", "source_row_ordinal", "source_record_sha256"]
        ).reset_index(drop=True)
        frame.to_parquet(staging / "report_events.parquet", index=False)
        frame.to_csv(staging / "report_events.csv", index=False)
        pd.DataFrame(report_rows).to_csv(staging / "raw_report_inventory.csv", index=False)
        claim_frame = pd.DataFrame(claim_rows)
        claim_frame.to_csv(staging / "claim_row_reconciliation.csv", index=False)

        mapping_plan = frame[
            [
                "event_id",
                "source_record_sha256",
                "report_month",
                "channel",
                "event_date",
                "raw_code",
                "normalized_local_code",
                "instrument_class",
            ]
        ].copy()
        mapping_plan["ric"] = ""
        mapping_plan["mapping_complete"] = False
        mapping_plan["required_evidence"] = (
            "independent point-in-time official identifier mapping for this exact instrument class and date"
        )
        mapping_plan["next_action"] = (
            "resolve without Refinitiv and independently validate before any shared-history admission"
        )
        mapping_plan.to_csv(staging / "mapping_completion_plan.csv", index=False)

        timestamp_plan = frame[
            ["event_id", "source_record_sha256", "report_month", "channel", "event_date"]
        ].copy()
        timestamp_plan["known_at"] = ""
        timestamp_plan["timestamp_complete"] = False
        timestamp_plan["required_evidence"] = (
            "contemporaneous official publication or notice timestamp proving when this row was knowable"
        )
        timestamp_plan["next_action"] = (
            "recover and independently validate source-backed timestamps; do not infer from report month"
        )
        timestamp_plan.to_csv(staging / "timestamp_completion_plan.csv", index=False)

        input_hash_rows = denominator_hash_rows + validation_hash_rows
        pd.DataFrame(input_hash_rows).to_csv(staging / "input_packet_hash_validation.csv", index=False)
        hashes = _artifact_hashes(staging, exclude={"manifest.json", "artifact_hashes.json"})
        hashes_path = staging / "artifact_hashes.json"
        hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "bounded_report_event_candidate_pending_independent_validation",
            "denominator_dir": str(denominator),
            "denominator_manifest_sha256": denominator_manifest_sha,
            "validation_dir": str(validation),
            "validation_manifest_sha256": validation_manifest_sha,
            "source_record_identity_version": SOURCE_RECORD_VERSION,
            "input_denominator_rows": int(len(denominator_rows)),
            "covered_month_channel_claims": int(len(covered)),
            "unique_reports": int(len(report_rows)),
            "explicit_report_rows": int(len(frame)),
            "rows_by_channel": {
                str(key): int(value) for key, value in frame.groupby("channel").size().items()
            },
            "rows_by_instrument_class": {
                str(key): int(value)
                for key, value in frame.groupby("instrument_class").size().items()
            },
            "input_date_occurrences": int(claim_frame["input_date_occurrences"].sum()),
            "row_minus_occurrence_delta": int(claim_frame["row_minus_occurrence_delta"].sum()),
            "claims_with_count_semantic_delta": int(
                claim_frame["row_minus_occurrence_delta"].ne(0).sum()
            ),
            "all_covered_claims_have_explicit_rows": bool(
                claim_frame["explicit_nonzero_row_set"].all()
            ),
            "absence_is_zero": False,
            "zero_inferred": False,
            "known_at_assigned": 0,
            "ric_assigned": 0,
            "events_admissible_to_shared_halt_history": 0,
            "refinitiv_used": False,
            "shared_halt_controller_scorecard_edited": False,
            "independent_exact_row_set_validation_status": "not_run",
            "write_once": True,
            "artifact_hashes_sha256": _sha_path(hashes_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return JapanG4HaltReportEventArtifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=root / "manifest.json",
        events_path=root / "report_events.parquet",
        manifest=manifest,
    )


def verify_report_event_artifact(root: str | Path) -> list[str]:
    path = Path(root)
    issues: list[str] = []
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        inventory_path = path / "artifact_hashes.json"
        if _sha_path(inventory_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("artifact_hash_inventory_digest_mismatch")
        declared = json.loads(inventory_path.read_text(encoding="utf-8"))
        for relative, expected in declared.items():
            if _unsafe_inventory_key(relative):
                issues.append(f"artifact_hash_path_unsafe:{relative}")
                continue
            item = path / relative
            if not item.is_file() or _sha_path(item) != expected:
                issues.append(f"artifact_output_hash_mismatch:{relative}")
        expected_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
        for item in path.rglob("*"):
            if item.is_file() and str(item.relative_to(path)) not in expected_files:
                issues.append(f"undeclared_artifact_output:{item.relative_to(path)}")
        denominator = Path(str(manifest.get("denominator_dir", "")))
        validation = Path(str(manifest.get("validation_dir", "")))
        denominator_sha = _sha_path(denominator / "manifest.json")
        validation_sha = _sha_path(validation / "manifest.json")
        if manifest.get("denominator_manifest_sha256") != denominator_sha:
            issues.append("denominator_manifest_lineage_mismatch")
        if manifest.get("validation_manifest_sha256") != validation_sha:
            issues.append("validation_manifest_lineage_mismatch")
        validation_manifest = json.loads(
            (validation / "manifest.json").read_text(encoding="utf-8")
        )
        if validation_manifest.get("producer_manifest_sha256") != denominator_sha:
            issues.append("denominator_validation_producer_lineage_mismatch")
    except Exception as exc:
        issues.append(f"artifact_verification_error:{type(exc).__name__}")
    return issues


def _section_boundary(line: str) -> bool:
    return bool(
        NEXT_EVENT_HEADING.match(line)
        or NUMBERED_REPORT.match(line)
        or line in {"別 紙", "別紙"}
        or line.startswith("変更日:")
    )


def _classify_instrument(prefix: str, current: str) -> str:
    if not prefix:
        return current
    raw_value = _canonical_text(prefix).replace("＜", "<").replace("＞", ">").casefold()
    if raw_value in {"<外国株>", "<foreign stocks>"}:
        return current or "equity"
    value = _class_key(prefix)
    equity = {
        "市場第一部",
        "1st section",
        "市場第二部",
        "2nd section",
        "マザーズ",
        "mothers",
        "jasdaq standard",
        "jasdaq growth",
        "jasdaqスタンダード",
        "jasdaqグロース",
        "prime",
        "standard",
        "growth",
        "プライム",
        "スタンダード",
        "グロース",
        "tokyo pro market",
    }
    classes = {
        **{key: "equity" for key in equity},
        "etf": "etf",
        "etfs": "etf",
        "etf(jdr)": "etf",
        "投信受益証券": "etf",
        "etn(jdr)": "etn",
        "etns(jdr)": "etn",
        "不動産投信(reit)": "reit",
        "reits": "reit",
        "投資証券": "investment_certificate",
        "investment certificate": "investment_certificate",
        "普通債": "straight_bond",
        "straight bonds": "straight_bond",
        "転換社債型新株予約権付社債": "convertible_bond",
        "convertible bonds": "convertible_bond",
        "有価証券オプション": "equity_option",
        "individual options": "equity_option",
        "指数先物": "index_future",
        "指数先物オプション": "index_option",
        "index options": "index_option",
        "topixオプション": "index_option",
        "topix options": "index_option",
        "国債先物オプション": "jgb_future_option",
        "options on jgb futures": "jgb_future_option",
    }
    if value in classes:
        return classes[value]
    raise ValueError(f"unrecognized instrument-class prefix: {prefix}")


def _class_key(value: str) -> str:
    result = _canonical_text(value).replace("（", "(").replace("）", ")")
    result = re.sub(r"\s*([()])\s*", r"\1", result)
    result = re.sub(r"\s*[<＜]外国株[>＞]\s*", "", result).strip()
    return result.casefold()


def _issue_name(body: str, channel: str, range_start: int | None) -> str:
    if channel == DELISTING_CHANNEL:
        return _canonical_text(body)
    end = range_start if range_start is not None else len(body)
    for marker in ("当該銘柄", "終日", "期間売買停止", "売買停止"):
        position = body.find(marker)
        if position >= 0:
            end = min(end, position)
    return _canonical_text(body[:end])


def _halt_times(body: str, event_date: date) -> dict[str, Any]:
    match = TIME_RANGE.search(body)
    if not match:
        single = list(re.finditer(TIME_TOKEN, body))
        if not single:
            return _blank_times()
        printed = _canonical_time(single[-1].group(0))
        return {
            "start_printed": printed,
            "end_printed": "",
            "start_jst": datetime.combine(event_date, _parse_time(printed))
            .replace(tzinfo=_jst())
            .isoformat(),
            "end_jst": "",
            "range_start": single[-1].start(),
        }
    start_printed = _canonical_time(match.group("start"))
    end_clock = _canonical_time(match.group("end"))
    end_date_printed = _canonical_date_token(match.group("end_date")) if match.group("end_date") else ""
    end_printed = f"{end_date_printed} {end_clock}".strip()
    start_clock = _parse_time(start_printed)
    end_clock_value = _parse_time(end_clock)
    end_day = _resolve_end_date(end_date_printed, event_date) if end_date_printed else event_date
    return {
        "start_printed": start_printed,
        "end_printed": end_printed,
        "start_jst": datetime.combine(event_date, start_clock).replace(tzinfo=_jst()).isoformat(),
        "end_jst": datetime.combine(end_day, end_clock_value).replace(tzinfo=_jst()).isoformat(),
        "range_start": match.start(),
    }


def _blank_times() -> dict[str, Any]:
    return {
        "start_printed": "",
        "end_printed": "",
        "start_jst": "",
        "end_jst": "",
        "range_start": None,
    }


def _jst() -> timezone:
    return timezone(timedelta(hours=9))


def _parse_date(value: str) -> date:
    parts = [int(item) for item in re.findall(r"\d+", value)]
    return date(parts[0], parts[1], parts[2])


def _resolve_end_date(value: str, start: date) -> date:
    parsed = _parse_date(value)
    return parsed if parsed else start


def _parse_time(value: str) -> time:
    hour, minute = [int(item) for item in re.findall(r"\d+", value)]
    return time(hour, minute)


def _canonical_time(value: str) -> str:
    hour, minute = [int(item) for item in re.findall(r"\d+", value)]
    return f"{hour:02d}:{minute:02d}"


def _canonical_date_token(value: str) -> str:
    parsed = _parse_date(value)
    return parsed.isoformat()


def _canonical_text(value: str) -> str:
    result = unicodedata.normalize("NFKC", str(value))
    result = " ".join(result.split())
    result = re.sub(r"\s+([),.;])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    result = re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff>])(?=[A-Za-z])", " ", result)
    result = re.sub(r"(?<=[A-Za-z.)])(?=\d{1,2}:\d{2})", " ", result)
    result = re.sub(r"\bFun d\b", "Fund", result)
    return result.strip()


def _extract_pages(body: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages]


def _verify_packet(root: Path, packet: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inventory_path = root / "artifact_hashes.json"
    inventory_sha = _sha_path(inventory_path)
    if inventory_sha != manifest.get("artifact_hashes_sha256"):
        raise ValueError(f"{packet} artifact hash inventory mismatch")
    declared = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = [
        {
            "packet": packet,
            "path": "artifact_hashes.json",
            "expected_sha256": manifest.get("artifact_hashes_sha256", ""),
            "observed_sha256": inventory_sha,
            "hash_match": True,
        }
    ]
    for relative, expected in sorted(declared.items()):
        if _unsafe_inventory_key(relative):
            raise ValueError(f"{packet} artifact hash inventory contains unsafe path: {relative}")
        path = root / relative
        observed = _sha_path(path) if path.is_file() else ""
        rows.append(
            {
                "packet": packet,
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_match": observed == expected,
            }
        )
        if observed != expected:
            raise ValueError(f"{packet} artifact file hash mismatch: {relative}")
    expected_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
    extras = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) not in expected_files
    ]
    if extras:
        raise ValueError(f"{packet} packet contains undeclared files: {extras[0]}")
    return manifest, rows


def _unsafe_inventory_key(value: object) -> bool:
    key = str(value)
    path = Path(key)
    return not key or path.is_absolute() or ".." in path.parts


def _bools(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
