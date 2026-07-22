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
import pdfplumber


HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
EXPECTED_INPUT_ROWS = 366
EXPECTED_CLAIMS = 194
EXPECTED_REPORTS = 155
IDENTITY_VERSION = "jpx-g4-report-row-v1"
EVENT_VERSION = "jpx-g4-report-event-v1"

SECTION_NAMES = {
    HALT_CHANNEL: "Suspension of Trading",
    DELISTING_CHANNEL: "Delisting",
}
EVENT_KINDS = {
    HALT_CHANNEL: "trading_halt_report_row",
    DELISTING_CHANNEL: "delisting_report_row",
}
DATE_EXPRESSION = r"20\d{2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{1,2}"
CLOCK_EXPRESSION = r"(?:[01]?\d|2[0-3])\s*[:：;.]\s*[0-5]\d"
DATED_ROW = re.compile(rf"(?P<date>{DATE_EXPRESSION})\s+(?P<remainder>.+)$")
LOCAL_CODE = re.compile(r"^(?P<value>[0-9][0-9A-Z]{3,9})(?:\s+|$)")
CLOCK_INTERVAL = re.compile(
    rf"(?P<open>{CLOCK_EXPRESSION})\s*[～~〜]\s*"
    rf"(?:(?P<close_date>{DATE_EXPRESSION})\s*)?(?P<close>{CLOCK_EXPRESSION})"
)

IndependentPageExtractor = Callable[[bytes], list[str]]

CORE_FIELDS = [
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
    "source_record_sha256",
]


@dataclass(frozen=True)
class JapanG4HaltReportEventValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def independently_reparse_report_rows(
    pages: list[str],
    *,
    report_month: str,
    channel: str,
    source_url: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Independent page-state parser for exact report-row validation."""
    if channel not in SECTION_NAMES:
        raise ValueError(f"unsupported independent parser channel: {channel}")
    pd.Period(report_month, freq="M")
    target = SECTION_NAMES[channel].casefold()
    active = False
    heading_count = 0
    current_class = ""
    rows: list[dict[str, Any]] = []

    for page_number, page in enumerate(pages, start=1):
        for raw_line in page.splitlines():
            line = _independent_normalize_text(raw_line)
            if not line:
                continue
            if line.casefold() == target:
                heading_count += 1
                active = True
                continue
            if not active:
                continue
            if _independent_end_of_section(line, target):
                active = False
                continue
            row_match = DATED_ROW.search(line)
            if row_match is None:
                continue
            label = line[: row_match.start()].strip()
            current_class = _independent_instrument_class(label, current_class)
            if not current_class:
                raise ValueError(f"independent class missing: {report_month} {channel} {line}")

            event_day = _independent_date(row_match.group("date"))
            remainder = row_match.group("remainder").strip()
            code_match = LOCAL_CODE.match(remainder)
            if code_match is not None:
                raw_code = code_match.group("value")
                body = remainder[code_match.end() :].strip()
            elif remainder.startswith("全銘柄"):
                raw_code = "全銘柄"
                body = remainder[len("全銘柄") :].strip()
            else:
                raw_code = ""
                body = remainder
            normalized_code = (
                unicodedata.normalize("NFKC", raw_code).strip().upper()
                if re.fullmatch(r"[0-9][0-9A-Z]{3,9}", raw_code)
                else ""
            )
            timing = _independent_timing(body, event_day) if channel == HALT_CHANNEL else _empty_timing()
            issue_name = _independent_issue_name(body, channel, timing["cut"])
            if not issue_name and raw_code:
                issue_name = raw_code
            if not issue_name:
                raise ValueError(f"independent issue name missing: {report_month} {channel} {line}")
            ordinal = len(rows) + 1
            identity = {
                "version": IDENTITY_VERSION,
                "report_month": report_month,
                "channel": channel,
                "source_sha256": source_sha256,
                "source_page": page_number,
                "source_row_ordinal": ordinal,
                "event_date": event_day.isoformat(),
                "raw_code": raw_code,
                "normalized_local_code": normalized_code,
                "instrument_class": current_class,
                "issue_name": issue_name,
                "halt_start_printed": timing["open_printed"],
                "halt_end_printed": timing["close_printed"],
            }
            row_sha = _independent_record_sha(identity)
            rows.append(
                {
                    "event_id": hashlib.sha256(
                        f"{EVENT_VERSION}|{row_sha}".encode("utf-8")
                    ).hexdigest()[:24],
                    "report_month": report_month,
                    "channel": channel,
                    "event_type": EVENT_KINDS[channel],
                    "event_date": event_day.isoformat(),
                    "raw_code": raw_code,
                    "normalized_local_code": normalized_code,
                    "instrument_class": current_class,
                    "issue_name": issue_name,
                    "halt_start_printed": timing["open_printed"],
                    "halt_end_printed": timing["close_printed"],
                    "halt_start_jst": timing["open_jst"],
                    "halt_end_jst": timing["close_jst"],
                    "source_url": source_url,
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "source_page": page_number,
                    "source_row_ordinal": ordinal,
                    "source_record_sha256": row_sha,
                }
            )
    if heading_count != 1:
        raise ValueError(
            f"independent parser expected one {channel} heading in {report_month}, got {heading_count}"
        )
    return rows


def validate_japan_g4_halt_report_events(
    *,
    run_id: str,
    producer_dir: str | Path,
    denominator_dir: str | Path,
    denominator_validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_report_events_validation",
    page_extractor: IndependentPageExtractor | None = None,
    require_expected_shape: bool = True,
) -> JapanG4HaltReportEventValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    producer = Path(producer_dir).resolve()
    denominator = Path(denominator_dir).resolve()
    upstream_validation = Path(denominator_validation_dir).resolve()
    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once independent validation artifact already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    issues: list[dict[str, str]] = []

    try:
        producer_manifest, producer_hash_rows = _audit_packet(producer, "producer", issues)
        denominator_manifest, denominator_hash_rows = _audit_packet(
            denominator, "denominator", issues
        )
        validation_manifest, validation_hash_rows = _audit_packet(
            upstream_validation, "denominator_validation", issues
        )
        producer_manifest_sha = _sha_path(producer / "manifest.json")
        denominator_manifest_sha = _sha_path(denominator / "manifest.json")
        denominator_validation_manifest_sha = _sha_path(
            upstream_validation / "manifest.json"
        )
        if validation_manifest.get("status") != "pass_partial_component":
            _issue(issues, "upstream_validation_not_passed", "manifest", str(validation_manifest.get("status", "")))
        if producer_manifest.get("denominator_manifest_sha256") != denominator_manifest_sha:
            _issue(
                issues,
                "event_producer_denominator_lineage_mismatch",
                "producer_manifest",
                str(producer_manifest.get("denominator_manifest_sha256", "")),
            )
        if producer_manifest.get("validation_manifest_sha256") != denominator_validation_manifest_sha:
            _issue(
                issues,
                "event_producer_validation_lineage_mismatch",
                "producer_manifest",
                str(producer_manifest.get("validation_manifest_sha256", "")),
            )
        if validation_manifest.get("producer_manifest_sha256") != denominator_manifest_sha:
            _issue(
                issues,
                "denominator_validation_producer_lineage_mismatch",
                "denominator_validation_manifest",
                str(validation_manifest.get("producer_manifest_sha256", "")),
            )

        source_input = _read_csv(denominator / "input_denominator.csv", issues, "denominator_input")
        copied_input = _read_csv(producer / "inputs" / "input_denominator.csv", issues, "producer_input_copy")
        coverage = _read_csv(denominator / "month_channel_coverage.csv", issues, "coverage")
        upstream_claims = _read_csv(
            upstream_validation / "claim_reconciliation.csv", issues, "upstream_claims"
        )
        producer_claims = _read_csv(
            producer / "claim_row_reconciliation.csv", issues, "producer_claims"
        )
        report_inventory = _read_csv(
            producer / "raw_report_inventory.csv", issues, "report_inventory"
        )
        producer_events = _read_parquet(producer / "report_events.parquet", issues, "producer_events")
        producer_csv = _read_csv(producer / "report_events.csv", issues, "producer_events_csv")

        input_checks = _validate_inputs(
            source_input,
            copied_input,
            coverage,
            upstream_claims,
            producer_claims,
            report_inventory,
            producer_manifest,
            require_expected_shape,
            issues,
        )
        pd.DataFrame(input_checks).to_csv(staging / "input_denominator_validation.csv", index=False)

        _validate_producer_event_files(producer_events, producer_csv, producer, issues)
        extractor = page_extractor or _extract_pages_with_pdfplumber
        independent_rows: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []
        claim_rows: list[dict[str, Any]] = []
        covered = coverage.loc[_truths(coverage.get("covered", pd.Series(dtype=str)))].copy()
        inventory_by_month = report_inventory.set_index("report_month") if not report_inventory.empty else pd.DataFrame()
        producer_by_claim = {
            (str(month), str(channel)): group.copy()
            for (month, channel), group in producer_events.groupby(["report_month", "channel"])
        } if not producer_events.empty else {}

        for month in sorted(set(covered.get("month", pd.Series(dtype=str)))):
            if inventory_by_month.empty or month not in inventory_by_month.index:
                _issue(issues, "report_inventory_month_missing", month, "")
                continue
            inventory_row = inventory_by_month.loc[month]
            if isinstance(inventory_row, pd.DataFrame):
                _issue(issues, "report_inventory_month_duplicate", month, str(len(inventory_row)))
                continue
            pdf_path = producer / "raw" / "lifecycle" / f"{month}.pdf"
            observed_sha = _sha_path(pdf_path) if pdf_path.is_file() else ""
            expected_sha = str(inventory_row.get("source_sha256", ""))
            hash_match = bool(observed_sha and observed_sha == expected_sha)
            if not hash_match:
                _issue(issues, "raw_report_hash_mismatch", month, observed_sha)
                report_rows.append(
                    {
                        "report_month": month,
                        "source_path": str(pdf_path),
                        "expected_sha256": expected_sha,
                        "observed_sha256": observed_sha,
                        "hash_match": False,
                        "channels_reparsed": 0,
                        "independent_rows": 0,
                    }
                )
                continue
            try:
                pages = extractor(pdf_path.read_bytes())
            except Exception as exc:
                _issue(issues, "independent_pdf_extract_error", month, type(exc).__name__)
                continue
            month_total = 0
            month_claims = covered.loc[covered["month"].eq(month)]
            for claim in month_claims.to_dict("records"):
                channel = claim["channel"]
                try:
                    parsed = independently_reparse_report_rows(
                        pages,
                        report_month=month,
                        channel=channel,
                        source_url=str(inventory_row.get("source_url", "")),
                        source_path=str(pdf_path),
                        source_sha256=observed_sha,
                    )
                except Exception as exc:
                    _issue(
                        issues,
                        "independent_report_parse_error",
                        f"{month}|{channel}",
                        f"{type(exc).__name__}:{exc}",
                    )
                    parsed = []
                independent_rows.extend(parsed)
                month_total += len(parsed)
                produced = producer_by_claim.get(
                    (month, channel), pd.DataFrame(columns=CORE_FIELDS)
                )
                produced_set = set(produced.get("source_record_sha256", pd.Series(dtype=str)))
                independent_set = {row["source_record_sha256"] for row in parsed}
                exact_set = produced_set == independent_set and bool(independent_set)
                if not independent_set:
                    _issue(issues, "inferred_zero_or_empty_claim_rejected", f"{month}|{channel}", "")
                if produced_set != independent_set:
                    _issue(
                        issues,
                        "claim_exact_row_set_mismatch",
                        f"{month}|{channel}",
                        f"producer={len(produced_set)} independent={len(independent_set)}",
                    )
                claim_rows.append(
                    {
                        "gap_id": claim["gap_id"],
                        "report_month": month,
                        "channel": channel,
                        "input_date_occurrences": int(claim["event_rows"]),
                        "producer_rows": len(produced),
                        "independent_rows": len(parsed),
                        "producer_unique_record_shas": len(produced_set),
                        "independent_unique_record_shas": len(independent_set),
                        "exact_nonempty_row_set": exact_set,
                        "missing_record_shas": "|".join(sorted(produced_set - independent_set)),
                        "unexpected_record_shas": "|".join(sorted(independent_set - produced_set)),
                        "zero_inferred": False,
                    }
                )
            report_rows.append(
                {
                    "report_month": month,
                    "source_path": str(pdf_path),
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "hash_match": hash_match,
                    "channels_reparsed": int(len(month_claims)),
                    "independent_rows": month_total,
                }
            )

        independent = pd.DataFrame(independent_rows, columns=CORE_FIELDS)
        independent.to_parquet(staging / "independent_report_rows.parquet", index=False)
        report_frame = pd.DataFrame(report_rows)
        report_frame.to_csv(staging / "report_reparse_validation.csv", index=False)
        claim_frame = pd.DataFrame(claim_rows)
        claim_frame.to_csv(staging / "claim_exact_row_set_reconciliation.csv", index=False)

        row_reconciliation = _exact_row_reconciliation(producer_events, independent, issues)
        row_reconciliation.to_csv(staging / "row_field_reconciliation.csv", index=False)
        _validate_record_hashes(producer_events, issues)
        _validate_completion_plans(producer, producer_events, issues)

        all_hash_rows = producer_hash_rows + denominator_hash_rows + validation_hash_rows
        pd.DataFrame(all_hash_rows).to_csv(staging / "all_input_hash_validation.csv", index=False)
        pd.DataFrame(issues, columns=["issue_code", "scope", "detail"]).to_csv(
            staging / "validation_issues.csv", index=False
        )
        hashes = _artifact_hashes(staging, {"manifest.json", "artifact_hashes.json"})
        hashes_path = staging / "artifact_hashes.json"
        hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exact_rows = int(row_reconciliation.get("exact_field_match", pd.Series(dtype=bool)).sum())
        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass_bounded_report_event_component" if not issues else "failed_validation",
            "producer_dir": str(producer),
            "producer_manifest_sha256": producer_manifest_sha,
            "denominator_dir": str(denominator),
            "denominator_manifest_sha256": denominator_manifest_sha,
            "denominator_validation_dir": str(upstream_validation),
            "denominator_validation_manifest_sha256": denominator_validation_manifest_sha,
            "input_denominator_rows_verified": int(len(source_input)),
            "covered_claims_reparsed": int(len(claim_frame)),
            "reports_reparsed": int(len(report_frame)),
            "producer_report_rows": int(len(producer_events)),
            "independent_report_rows": int(len(independent)),
            "exact_field_reconciled_rows": exact_rows,
            "exact_claim_row_sets": int(
                claim_frame.get("exact_nonempty_row_set", pd.Series(dtype=bool)).sum()
            ),
            "all_report_hashes_verified": bool(
                not report_frame.empty and report_frame["hash_match"].all()
            ),
            "all_input_hashes_verified": bool(
                all_hash_rows and all(bool(row["hash_match"]) for row in all_hash_rows)
            ),
            "absence_is_zero": False,
            "inferred_zero_rejected": True,
            "known_at_assigned": 0,
            "ric_assigned": 0,
            "refinitiv_used": False,
            "shared_halt_controller_scorecard_edited": False,
            "validation_issue_count": int(len(issues)),
            "validation_issue_codes": sorted({row["issue_code"] for row in issues}),
            "write_once": True,
            "artifact_hashes_sha256": _sha_path(hashes_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return JapanG4HaltReportEventValidationArtifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=root / "manifest.json",
        manifest=manifest,
    )


def verify_report_event_validation_artifact(root: str | Path) -> list[str]:
    path = Path(root)
    problems: list[str] = []
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        hashes_path = path / "artifact_hashes.json"
        if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
            problems.append("artifact_hash_inventory_digest_mismatch")
        declared = json.loads(hashes_path.read_text(encoding="utf-8"))
        for relative, expected in declared.items():
            if _unsafe_inventory_key(relative):
                problems.append(f"artifact_hash_path_unsafe:{relative}")
                continue
            item = path / relative
            if not item.is_file() or _sha_path(item) != expected:
                problems.append(f"artifact_output_hash_mismatch:{relative}")
        allowed = set(declared) | {"manifest.json", "artifact_hashes.json"}
        for item in path.rglob("*"):
            if item.is_file() and str(item.relative_to(path)) not in allowed:
                problems.append(f"undeclared_artifact_output:{item.relative_to(path)}")
        producer = Path(str(manifest.get("producer_dir", "")))
        denominator = Path(str(manifest.get("denominator_dir", "")))
        denominator_validation = Path(
            str(manifest.get("denominator_validation_dir", ""))
        )
        producer_sha = _sha_path(producer / "manifest.json")
        denominator_sha = _sha_path(denominator / "manifest.json")
        denominator_validation_sha = _sha_path(
            denominator_validation / "manifest.json"
        )
        if manifest.get("producer_manifest_sha256") != producer_sha:
            problems.append("producer_manifest_lineage_mismatch")
        if manifest.get("denominator_manifest_sha256") != denominator_sha:
            problems.append("denominator_manifest_lineage_mismatch")
        if (
            manifest.get("denominator_validation_manifest_sha256")
            != denominator_validation_sha
        ):
            problems.append("denominator_validation_manifest_lineage_mismatch")
        producer_manifest = json.loads(
            (producer / "manifest.json").read_text(encoding="utf-8")
        )
        denominator_validation_manifest = json.loads(
            (denominator_validation / "manifest.json").read_text(encoding="utf-8")
        )
        if producer_manifest.get("denominator_manifest_sha256") != denominator_sha:
            problems.append("event_manifest_denominator_lineage_disagreement")
        if (
            denominator_validation_manifest.get("producer_manifest_sha256")
            != denominator_sha
        ):
            problems.append("denominator_validation_producer_lineage_mismatch")
    except Exception as exc:
        problems.append(f"artifact_verification_error:{type(exc).__name__}")
    return problems


def _validate_inputs(
    source: pd.DataFrame,
    copied: pd.DataFrame,
    coverage: pd.DataFrame,
    upstream_claims: pd.DataFrame,
    producer_claims: pd.DataFrame,
    reports: pd.DataFrame,
    producer_manifest: dict[str, Any],
    require_expected: bool,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            _issue(issues, f"{name}_failed", "input", detail)

    check(
        "denominator_exact_copy",
        _frame_records(source) == _frame_records(copied),
        f"source={len(source)} copied={len(copied)}",
    )
    check(
        "denominator_row_count",
        len(source) == EXPECTED_INPUT_ROWS if require_expected else len(source) > 0,
        str(len(source)),
    )
    check(
        "denominator_gap_ids_unique",
        "gap_id" in source and not source["gap_id"].duplicated().any(),
        str(source.get("gap_id", pd.Series(dtype=str)).nunique()),
    )
    covered = coverage.loc[_truths(coverage.get("covered", pd.Series(dtype=str)))]
    check(
        "covered_claim_count",
        len(covered) == EXPECTED_CLAIMS if require_expected else len(covered) > 0,
        str(len(covered)),
    )
    check(
        "upstream_claim_count",
        len(upstream_claims) == EXPECTED_CLAIMS if require_expected else len(upstream_claims) > 0,
        str(len(upstream_claims)),
    )
    check(
        "producer_claim_count",
        len(producer_claims) == EXPECTED_CLAIMS if require_expected else len(producer_claims) > 0,
        str(len(producer_claims)),
    )
    check(
        "report_count",
        len(reports) == EXPECTED_REPORTS if require_expected else len(reports) > 0,
        str(len(reports)),
    )
    check(
        "claim_gap_sets_equal",
        set(covered.get("gap_id", []))
        == set(upstream_claims.get("gap_id", []))
        == set(producer_claims.get("gap_id", [])),
        "coverage/upstream-validation/producer",
    )
    nonzero = pd.to_numeric(producer_claims.get("explicit_report_rows"), errors="coerce").gt(0)
    no_inference = ~_truths(producer_claims.get("zero_inferred", pd.Series(dtype=str)))
    check(
        "no_inferred_zero",
        len(producer_claims) > 0 and nonzero.all() and no_inference.all(),
        "every covered claim must have a nonempty explicit row set",
    )
    check(
        "producer_manifest_zero_policy",
        producer_manifest.get("absence_is_zero") is False
        and producer_manifest.get("zero_inferred") is False,
        "absence_is_zero=false; zero_inferred=false",
    )
    return checks


def _validate_producer_event_files(
    parquet: pd.DataFrame,
    csv_frame: pd.DataFrame,
    producer: Path,
    issues: list[dict[str, str]],
) -> None:
    if _frame_records(parquet.astype(str).fillna("")) != _frame_records(csv_frame.astype(str).fillna("")):
        _issue(issues, "producer_csv_parquet_row_mismatch", "report_events", "")
    required = set(CORE_FIELDS) | {
        "known_at",
        "ric",
        "mapping_status",
        "timestamp_status",
        "admissible_to_shared_halt_history",
    }
    missing = required - set(parquet.columns)
    if missing:
        _issue(issues, "producer_event_columns_missing", "report_events", "|".join(sorted(missing)))
        return
    if parquet["event_id"].duplicated().any() or parquet["source_record_sha256"].duplicated().any():
        _issue(issues, "producer_event_identity_duplicate", "report_events", "")
    if parquet["known_at"].fillna("").astype(str).ne("").any():
        _issue(issues, "known_at_assigned_without_source", "report_events", "")
    if parquet["ric"].fillna("").astype(str).ne("").any():
        _issue(issues, "ric_assigned_without_source", "report_events", "")
    if _truths(parquet["admissible_to_shared_halt_history"]).any():
        _issue(issues, "unvalidated_event_marked_admissible", "report_events", "")
    expected_root = (producer / "raw" / "lifecycle").resolve()
    for value in parquet["source_path"].astype(str).unique():
        try:
            Path(value).resolve().relative_to(expected_root)
        except ValueError:
            _issue(issues, "event_source_path_outside_artifact", value, str(expected_root))


def _exact_row_reconciliation(
    producer: pd.DataFrame,
    independent: pd.DataFrame,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    left = producer[CORE_FIELDS].copy() if not producer.empty else pd.DataFrame(columns=CORE_FIELDS)
    right = independent[CORE_FIELDS].copy() if not independent.empty else pd.DataFrame(columns=CORE_FIELDS)
    left = left.add_prefix("producer_")
    right = right.add_prefix("independent_")
    merged = left.merge(
        right,
        left_on="producer_source_record_sha256",
        right_on="independent_source_record_sha256",
        how="outer",
        indicator=True,
    )
    comparisons = []
    for row in merged.to_dict("records"):
        present = row["_merge"] == "both"
        mismatched = []
        if present:
            for field in CORE_FIELDS:
                if str(row.get(f"producer_{field}", "")) != str(
                    row.get(f"independent_{field}", "")
                ):
                    mismatched.append(field)
        comparisons.append(
            {
                "source_record_sha256": row.get("producer_source_record_sha256")
                or row.get("independent_source_record_sha256"),
                "set_membership": row["_merge"],
                "exact_field_match": present and not mismatched,
                "mismatched_fields": "|".join(mismatched),
                "report_month": row.get("producer_report_month")
                or row.get("independent_report_month"),
                "channel": row.get("producer_channel") or row.get("independent_channel"),
                "raw_code": row.get("producer_raw_code") or row.get("independent_raw_code"),
            }
        )
    result = pd.DataFrame(comparisons)
    if len(result) != len(producer) or result.empty or not result["exact_field_match"].all():
        _issue(
            issues,
            "global_exact_row_field_reconciliation_failed",
            "report_events",
            f"producer={len(producer)} independent={len(independent)} exact={int(result.get('exact_field_match', pd.Series(dtype=bool)).sum())}",
        )
    return result


def _validate_record_hashes(events: pd.DataFrame, issues: list[dict[str, str]]) -> None:
    if events.empty:
        return
    for row in events.to_dict("records"):
        identity = {
            "version": IDENTITY_VERSION,
            "report_month": str(row["report_month"]),
            "channel": str(row["channel"]),
            "source_sha256": str(row["source_sha256"]),
            "source_page": int(row["source_page"]),
            "source_row_ordinal": int(row["source_row_ordinal"]),
            "event_date": str(row["event_date"]),
            "raw_code": str(row["raw_code"]),
            "normalized_local_code": str(row["normalized_local_code"]),
            "instrument_class": str(row["instrument_class"]),
            "issue_name": str(row["issue_name"]),
            "halt_start_printed": str(row["halt_start_printed"]),
            "halt_end_printed": str(row["halt_end_printed"]),
        }
        observed = _independent_record_sha(identity)
        expected_event = hashlib.sha256(
            f"{EVENT_VERSION}|{observed}".encode("utf-8")
        ).hexdigest()[:24]
        if observed != row["source_record_sha256"]:
            _issue(issues, "source_record_hash_invalid", str(row["event_id"]), observed)
        if expected_event != row["event_id"]:
            _issue(issues, "event_id_hash_invalid", str(row["event_id"]), expected_event)


def _validate_completion_plans(
    producer: Path, events: pd.DataFrame, issues: list[dict[str, str]]
) -> None:
    event_ids = set(events.get("event_id", []))
    for name, completion, blank_field in (
        ("mapping_completion_plan.csv", "mapping_complete", "ric"),
        ("timestamp_completion_plan.csv", "timestamp_complete", "known_at"),
    ):
        plan = _read_csv(producer / name, issues, name)
        if set(plan.get("event_id", [])) != event_ids or len(plan) != len(events):
            _issue(issues, "completion_plan_event_set_mismatch", name, str(len(plan)))
        if _truths(plan.get(completion, pd.Series(dtype=str))).any():
            _issue(issues, "unsupported_completion_claim", name, completion)
        if plan.get(blank_field, pd.Series(dtype=str)).fillna("").astype(str).ne("").any():
            _issue(issues, "unsupported_completion_value", name, blank_field)


def _independent_end_of_section(line: str, target: str) -> bool:
    folded = line.casefold()
    if folded == target:
        return False
    heading_starts = (
        "new listings",
        "delisting",
        "suspension of trading",
        "assignment to",
        "reassignment",
        "transfer to",
        "name change",
        "change of",
        "change in",
        "additional assignment",
        "exception from",
        "designation as",
        "ex-new",
    )
    return bool(
        folded.startswith(heading_starts)
        or re.match(r"^\d{1,2}(?:-\d+)?\s+.*(?:\(|（)", line)
        or line in {"別 紙", "別紙"}
        or line.startswith("変更日:")
    )


def _independent_instrument_class(label: str, prior: str) -> str:
    if not label:
        return prior
    normalized = _independent_normalize_text(label).replace("＜", "<").replace("＞", ">")
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
        "jgb_future_option": {"国債先物オプション", "options on jgb futures"},
    }
    for instrument_class, values in groups.items():
        if normalized in values:
            return instrument_class
    raise ValueError(f"independent unrecognized instrument label: {label}")


def _independent_timing(body: str, event_day: date) -> dict[str, Any]:
    interval = CLOCK_INTERVAL.search(body)
    jst = timezone(timedelta(hours=9))
    if interval is not None:
        open_clock = _independent_clock(interval.group("open"))
        close_clock = _independent_clock(interval.group("close"))
        close_day = (
            _independent_date(interval.group("close_date"))
            if interval.group("close_date")
            else event_day
        )
        open_printed = open_clock.strftime("%H:%M")
        close_date_printed = close_day.isoformat() if interval.group("close_date") else ""
        close_printed = f"{close_date_printed} {close_clock.strftime('%H:%M')}".strip()
        return {
            "open_printed": open_printed,
            "close_printed": close_printed,
            "open_jst": datetime.combine(event_day, open_clock).replace(tzinfo=jst).isoformat(),
            "close_jst": datetime.combine(close_day, close_clock).replace(tzinfo=jst).isoformat(),
            "cut": interval.start(),
        }
    singles = list(re.finditer(CLOCK_EXPRESSION, body))
    if singles:
        clock = _independent_clock(singles[-1].group(0))
        return {
            "open_printed": clock.strftime("%H:%M"),
            "close_printed": "",
            "open_jst": datetime.combine(event_day, clock).replace(tzinfo=jst).isoformat(),
            "close_jst": "",
            "cut": singles[-1].start(),
        }
    return _empty_timing()


def _empty_timing() -> dict[str, Any]:
    return {"open_printed": "", "close_printed": "", "open_jst": "", "close_jst": "", "cut": None}


def _independent_issue_name(body: str, channel: str, cut: int | None) -> str:
    if channel == DELISTING_CHANNEL:
        return _independent_normalize_text(body)
    end = len(body) if cut is None else cut
    for marker in ("当該銘柄", "終日", "期間売買停止", "売買停止"):
        found = body.find(marker)
        if found >= 0:
            end = min(end, found)
    return _independent_normalize_text(body[:end])


def _independent_record_sha(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _independent_date(value: str) -> date:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    return date(numbers[0], numbers[1], numbers[2])


def _independent_clock(value: str) -> time:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    return time(numbers[0], numbers[1])


def _independent_normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([),.;])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff>])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z.)])(?=\d{1,2}:\d{2})", " ", text)
    text = re.sub(r"\bFun d\b", "Fund", text)
    return text


def _extract_pages_with_pdfplumber(body: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _audit_packet(
    root: Path, packet: str, issues: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        _issue(issues, "packet_manifest_unreadable", packet, type(exc).__name__)
        return {}, []
    inventory_path = root / "artifact_hashes.json"
    inventory_sha = _sha_path(inventory_path) if inventory_path.is_file() else ""
    expected_inventory_sha = str(manifest.get("artifact_hashes_sha256", ""))
    rows = [
        {
            "packet": packet,
            "path": "artifact_hashes.json",
            "expected_sha256": expected_inventory_sha,
            "observed_sha256": inventory_sha,
            "hash_match": inventory_sha == expected_inventory_sha,
        }
    ]
    if inventory_sha != expected_inventory_sha:
        _issue(issues, "packet_hash_inventory_mismatch", packet, inventory_sha)
    try:
        declared = json.loads(inventory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _issue(issues, "packet_hash_inventory_unreadable", packet, type(exc).__name__)
        return manifest, rows
    for relative, expected in sorted(declared.items()):
        if _unsafe_inventory_key(relative):
            rows.append(
                {
                    "packet": packet,
                    "path": relative,
                    "expected_sha256": expected,
                    "observed_sha256": "",
                    "hash_match": False,
                }
            )
            _issue(issues, "packet_hash_path_unsafe", packet, relative)
            continue
        item = root / relative
        observed = _sha_path(item) if item.is_file() else ""
        match = observed == expected
        rows.append(
            {
                "packet": packet,
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_match": match,
            }
        )
        if not match:
            _issue(issues, "packet_output_hash_mismatch", f"{packet}:{relative}", observed)
    declared_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
    for item in root.rglob("*"):
        relative = str(item.relative_to(root))
        if item.is_file() and relative not in declared_files:
            _issue(issues, "packet_undeclared_output", packet, relative)
    return manifest, rows


def _unsafe_inventory_key(value: object) -> bool:
    key = str(value)
    path = Path(key)
    return not key or path.is_absolute() or ".." in path.parts


def _read_csv(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        _issue(issues, "csv_unreadable", scope, type(exc).__name__)
        return pd.DataFrame()


def _read_parquet(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path).fillna("")
    except Exception as exc:
        _issue(issues, "parquet_unreadable", scope, type(exc).__name__)
        return pd.DataFrame()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return []
    normalized = frame.astype(str).fillna("")
    return normalized.sort_values(list(normalized.columns)).to_dict("records")


def _truths(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _issue(issues: list[dict[str, str]], code: str, scope: str, detail: str) -> None:
    issues.append({"issue_code": code, "scope": scope, "detail": detail})


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
