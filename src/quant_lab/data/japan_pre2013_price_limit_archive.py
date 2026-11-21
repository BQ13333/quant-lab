from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from quant_lab.data.jpx_exceptional_price_limit_notices import (
    classify_notice_title,
    parse_notice_page,
)


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"
ARCHIVE_START = "2006-07"
ARCHIVE_END = "2013-08"
POSITIVE_CONTROL_MONTH = "2013-09"
MIGRATED_DETAIL_START = 634
MIGRATED_DETAIL_END = 673
LOWER_BOUNDARY_DETAIL_ID = 633
UPPER_BOUNDARY_DETAIL_ID = 674
JAPANESE_DETAIL_START = 823
JAPANESE_DETAIL_END = 867
JAPANESE_LOWER_BOUNDARY_DETAIL_ID = 822
JAPANESE_UPPER_BOUNDARY_DETAIL_ID = 868
OSE_DETAIL_SCAN_START = 23900
OSE_DETAIL_SCAN_END = 24122
EXPECTED_MONTHS = pd.period_range(ARCHIVE_START, ARCHIVE_END, freq="M")
PRE_2013_BLOCKER = (
    "official issue-specific price-limit archive before 2013-09 is not yet proven complete"
)
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


class Pre2013ArchiveError(RuntimeError):
    """Raised when the acquisition artifact cannot be written truthfully."""


@dataclass(frozen=True)
class Pre2013PriceLimitArchiveArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _Response:
    status: int
    body: bytes
    headers: dict[str, str]


def write_japan_pre2013_price_limit_archive(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/japan_pre2013_price_limit_archive",
    local_artifact_root: str | Path = "artifacts",
    fetcher: Fetcher | None = None,
    max_workers: int = 6,
) -> Pre2013PriceLimitArchiveArtifacts:
    """Acquire public JPX evidence and publish a fail-closed completeness artifact."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"immutable pre-2013 archive run already exists: {root}")
    root.mkdir(parents=True)
    fetch = fetcher or _default_fetch

    specs = _request_specs()
    responses = _fetch_all(specs["url"].tolist(), fetch, max_workers=max_workers)
    try:
        attempts = _build_source_attempts(root, specs, responses)
        migrated_pages, migrated_notices, migrated_events = _build_migrated_evidence(
            attempts, responses
        )
        japanese_pages, japanese_notices = _build_japanese_migrated_evidence(
            attempts, responses
        )
        ose_pages = _build_ose_migrated_evidence(attempts, responses)
        reconciliation = _bilingual_reconciliation(migrated_events, japanese_notices)
        coverage = _build_monthly_coverage(
            attempts, migrated_notices, japanese_notices, reconciliation
        )
        archive_catalogs = _build_official_archive_catalogs(attempts, responses)
        local_audit = _build_local_source_catalog_audit(Path(local_artifact_root).resolve())
        completion_plan = _completion_plan()
        validation_requirements = _independent_validation_requirements()
        denominator = _build_denominator_evidence(
            attempts,
            responses,
            migrated_pages,
            japanese_pages=japanese_pages,
            japanese_notices=japanese_notices,
            ose_pages=ose_pages,
            archive_catalogs=archive_catalogs,
        )
        vendor_plan = _vendor_route_plan()
        references = _source_references()
        completeness = _completeness_result(
            coverage=coverage,
            attempts=attempts,
            pages=migrated_pages,
            notices=migrated_notices,
            events=migrated_events,
            japanese_pages=japanese_pages,
            japanese_notices=japanese_notices,
            ose_pages=ose_pages,
            reconciliation=reconciliation,
            archive_catalogs=archive_catalogs,
        )

        frames = {
            "coverage_months.csv": coverage,
            "source_attempts.csv": attempts,
            "denominator_evidence.csv": denominator,
            "migrated_pages.csv": migrated_pages,
            "migrated_notices.csv": migrated_notices,
            "migrated_events.csv": migrated_events,
            "migrated_japanese_pages.csv": japanese_pages,
            "migrated_japanese_notices.csv": japanese_notices,
            "migrated_ose_pages.csv": ose_pages,
            "bilingual_notice_reconciliation.csv": reconciliation,
            "official_archive_catalogs.csv": archive_catalogs,
            "local_source_catalog_audit.csv": local_audit,
            "completion_plan.csv": completion_plan,
            "independent_validation_requirements.csv": validation_requirements,
            "vendor_route_plan.csv": vendor_plan,
            "source_references.csv": references,
        }
        for filename, frame in frames.items():
            _atomic_csv(root / filename, frame)
        _atomic_json(root / "completeness.json", completeness)
        _atomic_text(root / "summary.md", _summary_markdown(completeness))

        acquisition_complete = bool(
            len(attempts[attempts["scope"] == "research_month"]) == len(EXPECTED_MONTHS) * 4
            and attempts[attempts["scope"] == "research_month"]["request_completed"].all()
            and len(migrated_pages) == MIGRATED_DETAIL_END - MIGRATED_DETAIL_START + 1
            and migrated_pages["request_completed"].all()
            and len(japanese_pages) == JAPANESE_DETAIL_END - JAPANESE_DETAIL_START + 1
            and japanese_pages["request_completed"].all()
            and len(reconciliation) == len(migrated_notices)
            and reconciliation["reconciliation_status"].eq("matched").all()
        )
        outputs = _hash_inventory(root)
        manifest = {
            "schema_version": "2.0-immutable-fail-closed",
            "run_id": run_id,
            "artifact_type": "japan_pre_2013_exceptional_price_limit_archive_acquisition",
            "status": (
                "valid_incomplete_public_denominator"
                if acquisition_complete
                else "valid_incomplete_acquisition_and_public_denominator"
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "immutable": True,
            "official_public_http_calls_performed": True,
            "live_vendor_calls_performed": False,
            "completion_claimed": False,
            "archive_denominator_proven": False,
            "scorecard_eligible": False,
            "promotion_allowed": False,
            "promotion_policy": dict(PROMOTION_POLICY),
            "coverage": {
                "start_month": ARCHIVE_START,
                "end_month": ARCHIVE_END,
                "required_months": len(EXPECTED_MONTHS),
                "months_with_official_denominator": 0,
                "months_with_unresolved_denominator": len(EXPECTED_MONTHS),
            },
            "counts": {
                "source_attempts": len(attempts),
                "research_month_route_attempts": int(
                    attempts["scope"].eq("research_month").sum()
                ),
                "migrated_detail_pages": len(migrated_pages),
                "migrated_notice_pages": len(migrated_notices),
                "migrated_event_rows": len(migrated_events),
                "migrated_japanese_detail_pages": len(japanese_pages),
                "migrated_japanese_notice_pages": len(japanese_notices),
                "migrated_ose_scan_rows": len(ose_pages),
                "bilingual_matched_notice_rows": int(
                    reconciliation["reconciliation_status"].eq("matched").sum()
                ),
                "official_archive_catalog_rows": len(archive_catalogs),
                "local_source_catalog_audit_rows": len(local_audit),
                "completion_plan_rows": len(completion_plan),
                "independent_validation_requirement_rows": len(validation_requirements),
                "vendor_route_rows": len(vendor_plan),
            },
            "remaining_completion_blockers": [PRE_2013_BLOCKER],
            "critical_blockers": [PRE_2013_BLOCKER],
            "output_sha256": outputs,
        }
        _atomic_json(root / "manifest.json", manifest)
        validation = validate_japan_pre2013_price_limit_archive(root)
        if not validation["valid"]:
            raise Pre2013ArchiveError(
                "written artifact failed self-validation: " + "; ".join(validation["issues"])
            )
        return Pre2013PriceLimitArchiveArtifacts(run_id, root, root / "manifest.json", manifest)
    except Exception:
        _remove_tree(root)
        raise


def validate_japan_pre2013_price_limit_archive(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Validate integrity and the fail-closed denominator semantics independently."""
    root = Path(artifact_dir).resolve()
    issues: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "issues": ["missing_outputs:manifest.json"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"valid": False, "issues": [f"unreadable_manifest:{type(exc).__name__}"]}
    is_v2 = str(manifest.get("schema_version", "")).startswith("2.")
    required = {
        "manifest.json",
        "completeness.json",
        "coverage_months.csv",
        "source_attempts.csv",
        "denominator_evidence.csv",
        "migrated_pages.csv",
        "migrated_notices.csv",
        "migrated_events.csv",
        "vendor_route_plan.csv",
        "source_references.csv",
        "summary.md",
    }
    if is_v2:
        required.update(
            {
                "migrated_japanese_pages.csv",
                "migrated_japanese_notices.csv",
                "migrated_ose_pages.csv",
                "bilingual_notice_reconciliation.csv",
                "official_archive_catalogs.csv",
                "local_source_catalog_audit.csv",
                "completion_plan.csv",
                "independent_validation_requirements.csv",
            }
        )
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        return {"valid": False, "issues": [f"missing_outputs:{'|'.join(missing)}"]}

    try:
        completeness = json.loads((root / "completeness.json").read_text(encoding="utf-8"))
        coverage = pd.read_csv(root / "coverage_months.csv", dtype=str, keep_default_na=False)
        attempts = pd.read_csv(root / "source_attempts.csv", dtype=str, keep_default_na=False)
        denominator = pd.read_csv(
            root / "denominator_evidence.csv", dtype=str, keep_default_na=False
        )
        pages = pd.read_csv(root / "migrated_pages.csv", dtype=str, keep_default_na=False)
        notices = pd.read_csv(root / "migrated_notices.csv", dtype=str, keep_default_na=False)
        events = pd.read_csv(root / "migrated_events.csv", dtype=str, keep_default_na=False)
        vendor = pd.read_csv(root / "vendor_route_plan.csv", dtype=str, keep_default_na=False)
        if is_v2:
            japanese_pages = pd.read_csv(
                root / "migrated_japanese_pages.csv", dtype=str, keep_default_na=False
            )
            japanese_notices = pd.read_csv(
                root / "migrated_japanese_notices.csv", dtype=str, keep_default_na=False
            )
            ose_pages = pd.read_csv(
                root / "migrated_ose_pages.csv", dtype=str, keep_default_na=False
            )
            reconciliation = pd.read_csv(
                root / "bilingual_notice_reconciliation.csv",
                dtype=str,
                keep_default_na=False,
            )
            archive_catalogs = pd.read_csv(
                root / "official_archive_catalogs.csv", dtype=str, keep_default_na=False
            )
            completion_plan = pd.read_csv(
                root / "completion_plan.csv", dtype=str, keep_default_na=False
            )
            independent = pd.read_csv(
                root / "independent_validation_requirements.csv",
                dtype=str,
                keep_default_na=False,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return {"valid": False, "issues": [f"unreadable_output:{type(exc).__name__}"]}

    schemas = {
        "coverage_months.csv": (
            coverage,
            {
                "month",
                "official_notice_denominator",
                "official_denominator_proven",
                "missing_notice_count",
            },
        ),
        "source_attempts.csv": (
            attempts,
            {"request_id", "scope", "requested_url", "raw_path", "sha256"},
        ),
        "denominator_evidence.csv": (
            denominator,
            {"evidence_id", "scope", "denominator_proven"},
        ),
        "migrated_pages.csv": (
            pages,
            {"detail_id", "is_exceptional_price_limit"},
        ),
        "migrated_notices.csv": (notices, {"parsed_event_rows"}),
        "vendor_route_plan.csv": (
            vendor,
            {"execution_status", "live_call_performed"},
        ),
    }
    if is_v2:
        schemas.update(
            {
                "migrated_japanese_pages.csv": (
                    japanese_pages,
                    {"detail_id", "publication_date", "event_type", "security_code"},
                ),
                "migrated_japanese_notices.csv": (
                    japanese_notices,
                    {"notice_id", "publication_date", "event_type", "security_code"},
                ),
                "migrated_ose_pages.csv": (
                    ose_pages,
                    {"detail_id", "http_status", "is_equity_exceptional_price_limit"},
                ),
                "bilingual_notice_reconciliation.csv": (
                    reconciliation,
                    {"publication_date", "event_type", "security_code", "reconciliation_status"},
                ),
                "official_archive_catalogs.csv": (
                    archive_catalogs,
                    {"evidence_id", "denominator_proven", "conclusion"},
                ),
                "completion_plan.csv": (
                    completion_plan,
                    {"requirement_id", "query_start", "query_end", "acceptance_gate"},
                ),
                "independent_validation_requirements.csv": (
                    independent,
                    {"validation_id", "mandatory", "failure_result"},
                ),
            }
        )
    for filename, (frame, columns) in schemas.items():
        absent = sorted(columns - set(frame.columns))
        if absent:
            issues.append(f"schema_missing_columns:{filename}:{'|'.join(absent)}")
    if issues:
        return {"valid": False, "issues": sorted(set(issues))}

    if manifest.get("completion_claimed") is not False:
        issues.append("manifest_completion_must_be_false")
    if manifest.get("archive_denominator_proven") is not False:
        issues.append("manifest_archive_denominator_must_be_false")
    if manifest.get("live_vendor_calls_performed") is not False:
        issues.append("manifest_live_vendor_calls_must_be_false")
    if manifest.get("promotion_allowed") is not False:
        issues.append("manifest_promotion_must_be_false")

    expected_months = [str(period) for period in EXPECTED_MONTHS]
    if coverage.get("month", pd.Series(dtype=str)).tolist() != expected_months:
        issues.append("coverage_month_denominator_invalid")
    if len(coverage) and not coverage["official_denominator_proven"].eq("False").all():
        issues.append("coverage_contains_unproven_pass")
    if len(coverage) and not coverage["official_notice_denominator"].eq("").all():
        issues.append("unknown_notice_denominator_must_be_blank")
    if len(coverage) and not coverage["missing_notice_count"].eq("").all():
        issues.append("unknown_missing_notice_count_must_be_blank")
    in_range_denominator = denominator[denominator["evidence_id"].isin({"JPLA-DEN-001", "JPLA-DEN-002"})]
    if len(in_range_denominator) != 2 or not in_range_denominator["denominator_proven"].eq("False").all():
        issues.append("in_range_denominator_evidence_contains_pass")

    research = attempts[attempts.get("scope", "").eq("research_month")]
    expected_research_routes = len(EXPECTED_MONTHS) * (4 if is_v2 else 2)
    if len(research) != expected_research_routes:
        issues.append("research_route_attempt_denominator_invalid")
    if research.get("requested_url", pd.Series(dtype=str)).duplicated().any():
        issues.append("research_route_urls_not_unique")
    for value in attempts.get("requested_url", pd.Series(dtype=str)):
        if urlparse(value).hostname not in {"jpx.co.jp", "www.jpx.co.jp"}:
            issues.append("source_attempt_non_jpx_host")
            break

    expected_ids = [str(value) for value in range(MIGRATED_DETAIL_START, MIGRATED_DETAIL_END + 1)]
    if pages.get("detail_id", pd.Series(dtype=str)).tolist() != expected_ids:
        issues.append("migrated_detail_id_sequence_invalid")
    if len(notices) != int(pages.get("is_exceptional_price_limit", pd.Series(dtype=str)).eq("True").sum()):
        issues.append("migrated_notice_count_inconsistent")
    if len(events) != int(notices.get("parsed_event_rows", pd.Series(dtype=str)).replace("", "0").astype(int).sum()):
        issues.append("migrated_event_count_inconsistent")

    if len(vendor) and not vendor["execution_status"].eq("planned_not_executed").all():
        issues.append("vendor_route_was_marked_executed")
    if len(vendor) and not vendor["live_call_performed"].eq("False").all():
        issues.append("vendor_route_contains_live_call")
    if is_v2:
        expected_japanese_ids = [
            str(value) for value in range(JAPANESE_DETAIL_START, JAPANESE_DETAIL_END + 1)
        ]
        if japanese_pages["detail_id"].tolist() != expected_japanese_ids:
            issues.append("migrated_japanese_detail_id_sequence_invalid")
        if len(japanese_notices) != len(notices):
            issues.append("bilingual_notice_numerator_count_mismatch")
        if len(reconciliation) != len(notices) or not reconciliation[
            "reconciliation_status"
        ].eq("matched").all():
            issues.append("bilingual_notice_reconciliation_incomplete")
        expected_ose_ids = [
            str(value) for value in range(OSE_DETAIL_SCAN_START, OSE_DETAIL_SCAN_END + 1)
        ]
        if ose_pages["detail_id"].tolist() != expected_ose_ids:
            issues.append("migrated_ose_detail_id_sequence_invalid")
        if not ose_pages["is_equity_exceptional_price_limit"].eq("False").all():
            issues.append("ose_scan_contains_unadmitted_equity_notice")
        in_range_catalogs = archive_catalogs[
            archive_catalogs["scope"].eq("2006-07_through_2013-08")
        ]
        if len(in_range_catalogs) == 0 or not in_range_catalogs[
            "denominator_proven"
        ].eq("False").all():
            issues.append("official_archive_catalog_contains_in_range_pass")
        public_recheck = completion_plan[
            completion_plan["requirement_id"].eq("JPLA-COMP-004")
        ]
        pending = completion_plan[
            ~completion_plan["requirement_id"].eq("JPLA-COMP-004")
        ]
        if len(public_recheck) != 1 or not public_recheck["execution_status"].eq(
            "producer_executed_independent_recheck_required"
        ).all():
            issues.append("completion_plan_public_recheck_status_invalid")
        if not pending["execution_status"].eq("required_not_executed").all():
            issues.append("completion_plan_pending_route_status_invalid")
        if not independent["mandatory"].eq("True").all():
            issues.append("independent_validation_contains_optional_gate")

    notice_denominator = completeness.get("notice_denominator", {})
    if completeness.get("completion_claimed") is not False:
        issues.append("completeness_claim_must_be_false")
    if notice_denominator.get("value") is not None:
        issues.append("unknown_notice_denominator_must_be_null")
    if notice_denominator.get("missing_notice_count") is not None:
        issues.append("unknown_missing_notice_count_must_be_null")
    monthly = completeness.get("monthly_partition_denominator", {})
    if monthly.get("required_months") != len(EXPECTED_MONTHS):
        issues.append("completeness_required_months_invalid")
    if monthly.get("months_with_official_denominator") != 0:
        issues.append("completeness_contains_denominator_pass")

    hashes = manifest.get("output_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        issues.append("manifest_hash_inventory_missing")
    else:
        for relative, expected_hash in hashes.items():
            path = (root / relative).resolve()
            if not _is_relative_to(path, root) or not path.is_file():
                issues.append(f"hashed_output_missing:{relative}")
            elif _sha256(path) != expected_hash:
                issues.append(f"hashed_output_mismatch:{relative}")

    raw_rows = attempts[attempts.get("raw_path", "").ne("")]
    for row in raw_rows.to_dict("records"):
        path = (root / row["raw_path"]).resolve()
        if not _is_relative_to(path, root) or not path.is_file():
            issues.append(f"raw_source_missing:{row['request_id']}")
        elif _sha256(path) != row["sha256"]:
            issues.append(f"raw_source_hash_mismatch:{row['request_id']}")

    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "counts": {
            "coverage_months": len(coverage),
            "source_attempts": len(attempts),
            "migrated_pages": len(pages),
            "migrated_notices": len(notices),
            "migrated_events": len(events),
            "vendor_routes": len(vendor),
            **(
                {
                    "migrated_japanese_pages": len(japanese_pages),
                    "migrated_japanese_notices": len(japanese_notices),
                    "migrated_ose_scan_rows": len(ose_pages),
                    "bilingual_reconciliation_rows": len(reconciliation),
                    "official_archive_catalog_rows": len(archive_catalogs),
                    "completion_plan_rows": len(completion_plan),
                }
                if is_v2
                else {}
            ),
        },
    }


def _request_specs() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for period in EXPECTED_MONTHS:
        yyyymm = period.strftime("%Y%m")
        rows.extend(
            [
                {
                    "request_id": f"JPLA-M-{yyyymm}-JSON",
                    "scope": "research_month",
                    "month": str(period),
                    "route_type": "monthly_archive_json",
                    "url": f"{JPX_ROOT}/english/news/news_ym_{yyyymm}.json",
                },
                {
                    "request_id": f"JPLA-M-{yyyymm}-STATIC",
                    "scope": "research_month",
                    "month": str(period),
                    "route_type": "monthly_static_archive",
                    "url": f"{JPX_ROOT}/english/news/static-archives-{yyyymm}.html",
                },
                {
                    "request_id": f"JPLA-M-{yyyymm}-JSON-JA",
                    "scope": "research_month",
                    "month": str(period),
                    "route_type": "monthly_archive_json_ja",
                    "url": f"{JPX_ROOT}/news/news_ym_{yyyymm}.json",
                },
                {
                    "request_id": f"JPLA-M-{yyyymm}-STATIC-JA",
                    "scope": "research_month",
                    "month": str(period),
                    "route_type": "monthly_static_archive_ja",
                    "url": f"{JPX_ROOT}/news/static-archives-{yyyymm}.html",
                },
            ]
        )
    control = POSITIVE_CONTROL_MONTH.replace("-", "")
    rows.extend(
        [
            {
                "request_id": f"JPLA-C-{control}-JSON",
                "scope": "positive_control",
                "month": POSITIVE_CONTROL_MONTH,
                "route_type": "monthly_archive_json",
                "url": f"{JPX_ROOT}/english/news/news_ym_{control}.json",
            },
            {
                "request_id": f"JPLA-C-{control}-STATIC",
                "scope": "positive_control",
                "month": POSITIVE_CONTROL_MONTH,
                "route_type": "monthly_static_archive",
                "url": f"{JPX_ROOT}/english/news/static-archives-{control}.html",
            },
            {
                "request_id": f"JPLA-C-{control}-JSON-JA",
                "scope": "positive_control",
                "month": POSITIVE_CONTROL_MONTH,
                "route_type": "monthly_archive_json_ja",
                "url": f"{JPX_ROOT}/news/news_ym_{control}.json",
            },
            {
                "request_id": f"JPLA-C-{control}-STATIC-JA",
                "scope": "positive_control",
                "month": POSITIVE_CONTROL_MONTH,
                "route_type": "monthly_static_archive_ja",
                "url": f"{JPX_ROOT}/news/static-archives-{control}.html",
            },
        ]
    )
    for detail_id in range(LOWER_BOUNDARY_DETAIL_ID, UPPER_BOUNDARY_DETAIL_ID + 1):
        scope = (
            "migrated_detail_candidate"
            if MIGRATED_DETAIL_START <= detail_id <= MIGRATED_DETAIL_END
            else "migrated_detail_boundary"
        )
        rows.append(
            {
                "request_id": f"JPLA-D-{detail_id:05d}",
                "scope": scope,
                "month": "",
                "route_type": "official_migrated_detail",
                "url": f"{JPX_ROOT}/english/news/detail/detail_{detail_id}.html",
            }
        )
    for detail_id in range(
        JAPANESE_LOWER_BOUNDARY_DETAIL_ID, JAPANESE_UPPER_BOUNDARY_DETAIL_ID + 1
    ):
        scope = (
            "migrated_japanese_detail_candidate"
            if JAPANESE_DETAIL_START <= detail_id <= JAPANESE_DETAIL_END
            else "migrated_japanese_detail_boundary"
        )
        rows.append(
            {
                "request_id": f"JPLA-JA-D-{detail_id:05d}",
                "scope": scope,
                "month": "",
                "route_type": "official_migrated_detail_ja",
                "url": f"{JPX_ROOT}/news/detail/detail_{detail_id}.html",
            }
        )
    for detail_id in range(OSE_DETAIL_SCAN_START, OSE_DETAIL_SCAN_END + 1):
        rows.append(
            {
                "request_id": f"JPLA-OSE-D-{detail_id:05d}",
                "scope": "migrated_ose_detail_scan",
                "month": "",
                "route_type": "official_migrated_ose_detail",
                "url": f"{JPX_ROOT}/english/news/detail/detail_{detail_id}.html",
            }
        )
    rows.extend(
        [
            {
                "request_id": "JPLA-CAT-EN-INDEX",
                "scope": "official_archive_catalog",
                "month": "",
                "route_type": "archive_index_en",
                "url": f"{JPX_ROOT}/english/news/index.html",
            },
            {
                "request_id": "JPLA-CAT-JA-INDEX",
                "scope": "official_archive_catalog",
                "month": "",
                "route_type": "archive_index_ja",
                "url": f"{JPX_ROOT}/news/index.html",
            },
            {
                "request_id": "JPLA-CAT-EN-JS",
                "scope": "official_archive_catalog",
                "month": "",
                "route_type": "archive_catalog_javascript_en",
                "url": f"{JPX_ROOT}/english/news/news.js",
            },
            {
                "request_id": "JPLA-CAT-EN-SEARCH",
                "scope": "official_archive_catalog",
                "month": "",
                "route_type": "archive_search_counts_en",
                "url": f"{JPX_ROOT}/english/news/news_search_info.json",
            },
            {
                "request_id": "JPLA-CAT-EN-SEARCH-STATIC",
                "scope": "official_archive_catalog",
                "month": "",
                "route_type": "archive_search_counts_static_en",
                "url": f"{JPX_ROOT}/english/news/news_search_info_static.json",
            },
        ]
    )
    for language, prefix in (("en", "english/news"), ("ja", "news")):
        for yyyymm in ("200607", "201307", "201308", "201309"):
            for route_name, route in (
                ("archives", f"archives-{yyyymm}.html"),
                ("archive_dir", f"archive/{yyyymm}.html"),
                ("dated_index", f"{yyyymm[:4]}/{yyyymm[4:]}/index.html"),
            ):
                rows.append(
                    {
                        "request_id": f"JPLA-ALT-{language.upper()}-{yyyymm}-{route_name.upper()}",
                        "scope": "alternate_official_route_probe",
                        "month": f"{yyyymm[:4]}-{yyyymm[4:]}",
                        "route_type": f"alternate_{route_name}_{language}",
                        "url": f"{JPX_ROOT}/{prefix}/{route}",
                    }
                )
    return pd.DataFrame(rows)


def _fetch_all(
    urls: Iterable[str], fetcher: Fetcher, *, max_workers: int
) -> dict[str, _Response]:
    unique = list(dict.fromkeys(urls))
    responses: dict[str, _Response] = {}

    def one(url: str) -> tuple[str, _Response]:
        try:
            status, body, headers = fetcher(url)
            return url, _Response(int(status), bytes(body), dict(headers))
        except Exception as exc:
            return url, _Response(0, b"", {"x-acquisition-error": f"{type(exc).__name__}: {exc}"})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(one, url) for url in unique]
        for future in as_completed(futures):
            url, response = future.result()
            responses[url] = response
    return responses


def _build_source_attempts(
    root: Path, specs: pd.DataFrame, responses: dict[str, _Response]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs.to_dict("records"):
        response = responses[spec["url"]]
        digest = hashlib.sha256(response.body).hexdigest() if response.body else ""
        raw_path = ""
        if response.body:
            path_suffix = Path(urlparse(spec["url"]).path).suffix.lower()
            suffix = path_suffix if path_suffix in {".json", ".js", ".html"} else ".bin"
            raw = root / "raw" / "source_responses" / f"{digest}{suffix}"
            if not raw.exists():
                _atomic_bytes(raw, response.body)
            raw_path = str(raw.relative_to(root))
        rows.append(
            {
                "request_id": spec["request_id"],
                "scope": spec["scope"],
                "month": spec["month"],
                "route_type": spec["route_type"],
                "requested_url": spec["url"],
                "authority": "JPX/TSE",
                "http_status": response.status,
                "request_completed": response.status != 0,
                "bytes": len(response.body),
                "sha256": digest,
                "raw_path": raw_path,
                "response_headers": json.dumps(response.headers, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _build_migrated_evidence(
    attempts: pd.DataFrame, responses: dict[str, _Response]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    page_rows: list[dict[str, Any]] = []
    notice_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    subset = attempts[attempts["scope"] == "migrated_detail_candidate"].copy()
    subset["detail_id"] = subset["requested_url"].str.extract(r"detail_(\d+)\.html")[0].astype(int)
    subset = subset.sort_values("detail_id")
    for attempt in subset.to_dict("records"):
        detail_id = int(attempt["detail_id"])
        response = responses[attempt["requested_url"]]
        title = ""
        publication_date = ""
        category = ""
        event_type = ""
        parse_status = "not_in_scope"
        parse_error = ""
        parsed_rows: list[dict[str, Any]] = []
        if response.status == 200 and response.body:
            soup = BeautifulSoup(response.body, "html.parser")
            if soup.title:
                title = soup.title.get_text(" ", strip=True).rsplit(
                    " | Japan Exchange Group", 1
                )[0]
            date_node = soup.select_one(".date-news")
            date_text = date_node.get_text(" ", strip=True) if date_node else ""
            publication_date = _english_publication_date(date_text)
            category_node = soup.select_one(".date-news em")
            category = category_node.get_text(" ", strip=True) if category_node else ""
            event_type = classify_notice_title(title)
            if event_type:
                try:
                    parsed_rows = parse_notice_page(
                        response.body,
                        event_type=event_type,
                        archive_publication_date=publication_date,
                        source_url=attempt["requested_url"],
                        title=title,
                        source_sha256=attempt["sha256"],
                    )
                    parse_status = "parsed"
                except Exception as exc:
                    parse_status = "parse_failed"
                    parse_error = f"{type(exc).__name__}: {exc}"
        elif response.status == 0:
            parse_status = "request_failed"
        else:
            parse_status = f"http_{response.status}"

        is_exceptional = bool(event_type)
        page_rows.append(
            {
                "detail_id": detail_id,
                "publication_date": publication_date,
                "category": category,
                "title": title,
                "event_type": event_type,
                "is_exceptional_price_limit": is_exceptional,
                "http_status": response.status,
                "request_completed": response.status != 0,
                "source_url": attempt["requested_url"],
                "raw_path": attempt["raw_path"],
                "sha256": attempt["sha256"],
                "bytes": attempt["bytes"],
                "parse_status": parse_status,
                "parse_error": parse_error,
            }
        )
        if is_exceptional:
            notice_id = f"JPLA-201308-{detail_id}"
            notice_rows.append(
                {
                    "notice_id": notice_id,
                    "detail_id": detail_id,
                    "publication_date": publication_date,
                    "event_type": event_type,
                    "title": title,
                    "source_url": attempt["requested_url"],
                    "raw_path": attempt["raw_path"],
                    "sha256": attempt["sha256"],
                    "parse_status": parse_status,
                    "parsed_event_rows": len(parsed_rows),
                    "denominator_role": "observed_numerator_only",
                }
            )
            for event in parsed_rows:
                event_rows.append({"notice_id": notice_id, "detail_id": detail_id, **event})

    pages = pd.DataFrame(page_rows)
    notices = pd.DataFrame(
        notice_rows,
        columns=[
            "notice_id",
            "detail_id",
            "publication_date",
            "event_type",
            "title",
            "source_url",
            "raw_path",
            "sha256",
            "parse_status",
            "parsed_event_rows",
            "denominator_role",
        ],
    )
    events = pd.DataFrame(event_rows)
    return pages, notices, events


def _build_japanese_migrated_evidence(
    attempts: pd.DataFrame, responses: dict[str, _Response]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    page_rows: list[dict[str, Any]] = []
    notice_rows: list[dict[str, Any]] = []
    subset = attempts[
        attempts["scope"] == "migrated_japanese_detail_candidate"
    ].copy()
    subset["detail_id"] = subset["requested_url"].str.extract(r"detail_(\d+)\.html")[0].astype(int)
    for attempt in subset.sort_values("detail_id").to_dict("records"):
        detail_id = int(attempt["detail_id"])
        response = responses[attempt["requested_url"]]
        title = ""
        publication_date = ""
        category = ""
        event_type = ""
        security_code = ""
        if response.status == 200 and response.body:
            soup = BeautifulSoup(response.body, "html.parser")
            if soup.title:
                title = soup.title.get_text(" ", strip=True).rsplit(
                    " | 日本取引所グループ", 1
                )[0]
            date_node = soup.select_one(".date-news")
            date_text = date_node.get_text(" ", strip=True) if date_node else ""
            match = re.search(r"20\d{2}/\d{2}/\d{2}", date_text)
            publication_date = match.group(0).replace("/", "-") if match else ""
            category_node = soup.select_one(".date-news em")
            category = category_node.get_text(" ", strip=True) if category_node else ""
            event_type = _classify_japanese_notice_title(title)
            normalized = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
            code_match = re.search(r"コード\s*([0-9A-Z]{4})", normalized, re.I)
            security_code = code_match.group(1).upper() if code_match else ""
        page_rows.append(
            {
                "detail_id": detail_id,
                "publication_date": publication_date,
                "category": category,
                "title": title,
                "event_type": event_type,
                "security_code": security_code,
                "is_exceptional_price_limit": bool(event_type),
                "http_status": response.status,
                "request_completed": response.status != 0,
                "source_url": attempt["requested_url"],
                "raw_path": attempt["raw_path"],
                "sha256": attempt["sha256"],
                "bytes": attempt["bytes"],
                "denominator_role": "observed_bilingual_numerator_only" if event_type else "not_in_scope",
            }
        )
        if event_type:
            notice_rows.append(
                {
                    "notice_id": f"JPLA-JA-201308-{detail_id}",
                    "detail_id": detail_id,
                    "publication_date": publication_date,
                    "event_type": event_type,
                    "security_code": security_code,
                    "title": title,
                    "source_url": attempt["requested_url"],
                    "raw_path": attempt["raw_path"],
                    "sha256": attempt["sha256"],
                    "denominator_role": "observed_bilingual_numerator_only",
                }
            )
    return pd.DataFrame(page_rows), pd.DataFrame(notice_rows)


def _build_ose_migrated_evidence(
    attempts: pd.DataFrame, responses: dict[str, _Response]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = attempts[attempts["scope"] == "migrated_ose_detail_scan"].copy()
    subset["detail_id"] = subset["requested_url"].str.extract(r"detail_(\d+)\.html")[0].astype(int)
    for attempt in subset.sort_values("detail_id").to_dict("records"):
        response = responses[attempt["requested_url"]]
        title = ""
        publication_date = ""
        category = ""
        event_type = ""
        if response.status == 200 and response.body:
            soup = BeautifulSoup(response.body, "html.parser")
            if soup.title:
                title = soup.title.get_text(" ", strip=True).rsplit(
                    " | Japan Exchange Group", 1
                )[0]
            node = soup.select_one(".date-news")
            publication_date = _english_publication_date(
                node.get_text(" ", strip=True) if node else ""
            )
            category_node = soup.select_one(".date-news em")
            category = category_node.get_text(" ", strip=True) if category_node else ""
            event_type = classify_notice_title(title)
        rows.append(
            {
                "detail_id": int(attempt["detail_id"]),
                "publication_date": publication_date,
                "category": category,
                "title": title,
                "http_status": response.status,
                "request_completed": response.status != 0,
                "is_equity_exceptional_price_limit": bool(event_type),
                "source_url": attempt["requested_url"],
                "raw_path": attempt["raw_path"],
                "sha256": attempt["sha256"],
                "denominator_role": "adjacency_discovery_only",
            }
        )
    return pd.DataFrame(rows)


def _bilingual_reconciliation(
    english_events: pd.DataFrame, japanese_notices: pd.DataFrame
) -> pd.DataFrame:
    keys = ["publication_date", "event_type", "security_code"]
    english = english_events[
        ["notice_id", "detail_id", *keys, "source_url", "source_sha256"]
    ].rename(
        columns={
            "notice_id": "english_notice_id",
            "detail_id": "english_detail_id",
            "source_url": "english_source_url",
            "source_sha256": "english_sha256",
        }
    )
    japanese = japanese_notices[
        ["notice_id", "detail_id", *keys, "source_url", "sha256"]
    ].rename(
        columns={
            "notice_id": "japanese_notice_id",
            "detail_id": "japanese_detail_id",
            "source_url": "japanese_source_url",
            "sha256": "japanese_sha256",
        }
    )
    merged = english.merge(japanese, on=keys, how="outer", indicator=True)
    merged["reconciliation_status"] = merged["_merge"].map(
        {"both": "matched", "left_only": "missing_japanese", "right_only": "missing_english"}
    )
    return merged.drop(columns="_merge").sort_values(keys).reset_index(drop=True)


def _classify_japanese_notice_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title)
    patterns = (
        ("preliminary", r"^制限値幅の拡大について\(拡大予告\)"),
        ("cancel", r"^制限値幅の取扱いについて\(制限値幅を拡大しません\)"),
        ("activate", r"^制限値幅の拡大について\(制限値幅を拡大します\)"),
        ("return", r"^制限値幅の拡大解除について\(通常の制限値幅に戻ります\)"),
    )
    for event_type, pattern in patterns:
        if re.search(pattern, value):
            return event_type
    return ""


def _build_monthly_coverage(
    attempts: pd.DataFrame,
    notices: pd.DataFrame,
    japanese_notices: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in EXPECTED_MONTHS:
        month = str(period)
        current = attempts[(attempts["scope"] == "research_month") & (attempts["month"] == month)]
        statuses = dict(zip(current["route_type"], current["http_status"]))
        observed = int(notices["publication_date"].str.startswith(month).sum()) if len(notices) else 0
        observed_ja = (
            int(japanese_notices["publication_date"].str.startswith(month).sum())
            if len(japanese_notices)
            else 0
        )
        reconciled = (
            int(
                (
                    reconciliation["publication_date"].str.startswith(month)
                    & reconciliation["reconciliation_status"].eq("matched")
                ).sum()
            )
            if len(reconciliation)
            else 0
        )
        rows.append(
            {
                "month": month,
                "interval_start": period.start_time.date().isoformat(),
                "interval_end": period.end_time.date().isoformat(),
                "archive_json_http_status": statuses.get("monthly_archive_json", ""),
                "static_archive_http_status": statuses.get("monthly_static_archive", ""),
                "archive_json_ja_http_status": statuses.get("monthly_archive_json_ja", ""),
                "static_archive_ja_http_status": statuses.get("monthly_static_archive_ja", ""),
                "observed_public_notice_pages": observed,
                "observed_japanese_notice_pages": observed_ja,
                "bilingual_reconciled_notice_pages": reconciled,
                "official_notice_denominator": "",
                "official_denominator_proven": False,
                "missing_notice_count": "",
                "denominator_semantics": "unknown_not_zero",
                "coverage_status": (
                    "migrated_numerator_only_denominator_unknown"
                    if month == "2013-08" and observed
                    else "no_public_archive_denominator"
                ),
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _build_denominator_evidence(
    attempts: pd.DataFrame,
    responses: dict[str, _Response],
    pages: pd.DataFrame,
    *,
    japanese_pages: pd.DataFrame,
    japanese_notices: pd.DataFrame,
    ose_pages: pd.DataFrame,
    archive_catalogs: pd.DataFrame,
) -> pd.DataFrame:
    research = attempts[attempts["scope"] == "research_month"]
    control = attempts[attempts["scope"] == "positive_control"]
    control_json = control[control["route_type"] == "monthly_archive_json"].iloc[0]
    control_static = control[control["route_type"] == "monthly_static_archive"].iloc[0]
    json_items = _json_news_items(responses[control_json["requested_url"]].body)
    static_items = _static_detail_links(responses[control_static["requested_url"]].body)
    lower = attempts[attempts["request_id"] == f"JPLA-D-{LOWER_BOUNDARY_DETAIL_ID:05d}"].iloc[0]
    upper = attempts[attempts["request_id"] == f"JPLA-D-{UPPER_BOUNDARY_DETAIL_ID:05d}"].iloc[0]
    rows = [
            {
                "evidence_id": "JPLA-DEN-001",
                "evidence_type": "in_range_monthly_archive_json_routes",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "route_count": int(research["route_type"].eq("monthly_archive_json").sum()),
                "http_200_count": int(
                    (
                        research["route_type"].eq("monthly_archive_json")
                        & research["http_status"].eq(200)
                    ).sum()
                ),
                "http_404_count": int(
                    (
                        research["route_type"].eq("monthly_archive_json")
                        & research["http_status"].eq(404)
                    ).sum()
                ),
                "observed_items": "",
                "denominator_proven": False,
                "conclusion": "no in-range machine-readable official monthly denominator is published at this route",
            },
            {
                "evidence_id": "JPLA-DEN-002",
                "evidence_type": "in_range_static_archive_routes",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "route_count": int(research["route_type"].eq("monthly_static_archive").sum()),
                "http_200_count": int(
                    (
                        research["route_type"].eq("monthly_static_archive")
                        & research["http_status"].eq(200)
                    ).sum()
                ),
                "http_404_count": int(
                    (
                        research["route_type"].eq("monthly_static_archive")
                        & research["http_status"].eq(404)
                    ).sum()
                ),
                "observed_items": "",
                "denominator_proven": False,
                "conclusion": "no in-range official static monthly denominator is published at this route",
            },
            {
                "evidence_id": "JPLA-DEN-003",
                "evidence_type": "post_range_json_positive_control",
                "scope": POSITIVE_CONTROL_MONTH,
                "route_count": 1,
                "http_200_count": int(control_json["http_status"] == 200),
                "http_404_count": int(control_json["http_status"] == 404),
                "observed_items": len(json_items),
                "denominator_proven": bool(control_json["http_status"] == 200),
                "conclusion": "the official JSON route is live immediately after the unresolved range",
            },
            {
                "evidence_id": "JPLA-DEN-004",
                "evidence_type": "post_range_static_positive_control",
                "scope": POSITIVE_CONTROL_MONTH,
                "route_count": 1,
                "http_200_count": int(control_static["http_status"] == 200),
                "http_404_count": int(control_static["http_status"] == 404),
                "observed_items": len(static_items),
                "denominator_proven": bool(control_static["http_status"] == 200),
                "conclusion": "the official static archive route is live immediately after the unresolved range",
            },
            {
                "evidence_id": "JPLA-DEN-005",
                "evidence_type": "bounded_official_migrated_detail_sequence",
                "scope": f"detail_{MIGRATED_DETAIL_START}_through_detail_{MIGRATED_DETAIL_END}",
                "route_count": len(pages),
                "http_200_count": int(pages["http_status"].eq(200).sum()),
                "http_404_count": int(pages["http_status"].eq(404).sum()),
                "observed_items": int(pages["is_exceptional_price_limit"].sum()),
                "denominator_proven": False,
                "conclusion": "official migrated August pages provide a notice numerator but no official exhaustive archive count",
            },
            {
                "evidence_id": "JPLA-DEN-006",
                "evidence_type": "migrated_detail_lower_boundary",
                "scope": f"detail_{LOWER_BOUNDARY_DETAIL_ID}",
                "route_count": 1,
                "http_200_count": int(lower["http_status"] == 200),
                "http_404_count": int(lower["http_status"] == 404),
                "observed_items": "",
                "denominator_proven": False,
                "conclusion": "lower numeric boundary is a route observation, not proof that no earlier notices exist",
            },
            {
                "evidence_id": "JPLA-DEN-007",
                "evidence_type": "migrated_detail_successor_boundary",
                "scope": f"detail_{UPPER_BOUNDARY_DETAIL_ID}",
                "route_count": 1,
                "http_200_count": int(upper["http_status"] == 200),
                "http_404_count": int(upper["http_status"] == 404),
                "observed_items": 1 if upper["http_status"] == 200 else "",
                "denominator_proven": False,
                "conclusion": "the successor is the 2013-09-02 return notice and links the August S-Pool chain",
            },
        ]
    rows.extend(
        [
            {
                "evidence_id": "JPLA-DEN-008",
                "evidence_type": "in_range_japanese_monthly_archive_routes",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "route_count": int(
                    research["route_type"].isin(
                        {"monthly_archive_json_ja", "monthly_static_archive_ja"}
                    ).sum()
                ),
                "http_200_count": int(
                    (
                        research["route_type"].isin(
                            {"monthly_archive_json_ja", "monthly_static_archive_ja"}
                        )
                        & research["http_status"].eq(200)
                    ).sum()
                ),
                "http_404_count": int(
                    (
                        research["route_type"].isin(
                            {"monthly_archive_json_ja", "monthly_static_archive_ja"}
                        )
                        & research["http_status"].eq(404)
                    ).sum()
                ),
                "observed_items": "",
                "denominator_proven": False,
                "conclusion": "Japanese JSON and static monthly routes also provide no in-range official denominator",
            },
            {
                "evidence_id": "JPLA-DEN-009",
                "evidence_type": "bounded_official_japanese_migrated_detail_sequence",
                "scope": f"detail_{JAPANESE_DETAIL_START}_through_detail_{JAPANESE_DETAIL_END}",
                "route_count": len(japanese_pages),
                "http_200_count": int(japanese_pages["http_status"].eq(200).sum()),
                "http_404_count": int(japanese_pages["http_status"].eq(404).sum()),
                "observed_items": len(japanese_notices),
                "denominator_proven": False,
                "conclusion": "the authoritative-language migrated sequence corroborates the 19-page August numerator but is not an archive index",
            },
            {
                "evidence_id": "JPLA-DEN-010",
                "evidence_type": "bounded_official_ose_migrated_detail_scan",
                "scope": f"detail_{OSE_DETAIL_SCAN_START}_through_detail_{OSE_DETAIL_SCAN_END}",
                "route_count": len(ose_pages),
                "http_200_count": int(ose_pages["http_status"].eq(200).sum()),
                "http_404_count": int(ose_pages["http_status"].eq(404).sum()),
                "observed_items": int(
                    ose_pages["is_equity_exceptional_price_limit"].sum()
                ),
                "denominator_proven": False,
                "conclusion": "the surviving OSE adjacency pages are derivatives/index notices and provide no equity exceptional-price-limit denominator",
            },
            {
                "evidence_id": "JPLA-DEN-011",
                "evidence_type": "official_archive_catalog_boundary",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "route_count": len(archive_catalogs),
                "http_200_count": "",
                "http_404_count": "",
                "observed_items": "",
                "denominator_proven": False,
                "conclusion": "official selectors, JavaScript mappings, and search-count catalogs do not expose an exhaustive pre-September-2013 archive",
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_official_archive_catalogs(
    attempts: pd.DataFrame, responses: dict[str, _Response]
) -> pd.DataFrame:
    research = attempts[attempts["scope"] == "research_month"]
    alternate = attempts[attempts["scope"] == "alternate_official_route_probe"]

    def response_for(request_id: str) -> _Response:
        row = attempts[attempts["request_id"] == request_id].iloc[0]
        return responses[row["requested_url"]]

    en_static = response_for("JPLA-C-201309-STATIC")
    ja_static = response_for("JPLA-C-201309-STATIC-JA")
    en_options = _archive_selector_months(en_static.body)
    ja_options = _archive_selector_months(ja_static.body)
    script = response_for("JPLA-CAT-EN-JS").body.decode("utf-8", errors="replace")
    script_start = _javascript_date_pair(script, "start_year", "start_month")
    script_base = _javascript_date_pair(script, "baseYear", "baseMonth")
    cap_match = re.search(r"36\s*<\s*month_between", script)
    search_start, search_end = _json_date_bounds(response_for("JPLA-CAT-EN-SEARCH").body)
    static_start, static_end = _json_date_bounds(
        response_for("JPLA-CAT-EN-SEARCH-STATIC").body
    )
    return pd.DataFrame(
        [
            {
                "evidence_id": "JPLA-CAT-001",
                "evidence_type": "bilingual_monthly_archive_route_matrix",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "source_urls": "four JPX monthly route templates recorded in source_attempts.csv",
                "route_count": len(research),
                "http_200_count": int(research["http_status"].eq(200).sum()),
                "http_404_count": int(research["http_status"].eq(404).sum()),
                "observed_start": ARCHIVE_START,
                "observed_end": ARCHIVE_END,
                "denominator_proven": False,
                "conclusion": "all English and Japanese JSON/static month routes were queried; none publishes an in-range archive",
            },
            {
                "evidence_id": "JPLA-CAT-002",
                "evidence_type": "explicit_2013_09_archive_selector_en",
                "scope": POSITIVE_CONTROL_MONTH,
                "source_urls": f"{JPX_ROOT}/english/news/static-archives-201309.html",
                "route_count": 1,
                "http_200_count": int(en_static.status == 200),
                "http_404_count": int(en_static.status == 404),
                "observed_start": min(en_options) if en_options else "",
                "observed_end": max(en_options) if en_options else "",
                "denominator_proven": bool(en_static.status == 200),
                "conclusion": "the retained selector begins at September 2013 and exposes no August option",
            },
            {
                "evidence_id": "JPLA-CAT-003",
                "evidence_type": "explicit_2013_09_archive_selector_ja",
                "scope": POSITIVE_CONTROL_MONTH,
                "source_urls": f"{JPX_ROOT}/news/static-archives-201309.html",
                "route_count": 1,
                "http_200_count": int(ja_static.status == 200),
                "http_404_count": int(ja_static.status == 404),
                "observed_start": min(ja_options) if ja_options else "",
                "observed_end": max(ja_options) if ja_options else "",
                "denominator_proven": bool(ja_static.status == 200),
                "conclusion": "the authoritative-language selector also begins at September 2013",
            },
            {
                "evidence_id": "JPLA-CAT-004",
                "evidence_type": "current_news_javascript_mapping",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "source_urls": f"{JPX_ROOT}/english/news/news.js",
                "route_count": 1,
                "http_200_count": int(bool(script)),
                "http_404_count": 0,
                "observed_start": script_start,
                "observed_end": script_base,
                "denominator_proven": False,
                "conclusion": f"current mapping is bounded to {'36' if cap_match else 'an observed finite'} months and is not a historical pre-2013 catalog",
            },
            {
                "evidence_id": "JPLA-CAT-005",
                "evidence_type": "current_news_search_count_catalog",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "source_urls": f"{JPX_ROOT}/english/news/news_search_info.json",
                "route_count": 1,
                "http_200_count": int(bool(search_start)),
                "http_404_count": 0,
                "observed_start": search_start,
                "observed_end": search_end,
                "denominator_proven": False,
                "conclusion": "current search-count metadata does not reach the research interval",
            },
            {
                "evidence_id": "JPLA-CAT-006",
                "evidence_type": "static_news_search_count_catalog",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "source_urls": f"{JPX_ROOT}/english/news/news_search_info_static.json",
                "route_count": 1,
                "http_200_count": int(bool(static_start)),
                "http_404_count": 0,
                "observed_start": static_start,
                "observed_end": static_end,
                "denominator_proven": False,
                "conclusion": "static search-count metadata begins in 2023 and is not a pre-2013 denominator",
            },
            {
                "evidence_id": "JPLA-CAT-007",
                "evidence_type": "alternate_official_archive_route_probes",
                "scope": f"{ARCHIVE_START}_through_{ARCHIVE_END}",
                "source_urls": "archives-YYYYMM.html|archive/YYYYMM.html|YYYY/MM/index.html in both languages",
                "route_count": len(alternate),
                "http_200_count": int(alternate["http_status"].eq(200).sum()),
                "http_404_count": int(alternate["http_status"].eq(404).sum()),
                "observed_start": "2006-07",
                "observed_end": "2013-09",
                "denominator_proven": False,
                "conclusion": "bounded alternate-path probes yield no denominator-bearing pre-September route",
            },
        ]
    )


def _build_local_source_catalog_audit(local_root: Path) -> pd.DataFrame:
    sources = (
        (
            "LOCAL-001",
            "japan_market_rule_history/JPN-MARKET-RULE-HISTORY-200607-202606-V1/manifest.json",
            "JPX/TSE rules and transition evidence",
            "2006-07 through 2026-06 partial rule history",
            "aggregate rules do not enumerate issue-specific temporary notices",
        ),
        (
            "LOCAL-002",
            "jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200607-200711-EXT-V1/manifest.json",
            "TSE monthly quotation PDFs",
            "2006-07 through 2007-11",
            "monthly quotation summaries lack daily trigger/notice enumeration",
        ),
        (
            "LOCAL-003",
            "jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2/manifest.json",
            "TSE/JPX monthly quotation PDFs",
            "2007-12 onward with archival fallbacks",
            "monthly quotation summaries and mirror routes are not an official issue-notice denominator",
        ),
        (
            "LOCAL-004",
            "ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200607-200711-EXT-V1/manifest.json",
            "official migrated OSE monthly quotations",
            "2006-07 through 2007-11",
            "monthly price summaries cannot prove exceptional-notice absence or trigger state",
        ),
        (
            "LOCAL-005",
            "ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200712-201306-V1/manifest.json",
            "official migrated OSE monthly quotations",
            "2007-12 through 2013-06",
            "monthly price summaries are not an exhaustive OSE/TSE notice index",
        ),
        (
            "LOCAL-006",
            "ose_main_monthly_quotations/JPN-OSE-MAIN-MONTHLY-QUOTATIONS-200607-201009-V1/manifest.json",
            "official OSE main-market monthly quotations",
            "2006-07 through 2010-09",
            "main-market monthly summaries omit issue-notice and daily quote-state denominators",
        ),
        (
            "LOCAL-007",
            "japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V1/source_catalog.csv",
            "official halt/listing source catalog",
            "mixed current and historical event channels",
            "catalog has no pre-2013 exceptional-price-limit archive denominator",
        ),
        (
            "LOCAL-008",
            "japan_pre2013_price_limit_archive/JPN-PRE2013-PRICE-LIMIT-ARCHIVE-200607-201308-V1/manifest.json",
            "V1 public-route acquisition",
            "2006-07 through 2013-08",
            "V1 proves route failures and an August numerator but leaves all 86 denominators unresolved",
        ),
    )
    rows = []
    for source_id, relative, source_type, coverage, reason in sources:
        path = local_root / relative
        rows.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "coverage": coverage,
                "local_path": str(path),
                "file_exists": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else "",
                "official_publisher_content": True,
                "denominator_bearing": False,
                "admission_decision": "context_only_rejected_as_notice_denominator",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _completion_plan() -> pd.DataFrame:
    start = "2006-07-01T00:00:00+09:00"
    end = "2013-08-31T23:59:59.999999+09:00"
    return pd.DataFrame(
        [
            {
                "requirement_id": "JPLA-COMP-001",
                "source_authority": "JPX/TSE/OSE official publisher only",
                "route_or_delivery": "official market-news archive export or documented replacement endpoint",
                "query_start": start,
                "query_end": end,
                "query_filter": "corporation in [TSE,OSE] AND authoritative Japanese title in [preliminary,cancel,activate,continue,return] exceptional-equity families; include revisions and withdrawals",
                "required_fields": "archive_record_id|publication_timestamp_jst|effective_date|corporation|language|title|security_code|side|canonical_url|revision_of|raw_body",
                "pagination_requirement": "unbounded export or cursor traversal retaining declared total_hits, page_size, every cursor, final_cursor, and duplicate policy",
                "acceptance_gate": "one immutable record set whose declared total equals unique acquired records and whose 86 monthly counts sum exactly to that total",
                "independent_validation": "re-run the same query independently and compare record IDs, monthly counts, raw hashes, revisions, and zero-gap cursor chain",
                "execution_status": "required_not_executed",
            },
            {
                "requirement_id": "JPLA-COMP-002",
                "source_authority": "JPX/TSE/OSE official publisher only",
                "route_or_delivery": "legacy URL migration crosswalk export",
                "query_start": start,
                "query_end": end,
                "query_filter": "all legacy www.tse.or.jp and www.ose.or.jp equity market-news URLs, including deleted, redirected, translated, revised, and no-link records",
                "required_fields": "legacy_url|current_jpx_url|archive_record_id|publication_timestamp_jst|corporation|language|title|migration_status|deletion_reason",
                "pagination_requirement": "complete export with declared legacy-record total; no search-snippet or numeric-ID inference admitted",
                "acceptance_gate": "every legacy archive record maps to a retained JPX page or an official tombstone, with exact totals by month and corporation",
                "independent_validation": "verify bijection/declared one-to-many revisions, duplicate handling, date bounds, and 2013-08 English/Japanese checkpoints",
                "execution_status": "required_not_executed",
            },
            {
                "requirement_id": "JPLA-COMP-003",
                "source_authority": "JPX/TSE official market-data delivery",
                "route_or_delivery": "complete cash-equity venue-by-session trade/quote export",
                "query_start": start,
                "query_end": "2013-09-03T23:59:59.999999+09:00",
                "query_filter": "all TSE cash-equity instruments and, through 2013-07-15, all OSE cash-equity instruments; include inactive/delisted codes, closing auction, quotes, corrections, and zero-volume sessions",
                "required_fields": "venue|session_date|timestamp|security_code|listing_identity|event_type|trade_price|trade_size|bid|ask|bid_size|ask_size|quote_condition|auction_state|correction_sequence",
                "pagination_requirement": "venue/session file inventory with expected and delivered files, record totals, checksums, and explicit no-data status for every trading session",
                "acceptance_gate": "deterministic replay reproduces all 19 August official notices and yields an exact all-security event denominator for every one of 86 months",
                "independent_validation": "independent rule engine, calendar, listing universe, correction replay, and notice reconciliation with zero unexplained events",
                "execution_status": "required_not_executed",
            },
            {
                "requirement_id": "JPLA-COMP-004",
                "source_authority": "JPX/TSE/OSE official publisher only",
                "route_or_delivery": "public monthly route recheck matrix",
                "query_start": start,
                "query_end": end,
                "query_filter": "GET /english/news/news_ym_YYYYMM.json, /english/news/static-archives-YYYYMM.html, /news/news_ym_YYYYMM.json, /news/static-archives-YYYYMM.html for every YYYYMM",
                "required_fields": "requested_url|http_status|response_headers|bytes|sha256|raw_path|parsed_record_count|declared_total_if_any",
                "pagination_requirement": "exactly four responses per month, 344 total, with no inferred status and immutable raw error/success bodies",
                "acceptance_gate": "a month can pass only if an official response declares or structurally proves an exhaustive count and every listed record is retained",
                "independent_validation": "repeat all 344 requests, compare status/content hashes, and reject endpoint absence as evidence of zero notices",
                "execution_status": "producer_executed_independent_recheck_required",
            },
            {
                "requirement_id": "JPLA-COMP-005",
                "source_authority": "JPX records/compliance owner",
                "route_or_delivery": "signed archive completeness attestation accompanying COMP-001 or COMP-002",
                "query_start": start,
                "query_end": end,
                "query_filter": "identify archive system, included corporations/languages/content types, known losses, migration date, and revision treatment",
                "required_fields": "issuer|issued_at|archive_system|coverage_start|coverage_end|declared_total|known_exclusions|signature_or_official_publication_url",
                "pagination_requirement": "not applicable; attestation must bind to exact export job/hash inventory",
                "acceptance_gate": "attested declared total matches acquired unique records and monthly reconciliation exactly",
                "independent_validation": "verify authority, signature/publication URL, export hash binding, date scope, and exclusion statement",
                "execution_status": "required_not_executed",
            },
        ]
    )


def _independent_validation_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("JPLA-IV-001", "authority", "all admitted hosts/files are direct JPX/TSE/OSE publisher sources; mirrors and snippets are discovery-only", True, "reject_artifact"),
            ("JPLA-IV-002", "denominator", "declared total is numeric, source-bound, and equals unique acquired records after documented revision handling", True, "retain_blocker"),
            ("JPLA-IV-003", "monthly_partition", "exactly 86 ordered months cover 2006-07 through 2013-08 with no gap or overlap and monthly totals sum to denominator", True, "retain_blocker"),
            ("JPLA-IV-004", "pagination", "cursor/page chain starts at first page, terminates explicitly, has no repeated/missing cursor, and reconciles to total_hits", True, "retain_blocker"),
            ("JPLA-IV-005", "raw_integrity", "every response/export has immutable path, byte count, SHA-256, request metadata, and independently recomputed hash", True, "reject_artifact"),
            ("JPLA-IV-006", "bilingual", "Japanese authoritative records reconcile to English where translated; untranslated Japanese records remain admitted and visible", True, "retain_blocker"),
            ("JPLA-IV-007", "event_semantics", "five equity exceptional event families exclude ETF/ETN, new-listing, ordinary band, derivatives, and generic price-limit publications", True, "reject_rows"),
            ("JPLA-IV-008", "negative_evidence", "404, missing page, missing quote, zero monthly rows, and numeric-ID gaps are never interpreted as zero events", True, "retain_blocker"),
            ("JPLA-IV-009", "independent_reproduction", "second implementation reproduces IDs, dates, codes, event types, monthly counts, total denominator, and hashes", True, "retain_blocker"),
            ("JPLA-IV-010", "promotion", "completion_claimed and promotion_allowed remain false until IV-001 through IV-009 all pass with zero issues", True, "retain_blocker"),
        ],
        columns=["validation_id", "validation_area", "requirement", "mandatory", "failure_result"],
    )


def _archive_selector_months(body: bytes) -> list[str]:
    soup = BeautifulSoup(body, "html.parser")
    months: set[str] = set()
    for option in soup.select("select.backnumber option"):
        text = unicodedata.normalize("NFKC", option.get_text(" ", strip=True))
        english = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?:\s+)(20\d{2})",
            text,
        )
        if english:
            month_number = {
                name: index
                for index, name in enumerate(
                    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
                    1,
                )
            }[english.group(1)]
            months.add(f"{english.group(2)}-{month_number:02d}")
            continue
        japanese = re.search(r"(20\d{2})年(\d{1,2})月", text)
        if japanese:
            months.add(f"{japanese.group(1)}-{int(japanese.group(2)):02d}")
    return sorted(months)


def _javascript_date_pair(script: str, year_name: str, month_name: str) -> str:
    year = re.search(rf"var\s+{re.escape(year_name)}\s*=\s*parseInt\((20\d{{2}})", script)
    if not year:
        year = re.search(rf"var\s+{re.escape(year_name)}\s*=\s*(20\d{{2}})", script)
    month = re.search(rf"var\s+{re.escape(month_name)}\s*=\s*parseInt\((\d{{1,2}})", script)
    if not month:
        month = re.search(rf"var\s+{re.escape(month_name)}\s*=\s*(\d{{1,2}})", script)
    return f"{year.group(1)}-{int(month.group(1)):02d}" if year and month else ""


def _json_date_bounds(body: bytes) -> tuple[str, str]:
    if not body:
        return "", ""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", ""
    dates: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if re.fullmatch(r"20\d{2}/\d{2}/\d{2}", str(key)):
                    dates.append(str(key).replace("/", "-"))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return (min(dates), max(dates)) if dates else ("", "")


def _vendor_route_plan() -> pd.DataFrame:
    common_required = (
        "vendor_job_id|entitlement_identity|request_payload|response_metadata|raw_bytes|"
        "row_count|pagination_or_cursor_audit|sha256|independent_validation"
    )
    return pd.DataFrame(
        [
            {
                "route_id": "LSEG-EPL-001",
                "priority": 1,
                "product": "LSEG Workspace Data Item Browser",
                "interface": "manual field discovery only",
                "purpose": "discover exchange-specific historical price-limit, quote-state, and auction fields without guessing field names",
                "field_or_schema_candidates": "search terms: daily price limit|upper limit|lower limit|limit state|closing quote|zero volume",
                "request_shape": "one known 2013 Tokyo RIC and one dead historical Tokyo RIC; inspect field history depth and parameters",
                "denominator_evidence_required": "exported field definitions, parameters, history depth, content source, and entitlement result",
                "limitation": "a field name is not admitted until confirmed in the licensed Data Item Browser",
                "required_retained_evidence": common_required,
                "execution_status": "planned_not_executed",
                "live_call_performed": False,
            },
            {
                "route_id": "LSEG-EPL-002",
                "priority": 2,
                "product": "LSEG Data Library Historical Pricing",
                "interface": "ld.get_history",
                "purpose": "obtain raw unadjusted daily trade and quote checkpoints for deterministic event reconstruction",
                "field_or_schema_candidates": "OPEN_PRC|HIGH_1|LOW_1|TRDPRC_1|ACVOL_UNS|BID|ASK; adjustments=unadjusted; discover size/status/qualifier fields",
                "request_shape": "PIT Tokyo RIC batches, daily, 2006-07-01 through 2013-09-03, with no corporate-action adjustment",
                "denominator_evidence_required": "all sessions and all PIT listings, explicit null/status semantics, corrections, and zero-truncation batch audit",
                "limitation": "daily bars alone cannot prove closing unexecuted limit quotes or the official notice archive denominator",
                "required_retained_evidence": common_required,
                "execution_status": "planned_not_executed",
                "live_call_performed": False,
            },
            {
                "route_id": "LSEG-EPL-003",
                "priority": 3,
                "product": "Refinitiv Tick History / DataScope Select",
                "interface": "Tick History REST extraction",
                "purpose": "reconstruct TSE trigger states from complete trade, quote, auction, correction, and market-depth messages",
                "field_or_schema_candidates": "select exact venue timestamp|RIC|event type|price|size|quote side|qualifier|sequence|correction fields from licensed template schema",
                "request_shape": "all TSE venue sessions and all historical instruments for 2006-07-01 through 2013-09-03",
                "denominator_evidence_required": "venue/session file inventory, extraction notes, historical-instrument inclusion, record counts, and no-gap delivery metadata",
                "limitation": "separate Tick History entitlement is required; schema labels must come from the licensed template, not this plan",
                "required_retained_evidence": common_required,
                "execution_status": "planned_not_executed",
                "live_call_performed": False,
            },
            {
                "route_id": "LSEG-EPL-004",
                "priority": 4,
                "product": "LSEG Symbology",
                "interface": "lookup with showHistory/reference history",
                "purpose": "build complete effective-dated RIC identity history including delisted and renamed Tokyo listings",
                "field_or_schema_candidates": "PermID|RIC|effectiveFrom|effectiveTo|status|exchange/venue identity",
                "request_shape": "all official TSE codes observed in PIT listing files plus successor/predecessor identifier expansion",
                "denominator_evidence_required": "effective-dated one-to-many identifier map with unresolved and conflicting mappings quarantined",
                "limitation": "symbology proves identity coverage, not price-limit events",
                "required_retained_evidence": common_required,
                "execution_status": "planned_not_executed",
                "live_call_performed": False,
            },
            {
                "route_id": "LSEG-EPL-005",
                "priority": 5,
                "product": "LSEG historical news archive / MRN commercial route",
                "interface": "entitlement and archive-export discovery",
                "purpose": "seek a source-complete TSE/JPX headline and story export for direct notice reconciliation",
                "field_or_schema_candidates": "storyId|versionCreated|sourceCode|headline|story body|language|RIC subjects|revision chain",
                "request_shape": "source-filtered exhaustive export for 2006-07-01 through 2013-08-31 with all pages/cursors",
                "denominator_evidence_required": "vendor-certified query scope, source-code definition, full pagination, total hits, revisions, and zero truncation",
                "limitation": "the standard Data Library headline endpoint documents only 15 months of history and cannot fill this gap",
                "required_retained_evidence": common_required,
                "execution_status": "planned_not_executed",
                "live_call_performed": False,
            },
        ]
    )


def _source_references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_id": "SRC-JPX-001",
                "authority": "JPX/TSE",
                "url": f"{JPX_ROOT}/english/news/news_ym_201309.json",
                "role": "first post-range machine-readable archive positive control",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-002",
                "authority": "JPX/TSE",
                "url": f"{JPX_ROOT}/english/news/static-archives-201309.html",
                "role": "first post-range static archive positive control",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-003",
                "authority": "JPX/TSE",
                "url": f"{JPX_ROOT}/english/news/detail/detail_{MIGRATED_DETAIL_START}.html",
                "role": "first reachable official migrated August 2013 English market-news page",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-004",
                "authority": "JPX/TSE",
                "url": f"{JPX_ROOT}/english/news/detail/detail_{UPPER_BOUNDARY_DETAIL_ID}.html",
                "role": "September successor return notice for the August S-Pool event chain",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-005",
                "authority": "JPX/TSE",
                "url": f"{JPX_ROOT}/news/detail/detail_{JAPANESE_DETAIL_START}.html",
                "role": "first reachable official Japanese August 2013 migrated market-news page",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-006",
                "authority": "JPX/OSE",
                "url": f"{JPX_ROOT}/english/news/detail/detail_24102.html",
                "role": "first reachable August page in the separate migrated OSE detail series",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-007",
                "authority": "JPX",
                "url": f"{JPX_ROOT}/english/news/news.js",
                "role": "official machine-readable monthly archive route mapping and finite current window",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-JPX-008",
                "authority": "JPX",
                "url": f"{JPX_ROOT}/english/news/news_search_info_static.json",
                "role": "official static search-count catalog boundary evidence",
                "access_mode": "producer_public_http",
            },
            {
                "reference_id": "SRC-LSEG-001",
                "authority": "LSEG Developer Portal",
                "url": "https://developers.lseg.com/en/api-catalog/lseg-data-platform/lseg-data-library-for-python/tutorials/content-tutorials/historical-pricing",
                "role": "documents Historical Pricing summaries/events and standard trade/quote fields",
                "access_mode": "plan_reference_not_called",
            },
            {
                "reference_id": "SRC-LSEG-002",
                "authority": "LSEG Developer Portal",
                "url": "https://developers.lseg.com/en/article-catalog/article/workspace-corporate-actions-content-set-guide",
                "role": "documents unadjusted Historical Pricing requests",
                "access_mode": "plan_reference_not_called",
            },
            {
                "reference_id": "SRC-LSEG-003",
                "authority": "LSEG Developer Portal",
                "url": "https://developers.lseg.com/content/dam/devportal/api-families/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/symbology_user_guide.pdf",
                "role": "documents effective-dated symbology history",
                "access_mode": "plan_reference_not_called",
            },
            {
                "reference_id": "SRC-LSEG-004",
                "authority": "LSEG Developer Portal",
                "url": "https://developers.lseg.com/en/article-catalog/article/lseg-data-library-for-python--news-pagination",
                "role": "documents the standard headline endpoint's 15-month history limit",
                "access_mode": "plan_reference_not_called",
            },
            {
                "reference_id": "SRC-LSEG-005",
                "authority": "LSEG Developer Portal",
                "url": "https://developers.lseg.com/content/dam/devportal/api-families/thomson-reuters-tick-history-trth/thomson-reuters-tick-history-trth-rest-api/documentation/tick_hist_user_guide_jan2021.pdf",
                "role": "documents Tick History and Venue By Day historical market-data routes",
                "access_mode": "plan_reference_not_called",
            },
        ]
    )


def _completeness_result(
    *,
    coverage: pd.DataFrame,
    attempts: pd.DataFrame,
    pages: pd.DataFrame,
    notices: pd.DataFrame,
    events: pd.DataFrame,
    japanese_pages: pd.DataFrame,
    japanese_notices: pd.DataFrame,
    ose_pages: pd.DataFrame,
    reconciliation: pd.DataFrame,
    archive_catalogs: pd.DataFrame,
) -> dict[str, Any]:
    research = attempts[attempts["scope"] == "research_month"]
    return {
        "schema_version": "2.0-fail-closed",
        "scope": {"start_month": ARCHIVE_START, "end_month": ARCHIVE_END},
        "result": "incomplete_public_denominator_unavailable",
        "completion_claimed": False,
        "public_sources_exhausted": False,
        "monthly_partition_denominator": {
            "required_months": len(EXPECTED_MONTHS),
            "months_with_official_denominator": 0,
            "months_with_unresolved_denominator": len(coverage),
            "official_route_attempts": len(research),
            "official_route_http_200": int(research["http_status"].eq(200).sum()),
            "official_route_http_404": int(research["http_status"].eq(404).sum()),
        },
        "notice_denominator": {
            "value": None,
            "numeric_denominator_known": False,
            "missing_notice_count": None,
            "semantics": "unknown_not_zero",
        },
        "observed_public_numerator": {
            "official_migrated_august_detail_pages": len(pages),
            "exceptional_price_limit_notice_pages": len(notices),
            "parsed_issue_event_rows": len(events),
            "official_japanese_august_detail_pages": len(japanese_pages),
            "japanese_exceptional_notice_pages": len(japanese_notices),
            "bilingual_matched_notice_pages": int(
                reconciliation["reconciliation_status"].eq("matched").sum()
            ),
            "ose_migrated_id_rows_scanned": len(ose_pages),
            "ose_reachable_pages": int(ose_pages["http_status"].eq(200).sum()),
            "ose_equity_exceptional_notices": int(
                ose_pages["is_equity_exceptional_price_limit"].sum()
            ),
            "earliest_notice_date": (
                notices["publication_date"].min() if len(notices) else None
            ),
            "latest_notice_date": notices["publication_date"].max() if len(notices) else None,
            "denominator_role": "numerator_only",
        },
        "exact_unresolved_denominator": {
            "monthly_partitions_requiring_exhaustive_official_evidence": len(
                EXPECTED_MONTHS
            ),
            "monthly_partitions_satisfied": 0,
            "monthly_partitions_unsatisfied": len(EXPECTED_MONTHS),
            "total_required_notice_count": None,
            "reason_total_is_null": (
                "None of the four English/Japanese official monthly JSON/static routes exists for "
                "any of the 86 in-range months. English and Japanese migrated August sequences "
                "agree on 19 observed notices, but selectors, search catalogs, and migrated numeric "
                "adjacency provide no exhaustive pre-September archive count."
            ),
        },
        "official_completion_evidence_required": [
            "A source-complete JPX/TSE/OSE archive export with a declared total, complete pagination/cursor audit, and exact 2006-07-01 through 2013-08-31 JST bounds.",
            "A legacy TSE/OSE-to-JPX migration crosswalk including deleted, untranslated, revised, withdrawn, and no-link archive records.",
            "Alternatively, complete official venue/session cash-equity trade, quote, auction, correction, and listing-universe data sufficient to reconstruct every trigger.",
            "Month-by-month reconciliation to all 19 bilingual August checkpoints and the complete September 2013 positive-control archive.",
            "Independent reproduction satisfying every row in independent_validation_requirements.csv.",
        ],
        "official_archive_catalog_rows": len(archive_catalogs),
        "completion_plan_execution_status": "producer_evidence_complete_independent_completion_pending",
        "remaining_blocker": PRE_2013_BLOCKER,
        "promotion_allowed": False,
    }


def _summary_markdown(result: dict[str, Any]) -> str:
    numerator = result["observed_public_numerator"]
    unresolved = result["exact_unresolved_denominator"]
    return f"""# Pre-September-2013 Exceptional Price-Limit Archive V2

Status: incomplete; no public denominator pass is claimed.

- Scope: {ARCHIVE_START} through {ARCHIVE_END} ({len(EXPECTED_MONTHS)} monthly partitions).
- Official bilingual monthly archive routes with an in-range HTTP 200 response: 0 of {len(EXPECTED_MONTHS) * 4} route attempts.
- Retained official migrated August detail pages: {numerator['official_migrated_august_detail_pages']}.
- Observed exceptional price-limit notice pages: {numerator['exceptional_price_limit_notice_pages']}.
- Parsed issue-level event rows: {numerator['parsed_issue_event_rows']}.
- Retained Japanese migrated August pages/notices: {numerator['official_japanese_august_detail_pages']}/{numerator['japanese_exceptional_notice_pages']}.
- Bilingual notice matches: {numerator['bilingual_matched_notice_pages']}.
- OSE migrated IDs scanned/reachable/equity notices: {numerator['ose_migrated_id_rows_scanned']}/{numerator['ose_reachable_pages']}/{numerator['ose_equity_exceptional_notices']}.
- Months with an independently proven official denominator: {unresolved['monthly_partitions_satisfied']} of {unresolved['monthly_partitions_requiring_exhaustive_official_evidence']}.
- Total notice denominator: unknown, represented as null rather than zero.
- Live Refinitiv/LSEG calls: none.

The bilingual August migrated pages are admitted as a public numerator and chain checkpoint only. Completion requires the exact official query/export evidence in `completion_plan.csv` and every independent gate in `independent_validation_requirements.csv`.
"""


def _json_news_items(body: bytes) -> list[dict[str, str]]:
    if not body:
        return []
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    items: dict[str, dict[str, str]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            title = value.get("title")
            target = value.get("url")
            if isinstance(title, str) and isinstance(target, str):
                absolute = urljoin(JPX_ROOT, target)
                items.setdefault(absolute, {"title": title, "url": absolute})
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return list(items.values())


def _static_detail_links(body: bytes) -> list[str]:
    if not body:
        return []
    soup = BeautifulSoup(body, "html.parser")
    return sorted(
        {
            urljoin(JPX_ROOT, str(node.get("href")))
            for node in soup.select('a[href*="/english/news/detail/detail_"]')
        }
    )


def _english_publication_date(value: str) -> str:
    match = re.search(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s+\d{2},\s+\d{4}",
        value,
    )
    if not match:
        return ""
    return datetime.strptime(match.group(0), "%b. %d, %Y").date().isoformat()


def _default_fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quant-lab-jpx-archive-research/1.0",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _hash_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire or validate the fail-closed pre-2013 JPX price-limit archive artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce", help="run official public acquisition")
    produce.add_argument("--run-id", required=True)
    produce.add_argument(
        "--artifact-root", default="artifacts/japan_pre2013_price_limit_archive"
    )
    produce.add_argument("--max-workers", type=int, default=6)
    validate = subparsers.add_parser("validate", help="validate an existing artifact")
    validate.add_argument("artifact_dir")
    args = parser.parse_args(argv)

    if args.command == "produce":
        result = write_japan_pre2013_price_limit_archive(
            run_id=args.run_id,
            artifact_root=args.artifact_root,
            max_workers=args.max_workers,
        )
        print(result.artifact_dir)
        return 0
    result = validate_japan_pre2013_price_limit_archive(args.artifact_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
