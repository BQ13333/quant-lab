from __future__ import annotations

import calendar
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc
ARCHIVE_START = date(2006, 7, 1)
ARCHIVE_END = date(2013, 8, 31)
REGIME_CHANGE_DATE = date(2020, 8, 3)
REQUIRED_COMPLETION_BLOCKERS = (
    "official issue-specific price-limit archive before 2013-09 is not yet proven complete",
    "post-2020 return-to-normal dates require admitted daily price/calendar evidence because JPX discontinued removal notices",
)
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
ARCHIVE_EVENT_TYPES = (
    "PRELIMINARY",
    "CANCELLED",
    "BROADENING",
    "CONTINUATION",
    "RETURN_NORMAL",
)
PRODUCER_OUTPUTS = (
    "notice_plan.csv",
    "page_state.csv",
    "price_limit_events.csv",
    "event_quality.csv",
)
RECONCILIATION_OUTPUTS = (
    "counterpart_fetch_state.csv",
    "reconciliation.csv",
    "corrected_candidate_events.csv",
)
VALIDATION_OUTPUTS = (
    "validation_summary.csv",
    "event_validation_detail.csv",
)
PLAN_OUTPUTS = (
    "source_queue.csv",
    "event_evidence_requirements.csv",
    "extraction_queue.csv",
    "admissibility_rules.json",
)


class CompletionPlanInputError(ValueError):
    """Raised when immutable upstream evidence does not satisfy the plan contract."""


@dataclass(frozen=True)
class JapanPriceLimitCompletionPlanArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_price_limit_completion_plan(
    *,
    run_id: str,
    producer_dir: str | Path,
    reconciliation_dir: str | Path,
    validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_price_limit_completion_plan",
    return_batch_size: int = 25,
    max_return_sessions: int = 20,
    max_return_lookahead_days: int = 35,
    max_archive_index_requests_per_month: int = 4,
    max_archive_notices_per_month: int = 500,
) -> JapanPriceLimitCompletionPlanArtifacts:
    """Publish an immutable, non-executing plan for the two retained blockers."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    bounds = {
        "return_batch_size": return_batch_size,
        "max_return_sessions": max_return_sessions,
        "max_return_lookahead_days": max_return_lookahead_days,
        "max_archive_index_requests_per_month": max_archive_index_requests_per_month,
        "max_archive_notices_per_month": max_archive_notices_per_month,
    }
    if any(not isinstance(value, int) or value <= 0 for value in bounds.values()):
        raise ValueError("all queue bounds must be positive integers")

    producer = Path(producer_dir).resolve()
    reconciliation = Path(reconciliation_dir).resolve()
    validation = Path(validation_dir).resolve()
    output_root = Path(artifact_root).resolve()
    root = output_root / run_id
    for source in (producer, reconciliation, validation):
        if root == source or _is_relative_to(root, source):
            raise ValueError("completion-plan output must be separate from immutable inputs")
    if root.exists():
        raise FileExistsError(f"immutable completion-plan run already exists: {root}")

    manifests, input_sha256 = _load_and_verify_inputs(
        producer=producer,
        reconciliation=reconciliation,
        validation=validation,
    )
    events = pd.read_csv(producer / "price_limit_events.csv", dtype={"code": str})
    details = pd.read_csv(validation / "event_validation_detail.csv", dtype={"code": str})

    archive_requirements = _archive_requirements(
        max_index_requests=max_archive_index_requests_per_month,
        max_notices=max_archive_notices_per_month,
    )
    event_requirements = build_return_normal_evidence_requirements(
        events,
        details,
        max_sessions=max_return_sessions,
        max_lookahead_days=max_return_lookahead_days,
    )
    source_queue = _source_queue(archive_requirements, event_requirements)
    extraction_queue = _extraction_queue(
        archive_requirements,
        event_requirements,
        return_batch_size=return_batch_size,
        max_sessions=max_return_sessions,
    )
    rules = _admissibility_rules(
        archive_months=len(archive_requirements),
        return_cases=len(event_requirements),
        bounds=bounds,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        output_paths = {name: root / name for name in PLAN_OUTPUTS}
        _atomic_csv(output_paths["source_queue.csv"], source_queue)
        _atomic_csv(output_paths["event_evidence_requirements.csv"], event_requirements)
        _atomic_csv(output_paths["extraction_queue.csv"], extraction_queue)
        _atomic_json(output_paths["admissibility_rules.json"], rules)
        output_sha256 = {name: _sha256(path) for name, path in output_paths.items()}

        manifest = {
            "schema_version": "1.0-immutable-planning-only",
            "run_id": run_id,
            "artifact_type": "japan_exceptional_price_limit_completion_plan",
            "status": "planning_complete_blockers_unresolved",
            "component": "temporary_price_limit_events",
            "planning_only": True,
            "immutable": True,
            "atomic_writes": True,
            "source_access_performed": False,
            "inference_performed": False,
            "completion_claimed": False,
            "blockers_solved": [],
            "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
            "critical_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "promotion_allowed": False,
            "promotion_policy": dict(PROMOTION_POLICY),
            "archive_coverage": {
                "start_date": ARCHIVE_START.isoformat(),
                "end_date": ARCHIVE_END.isoformat(),
                "monthly_requirements": len(archive_requirements),
                "coverage_proven": False,
            },
            "return_normal_evidence": {
                "regime_start_date": REGIME_CHANGE_DATE.isoformat(),
                "event_cases": len(event_requirements),
                "inferences_admitted": 0,
                "all_cases_resolved": False,
            },
            "counts": {
                "source_queue_rows": len(source_queue),
                "archive_month_rows": len(archive_requirements),
                "return_normal_event_cases": len(event_requirements),
                "extraction_queue_rows": len(extraction_queue),
                "enabled_extraction_rows": int(extraction_queue["execution_enabled"].sum()),
                "promotion_rows": int(extraction_queue["promotion_allowed"].sum()),
            },
            "queue_bounds": bounds,
            "upstream": {
                "producer_run_id": manifests["producer"].get("run_id"),
                "reconciliation_run_id": manifests["reconciliation"].get("run_id"),
                "validation_run_id": manifests["validation"].get("run_id"),
                "validation_status": manifests["validation"].get("status"),
            },
            "input_sha256": input_sha256,
            "artifacts": {name: str(path) for name, path in output_paths.items()},
            "output_sha256": output_sha256,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
    except Exception:
        shutil.rmtree(root)
        raise

    return JapanPriceLimitCompletionPlanArtifacts(run_id, root, manifest_path, manifest)


def build_return_normal_evidence_requirements(
    events: pd.DataFrame,
    validation_detail: pd.DataFrame,
    *,
    max_sessions: int = 20,
    max_lookahead_days: int = 35,
) -> pd.DataFrame:
    """Enumerate one unresolved evidence case for every post-regime broadening."""
    event_columns = {
        "publication_date",
        "effective_date",
        "event_type",
        "code",
        "source_url",
    }
    detail_columns = {
        "publication_date",
        "event_type",
        "code",
        "source_url",
        "resolved_side",
        "validated_base_price_yen",
        "validated_expanded_range_yen",
        "validated_normal_range_yen",
        "validated_observed_multiplier",
        "post_overlay_numeric_discrepancy",
        "post_overlay_regime_discrepancy",
        "expected_multiplier",
    }
    if not event_columns.issubset(events.columns):
        raise CompletionPlanInputError("producer event schema is incomplete")
    if not detail_columns.issubset(validation_detail.columns):
        raise CompletionPlanInputError("validation detail schema is incomplete")
    if max_sessions <= 0 or max_lookahead_days <= 0:
        raise ValueError("return evidence bounds must be positive")

    key = ["publication_date", "event_type", "code", "source_url"]
    source = events.copy()
    detail = validation_detail.copy()
    for frame in (source, detail):
        frame["code"] = frame["code"].astype("string").str.strip().str.upper()
        for column in ("publication_date", "event_type", "source_url"):
            frame[column] = frame[column].astype("string").str.strip()
    if source.duplicated(key).any() or detail.duplicated(key).any():
        raise CompletionPlanInputError("event evidence keys must be unique")
    if set(map(tuple, source[key].itertuples(index=False, name=None))) != set(
        map(tuple, detail[key].itertuples(index=False, name=None))
    ):
        raise CompletionPlanInputError("producer and validation event keys differ")

    wanted_detail = detail[sorted(detail_columns)].copy()
    merged = source.merge(wanted_detail, on=key, how="left", validate="one_to_one")
    merged["publication_stamp"] = pd.to_datetime(
        merged["publication_date"], format="%Y-%m-%d", errors="coerce"
    )
    merged["effective_stamp"] = pd.to_datetime(
        merged["effective_date"], format="%Y-%m-%d", errors="coerce"
    )
    if merged[["publication_stamp", "effective_stamp"]].isna().any().any():
        raise CompletionPlanInputError("event dates must be valid ISO dates")
    if (merged["effective_stamp"] < merged["publication_stamp"]).any():
        raise CompletionPlanInputError("event effective date precedes publication date")

    post = merged.loc[merged["publication_stamp"].dt.date >= REGIME_CHANGE_DATE].copy()
    if post.empty:
        raise CompletionPlanInputError("post-regime event denominator is empty")
    if not set(post["event_type"]).issubset({"BROADENING", "CONTINUATION"}):
        raise CompletionPlanInputError("legacy event family appears after the regime change")
    for column in ("post_overlay_numeric_discrepancy", "post_overlay_regime_discrepancy"):
        if post[column].map(_to_bool).any():
            raise CompletionPlanInputError(f"unresolved validation discrepancy: {column}")
    for column in (
        "validated_base_price_yen",
        "validated_expanded_range_yen",
        "validated_normal_range_yen",
        "validated_observed_multiplier",
    ):
        values = pd.to_numeric(post[column], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise CompletionPlanInputError(f"invalid admitted event numeric field: {column}")
        post[column] = values
    expected = pd.to_numeric(post["expected_multiplier"], errors="coerce")
    if expected.isna().any() or (expected - 4.0).abs().gt(1e-9).any():
        raise CompletionPlanInputError("post-regime expected multiplier must be 4x")
    if (post["validated_observed_multiplier"] - 4.0).abs().gt(1e-9).any():
        raise CompletionPlanInputError("post-regime validated multiplier must be 4x")
    post["resolved_side"] = post["resolved_side"].astype("string").str.upper().str.strip()
    if not post["resolved_side"].isin({"UPPER", "LOWER"}).all():
        raise CompletionPlanInputError("post-regime event side is not fully resolved")

    segments: list[dict[str, Any]] = []
    ordered = post.sort_values(
        ["code", "publication_stamp", "effective_stamp", "source_url", "event_type"]
    )
    for code, group in ordered.groupby("code", sort=True):
        active: list[dict[str, Any]] | None = None
        code_segments: list[list[dict[str, Any]]] = []
        for row in group.to_dict("records"):
            if row["event_type"] == "BROADENING":
                if active is not None:
                    code_segments.append(active)
                active = [row]
            elif active is None:
                raise CompletionPlanInputError(
                    f"post-regime continuation lacks a preceding broadening: {code}"
                )
            else:
                active.append(row)
        if active is not None:
            code_segments.append(active)
        for index, chain in enumerate(code_segments):
            next_chain = code_segments[index + 1] if index + 1 < len(code_segments) else None
            segments.append({"events": chain, "next": next_chain[0] if next_chain else None})

    segments.sort(
        key=lambda item: (
            item["events"][0]["publication_date"],
            item["events"][0]["effective_date"],
            item["events"][0]["code"],
            item["events"][0]["source_url"],
        )
    )
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, 1):
        chain = segment["events"]
        first = chain[0]
        last = chain[-1]
        if {row["resolved_side"] for row in chain} != {first["resolved_side"]}:
            raise CompletionPlanInputError(
                f"event side changes inside an unresolved chain: {first['code']}"
            )
        start = _as_date(first["effective_date"])
        anchor = _as_date(last["effective_date"])
        end = anchor + timedelta(days=max_lookahead_days - 1)
        next_event = segment["next"]
        truncated = False
        if next_event is not None:
            before_next = _as_date(next_event["effective_date"]) - timedelta(days=1)
            if before_next < end:
                end = before_next
                truncated = True
        if end < anchor or anchor < start:
            raise CompletionPlanInputError(
                f"non-positive return evidence window: {first['code']}:{first['publication_date']}"
            )
        event_keys = [
            f"{row['publication_date']}:{row['event_type']}:{row['code']}" for row in chain
        ]
        rows.append(
            {
                "case_id": f"JPLCP-RN-{index:05d}",
                "blocker_id": "post_2020_return_normal_inference",
                "code": first["code"],
                "side": first["resolved_side"],
                "broadening_publication_date": first["publication_date"],
                "broadening_effective_date": first["effective_date"],
                "broadening_source_url": first["source_url"],
                "broadening_base_price_yen": first["validated_base_price_yen"],
                "broadening_expanded_range_yen": first["validated_expanded_range_yen"],
                "broadening_normal_range_yen": first["validated_normal_range_yen"],
                "broadening_multiplier": first["validated_observed_multiplier"],
                "continuation_count": len(chain) - 1,
                "last_chain_event_type": last["event_type"],
                "last_chain_publication_date": last["publication_date"],
                "last_chain_effective_date": last["effective_date"],
                "source_event_keys": "|".join(event_keys),
                "source_event_urls": "|".join(row["source_url"] for row in chain),
                "next_broadening_publication_date": (
                    next_event["publication_date"] if next_event is not None else ""
                ),
                "next_broadening_effective_date": (
                    next_event["effective_date"] if next_event is not None else ""
                ),
                "evidence_start_date": start.isoformat(),
                "inference_search_anchor_date": anchor.isoformat(),
                "bounded_evidence_end_date": end.isoformat(),
                "window_truncated_by_next_broadening": truncated,
                "max_sessions": max_sessions,
                "max_lookahead_calendar_days": max_lookahead_days,
                "required_daily_fields": "trade_date|open|high|low|close|session_status",
                "required_calendar_fields": "session_date|is_trading_session|session_status",
                "inference_status": "not_attempted_pending_admitted_evidence",
                "return_normal_date": "",
                "promotion_allowed": False,
            }
        )
    broadening_count = int(post["event_type"].eq("BROADENING").sum())
    if len(rows) != broadening_count:
        raise CompletionPlanInputError("return evidence denominator does not equal broadenings")
    return pd.DataFrame(rows)


def _load_and_verify_inputs(
    *,
    producer: Path,
    reconciliation: Path,
    validation: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    directories = {
        "producer": producer,
        "reconciliation": reconciliation,
        "validation": validation,
    }
    manifests: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for role, directory in directories.items():
        manifest_path = directory / "manifest.json"
        manifests[role] = _read_json(manifest_path)
        hashes[f"{role}/manifest.json"] = _sha256(manifest_path)

    producer_manifest = manifests["producer"]
    reconciliation_manifest = manifests["reconciliation"]
    validation_manifest = manifests["validation"]
    if producer_manifest.get("artifact_type") != "japan_exceptional_price_limit_event_history_candidate":
        raise CompletionPlanInputError("producer artifact type is invalid")
    if (
        reconciliation_manifest.get("artifact_type")
        != "japan_exceptional_price_limit_bilingual_reconciliation_candidate"
        or reconciliation_manifest.get("status") != "partial_candidate"
        or reconciliation_manifest.get("reconciliation_blockers")
    ):
        raise CompletionPlanInputError("reconciliation contract is invalid")
    if (
        validation_manifest.get("artifact_type")
        != "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation"
        or validation_manifest.get("status") != "pass_partial_component"
        or validation_manifest.get("independent_validation") is not True
        or validation_manifest.get("validation_issues")
    ):
        raise CompletionPlanInputError("independent validation contract is invalid")
    for role, manifest in manifests.items():
        if list(manifest.get("remaining_completion_blockers", [])) != list(
            REQUIRED_COMPLETION_BLOCKERS
        ):
            raise CompletionPlanInputError(f"{role} does not retain exactly the two blockers")
        if manifest.get("promotion_policy") != PROMOTION_POLICY:
            raise CompletionPlanInputError(f"{role} promotion policy is not fail-closed")
        if manifest.get("promotion_allowed") is True:
            raise CompletionPlanInputError(f"{role} permits promotion")
    for role in ("reconciliation", "validation"):
        if list(manifests[role].get("critical_blockers", [])) != list(
            REQUIRED_COMPLETION_BLOCKERS
        ):
            raise CompletionPlanInputError(f"{role} critical blockers are not exact")

    required_by_role = {
        "producer": PRODUCER_OUTPUTS,
        "reconciliation": RECONCILIATION_OUTPUTS,
        "validation": VALIDATION_OUTPUTS,
    }
    for role, required in required_by_role.items():
        declared = manifests[role].get("output_sha256")
        if not isinstance(declared, dict) or not set(required).issubset(declared):
            raise CompletionPlanInputError(f"{role} output hashes are incomplete")
        for filename in required:
            path = directories[role] / filename
            digest = _sha256(path)
            if digest != declared[filename]:
                raise CompletionPlanInputError(f"{role} output hash mismatch: {filename}")
            hashes[f"{role}/{filename}"] = digest

    if reconciliation_manifest.get("source_manifest_sha256") != hashes["producer/manifest.json"]:
        raise CompletionPlanInputError("reconciliation source manifest hash mismatch")
    repeated = reconciliation_manifest.get("source_output_sha256")
    if not isinstance(repeated, dict) or any(
        repeated.get(filename) != hashes[f"producer/{filename}"]
        for filename in PRODUCER_OUTPUTS
    ):
        raise CompletionPlanInputError("reconciliation does not repeat producer output hashes")
    if validation_manifest.get("source_manifest_sha256") != hashes["producer/manifest.json"]:
        raise CompletionPlanInputError("validation source manifest hash mismatch")
    if (
        validation_manifest.get("reconciliation_manifest_sha256")
        != hashes["reconciliation/manifest.json"]
    ):
        raise CompletionPlanInputError("validation reconciliation manifest hash mismatch")
    return manifests, dict(sorted(hashes.items()))


def _archive_requirements(*, max_index_requests: int, max_notices: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stamp = ARCHIVE_START.replace(day=1)
    end_month = ARCHIVE_END.replace(day=1)
    index = 1
    while stamp <= end_month:
        month_end = date(stamp.year, stamp.month, calendar.monthrange(stamp.year, stamp.month)[1])
        start = max(stamp, ARCHIVE_START)
        end = min(month_end, ARCHIVE_END)
        rows.append(
            {
                "requirement_id": f"JPLCP-ARCH-{index:03d}",
                "blocker_id": "official_issue_specific_archive_coverage",
                "coverage_month": f"{stamp:%Y-%m}",
                "interval_start": start.isoformat(),
                "interval_end": end.isoformat(),
                "code_scope": "ALL_ISSUE_CODES_IN_OFFICIAL_ARCHIVE",
                "event_type_scope": "|".join(ARCHIVE_EVENT_TYPES),
                "source_authority": "official_JPX_or_TSE_primary_issue_specific_archive",
                "allowed_hosts": "jpx.co.jp|www.jpx.co.jp",
                "max_index_requests": max_index_requests,
                "max_notice_documents": max_notices,
                "coverage_status": "not_proven",
                "promotion_allowed": False,
            }
        )
        stamp = date(stamp.year + (stamp.month == 12), stamp.month % 12 + 1, 1)
        index += 1
    return pd.DataFrame(rows)


def _source_queue(archive: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in archive.to_dict("records"):
        rows.append(
            {
                "source_request_id": f"JPLCP-S-{row['requirement_id']}",
                "lane": "official_archive",
                "blocker_id": row["blocker_id"],
                "requirement_id": row["requirement_id"],
                "interval_start": row["interval_start"],
                "interval_end": row["interval_end"],
                "code": row["code_scope"],
                "event_type": row["event_type_scope"],
                "source_event_key": "",
                "source_authority": row["source_authority"],
                "allowed_hosts": row["allowed_hosts"],
                "required_evidence": "complete_issue_notice_index|raw_notice_bytes|publication_date|effective_date|event_type|code|side|source_url|sha256",
                "max_index_requests": row["max_index_requests"],
                "max_documents": row["max_notice_documents"],
                "max_sessions": 0,
                "execution_enabled": False,
                "status": "pending_bounded_official_source_discovery",
                "promotion_allowed": False,
            }
        )
    for row in events.to_dict("records"):
        rows.append(
            {
                "source_request_id": f"JPLCP-S-{row['case_id']}",
                "lane": "return_normal_evidence",
                "blocker_id": row["blocker_id"],
                "requirement_id": row["case_id"],
                "interval_start": row["evidence_start_date"],
                "interval_end": row["bounded_evidence_end_date"],
                "code": row["code"],
                "event_type": "INFER_RETURN_NORMAL",
                "source_event_key": row["source_event_keys"],
                "source_authority": "preexisting_independently_admitted_JPX_daily_price_and_calendar_artifacts",
                "allowed_hosts": "",
                "required_evidence": "unadjusted_daily_ohlc|complete_JPX_session_calendar|direct_code_identity|immutable_manifests|sha256",
                "max_index_requests": 0,
                "max_documents": 2,
                "max_sessions": row["max_sessions"],
                "execution_enabled": False,
                "status": "pending_admitted_local_evidence",
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _extraction_queue(
    archive: pd.DataFrame,
    events: pd.DataFrame,
    *,
    return_batch_size: int,
    max_sessions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(archive.to_dict("records"), 1):
        rows.append(
            {
                "batch_id": f"JPLCP-X-ARCH-{index:03d}",
                "lane": "official_archive",
                "blocker_id": row["blocker_id"],
                "requirement_count": 1,
                "requirement_ids": row["requirement_id"],
                "codes": row["code_scope"],
                "interval_start": row["interval_start"],
                "interval_end": row["interval_end"],
                "max_records": row["max_notice_documents"],
                "max_sessions_per_case": 0,
                "required_inputs": "admitted_official_archive_sources",
                "execution_enabled": False,
                "status": "blocked_pending_source_admission",
                "promotion_allowed": False,
            }
        )
    records = events.to_dict("records")
    for offset in range(0, len(records), return_batch_size):
        batch = records[offset : offset + return_batch_size]
        index = offset // return_batch_size + 1
        rows.append(
            {
                "batch_id": f"JPLCP-X-RN-{index:03d}",
                "lane": "return_normal_evidence",
                "blocker_id": "post_2020_return_normal_inference",
                "requirement_count": len(batch),
                "requirement_ids": "|".join(row["case_id"] for row in batch),
                "codes": "|".join(sorted({row["code"] for row in batch})),
                "interval_start": min(row["evidence_start_date"] for row in batch),
                "interval_end": max(row["bounded_evidence_end_date"] for row in batch),
                "max_records": len(batch) * max_sessions * 2,
                "max_sessions_per_case": max_sessions,
                "required_inputs": "admitted_daily_prices|admitted_JPX_calendar|admitted_code_identity|effective_price_limit_rules",
                "execution_enabled": False,
                "status": "blocked_pending_source_admission",
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _admissibility_rules(
    *, archive_months: int, return_cases: int, bounds: dict[str, int]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0-fail-closed",
        "default_decision": "reject",
        "planning_only": True,
        "completion_claimed": False,
        "required_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "common_rules": [
            "Every admitted input has an immutable manifest, exact file path, byte count, and SHA-256 digest.",
            "Every admitted input passes an independent validator with zero validation issues.",
            "Missing, duplicate, conflicting, adjusted, imputed, or hash-mismatched rows are rejected.",
            "Absence of a notice or price row is never affirmative evidence.",
            "Partial admission leaves the applicable blocker unresolved.",
        ],
        "official_archive_rules": {
            "required_months": archive_months,
            "coverage_start": ARCHIVE_START.isoformat(),
            "coverage_end": ARCHIVE_END.isoformat(),
            "allowed_authority": "official JPX or predecessor TSE primary issue-specific archive only",
            "allowed_hosts": ["jpx.co.jp", "www.jpx.co.jp"],
            "required_event_types": list(ARCHIVE_EVENT_TYPES),
            "required_fields": [
                "publication_date",
                "effective_date",
                "event_type",
                "code",
                "side",
                "source_url",
                "raw_path",
                "sha256",
            ],
            "completion_gate": "all 86 monthly partitions independently prove exhaustive issue-specific coverage with zero gaps, conflicts, or parse failures",
            "rejected_evidence": [
                "secondary summaries",
                "search result snippets",
                "aggregate rule tables without issue-level notices",
                "an archive endpoint without an independently proven complete denominator",
            ],
        },
        "return_normal_rules": {
            "required_cases": return_cases,
            "required_price_fields": ["trade_date", "open", "high", "low", "close", "session_status"],
            "required_calendar_fields": ["session_date", "is_trading_session", "session_status"],
            "price_basis": "raw unadjusted JPY daily prices for the exact JPX code and listing identity",
            "calendar_basis": "complete JPX session calendar including holidays and exceptional closures",
            "inference_protocol": [
                "Join each exact event chain to complete daily prices and calendar sessions from the broadening effective date.",
                "Recompute the effective exceptional-limit state using the admitted price-limit rules and every intervening JPX session.",
                "Admit a return date only as the first session for which the normal-range state is positively proven.",
                "Do not infer from elapsed time, a later broadening, missing observations, adjusted prices, or calendar-day adjacency.",
                "If the bounded window is exhausted without proof, retain the blocker and require a separately reviewed bounded extension.",
            ],
            "completion_gate": "all exact event cases have source-hashed price/calendar evidence, deterministic return dates, and independent validation",
        },
        "queue_bounds": bounds,
        "execution_enabled": False,
        "promotion_allowed": False,
        "promotion_policy": dict(PROMOTION_POLICY),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionPlanInputError(f"required manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletionPlanInputError(f"required manifest is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise CompletionPlanInputError(f"required manifest is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise CompletionPlanInputError(f"required input missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    _atomic_bytes(path, body)


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _as_date(value: Any) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CompletionPlanInputError(f"invalid ISO date: {value}") from exc


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
