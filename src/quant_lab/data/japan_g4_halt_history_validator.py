from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

from quant_lab.data.japan_g4_halt_history import (
    CHANNELS,
    COMPLETION_COLUMNS,
    DELISTING_CHANNEL,
    EVENT_COLUMNS,
    GAP_COLUMNS,
    G4_END,
    G4_START,
    HALT_CHANNEL,
    PIT_MAPPING_STATUSES,
    PIT_SAFE_STATUSES,
    REPORT_EVENT_LINEAGE_COLUMNS,
    TSE_WAYBACK_VALIDATOR_VERSION,
    _apply_lifecycle_taxonomy_repairs,
    _validate_official_timestamp_v14_contract,
)
from quant_lab.data.japan_g4_halt_report_admission import REPORT_ADMISSION_COLUMNS
from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    NORMALIZED_DISPOSITION,
    normalization_repair_for,
    verify_halt_report_normalization_repair,
)
from quant_lab.data.japan_g4_halt_report_normalization_repair_validator import (
    verify_halt_report_normalization_repair_validation,
)
from quant_lab.data.japan_g4_halt_timestamp_sidecar_admission import (
    normalize_accepted_timestamp_sidecar,
)
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair import (
    REPAIR_COLUMNS as LIFECYCLE_TAXONOMY_REPAIR_COLUMNS,
    verify_japan_g4_lifecycle_taxonomy_repair,
)
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair_validator import (
    verify_japan_g4_lifecycle_taxonomy_repair_validation,
)
from quant_lab.data.japan_g4_tse_wayback_lifecycle_validator import (
    COPIED_FILE_ROUTE,
    COPIED_PRESERVATION_TRANSPORT,
    DIRECT_ARCHIVED_PDF_LINEAGE,
    INDEPENDENT_COPIED_FILE_REGISTRY,
    WEAKER_ARCHIVED_MONTH_LINEAGE,
    independently_parse_pdf_pages,
)


UTC = timezone.utc


@dataclass(frozen=True)
class JapanG4HaltValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _LifecyclePacketValidation:
    claims: frozenset[tuple[str, str]]
    packet_count: int
    months: tuple[str, ...]
    source_rows: int


def validate_japan_g4_halt_history(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_history_validation",
) -> JapanG4HaltValidationArtifacts:
    """Independently re-read raw pages and recompute the fail-closed lane checks."""
    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    producer_manifest_path = producer / "manifest.json"
    source_manifest = _json(producer_manifest_path)
    issues: list[str] = []
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    if run_id == source_manifest.get("run_id"):
        issues.append("independence_failure_same_run_id")
    if not (
        source_manifest.get("component") == "halt_suspension_events"
        and source_manifest.get("scorecard_eligible") is False
        and source_manifest.get("independent_validation_required") is True
    ):
        issues.append("producer_contract_invalid")
    for filename, digest in source_manifest.get("output_sha256", {}).items():
        relative = Path(str(filename))
        path = (producer / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not _within(path, producer) or not path.is_file() or _sha256(path) != digest:
            issues.append(f"producer_output_hash_mismatch:{filename}")

    catalog = _csv(producer / "source_catalog.csv")
    requests = _csv(producer / "request_manifest.csv")
    coverage = _csv(producer / "acquisition_plan.csv")
    no_event = _csv(producer / "no_event_evidence.csv")
    inventory = _csv(producer / "source_inventory.csv")
    gaps = _csv(producer / "unresolved_gaps.csv")
    completion_plan = _csv(producer / "completion_plan.csv")
    events = pd.read_parquet(producer / "halt_suspension_events.parquet").fillna("")
    outside = pd.read_parquet(producer / "out_of_scope_events.parquet").fillna("")
    superseded_path = producer / "superseded_source_events.parquet"
    superseded = (
        pd.read_parquet(superseded_path).fillna("")
        if superseded_path.is_file()
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    report_admission_path = producer / "report_event_admission.csv"
    report_admission = (
        _csv(report_admission_path)
        if report_admission_path.is_file()
        else pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
    )
    report_lineage_path = producer / "report_event_lineage.csv"
    report_lineage = (
        _csv(report_lineage_path)
        if report_lineage_path.is_file()
        else pd.DataFrame(columns=REPORT_EVENT_LINEAGE_COLUMNS)
    )
    symbol_map = _csv(producer / "inputs" / "symbol_map.csv")
    g1_map_path = producer / "inputs" / "g1_reference_symbol_map.csv"
    g1_security_path = producer / "inputs" / "g1_security_master_candidates.csv"
    g1_lineage_path = producer / "inputs" / "g1_lineage.json"
    g1_map = _csv(g1_map_path) if g1_map_path.is_file() else pd.DataFrame()
    g1_security = _csv(g1_security_path) if g1_security_path.is_file() else pd.DataFrame()
    if g1_lineage_path.is_file():
        _validate_g1_lineage(_json(g1_lineage_path), g1_map_path, g1_security, issues)
    all_events = pd.concat([events, outside, superseded], ignore_index=True).fillna("")

    if (
        tuple(events.columns) != EVENT_COLUMNS
        or tuple(outside.columns) != EVENT_COLUMNS
        or tuple(superseded.columns) != EVENT_COLUMNS
    ):
        issues.append("event_schema_invalid")
    if tuple(report_admission.columns) != REPORT_ADMISSION_COLUMNS:
        issues.append("report_event_admission_schema_invalid")
    if tuple(report_lineage.columns) != REPORT_EVENT_LINEAGE_COLUMNS:
        issues.append("report_event_lineage_schema_invalid")
    complete_requests = requests.loc[requests["status"].eq("completed")]
    source_index = catalog.set_index("source_id").to_dict("index") if len(catalog) else {}
    request_index = complete_requests.set_index("source_id").to_dict("index") if len(complete_requests) else {}
    independently_parsed: dict[str, set[str]] = {}
    for source_id, request in request_index.items():
        source = source_index.get(source_id)
        if source is None:
            issues.append(f"completed_source_not_in_catalog:{source_id}")
            continue
        raw = Path(request["raw_path"])
        if not raw.is_file() or _sha256(raw) != request["raw_sha256"]:
            issues.append(f"raw_source_hash_mismatch:{source_id}")
            continue
        records = _independent_records(raw, source)
        independently_parsed[source_id] = records
        if len(records) != _integer(request["source_rows"]):
            issues.append(f"raw_source_row_count_mismatch:{source_id}")
        produced = set(all_events.loc[all_events["source_id"].eq(source_id), "source_record_id"])
        if records != produced:
            issues.append(f"independent_source_record_mismatch:{source_id}:{len(records)}/{len(produced)}")

    _validate_denominator(source_manifest, coverage, issues)
    _validate_event_semantics(events, issues)
    normalization_repairs = _validate_halt_normalization_sidecar(
        producer, source_manifest, issues
    )
    lifecycle_taxonomy_repairs = _validate_lifecycle_taxonomy_sidecar(
        producer, source_manifest, events, report_admission, issues
    )
    lifecycle = _validate_tse_wayback_lifecycle_packet(
        producer,
        source_manifest,
        events,
        superseded,
        report_admission,
        coverage,
        issues,
        apply_normalization_repairs=normalization_repairs,
        taxonomy_repairs=lifecycle_taxonomy_repairs,
    )
    _validate_halt_report_packet(
        producer,
        events,
        superseded,
        report_admission,
        coverage,
        issues,
        excluded_claims=set(lifecycle.claims),
        apply_normalization_repairs=normalization_repairs,
    )
    _validate_report_event_lineage(report_lineage, report_admission, issues)
    _validate_halt_mapping_recovery(producer, events, issues)
    _validate_coverage(
        catalog,
        complete_requests,
        coverage,
        no_event,
        events,
        issues,
        report_admission=report_admission,
    )
    _validate_mappings(
        events,
        g1_map if not g1_map.empty else symbol_map,
        g1_security,
        issues,
    )
    _validate_timestamp_recovery(producer, events, issues)
    _validate_official_timestamp_sidecar(producer, source_manifest, events, issues)
    _validate_official_timestamp_v14_sidecar(
        producer, source_manifest, events, issues
    )
    _validate_timestamps(events, g1_map, issues)
    _validate_completion_outputs(
        coverage,
        events,
        gaps,
        completion_plan,
        issues,
        report_admission=report_admission,
    )
    _validate_inventory(inventory, issues)

    producer_blockers = list(source_manifest.get("producer_blockers", []))
    complete = bool(
        not issues
        and source_manifest.get("coverage", {}).get("start_date") == G4_START.isoformat()
        and source_manifest.get("coverage", {}).get("end_date") == G4_END.isoformat()
        and len(coverage) == 480
        and coverage["coverage_status"].eq("complete").all()
    )
    status = "pass_component" if complete else "failed_validation" if issues else "pass_partial_component"
    critical = [] if complete else [*issues, *producer_blockers]
    summary = pd.DataFrame(
        [
            {"check": "producer_output_hashes", "value": len(source_manifest.get("output_sha256", {})), "passed": not any(value.startswith("producer_output_hash_mismatch") for value in issues)},
            {"check": "independent_raw_sources", "value": len(independently_parsed), "passed": not any("source_record" in value or "raw_source" in value for value in issues)},
            {"check": "month_channel_denominator", "value": len(coverage), "passed": len(coverage) == 480 and coverage["plan_id"].nunique() == 480},
            {"check": "complete_month_channels", "value": int(coverage["coverage_status"].eq("complete").sum()), "passed": complete},
            {"check": "normalized_events", "value": len(events), "passed": not any(value.startswith("event_") for value in issues)},
            {"check": "archive_inventory_files", "value": len(inventory), "passed": not any(value.startswith("inventory_") for value in issues)},
            {"check": "RIC_resolved_events", "value": int(events["refinitiv_ric"].ne("").sum()), "passed": not any(value.startswith("mapping_") for value in issues)},
            {"check": "historical_PIT_events", "value": int(events["known_at_status"].isin(PIT_SAFE_STATUSES).sum()), "passed": not any(value.startswith("timestamp_") for value in issues)},
            {"check": "machine_readable_completion_plan", "value": len(gaps), "passed": not any(value.startswith("completion_") for value in issues)},
        ]
    )
    validated_events_path = root / "validated_halt_suspension_events.parquet"
    validated_coverage_path = root / "validated_acquisition_plan.csv"
    summary_path = root / "validation_summary.csv"
    _atomic_parquet(validated_events_path, events)
    _atomic_csv(validated_coverage_path, coverage)
    _atomic_csv(summary_path, summary)
    outputs = (validated_events_path, validated_coverage_path, summary_path)
    manifest = {
        "schema_version": "1.0-independent",
        "run_id": run_id,
        "producer_run_id": source_manifest.get("run_id"),
        "artifact_type": "japan_g4_halt_suspension_history_independent_validation",
        "status": status,
        "component": "halt_suspension_events",
        "independent_validation": True,
        "official_jpx_source": source_manifest.get("official_jpx_source") is True,
        "component_scorecard_eligible": complete,
        "scorecard_eligible": False,
        "historical_archive_complete": complete,
        "source_denominator_exact": complete,
        "coverage": {
            "start_date": source_manifest.get("coverage", {}).get("start_date"),
            "end_date": source_manifest.get("coverage", {}).get("end_date"),
            "months": int(coverage["month"].nunique()) if len(coverage) else 0,
            "month_channels": len(coverage),
            "complete_month_channels": int(coverage["coverage_status"].eq("complete").sum()),
            "complete": complete,
        },
        "counts": {
            "source_rows": len(events),
            "event_rows": len(events),
            "raw_source_rows": sum(len(value) for value in independently_parsed.values()),
            "no_event_evidence_rows": len(no_event),
            "inventory_files": len(inventory),
            "ric_resolved_rows": int(events["refinitiv_ric"].ne("").sum()),
            "ric_unresolved_rows": int(events["refinitiv_ric"].eq("").sum()),
            "historical_pit_timestamp_rows": int(events["known_at_status"].isin(PIT_SAFE_STATUSES).sum()),
            "historical_pit_timestamp_gap_rows": int((~events["known_at_status"].isin(PIT_SAFE_STATUSES)).sum()),
            "completion_gap_rows": len(gaps),
            "report_event_source_rows": len(report_admission),
            "superseded_source_event_rows": len(superseded),
            "tse_wayback_lifecycle_packets": lifecycle.packet_count,
            "tse_wayback_lifecycle_months": len(lifecycle.months),
            "tse_wayback_lifecycle_claims": len(lifecycle.claims),
            "tse_wayback_lifecycle_source_rows": lifecycle.source_rows,
        },
        "tse_wayback_lifecycle_reparsed": bool(lifecycle.claims),
        "validation_issues": issues,
        "critical_blocker_count": len(critical),
        "critical_blockers": critical,
        "producer_manifest": str(producer_manifest_path),
        "producer_manifest_sha256": _sha256(producer_manifest_path),
        "artifacts": {
            "validated_halt_suspension_events": str(validated_events_path),
            "validated_acquisition_plan": str(validated_coverage_path),
            "validation_summary": str(summary_path),
        },
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "promotion_policy": {
            "alpha": False,
            "historical_backtest": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG4HaltValidationArtifacts(run_id, root, manifest_path, manifest)


def _independent_records(path: Path, source: dict[str, str]) -> set[str]:
    if source["source_kind"] == "inventory_only":
        return set()
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    records: set[str] = set()
    if source["source_kind"] == "jpx_halt_html":
        for table in soup.find_all("table")[:2]:
            for cells in _expanded_rows(table):
                if len(cells) < 7 or not re.fullmatch(r"(?=.*\d)[0-9A-Z]{4}", cells[1].strip().upper()):
                    continue
                stamp = _english_date(cells[2])
                if stamp:
                    records.add("|".join((stamp, cells[1].strip().upper(), cells[3], cells[4], cells[5], cells[6])))
    else:
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [" ".join(cell.stripped_strings) for cell in tr.find_all("td")]
                if len(cells) < 5:
                    continue
                stamp = _english_date(cells[0])
                code = cells[2].strip().upper()
                if stamp and code:
                    records.add("|".join((stamp, code, cells[4])))
    return records


def _expanded_rows(table: Any) -> list[list[str]]:
    output: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    for tr in table.find_all("tr"):
        slots: dict[int, str] = {}
        for column, (remaining, value) in list(pending.items()):
            slots[column] = value
            if remaining == 1:
                del pending[column]
            else:
                pending[column] = (remaining - 1, value)
        column = 0
        for cell in tr.find_all("td"):
            while column in slots:
                column += 1
            value = " ".join(cell.stripped_strings)
            colspan = max(1, int(cell.get("colspan", 1)))
            rowspan = max(1, int(cell.get("rowspan", 1)))
            for offset in range(colspan):
                target = column + offset
                slots[target] = value
                if rowspan > 1:
                    pending[target] = (rowspan - 1, value)
            column += colspan
        if slots:
            output.append([slots.get(index, "") for index in range(max(slots) + 1)])
    return output


def _english_date(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.strip())
    for pattern in ("%b. %d, %Y", "%b %d, %Y", "%b. %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(clean, pattern).date().isoformat()
        except ValueError:
            pass
    return ""


def _validate_denominator(manifest: dict[str, Any], coverage: pd.DataFrame, issues: list[str]) -> None:
    try:
        start = date.fromisoformat(manifest["coverage"]["start_date"])
        end = date.fromisoformat(manifest["coverage"]["end_date"])
    except (KeyError, TypeError, ValueError):
        issues.append("coverage_bounds_invalid")
        return
    months = pd.period_range(start, end, freq="M")
    expected = {f"{month}-{channel}" for month in months for channel in CHANNELS}
    actual = set(coverage["plan_id"])
    if len(coverage) != len(expected) or coverage["plan_id"].duplicated().any() or actual != expected:
        issues.append(f"month_channel_denominator_invalid:{len(actual)}/{len(expected)}")


def _validate_event_semantics(events: pd.DataFrame, issues: list[str]) -> None:
    if events.empty:
        return
    if events["event_id"].duplicated().any() or events["source_record_id"].duplicated().any():
        issues.append("event_identifiers_not_unique")
    if not events["event_type"].isin({"trading_halt", "delisting_non_trading_interval"}).all():
        issues.append("event_type_invalid")
    starts = pd.to_datetime(events["event_start_jst"], errors="coerce", utc=True)
    ends = pd.to_datetime(events["interval_end_jst"], errors="coerce", utc=True)
    present = events["interval_end_jst"].astype(str).str.strip().ne("")
    if starts.isna().any() or ends[present].isna().any() or (ends[present] < starts[present]).any():
        issues.append("event_interval_invalid")


def _validate_coverage(
    catalog: pd.DataFrame,
    requests: pd.DataFrame,
    coverage: pd.DataFrame,
    no_event: pd.DataFrame,
    events: pd.DataFrame,
    issues: list[str],
    *,
    report_admission: pd.DataFrame | None = None,
) -> None:
    completed = set(requests["source_id"])
    source_index = catalog.set_index("source_id").to_dict("index") if len(catalog) else {}
    request_index = requests.set_index("source_id").to_dict("index") if len(requests) else {}
    report = (
        report_admission.astype(str).fillna("")
        if report_admission is not None and len(report_admission)
        else pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
    )
    event_index = events.set_index("event_id", drop=False) if len(events) else pd.DataFrame()
    for row in coverage.to_dict("records"):
        report_rows = report.loc[
            report["report_month"].eq(row["month"])
            & report["channel"].eq(row["channel"])
        ].copy()
        report_claim = bool(
            len(report_rows)
            and report_rows["claim_exact_set_validated"].map(_truth).all()
        )
        research_rows = report_rows.loc[
            report_rows["in_research_scope"].map(_truth)
        ].copy()
        if report_claim:
            primary_ids = [
                value for value in research_rows["primary_event_id"].tolist() if value
            ]
            channel_events = (
                event_index.loc[event_index.index.intersection(primary_ids)].copy()
                if primary_ids and not event_index.empty
                else pd.DataFrame(columns=EVENT_COLUMNS)
            )
        else:
            channel_events = events.loc[
                events["event_date"].str[:7].eq(row["month"])
                & events["event_type"].map(
                    lambda value: HALT_CHANNEL
                    if value == "trading_halt"
                    else DELISTING_CHANNEL
                ).eq(row["channel"])
            ]
        if len(channel_events) != _integer(row["event_rows"]):
            issues.append(f"coverage_event_count_mismatch:{row['plan_id']}")
        claimed = [value for value in row["source_ids"].split("|") if value]
        for source_id in claimed:
            if source_id == f"jpx-monthly-report-{row['month']}-{row['channel']}":
                if not report_claim:
                    issues.append(f"coverage_report_source_claim_invalid:{row['plan_id']}")
                continue
            source = source_index.get(source_id, {})
            if source_id not in completed or not _truth(source.get("denominator_complete")) or source.get("coverage_start", "") > row["window_start"] or source.get("coverage_end", "") < row["window_end"]:
                issues.append(f"coverage_source_claim_invalid:{row['plan_id']}:{source_id}")
        evidence = no_event.loc[(no_event["month"].eq(row["month"])) & (no_event["channel"].eq(row["channel"]))]
        if _truth(row["no_event_evidence"]) != bool(len(evidence)):
            issues.append(f"no_event_claim_mismatch:{row['plan_id']}")
        for item in evidence.to_dict("records"):
            if item["source_id"] == f"jpx-monthly-report-{row['month']}-{row['channel']}":
                if not (
                    report_claim
                    and research_rows.empty
                    and "zero common-equity rows" in item["assertion"]
                ):
                    issues.append(f"report_no_event_evidence_invalid:{item['evidence_id']}")
                continue
            source = source_index.get(item["source_id"], {})
            request = request_index.get(item["source_id"], {})
            if len(channel_events) or not _truth(source.get("no_event_capable")) or item["source_sha256"] != request.get("raw_sha256"):
                issues.append(f"no_event_evidence_invalid:{item['evidence_id']}")
        source_covered = _truth(row["source_covered"])
        expected_source_covered = bool(
            report_claim
            or any(
                source_id in completed
                and _truth(source_index.get(source_id, {}).get("denominator_complete"))
                for source_id in claimed
                if not source_id.startswith("jpx-monthly-report-")
            )
        )
        if source_covered != expected_source_covered:
            issues.append(f"coverage_source_status_mismatch:{row['plan_id']}")
        report_normalized = bool(
            not report_claim
            or research_rows.empty
            or research_rows["normalization_status"].eq("normalized").all()
        )
        if report_claim and research_rows.empty:
            expected_pit = False
        elif report_claim:
            expected_pit = bool(
                report_normalized
                and research_rows["timestamp_complete"].map(_truth).all()
            )
        elif len(channel_events):
            expected_pit = source_covered and channel_events["known_at_status"].isin(PIT_SAFE_STATUSES).all()
        else:
            eligible = catalog.loc[catalog["source_id"].isin(claimed)]
            expected_pit = bool(
                source_covered
                and len(evidence)
                and len(eligible)
                and eligible["pit_timestamp_status"].eq("exact_publication_timestamp").all()
            )
        if report_claim:
            expected_mapping = bool(
                research_rows.empty
                or (
                    report_normalized
                    and research_rows["mapping_complete"].map(_truth).all()
                )
            )
        else:
            expected_mapping = bool(
                channel_events.empty
                or channel_events["mapping_status"].isin(PIT_MAPPING_STATUSES).all()
            )
        if _truth(row["pit_timestamp_complete"]) != bool(expected_pit):
            issues.append(f"coverage_pit_claim_mismatch:{row['plan_id']}")
        if _truth(row["listing_ric_mapping_complete"]) != expected_mapping:
            issues.append(f"coverage_mapping_claim_mismatch:{row['plan_id']}")
        expected_complete = (
            _truth(row["source_covered"])
            and report_normalized
            and (len(channel_events) > 0 or len(evidence) > 0)
            and _truth(row["pit_timestamp_complete"])
            and _truth(row["listing_ric_mapping_complete"])
        )
        if (row["coverage_status"] == "complete") != expected_complete:
            issues.append(f"coverage_status_invalid:{row['plan_id']}")


def _validate_mappings(
    events: pd.DataFrame,
    symbol_map: pd.DataFrame,
    security_candidates: pd.DataFrame,
    issues: list[str],
) -> None:
    if events.empty:
        return
    for event in events.to_dict("records"):
        status = event["mapping_status"]
        if status not in PIT_MAPPING_STATUSES | {
            "deterministic_reference_post_event",
            "missing_admitted_mapping",
        }:
            issues.append(f"mapping_status_unsupported:{event['event_id']}:{status}")
            continue
        if status == "pit_exact_official_monthly_quote_recovery":
            continue
        if status in {"pit_exact_g1_identity_retire_bridge", "deterministic_reference_post_event"}:
            candidates = _independent_retirement_candidates(event, security_candidates)
            if not (
                len(candidates) == 1
                and candidates.iloc[0]["candidate_ric"] == event["refinitiv_ric"]
                and candidates.iloc[0]["admitted_source_sha256"] == event["mapping_source_sha256"]
                and candidates.iloc[0]["source_record_id"] == event["mapping_source_record_id"]
            ):
                issues.append(f"mapping_retirement_recompute_failed:{event['event_id']}")
                continue
            relations = symbol_map.loc[
                symbol_map["local_code"].eq(event["local_code"])
                & symbol_map["refinitiv_ric"].eq(event["refinitiv_ric"])
                & symbol_map["known_at"].ne("")
                & symbol_map["known_at"].str[:10].le(event["event_date"])
            ]
            if status == "pit_exact_g1_identity_retire_bridge" and relations.empty:
                issues.append(f"mapping_bridge_identity_missing:{event['event_id']}")
            if status == "deterministic_reference_post_event" and not relations.empty:
                issues.append(f"mapping_post_event_status_too_weak:{event['event_id']}")
            continue
        if status == "missing_admitted_mapping":
            if event["refinitiv_ric"] or len(_independent_retirement_candidates(event, security_candidates)):
                issues.append(f"mapping_missing_claim_invalid:{event['event_id']}")
            continue
        candidates = symbol_map.loc[symbol_map["local_code"].eq(event["local_code"])].copy()
        mapping_day = (
            (date.fromisoformat(event["event_date"]) - timedelta(days=1)).isoformat()
            if event["event_type"] == "delisting_non_trading_interval"
            else event["event_date"]
        )
        if "listing_start" in candidates:
            candidates = candidates.loc[candidates["listing_start"].eq("") | candidates["listing_start"].le(mapping_day)]
        if "listing_end" in candidates:
            right_open = candidates["right_open"].map(_truth) if "right_open" in candidates else pd.Series(False, index=candidates.index)
            candidates = candidates.loc[
                right_open
                | candidates["listing_end"].eq("")
                | candidates["listing_end"].ge(mapping_day)
            ]
        if "pit_usable" in candidates:
            candidates = candidates.loc[candidates["pit_usable"].map(_truth)]
        if "known_at" in candidates:
            candidates = candidates.loc[candidates["known_at"].ne("") & candidates["known_at"].str[:10].le(event["event_date"])]
        candidates = candidates.loc[candidates["refinitiv_ric"].eq(event["refinitiv_ric"])]
        if len(candidates) != 1:
            issues.append(f"pit_mapping_recompute_failed:{event['event_id']}")


def _independent_retirement_candidates(
    event: dict[str, str], security: pd.DataFrame
) -> pd.DataFrame:
    if security.empty or event["event_type"] != "delisting_non_trading_interval":
        return pd.DataFrame()
    code = re.escape(event["local_code"])
    return security.loc[
        security["local_code"].eq(event["local_code"])
        & security["retire_date"].eq(event["event_date"])
        & security["exchange_code"].eq("TYO")
        & security["candidate_ric"].str.match(
            rf"^{code}\.T(?:\^.*)?$", case=False, na=False
        )
    ].drop_duplicates("candidate_ric")


def _validate_timestamps(
    events: pd.DataFrame, symbol_map: pd.DataFrame, issues: list[str]
) -> None:
    map_by_listing = (
        symbol_map.set_index("stable_listing_id", drop=False).to_dict("index")
        if not symbol_map.empty
        and "stable_listing_id" in symbol_map
        and not symbol_map["stable_listing_id"].duplicated().any()
        else {}
    )
    for event in events.to_dict("records"):
        status = event["known_at_status"]
        if status == "conservative_halt_start_market_state":
            expected = pd.Timestamp(event["event_start_jst"]).tz_convert("UTC").isoformat()
            if event["known_at_utc"] != expected or event["known_at_evidence_sha256"] != event["source_sha256"]:
                issues.append(f"timestamp_halt_start_recompute_failed:{event['event_id']}")
        elif status == "conservative_g1_monthly_absence_observation":
            mapping = map_by_listing.get(event["stable_listing_id"], {})
            expected = str(mapping.get("window_end_known_at", ""))
            if not (
                expected
                and str(mapping.get("listing_end", "")) <= event["event_date"]
                and expected[:10] >= event["event_date"]
                and event["known_at_utc"] == expected
            ):
                issues.append(f"timestamp_g1_absence_recompute_failed:{event['event_id']}")
        elif status == "missing_historical_publication_or_absence_timestamp":
            if event["known_at_utc"]:
                issues.append(f"timestamp_missing_claim_has_value:{event['event_id']}")
        elif status == "official_jpx_publication_date_recovery":
            if not event["known_at_utc"] or not re.fullmatch(
                r"[0-9a-f]{64}", str(event["known_at_evidence_sha256"])
            ):
                issues.append(f"timestamp_recovery_value_invalid:{event['event_id']}")
        elif status not in PIT_SAFE_STATUSES:
            issues.append(f"timestamp_status_unsupported:{event['event_id']}:{status}")


def _validate_timestamp_recovery(
    producer: Path, events: pd.DataFrame, issues: list[str]
) -> None:
    root = producer / "inputs" / "timestamp_recovery"
    if not root.exists():
        return
    producer_manifest_path = root / "producer_manifest.json"
    validation_manifest_path = root / "validation_manifest.json"
    lineage_path = root / "lineage.json"
    recovery_path = root / "timestamp_recovery.csv"
    recovery_manifest = _json(producer_manifest_path)
    validation = _json(validation_manifest_path)
    lineage = _json(lineage_path)
    if not (
        recovery_manifest.get("status") == "recovery_partial"
        and validation.get("status") == "pass"
        and validation.get("independent_validation_passed") is True
        and validation.get("producer_manifest_sha256") == _sha256(producer_manifest_path)
        and lineage.get("producer_manifest_sha256") == _sha256(producer_manifest_path)
        and lineage.get("validation_manifest_sha256") == _sha256(validation_manifest_path)
    ):
        issues.append("timestamp_recovery_lineage_invalid")
        return
    for filename, digest in recovery_manifest.get("output_sha256", {}).items():
        path = root / filename
        if not path.is_file() or _sha256(path) != digest:
            issues.append(f"timestamp_recovery_output_hash_mismatch:{filename}")
    for filename, digest in validation.get("output_sha256", {}).items():
        path = root / "validation" / filename
        if not path.is_file() or _sha256(path) != digest:
            issues.append(f"timestamp_recovery_validation_hash_mismatch:{filename}")
    recovery = _csv(recovery_path)
    recovered = recovery.loc[recovery["recovery_status"].eq("recovered")]
    if (
        recovered.empty
        or recovered["event_id"].duplicated().any()
        or len(recovered) != _integer(lineage.get("recovered_rows", -1))
    ):
        issues.append("timestamp_recovery_denominator_invalid")
        return
    event_index = events.set_index("event_id", drop=False).to_dict("index")
    for row in recovered.to_dict("records"):
        event = event_index.get(row["event_id"])
        raw_candidates = list((root / "raw").glob(f"{row['notice_source_sha256']}.*"))
        if (
            event is None
            or event["known_at_status"] != "official_jpx_publication_date_recovery"
            or event["known_at_utc"] != pd.Timestamp(row["known_at_utc"]).tz_convert("UTC").isoformat()
            or event["known_at_evidence_sha256"] != row["notice_source_sha256"]
            or len(raw_candidates) != 1
            or _sha256(raw_candidates[0]) != row["notice_source_sha256"]
        ):
            issues.append(f"timestamp_recovery_event_mismatch:{row['event_id']}")


def _validate_halt_normalization_sidecar(
    producer: Path, source_manifest: dict[str, Any], issues: list[str]
) -> bool:
    root = producer / "inputs" / "halt_normalization_repair"
    admitted = source_manifest.get("halt_normalization_repair_admitted") is True
    if not admitted:
        if root.exists():
            issues.append("halt_normalization_sidecar_present_without_manifest_admission")
        return False
    if not root.is_dir():
        issues.append("halt_normalization_sidecar_missing")
        return False
    start_issues = len(issues)
    producer_packet = root / "producer"
    validation_packet = root / "validation"
    for value in verify_halt_report_normalization_repair(producer_packet):
        issues.append(f"halt_normalization_sidecar_producer:{value}")
    for value in verify_halt_report_normalization_repair_validation(
        validation_packet
    ):
        issues.append(f"halt_normalization_sidecar_validation:{value}")
    lineage = _json(root / "lineage.json") if (root / "lineage.json").is_file() else {}
    repairs = _csv(producer_packet / "normalization_repairs.csv")
    if not (
        len(repairs) == 10
        and repairs["source_record_sha256"].is_unique
        and _integer(lineage.get("repair_rows_reconciled", -1)) == 10
        and _integer(lineage.get("instrument_event_rows", -1)) == 1
        and _integer(lineage.get("appendix_reference_rows", -1)) == 3
        and _integer(lineage.get("market_aggregate_rows", -1)) == 6
        and _integer(lineage.get("invented_local_codes", -1)) == 0
        and lineage.get("source_hashes_verified") is True
        and lineage.get("independent_semantics_verified") is True
        and lineage.get("producer_manifest_sha256")
        == _sha256(producer_packet / "manifest.json")
        and lineage.get("validation_manifest_sha256")
        == _sha256(validation_packet / "manifest.json")
    ):
        issues.append("halt_normalization_sidecar_lineage_invalid")
    return len(issues) == start_issues


def _validate_lifecycle_taxonomy_sidecar(
    producer: Path,
    source_manifest: dict[str, Any],
    events: pd.DataFrame,
    admission: pd.DataFrame,
    issues: list[str],
) -> pd.DataFrame:
    root = producer / "inputs" / "lifecycle_taxonomy_repair"
    admitted = source_manifest.get("lifecycle_taxonomy_repair_admitted") is True
    if not admitted:
        if root.exists():
            issues.append("lifecycle_taxonomy_sidecar_present_without_admission")
        return pd.DataFrame(columns=LIFECYCLE_TAXONOMY_REPAIR_COLUMNS)
    if not root.is_dir():
        issues.append("lifecycle_taxonomy_sidecar_missing")
        return pd.DataFrame(columns=LIFECYCLE_TAXONOMY_REPAIR_COLUMNS)

    producer_packet = root / "producer"
    validation_packet = root / "validation"
    for value in verify_japan_g4_lifecycle_taxonomy_repair(producer_packet):
        issues.append(f"lifecycle_taxonomy_sidecar_producer:{value}")
    for value in verify_japan_g4_lifecycle_taxonomy_repair_validation(
        validation_packet
    ):
        issues.append(f"lifecycle_taxonomy_sidecar_validation:{value}")
    repairs = _csv(producer_packet / "taxonomy_repairs.csv")
    lineage = _json(root / "lineage.json") if (root / "lineage.json").is_file() else {}
    producer_manifest = _json(producer_packet / "manifest.json")
    validation_manifest = _json(validation_packet / "manifest.json")
    if not (
        tuple(repairs.columns) == tuple(LIFECYCLE_TAXONOMY_REPAIR_COLUMNS)
        and len(repairs) == 9
        and repairs["event_id"].is_unique
        and repairs["source_record_sha256"].is_unique
        and _integer(source_manifest.get("lifecycle_taxonomy_repair_rows", -1))
        == 9
        and _integer(lineage.get("repair_rows", -1)) == 9
        and _integer(lineage.get("warrant_rows", -1)) == 5
        and _integer(lineage.get("foreign_share_rows", -1)) == 4
        and _integer(lineage.get("research_scope_rows_after", -1)) == 0
        and _integer(lineage.get("mappings_created", -1)) == 0
        and _integer(lineage.get("timestamps_created", -1)) == 0
        and _integer(lineage.get("zero_events_created", -1)) == 0
        and _integer(lineage.get("classification_inferred_rows", -1)) == 0
        and lineage.get("independently_validated") is True
        and lineage.get("producer_manifest_sha256")
        == _sha256(producer_packet / "manifest.json")
        and lineage.get("validation_manifest_sha256")
        == _sha256(validation_packet / "manifest.json")
        and validation_manifest.get("producer_manifest_sha256")
        == _sha256(producer_packet / "manifest.json")
        and validation_manifest.get("producer_run_id")
        == producer_manifest.get("run_id")
    ):
        issues.append("lifecycle_taxonomy_sidecar_lineage_invalid")

    admission_index = admission.set_index("report_event_id", drop=False).to_dict(
        "index"
    )
    primary_ids = set(events["event_id"])
    for repair in repairs.to_dict("records"):
        row = admission_index.get(repair["event_id"])
        if (
            row is None
            or row["source_record_sha256"] != repair["source_record_sha256"]
            or row["instrument_class"]
            != repair["reclassified_instrument_class"]
            or _truth(row["in_research_scope"])
            or row["normalization_status"] != "out_of_research_universe"
            or row["primary_event_id"]
            or f"jpx-report-{repair['event_id']}" in primary_ids
        ):
            issues.append(
                f"lifecycle_taxonomy_sidecar_primary_mismatch:{repair['event_id']}"
            )
    return repairs


def _validate_official_timestamp_sidecar(
    producer: Path,
    source_manifest: dict[str, Any],
    events: pd.DataFrame,
    issues: list[str],
) -> None:
    root = producer / "inputs" / "official_timestamp_recovery"
    admitted = source_manifest.get("official_timestamp_recovery_admitted") is True
    if not admitted:
        if root.exists():
            issues.append("official_timestamp_sidecar_present_without_manifest_admission")
        return
    if not root.is_dir():
        issues.append("official_timestamp_sidecar_missing")
        return
    producer_packet = root / "producer"
    validation_packet = root / "validation"
    producer_manifest = _json(producer_packet / "manifest.json")
    validation_manifest = _json(validation_packet / "manifest.json")
    try:
        accepted = normalize_accepted_timestamp_sidecar(
            _csv(producer_packet / "accepted_timestamp_recovery.csv")
        )
    except Exception as exc:
        issues.append(
            f"official_timestamp_sidecar_claim_validation_failed:{type(exc).__name__}"
        )
        return
    if not _validate_copied_timestamp_sidecar_files(
        producer_packet,
        validation_packet,
        producer_manifest,
        validation_manifest,
        issues,
    ):
        return
    lineage = _json(root / "lineage.json") if (root / "lineage.json").is_file() else {}
    if not (
        len(accepted) == 10
        and accepted["event_id"].is_unique
        and _integer(source_manifest.get("official_timestamp_recovery_rows", -1))
        == len(accepted)
        and _integer(lineage.get("target_events", -1)) == len(accepted)
        and _integer(lineage.get("source_responses", -1)) == 12
        and lineage.get("known_at_rule")
        == "date_only_conservative_upper_bound_next_midnight_jst"
    ):
        issues.append("official_timestamp_sidecar_lineage_invalid")
    event_index = events.set_index("event_id", drop=False).to_dict("index")
    for item in accepted.to_dict("records"):
        event = event_index.get(item["event_id"])
        if (
            event is None
            or event["known_at_utc"] != item["known_at_utc"]
            or event["known_at_status"]
            != "official_jpx_publication_date_recovery"
            or event["known_at_evidence_sha256"] != item["source_sha256"]
        ):
            issues.append(
                f"official_timestamp_sidecar_event_invalid:{item['event_id']}"
            )


def _validate_official_timestamp_v14_sidecar(
    producer: Path,
    source_manifest: dict[str, Any],
    events: pd.DataFrame,
    issues: list[str],
) -> None:
    root = producer / "inputs" / "official_timestamp_v14_recovery"
    admitted = source_manifest.get("official_timestamp_v14_recovery_admitted") is True
    if not admitted:
        if root.exists():
            issues.append("official_timestamp_v14_sidecar_present_without_admission")
        return
    if not root.is_dir():
        issues.append("official_timestamp_v14_sidecar_missing")
        return
    producer_packet = root / "producer"
    validation_packet = root / "validation"
    try:
        accepted, producer_files, validation_files = (
            _validate_official_timestamp_v14_contract(
                producer_packet, validation_packet
            )
        )
    except Exception as exc:
        issues.append(
            f"official_timestamp_v14_sidecar_contract_failed:{type(exc).__name__}"
        )
        return
    lineage = _json(root / "lineage.json") if (root / "lineage.json").is_file() else {}
    if not (
        len(accepted) == 6
        and accepted["event_id"].is_unique
        and _integer(source_manifest.get("official_timestamp_v14_recovery_rows", -1))
        == 6
        and _integer(lineage.get("target_events", -1)) == 6
        and _integer(lineage.get("source_responses", -1)) == 6
        and lineage.get("known_at_rule")
        == "date_only_conservative_upper_bound_next_midnight_jst"
        and lineage.get("official_jpx_only") is True
        and lineage.get("independently_validated") is True
        and lineage.get("refinitiv_calls") is False
        and lineage.get("producer_manifest_sha256")
        == _sha256(producer_packet / "manifest.json")
        and lineage.get("validation_manifest_sha256")
        == _sha256(validation_packet / "manifest.json")
        and len(producer_files) == 20
        and len(validation_files) == 4
    ):
        issues.append("official_timestamp_v14_sidecar_lineage_invalid")
    event_index = events.set_index("event_id", drop=False).to_dict("index")
    for item in accepted.to_dict("records"):
        event = event_index.get(item["event_id"])
        if (
            event is None
            or event["known_at_utc"] != item["known_at_utc"]
            or event["known_at_status"]
            != "official_jpx_publication_date_recovery"
            or event["known_at_evidence_sha256"] != item["source_sha256"]
            or event["local_code"] != item["local_code"]
            or event["event_date"] != item["event_date"]
            or event["source_record_id"] != item["source_record_id"]
        ):
            issues.append(
                f"official_timestamp_v14_sidecar_event_invalid:{item['event_id']}"
            )


def _validate_copied_timestamp_sidecar_files(
    producer: Path,
    validation: Path,
    producer_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    issues: list[str],
) -> bool:
    valid = True
    producer_sha = _sha256(producer / "manifest.json")
    producer_outputs = producer_manifest.get("outputs", {})
    validation_outputs = validation_manifest.get("outputs", {})
    for root, outputs, label in (
        (producer, producer_outputs, "producer"),
        (validation, validation_outputs, "validation"),
    ):
        if not isinstance(outputs, dict):
            issues.append(f"official_timestamp_sidecar_{label}_outputs_invalid")
            valid = False
            continue
        for relative_name, metadata in outputs.items():
            relative = Path(str(relative_name))
            expected = metadata.get("sha256", "") if isinstance(metadata, dict) else ""
            path = root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not _within(path.resolve(), root.resolve())
                or path.is_symlink()
                or not path.is_file()
                or _sha256(path) != expected
            ):
                issues.append(
                    f"official_timestamp_sidecar_{label}_output_invalid:{relative_name}"
                )
                valid = False
    sources = _csv(producer / "official_source_manifest.csv")
    if len(sources) != 12 or not sources["source_id"].is_unique:
        issues.append("official_timestamp_sidecar_source_denominator_invalid")
        valid = False
    for item in sources.to_dict("records"):
        raw = producer / "raw" / Path(item["raw_path"]).name
        metadata = producer / "raw" / "metadata" / Path(item["metadata_path"]).name
        if (
            raw.is_symlink()
            or metadata.is_symlink()
            or not raw.is_file()
            or not metadata.is_file()
            or _sha256(raw) != item["sha256"]
            or _sha256(metadata) != item["metadata_sha256"]
        ):
            issues.append(
                f"official_timestamp_sidecar_source_invalid:{item['source_id']}"
            )
            valid = False
    counts = validation_manifest.get("counts", {})
    if not (
        producer_manifest.get("status") == "complete"
        and producer_manifest.get("refinitiv_calls") is False
        and validation_manifest.get("status") == "pass"
        and validation_manifest.get("independent_implementation") is True
        and validation_manifest.get("retained_bytes_reparsed") is True
        and validation_manifest.get("producer_manifest", {}).get("sha256")
        == producer_sha
        and _integer(counts.get("target_events", -1)) == 10
        and _integer(counts.get("independently_validated_events", -1)) == 10
        and _integer(counts.get("validation_issues", -1)) == 0
        and validation_manifest.get("validation_issues") == []
    ):
        issues.append("official_timestamp_sidecar_manifest_contract_invalid")
        valid = False
    return valid


def _validate_tse_wayback_lifecycle_packet(
    producer: Path,
    source_manifest: dict[str, Any],
    events: pd.DataFrame,
    superseded: pd.DataFrame,
    admission: pd.DataFrame,
    coverage: pd.DataFrame,
    issues: list[str],
    *,
    apply_normalization_repairs: bool = False,
    taxonomy_repairs: pd.DataFrame | None = None,
) -> _LifecyclePacketValidation:
    root = producer / "inputs" / "tse_wayback_lifecycle"
    if not root.exists():
        if source_manifest.get("tse_wayback_lifecycle_packet_admitted") is True:
            issues.append("tse_wayback_lifecycle_packet_missing")
        return _LifecyclePacketValidation(frozenset(), 0, (), 0)

    lineage_path = root / "lineage.json"
    packet_root = root / "packets"
    if not lineage_path.is_file() or not packet_root.is_dir():
        issues.append("tse_wayback_lifecycle_collection_layout_invalid")
        return _LifecyclePacketValidation(frozenset(), 0, (), 0)
    aggregate = _json(lineage_path)
    declared_packets = aggregate.get("packets", [])
    if not isinstance(declared_packets, list):
        issues.append("tse_wayback_lifecycle_packet_lineage_invalid")
        declared_packets = []
    declared_ids = [str(item.get("packet_id", "")) for item in declared_packets]
    discovered_ids = sorted(path.name for path in packet_root.iterdir() if path.is_dir())
    direct_files = [path.name for path in packet_root.iterdir() if path.is_file()]
    if (
        any(re.fullmatch(r"packet-\d{4}", value) is None for value in declared_ids)
        or len(set(declared_ids)) != len(declared_ids)
        or sorted(declared_ids) != discovered_ids
        or direct_files
        or any(path.is_symlink() for path in packet_root.iterdir())
        or set(path.name for path in root.iterdir()) != {"lineage.json", "packets"}
    ):
        issues.append("tse_wayback_lifecycle_packet_directory_denominator_invalid")

    all_claims: set[tuple[str, str]] = set()
    all_months: set[str] = set()
    source_frames: list[pd.DataFrame] = []
    for item in declared_packets:
        packet_id = str(item.get("packet_id", ""))
        if packet_id not in discovered_ids:
            continue
        expected_path = f"packets/{packet_id}"
        if item.get("packet_path") != expected_path:
            issues.append(f"tse_wayback_lifecycle_packet_path_invalid:{packet_id}")
        packet_claims, packet_months, source_events = _validate_lifecycle_packet_copy(
            packet_root / packet_id, item, packet_id, issues
        )
        overlap = all_claims & packet_claims
        if overlap:
            month, channel = sorted(overlap)[0]
            issues.append(
                f"tse_wayback_lifecycle_claim_overlap:{month}:{channel}"
            )
        all_claims.update(packet_claims)
        all_months.update(packet_months)
        if not source_events.empty:
            source_frames.append(source_events)

    source_events = (
        pd.concat(source_frames, ignore_index=True)
        if source_frames
        else pd.DataFrame()
    )
    primary_source_events = (
        _apply_lifecycle_taxonomy_repairs(source_events, taxonomy_repairs)
        if taxonomy_repairs is not None and not taxonomy_repairs.empty
        else source_events
    )
    packet_summary = source_manifest.get("tse_wayback_lifecycle_packet", {})
    expected_common_equity = int(
        source_events.get("instrument_class", pd.Series(dtype=str)).eq("equity").sum()
    )
    expected_non_equity = len(source_events) - expected_common_equity
    aggregate_contract = (
        source_manifest.get("tse_wayback_lifecycle_packet_admitted") is True
        and _integer(aggregate.get("packet_count", -1)) == len(declared_packets)
        and sorted(aggregate.get("packet_ids", [])) == sorted(declared_ids)
        and sorted(aggregate.get("selected_months", [])) == sorted(all_months)
        and _integer(aggregate.get("selected_month_count", -1)) == len(all_months)
        and _integer(aggregate.get("covered_claims", -1)) == len(all_claims)
        and _integer(aggregate.get("claim_count", -1)) == len(all_claims)
        and _integer(aggregate.get("source_rows", -1)) == len(source_events)
        and _integer(aggregate.get("source_event_count", -1)) == len(source_events)
        and _integer(aggregate.get("common_equity_rows", -1))
        == expected_common_equity
        and _integer(aggregate.get("preserved_non_equity_rows", -1))
        == expected_non_equity
        and aggregate.get("absence_is_zero") is False
        and packet_summary.get("packets") == declared_packets
        and _integer(packet_summary.get("packet_count", -1))
        == len(declared_packets)
        and sorted(packet_summary.get("months", [])) == sorted(all_months)
        and _integer(packet_summary.get("month_count", -1)) == len(all_months)
        and _integer(packet_summary.get("covered_claims", -1)) == len(all_claims)
        and _integer(packet_summary.get("claim_count", -1)) == len(all_claims)
        and _integer(packet_summary.get("source_rows", -1)) == len(source_events)
        and _integer(packet_summary.get("common_equity_rows", -1))
        == expected_common_equity
        and _integer(packet_summary.get("preserved_non_equity_rows", -1))
        == expected_non_equity
        and _integer(packet_summary.get("zero_event_claims", -1)) == 0
    )
    if not aggregate_contract:
        issues.append("tse_wayback_lifecycle_aggregate_lineage_invalid")

    admission_claims = pd.Series(
        list(zip(admission["report_month"], admission["channel"], strict=False)),
        index=admission.index,
        dtype=object,
    )
    lifecycle_admission = admission.loc[admission_claims.isin(all_claims)].copy()
    if (
        len(lifecycle_admission) != len(source_events)
        or lifecycle_admission["source_record_sha256"].duplicated().any()
        or set(lifecycle_admission["source_record_sha256"])
        != set(source_events.get("source_record_sha256", pd.Series(dtype=str)))
    ):
        issues.append("tse_wayback_lifecycle_primary_admission_set_invalid")
    elif len(source_events):
        _validate_lifecycle_primary_admission(
            primary_source_events,
            lifecycle_admission,
            events,
            all_claims,
            issues,
            apply_normalization_repairs=apply_normalization_repairs,
        )

    claim_series = pd.Series(
        list(zip(events["event_date"].str[:7], events["event_type"].map(_event_channel), strict=False)),
        index=events.index,
        dtype=object,
    )
    non_report_primary = events.loc[
        claim_series.isin(all_claims)
        & ~events["source_id"].str.startswith("jpx-monthly-report-")
    ]
    superseded_claims = pd.Series(
        list(
            zip(
                superseded["event_date"].str[:7],
                superseded["event_type"].map(_event_channel),
                strict=False,
            )
        ),
        index=superseded.index,
        dtype=object,
    )
    invalid_superseded = superseded.loc[
        superseded_claims.isin(all_claims)
        & superseded["source_id"].str.startswith("jpx-monthly-report-")
    ]
    if len(non_report_primary) or len(invalid_superseded):
        issues.append("tse_wayback_lifecycle_supersession_invalid")

    coverage_index = coverage.set_index(["month", "channel"], drop=False)
    for claim in all_claims:
        if claim not in coverage_index.index:
            issues.append(f"tse_wayback_lifecycle_coverage_missing:{claim[0]}:{claim[1]}")
            continue
        row = coverage_index.loc[claim]
        expected_source = f"jpx-monthly-report-{claim[0]}-{claim[1]}"
        claim_sources = primary_source_events.loc[
            primary_source_events["report_month"].eq(claim[0])
            & primary_source_events["channel"].eq(claim[1])
        ]
        expected_no_equity = not claim_sources["instrument_class"].eq("equity").any()
        if isinstance(row, pd.DataFrame) or not (
            _truth(row["source_covered"])
            and expected_source in row["source_ids"].split("|")
            and _truth(row["no_event_evidence"]) == expected_no_equity
        ):
            issues.append(f"tse_wayback_lifecycle_coverage_invalid:{claim[0]}:{claim[1]}")
    return _LifecyclePacketValidation(
        frozenset(all_claims), len(declared_packets), tuple(sorted(all_months)), len(source_events)
    )


def _validate_lifecycle_packet_copy(
    root: Path,
    lineage: dict[str, Any],
    packet_id: str,
    issues: list[str],
) -> tuple[set[tuple[str, str]], tuple[str, ...], pd.DataFrame]:
    producer_packet = root / "producer"
    validation_packet = root / "validation"
    if (
        not root.is_dir()
        or set(path.name for path in root.iterdir())
        != {"producer", "validation", "lineage.json"}
        or any(path.is_symlink() for path in root.iterdir())
    ):
        issues.append(f"tse_wayback_lifecycle_packet_layout_invalid:{packet_id}")
    producer_manifest = _validate_sealed_packet_copy(
        producer_packet, f"tse_wayback_producer:{packet_id}", issues
    )
    validation_manifest = _validate_sealed_packet_copy(
        validation_packet, f"tse_wayback_validation:{packet_id}", issues
    )
    packet_lineage_path = root / "lineage.json"
    required = (
        producer_packet / "exact_events.csv",
        producer_packet / "exact_events.parquet",
        producer_packet / "exact_nonempty_claims.csv",
        producer_packet / "source_inventory.csv",
        producer_packet / "transport_inventory.csv",
        validation_packet / "month_validation.csv",
        validation_packet / "event_set_reconciliation.csv",
        validation_packet / "independent_exact_events.csv",
        packet_lineage_path,
    )
    if not all(path.is_file() for path in required):
        issues.append(f"tse_wayback_lifecycle_packet_files_missing:{packet_id}")
        return set(), (), pd.DataFrame()
    packet_lineage = _json(packet_lineage_path)
    producer_manifest_path = producer_packet / "manifest.json"
    validation_manifest_path = validation_packet / "manifest.json"
    months = tuple(str(value) for value in producer_manifest.get("selected_months", []))
    month_count = len(months)
    claim_count = _integer(producer_manifest.get("exact_nonempty_claim_count", -1))
    event_count = _integer(producer_manifest.get("exact_event_count", -1))
    expected_claims = {(month, channel) for month in months for channel in CHANNELS}
    copied_route = producer_manifest.get("route_modes") == [COPIED_FILE_ROUTE]
    expected_producer_version = (
        "1.1-jpx-copied-file-lifecycle"
        if copied_route
        else "1.0-tse-wayback-lifecycle"
    )
    expected_validation_version = (
        "2.1-independent-jpx-copied-file-pdfplumber"
        if copied_route
        else TSE_WAYBACK_VALIDATOR_VERSION
    )
    contract = (
        bool(months)
        and len(set(months)) == month_count
        and list(months) == sorted(months)
        and all(re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", month) for month in months)
        and producer_manifest.get("artifact_type")
        == "bounded_tse_wayback_lifecycle_source_acquisition"
        and producer_manifest.get("status") == "bounded_acquisition_complete"
        and producer_manifest.get("version") == expected_producer_version
        and _integer(producer_manifest.get("selected_month_count", -1)) == month_count
        and _integer(producer_manifest.get("acquired_exact_pdf_month_count", -1))
        == month_count
        and claim_count == len(expected_claims)
        and event_count > 0
        and _integer(producer_manifest.get("failure_count", -1)) == 0
        and _integer(producer_manifest.get("zero_event_claims_published", -1)) == 0
        and producer_manifest.get("absence_is_zero") is False
        and producer_manifest.get("preservation_transport")
        == (
            COPIED_PRESERVATION_TRANSPORT
            if copied_route
            else "Internet Archive Wayback Machine"
        )
        and producer_manifest.get("original_source")
        == "Tokyo Stock Exchange / JPX official monthly PDF bytes"
        and producer_manifest.get("refinitiv_used") is False
        and producer_manifest.get("write_once_after_manifest") is True
        and (
            not copied_route
            or (
                set(months).issubset(INDEPENDENT_COPIED_FILE_REGISTRY)
                and _integer(
                    producer_manifest.get("official_copied_file_registry_month_count", -1)
                )
                == month_count
                and _integer(producer_manifest.get("registry_entries_guessed", -1)) == 0
                and producer_manifest.get("live_official_pdf_response_preserved") is True
            )
        )
        and validation_manifest.get("artifact_type")
        == "independent_tse_wayback_lifecycle_validation"
        and validation_manifest.get("version") == expected_validation_version
        and validation_manifest.get("status")
        == "pass_bounded_exact_nonempty_evidence"
        and validation_manifest.get("producer_manifest_sha256")
        == _sha256(producer_manifest_path)
        and _integer(validation_manifest.get("selected_month_count", -1))
        == month_count
        and _integer(validation_manifest.get("capture_valid_month_count", -1))
        == month_count
        and _integer(validation_manifest.get("independent_exact_event_count", -1))
        == event_count
        and _integer(validation_manifest.get("reconciled_nonempty_claim_count", -1))
        == claim_count
        and _integer(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("validation_issue_codes") == []
        and validation_manifest.get("absence_is_zero") is False
        and validation_manifest.get("missing_capture_fails_closed") is True
        and validation_manifest.get("parser_disagreement_fails_closed") is True
        and validation_manifest.get("refinitiv_used") is False
        and validation_manifest.get("write_once") is True
        and (
            not copied_route
            or (
                validation_manifest.get("route_modes") == [COPIED_FILE_ROUTE]
                and validation_manifest.get("archived_parent_lineage_strengths")
                == sorted(
                    {
                        INDEPENDENT_COPIED_FILE_REGISTRY[month][
                            "archived_parent_lineage_strength"
                        ]
                        for month in months
                    }
                )
                and validation_manifest.get(
                    "live_official_pdf_response_independently_validated"
                )
                is True
            )
        )
        and lineage.get("producer_manifest_sha256") == _sha256(producer_manifest_path)
        and lineage.get("validation_manifest_sha256")
        == _sha256(validation_manifest_path)
        and packet_lineage == {
            key: value
            for key, value in lineage.items()
            if key not in {"packet_id", "packet_path"}
        }
    )
    if not contract:
        issues.append(f"tse_wayback_lifecycle_lineage_or_contract_invalid:{packet_id}")

    source_events = _csv(producer_packet / "exact_events.csv")
    parquet_events = pd.read_parquet(producer_packet / "exact_events.parquet").fillna("")
    parquet_events = parquet_events.astype(str).reindex(columns=source_events.columns)
    claims = _csv(producer_packet / "exact_nonempty_claims.csv")
    inventory = _csv(producer_packet / "source_inventory.csv")
    transport = _csv(producer_packet / "transport_inventory.csv")
    month_validation = _csv(validation_packet / "month_validation.csv")
    reconciliation = _csv(validation_packet / "event_set_reconciliation.csv")
    independent = _csv(validation_packet / "independent_exact_events.csv")
    claim_keys = set(zip(claims["report_month"], claims["channel"], strict=False))
    grouped = source_events.groupby(["report_month", "channel"]).size().to_dict()
    claim_counts = {
        (row["report_month"], row["channel"]): _integer(row["event_count"])
        for row in claims.to_dict("records")
    }
    exact = (
        len(source_events) == len(parquet_events) == event_count
        and source_events.astype(str).equals(parquet_events)
        and source_events["event_id"].nunique() == event_count
        and source_events["source_record_sha256"].nunique() == event_count
        and len(claims) == claim_count
        and claim_keys == expected_claims
        and grouped == claim_counts
        and sum(claim_counts.values()) == event_count
        and claims["claim_kind"].eq("exact_nonempty_official_report_event_set").all()
        and claims["event_count"].map(_integer).gt(0).all()
        and not claims["zero_event_asserted"].map(_truth).any()
        and source_events["known_at"].eq("").all()
        and source_events["ric"].eq("").all()
        and not source_events["admissible_to_shared_halt_history"].map(_truth).any()
        and len(month_validation) == month_count
        and set(month_validation["report_month"]) == set(months)
        and month_validation["capture_valid"].map(_truth).all()
        and (
            not copied_route
            or _copied_lifecycle_validation_rows_exact(
                month_validation,
                months,
                source_events,
            )
        )
        and len(reconciliation) == claim_count
        and reconciliation["validated"].map(_truth).all()
        and not reconciliation["zero_inferred"].map(_truth).any()
        and len(independent) == event_count
        and _lifecycle_event_counter(source_events) == _lifecycle_event_counter(independent)
    )
    if not exact:
        issues.append(f"tse_wayback_lifecycle_exact_denominator_invalid:{packet_id}")

    reparsed_rows: list[dict[str, Any]] = []
    if len(inventory) != month_count or set(inventory["report_month"]) != set(months):
        issues.append(f"tse_wayback_lifecycle_source_inventory_invalid:{packet_id}")
    else:
        inventory_by_month = inventory.set_index("report_month").to_dict("index")
        if copied_route and not _copied_lifecycle_transport_exact(
            producer_packet,
            months,
            inventory_by_month,
            transport,
        ):
            issues.append(f"tse_wayback_lifecycle_transport_invalid:{packet_id}")
        for source in inventory.to_dict("records"):
            month = source["report_month"]
            relative = Path(str(source["pdf_path"]))
            pdf_path = (producer_packet / relative).resolve()
            official = source["official_pdf_url"]
            common_source = (
                relative.as_posix() == f"raw/pdfs/{month}.pdf"
                and not relative.is_absolute()
                and ".." not in relative.parts
                and _within(pdf_path, producer_packet)
                and pdf_path.is_file()
                and _sha256(pdf_path) == source["pdf_sha256"]
            )
            route_source = (
                _copied_lifecycle_source_exact(
                    producer_packet,
                    month,
                    source,
                )
                if copied_route
                else (
                    _official_tse_url(official)
                    and _official_tse_url(source["official_monthly_page_url"])
                    and _wayback_transport_url(
                        source["pdf_wayback_url"],
                        source["pdf_cdx_timestamp"],
                        official,
                    )
                )
            )
            if not common_source or not route_source:
                issues.append(
                    f"tse_wayback_lifecycle_source_lineage_invalid:{packet_id}:{month}"
                )
                continue
            try:
                with pdfplumber.open(io.BytesIO(pdf_path.read_bytes())) as document:
                    pages = [page.extract_text() or "" for page in document.pages]
                parsed = independently_parse_pdf_pages(
                    pages,
                    month,
                    source_url=official,
                    source_path=relative.as_posix(),
                    source_sha256=source["pdf_sha256"],
                )
            except Exception as exc:
                issues.append(
                    "tse_wayback_lifecycle_independent_reparse_failed:"
                    f"{packet_id}:{month}:{type(exc).__name__}"
                )
                continue
            if not parsed.get("exact_month"):
                issues.append(
                    f"tse_wayback_lifecycle_pdf_month_invalid:{packet_id}:{month}"
                )
            reparsed_rows.extend(parsed.get("events", []))
        if copied_route and not _copied_lifecycle_claim_and_event_sources_exact(
            claims,
            source_events,
            inventory_by_month,
        ):
            issues.append(f"tse_wayback_lifecycle_source_claim_mismatch:{packet_id}")
    reparsed = pd.DataFrame(reparsed_rows).reindex(columns=source_events.columns).fillna("")
    if len(reparsed) != event_count or not source_events.astype(str).equals(
        reparsed.astype(str)
    ):
        issues.append(f"tse_wayback_lifecycle_independent_event_set_mismatch:{packet_id}")
    return expected_claims, months, source_events


def _validate_lifecycle_primary_admission(
    source_events: pd.DataFrame,
    admission: pd.DataFrame,
    events: pd.DataFrame,
    claims: set[tuple[str, str]],
    issues: list[str],
    *,
    apply_normalization_repairs: bool = False,
) -> None:
    source_index = source_events.set_index("source_record_sha256", drop=False).to_dict(
        "index"
    )
    event_index = events.set_index("event_id", drop=False).to_dict("index")
    expected_primary: set[str] = set()
    for admitted in admission.to_dict("records"):
        source, expected_status, expected_in_scope = _normalization_contract(
            source_index[admitted["source_record_sha256"]],
            apply_repairs=apply_normalization_repairs,
        )
        primary_id = (
            f"jpx-report-{source['event_id']}" if expected_status == "normalized" else ""
        )
        if (
            admitted["normalization_status"] != expected_status
            or admitted["primary_event_id"] != primary_id
            or _truth(admitted["in_research_scope"])
            != expected_in_scope
            or admitted["source_sha256"] != source["source_sha256"]
        ):
            issues.append(f"tse_wayback_lifecycle_admission_invalid:{source['event_id']}")
        if primary_id:
            expected_primary.add(primary_id)
            primary = event_index.get(primary_id)
            if not primary or any(
                (
                    primary["local_code"] != source["normalized_local_code"],
                    primary["event_date"] != source["event_date"],
                    primary["source_record_id"] != source["source_record_sha256"],
                    primary["source_url"] != source["source_url"],
                )
            ):
                issues.append(
                    f"tse_wayback_lifecycle_primary_event_invalid:{source['event_id']}"
                )
    lifecycle_sources = {
        f"jpx-monthly-report-{month}-{channel}" for month, channel in claims
    }
    actual_primary = set(events.loc[events["source_id"].isin(lifecycle_sources), "event_id"])
    if actual_primary != expected_primary:
        issues.append("tse_wayback_lifecycle_primary_event_set_invalid")


def _validate_report_event_lineage(
    lineage: pd.DataFrame, admission: pd.DataFrame, issues: list[str]
) -> None:
    if admission.empty and lineage.empty:
        return
    if lineage.empty:
        issues.append("report_event_lineage_missing")
        return
    if (
        lineage.duplicated(["source_packet", "source_record_sha256"]).any()
        or not lineage["disposition"].isin(
            {"admitted", "duplicate_equivalent_superseded"}
        ).all()
    ):
        issues.append("report_event_lineage_identity_invalid")
        return
    selected = lineage.loc[lineage["disposition"].eq("admitted")].copy()
    if (
        selected["source_record_sha256"].duplicated().any()
        or set(selected["source_record_sha256"])
        != set(admission["source_record_sha256"])
    ):
        issues.append("report_event_lineage_admission_set_invalid")
        return
    admission_index = admission.set_index("source_record_sha256", drop=False).to_dict(
        "index"
    )
    selected_keys = set(
        zip(selected["source_packet"], selected["source_record_sha256"], strict=False)
    )
    for row in lineage.to_dict("records"):
        admitted = admission_index.get(row["selected_source_record_sha256"])
        if (
            (row["selected_packet"], row["selected_source_record_sha256"])
            not in selected_keys
            or admitted is None
            or row["selected_report_event_id"] != admitted["report_event_id"]
            or row["primary_event_id"] != admitted["primary_event_id"]
        ):
            issues.append(
                f"report_event_lineage_target_invalid:{row['source_packet']}:{row['source_event_id']}"
            )


def _report_normalization_status(source: dict[str, str]) -> str:
    if source["instrument_class"] != "equity":
        return "out_of_research_universe"
    if not source["normalized_local_code"]:
        return "missing_local_code"
    if date.fromisoformat(source["event_date"]) > pd.Period(
        source["report_month"], freq="M"
    ).end_time.date():
        return "future_dated_source_record"
    if source["channel"] == HALT_CHANNEL and not source["halt_start_jst"]:
        return "missing_halt_start"
    if source["channel"] == HALT_CHANNEL and source["halt_end_jst"]:
        start = pd.to_datetime(source["halt_start_jst"], errors="coerce", utc=True)
        end = pd.to_datetime(source["halt_end_jst"], errors="coerce", utc=True)
        if pd.isna(end) or end < start:
            return "invalid_halt_interval"
    return "normalized"


def _normalization_contract(
    source: dict[str, str], *, apply_repairs: bool
) -> tuple[dict[str, str], str, bool]:
    normalized = dict(source)
    repair = normalization_repair_for(normalized) if apply_repairs else None
    if repair is not None and repair.normalization_disposition == NORMALIZED_DISPOSITION:
        normalized["halt_start_printed"] = repair.repaired_halt_start_printed
        normalized["halt_end_printed"] = repair.repaired_halt_end_printed
        normalized["halt_start_jst"] = repair.repaired_halt_start_jst
        normalized["halt_end_jst"] = repair.repaired_halt_end_jst
    in_scope = normalized["instrument_class"] == "equity" and not (
        repair is not None and not repair.in_research_scope_after
    )
    status = (
        repair.normalization_disposition
        if repair is not None and not repair.in_research_scope_after
        else _report_normalization_status(normalized)
    )
    return normalized, status, in_scope


def _event_channel(event_type: str) -> str:
    return (
        DELISTING_CHANNEL
        if event_type == "delisting_non_trading_interval"
        else HALT_CHANNEL
    )


def _lifecycle_event_counter(frame: pd.DataFrame) -> Counter[tuple[str, ...]]:
    return Counter(
        (
            str(row.get("report_month", "")),
            str(row.get("channel", "")),
            str(row.get("event_date", "")),
            str(row.get("raw_code", "")),
            _canonical_report_text(str(row.get("issue_name", ""))),
        )
        for row in frame.to_dict("records")
    )


def _canonical_report_text(value: str) -> str:
    result = unicodedata.normalize("NFKC", value)
    result = " ".join(result.split())
    result = re.sub(r"\s+([),.;])", r"\1", result)
    result = re.sub(r"([(])\s+", r"\1", result)
    return result.strip()


def _official_tse_url(value: Any) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == "www.tse.or.jp"
        and parsed.path.startswith("/english/market/data/geppo/")
        and not parsed.query
        and not parsed.fragment
    )


def _wayback_transport_url(value: Any, timestamp: Any, original: Any) -> bool:
    prefix = f"https://web.archive.org/web/{timestamp}id_/"
    archived = str(value)
    return bool(
        _official_tse_url(original)
        and re.fullmatch(r"\d{14}", str(timestamp))
        and archived.startswith(prefix)
        and _same_official_url(archived[len(prefix) :], str(original))
    )


def _same_official_url(left: str, right: str) -> bool:
    def normalized(value: str) -> tuple[str, str, str, str, str]:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        authority = parsed.hostname or ""
        if port is not None and not (
            (parsed.scheme == "http" and port == 80)
            or (parsed.scheme == "https" and port == 443)
        ):
            authority = f"{authority}:{port}"
        return parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment

    return _official_tse_url(left) and normalized(left) == normalized(right)


def _copied_lifecycle_validation_rows_exact(
    validation: pd.DataFrame,
    months: tuple[str, ...],
    events: pd.DataFrame,
) -> bool:
    required = {
        "report_month",
        "route_mode",
        "archived_parent_lineage_strength",
        "capture_valid",
        "html_sha256_match",
        "pdf_sha256_match",
        "byte_lengths_match",
        "pdf_magic_valid",
        "cdx_contract_valid",
        "url_month_contract_valid",
        "archived_html_exact_link_valid",
        "transport_contract_valid",
        "independent_exact_month",
        "required_sections_valid",
        "independent_delisting_events",
        "independent_halt_events",
        "parse_error",
    }
    if not required.issubset(validation.columns):
        return False
    boolean_columns = (
        "capture_valid",
        "html_sha256_match",
        "pdf_sha256_match",
        "byte_lengths_match",
        "pdf_magic_valid",
        "cdx_contract_valid",
        "url_month_contract_valid",
        "archived_html_exact_link_valid",
        "transport_contract_valid",
        "independent_exact_month",
        "required_sections_valid",
    )
    counts = events.groupby(["report_month", "channel"]).size().to_dict()
    for month in months:
        selected = validation.loc[validation["report_month"].eq(month)]
        route = INDEPENDENT_COPIED_FILE_REGISTRY.get(month)
        if len(selected) != 1 or route is None:
            return False
        row = selected.iloc[0]
        if not (
            row["route_mode"] == COPIED_FILE_ROUTE
            and row["archived_parent_lineage_strength"]
            == route["archived_parent_lineage_strength"]
            and all(_truth(row[column]) for column in boolean_columns)
            and str(row["parse_error"]) == ""
            and _integer(row["independent_delisting_events"])
            == counts.get((month, DELISTING_CHANNEL), 0)
            and _integer(row["independent_halt_events"])
            == counts.get((month, HALT_CHANNEL), 0)
        ):
            return False
    return True


def _copied_lifecycle_source_exact(
    producer: Path,
    month: str,
    source: dict[str, Any],
) -> bool:
    route = INDEPENDENT_COPIED_FILE_REGISTRY.get(month)
    if route is None:
        return False
    official = str(source.get("official_pdf_url", ""))
    parent = str(source.get("parent_index_cdx_original", ""))
    timestamp = str(source.get("parent_index_cdx_timestamp", ""))
    parent_path = (producer / str(source.get("parent_index_path", ""))).resolve()
    pdf_path = (producer / str(source.get("pdf_path", ""))).resolve()
    cdx_path = (producer / f"raw/cdx/{month}_parent_index.json").resolve()
    direct = route["archived_parent_lineage_strength"] == DIRECT_ARCHIVED_PDF_LINEAGE
    source_contract = (
        source.get("route_mode") == COPIED_FILE_ROUTE
        and official == route["official_pdf_url"]
        and _official_jpx_lifecycle_pdf(official, month)
        and source.get("official_monthly_page_url") == route["parent_link_url"]
        and source.get("parent_index_url") == route["parent_index_url"]
        and parent == route["parent_index_url"]
        and timestamp == route["parent_capture_timestamp"]
        and source.get("parent_link_url") == route["parent_link_url"]
        and source.get("archived_parent_lineage_strength")
        == route["archived_parent_lineage_strength"]
        and _truth(source.get("archived_parent_direct_pdf_link")) == direct
        and (
            source.get("parent_link_url") == official
            if direct
            else _official_tse_monthly_page(source.get("parent_link_url"), month)
        )
        and source.get("preservation_transport", "").casefold()
        == COPIED_PRESERVATION_TRANSPORT.casefold()
        and source.get("original_source_publisher") == "Tokyo Stock Exchange / JPX"
        and not _truth(source.get("registry_entry_guessed"))
        and not _truth(source.get("zero_inferred"))
        and source.get("parent_index_cdx_statuscode") == "200"
        and source.get("parent_index_cdx_mimetype") == "text/html"
        and re.fullmatch(r"[A-Z2-7]{32}", str(source.get("parent_index_cdx_digest", "")))
        is not None
        and _integer(source.get("parent_index_cdx_length")) > 0
        and source.get("parent_index_http_status") == "200"
        and source.get("parent_index_path") == f"raw/pages/{month}.html"
        and _truth(source.get("parent_index_exact_link_proved"))
        and source.get("pdf_final_url") == official
        and source.get("pdf_http_status") == "200"
        and source.get("pdf_content_type") == "application/pdf"
        and source.get("pdf_magic") == "%PDF-"
        and _copied_wayback_url(
            source.get("parent_index_wayback_url"), timestamp, parent
        )
    )
    retained_contract = (
        _within(parent_path, producer)
        and parent_path.is_file()
        and _sha256(parent_path) == source.get("parent_index_response_sha256")
        and parent_path.stat().st_size
        == _integer(source.get("parent_index_response_byte_length"))
        and _copied_parent_index_lineage_exact(parent_path, source)
        and _within(pdf_path, producer)
        and pdf_path.is_file()
        and pdf_path.read_bytes().startswith(b"%PDF-")
        and _sha256(pdf_path) == source.get("pdf_sha256")
        and pdf_path.stat().st_size == _integer(source.get("pdf_byte_length"))
        and _within(cdx_path, producer)
        and cdx_path.is_file()
        and _copied_cdx_contains_capture(
            cdx_path,
            timestamp=timestamp,
            original=parent,
            statuscode=str(source.get("parent_index_cdx_statuscode", "")),
            digest=str(source.get("parent_index_cdx_digest", "")),
            length=str(source.get("parent_index_cdx_length", "")),
            mimetype=str(source.get("parent_index_cdx_mimetype", "")),
        )
    )
    return bool(source_contract and retained_contract)


def _copied_lifecycle_transport_exact(
    producer: Path,
    months: tuple[str, ...],
    inventory: dict[str, dict[str, Any]],
    transport: pd.DataFrame,
) -> bool:
    required = {
        "report_month",
        "request_kind",
        "request_url",
        "final_url",
        "http_status",
        "raw_path",
        "byte_length",
        "sha256",
    }
    kinds = {"parent_index_cdx", "parent_index_capture", "official_pdf_live"}
    if (
        not required.issubset(transport.columns)
        or len(transport) != len(months) * len(kinds)
        or set(transport["report_month"]) != set(months)
        or set(transport["request_kind"]) != kinds
        or transport.duplicated(["report_month", "request_kind"]).any()
        or not transport["http_status"].eq("200").all()
    ):
        return False
    for month in months:
        source = inventory.get(month)
        if source is None:
            return False
        rows = transport.loc[transport["report_month"].eq(month)].set_index(
            "request_kind"
        )
        cdx = rows.loc["parent_index_cdx"]
        capture = rows.loc["parent_index_capture"]
        pdf = rows.loc["official_pdf_live"]
        cdx_path = f"raw/cdx/{month}_parent_index.json"
        if not (
            _copied_cdx_request_url(
                str(cdx["request_url"]), str(source["parent_index_cdx_original"])
            )
            and cdx["request_url"] == cdx["final_url"]
            and _transport_file_exact(producer, cdx, cdx_path)
            and capture["request_url"] == source["parent_index_wayback_url"]
            and capture["final_url"] == source["parent_index_wayback_url"]
            and _transport_file_exact(
                producer, capture, f"raw/pages/{month}.html"
            )
            and capture["sha256"] == source["parent_index_response_sha256"]
            and _integer(capture["byte_length"])
            == _integer(source["parent_index_response_byte_length"])
            and pdf["request_url"] == source["official_pdf_url"]
            and pdf["final_url"] == source["official_pdf_url"]
            and _transport_file_exact(producer, pdf, f"raw/pdfs/{month}.pdf")
            and pdf["sha256"] == source["pdf_sha256"]
            and _integer(pdf["byte_length"])
            == _integer(source["pdf_byte_length"])
        ):
            return False
    return True


def _copied_lifecycle_claim_and_event_sources_exact(
    claims: pd.DataFrame,
    events: pd.DataFrame,
    inventory: dict[str, dict[str, Any]],
) -> bool:
    for frame in (claims, events):
        if set(frame["report_month"]) - set(inventory):
            return False
    for month, source in inventory.items():
        official = source["official_pdf_url"]
        digest = source["pdf_sha256"]
        month_claims = claims.loc[claims["report_month"].eq(month)]
        month_events = events.loc[events["report_month"].eq(month)]
        if not (
            len(month_claims) == len(CHANNELS)
            and month_claims["official_source_url"].eq(official).all()
            and month_claims["source_sha256"].eq(digest).all()
            and month_claims["archive_timestamp"].eq(
                source["parent_index_cdx_timestamp"]
            ).all()
            and month_claims["wayback_url"].eq(
                source["parent_index_wayback_url"]
            ).all()
            and month_claims["validation_status"].eq(
                "pending_independent_validation"
            ).all()
            and not month_claims["zero_event_asserted"].map(_truth).any()
            and month_events["source_url"].eq(official).all()
            and month_events["source_path"].eq(f"raw/pdfs/{month}.pdf").all()
            and month_events["source_sha256"].eq(digest).all()
            and month_events["known_at"].eq("").all()
            and month_events["ric"].eq("").all()
            and not month_events["admissible_to_shared_halt_history"].map(_truth).any()
        ):
            return False
    return True


def _official_jpx_lifecycle_pdf(value: Any, month: str) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    expected_suffix = f"_idou{month[2:4]}{month[5:7]}.pdf"
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
        and (
            parsed.path.startswith("/files/tse/english/market/data/geppo/")
            or parsed.path.startswith(
                "/english/markets/statistics-equities/monthly/"
            )
        )
        and parsed.path.endswith(expected_suffix)
        and not parsed.query
        and not parsed.fragment
    )


def _official_tse_monthly_page(value: Any, month: str) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return bool(
        _official_tse_url(value)
        and parsed.path
        == f"/english/market/data/geppo/{month.replace('-', '')}.html"
    )


def _copied_parent_index(value: Any) -> bool:
    return str(value) in {
        route["parent_index_url"]
        for route in INDEPENDENT_COPIED_FILE_REGISTRY.values()
    }


def _copied_wayback_url(value: Any, timestamp: str, original: str) -> bool:
    return bool(
        _copied_parent_index(original)
        and re.fullmatch(r"\d{14}", timestamp)
        and str(value) == f"https://web.archive.org/web/{timestamp}id_/{original}"
    )


def _copied_cdx_request_url(value: str, original: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    query = urllib.parse.parse_qs(parsed.query)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "web.archive.org"
        and parsed.path == "/cdx/search/cdx"
        and query.get("url") == [original]
        and query.get("output") == ["json"]
        and query.get("fl")
        == ["timestamp,original,statuscode,digest,length,mimetype"]
        and query.get("filter") == ["statuscode:200"]
        and query.get("collapse") == ["digest"]
    )


def _copied_cdx_contains_capture(
    path: Path,
    *,
    timestamp: str,
    original: str,
    statuscode: str,
    digest: str,
    length: str,
    mimetype: str,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        header = payload[0]
        rows = [dict(zip(header, row, strict=True)) for row in payload[1:]]
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "timestamp": timestamp,
        "original": original,
        "statuscode": statuscode,
        "digest": digest,
        "length": length,
        "mimetype": mimetype,
    }
    return expected in rows


def _copied_parent_index_lineage_exact(
    path: Path,
    source: dict[str, Any],
) -> bool:
    try:
        headers = json.loads(str(source.get("parent_index_http_headers", "")))
        response = path.read_bytes()
        if str(headers.get("content-encoding", "")).casefold() == "gzip":
            decoded = gzip.decompress(response)
        else:
            decoded = response
        links = {
            urllib.parse.urljoin(str(source.get("parent_index_url", "")), href)
            for node in BeautifulSoup(decoded, "html.parser").find_all("a", href=True)
            if (href := node.get("href"))
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    target = str(source.get("parent_link_url", ""))
    return bool(
        _sha256_bytes(decoded) == source.get("parent_index_decoded_sha256")
        and len(decoded) == _integer(source.get("parent_index_decoded_byte_length"))
        and any(_same_web_url(link, target) for link in links)
    )


def _same_web_url(left: str, right: str) -> bool:
    def normalized(value: str) -> tuple[str, str, int | None, str, str, str]:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        if (parsed.scheme, port) in {("http", 80), ("https", 443)}:
            port = None
        return (
            parsed.scheme,
            parsed.hostname or "",
            port,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )

    return normalized(left) == normalized(right)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _transport_file_exact(producer: Path, row: pd.Series, relative_name: str) -> bool:
    relative = Path(str(row["raw_path"]))
    path = (producer / relative).resolve()
    return bool(
        relative.as_posix() == relative_name
        and not relative.is_absolute()
        and ".." not in relative.parts
        and _within(path, producer)
        and path.is_file()
        and path.stat().st_size == _integer(row["byte_length"])
        and _sha256(path) == row["sha256"]
    )


def _validate_halt_report_packet(
    producer: Path,
    events: pd.DataFrame,
    superseded: pd.DataFrame,
    admission: pd.DataFrame,
    coverage: pd.DataFrame,
    issues: list[str],
    *,
    excluded_claims: set[tuple[str, str]] | None = None,
    apply_normalization_repairs: bool = False,
) -> None:
    excluded = excluded_claims or set()
    admission_claims = pd.Series(
        list(zip(admission["report_month"], admission["channel"], strict=False)),
        index=admission.index,
        dtype=object,
    )
    selected_admission = admission.loc[~admission_claims.isin(excluded)].copy()
    root = producer / "inputs" / "halt_report_events"
    if not root.exists():
        if len(selected_admission):
            issues.append("report_event_admission_without_packet")
        return
    producer_packet = root / "producer"
    validation_packet = root / "validation"
    producer_manifest = _validate_sealed_packet_copy(
        producer_packet, "report_producer", issues
    )
    validation_manifest = _validate_sealed_packet_copy(
        validation_packet, "report_validation", issues
    )
    lineage_path = root / "lineage.json"
    lineage = _json(lineage_path) if lineage_path.is_file() else {}
    producer_manifest_path = producer_packet / "manifest.json"
    validation_manifest_path = validation_packet / "manifest.json"
    producer_sha = _sha256(producer_manifest_path)
    validation_sha = _sha256(validation_manifest_path)
    expected_rows = _integer(producer_manifest.get("explicit_report_rows", -1))
    expected_claims = _integer(
        producer_manifest.get("covered_month_channel_claims", -1)
    )
    if not (
        producer_manifest.get("status")
        == "bounded_report_event_candidate_pending_independent_validation"
        and validation_manifest.get("status")
        == "pass_bounded_report_event_component"
        and validation_manifest.get("producer_manifest_sha256") == producer_sha
        and _integer(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("all_report_hashes_verified") is True
        and validation_manifest.get("all_input_hashes_verified") is True
        and validation_manifest.get("absence_is_zero") is False
        and validation_manifest.get("inferred_zero_rejected") is True
        and _integer(validation_manifest.get("producer_report_rows", -1))
        == expected_rows
        and _integer(validation_manifest.get("independent_report_rows", -1))
        == expected_rows
        and _integer(validation_manifest.get("exact_field_reconciled_rows", -1))
        == expected_rows
        and _integer(validation_manifest.get("covered_claims_reparsed", -1))
        == expected_claims
        and _integer(validation_manifest.get("exact_claim_row_sets", -1))
        == expected_claims
        and producer_manifest.get("denominator_manifest_sha256")
        == validation_manifest.get("denominator_manifest_sha256")
        and producer_manifest.get("validation_manifest_sha256")
        == validation_manifest.get("denominator_validation_manifest_sha256")
        and lineage.get("producer_manifest_sha256") == producer_sha
        and lineage.get("validation_manifest_sha256") == validation_sha
        and _integer(lineage.get("report_rows", -1)) == expected_rows
        and _integer(lineage.get("covered_claims", -1)) == expected_claims
        and lineage.get("absence_is_zero") is False
        and lineage.get("research_instrument_class") == "equity"
    ):
        issues.append("report_event_packet_lineage_or_contract_invalid")

    report_events = pd.read_parquet(
        producer_packet / "report_events.parquet"
    ).fillna("")
    claims = _csv(validation_packet / "claim_exact_row_set_reconciliation.csv")
    report_claims = pd.Series(
        list(zip(report_events["report_month"], report_events["channel"], strict=False)),
        index=report_events.index,
        dtype=object,
    )
    selected_report_events = report_events.loc[~report_claims.isin(excluded)].copy()
    claim_index = pd.Series(
        list(zip(claims["report_month"], claims["channel"], strict=False)),
        index=claims.index,
        dtype=object,
    )
    selected_claims = claims.loc[~claim_index.isin(excluded)].copy()
    if (
        len(report_events) != expected_rows
        or len(selected_admission) != len(selected_report_events)
        or report_events["source_record_sha256"].duplicated().any()
        or selected_admission["source_record_sha256"].duplicated().any()
        or set(selected_report_events["source_record_sha256"])
        != set(selected_admission["source_record_sha256"])
        or len(claims) != expected_claims
        or not claims["exact_nonempty_row_set"].map(_truth).all()
        or claims["zero_inferred"].map(_truth).any()
    ):
        issues.append("report_event_exact_source_set_invalid")
        return

    event_index = events.set_index("event_id", drop=False).to_dict("index")
    report_index = selected_report_events.set_index("source_record_sha256", drop=False).to_dict(
        "index"
    )
    expected_primary_ids: set[str] = set()
    for admitted in selected_admission.to_dict("records"):
        source = report_index.get(admitted["source_record_sha256"])
        if source is None:
            issues.append(
                f"report_event_source_record_missing:{admitted['source_record_sha256']}"
            )
            continue
        for target, source_key in (
            ("report_event_id", "event_id"),
            ("report_month", "report_month"),
            ("channel", "channel"),
            ("instrument_class", "instrument_class"),
            ("raw_code", "raw_code"),
            ("local_code", "normalized_local_code"),
            ("event_date", "event_date"),
            ("source_sha256", "source_sha256"),
        ):
            if admitted[target] != str(source[source_key]):
                issues.append(
                    f"report_event_admission_field_mismatch:{admitted['report_event_id']}:{target}"
                )
        source, expected_status, in_scope = _normalization_contract(
            source, apply_repairs=apply_normalization_repairs
        )
        if (
            _truth(admitted["in_research_scope"]) != in_scope
            or admitted["normalization_status"] != expected_status
            or not _truth(admitted["claim_exact_set_validated"])
        ):
            issues.append(f"report_event_scope_or_status_invalid:{source['event_id']}")
        primary_id = "jpx-report-" + source["event_id"] if expected_status == "normalized" else ""
        if admitted["primary_event_id"] != primary_id:
            issues.append(f"report_event_primary_id_invalid:{source['event_id']}")
        primary = event_index.get(primary_id) if primary_id else None
        if expected_status == "normalized":
            expected_primary_ids.add(primary_id)
            expected_type = (
                "trading_halt"
                if source["channel"] == HALT_CHANNEL
                else "delisting_non_trading_interval"
            )
            expected_start = (
                source["halt_start_jst"]
                if source["channel"] == HALT_CHANNEL
                else pd.Timestamp(source["event_date"], tz="Asia/Tokyo").isoformat()
            )
            expected_end = (
                source["halt_end_jst"]
                if source["channel"] == HALT_CHANNEL
                else ""
            )
            if not primary or any(
                (
                    primary["event_type"] != expected_type,
                    primary["local_code"] != source["normalized_local_code"],
                    primary["issue_name"] != source["issue_name"],
                    primary["event_date"] != source["event_date"],
                    primary["event_start_jst"] != expected_start,
                    primary["interval_end_jst"] != expected_end,
                    primary["source_sha256"] != source["source_sha256"],
                    primary["source_record_id"] != source["source_record_sha256"],
                    primary["source_url"] != source["source_url"],
                )
            ):
                issues.append(f"report_event_primary_normalization_invalid:{source['event_id']}")
        elif primary is not None:
            issues.append(f"report_event_quarantine_bypassed:{source['event_id']}")
        mapping = bool(primary and primary["mapping_status"] in PIT_MAPPING_STATUSES)
        timestamp = bool(primary and primary["known_at_status"] in PIT_SAFE_STATUSES)
        ready = bool(expected_status == "normalized" and mapping and timestamp)
        if (
            _truth(admitted["mapping_complete"]) != mapping
            or _truth(admitted["timestamp_complete"]) != timestamp
            or _truth(admitted["admission_ready"]) != ready
        ):
            issues.append(f"report_event_admission_state_invalid:{source['event_id']}")

    selected_sources = {
        f"jpx-monthly-report-{month}-{channel}"
        for month, channel in set(
            zip(selected_claims["report_month"], selected_claims["channel"], strict=False)
        )
    }
    actual_primary_ids = set(
        events.loc[events["source_id"].isin(selected_sources), "event_id"]
    )
    if actual_primary_ids != expected_primary_ids:
        issues.append("report_event_primary_event_set_invalid")
    if len(superseded) and superseded["source_id"].str.startswith(
        "jpx-monthly-report-"
    ).any():
        issues.append("report_event_present_in_superseded_ledger")

    claim_keys = set(
        zip(selected_claims["report_month"], selected_claims["channel"], strict=False)
    )
    admission_keys = set(
        zip(
            selected_admission["report_month"],
            selected_admission["channel"],
            strict=False,
        )
    )
    coverage_index = coverage.set_index(["month", "channel"], drop=False)
    if claim_keys != admission_keys:
        issues.append("report_event_claim_key_set_invalid")
    for month, channel in claim_keys:
        if (month, channel) not in coverage_index.index:
            issues.append(f"report_event_coverage_claim_missing:{month}:{channel}")
            continue
        covered = coverage_index.loc[(month, channel)]
        if isinstance(covered, pd.DataFrame):
            issues.append(f"report_event_coverage_claim_duplicate:{month}:{channel}")
            continue
        expected_source = f"jpx-monthly-report-{month}-{channel}"
        if not _truth(covered["source_covered"]) or expected_source not in covered[
            "source_ids"
        ].split("|"):
            issues.append(f"report_event_coverage_source_invalid:{month}:{channel}")


def _validate_sealed_packet_copy(
    root: Path, packet: str, issues: list[str]
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    inventory_path = root / "artifact_hashes.json"
    if not manifest_path.is_file() or not inventory_path.is_file():
        issues.append(f"{packet}_packet_missing")
        return {}
    manifest = _json(manifest_path)
    if _sha256(inventory_path) != manifest.get("artifact_hashes_sha256"):
        issues.append(f"{packet}_inventory_digest_mismatch")
    declared = _json(inventory_path)
    for relative_name, digest in declared.items():
        relative = Path(str(relative_name))
        candidate = root / relative
        target = candidate.resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _within(target, root)
            or any(
                (root / Path(*relative.parts[:index])).is_symlink()
                for index in range(1, len(relative.parts) + 1)
            )
            or not target.is_file()
            or _sha256(target) != str(digest)
        ):
            issues.append(f"{packet}_output_hash_mismatch:{relative_name}")
    expected = set(declared) | {"manifest.json", "artifact_hashes.json"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        issues.append(f"{packet}_undeclared_or_missing_files")
    return manifest


def _validate_halt_mapping_recovery(
    producer: Path, events: pd.DataFrame, issues: list[str]
) -> None:
    root = producer / "inputs" / "halt_mapping_recovery"
    recovered_events = events.loc[
        events["mapping_status"].eq("pit_exact_official_monthly_quote_recovery")
    ]
    if not root.exists():
        if len(recovered_events):
            issues.append("halt_mapping_recovery_rows_without_packet")
        return
    packet_root = root / "packets"
    packet_dirs = (
        sorted(path for path in packet_root.iterdir() if path.is_dir())
        if packet_root.is_dir()
        else [root]
    )
    if not packet_dirs:
        issues.append("halt_mapping_recovery_packet_denominator_empty")
        return

    accepted_frames: list[pd.DataFrame] = []
    for packet in packet_dirs:
        producer_packet = packet / "producer"
        validation_packet = packet / "validation"
        recovery_manifest = _validate_sealed_packet_copy(
            producer_packet, "halt_mapping_recovery_producer", issues
        )
        validation_manifest = _validate_sealed_packet_copy(
            validation_packet, "halt_mapping_recovery_validation", issues
        )
        producer_manifest_path = producer_packet / "manifest.json"
        validation_manifest_path = validation_packet / "manifest.json"
        lineage_path = packet / "lineage.json"
        lineage = _json(lineage_path) if lineage_path.is_file() else {}
        denominator = _integer(recovery_manifest.get("denominator_rows", -1))
        accepted_count = _integer(recovery_manifest.get("accepted_rows", -1))
        rejected_count = _integer(recovery_manifest.get("rejected_rows", -1))
        contract_valid = (
            recovery_manifest.get("status") == "complete_candidate"
            and recovery_manifest.get("artifact_type")
            in {
                "japan_g4_halt_mapping_recovery_packet",
                "japan_g4_halt_mapping_recovery_v8_packet",
            }
            and recovery_manifest.get("fail_closed") is True
            and recovery_manifest.get("producer_checks_pass") is True
            and recovery_manifest.get("latest_eligible_official_observation_only")
            is True
            and recovery_manifest.get("successor_ric_allowed") is False
            and recovery_manifest.get("inferred_code_t_allowed", False) is False
            and recovery_manifest.get("window_extension") is False
            and denominator > 0
            and accepted_count > 0
            and rejected_count >= 0
            and accepted_count + rejected_count == denominator
            and validation_manifest.get("status") == "pass"
            and validation_manifest.get("artifact_type")
            in {
                "japan_g4_halt_mapping_recovery_independent_validation",
                "japan_g4_halt_mapping_recovery_v8_independent_validation",
            }
            and validation_manifest.get("component_validation_pass") is True
            and validation_manifest.get("producer_manifest_sha256")
            == _sha256(producer_manifest_path)
            and _integer(validation_manifest.get("denominator_rows", -1))
            == denominator
            and _integer(validation_manifest.get("accepted_rows", -1))
            == accepted_count
            and _integer(validation_manifest.get("rejected_rows", -1))
            == rejected_count
            and _integer(validation_manifest.get("accepted_rows_validated", -1))
            == accepted_count
            and _integer(validation_manifest.get("validation_issue_count", -1)) == 0
            and validation_manifest.get("all_lineage_hashes_verified") is True
            and validation_manifest.get("known_at_recomputed") is True
            and validation_manifest.get("candidate_uniqueness_recomputed") is True
            and lineage.get("producer_manifest_sha256")
            == _sha256(producer_manifest_path)
            and lineage.get("validation_manifest_sha256")
            == _sha256(validation_manifest_path)
            and _integer(lineage.get("accepted_rows", -1)) == accepted_count
        )
        if not contract_valid:
            issues.append("halt_mapping_recovery_lineage_or_contract_invalid")

        accepted = _csv(producer_packet / "accepted_mappings.csv")
        if (
            len(accepted) != accepted_count
            or accepted["event_id"].duplicated().any()
            or not accepted["decision"].eq("accepted").all()
            or not accepted["candidate_count"].eq("1").all()
        ):
            issues.append("halt_mapping_recovery_packet_event_denominator_invalid")
            continue
        accepted_frames.append(accepted)

        evidence = _csv(packet / "evidence_inventory.csv")
        if evidence.empty or _integer(lineage.get("evidence_files", -1)) != len(
            evidence
        ):
            issues.append("halt_mapping_recovery_evidence_inventory_invalid")
        for item in evidence.to_dict("records"):
            relative = Path(item["copied_path"])
            path = (packet / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not _within(path, packet)
                or not path.is_file()
                or _sha256(path) != item["sha256"]
            ):
                issues.append(
                    f"halt_mapping_recovery_evidence_hash_invalid:{item['sha256']}"
                )
        inputs = _csv(producer_packet / "input_inventory.csv")
        copies = lineage.get("input_copies", {})
        if not isinstance(copies, dict) or set(copies) != set(inputs["input_id"]):
            issues.append("halt_mapping_recovery_input_copy_denominator_invalid")
        else:
            for item in inputs.to_dict("records"):
                relative = Path(str(copies[item["input_id"]]))
                path = (packet / relative).resolve()
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not _within(path, packet)
                    or not path.is_file()
                    or _sha256(path) != item["expected_sha256"]
                ):
                    issues.append(
                        f"halt_mapping_recovery_input_copy_invalid:{item['input_id']}"
                    )

    if not accepted_frames:
        issues.append("halt_mapping_recovery_event_denominator_invalid")
        return
    accepted = pd.concat(accepted_frames, ignore_index=True)
    if (
        accepted["event_id"].duplicated().any()
        or set(accepted["event_id"]) != set(recovered_events["event_id"])
    ):
        issues.append("halt_mapping_recovery_event_denominator_invalid")
        return
    if packet_root.is_dir():
        aggregate_lineage = _json(root / "lineage.json")
        if not (
            _integer(aggregate_lineage.get("packet_count", -1)) == len(packet_dirs)
            and _integer(aggregate_lineage.get("accepted_rows", -1)) == len(accepted)
            and aggregate_lineage.get("event_ids_unique") is True
        ):
            issues.append("halt_mapping_recovery_aggregate_lineage_invalid")
    event_index = recovered_events.set_index("event_id", drop=False).to_dict("index")
    for item in accepted.to_dict("records"):
        event = event_index.get(item["event_id"])
        candidate = item["candidate_ric"]
        source_known_at = pd.Timestamp(item["source_known_at_utc"])
        event_day = pd.Timestamp(item["event_date"], tz="UTC")
        if (
            event is None
            or event["event_date"] != item["event_date"]
            or event["local_code"] != item["local_code"]
            or event["refinitiv_ric"] != candidate
            or event["stable_listing_id"] != item["identity_stable_instrument_id"]
            or event["mapping_known_at"] != item["source_known_at_utc"]
            or event["mapping_route"]
            != "official_monthly_quote_pre_event_plus_unique_g1_identity"
            or event["mapping_source_sha256"] != item["source_pdf_sha256"]
            or event["mapping_source_record_id"] != item["source_record_sha256"]
            or not re.fullmatch(
                rf"{re.escape(item['local_code'])}\.T(?:\^.*)?",
                candidate,
                flags=re.IGNORECASE,
            )
            or source_known_at.tzinfo is None
            or source_known_at > event_day
        ):
            issues.append(f"halt_mapping_recovery_event_invalid:{item['event_id']}")



def _validate_g1_lineage(
    lineage: dict[str, Any],
    copied_map_path: Path,
    copied_security: pd.DataFrame,
    issues: list[str],
) -> None:
    if _sha256(copied_map_path) != lineage.get("reference_symbol_map_sha256"):
        issues.append("g1_reference_map_copy_hash_mismatch")
    for label in ("g1_producer_manifest", "g1_validation_manifest"):
        path = Path(str(lineage.get(label, "")))
        if not path.is_file() or _sha256(path) != lineage.get(f"{label}_sha256"):
            issues.append(f"g1_lineage_hash_mismatch:{label}")
    expected_records: set[str] = set()
    event_codes = set(copied_security["local_code"]) if not copied_security.empty else set()
    admitted_hashes: set[str] = set()
    for source in lineage.get("security_master_sources", []):
        manifest_path = Path(source["manifest_path"])
        data_path = Path(source["security_master_path"])
        if not manifest_path.is_file() or _sha256(manifest_path) != source["manifest_sha256"]:
            issues.append(f"g1_security_manifest_hash_mismatch:{source['source_key']}")
            continue
        if not data_path.is_file() or _sha256(data_path) != source["security_master_sha256"]:
            issues.append(f"g1_security_data_hash_mismatch:{source['source_key']}")
            continue
        admitted_hashes.add(source["security_master_sha256"])
        raw = pd.read_csv(data_path, dtype=str).fillna("")
        for column in ("mapping_local_code_override", "direct_code", "ticker_symbol", "ric", "refinitiv_ric", "exchange_code", "first_trade_date", "retire_date"):
            if column not in raw:
                raw[column] = ""
        local_code = raw["mapping_local_code_override"].mask(raw["mapping_local_code_override"].eq(""), raw["direct_code"]).mask(lambda values: values.eq(""), raw["ticker_symbol"])
        ric = raw["ric"].mask(raw["ric"].eq(""), raw["refinitiv_ric"])
        for index in raw.index[local_code.isin(event_codes)]:
            value = "|".join((source["security_master_sha256"], local_code[index], ric[index], raw.at[index, "exchange_code"], raw.at[index, "first_trade_date"], raw.at[index, "retire_date"]))
            expected_records.add(hashlib.sha256(value.encode("utf-8")).hexdigest()[:24])
    observed_records = set(copied_security["source_record_id"]) if not copied_security.empty else set()
    if expected_records != observed_records:
        issues.append(f"g1_security_candidate_rebuild_mismatch:{len(expected_records)}/{len(observed_records)}")
    if not copied_security.empty and not set(copied_security["admitted_source_sha256"]).issubset(admitted_hashes):
        issues.append("g1_security_candidate_unadmitted_hash")


def _validate_completion_outputs(
    coverage: pd.DataFrame,
    events: pd.DataFrame,
    gaps: pd.DataFrame,
    completion: pd.DataFrame,
    issues: list[str],
    *,
    report_admission: pd.DataFrame | None = None,
) -> None:
    if tuple(gaps.columns) != GAP_COLUMNS or tuple(completion.columns) != COMPLETION_COLUMNS:
        issues.append("completion_schema_invalid")
        return
    report = (
        report_admission.astype(str).fillna("")
        if report_admission is not None and len(report_admission)
        else pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
    )
    expected = {
        "source_denominator_uncovered": int((~coverage["source_covered"].map(_truth)).sum()),
        "historical_pit_timestamp_unresolved": int((~events["known_at_status"].isin(PIT_SAFE_STATUSES)).sum()),
        "ric_mapping_not_historical_pit": int(events["mapping_status"].eq("deterministic_reference_post_event").sum()),
        "ric_mapping_unresolved": int(events["refinitiv_ric"].eq("").sum()),
        "report_event_normalization_unresolved": int(
            (
                report["in_research_scope"].map(_truth)
                & ~report["normalization_status"].eq("normalized")
            ).sum()
        ),
        "report_zero_equity_publication_timestamp_unresolved": int(
            coverage.get(
                "blockers", pd.Series("", index=coverage.index, dtype=str)
            ).str.contains(
                "report_zero_equity_publication_timestamp_missing", regex=False
            ).sum()
        ),
    }
    observed = gaps.groupby("gap_type").size().to_dict() if len(gaps) else {}
    if {key: value for key, value in expected.items() if value} != observed:
        issues.append(f"completion_gap_counts_invalid:{expected}/{observed}")
    if gaps["gap_id"].duplicated().any() or not gaps["irreducible_with_local_sources"].map(_truth).all():
        issues.append("completion_gap_identity_or_irreducibility_invalid")
    aggregate = completion.set_index("gap_type")["gap_rows"].map(_integer).to_dict() if len(completion) else {}
    if aggregate != observed:
        issues.append("completion_aggregate_counts_invalid")


def _validate_inventory(inventory: pd.DataFrame, issues: list[str]) -> None:
    if inventory.empty:
        return
    if inventory["inventory_id"].duplicated().any():
        issues.append("inventory_ids_not_unique")
    for item in inventory.to_dict("records"):
        path = Path(item["absolute_path"])
        if not path.is_file() or path.stat().st_size != _integer(item["bytes"]) or _sha256(path) != item["sha256"]:
            issues.append(f"inventory_source_changed:{item['inventory_id']}")


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
