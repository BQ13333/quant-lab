from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from quant_lab.data.jpx_market_rules_public import parse_halts
from quant_lab.data.jpx_public_listing_events import parse_delisting_html
from quant_lab.data.japan_g4_halt_report_admission import (
    REPORT_ADMISSION_COLUMNS,
    copy_validated_halt_report_packet,
    covered_report_claims,
    normalize_halt_report_events,
    update_halt_report_admission,
)
from quant_lab.data.japan_g4_halt_normalization_sidecar_admission import (
    copy_validated_halt_normalization_sidecar,
)
from quant_lab.data.japan_g4_halt_timestamp_sidecar_admission import (
    copy_validated_halt_timestamp_sidecar,
)
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair import (
    REPAIR_COLUMNS as LIFECYCLE_TAXONOMY_REPAIR_COLUMNS,
    verify_japan_g4_lifecycle_taxonomy_repair,
)
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair_validator import (
    verify_japan_g4_lifecycle_taxonomy_repair_validation,
)


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
G4_START = date(2006, 7, 1)
G4_END = date(2026, 6, 30)
HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
CHANNELS = (HALT_CHANNEL, DELISTING_CHANNEL)
SOURCE_KINDS = ("jpx_halt_html", "jpx_delisting_html", "inventory_only")
MAX_RESPONSE_BYTES = 64 * 1024 * 1024

SOURCE_CATALOG_COLUMNS = (
    "source_id",
    "source_kind",
    "channel",
    "official_source",
    "source_url",
    "local_path",
    "expected_sha256",
    "observed_at_utc",
    "coverage_start",
    "coverage_end",
    "denominator_complete",
    "no_event_capable",
    "pit_timestamp_status",
)
REQUEST_COLUMNS = (
    "source_id",
    "status",
    "attempts",
    "observed_at_utc",
    "completed_at_utc",
    "raw_path",
    "raw_sha256",
    "source_rows",
    "error_type",
    "error",
)
EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "exchange",
    "local_code",
    "issue_name",
    "event_date",
    "event_start_jst",
    "suspension_end_jst",
    "resumption_jst",
    "interval_end_jst",
    "known_at_utc",
    "known_at_status",
    "source_observed_at_utc",
    "known_at_evidence_sha256",
    "reason",
    "source_id",
    "source_url",
    "source_sha256",
    "source_record_id",
    "stable_listing_id",
    "refinitiv_ric",
    "mapping_known_at",
    "mapping_status",
    "mapping_route",
    "mapping_source_sha256",
    "mapping_source_record_id",
)
COVERAGE_COLUMNS = (
    "plan_id",
    "month",
    "window_start",
    "window_end",
    "channel",
    "source_covered",
    "event_rows",
    "no_event_evidence",
    "pit_timestamp_complete",
    "listing_ric_mapping_complete",
    "coverage_status",
    "source_ids",
    "blockers",
    "next_action",
)
NO_EVENT_COLUMNS = (
    "evidence_id",
    "month",
    "channel",
    "source_id",
    "source_url",
    "source_sha256",
    "observed_at_utc",
    "coverage_start",
    "coverage_end",
    "assertion",
)
INVENTORY_COLUMNS = (
    "inventory_id",
    "archive_root",
    "relative_path",
    "absolute_path",
    "bytes",
    "sha256",
    "suffix",
)
MAPPING_AUDIT_COLUMNS = (
    "event_id",
    "event_date",
    "local_code",
    "refinitiv_ric",
    "mapping_status",
    "mapping_route",
    "mapping_known_at",
    "mapping_source_sha256",
    "mapping_source_record_id",
)
PIT_AUDIT_COLUMNS = (
    "event_id",
    "event_date",
    "local_code",
    "known_at_utc",
    "known_at_status",
    "source_observed_at_utc",
    "known_at_evidence_sha256",
    "historical_pit_usable",
)
GAP_COLUMNS = (
    "gap_id",
    "gap_type",
    "scope",
    "month",
    "event_id",
    "local_code",
    "current_evidence",
    "required_evidence",
    "irreducible_with_local_sources",
    "next_action",
)
COMPLETION_COLUMNS = (
    "plan_id",
    "gap_type",
    "gap_rows",
    "affected_months",
    "affected_events",
    "irreducible_rows",
    "status",
    "required_evidence",
    "next_action",
)
REPORT_EVENT_LINEAGE_COLUMNS = (
    "source_packet",
    "source_record_sha256",
    "report_month",
    "channel",
    "source_event_id",
    "disposition",
    "selected_packet",
    "selected_source_record_sha256",
    "selected_report_event_id",
    "primary_event_id",
)
TSE_WAYBACK_VALIDATOR_VERSION = "2.0-institutional-independent-pdfplumber"
PIT_SAFE_STATUSES = frozenset(
    {
        "conservative_halt_start_market_state",
        "conservative_g1_monthly_absence_observation",
        "exact_publication_timestamp",
        "official_jpx_publication_date_recovery",
    }
)
PIT_MAPPING_STATUSES = frozenset(
    {
        "pit_exact",
        "pit_exact_g1_monthly",
        "pit_exact_g1_identity_retire_bridge",
        "pit_exact_official_monthly_quote_recovery",
    }
)

Fetcher = Callable[[str, float], bytes]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class JapanG4HaltArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    events_path: Path
    coverage_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _G1Evidence:
    symbol_map: pd.DataFrame
    security_candidates: pd.DataFrame
    symbol_map_sha256: str
    lineage: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedTseWaybackLifecycle:
    events: pd.DataFrame
    claims: pd.DataFrame
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedLifecycleTaxonomyRepair:
    repairs: pd.DataFrame
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedOfficialTimestampExtension:
    accepted: pd.DataFrame
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]


def build_halt_acquisition_plan(
    start_date: str | date = G4_START,
    end_date: str | date = G4_END,
) -> pd.DataFrame:
    """Build the exact month-channel denominator before any source is inspected."""
    start = _date(start_date, "start_date")
    end = _date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    rows: list[dict[str, Any]] = []
    for period in pd.period_range(start, end, freq="M"):
        month_start = max(start, period.start_time.date())
        month_end = min(end, period.end_time.date())
        for channel in CHANNELS:
            rows.append(
                {
                    "plan_id": f"{period}-{channel}",
                    "month": str(period),
                    "window_start": month_start.isoformat(),
                    "window_end": month_end.isoformat(),
                    "channel": channel,
                    "source_covered": False,
                    "event_rows": 0,
                    "no_event_evidence": False,
                    "pit_timestamp_complete": False,
                    "listing_ric_mapping_complete": False,
                    "coverage_status": "uncovered",
                    "source_ids": "",
                    "blockers": "source_denominator_uncovered",
                    "next_action": _next_action(channel, "uncovered"),
                }
            )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def discover_local_jpx_source_catalog(
    *,
    halt_rules_dirs: Iterable[str | Path] = (),
    listing_events_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Describe already acquired official pages without granting inferred coverage."""
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for directory_value in halt_rules_dirs:
        directory = Path(directory_value).resolve()
        raw_manifest_path = directory / "raw_manifest.csv"
        if not raw_manifest_path.is_file():
            raise FileNotFoundError(f"missing raw_manifest.csv: {directory}")
        raw_manifest = pd.read_csv(raw_manifest_path, dtype=str).fillna("")
        matches = raw_manifest.loc[raw_manifest["component"].eq("halts")]
        if len(matches) != 1:
            raise ValueError(f"expected one halts row in {raw_manifest_path}")
        item = matches.iloc[0]
        path = Path(item["raw_path"]).resolve()
        digest = _sha256(path)
        if item["sha256"] != digest:
            raise ValueError(f"halt raw hash mismatch: {path}")
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        rows.append(
            _source_row(
                source_id=f"jpx-halt-{digest[:16]}",
                source_kind="jpx_halt_html",
                channel=HALT_CHANNEL,
                source_url=item["url"],
                local_path=path,
                expected_sha256=digest,
                observed_at_utc=item["observed_at_utc"],
                pit_timestamp_status="snapshot_observation_only",
            )
        )

    if listing_events_dir is not None:
        directory = Path(listing_events_dir).resolve()
        plan = pd.read_csv(directory / "request_plan.csv", dtype=str).fillna("")
        status = pd.read_csv(directory / "page_status.csv", dtype=str).fillna("")
        merged = plan.merge(status, on="page_id", how="left")
        merged = merged.loc[
            merged["event_type"].eq("delisting") & merged["status"].eq("completed")
        ]
        for item in merged.to_dict("records"):
            year = str(item["archive_year"])
            path = Path(item["raw_path"]).resolve()
            digest = _sha256(path)
            if digest != item["sha256"]:
                raise ValueError(f"delisting raw hash mismatch: {path}")
            rows.append(
                _source_row(
                    source_id=f"jpx-delisting-{year}-{digest[:12]}",
                    source_kind="jpx_delisting_html",
                    channel=DELISTING_CHANNEL,
                    source_url=item["url"],
                    local_path=path,
                    expected_sha256=digest,
                    observed_at_utc=item["observed_at_utc"],
                    coverage_start=f"{year}-01-01" if year.isdigit() else "",
                    coverage_end=f"{year}-12-31" if year.isdigit() else "",
                    denominator_complete=year.isdigit(),
                    no_event_capable=year.isdigit(),
                    pit_timestamp_status="event_date_only",
                )
            )
    result = pd.DataFrame(rows, columns=SOURCE_CATALOG_COLUMNS)
    if not result.empty and result["source_id"].duplicated().any():
        raise ValueError("discovered source_id values are not unique")
    return result


def inventory_official_archives(archive_roots: Iterable[str | Path]) -> pd.DataFrame:
    """Hash every file under caller-declared official archive roots."""
    rows: list[dict[str, Any]] = []
    for root_value in archive_roots:
        root = Path(root_value).resolve()
        if not root.exists():
            raise FileNotFoundError(f"archive root does not exist: {root}")
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            relative = path.name if root.is_file() else path.relative_to(root).as_posix()
            digest = _sha256(path)
            rows.append(
                {
                    "inventory_id": hashlib.sha256(
                        f"{root}|{relative}|{digest}".encode("utf-8")
                    ).hexdigest()[:24],
                    "archive_root": str(root),
                    "relative_path": relative,
                    "absolute_path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "suffix": path.suffix.lower(),
                }
            )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def produce_japan_g4_halt_history(
    *,
    run_id: str,
    source_catalog: pd.DataFrame | str | Path,
    symbol_map_path: str | Path | None,
    g1_producer_manifest_path: str | Path | None = None,
    g1_validation_manifest_path: str | Path | None = None,
    timestamp_recovery_dir: str | Path | None = None,
    official_timestamp_recovery_dir: str | Path | None = None,
    official_timestamp_recovery_validation_dir: str | Path | None = None,
    official_timestamp_v14_recovery_dir: str | Path | None = None,
    official_timestamp_v14_recovery_validation_dir: str | Path | None = None,
    halt_report_events_dir: str | Path | None = None,
    halt_report_events_validation_dir: str | Path | None = None,
    halt_normalization_repair_dir: str | Path | None = None,
    halt_normalization_repair_validation_dir: str | Path | None = None,
    lifecycle_taxonomy_repair_dir: str | Path | None = None,
    lifecycle_taxonomy_repair_validation_dir: str | Path | None = None,
    tse_wayback_lifecycle_dir: str | Path | None = None,
    tse_wayback_lifecycle_validation_dir: str | Path | None = None,
    tse_wayback_lifecycle_pairs: Iterable[
        tuple[str | Path, str | Path]
    ] | None = None,
    halt_mapping_recovery_dir: str | Path | None = None,
    halt_mapping_recovery_validation_dir: str | Path | None = None,
    halt_mapping_recovery_pairs: Iterable[
        tuple[str | Path, str | Path]
    ] | None = None,
    archive_roots: Iterable[str | Path] = (),
    artifact_root: str | Path = "artifacts/japan_g4_halt_history",
    start_date: str | date = G4_START,
    end_date: str | date = G4_END,
    max_sources: int = 1000,
    timeout_seconds: float = 30.0,
    retries: int = 2,
    resume: bool = True,
    fetcher: Fetcher | None = None,
    clock: Clock | None = None,
) -> JapanG4HaltArtifacts:
    """Acquire, checkpoint, normalize, map, and fail-close the Japan G4 halt lane."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    if max_sources <= 0 or timeout_seconds <= 0 or retries < 0:
        raise ValueError("max_sources/timeout_seconds must be positive and retries non-negative")
    if (halt_report_events_dir is None) != (halt_report_events_validation_dir is None):
        raise ValueError("both halt report producer and validation directories are required")
    if (official_timestamp_recovery_dir is None) != (
        official_timestamp_recovery_validation_dir is None
    ):
        raise ValueError(
            "both official timestamp recovery and validation directories are required"
        )
    if (official_timestamp_v14_recovery_dir is None) != (
        official_timestamp_v14_recovery_validation_dir is None
    ):
        raise ValueError(
            "both V14 official timestamp recovery and validation directories are required"
        )
    if (halt_normalization_repair_dir is None) != (
        halt_normalization_repair_validation_dir is None
    ):
        raise ValueError(
            "both halt normalization repair and validation directories are required"
        )
    if (lifecycle_taxonomy_repair_dir is None) != (
        lifecycle_taxonomy_repair_validation_dir is None
    ):
        raise ValueError(
            "both lifecycle taxonomy repair and validation directories are required"
        )
    lifecycle_pairs = _normalize_tse_wayback_lifecycle_pairs(
        tse_wayback_lifecycle_dir,
        tse_wayback_lifecycle_validation_dir,
        tse_wayback_lifecycle_pairs,
    )
    mapping_recovery_pairs = _normalize_halt_mapping_recovery_pairs(
        halt_mapping_recovery_dir,
        halt_mapping_recovery_validation_dir,
        halt_mapping_recovery_pairs,
    )
    start = _date(start_date, "start_date")
    end = _date(end_date, "end_date")
    catalog = _catalog_frame(source_catalog)
    _validate_catalog(catalog)
    now = clock or (lambda: datetime.now(UTC))
    fetch = fetcher or _fetch

    root = (Path(artifact_root) / run_id).resolve()
    raw_dir = root / "raw"
    input_dir = root / "inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(root)
    try:
        catalog_path = root / "source_catalog.csv"
        request_path = root / "request_manifest.csv"
        inventory_path = root / "source_inventory.csv"
        mapping_path = input_dir / "symbol_map.csv"
        g1_map_path = input_dir / "g1_reference_symbol_map.csv"
        g1_security_path = input_dir / "g1_security_master_candidates.csv"
        g1_lineage_path = input_dir / "g1_lineage.json"
        plan_path = root / "acquisition_plan.csv"
        events_path = root / "halt_suspension_events.parquet"
        out_of_scope_path = root / "out_of_scope_events.parquet"
        superseded_path = root / "superseded_source_events.parquet"
        no_event_path = root / "no_event_evidence.csv"
        mapping_audit_path = root / "mapping_audit.csv"
        pit_audit_path = root / "pit_timestamp_audit.csv"
        gaps_path = root / "unresolved_gaps.csv"
        completion_path = root / "completion_plan.csv"
        manifest_path = root / "manifest.json"
        summary_path = root / "summary.md"
        report_admission_path = root / "report_event_admission.csv"
        report_lineage_path = root / "report_event_lineage.csv"
        timestamp_recovery_paths: list[Path] = []
        official_timestamp_recovery_paths: list[Path] = []
        official_timestamp_v14_recovery_paths: list[Path] = []
        halt_report_paths: list[Path] = []
        halt_normalization_repair_paths: list[Path] = []
        lifecycle_taxonomy_repair_paths: list[Path] = []
        tse_wayback_paths: list[Path] = []
        halt_mapping_recovery_paths: list[Path] = []
        report_admission = pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
        report_lineage = pd.DataFrame(columns=REPORT_EVENT_LINEAGE_COLUMNS)
        superseded = pd.DataFrame(columns=EVENT_COLUMNS)

        if catalog_path.is_file() and resume:
            existing = pd.read_csv(catalog_path, dtype=str).fillna("")
            if not existing.equals(catalog.astype(str).fillna("")):
                raise ValueError("resume source catalog differs from the immutable run plan")
        _atomic_csv(catalog_path, catalog)
        inventory = inventory_official_archives(archive_roots)
        _atomic_csv(inventory_path, inventory)
        symbol_map = _copy_symbol_map(symbol_map_path, mapping_path)

        requests = _load_requests(request_path) if resume else pd.DataFrame(columns=REQUEST_COLUMNS)
        requests = _invalidate_stale_requests(requests)
        completed = set(requests.loc[requests["status"].eq("completed"), "source_id"])
        pending = catalog.loc[~catalog["source_id"].isin(completed)].head(max_sources)
        rows = requests.to_dict("records")
        for item in pending.to_dict("records"):
            source_id = item["source_id"]
            attempts = 0
            payload: bytes | None = None
            error: Exception | None = None
            observed = item["observed_at_utc"]
            for attempt in range(1, retries + 2):
                attempts = attempt
                try:
                    payload = _source_payload(item, fetch, timeout_seconds)
                    if len(payload) > MAX_RESPONSE_BYTES:
                        raise ValueError("source exceeds response safety limit")
                    error = None
                    break
                except Exception as exc:
                    error = exc
            previous = next((row for row in rows if row["source_id"] == source_id), None)
            total_attempts = attempts + _int(previous.get("attempts", 0) if previous else 0)
            raw_path = raw_dir / f"{source_id}{_raw_suffix(item)}"
            if payload is not None and error is None:
                _atomic_bytes(raw_path, payload)
                digest = _sha256(raw_path)
                expected = item["expected_sha256"]
                if expected and digest != expected:
                    error = ValueError(f"expected source hash {expected}, got {digest}")
                else:
                    if not observed:
                        observed = _utc_iso(now())
                    source_rows = len(_parse_source(raw_path, item, observed))
                    row = {
                        "source_id": source_id,
                        "status": "completed",
                        "attempts": total_attempts,
                        "observed_at_utc": observed,
                        "completed_at_utc": _utc_iso(now()),
                        "raw_path": str(raw_path),
                        "raw_sha256": digest,
                        "source_rows": source_rows,
                        "error_type": "",
                        "error": "",
                    }
            if payload is None or error is not None:
                row = {
                    "source_id": source_id,
                    "status": "failed",
                    "attempts": total_attempts,
                    "observed_at_utc": observed,
                    "completed_at_utc": _utc_iso(now()),
                    "raw_path": str(raw_path) if raw_path.is_file() else "",
                    "raw_sha256": _sha256(raw_path) if raw_path.is_file() else "",
                    "source_rows": 0,
                    "error_type": type(error).__name__ if error else "",
                    "error": str(error) if error else "source acquisition failed",
                }
            rows = [old for old in rows if old["source_id"] != source_id] + [row]
            _atomic_csv(request_path, pd.DataFrame(rows, columns=REQUEST_COLUMNS))

        requests = pd.DataFrame(rows, columns=REQUEST_COLUMNS).fillna("")
        events = _events_from_sources(catalog, requests)
        in_scope = events.loc[
            events["event_date"].between(start.isoformat(), end.isoformat())
        ].copy()
        out_of_scope = events.loc[~events.index.isin(in_scope.index)].copy()
        report_packets: list[tuple[str, pd.DataFrame]] = []
        if halt_report_events_dir is not None:
            packet = copy_validated_halt_report_packet(
                halt_report_events_dir,
                halt_report_events_validation_dir,
                input_dir / "halt_report_events",
                atomic_write=_atomic_bytes,
            )
            halt_report_paths = list(packet.copied_paths)
            report_packets.append(("halt_report_events", packet.events))
        lifecycle_packet: _ValidatedTseWaybackLifecycle | None = None
        if lifecycle_pairs:
            lifecycle_packet = _copy_validated_tse_wayback_lifecycles(
                lifecycle_pairs,
                input_dir / "tse_wayback_lifecycle",
            )
            tse_wayback_paths = list(lifecycle_packet.copied_paths)
            report_packets.append(("tse_wayback_lifecycle", lifecycle_packet.events))
        if report_packets:
            merged_report_events, report_lineage = _merge_report_event_packets(
                report_packets
            )
            taxonomy_sidecar: _ValidatedLifecycleTaxonomyRepair | None = None
            if lifecycle_taxonomy_repair_dir is not None:
                taxonomy_sidecar = _copy_validated_lifecycle_taxonomy_repair(
                    lifecycle_taxonomy_repair_dir,
                    lifecycle_taxonomy_repair_validation_dir,
                    input_dir / "lifecycle_taxonomy_repair",
                )
                lifecycle_taxonomy_repair_paths = list(
                    taxonomy_sidecar.copied_paths
                )
                merged_report_events = _apply_lifecycle_taxonomy_repairs(
                    merged_report_events, taxonomy_sidecar.repairs
                )
            apply_normalization_repairs = False
            if halt_normalization_repair_dir is not None:
                normalization_sidecar = copy_validated_halt_normalization_sidecar(
                    halt_normalization_repair_dir,
                    halt_normalization_repair_validation_dir,
                    input_dir / "halt_normalization_repair",
                    atomic_write=_atomic_bytes,
                )
                halt_normalization_repair_paths = list(
                    normalization_sidecar.copied_paths
                )
                apply_normalization_repairs = True
            normalized_report_events, report_admission = normalize_halt_report_events(
                merged_report_events,
                event_columns=EVENT_COLUMNS,
                apply_bounded_repairs=apply_normalization_repairs,
            )
            normalized_report_events, report_admission = (
                _quarantine_invalid_report_intervals(
                    normalized_report_events, report_admission
                )
            )
            report_lineage = _finish_report_event_lineage(
                report_lineage, report_admission
            )
            covered = covered_report_claims(report_admission)
            existing_keys = pd.Series(
                list(
                    zip(
                        in_scope["event_date"].str[:7],
                        in_scope["event_type"].map(_event_channel),
                        strict=False,
                    )
                ),
                index=in_scope.index,
            )
            superseded = in_scope.loc[existing_keys.isin(covered)].copy()
            in_scope = in_scope.loc[~existing_keys.isin(covered)].copy()
            report_inside = normalized_report_events.loc[
                normalized_report_events["event_date"].between(
                    start.isoformat(), end.isoformat()
                )
            ].copy()
            report_outside = normalized_report_events.loc[
                ~normalized_report_events.index.isin(report_inside.index)
            ].copy()
            in_scope = pd.concat([in_scope, report_inside], ignore_index=True).reindex(
                columns=EVENT_COLUMNS
            )
            out_of_scope = pd.concat(
                [out_of_scope, report_outside], ignore_index=True
            ).reindex(columns=EVENT_COLUMNS)
        g1 = _load_g1_evidence(
            producer_manifest_path=g1_producer_manifest_path,
            validation_manifest_path=g1_validation_manifest_path,
            event_codes=set(in_scope["local_code"]) | set(out_of_scope["local_code"]),
        )
        if g1 is not None:
            symbol_map = g1.symbol_map
            _atomic_csv(g1_map_path, symbol_map)
            _atomic_csv(g1_security_path, g1.security_candidates)
            _atomic_json(g1_lineage_path, g1.lineage)
        in_scope = _attach_pit_mapping(
            in_scope,
            symbol_map,
            security_candidates=g1.security_candidates if g1 else None,
            symbol_map_sha256=g1.symbol_map_sha256 if g1 else _sha256(mapping_path),
        )
        out_of_scope = _attach_pit_mapping(
            out_of_scope,
            symbol_map,
            security_candidates=g1.security_candidates if g1 else None,
            symbol_map_sha256=g1.symbol_map_sha256 if g1 else _sha256(mapping_path),
        )
        if mapping_recovery_pairs:
            mapping_recovery, halt_mapping_recovery_paths = (
                _copy_validated_halt_mapping_recoveries(
                    mapping_recovery_pairs,
                    input_dir / "halt_mapping_recovery",
                )
            )
            in_scope = _apply_halt_mapping_recovery(in_scope, mapping_recovery)
        in_scope = _resolve_pit_timestamps(
            in_scope, symbol_map, g1.symbol_map_sha256 if g1 else _sha256(mapping_path)
        )
        out_of_scope = _resolve_pit_timestamps(
            out_of_scope, symbol_map, g1.symbol_map_sha256 if g1 else _sha256(mapping_path)
        )
        if timestamp_recovery_dir is not None:
            recovery, timestamp_recovery_paths = _copy_validated_timestamp_recovery(
                Path(timestamp_recovery_dir), input_dir / "timestamp_recovery"
            )
            in_scope = _apply_timestamp_recovery(in_scope, recovery)
            out_of_scope = _apply_timestamp_recovery(out_of_scope, recovery)
        if official_timestamp_recovery_dir is not None:
            timestamp_sidecar = copy_validated_halt_timestamp_sidecar(
                official_timestamp_recovery_dir,
                official_timestamp_recovery_validation_dir,
                input_dir / "official_timestamp_recovery",
            )
            official_timestamp_recovery_paths = list(timestamp_sidecar.copied_paths)
            in_scope = _apply_official_timestamp_sidecar(
                in_scope, timestamp_sidecar.accepted
            )
            out_of_scope = _apply_official_timestamp_sidecar(
                out_of_scope, timestamp_sidecar.accepted, require_all=False
            )
        if official_timestamp_v14_recovery_dir is not None:
            timestamp_v14_sidecar = _copy_validated_official_timestamp_v14_extension(
                official_timestamp_v14_recovery_dir,
                official_timestamp_v14_recovery_validation_dir,
                input_dir / "official_timestamp_v14_recovery",
            )
            official_timestamp_v14_recovery_paths = list(
                timestamp_v14_sidecar.copied_paths
            )
            in_scope = _apply_official_timestamp_sidecar(
                in_scope, timestamp_v14_sidecar.accepted
            )
            out_of_scope = _apply_official_timestamp_sidecar(
                out_of_scope, timestamp_v14_sidecar.accepted, require_all=False
            )
        if report_packets:
            report_admission = update_halt_report_admission(
                report_admission,
                in_scope,
                pit_mapping_statuses=PIT_MAPPING_STATUSES,
                pit_timestamp_statuses=PIT_SAFE_STATUSES,
            )
            _atomic_csv(report_admission_path, report_admission)
            report_lineage = _finish_report_event_lineage(
                report_lineage, report_admission
            )
            _atomic_csv(report_lineage_path, report_lineage)
        _validate_events(in_scope)
        _atomic_parquet(events_path, in_scope)
        _atomic_parquet(out_of_scope_path, out_of_scope)
        _atomic_parquet(superseded_path, superseded)

        coverage, no_event = _build_coverage(
            start=start,
            end=end,
            catalog=catalog,
            requests=requests,
            events=in_scope,
            report_admission=report_admission,
        )
        _atomic_csv(plan_path, coverage)
        _atomic_csv(no_event_path, no_event)
        mapping_audit = in_scope.reindex(columns=MAPPING_AUDIT_COLUMNS)
        pit_audit = _pit_audit(in_scope)
        gaps, completion = _build_completion_outputs(
            coverage, in_scope, report_admission=report_admission
        )
        _atomic_csv(mapping_audit_path, mapping_audit)
        _atomic_csv(pit_audit_path, pit_audit)
        _atomic_csv(gaps_path, gaps)
        _atomic_csv(completion_path, completion)
        complete = bool(len(coverage) and coverage["coverage_status"].eq("complete").all())
        blockers = sorted(
            {
                blocker
                for value in coverage["blockers"]
                for blocker in str(value).split("|")
                if blocker
            }
        )
        output_paths = [
            catalog_path,
            request_path,
            inventory_path,
            mapping_path,
            plan_path,
            events_path,
            out_of_scope_path,
            superseded_path,
            no_event_path,
            mapping_audit_path,
            pit_audit_path,
            gaps_path,
            completion_path,
            summary_path,
        ]
        if g1 is not None:
            output_paths.extend((g1_map_path, g1_security_path, g1_lineage_path))
        output_paths.extend(timestamp_recovery_paths)
        output_paths.extend(official_timestamp_recovery_paths)
        output_paths.extend(official_timestamp_v14_recovery_paths)
        if report_packets:
            output_paths.extend(halt_report_paths)
            output_paths.extend(tse_wayback_paths)
            output_paths.extend(halt_normalization_repair_paths)
            output_paths.extend(lifecycle_taxonomy_repair_paths)
            output_paths.extend((report_admission_path, report_lineage_path))
        output_paths.extend(halt_mapping_recovery_paths)
        _atomic_bytes(summary_path, _summary(coverage, in_scope, out_of_scope, inventory).encode("utf-8"))
        manifest = {
            "schema_version": "2.0-g1-admitted",
            "run_id": run_id,
            "artifact_type": "japan_g4_halt_suspension_history_producer",
            "status": "complete_candidate" if complete else "partial_candidate",
            "component": "halt_suspension_events",
            "official_jpx_source": bool(
                len(catalog) and _bool_series(catalog["official_source"]).all()
            ),
            "historical_archive_complete": complete,
            "source_denominator_exact": complete,
            "scorecard_eligible": False,
            "independent_validation_required": True,
            "coverage": {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "months": int(coverage["month"].nunique()),
                "month_channels": len(coverage),
                "complete_month_channels": int(coverage["coverage_status"].eq("complete").sum()),
                "complete": complete,
            },
            "counts": {
                "catalog_sources": len(catalog),
                "completed_sources": int(requests["status"].eq("completed").sum()),
                "source_rows": int(pd.to_numeric(requests["source_rows"], errors="coerce").fillna(0).sum()),
                "event_rows": len(in_scope),
                "out_of_scope_event_rows": len(out_of_scope),
                "superseded_source_event_rows": len(superseded),
                "halt_event_rows": int(in_scope["event_type"].eq("trading_halt").sum()),
                "delisting_interval_rows": int(in_scope["event_type"].eq("delisting_non_trading_interval").sum()),
                "no_event_evidence_rows": len(no_event),
                "inventory_files": len(inventory),
                "inventory_unique_hashes": int(inventory["sha256"].nunique()) if len(inventory) else 0,
                "ric_resolved_rows": int(in_scope["refinitiv_ric"].ne("").sum()),
                "ric_unresolved_rows": int(in_scope["refinitiv_ric"].eq("").sum()),
                "pit_mapped_rows": int(in_scope["mapping_status"].isin(PIT_MAPPING_STATUSES).sum()),
                "non_pit_reference_mapping_rows": int(in_scope["mapping_status"].eq("deterministic_reference_post_event").sum()),
                "historical_pit_timestamp_rows": int(in_scope["known_at_status"].isin(PIT_SAFE_STATUSES).sum()),
                "historical_pit_timestamp_gap_rows": int((~in_scope["known_at_status"].isin(PIT_SAFE_STATUSES)).sum()),
                "completion_gap_rows": len(gaps),
                "tse_wayback_lifecycle_packets": int(
                    lifecycle_packet.lineage.get("packet_count", 0)
                )
                if lifecycle_packet is not None
                else 0,
                "tse_wayback_lifecycle_months": int(
                    lifecycle_packet.lineage.get("selected_month_count", 0)
                )
                if lifecycle_packet is not None
                else 0,
                "tse_wayback_lifecycle_claims": len(lifecycle_packet.claims)
                if lifecycle_packet is not None
                else 0,
                "tse_wayback_lifecycle_source_rows": len(lifecycle_packet.events)
                if lifecycle_packet is not None
                else 0,
            },
            "g1_lineage_admitted": g1 is not None,
            "timestamp_recovery_admitted": timestamp_recovery_dir is not None,
            "official_timestamp_recovery_admitted": (
                official_timestamp_recovery_dir is not None
            ),
            "official_timestamp_recovery_rows": (
                len(timestamp_sidecar.accepted)
                if official_timestamp_recovery_dir is not None
                else 0
            ),
            "official_timestamp_v14_recovery_admitted": (
                official_timestamp_v14_recovery_dir is not None
            ),
            "official_timestamp_v14_recovery_rows": (
                len(timestamp_v14_sidecar.accepted)
                if official_timestamp_v14_recovery_dir is not None
                else 0
            ),
            "halt_report_event_packet_admitted": halt_report_events_dir is not None,
            "halt_normalization_repair_admitted": (
                halt_normalization_repair_dir is not None
            ),
            "lifecycle_taxonomy_repair_admitted": (
                lifecycle_taxonomy_repair_dir is not None
            ),
            "lifecycle_taxonomy_repair_rows": (
                len(taxonomy_sidecar.repairs)
                if lifecycle_taxonomy_repair_dir is not None
                else 0
            ),
            "tse_wayback_lifecycle_packet_admitted": bool(lifecycle_pairs),
            "halt_mapping_recovery_admitted": bool(mapping_recovery_pairs),
            "halt_mapping_recovery_rows": int(
                in_scope["mapping_status"]
                .eq("pit_exact_official_monthly_quote_recovery")
                .sum()
            ),
            "halt_report_event_packet": {
                "source_rows": len(report_admission),
                "covered_claims": len(covered_report_claims(report_admission))
                if len(report_admission)
                else 0,
                "research_scope_rows": int(
                    _bool_series(report_admission["in_research_scope"]).sum()
                )
                if len(report_admission)
                else 0,
                "normalized_rows": int(
                    report_admission["normalization_status"].eq("normalized").sum()
                )
                if len(report_admission)
                else 0,
                "admission_ready_rows": int(
                    _bool_series(report_admission["admission_ready"]).sum()
                )
                if len(report_admission)
                else 0,
            },
            "tse_wayback_lifecycle_packet": {
                "packet_count": int(
                    lifecycle_packet.lineage.get("packet_count", 0)
                )
                if lifecycle_packet is not None
                else 0,
                "months": list(lifecycle_packet.lineage.get("selected_months", []))
                if lifecycle_packet is not None
                else [],
                "month_count": int(
                    lifecycle_packet.lineage.get("selected_month_count", 0)
                )
                if lifecycle_packet is not None
                else 0,
                "source_rows": len(lifecycle_packet.events)
                if lifecycle_packet is not None
                else 0,
                "covered_claims": len(lifecycle_packet.claims)
                if lifecycle_packet is not None
                else 0,
                "claim_count": len(lifecycle_packet.claims)
                if lifecycle_packet is not None
                else 0,
                "common_equity_rows": int(
                    lifecycle_packet.events["instrument_class"].eq("equity").sum()
                )
                if lifecycle_packet is not None
                else 0,
                "preserved_non_equity_rows": int(
                    lifecycle_packet.events["instrument_class"].ne("equity").sum()
                )
                if lifecycle_packet is not None
                else 0,
                "zero_event_claims": 0,
                "packets": list(lifecycle_packet.lineage.get("packets", []))
                if lifecycle_packet is not None
                else [],
            },
            "producer_blockers": blockers,
            "critical_blockers": blockers,
            "artifacts": {path.stem: str(path) for path in output_paths},
            "output_sha256": {str(path.relative_to(root)): _sha256(path) for path in output_paths},
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "generated_at_utc": _utc_iso(now()),
        }
        _atomic_json(manifest_path, manifest)
        return JapanG4HaltArtifacts(run_id, root, manifest_path, events_path, plan_path, manifest)
    finally:
        lock.close()


def _source_row(
    *,
    source_id: str,
    source_kind: str,
    channel: str,
    source_url: str,
    local_path: Path,
    expected_sha256: str,
    observed_at_utc: str,
    coverage_start: str = "",
    coverage_end: str = "",
    denominator_complete: bool = False,
    no_event_capable: bool = False,
    pit_timestamp_status: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "channel": channel,
        "official_source": True,
        "source_url": source_url,
        "local_path": str(local_path),
        "expected_sha256": expected_sha256,
        "observed_at_utc": observed_at_utc,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "denominator_complete": denominator_complete,
        "no_event_capable": no_event_capable,
        "pit_timestamp_status": pit_timestamp_status,
    }


def _catalog_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value, dtype=str)
    return frame.reindex(columns=SOURCE_CATALOG_COLUMNS).fillna("")


def _validate_catalog(catalog: pd.DataFrame) -> None:
    if catalog.empty:
        raise ValueError("source catalog cannot be empty")
    if catalog["source_id"].astype(str).str.strip().eq("").any() or catalog["source_id"].duplicated().any():
        raise ValueError("source catalog requires unique nonblank source_id values")
    if not catalog["source_kind"].isin(SOURCE_KINDS).all():
        raise ValueError("source catalog contains unsupported source_kind")
    if not catalog["channel"].isin(CHANNELS).all():
        raise ValueError("source catalog contains unsupported channel")
    for item in catalog.to_dict("records"):
        if not item["local_path"] and not item["source_url"]:
            raise ValueError(f"source has neither local_path nor URL: {item['source_id']}")
        if _truth(item["denominator_complete"]):
            if not item["coverage_start"] or not item["coverage_end"]:
                raise ValueError(f"complete source lacks coverage bounds: {item['source_id']}")
            if not _truth(item["no_event_capable"]):
                raise ValueError(f"complete source cannot preserve no-event evidence: {item['source_id']}")


def _source_payload(item: dict[str, Any], fetcher: Fetcher, timeout: float) -> bytes:
    if item["local_path"]:
        return Path(item["local_path"]).read_bytes()
    return fetcher(item["source_url"], timeout)


def _raw_suffix(item: dict[str, Any]) -> str:
    suffix = Path(item["local_path"] or item["source_url"]).suffix.lower()
    return suffix if suffix in {".html", ".htm", ".csv", ".json", ".pdf", ".xml"} else ".bin"


def _parse_source(path: Path, item: dict[str, Any], observed_at: str) -> pd.DataFrame:
    if item["source_kind"] == "inventory_only":
        return pd.DataFrame(columns=EVENT_COLUMNS)
    digest = _sha256(path)
    if item["source_kind"] == "jpx_halt_html":
        parsed = parse_halts(path.read_bytes()).fillna("")
        rows = [_halt_event(row, item, digest, observed_at) for row in parsed.to_dict("records")]
    else:
        parsed = parse_delisting_html(path.read_bytes(), item["source_url"], "").fillna("")
        rows = [
            _delisting_event(row, item, digest, observed_at)
            for row in parsed.to_dict("records")
        ]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _halt_event(row: dict[str, Any], source: dict[str, Any], digest: str, observed: str) -> dict[str, Any]:
    event_date = row["date"]
    start = _jst_timestamp(event_date, row["suspension_start"])
    suspension_end = _jst_timestamp(event_date, row["suspension_end"])
    resumption = _jst_timestamp(event_date, row["restart_time"])
    record = "|".join(str(row.get(key, "")).strip() for key in ("date", "code", "suspension_start", "suspension_end", "restart_time", "reason"))
    return {
        "event_id": "jpx-halt-" + hashlib.sha256(record.encode("utf-8")).hexdigest()[:24],
        "event_type": "trading_halt",
        "exchange": "TSE",
        "local_code": str(row["code"]).strip().upper(),
        "issue_name": row["issue_name"],
        "event_date": event_date,
        "event_start_jst": start,
        "suspension_end_jst": suspension_end,
        "resumption_jst": resumption,
        "interval_end_jst": resumption,
        "known_at_utc": observed,
        "known_at_status": source["pit_timestamp_status"] or "snapshot_observation_only",
        "source_observed_at_utc": observed,
        "known_at_evidence_sha256": digest,
        "reason": row["reason"],
        "source_id": source["source_id"],
        "source_url": source["source_url"],
        "source_sha256": digest,
        "source_record_id": record,
        "stable_listing_id": "",
        "refinitiv_ric": "",
        "mapping_known_at": "",
        "mapping_status": "unmapped",
        "mapping_route": "",
        "mapping_source_sha256": "",
        "mapping_source_record_id": "",
    }


def _delisting_event(
    row: dict[str, Any], source: dict[str, Any], digest: str, observed: str
) -> dict[str, Any]:
    record = "|".join(str(row.get(key, "")).strip() for key in ("event_date", "code", "reason"))
    event_date = row["event_date"]
    return {
        "event_id": "jpx-delist-" + hashlib.sha256(record.encode("utf-8")).hexdigest()[:24],
        "event_type": "delisting_non_trading_interval",
        "exchange": "TSE",
        "local_code": str(row["code"]).strip().upper(),
        "issue_name": row["issue_name"],
        "event_date": event_date,
        "event_start_jst": datetime.combine(date.fromisoformat(event_date), time(), JST).isoformat(),
        "suspension_end_jst": "",
        "resumption_jst": "",
        "interval_end_jst": "",
        "known_at_utc": row.get("known_at_utc", ""),
        "known_at_status": row.get("known_at_status", "") or source["pit_timestamp_status"],
        "source_observed_at_utc": observed,
        "known_at_evidence_sha256": "",
        "reason": row["reason"],
        "source_id": source["source_id"],
        "source_url": source["source_url"],
        "source_sha256": digest,
        "source_record_id": record,
        "stable_listing_id": "",
        "refinitiv_ric": "",
        "mapping_known_at": "",
        "mapping_status": "unmapped",
        "mapping_route": "",
        "mapping_source_sha256": "",
        "mapping_source_record_id": "",
    }


def _events_from_sources(catalog: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    complete = requests.loc[requests["status"].eq("completed")]
    rows: list[pd.DataFrame] = []
    for item in catalog.merge(complete, on="source_id", suffixes=("", "_request")).to_dict("records"):
        path = Path(item["raw_path"])
        if not path.is_file() or _sha256(path) != item["raw_sha256"]:
            raise ValueError(f"completed source hash mismatch: {item['source_id']}")
        rows.append(_parse_source(path, item, item["observed_at_utc_request"]))
    events = pd.concat(rows, ignore_index=True).reindex(columns=EVENT_COLUMNS) if rows else pd.DataFrame(columns=EVENT_COLUMNS)
    return events.drop_duplicates("event_id").sort_values(["event_date", "local_code", "event_id"]).reset_index(drop=True)


def _copy_symbol_map(path_value: str | Path | None, target: Path) -> pd.DataFrame:
    if path_value is None:
        frame = pd.DataFrame(columns=["local_code", "refinitiv_ric", "listing_start", "listing_end", "known_at", "pit_usable", "stable_listing_id"])
        _atomic_csv(target, frame)
        return frame
    source = Path(path_value).resolve()
    frame = pd.read_csv(source, dtype=str).fillna("")
    if "local_code" not in frame or "refinitiv_ric" not in frame:
        raise ValueError("symbol map requires local_code and refinitiv_ric")
    _atomic_bytes(target, source.read_bytes())
    return frame


def _load_g1_evidence(
    *,
    producer_manifest_path: str | Path | None,
    validation_manifest_path: str | Path | None,
    event_codes: set[str],
) -> _G1Evidence | None:
    if producer_manifest_path is None and validation_manifest_path is None:
        return None
    if producer_manifest_path is None or validation_manifest_path is None:
        raise ValueError("both G1 producer and validation manifests are required")
    producer_path = Path(producer_manifest_path).resolve()
    validation_path = Path(validation_manifest_path).resolve()
    producer = _json(producer_path)
    validation = _json(validation_path)
    if not (
        producer.get("status") == "complete"
        and producer.get("historical_pit_ready") is True
        and not producer.get("critical_blockers")
        and validation.get("status") == "pass"
        and validation.get("independent_audit") is True
        and validation.get("source_checksums_verified") is True
        and validation.get("scorecard_eligible") is True
        and validation.get("producer_run_id") == producer.get("run_id")
    ):
        raise ValueError("G1 producer/validator admission contract failed")
    for filename, digest in producer.get("output_sha256", {}).items():
        path = producer_path.parent / filename
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"G1 producer output hash mismatch: {filename}")
    for filename, digest in validation.get("output_sha256", {}).items():
        path = Path(filename)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"G1 validator output hash mismatch: {filename}")

    reference_dir = Path(validation["output_directories"]["reference_map"])
    reference_path = reference_dir / "symbol_map.csv"
    reference_manifest = _json(reference_dir / "manifest.json")
    reference_digest = _sha256(reference_path)
    if not (
        reference_manifest.get("status") == "pass"
        and reference_manifest.get("independent_audit") is True
        and reference_manifest.get("map_role") == "reference"
        and reference_manifest.get("symbol_map_sha256") == reference_digest
    ):
        raise ValueError("validated G1 reference-map contract failed")
    symbol_map = pd.read_csv(reference_path, dtype=str).fillna("")

    security_frames: list[pd.DataFrame] = []
    security_lineage: list[dict[str, str]] = []
    for key, manifest_value in producer.get("input_manifests", {}).items():
        match = re.match(r"security_master_(\d{2})_", key)
        if not match:
            continue
        index = match.group(1)
        manifest_path = Path(manifest_value)
        manifest_digest = _sha256(manifest_path)
        if manifest_digest != producer.get("input_manifest_sha256", {}).get(key):
            raise ValueError(f"G1 input manifest hash mismatch: {key}")
        manifest = _json(manifest_path)
        source_path = Path(str(manifest.get("artifacts", {}).get("security_master", "")))
        source_digest = _sha256(source_path)
        if source_digest != producer.get("input_data_sha256", {}).get(
            f"security_master_{index}.csv"
        ):
            raise ValueError(f"G1 security-master data hash mismatch: {key}")
        frame = pd.read_csv(source_path, dtype=str).fillna("")
        for column in (
            "mapping_local_code_override",
            "direct_code",
            "ticker_symbol",
            "ric",
            "refinitiv_ric",
            "exchange_code",
            "retire_date",
            "first_trade_date",
            "source_loaded_at_utc",
        ):
            if column not in frame:
                frame[column] = ""
        frame["local_code"] = frame["mapping_local_code_override"]
        frame.loc[frame["local_code"].eq(""), "local_code"] = frame.loc[
            frame["local_code"].eq(""), "direct_code"
        ]
        frame.loc[frame["local_code"].eq(""), "local_code"] = frame.loc[
            frame["local_code"].eq(""), "ticker_symbol"
        ]
        frame["candidate_ric"] = frame["ric"]
        frame.loc[frame["candidate_ric"].eq(""), "candidate_ric"] = frame.loc[
            frame["candidate_ric"].eq(""), "refinitiv_ric"
        ]
        frame = frame.loc[frame["local_code"].isin(event_codes)].copy()
        frame["admitted_source_path"] = str(source_path)
        frame["admitted_source_sha256"] = source_digest
        frame["admitted_manifest_path"] = str(manifest_path)
        frame["admitted_manifest_sha256"] = manifest_digest
        frame["source_record_id"] = frame.apply(
            lambda row: hashlib.sha256(
                "|".join(
                    (
                        source_digest,
                        row["local_code"],
                        row["candidate_ric"],
                        row["exchange_code"],
                        row["first_trade_date"],
                        row["retire_date"],
                    )
                ).encode("utf-8")
            ).hexdigest()[:24],
            axis=1,
        )
        security_frames.append(frame)
        security_lineage.append(
            {
                "source_key": key,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_digest,
                "security_master_path": str(source_path),
                "security_master_sha256": source_digest,
            }
        )
    security = (
        pd.concat(security_frames, ignore_index=True, sort=False).fillna("")
        if security_frames
        else pd.DataFrame()
    )
    lineage = {
        "g1_producer_manifest": str(producer_path),
        "g1_producer_manifest_sha256": _sha256(producer_path),
        "g1_validation_manifest": str(validation_path),
        "g1_validation_manifest_sha256": _sha256(validation_path),
        "reference_symbol_map": str(reference_path),
        "reference_symbol_map_sha256": reference_digest,
        "security_master_sources": security_lineage,
    }
    return _G1Evidence(symbol_map, security, reference_digest, lineage)


def _attach_pit_mapping(
    events: pd.DataFrame,
    symbol_map: pd.DataFrame,
    *,
    security_candidates: pd.DataFrame | None = None,
    symbol_map_sha256: str = "",
) -> pd.DataFrame:
    result = events.copy()
    if result.empty:
        return result.reindex(columns=EVENT_COLUMNS)
    mapped_rows = []
    for event in result.to_dict("records"):
        candidates = symbol_map.loc[symbol_map["local_code"].astype(str).eq(event["local_code"])].copy()
        event_day = event["event_date"]
        mapping_day = (
            (date.fromisoformat(event_day) - timedelta(days=1)).isoformat()
            if event["event_type"] == "delisting_non_trading_interval"
            else event_day
        )
        if "listing_start" in candidates:
            candidates = candidates.loc[candidates["listing_start"].eq("") | candidates["listing_start"].le(mapping_day)]
        if "listing_end" in candidates:
            right_open = _bool_series(candidates["right_open"]) if "right_open" in candidates else pd.Series(False, index=candidates.index)
            candidates = candidates.loc[
                right_open
                | candidates["listing_end"].eq("")
                | candidates["listing_end"].ge(mapping_day)
            ]
        if "pit_usable" in candidates:
            candidates = candidates.loc[_bool_series(candidates["pit_usable"])]
        if "known_at" in candidates:
            candidates = candidates.loc[candidates["known_at"].ne("") & candidates["known_at"].str[:10].le(event_day)]
        candidates = candidates.loc[candidates["refinitiv_ric"].ne("")]
        if len(candidates) == 1:
            match = candidates.iloc[0]
            event["stable_listing_id"] = match.get("stable_listing_id", "")
            event["refinitiv_ric"] = match["refinitiv_ric"]
            event["mapping_known_at"] = match.get("known_at", "")
            event["mapping_status"] = (
                "pit_exact_g1_monthly" if security_candidates is not None else "pit_exact"
            )
            event["mapping_route"] = "g1_validated_monthly_window" if security_candidates is not None else "provided_symbol_map"
            event["mapping_source_sha256"] = symbol_map_sha256
            event["mapping_source_record_id"] = match.get("stable_listing_id", "")
        elif len(candidates) > 1:
            event["mapping_status"] = "ambiguous_pit_candidates"
        else:
            fallback = _retirement_mapping_candidate(
                event, security_candidates if security_candidates is not None else pd.DataFrame()
            )
            if fallback is None:
                event["mapping_status"] = "missing_admitted_mapping"
                event["mapping_route"] = "no_unique_admitted_candidate"
            else:
                relations = symbol_map.loc[
                    symbol_map["local_code"].astype(str).eq(event["local_code"])
                    & symbol_map["refinitiv_ric"].eq(fallback["candidate_ric"])
                ].copy()
                if "known_at" in relations:
                    relations = relations.loc[
                        relations["known_at"].ne("")
                        & relations["known_at"].str[:10].le(event_day)
                    ]
                relations = relations.sort_values(["listing_end", "listing_start"])
                event["refinitiv_ric"] = fallback["candidate_ric"]
                event["mapping_source_sha256"] = fallback["admitted_source_sha256"]
                event["mapping_source_record_id"] = fallback["source_record_id"]
                if len(relations):
                    relation = relations.iloc[-1]
                    event["stable_listing_id"] = relation.get("stable_listing_id", "")
                    event["mapping_known_at"] = relation.get("known_at", "")
                    event["mapping_status"] = "pit_exact_g1_identity_retire_bridge"
                    event["mapping_route"] = "official_delisting_code_plus_g1_identity_plus_exact_retire_date"
                else:
                    event["mapping_known_at"] = fallback.get("source_loaded_at_utc", "")
                    event["mapping_status"] = "deterministic_reference_post_event"
                    event["mapping_route"] = "official_delisting_code_plus_exact_admitted_retire_date_post_event"
        mapped_rows.append(event)
    return pd.DataFrame(mapped_rows, columns=EVENT_COLUMNS)


def _retirement_mapping_candidate(
    event: dict[str, Any], security_candidates: pd.DataFrame
) -> dict[str, Any] | None:
    if event["event_type"] != "delisting_non_trading_interval" or security_candidates.empty:
        return None
    code = re.escape(event["local_code"])
    candidates = security_candidates.loc[
        security_candidates["local_code"].eq(event["local_code"])
        & security_candidates["retire_date"].eq(event["event_date"])
        & security_candidates["exchange_code"].eq("TYO")
        & security_candidates["candidate_ric"].str.match(
            rf"^{code}\.T(?:\^.*)?$", case=False, na=False
        )
    ].drop_duplicates("candidate_ric")
    return candidates.iloc[0].to_dict() if len(candidates) == 1 else None


def _resolve_pit_timestamps(
    events: pd.DataFrame, symbol_map: pd.DataFrame, symbol_map_sha256: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_listing = (
        symbol_map.set_index("stable_listing_id", drop=False).to_dict("index")
        if "stable_listing_id" in symbol_map and not symbol_map["stable_listing_id"].duplicated().any()
        else {}
    )
    for event in events.to_dict("records"):
        if event["event_type"] == "trading_halt" and event["event_start_jst"]:
            event["known_at_utc"] = pd.Timestamp(event["event_start_jst"]).tz_convert("UTC").isoformat()
            event["known_at_status"] = "conservative_halt_start_market_state"
            event["known_at_evidence_sha256"] = event["source_sha256"]
        elif event["event_type"] == "delisting_non_trading_interval":
            mapping = by_listing.get(event["stable_listing_id"], {})
            end_known = str(mapping.get("window_end_known_at", ""))
            listing_end = str(mapping.get("listing_end", ""))
            if (
                end_known
                and listing_end
                and listing_end <= event["event_date"]
                and end_known[:10] >= event["event_date"]
            ):
                event["known_at_utc"] = end_known
                event["known_at_status"] = "conservative_g1_monthly_absence_observation"
                event["known_at_evidence_sha256"] = symbol_map_sha256
            else:
                event["known_at_utc"] = ""
                event["known_at_status"] = "missing_historical_publication_or_absence_timestamp"
                event["known_at_evidence_sha256"] = ""
        rows.append(event)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _copy_validated_timestamp_recovery(
    source_dir: Path, destination: Path
) -> tuple[pd.DataFrame, list[Path]]:
    source = source_dir.resolve()
    producer_manifest_path = source / "manifest.json"
    validation_manifest_path = source / "validation" / "manifest.json"
    producer = _json(producer_manifest_path)
    validation = _json(validation_manifest_path)
    if not (
        producer.get("status") == "recovery_partial"
        and validation.get("status") == "pass"
        and validation.get("independent_validation_passed") is True
        and validation.get("producer_manifest_sha256") == _sha256(producer_manifest_path)
        and not validation.get("validation_issues")
    ):
        raise ValueError("timestamp recovery independent validation contract failed")

    output_hashes = producer.get("output_sha256", {})
    if not isinstance(output_hashes, dict) or "timestamp_recovery.csv" not in output_hashes:
        raise ValueError("timestamp recovery output denominator is incomplete")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for relative_name, digest in output_hashes.items():
        relative = Path(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("timestamp recovery output path is unsafe")
        source_path = source / relative
        if not source_path.is_file() or _sha256(source_path) != str(digest):
            raise ValueError(f"timestamp recovery output hash mismatch: {relative_name}")
        target = destination / relative
        _atomic_bytes(target, source_path.read_bytes())
        copied.append(target)

    producer_copy = destination / "producer_manifest.json"
    validation_copy = destination / "validation_manifest.json"
    _atomic_bytes(producer_copy, producer_manifest_path.read_bytes())
    _atomic_bytes(validation_copy, validation_manifest_path.read_bytes())
    copied.extend((producer_copy, validation_copy))
    validation_hashes = validation.get("output_sha256", {})
    for relative_name, digest in validation_hashes.items():
        relative = Path(str(relative_name))
        source_path = source / "validation" / relative
        if relative.is_absolute() or ".." in relative.parts or not source_path.is_file():
            raise ValueError("timestamp recovery validation output path is invalid")
        if _sha256(source_path) != str(digest):
            raise ValueError(f"timestamp recovery validation hash mismatch: {relative_name}")
        target = destination / "validation" / relative
        _atomic_bytes(target, source_path.read_bytes())
        copied.append(target)

    recovery = pd.read_csv(destination / "timestamp_recovery.csv", dtype=str).fillna("")
    recovered = recovery.loc[recovery["recovery_status"].eq("recovered")].copy()
    if recovered.empty or recovered["event_id"].duplicated().any():
        raise ValueError("timestamp recovery accepted event denominator is invalid")
    raw_dir = destination / "raw"
    for row in recovered.to_dict("records"):
        raw_source = Path(row["notice_source_path"])
        digest = row["notice_source_sha256"]
        if not raw_source.is_file() or _sha256(raw_source) != digest:
            raise ValueError(f"timestamp recovery notice hash mismatch: {row['event_id']}")
        target = raw_dir / f"{digest}{raw_source.suffix or '.html'}"
        _atomic_bytes(target, raw_source.read_bytes())
        copied.append(target)
    lineage_path = destination / "lineage.json"
    _atomic_json(
        lineage_path,
        {
            "producer_run_id": producer.get("run_id"),
            "producer_manifest_sha256": _sha256(producer_copy),
            "validation_manifest_sha256": _sha256(validation_copy),
            "recovered_rows": len(recovered),
            "raw_notice_rows": len(recovered),
        },
    )
    copied.append(lineage_path)
    return recovered, copied


def _apply_timestamp_recovery(events: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    if events.empty or recovery.empty:
        return events
    by_event = recovery.set_index("event_id", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        item = by_event.get(event["event_id"])
        if item is not None:
            known_at = pd.Timestamp(item["known_at_utc"])
            publication = date.fromisoformat(item["publication_date"])
            if (
                known_at.tzinfo is None
                or known_at.tz_convert("UTC").date() < publication
                or item["timestamp_rule"] != "first_instant_after_official_publication_date_in_JST"
            ):
                raise ValueError(f"timestamp recovery semantics invalid: {event['event_id']}")
            event["known_at_utc"] = known_at.tz_convert("UTC").isoformat()
            event["known_at_status"] = "official_jpx_publication_date_recovery"
            event["known_at_evidence_sha256"] = item["notice_source_sha256"]
        rows.append(event)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _apply_official_timestamp_sidecar(
    events: pd.DataFrame,
    recovery: pd.DataFrame,
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    if events.empty or recovery.empty:
        if require_all and not recovery.empty:
            raise ValueError("official timestamp sidecar event denominator missing")
        return events
    by_event = recovery.set_index("event_id", drop=False).to_dict("index")
    matched: set[str] = set()
    rows: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        item = by_event.get(event["event_id"])
        if item is not None:
            known_at = pd.Timestamp(item["known_at_utc"])
            event_day = pd.Timestamp(event["event_date"], tz="Asia/Tokyo")
            if (
                known_at.tzinfo is None
                or known_at.tz_convert("Asia/Tokyo").date() > event_day.date()
                or item["known_at_rule"]
                != "date_only_conservative_upper_bound_next_midnight_jst"
            ):
                raise ValueError(
                    f"official timestamp sidecar semantics invalid: {event['event_id']}"
                )
            existing_known_at = (
                pd.Timestamp(event["known_at_utc"])
                if event["known_at_utc"]
                else None
            )
            replaceable_g1_observation = bool(
                event["known_at_status"]
                == "conservative_g1_monthly_absence_observation"
                and existing_known_at is not None
                and existing_known_at.tzinfo is not None
                and known_at < existing_known_at
            )
            if not (
                event["known_at_status"]
                == "missing_historical_publication_or_absence_timestamp"
                and not event["known_at_utc"]
            ) and not replaceable_g1_observation:
                raise ValueError(
                    "official timestamp sidecar would overwrite PIT evidence: "
                    f"{event['event_id']}"
                )
            event["known_at_utc"] = known_at.tz_convert("UTC").isoformat()
            event["known_at_status"] = "official_jpx_publication_date_recovery"
            event["known_at_evidence_sha256"] = item["source_sha256"]
            matched.add(event["event_id"])
        rows.append(event)
    if require_all and matched != set(recovery["event_id"]):
        missing = sorted(set(recovery["event_id"]) - matched)
        raise ValueError(
            f"official timestamp sidecar event denominator mismatch: {missing[:5]}"
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _normalize_tse_wayback_lifecycle_pairs(
    producer_dir: str | Path | None,
    validation_dir: str | Path | None,
    pairs: Iterable[tuple[str | Path, str | Path]] | None,
) -> tuple[tuple[Path, Path], ...]:
    if (producer_dir is None) != (validation_dir is None):
        raise ValueError(
            "both TSE Wayback lifecycle and validation directories are required"
        )
    supplied = tuple(pairs or ())
    if supplied and producer_dir is not None:
        raise ValueError(
            "use either the legacy TSE Wayback lifecycle pair or multiple pairs, not both"
        )
    if producer_dir is not None:
        supplied = ((producer_dir, validation_dir),)
    normalized: list[tuple[Path, Path]] = []
    for index, pair in enumerate(supplied, start=1):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"TSE Wayback lifecycle pair {index} must contain producer and validation paths"
            )
        producer_value, validation_value = pair
        if producer_value is None or validation_value is None:
            raise ValueError(
                f"TSE Wayback lifecycle pair {index} must contain producer and validation paths"
            )
        normalized.append((Path(producer_value), Path(validation_value)))
    return tuple(normalized)


def _copy_validated_lifecycle_taxonomy_repair(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: Path,
) -> _ValidatedLifecycleTaxonomyRepair:
    producer = Path(producer_dir).resolve()
    validation = Path(validation_dir).resolve()
    producer_manifest = _json(producer / "manifest.json")
    validation_manifest = _json(validation / "manifest.json")
    producer_sha = _sha256(producer / "manifest.json")
    producer_issues = verify_japan_g4_lifecycle_taxonomy_repair(producer)
    validation_issues = verify_japan_g4_lifecycle_taxonomy_repair_validation(
        validation
    )
    contract = (
        producer_manifest.get("schema_version")
        == "1.0-bounded-v14-lifecycle-taxonomy-repair"
        and producer_manifest.get("status")
        == "bounded_taxonomy_repair_candidate_pending_independent_validation"
        and int(producer_manifest.get("repair_rows", -1)) == 9
        and int(producer_manifest.get("warrant_rows", -1)) == 5
        and int(producer_manifest.get("foreign_share_rows", -1)) == 4
        and int(producer_manifest.get("classification_inferred_rows", -1)) == 0
        and int(producer_manifest.get("mappings_created", -1)) == 0
        and int(producer_manifest.get("timestamps_created", -1)) == 0
        and int(producer_manifest.get("zero_events_created", -1)) == 0
        and producer_manifest.get("refinitiv_used") is False
        and validation_manifest.get("schema_version")
        == "1.0-independent-pdfplumber-taxonomy-validation"
        and validation_manifest.get("status")
        == "pass_bounded_source_backed_taxonomy_repair"
        and validation_manifest.get("producer_run_id")
        == producer_manifest.get("run_id")
        and validation_manifest.get("producer_manifest_sha256") == producer_sha
        and int(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("validation_issue_codes") == []
        and int(validation_manifest.get("repair_rows_reconciled", -1)) == 9
        and int(validation_manifest.get("warrant_rows_reconciled", -1)) == 5
        and int(validation_manifest.get("foreign_share_rows_reconciled", -1))
        == 4
        and validation_manifest.get("source_record_hashes_recomputed") is True
        and validation_manifest.get("event_ids_recomputed") is True
        and int(validation_manifest.get("classification_inferred_rows", -1)) == 0
        and int(validation_manifest.get("mappings_created", -1)) == 0
        and int(validation_manifest.get("timestamps_created", -1)) == 0
        and int(validation_manifest.get("zero_events_created", -1)) == 0
        and validation_manifest.get("refinitiv_used") is False
    )
    if producer_issues or validation_issues or not contract:
        raise ValueError("lifecycle taxonomy repair institutional contract failed")

    repairs = pd.read_csv(producer / "taxonomy_repairs.csv", dtype=str).fillna("")
    if (
        tuple(repairs.columns) != tuple(LIFECYCLE_TAXONOMY_REPAIR_COLUMNS)
        or len(repairs) != 9
        or repairs["event_id"].duplicated().any()
        or repairs["source_record_sha256"].duplicated().any()
        or not repairs["reclassified_instrument_class"].isin(
            {"warrant", "foreign_share"}
        ).all()
        or _bool_series(repairs["in_research_scope_after"]).any()
        or _bool_series(repairs["mapping_created"]).any()
        or _bool_series(repairs["timestamp_created"]).any()
        or _bool_series(repairs["zero_event_created"]).any()
        or _bool_series(repairs["classification_inferred"]).any()
    ):
        raise ValueError("lifecycle taxonomy repair exact denominator failed")

    copied: list[Path] = []
    copied.extend(_copy_sealed_tree(producer, destination / "producer"))
    copied.extend(_copy_sealed_tree(validation, destination / "validation"))
    copied_repairs = pd.read_csv(
        destination / "producer" / "taxonomy_repairs.csv", dtype=str
    ).fillna("")
    if (
        verify_japan_g4_lifecycle_taxonomy_repair(destination / "producer")
        or verify_japan_g4_lifecycle_taxonomy_repair_validation(
            destination / "validation"
        )
        or not repairs.equals(copied_repairs)
    ):
        raise ValueError("lifecycle taxonomy repair copied packet changed")
    lineage = {
        "producer_run_id": producer_manifest["run_id"],
        "producer_manifest_sha256": _sha256(
            destination / "producer" / "manifest.json"
        ),
        "validation_run_id": validation_manifest["run_id"],
        "validation_manifest_sha256": _sha256(
            destination / "validation" / "manifest.json"
        ),
        "repair_rows": 9,
        "warrant_rows": 5,
        "foreign_share_rows": 4,
        "research_scope_rows_after": 0,
        "mappings_created": 0,
        "timestamps_created": 0,
        "zero_events_created": 0,
        "classification_inferred_rows": 0,
        "independently_validated": True,
        "refinitiv_used": False,
    }
    lineage_path = destination / "lineage.json"
    _atomic_json(lineage_path, lineage)
    copied.append(lineage_path)
    return _ValidatedLifecycleTaxonomyRepair(
        repairs=copied_repairs,
        copied_paths=tuple(copied),
        lineage=lineage,
    )


def _validate_official_timestamp_v14_contract(
    producer: Path, validation: Path
) -> tuple[pd.DataFrame, tuple[Path, ...], tuple[Path, ...]]:
    producer_manifest = _json(producer / "manifest.json")
    validation_manifest = _json(validation / "manifest.json")
    producer_sha = _sha256(producer / "manifest.json")
    producer_counts = producer_manifest.get("counts", {})
    validation_counts = validation_manifest.get("counts", {})
    policy = producer_manifest.get("source_policy", {})
    contract = (
        producer_manifest.get("component")
        == "japan_g4_halt_v14_six_official_publication_timestamp_recovery"
        and producer_manifest.get("schema_version")
        == "1.0-v14-six-bounded-immutable"
        and producer_manifest.get("status") == "complete"
        and int(producer_counts.get("target_events", -1)) == 6
        and int(producer_counts.get("accepted_events", -1)) == 6
        and int(producer_counts.get("rejected_events", -1)) == 0
        and int(producer_counts.get("retained_source_responses", -1)) == 6
        and policy.get("official_jpx_only") is True
        and policy.get("redirects_accepted") is False
        and policy.get("inferred_publication_time_used") is False
        and policy.get("issuer_domains_accepted") is False
        and policy.get("excluded_codes") == ["4283"]
        and policy.get("zero_equity_channel_included") is False
        and producer_manifest.get("refinitiv_calls") is False
        and validation_manifest.get("component")
        == "japan_g4_halt_v14_six_official_publication_timestamp_recovery_validation"
        and validation_manifest.get("schema_version")
        == "1.0-independent-v14-six-retained-byte-validation"
        and validation_manifest.get("status") == "pass"
        and validation_manifest.get("producer_run_id")
        == producer_manifest.get("run_id")
        and validation_manifest.get("producer_manifest", {}).get("sha256")
        == producer_sha
        and validation_manifest.get("independent_implementation") is True
        and validation_manifest.get("retained_bytes_reparsed") is True
        and int(validation_counts.get("target_events", -1)) == 6
        and int(validation_counts.get("independently_validated_events", -1))
        == 6
        and int(validation_counts.get("validation_issues", -1)) == 0
        and validation_manifest.get("validation_issues") == []
        and validation_manifest.get("refinitiv_calls") is False
    )
    if not contract:
        raise ValueError("V14 official timestamp institutional contract failed")

    producer_files: set[Path] = {Path("manifest.json")}
    for relative_name, descriptor in producer_manifest.get("outputs", {}).items():
        relative = Path(str(relative_name))
        path = producer / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or not isinstance(descriptor, dict)
            or _sha256(path) != descriptor.get("sha256")
            or path.stat().st_size != int(descriptor.get("bytes", -1))
        ):
            raise ValueError(f"V14 timestamp producer output invalid: {relative_name}")
        producer_files.add(relative)
    sources = pd.read_csv(
        producer / "official_source_manifest.csv", dtype=str
    ).fillna("")
    if (
        len(sources) != 6
        or not sources["source_id"].is_unique
        or not _bool_series(sources["accepted"]).all()
        or sources["quarantine_reason"].ne("").any()
        or sources["http_status"].ne("200").any()
        or sources["requested_url"].ne(sources["final_url"]).any()
    ):
        raise ValueError("V14 timestamp source denominator invalid")
    for row in sources.to_dict("records"):
        raw_original = Path(row["raw_path"])
        metadata_original = Path(row["metadata_path"])
        for original, digest, label in (
            (raw_original, row["sha256"], "raw"),
            (metadata_original, row["metadata_sha256"], "metadata"),
        ):
            try:
                relative = original.resolve().relative_to(producer)
            except ValueError:
                relative = (
                    Path("raw") / original.name
                    if label == "raw"
                    else Path("raw") / "metadata" / original.name
                )
            path = producer / relative
            if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"V14 timestamp {label} source invalid")
            producer_files.add(relative)

    accepted = pd.read_csv(
        producer / "accepted_timestamp_recovery.csv", dtype=str
    ).fillna("")
    if (
        len(accepted) != 6
        or not accepted["event_id"].is_unique
        or accepted["status"].ne("accepted").any()
        or accepted["known_at_rule"].ne(
            "date_only_conservative_upper_bound_next_midnight_jst"
        ).any()
        or accepted["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").ne(True).any()
    ):
        raise ValueError("V14 timestamp accepted claim denominator invalid")

    validation_files: set[Path] = {Path("manifest.json")}
    for relative_name, descriptor in validation_manifest.get("outputs", {}).items():
        relative = Path(str(relative_name))
        path = validation / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or not isinstance(descriptor, dict)
            or _sha256(path) != descriptor.get("sha256")
            or path.stat().st_size != int(descriptor.get("bytes", -1))
        ):
            raise ValueError(f"V14 timestamp validation output invalid: {relative_name}")
        validation_files.add(relative)
    actual_producer = {
        path.relative_to(producer) for path in producer.rglob("*") if path.is_file()
    }
    actual_validation = {
        path.relative_to(validation) for path in validation.rglob("*") if path.is_file()
    }
    if actual_producer != producer_files or actual_validation != validation_files:
        raise ValueError("V14 timestamp packet file denominator changed")
    return accepted, tuple(sorted(producer_files)), tuple(sorted(validation_files))


def _copy_validated_official_timestamp_v14_extension(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: Path,
) -> _ValidatedOfficialTimestampExtension:
    producer = Path(producer_dir).resolve()
    validation = Path(validation_dir).resolve()
    accepted, producer_files, validation_files = (
        _validate_official_timestamp_v14_contract(producer, validation)
    )
    copied: list[Path] = []
    for source, target_root, files in (
        (producer, destination / "producer", producer_files),
        (validation, destination / "validation", validation_files),
    ):
        for relative in files:
            target = target_root / relative
            _atomic_bytes(target, (source / relative).read_bytes())
            copied.append(target)
    copied_accepted = pd.read_csv(
        destination / "producer" / "accepted_timestamp_recovery.csv", dtype=str
    ).fillna("")
    if not accepted.equals(copied_accepted):
        raise ValueError("V14 timestamp copied claims changed")
    lineage = {
        "producer_run_id": _json(producer / "manifest.json")["run_id"],
        "producer_manifest_sha256": _sha256(
            destination / "producer" / "manifest.json"
        ),
        "validation_run_id": _json(validation / "manifest.json")["run_id"],
        "validation_manifest_sha256": _sha256(
            destination / "validation" / "manifest.json"
        ),
        "target_events": 6,
        "source_responses": 6,
        "known_at_rule": "date_only_conservative_upper_bound_next_midnight_jst",
        "official_jpx_only": True,
        "independently_validated": True,
        "refinitiv_calls": False,
    }
    lineage_path = destination / "lineage.json"
    _atomic_json(lineage_path, lineage)
    copied.append(lineage_path)
    return _ValidatedOfficialTimestampExtension(
        accepted=copied_accepted,
        copied_paths=tuple(copied),
        lineage=lineage,
    )


def _apply_lifecycle_taxonomy_repairs(
    report_events: pd.DataFrame, repairs: pd.DataFrame
) -> pd.DataFrame:
    output = report_events.astype(str).fillna("").copy()
    if output["event_id"].duplicated().any():
        raise ValueError("report event IDs must be unique before taxonomy repair")
    event_index = output.set_index("event_id", drop=False).to_dict("index")
    for repair in repairs.to_dict("records"):
        event_id = str(repair["event_id"])
        event = event_index.get(event_id)
        if event is None or any(
            str(event[source_field]) != str(repair[repair_field])
            for source_field, repair_field in (
                ("source_record_sha256", "source_record_sha256"),
                ("report_month", "report_month"),
                ("channel", "channel"),
                ("event_date", "event_date"),
                ("raw_code", "raw_code"),
                ("normalized_local_code", "normalized_local_code"),
                ("instrument_class", "source_instrument_class"),
                ("source_sha256", "source_sha256"),
                ("source_page", "source_page"),
                ("source_row_ordinal", "source_row_ordinal"),
                ("source_record_text", "source_record_text"),
            )
        ):
            raise ValueError(
                f"lifecycle taxonomy repair source mismatch: {event_id}"
            )
        output.loc[
            output["event_id"].eq(event_id), "instrument_class"
        ] = str(repair["reclassified_instrument_class"])
    return output


def _copy_validated_tse_wayback_lifecycles(
    pairs: Iterable[tuple[Path, Path]],
    destination: Path,
) -> _ValidatedTseWaybackLifecycle:
    normalized = tuple(pairs)
    if not normalized:
        raise ValueError("at least one TSE Wayback lifecycle pair is required")
    packets: list[_ValidatedTseWaybackLifecycle] = []
    copied: list[Path] = []
    claimed: set[tuple[str, str]] = set()
    packet_lineage: list[dict[str, Any]] = []
    packet_root = destination / "packets"
    expected_directories: set[str] = set()
    for index, (producer, validation) in enumerate(normalized, start=1):
        packet_id = f"packet-{index:04d}"
        expected_directories.add(packet_id)
        packet = _copy_validated_tse_wayback_lifecycle(
            producer, validation, packet_root / packet_id
        )
        packet_claims = set(
            zip(packet.claims["report_month"], packet.claims["channel"], strict=False)
        )
        overlap = claimed & packet_claims
        if overlap:
            month, channel = sorted(overlap)[0]
            raise ValueError(
                "overlapping TSE Wayback lifecycle month/channel claim: "
                f"{month}:{channel}"
            )
        claimed.update(packet_claims)
        packets.append(packet)
        copied.extend(packet.copied_paths)
        packet_lineage.append(
            {
                "packet_id": packet_id,
                "packet_path": f"packets/{packet_id}",
                **packet.lineage,
            }
        )
    actual_directories = (
        {path.name for path in packet_root.iterdir() if path.is_dir()}
        if packet_root.is_dir()
        else set()
    )
    if actual_directories != expected_directories:
        raise ValueError("TSE Wayback lifecycle packet directory denominator changed")

    events = pd.concat([packet.events for packet in packets], ignore_index=True)
    claims = pd.concat([packet.claims for packet in packets], ignore_index=True)
    months = sorted(set(claims["report_month"]))
    lineage = {
        "schema_version": "2.0-multi-packet",
        "packet_count": len(packets),
        "packet_ids": sorted(expected_directories),
        "selected_months": months,
        "selected_month_count": len(months),
        "covered_claims": len(claims),
        "claim_count": len(claims),
        "source_rows": len(events),
        "source_event_count": len(events),
        "common_equity_rows": int(events["instrument_class"].eq("equity").sum()),
        "preserved_non_equity_rows": int(
            events["instrument_class"].ne("equity").sum()
        ),
        "absence_is_zero": False,
        "official_source_only": True,
        "wayback_transport_only": all(
            bool(packet.lineage.get("wayback_transport_only")) for packet in packets
        ),
        "official_copied_file_packet_count": sum(
            bool(packet.lineage.get("official_copied_file_transport"))
            for packet in packets
        ),
        "packets": packet_lineage,
    }
    lineage_path = destination / "lineage.json"
    _atomic_json(lineage_path, lineage)
    copied.append(lineage_path)
    return _ValidatedTseWaybackLifecycle(events, claims, tuple(copied), lineage)


def _copy_validated_tse_wayback_lifecycle(
    producer_dir: Path,
    validation_dir: Path,
    destination: Path,
) -> _ValidatedTseWaybackLifecycle:
    producer = producer_dir.resolve()
    validation = validation_dir.resolve()
    producer_manifest, validation_manifest, events, claims = (
        _validate_tse_wayback_lifecycle_contract(producer, validation)
    )
    copied: list[Path] = []
    copied.extend(_copy_sealed_tree(producer, destination / "producer"))
    copied.extend(_copy_sealed_tree(validation, destination / "validation"))

    copied_producer = destination / "producer"
    copied_validation = destination / "validation"
    _, _, copied_events, copied_claims = _validate_tse_wayback_lifecycle_contract(
        copied_producer, copied_validation
    )
    if not events.equals(copied_events) or not claims.equals(copied_claims):
        raise ValueError("TSE Wayback lifecycle packet copy changed exact rows")

    lineage = {
        "producer_run_id": producer_manifest["run_id"],
        "producer_manifest_sha256": _sha256(copied_producer / "manifest.json"),
        "validation_run_id": validation_manifest["run_id"],
        "validation_manifest_sha256": _sha256(
            copied_validation / "manifest.json"
        ),
        "selected_months": list(producer_manifest["selected_months"]),
        "selected_month_count": int(producer_manifest["selected_month_count"]),
        "covered_claims": int(producer_manifest["exact_nonempty_claim_count"]),
        "source_rows": int(producer_manifest["exact_event_count"]),
        "common_equity_rows": int(copied_events["instrument_class"].eq("equity").sum()),
        "preserved_non_equity_rows": int(
            copied_events["instrument_class"].ne("equity").sum()
        ),
        "absence_is_zero": False,
        "official_source_only": True,
        "wayback_transport_only": producer_manifest.get("preservation_transport")
        == "Internet Archive Wayback Machine",
        "official_copied_file_transport": producer_manifest.get(
            "preservation_transport"
        )
        == "Official JPX live PDF plus Internet Archive parent-index lineage",
    }
    lineage_path = destination / "lineage.json"
    _atomic_json(lineage_path, lineage)
    copied.append(lineage_path)
    return _ValidatedTseWaybackLifecycle(
        copied_events, copied_claims, tuple(copied), lineage
    )


def _validate_tse_wayback_lifecycle_contract(
    producer: Path, validation: Path
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    producer_manifest_path = producer / "manifest.json"
    validation_manifest_path = validation / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    validation_manifest = _json(validation_manifest_path)
    selected_months = tuple(
        str(month) for month in producer_manifest.get("selected_months", [])
    )
    if (
        not selected_months
        or len(set(selected_months)) != len(selected_months)
        or list(selected_months) != sorted(selected_months)
        or any(re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", month) is None for month in selected_months)
    ):
        raise ValueError("TSE Wayback lifecycle selected-month manifest is invalid")
    month_count = len(selected_months)
    claim_count = int(producer_manifest.get("exact_nonempty_claim_count", -1))
    event_count = int(producer_manifest.get("exact_event_count", -1))
    expected_claims = {
        (month, channel) for month in selected_months for channel in CHANNELS
    }
    copied_route = producer_manifest.get("route_modes") == [
        "official_jpx_copied_file_with_parent_index_wayback_lineage"
    ]
    producer_contract = (
        producer_manifest.get("artifact_type")
        == "bounded_tse_wayback_lifecycle_source_acquisition"
        and producer_manifest.get("status") == "bounded_acquisition_complete"
        and producer_manifest.get("version")
        == (
            "1.1-jpx-copied-file-lifecycle"
            if copied_route
            else "1.0-tse-wayback-lifecycle"
        )
        and producer_manifest.get("selected_months") == list(selected_months)
        and int(producer_manifest.get("selected_month_count", -1)) == month_count
        and int(producer_manifest.get("acquired_exact_pdf_month_count", -1))
        == month_count
        and claim_count == len(expected_claims)
        and event_count > 0
        and int(producer_manifest.get("failure_count", -1)) == 0
        and int(producer_manifest.get("zero_event_claims_published", -1)) == 0
        and producer_manifest.get("absence_is_zero") is False
        and producer_manifest.get("preservation_transport")
        == (
            "Official JPX live PDF plus Internet Archive parent-index lineage"
            if copied_route
            else "Internet Archive Wayback Machine"
        )
        and producer_manifest.get("original_source")
        == "Tokyo Stock Exchange / JPX official monthly PDF bytes"
        and producer_manifest.get("refinitiv_used") is False
        and producer_manifest.get("write_once_after_manifest") is True
    )
    validation_contract = (
        validation_manifest.get("artifact_type")
        == "independent_tse_wayback_lifecycle_validation"
        and validation_manifest.get("status")
        == "pass_bounded_exact_nonempty_evidence"
        and validation_manifest.get("version")
        == (
            "2.1-independent-jpx-copied-file-pdfplumber"
            if copied_route
            else TSE_WAYBACK_VALIDATOR_VERSION
        )
        and validation_manifest.get("producer_manifest_sha256")
        == _sha256(producer_manifest_path)
        and int(validation_manifest.get("selected_month_count", -1)) == month_count
        and int(validation_manifest.get("capture_valid_month_count", -1))
        == month_count
        and int(validation_manifest.get("independent_exact_event_count", -1))
        == event_count
        and int(validation_manifest.get("reconciled_nonempty_claim_count", -1))
        == claim_count
        and int(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("validation_issue_codes") == []
        and validation_manifest.get("absence_is_zero") is False
        and validation_manifest.get("missing_capture_fails_closed") is True
        and validation_manifest.get("parser_disagreement_fails_closed") is True
        and validation_manifest.get("refinitiv_used") is False
        and validation_manifest.get("write_once") is True
        and (
            not copied_route
            or validation_manifest.get(
                "live_official_pdf_response_independently_validated"
            )
            is True
        )
    )
    if not producer_contract or not validation_contract:
        raise ValueError(
            "TSE Wayback lifecycle producer-validation institutional contract failed"
        )
    _verify_sealed_tree(producer, "TSE Wayback lifecycle producer")
    _verify_sealed_tree(validation, "TSE Wayback lifecycle validation")

    events = pd.read_csv(producer / "exact_events.csv", dtype=str).fillna("")
    parquet_events = pd.read_parquet(producer / "exact_events.parquet").fillna("")
    parquet_events = parquet_events.astype(str).reindex(columns=events.columns)
    claims = pd.read_csv(
        producer / "exact_nonempty_claims.csv", dtype=str
    ).fillna("")
    inventory = pd.read_csv(producer / "source_inventory.csv", dtype=str).fillna("")
    transport = pd.read_csv(
        producer / "transport_inventory.csv", dtype=str
    ).fillna("")
    reconciliation = pd.read_csv(
        validation / "event_set_reconciliation.csv", dtype=str
    ).fillna("")
    month_validation = pd.read_csv(
        validation / "month_validation.csv", dtype=str
    ).fillna("")
    independent = pd.read_csv(
        validation / "independent_exact_events.csv", dtype=str
    ).fillna("")

    claim_keys = set(zip(claims["report_month"], claims["channel"], strict=False))
    event_keys = set(zip(events["report_month"], events["channel"], strict=False))
    grouped = events.groupby(["report_month", "channel"]).size().to_dict()
    claimed = {
        (row["report_month"], row["channel"]): int(row["event_count"])
        for row in claims.to_dict("records")
    }
    exact_rows = (
        len(events) == event_count
        and len(parquet_events) == event_count
        and events.astype(str).equals(parquet_events)
        and len(claims) == claim_count
        and claim_keys == event_keys == expected_claims
        and grouped == claimed
        and sum(claimed.values()) == event_count
        and events["event_id"].nunique() == event_count
        and events["source_record_sha256"].nunique() == event_count
        and events["known_at"].eq("").all()
        and events["ric"].eq("").all()
        and not _bool_series(events["admissible_to_shared_halt_history"]).any()
        and claims["claim_kind"].eq(
            "exact_nonempty_official_report_event_set"
        ).all()
        and claims["event_count"].astype(int).gt(0).all()
        and not _bool_series(claims["zero_event_asserted"]).any()
    )
    validation_rows = (
        len(month_validation) == month_count
        and set(month_validation["report_month"]) == set(selected_months)
        and _bool_series(month_validation["capture_valid"]).all()
        and len(reconciliation) == claim_count
        and set(zip(reconciliation["report_month"], reconciliation["channel"], strict=False))
        == expected_claims
        and _bool_series(reconciliation["validated"]).all()
        and not _bool_series(reconciliation["zero_inferred"]).any()
        and len(independent) == event_count
        and _report_event_counter(events) == _report_event_counter(independent)
    )
    if not exact_rows or not validation_rows:
        raise ValueError("TSE Wayback lifecycle exact manifest denominator failed")

    if len(inventory) != month_count or set(inventory["report_month"]) != set(
        selected_months
    ):
        raise ValueError("TSE Wayback lifecycle source inventory denominator failed")
    for row in inventory.to_dict("records"):
        month = row["report_month"]
        if copied_route:
            lineage_strength = row.get("archived_parent_lineage_strength")
            direct_parent_link = _truth(row.get("archived_parent_direct_pdf_link"))
            valid_lineage = (
                lineage_strength == "archived_parent_index_directly_links_official_pdf"
                and direct_parent_link
                and row.get("parent_link_url") == row.get("official_pdf_url")
            ) or (
                lineage_strength == "archived_parent_index_links_month_page_only"
                and not direct_parent_link
                and _official_tse_url(row.get("parent_link_url"))
            )
            valid_source = (
                row.get("route_mode")
                == "official_jpx_copied_file_with_parent_index_wayback_lineage"
                and (
                    _official_tse_url(row.get("official_monthly_page_url"))
                    or _official_jpx_lifecycle_pdf(
                        row.get("official_monthly_page_url")
                    )
                )
                and (
                    _official_tse_url(row.get("parent_index_cdx_original"))
                    or _official_jpx_lifecycle_parent_index(
                        row.get("parent_index_cdx_original")
                    )
                )
                and _official_jpx_lifecycle_pdf(row.get("official_pdf_url"))
                and row.get("parent_index_url")
                == row.get("parent_index_cdx_original")
                and row.get("parent_index_path") == f"raw/pages/{month}.html"
                and row.get("pdf_path") == f"raw/pdfs/{month}.pdf"
                and row.get("pdf_final_url") == row.get("official_pdf_url")
                and row.get("pdf_http_status") == "200"
                and row.get("pdf_content_type") == "application/pdf"
                and row.get("preservation_transport")
                == "official JPX live PDF plus Internet Archive parent-index lineage"
                and not _truth(row.get("registry_entry_guessed"))
                and not _truth(row.get("zero_inferred"))
                and valid_lineage
                and _wayback_transport_url(
                    row.get("parent_index_wayback_url"),
                    row.get("parent_index_cdx_timestamp"),
                    row.get("parent_index_cdx_original"),
                )
            )
        else:
            valid_source = (
                _official_tse_url(row["official_monthly_page_url"])
                and _official_tse_url(row["html_cdx_original"])
                and _official_tse_url(row["official_pdf_url"])
                and _official_tse_url(row["pdf_cdx_original"])
                and row["html_path"] == f"raw/pages/{month}.html"
                and row["pdf_path"] == f"raw/pdfs/{month}.pdf"
                and row["preservation_transport"] == "Internet Archive Wayback Machine"
                and not _truth(row["zero_inferred"])
                and _wayback_transport_url(
                    row["html_wayback_url"], row["html_cdx_timestamp"], row["html_cdx_original"]
                )
                and _wayback_transport_url(
                    row["pdf_wayback_url"], row["pdf_cdx_timestamp"], row["pdf_cdx_original"]
                )
            )
        if not valid_source:
            raise ValueError(f"TSE Wayback lifecycle official-source transport contract failed: {month}")
    if copied_route:
        inventory_by_month = inventory.set_index("report_month").to_dict("index")
        transport_ok = (
            events["source_url"].map(_official_jpx_lifecycle_pdf).all()
            and claims["official_source_url"].map(_official_jpx_lifecycle_pdf).all()
            and all(
                row["wayback_url"]
                == inventory_by_month[row["report_month"]]["parent_index_wayback_url"]
                and row["source_sha256"]
                == inventory_by_month[row["report_month"]]["pdf_sha256"]
                for row in claims.to_dict("records")
            )
            and len(transport) == month_count * 3
            and transport["http_status"].eq("200").all()
            and set(transport["request_kind"])
            == {"parent_index_cdx", "parent_index_capture", "official_pdf_live"}
        )
    else:
        transport_ok = (
            events["source_url"].map(_official_tse_url).all()
            and claims["official_source_url"].map(_official_tse_url).all()
            and all(
                _wayback_transport_url(
                    row["wayback_url"], row["archive_timestamp"], row["official_source_url"]
                )
                for row in claims.to_dict("records")
            )
            and len(transport) == month_count * 4
            and transport["request_url"].map(_is_wayback_url).all()
            and transport["final_url"].map(_is_wayback_url).all()
            and transport["http_status"].eq("200").all()
        )
    if not transport_ok:
        raise ValueError("TSE Wayback lifecycle uses non-official source or non-Wayback transport")
    return producer_manifest, validation_manifest, events, claims


def _verify_sealed_tree(root: Path, label: str) -> dict[str, str]:
    manifest = _json(root / "manifest.json")
    inventory_path = root / "artifact_hashes.json"
    if not inventory_path.is_file() or _sha256(inventory_path) != manifest.get(
        "artifact_hashes_sha256"
    ):
        raise ValueError(f"{label} hash inventory mismatch")
    declared = _json(inventory_path)
    for relative_name, digest in declared.items():
        relative = Path(str(relative_name))
        candidate = root / relative
        target = candidate.resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _within(target, root.resolve())
            or any(
                (root / Path(*relative.parts[:index])).is_symlink()
                for index in range(1, len(relative.parts) + 1)
            )
            or not target.is_file()
            or _sha256(target) != str(digest)
        ):
            raise ValueError(f"{label} unsafe path or hash mismatch: {relative_name}")
    expected = set(declared) | {"manifest.json", "artifact_hashes.json"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"{label} contains undeclared or missing files")
    return {str(key): str(value) for key, value in declared.items()}


def _copy_sealed_tree(source: Path, destination: Path) -> list[Path]:
    declared = _verify_sealed_tree(source, "sealed packet")
    copied: list[Path] = []
    for relative_name in sorted(declared):
        target = destination / relative_name
        _atomic_bytes(target, (source / relative_name).read_bytes())
        copied.append(target)
    for name in ("manifest.json", "artifact_hashes.json"):
        target = destination / name
        _atomic_bytes(target, (source / name).read_bytes())
        copied.append(target)
    return copied


def _merge_report_event_packets(
    packets: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    packet_claims: list[tuple[str, dict[tuple[str, str], pd.DataFrame]]] = []
    for packet_name, frame in packets:
        rows = frame.astype(str).fillna("").copy()
        if (
            rows.empty
            or rows["event_id"].duplicated().any()
            or rows["source_record_sha256"].duplicated().any()
        ):
            raise ValueError(f"report packet rows are empty or duplicated: {packet_name}")
        claims = {
            (month, channel): group.copy()
            for (month, channel), group in rows.groupby(
                ["report_month", "channel"], sort=False
            )
        }
        packet_claims.append((packet_name, claims))

    selected_owner: dict[tuple[str, str], str] = {}
    selected_rows: dict[tuple[str, str], pd.DataFrame] = {}
    for packet_name, claims in packet_claims:
        for claim, rows in claims.items():
            if claim in selected_rows and Counter(
                _report_merge_identity(row) for row in selected_rows[claim].to_dict("records")
            ) != Counter(_report_merge_identity(row) for row in rows.to_dict("records")):
                raise ValueError(
                    f"overlapping report claim exact row sets disagree: {claim[0]}:{claim[1]}"
                )
            selected_owner[claim] = packet_name
            selected_rows[claim] = rows

    lineage_rows: list[dict[str, str]] = []
    for packet_name, claims in packet_claims:
        for claim, rows in claims.items():
            selected = selected_rows[claim]
            selected_by_identity: dict[tuple[str, ...], list[dict[str, str]]] = {}
            for selected_row in selected.to_dict("records"):
                selected_by_identity.setdefault(
                    _report_merge_identity(selected_row), []
                ).append(selected_row)
            offsets: Counter[tuple[str, ...]] = Counter()
            for row in rows.to_dict("records"):
                identity = _report_merge_identity(row)
                target = selected_by_identity[identity][offsets[identity]]
                offsets[identity] += 1
                admitted = packet_name == selected_owner[claim]
                lineage_rows.append(
                    {
                        "source_packet": packet_name,
                        "source_record_sha256": row["source_record_sha256"],
                        "report_month": claim[0],
                        "channel": claim[1],
                        "source_event_id": row["event_id"],
                        "disposition": "admitted" if admitted else "duplicate_equivalent_superseded",
                        "selected_packet": selected_owner[claim],
                        "selected_source_record_sha256": target["source_record_sha256"],
                        "selected_report_event_id": target["event_id"],
                        "primary_event_id": "",
                    }
                )
    merged = pd.concat(
        [selected_rows[key] for key in sorted(selected_rows)], ignore_index=True
    )
    if merged["event_id"].duplicated().any() or merged["source_record_sha256"].duplicated().any():
        raise ValueError("merged report packets contain duplicate admitted events")
    lineage = pd.DataFrame(lineage_rows, columns=REPORT_EVENT_LINEAGE_COLUMNS)
    if lineage.duplicated(["source_packet", "source_record_sha256"]).any():
        raise ValueError("report packet lineage is not one-to-one")
    return merged, lineage


def _finish_report_event_lineage(
    lineage: pd.DataFrame, admission: pd.DataFrame
) -> pd.DataFrame:
    if lineage.empty:
        return lineage.reindex(columns=REPORT_EVENT_LINEAGE_COLUMNS)
    admitted = admission.set_index("source_record_sha256", drop=False).to_dict("index")
    output = lineage.copy()
    output["primary_event_id"] = output["selected_source_record_sha256"].map(
        lambda digest: admitted.get(digest, {}).get("primary_event_id", "")
    )
    return output.reindex(columns=REPORT_EVENT_LINEAGE_COLUMNS)


def _quarantine_invalid_report_intervals(
    events: pd.DataFrame, admission: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events, admission
    starts = pd.to_datetime(events["event_start_jst"], errors="coerce", utc=True)
    ends = pd.to_datetime(events["interval_end_jst"], errors="coerce", utc=True)
    invalid = events["interval_end_jst"].ne("") & (ends.isna() | (ends < starts))
    invalid_ids = set(events.loc[invalid, "event_id"])
    if not invalid_ids:
        return events, admission
    output_admission = admission.copy()
    affected = output_admission["primary_event_id"].isin(invalid_ids)
    output_admission.loc[affected, "normalization_status"] = "invalid_halt_interval"
    output_admission.loc[affected, "primary_event_id"] = ""
    return events.loc[~invalid].copy(), output_admission


def _report_merge_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(column, ""))
        for column in (
            "report_month",
            "channel",
            "event_type",
            "event_date",
            "raw_code",
            "normalized_local_code",
            "instrument_class",
            "issue_name",
            "halt_start_jst",
            "halt_end_jst",
        )
    )


def _report_event_counter(frame: pd.DataFrame) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(
            str(row.get(column, ""))
            for column in (
                "report_month",
                "channel",
                "event_date",
                "raw_code",
                "issue_name",
            )
        )
        for row in frame.to_dict("records")
    )


def _official_tse_url(value: Any) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == "www.tse.or.jp"
        and parsed.path.startswith("/english/market/data/geppo/")
        and not parsed.query
        and not parsed.fragment
    )


def _official_jpx_lifecycle_pdf(value: Any) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
        and (
            parsed.path.startswith("/files/tse/english/market/data/geppo/")
            or parsed.path.startswith(
                "/english/markets/statistics-equities/monthly/"
            )
        )
        and parsed.path.endswith(".pdf")
        and not parsed.query
        and not parsed.fragment
    )


def _official_jpx_lifecycle_parent_index(value: Any) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
        and parsed.path
        == "/english/markets/statistics-equities/monthly/00-archives-10.html"
        and not parsed.query
        and not parsed.fragment
    )


def _is_wayback_url(value: Any) -> bool:
    parsed = urllib.parse.urlsplit(str(value))
    return parsed.scheme == "https" and parsed.hostname == "web.archive.org"


def _wayback_transport_url(value: Any, timestamp: Any, original: Any) -> bool:
    prefix = f"https://web.archive.org/web/{timestamp}id_/"
    archived = str(value)
    return bool(
        (
            _official_tse_url(original)
            or _official_jpx_lifecycle_parent_index(original)
        )
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
        return (
            parsed.scheme,
            authority,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )

    return (
        _official_tse_url(left) or _official_jpx_lifecycle_parent_index(left)
    ) and normalized(left) == normalized(right)


def _validated_halt_mapping_recovery_counts(
    producer_manifest: dict[str, Any], validation_manifest: dict[str, Any]
) -> tuple[int, int, int]:
    producer_denominator = int(producer_manifest.get("denominator_rows", -1))
    producer_accepted = int(producer_manifest.get("accepted_rows", -1))
    producer_rejected = int(producer_manifest.get("rejected_rows", -1))
    validation_denominator = int(validation_manifest.get("denominator_rows", -1))
    validation_accepted = int(validation_manifest.get("accepted_rows", -1))
    validation_rejected = int(validation_manifest.get("rejected_rows", -1))
    validation_accepted_validated = int(
        validation_manifest.get("accepted_rows_validated", -1)
    )
    if not (
        producer_denominator > 0
        and producer_accepted > 0
        and producer_rejected >= 0
        and producer_accepted + producer_rejected == producer_denominator
        and validation_denominator == producer_denominator
        and validation_accepted == producer_accepted
        and validation_rejected == producer_rejected
        and validation_accepted_validated == producer_accepted
    ):
        raise ValueError("halt mapping recovery count contract failed")
    return producer_denominator, producer_accepted, producer_rejected


def _normalize_halt_mapping_recovery_pairs(
    producer_dir: str | Path | None,
    validation_dir: str | Path | None,
    pairs: Iterable[tuple[str | Path, str | Path]] | None,
) -> tuple[tuple[Path, Path], ...]:
    if (producer_dir is None) != (validation_dir is None):
        raise ValueError(
            "both halt mapping recovery and validation directories are required"
        )
    supplied = tuple(pairs or ())
    if supplied and producer_dir is not None:
        raise ValueError(
            "use either the legacy halt mapping recovery pair or multiple pairs, not both"
        )
    if producer_dir is not None:
        supplied = ((producer_dir, validation_dir),)
    normalized: list[tuple[Path, Path]] = []
    for index, pair in enumerate(supplied, start=1):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"halt mapping recovery pair {index} must contain producer and validation paths"
            )
        producer_value, validation_value = pair
        if producer_value is None or validation_value is None:
            raise ValueError(
                f"halt mapping recovery pair {index} must contain producer and validation paths"
            )
        normalized.append((Path(producer_value), Path(validation_value)))
    return tuple(normalized)


def _copy_validated_halt_mapping_recoveries(
    pairs: Iterable[tuple[Path, Path]], destination: Path
) -> tuple[pd.DataFrame, list[Path]]:
    normalized = tuple(pairs)
    if not normalized:
        raise ValueError("at least one halt mapping recovery pair is required")
    frames: list[pd.DataFrame] = []
    copied: list[Path] = []
    event_ids: set[str] = set()
    packet_rows: list[dict[str, Any]] = []
    for index, (producer, validation) in enumerate(normalized, start=1):
        packet_id = f"packet-{index:04d}"
        frame, packet_paths = _copy_validated_halt_mapping_recovery(
            producer, validation, destination / "packets" / packet_id
        )
        overlap = event_ids & set(frame["event_id"])
        if overlap:
            raise ValueError(
                "overlapping halt mapping recovery event_id: "
                f"{sorted(overlap)[0]}"
            )
        event_ids.update(frame["event_id"])
        frames.append(frame)
        copied.extend(packet_paths)
        packet_rows.append(
            {
                "packet_id": packet_id,
                "producer_run_id": _json(producer / "manifest.json").get("run_id"),
                "validation_run_id": _json(validation / "manifest.json").get("run_id"),
                "accepted_rows": len(frame),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    lineage_path = destination / "lineage.json"
    _atomic_json(
        lineage_path,
        {
            "schema_version": "2.0-multi-packet",
            "packet_count": len(packet_rows),
            "accepted_rows": len(combined),
            "event_ids_unique": not combined["event_id"].duplicated().any(),
            "packets": packet_rows,
        },
    )
    copied.append(lineage_path)
    return combined, copied


def _copy_validated_halt_mapping_recovery(
    producer_dir: Path,
    validation_dir: Path,
    destination: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    producer = producer_dir.resolve()
    validation = validation_dir.resolve()
    producer_manifest_path = producer / "manifest.json"
    validation_manifest_path = validation / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    validation_manifest = _json(validation_manifest_path)
    _, accepted_count, _ = _validated_halt_mapping_recovery_counts(
        producer_manifest, validation_manifest
    )
    if not (
        producer_manifest.get("status") == "complete_candidate"
        and producer_manifest.get("artifact_type")
        in {
            "japan_g4_halt_mapping_recovery_packet",
            "japan_g4_halt_mapping_recovery_v8_packet",
        }
        and producer_manifest.get("fail_closed") is True
        and producer_manifest.get("producer_checks_pass") is True
        and producer_manifest.get("latest_eligible_official_observation_only") is True
        and producer_manifest.get("successor_ric_allowed") is False
        and producer_manifest.get("inferred_code_t_allowed", False) is False
        and producer_manifest.get("window_extension") is False
        and validation_manifest.get("status") == "pass"
        and validation_manifest.get("artifact_type")
        in {
            "japan_g4_halt_mapping_recovery_independent_validation",
            "japan_g4_halt_mapping_recovery_v8_independent_validation",
        }
        and validation_manifest.get("independent_validation") is True
        and validation_manifest.get("component_validation_pass") is True
        and validation_manifest.get("producer_manifest_sha256")
        == _sha256(producer_manifest_path)
        and int(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("all_lineage_hashes_verified") is True
        and validation_manifest.get("known_at_recomputed") is True
        and validation_manifest.get("candidate_uniqueness_recomputed") is True
    ):
        raise ValueError("halt mapping recovery independent validation contract failed")

    copied: list[Path] = []
    for source, target in (
        (producer, destination / "producer"),
        (validation, destination / "validation"),
    ):
        manifest = _json(source / "manifest.json")
        inventory_path = source / "artifact_hashes.json"
        if _sha256(inventory_path) != manifest.get("artifact_hashes_sha256"):
            raise ValueError("halt mapping recovery hash inventory mismatch")
        inventory = _json(inventory_path)
        expected_files = set(inventory) | {"manifest.json", "artifact_hashes.json"}
        actual_files = {
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("halt mapping recovery packet contains undeclared files")
        for relative_name, digest in inventory.items():
            relative = Path(str(relative_name))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("halt mapping recovery inventory path is unsafe")
            source_path = source / relative
            if not source_path.is_file() or _sha256(source_path) != str(digest):
                raise ValueError(
                    f"halt mapping recovery output hash mismatch: {relative_name}"
                )
            target_path = target / relative
            _atomic_bytes(target_path, source_path.read_bytes())
            copied.append(target_path)
        for name in ("manifest.json", "artifact_hashes.json"):
            target_path = target / name
            _atomic_bytes(target_path, (source / name).read_bytes())
            copied.append(target_path)

    accepted = pd.read_csv(
        destination / "producer" / "accepted_mappings.csv", dtype=str
    ).fillna("")
    if (
        len(accepted) != accepted_count
        or accepted["event_id"].duplicated().any()
        or not accepted["decision"].eq("accepted").all()
        or not accepted["candidate_count"].eq("1").all()
    ):
        raise ValueError("halt mapping recovery accepted denominator is invalid")
    event_dates = pd.to_datetime(accepted["event_date"], utc=True)
    known_at = pd.to_datetime(accepted["source_known_at_utc"], utc=True)
    if known_at.isna().any() or (known_at > event_dates).any():
        raise ValueError("halt mapping recovery contains post-event source evidence")

    inputs = pd.read_csv(
        destination / "producer" / "input_inventory.csv", dtype=str
    ).fillna("")
    input_root = destination / "inputs"
    input_copies: dict[str, str] = {}
    for item in inputs.to_dict("records"):
        source_path = Path(item["path"])
        digest = item["expected_sha256"]
        if not source_path.is_file() or _sha256(source_path) != digest:
            raise ValueError(f"halt mapping recovery input changed: {item['input_id']}")
        suffix = source_path.suffix or ".bin"
        target = input_root / f"{item['input_id']}{suffix}"
        _atomic_bytes(target, source_path.read_bytes())
        copied.append(target)
        input_copies[item["input_id"]] = str(target.relative_to(destination))

    evidence_rows: list[dict[str, str]] = []
    for kind, path_field, hash_field in (
        ("raw_pdf", "source_pdf_path", "source_pdf_sha256"),
        ("normalized", "source_normalized_path", "source_normalized_sha256"),
    ):
        unique = accepted[[path_field, hash_field]].drop_duplicates()
        for item in unique.to_dict("records"):
            source_path = Path(item[path_field])
            digest = item[hash_field]
            if not source_path.is_file() or _sha256(source_path) != digest:
                raise ValueError(f"halt mapping recovery source changed: {source_path}")
            target = destination / "evidence" / kind / f"{digest}{source_path.suffix}"
            _atomic_bytes(target, source_path.read_bytes())
            copied.append(target)
            evidence_rows.append(
                {
                    "kind": kind,
                    "source_path": str(source_path),
                    "copied_path": str(target.relative_to(destination)),
                    "sha256": digest,
                }
            )
    evidence_inventory_path = destination / "evidence_inventory.csv"
    _atomic_csv(evidence_inventory_path, pd.DataFrame(evidence_rows))
    copied.append(evidence_inventory_path)
    lineage_path = destination / "lineage.json"
    _atomic_json(
        lineage_path,
        {
            "producer_run_id": producer_manifest.get("run_id"),
            "producer_manifest_sha256": _sha256(
                destination / "producer" / "manifest.json"
            ),
            "validation_run_id": validation_manifest.get("run_id"),
            "validation_manifest_sha256": _sha256(
                destination / "validation" / "manifest.json"
            ),
            "accepted_rows": len(accepted),
            "input_copies": input_copies,
            "evidence_files": len(evidence_rows),
        },
    )
    copied.append(lineage_path)
    return accepted, copied


def _apply_halt_mapping_recovery(
    events: pd.DataFrame, recovery: pd.DataFrame
) -> pd.DataFrame:
    if events.empty or recovery.empty:
        return events
    accepted = recovery.set_index("event_id", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    matched: set[str] = set()
    for event in events.to_dict("records"):
        item = accepted.get(event["event_id"])
        if item is not None:
            if (
                event["event_date"] != item["event_date"]
                or event["local_code"] != item["local_code"]
                or event["refinitiv_ric"]
                or event["mapping_status"] in PIT_MAPPING_STATUSES
                or int(item["candidate_count"]) != 1
                or not re.fullmatch(
                    rf"{re.escape(event['local_code'])}\.T(?:\^.*)?",
                    item["candidate_ric"],
                    flags=re.IGNORECASE,
                )
                or pd.Timestamp(item["source_known_at_utc"])
                > pd.Timestamp(event["event_date"], tz="UTC")
            ):
                raise ValueError(f"halt mapping recovery semantics invalid: {event['event_id']}")
            event["stable_listing_id"] = item["identity_stable_instrument_id"]
            event["refinitiv_ric"] = item["candidate_ric"]
            event["mapping_known_at"] = item["source_known_at_utc"]
            event["mapping_status"] = "pit_exact_official_monthly_quote_recovery"
            event["mapping_route"] = (
                "official_monthly_quote_pre_event_plus_unique_g1_identity"
            )
            event["mapping_source_sha256"] = item["source_pdf_sha256"]
            event["mapping_source_record_id"] = item["source_record_sha256"]
            matched.add(event["event_id"])
        rows.append(event)
    if matched != set(recovery["event_id"]):
        missing = sorted(set(recovery["event_id"]) - matched)
        raise ValueError(f"halt mapping recovery event denominator mismatch: {missing[:5]}")
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _build_coverage(
    *,
    start: date,
    end: date,
    catalog: pd.DataFrame,
    requests: pd.DataFrame,
    events: pd.DataFrame,
    report_admission: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = build_halt_acquisition_plan(start, end)
    complete_ids = set(requests.loc[requests["status"].eq("completed"), "source_id"])
    completed_sources = catalog.loc[catalog["source_id"].isin(complete_ids)].copy()
    request_index = requests.set_index("source_id").to_dict("index") if len(requests) else {}
    report = (
        report_admission.astype(str).fillna("")
        if report_admission is not None and len(report_admission)
        else pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
    )
    event_index = events.set_index("event_id", drop=False) if len(events) else pd.DataFrame()
    no_event_rows: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for item in plan.to_dict("records"):
        report_rows = report.loc[
            report["report_month"].eq(item["month"])
            & report["channel"].eq(item["channel"])
        ].copy()
        report_claim = bool(
            len(report_rows)
            and _bool_series(report_rows["claim_exact_set_validated"]).all()
        )
        research_rows = report_rows.loc[
            _bool_series(report_rows["in_research_scope"])
        ].copy()
        if report_claim:
            primary_ids = [
                value for value in research_rows["primary_event_id"].tolist() if value
            ]
            month_events = (
                event_index.loc[event_index.index.intersection(primary_ids)].copy()
                if len(primary_ids) and not event_index.empty
                else pd.DataFrame(columns=EVENT_COLUMNS)
            )
        else:
            month_events = events.loc[
                events["event_date"].str[:7].eq(item["month"])
                & events["event_type"].map(_event_channel).eq(item["channel"])
            ]
        eligible = completed_sources.loc[
            completed_sources["channel"].eq(item["channel"])
            & _bool_series(completed_sources["denominator_complete"])
            & completed_sources["coverage_start"].le(item["window_start"])
            & completed_sources["coverage_end"].ge(item["window_end"])
        ]
        source_covered = bool(not eligible.empty or report_claim)
        source_ids = sorted(eligible["source_id"].tolist())
        if report_claim:
            source_ids.append(f"jpx-monthly-report-{item['month']}-{item['channel']}")
        no_event = False
        if source_covered and month_events.empty:
            capable = eligible.loc[_bool_series(eligible["no_event_capable"])]
            no_event = not capable.empty
            for source in capable.to_dict("records"):
                request = request_index[source["source_id"]]
                evidence_key = f"{item['month']}|{item['channel']}|{source['source_id']}|{request['raw_sha256']}"
                no_event_rows.append(
                    {
                        "evidence_id": hashlib.sha256(evidence_key.encode("utf-8")).hexdigest()[:24],
                        "month": item["month"],
                        "channel": item["channel"],
                        "source_id": source["source_id"],
                        "source_url": source["source_url"],
                        "source_sha256": request["raw_sha256"],
                        "observed_at_utc": request["observed_at_utc"],
                        "coverage_start": source["coverage_start"],
                        "coverage_end": source["coverage_end"],
                        "assertion": "official complete source contains zero normalized events for month-channel",
                    }
                )
        if report_claim and research_rows.empty:
            source_hashes = sorted(set(report_rows["source_sha256"]))
            evidence_key = (
                f"{item['month']}|{item['channel']}|exact-report-equity-scope|"
                + "|".join(source_hashes)
            )
            no_event = True
            no_event_rows.append(
                {
                    "evidence_id": hashlib.sha256(evidence_key.encode("utf-8")).hexdigest()[:24],
                    "month": item["month"],
                    "channel": item["channel"],
                    "source_id": f"jpx-monthly-report-{item['month']}-{item['channel']}",
                    "source_url": "",
                    "source_sha256": "|".join(source_hashes),
                    "observed_at_utc": "",
                    "coverage_start": item["window_start"],
                    "coverage_end": item["window_end"],
                    "assertion": "independently validated exact report row set contains zero common-equity rows; publication timestamp unresolved",
                }
            )
        report_normalized = bool(
            not report_claim
            or research_rows.empty
            or research_rows["normalization_status"].eq("normalized").all()
        )
        if report_claim and research_rows.empty:
            pit_complete = False
        elif report_claim:
            pit_complete = bool(
                report_normalized
                and _bool_series(research_rows["timestamp_complete"]).all()
            )
        elif month_events.empty:
            pit_complete = bool(
                source_covered
                and no_event
                and eligible["pit_timestamp_status"].eq("exact_publication_timestamp").all()
            )
        else:
            pit_complete = bool(
                source_covered
                and month_events["known_at_status"].isin(PIT_SAFE_STATUSES).all()
            )
        if report_claim:
            mapping_complete = bool(
                research_rows.empty
                or (
                    report_normalized
                    and _bool_series(research_rows["mapping_complete"]).all()
                )
            )
        else:
            mapping_complete = bool(
                month_events.empty
                or month_events["mapping_status"].isin(PIT_MAPPING_STATUSES).all()
            )
        blockers: list[str] = []
        if not source_covered:
            blockers.append("source_denominator_uncovered")
        if report_claim and not report_normalized:
            blockers.append("report_event_normalization_incomplete")
        if source_covered and month_events.empty and not no_event:
            blockers.append("no_event_evidence_missing")
        if report_claim and research_rows.empty:
            blockers.append("report_zero_equity_publication_timestamp_missing")
        if not pit_complete:
            blockers.append("pit_timestamp_incomplete")
        if not mapping_complete:
            blockers.append("listing_ric_mapping_incomplete")
        complete = (
            source_covered
            and report_normalized
            and (not month_events.empty or no_event)
            and pit_complete
            and mapping_complete
        )
        if complete:
            status = "complete"
        elif not source_covered and not month_events.empty:
            status = "event_evidence_only"
        elif source_covered:
            status = "source_complete_pit_or_mapping_incomplete"
        else:
            status = "uncovered"
        item.update(
            {
                "source_covered": source_covered,
                "event_rows": len(month_events),
                "no_event_evidence": no_event,
                "pit_timestamp_complete": pit_complete,
                "listing_ric_mapping_complete": mapping_complete,
                "coverage_status": status,
                "source_ids": "|".join(source_ids),
                "blockers": "|".join(blockers),
                "next_action": _next_action(item["channel"], status),
            }
        )
        output.append(item)
    return pd.DataFrame(output, columns=COVERAGE_COLUMNS), pd.DataFrame(no_event_rows, columns=NO_EVENT_COLUMNS)


def _pit_audit(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.reindex(columns=PIT_AUDIT_COLUMNS[:-1]).copy()
    frame["historical_pit_usable"] = events["known_at_status"].isin(PIT_SAFE_STATUSES).to_numpy()
    return frame.reindex(columns=PIT_AUDIT_COLUMNS)


def _build_completion_outputs(
    coverage: pd.DataFrame,
    events: pd.DataFrame,
    *,
    report_admission: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for item in coverage.loc[~_bool_series(coverage["source_covered"])].to_dict("records"):
        channel = item["channel"]
        rows.append(
            _gap_row(
                gap_type="source_denominator_uncovered",
                scope=channel,
                month=item["month"],
                current_evidence=(
                    "event_rows_without_complete_denominator"
                    if _int(item["event_rows"])
                    else "no_admitted_complete_source"
                ),
                required_evidence=(
                    "official immutable halt snapshot/export explicitly covering the full month and preserving zero-event responses"
                    if channel == HALT_CHANNEL
                    else "official JPX lifecycle/monthly-change evidence covering the full month"
                ),
                next_action=(
                    "acquire licensed JPX historical halt export or contemporaneous official snapshots; never infer zero events"
                    if channel == HALT_CHANNEL
                    else "parse/admit official pre-2017 monthly lifecycle archives with exact source hashes"
                ),
            )
        )
    for event in events.loc[~events["known_at_status"].isin(PIT_SAFE_STATUSES)].to_dict("records"):
        rows.append(
            _gap_row(
                gap_type="historical_pit_timestamp_unresolved",
                scope=event["event_type"],
                month=event["event_date"][:7],
                event_id=event["event_id"],
                local_code=event["local_code"],
                current_evidence=event["known_at_status"],
                required_evidence="official publication/decision timestamp or a later admitted contemporaneous observation proving the event was known",
                next_action="acquire timestamped JPX notice metadata; source-page observation after the target period is not historical PIT evidence",
            )
        )
    for event in events.loc[~events["mapping_status"].isin(PIT_MAPPING_STATUSES)].to_dict("records"):
        resolved = bool(event["refinitiv_ric"])
        rows.append(
            _gap_row(
                gap_type=(
                    "ric_mapping_not_historical_pit" if resolved else "ric_mapping_unresolved"
                ),
                scope=event["event_type"],
                month=event["event_date"][:7],
                event_id=event["event_id"],
                local_code=event["local_code"],
                current_evidence=event["mapping_status"],
                required_evidence=(
                    "pre-event admitted code-to-RIC identity observation"
                    if resolved
                    else "admitted security-master row uniquely linking local code and Tokyo RIC"
                ),
                next_action=(
                    "admit a contemporaneous G1 identifier observation"
                    if resolved
                    else "target the missing code in a future independently validated security-master acquisition"
                ),
            )
        )
    report = (
        report_admission.astype(str).fillna("")
        if report_admission is not None and len(report_admission)
        else pd.DataFrame(columns=REPORT_ADMISSION_COLUMNS)
    )
    unresolved_report = report.loc[
        _bool_series(report["in_research_scope"])
        & ~report["normalization_status"].eq("normalized")
    ]
    for item in unresolved_report.to_dict("records"):
        rows.append(
            _gap_row(
                gap_type="report_event_normalization_unresolved",
                scope=item["channel"],
                month=item["report_month"],
                event_id=item["report_event_id"],
                local_code=item["local_code"],
                current_evidence=item["normalization_status"],
                required_evidence="corrected official JPX source row or independently validated date/time/code repair",
                next_action="quarantine the malformed report row until exact source-backed repair is available",
            )
        )
    coverage_blockers = coverage.get(
        "blockers", pd.Series("", index=coverage.index, dtype=str)
    )
    for item in coverage.loc[
        coverage_blockers.str.contains(
            "report_zero_equity_publication_timestamp_missing", regex=False
        )
    ].to_dict("records"):
        rows.append(
            _gap_row(
                gap_type="report_zero_equity_publication_timestamp_unresolved",
                scope=item["channel"],
                month=item["month"],
                current_evidence="exact row set contains zero common-equity rows but report publication time is absent",
                required_evidence="official report publication timestamp",
                next_action="recover and independently validate JPX monthly report publication metadata",
            )
        )
    gaps = pd.DataFrame(rows, columns=GAP_COLUMNS)
    completion_rows: list[dict[str, Any]] = []
    if not gaps.empty:
        for gap_type, group in gaps.groupby("gap_type", sort=True):
            completion_rows.append(
                {
                    "plan_id": f"complete-{gap_type}",
                    "gap_type": gap_type,
                    "gap_rows": len(group),
                    "affected_months": group["month"].replace("", pd.NA).nunique(),
                    "affected_events": group["event_id"].replace("", pd.NA).nunique(),
                    "irreducible_rows": int(_bool_series(group["irreducible_with_local_sources"]).sum()),
                    "status": "blocked_external_evidence_required",
                    "required_evidence": " | ".join(sorted(set(group["required_evidence"]))),
                    "next_action": " | ".join(sorted(set(group["next_action"]))),
                }
            )
    return gaps, pd.DataFrame(completion_rows, columns=COMPLETION_COLUMNS)


def _gap_row(
    *,
    gap_type: str,
    scope: str,
    month: str,
    current_evidence: str,
    required_evidence: str,
    next_action: str,
    event_id: str = "",
    local_code: str = "",
) -> dict[str, Any]:
    key = "|".join((gap_type, scope, month, event_id, local_code))
    return {
        "gap_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:24],
        "gap_type": gap_type,
        "scope": scope,
        "month": month,
        "event_id": event_id,
        "local_code": local_code,
        "current_evidence": current_evidence,
        "required_evidence": required_evidence,
        "irreducible_with_local_sources": True,
        "next_action": next_action,
    }


def _validate_events(events: pd.DataFrame) -> None:
    if tuple(events.columns) != EVENT_COLUMNS:
        raise ValueError("event schema mismatch")
    if events.empty:
        return
    for column in ("event_id", "event_type", "exchange", "local_code", "event_date", "event_start_jst", "source_id", "source_sha256", "source_record_id"):
        if events[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"events contain blank {column}")
    if events["event_id"].duplicated().any():
        raise ValueError("events contain duplicate event_id")
    starts = pd.to_datetime(events["event_start_jst"], errors="coerce", utc=True)
    if starts.isna().any():
        raise ValueError("events contain invalid event_start_jst")
    ends = pd.to_datetime(events["interval_end_jst"], errors="coerce", utc=True)
    present = events["interval_end_jst"].astype(str).str.strip().ne("")
    if ends[present].isna().any() or (ends[present] < starts[present]).any():
        raise ValueError("events contain invalid interval end")


def _jst_timestamp(day: str, value: Any) -> str:
    clean = str(value).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", clean)
    if not match:
        return ""
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return ""
    return datetime.combine(date.fromisoformat(day), time(hour, minute), JST).isoformat()


def _event_channel(value: str) -> str:
    return HALT_CHANNEL if value == "trading_halt" else DELISTING_CHANNEL


def _next_action(channel: str, status: str) -> str:
    if status == "complete":
        return "none"
    if channel == HALT_CHANNEL:
        return "acquire an official complete historical halt export or immutable daily JPX snapshots with explicit retention coverage and publication timestamps"
    return "parse the official monthly changes archive or acquire a JPX lifecycle export with decision/publication timestamps, then resolve PIT listing-to-RIC mapping"


def _summary(coverage: pd.DataFrame, events: pd.DataFrame, out_scope: pd.DataFrame, inventory: pd.DataFrame) -> str:
    return (
        "# Japan G4 Halt/Suspension Acquisition\n\n"
        f"- Month-channel denominator: {len(coverage)}\n"
        f"- Complete month-channels: {int(coverage['coverage_status'].eq('complete').sum())}\n"
        f"- In-scope normalized events: {len(events)}\n"
        f"- Events with resolved RIC: {int(events['refinitiv_ric'].ne('').sum())}\n"
        f"- Events with historical PIT timestamp: {int(events['known_at_status'].isin(PIT_SAFE_STATUSES).sum())}\n"
        f"- Preserved out-of-scope source events: {len(out_scope)}\n"
        f"- Locally inventoried official archive files: {len(inventory)}\n\n"
        "Unknown coverage is retained as unknown. Event absence is admitted only when a hashed source explicitly covers the full month-channel and supports no-event evidence.\n"
    )


def _fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-japan-g4-halts/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(MAX_RESPONSE_BYTES + 1)


def _load_requests(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).reindex(columns=REQUEST_COLUMNS).fillna("") if path.is_file() else pd.DataFrame(columns=REQUEST_COLUMNS)


def _invalidate_stale_requests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        path = Path(row["raw_path"]) if row["raw_path"] else Path("/")
        if row["status"] == "completed" and (not path.is_file() or _sha256(path) != row["raw_sha256"]):
            row["status"] = "stale"
            row["error_type"] = "HashMismatch"
            row["error"] = "checkpoint raw source is missing or changed"
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUEST_COLUMNS)


def _acquire_lock(root: Path):
    handle = (root / ".producer.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"producer run is already active: {root}") from exc
    return handle


def _date(value: str | date, label: str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO date") from exc


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return timezone-aware datetime")
    return value.astimezone(UTC).isoformat()


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
