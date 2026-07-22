from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_lab.data.usa_index_constituents_pit import (
    DATASET_ID,
    PIT_CONSTITUENT_COLUMNS,
    USA_INDEX_SCOPE_PROFILE_FULL,
    resolve_usa_index_scope_profile,
    usa_index_catalog_for_scope_profile,
)


LSEG_CURRENT_CHAIN_FIELD = "TR.PriceClose"
LSEG_INITIAL_SNAPSHOT_FIELD = LSEG_CURRENT_CHAIN_FIELD
LSEG_JOINER_LEAVER_FIELDS = (
    "TR.IndexJLConstituentChangeDate",
    "TR.IndexJLConstituentRIC",
    "TR.IndexJLConstituentName",
    "TR.IndexJLConstituentituentChange",
)
LSEG_DATED_SNAPSHOT_PROBE_FIELD = "TR.IndexConstituentRIC"
LSEG_SUPPORTED_INDEX_RICS = (".SPX", ".NDX", ".DJI")
LSEG_RUSSELL_INDEX_RICS = (".RUI", ".RUT", ".RUA")
LSEG_VERIFIED_2020_EVENT_COUNTS = {".SPX": 34, ".NDX": 25, ".DJI": 6}
LSEG_IDENTITY_RESOLUTION_FIELDS = ("TR.RIC", "TR.CommonName", "TR.SEDOL", "TR.PrimaryRIC")
LSEG_IDENTITY_RESOLUTION_CHUNK_SIZE = 100
LSEG_MAX_IDENTITY_RESOLUTION_REQUESTS = 10
LSEG_RESET_SENTINEL_MIN_ANCHOR_MEMBERS = 20
LSEG_RESET_SENTINEL_MIN_ANCHOR_FRACTION = 0.80

REQUEST_MANIFEST_COLUMNS = (
    "source_request_id",
    "index_id",
    "index_ric",
    "request_kind",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "response_rows",
    "normalized_rows",
    "raw_response_path",
    "raw_response_sha256",
    "normalized_path",
    "error_type",
    "error",
)

IDENTITY_RESOLUTION_MANIFEST_COLUMNS = (
    "source_request_id",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "requested_identifiers",
    "response_rows",
    "raw_response_path",
    "raw_response_sha256",
    "normalized_path",
    "normalized_sha256",
    "error_type",
    "error",
)

IDENTITY_MAP_COLUMNS = (
    "raw_identifier",
    "resolved_ric",
    "common_name",
    "sedol",
    "primary_ric",
    "resolution_status",
    "source_request_id",
    "observed_at_utc",
    "raw_response_sha256",
)

QUARANTINE_AUDIT_COLUMNS = (
    "quarantine_reason",
    "quarantine_rule",
    "batch_member_count",
    "current_anchor_member_count",
    "anchor_overlap_count",
    "anchor_overlap_fraction",
)

RECONSTRUCTION_ISSUE_COLUMNS = (
    "index_id",
    "member_ric",
    "effective_date",
    "source_record_id",
    "issue_code",
    "detail",
)

INTERVAL_COLUMNS = (
    "index_id",
    "index_ric",
    "member_ric",
    "member_name",
    "valid_from",
    "valid_to",
    "known_at_utc",
    "opening_source_record_id",
    "closing_source_record_id",
    "current_anchor_request_id",
    "current_anchor_chain",
    "current_anchor_raw_payload_sha256",
)


@dataclass(frozen=True)
class LSEGIndexReconstruction:
    intervals: pd.DataFrame
    issues: pd.DataFrame


@dataclass(frozen=True)
class LSEGUSAIndexConstituentArtifacts:
    run_id: str
    artifact_dir: Path
    request_plan_path: Path
    request_manifest_path: Path
    coverage_path: Path
    pit_rows_path: Path | None
    intervals_path: Path | None
    reconstruction_issues_path: Path
    checkpoint_manifest_path: Path
    identity_resolution_manifest_path: Path
    identity_map_path: Path
    quarantine_path: Path
    manifest_path: Path
    request_plan: pd.DataFrame
    request_manifest: pd.DataFrame
    coverage: pd.DataFrame
    pit_rows: pd.DataFrame
    identity_resolution_manifest: pd.DataFrame
    identity_map: pd.DataFrame
    quarantined_rows: pd.DataFrame
    reconstruction: LSEGIndexReconstruction


def build_lseg_usa_index_constituent_plan(
    start_date: str,
    end_date: str,
    index_catalog: pd.DataFrame | None = None,
    chunk_years: int = 1,
    anchor_date: str | None = None,
    scope_profile: str = USA_INDEX_SCOPE_PROFILE_FULL,
) -> pd.DataFrame:
    """Build a current-chain anchor plus official joiner/leaver request sequence."""

    start = _date(start_date, "start_date")
    end = _date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    if chunk_years <= 0:
        raise ValueError("chunk_years must be positive")
    anchor = _date(anchor_date, "anchor_date") if anchor_date is not None else max(end, pd.Timestamp.now(tz="UTC").tz_localize(None).normalize())
    if anchor < end:
        raise ValueError("anchor_date must be on or after end_date")

    profile = resolve_usa_index_scope_profile(scope_profile)
    catalog = index_catalog.copy() if index_catalog is not None else usa_index_catalog_for_scope_profile(profile.name)
    required = {"index_id", "index_family", "index_name", "candidate_index_ric"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"index catalog missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for item in catalog.itertuples(index=False):
        ric = str(item.candidate_index_ric)
        entitled = ric in LSEG_SUPPORTED_INDEX_RICS
        entitlement_status = (
            "validated_current_chain_and_joiner_leaver_available"
            if entitled
            else "missing_constituent_entitlement"
            if ric in LSEG_RUSSELL_INDEX_RICS
            else "unvalidated_index_ric"
        )
        chain = f"0#{ric}"
        rows.append(
            _plan_row(
                request_id=f"usa-pit-index-{item.index_id}-snapshot-{anchor.strftime('%Y%m%d')}",
                item=item,
                request_kind="current_snapshot",
                universe=chain,
                fields=(LSEG_CURRENT_CHAIN_FIELD,),
                parameters={},
                request_start=anchor,
                request_end=anchor,
                entitled=entitled,
                entitlement_status=entitlement_status,
            )
        )
        for chunk_start, chunk_end in _year_chunks(start, anchor, chunk_years=chunk_years):
            rows.append(
                _plan_row(
                    request_id=(
                        f"usa-pit-index-{item.index_id}-jl-"
                        f"{chunk_start.strftime('%Y%m%d')}-{chunk_end.strftime('%Y%m%d')}"
                    ),
                    item=item,
                    request_kind="joiner_leaver",
                    universe=ric,
                    fields=LSEG_JOINER_LEAVER_FIELDS,
                    parameters={
                        "SDATE": chunk_start.date().isoformat(),
                        "EDATE": chunk_end.date().isoformat(),
                        "IC": "B",
                    },
                    request_start=chunk_start,
                    request_end=chunk_end,
                    entitled=entitled,
                    entitlement_status=entitlement_status,
                )
            )
    return pd.DataFrame(rows)


def validate_lseg_joiner_leaver_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized JL events before they are allowed to alter membership."""

    issues: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(columns=RECONSTRUCTION_ISSUE_COLUMNS)

    work = events.reset_index(drop=True)
    effective = pd.to_datetime(work.get("effective_date"), errors="coerce")
    known_at = pd.to_datetime(work.get("known_at_utc"), errors="coerce", utc=True)
    for row_number, row in work.iterrows():
        base = {
            "index_id": _text(row.get("index_id")),
            "member_ric": _text(row.get("member_ric")),
            "effective_date": _date_text(effective.loc[row_number]),
            "source_record_id": _text(row.get("source_record_id")),
        }
        if not base["member_ric"]:
            issues.append({**base, "issue_code": "missing_member_ric", "detail": "JL event has no constituent RIC"})
        if pd.isna(effective.loc[row_number]):
            issues.append({**base, "issue_code": "missing_effective_date", "detail": "JL event has no parseable change date"})
        if _text(row.get("membership_action")) not in {"added", "removed"}:
            issues.append({**base, "issue_code": "invalid_change_action", "detail": "JL change is not a joiner or leaver"})
        if pd.isna(known_at.loc[row_number]):
            issues.append({**base, "issue_code": "missing_known_at", "detail": "JL event has no observable timestamp"})

    duplicate_key = ["index_id", "member_ric", "effective_date", "membership_action"]
    duplicates = work.duplicated(duplicate_key, keep=False)
    for row_number in work.index[duplicates]:
        row = work.loc[row_number]
        issues.append(
            {
                "index_id": _text(row.get("index_id")),
                "member_ric": _text(row.get("member_ric")),
                "effective_date": _date_text(effective.loc[row_number]),
                "source_record_id": _text(row.get("source_record_id")),
                "issue_code": "duplicate_joiner_leaver_event",
                "detail": "duplicate index/member/date/action event",
            }
        )

    return pd.DataFrame(issues, columns=RECONSTRUCTION_ISSUE_COLUMNS).drop_duplicates(ignore_index=True)


def reconstruct_lseg_constituent_intervals(
    pit_rows: pd.DataFrame,
    start_date: str | None,
    end_date: str,
    checkpoint_dir: str | Path | None = None,
) -> LSEGIndexReconstruction:
    """Reverse from a current chain, then build leakage-safe event-time intervals."""

    start = _date(start_date, "start_date") if start_date is not None else None
    end = _date(end_date, "end_date")
    if pit_rows.empty:
        return LSEGIndexReconstruction(
            intervals=pd.DataFrame(columns=INTERVAL_COLUMNS),
            issues=pd.DataFrame(columns=RECONSTRUCTION_ISSUE_COLUMNS),
        )

    rows = pit_rows.copy()
    snapshots = rows[rows["membership_action"].eq("snapshot_member")].copy()
    changes = _neutralize_same_day_round_trips(
        rows[rows["membership_action"].isin(["added", "removed"])].copy()
    )
    issue_frames = [validate_lseg_joiner_leaver_events(changes)]
    interval_rows: list[dict[str, Any]] = []

    for index_id in sorted(rows["index_id"].dropna().astype(str).unique()):
        anchor = snapshots[snapshots["index_id"].astype(str).eq(index_id)].copy()
        events = changes[changes["index_id"].astype(str).eq(index_id)].copy()
        if anchor.empty:
            issue_frames.append(
                pd.DataFrame(
                    [{
                        "index_id": index_id,
                        "member_ric": "",
                        "effective_date": "",
                        "source_record_id": "",
                        "issue_code": "missing_current_anchor",
                        "detail": "reconstruction requires a non-empty observed current-chain anchor",
                    }],
                    columns=RECONSTRUCTION_ISSUE_COLUMNS,
                )
            )
            continue

        anchor_dates = pd.to_datetime(anchor["membership_as_of_date"], errors="coerce").dropna().dt.normalize().unique()
        if len(anchor_dates) != 1:
            issue_frames.append(
                pd.DataFrame(
                    [{
                        "index_id": index_id,
                        "member_ric": "",
                        "effective_date": "",
                        "source_record_id": "",
                        "issue_code": "ambiguous_current_anchor",
                        "detail": "anchor rows must share exactly one snapshot date",
                    }],
                    columns=RECONSTRUCTION_ISSUE_COLUMNS,
                )
            )
            continue

        anchor_date = pd.Timestamp(anchor_dates[0])
        reconstruction_start = start if start is not None else _inferred_start(events, anchor_date)
        state: dict[str, dict[str, Any]] = {}
        anchor_chain = f"0#{_text(anchor.iloc[0].get('index_ric'))}"
        anchor_request_id = _text(anchor.iloc[0].get("source_request_id"))
        anchor_hash = _text(anchor.iloc[0].get("raw_payload_sha256"))
        for _, row in anchor.iterrows():
            member = _text(row.get("member_ric"))
            if not member or member in state:
                issue_frames.append(
                    pd.DataFrame(
                        [{
                            "index_id": index_id,
                            "member_ric": member,
                            "effective_date": anchor_date.date().isoformat(),
                            "source_record_id": _text(row.get("source_record_id")),
                            "issue_code": "duplicate_current_anchor_member",
                            "detail": "current-chain anchor contains a missing or duplicate member RIC",
                        }],
                        columns=RECONSTRUCTION_ISSUE_COLUMNS,
                    )
                )
                continue
            state[member] = {
                "member_ric": member,
                "member_name": _text(row.get("member_name")),
                "known_at_utc": _text(row.get("known_at_utc")),
                "source_record_id": _text(row.get("source_record_id")),
            }

        if not events.empty:
            events["_effective"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
            events["_known_at"] = pd.to_datetime(events["known_at_utc"], errors="coerce", utc=True)
            known_dates = events["_known_at"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
            events["_application_date"] = pd.concat([events["_effective"], known_dates], axis=1).max(axis=1)
            events["_action_order"] = events["membership_action"].map({"added": 0, "removed": 1}).fillna(9)
        else:
            events["_effective"] = pd.Series(dtype="datetime64[ns]")
            events["_known_at"] = pd.Series(dtype="datetime64[ns, UTC]")
            events["_application_date"] = pd.Series(dtype="datetime64[ns]")
            events["_action_order"] = pd.Series(dtype="float64")

        reverse_events = events.loc[
            events.get("_application_date", pd.Series(dtype="datetime64[ns]")).gt(reconstruction_start)
            & events.get("_application_date", pd.Series(dtype="datetime64[ns]")).le(anchor_date)
        ].sort_values(
            ["_application_date", "_action_order", "member_ric", "source_record_id"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        event_fingerprint = _events_fingerprint(reverse_events)
        checkpoint_root = Path(checkpoint_dir) / index_id if checkpoint_dir is not None else None
        restored = _restore_checkpoint(checkpoint_root, event_fingerprint, anchor_hash)
        processed = 0
        if restored is not None:
            state, processed = restored
        if restored is None:
            _persist_checkpoint(
                checkpoint_root,
                index_id,
                state,
                anchor_date,
                0,
                anchor_request_id,
                anchor_chain,
                anchor_hash,
                event_fingerprint,
            )
        remaining_reverse_events = reverse_events.iloc[processed:]
        for application_date, date_events in remaining_reverse_events.groupby("_application_date", sort=False):
            for _, row in date_events.iterrows():
                member = _text(row.get("member_ric"))
                action = _text(row.get("membership_action"))
                if not member or action not in {"added", "removed"}:
                    continue
                if action == "added":
                    if member not in state:
                        issue_frames.append(
                            _state_issue(row, index_id, "reverse_joiner_not_in_state", "reverse Joiner cannot remove an absent member")
                        )
                        continue
                    state.pop(member)
                elif member in state:
                    issue_frames.append(
                        _state_issue(row, index_id, "reverse_leaver_already_in_state", "reverse Leaver cannot add an active member")
                    )
                else:
                    state[member] = {
                        "member_ric": member,
                        "member_name": _text(row.get("member_name")),
                        "known_at_utc": _text(row.get("known_at_utc")),
                        "source_record_id": _text(row.get("source_record_id")),
                    }
            processed += len(date_events)
            _persist_checkpoint(
                checkpoint_root,
                index_id,
                state,
                pd.Timestamp(application_date),
                processed,
                anchor_request_id,
                anchor_chain,
                anchor_hash,
                event_fingerprint,
            )

        start_state = {member: dict(metadata) for member, metadata in state.items()}
        active: dict[str, dict[str, Any]] = {
            member: _interval_from_state(
                index_id=index_id,
                index_ric=_text(anchor.iloc[0].get("index_ric")),
                metadata=metadata,
                valid_from=reconstruction_start,
                anchor_request_id=anchor_request_id,
                anchor_chain=anchor_chain,
                anchor_hash=anchor_hash,
            )
            for member, metadata in start_state.items()
        }
        forward_events = events.loc[
            events.get("_application_date", pd.Series(dtype="datetime64[ns]")).gt(reconstruction_start)
            & events.get("_application_date", pd.Series(dtype="datetime64[ns]")).le(end)
        ].sort_values(
            ["_application_date", "_action_order", "member_ric", "source_record_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        for _, row in forward_events.iterrows():
            member = _text(row.get("member_ric"))
            action = _text(row.get("membership_action"))
            application_date = pd.Timestamp(row.get("_application_date")).normalize()
            if action == "removed":
                if member not in active:
                    continue
                closed = active.pop(member)
                closed["valid_to"] = (application_date - pd.Timedelta(days=1)).date().isoformat()
                closed["closing_source_record_id"] = _text(row.get("source_record_id"))
                interval_rows.append(closed)
            elif action == "added" and member not in active:
                active[member] = _interval_from_event(
                    row,
                    index_id,
                    application_date,
                    anchor_request_id,
                    anchor_chain,
                    anchor_hash,
                )
        for open_interval in active.values():
            open_interval["valid_to"] = end.date().isoformat()
            interval_rows.append(open_interval)

    issues = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame(columns=RECONSTRUCTION_ISSUE_COLUMNS)
    issues = issues.drop_duplicates(ignore_index=True)
    intervals = pd.DataFrame(interval_rows, columns=INTERVAL_COLUMNS)
    if not intervals.empty:
        intervals = intervals.sort_values(["index_id", "member_ric", "valid_from"], kind="stable").reset_index(drop=True)
    return LSEGIndexReconstruction(intervals=intervals, issues=issues)


def _neutralize_same_day_round_trips(events: pd.DataFrame) -> pd.DataFrame:
    """Cancel RICs reported as both joiner and leaver on the same effective date."""

    if events.empty:
        return events.copy()
    work = events.copy()
    key = ["index_id", "member_ric", "effective_date"]
    action_count = work.groupby(key, dropna=False)["membership_action"].transform(
        lambda values: values.dropna().astype(str).nunique()
    )
    return work.loc[action_count.lt(2)].copy().reset_index(drop=True)


def lseg_constituents_as_of(
    intervals: pd.DataFrame,
    as_of_date: str,
    known_at_utc: str | None = None,
) -> pd.DataFrame:
    """Select membership valid on a date and observable by the supplied knowledge cutoff."""

    as_of = _date(as_of_date, "as_of_date")
    cutoff = pd.Timestamp(known_at_utc) if known_at_utc is not None else as_of.tz_localize("UTC") + pd.Timedelta(days=1)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    valid_from = pd.to_datetime(intervals.get("valid_from"), errors="coerce")
    valid_to = pd.to_datetime(intervals.get("valid_to"), errors="coerce")
    known = pd.to_datetime(intervals.get("known_at_utc"), errors="coerce", utc=True)
    return intervals.loc[valid_from.le(as_of) & valid_to.ge(as_of) & known.le(cutoff)].copy().reset_index(drop=True)


def quarantine_lseg_snapshot_like_batches(pit_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarantine only source batches that are demonstrably anchor-like reset sentinels."""

    quarantine_columns = [*PIT_CONSTITUENT_COLUMNS, *QUARANTINE_AUDIT_COLUMNS]
    if pit_rows.empty:
        return pit_rows.copy(), pd.DataFrame(columns=quarantine_columns)

    rows = pit_rows.copy()
    snapshots = rows[rows["membership_action"].eq("snapshot_member")]
    changes = rows[rows["membership_action"].isin(["added", "removed"])]
    quarantined: list[pd.DataFrame] = []
    quarantine_indexes: set[Any] = set()
    group_columns = ["index_id", "source_request_id", "effective_date"]
    for (index_id, _, _), batch in changes.groupby(group_columns, dropna=False, sort=False):
        anchor_members = {
            _text(value)
            for value in snapshots.loc[snapshots["index_id"].astype(str).eq(str(index_id)), "member_ric"]
            if _text(value)
        }
        batch_members = {_text(value) for value in batch["member_ric"] if _text(value)}
        overlap = batch_members.intersection(anchor_members)
        anchor_count = len(anchor_members)
        overlap_fraction = len(overlap) / anchor_count if anchor_count else 0.0
        actions = set(batch["membership_action"].dropna().astype(str))
        is_strict_reset_sentinel = bool(
            anchor_count >= LSEG_RESET_SENTINEL_MIN_ANCHOR_MEMBERS
            and actions == {"added"}
            and len(batch) == len(batch_members)
            and batch_members.issubset(anchor_members)
            and overlap_fraction >= LSEG_RESET_SENTINEL_MIN_ANCHOR_FRACTION
        )
        if not is_strict_reset_sentinel:
            continue
        audit = batch.copy()
        audit["quarantine_reason"] = "snapshot_like_all_added_reset_sentinel"
        audit["quarantine_rule"] = (
            "single source/date batch; anchor>=20; additions only; unique members; all members in current anchor; "
            "anchor overlap>=0.80"
        )
        audit["batch_member_count"] = len(batch_members)
        audit["current_anchor_member_count"] = anchor_count
        audit["anchor_overlap_count"] = len(overlap)
        audit["anchor_overlap_fraction"] = overlap_fraction
        quarantined.append(audit.reindex(columns=quarantine_columns))
        quarantine_indexes.update(batch.index)

    quarantine = (
        pd.concat(quarantined, ignore_index=True)
        if quarantined
        else pd.DataFrame(columns=quarantine_columns)
    )
    retained = rows.drop(index=list(quarantine_indexes)).reset_index(drop=True)
    return retained, quarantine


def execute_lseg_usa_index_constituents(
    client: Any,
    artifact_root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    index_catalog: pd.DataFrame | None = None,
    chunk_years: int = 1,
    max_requests: int | None = 10,
    resume: bool = True,
    sleep_seconds: float = 0.05,
    session_name: str | None = None,
    observed_at_utc: str | None = None,
    anchor_date: str | None = None,
    identifier_resolution_chunk_size: int = LSEG_IDENTITY_RESOLUTION_CHUNK_SIZE,
    max_identifier_resolution_requests: int = LSEG_MAX_IDENTITY_RESOLUTION_REQUESTS,
    scope_profile: str = USA_INDEX_SCOPE_PROFILE_FULL,
) -> LSEGUSAIndexConstituentArtifacts:
    """Execute a bounded, resumable official LSEG constituent extraction."""

    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive or None")
    if identifier_resolution_chunk_size <= 0:
        raise ValueError("identifier_resolution_chunk_size must be positive")
    if max_identifier_resolution_requests <= 0:
        raise ValueError("max_identifier_resolution_requests must be positive")
    artifact_dir = Path(artifact_root) / run_id
    raw_dir = artifact_dir / "raw_responses"
    normalized_dir = artifact_dir / "normalized_requests"
    identity_dir = artifact_dir / "identity_resolution"
    identity_raw_dir = identity_dir / "raw_responses"
    identity_normalized_dir = identity_dir / "normalized_requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    identity_raw_dir.mkdir(parents=True, exist_ok=True)
    identity_normalized_dir.mkdir(parents=True, exist_ok=True)

    profile = resolve_usa_index_scope_profile(scope_profile)
    catalog = index_catalog.copy() if index_catalog is not None else usa_index_catalog_for_scope_profile(profile.name)
    observed_timestamp = _utc_timestamp(observed_at_utc)
    observed_date = pd.Timestamp(observed_timestamp).date().isoformat()
    if anchor_date is not None and _date(anchor_date, "anchor_date").date().isoformat() != observed_date:
        raise ValueError("anchor_date must equal the raw-response observation date for a current-chain anchor")
    effective_anchor_date = observed_date
    plan = build_lseg_usa_index_constituent_plan(
        start_date,
        end_date,
        catalog,
        chunk_years=chunk_years,
        anchor_date=effective_anchor_date,
        scope_profile=profile.name,
    )
    request_plan_path = artifact_dir / "request_plan.csv"
    request_manifest_path = artifact_dir / "request_manifest.csv"
    coverage_path = artifact_dir / "coverage.csv"
    pit_rows_path_candidate = artifact_dir / "pit_constituent_rows.parquet"
    intervals_path_candidate = artifact_dir / "constituent_intervals.parquet"
    issues_path = artifact_dir / "reconstruction_issues.csv"
    checkpoint_dir = artifact_dir / "reconstruction_checkpoints"
    checkpoint_manifest_path = artifact_dir / "reconstruction_checkpoint_manifest.csv"
    identity_resolution_manifest_path = identity_dir / "request_manifest.csv"
    identity_map_path = identity_dir / "identity_map.csv"
    quarantine_path = artifact_dir / "quarantined_joiner_leaver_batches.csv"
    manifest_path = artifact_dir / "manifest.json"
    plan.to_csv(request_plan_path, index=False)

    existing = _read_request_manifest(request_manifest_path) if resume else pd.DataFrame(columns=REQUEST_MANIFEST_COLUMNS)
    planned_request_ids = set(plan["source_request_id"].astype(str))
    ledger = {
        str(row["source_request_id"]): row
        for row in existing.to_dict("records")
        if str(row["source_request_id"]) in planned_request_ids
    }
    for row in plan.loc[~plan["live_request_allowed"]].to_dict("records"):
        request_id = str(row["source_request_id"])
        ledger[request_id] = _manifest_row(
            row,
            status="blocked_missing_entitlement",
            error="Russell constituent history is unavailable under the confirmed entitlement",
        )

    completed = {
        request_id
        for request_id, row in ledger.items()
        if _resumable_request(row)
    }
    pending = plan.loc[plan["live_request_allowed"] & ~plan["source_request_id"].astype(str).isin(completed)].copy()
    if max_requests is not None:
        pending = pending.head(max_requests)

    for request in pending.to_dict("records"):
        request_id = str(request["source_request_id"])
        started_at = _utc_timestamp(observed_at_utc)
        raw_path = raw_dir / f"{request_id}.json"
        normalized_path = normalized_dir / f"{request_id}.parquet"
        try:
            raw = _call_get_data(client, request)
            observed = _utc_timestamp(observed_at_utc)
            frame = _response_to_frame(raw)
            raw_hash = _persist_raw_response(raw_path, request, frame, observed)
            normalized, normalization_errors = _normalize_response(frame, request, observed, raw_hash)
            normalized.to_parquet(normalized_path, index=False)
            if normalization_errors:
                status = "failed_validation"
                error = " | ".join(normalization_errors)
            elif request["request_kind"] == "current_snapshot" and normalized.empty:
                status = "failed_validation"
                error = "current-chain snapshot returned no constituent RICs"
            else:
                status = "completed_empty" if normalized.empty else "completed"
                error = ""
            ledger[request_id] = _manifest_row(
                request,
                status=status,
                started_at=started_at,
                completed_at=_utc_timestamp(observed_at_utc),
                response_rows=len(frame),
                normalized_rows=len(normalized),
                raw_path=raw_path,
                raw_hash=raw_hash,
                normalized_path=normalized_path,
                error=error,
            )
        except Exception as exc:  # pragma: no cover - exercised by live vendor failures
            ledger[request_id] = _manifest_row(
                request,
                status="failed",
                started_at=started_at,
                completed_at=_utc_timestamp(observed_at_utc),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        _write_request_manifest(request_manifest_path, ledger)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _write_request_manifest(request_manifest_path, ledger)
    request_manifest = _read_request_manifest(request_manifest_path)
    pit_rows = _load_completed_normalized(request_manifest)
    identity_resolution_manifest, identity_map = _resolve_lseg_identity_drift(
        client=client,
        pit_rows=pit_rows,
        raw_dir=identity_raw_dir,
        normalized_dir=identity_normalized_dir,
        manifest_path=identity_resolution_manifest_path,
        identity_map_path=identity_map_path,
        observed_at_utc=observed_at_utc,
        resume=resume,
        chunk_size=identifier_resolution_chunk_size,
        max_requests=max_identifier_resolution_requests,
        sleep_seconds=sleep_seconds,
    )
    pit_rows, identity_issues = _apply_lseg_identity_map(pit_rows, identity_map)
    reconstruction_rows, quarantined_rows = quarantine_lseg_snapshot_like_batches(pit_rows)
    _write_csv_atomic(
        quarantine_path,
        quarantined_rows.reindex(columns=[*PIT_CONSTITUENT_COLUMNS, *QUARANTINE_AUDIT_COLUMNS]),
    )
    base_reconstruction = reconstruct_lseg_constituent_intervals(
        reconstruction_rows,
        start_date=start_date,
        end_date=end_date,
        checkpoint_dir=checkpoint_dir,
    )
    combined_issues = pd.concat([identity_issues, base_reconstruction.issues], ignore_index=True).drop_duplicates(
        ignore_index=True
    )
    reconstruction = LSEGIndexReconstruction(intervals=base_reconstruction.intervals, issues=combined_issues)
    reconstruction.issues.to_csv(issues_path, index=False)
    _checkpoint_manifest(checkpoint_dir).to_csv(checkpoint_manifest_path, index=False)

    pit_rows_path: Path | None = None
    if not pit_rows.empty:
        pit_rows.to_parquet(pit_rows_path_candidate, index=False)
        pit_rows_path = pit_rows_path_candidate
    intervals_path: Path | None = None
    if not reconstruction.intervals.empty:
        reconstruction.intervals.to_parquet(intervals_path_candidate, index=False)
        intervals_path = intervals_path_candidate

    coverage = _build_coverage(catalog, plan, request_manifest, reconstruction)
    coverage.to_csv(coverage_path, index=False)
    incomplete = coverage.loc[~coverage["status"].eq("complete"), "index_id"].astype(str).tolist()
    missing_russell = coverage.loc[
        coverage["status"].eq("incomplete_missing_constituent_entitlement"), "index_id"
    ].astype(str).tolist()
    catalog_index_ids = catalog["index_id"].astype(str).tolist()
    scope_match = bool(
        len(catalog_index_ids) == len(profile.required_index_ids)
        and set(catalog_index_ids) == set(profile.required_index_ids)
    )
    coverage_status = coverage.set_index("index_id")["status"].astype(str).to_dict()
    incomplete_required = [
        index_id for index_id in profile.required_index_ids if coverage_status.get(index_id) != "complete"
    ]
    zero_reconstruction_issues = reconstruction.issues.empty
    completion_gates = {
        "catalog_exactly_matches_scope_profile": scope_match,
        "all_required_indexes_complete": not incomplete_required,
        "zero_reconstruction_issues": zero_reconstruction_issues,
    }
    complete = all(completion_gates.values())
    status = "complete" if complete else "incomplete"
    scorecard_eligible = complete
    manifest = {
        "run_id": run_id,
        "dataset_id": DATASET_ID,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "algorithm": "current_chain_anchor_plus_reverse_official_lseg_joiner_leaver",
        "current_snapshot_field": LSEG_CURRENT_CHAIN_FIELD,
        "joiner_leaver_fields": list(LSEG_JOINER_LEAVER_FIELDS),
        "joiner_leaver_parameters": ["SDATE", "EDATE", "IC=B"],
        "start_date": _date(start_date, "start_date").date().isoformat(),
        "end_date": _date(end_date, "end_date").date().isoformat(),
        "anchor_date": _date(effective_anchor_date, "anchor_date").date().isoformat(),
        "scope_profile": profile.name,
        "required_index_ids": list(profile.required_index_ids),
        "excluded_index_ids": list(profile.excluded_index_ids),
        "excluded_index_reason": profile.excluded_index_reason,
        "scope_match": scope_match,
        "status": status,
        "scorecard_eligible": scorecard_eligible,
        "completion_gates": completion_gates,
        "planned_requests": int(len(plan)),
        "bounded_request_limit": max_requests,
        "identity_resolution_fields": list(LSEG_IDENTITY_RESOLUTION_FIELDS),
        "identity_resolution_chunk_size": identifier_resolution_chunk_size,
        "bounded_identity_resolution_request_limit": max_identifier_resolution_requests,
        "identity_resolution_requests": int(len(identity_resolution_manifest)),
        "identity_resolution_completed_requests": int(identity_resolution_manifest["status"].eq("completed").sum()),
        "identity_map_rows": int(len(identity_map)),
        "identity_map_sha256": _file_sha256(identity_map_path),
        "quarantined_joiner_leaver_rows": int(len(quarantined_rows)),
        "quarantine_sha256": _file_sha256(quarantine_path),
        "reset_sentinel_rule": {
            "minimum_current_anchor_members": LSEG_RESET_SENTINEL_MIN_ANCHOR_MEMBERS,
            "minimum_anchor_overlap_fraction": LSEG_RESET_SENTINEL_MIN_ANCHOR_FRACTION,
            "requires_no_removals": True,
        },
        "completed_requests": int(request_manifest["status"].isin(["completed", "completed_empty"]).sum()),
        "failed_requests": int(request_manifest["status"].isin(["failed", "failed_validation"]).sum()),
        "blocked_requests": int(request_manifest["status"].eq("blocked_missing_entitlement").sum()),
        "pit_rows": int(len(pit_rows)),
        "constituent_intervals": int(len(reconstruction.intervals)),
        "reconstruction_issue_count": int(len(reconstruction.issues)),
        "incomplete_index_ids": incomplete,
        "incomplete_required_index_ids": incomplete_required,
        "missing_russell_coverage": missing_russell,
        "known_at_policy": (
            "Current-chain members are known only when the raw response is observed. Historical JL events use the "
            "official change date once effective; forward-dated events use response observation time. Changes apply "
            "at max(effective_date, known_at_date), and reverse-derived start members retain their later evidence time."
        ),
        "current_anchor_provenance_retained": True,
        "dated_chain_historical_anchor_accepted": False,
        "dated_chain_rejection_reason": "live dated-chain probe returned modern members for a historical date",
        "verified_2020_joiner_leaver_counts": LSEG_VERIFIED_2020_EVENT_COUNTS,
        "raw_responses_retained_and_hashed": True,
        "constituent_history_fabricated": False,
        "blockers": [
            *(["Russell .RUI/.RUT/.RUA constituent entitlement is missing"] if missing_russell else []),
            *(
                ["selected index catalog does not exactly match the required scope profile"]
                if not scope_match
                else []
            ),
            *(["one or more required indexes are incomplete"] if incomplete_required else []),
            *(["one or more enabled LSEG requests are incomplete"] if _has_incomplete_enabled_requests(plan, request_manifest) else []),
            *(["joiner/leaver reconstruction validation failed"] if not reconstruction.issues.empty else []),
        ],
        "artifacts": {
            "request_plan": str(request_plan_path.resolve()),
            "request_manifest": str(request_manifest_path.resolve()),
            "coverage": str(coverage_path.resolve()),
            "pit_constituent_rows": str(pit_rows_path.resolve()) if pit_rows_path else None,
            "constituent_intervals": str(intervals_path.resolve()) if intervals_path else None,
            "reconstruction_issues": str(issues_path.resolve()),
            "reconstruction_checkpoint_manifest": str(checkpoint_manifest_path.resolve()),
            "reconstruction_checkpoint_dir": str(checkpoint_dir.resolve()),
            "raw_response_dir": str(raw_dir.resolve()),
            "identity_resolution_manifest": str(identity_resolution_manifest_path.resolve()),
            "identity_map": str(identity_map_path.resolve()),
            "quarantined_joiner_leaver_batches": str(quarantine_path.resolve()),
        },
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return LSEGUSAIndexConstituentArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        request_plan_path=request_plan_path,
        request_manifest_path=request_manifest_path,
        coverage_path=coverage_path,
        pit_rows_path=pit_rows_path,
        intervals_path=intervals_path,
        reconstruction_issues_path=issues_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        identity_resolution_manifest_path=identity_resolution_manifest_path,
        identity_map_path=identity_map_path,
        quarantine_path=quarantine_path,
        manifest_path=manifest_path,
        request_plan=plan,
        request_manifest=request_manifest,
        coverage=coverage,
        pit_rows=pit_rows,
        identity_resolution_manifest=identity_resolution_manifest,
        identity_map=identity_map,
        quarantined_rows=quarantined_rows,
        reconstruction=reconstruction,
    )


def _plan_row(
    request_id: str,
    item: Any,
    request_kind: str,
    universe: str,
    fields: Sequence[str],
    parameters: dict[str, str],
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    entitled: bool,
    entitlement_status: str,
) -> dict[str, Any]:
    return {
        "source_request_id": request_id,
        "dataset_id": DATASET_ID,
        "index_id": item.index_id,
        "index_family": item.index_family,
        "index_name": item.index_name,
        "index_ric": item.candidate_index_ric,
        "request_kind": request_kind,
        "universe": universe,
        "fields_json": json.dumps(list(fields), separators=(",", ":")),
        "parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        "request_start_date": request_start.date().isoformat(),
        "request_end_date": request_end.date().isoformat(),
        "route": "lseg.data.get_data",
        "entitlement_status": entitlement_status,
        "live_request_allowed": entitled,
        "request_status": "pending" if entitled else "blocked_missing_entitlement",
    }


def _call_get_data(client: Any, request: dict[str, Any]) -> Any:
    get_data = getattr(client, "get_data", None)
    if get_data is None:
        raise TypeError("client must expose get_data(universe=..., fields=..., parameters=...)")
    universe = str(request["universe"])
    fields = json.loads(str(request["fields_json"]))
    parameters = json.loads(str(request["parameters_json"]))
    if parameters:
        return get_data(universe=universe, fields=fields, parameters=parameters)
    return get_data(universe=universe, fields=fields)


def _response_to_frame(raw: Any) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    data = getattr(raw, "data", None)
    if data is not None and isinstance(getattr(data, "df", None), pd.DataFrame):
        return data.df.copy()
    if isinstance(getattr(raw, "df", None), pd.DataFrame):
        return raw.df.copy()
    if isinstance(raw, dict):
        return pd.DataFrame(raw)
    raise TypeError(f"unsupported LSEG response type: {type(raw).__name__}")


def _persist_raw_response(path: Path, request: dict[str, Any], frame: pd.DataFrame, observed_at: str) -> str:
    envelope = {
        "schema_version": 1,
        "source": "lseg.data.get_data",
        "observed_at_utc": observed_at,
        "request": {
            "source_request_id": request["source_request_id"],
            "universe": request["universe"],
            "fields": json.loads(str(request["fields_json"])),
            "parameters": json.loads(str(request["parameters_json"])),
        },
        "response": {
            "columns": [str(column) for column in frame.columns],
            "data": [[_json_value(value) for value in row] for row in frame.itertuples(index=False, name=None)],
        },
    }
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _resolve_lseg_identity_drift(
    client: Any,
    pit_rows: pd.DataFrame,
    raw_dir: Path,
    normalized_dir: Path,
    manifest_path: Path,
    identity_map_path: Path,
    observed_at_utc: str | None,
    resume: bool,
    chunk_size: int,
    max_requests: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = sorted(
        {
            _identifier_text(value)
            for value in pit_rows.loc[
                pit_rows["membership_action"].isin(["added", "removed"]), "member_ric"
            ]
            if _is_lseg_identity_drift_candidate(value)
        },
        key=_identifier_key,
    )
    existing = (
        _read_identity_resolution_manifest(manifest_path)
        if resume
        else pd.DataFrame(columns=IDENTITY_RESOLUTION_MANIFEST_COLUMNS)
    )
    ledger = {
        _text(row.get("source_request_id")): row
        for row in existing.to_dict("records")
        if _text(row.get("source_request_id"))
    }
    existing_map = _load_resumable_identity_map(pd.DataFrame(ledger.values()))
    covered = {_identifier_key(value) for value in existing_map.get("raw_identifier", pd.Series(dtype=str))}
    outstanding = [value for value in candidates if _identifier_key(value) not in covered]
    chunks = [outstanding[offset : offset + chunk_size] for offset in range(0, len(outstanding), chunk_size)]

    for identifiers in chunks[:max_requests]:
        fingerprint = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()[:20]
        request_id = f"usa-pit-index-identity-{fingerprint}"
        request = {
            "source_request_id": request_id,
            "universe": identifiers,
            "fields_json": json.dumps(list(LSEG_IDENTITY_RESOLUTION_FIELDS), separators=(",", ":")),
            "parameters_json": "{}",
        }
        started_at = _utc_timestamp(observed_at_utc)
        raw_path = raw_dir / f"{request_id}.json"
        normalized_path = normalized_dir / f"{request_id}.parquet"
        try:
            raw = client.get_data(universe=identifiers, fields=list(LSEG_IDENTITY_RESOLUTION_FIELDS))
            observed = _utc_timestamp(observed_at_utc)
            frame = _response_to_frame(raw)
            raw_hash = _persist_raw_response(raw_path, request, frame, observed)
            normalized = _normalize_identity_resolution(frame, identifiers, request_id, observed, raw_hash)
            normalized.to_parquet(normalized_path, index=False)
            normalized_hash = _file_sha256(normalized_path)
            ledger[request_id] = _identity_manifest_row(
                request_id=request_id,
                status="completed",
                identifiers=identifiers,
                started_at=started_at,
                completed_at=_utc_timestamp(observed_at_utc),
                response_rows=len(frame),
                raw_path=raw_path,
                raw_hash=raw_hash,
                normalized_path=normalized_path,
                normalized_hash=normalized_hash,
            )
        except Exception as exc:  # pragma: no cover - exercised by live vendor failures
            ledger[request_id] = _identity_manifest_row(
                request_id=request_id,
                status="failed",
                identifiers=identifiers,
                started_at=started_at,
                completed_at=_utc_timestamp(observed_at_utc),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        _write_identity_resolution_manifest(manifest_path, ledger)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _write_identity_resolution_manifest(manifest_path, ledger)
    identity_manifest = _read_identity_resolution_manifest(manifest_path)
    identity_map = _load_resumable_identity_map(identity_manifest)
    candidate_keys = {_identifier_key(value) for value in candidates}
    identity_map = identity_map.loc[
        identity_map.get("raw_identifier", pd.Series(dtype=str)).map(_identifier_key).isin(candidate_keys)
    ].copy()
    if not identity_map.empty:
        identity_map = identity_map.sort_values(["raw_identifier", "source_request_id"], kind="stable").reset_index(
            drop=True
        )
    _write_csv_atomic(identity_map_path, identity_map.reindex(columns=IDENTITY_MAP_COLUMNS))
    return identity_manifest, identity_map.reindex(columns=IDENTITY_MAP_COLUMNS)


def _normalize_identity_resolution(
    frame: pd.DataFrame,
    identifiers: Sequence[str],
    request_id: str,
    observed_at: str,
    raw_hash: str,
) -> pd.DataFrame:
    instrument_column = _find_column(frame, ("instrument",))
    ric_column = _find_column(frame, ("ric",))
    name_column = _find_column(frame, ("commonname", "name"))
    sedol_column = _find_column(frame, ("sedol",))
    primary_ric_column = _find_column(frame, ("primaryric",))
    rows: list[dict[str, Any]] = []

    for raw_identifier in identifiers:
        raw_key = _identifier_key(raw_identifier)
        matching_rows: list[pd.Series] = []
        for _, source_row in frame.iterrows():
            response_keys = {
                _identifier_key(source_row.get(column))
                for column in (instrument_column, sedol_column)
                if column is not None and _identifier_key(source_row.get(column))
            }
            if raw_key in response_keys:
                matching_rows.append(source_row)

        resolved_values: list[str] = []
        for source_row in matching_rows:
            for column in (primary_ric_column, ric_column):
                value = _text(source_row.get(column)) if column is not None else ""
                if value and _identifier_key(value) != raw_key:
                    resolved_values.append(value)
                    break
        unique_resolved = list(dict.fromkeys(resolved_values))
        if not matching_rows:
            status = "unresolved_no_matching_response"
            resolved_ric = ""
        elif not unique_resolved:
            status = "unresolved_no_ric"
            resolved_ric = ""
        elif len(unique_resolved) > 1:
            status = "conflicting_resolved_ric"
            resolved_ric = ""
        else:
            status = "resolved"
            resolved_ric = unique_resolved[0]
        source_row = matching_rows[0] if matching_rows else pd.Series(dtype=object)
        rows.append(
            {
                "raw_identifier": raw_identifier,
                "resolved_ric": resolved_ric,
                "common_name": _text(source_row.get(name_column)) if name_column is not None else "",
                "sedol": _identifier_text(source_row.get(sedol_column)) if sedol_column is not None else "",
                "primary_ric": _text(source_row.get(primary_ric_column)) if primary_ric_column is not None else "",
                "resolution_status": status,
                "source_request_id": request_id,
                "observed_at_utc": observed_at,
                "raw_response_sha256": raw_hash,
            }
        )
    return pd.DataFrame(rows, columns=IDENTITY_MAP_COLUMNS)


def _apply_lseg_identity_map(
    pit_rows: pd.DataFrame,
    identity_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pit_rows.copy()
    issues: list[dict[str, Any]] = []
    map_by_identifier: dict[str, pd.DataFrame] = {
        key: group
        for key, group in identity_map.assign(
            _raw_key=identity_map.get("raw_identifier", pd.Series(dtype=str)).map(_identifier_key)
        ).groupby("_raw_key", sort=False)
        if key
    }
    event_mask = rows["membership_action"].isin(["added", "removed"])
    for row_number, row in rows.loc[event_mask].iterrows():
        raw_identifier = _text(row.get("member_ric"))
        if not _is_lseg_identity_drift_candidate(raw_identifier):
            continue
        matches = map_by_identifier.get(_identifier_key(raw_identifier), pd.DataFrame(columns=IDENTITY_MAP_COLUMNS))
        resolved = list(
            dict.fromkeys(
                matches.loc[matches.get("resolution_status").eq("resolved"), "resolved_ric"].dropna().astype(str)
            )
        )
        if len(resolved) == 1:
            rows.at[row_number, "member_ric"] = resolved[0]
            names = matches.loc[matches.get("resolution_status").eq("resolved"), "common_name"].dropna().astype(str)
            if not _text(row.get("member_name")) and not names.empty:
                rows.at[row_number, "member_name"] = names.iloc[0]
            continue
        statuses = sorted(set(matches.get("resolution_status", pd.Series(dtype=str)).dropna().astype(str)))
        detail = (
            f"LSEG identifier resolution did not produce one unique RIC; raw_identifier={raw_identifier}; "
            f"statuses={statuses or ['not_requested_or_request_incomplete']}"
        )
        issues.append(
            {
                "index_id": _text(row.get("index_id")),
                "member_ric": raw_identifier,
                "effective_date": _date_text(row.get("effective_date")),
                "source_record_id": _text(row.get("source_record_id")),
                "issue_code": "unresolved_member_identifier",
                "detail": detail,
            }
        )
    return rows, pd.DataFrame(issues, columns=RECONSTRUCTION_ISSUE_COLUMNS).drop_duplicates(ignore_index=True)


def _identity_manifest_row(
    request_id: str,
    status: str,
    identifiers: Sequence[str],
    started_at: str = "",
    completed_at: str = "",
    response_rows: int = 0,
    raw_path: Path | None = None,
    raw_hash: str = "",
    normalized_path: Path | None = None,
    normalized_hash: str = "",
    error_type: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "source_request_id": request_id,
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "requested_identifiers": json.dumps(list(identifiers), separators=(",", ":")),
        "response_rows": response_rows,
        "raw_response_path": str(raw_path.resolve()) if raw_path else "",
        "raw_response_sha256": raw_hash,
        "normalized_path": str(normalized_path.resolve()) if normalized_path else "",
        "normalized_sha256": normalized_hash,
        "error_type": error_type,
        "error": error,
    }


def _read_identity_resolution_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=IDENTITY_RESOLUTION_MANIFEST_COLUMNS)
    frame = pd.read_csv(path, low_memory=False, keep_default_na=False)
    for column in IDENTITY_RESOLUTION_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[list(IDENTITY_RESOLUTION_MANIFEST_COLUMNS)]


def _write_identity_resolution_manifest(path: Path, ledger: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(ledger.values(), columns=IDENTITY_RESOLUTION_MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("source_request_id", kind="stable")
    _write_csv_atomic(path, frame)


def _load_resumable_identity_map(manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in manifest.to_dict("records"):
        if not _resumable_identity_request(row):
            continue
        frame = pd.read_parquet(str(row["normalized_path"]))
        if not frame.empty:
            frames.append(frame.reindex(columns=IDENTITY_MAP_COLUMNS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=IDENTITY_MAP_COLUMNS)


def _resumable_identity_request(row: dict[str, Any]) -> bool:
    if _text(row.get("status")) != "completed":
        return False
    raw_path = Path(_text(row.get("raw_response_path")))
    normalized_path = Path(_text(row.get("normalized_path")))
    expected_hash = _text(row.get("raw_response_sha256"))
    expected_normalized_hash = _text(row.get("normalized_sha256"))
    return bool(
        raw_path.is_file()
        and normalized_path.is_file()
        and expected_hash
        and expected_normalized_hash
        and _file_sha256(raw_path) == expected_hash
        and _file_sha256(normalized_path) == expected_normalized_hash
    )


def _normalize_response(
    frame: pd.DataFrame,
    request: dict[str, Any],
    observed_at: str,
    raw_hash: str,
) -> tuple[pd.DataFrame, list[str]]:
    if request["request_kind"] == "current_snapshot":
        return _normalize_snapshot(frame, request, observed_at, raw_hash)
    return _normalize_joiner_leaver(frame, request, observed_at, raw_hash)


def _normalize_snapshot(
    frame: pd.DataFrame,
    request: dict[str, Any],
    observed_at: str,
    raw_hash: str,
) -> tuple[pd.DataFrame, list[str]]:
    member_column = _find_column(frame, ("instrument", "constituentric", "indexconstituentric", "ric"))
    if member_column is None:
        return _empty_pit_rows(), ["initial snapshot response is missing a constituent RIC/Instrument column"]
    snapshot_date = pd.Timestamp(observed_at).tz_convert("UTC").tz_localize(None).normalize()
    known_at = _utc_timestamp(observed_at)
    known_source = "lseg_raw_response_observed_at_utc"
    rows: list[dict[str, Any]] = []
    for ordinal, source_row in frame.reset_index(drop=True).iterrows():
        member = _text(source_row.get(member_column))
        if not member or member == str(request["index_ric"]):
            continue
        rows.append(
            _pit_row(
                request=request,
                member_ric=member,
                member_name="",
                action="snapshot_member",
                membership_as_of_date=snapshot_date.date().isoformat(),
                effective_date=None,
                known_at=known_at,
                known_source=known_source,
                observed_at=observed_at,
                raw_hash=raw_hash,
                ordinal=ordinal,
            )
        )
    return pd.DataFrame(rows, columns=PIT_CONSTITUENT_COLUMNS), []


def _normalize_joiner_leaver(
    frame: pd.DataFrame,
    request: dict[str, Any],
    observed_at: str,
    raw_hash: str,
) -> tuple[pd.DataFrame, list[str]]:
    aliases = {
        "effective": ("indexjlconstituentchangedate", "constituentchangedate", "changedate", "date"),
        "member": ("indexjlconstituentric", "constituentric", "ric"),
        "name": ("indexjlconstituentname", "constituentname", "name"),
        "change": (
            "indexjlconstituentituentchange",
            "indexjlconstituentchange",
            "constituentchange",
            "change",
        ),
    }
    columns = {role: _find_column(frame, candidates) for role, candidates in aliases.items()}
    missing = [role for role in ("effective", "member", "change") if columns[role] is None]
    if missing:
        return _empty_pit_rows(), [f"joiner/leaver response is missing required columns: {','.join(missing)}"]

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for ordinal, source_row in frame.reset_index(drop=True).iterrows():
        raw_date = source_row.get(columns["effective"])
        effective = pd.to_datetime(raw_date, errors="coerce")
        member = _text(source_row.get(columns["member"]))
        raw_change = source_row.get(columns["change"])
        action = _normalize_action(raw_change)
        if pd.isna(effective) and not member and not _text(raw_change):
            # LSEG represents a valid no-change window with one blank sentinel row.
            continue
        if pd.isna(effective) or not member or action is None:
            errors.append(f"row {ordinal} has invalid change date, member RIC, or action")
            continue
        effective = pd.Timestamp(effective).normalize()
        known_at, known_source = _known_at(effective, observed_at, LSEG_JOINER_LEAVER_FIELDS[0])
        name = _text(source_row.get(columns["name"])) if columns["name"] is not None else ""
        rows.append(
            _pit_row(
                request=request,
                member_ric=member,
                member_name=name,
                action=action,
                membership_as_of_date=None,
                effective_date=effective.date().isoformat(),
                known_at=known_at,
                known_source=known_source,
                observed_at=observed_at,
                raw_hash=raw_hash,
                ordinal=ordinal,
            )
        )
    normalized = pd.DataFrame(rows, columns=PIT_CONSTITUENT_COLUMNS)
    validation = validate_lseg_joiner_leaver_events(normalized)
    errors.extend(validation["detail"].astype(str).tolist())
    return normalized, list(dict.fromkeys(errors))


def _pit_row(
    request: dict[str, Any],
    member_ric: str,
    member_name: str,
    action: str,
    membership_as_of_date: str | None,
    effective_date: str | None,
    known_at: str,
    known_source: str,
    observed_at: str,
    raw_hash: str,
    ordinal: int,
) -> dict[str, Any]:
    identity = "|".join(
        [str(request["source_request_id"]), member_ric, action, str(membership_as_of_date or effective_date), str(ordinal)]
    )
    return {
        "index_id": request["index_id"],
        "index_family": request["index_family"],
        "index_name": request["index_name"],
        "index_ric": request["index_ric"],
        "member_ric": member_ric,
        "member_perm_id": pd.NA,
        "member_isin": pd.NA,
        "member_ticker": pd.NA,
        "member_name": member_name or pd.NA,
        "membership_action": action,
        "membership_as_of_date": membership_as_of_date or pd.NA,
        "effective_date": effective_date or pd.NA,
        "weight_percent": pd.NA,
        "known_at_utc": known_at,
        "known_at_source_field": known_source,
        "source_provider": "LSEG",
        "source_dataset": "official_index_current_chain_and_joiner_leaver",
        "source_record_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "source_request_id": request["source_request_id"],
        "source_loaded_at_utc": observed_at,
        "raw_payload_sha256": raw_hash,
    }


def _known_at(event_date: pd.Timestamp, observed_at: str, date_source: str) -> tuple[str, str]:
    event_utc = pd.Timestamp(event_date).normalize().tz_localize("UTC")
    observed = pd.Timestamp(observed_at)
    observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    if event_utc <= observed:
        return event_utc.isoformat(), date_source
    return observed.isoformat(), "lseg_raw_response_observed_at_utc"


def _interval_from_state(
    index_id: str,
    index_ric: str,
    metadata: dict[str, Any],
    valid_from: pd.Timestamp,
    anchor_request_id: str,
    anchor_chain: str,
    anchor_hash: str,
) -> dict[str, Any]:
    return {
        "index_id": index_id,
        "index_ric": index_ric,
        "member_ric": _text(metadata.get("member_ric")),
        "member_name": _text(metadata.get("member_name")),
        "valid_from": valid_from.date().isoformat(),
        "valid_to": "",
        "known_at_utc": _text(metadata.get("known_at_utc")),
        "opening_source_record_id": _text(metadata.get("source_record_id")),
        "closing_source_record_id": "",
        "current_anchor_request_id": anchor_request_id,
        "current_anchor_chain": anchor_chain,
        "current_anchor_raw_payload_sha256": anchor_hash,
    }


def _interval_from_event(
    row: pd.Series,
    index_id: str,
    valid_from: pd.Timestamp,
    anchor_request_id: str,
    anchor_chain: str,
    anchor_hash: str,
) -> dict[str, Any]:
    return _interval_from_state(
        index_id=index_id,
        index_ric=_text(row.get("index_ric")),
        metadata={
            "member_ric": row.get("member_ric"),
            "member_name": row.get("member_name"),
            "known_at_utc": row.get("known_at_utc"),
            "source_record_id": row.get("source_record_id"),
        },
        valid_from=valid_from,
        anchor_request_id=anchor_request_id,
        anchor_chain=anchor_chain,
        anchor_hash=anchor_hash,
    )


def _state_issue(row: Any, index_id: str, code: str, detail: str) -> pd.DataFrame:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
    return pd.DataFrame(
        [{
            "index_id": index_id,
            "member_ric": _text(getter("member_ric", "")),
            "effective_date": _date_text(getter("effective_date", pd.NaT)),
            "source_record_id": _text(getter("source_record_id", "")),
            "issue_code": code,
            "detail": detail,
        }],
        columns=RECONSTRUCTION_ISSUE_COLUMNS,
    )


def _inferred_start(events: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.Timestamp:
    if events.empty:
        return anchor_date
    effective = pd.to_datetime(events.get("effective_date"), errors="coerce").dropna()
    return effective.min().normalize() if not effective.empty else anchor_date


def _events_fingerprint(events: pd.DataFrame) -> str:
    material = [
        "|".join(
            [
                _text(row.get("source_record_id")),
                _text(row.get("member_ric")),
                _text(row.get("membership_action")),
                _date_text(row.get("_application_date")),
            ]
        )
        for _, row in events.iterrows()
    ]
    return hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()


def _persist_checkpoint(
    root: Path | None,
    index_id: str,
    state: dict[str, dict[str, Any]],
    checkpoint_date: pd.Timestamp,
    processed_events: int,
    anchor_request_id: str,
    anchor_chain: str,
    anchor_hash: str,
    event_fingerprint: str,
) -> None:
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{pd.Timestamp(checkpoint_date).strftime('%Y%m%d')}-{processed_events:06d}"
    state_path = root / f"{stem}.parquet"
    temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp.parquet")
    state_rows = [
        {
            "index_id": index_id,
            "member_ric": member,
            "member_name": _text(metadata.get("member_name")),
            "known_at_utc": _text(metadata.get("known_at_utc")),
            "source_record_id": _text(metadata.get("source_record_id")),
        }
        for member, metadata in sorted(state.items())
    ]
    pd.DataFrame(
        state_rows,
        columns=["index_id", "member_ric", "member_name", "known_at_utc", "source_record_id"],
    ).to_parquet(temporary, index=False)
    temporary.replace(state_path)
    metadata = {
        "index_id": index_id,
        "checkpoint_date": pd.Timestamp(checkpoint_date).date().isoformat(),
        "direction": "descending_reverse_joiner_leaver",
        "processed_events": processed_events,
        "member_count": len(state),
        "anchor_request_id": anchor_request_id,
        "anchor_chain": anchor_chain,
        "anchor_raw_payload_sha256": anchor_hash,
        "event_fingerprint_sha256": event_fingerprint,
        "state_path": str(state_path.resolve()),
        "state_sha256": _file_sha256(state_path),
    }
    _atomic_write(
        root / f"{stem}.json",
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
    )


def _restore_checkpoint(
    root: Path | None,
    event_fingerprint: str,
    anchor_hash: str,
) -> tuple[dict[str, dict[str, Any]], int] | None:
    if root is None or not root.exists():
        return None
    candidates: list[tuple[int, dict[str, Any]]] = []
    for metadata_path in root.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("event_fingerprint_sha256") != event_fingerprint:
            continue
        if metadata.get("anchor_raw_payload_sha256") != anchor_hash:
            continue
        state_path = Path(str(metadata.get("state_path", "")))
        if not state_path.is_file() or _file_sha256(state_path) != metadata.get("state_sha256"):
            continue
        candidates.append((int(metadata.get("processed_events", 0)), metadata))
    if not candidates:
        return None
    processed, metadata = max(candidates, key=lambda item: item[0])
    frame = pd.read_parquet(str(metadata["state_path"]))
    state = {
        _text(row.member_ric): {
            "member_ric": _text(row.member_ric),
            "member_name": _text(row.member_name),
            "known_at_utc": _text(row.known_at_utc),
            "source_record_id": _text(row.source_record_id),
        }
        for row in frame.itertuples(index=False)
    }
    return state, processed


def _checkpoint_manifest(root: Path) -> pd.DataFrame:
    columns = (
        "index_id",
        "checkpoint_date",
        "direction",
        "processed_events",
        "member_count",
        "anchor_request_id",
        "anchor_chain",
        "anchor_raw_payload_sha256",
        "event_fingerprint_sha256",
        "state_path",
        "state_sha256",
    )
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.glob("*/*.json")):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return pd.DataFrame(rows, columns=columns)


def _build_coverage(
    catalog: pd.DataFrame,
    plan: pd.DataFrame,
    request_manifest: pd.DataFrame,
    reconstruction: LSEGIndexReconstruction,
) -> pd.DataFrame:
    status_by_request = request_manifest.set_index("source_request_id")["status"].astype(str).to_dict()
    issue_indexes = set(reconstruction.issues["index_id"].dropna().astype(str))
    interval_indexes = set(reconstruction.intervals["index_id"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for item in catalog.itertuples(index=False):
        target = plan[plan["index_id"].astype(str).eq(str(item.index_id))]
        snapshot = target[target["request_kind"].eq("current_snapshot")]
        jl = target[target["request_kind"].eq("joiner_leaver")]
        snapshot_status = status_by_request.get(str(snapshot.iloc[0]["source_request_id"]), "pending")
        jl_statuses = [status_by_request.get(str(request_id), "pending") for request_id in jl["source_request_id"]]
        ric = str(item.candidate_index_ric)
        if ric in LSEG_RUSSELL_INDEX_RICS:
            status = "incomplete_missing_constituent_entitlement"
        elif snapshot_status != "completed":
            status = "incomplete_no_current_anchor"
        elif not jl_statuses or not all(value in {"completed", "completed_empty"} for value in jl_statuses):
            status = "incomplete_request_coverage"
        elif str(item.index_id) in issue_indexes or str(item.index_id) not in interval_indexes:
            status = "incomplete_reconstruction_validation"
        else:
            status = "complete"
        rows.append(
            {
                "index_id": item.index_id,
                "index_family": item.index_family,
                "index_name": item.index_name,
                "index_ric": ric,
                "status": status,
                "current_anchor_status": snapshot_status,
                "joiner_leaver_requests": len(jl_statuses),
                "joiner_leaver_completed": sum(value in {"completed", "completed_empty"} for value in jl_statuses),
                "reconstruction_issues": int((reconstruction.issues["index_id"].astype(str) == str(item.index_id)).sum()),
                "interval_rows": int((reconstruction.intervals["index_id"].astype(str) == str(item.index_id)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _load_completed_normalized(request_manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    completed = request_manifest[request_manifest["status"].isin(["completed", "completed_empty"])]
    for path_text in completed["normalized_path"].dropna().astype(str):
        if path_text and Path(path_text).exists():
            frame = pd.read_parquet(path_text)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else _empty_pit_rows()


def _has_incomplete_enabled_requests(plan: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    enabled = set(plan.loc[plan["live_request_allowed"], "source_request_id"].astype(str))
    completed = set(
        manifest.loc[manifest["status"].isin(["completed", "completed_empty"]), "source_request_id"].astype(str)
    )
    return bool(enabled.difference(completed))


def _manifest_row(
    request: dict[str, Any],
    status: str,
    started_at: str = "",
    completed_at: str = "",
    response_rows: int = 0,
    normalized_rows: int = 0,
    raw_path: Path | None = None,
    raw_hash: str = "",
    normalized_path: Path | None = None,
    error_type: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "source_request_id": request["source_request_id"],
        "index_id": request["index_id"],
        "index_ric": request.get("index_ric", ""),
        "request_kind": request["request_kind"],
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "response_rows": response_rows,
        "normalized_rows": normalized_rows,
        "raw_response_path": str(raw_path.resolve()) if raw_path else "",
        "raw_response_sha256": raw_hash,
        "normalized_path": str(normalized_path.resolve()) if normalized_path else "",
        "error_type": error_type,
        "error": error,
    }


def _resumable_request(row: dict[str, Any]) -> bool:
    if str(row.get("status")) not in {"completed", "completed_empty"}:
        return False
    raw_path = Path(str(row.get("raw_response_path", "")))
    normalized_path = Path(str(row.get("normalized_path", "")))
    expected_hash = str(row.get("raw_response_sha256", ""))
    return bool(
        raw_path.is_file()
        and normalized_path.is_file()
        and expected_hash
        and _file_sha256(raw_path) == expected_hash
    )


def _read_request_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUEST_MANIFEST_COLUMNS)
    frame = pd.read_csv(path, low_memory=False, keep_default_na=False)
    for column in REQUEST_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[list(REQUEST_MANIFEST_COLUMNS)]


def _write_request_manifest(path: Path, ledger: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(ledger.values(), columns=REQUEST_MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["index_id", "request_kind", "source_request_id"], kind="stable")
    payload = frame.to_csv(index=False).encode("utf-8")
    _atomic_write(path, payload)


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Any | None:
    by_token = {_column_token(column): column for column in frame.columns}
    for alias in aliases:
        if alias in by_token:
            return by_token[alias]
    for alias in aliases:
        matches = [column for token, column in by_token.items() if token.endswith(alias)]
        if matches:
            return matches[0]
    return None


def _column_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _identifier_text(value: Any) -> str:
    text = _text(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _identifier_key(value: Any) -> str:
    return _identifier_text(value).upper()


def _is_lseg_identity_drift_candidate(value: Any) -> bool:
    identifier = _identifier_key(value)
    return bool(re.fullmatch(r"[A-Z0-9]{7}", identifier) and any(character.isdigit() for character in identifier))


def _normalize_action(value: Any) -> str | None:
    token = _column_token(value)
    if token in {"joiner", "join", "joined", "add", "added", "addition", "in", "j", "constituentadded"}:
        return "added"
    if token in {"leaver", "leave", "left", "remove", "removed", "delete", "deletion", "out", "l", "constituentremoved"}:
        return "removed"
    return None


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp, chunk_years: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, pd.Timestamp(year=cursor.year + chunk_years - 1, month=12, day=31))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def _date(value: str, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid date")
    parsed = pd.Timestamp(parsed)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.normalize()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _empty_pit_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=PIT_CONSTITUENT_COLUMNS)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _utc_timestamp(value: str | None = None) -> str:
    timestamp = pd.Timestamp(value) if value is not None else pd.Timestamp(datetime.now(timezone.utc))
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, frame.to_csv(index=False).encode("utf-8"))


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
