from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from quant_lab.data.security_master_discovery import (
    DISCOVERY_ENRICHMENT_FIELDS,
    normalize_security_master_enrichment,
)


DEFAULT_JAPAN_G1_SOURCE_DIR = Path(
    "artifacts/research_universes/JPX-UNIVERSE-RECONCILIATION-V2-20260711"
)
DEFAULT_JAPAN_G1_ARTIFACT_ROOT = Path("artifacts/japan_g1_enrichment")
ACTIVE_UNIVERSE_FILE = "japan_jpx_active_operating_core_v1.csv"
REFERENCE_UNIVERSE_FILE = "japan_jpx_historical_listed_reference_v1.csv"

TARGET_ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "price_currency",
    "first_trade_date",
    "retire_date",
    "isin",
    "sedol",
    "perm_id",
)
JAPAN_G1_ENRICHMENT_FIELDS: tuple[str, ...] = (*DISCOVERY_ENRICHMENT_FIELDS, "CF_CURR")
NORMALIZED_COLUMNS: tuple[str, ...] = (
    "ric",
    "ric_enriched",
    "common_name_enriched",
    "exchange_name_enriched",
    "exchange_country_code",
    "asset_status",
    "retire_date",
    "first_trade_date",
    "isin",
    "sedol",
    "organization_id",
    "perm_id",
    "price_currency",
    "attempt_id",
    "requested_ric",
    "observed_at_utc",
    "evidence_scope",
    "historical_pit_claimed",
    "eligible_for_historical_pit",
    "eligible_for_alpha_promotion",
    "eligible_for_production_promotion",
)
STATUS_COLUMNS: tuple[str, ...] = (
    "ric",
    "in_active_universe",
    "in_reference_universe",
    "source_scope",
    "source_order",
    "status",
    "attempt_count",
    "last_attempt_id",
    "observed_at_utc",
    "response_rows",
    "fields_present_count",
    "error_type",
    "error_message",
    "retry_eligible",
    "historical_pit_claimed",
    "eligible_for_historical_pit",
    "eligible_for_alpha_promotion",
    "eligible_for_production_promotion",
)


@dataclass(frozen=True)
class JapanG1EnrichmentArtifacts:
    run_id: str
    artifact_dir: Path
    queue_path: Path
    raw_responses_path: Path
    normalized_responses_path: Path
    ric_status_path: Path
    quality_path: Path
    checksums_path: Path
    manifest_path: Path
    status: pd.DataFrame
    quality: pd.DataFrame
    manifest: dict[str, Any]


def default_japan_g1_run_id(source_dir: str | Path = DEFAULT_JAPAN_G1_SOURCE_DIR) -> str:
    return f"{Path(source_dir).name}-G1-CURRENT-ENRICHMENT"


def write_japan_g1_enrichment_artifacts(
    get_data_client: Any,
    *,
    source_dir: str | Path = DEFAULT_JAPAN_G1_SOURCE_DIR,
    artifact_root: str | Path = DEFAULT_JAPAN_G1_ARTIFACT_ROOT,
    run_id: str | None = None,
    max_rics: int = 100,
    request_chunk_size: int = 25,
    sleep_seconds: float = 0.0,
    session_name: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> JapanG1EnrichmentArtifacts:
    """Resume a bounded current-observation enrichment of the reconciled JPX universes.

    The output intentionally does not represent historical point-in-time evidence. A
    successful current observation can fill metadata coverage, but it cannot promote
    Japan into historical backtests, alpha, or production.
    """

    if max_rics <= 0:
        raise ValueError("max_rics must be positive")
    if request_chunk_size <= 0:
        raise ValueError("request_chunk_size must be positive")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative")

    source = Path(source_dir)
    active_path = source / ACTIVE_UNIVERSE_FILE
    reference_path = source / REFERENCE_UNIVERSE_FILE
    _require_source(active_path)
    _require_source(reference_path)

    resolved_run_id = run_id or default_japan_g1_run_id(source)
    if not resolved_run_id.strip() or Path(resolved_run_id).name != resolved_run_id:
        raise ValueError("run_id must be a non-empty path-safe name")

    artifact_dir = Path(artifact_root) / resolved_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifact_dir)
    now = clock or (lambda: datetime.now(timezone.utc))
    source_hashes = {
        "active_operating_core": _sha256(active_path),
        "historical_listed_reference": _sha256(reference_path),
    }

    lock_handle = _acquire_run_lock(artifact_dir)
    try:
        prior_manifest = _read_json(paths["manifest"])
        _verify_resume_sources(prior_manifest, source_hashes)

        queue = _build_queue(active_path, reference_path)
        _verify_or_write_queue(paths["queue"], queue)
        status = _load_status(paths["ric_status"], queue)
        raw_events = _read_json_lines(paths["raw_responses"])
        normalized = _read_csv(paths["normalized_responses"], NORMALIZED_COLUMNS)

        candidates = _resume_candidates(status).head(max_rics)
        selected_rics = candidates["ric"].astype(str).tolist()
        invocation_attempted: list[str] = []
        invocation_failure: str | None = None

        getter = _resolve_getter(get_data_client)
        for rics in _chunks(selected_rics, min(request_chunk_size, max_rics)):
            invocation_attempted.extend(rics)
            pending_chunks = [list(rics)]
            while pending_chunks and not invocation_failure:
                request_rics = pending_chunks.pop(0)
                attempt_id = uuid.uuid4().hex
                observed_at = _as_utc_iso(now())
                try:
                    response = getter(
                        universe=request_rics,
                        fields=list(JAPAN_G1_ENRICHMENT_FIELDS),
                    )
                    raw_frame = _response_frame(response)
                    normalized_chunk = normalize_security_master_enrichment(raw_frame)
                    normalized_chunk = _apply_japan_currency_fallback(normalized_chunk, raw_frame)
                    normalized_chunk = _decorate_normalized(
                        normalized_chunk,
                        attempt_id=attempt_id,
                        observed_at_utc=observed_at,
                    )
                    status, _ = _record_chunk_status(
                        status,
                        request_rics,
                        normalized_chunk,
                        attempt_id=attempt_id,
                        observed_at_utc=observed_at,
                    )
                    raw_events.append(
                        _raw_event(
                            raw_frame,
                            request_rics,
                            attempt_id=attempt_id,
                            observed_at_utc=observed_at,
                        )
                    )
                    if normalized.empty:
                        normalized = normalized_chunk.copy()
                    elif not normalized_chunk.empty:
                        normalized = pd.concat(
                            [normalized, normalized_chunk], ignore_index=True, sort=False
                        )[list(NORMALIZED_COLUMNS)]
                except Exception as exc:  # vendor errors are retained as route evidence
                    raw_events.append(
                        _raw_event(
                            pd.DataFrame(),
                            request_rics,
                            attempt_id=attempt_id,
                            observed_at_utc=observed_at,
                            error=exc,
                        )
                    )
                    if _is_throttle_error(exc):
                        status = _record_chunk_error(
                            status,
                            request_rics,
                            exc,
                            attempt_id=attempt_id,
                            observed_at_utc=observed_at,
                        )
                        invocation_failure = "failed_closed_vendor_throttle"
                    elif len(request_rics) > 1:
                        midpoint = len(request_rics) // 2
                        pending_chunks = [
                            request_rics[:midpoint],
                            request_rics[midpoint:],
                            *pending_chunks,
                        ]
                    else:
                        status = _record_chunk_error(
                            status,
                            request_rics,
                            exc,
                            attempt_id=attempt_id,
                            observed_at_utc=observed_at,
                        )

            quality = _build_quality(queue, status, normalized)
            manifest = _persist_state(
                paths=paths,
                run_id=resolved_run_id,
                source_dir=source,
                source_paths={
                    "active_operating_core": active_path,
                    "historical_listed_reference": reference_path,
                },
                source_hashes=source_hashes,
                queue=queue,
                raw_events=raw_events,
                normalized=normalized,
                status=status,
                quality=quality,
                max_rics=max_rics,
                request_chunk_size=request_chunk_size,
                selected_rics=selected_rics,
                attempted_rics=invocation_attempted,
                invocation_failure=invocation_failure,
                session_name=session_name,
                generated_at_utc=observed_at,
            )
            if invocation_failure:
                break
            if sleep_seconds > 0 and len(invocation_attempted) < len(selected_rics):
                time.sleep(sleep_seconds)

        if not invocation_attempted:
            generated_at = _as_utc_iso(now())
            quality = _build_quality(queue, status, normalized)
            manifest = _persist_state(
                paths=paths,
                run_id=resolved_run_id,
                source_dir=source,
                source_paths={
                    "active_operating_core": active_path,
                    "historical_listed_reference": reference_path,
                },
                source_hashes=source_hashes,
                queue=queue,
                raw_events=raw_events,
                normalized=normalized,
                status=status,
                quality=quality,
                max_rics=max_rics,
                request_chunk_size=request_chunk_size,
                selected_rics=selected_rics,
                attempted_rics=invocation_attempted,
                invocation_failure=None,
                session_name=session_name,
                generated_at_utc=generated_at,
            )

        if source_hashes != {
            "active_operating_core": _sha256(active_path),
            "historical_listed_reference": _sha256(reference_path),
        }:
            raise RuntimeError("Japan G1 source universes changed during enrichment")

        return JapanG1EnrichmentArtifacts(
            run_id=resolved_run_id,
            artifact_dir=artifact_dir,
            queue_path=paths["queue"],
            raw_responses_path=paths["raw_responses"],
            normalized_responses_path=paths["normalized_responses"],
            ric_status_path=paths["ric_status"],
            quality_path=paths["quality"],
            checksums_path=paths["checksums"],
            manifest_path=paths["manifest"],
            status=status.reset_index(drop=True),
            quality=quality,
            manifest=manifest,
        )
    finally:
        lock_handle.close()


run_japan_g1_enrichment = write_japan_g1_enrichment_artifacts


def _artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "queue": artifact_dir / "source_universe_queue.csv",
        "raw_responses": artifact_dir / "raw_responses.jsonl",
        "normalized_responses": artifact_dir / "normalized_responses.csv",
        "ric_status": artifact_dir / "ric_status.csv",
        "quality": artifact_dir / "quality_metrics.csv",
        "checksums": artifact_dir / "checksums.sha256",
        "manifest": artifact_dir / "manifest.json",
    }


def _require_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required Japan G1 universe CSV is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"required Japan G1 universe CSV is empty: {path}")


def _build_queue(active_path: Path, reference_path: Path) -> pd.DataFrame:
    active = pd.read_csv(active_path, dtype=object)
    reference = pd.read_csv(reference_path, dtype=object)
    active_rics = _universe_rics(active, active_path)
    reference_rics = _universe_rics(reference, reference_path)
    active_set = set(active_rics)
    reference_set = set(reference_rics)
    ordered = list(dict.fromkeys([*active_rics, *reference_rics]))
    if not ordered:
        raise ValueError("Japan G1 source universes contain no usable RICs")
    return pd.DataFrame(
        {
            "ric": ordered,
            "in_active_universe": [ric in active_set for ric in ordered],
            "in_reference_universe": [ric in reference_set for ric in ordered],
            "source_scope": [
                "active_and_reference"
                if ric in active_set and ric in reference_set
                else "active"
                if ric in active_set
                else "reference"
                for ric in ordered
            ],
            "source_order": range(1, len(ordered) + 1),
        }
    )


def _universe_rics(frame: pd.DataFrame, path: Path) -> list[str]:
    column = "refinitiv_ric" if "refinitiv_ric" in frame.columns else "ric" if "ric" in frame.columns else None
    if column is None:
        raise ValueError(f"Japan G1 source CSV has no ric/refinitiv_ric column: {path}")
    values = frame[column].dropna().astype(str).str.strip()
    values = values.loc[values.ne("") & values.str.casefold().ne("nan")]
    return list(dict.fromkeys(values.tolist()))


def _verify_or_write_queue(path: Path, queue: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        left = existing.astype(object).where(pd.notna(existing), None).to_dict("records")
        right = queue.astype(object).where(pd.notna(queue), None).to_dict("records")
        if left != right:
            raise RuntimeError("persisted Japan G1 queue does not match immutable source universes")
        return
    _atomic_write_csv(queue, path)


def _load_status(path: Path, queue: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        status = pd.read_csv(path, dtype=object)
        missing = set(STATUS_COLUMNS).difference(status.columns)
        if missing:
            raise RuntimeError(f"persisted Japan G1 status is missing columns: {sorted(missing)}")
        if status["ric"].astype(str).tolist() != queue["ric"].astype(str).tolist():
            raise RuntimeError("persisted Japan G1 status RICs do not match the source queue")
        status["attempt_count"] = pd.to_numeric(status["attempt_count"], errors="raise").astype(int)
        status["source_order"] = pd.to_numeric(status["source_order"], errors="raise").astype(int)
        return status[list(STATUS_COLUMNS)]

    status = queue.copy()
    status["status"] = "pending"
    status["attempt_count"] = 0
    status["last_attempt_id"] = ""
    status["observed_at_utc"] = ""
    status["response_rows"] = 0
    status["fields_present_count"] = 0
    status["error_type"] = ""
    status["error_message"] = ""
    status["retry_eligible"] = True
    status["historical_pit_claimed"] = False
    status["eligible_for_historical_pit"] = False
    status["eligible_for_alpha_promotion"] = False
    status["eligible_for_production_promotion"] = False
    return status[list(STATUS_COLUMNS)]


def _resume_candidates(status: pd.DataFrame) -> pd.DataFrame:
    remaining = status.loc[~status["status"].astype(str).eq("success")].copy()
    remaining["retry_rank"] = pd.to_numeric(remaining["attempt_count"], errors="coerce").fillna(0).gt(0).astype(int)
    return remaining.sort_values(["retry_rank", "source_order"], kind="stable").drop(columns="retry_rank")


def _resolve_getter(client: Any) -> Callable[..., Any]:
    getter = getattr(client, "get_data", None)
    if getter is None and callable(client):
        getter = client
    if getter is None:
        raise TypeError("get_data_client must expose get_data(universe=..., fields=...) or be callable")
    return getter


def _is_throttle_error(error: Exception) -> bool:
    message = str(error).casefold()
    return "429" in message or "too many requests" in message or "rate limit" in message


def _response_frame(response: Any) -> pd.DataFrame:
    if isinstance(response, tuple) and response:
        response = response[0]
    if response is None:
        return pd.DataFrame()
    if isinstance(response, pd.DataFrame):
        return response.copy()
    return pd.DataFrame(response)


def _apply_japan_currency_fallback(
    normalized: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    """Use the entitlement-supported realtime currency field when TR.PriceCurrency is denied."""

    if normalized.empty or raw.empty:
        return normalized
    currency_column = next(
        (column for column in ("CF_CURR", "Currency", "TR.PriceClose.currency") if column in raw.columns),
        None,
    )
    if currency_column is None:
        return normalized
    ric_column = next(
        (column for column in ("Instrument", "RIC", "ric") if column in raw.columns),
        None,
    )
    if ric_column is None:
        return normalized
    currency_by_ric = (
        raw[[ric_column, currency_column]]
        .dropna(subset=[ric_column])
        .drop_duplicates(ric_column, keep="first")
        .set_index(ric_column)[currency_column]
    )
    result = normalized.copy()
    fallback = result["ric"].map(currency_by_ric)
    current = result["price_currency"].astype(object)
    result["price_currency"] = current.where(current.notna(), fallback.astype(object))
    return result


def _decorate_normalized(
    frame: pd.DataFrame,
    *,
    attempt_id: str,
    observed_at_utc: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["attempt_id"] = attempt_id
    result["requested_ric"] = result.get("ric", pd.Series(dtype=object)).astype("string")
    result["observed_at_utc"] = observed_at_utc
    result["evidence_scope"] = "refinitiv_current_observation_only"
    result["historical_pit_claimed"] = False
    result["eligible_for_historical_pit"] = False
    result["eligible_for_alpha_promotion"] = False
    result["eligible_for_production_promotion"] = False
    for column in NORMALIZED_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[list(NORMALIZED_COLUMNS)].reset_index(drop=True)


def _record_chunk_status(
    status: pd.DataFrame,
    requested_rics: list[str],
    normalized: pd.DataFrame,
    *,
    attempt_id: str,
    observed_at_utc: str,
) -> tuple[pd.DataFrame, list[str]]:
    result = status.copy()
    normalized_ric = normalized["ric"].astype("string") if "ric" in normalized.columns else pd.Series(dtype="string")
    empty_rics: list[str] = []
    for ric in requested_rics:
        rows = normalized.loc[normalized_ric.eq(ric)].copy()
        payload_rows = rows.loc[rows[list(TARGET_ENRICHMENT_COLUMNS)].notna().any(axis=1)] if not rows.empty else rows
        mask = result["ric"].astype(str).eq(ric)
        result.loc[mask, "attempt_count"] = int(result.loc[mask, "attempt_count"].iloc[0]) + 1
        result.loc[mask, "last_attempt_id"] = attempt_id
        result.loc[mask, "observed_at_utc"] = observed_at_utc
        result.loc[mask, "response_rows"] = len(rows)
        result.loc[mask, "fields_present_count"] = (
            int(rows[list(TARGET_ENRICHMENT_COLUMNS)].notna().any(axis=0).sum()) if not rows.empty else 0
        )
        result.loc[mask, ["error_type", "error_message"]] = ""
        if payload_rows.empty:
            empty_rics.append(ric)
            result.loc[mask, "status"] = "failed_empty"
            result.loc[mask, "error_type"] = "EmptyResponse"
            result.loc[mask, "error_message"] = "no normalized enrichment values returned for requested RIC"
            result.loc[mask, "retry_eligible"] = True
        else:
            result.loc[mask, "status"] = "success"
            result.loc[mask, "retry_eligible"] = False
    return result[list(STATUS_COLUMNS)], empty_rics


def _record_chunk_error(
    status: pd.DataFrame,
    requested_rics: list[str],
    error: Exception,
    *,
    attempt_id: str,
    observed_at_utc: str,
) -> pd.DataFrame:
    result = status.copy()
    for ric in requested_rics:
        mask = result["ric"].astype(str).eq(ric)
        result.loc[mask, "attempt_count"] = int(result.loc[mask, "attempt_count"].iloc[0]) + 1
        result.loc[mask, "last_attempt_id"] = attempt_id
        result.loc[mask, "observed_at_utc"] = observed_at_utc
        result.loc[mask, "response_rows"] = 0
        result.loc[mask, "fields_present_count"] = 0
        result.loc[mask, "status"] = "error"
        result.loc[mask, "error_type"] = type(error).__name__
        result.loc[mask, "error_message"] = str(error)[:2000]
        result.loc[mask, "retry_eligible"] = True
    return result[list(STATUS_COLUMNS)]


def _raw_event(
    frame: pd.DataFrame,
    requested_rics: list[str],
    *,
    attempt_id: str,
    observed_at_utc: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    raw = frame.copy()
    raw.columns = [_column_name(column) for column in raw.columns]
    if not isinstance(raw.index, pd.RangeIndex):
        raw = raw.reset_index()
        raw.columns = [_column_name(column) for column in raw.columns]
    rows = [_json_safe(record) for record in raw.to_dict("records")]
    return {
        "attempt_id": attempt_id,
        "requested_rics": list(requested_rics),
        "fields": list(JAPAN_G1_ENRICHMENT_FIELDS),
        "observed_at_utc": observed_at_utc,
        "raw_row_count": int(len(raw)),
        "raw_rows": rows,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error)[:2000] if error else None,
        "evidence_scope": "refinitiv_current_observation_only",
        "historical_pit_claimed": False,
    }


def _build_quality(queue: pd.DataFrame, status: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    latest = normalized.copy()
    if not latest.empty:
        latest = latest.drop_duplicates("requested_ric", keep="last").set_index("requested_ric", drop=False)
    rows: list[dict[str, Any]] = []
    scopes = (
        ("active", status["in_active_universe"].map(_as_bool)),
        ("reference", status["in_reference_universe"].map(_as_bool)),
        ("unique", pd.Series(True, index=status.index)),
    )
    for scope, mask in scopes:
        scoped = status.loc[mask].copy()
        success_rics = scoped.loc[scoped["status"].astype(str).eq("success"), "ric"].astype(str).tolist()
        enriched = latest.reindex(success_rics) if success_rics and not latest.empty else pd.DataFrame()
        total = len(scoped)
        succeeded = int(scoped["status"].astype(str).eq("success").sum())
        row: dict[str, Any] = {
            "scope": scope,
            "source_rows": int(total),
            "successful_rics": succeeded,
            "failed_empty_rics": int(scoped["status"].astype(str).eq("failed_empty").sum()),
            "error_rics": int(scoped["status"].astype(str).eq("error").sum()),
            "never_attempted_rics": int(scoped["status"].astype(str).eq("pending").sum()),
            "current_observation_coverage": round(succeeded / total, 8) if total else 0.0,
            "quality_status": "complete_current_observation" if total and succeeded == total else "incomplete",
            "historical_pit_eligible": False,
            "alpha_promotion_allowed": False,
            "production_promotion_allowed": False,
        }
        for column in TARGET_ENRICHMENT_COLUMNS:
            covered = int(enriched[column].notna().sum()) if not enriched.empty else 0
            row[f"{column}_coverage"] = round(covered / succeeded, 8) if succeeded else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _persist_state(
    *,
    paths: dict[str, Path],
    run_id: str,
    source_dir: Path,
    source_paths: dict[str, Path],
    source_hashes: dict[str, str],
    queue: pd.DataFrame,
    raw_events: list[dict[str, Any]],
    normalized: pd.DataFrame,
    status: pd.DataFrame,
    quality: pd.DataFrame,
    max_rics: int,
    request_chunk_size: int,
    selected_rics: list[str],
    attempted_rics: list[str],
    invocation_failure: str | None,
    session_name: str | None,
    generated_at_utc: str,
) -> dict[str, Any]:
    _atomic_write_csv(queue, paths["queue"])
    _atomic_write_json_lines(raw_events, paths["raw_responses"])
    _atomic_write_csv(normalized[list(NORMALIZED_COLUMNS)], paths["normalized_responses"])
    _atomic_write_csv(status[list(STATUS_COLUMNS)], paths["ric_status"])
    _atomic_write_csv(quality, paths["quality"])

    checksummed = {
        key: path
        for key, path in paths.items()
        if key in {"queue", "raw_responses", "normalized_responses", "ric_status", "quality"}
    }
    artifact_sha256 = {key: _sha256(path) for key, path in checksummed.items()}
    checksum_text = "".join(
        f"{artifact_sha256[key]}  {path.name}\n" for key, path in sorted(checksummed.items())
    )
    _atomic_write_text(paths["checksums"], checksum_text)
    artifact_sha256["checksums"] = _sha256(paths["checksums"])

    counts = status["status"].astype(str).value_counts().to_dict()
    success = int(counts.get("success", 0))
    total = int(len(status))
    unresolved_errors = int(counts.get("error", 0)) + int(counts.get("failed_empty", 0))
    manifest_status = invocation_failure or (
        "complete_current_observation"
        if total > 0 and success == total
        else "bounded_in_progress_with_retryable_errors"
        if unresolved_errors
        else "bounded_in_progress"
    )
    observed = status["observed_at_utc"].dropna().astype(str)
    observed = observed.loc[observed.ne("") & observed.str.casefold().ne("nan")]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": manifest_status,
        "artifact_dir": str(paths["manifest"].parent.resolve()),
        "source_universe_dir": str(source_dir.resolve()),
        "source_files": {key: str(path.resolve()) for key, path in source_paths.items()},
        "input_sha256": source_hashes,
        "input_checksums_verified_for_resume": True,
        "input_checksums_verified_after_invocation": True,
        "fields": list(JAPAN_G1_ENRICHMENT_FIELDS),
        "target_normalized_fields": list(TARGET_ENRICHMENT_COLUMNS),
        "session_name": session_name,
        "bounds": {
            "max_rics": int(max_rics),
            "request_chunk_size": int(request_chunk_size),
            "selected_rics": int(len(selected_rics)),
            "attempted_rics": int(len(attempted_rics)),
        },
        "counts": {
            "source_unique_rics": total,
            "success": success,
            "failed_empty": int(counts.get("failed_empty", 0)),
            "error": int(counts.get("error", 0)),
            "never_attempted": int(counts.get("pending", 0)),
            "retry_eligible": int(status["retry_eligible"].map(_as_bool).sum()),
            "remaining_without_success": int(total - success),
            "raw_request_events": int(len(raw_events)),
            "normalized_response_rows": int(len(normalized)),
        },
        "resume": {
            "resumable": True,
            "successful_rics_are_skipped": True,
            "never_attempted_rics_are_prioritized": True,
            "failed_rics_remain_retry_eligible": True,
            "next_source_order": _next_source_order(status),
        },
        "observed_at_utc_min": observed.min() if not observed.empty else None,
        "observed_at_utc_max": observed.max() if not observed.empty else None,
        "evidence_scope": "refinitiv_current_observation_only",
        "current_observation_only": True,
        "does_not_assert_historical_pit": True,
        "historical_pit_ready": False,
        "historical_backtest_eligible": False,
        "point_in_time_membership_claimed": False,
        "alpha_promotion_allowed": False,
        "production_promotion_allowed": False,
        "ibkr_promotion_allowed": False,
        "live_trading_allowed": False,
        "promotion_policy": {
            "alpha": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
        "pit_warning": (
            "Currency, first-trade, retire-date, and identifier values were observed from the current "
            "Refinitiv get_data route. They are not historical point-in-time observations and do not "
            "establish historical membership, identifier validity windows, or source availability."
        ),
        "fail_closed_on_empty_response": True,
        "single_writer_lock": True,
        "atomic_writes": True,
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
        "artifact_sha256": artifact_sha256,
        "generated_at_utc": generated_at_utc,
    }
    _atomic_write_json(manifest, paths["manifest"])
    return manifest


def _next_source_order(status: pd.DataFrame) -> int | None:
    candidates = _resume_candidates(status)
    return int(candidates.iloc[0]["source_order"]) if not candidates.empty else None


def _verify_resume_sources(manifest: dict[str, Any], current: dict[str, str]) -> None:
    expected = manifest.get("input_sha256") if manifest else None
    if expected is not None and expected != current:
        raise RuntimeError("Japan G1 source universe checksum changed; refusing unsafe resume")


def _acquire_run_lock(artifact_dir: Path):
    lock_path = artifact_dir / ".enrichment.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another Japan G1 enrichment writer owns the lock: {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=object)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[list(columns)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _atomic_write_json_lines(rows: list[dict[str, Any]], path: Path) -> None:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _column_name(value: Any) -> str:
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value if str(item))
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}
