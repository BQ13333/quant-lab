from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quant_lab.data.japan_g4_halt_report_events import verify_report_event_artifact
from quant_lab.data.japan_g4_halt_report_events_validator import (
    verify_report_event_validation_artifact,
)
from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    NORMALIZED_DISPOSITION,
    normalization_repair_for,
)


JST = ZoneInfo("Asia/Tokyo")
HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
RESEARCH_INSTRUMENT_CLASS = "equity"

REPORT_ADMISSION_COLUMNS = (
    "report_event_id",
    "source_record_sha256",
    "report_month",
    "channel",
    "instrument_class",
    "raw_code",
    "local_code",
    "event_date",
    "source_sha256",
    "claim_exact_set_validated",
    "in_research_scope",
    "normalization_status",
    "primary_event_id",
    "mapping_complete",
    "timestamp_complete",
    "admission_ready",
)


@dataclass(frozen=True)
class ValidatedHaltReportPacket:
    events: pd.DataFrame
    claims: pd.DataFrame
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]


def copy_validated_halt_report_packet(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    atomic_write: Callable[[Path, bytes], None],
) -> ValidatedHaltReportPacket:
    producer = Path(producer_dir).resolve()
    validation = Path(validation_dir).resolve()
    target = Path(destination).resolve()
    producer_issues = verify_report_event_artifact(producer)
    validation_issues = verify_report_event_validation_artifact(validation)
    if producer_issues or validation_issues:
        raise ValueError(
            "halt report packet verification failed: "
            + "|".join(producer_issues + validation_issues)
        )

    producer_manifest_path = producer / "manifest.json"
    validation_manifest_path = validation / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    validation_manifest = _json(validation_manifest_path)
    producer_sha = _sha256(producer_manifest_path)
    validation_sha = _sha256(validation_manifest_path)
    expected_rows = int(producer_manifest.get("explicit_report_rows", -1))
    expected_claims = int(producer_manifest.get("covered_month_channel_claims", -1))
    contract = (
        producer_manifest.get("status")
        == "bounded_report_event_candidate_pending_independent_validation"
        and validation_manifest.get("status")
        == "pass_bounded_report_event_component"
        and validation_manifest.get("producer_manifest_sha256") == producer_sha
        and int(validation_manifest.get("validation_issue_count", -1)) == 0
        and validation_manifest.get("all_report_hashes_verified") is True
        and validation_manifest.get("all_input_hashes_verified") is True
        and validation_manifest.get("absence_is_zero") is False
        and validation_manifest.get("inferred_zero_rejected") is True
        and expected_rows > 0
        and expected_claims > 0
        and int(validation_manifest.get("producer_report_rows", -1)) == expected_rows
        and int(validation_manifest.get("independent_report_rows", -1)) == expected_rows
        and int(validation_manifest.get("exact_field_reconciled_rows", -1)) == expected_rows
        and int(validation_manifest.get("covered_claims_reparsed", -1)) == expected_claims
        and int(validation_manifest.get("exact_claim_row_sets", -1)) == expected_claims
        and producer_manifest.get("denominator_manifest_sha256")
        == validation_manifest.get("denominator_manifest_sha256")
        and producer_manifest.get("validation_manifest_sha256")
        == validation_manifest.get("denominator_validation_manifest_sha256")
    )
    if not contract:
        raise ValueError("halt report packet independent validation contract failed")

    copied: list[Path] = []
    copied.extend(
        _copy_sealed_packet(producer, target / "producer", atomic_write=atomic_write)
    )
    copied.extend(
        _copy_sealed_packet(validation, target / "validation", atomic_write=atomic_write)
    )
    producer_copy = target / "producer" / "manifest.json"
    validation_copy = target / "validation" / "manifest.json"
    if _sha256(producer_copy) != producer_sha or _sha256(validation_copy) != validation_sha:
        raise ValueError("halt report packet manifest copy mismatch")

    events = pd.read_parquet(target / "producer" / "report_events.parquet").fillna("")
    claims = pd.read_csv(
        target / "validation" / "claim_exact_row_set_reconciliation.csv",
        dtype=str,
    ).fillna("")
    if (
        len(events) != expected_rows
        or events["source_record_sha256"].duplicated().any()
        or len(claims) != expected_claims
        or not _truths(claims["exact_nonempty_row_set"]).all()
        or _truths(claims["zero_inferred"]).any()
    ):
        raise ValueError("halt report packet exact event denominator is invalid")
    claim_keys = set(zip(claims["report_month"], claims["channel"], strict=False))
    event_keys = set(zip(events["report_month"], events["channel"], strict=False))
    if claim_keys != event_keys:
        raise ValueError("halt report claim and event sets differ")

    lineage = {
        "producer_run_id": producer_manifest.get("run_id"),
        "producer_manifest_sha256": producer_sha,
        "validation_run_id": validation_manifest.get("run_id"),
        "validation_manifest_sha256": validation_sha,
        "denominator_manifest_sha256": producer_manifest.get(
            "denominator_manifest_sha256"
        ),
        "denominator_validation_manifest_sha256": producer_manifest.get(
            "validation_manifest_sha256"
        ),
        "report_rows": expected_rows,
        "covered_claims": expected_claims,
        "exact_claim_row_sets": int(validation_manifest["exact_claim_row_sets"]),
        "absence_is_zero": False,
        "research_instrument_class": RESEARCH_INSTRUMENT_CLASS,
    }
    lineage_path = target / "lineage.json"
    atomic_write(
        lineage_path,
        (json.dumps(lineage, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    copied.append(lineage_path)
    return ValidatedHaltReportPacket(events, claims, tuple(copied), lineage)


def normalize_halt_report_events(
    report_events: pd.DataFrame,
    *,
    event_columns: tuple[str, ...],
    apply_bounded_repairs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize report rows, optionally applying the sealed ten-record repair set."""
    normalized: list[dict[str, Any]] = []
    admission: list[dict[str, Any]] = []
    for original_source in report_events.to_dict("records"):
        source = dict(original_source)
        repair = normalization_repair_for(source) if apply_bounded_repairs else None
        if repair is not None and repair.normalization_disposition == NORMALIZED_DISPOSITION:
            source["halt_start_printed"] = repair.repaired_halt_start_printed
            source["halt_end_printed"] = repair.repaired_halt_end_printed
            source["halt_start_jst"] = repair.repaired_halt_start_jst
            source["halt_end_jst"] = repair.repaired_halt_end_jst
        report_month = str(source["report_month"])
        channel = str(source["channel"])
        event_date = str(source["event_date"])
        instrument_class = str(source["instrument_class"])
        local_code = str(source["normalized_local_code"])
        in_scope = instrument_class == RESEARCH_INSTRUMENT_CLASS and not (
            repair is not None and not repair.in_research_scope_after
        )
        status = "out_of_research_universe"
        primary_id = ""
        row: dict[str, Any] | None = None
        month_end = pd.Period(report_month, freq="M").end_time.date()
        future_dated = date.fromisoformat(event_date) > month_end
        if repair is not None and not repair.in_research_scope_after:
            status = repair.normalization_disposition
        elif in_scope and not local_code:
            status = "missing_local_code"
        elif in_scope and future_dated:
            status = "future_dated_source_record"
        elif in_scope and channel == HALT_CHANNEL and not str(source["halt_start_jst"]):
            status = "missing_halt_start"
        elif in_scope:
            primary_id = "jpx-report-" + str(source["event_id"])
            status = "normalized"
            start = (
                str(source["halt_start_jst"])
                if channel == HALT_CHANNEL
                else datetime.combine(date.fromisoformat(event_date), time(), JST).isoformat()
            )
            end = str(source["halt_end_jst"]) if channel == HALT_CHANNEL else ""
            event_type = (
                "trading_halt"
                if channel == HALT_CHANNEL
                else "delisting_non_trading_interval"
            )
            row = {
                "event_id": primary_id,
                "event_type": event_type,
                "exchange": "TSE",
                "local_code": local_code,
                "issue_name": str(source["issue_name"]),
                "event_date": event_date,
                "event_start_jst": start,
                "suspension_end_jst": end,
                "resumption_jst": end,
                "interval_end_jst": end,
                "known_at_utc": "",
                "known_at_status": "pending_report_event_pit_resolution",
                "source_observed_at_utc": "",
                "known_at_evidence_sha256": "",
                "reason": (
                    "official JPX monthly report row; sealed-byte wrapped-time normalization repair"
                    if repair is not None
                    else "official JPX monthly report row"
                ),
                "source_id": f"jpx-monthly-report-{report_month}-{channel}",
                "source_url": str(source["source_url"]),
                "source_sha256": str(source["source_sha256"]),
                "source_record_id": str(source["source_record_sha256"]),
                "stable_listing_id": "",
                "refinitiv_ric": "",
                "mapping_known_at": "",
                "mapping_status": "unmapped",
                "mapping_route": "",
                "mapping_source_sha256": "",
                "mapping_source_record_id": "",
            }
            normalized.append(row)
        admission.append(
            {
                "report_event_id": str(source["event_id"]),
                "source_record_sha256": str(source["source_record_sha256"]),
                "report_month": report_month,
                "channel": channel,
                "instrument_class": instrument_class,
                "raw_code": str(source["raw_code"]),
                "local_code": local_code,
                "event_date": event_date,
                "source_sha256": str(source["source_sha256"]),
                "claim_exact_set_validated": True,
                "in_research_scope": in_scope,
                "normalization_status": status,
                "primary_event_id": primary_id,
                "mapping_complete": False,
                "timestamp_complete": False,
                "admission_ready": False,
            }
        )
    events = pd.DataFrame(normalized, columns=event_columns)
    admissions = pd.DataFrame(admission, columns=REPORT_ADMISSION_COLUMNS)
    if admissions["source_record_sha256"].duplicated().any():
        raise ValueError("halt report admission contains duplicate source records")
    if len(events) and events["event_id"].duplicated().any():
        raise ValueError("halt report normalized events contain duplicate IDs")
    return events, admissions


def update_halt_report_admission(
    admission: pd.DataFrame,
    mapped_events: pd.DataFrame,
    *,
    pit_mapping_statuses: frozenset[str],
    pit_timestamp_statuses: frozenset[str],
) -> pd.DataFrame:
    by_id = (
        mapped_events.set_index("event_id", drop=False).to_dict("index")
        if len(mapped_events)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for row in admission.to_dict("records"):
        event = by_id.get(row["primary_event_id"])
        mapping = bool(event and event["mapping_status"] in pit_mapping_statuses)
        timestamp = bool(event and event["known_at_status"] in pit_timestamp_statuses)
        row["mapping_complete"] = mapping
        row["timestamp_complete"] = timestamp
        row["admission_ready"] = bool(
            _truth(row["in_research_scope"])
            and row["normalization_status"] == "normalized"
            and mapping
            and timestamp
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=REPORT_ADMISSION_COLUMNS)


def covered_report_claims(admission: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(admission["report_month"], admission["channel"], strict=False))


def _copy_sealed_packet(
    source: Path,
    destination: Path,
    *,
    atomic_write: Callable[[Path, bytes], None],
) -> list[Path]:
    manifest_path = source / "manifest.json"
    inventory_path = source / "artifact_hashes.json"
    manifest = _json(manifest_path)
    if _sha256(inventory_path) != manifest.get("artifact_hashes_sha256"):
        raise ValueError("halt report packet inventory digest mismatch")
    declared = _json(inventory_path)
    copied: list[Path] = []
    for relative_name, digest in declared.items():
        relative = Path(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("halt report packet inventory path is unsafe")
        source_path = source / relative
        if not source_path.is_file() or _sha256(source_path) != str(digest):
            raise ValueError(f"halt report packet hash mismatch: {relative_name}")
        target = destination / relative
        atomic_write(target, source_path.read_bytes())
        copied.append(target)
    for name in ("manifest.json", "artifact_hashes.json"):
        target = destination / name
        atomic_write(target, (source / name).read_bytes())
        copied.append(target)
    expected = set(declared) | {"manifest.json", "artifact_hashes.json"}
    actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("halt report packet contains undeclared files")
    return copied


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def _truths(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
