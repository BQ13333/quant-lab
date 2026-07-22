from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


DEFAULT_JPX_RECONCILIATION_DIR = Path(
    "artifacts/research_universes/JPX-UNIVERSE-RECONCILIATION-V2-20260711"
)
DEFAULT_G1_ENRICHMENT_DIR = Path(
    "artifacts/japan_g1_enrichment/"
    "JPX-UNIVERSE-RECONCILIATION-V2-20260711-G1-CURRENT-ENRICHMENT-V2"
)
DEFAULT_JAPAN_LISTING_VALID_TIME_ARTIFACT_ROOT = Path(
    "artifacts/japan_listing_valid_time"
)

ACTIVE_MAP_FILE = "japan_jpx_active_operating_core_v1.csv"
REFERENCE_MAP_FILE = "japan_jpx_historical_listed_reference_v1.csv"
NORMALIZED_FILE = "normalized_responses.csv"
STATUS_FILE = "ric_status.csv"
INPUT_CHECKSUMS_FILE = "checksums.sha256"

LISTING_INTERVAL_COLUMNS: tuple[str, ...] = (
    "stable_instrument_id",
    "stable_identifier_type",
    "isin",
    "exchange_code",
    "listing_start",
    "retire_date",
    "unknown_start",
    "unknown_end",
    "boundary_state",
    "scheduled_retirement",
    "retired_as_of",
    "current_active_map_member",
    "historical_reference_map_member",
    "ric_count",
    "rics",
    "as_of_date",
    "known_at_utc",
    "known_at_source",
    "valid_time_source",
    "boundary_convention",
)

RIC_QUOTE_INTERVAL_COLUMNS: tuple[str, ...] = (
    "stable_instrument_id",
    "stable_identifier_type",
    "isin",
    "ric",
    "local_code",
    "exchange_code",
    "listing_start",
    "retire_date",
    "unknown_start",
    "unknown_end",
    "boundary_state",
    "scheduled_retirement",
    "retired_as_of",
    "current_active_map_member",
    "historical_reference_map_member",
    "as_of_date",
    "known_at_utc",
    "known_at_source",
    "listing_start_source",
    "retire_date_source",
    "date_evidence_conflict",
)

CURRENT_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "stable_instrument_id",
    "ric",
    "attribute_name",
    "attribute_value",
    "valid_from",
    "valid_to",
    "unknown_start",
    "unknown_end",
    "undated_current_observation",
    "observed_at_utc",
    "known_at_utc",
    "known_at_source",
    "pit_usable",
    "pit_block_reason",
    "evidence_scope",
)

VALID_TIME_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "stable_instrument_id",
    "exchange_code",
    "membership_start",
    "membership_end",
    "unknown_start",
    "unknown_end",
    "membership_state",
    "current_active_map_member",
    "historical_reference_map_member",
    "as_of_date",
    "membership_basis",
)

PIT_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "stable_instrument_id",
    "exchange_code",
    "membership_start",
    "membership_end",
    "membership_state",
    "known_at_utc",
    "known_at_source",
    "pit_usable",
    "pit_block_reason",
    "as_of_date",
    "membership_basis",
)


@dataclass(frozen=True)
class JapanListingValidTimeTables:
    listing_intervals: pd.DataFrame
    ric_quote_intervals: pd.DataFrame
    current_observations: pd.DataFrame
    valid_time_membership: pd.DataFrame
    pit_membership: pd.DataFrame
    quality: pd.DataFrame
    as_of_date: str


@dataclass(frozen=True)
class JapanListingValidTimeArtifacts:
    run_id: str
    artifact_dir: Path
    listing_intervals_path: Path
    ric_quote_intervals_path: Path
    current_observations_path: Path
    valid_time_membership_path: Path
    pit_membership_path: Path
    quality_path: Path
    input_files_path: Path
    checksums_path: Path
    manifest_path: Path
    tables: JapanListingValidTimeTables
    manifest: dict[str, Any]


def default_japan_listing_valid_time_run_id(
    reconciliation_dir: str | Path = DEFAULT_JPX_RECONCILIATION_DIR,
) -> str:
    return f"{Path(reconciliation_dir).name}-LISTING-VALID-TIME-BASELINE"


def build_japan_listing_valid_time_tables(
    active_map: pd.DataFrame,
    reference_map: pd.DataFrame,
    normalized: pd.DataFrame,
    status: pd.DataFrame,
    *,
    as_of_date: str | date | datetime | None = None,
) -> JapanListingValidTimeTables:
    """Build a valid-time baseline without converting current snapshots into PIT history."""

    for name, frame in (("active map", active_map), ("reference map", reference_map)):
        _require_columns(
            frame,
            {"ric", "stable_instrument_id", "exchange_code", "is_direct_jpx_quote"},
            name,
        )
    _require_columns(normalized, {"ric", "observed_at_utc"}, "normalized enrichment")
    _require_columns(status, {"ric", "status", "observed_at_utc"}, "RIC status")

    latest = _latest_normalized(normalized)
    baseline_date = _resolve_as_of_date(as_of_date, latest, status)
    direct_rics = _direct_tyo_rics(active_map) | _direct_tyo_rics(reference_map)
    _verify_completed_rows(direct_rics, latest, status)

    enrich_by_ric = {
        _text(row.get("ric")): row for row in latest.to_dict("records")
    }
    quote_records = _quote_records(
        active_map, reference_map, enrich_by_ric, baseline_date
    )
    quote_intervals = pd.DataFrame(quote_records, columns=RIC_QUOTE_INTERVAL_COLUMNS)
    quote_intervals = quote_intervals.sort_values(
        ["stable_instrument_id", "ric"], kind="stable"
    ).reset_index(drop=True)

    listing_records = [
        _listing_record(group, baseline_date)
        for _, group in quote_intervals.groupby(
            ["stable_instrument_id", "exchange_code"], sort=True, dropna=False
        )
    ]
    listing_intervals = pd.DataFrame(
        listing_records, columns=LISTING_INTERVAL_COLUMNS
    ).sort_values(["stable_instrument_id", "exchange_code"], kind="stable")
    listing_intervals = listing_intervals.reset_index(drop=True)

    observations = _current_observations(latest, quote_intervals)
    valid_time_membership = _valid_time_membership(listing_intervals)
    pit_membership = _pit_membership(listing_intervals)
    quality = _quality_metrics(
        listing_intervals,
        quote_intervals,
        observations,
        valid_time_membership,
        pit_membership,
    )
    _validate_pit_invariants(observations, pit_membership)
    return JapanListingValidTimeTables(
        listing_intervals=listing_intervals,
        ric_quote_intervals=quote_intervals,
        current_observations=observations,
        valid_time_membership=valid_time_membership,
        pit_membership=pit_membership,
        quality=quality,
        as_of_date=baseline_date.isoformat(),
    )


def write_japan_listing_valid_time_artifacts(
    *,
    reconciliation_dir: str | Path = DEFAULT_JPX_RECONCILIATION_DIR,
    enrichment_dir: str | Path = DEFAULT_G1_ENRICHMENT_DIR,
    artifact_root: str | Path = DEFAULT_JAPAN_LISTING_VALID_TIME_ARTIFACT_ROOT,
    run_id: str | None = None,
    as_of_date: str | date | datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> JapanListingValidTimeArtifacts:
    """Validate inputs and atomically publish one immutable baseline artifact."""

    reconciliation = Path(reconciliation_dir)
    enrichment = Path(enrichment_dir)
    source_paths = {
        "active_map": reconciliation / ACTIVE_MAP_FILE,
        "reference_map": reconciliation / REFERENCE_MAP_FILE,
        "normalized": enrichment / NORMALIZED_FILE,
        "status": enrichment / STATUS_FILE,
        "enrichment_checksums": enrichment / INPUT_CHECKSUMS_FILE,
        "enrichment_manifest": enrichment / "manifest.json",
    }
    source_manifest_path = reconciliation / "manifest.json"
    if source_manifest_path.is_file():
        source_paths["reconciliation_manifest"] = source_manifest_path
    for path in source_paths.values():
        _require_source(path)

    resolved_run_id = run_id or default_japan_listing_valid_time_run_id(
        reconciliation
    )
    if not resolved_run_id.strip() or Path(resolved_run_id).name != resolved_run_id:
        raise ValueError("run_id must be a non-empty path-safe name")

    root = Path(artifact_root)
    final_dir = root / resolved_run_id
    if final_dir.exists():
        raise FileExistsError(f"immutable artifact directory already exists: {final_dir}")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{resolved_run_id}.lock"
    lock_fd = _acquire_lock(lock_path)
    stage_dir: Path | None = None
    try:
        if final_dir.exists():
            raise FileExistsError(
                f"immutable artifact directory already exists: {final_dir}"
            )
        stage_dir = Path(tempfile.mkdtemp(prefix=f".{resolved_run_id}.tmp-", dir=root))
        input_hashes = {key: _sha256(path) for key, path in source_paths.items()}
        verification = _verify_input_controls(source_paths, input_hashes)

        active = pd.read_csv(source_paths["active_map"], dtype=object)
        reference = pd.read_csv(source_paths["reference_map"], dtype=object)
        normalized = pd.read_csv(source_paths["normalized"], dtype=object)
        status = pd.read_csv(source_paths["status"], dtype=object)
        tables = build_japan_listing_valid_time_tables(
            active,
            reference,
            normalized,
            status,
            as_of_date=as_of_date,
        )

        input_files = pd.DataFrame(
            [
                {
                    "input_key": key,
                    "path": str(path.resolve()),
                    "sha256": input_hashes[key],
                    "bytes": path.stat().st_size,
                    "declared_checksum_verified": verification.get(key, False),
                }
                for key, path in source_paths.items()
            ]
        )
        stage_paths = {
            "listing_intervals": stage_dir / "listing_intervals.csv",
            "ric_quote_intervals": stage_dir / "ric_quote_intervals.csv",
            "current_observations": stage_dir / "current_observations.csv",
            "valid_time_membership": stage_dir / "valid_time_membership.csv",
            "pit_membership": stage_dir / "pit_membership.csv",
            "quality": stage_dir / "quality_metrics.csv",
            "input_files": stage_dir / "input_files.csv",
        }
        _write_csv(tables.listing_intervals, stage_paths["listing_intervals"])
        _write_csv(tables.ric_quote_intervals, stage_paths["ric_quote_intervals"])
        _write_csv(tables.current_observations, stage_paths["current_observations"])
        _write_csv(tables.valid_time_membership, stage_paths["valid_time_membership"])
        _write_csv(tables.pit_membership, stage_paths["pit_membership"])
        _write_csv(tables.quality, stage_paths["quality"])
        _write_csv(input_files, stage_paths["input_files"])

        output_hashes = {key: _sha256(path) for key, path in stage_paths.items()}
        checksums_path = stage_dir / "checksums.sha256"
        _write_text(
            checksums_path,
            "".join(
                f"{output_hashes[key]}  {stage_paths[key].name}\n"
                for key in sorted(stage_paths)
            ),
        )
        checksums_hash = _sha256(checksums_path)

        current_hashes = {key: _sha256(path) for key, path in source_paths.items()}
        if current_hashes != input_hashes:
            raise RuntimeError("Japan listing valid-time input changed during build")

        generated_at = _as_utc_iso(
            (clock or (lambda: datetime.now(timezone.utc)))()
        )
        counts = _manifest_counts(tables)
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "japan_listing_valid_time_baseline",
            "run_id": resolved_run_id,
            "status": "complete_valid_time_baseline_current_observation_only",
            "artifact_dir": str(final_dir.resolve()),
            "as_of_date": tables.as_of_date,
            "immutable": True,
            "atomic_publish": True,
            "direct_listing_scope": {
                "country": "Japan",
                "exchange_code": "TYO",
                "requires_direct_jpx_quote": True,
            },
            "counts": counts,
            "input_sha256": input_hashes,
            "inputs": {
                key: {
                    "path": str(path.resolve()),
                    "sha256": input_hashes[key],
                    "bytes": path.stat().st_size,
                    "declared_checksum_verified": verification.get(key, False),
                }
                for key, path in source_paths.items()
            },
            "input_checksums_verified_before_build": True,
            "input_checksums_verified_after_build": True,
            "artifacts": {
                key: str((final_dir / path.name).resolve())
                for key, path in stage_paths.items()
            }
            | {"checksums": str((final_dir / checksums_path.name).resolve())},
            "artifact_sha256": output_hashes | {"checksums": checksums_hash},
            "valid_time_baseline_ready": True,
            "current_observation_only": True,
            "does_not_assert_historical_pit": True,
            "historical_pit_ready": False,
            "historical_backtest_eligible": False,
            "alpha_promotion_allowed": False,
            "production_promotion_allowed": False,
            "ibkr_promotion_allowed": False,
            "live_trading_allowed": False,
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "known_at_policy": (
                "Membership known_at_utc is copied only from explicit reconciliation "
                "source timestamps; null known_at_utc always implies pit_usable=false."
            ),
            "current_observation_policy": (
                "ISIN, SEDOL, and currency are undated current observations. Their "
                "retrieval timestamps do not create historical validity windows."
            ),
            "retirement_policy": (
                "Retire dates after as_of_date are scheduled boundaries and are not "
                "classified as retired."
            ),
            "generated_at_utc": generated_at,
        }
        manifest_path = stage_dir / "manifest.json"
        _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if final_dir.exists():
            raise FileExistsError(
                f"immutable artifact directory already exists: {final_dir}"
            )
        os.replace(stage_dir, final_dir)
        stage_dir = None
        return JapanListingValidTimeArtifacts(
            run_id=resolved_run_id,
            artifact_dir=final_dir,
            listing_intervals_path=final_dir / stage_paths["listing_intervals"].name,
            ric_quote_intervals_path=final_dir / stage_paths["ric_quote_intervals"].name,
            current_observations_path=final_dir / stage_paths["current_observations"].name,
            valid_time_membership_path=final_dir
            / stage_paths["valid_time_membership"].name,
            pit_membership_path=final_dir / stage_paths["pit_membership"].name,
            quality_path=final_dir / stage_paths["quality"].name,
            input_files_path=final_dir / stage_paths["input_files"].name,
            checksums_path=final_dir / checksums_path.name,
            manifest_path=final_dir / manifest_path.name,
            tables=tables,
            manifest=manifest,
        )
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


run_japan_listing_valid_time = write_japan_listing_valid_time_artifacts


def _quote_records(
    active: pd.DataFrame,
    reference: pd.DataFrame,
    enrich_by_ric: dict[str, dict[str, Any]],
    as_of: date,
) -> list[dict[str, Any]]:
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for frame, scope in ((reference, "reference"), (active, "active")):
        for source_row in frame.to_dict("records"):
            if not _is_direct_tyo(source_row):
                continue
            ric = _text(source_row.get("ric"))
            enrich = enrich_by_ric.get(ric, {})
            source_isin = _text(source_row.get("isin"))
            observed_isin = _text(enrich.get("isin"))
            stable_id = _text(source_row.get("stable_instrument_id"))
            if not stable_id:
                fallback_isin = source_isin or observed_isin
                if not fallback_isin:
                    raise ValueError(f"direct TYO RIC {ric} lacks stable instrument/ISIN")
                stable_id = f"isin:{fallback_isin}"
            if stable_id.casefold().startswith("isin:") and observed_isin:
                stable_isin = stable_id.split(":", 1)[1]
                if stable_isin != observed_isin:
                    raise ValueError(
                        f"stable ISIN conflicts with completed enrichment for {ric}"
                    )
            key = (stable_id, ric)
            record = combined.setdefault(
                key,
                {
                    "source_rows": [],
                    "in_active": False,
                    "in_reference": False,
                },
            )
            record["source_rows"].append(source_row)
            record[f"in_{scope}"] = True

    records: list[dict[str, Any]] = []
    for (stable_id, ric), item in combined.items():
        source_rows = item["source_rows"]
        enrich = enrich_by_ric[ric]
        source_start = _first_date(source_rows, ("first_trade_date", "jpx_first_trade_date"))
        source_retire = _first_date(source_rows, ("retire_date",))
        enriched_start = _date_value(enrich.get("first_trade_date"), "first_trade_date")
        enriched_retire = _date_value(enrich.get("retire_date"), "retire_date")
        listing_start = enriched_start or source_start
        retire_date = enriched_retire or source_retire
        current_active = bool(item["in_active"])
        state = _boundary_state(current_active, retire_date, as_of)
        known_values: list[tuple[str, str]] = []
        for row in source_rows:
            for column in ("jpx_known_at_utc", "security_master_known_at_utc"):
                value = _utc_value(row.get(column), column)
                if value:
                    known_values.append((value, column))
        known_at = max((value for value, _ in known_values), default=None)
        known_sources = "|".join(sorted({source for _, source in known_values}))
        source_isin = _first_text(source_rows, ("isin",))
        observed_isin = _text(enrich.get("isin"))
        records.append(
            {
                "stable_instrument_id": stable_id,
                "stable_identifier_type": _first_text(
                    source_rows, ("stable_identifier_type",)
                )
                or ("isin" if stable_id.casefold().startswith("isin:") else "stable_id"),
                "isin": source_isin or observed_isin or None,
                "ric": ric,
                "local_code": _first_text(source_rows, ("local_code", "ticker_symbol"))
                or None,
                "exchange_code": "TYO",
                "listing_start": listing_start.isoformat() if listing_start else None,
                "retire_date": retire_date.isoformat() if retire_date else None,
                "unknown_start": listing_start is None,
                "unknown_end": retire_date is None,
                "boundary_state": state,
                "scheduled_retirement": retire_date is not None and retire_date > as_of,
                "retired_as_of": state == "retired",
                "current_active_map_member": current_active,
                "historical_reference_map_member": bool(item["in_reference"]),
                "as_of_date": as_of.isoformat(),
                "known_at_utc": known_at,
                "known_at_source": known_sources or None,
                "listing_start_source": (
                    "g1_normalized_current_observation"
                    if enriched_start
                    else "jpx_reconciliation_map"
                    if source_start
                    else None
                ),
                "retire_date_source": (
                    "g1_normalized_current_observation"
                    if enriched_retire
                    else "jpx_reconciliation_map"
                    if source_retire
                    else None
                ),
                "date_evidence_conflict": bool(
                    (enriched_start and source_start and enriched_start != source_start)
                    or (
                        enriched_retire
                        and source_retire
                        and enriched_retire != source_retire
                    )
                ),
            }
        )
    if not records:
        raise ValueError("no direct TYO listing rows were found")
    return records


def _listing_record(group: pd.DataFrame, as_of: date) -> dict[str, Any]:
    starts = [_date_value(value, "listing_start") for value in group["listing_start"]]
    starts = [value for value in starts if value is not None]
    current = bool(group["current_active_map_member"].astype(bool).any())
    if current:
        end_values = [
            _date_value(value, "retire_date")
            for value in group.loc[group["current_active_map_member"].astype(bool), "retire_date"]
        ]
        ends = [value for value in end_values if value is not None]
        retire = min(ends) if ends else None
    else:
        end_values = [_date_value(value, "retire_date") for value in group["retire_date"]]
        if end_values and all(value is not None for value in end_values):
            retire = max(value for value in end_values if value is not None)
        else:
            retire = None
    start = min(starts) if starts else None
    state = _boundary_state(current, retire, as_of)
    known = [_text(value) for value in group["known_at_utc"]]
    known_at = max(known) if known and all(known) else None
    isins = sorted({_text(value) for value in group["isin"] if _text(value)})
    return {
        "stable_instrument_id": str(group.iloc[0]["stable_instrument_id"]),
        "stable_identifier_type": str(group.iloc[0]["stable_identifier_type"]),
        "isin": isins[0] if len(isins) == 1 else None,
        "exchange_code": str(group.iloc[0]["exchange_code"]),
        "listing_start": start.isoformat() if start else None,
        "retire_date": retire.isoformat() if retire else None,
        "unknown_start": start is None,
        "unknown_end": retire is None,
        "boundary_state": state,
        "scheduled_retirement": retire is not None and retire > as_of,
        "retired_as_of": state == "retired",
        "current_active_map_member": current,
        "historical_reference_map_member": bool(
            group["historical_reference_map_member"].astype(bool).any()
        ),
        "ric_count": int(group["ric"].nunique()),
        "rics": "|".join(sorted(group["ric"].astype(str).unique())),
        "as_of_date": as_of.isoformat(),
        "known_at_utc": known_at,
        "known_at_source": (
            "|".join(
                sorted(
                    {
                        _text(value)
                        for value in group["known_at_source"]
                        if _text(value)
                    }
                )
            )
            or None
        ),
        "valid_time_source": "jpx_reconciliation_v2_plus_g1_current_enrichment_v2",
        "boundary_convention": "retire_date_as_reported_no_intraday_inference",
    }


def _current_observations(
    latest: pd.DataFrame, quote_intervals: pd.DataFrame
) -> pd.DataFrame:
    stable_by_ric = {
        str(row.ric): str(row.stable_instrument_id)
        for row in quote_intervals.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    specs = (("isin", "isin"), ("sedol", "sedol"), ("currency", "price_currency"))
    for item in latest.to_dict("records"):
        ric = _text(item.get("ric"))
        stable_id = stable_by_ric.get(ric)
        if not stable_id:
            continue
        observed_at = _utc_value(item.get("observed_at_utc"), "observed_at_utc")
        for attribute_name, column in specs:
            value = _text(item.get(column))
            if not value:
                continue
            rows.append(
                {
                    "stable_instrument_id": stable_id,
                    "ric": ric,
                    "attribute_name": attribute_name,
                    "attribute_value": value,
                    "valid_from": None,
                    "valid_to": None,
                    "unknown_start": True,
                    "unknown_end": True,
                    "undated_current_observation": True,
                    "observed_at_utc": observed_at,
                    "known_at_utc": observed_at,
                    "known_at_source": "g1_enrichment_observed_at_utc",
                    "pit_usable": False,
                    "pit_block_reason": "current_observation_has_no_validity_window",
                    "evidence_scope": "refinitiv_current_observation_only",
                }
            )
    return pd.DataFrame(rows, columns=CURRENT_OBSERVATION_COLUMNS).sort_values(
        ["stable_instrument_id", "ric", "attribute_name"], kind="stable"
    ).reset_index(drop=True)


def _valid_time_membership(listings: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "stable_instrument_id": listings["stable_instrument_id"],
            "exchange_code": listings["exchange_code"],
            "membership_start": listings["listing_start"],
            "membership_end": listings["retire_date"],
            "unknown_start": listings["unknown_start"],
            "unknown_end": listings["unknown_end"],
            "membership_state": listings["boundary_state"],
            "current_active_map_member": listings["current_active_map_member"],
            "historical_reference_map_member": listings[
                "historical_reference_map_member"
            ],
            "as_of_date": listings["as_of_date"],
            "membership_basis": "valid_time_listing_boundaries",
        }
    )
    return result[list(VALID_TIME_MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def _pit_membership(listings: pd.DataFrame) -> pd.DataFrame:
    known = listings["known_at_utc"].map(_text)
    usable = known.ne("")
    result = pd.DataFrame(
        {
            "stable_instrument_id": listings["stable_instrument_id"],
            "exchange_code": listings["exchange_code"],
            "membership_start": listings["listing_start"],
            "membership_end": listings["retire_date"],
            "membership_state": listings["boundary_state"],
            "known_at_utc": known.where(usable, None),
            "known_at_source": listings["known_at_source"],
            "pit_usable": usable,
            "pit_block_reason": pd.Series(
                [None if value else "missing_source_backed_known_at_utc" for value in usable]
            ),
            "as_of_date": listings["as_of_date"],
            "membership_basis": "source_known_at_gated_listing_membership",
        }
    )
    return result[list(PIT_MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def _quality_metrics(
    listings: pd.DataFrame,
    quotes: pd.DataFrame,
    observations: pd.DataFrame,
    valid_membership: pd.DataFrame,
    pit_membership: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "listing_intervals": len(listings),
        "ric_quote_intervals": len(quotes),
        "current_observations": len(observations),
        "valid_time_membership_rows": len(valid_membership),
        "pit_membership_rows": len(pit_membership),
        "unknown_listing_start_rows": int(listings["unknown_start"].astype(bool).sum()),
        "unknown_listing_end_rows": int(listings["unknown_end"].astype(bool).sum()),
        "scheduled_retirement_rows": int(
            listings["scheduled_retirement"].astype(bool).sum()
        ),
        "retired_as_of_rows": int(listings["retired_as_of"].astype(bool).sum()),
        "null_membership_known_at_rows": int(
            pit_membership["known_at_utc"].map(_text).eq("").sum()
        ),
        "pit_usable_membership_rows": int(
            pit_membership["pit_usable"].astype(bool).sum()
        ),
        "date_evidence_conflict_rows": int(
            quotes["date_evidence_conflict"].astype(bool).sum()
        ),
    }
    return pd.DataFrame(
        [{"metric": key, "value": int(value)} for key, value in metrics.items()]
    )


def _verify_input_controls(
    paths: dict[str, Path], hashes: dict[str, str]
) -> dict[str, bool]:
    manifest = _read_json(paths["enrichment_manifest"])
    if manifest.get("status") != "complete_current_observation":
        raise ValueError("G1 enrichment is not complete_current_observation")
    if manifest.get("current_observation_only") is not True:
        raise ValueError("G1 enrichment must declare current_observation_only")
    if manifest.get("historical_pit_ready") is not False:
        raise ValueError("G1 enrichment unexpectedly claims historical PIT readiness")

    declared = _read_checksums(paths["enrichment_checksums"])
    for key in ("normalized", "status"):
        path = paths[key]
        if declared.get(path.name) != hashes[key]:
            raise ValueError(f"G1 checksum mismatch for {path.name}")
    manifest_hashes = manifest.get("artifact_sha256", {})
    if manifest_hashes.get("normalized_responses") != hashes["normalized"]:
        raise ValueError("G1 manifest normalized checksum mismatch")
    if manifest_hashes.get("ric_status") != hashes["status"]:
        raise ValueError("G1 manifest status checksum mismatch")
    if manifest_hashes.get("checksums") != hashes["enrichment_checksums"]:
        raise ValueError("G1 manifest checksums-file hash mismatch")

    verified = {
        "normalized": True,
        "status": True,
        "enrichment_checksums": True,
        "enrichment_manifest": True,
        "active_map": False,
        "reference_map": False,
    }
    if "reconciliation_manifest" in paths:
        source_manifest = _read_json(paths["reconciliation_manifest"])
        source_hashes = source_manifest.get("artifact_sha256", {})
        for key, declared_key in (
            ("active_map", "active_operating_core"),
            ("reference_map", "historical_listed_reference"),
        ):
            if source_hashes.get(declared_key) != hashes[key]:
                raise ValueError(f"reconciliation manifest checksum mismatch for {key}")
            verified[key] = True
        verified["reconciliation_manifest"] = True
    return verified


def _latest_normalized(normalized: pd.DataFrame) -> pd.DataFrame:
    data = normalized.copy()
    data["ric"] = data["ric"].map(_text)
    data["_observed_sort"] = data["observed_at_utc"].map(_text)
    data = data.sort_values(["ric", "_observed_sort"], kind="stable")
    return data.drop_duplicates("ric", keep="last").drop(columns="_observed_sort")


def _verify_completed_rows(
    required_rics: set[str], latest: pd.DataFrame, status: pd.DataFrame
) -> None:
    status_data = status.copy()
    status_data["ric"] = status_data["ric"].map(_text)
    latest_status = status_data.drop_duplicates("ric", keep="last").set_index("ric")
    missing_status = sorted(required_rics.difference(latest_status.index))
    if missing_status:
        raise ValueError(f"completed G1 status lacks direct TYO RICs: {missing_status[:5]}")
    failed = latest_status.loc[sorted(required_rics), "status"].astype(str).ne("success")
    if failed.any():
        raise ValueError("completed G1 status contains non-success direct TYO RICs")
    missing_normalized = required_rics.difference(set(latest["ric"].map(_text)))
    if missing_normalized:
        raise ValueError(
            f"normalized G1 enrichment lacks direct TYO RICs: {sorted(missing_normalized)[:5]}"
        )


def _resolve_as_of_date(
    supplied: str | date | datetime | None,
    latest: pd.DataFrame,
    status: pd.DataFrame,
) -> date:
    if supplied is not None:
        value = pd.Timestamp(supplied)
        if pd.isna(value):
            raise ValueError("as_of_date is invalid")
        return value.date()
    values = pd.concat(
        [latest["observed_at_utc"], status["observed_at_utc"]], ignore_index=True
    )
    parsed = pd.to_datetime(values, errors="coerce", utc=True).dropna()
    if parsed.empty:
        raise ValueError("cannot derive as_of_date without enrichment observation time")
    return parsed.max().date()


def _boundary_state(current: bool, retire: date | None, as_of: date) -> str:
    if retire is not None and retire > as_of:
        return "scheduled_retirement"
    if current and retire is not None and retire <= as_of:
        return "active_map_retire_boundary_conflict"
    if current:
        return "active_open"
    if retire is not None:
        return "retired"
    return "historical_unknown_end"


def _validate_pit_invariants(
    observations: pd.DataFrame, pit_membership: pd.DataFrame
) -> None:
    for frame in (observations, pit_membership):
        null_known = frame["known_at_utc"].map(_text).eq("")
        if frame.loc[null_known, "pit_usable"].astype(bool).any():
            raise AssertionError("null known_at_utc cannot be PIT usable")
    if not observations.empty:
        if observations[["valid_from", "valid_to"]].notna().any().any():
            raise AssertionError("current identifier/currency observations must be undated")
        if observations["pit_usable"].astype(bool).any():
            raise AssertionError("undated current observations cannot be PIT usable")


def _manifest_counts(tables: JapanListingValidTimeTables) -> dict[str, int]:
    return {
        "listing_intervals": int(len(tables.listing_intervals)),
        "ric_quote_intervals": int(len(tables.ric_quote_intervals)),
        "current_observations": int(len(tables.current_observations)),
        "valid_time_membership_rows": int(len(tables.valid_time_membership)),
        "pit_membership_rows": int(len(tables.pit_membership)),
        "pit_usable_membership_rows": int(
            tables.pit_membership["pit_usable"].astype(bool).sum()
        ),
        "scheduled_retirement_rows": int(
            tables.listing_intervals["scheduled_retirement"].astype(bool).sum()
        ),
        "retired_as_of_rows": int(
            tables.listing_intervals["retired_as_of"].astype(bool).sum()
        ),
        "unknown_start_rows": int(
            tables.listing_intervals["unknown_start"].astype(bool).sum()
        ),
        "unknown_end_rows": int(
            tables.listing_intervals["unknown_end"].astype(bool).sum()
        ),
    }


def _direct_tyo_rics(frame: pd.DataFrame) -> set[str]:
    return {
        _text(row.get("ric"))
        for row in frame.to_dict("records")
        if _is_direct_tyo(row) and _text(row.get("ric"))
    }


def _is_direct_tyo(row: dict[str, Any]) -> bool:
    direct = _as_bool(row.get("is_direct_jpx_quote"))
    evidence = row.get("has_direct_jpx_listing_evidence")
    if evidence is not None and _text(evidence):
        direct = direct and _as_bool(evidence)
    return direct and _text(row.get("exchange_code")).upper() == "TYO"


def _first_date(rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> date | None:
    for row in rows:
        for column in columns:
            value = _date_value(row.get(column), column)
            if value:
                return value
    return None


def _first_text(rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> str:
    for row in rows:
        for column in columns:
            value = _text(row.get(column))
            if value:
                return value
    return ""


def _date_value(value: Any, column: str) -> date | None:
    text = _text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid {column}: {text}")
    return parsed.date()


def _utc_value(value: Any, column: str) -> str | None:
    text = _text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"invalid {column}: {text}")
    return parsed.isoformat().replace("+00:00", "+00:00")


def _as_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "nat"} else text


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true", "yes", "y"}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _require_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required Japan valid-time input is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"required Japan valid-time input is empty: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON control must be an object: {path}")
    return value


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid checksum line in {path}: {line}")
        if parts[1] in checksums:
            raise ValueError(f"duplicate checksum entry in {path}: {parts[1]}")
        checksums[parts[1]] = parts[0]
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _write_text(path, frame.to_csv(index=False))


def _write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Japan listing valid-time run is already locked: {path}") from exc


__all__ = [
    "ACTIVE_MAP_FILE",
    "CURRENT_OBSERVATION_COLUMNS",
    "DEFAULT_G1_ENRICHMENT_DIR",
    "DEFAULT_JAPAN_LISTING_VALID_TIME_ARTIFACT_ROOT",
    "DEFAULT_JPX_RECONCILIATION_DIR",
    "INPUT_CHECKSUMS_FILE",
    "JapanListingValidTimeArtifacts",
    "JapanListingValidTimeTables",
    "LISTING_INTERVAL_COLUMNS",
    "NORMALIZED_FILE",
    "PIT_MEMBERSHIP_COLUMNS",
    "REFERENCE_MAP_FILE",
    "RIC_QUOTE_INTERVAL_COLUMNS",
    "STATUS_FILE",
    "VALID_TIME_MEMBERSHIP_COLUMNS",
    "build_japan_listing_valid_time_tables",
    "default_japan_listing_valid_time_run_id",
    "run_japan_listing_valid_time",
    "write_japan_listing_valid_time_artifacts",
]
