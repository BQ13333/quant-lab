from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.japan_g4_evidence_contracts import full_component_blockers


UTC = timezone.utc


@dataclass(frozen=True)
class JapanG4EvidenceArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def assemble_japan_g4_evidence(
    *,
    run_id: str,
    base_rules_dir: str | Path,
    lot_size_validation_dir: str | Path,
    calendar_validation_dir: str | Path | None = None,
    rule_history_validation_dir: str | Path | None = None,
    tick_assignment_validation_dir: str | Path | None = None,
    halt_validation_dir: str | Path | None = None,
    currency_validation_dir: str | Path | None = None,
    tick_rule_validation_dir: str | Path | None = None,
    pit_tick_assignment_validation_dir: str | Path | None = None,
    normal_price_limit_validation_dir: str | Path | None = None,
    exceptional_price_limit_validation_dir: str | Path | None = None,
    tca_validation_dir: str | Path | None = None,
    artifact_root: str | Path = "artifacts/fx_calendar_tca",
) -> JapanG4EvidenceArtifacts:
    """Assemble independently admitted G4 components without upgrading incomplete ones."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    base = Path(base_rules_dir).resolve()
    lot = Path(lot_size_validation_dir).resolve()
    calendar = Path(calendar_validation_dir).resolve() if calendar_validation_dir else None
    rule_validation = (
        Path(rule_history_validation_dir).resolve() if rule_history_validation_dir else None
    )
    tick_validation = (
        Path(tick_assignment_validation_dir).resolve()
        if tick_assignment_validation_dir
        else None
    )
    full_component_dirs = {
        component: Path(value).resolve()
        for component, value in {
            "halt_suspension_events": halt_validation_dir,
            "jpy_currency_identity": currency_validation_dir,
            "tick_size_rules": tick_rule_validation_dir,
            "pit_narrow_tick_assignment": pit_tick_assignment_validation_dir,
            "normal_price_limits": normal_price_limit_validation_dir,
            "exceptional_price_limits": exceptional_price_limit_validation_dir,
            "tca_assumptions": tca_validation_dir,
        }.items()
        if value is not None
    }
    root = (Path(artifact_root) / run_id).resolve()
    if root in {base, lot, calendar, rule_validation, tick_validation, *full_component_dirs.values()}:
        raise ValueError("output must differ from source evidence directories")

    base_manifest_path = base / "manifest.json"
    lot_manifest_path = lot / "manifest.json"
    base_manifest = _read_json(base_manifest_path)
    lot_manifest = _read_json(lot_manifest_path)
    blockers = _base_blockers(base, base_manifest) + _lot_blockers(lot, lot_manifest)
    calendar_manifest_path = calendar / "manifest.json" if calendar else None
    calendar_manifest = _read_json(calendar_manifest_path) if calendar_manifest_path else {}
    if calendar is not None:
        blockers += _calendar_blockers(calendar, calendar_manifest)
    rule_validation_manifest_path = (
        rule_validation / "manifest.json" if rule_validation else None
    )
    rule_validation_manifest = (
        _read_json(rule_validation_manifest_path) if rule_validation_manifest_path else {}
    )
    if rule_validation is not None:
        blockers += _partial_rule_blockers(rule_validation_manifest)
    tick_validation_manifest_path = (
        tick_validation / "manifest.json" if tick_validation else None
    )
    tick_validation_manifest = (
        _read_json(tick_validation_manifest_path)
        if tick_validation_manifest_path
        else {}
    )
    if tick_validation is not None:
        blockers += _partial_tick_assignment_blockers(tick_validation_manifest)
    full_component_manifests: dict[str, dict[str, Any]] = {}
    for component, directory in full_component_dirs.items():
        manifest = _read_json(directory / "manifest.json")
        full_component_manifests[component] = manifest
        blockers += full_component_blockers(
            directory,
            manifest,
            expected_component=component,
        )
    if blockers:
        raise ValueError("G4 evidence admission failed: " + "; ".join(blockers))

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        _copy_base_evidence(base, root)
        validated_history_source = Path(
            lot_manifest["artifacts"]["validated_monthly_lot_size_history"]
        ).resolve()
        validated_history = root / "validated_monthly_lot_size_history.parquet"
        shutil.copy2(validated_history_source, validated_history)
        validated_calendar = root / "validated_exchange_calendar.parquet"
        if calendar is not None:
            validated_calendar_source = Path(
                calendar_manifest["artifacts"]["validated_exchange_calendar"]
            ).resolve()
            shutil.copy2(validated_calendar_source, validated_calendar)
        rule_producer = None
        if rule_validation is not None:
            rule_producer_manifest_path = Path(
                rule_validation_manifest["producer_manifest"]
            ).resolve()
            rule_producer = rule_producer_manifest_path.parent
            _copy_directory_evidence(rule_producer, root / "rule_history")
            shutil.copy2(
                rule_validation_manifest_path,
                root / "rule_history_validation_manifest.json",
            )
        if tick_validation is not None:
            tick_producer_manifest_path = Path(
                tick_validation_manifest["producer_manifest"]
            ).resolve()
            _copy_directory_evidence(
                tick_producer_manifest_path.parent,
                root / "tick_assignment_history",
            )
            shutil.copy2(
                tick_validation_manifest_path,
                root / "tick_assignment_validation_manifest.json",
            )
        for component, directory in full_component_dirs.items():
            _copy_validated_component_evidence(
                directory,
                root / "full_component_evidence" / component,
                full_component_manifests[component],
            )

        quality_path = root / "evidence_quality.csv"
        quality = pd.read_csv(quality_path)
        jpy_mask = quality["component"].astype(str).eq("jpy_fx_rates")
        quality.loc[jpy_mask, "status"] = "pass"
        quality.loc[jpy_mask, "validated"] = True
        quality.loc[jpy_mask, "coverage"] = 1.0
        lot_mask = quality["component"].astype(str).eq("lot_sizes")
        admitted_row = {
            "component": "lot_sizes",
            "status": "pass",
            "validated": True,
            "coverage": 1.0,
            "source_rows": int(lot_manifest["counts"]["history_rows"]),
            "notes": (
                "Independent validation of direct official monthly JPX issuer trading-unit "
                "observations for 2006-07 through 2026-06; no future fill"
            ),
        }
        if lot_mask.any():
            for key, value in admitted_row.items():
                quality.loc[lot_mask, key] = value
        else:
            quality = pd.concat([quality, pd.DataFrame([admitted_row])], ignore_index=True)
        if calendar is not None:
            calendar_row = {
                "component": "exchange_calendars",
                "status": "pass",
                "validated": True,
                "coverage": 1.0,
                "source_rows": int(calendar_manifest["counts"]["sessions"]),
                "notes": (
                    "Independent date/time reconciliation for 2006-07 through 2026-06; "
                    "240 monthly session counts validated against official JPX quotation PDFs"
                ),
            }
            calendar_mask = quality["component"].astype(str).eq("exchange_calendars")
            if calendar_mask.any():
                for key, value in calendar_row.items():
                    quality.loc[calendar_mask, key] = value
            else:
                quality = pd.concat([quality, pd.DataFrame([calendar_row])], ignore_index=True)
        if rule_validation is not None:
            counts = rule_validation_manifest.get("counts", {})
            remaining = rule_validation_manifest.get("remaining_completion_blockers", [])
            rule_row = {
                "component": "price_limits",
                "status": "partial_validated_rule_history",
                "validated": False,
                "coverage": 0.0,
                "source_rows": int(counts.get("tick_band_rows", 0))
                + int(counts.get("assignment_snapshot_rows", 0))
                + int(counts.get("price_limit_band_rows", 0)),
                "notes": "; ".join(map(str, remaining)),
            }
            rule_mask = quality["component"].astype(str).eq("price_limits")
            if rule_mask.any():
                for key, value in rule_row.items():
                    quality.loc[rule_mask, key] = value
            else:
                quality = pd.concat([quality, pd.DataFrame([rule_row])], ignore_index=True)
        if tick_validation is not None:
            counts = tick_validation_manifest.get("counts", {})
            remaining = tick_validation_manifest.get("remaining_completion_blockers", [])
            assignment_row = {
                "component": "pit_narrow_tick_assignment",
                "status": "partial_validated_notice_replay",
                "validated": False,
                "coverage": 0.0,
                "source_rows": int(counts.get("assignment_events", 0)),
                "notes": "; ".join(map(str, remaining)),
            }
            assignment_mask = quality["component"].astype(str).eq(
                "pit_narrow_tick_assignment"
            )
            if assignment_mask.any():
                for key, value in assignment_row.items():
                    quality.loc[assignment_mask, key] = value
            else:
                quality = pd.concat(
                    [quality, pd.DataFrame([assignment_row])], ignore_index=True
                )
        full_quality_rows = {
            "jpy_currency_identity": ("jpy_fx_rates", "Full hash-verified JPY currency identity"),
            "halt_suspension_events": (
                "halt_suspension_events",
                "Full hash-verified official historical halt and suspension archive",
            ),
            "tca_assumptions": (
                "tca_assumptions",
                "Independent recomputation of liquidity-aware Japan market-data calibration",
            ),
        }
        for component, (quality_component, note) in full_quality_rows.items():
            manifest = full_component_manifests.get(component)
            if manifest is None:
                continue
            counts = manifest.get("counts", {})
            source_rows = int(
                counts.get(
                    "event_rows",
                    counts.get("calibration_rows", counts.get("source_rows", 0)),
                )
            )
            quality = _upsert_quality(
                quality,
                {
                    "component": quality_component,
                    "status": "pass",
                    "validated": True,
                    "coverage": 1.0,
                    "source_rows": source_rows,
                    "notes": note,
                },
            )
        price_evidence_components = {"normal_price_limits", "exceptional_price_limits"}
        full_tick_assignment = "pit_narrow_tick_assignment" in full_component_manifests
        tick_sizes_complete = (
            "tick_size_rules" in full_component_manifests and full_tick_assignment
        )
        price_limits_complete = price_evidence_components.issubset(
            full_component_manifests
        )
        if price_limits_complete:
            source_rows = sum(
                int(
                    full_component_manifests[component]
                    .get("counts", {})
                    .get(
                        "rule_rows",
                        full_component_manifests[component]
                        .get("counts", {})
                        .get("event_rows", 0),
                    )
                )
                for component in sorted(price_evidence_components)
            )
            quality = _upsert_quality(
                quality,
                {
                    "component": "price_limits",
                    "status": "pass",
                    "validated": True,
                    "coverage": 1.0,
                    "source_rows": source_rows,
                    "notes": (
                        "Full hash-verified normal and exceptional price-limit history"
                    ),
                },
            )
        if tick_sizes_complete:
            tick_rows = int(
                full_component_manifests["tick_size_rules"]
                .get("counts", {})
                .get("rule_rows", 0)
            ) + int(
                full_component_manifests["pit_narrow_tick_assignment"]
                .get("counts", {})
                .get("assignment_rows", 0)
            )
            quality = _upsert_quality(
                quality,
                {
                    "component": "tick_sizes",
                    "status": "pass",
                    "validated": True,
                    "coverage": 1.0,
                    "source_rows": tick_rows,
                    "notes": "Full hash-verified PIT tick rules and issue assignment history",
                },
            )
            quality = _upsert_quality(
                quality,
                {
                    "component": "pit_narrow_tick_assignment",
                    "status": "pass",
                    "validated": True,
                    "coverage": 1.0,
                    "source_rows": int(
                        full_component_manifests["pit_narrow_tick_assignment"]
                        .get("counts", {})
                        .get("assignment_rows", 0)
                    ),
                    "notes": "Full hash-verified PIT narrow-tick assignment history",
                },
            )
        _atomic_csv(quality_path, quality)

        required_components = [
            "jpy_fx_rates",
            "exchange_calendars",
            "halt_suspension_events",
            "tick_sizes",
            "price_limits",
            "lot_sizes",
            "tca_assumptions",
        ]
        complete = ["jpy_fx_rates", "lot_sizes"]
        if calendar is not None:
            complete.append("exchange_calendars")
        if "halt_suspension_events" in full_component_manifests:
            complete.append("halt_suspension_events")
        if "tca_assumptions" in full_component_manifests:
            complete.append("tca_assumptions")
        if tick_sizes_complete:
            complete.append("tick_sizes")
        if price_limits_complete:
            complete.append("price_limits")
        complete = [value for value in required_components if value in complete]
        partial = [value for value in required_components if value not in complete]
        critical = [
            value
            for value in base_manifest.get("critical_blockers", [])
            if value != "issuer-level lot-size history before 2018-10-01 is missing"
            and not (
                calendar is not None
                and value == "full 2006-present official JPX session calendar is missing"
            )
            and not (
                "halt_suspension_events" in full_component_manifests
                and value == "full historical halt/suspension archive is missing"
            )
            and not (
                "tca_assumptions" in full_component_manifests
                and value == "TCA is not calibrated against admitted Japan market data"
            )
            and not (
                price_limits_complete
                and value == "historical price-limit and tick-size validity windows are missing"
            )
        ]
        if rule_validation is not None and not price_limits_complete:
            critical = [
                value
                for value in critical
                if value != "historical price-limit and tick-size validity windows are missing"
            ]
            critical.extend(rule_validation_manifest["remaining_completion_blockers"])
        if tick_validation is not None and not full_tick_assignment:
            critical = [
                value
                for value in critical
                if not value.startswith(
                    "complete PIT TOPIX100/TOPIX500 constituent-change ledger"
                )
            ]
            critical.extend(
                tick_validation_manifest["remaining_completion_blockers"]
            )
        missing_blockers = {
            "exchange_calendars": "full 2006-present official JPX session calendar is missing",
            "halt_suspension_events": "full historical halt/suspension archive is missing",
            "tick_sizes": "full PIT tick-size rules and issue assignment history are missing",
            "price_limits": (
                "full normal and exceptional price-limit history is missing; exceptional "
                "archive before 2013-09 remains unproven"
            ),
            "tca_assumptions": "TCA is not calibrated against admitted Japan market data",
        }
        critical.extend(
            blocker
            for component, blocker in missing_blockers.items()
            if component not in complete
        )
        for component in list(complete):
            if not _quality_component_admitted(quality, component):
                complete.remove(component)
                critical.append(f"component_quality_contract_invalid:{component}")
        partial = [value for value in required_components if value not in complete]
        critical = list(dict.fromkeys(critical))
        artifacts = {
            key: str((root / Path(value).name).resolve())
            for key, value in base_manifest.get("artifacts", {}).items()
        }
        artifacts["validated_monthly_lot_size_history"] = str(validated_history.resolve())
        if calendar is not None:
            artifacts["validated_exchange_calendar"] = str(validated_calendar.resolve())
        if rule_validation is not None:
            artifacts.update(
                {
                    "rule_history_manifest": str(
                        (root / "rule_history/manifest.json").resolve()
                    ),
                    "rule_history_validation_manifest": str(
                        (root / "rule_history_validation_manifest.json").resolve()
                    ),
                }
            )
        if tick_validation is not None:
            artifacts.update(
                {
                    "tick_assignment_history_manifest": str(
                        (root / "tick_assignment_history/manifest.json").resolve()
                    ),
                    "tick_assignment_validation_manifest": str(
                        (root / "tick_assignment_validation_manifest.json").resolve()
                    ),
                }
            )
        for component in full_component_manifests:
            artifacts[f"{component}_validation_manifest"] = str(
                (root / "full_component_evidence" / component / "manifest.json").resolve()
            )
        counts = dict(base_manifest.get("counts", {}))
        counts.update(
            {
                "lot_size_history_rows": int(lot_manifest["counts"]["history_rows"]),
                "lot_size_history_months": int(lot_manifest["counts"]["months"]),
                "lot_size_venue_variants": int(
                    lot_manifest["counts"].get("venue_specific_variant_rows", 0)
                ),
            }
        )
        if calendar is not None:
            counts.update(
                {
                    "validated_calendar_sessions": int(calendar_manifest["counts"]["sessions"]),
                    "validated_calendar_months": int(calendar_manifest["counts"]["months"]),
                    "calendar_official_source_files": int(
                        calendar_manifest["counts"]["source_files"]
                    ),
                }
            )
        if rule_validation is not None:
            rule_counts = rule_validation_manifest["counts"]
            counts.update(
                {
                    "validated_partial_rule_sources": int(rule_counts["sources"]),
                    "validated_partial_tick_band_rows": int(rule_counts["tick_band_rows"]),
                    "validated_partial_assignment_rows": int(
                        rule_counts["assignment_snapshot_rows"]
                    ),
                }
            )
        if tick_validation is not None:
            tick_counts = tick_validation_manifest["counts"]
            counts.update(
                {
                    "validated_tick_assignment_archive_sources": int(tick_counts["archive_sources"]),
                    "validated_tick_assignment_notice_pages": int(tick_counts["unique_notice_pages"]),
                    "validated_tick_assignment_events": int(tick_counts["assignment_events"]),
                    "validated_tick_assignment_checkpoints": int(tick_counts["checkpoint_exact_matches"]),
                    "validated_tick_assignment_cutoff_names": int(tick_counts["cutoff_assignment_names"]),
                }
            )
        for component, component_manifest in full_component_manifests.items():
            for key, value in component_manifest.get("counts", {}).items():
                if isinstance(value, (bool, int, float, str)):
                    counts[f"{component}_{key}"] = value
        component_lineage = {
            "base_rules_manifest": str(base_manifest_path),
            "base_rules_manifest_sha256": _sha256(base_manifest_path),
            "lot_size_validator_manifest": str(lot_manifest_path),
            "lot_size_validator_manifest_sha256": _sha256(lot_manifest_path),
            "lot_size_producer_manifest": lot_manifest["producer_manifest"],
            "lot_size_producer_manifest_sha256": lot_manifest[
                "producer_manifest_sha256"
            ],
        }
        if calendar_manifest_path is not None:
            component_lineage.update(
                {
                    "calendar_validator_manifest": str(calendar_manifest_path),
                    "calendar_validator_manifest_sha256": _sha256(calendar_manifest_path),
                    "calendar_producer_manifest": calendar_manifest["producer_manifest"],
                    "calendar_producer_manifest_sha256": calendar_manifest[
                        "producer_manifest_sha256"
                    ],
                }
            )
        if rule_validation_manifest_path is not None:
            component_lineage.update(
                {
                    "rule_history_validator_manifest": str(
                        rule_validation_manifest_path
                    ),
                    "rule_history_validator_manifest_sha256": _sha256(
                        rule_validation_manifest_path
                    ),
                    "rule_history_producer_manifest": rule_validation_manifest[
                        "producer_manifest"
                    ],
                    "rule_history_producer_manifest_sha256": rule_validation_manifest[
                        "producer_manifest_sha256"
                    ],
                }
            )
        if tick_validation_manifest_path is not None:
            component_lineage.update(
                {
                    "tick_assignment_validator_manifest": str(tick_validation_manifest_path),
                    "tick_assignment_validator_manifest_sha256": _sha256(tick_validation_manifest_path),
                    "tick_assignment_producer_manifest": tick_validation_manifest["producer_manifest"],
                    "tick_assignment_producer_manifest_sha256": tick_validation_manifest["producer_manifest_sha256"],
                }
            )
        for component, directory in full_component_dirs.items():
            validation_manifest_path = directory / "manifest.json"
            component_manifest = full_component_manifests[component]
            component_lineage.update(
                {
                    f"{component}_validator_manifest": str(validation_manifest_path),
                    f"{component}_validator_manifest_sha256": _sha256(
                        validation_manifest_path
                    ),
                    f"{component}_producer_manifest": component_manifest[
                        "producer_manifest"
                    ],
                    f"{component}_producer_manifest_sha256": component_manifest[
                        "producer_manifest_sha256"
                    ],
                }
            )
        aggregate_complete = set(required_components).issubset(complete) and not critical
        manifest = {
            **base_manifest,
            "run_id": run_id,
            "artifact_type": (
                "jpx_g4_full_evidence_assembly"
                if aggregate_complete
                else "jpx_g4_evidence_assembly_incomplete"
            ),
            "status": "complete" if aggregate_complete else "incomplete",
            "scorecard_eligible": aggregate_complete,
            "evidence_scope": (
                "full_hash_verified_g4_2006_07_to_2026_06"
                if aggregate_complete
                else "partial_hash_verified_g4_fail_closed"
            ),
            "complete_components": complete,
            "partial_components": partial,
            "counts": counts,
            "critical_blockers": critical,
            "artifacts": artifacts,
            "official_jpx_calendar": bool(calendar is not None),
            "tca_calibrated": "tca_assumptions" in complete,
            "component_admission": {
                "currency_identity": "jpy_fx_rates" in complete,
                "exchange_calendar": "exchange_calendars" in complete,
                "historical_halts_suspensions": "halt_suspension_events" in complete,
                "tick_sizes": "tick_sizes" in complete,
                "pit_tick_rules": "tick_size_rules" in full_component_manifests,
                "pit_tick_assignment": full_tick_assignment,
                "pit_lot_sizes": "lot_sizes" in complete,
                "normal_price_limits": "normal_price_limits" in full_component_manifests,
                "exceptional_price_limits": "exceptional_price_limits" in full_component_manifests,
                "price_limits": "price_limits" in complete,
                "liquidity_aware_tca": "tca_assumptions" in complete,
            },
            "component_lineage": component_lineage,
            "sha256": _artifact_hashes(root),
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
        return JapanG4EvidenceArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def _base_blockers(root: Path, manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not (
        manifest.get("official_jpx_source") is True
        and manifest.get("research_only") is True
        and manifest.get("production_ready") is False
        and manifest.get("live_execution_ready") is False
        and manifest.get("currency") == "JPY"
        and manifest.get("scorecard_eligible") is False
    ):
        blockers.append("base_rules_contract_invalid")
    hashes = manifest.get("sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        blockers.append("base_rules_hash_inventory_empty")
        hashes = {}
    for relative_name, digest in hashes.items():
        path = root / relative_name
        if not path.is_file() or _sha256(path) != digest:
            blockers.append(f"base_rules_hash_mismatch:{relative_name}")
    for name, value in manifest.get("artifacts", {}).items():
        path = Path(str(value)).resolve()
        try:
            relative = path.relative_to(root.resolve())
        except ValueError:
            blockers.append(f"base_rules_artifact_path_invalid:{name}")
            continue
        if str(relative) not in hashes:
            blockers.append(f"base_rules_artifact_not_hash_verified:{name}")
    quality_path = root / "evidence_quality.csv"
    if not quality_path.is_file():
        blockers.append("base_rules_quality_missing")
    else:
        quality = pd.read_csv(quality_path)
        required = {"component", "status", "validated", "coverage", "source_rows"}
        if not required.issubset(quality.columns):
            blockers.append("base_rules_quality_schema_invalid")
        else:
            currency = quality.loc[quality["component"].astype(str).eq("jpy_fx_rates")]
            if (
                len(currency) != 1
                or not _truthy(currency.iloc[0]["validated"])
                or float(currency.iloc[0]["coverage"]) != 1.0
                or int(currency.iloc[0]["source_rows"]) <= 0
                or "pass" not in str(currency.iloc[0]["status"])
            ):
                blockers.append("base_rules_currency_identity_quality_invalid")
    raw_manifest_path = root / "raw_manifest.csv"
    if raw_manifest_path.is_file():
        raw = pd.read_csv(raw_manifest_path, dtype=str).fillna("")
        required_raw = {"component", "http_status", "raw_path", "sha256"}
        if not required_raw.issubset(raw.columns) or "trading_unit" not in set(raw.get("component", [])):
            blockers.append("base_rules_raw_manifest_contract_invalid")
        else:
            for row in raw.to_dict("records"):
                path = Path(row["raw_path"])
                if row["http_status"] != "200" or not path.is_file() or _sha256(path) != row["sha256"]:
                    blockers.append(f"base_rules_raw_hash_mismatch:{row['component']}")
    else:
        blockers.append("base_rules_raw_manifest_missing")
    counts = manifest.get("counts", {})
    if "halt_source_rows" in counts:
        halt_rows = int(counts.get("halt_rows", -1))
        source_rows = int(counts.get("halt_source_rows", -2))
        if halt_rows <= 0 or halt_rows != source_rows:
            blockers.append(f"base_rules_halt_source_output_parity_failed:{halt_rows}/{source_rows}")
        elif quality_path.is_file():
            quality = pd.read_csv(quality_path)
            halt_quality = quality.loc[quality["component"].astype(str).eq("halt_suspension_events")]
            if (
                len(halt_quality) != 1
                or int(halt_quality.iloc[0]["source_rows"]) != halt_rows
                or "parity_passed" not in str(halt_quality.iloc[0]["status"])
            ):
                blockers.append("base_rules_halt_quality_parity_contract_invalid")
    return blockers


def _lot_blockers(root: Path, manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    promotion = manifest.get("promotion_policy", {})
    if not (
        manifest.get("status") == "pass_component"
        and manifest.get("component") == "lot_sizes"
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("scorecard_eligible") is False
        and manifest.get("independent_validation") is True
        and manifest.get("official_jpx_source") is True
        and manifest.get("critical_blocker_count") == 0
        and not manifest.get("critical_blockers")
        and all(promotion.get(key) is False for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading"))
    ):
        blockers.append("lot_size_validation_contract_invalid")
    blockers.extend(_validation_output_blockers(root, manifest, "lot_size_validation"))
    producer_manifest = Path(str(manifest.get("producer_manifest", "")))
    if not producer_manifest.is_file() or _sha256(producer_manifest) != manifest.get(
        "producer_manifest_sha256"
    ):
        blockers.append("lot_size_producer_lineage_mismatch")
    else:
        blockers.extend(_producer_output_blockers(producer_manifest, "lot_size"))
    if int(manifest.get("counts", {}).get("history_rows", 0)) <= 0:
        blockers.append("lot_size_history_empty")
    return blockers


def _calendar_blockers(root: Path, manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    promotion = manifest.get("promotion_policy", {})
    if not (
        manifest.get("status") == "pass_component"
        and manifest.get("component") == "exchange_calendars"
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("scorecard_eligible") is False
        and manifest.get("independent_validation") is True
        and manifest.get("official_jpx_calendar") is True
        and manifest.get("currency") == "JPY"
        and manifest.get("critical_blocker_count") == 0
        and not manifest.get("critical_blockers")
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        blockers.append("calendar_validation_contract_invalid")
    blockers.extend(_validation_output_blockers(root, manifest, "calendar_validation"))
    producer_manifest = Path(str(manifest.get("producer_manifest", "")))
    if not producer_manifest.is_file() or _sha256(producer_manifest) != manifest.get(
        "producer_manifest_sha256"
    ):
        blockers.append("calendar_producer_lineage_mismatch")
    else:
        blockers.extend(_producer_output_blockers(producer_manifest, "calendar"))
    counts = manifest.get("counts", {})
    if int(counts.get("months", 0)) != 240 or int(counts.get("sessions", 0)) != 4_891:
        blockers.append("calendar_denominator_invalid")
    return blockers


def _partial_rule_blockers(manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    promotion = manifest.get("promotion_policy", {})
    if not (
        manifest.get("status") == "pass_partial_component"
        and manifest.get("component") == "price_limits"
        and manifest.get("independent_validation") is True
        and manifest.get("component_scorecard_eligible") is False
        and manifest.get("scorecard_eligible") is False
        and manifest.get("official_jpx_source") is True
        and not manifest.get("validation_issues")
        and bool(manifest.get("remaining_completion_blockers"))
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        blockers.append("partial_rule_validation_contract_invalid")
    producer_manifest = Path(str(manifest.get("producer_manifest", "")))
    if not producer_manifest.is_file() or _sha256(producer_manifest) != manifest.get(
        "producer_manifest_sha256"
    ):
        blockers.append("partial_rule_producer_lineage_mismatch")
    producer = _read_json(producer_manifest)
    producer_hashes = producer.get("output_sha256", {})
    if not isinstance(producer_hashes, dict) or not producer_hashes:
        blockers.append("partial_rule_producer_hash_inventory_empty")
        producer_hashes = {}
    for filename, digest in producer_hashes.items():
        path = producer_manifest.parent / filename
        if not path.is_file() or _sha256(path) != digest:
            blockers.append(f"partial_rule_output_hash_mismatch:{filename}")
    return blockers


def _partial_tick_assignment_blockers(manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    promotion = manifest.get("promotion_policy", {})
    counts = manifest.get("counts", {})
    if not (
        manifest.get("status") == "pass_partial_component"
        and manifest.get("component") == "pit_narrow_tick_assignment"
        and manifest.get("independent_validation") is True
        and manifest.get("component_scorecard_eligible") is False
        and manifest.get("scorecard_eligible") is False
        and manifest.get("official_jpx_source") is True
        and not manifest.get("validation_issues")
        and bool(manifest.get("remaining_completion_blockers"))
        and int(counts.get("archive_sources", 0)) == 154
        and int(counts.get("unique_notice_pages", 0)) == 40
        and int(counts.get("assignment_events", 0)) == 145
        and int(counts.get("checkpoint_exact_matches", 0)) == 14
        and int(counts.get("cutoff_assignment_names", 0)) == 493
        and counts.get("cutoff_master_exact_match") is True
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        blockers.append("partial_tick_assignment_validation_contract_invalid")
    producer_manifest = Path(str(manifest.get("producer_manifest", "")))
    if not producer_manifest.is_file() or _sha256(producer_manifest) != manifest.get(
        "producer_manifest_sha256"
    ):
        blockers.append("partial_tick_assignment_producer_lineage_mismatch")
    producer = _read_json(producer_manifest)
    producer_hashes = producer.get("output_sha256", {})
    if not isinstance(producer_hashes, dict) or not producer_hashes:
        blockers.append("partial_tick_assignment_producer_hash_inventory_empty")
        producer_hashes = {}
    for filename, digest in producer_hashes.items():
        path = producer_manifest.parent / filename
        if not path.is_file() or _sha256(path) != digest:
            blockers.append(f"partial_tick_assignment_output_hash_mismatch:{filename}")
    return blockers


def _validation_output_blockers(
    root: Path, manifest: dict[str, Any], prefix: str
) -> list[str]:
    hashes = manifest.get("output_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        return [f"{prefix}_hash_inventory_empty"]
    blockers = []
    verified: set[Path] = set()
    for filename, digest in hashes.items():
        relative = Path(str(filename))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _within(path, root)
            or not path.is_file()
            or _sha256(path) != digest
        ):
            blockers.append(f"{prefix}_hash_mismatch:{filename}")
        else:
            verified.add(path)
    for name, value in manifest.get("artifacts", {}).items():
        path = Path(str(value)).resolve()
        if not _within(path, root) or path not in verified:
            blockers.append(f"{prefix}_artifact_not_hash_verified:{name}")
    return blockers


def _producer_output_blockers(manifest_path: Path, prefix: str) -> list[str]:
    producer = _read_json(manifest_path)
    hashes = producer.get("output_sha256", producer.get("artifact_sha256", {}))
    if not isinstance(hashes, dict) or not hashes:
        return [f"{prefix}_producer_hash_inventory_empty"]
    blockers = []
    for filename, digest in hashes.items():
        relative = Path(str(filename))
        path = (manifest_path.parent / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _within(path, manifest_path.parent)
            or not path.is_file()
            or _sha256(path) != digest
        ):
            blockers.append(f"{prefix}_producer_output_hash_mismatch:{filename}")
    return blockers


def _upsert_quality(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    mask = frame["component"].astype(str).eq(str(row["component"]))
    if not mask.any():
        return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    result = frame.copy()
    for key, value in row.items():
        result.loc[mask, key] = value
    return result


def _quality_component_admitted(frame: pd.DataFrame, component: str) -> bool:
    rows = frame.loc[frame["component"].astype(str).eq(component)]
    if len(rows) != 1:
        return False
    row = rows.iloc[0]
    try:
        coverage = float(row["coverage"])
        source_rows = int(row["source_rows"])
    except (TypeError, ValueError):
        return False
    return (
        str(row["status"]) == "pass"
        and _truthy(row["validated"])
        and coverage == 1.0
        and source_rows > 0
    )


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _copy_base_evidence(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_directory_evidence(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith((".tmp", ".lock")):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_validated_component_evidence(
    source: Path, destination: Path, manifest: dict[str, Any]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    for filename in manifest["output_sha256"]:
        relative = Path(str(filename))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json"} and not path.name.endswith(".lock")
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


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
