from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.pit_data_plan import (
    PITDatasetSpec,
    _build_extraction_batches,
    _dataset_requirements_frame,
    _field_probe_queue,
    _readiness_gates,
    _target_artifacts,
    default_pit_dataset_specs,
)


G3_DATASETS: tuple[str, ...] = (
    "fundamentals_pit",
    "ibes_estimates_revisions",
    "earnings_events_actuals",
)


@dataclass(frozen=True)
class JapanPITDataPlanResult:
    run_id: str
    artifact_dir: Path
    selected_universe_path: Path
    manifest_path: Path
    selected_names: int
    extraction_batches: int
    scorecard_eligible: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _g1_is_admitted(manifest: dict[str, Any]) -> bool:
    independent = manifest.get("independent_validation") is True or bool(
        manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
    )
    return bool(
        str(manifest.get("status", "")).lower() in {"complete", "pass", "accepted_backtest"}
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
        and independent
        and int(manifest.get("critical_blocker_count", 0) or 0) == 0
        and manifest.get("source_checksums_verified") is True
        and not any(manifest.get(flag) is True for flag in ("partial", "smoke", "current_survivor"))
    )


def _normalise_current_universe(source: pd.DataFrame, *, g1_admitted: bool) -> pd.DataFrame:
    ric_col = "refinitiv_ric" if "refinitiv_ric" in source.columns else "ric"
    if ric_col not in source.columns:
        raise ValueError("Japan universe must contain refinitiv_ric or ric")

    frame = source.copy()
    frame["refinitiv_ric"] = frame[ric_col].astype("string").str.strip()
    frame = frame.loc[frame["refinitiv_ric"].notna() & frame["refinitiv_ric"].ne("")].copy()
    window_columns = {
        "stable_listing_id",
        "stable_instrument_id",
        "listing_start",
        "listing_end",
        "known_at",
        "pit_usable",
    }
    has_window_identity = window_columns.issubset(frame.columns)
    if has_window_identity:
        frame = frame.drop_duplicates(
            ["stable_listing_id", "refinitiv_ric", "listing_start", "listing_end"],
            keep="first",
        )
    else:
        frame = frame.drop_duplicates("refinitiv_ric", keep="first")
    if frame.empty:
        raise ValueError("Japan universe contains no usable Refinitiv RICs")

    aliases = {
        "ticker": ("ticker", "ticker_symbol", "local_code"),
        "name": ("name", "common_name", "document_title"),
        "sid": ("sid", "stable_instrument_id", "instrument_permid"),
        "exchange": ("exchange", "exchange_code", "exchange_name"),
        "currency": ("currency", "price_currency"),
    }
    for target, candidates in aliases.items():
        if target in frame.columns:
            continue
        source_col = next((column for column in candidates if column in frame.columns), None)
        frame[target] = frame[source_col] if source_col else pd.NA

    frame["country"] = "Japan"
    frame["currency"] = frame["currency"].fillna("JPY")
    frame["selected_for_u2"] = True
    frame["selected_for_u3"] = False
    frame["has_refinitiv_ric"] = True
    pit_windows = bool(
        g1_admitted
        and has_window_identity
        and frame["pit_usable"].fillna(False).astype(bool).all()
        and frame[["stable_listing_id", "listing_start", "listing_end", "known_at"]]
        .notna()
        .all()
        .all()
    )
    frame["point_in_time_membership_claimed"] = pit_windows
    frame["planning_universe_status"] = (
        "g1_admitted_pit_listing_window" if pit_windows
        else "current_survivor_or_incomplete_window_identity"
    )
    for column in (
        "stable_listing_id",
        "stable_instrument_id",
        "listing_start",
        "listing_end",
        "known_at",
        "window_end_known_at",
        "right_open",
        "pit_usable",
    ):
        if column not in frame:
            frame[column] = pd.NA
    keep = [
        "ticker",
        "refinitiv_ric",
        "sid",
        "name",
        "country",
        "exchange",
        "currency",
        "selected_for_u2",
        "selected_for_u3",
        "has_refinitiv_ric",
        "point_in_time_membership_claimed",
        "planning_universe_status",
        "stable_listing_id",
        "stable_instrument_id",
        "listing_start",
        "listing_end",
        "known_at",
        "window_end_known_at",
        "right_open",
        "pit_usable",
    ]
    return frame[keep].sort_values("refinitiv_ric").reset_index(drop=True)


def _selected_specs() -> list[PITDatasetSpec]:
    wanted = set(G3_DATASETS)
    return [spec for spec in default_pit_dataset_specs() if spec.dataset_id in wanted]


def build_japan_g3_pit_data_plan(
    universe_path: str | Path,
    g1_validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/pit_data_plan",
    run_id: str = "PIT-DATA-PLAN-JPN-U2-20260711",
    start_date: str = "2006-01-01",
    end_date: str = "2026-07-06",
    batch_size: int = 25,
    lseg_connector_status: str = "not_checked",
) -> JapanPITDataPlanResult:
    """Build the Japan G3 queue without promoting a current-survivor universe."""

    source_path = Path(universe_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Japan universe not found: {source_path}")
    validation_manifest_path = Path(g1_validation_dir) / "manifest.json"
    validation_manifest = _read_json(validation_manifest_path)
    g1_admitted = _g1_is_admitted(validation_manifest)

    selected = _normalise_current_universe(
        pd.read_csv(source_path, low_memory=False), g1_admitted=g1_admitted
    )
    window_denominator_ready = bool(
        selected["point_in_time_membership_claimed"].fillna(False).astype(bool).all()
    )
    plan_admitted = g1_admitted and window_denominator_ready
    specs = _selected_specs()
    requirements = _dataset_requirements_frame(specs)
    probes = _field_probe_queue(specs)
    batches = _build_extraction_batches(
        selected,
        specs,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        tiers=("u2",),
    )
    if not plan_admitted:
        batches["status"] = "blocked_upstream_g1"

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, tuple[pd.DataFrame, Path]] = {
        "selected_universe": (selected, artifact_dir / "selected_universe.csv"),
        "dataset_requirements": (requirements, artifact_dir / "dataset_requirements.csv"),
        "field_probe_queue": (probes, artifact_dir / "field_probe_queue.csv"),
        "extraction_batches": (batches, artifact_dir / "extraction_batches.csv"),
        "target_artifacts": (_target_artifacts(specs), artifact_dir / "target_artifacts.csv"),
        "readiness_gates": (_readiness_gates(specs), artifact_dir / "readiness_gates.csv"),
    }
    for frame, path in outputs.values():
        frame.to_csv(path, index=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    summary_path = artifact_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# {run_id}",
                "",
                "## Scope",
                "- Market: Japan / JPX.",
                "- Lane: G3 PIT fundamentals, IBES revisions, and earnings events.",
                f"- Planning denominator: `{len(selected):,}` listing windows across `{selected['refinitiv_ric'].nunique():,}` RICs.",
                f"- Extraction batches: `{len(batches):,}`.",
                f"- Independent G1 admitted: `{g1_admitted}`.",
                f"- PIT listing-window denominator preserved: `{window_denominator_ready}`.",
                "",
                "## Admission",
                "This is a resumable acquisition plan. It is not scorecard-eligible until the independent G1 security-master validator admits a full point-in-time universe.",
                "Current listings remain explicitly labeled as current-survivor planning rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_hashes = {name: _sha256(path) for name, (_, path) in outputs.items()}
    artifact_hashes["summary"] = _sha256(summary_path)
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "complete" if plan_admitted else "blocked_upstream",
        "scorecard_eligible": plan_admitted,
        "evidence_scope": "full" if plan_admitted else "planning_only",
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "point_in_time_plan": True,
        "current_survivor": not window_denominator_ready,
        "country": "Japan",
        "country_code": "JPN",
        "exchange_scope": "JPX",
        "tier": "u2",
        "datasets": list(G3_DATASETS),
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": batch_size,
        "selected_names": int(selected["refinitiv_ric"].nunique()),
        "selected_listing_windows": int(len(selected)),
        "window_denominator_preserved": window_denominator_ready,
        "selected_tickers": int(selected["ticker"].notna().sum()),
        "names_with_refinitiv_ric": int(selected["has_refinitiv_ric"].sum()),
        "extraction_batches": int(len(batches)),
        "blocked_batches": int(batches["status"].ne("ready_after_field_validation").sum()),
        "lseg_connector_status": lseg_connector_status,
        "g1_independent_validation_admitted": g1_admitted,
        "generated_at_utc": generated_at,
        "provenance": {
            "universe_source": {
                "path": str(source_path.resolve()),
                "sha256": _sha256(source_path),
                "point_in_time_membership_claimed": window_denominator_ready,
            },
            "g1_independent_validation": {
                "path": str(validation_manifest_path.resolve()),
                "sha256": _sha256(validation_manifest_path) if validation_manifest_path.exists() else "",
                "admitted": g1_admitted,
            },
        },
        "artifact_sha256": artifact_hashes,
        "artifacts": {name: str(path.resolve()) for name, (_, path) in outputs.items()},
    }
    manifest["artifacts"]["summary"] = str(summary_path.resolve())
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return JapanPITDataPlanResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        selected_universe_path=outputs["selected_universe"][1],
        manifest_path=manifest_path,
        selected_names=int(selected["refinitiv_ric"].nunique()),
        extraction_batches=len(batches),
        scorecard_eligible=plan_admitted,
    )
