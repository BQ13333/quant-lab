from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from quant_lab.data.vendor_adapters import (
    RefinitivRetryableProviderError,
    is_refinitiv_retryable_provider_error,
)


SPECIAL_FIELD_PREFIXES = ("IBES:", "QA:")
SPECIAL_FIELD_VALUES = {"exchange_calendar", "manual_policy_table"}

CRITICAL_ROUTE_EXPANSION_BUNDLES: dict[str, tuple[dict[str, object], ...]] = {
    "security_master_pit": (
        {
            "field_role": "listing_status",
            "candidate_fields": (
                "TR.ListingStatus",
                "TR.InstrumentStatus",
                "TR.IssueStatus",
                "TR.SecurityStatus",
                "TR.AssetStatus",
                "TR.QuoteStatus",
            ),
            "required": True,
            "probe_reason": "current accepted map has no listing-status route",
        },
        {
            "field_role": "currency",
            "candidate_fields": (
                "TR.Currency",
                "TR.PriceCurrency",
                "TR.CurrencyCode",
                "TR.TradingCurrency",
                "TR.QuoteCurrency",
                "TR.PrimaryQuoteCurrency",
            ),
            "required": True,
            "probe_reason": "price-currency metadata is absent in U2 artifacts",
        },
        {
            "field_role": "delisting_date",
            "candidate_fields": ("TR.DelistingDate", "TR.RetireDate", "TR.LastTradeDate"),
            "required": False,
            "probe_reason": "current survivor universe needs listing-window evidence",
        },
    ),
    "corporate_actions": (
        {
            "field_role": "dividend_event",
            "candidate_fields": (
                "TR.DivExDate",
                "TR.DivPayDate",
                "TR.DivRecordDate",
                "TR.DivAnnouncementDate",
                "TR.DivUnadjustedGross",
                "TR.DivCurrency",
                "TR.DivType",
            ),
            "required": True,
            "probe_reason": "dividend snapshot routes exist but historical event route is unresolved",
        },
        {
            "field_role": "split_event",
            "candidate_fields": (
                "TR.SplitFactor",
                "TR.SplitEffectiveDate",
                "TR.CapitalChangeType",
                "TR.CapitalChangeEffectiveDate",
                "TR.CapitalChangeRatio",
                "TR.CAEventType",
                "TR.CAEffectiveDate",
                "TR.CAExDate",
                "TR.CAAdjustmentFactor",
            ),
            "required": True,
            "probe_reason": "split/adjustment event history is required for adjusted-return integrity",
        },
        {
            "field_role": "action_announcement",
            "candidate_fields": ("TR.CAAnnouncementDate", "TR.CapitalChangeAnnouncementDate"),
            "required": False,
            "probe_reason": "announcement dates are needed for leakage-safe corporate-action handling",
        },
    ),
    "delistings_listing_history": (
        {
            "field_role": "delisting_date",
            "candidate_fields": ("TR.DelistingDate", "TR.RetireDate", "TR.LastTradeDate"),
            "required": True,
            "probe_reason": "delisted names/listing windows are required to reduce survivorship bias",
        },
        {
            "field_role": "delisting_reason",
            "candidate_fields": ("TR.DelistingReason", "TR.RetireReason"),
            "required": False,
            "probe_reason": "delisting reasons help classify investable vs corporate-event exits",
        },
        {
            "field_role": "ric_change",
            "candidate_fields": (
                "TR.RICChangeDate",
                "TR.NewRIC",
                "TR.OldRIC",
                "TR.InstrumentChangeDate",
                "TR.SymbolChangeDate",
            ),
            "required": False,
            "probe_reason": "ticker/RIC change history is required for point-in-time joins",
        },
    ),
    "fundamentals_pit": (
        {
            "field_role": "release_date",
            "candidate_fields": (
                "TR.F.ReportDate",
                "TR.F.FilingDate",
                "TR.F.AnnouncementDate",
                "TR.F.ReleaseDate",
                "TR.F.DateReported",
                "TR.F.PeriodEndDate",
                "TR.FilingDate",
            ),
            "required": True,
            "probe_reason": "period end is not a true known-at date; search for report/filing dates",
        },
        {
            "field_role": "eps",
            "candidate_fields": (
                "TR.F.EPSDilutedExclExtra",
                "TR.F.EPSDiluted",
                "TR.F.EPSBasic",
                "TR.EPSActual",
            ),
            "required": True,
            "probe_reason": "fundamental EPS was unresolved in first route probe",
        },
    ),
    "ibes_estimates_revisions": (
        {
            "field_role": "target_price",
            "candidate_fields": (
                "TR.TargetPriceMean",
                "TR.PriceTargetMean",
                "TR.TargetMeanPrice",
                "TR.PriceTargetMedian",
                "TR.PriceTargetHigh",
                "TR.PriceTargetLow",
            ),
            "required": False,
            "probe_reason": "target-price history is unresolved",
        },
        {
            "field_role": "revision_time",
            "candidate_fields": (
                "TR.EstimateDate",
                "TR.EPSMeanDate",
                "TR.RevenueMeanDate",
                "TR.IBESDate",
                "TR.RevisionDate",
                "TR.ConsensusDate",
                "TR.LastUpdatedDate",
            ),
            "required": True,
            "probe_reason": "consensus histories need source/update dates for true PIT semantics",
        },
    ),
    "earnings_events_actuals": (
        {
            "field_role": "actual_eps",
            "candidate_fields": (
                "TR.EPSActual",
                "TR.EPSActValue",
                "TR.EPSActualValue",
                "TR.ActualEPS",
                "TR.IBESActualEPS",
                "TR.EPSAct",
            ),
            "required": True,
            "probe_reason": "actual EPS is unresolved",
        },
        {
            "field_role": "surprise",
            "candidate_fields": (
                "TR.EPSSurprise",
                "TR.EPSSurpriseValue",
                "TR.EPSSurprisePct",
                "TR.EPSActSurprise",
                "TR.IBESSurprise",
            ),
            "required": True,
            "probe_reason": "earnings surprise is unresolved",
        },
        {
            "field_role": "earnings_release",
            "candidate_fields": (
                "TR.EventStartDate",
                "TR.EarningsReleaseDate",
                "TR.EarningsAnnouncementDate",
                "TR.ResultsAnnouncementDate",
            ),
            "required": True,
            "probe_reason": "existing event route is stale for Singapore and needs alternatives",
        },
    ),
    "liquidity_market_cap_free_float": (
        {
            "field_role": "adv_value",
            "candidate_fields": ("TR.AvgDailyValTraded30D",),
            "required": True,
            "probe_reason": "direct historical ADV route is unresolved",
        },
        {
            "field_role": "turnover_value",
            "candidate_fields": ("TR.Turnover", "TR.TurnoverValue", "TR.TradeValue", "TR.ValueTraded"),
            "required": True,
            "probe_reason": "historical turnover/value-traded route can support lagged ADV and TCA",
        },
        {
            "field_role": "volume_history",
            "candidate_fields": ("ACVOL_UNS",),
            "required": False,
            "probe_reason": "vendor volume history can cross-check OHLCV volume and turnover derivations",
        },
    ),
}


@dataclass(frozen=True)
class PITPlanArtifacts:
    plan_dir: Path
    selected_universe: pd.DataFrame
    field_probe_queue: pd.DataFrame
    dataset_requirements: pd.DataFrame
    extraction_batches: pd.DataFrame


@dataclass(frozen=True)
class RefinitivPITFieldProbeArtifacts:
    run_id: str
    artifact_dir: Path
    sample_path: Path
    results_path: Path
    accepted_field_map_path: Path
    summary_path: Path
    manifest_path: Path
    results: pd.DataFrame
    accepted_field_map: pd.DataFrame


@dataclass(frozen=True)
class RefinitivCurrentSnapshotArtifacts:
    run_id: str
    artifact_dir: Path
    snapshot_parquet_path: Path
    snapshot_csv_path: Path
    field_map_path: Path
    quality_path: Path
    chunk_errors_path: Path
    manifest_path: Path
    snapshot: pd.DataFrame
    quality: pd.DataFrame
    chunk_errors: pd.DataFrame


@dataclass(frozen=True)
class RefinitivPITHistoricalProbeArtifacts:
    run_id: str
    artifact_dir: Path
    results_path: Path
    history_field_map_path: Path
    summary_path: Path
    manifest_path: Path
    results: pd.DataFrame
    history_field_map: pd.DataFrame


@dataclass(frozen=True)
class RefinitivPITHistoricalRunArtifacts:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    manifest_path: Path
    quality_path: Path
    errors_path: Path
    run_manifest_path: Path
    output_root: Path
    plan: pd.DataFrame
    manifest: pd.DataFrame
    quality: pd.DataFrame
    errors: pd.DataFrame


@dataclass(frozen=True)
class RefinitivPITRouteProbeArtifacts:
    run_id: str
    artifact_dir: Path
    results_path: Path
    summary_path: Path
    manifest_path: Path
    results: pd.DataFrame


@dataclass(frozen=True)
class RefinitivPITRouteHistoryMapArtifacts:
    run_id: str
    artifact_dir: Path
    route_results_path: Path
    history_field_map_path: Path
    summary_path: Path
    manifest_path: Path
    route_results: pd.DataFrame
    history_field_map: pd.DataFrame


def load_pit_plan_artifacts(plan_dir: str | Path) -> PITPlanArtifacts:
    """Load the PIT data plan files produced by `pit_data_plan.py`."""

    root = Path(plan_dir)
    required = {
        "selected_universe": root / "selected_universe.csv",
        "field_probe_queue": root / "field_probe_queue.csv",
        "dataset_requirements": root / "dataset_requirements.csv",
        "extraction_batches": root / "extraction_batches.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing PIT plan artifact(s): {missing}")
    return PITPlanArtifacts(
        plan_dir=root,
        selected_universe=pd.read_csv(required["selected_universe"], low_memory=False),
        field_probe_queue=pd.read_csv(required["field_probe_queue"], low_memory=False),
        dataset_requirements=pd.read_csv(required["dataset_requirements"], low_memory=False),
        extraction_batches=pd.read_csv(required["extraction_batches"], low_memory=False),
    )


def is_special_refinitiv_pit_candidate(field: object) -> bool:
    """Return True for plan entries that need a route other than `ld.get_data`."""

    text = str(field).strip()
    upper = text.upper()
    if not text:
        return True
    if upper.startswith(SPECIAL_FIELD_PREFIXES):
        return True
    if text in SPECIAL_FIELD_VALUES:
        return True
    if "=" in text and not upper.startswith("TR."):
        return True
    return False


def select_pit_probe_samples(
    selected_universe: pd.DataFrame,
    sample_per_country: int = 3,
    tier: str = "u3",
    ric_col: str = "refinitiv_ric",
) -> pd.DataFrame:
    """Select a small liquid RIC sample per country from the PIT plan universe."""

    if sample_per_country <= 0 or selected_universe.empty:
        return pd.DataFrame(columns=selected_universe.columns)
    if ric_col not in selected_universe.columns:
        raise ValueError(f"selected universe missing `{ric_col}`")

    data = selected_universe[selected_universe[ric_col].notna()].copy()
    tier_col = f"selected_for_{tier.lower()}"
    if tier_col in data.columns:
        tier_data = data[data[tier_col].fillna(False).astype(bool)].copy()
        if not tier_data.empty:
            data = tier_data
    if data.empty:
        return data

    if "adv_60_country_rank" in data.columns:
        adv_rank = pd.to_numeric(data["adv_60_country_rank"], errors="coerce")
    else:
        adv_rank = pd.Series(10**9, index=data.index, dtype="float64")
    data["_adv_rank_sort"] = adv_rank.fillna(10**9)
    sort_cols = [col for col in ("country", "_adv_rank_sort", "ticker", ric_col) if col in data.columns]
    sampled = (
        data.sort_values(sort_cols)
        .groupby("country", dropna=False, group_keys=False)
        .head(sample_per_country)
        .drop(columns=["_adv_rank_sort"], errors="ignore")
        .reset_index(drop=True)
    )
    return sampled


def filter_field_probe_queue(
    field_probe_queue: pd.DataFrame,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    if datasets is None:
        queue = field_probe_queue.copy()
    else:
        wanted = {dataset.strip() for dataset in datasets if str(dataset).strip()}
        queue = field_probe_queue[field_probe_queue["dataset_id"].astype(str).isin(wanted)].copy()
    queue = queue.reset_index(drop=True)
    queue["candidate_order"] = range(len(queue))
    return queue


def critical_refinitiv_route_probe_queue(
    datasets: Sequence[str] | None = None,
    field_roles: Sequence[str] | None = None,
    include_probe_reason: bool = True,
) -> pd.DataFrame:
    """Build a deterministic expanded queue for unresolved production blockers.

    These are candidate routes, not accepted fields. A row becomes usable only
    after a live route probe returns source data and later history/leakage QA
    promotes it.
    """

    wanted_datasets = {item.strip() for item in datasets or () if str(item).strip()}
    wanted_roles = {item.strip() for item in field_roles or () if str(item).strip()}
    rows: list[dict[str, Any]] = []
    for dataset_id, bundle in CRITICAL_ROUTE_EXPANSION_BUNDLES.items():
        if wanted_datasets and dataset_id not in wanted_datasets:
            continue
        for entry in bundle:
            field_role = str(entry["field_role"])
            if wanted_roles and field_role not in wanted_roles:
                continue
            for candidate_field in entry["candidate_fields"]:
                row = {
                    "dataset_id": dataset_id,
                    "owner_agent": _default_owner_agent(dataset_id),
                    "source_family": _default_source_family(dataset_id),
                    "field_role": field_role,
                    "candidate_field": candidate_field,
                    "required": bool(entry.get("required")),
                    "validation_status": "needs_live_route_probe",
                    "validation_method": "LSEG Workspace get_data/get_history U3 smoke before any full backfill.",
                    "candidate_source": "critical_route_expansion_v1",
                }
                if include_probe_reason:
                    row["probe_reason"] = str(entry.get("probe_reason", ""))
                rows.append(row)
    queue = pd.DataFrame(rows)
    if queue.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "owner_agent",
                "source_family",
                "field_role",
                "candidate_field",
                "required",
                "validation_status",
                "validation_method",
                "candidate_source",
                "probe_reason",
            ]
        )
    return queue.drop_duplicates(["dataset_id", "field_role", "candidate_field"]).reset_index(drop=True)


def probe_refinitiv_pit_fields(
    ld: Any,
    selected_universe: pd.DataFrame,
    field_probe_queue: pd.DataFrame,
    sample_per_country: int = 3,
    tier: str = "u3",
    datasets: Sequence[str] | None = None,
    sleep_seconds: float = 0.05,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Probe candidate Refinitiv fields with `ld.get_data` on a country sample."""

    samples = select_pit_probe_samples(selected_universe, sample_per_country=sample_per_country, tier=tier)
    queue = filter_field_probe_queue(field_probe_queue, datasets=datasets)
    sample_rics = samples["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist() if not samples.empty else []
    rows: list[dict[str, Any]] = []
    total = len(queue)

    for completed, candidate in enumerate(queue.itertuples(index=False), start=1):
        candidate_dict = candidate._asdict()
        field = str(candidate_dict.get("candidate_field", "")).strip()
        row = {
            "dataset_id": candidate_dict.get("dataset_id"),
            "owner_agent": candidate_dict.get("owner_agent"),
            "source_family": candidate_dict.get("source_family"),
            "field_role": candidate_dict.get("field_role"),
            "candidate_field": field,
            "candidate_order": candidate_dict.get("candidate_order"),
            "required": bool(candidate_dict.get("required")),
            "tier": tier,
            "sample_count": len(sample_rics),
            "sample_rics": ",".join(sample_rics),
            "returned_rows": 0,
            "returned_columns": "",
            "returned_data_columns": "",
            "returned_data_column_count": 0,
            "non_null_values": 0,
            "coverage": 0.0,
            "status": "not_run",
            "error": "",
            "probed_at_utc": _now_utc(),
        }
        if not sample_rics:
            row["status"] = "blocked_no_sample_rics"
            rows.append(row)
            continue
        if is_special_refinitiv_pit_candidate(field):
            row["status"] = "needs_special_provider"
            row["error"] = "candidate requires a route other than ld.get_data"
            rows.append(row)
            continue

        try:
            frame = _get_data_frame(ld, universe=sample_rics, fields=[field])
            row.update(_get_data_stats(frame, sample_count=len(sample_rics)))
            row["status"] = "validated" if row["non_null_values"] > 0 else "field_returned_empty"
        except Exception as exc:
            retryable = isinstance(
                exc, RefinitivRetryableProviderError
            ) or is_refinitiv_retryable_provider_error(exc)
            row["status"] = "failed_retryable_provider" if retryable else "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

        if row["status"] == "failed_retryable_provider":
            break

        if progress_every > 0 and (completed % progress_every == 0 or completed == total):
            print(f"Probed PIT fields {completed}/{total}: {field}", file=sys.stderr, flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return pd.DataFrame(rows)


def accepted_refinitiv_pit_field_map(probe_results: pd.DataFrame) -> pd.DataFrame:
    """Pick the first live-validated candidate per dataset/field role."""

    if probe_results.empty:
        return pd.DataFrame()
    accepted = probe_results[probe_results["status"].eq("validated")].copy()
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "field_role",
                "candidate_field",
                "returned_data_columns",
                "required",
                "sample_count",
                "non_null_values",
                "coverage",
                "acceptance_status",
            ]
        )

    accepted["candidate_order"] = pd.to_numeric(accepted["candidate_order"], errors="coerce").fillna(10**9)
    accepted = accepted.sort_values(
        ["dataset_id", "field_role", "candidate_order", "coverage", "non_null_values"],
        ascending=[True, True, True, False, False],
    )
    accepted = accepted.drop_duplicates(["dataset_id", "field_role"], keep="first").copy()
    accepted["acceptance_status"] = "accepted_live_probe"
    keep = [
        "dataset_id",
        "owner_agent",
        "source_family",
        "field_role",
        "candidate_field",
        "returned_data_columns",
        "required",
        "sample_count",
        "non_null_values",
        "coverage",
        "acceptance_status",
        "probed_at_utc",
    ]
    return accepted[[col for col in keep if col in accepted.columns]].reset_index(drop=True)


def write_refinitiv_pit_field_probe_artifacts(
    artifact_root: str | Path,
    run_id: str,
    selected_universe: pd.DataFrame,
    field_probe_queue: pd.DataFrame,
    results: pd.DataFrame,
    sample_per_country: int,
    tier: str,
    datasets: Sequence[str] | None,
    session_name: str | None,
) -> RefinitivPITFieldProbeArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    samples = select_pit_probe_samples(selected_universe, sample_per_country=sample_per_country, tier=tier)
    accepted = accepted_refinitiv_pit_field_map(results)

    sample_path = artifact_dir / "sample_universe.csv"
    results_path = artifact_dir / "field_probe_results.csv"
    accepted_path = artifact_dir / "accepted_field_map.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    samples.to_csv(sample_path, index=False)
    results.to_csv(results_path, index=False)
    accepted.to_csv(accepted_path, index=False)
    summary_path.write_text(
        _field_probe_summary_md(run_id, results, accepted, datasets=datasets),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "tier": tier,
        "sample_per_country": sample_per_country,
        "datasets": list(datasets) if datasets else [],
        "sample_rics": int(samples["refinitiv_ric"].nunique()) if "refinitiv_ric" in samples else 0,
        "queue_rows": int(len(field_probe_queue)),
        "probe_rows": int(len(results)),
        "validated_fields": int(results["status"].eq("validated").sum()) if "status" in results else 0,
        "accepted_fields": int(len(accepted)),
        "special_route_fields": int(results["status"].eq("needs_special_provider").sum()) if "status" in results else 0,
        "error_fields": int(results["status"].eq("error").sum()) if "status" in results else 0,
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "sample_universe": str(sample_path.resolve()),
            "field_probe_results": str(results_path.resolve()),
            "accepted_field_map": str(accepted_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivPITFieldProbeArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        sample_path=sample_path,
        results_path=results_path,
        accepted_field_map_path=accepted_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        results=results,
        accepted_field_map=accepted,
    )


def extract_refinitiv_current_snapshot(
    ld: Any,
    selected_universe: pd.DataFrame,
    accepted_field_map: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    tier: str = "u2",
    datasets: Sequence[str] | None = None,
    chunk_size: int = 50,
    sleep_seconds: float = 0.05,
    session_name: str | None = None,
) -> RefinitivCurrentSnapshotArtifacts:
    """Extract a current Refinitiv snapshot for accepted get_data fields."""

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    universe = _filter_selected_universe(selected_universe, tier=tier)
    fmap = _filter_accepted_field_map(accepted_field_map, datasets=datasets)
    fields = list(dict.fromkeys(fmap["candidate_field"].dropna().astype(str).tolist())) if not fmap.empty else []
    loaded_at = _now_utc()
    asof_date = datetime.now(timezone.utc).date().isoformat()

    frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, Any]] = []
    rics = universe["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist() if "refinitiv_ric" in universe else []
    for chunk_number, chunk in enumerate(_chunked(rics, max(1, chunk_size)), start=1):
        if not fields:
            error_rows.append(
                {
                    "chunk_number": chunk_number,
                    "ric_count": len(chunk),
                    "rics": ",".join(chunk),
                    "field": "",
                    "error": "no accepted fields",
                }
            )
            continue
        frame, errors = _get_data_with_field_fallback(ld, universe=chunk, fields=fields)
        for error in errors:
            error_rows.append({"chunk_number": chunk_number, "ric_count": len(chunk), "rics": ",".join(chunk), **error})
        if not frame.empty:
            frame = _attach_requested_ric(frame, chunk)
            frame["source_loaded_at_utc"] = loaded_at
            frame["asof_date"] = asof_date
            frame["snapshot_scope"] = "current_snapshot_only"
            frames.append(frame)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    snapshot = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    snapshot = _merge_universe_metadata(snapshot, universe)
    quality = _snapshot_quality(snapshot, field_map=fmap, requested_rics=rics)
    chunk_errors = pd.DataFrame(error_rows, columns=["chunk_number", "ric_count", "rics", "field", "error"])

    snapshot_parquet_path = artifact_dir / "current_snapshot.parquet"
    snapshot_csv_path = artifact_dir / "current_snapshot.csv"
    field_map_path = artifact_dir / "snapshot_field_map.csv"
    quality_path = artifact_dir / "snapshot_quality.csv"
    chunk_errors_path = artifact_dir / "chunk_errors.csv"
    manifest_path = artifact_dir / "manifest.json"

    if snapshot.empty:
        snapshot.to_csv(snapshot_csv_path, index=False)
    else:
        snapshot.to_parquet(snapshot_parquet_path, index=False)
        snapshot.to_csv(snapshot_csv_path, index=False)
    fmap.to_csv(field_map_path, index=False)
    quality.to_csv(quality_path, index=False)
    chunk_errors.to_csv(chunk_errors_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "tier": tier,
        "datasets": list(datasets) if datasets else [],
        "scope": "current_snapshot_only",
        "asof_date": asof_date,
        "source_loaded_at_utc": loaded_at,
        "requested_rics": int(len(rics)),
        "accepted_fields": int(len(fields)),
        "snapshot_rows": int(len(snapshot)),
        "chunk_errors": int(len(chunk_errors)),
        "non_empty_snapshot": bool(not snapshot.empty),
        "artifacts": {
            "snapshot_parquet": str(snapshot_parquet_path.resolve()) if snapshot_parquet_path.exists() else "",
            "snapshot_csv": str(snapshot_csv_path.resolve()),
            "snapshot_field_map": str(field_map_path.resolve()),
            "snapshot_quality": str(quality_path.resolve()),
            "chunk_errors": str(chunk_errors_path.resolve()),
        },
        "warning": "Current snapshot is for security-master/liquidity QA only; it is not historical point-in-time evidence.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivCurrentSnapshotArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        snapshot_parquet_path=snapshot_parquet_path,
        snapshot_csv_path=snapshot_csv_path,
        field_map_path=field_map_path,
        quality_path=quality_path,
        chunk_errors_path=chunk_errors_path,
        manifest_path=manifest_path,
        snapshot=snapshot,
        quality=quality,
        chunk_errors=chunk_errors,
    )


def probe_refinitiv_pit_historical_fields(
    ld: Any,
    selected_universe: pd.DataFrame,
    accepted_field_map: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str = "1M",
    sample_per_country: int = 2,
    tier: str = "u3",
    datasets: Sequence[str] | None = None,
    sleep_seconds: float = 0.05,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Probe accepted snapshot fields through `ld.get_history` before batch extraction."""

    samples = select_pit_probe_samples(selected_universe, sample_per_country=sample_per_country, tier=tier)
    sample_rics = samples["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist() if not samples.empty else []
    fmap = _filter_accepted_field_map(accepted_field_map, datasets=datasets)
    rows: list[dict[str, Any]] = []
    total = len(fmap)
    for completed, item in enumerate(fmap.itertuples(index=False), start=1):
        item_dict = item._asdict()
        field = str(item_dict.get("candidate_field", "")).strip()
        row = {
            "dataset_id": item_dict.get("dataset_id"),
            "owner_agent": item_dict.get("owner_agent"),
            "source_family": item_dict.get("source_family"),
            "field_role": item_dict.get("field_role"),
            "candidate_field": field,
            "returned_data_columns": item_dict.get("returned_data_columns", ""),
            "tier": tier,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "sample_count": len(sample_rics),
            "sample_rics": ",".join(sample_rics),
            "returned_rows": 0,
            "returned_columns": "",
            "non_null_values": 0,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "inside_requested_rows": 0,
            "outside_requested_rows": 0,
            "coverage": 0.0,
            "status": "not_run",
            "error": "",
            "probed_at_utc": _now_utc(),
        }
        if not sample_rics:
            row["status"] = "blocked_no_sample_rics"
            rows.append(row)
            continue
        try:
            frame = _get_history_frame(
                ld,
                universe=sample_rics,
                fields=[field],
                start=start_date,
                end=end_date,
                interval=interval,
            )
            row.update(_history_stats(frame, sample_rics, start_date=start_date, end_date=end_date))
            row["status"] = _historical_probe_status(row)
        except Exception as exc:
            retryable = isinstance(
                exc, RefinitivRetryableProviderError
            ) or is_refinitiv_retryable_provider_error(exc)
            row["status"] = "failed_retryable_provider" if retryable else "history_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if row["status"] == "failed_retryable_provider":
            break
        if progress_every > 0 and (completed % progress_every == 0 or completed == total):
            print(f"Probed PIT history {completed}/{total}: {field}", file=sys.stderr, flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def accepted_refinitiv_pit_history_field_map(historical_probe_results: pd.DataFrame) -> pd.DataFrame:
    """Return fields that have usable rows inside the requested history window."""

    if historical_probe_results.empty:
        return pd.DataFrame()
    accepted = historical_probe_results[historical_probe_results["status"].eq("historical_validated")].copy()
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "field_role",
                "candidate_field",
                "returned_data_columns",
                "interval",
                "history_status",
                "coverage",
            ]
        )
    accepted["history_status"] = accepted["status"]
    keep = [
        "dataset_id",
        "owner_agent",
        "source_family",
        "field_role",
        "candidate_field",
        "returned_data_columns",
        "interval",
        "sample_count",
        "non_null_values",
        "inside_requested_rows",
        "outside_requested_rows",
        "coverage",
        "history_status",
        "probed_at_utc",
    ]
    return accepted[[col for col in keep if col in accepted.columns]].reset_index(drop=True)


def write_refinitiv_pit_historical_probe_artifacts(
    artifact_root: str | Path,
    run_id: str,
    results: pd.DataFrame,
    datasets: Sequence[str] | None,
    session_name: str | None,
) -> RefinitivPITHistoricalProbeArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_field_map = accepted_refinitiv_pit_history_field_map(results)

    results_path = artifact_dir / "historical_probe_results.csv"
    history_field_map_path = artifact_dir / "history_field_map.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    results.to_csv(results_path, index=False)
    history_field_map.to_csv(history_field_map_path, index=False)
    summary_path.write_text(_historical_probe_summary_md(run_id, results, history_field_map, datasets=datasets), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "datasets": list(datasets) if datasets else [],
        "probe_rows": int(len(results)),
        "historical_fields": int(len(history_field_map)),
        "status_counts": {str(key): int(value) for key, value in results["status"].value_counts(dropna=False).items()}
        if "status" in results
        else {},
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "historical_probe_results": str(results_path.resolve()),
            "history_field_map": str(history_field_map_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivPITHistoricalProbeArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        results_path=results_path,
        history_field_map_path=history_field_map_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        results=results,
        history_field_map=history_field_map,
    )


def build_refinitiv_pit_historical_plan(
    selected_universe: pd.DataFrame,
    history_field_map: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    tier: str = "u2",
    datasets: Sequence[str] | None = None,
    batch_size: int = 25,
    window_years: int = 5,
    interval: str = "1M",
) -> pd.DataFrame:
    """Create a field-by-field PIT history extraction plan."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    universe = _filter_selected_universe(selected_universe, tier=tier)
    fmap = _filter_history_field_map(history_field_map, datasets=datasets)
    if universe.empty:
        raise ValueError("selected universe is empty")
    if fmap.empty:
        raise ValueError("history field map is empty")
    windows = _date_windows(start_date=start_date, end_date=end_date, window_years=window_years)
    output_root = Path(artifact_root) / run_id / "history"
    rows: list[dict[str, Any]] = []
    for country, country_group in universe.groupby("country", dropna=False, sort=True):
        country_rics = country_group["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist()
        for batch_number, rics in enumerate(_chunked(country_rics, batch_size), start=1):
            for fmap_row in fmap.itertuples(index=False):
                field_item = fmap_row._asdict()
                dataset_id = str(field_item.get("dataset_id"))
                field_role = str(field_item.get("field_role"))
                field = str(field_item.get("candidate_field"))
                for window_number, (window_start, window_end) in enumerate(windows, start=1):
                    row_id = (
                        f"{_slugish(dataset_id)}-{_slugish(field_role)}-{_slugish(country)}-"
                        f"{batch_number:03d}-{window_number:03d}"
                    )
                    output_path = (
                        output_root
                        / f"dataset_id={_slugish(dataset_id)}"
                        / f"field_role={_slugish(field_role)}"
                        / f"country={_slugish(country)}"
                        / f"batch={batch_number:03d}"
                        / f"window={window_number:03d}_{window_start}_{window_end}.parquet"
                    )
                    rows.append(
                        {
                            "plan_row_id": row_id,
                            "run_id": run_id,
                            "dataset_id": dataset_id,
                            "owner_agent": field_item.get("owner_agent", ""),
                            "source_family": field_item.get("source_family", ""),
                            "field_role": field_role,
                            "candidate_field": field,
                            "returned_data_columns": field_item.get("returned_data_columns", ""),
                            "country": country,
                            "batch_number": batch_number,
                            "window_number": window_number,
                            "start_date": window_start,
                            "end_date": window_end,
                            "interval": interval,
                            "tier": tier,
                            "requested_rics": len(rics),
                            "ric_list": ",".join(rics),
                            "output_path": str(output_path),
                            "status": "pending",
                        }
                    )
    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("PIT historical plan is empty")
    return plan


def execute_refinitiv_pit_historical_plan(
    ld: Any,
    plan: pd.DataFrame,
    selected_universe: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    max_plan_rows: int | None = None,
    resume: bool = True,
    skip_failed_on_resume: bool = False,
    sleep_seconds: float = 0.05,
    session_name: str | None = None,
    request_timeout_seconds: float | None = None,
) -> RefinitivPITHistoricalRunArtifacts:
    """Execute a PIT history plan, writing each batch to a small parquet file."""

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_dir / "historical_plan.csv"
    manifest_path = artifact_dir / "historical_manifest.csv"
    quality_path = artifact_dir / "historical_quality.csv"
    errors_path = artifact_dir / "historical_errors.csv"
    run_manifest_path = artifact_dir / "manifest.json"
    output_root = artifact_dir / "history"
    stage_refinitiv_pit_historical_plan(plan, plan_path)
    lock_handle = _acquire_historical_run_lock(artifact_dir / ".historical_backfill.lock")

    existing_manifest = pd.read_csv(manifest_path, low_memory=False) if resume and manifest_path.exists() else pd.DataFrame()
    existing_manifest = _matching_historical_manifest_rows(existing_manifest, plan)
    resume_statuses = ["completed", "completed_partial", "completed_empty"]
    if skip_failed_on_resume:
        resume_statuses.append("failed")
    completed = (
        set(existing_manifest.loc[existing_manifest["status"].isin(resume_statuses), "plan_row_id"].astype(str))
        if not existing_manifest.empty
        else set()
    )
    pending = plan[~plan["plan_row_id"].astype(str).isin(completed)].copy()
    if max_plan_rows is not None:
        if max_plan_rows <= 0:
            raise ValueError("max_plan_rows must be positive")
        pending = pending.head(max_plan_rows)

    manifest_rows = existing_manifest.to_dict("records") if not existing_manifest.empty else []
    quality_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    universe = selected_universe.copy()
    if "refinitiv_ric" in universe.columns:
        universe["refinitiv_ric"] = universe["refinitiv_ric"].astype(str)

    provider_halt: dict[str, Any] | None = None
    for item in pending.itertuples(index=False):
        stop_after_row = False
        row = item._asdict()
        rics = [ric for ric in str(row.get("ric_list", "")).split(",") if ric]
        started_at = _now_utc()
        output_path = Path(str(row["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history, identifier_errors = _get_history_frame_with_single_ric_fallback(
                ld,
                universe=rics,
                fields=[str(row["candidate_field"])],
                start=str(row["start_date"]),
                end=str(row["end_date"]),
                interval=str(row["interval"]),
                request_timeout_seconds=request_timeout_seconds,
            )
            stats = _history_stats(history, rics, start_date=str(row["start_date"]), end_date=str(row["end_date"]))
            long = _history_frame_to_long(history, rics=rics, plan_row=row, universe=universe, source_loaded_at_utc=started_at)
            if not long.empty:
                long.to_parquet(output_path, index=False)
                long_csv = output_path.with_suffix(".csv")
                long.to_csv(long_csv, index=False)
            if stats["non_null_values"] <= 0 or stats["inside_requested_rows"] <= 0:
                status = "completed_empty"
            else:
                status = "completed_partial" if identifier_errors else "completed"
            manifest_row = {
                **row,
                **stats,
                "history_rows": int(len(long)),
                "status": status,
                "started_at_utc": started_at,
                "completed_at_utc": _now_utc(),
                "error_type": "partial_identifier_failures" if identifier_errors else "",
                "error": " | ".join(identifier_errors),
            }
            quality_rows.append(_quality_row_from_manifest(manifest_row))
        except Exception as exc:
            retryable = is_refinitiv_retryable_provider_error(exc)
            error = RefinitivRetryableProviderError(str(exc)) if retryable else exc
            manifest_row = {
                **row,
                "returned_rows": 0,
                "returned_columns": "",
                "non_null_values": 0,
                "first_available": pd.NaT,
                "last_available": pd.NaT,
                "inside_requested_rows": 0,
                "outside_requested_rows": 0,
                "coverage": 0.0,
                "history_rows": 0,
                "status": "failed_retryable_provider" if retryable else "failed",
                "started_at_utc": started_at,
                "completed_at_utc": _now_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            error_rows.append(
                {
                    "plan_row_id": row.get("plan_row_id"),
                    "dataset_id": row.get("dataset_id"),
                    "field_role": row.get("field_role"),
                    "candidate_field": row.get("candidate_field"),
                    "country": row.get("country"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if retryable:
                provider_halt = {
                    "plan_row_id": str(row.get("plan_row_id", "")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                stop_after_row = True
        manifest_rows = [existing for existing in manifest_rows if str(existing.get("plan_row_id")) != str(row.get("plan_row_id"))]
        manifest_rows.append(manifest_row)
        _atomic_write_csv(
            pd.DataFrame(manifest_rows).sort_values(
                ["dataset_id", "field_role", "country", "batch_number", "window_number"]
            ),
            manifest_path,
        )
        if stop_after_row:
            break

    manifest = pd.read_csv(manifest_path, low_memory=False) if manifest_path.exists() else pd.DataFrame(manifest_rows)
    quality = _historical_quality(manifest)
    errors = _historical_errors(manifest)
    _atomic_write_csv(quality, quality_path)
    _atomic_write_csv(errors, errors_path)
    run_manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "scope": "historical_pit_batch",
        "planned_rows": int(len(plan)),
        "executed_manifest_rows": int(len(manifest)),
        "completed_rows": int(manifest["status"].isin(["completed", "completed_partial"]).sum()) if "status" in manifest else 0,
        "failed_rows": int(manifest["status"].astype(str).str.startswith("failed").sum()) if "status" in manifest else 0,
        "empty_rows": int(manifest["status"].eq("completed_empty").sum()) if "status" in manifest else 0,
        "history_rows": int(pd.to_numeric(manifest.get("history_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not manifest.empty
        else 0,
        "output_root": str(output_root.resolve()),
        "halted_early": provider_halt is not None,
        "provider_halt": provider_halt,
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "historical_plan": str(plan_path.resolve()),
            "historical_manifest": str(manifest_path.resolve()),
            "historical_quality": str(quality_path.resolve()),
            "historical_errors": str(errors_path.resolve()),
        },
    }
    _atomic_write_json(run_manifest, run_manifest_path)
    artifacts = RefinitivPITHistoricalRunArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        plan_path=plan_path,
        manifest_path=manifest_path,
        quality_path=quality_path,
        errors_path=errors_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
        plan=plan,
        manifest=manifest,
        quality=quality,
        errors=errors,
    )
    lock_handle.close()
    return artifacts


def stage_refinitiv_pit_historical_plan(plan: pd.DataFrame, plan_path: str | Path) -> str:
    """Write a PIT history plan once and reject any resume-time drift."""

    path = Path(plan_path)
    checksum_path = path.with_suffix(".sha256")
    payload = plan.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"immutable PIT historical plan drift: {path}")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        if expected != digest:
            raise ValueError(f"immutable PIT historical plan checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    if not checksum_path.exists():
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{checksum_path.name}.", suffix=".tmp", dir=checksum_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(digest + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, checksum_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return digest


def _matching_historical_manifest_rows(existing: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Keep resume rows only when their complete request identity matches the current plan."""

    if existing.empty or plan.empty or "plan_row_id" not in existing or "plan_row_id" not in plan:
        return existing.iloc[0:0].copy()
    identity_columns = [
        "run_id",
        "dataset_id",
        "field_role",
        "candidate_field",
        "country",
        "batch_number",
        "window_number",
        "start_date",
        "end_date",
        "interval",
        "tier",
        "ric_list",
        "output_path",
    ]
    plan_by_id = {str(row["plan_row_id"]): row for row in plan.to_dict("records")}
    keep: list[bool] = []
    for row in existing.to_dict("records"):
        planned = plan_by_id.get(str(row.get("plan_row_id")))
        keep.append(
            planned is not None
            and all(
                _resume_identity_value(row.get(column)) == _resume_identity_value(planned.get(column))
                for column in identity_columns
            )
        )
    return existing.loc[keep].copy().reset_index(drop=True)


def _resume_identity_value(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return ""
    return str(value).strip()


def recover_refinitiv_pit_historical_manifest(
    plan: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    session_name: str | None = None,
) -> RefinitivPITHistoricalRunArtifacts:
    """Rebuild a historical manifest from validated parquet checkpoints."""

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_dir / "historical_plan.csv"
    manifest_path = artifact_dir / "historical_manifest.csv"
    quality_path = artifact_dir / "historical_quality.csv"
    errors_path = artifact_dir / "historical_errors.csv"
    run_manifest_path = artifact_dir / "manifest.json"
    output_root = artifact_dir / "history"
    stage_refinitiv_pit_historical_plan(plan, plan_path)
    lock_handle = _acquire_historical_run_lock(artifact_dir / ".historical_backfill.lock")

    recovered: list[dict[str, Any]] = []
    for row in plan.to_dict("records"):
        output_path = Path(str(row["output_path"]))
        if not output_path.exists():
            continue
        long = pd.read_parquet(output_path)
        required = {"plan_row_id", "asof_date", "requested_ric", "value"}
        if not required.issubset(long.columns):
            raise ValueError(f"checkpoint missing required columns: {output_path}")
        checkpoint_ids = set(long["plan_row_id"].dropna().astype(str))
        if checkpoint_ids != {str(row["plan_row_id"])}:
            raise ValueError(f"checkpoint plan-row identity mismatch: {output_path}")
        dates = pd.to_datetime(long["asof_date"], errors="coerce")
        values = long["value"].notna()
        start = pd.Timestamp(str(row["start_date"]))
        end = pd.Timestamp(str(row["end_date"]))
        valid_dates = dates[values & dates.notna()]
        inside_dates = valid_dates[(valid_dates >= start) & (valid_dates <= end)].drop_duplicates()
        outside_dates = valid_dates[(valid_dates < start) | (valid_dates > end)].drop_duplicates()
        non_null_values = int(values.sum())
        requested_rics = max(1, int(row.get("requested_rics") or 0))
        coverage = min(1.0, non_null_values / max(1, requested_rics * len(inside_dates))) if len(inside_dates) else 0.0
        loaded_source = (
            long["source_loaded_at_utc"]
            if "source_loaded_at_utc" in long.columns
            else pd.Series(pd.NaT, index=long.index)
        )
        loaded = pd.to_datetime(loaded_source, errors="coerce", utc=True)
        recovered.append(
            {
                **row,
                "returned_rows": int(valid_dates.nunique()),
                "returned_columns": "|".join(sorted(long.loc[values, "requested_ric"].dropna().astype(str).unique())),
                "non_null_values": non_null_values,
                "first_available": valid_dates.min() if not valid_dates.empty else pd.NaT,
                "last_available": valid_dates.max() if not valid_dates.empty else pd.NaT,
                "inside_requested_rows": int(len(inside_dates)),
                "outside_requested_rows": int(len(outside_dates)),
                "coverage": float(coverage),
                "history_rows": int(len(long)),
                "status": "completed" if non_null_values > 0 and len(inside_dates) > 0 else "completed_empty",
                "started_at_utc": loaded.min().isoformat() if loaded.notna().any() else "",
                "completed_at_utc": loaded.max().isoformat() if loaded.notna().any() else "",
                "error_type": "",
                "error": "",
            }
        )

    manifest = pd.DataFrame(recovered)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["dataset_id", "field_role", "country", "batch_number", "window_number"]
        ).reset_index(drop=True)
    quality = _historical_quality(manifest)
    errors = _historical_errors(manifest)
    _atomic_write_csv(manifest, manifest_path)
    _atomic_write_csv(quality, quality_path)
    _atomic_write_csv(errors, errors_path)
    run_manifest = _historical_run_manifest(
        run_id=run_id,
        artifact_dir=artifact_dir,
        session_name=session_name,
        plan=plan,
        manifest=manifest,
        output_root=output_root,
        plan_path=plan_path,
        manifest_path=manifest_path,
        quality_path=quality_path,
        errors_path=errors_path,
    )
    run_manifest["recovered_from_checkpoints"] = True
    _atomic_write_json(run_manifest, run_manifest_path)
    artifacts = RefinitivPITHistoricalRunArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        plan_path=plan_path,
        manifest_path=manifest_path,
        quality_path=quality_path,
        errors_path=errors_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
        plan=plan,
        manifest=manifest,
        quality=quality,
        errors=errors,
    )
    lock_handle.close()
    return artifacts


def _historical_run_manifest(
    *,
    run_id: str,
    artifact_dir: Path,
    session_name: str | None,
    plan: pd.DataFrame,
    manifest: pd.DataFrame,
    output_root: Path,
    plan_path: Path,
    manifest_path: Path,
    quality_path: Path,
    errors_path: Path,
) -> dict[str, Any]:
    status = manifest.get("status", pd.Series(dtype=str)).astype(str) if not manifest.empty else pd.Series(dtype=str)
    history_rows = pd.to_numeric(
        manifest.get("history_rows", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "scope": "historical_pit_batch",
        "planned_rows": int(len(plan)),
        "executed_manifest_rows": int(len(manifest)),
        "completed_rows": int(status.isin(["completed", "completed_partial"]).sum()),
        "failed_rows": int(status.eq("failed").sum()),
        "empty_rows": int(status.eq("completed_empty").sum()),
        "history_rows": int(history_rows.sum()),
        "output_root": str(output_root.resolve()),
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "historical_plan": str(plan_path.resolve()),
            "historical_manifest": str(manifest_path.resolve()),
            "historical_quality": str(quality_path.resolve()),
            "historical_errors": str(errors_path.resolve()),
        },
    }


def _acquire_historical_run_lock(path: Path):
    import fcntl

    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"historical backfill already has an active writer: {path}") from exc
    return handle


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def _vendor_request_timeout(seconds: float | None):
    """Bound a synchronous vendor request on platforms that expose SIGALRM."""

    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"vendor request exceeded {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def probe_refinitiv_pit_route_candidates(
    ld: Any,
    selected_universe: pd.DataFrame,
    field_probe_queue: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str = "1M",
    sample_per_country: int = 2,
    tier: str = "u3",
    datasets: Sequence[str] | None = None,
    routes: Sequence[str] = ("get_data", "get_history"),
    sleep_seconds: float = 0.05,
    progress_every: int = 0,
    request_timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Probe unresolved PIT families across snapshot and history routes."""

    samples = select_pit_probe_samples(selected_universe, sample_per_country=sample_per_country, tier=tier)
    sample_rics = samples["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist() if not samples.empty else []
    queue = filter_field_probe_queue(field_probe_queue, datasets=datasets)
    rows: list[dict[str, Any]] = []
    work = [(item._asdict(), route) for item in queue.itertuples(index=False) for route in routes]
    for completed, (candidate, route) in enumerate(work, start=1):
        field = str(candidate.get("candidate_field", "")).strip()
        row = {
            "dataset_id": candidate.get("dataset_id"),
            "owner_agent": candidate.get("owner_agent"),
            "source_family": candidate.get("source_family"),
            "field_role": candidate.get("field_role"),
            "candidate_field": field,
            "required": bool(candidate.get("required")),
            "route": route,
            "tier": tier,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "sample_count": len(sample_rics),
            "sample_rics": ",".join(sample_rics),
            "returned_rows": 0,
            "returned_columns": "",
            "non_null_values": 0,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "inside_requested_rows": 0,
            "outside_requested_rows": 0,
            "coverage": 0.0,
            "status": "not_run",
            "error": "",
            "probed_at_utc": _now_utc(),
        }
        if not sample_rics:
            row["status"] = "blocked_no_sample_rics"
            rows.append(row)
            continue
        if is_special_refinitiv_pit_candidate(field):
            row["status"] = "needs_special_provider"
            row["error"] = "candidate requires a specialized endpoint/export route"
            rows.append(row)
            continue
        try:
            if route == "get_data":
                with _vendor_request_timeout(request_timeout_seconds):
                    frame = _get_data_frame(ld, universe=sample_rics, fields=[field])
                row.update(_get_data_stats(frame, sample_count=len(sample_rics)))
                row["status"] = "snapshot_validated" if row["non_null_values"] > 0 else "snapshot_empty"
            elif route == "get_history":
                with _vendor_request_timeout(request_timeout_seconds):
                    history = _get_history_frame(
                        ld,
                        universe=sample_rics,
                        fields=[field],
                        start=start_date,
                        end=end_date,
                        interval=interval,
                    )
                row.update(_history_stats(history, sample_rics, start_date=start_date, end_date=end_date))
                row["status"] = _historical_probe_status(row)
            else:
                row["status"] = "unsupported_route"
                row["error"] = f"unsupported route: {route}"
        except Exception as exc:
            row["status"] = f"{route}_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress_every > 0 and (completed % progress_every == 0 or completed == len(work)):
            print(f"Probed PIT route {completed}/{len(work)}: {route} {field}", file=sys.stderr, flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def write_refinitiv_pit_route_probe_artifacts(
    artifact_root: str | Path,
    run_id: str,
    results: pd.DataFrame,
    datasets: Sequence[str] | None,
    session_name: str | None,
) -> RefinitivPITRouteProbeArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / "route_probe_results.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    results.to_csv(results_path, index=False)
    summary_path.write_text(_route_probe_summary_md(run_id, results, datasets=datasets), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "datasets": list(datasets) if datasets else [],
        "probe_rows": int(len(results)),
        "status_counts": {str(key): int(value) for key, value in results["status"].value_counts(dropna=False).items()}
        if "status" in results
        else {},
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "route_probe_results": str(results_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivPITRouteProbeArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        results_path=results_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        results=results,
    )


def history_field_map_from_route_probe(
    route_probe_results: pd.DataFrame,
    datasets: Sequence[str] | None = None,
    min_inside_rows: int = 1,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Promote usable `get_history` route-probe rows into an accepted field map."""

    if route_probe_results.empty:
        return pd.DataFrame()
    data = route_probe_results.copy()
    if datasets is not None:
        wanted = {dataset.strip() for dataset in datasets if str(dataset).strip()}
        data = data[data["dataset_id"].astype(str).isin(wanted)].copy()
    data["inside_requested_rows"] = pd.to_numeric(data.get("inside_requested_rows"), errors="coerce").fillna(0)
    data["coverage"] = pd.to_numeric(data.get("coverage"), errors="coerce").fillna(0.0)
    data["non_null_values"] = pd.to_numeric(data.get("non_null_values"), errors="coerce").fillna(0)
    accepted = data[
        data["route"].astype(str).eq("get_history")
        & data["status"].astype(str).eq("historical_validated")
        & data["inside_requested_rows"].ge(min_inside_rows)
        & data["coverage"].ge(min_coverage)
    ].copy()
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "owner_agent",
                "source_family",
                "field_role",
                "candidate_field",
                "returned_data_columns",
                "interval",
                "sample_count",
                "non_null_values",
                "inside_requested_rows",
                "outside_requested_rows",
                "coverage",
                "history_status",
                "probed_at_utc",
            ]
        )

    if "required" in accepted.columns:
        accepted["_required_rank"] = accepted["required"].fillna(False).astype(bool).astype(int)
    else:
        accepted["_required_rank"] = 0
    accepted["_original_order"] = range(len(accepted))
    accepted = accepted.sort_values(
        ["dataset_id", "field_role", "_required_rank", "coverage", "non_null_values", "inside_requested_rows", "_original_order"],
        ascending=[True, True, False, False, False, False, True],
    )
    accepted = accepted.drop_duplicates(["dataset_id", "field_role"], keep="first").copy()
    accepted["history_status"] = "accepted_from_route_probe"
    if "returned_data_columns" not in accepted.columns:
        accepted["returned_data_columns"] = ""
    keep = [
        "dataset_id",
        "owner_agent",
        "source_family",
        "field_role",
        "candidate_field",
        "returned_data_columns",
        "interval",
        "sample_count",
        "non_null_values",
        "inside_requested_rows",
        "outside_requested_rows",
        "coverage",
        "history_status",
        "probed_at_utc",
    ]
    return accepted[[col for col in keep if col in accepted.columns]].reset_index(drop=True)


def write_refinitiv_pit_route_history_map_artifacts(
    artifact_root: str | Path,
    run_id: str,
    route_probe_results: pd.DataFrame,
    datasets: Sequence[str] | None,
    min_inside_rows: int = 1,
    min_coverage: float = 0.0,
) -> RefinitivPITRouteHistoryMapArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_field_map = history_field_map_from_route_probe(
        route_probe_results,
        datasets=datasets,
        min_inside_rows=min_inside_rows,
        min_coverage=min_coverage,
    )
    route_results_path = artifact_dir / "route_probe_results.csv"
    history_field_map_path = artifact_dir / "history_field_map.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    route_probe_results.to_csv(route_results_path, index=False)
    history_field_map.to_csv(history_field_map_path, index=False)
    summary_path.write_text(
        _route_history_map_summary_md(run_id, route_probe_results, history_field_map, datasets=datasets),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "datasets": list(datasets) if datasets else [],
        "route_rows": int(len(route_probe_results)),
        "history_fields": int(len(history_field_map)),
        "min_inside_rows": int(min_inside_rows),
        "min_coverage": float(min_coverage),
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "route_probe_results": str(route_results_path.resolve()),
            "history_field_map": str(history_field_map_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivPITRouteHistoryMapArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        route_results_path=route_results_path,
        history_field_map_path=history_field_map_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        route_results=route_probe_results,
        history_field_map=history_field_map,
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _default_owner_agent(dataset_id: str) -> str:
    return {
        "security_master_pit": "asean_security_master_agent",
        "corporate_actions": "corporate_actions_agent",
        "delistings_listing_history": "delisting_survivorship_agent",
        "fundamentals_pit": "fundamentals_pit_agent",
        "ibes_estimates_revisions": "analyst_revisions_agent",
        "earnings_events_actuals": "earnings_events_agent",
        "liquidity_market_cap_free_float": "liquidity_adv_agent",
        "fx_country_risk": "fx_benchmark_agent",
    }.get(dataset_id, "data_extraction_orchestrator_agent")


def _default_source_family(dataset_id: str) -> str:
    return {
        "security_master_pit": "refinitiv_reference_or_discovery",
        "corporate_actions": "refinitiv_corporate_actions",
        "delistings_listing_history": "refinitiv_reference_or_exchange_history",
        "fundamentals_pit": "worldscope_or_qa_company_fundamentals",
        "ibes_estimates_revisions": "ibes_history",
        "earnings_events_actuals": "ibes_actuals_company_events_or_exchange_calendar",
        "liquidity_market_cap_free_float": "refinitiv_reference_history_plus_ohlcv",
        "fx_country_risk": "lseg_fx_macro_exchange_calendars",
    }.get(dataset_id, "refinitiv_or_special_provider")


def _get_data_frame(ld: Any, universe: Sequence[str], fields: Sequence[str]) -> pd.DataFrame:
    raw = ld.get_data(universe=list(universe), fields=list(fields))
    if isinstance(raw, tuple):
        raw = raw[0]
    if raw is None:
        return pd.DataFrame()
    if not isinstance(raw, pd.DataFrame):
        return pd.DataFrame(raw)
    return raw.copy()


def _get_history_frame(
    ld: Any,
    universe: Sequence[str],
    fields: Sequence[str],
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    raw = ld.get_history(
        universe=list(universe),
        fields=list(fields),
        start=start,
        end=end,
        interval=interval,
    )
    if isinstance(raw, tuple):
        raw = raw[0]
    if raw is None:
        return pd.DataFrame()
    if not isinstance(raw, pd.DataFrame):
        return pd.DataFrame(raw)
    return raw.copy()


def _get_history_frame_with_single_ric_fallback(
    ld: Any,
    universe: Sequence[str],
    fields: Sequence[str],
    start: str,
    end: str,
    interval: str,
    request_timeout_seconds: float | None,
) -> tuple[pd.DataFrame, list[str]]:
    original_error: Exception | None = None
    try:
        with _vendor_request_timeout(request_timeout_seconds):
            return (
                _get_history_frame(
                    ld,
                    universe=universe,
                    fields=fields,
                    start=start,
                    end=end,
                    interval=interval,
                ),
                [],
            )
    except Exception as exc:
        if is_refinitiv_retryable_provider_error(exc):
            raise RefinitivRetryableProviderError(str(exc)) from exc
        if len(universe) <= 1:
            raise
        original_error = exc

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for ric in universe:
        try:
            with _vendor_request_timeout(request_timeout_seconds):
                frame = _get_history_frame(
                    ld,
                    universe=[ric],
                    fields=fields,
                    start=start,
                    end=end,
                    interval=interval,
                )
        except Exception as exc:
            if is_refinitiv_retryable_provider_error(exc):
                raise RefinitivRetryableProviderError(str(exc)) from exc
            errors.append(f"{ric}:{type(exc).__name__}:{exc}")
            continue
        if frame.empty:
            errors.append(f"{ric}:empty_history")
            continue
        tagged = frame.copy()
        if isinstance(tagged.columns, pd.MultiIndex):
            has_ric = any(ric in {str(part) for part in column if pd.notna(part)} for column in tagged.columns)
            if not has_ric:
                tagged.columns = pd.MultiIndex.from_tuples([(ric, *tuple(column)) for column in tagged.columns])
        else:
            tagged.columns = pd.MultiIndex.from_tuples([(ric, str(column)) for column in tagged.columns])
        frames.append(tagged)
    if not frames:
        if original_error is not None:
            raise original_error
        raise RuntimeError("history fallback produced no frames")
    return pd.concat(frames, axis=1).sort_index(), errors


def _data_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"instrument"}
    return [str(col) for col in frame.columns if str(col).strip().lower() not in excluded]


def _get_data_stats(frame: pd.DataFrame, sample_count: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "returned_rows": 0,
            "returned_columns": "",
            "returned_data_columns": "",
            "returned_data_column_count": 0,
            "non_null_values": 0,
            "coverage": 0.0,
        }
    data_cols = _data_columns(frame)
    non_null_values = int(frame[data_cols].notna().sum().sum()) if data_cols else 0
    denominator = max(1, sample_count * max(1, len(data_cols)))
    return {
        "returned_rows": int(len(frame)),
        "returned_columns": "|".join(str(col) for col in frame.columns),
        "returned_data_columns": "|".join(data_cols),
        "returned_data_column_count": int(len(data_cols)),
        "non_null_values": non_null_values,
        "coverage": float(non_null_values / denominator),
    }


def _history_stats(history: pd.DataFrame, rics: Sequence[str], start_date: str, end_date: str) -> dict[str, Any]:
    if history.empty:
        return {
            "returned_rows": 0,
            "returned_columns": "",
            "non_null_values": 0,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "inside_requested_rows": 0,
            "outside_requested_rows": 0,
            "coverage": 0.0,
        }

    frame = history.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    non_null = frame.notna()
    any_non_null = non_null.any(axis=1)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    valid_dates = pd.Series(frame.index, index=frame.index).notna()
    inside = (frame.index >= start) & (frame.index <= end) & valid_dates.to_numpy()
    inside_non_null = any_non_null & inside
    outside_non_null = any_non_null & ~inside
    inside_row_count = int(inside_non_null.sum())
    non_null_values = int(non_null.sum().sum())
    denominator = max(1, len(list(rics)) * max(1, inside_row_count))
    return {
        "returned_rows": int(frame.shape[0]),
        "returned_columns": _history_columns_text(frame),
        "non_null_values": non_null_values,
        "first_available": frame.index[any_non_null].min() if any_non_null.any() else pd.NaT,
        "last_available": frame.index[any_non_null].max() if any_non_null.any() else pd.NaT,
        "inside_requested_rows": inside_row_count,
        "outside_requested_rows": int(outside_non_null.sum()),
        "coverage": min(1.0, float(non_null_values / denominator)) if inside_row_count > 0 else 0.0,
    }


def _historical_probe_status(row: dict[str, Any]) -> str:
    non_null = int(row.get("non_null_values") or 0)
    inside = int(row.get("inside_requested_rows") or 0)
    outside = int(row.get("outside_requested_rows") or 0)
    if non_null <= 0:
        return "history_empty"
    if inside > 0:
        return "historical_validated"
    if outside > 0:
        return "current_or_static_only"
    return "history_empty"


def _field_probe_summary_md(
    run_id: str,
    results: pd.DataFrame,
    accepted: pd.DataFrame,
    datasets: Sequence[str] | None,
) -> str:
    lines = [
        f"# {run_id} Refinitiv PIT Field Probe",
        "",
        "## Scope",
        f"- Datasets: `{','.join(datasets) if datasets else 'all'}`",
        "- Route: local LSEG Workspace `lseg.data.get_data`.",
        "- Special provider rows are deliberately not forced through `get_data`.",
        "",
        "## Counts",
        f"- Probe rows: `{len(results)}`",
        f"- Accepted live fields: `{len(accepted)}`",
    ]
    if not results.empty and "status" in results.columns:
        for status, count in results["status"].value_counts(dropna=False).sort_index().items():
            lines.append(f"- `{status}`: `{int(count)}`")
    lines.extend(
        [
            "",
            "## Readiness Note",
            "This field map validates current/snapshot calls only. Historical PIT extraction still needs batch manifests, timestamp policy, and leakage QA before entering alpha research.",
        ]
    )
    return "\n".join(lines) + "\n"


def _historical_probe_summary_md(
    run_id: str,
    results: pd.DataFrame,
    history_field_map: pd.DataFrame,
    datasets: Sequence[str] | None,
) -> str:
    lines = [
        f"# {run_id} Refinitiv PIT Historical Probe",
        "",
        "## Scope",
        f"- Datasets: `{','.join(datasets) if datasets else 'all'}`",
        "- Route: local LSEG Workspace `lseg.data.get_history`.",
        "- Only fields with rows inside the requested window become eligible for batch history extraction.",
        "",
        "## Counts",
        f"- Probe rows: `{len(results)}`",
        f"- Historical fields accepted: `{len(history_field_map)}`",
    ]
    if not results.empty and "status" in results.columns:
        for status, count in results["status"].value_counts(dropna=False).sort_index().items():
            lines.append(f"- `{status}`: `{int(count)}`")
    if not history_field_map.empty:
        lines.extend(["", "## Accepted Historical Fields"])
        for item in history_field_map.itertuples(index=False):
            row = item._asdict()
            lines.append(f"- `{row.get('dataset_id')}` / `{row.get('field_role')}`: `{row.get('candidate_field')}`")
    lines.extend(
        [
            "",
            "## Readiness Note",
            "This probe validates the vendor history route only. Batch outputs still need per-window manifests and leakage gating before model use.",
        ]
    )
    return "\n".join(lines) + "\n"


def _route_probe_summary_md(run_id: str, results: pd.DataFrame, datasets: Sequence[str] | None) -> str:
    lines = [
        f"# {run_id} Refinitiv PIT Route Probe",
        "",
        "## Scope",
        f"- Datasets: `{','.join(datasets) if datasets else 'all'}`",
        "- Routes tested: snapshot `get_data` and historical `get_history` where applicable.",
        "- `IBES:` and `QA:` aliases remain special-provider/export routes until mapped to callable LSEG fields.",
        "",
        "## Counts",
        f"- Probe rows: `{len(results)}`",
    ]
    if not results.empty and "status" in results.columns:
        for status, count in results["status"].value_counts(dropna=False).sort_index().items():
            lines.append(f"- `{status}`: `{int(count)}`")
    if not results.empty:
        usable = results[results["status"].isin(["snapshot_validated", "historical_validated"])].copy()
        if not usable.empty:
            lines.extend(["", "## Usable Routes Found"])
            for item in usable.itertuples(index=False):
                row = item._asdict()
                lines.append(
                    f"- `{row.get('dataset_id')}` / `{row.get('field_role')}` / `{row.get('route')}`: `{row.get('candidate_field')}`"
                )
    return "\n".join(lines) + "\n"


def _route_history_map_summary_md(
    run_id: str,
    route_probe_results: pd.DataFrame,
    history_field_map: pd.DataFrame,
    datasets: Sequence[str] | None,
) -> str:
    lines = [
        f"# {run_id} Refinitiv PIT Promoted History Field Map",
        "",
        "## Scope",
        f"- Datasets: `{','.join(datasets) if datasets else 'all'}`",
        "- Source: route-probe rows with `route=get_history` and `status=historical_validated`.",
        "- Duplicate dataset/field roles are ranked by required flag, coverage, non-null count, and original probe order.",
        "",
        "## Counts",
        f"- Route rows reviewed: `{len(route_probe_results)}`",
        f"- Promoted historical fields: `{len(history_field_map)}`",
    ]
    if not history_field_map.empty:
        lines.extend(["", "## Promoted Fields"])
        for item in history_field_map.itertuples(index=False):
            row = item._asdict()
            lines.append(
                f"- `{row.get('dataset_id')}` / `{row.get('field_role')}`: `{row.get('candidate_field')}` "
                f"(coverage `{float(row.get('coverage', 0.0)):.3f}`)"
            )
    return "\n".join(lines) + "\n"


def _filter_selected_universe(selected_universe: pd.DataFrame, tier: str) -> pd.DataFrame:
    if selected_universe.empty:
        return selected_universe.copy()
    tier_col = f"selected_for_{tier.lower()}"
    data = selected_universe.copy()
    if tier_col in data.columns:
        data = data[data[tier_col].fillna(False).astype(bool)].copy()
    if "refinitiv_ric" not in data.columns:
        raise ValueError("selected universe missing `refinitiv_ric`")
    return data[data["refinitiv_ric"].notna()].drop_duplicates("refinitiv_ric").reset_index(drop=True)


def _filter_accepted_field_map(
    accepted_field_map: pd.DataFrame,
    datasets: Sequence[str] | None,
) -> pd.DataFrame:
    if accepted_field_map.empty:
        return accepted_field_map.copy()
    fmap = accepted_field_map.copy()
    if datasets is not None:
        wanted = {dataset.strip() for dataset in datasets if str(dataset).strip()}
        fmap = fmap[fmap["dataset_id"].astype(str).isin(wanted)].copy()
    if "candidate_field" not in fmap.columns:
        raise ValueError("accepted field map missing `candidate_field`")
    return fmap.dropna(subset=["candidate_field"]).reset_index(drop=True)


def _filter_history_field_map(history_field_map: pd.DataFrame, datasets: Sequence[str] | None) -> pd.DataFrame:
    if history_field_map.empty:
        return history_field_map.copy()
    fmap = history_field_map.copy()
    if datasets is not None:
        wanted = {dataset.strip() for dataset in datasets if str(dataset).strip()}
        fmap = fmap[fmap["dataset_id"].astype(str).isin(wanted)].copy()
    if "candidate_field" not in fmap.columns:
        raise ValueError("history field map missing `candidate_field`")
    return fmap.dropna(subset=["candidate_field"]).reset_index(drop=True)


def _get_data_with_field_fallback(
    ld: Any,
    universe: Sequence[str],
    fields: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        return _get_data_frame(ld, universe=universe, fields=fields), []
    except Exception as exc:
        errors = [{"field": ",".join(fields), "error": f"bulk call failed: {type(exc).__name__}: {exc}"}]

    frames: list[pd.DataFrame] = []
    for field in fields:
        try:
            frame = _get_data_frame(ld, universe=universe, fields=[field])
            frames.append(frame)
        except Exception as exc:
            errors.append({"field": str(field), "error": f"{type(exc).__name__}: {exc}"})
    if not frames:
        return pd.DataFrame(), errors

    merged = frames[0]
    for frame in frames[1:]:
        if merged.empty:
            merged = frame
            continue
        if frame.empty:
            continue
        if "Instrument" in merged.columns and "Instrument" in frame.columns:
            overlap = [col for col in frame.columns if col in merged.columns and col != "Instrument"]
            merged = merged.merge(frame.drop(columns=overlap, errors="ignore"), on="Instrument", how="outer")
        else:
            merged = pd.concat([merged, frame], axis=1)
    return merged, errors


def _history_frame_to_long(
    history: pd.DataFrame,
    rics: Sequence[str],
    plan_row: dict[str, Any],
    universe: pd.DataFrame,
    source_loaded_at_utc: str,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    ric_set = {str(ric) for ric in rics}
    returned_column_hint = str(plan_row.get("returned_data_columns", "") or plan_row.get("candidate_field", ""))
    start = pd.Timestamp(str(plan_row["start_date"]))
    end = pd.Timestamp(str(plan_row["end_date"]))
    parts: list[pd.DataFrame] = []

    if isinstance(frame.columns, pd.MultiIndex):
        for column in frame.columns:
            column_parts = [str(part) for part in column if pd.notna(part)]
            ric = next((part for part in column_parts if part in ric_set), None)
            if ric is None and len(ric_set) == 1:
                ric = next(iter(ric_set))
            returned_column = next((part for part in column_parts if part != ric), returned_column_hint)
            if ric is None:
                continue
            series = frame[column].dropna()
            if series.empty:
                continue
            parts.append(_series_to_history_long(series, ric, returned_column, plan_row, start, end, source_loaded_at_utc))
    else:
        frame_column_name = str(frame.columns.name) if frame.columns.name is not None else returned_column_hint
        for column in frame.columns:
            column_text = str(column)
            ric = column_text if column_text in ric_set else None
            returned_column = frame_column_name
            if ric is None and len(ric_set) == 1:
                ric = next(iter(ric_set))
                returned_column = column_text if column_text not in ric_set else frame_column_name
            if ric is None:
                continue
            series = frame[column].dropna()
            if series.empty:
                continue
            parts.append(_series_to_history_long(series, ric, returned_column, plan_row, start, end, source_loaded_at_utc))

    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True, sort=False)
    return _merge_history_universe_metadata(long, universe)


def history_frame_to_long(
    history: pd.DataFrame,
    rics: Sequence[str],
    plan_row: dict[str, Any],
    universe: pd.DataFrame,
    source_loaded_at_utc: str,
) -> pd.DataFrame:
    """Public normalized adapter for Refinitiv wide or multi-index history frames."""

    return _history_frame_to_long(
        history,
        rics=rics,
        plan_row=plan_row,
        universe=universe,
        source_loaded_at_utc=source_loaded_at_utc,
    )


def _series_to_history_long(
    series: pd.Series,
    ric: str,
    returned_column: str,
    plan_row: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_loaded_at_utc: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(series.index, errors="coerce").date,
            "requested_ric": ric,
            "value": series.to_numpy(),
        }
    )
    dates = pd.to_datetime(out["asof_date"], errors="coerce")
    out["inside_requested_window"] = (dates >= start) & (dates <= end)
    out["dataset_id"] = plan_row.get("dataset_id")
    out["field_role"] = plan_row.get("field_role")
    out["candidate_field"] = plan_row.get("candidate_field")
    out["returned_column"] = returned_column
    out["country"] = plan_row.get("country")
    out["tier"] = plan_row.get("tier")
    out["interval"] = plan_row.get("interval")
    out["window_start"] = plan_row.get("start_date")
    out["window_end"] = plan_row.get("end_date")
    out["batch_number"] = plan_row.get("batch_number")
    out["window_number"] = plan_row.get("window_number")
    out["plan_row_id"] = plan_row.get("plan_row_id")
    out["source_loaded_at_utc"] = source_loaded_at_utc
    out["route"] = "get_history"
    return out


def _attach_requested_ric(frame: pd.DataFrame, chunk: Sequence[str]) -> pd.DataFrame:
    data = frame.copy()
    if "Instrument" in data.columns:
        data["requested_ric"] = data["Instrument"].astype(str)
    elif len(data) == len(chunk):
        data["requested_ric"] = list(chunk)
    else:
        data["requested_ric"] = pd.NA
    return data


def _merge_universe_metadata(snapshot: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty or universe.empty or "requested_ric" not in snapshot.columns:
        return snapshot
    keep = [
        "refinitiv_ric",
        "sid",
        "ticker",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
        "selected_for_u2",
        "selected_for_u3",
    ]
    meta = universe[[col for col in keep if col in universe.columns]].drop_duplicates("refinitiv_ric")
    merged = snapshot.merge(meta, left_on="requested_ric", right_on="refinitiv_ric", how="left")
    front = [
        col
        for col in (
            "requested_ric",
            "refinitiv_ric",
            "sid",
            "ticker",
            "local_symbol",
            "name",
            "country",
            "exchange",
            "currency",
            "asof_date",
            "source_loaded_at_utc",
            "snapshot_scope",
        )
        if col in merged.columns
    ]
    remaining = [col for col in merged.columns if col not in front]
    return merged[front + remaining]


def _merge_history_universe_metadata(long: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if long.empty or universe.empty or "requested_ric" not in long.columns:
        return long
    keep = [
        "refinitiv_ric",
        "sid",
        "ticker",
        "local_symbol",
        "name",
        "exchange",
        "currency",
        "selected_for_u2",
        "selected_for_u3",
    ]
    available = [col for col in keep if col in universe.columns]
    if "refinitiv_ric" not in available:
        return long
    meta = universe[available].drop_duplicates("refinitiv_ric")
    merged = long.merge(meta, left_on="requested_ric", right_on="refinitiv_ric", how="left")
    front = [
        col
        for col in (
            "asof_date",
            "requested_ric",
            "refinitiv_ric",
            "sid",
            "ticker",
            "local_symbol",
            "name",
            "country",
            "exchange",
            "currency",
            "dataset_id",
            "field_role",
            "candidate_field",
            "returned_column",
            "value",
            "inside_requested_window",
            "source_loaded_at_utc",
        )
        if col in merged.columns
    ]
    remaining = [col for col in merged.columns if col not in front]
    return merged[front + remaining]


def _snapshot_quality(snapshot: pd.DataFrame, field_map: pd.DataFrame, requested_rics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if field_map.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "field_role",
                "field",
                "returned_column",
                "requested_rics",
                "snapshot_rows",
                "non_null_rows",
                "coverage",
            ]
        )
    field_rows = field_map.to_dict("records")
    if snapshot.empty:
        for item in field_rows:
            rows.append(
                {
                    "dataset_id": item.get("dataset_id", ""),
                    "field_role": item.get("field_role", ""),
                    "field": item.get("candidate_field", ""),
                    "returned_column": "",
                    "requested_rics": len(requested_rics),
                    "snapshot_rows": 0,
                    "non_null_rows": 0,
                    "coverage": 0.0,
                }
            )
        return pd.DataFrame(rows)

    lookup_cols = set(_data_columns(snapshot))
    for item in field_rows:
        field = str(item.get("candidate_field", ""))
        returned = _best_returned_column_for_field(field, lookup_cols, hinted_columns=str(item.get("returned_data_columns", "")))
        non_null = int(snapshot[returned].notna().sum()) if returned in snapshot.columns else 0
        rows.append(
            {
                "dataset_id": item.get("dataset_id", ""),
                "field_role": item.get("field_role", ""),
                "field": field,
                "returned_column": returned,
                "requested_rics": len(requested_rics),
                "snapshot_rows": len(snapshot),
                "non_null_rows": non_null,
                "coverage": float(non_null / max(1, len(requested_rics))),
            }
        )
    return pd.DataFrame(rows)


def _historical_quality(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "field_role",
        "candidate_field",
        "country",
        "plan_rows",
        "completed_rows",
        "failed_rows",
        "empty_rows",
        "history_rows",
        "non_null_values",
        "first_available",
        "last_available",
        "avg_coverage",
        "min_coverage",
    ]
    if manifest.empty:
        return pd.DataFrame(columns=columns)
    data = manifest.copy()
    for col in ("history_rows", "non_null_values", "coverage"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    for col in ("first_available", "last_available"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    quality = (
        data.groupby(["dataset_id", "field_role", "candidate_field", "country"], dropna=False)
        .agg(
            plan_rows=("plan_row_id", "nunique"),
            completed_rows=("status", lambda values: int(pd.Series(values).isin(["completed", "completed_partial"]).sum())),
            failed_rows=("status", lambda values: int(pd.Series(values).astype(str).str.startswith("failed").sum())),
            empty_rows=("status", lambda values: int(pd.Series(values).eq("completed_empty").sum())),
            history_rows=("history_rows", "sum"),
            non_null_values=("non_null_values", "sum"),
            first_available=("first_available", "min"),
            last_available=("last_available", "max"),
            avg_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
        )
        .reset_index()
    )
    return quality[columns].sort_values(["dataset_id", "field_role", "country"]).reset_index(drop=True)


def _historical_errors(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = ["plan_row_id", "dataset_id", "field_role", "candidate_field", "country", "error_type", "error"]
    if manifest.empty or "status" not in manifest.columns:
        return pd.DataFrame(columns=columns)
    failed = manifest[manifest["status"].astype(str).str.startswith("failed")].copy()
    if failed.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in failed.columns:
            failed[column] = ""
    return failed[columns].reset_index(drop=True)


def _quality_row_from_manifest(manifest_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": manifest_row.get("dataset_id"),
        "field_role": manifest_row.get("field_role"),
        "candidate_field": manifest_row.get("candidate_field"),
        "country": manifest_row.get("country"),
        "status": manifest_row.get("status"),
        "history_rows": manifest_row.get("history_rows", 0),
        "coverage": manifest_row.get("coverage", 0.0),
    }


def _best_returned_column_for_field(field: str, columns: set[str], hinted_columns: str = "") -> str:
    if field in columns:
        return field
    for hinted in [part.strip() for part in hinted_columns.split("|") if part.strip()]:
        if hinted in columns:
            return hinted
    normalized_field = _normalize_text(field.split(".")[-1])
    normalized_aliases = {normalized_field}
    if normalized_field.endswith("PCT"):
        normalized_aliases.add(f"{normalized_field[:-3]}PERCENT")
    if "PCT" in normalized_field:
        normalized_aliases.add(normalized_field.replace("PCT", "PERCENT"))
    for column in columns:
        normalized_column = _normalize_text(column)
        if any(alias and alias in normalized_column for alias in normalized_aliases):
            return column
    return ""


def _normalize_text(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _history_columns_text(frame: pd.DataFrame) -> str:
    if isinstance(frame.columns, pd.MultiIndex):
        return "|".join("::".join(str(part) for part in col if pd.notna(part)) for col in frame.columns)
    return "|".join(str(col) for col in frame.columns)


def _date_windows(start_date: str, end_date: str, window_years: int) -> list[tuple[str, str]]:
    if window_years <= 0:
        raise ValueError("window_years must be positive")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.DateOffset(years=window_years) - pd.Timedelta(days=1), end)
        windows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def _slugish(value: object) -> str:
    text = str(value).strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"
