from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class JapanG3ExecutionPlanArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    fundamentals_plan: pd.DataFrame
    consensus_plan: pd.DataFrame
    earnings_plan: pd.DataFrame
    analyst_detail_plan: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _g1_admitted(manifest: dict[str, Any]) -> bool:
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
    )


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _windows(start_date: str, end_date: str, years: int) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), end)
        rows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end + pd.Timedelta(days=1)
    return rows


def _status(g1_ok: bool, lane: str) -> str:
    if not g1_ok:
        return "blocked_upstream_g1"
    if lane == "analyst_detail":
        return "blocked_requires_bulk_or_explicit_long_run"
    return "ready_after_vendor_health"


def build_japan_g3_execution_plans(
    selected_universe: pd.DataFrame,
    sector_profile: pd.DataFrame,
    start_date: str,
    end_date: str,
    batch_size: int,
    window_years: int,
    g1_ok: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if batch_size <= 0 or window_years <= 0:
        raise ValueError("batch_size and window_years must be positive")
    selected = selected_universe.loc[
        selected_universe["selected_for_u2"].fillna(False).astype(bool), ["refinitiv_ric"]
    ].dropna().drop_duplicates()
    if selected.empty:
        raise ValueError("Japan G3 selected universe is empty")
    routed = selected.merge(
        sector_profile[["ric", "fundamental_route_class"]].drop_duplicates("ric"),
        left_on="refinitiv_ric",
        right_on="ric",
        how="left",
    )
    routed["fundamental_route_class"] = routed["fundamental_route_class"].fillna("unresolved")
    if routed["fundamental_route_class"].eq("unresolved").any():
        raise ValueError("sector routing does not cover the full selected universe")

    windows = _windows(start_date, end_date, window_years)
    fundamental_fields = {
        "general": "TR.Revenue,TR.GrossProfit,TR.EPSActValue,TR.BookValuePerShare",
        "bank": "TR.BankTotalRevenue,TR.NetInterestIncome,TR.InterestIncomeBank,TR.EPSActValue,TR.BookValuePerShare",
        "insurance": "TR.Revenue,TR.OperatingIncome,TR.EPSActValue,TR.BookValuePerShare",
    }
    fundamental_rows: list[dict[str, Any]] = []
    for route_class, group in routed.groupby("fundamental_route_class", sort=True):
        rics = sorted(group["refinitiv_ric"].astype(str))
        for batch_number, chunk in enumerate(_chunks(rics, batch_size), start=1):
            for window_number, (window_start, window_end) in enumerate(windows, start=1):
                for frequency in ("FY", "FQ"):
                    fundamental_rows.append(
                        {
                            "plan_row_id": f"fund-{route_class}-{batch_number:04d}-{window_number:02d}-{frequency.lower()}",
                            "dataset_id": "fundamentals_pit",
                            "route_class": route_class,
                            "route": "get_data_parameterized",
                            "frequency": frequency,
                            "start_date": window_start,
                            "end_date": window_end,
                            "batch_number": batch_number,
                            "window_number": window_number,
                            "ric_count": len(chunk),
                            "ric_list": ",".join(chunk),
                            "source_fields": fundamental_fields[route_class],
                            "status": _status(g1_ok, "fundamentals"),
                        }
                    )

    all_rics = sorted(routed["refinitiv_ric"].astype(str))
    consensus_rows: list[dict[str, Any]] = []
    earnings_rows: list[dict[str, Any]] = []
    for batch_number, chunk in enumerate(_chunks(all_rics, batch_size), start=1):
        for window_number, (window_start, window_end) in enumerate(windows, start=1):
            consensus_rows.append(
                {
                    "plan_row_id": f"consensus-{batch_number:04d}-{window_number:02d}",
                    "dataset_id": "ibes_estimates_revisions",
                    "route": "get_history",
                    "interval": "1M",
                    "start_date": window_start,
                    "end_date": window_end,
                    "batch_number": batch_number,
                    "window_number": window_number,
                    "ric_count": len(chunk),
                    "ric_list": ",".join(chunk),
                    "source_fields": "TR.EPSMean,TR.RevenueMean,TR.RecMean,TR.PriceTargetMean",
                    "status": _status(g1_ok, "consensus"),
                }
            )
            for frequency in ("FY", "FQ"):
                earnings_rows.append(
                    {
                        "plan_row_id": f"earnings-{batch_number:04d}-{window_number:02d}-{frequency.lower()}",
                        "dataset_id": "earnings_events_actuals",
                        "route": "get_data_parameterized",
                        "frequency": frequency,
                        "start_date": window_start,
                        "end_date": window_end,
                        "batch_number": batch_number,
                        "window_number": window_number,
                        "ric_count": len(chunk),
                        "ric_list": ",".join(chunk),
                        "source_fields": "TR.EPSActValue,TR.EPSActSurprise",
                        "status": _status(g1_ok, "earnings"),
                    }
                )

    analyst_rows: list[dict[str, Any]] = []
    annual_windows = _windows(start_date, end_date, 1)
    for ric_number, ric in enumerate(all_rics, start=1):
        for window_number, (window_start, window_end) in enumerate(annual_windows, start=1):
            for estimate_type, fields in (
                ("recommendation", "TR.RecEstValue.*"),
                ("target_price", "TR.TPEstValue.*"),
            ):
                analyst_rows.append(
                    {
                        "plan_row_id": f"analyst-{ric_number:04d}-{window_number:02d}-{estimate_type}",
                        "dataset_id": "ibes_estimates_revisions",
                        "route": "get_data_parameterized_self_keyed",
                        "estimate_type": estimate_type,
                        "start_date": window_start,
                        "end_date": window_end,
                        "ric_count": 1,
                        "ric_list": ric,
                        "source_fields": fields,
                        "status": _status(g1_ok, "analyst_detail"),
                    }
                )
    return (
        pd.DataFrame(fundamental_rows),
        pd.DataFrame(consensus_rows),
        pd.DataFrame(earnings_rows),
        pd.DataFrame(analyst_rows),
    )


def write_japan_g3_execution_plan_artifacts(
    plan_dir: str | Path,
    sector_snapshot_dir: str | Path,
    route_registry_dir: str | Path,
    g1_validation_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    start_date: str = "2006-01-01",
    end_date: str = "2026-07-06",
    batch_size: int = 25,
    window_years: int = 5,
) -> JapanG3ExecutionPlanArtifacts:
    plan_manifest_path = Path(plan_dir) / "manifest.json"
    universe_path = Path(plan_dir) / "selected_universe.csv"
    sector_manifest_path = Path(sector_snapshot_dir) / "manifest.json"
    sector_path = Path(sector_snapshot_dir) / "sector_routing_profile.csv"
    route_manifest_path = Path(route_registry_dir) / "manifest.json"
    route_path = Path(route_registry_dir) / "accepted_route_registry.csv"
    g1_manifest_path = Path(g1_validation_dir) / "manifest.json"
    required_paths = [
        plan_manifest_path,
        universe_path,
        sector_manifest_path,
        sector_path,
        route_manifest_path,
        route_path,
        g1_manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Japan G3 plan inputs: {missing}")
    plan_manifest = _read_json(plan_manifest_path)
    sector_manifest = _read_json(sector_manifest_path)
    route_manifest = _read_json(route_manifest_path)
    g1_manifest = _read_json(g1_manifest_path)
    if int(plan_manifest.get("selected_names", 0)) <= 0:
        raise ValueError("Japan PIT plan has no selected names")
    if float(sector_manifest.get("coverage", 0.0)) != 1.0:
        raise ValueError("Japan sector routing is incomplete")
    if int(route_manifest.get("unresolved_roles", -1)) != 0:
        raise ValueError("Japan G3 route registry has unresolved roles")

    g1_ok = _g1_admitted(g1_manifest)
    outputs = build_japan_g3_execution_plans(
        pd.read_csv(universe_path, low_memory=False),
        pd.read_csv(sector_path, low_memory=False),
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        window_years=window_years,
        g1_ok=g1_ok,
    )
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    names = ("fundamentals_plan", "consensus_plan", "earnings_plan", "analyst_detail_plan")
    paths: dict[str, Path] = {}
    for name, frame in zip(names, outputs):
        path = artifact_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    total_plan_rows = sum(len(frame) for frame in outputs)
    analyst_plan_rows = len(outputs[3])
    ordinary_plan_rows = total_plan_rows - analyst_plan_rows
    ordinary_estimated_calls = int(
        sum(
            frame["source_fields"].astype(str).str.split(",").map(len).sum()
            for frame in outputs[:3]
        )
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "ready" if g1_ok else "blocked_upstream",
        "evidence_scope": "full_execution_plan",
        "scorecard_eligible": g1_ok,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "g1_independent_validation_admitted": g1_ok,
        "selected_names": int(plan_manifest["selected_names"]),
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": batch_size,
        "window_years": window_years,
        "fundamentals_plan_rows": len(outputs[0]),
        "consensus_plan_rows": len(outputs[1]),
        "earnings_plan_rows": len(outputs[2]),
        "analyst_detail_plan_rows": analyst_plan_rows,
        "total_plan_rows": total_plan_rows,
        "ordinary_workspace_plan_rows": ordinary_plan_rows,
        "ordinary_workspace_estimated_field_calls": ordinary_estimated_calls,
        "analyst_detail_estimated_calls": analyst_plan_rows,
        "analyst_detail_execution_policy": "bulk_feed_or_explicit_long_run_only",
        "full_workspace_analyst_detail_auto_start": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {str(path.resolve()): _sha256(path) for path in required_paths},
        "artifact_sha256": {name: _sha256(path) for name, path in paths.items()},
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3ExecutionPlanArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        fundamentals_plan=outputs[0],
        consensus_plan=outputs[1],
        earnings_plan=outputs[2],
        analyst_detail_plan=outputs[3],
    )
