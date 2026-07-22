from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


UTC = timezone.utc


@dataclass(frozen=True)
class JapanCombinedMonthlyMembershipQAArtifacts:
    run_id: str
    artifact_dir: Path
    membership_path: Path
    monthly_coverage_path: Path
    monthly_churn_path: Path
    anomalies_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def run_japan_combined_monthly_membership_qa(
    *,
    source_dirs: Iterable[str | Path],
    run_id: str,
    artifact_root: str | Path = "artifacts/japan_combined_monthly_membership_qa",
    start_month: str = "2006-07",
    end_month: str = "2026-06",
    ose_end_month: str = "2013-06",
    max_monthly_churn_rate: float = 0.15,
    min_monthly_codes: int = 1_000,
    required_months: int = 240,
) -> JapanCombinedMonthlyMembershipQAArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    sources = [Path(path) for path in source_dirs]
    if not sources:
        raise ValueError("at least one source directory is required")
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    ose_end = pd.Period(ose_end_month, freq="M")
    expected_months = list(pd.period_range(start, end, freq="M"))

    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, dict[str, str]] = {}
    input_blockers: list[str] = []
    observed_exchange_months: set[tuple[str, str]] = set()
    completed_files = 0
    planned_files = 0
    for source in sources:
        manifest_path = source / "manifest.json"
        plan_path = source / "request_plan.csv"
        status_path = source / "file_status.csv"
        quality_path = source / "file_quality.csv"
        manifest = _read_json(manifest_path)
        source_kind = _source_kind(manifest)
        if manifest.get("status") != "complete_candidate_source":
            input_blockers.append(f"{source.name}:source_not_complete_candidate")
        if source_kind not in {"TSE", "JASDAQ_OSE_COMBINED", "OSE_MAIN"}:
            input_blockers.append(f"{source.name}:unknown_source_family")
            continue
        if not all(path.is_file() for path in (manifest_path, plan_path, status_path, quality_path)):
            input_blockers.append(f"{source.name}:required_ledger_missing")
            continue
        plan = pd.read_csv(plan_path, dtype=str).fillna("")
        status = pd.read_csv(status_path, dtype=str).fillna("")
        complete = status.loc[status["status"].eq("completed")].copy()
        if complete["request_id"].duplicated().any():
            input_blockers.append(f"{source.name}:duplicate_completed_request_ids")
            continue
        joined = plan.merge(complete, on="request_id", how="inner", validate="one_to_one")
        planned_files += len(plan)
        completed_files += len(joined)
        if len(joined) != len(plan):
            input_blockers.append(f"{source.name}:incomplete_file_ledger")
        source_hashes[source.name] = {
            "manifest.json": _sha256(manifest_path),
            "request_plan.csv": _sha256(plan_path),
            "file_status.csv": _sha256(status_path),
            "file_quality.csv": _sha256(quality_path),
        }
        for row in joined.to_dict("records"):
            normalized_path = Path(str(row["normalized_path"]))
            expected_hash = str(row.get("normalized_sha256", ""))
            if not normalized_path.is_file() or len(expected_hash) != 64 or _sha256(normalized_path) != expected_hash:
                input_blockers.append(f"{source.name}:{row['request_id']}:normalized_hash_failure")
                continue
            frame = pd.read_parquet(normalized_path)
            required = {"observation_month", "code", "known_at_utc", "currency", "pit_usable", "source_sha256"}
            if not required.issubset(frame.columns):
                input_blockers.append(f"{source.name}:{row['request_id']}:normalized_schema_failure")
                continue
            frame = frame.copy()
            observation_month = str(row["observation_month"])
            family = _source_family_for_month(source_kind, observation_month)
            frame["source_family"] = family
            frame["source_artifact"] = source.name
            frames.append(frame)
            observed_exchange_months.add((family, str(row["observation_month"])))

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    membership_path = root / "monthly_membership.parquet"
    coverage_path = root / "monthly_coverage.csv"
    churn_path = root / "monthly_churn.csv"
    anomalies_path = root / "anomalies.csv"
    manifest_path = root / "manifest.json"
    anomaly_rows: list[dict[str, Any]] = [
        {"observation_month": "", "type": "input_failure", "value": blocker}
        for blocker in input_blockers
    ]
    membership_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    month_codes: dict[str, set[str]] = {}
    month_passes: dict[str, bool] = {}

    for period in expected_months:
        month = str(period)
        group = data.loc[data.get("observation_month", pd.Series(dtype=str)).astype(str).eq(month)].copy()
        if period <= pd.Period("2010-09", freq="M"):
            expected_families = {"TSE", "JASDAQ", "OSE_MAIN"}
        elif period <= ose_end:
            expected_families = {"TSE", "OSE_COMBINED"}
        else:
            expected_families = {"TSE"}
        present_families = set(group.get("source_family", pd.Series(dtype=str)).astype(str))
        missing_families = sorted(expected_families - present_families)
        group["code"] = group.get("code", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
        within_source_duplicates = int(group.duplicated(["source_family", "code"]).sum()) if not group.empty else 0
        cross_source_duplicates = int(group.duplicated("code").sum()) if not group.empty else 0
        timestamps = pd.to_datetime(group.get("known_at_utc"), errors="coerce", utc=True) if not group.empty else pd.Series(dtype="datetime64[ns, UTC]")
        timestamp_coverage = float(timestamps.notna().mean()) if not group.empty else 0.0
        jpy_coverage = float(group.get("currency", pd.Series(dtype=str)).eq("JPY").mean()) if not group.empty else 0.0
        pit_coverage = float(group.get("pit_usable", pd.Series(dtype=bool)).astype(bool).mean()) if not group.empty else 0.0
        unique_codes = int(group["code"].nunique()) if not group.empty else 0
        passed = bool(
            not missing_families
            and within_source_duplicates == 0
            and unique_codes >= min_monthly_codes
            and timestamp_coverage == 1.0
            and jpy_coverage == 1.0
            and pit_coverage == 1.0
        )
        coverage_rows.append(
            {
                "observation_month": month,
                "expected_source_families": "|".join(sorted(expected_families)),
                "present_source_families": "|".join(sorted(present_families)),
                "missing_source_families": "|".join(missing_families),
                "source_rows": len(group),
                "unique_codes": unique_codes,
                "within_source_duplicate_codes": within_source_duplicates,
                "cross_source_duplicate_codes": cross_source_duplicates,
                "known_at_coverage": timestamp_coverage,
                "jpy_coverage": jpy_coverage,
                "pit_usable_coverage": pit_coverage,
                "month_pass": passed,
            }
        )
        if not passed:
            anomaly_rows.append({"observation_month": month, "type": "monthly_quality_failure", "value": "|".join(missing_families)})
        month_codes[month] = set(group["code"]) if not group.empty else set()
        month_passes[month] = passed

        if not group.empty:
            group["source_rank"] = group["source_family"].map(
                {"TSE": 0, "JASDAQ": 1, "OSE_MAIN": 2, "OSE_COMBINED": 1}
            ).fillna(9)
            for code, code_group in group.sort_values(["source_rank", "source_artifact"]).groupby("code", sort=True):
                first = code_group.iloc[0]
                membership_rows.append(
                    {
                        "observation_month": month,
                        "membership_effective_month_end": period.end_time.date().isoformat(),
                        "known_at_utc": timestamps.loc[code_group.index].max().isoformat(),
                        "known_at_policy": "official_monthly_publication_lagged_to_21st_calendar_day_00_jst",
                        "code": code,
                        "issue_name_japanese": str(first.get("issue_name_japanese", "")),
                        "source_families": "|".join(sorted(set(code_group["source_family"].astype(str)))),
                        "source_artifacts": "|".join(sorted(set(code_group["source_artifact"].astype(str)))),
                        "source_sha256": "|".join(sorted(set(code_group["source_sha256"].astype(str)))),
                        "evidence_rows": len(code_group),
                        "currency": "JPY",
                        "membership_observed": True,
                        "pit_usable": bool(code_group["pit_usable"].astype(bool).all()),
                    }
                )

    churn_rows = []
    ordered = sorted(month_codes)
    for previous, current in zip(ordered, ordered[1:]):
        if not month_passes[previous] or not month_passes[current]:
            continue
        before = month_codes[previous]
        after = month_codes[current]
        entrants = after - before
        exits = before - after
        churn_rate = (len(entrants) + len(exits)) / max(len(before), 1)
        gap = pd.Period(current, freq="M") - pd.Period(previous, freq="M")
        consecutive = getattr(gap, "n", gap) == 1
        passed = bool(consecutive and churn_rate <= max_monthly_churn_rate)
        churn_rows.append(
            {
                "previous_month": previous,
                "observation_month": current,
                "entrants": len(entrants),
                "exits": len(exits),
                "churn_rate": churn_rate,
                "consecutive_months": consecutive,
                "churn_pass": passed,
            }
        )
        if not passed:
            anomaly_rows.append({"observation_month": current, "type": "continuity_or_churn_failure", "value": churn_rate})

    membership = pd.DataFrame(membership_rows)
    coverage = pd.DataFrame(coverage_rows)
    churn = pd.DataFrame(churn_rows)
    anomalies = pd.DataFrame(anomaly_rows, columns=["observation_month", "type", "value"])
    _atomic_write_parquet(membership_path, membership)
    _atomic_write_csv(coverage_path, coverage)
    _atomic_write_csv(churn_path, churn)
    _atomic_write_csv(anomalies_path, anomalies)
    passed_months = int(coverage["month_pass"].sum()) if not coverage.empty else 0
    full_source = bool(
        len(expected_months) == required_months
        and passed_months == len(expected_months)
        and anomalies.empty
        and not membership.duplicated(["observation_month", "code"]).any()
    )
    manifest = {
        "run_id": run_id,
        "artifact_type": "independent_combined_tse_ose_monthly_membership_qa",
        "status": "pass_candidate_source" if full_source else "in_progress",
        "scorecard_eligible": False,
        "evidence_scope": "full_20_year_monthly_tse_ose_union" if full_source else "bounded_partial",
        "independent_validation": True,
        "source_start_month": start_month,
        "source_end_month": end_month,
        "planned_months": len(expected_months),
        "passed_months": passed_months,
        "planned_files": planned_files,
        "completed_files": completed_files,
        "membership_rows": len(membership),
        "anomalies": len(anomalies),
        "max_monthly_churn_rate": max_monthly_churn_rate,
        "min_monthly_codes": min_monthly_codes,
        "required_months": required_months,
        "source_hashes": source_hashes,
        "output_sha256": {
            "monthly_membership.parquet": _sha256(membership_path),
            "monthly_coverage.csv": _sha256(coverage_path),
            "monthly_churn.csv": _sha256(churn_path),
            "anomalies.csv": _sha256(anomalies_path),
        },
        "artifacts": {
            "monthly_membership": str(membership_path.resolve()),
            "monthly_coverage": str(coverage_path.resolve()),
            "monthly_churn": str(churn_path.resolve()),
            "anomalies": str(anomalies_path.resolve()),
        },
        "promotion_policy": {"alpha": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(manifest_path, manifest)
    return JapanCombinedMonthlyMembershipQAArtifacts(
        run_id, root, membership_path, coverage_path, churn_path, anomalies_path, manifest_path, manifest
    )


def _source_kind(manifest: dict[str, Any]) -> str:
    artifact_type = str(manifest.get("artifact_type", "")).lower()
    if "ose_main_market_monthly" in artifact_type:
        return "OSE_MAIN"
    if "jasdaq_then_ose_combined_monthly" in artifact_type:
        return "JASDAQ_OSE_COMBINED"
    if "jpx_monthly" in artifact_type or "tse" in str(manifest.get("publisher", "")).lower():
        return "TSE"
    return ""


def _source_family_for_month(source_kind: str, observation_month: str) -> str:
    if source_kind == "JASDAQ_OSE_COMBINED":
        return (
            "JASDAQ"
            if pd.Period(observation_month, freq="M") <= pd.Period("2010-09", freq="M")
            else "OSE_COMBINED"
        )
    return source_kind


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)
