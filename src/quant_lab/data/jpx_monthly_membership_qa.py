from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc


@dataclass(frozen=True)
class JPXMonthlyMembershipQAArtifacts:
    run_id: str
    artifact_dir: Path
    monthly_coverage_path: Path
    monthly_churn_path: Path
    anomalies_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def run_jpx_monthly_membership_qa(
    *,
    source_dir: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts/jpx_monthly_membership_qa",
    max_monthly_churn_rate: float = 0.15,
) -> JPXMonthlyMembershipQAArtifacts:
    source = Path(source_dir)
    plan_path = source / "request_plan.csv"
    status_path = source / "file_status.csv"
    quality_path = source / "file_quality.csv"
    if not plan_path.is_file() or not status_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError("monthly quotation plan, status, and quality artifacts are required")
    plan = pd.read_csv(plan_path, dtype=str).fillna("")
    status = pd.read_csv(status_path, dtype=str).fillna("")
    quality = pd.read_csv(quality_path, dtype=str).fillna("")
    completed = status.loc[status["status"].eq("completed")].copy()
    joined = plan.merge(completed, on="request_id", how="inner", validate="one_to_one")
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    coverage_path = root / "monthly_coverage.csv"
    churn_path = root / "monthly_churn.csv"
    anomalies_path = root / "anomalies.csv"
    manifest_path = root / "manifest.json"

    month_codes: dict[str, set[str]] = {}
    month_passes: dict[str, bool] = {}
    coverage_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    for month, group in joined.groupby("observation_month", sort=True):
        frames = []
        hash_failures = 0
        for row in group.to_dict("records"):
            normalized = Path(str(row["normalized_path"]))
            expected = str(row.get("normalized_sha256", ""))
            if not normalized.is_file() or len(expected) != 64 or _sha256(normalized) != expected:
                hash_failures += 1
                continue
            frames.append(
                pd.read_parquet(
                    normalized,
                    columns=["code", "known_at_utc", "currency", "pit_usable", "source_sha256"],
                )
            )
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        codes = set(data["code"].astype(str)) if not data.empty else set()
        duplicates = int(data.duplicated("code").sum()) if not data.empty else 0
        expected_files = int(plan.loc[plan["observation_month"].eq(month), "request_id"].nunique())
        completed_files = int(group["request_id"].nunique())
        timestamp_coverage = float(pd.to_datetime(data.get("known_at_utc"), errors="coerce", utc=True).notna().mean()) if not data.empty else 0.0
        currency_coverage = float(data.get("currency", pd.Series(dtype=str)).eq("JPY").mean()) if not data.empty else 0.0
        pit_coverage = float(data.get("pit_usable", pd.Series(dtype=bool)).astype(bool).mean()) if not data.empty else 0.0
        passed = bool(
            expected_files == completed_files
            and hash_failures == 0
            and len(codes) > 0
            and duplicates == 0
            and timestamp_coverage == 1.0
            and currency_coverage == 1.0
            and pit_coverage == 1.0
        )
        coverage_rows.append(
            {
                "observation_month": month,
                "expected_files": expected_files,
                "completed_files": completed_files,
                "normalized_rows": len(data),
                "unique_codes": len(codes),
                "duplicate_codes": duplicates,
                "hash_failures": hash_failures,
                "known_at_coverage": timestamp_coverage,
                "jpy_coverage": currency_coverage,
                "pit_usable_coverage": pit_coverage,
                "month_pass": passed,
            }
        )
        if not passed:
            anomaly_rows.append({"observation_month": month, "type": "monthly_quality_failure", "value": ""})
        month_codes[month] = codes
        month_passes[month] = passed

    churn_rows = []
    ordered = sorted(month_codes)
    for previous, current in zip(ordered, ordered[1:]):
        if not month_passes[previous] or not month_passes[current]:
            continue
        before = month_codes[previous]
        after = month_codes[current]
        entrants = after - before
        exits = before - after
        denominator = max(len(before), 1)
        churn_rate = (len(entrants) + len(exits)) / denominator
        period_gap = pd.Period(current, freq="M") - pd.Period(previous, freq="M")
        consecutive = getattr(period_gap, "n", period_gap) == 1
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
            anomaly_rows.append(
                {"observation_month": current, "type": "continuity_or_churn_failure", "value": churn_rate}
            )

    coverage = pd.DataFrame(coverage_rows)
    churn = pd.DataFrame(
        churn_rows,
        columns=[
            "previous_month",
            "observation_month",
            "entrants",
            "exits",
            "churn_rate",
            "consecutive_months",
            "churn_pass",
        ],
    )
    anomalies = pd.DataFrame(anomaly_rows, columns=["observation_month", "type", "value"])
    _atomic_write_csv(coverage_path, coverage)
    _atomic_write_csv(churn_path, churn)
    _atomic_write_csv(anomalies_path, anomalies)
    completed_months = int(coverage["month_pass"].sum()) if not coverage.empty else 0
    planned_months = int(plan["observation_month"].nunique())
    full_source = bool(
        completed_months == planned_months
        and not anomalies.shape[0]
        and len(completed) == len(plan)
    )
    manifest = {
        "run_id": run_id,
        "artifact_type": "independent_jpx_monthly_membership_continuity_qa",
        "status": "pass_candidate_source" if full_source else "in_progress",
        "scorecard_eligible": False,
        "evidence_scope": "full" if full_source else "bounded_partial",
        "independent_validation": True,
        "planned_files": len(plan),
        "completed_files": len(completed),
        "planned_months": planned_months,
        "completed_months": completed_months,
        "anomalies": len(anomalies),
        "max_monthly_churn_rate": max_monthly_churn_rate,
        "source_hashes": {
            "request_plan.csv": _sha256(plan_path),
            "file_status.csv": _sha256(status_path),
            "file_quality.csv": _sha256(quality_path),
        },
        "artifacts": {
            "monthly_coverage": str(coverage_path.resolve()),
            "monthly_churn": str(churn_path.resolve()),
            "anomalies": str(anomalies_path.resolve()),
        },
        "promotion_policy": {"alpha": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return JPXMonthlyMembershipQAArtifacts(
        run_id=run_id,
        artifact_dir=root,
        monthly_coverage_path=coverage_path,
        monthly_churn_path=churn_path,
        anomalies_path=anomalies_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


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


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        handle.write(payload)
        temp = Path(handle.name)
    os.replace(temp, path)
