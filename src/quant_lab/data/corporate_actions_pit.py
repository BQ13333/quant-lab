from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from quant_lab.data.vendor_adapters import (
    RefinitivRetryableProviderError,
    is_refinitiv_retryable_provider_error,
)


CORPORATE_ACTION_EVENT_ROUTE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "event_family": "dividend",
        "field_role": "dividend_event",
        "date_type": "ED",
        "fields": (
            "TR.DivAnnouncementDate",
            "TR.DivExDate",
            "TR.DivRecordDate",
            "TR.DivPayDate",
            "TR.DivUnadjustedGross",
            "TR.DivCurrency",
            "TR.DivType",
        ),
    },
    {
        "event_family": "capital_change",
        "field_role": "split_event",
        "date_type": "ED",
        "fields": (
            "TR.CAAnnouncementDate",
            "TR.CAEventType",
            "TR.CAEffectiveDate",
            "TR.CAExDate",
            "TR.CAAdjustmentFactor",
            "TR.SplitFactor",
        ),
    },
)


@dataclass(frozen=True)
class CorporateActionEventRunArtifacts:
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


def build_corporate_action_event_plan(
    selected_universe: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    tier: str = "u2",
    batch_size: int = 50,
    event_families: Sequence[str] | None = None,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    universe = _filter_universe(selected_universe, tier=tier)
    if universe.empty:
        raise ValueError("selected universe is empty")
    wanted = {item for item in event_families or [] if str(item).strip()}
    specs = [spec for spec in CORPORATE_ACTION_EVENT_ROUTE_SPECS if not wanted or spec["event_family"] in wanted]
    if not specs:
        raise ValueError("no corporate-action event specs selected")

    output_root = Path(artifact_root) / run_id / "events"
    rows: list[dict[str, Any]] = []
    for country, country_group in universe.groupby("country", dropna=False, sort=True):
        country_rics = country_group["refinitiv_ric"].dropna().astype(str).drop_duplicates().tolist()
        for batch_number, rics in enumerate(_chunked(country_rics, batch_size), start=1):
            for spec in specs:
                event_family = str(spec["event_family"])
                field_role = str(spec["field_role"])
                date_type = str(spec.get("date_type", "ED"))
                expressions = list(spec["fields"])
                row_id = f"{_slugish(event_family)}-{_slugish(country)}-{batch_number:03d}"
                output_path = (
                    output_root
                    / f"event_family={_slugish(event_family)}"
                    / f"country={_slugish(country)}"
                    / f"batch={batch_number:03d}_{start_date}_{end_date}.parquet"
                )
                rows.append(
                    {
                        "plan_row_id": row_id,
                        "run_id": run_id,
                        "dataset_id": "corporate_actions",
                        "field_role": field_role,
                        "event_family": event_family,
                        "route": "get_data_event",
                        "country": country,
                        "batch_number": batch_number,
                        "start_date": start_date,
                        "end_date": end_date,
                        "date_type": date_type,
                        "tier": tier,
                        "requested_rics": len(rics),
                        "ric_list": ",".join(rics),
                        "candidate_fields": "|".join(spec["fields"]),
                        "field_expressions": "|".join(expressions),
                        "route_parameters": json.dumps({"SDate": start_date, "EDate": end_date, "DateType": date_type}),
                        "output_path": str(output_path),
                        "status": "pending",
                    }
                )
    return pd.DataFrame(rows)


def execute_corporate_action_event_plan(
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
) -> CorporateActionEventRunArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_dir / "event_plan.csv"
    manifest_path = artifact_dir / "event_manifest.csv"
    quality_path = artifact_dir / "event_quality.csv"
    errors_path = artifact_dir / "event_errors.csv"
    run_manifest_path = artifact_dir / "manifest.json"
    output_root = artifact_dir / "events"
    plan_sha256 = stage_corporate_action_event_plan(plan, plan_path)

    existing_manifest = pd.read_csv(manifest_path, low_memory=False) if resume and manifest_path.exists() else pd.DataFrame()
    resume_statuses = ["completed", "completed_empty"]
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

    universe = selected_universe.copy()
    if "refinitiv_ric" in universe.columns:
        universe["refinitiv_ric"] = universe["refinitiv_ric"].astype(str)
    manifest_rows = existing_manifest.to_dict("records") if not existing_manifest.empty else []

    provider_halt: dict[str, Any] | None = None
    for item in pending.itertuples(index=False):
        stop_after_row = False
        row = item._asdict()
        rics = [ric for ric in str(row.get("ric_list", "")).split(",") if ric]
        fields = [field for field in str(row.get("field_expressions", "")).split("|") if field]
        parameters = json.loads(str(row.get("route_parameters") or "{}"))
        started_at = _now_utc()
        output_path = Path(str(row["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = _get_data_frame(ld, universe=rics, fields=fields, parameters=parameters)
            events = normalize_corporate_action_event_frame(
                raw,
                rics=rics,
                plan_row=row,
                universe=universe,
                source_loaded_at_utc=started_at,
            )
            stats = _event_stats(events, requested_rics=len(rics))
            stats["event_ric_coverage"] = stats.pop("coverage")
            stats["coverage"] = 1.0
            if not events.empty:
                events.to_parquet(output_path, index=False)
                events.to_csv(output_path.with_suffix(".csv"), index=False)
            status = "completed" if stats["event_rows"] > 0 else "completed_empty"
            manifest_row = {
                **row,
                **stats,
                "status": status,
                "started_at_utc": started_at,
                "completed_at_utc": _now_utc(),
                "error_type": "",
                "error": "",
                "response_success": True,
            }
        except Exception as exc:
            retryable = is_refinitiv_retryable_provider_error(exc)
            error = RefinitivRetryableProviderError(str(exc)) if retryable else exc
            manifest_row = {
                **row,
                "returned_rows": 0,
                "event_rows": 0,
                "known_at_rows": 0,
                "valid_event_date_rows": 0,
                "alpha_usable_rows": 0,
                "coverage": 0.0,
                "event_ric_coverage": 0.0,
                "status": "failed_retryable_provider" if retryable else "failed",
                "started_at_utc": started_at,
                "completed_at_utc": _now_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "response_success": False,
            }
            if retryable:
                provider_halt = {
                    "plan_row_id": str(row.get("plan_row_id", "")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                stop_after_row = True
        manifest_rows = [existing for existing in manifest_rows if str(existing.get("plan_row_id")) != str(row.get("plan_row_id"))]
        manifest_rows.append(manifest_row)
        _atomic_to_csv(
            pd.DataFrame(manifest_rows).sort_values(["event_family", "country", "batch_number"]),
            manifest_path,
        )
        if stop_after_row:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    manifest = pd.read_csv(manifest_path, low_memory=False) if manifest_path.exists() else pd.DataFrame(manifest_rows)
    quality = corporate_action_event_quality(manifest)
    errors = _event_errors(manifest)
    _atomic_to_csv(quality, quality_path)
    _atomic_to_csv(errors, errors_path)
    successful_status = manifest.get("status", pd.Series(dtype=str)).astype(str).isin(
        {"completed", "completed_empty"}
    )
    failed_status = manifest.get("status", pd.Series(dtype=str)).astype(str).str.startswith("failed")
    run_manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "scope": "corporate_action_event_backfill",
        "planned_rows": int(len(plan)),
        "plan_sha256": plan_sha256,
        "executed_manifest_rows": int(len(manifest)),
        "completed_rows": int(successful_status.sum()),
        "failed_rows": int(failed_status.sum()),
        "empty_rows": int(manifest["status"].eq("completed_empty").sum()) if "status" in manifest else 0,
        "event_rows": int(pd.to_numeric(manifest.get("event_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not manifest.empty
        else 0,
        "alpha_usable_rows": int(
            pd.to_numeric(manifest.get("alpha_usable_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        )
        if not manifest.empty
        else 0,
        "official_source": True,
        "halted_early": provider_halt is not None,
        "provider_halt": provider_halt,
        "status": "complete" if len(manifest) == len(plan) and not failed_status.any() else "incomplete_or_failed",
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "event_plan": str(plan_path.resolve()),
            "event_manifest": str(manifest_path.resolve()),
            "event_quality": str(quality_path.resolve()),
            "event_errors": str(errors_path.resolve()),
        },
    }
    _atomic_write_json(run_manifest_path, run_manifest)
    return CorporateActionEventRunArtifacts(
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


def refinitiv_field_expression(field: str, parameters: dict[str, str]) -> str:
    if "(" in field and field.endswith(")"):
        return field
    params = ",".join(f"{key}={value}" for key, value in parameters.items())
    return f"{field}({params})"


def stage_corporate_action_event_plan(plan: pd.DataFrame, plan_path: str | Path) -> str:
    """Write-once event plan staging with a persistent SHA-256 guard."""

    path = Path(plan_path)
    payload = plan.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    checksum_path = path.with_suffix(".sha256")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"immutable corporate-action plan drift: {path}")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        if expected != digest:
            raise ValueError(f"corporate-action plan checksum mismatch: {path}")
    if not path.exists():
        _atomic_write_bytes(path, payload)
    if not checksum_path.exists():
        _atomic_write_text(checksum_path, f"{digest}  {path}\n")
    return digest


def normalize_corporate_action_event_frame(
    frame: pd.DataFrame,
    rics: Sequence[str],
    plan_row: dict[str, Any],
    universe: pd.DataFrame,
    source_loaded_at_utc: str,
) -> pd.DataFrame:
    columns = [
        "requested_ric",
        "event_family",
        "field_role",
        "event_date",
        "known_at_date",
        "alpha_usable",
        "eligible_for_return_reconstruction",
        "event_key",
        "source_route",
        "source_loaded_at_utc",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    if "Instrument" in data.columns:
        data["requested_ric"] = data["Instrument"].astype(str)
    elif len(data) == len(rics):
        data["requested_ric"] = list(rics)
    else:
        data["requested_ric"] = pd.NA
    data["event_family"] = plan_row.get("event_family")
    data["field_role"] = plan_row.get("field_role")
    data["event_date"] = _best_date_column(data, ["ex", "effective", "pay", "record", "date"])
    data["known_at_date"] = _best_date_column(data, ["announce", "announcement"])
    data["alpha_usable"] = pd.to_datetime(data["known_at_date"], errors="coerce").notna()
    data["eligible_for_return_reconstruction"] = pd.to_datetime(data["event_date"], errors="coerce").notna()
    data["source_route"] = "get_data_event"
    data["route_parameters"] = plan_row.get("route_parameters", "")
    data["source_loaded_at_utc"] = source_loaded_at_utc
    data["event_key"] = [_event_key(row) for row in data.to_dict("records")]
    data = _merge_universe_metadata(data, universe)
    front = [column for column in columns if column in data.columns]
    remaining = [column for column in data.columns if column not in front]
    return data[front + remaining]


def corporate_action_event_quality(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "field_role",
        "event_family",
        "country",
        "plan_rows",
        "completed_rows",
        "failed_rows",
        "empty_rows",
        "event_rows",
        "known_at_rows",
        "valid_event_date_rows",
        "alpha_usable_rows",
        "avg_coverage",
        "min_coverage",
    ]
    if manifest.empty:
        return pd.DataFrame(columns=columns)
    data = manifest.copy()
    for column in ("event_rows", "known_at_rows", "valid_event_date_rows", "alpha_usable_rows", "coverage"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    quality = (
        data.groupby(["dataset_id", "field_role", "event_family", "country"], dropna=False)
        .agg(
            plan_rows=("plan_row_id", "nunique"),
            completed_rows=(
                "status",
                lambda values: int(pd.Series(values).isin({"completed", "completed_empty"}).sum()),
            ),
            failed_rows=("status", lambda values: int(pd.Series(values).astype(str).str.startswith("failed").sum())),
            empty_rows=("status", lambda values: int(pd.Series(values).eq("completed_empty").sum())),
            event_rows=("event_rows", "sum"),
            known_at_rows=("known_at_rows", "sum"),
            valid_event_date_rows=("valid_event_date_rows", "sum"),
            alpha_usable_rows=("alpha_usable_rows", "sum"),
            avg_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
        )
        .reset_index()
    )
    return quality[columns].sort_values(["event_family", "country"]).reset_index(drop=True)


def _event_stats(events: pd.DataFrame, requested_rics: int) -> dict[str, Any]:
    if events.empty:
        return {
            "returned_rows": 0,
            "event_rows": 0,
            "known_at_rows": 0,
            "valid_event_date_rows": 0,
            "alpha_usable_rows": 0,
            "coverage": 0.0,
        }
    event_rows = int(len(events))
    known_at_rows = int(events.get("known_at_date", pd.Series(dtype=object)).notna().sum())
    valid_event_date_rows = int(pd.to_datetime(events.get("event_date", pd.Series(dtype=object)), errors="coerce").notna().sum())
    alpha_usable_rows = int(events.get("alpha_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    ric_count = int(events.get("requested_ric", pd.Series(dtype=object)).dropna().astype(str).nunique())
    return {
        "returned_rows": event_rows,
        "event_rows": event_rows,
        "known_at_rows": known_at_rows,
        "valid_event_date_rows": valid_event_date_rows,
        "alpha_usable_rows": alpha_usable_rows,
        "coverage": float(ric_count / max(1, requested_rics)),
    }


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _event_errors(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = ["plan_row_id", "dataset_id", "field_role", "event_family", "country", "error_type", "error"]
    if manifest.empty or "status" not in manifest.columns:
        return pd.DataFrame(columns=columns)
    failed = manifest[manifest["status"].astype(str).str.startswith("failed")].copy()
    if failed.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in failed.columns:
            failed[column] = ""
    return failed[columns].reset_index(drop=True)


def _get_data_frame(ld: Any, universe: Sequence[str], fields: Sequence[str], parameters: dict[str, Any] | None = None) -> pd.DataFrame:
    try:
        raw = ld.get_data(universe=list(universe), fields=list(fields), parameters=parameters or None)
    except TypeError:
        raw = ld.get_data(universe=list(universe), fields=list(fields))
    if isinstance(raw, tuple):
        raw = raw[0]
    if raw is None:
        return pd.DataFrame()
    if not isinstance(raw, pd.DataFrame):
        return pd.DataFrame(raw)
    return raw.copy()


def _filter_universe(selected_universe: pd.DataFrame, tier: str) -> pd.DataFrame:
    if selected_universe.empty:
        return selected_universe.copy()
    data = selected_universe.copy()
    tier_col = f"selected_for_{tier.lower()}"
    if tier_col in data.columns:
        data = data[data[tier_col].fillna(False).astype(bool)].copy()
    if "refinitiv_ric" not in data.columns:
        raise ValueError("selected universe missing `refinitiv_ric`")
    return data[data["refinitiv_ric"].notna()].drop_duplicates("refinitiv_ric").reset_index(drop=True)


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _best_date_column(frame: pd.DataFrame, hints: Sequence[str]) -> pd.Series:
    for column in frame.columns:
        text = str(column).lower()
        if any(hint in text for hint in hints):
            values = pd.to_datetime(frame[column], errors="coerce")
            if values.notna().any():
                return values.dt.date
    return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")


def _merge_universe_metadata(events: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if events.empty or universe.empty or "requested_ric" not in events.columns:
        return events
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
    available = [column for column in keep if column in universe.columns]
    if "refinitiv_ric" not in available:
        return events
    meta = universe[available].drop_duplicates("refinitiv_ric")
    return events.merge(meta, left_on="requested_ric", right_on="refinitiv_ric", how="left")


def _event_key(row: dict[str, Any]) -> str:
    material = "|".join(
        str(row.get(key, ""))
        for key in ("requested_ric", "event_family", "event_date", "known_at_date", "TR.DivExDate", "TR.CAExDate")
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _slugish(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("/", "_").replace(":", "_")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
