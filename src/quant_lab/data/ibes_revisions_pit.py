from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DATASET_ID = "ibes_estimates_revisions"
FIELD_ROLE = "revision_time"
ROUTE = "get_data"

REQUESTED_COUNTRIES = ("MYS", "SGP", "IDN")

GENERIC_FAILED_REVISION_DATE_FIELDS = (
    "TR.EstimateDate",
    "TR.EPSMeanDate",
    "TR.RevenueMeanDate",
    "TR.IBESDate",
    "TR.RevisionDate",
    "TR.ConsensusDate",
    "TR.LastUpdatedDate",
)

CONSENSUS_REVISION_FIELD_CANDIDATES = (
    "TR.EPSMean.date",
    "TR.EPSMean.Date",
    "TR.EPSMean.CalcDate",
    "TR.EPSMean.rfperiod",
    "TR.EPSMean",
    "TR.RevenueMean.date",
    "TR.RevenueMean.CalcDate",
    "TR.RevenueMean.rfperiod",
    "TR.RevenueMean",
    "TR.RecMean.date",
    "TR.RecMean.CalcDate",
    "TR.RecMean",
    "TR.PriceTargetMean.date",
    "TR.PriceTargetMean.CalcDate",
    "TR.PriceTargetMean",
)

ANALYST_DETAIL_BASE_FIELDS = (
    "TR.AnalystName",
    "TR.RecLabelEstBrokerName",
    "TR.RecEstValue",
    "TR.BrkRecLabel",
    "TR.TPEstValue",
    "TR.EPSLTGEstValue",
)

RESULT_COLUMNS = [
    "dataset_id",
    "field_role",
    "route",
    "field",
    "parameter_set",
    "status",
    "country",
    "sample_rows",
    "non_null_date_rows",
    "date_min",
    "date_max",
    "unique_date_count",
    "rejection_reason",
]

COUNTRY_ALIASES = {
    "MYS": "MYS",
    "MY": "MYS",
    "MALAYSIA": "MYS",
    "SGP": "SGP",
    "SG": "SGP",
    "SINGAPORE": "SGP",
    "IDN": "IDN",
    "ID": "IDN",
    "INDONESIA": "IDN",
    "JPN": "JPN",
    "JP": "JPN",
    "JAPAN": "JPN",
}


@dataclass(frozen=True)
class IBESRevisionRouteProbeArtifacts:
    run_id: str
    artifact_dir: Path
    route_results_path: Path
    accepted_map_path: Path
    manifest_path: Path
    route_results: pd.DataFrame
    accepted_revision_time_map: pd.DataFrame


def default_ibes_revision_parameter_grid(start_date: str, end_date: str) -> list[dict[str, str]]:
    """Return daily FY1/FY2 parameter sets for IBES revision-style `get_data` probes."""

    return [
        {"SDate": start_date, "EDate": end_date, "Frq": "D", "Period": period}
        for period in ("FY1", "FY2")
    ]


def ibes_revision_timestamp_candidate_fields(include_analyst_detail: bool = True) -> list[str]:
    """Build the unresolved IBES revision timestamp field list from the audit summary."""

    fields = list(CONSENSUS_REVISION_FIELD_CANDIDATES)
    if include_analyst_detail:
        for base_field in ANALYST_DETAIL_BASE_FIELDS:
            fields.append(base_field)
            fields.extend(f"{base_field}{suffix}" for suffix in (".date", ".Date", ".CalcDate"))
    failed = set(GENERIC_FAILED_REVISION_DATE_FIELDS)
    return list(dict.fromkeys(field for field in fields if field not in failed))


def build_ibes_revision_route_probe_queue(
    start_date: str,
    end_date: str,
    fields: Sequence[str] | None = None,
    parameter_grid: Sequence[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build deterministic `get_data` field/parameter combinations for route probing."""

    candidate_fields = list(fields) if fields is not None else ibes_revision_timestamp_candidate_fields()
    parameters = (
        list(parameter_grid)
        if parameter_grid is not None
        else default_ibes_revision_parameter_grid(start_date=start_date, end_date=end_date)
    )
    rows: list[dict[str, Any]] = []
    for field in candidate_fields:
        if str(field) in GENERIC_FAILED_REVISION_DATE_FIELDS:
            continue
        for params in parameters:
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "field_role": FIELD_ROLE,
                    "route": ROUTE,
                    "field": str(field),
                    "parameter_set": _parameter_key(params),
                    "parameters": dict(params),
                }
            )
    return pd.DataFrame(rows)


def probe_ibes_revision_timestamp_routes(
    client: Any,
    selected_universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    countries: Sequence[str] = REQUESTED_COUNTRIES,
    fields: Sequence[str] | None = None,
    parameter_grid: Sequence[dict[str, Any]] | None = None,
    sample_per_country: int = 1,
    tier: str = "u3",
    run_date: str | None = None,
    load_date: str | None = None,
    sleep_seconds: float = 0.05,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Probe source-backed IBES revision timestamp routes by country.

    The caller supplies a live LSEG client or a fake with a compatible
    `get_data(universe, fields, parameters=...)` method.
    """

    if sample_per_country <= 0:
        raise ValueError("sample_per_country must be positive")

    requested = _normalize_country_list(countries)
    queue = build_ibes_revision_route_probe_queue(
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        parameter_grid=parameter_grid,
    )
    samples_by_country = _sample_rics_by_country(
        selected_universe,
        countries=requested,
        sample_per_country=sample_per_country,
        tier=tier,
    )

    rows: list[dict[str, Any]] = []
    work = [(country, item._asdict()) for country in requested for item in queue.itertuples(index=False)]
    total = len(work)
    for completed, (country, candidate) in enumerate(work, start=1):
        field = str(candidate["field"])
        parameter_set = str(candidate["parameter_set"])
        parameters = dict(candidate["parameters"])
        rics = samples_by_country.get(country, [])
        base_row = {
            "dataset_id": DATASET_ID,
            "field_role": FIELD_ROLE,
            "route": ROUTE,
            "field": field,
            "parameter_set": parameter_set,
            "status": "not_run",
            "country": country,
            "sample_rows": 0,
            "non_null_date_rows": 0,
            "date_min": "",
            "date_max": "",
            "unique_date_count": 0,
            "rejection_reason": "",
        }
        if not rics:
            rows.append({**base_row, "status": "blocked_no_sample_rics", "rejection_reason": "no_sample_rics"})
            continue

        try:
            frame = _get_data_frame(client, universe=rics, fields=[field], parameters=parameters)
            rows.append(
                validate_ibes_revision_date_frame(
                    frame,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters,
                    run_date=run_date,
                    load_date=load_date,
                    base_row=base_row,
                )
            )
        except Exception as exc:  # pragma: no cover - live vendor path
            rows.append(
                {
                    **base_row,
                    "status": "failed",
                    "rejection_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if progress_every > 0 and (completed % progress_every == 0 or completed == total):
            print(f"Probed IBES revision routes {completed}/{total}: {country} {field}", file=sys.stderr, flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows)[RESULT_COLUMNS]


def validate_ibes_revision_date_frame(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    parameters: dict[str, Any],
    run_date: str | None = None,
    load_date: str | None = None,
    base_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a returned frame as a source-backed PIT revision timestamp route."""

    row = dict(base_row or {})
    for column in RESULT_COLUMNS:
        row.setdefault(column, "" if column in {"date_min", "date_max", "rejection_reason"} else 0)
    row.update({"status": "rejected", "sample_rows": int(len(frame))})

    dates = _best_date_series(frame)
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return _reject(row, "no_source_dates")

    normalized_dates = valid_dates.dt.normalize()
    unique_dates = sorted({item.date().isoformat() for item in normalized_dates})
    row["non_null_date_rows"] = int(valid_dates.shape[0])
    row["date_min"] = normalized_dates.min().date().isoformat()
    row["date_max"] = normalized_dates.max().date().isoformat()
    row["unique_date_count"] = int(len(unique_dates))

    sentinel_dates = {
        pd.Timestamp(start_date).date().isoformat(),
        pd.Timestamp(end_date).date().isoformat(),
    }
    if run_date:
        sentinel_dates.add(pd.Timestamp(run_date).date().isoformat())
    else:
        sentinel_dates.add(pd.Timestamp.utcnow().date().isoformat())
    if load_date:
        sentinel_dates.add(pd.Timestamp(load_date).date().isoformat())
    if set(unique_dates).issubset(sentinel_dates):
        return _reject(row, "sentinel_date_pattern")

    start = pd.Timestamp(start_date).normalize().tz_localize(None)
    end = pd.Timestamp(end_date).normalize().tz_localize(None)
    in_window = normalized_dates.ge(start) & normalized_dates.le(end)
    if not bool(in_window.any()):
        return _reject(row, "no_dates_inside_window")
    if (normalized_dates > end).any():
        return _reject(row, "future_dates_outside_window")

    active_run_date = (
        pd.Timestamp(run_date).normalize()
        if run_date
        else pd.Timestamp.utcnow().tz_localize(None).normalize()
    )
    active_run_date = active_run_date.tz_localize(None) if getattr(active_run_date, "tzinfo", None) else active_run_date
    if (normalized_dates > active_run_date).any():
        return _reject(row, "future_leakage")

    if str(parameters.get("Frq", "")).upper() != "D" or not str(parameters.get("Period", "")).strip():
        return _reject(row, "unclear_grain")

    in_window_unique_dates = sorted({item.date().isoformat() for item in normalized_dates[in_window]})
    if len(in_window_unique_dates) < 2:
        return _reject(row, "no_varying_source_dates")

    row["status"] = "accepted"
    row["rejection_reason"] = ""
    return row


def accepted_ibes_revision_time_map(
    route_probe_results: pd.DataFrame,
    countries: Sequence[str] = REQUESTED_COUNTRIES,
) -> pd.DataFrame:
    """Promote the strongest accepted route per requested country."""

    requested = set(_normalize_country_list(countries))
    if route_probe_results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    data = route_probe_results.copy()
    data = data[
        data["dataset_id"].astype(str).eq(DATASET_ID)
        & data["field_role"].astype(str).eq(FIELD_ROLE)
        & data["status"].astype(str).eq("accepted")
        & data["country"].astype(str).isin(requested)
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    data["_probe_order"] = range(len(data))
    for column in ("unique_date_count", "non_null_date_rows", "sample_rows"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data = data.sort_values(
        ["country", "unique_date_count", "non_null_date_rows", "sample_rows", "_probe_order"],
        ascending=[True, False, False, False, True],
    )
    promoted = data.drop_duplicates("country", keep="first").drop(columns=["_probe_order"])
    promoted = promoted.sort_values("country").reset_index(drop=True)
    return promoted[RESULT_COLUMNS]


def write_ibes_revision_route_probe_artifacts(
    artifact_root: str | Path,
    run_id: str,
    route_probe_results: pd.DataFrame,
    countries: Sequence[str] = REQUESTED_COUNTRIES,
    session_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> IBESRevisionRouteProbeArtifacts:
    """Write route results, accepted map, and manifest for the IBES timestamp probe."""

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = route_probe_results.copy()
    if results.empty:
        results = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        results = results[RESULT_COLUMNS]
    accepted = accepted_ibes_revision_time_map(results, countries=countries)

    route_results_path = artifact_dir / "route_probe_results.csv"
    accepted_map_path = artifact_dir / "accepted_revision_time_map.csv"
    manifest_path = artifact_dir / "manifest.json"
    results.to_csv(route_results_path, index=False)
    accepted.to_csv(accepted_map_path, index=False)

    requested_countries = _normalize_country_list(countries)
    accepted_countries = sorted(accepted["country"].astype(str).dropna().unique().tolist()) if not accepted.empty else []
    missing_countries = [country for country in requested_countries if country not in set(accepted_countries)]
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "session_name": session_name,
        "dataset_id": DATASET_ID,
        "field_role": FIELD_ROLE,
        "promotion_target": f"{DATASET_ID}/{FIELD_ROLE}",
        "route": ROUTE,
        "start_date": start_date,
        "end_date": end_date,
        "requested_countries": requested_countries,
        "accepted_countries": accepted_countries,
        "missing_countries": missing_countries,
        "probe_rows": int(len(results)),
        "accepted_routes": int(len(accepted)),
        "status_counts": {
            str(key): int(value) for key, value in results["status"].value_counts(dropna=False).items()
        }
        if "status" in results
        else {},
        "status": "complete" if not missing_countries else "incomplete",
        "completed_at_utc": _now_utc(),
        "artifacts": {
            "route_probe_results": str(route_results_path.resolve()),
            "accepted_revision_time_map": str(accepted_map_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return IBESRevisionRouteProbeArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        route_results_path=route_results_path,
        accepted_map_path=accepted_map_path,
        manifest_path=manifest_path,
        route_results=results,
        accepted_revision_time_map=accepted,
    )


def _get_data_frame(client: Any, universe: Sequence[str], fields: Sequence[str], parameters: dict[str, Any]) -> pd.DataFrame:
    get_data = getattr(client, "get_data", client if callable(client) else None)
    if get_data is None:
        raise TypeError("client must expose get_data(...) or be callable")
    try:
        raw = get_data(universe=list(universe), fields=list(fields), parameters=dict(parameters))
    except TypeError:
        raw = get_data(list(universe), list(fields), dict(parameters))
    if isinstance(raw, tuple):
        raw = raw[0]
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    else:
        frame = pd.DataFrame(raw)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["|".join(str(part) for part in column if str(part)) for column in frame.columns]
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    return frame


def _best_date_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="datetime64[ns]")

    best = pd.Series(dtype="datetime64[ns]")
    best_count = 0
    for column in frame.columns:
        if _is_identifier_column(column):
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        count = int(parsed.notna().sum())
        if count > best_count:
            best = parsed.dt.tz_convert(None)
            best_count = count
    return best


def _sample_rics_by_country(
    selected_universe: pd.DataFrame,
    countries: Sequence[str],
    sample_per_country: int,
    tier: str,
) -> dict[str, list[str]]:
    if selected_universe.empty or "refinitiv_ric" not in selected_universe.columns:
        return {country: [] for country in countries}

    data = selected_universe[selected_universe["refinitiv_ric"].notna()].copy()
    data["_country_code"] = data.apply(_country_code_from_row, axis=1)
    data = data[data["_country_code"].isin(set(countries))].copy()
    tier_col = f"selected_for_{tier.lower()}"
    if tier_col in data.columns:
        tier_data = data[data[tier_col].fillna(False).astype(bool)].copy()
        if not tier_data.empty:
            data = tier_data
    if data.empty:
        return {country: [] for country in countries}

    if "adv_60_country_rank" in data.columns:
        adv_rank = pd.to_numeric(data["adv_60_country_rank"], errors="coerce")
    else:
        adv_rank = pd.Series(10**9, index=data.index, dtype="float64")
    data["_adv_rank_sort"] = adv_rank.fillna(10**9)
    sort_cols = [col for col in ("_country_code", "_adv_rank_sort", "ticker", "refinitiv_ric") if col in data.columns]
    sampled = data.sort_values(sort_cols).groupby("_country_code", group_keys=False).head(sample_per_country)
    return {
        country: sampled.loc[sampled["_country_code"].eq(country), "refinitiv_ric"].astype(str).drop_duplicates().tolist()
        for country in countries
    }


def _country_code_from_row(row: pd.Series) -> str:
    for column in ("country_code", "country_iso3", "iso3", "country"):
        if column in row and pd.notna(row[column]):
            code = COUNTRY_ALIASES.get(str(row[column]).strip().upper())
            if code:
                return code
    return ""


def _normalize_country_list(countries: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for country in countries:
        code = COUNTRY_ALIASES.get(str(country).strip().upper(), str(country).strip().upper())
        if code and code not in normalized:
            normalized.append(code)
    return normalized


def _is_identifier_column(column: object) -> bool:
    text = str(column).strip().lower().replace(" ", "_")
    return text in {
        "instrument",
        "instruments",
        "ric",
        "requested_ric",
        "ticker",
        "symbol",
        "index",
    }


def _parameter_key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _reject(row: dict[str, Any], reason: str) -> dict[str, Any]:
    row["status"] = "rejected"
    row["rejection_reason"] = reason
    return row


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
