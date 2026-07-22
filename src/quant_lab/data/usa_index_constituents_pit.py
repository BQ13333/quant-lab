from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DATASET_ID = "usa_pit_index_constituents"
INDEX_FAMILIES = ("Russell", "S&P", "Nasdaq", "Dow")
MEMBERSHIP_ACTIONS = ("snapshot_member", "added", "removed")

USA_INDEX_SCOPE_PROFILE_FULL = "full"
USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL = "backtest_no_russell"

_FULL_INDEX_IDS = (
    "russell_1000",
    "russell_2000",
    "russell_3000",
    "sp_500",
    "nasdaq_100",
    "dow_jones_industrial_average",
)
_BACKTEST_NO_RUSSELL_INDEX_IDS = (
    "sp_500",
    "nasdaq_100",
    "dow_jones_industrial_average",
)
_RUSSELL_INDEX_IDS = (
    "russell_1000",
    "russell_2000",
    "russell_3000",
)
_BACKTEST_NO_RUSSELL_EXCLUSION_REASON = (
    "Russell 1000/2000/3000 are deliberately excluded by the approved USA investment mandate; "
    "the backtest scope requires only point-in-time S&P 500, Nasdaq-100, and Dow membership."
)

INDEX_CATALOG_COLUMNS = (
    "index_id",
    "index_family",
    "index_name",
    "candidate_index_ric",
    "candidate_ric_status",
    "scope",
)

PIT_CONSTITUENT_COLUMNS = (
    "index_id",
    "index_family",
    "index_name",
    "index_ric",
    "member_ric",
    "member_perm_id",
    "member_isin",
    "member_ticker",
    "member_name",
    "membership_action",
    "membership_as_of_date",
    "effective_date",
    "weight_percent",
    "known_at_utc",
    "known_at_source_field",
    "source_provider",
    "source_dataset",
    "source_record_id",
    "source_request_id",
    "source_loaded_at_utc",
    "raw_payload_sha256",
)

DERIVED_PIT_COLUMNS = ("pit_usable", "quarantine_reason")
ISSUE_COLUMNS = (
    "row_number",
    "source_record_id",
    "index_id",
    "severity",
    "issue_code",
    "column",
    "detail",
)

_PROVENANCE_COLUMNS = (
    "known_at_utc",
    "known_at_source_field",
    "source_provider",
    "source_dataset",
    "source_record_id",
    "source_request_id",
    "source_loaded_at_utc",
    "raw_payload_sha256",
)
_STRING_COLUMNS = tuple(
    column
    for column in PIT_CONSTITUENT_COLUMNS
    if column not in {"membership_as_of_date", "effective_date", "weight_percent", "known_at_utc", "source_loaded_at_utc"}
)
_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_DISALLOWED_KNOWN_AT_FIELDS = {
    "as_of_date",
    "effective_date",
    "ingested_at",
    "load_date",
    "loaded_at",
    "retrieved_at",
    "run_date",
    "source_loaded_at_utc",
}


@dataclass(frozen=True)
class PITConstituentValidation:
    rows: pd.DataFrame
    issues: pd.DataFrame
    supplied_rows: int
    usable_rows: int
    covered_index_ids: tuple[str, ...]
    covered_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class USAIndexScopeProfile:
    name: str
    required_index_ids: tuple[str, ...]
    excluded_index_ids: tuple[str, ...]
    excluded_index_reason: str | None


@dataclass(frozen=True)
class USAIndexConstituentPlanArtifacts:
    run_id: str
    artifact_dir: Path
    index_catalog_path: Path
    schema_path: Path
    field_probe_plan_path: Path
    request_plan_path: Path
    validation_issues_path: Path
    pit_rows_path: Path | None
    summary_path: Path
    manifest_path: Path
    index_catalog: pd.DataFrame
    field_probe_plan: pd.DataFrame
    request_plan: pd.DataFrame
    validation: PITConstituentValidation


def default_usa_index_catalog() -> pd.DataFrame:
    """Return benchmark scope with the confirmed July 2026 LSEG entitlement status."""

    rows = (
        ("russell_1000", "Russell", "Russell 1000", ".RUI"),
        ("russell_2000", "Russell", "Russell 2000", ".RUT"),
        ("russell_3000", "Russell", "Russell 3000", ".RUA"),
        ("sp_500", "S&P", "S&P 500", ".SPX"),
        ("nasdaq_100", "Nasdaq", "Nasdaq-100", ".NDX"),
        ("dow_jones_industrial_average", "Dow", "Dow Jones Industrial Average", ".DJI"),
    )
    return pd.DataFrame(
        [
            {
                "index_id": index_id,
                "index_family": family,
                "index_name": name,
                "candidate_index_ric": ric,
                "candidate_ric_status": (
                    "missing_constituent_entitlement"
                    if ric in {".RUI", ".RUT", ".RUA"}
                    else "validated_current_chain_and_joiner_leaver_route"
                ),
                "scope": (
                    "incomplete_missing_russell_constituent_entitlement"
                    if ric in {".RUI", ".RUT", ".RUA"}
                    else "official_current_chain_reverse_joiner_leaver_reconstruction"
                ),
            }
            for index_id, family, name, ric in rows
        ],
        columns=INDEX_CATALOG_COLUMNS,
    )


def resolve_usa_index_scope_profile(scope_profile: str = USA_INDEX_SCOPE_PROFILE_FULL) -> USAIndexScopeProfile:
    """Resolve an auditable USA index scope profile, accepting CLI-friendly aliases."""

    normalized = str(scope_profile).strip().lower().replace("-", "_")
    aliases = {
        "default": USA_INDEX_SCOPE_PROFILE_FULL,
        "full_catalog": USA_INDEX_SCOPE_PROFILE_FULL,
        "no_russell_backtest": USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == USA_INDEX_SCOPE_PROFILE_FULL:
        return USAIndexScopeProfile(
            name=USA_INDEX_SCOPE_PROFILE_FULL,
            required_index_ids=_FULL_INDEX_IDS,
            excluded_index_ids=(),
            excluded_index_reason=None,
        )
    if normalized == USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL:
        return USAIndexScopeProfile(
            name=USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
            required_index_ids=_BACKTEST_NO_RUSSELL_INDEX_IDS,
            excluded_index_ids=_RUSSELL_INDEX_IDS,
            excluded_index_reason=_BACKTEST_NO_RUSSELL_EXCLUSION_REASON,
        )
    raise ValueError(
        f"unknown USA index scope profile {scope_profile!r}; expected "
        f"{USA_INDEX_SCOPE_PROFILE_FULL!r} or {USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL!r}"
    )


def usa_index_catalog_for_scope_profile(
    scope_profile: str = USA_INDEX_SCOPE_PROFILE_FULL,
) -> pd.DataFrame:
    """Return the default catalog filtered to exactly the selected scope contract."""

    profile = resolve_usa_index_scope_profile(scope_profile)
    catalog = default_usa_index_catalog().set_index("index_id", drop=False)
    return catalog.loc[list(profile.required_index_ids)].reset_index(drop=True)


def pit_constituent_schema() -> pd.DataFrame:
    """Describe the normalized bitemporal constituent-row contract."""

    descriptions = {
        "index_id": ("string", "required", "Stable quant_lab index key from index_catalog.csv."),
        "index_family": ("string", "required", "Russell, S&P, Nasdaq, or Dow."),
        "index_name": ("string", "required", "Human-readable benchmark name."),
        "index_ric": ("string", "required", "Source-returned and probe-validated index RIC."),
        "member_ric": ("string", "identifier", "Source-returned constituent RIC, when available."),
        "member_perm_id": ("string", "identifier", "Permanent entity/security identifier, when available."),
        "member_isin": ("string", "identifier", "ISIN at the source observation time, when available."),
        "member_ticker": ("string", "identifier", "Ticker at the source observation time, when available."),
        "member_name": ("string", "optional", "Constituent display name returned by the source."),
        "membership_action": (
            "enum",
            "required",
            "snapshot_member, added, or removed; never inferred from the current universe.",
        ),
        "membership_as_of_date": (
            "date",
            "conditional",
            "Source snapshot date; required for snapshot_member rows.",
        ),
        "effective_date": ("date", "conditional", "Source effective date; required for added/removed rows."),
        "weight_percent": ("float64", "optional", "Source-returned index weight; no synthetic equal weights."),
        "known_at_utc": (
            "timestamp[UTC]",
            "required",
            "Source availability/announcement timestamp used by point-in-time joins.",
        ),
        "known_at_source_field": (
            "string",
            "required",
            "Exact source field carrying known_at_utc; load/run timestamps are rejected.",
        ),
        "source_provider": ("string", "required", "Vendor or official index publisher supplying the row."),
        "source_dataset": ("string", "required", "Vendor dataset, feed, or licensed history product."),
        "source_record_id": ("string", "required", "Stable source row/event identifier."),
        "source_request_id": (
            "string",
            "required",
            "Logical request_plan.csv id proving requested-window coverage.",
        ),
        "source_loaded_at_utc": ("timestamp[UTC]", "required", "UTC ingestion timestamp, separate from known_at."),
        "raw_payload_sha256": (
            "string",
            "required",
            "SHA-256 of the retained raw source payload; never generated from the normalized row.",
        ),
    }
    rows = []
    for ordinal, column in enumerate(PIT_CONSTITUENT_COLUMNS, start=1):
        dtype, requirement, description = descriptions[column]
        rows.append(
            {
                "ordinal": ordinal,
                "column": column,
                "dtype": dtype,
                "requirement": requirement,
                "required_for_manifest_completion": requirement in {"required", "identifier", "conditional"},
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def load_refinitiv_field_catalog(path: str | Path | None) -> pd.DataFrame:
    """Load an existing field catalog for annotations only; it never accepts PIT routes."""

    if path is None:
        return pd.DataFrame()
    catalog_path = Path(path).expanduser()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Refinitiv field catalog not found: {catalog_path}")
    catalog = pd.read_csv(catalog_path, low_memory=False)
    if "field" not in catalog.columns:
        raise ValueError("Refinitiv field catalog must contain a `field` column")
    return catalog


def build_refinitiv_index_field_probe_plan(field_catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Record the verified official route and the rejected dated-snapshot anchor."""

    candidates: tuple[tuple[str, str, bool, str, bool, str], ...] = (
        ("current_chain_member", "TR.PriceClose", True, "get_data_current_chain", True, "validated_live"),
        (
            "effective_date",
            "TR.IndexJLConstituentChangeDate",
            True,
            "get_data_joiner_leaver_ic_b",
            True,
            "validated_live",
        ),
        (
            "member_ric",
            "TR.IndexJLConstituentRIC",
            True,
            "get_data_joiner_leaver_ic_b",
            True,
            "validated_live",
        ),
        (
            "member_name",
            "TR.IndexJLConstituentName",
            True,
            "get_data_joiner_leaver_ic_b",
            True,
            "validated_live",
        ),
        (
            "membership_action",
            "TR.IndexJLConstituentituentChange",
            True,
            "get_data_joiner_leaver_ic_b",
            True,
            "validated_live",
        ),
        (
            "dated_snapshot_member",
            "TR.IndexConstituentRIC",
            False,
            "get_data_dated_snapshot_probe_only",
            False,
            "rejected_historical_anchor_modern_members_returned",
        ),
    )
    catalog = field_catalog.copy() if field_catalog is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for candidate_order, (role, field, required, route, accepted, status) in enumerate(candidates, start=1):
        matches = catalog[catalog["field"].astype(str).eq(field)].copy() if "field" in catalog else pd.DataFrame()
        data_families = _joined_unique(matches.get("data_family", pd.Series(dtype=object)))
        catalog_statuses = _joined_unique(matches.get("validation_status", pd.Series(dtype=object)))
        rows.append(
            {
                "probe_id": f"usa-index-constituents-{candidate_order:02d}",
                "dataset_id": DATASET_ID,
                "source_family": "refinitiv_index_constituent_history",
                "field_role": role,
                "candidate_field": field,
                "route": route,
                "required": required,
                "candidate_order": candidate_order,
                "candidate_source": "official_lseg_algorithm_and_verified_live_route_20260711",
                "catalog_match_count": int(len(matches)),
                "catalog_data_families": data_families,
                "catalog_validation_statuses": catalog_statuses,
                "catalog_evidence_only": bool(len(matches)),
                "validation_status": status,
                "live_request_executed": True,
                "accepted_for_pit": accepted,
                "validation_method": (
                    "Verified on .SPX/.NDX/.DJI; JL 2020 event counts were 34/25/6. Russell returned one blank row."
                ),
                "rejection_guard": (
                    "Never use a historical dated chain as the initial set unless its membership is independently "
                    "validated; use current-chain reverse JL reconstruction."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_usa_index_constituent_request_plan(
    start_date: str,
    end_date: str,
    index_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build logical annual history requests without selecting or calling a live route."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date and end_date must be valid dates")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    catalog = index_catalog.copy() if index_catalog is not None else default_usa_index_catalog()
    missing = set(INDEX_CATALOG_COLUMNS).difference(catalog.columns)
    if missing:
        raise ValueError(f"index catalog missing columns: {sorted(missing)}")

    required_probe_roles = "member_ric|membership_action|effective_date|known_at_utc"
    rows: list[dict[str, Any]] = []
    for item in catalog.itertuples(index=False):
        for chunk_start, chunk_end in _annual_chunks(start, end):
            request_id = f"usa-pit-index-{item.index_id}-{chunk_start.year}"
            rows.append(
                {
                    "source_request_id": request_id,
                    "dataset_id": DATASET_ID,
                    "index_id": item.index_id,
                    "index_family": item.index_family,
                    "index_name": item.index_name,
                    "candidate_index_ric": item.candidate_index_ric,
                    "candidate_ric_status": item.candidate_ric_status,
                    "request_start_date": chunk_start.date().isoformat(),
                    "request_end_date": chunk_end.date().isoformat(),
                    "logical_request_kind": "historical_constituent_snapshot_or_change_events",
                    "route": "unresolved_pending_live_probe",
                    "required_probe_roles": required_probe_roles,
                    "parameters_json": json.dumps(
                        {
                            "SDate": chunk_start.date().isoformat(),
                            "EDate": chunk_end.date().isoformat(),
                            "Frq": "D",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "request_status": "blocked_pending_accepted_pit_route",
                    "live_request_allowed": False,
                    "expected_output_schema": "constituent_schema.csv",
                    "completion_requirement": (
                        "At least one source-backed PIT row with known_at_utc and raw-payload provenance."
                    ),
                }
            )
    return pd.DataFrame(rows)


def read_pit_constituent_rows(path: str | Path) -> pd.DataFrame:
    """Read caller-supplied PIT rows; this function never creates or infers history."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"PIT constituent rows not found: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, low_memory=False)
    raise ValueError("PIT constituent rows must be supplied as CSV or Parquet")


def validate_pit_constituent_rows(
    pit_rows: pd.DataFrame | None,
    request_plan: pd.DataFrame,
    index_catalog: pd.DataFrame | None = None,
) -> PITConstituentValidation:
    """Validate source-backed rows without filling dates, identifiers, or provenance."""

    catalog = index_catalog.copy() if index_catalog is not None else default_usa_index_catalog()
    expected_requests = set(request_plan.get("source_request_id", pd.Series(dtype=object)).dropna().astype(str))
    target_by_id = catalog.set_index("index_id").to_dict("index")
    supplied = pit_rows.copy() if pit_rows is not None else pd.DataFrame()
    issues: list[dict[str, Any]] = []

    if supplied.empty:
        empty = pd.DataFrame(columns=(*PIT_CONSTITUENT_COLUMNS, *DERIVED_PIT_COLUMNS))
        issues.append(_issue(-1, "", "", "no_pit_rows_supplied", "", "no constituent history rows were supplied"))
        return PITConstituentValidation(
            rows=empty,
            issues=pd.DataFrame(issues, columns=ISSUE_COLUMNS),
            supplied_rows=0,
            usable_rows=0,
            covered_index_ids=(),
            covered_request_ids=(),
        )

    missing_columns = [column for column in PIT_CONSTITUENT_COLUMNS if column not in supplied.columns]
    for column in missing_columns:
        issues.append(_issue(-1, "", "", "missing_schema_column", column, f"required schema column `{column}` is absent"))
        supplied[column] = pd.NA

    rows = supplied[list(PIT_CONSTITUENT_COLUMNS)].copy().reset_index(drop=True)

    for column in _STRING_COLUMNS:
        rows[column] = rows[column].astype("string").str.strip()
        rows.loc[rows[column].eq(""), column] = pd.NA

    parsed_dates = {
        "membership_as_of_date": pd.to_datetime(rows["membership_as_of_date"], errors="coerce"),
        "effective_date": pd.to_datetime(rows["effective_date"], errors="coerce"),
        "known_at_utc": pd.to_datetime(rows["known_at_utc"], errors="coerce", utc=True),
        "source_loaded_at_utc": pd.to_datetime(rows["source_loaded_at_utc"], errors="coerce", utc=True),
    }
    rows["weight_percent"] = pd.to_numeric(rows["weight_percent"], errors="coerce")

    for row_number, row in rows.iterrows():
        row_id = _text(row.get("source_record_id"))
        index_id = _text(row.get("index_id"))

        target = target_by_id.get(index_id)
        if target is None:
            issues.append(_issue(row_number, row_id, index_id, "unknown_index_id", "index_id", "index is outside the target catalog"))
        else:
            expected_values = {
                "index_family": target["index_family"],
                "index_name": target["index_name"],
            }
            for column, expected in expected_values.items():
                if _text(row.get(column)) != str(expected):
                    issues.append(
                        _issue(
                            row_number,
                            row_id,
                            index_id,
                            "index_metadata_mismatch",
                            column,
                            f"expected `{expected}` from the target catalog",
                        )
                    )

        if not _text(row.get("index_ric")):
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "missing_source_index_ric",
                    "index_ric",
                    "a source-returned, route-validated index RIC is required",
                )
            )

        identifiers = [_text(row.get(column)) for column in ("member_ric", "member_perm_id", "member_isin", "member_ticker")]
        if not any(identifiers):
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "missing_member_identifier",
                    "member_ric",
                    "at least one member RIC, PermID, ISIN, or ticker is required",
                )
            )

        action = _text(row.get("membership_action"))
        if action not in MEMBERSHIP_ACTIONS:
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "invalid_membership_action",
                    "membership_action",
                    f"expected one of {list(MEMBERSHIP_ACTIONS)}",
                )
            )
        if action == "snapshot_member" and pd.isna(parsed_dates["membership_as_of_date"].loc[row_number]):
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "missing_membership_as_of_date",
                    "membership_as_of_date",
                    "snapshot_member rows require a source snapshot date",
                )
            )
        if action in {"added", "removed"} and pd.isna(parsed_dates["effective_date"].loc[row_number]):
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "missing_effective_date",
                    "effective_date",
                    "added/removed rows require a source effective date",
                )
            )

        for column in _PROVENANCE_COLUMNS:
            if not _text(row.get(column)):
                issues.append(
                    _issue(row_number, row_id, index_id, "missing_source_provenance", column, f"`{column}` is required")
                )

        known_at = parsed_dates["known_at_utc"].loc[row_number]
        loaded_at = parsed_dates["source_loaded_at_utc"].loc[row_number]
        if _text(row.get("known_at_utc")) and pd.isna(known_at):
            issues.append(_issue(row_number, row_id, index_id, "invalid_timestamp", "known_at_utc", "timestamp is not parseable"))
        if _text(row.get("source_loaded_at_utc")) and pd.isna(loaded_at):
            issues.append(
                _issue(row_number, row_id, index_id, "invalid_timestamp", "source_loaded_at_utc", "timestamp is not parseable")
            )
        if pd.notna(known_at) and pd.notna(loaded_at) and known_at > loaded_at:
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "known_at_after_load",
                    "known_at_utc",
                    "known_at_utc cannot be later than source_loaded_at_utc",
                )
            )

        known_at_field = _text(row.get("known_at_source_field")).lower()
        if known_at_field in _DISALLOWED_KNOWN_AT_FIELDS or "load" in known_at_field or "retriev" in known_at_field:
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "load_timestamp_used_as_known_at",
                    "known_at_source_field",
                    "known_at must come from a source availability/announcement field, not ingestion metadata",
                )
            )

        request_id = _text(row.get("source_request_id"))
        if request_id and request_id not in expected_requests:
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "unplanned_source_request",
                    "source_request_id",
                    "request id is not present in request_plan.csv",
                )
            )

        payload_hash = _text(row.get("raw_payload_sha256"))
        if payload_hash and not _HASH_PATTERN.fullmatch(payload_hash):
            issues.append(
                _issue(
                    row_number,
                    row_id,
                    index_id,
                    "invalid_raw_payload_hash",
                    "raw_payload_sha256",
                    "expected a 64-character hexadecimal SHA-256",
                )
            )

    duplicated = rows.duplicated(["source_provider", "source_dataset", "source_record_id"], keep=False)
    for row_number in rows.index[duplicated]:
        row = rows.loc[row_number]
        issues.append(
            _issue(
                row_number,
                _text(row.get("source_record_id")),
                _text(row.get("index_id")),
                "duplicate_source_record",
                "source_record_id",
                "source provider/dataset/record id must be unique",
            )
        )

    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    bad_rows = set(issue_frame.loc[issue_frame["row_number"].ge(0), "row_number"].astype(int))
    global_error = bool((issue_frame["row_number"] < 0).any())
    rows["pit_usable"] = [not global_error and row_number not in bad_rows for row_number in rows.index]
    reasons_by_row = (
        issue_frame[issue_frame["row_number"].ge(0)]
        .groupby("row_number")["issue_code"]
        .agg(lambda values: "|".join(dict.fromkeys(values.astype(str))))
        .to_dict()
    )
    rows["quarantine_reason"] = [reasons_by_row.get(row_number, "") for row_number in rows.index]

    usable = rows[rows["pit_usable"]]
    covered_indexes = tuple(sorted(usable["index_id"].dropna().astype(str).unique()))
    covered_requests = tuple(sorted(usable["source_request_id"].dropna().astype(str).unique()))
    return PITConstituentValidation(
        rows=rows.reset_index(drop=True),
        issues=issue_frame,
        supplied_rows=int(len(rows)),
        usable_rows=int(rows["pit_usable"].sum()),
        covered_index_ids=covered_indexes,
        covered_request_ids=covered_requests,
    )


def write_usa_index_constituent_plan_artifacts(
    artifact_root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    field_catalog_path: str | Path | None = None,
    pit_rows: pd.DataFrame | None = None,
    scope_profile: str = USA_INDEX_SCOPE_PROFILE_FULL,
) -> USAIndexConstituentPlanArtifacts:
    """Write an offline acquisition scaffold and validate only caller-supplied PIT rows."""

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    profile = resolve_usa_index_scope_profile(scope_profile)
    index_catalog = usa_index_catalog_for_scope_profile(profile.name)
    schema = pit_constituent_schema()
    field_catalog = load_refinitiv_field_catalog(field_catalog_path)
    field_probe_plan = build_refinitiv_index_field_probe_plan(field_catalog=field_catalog)
    request_plan = build_usa_index_constituent_request_plan(
        start_date=start_date,
        end_date=end_date,
        index_catalog=index_catalog,
    )
    validation = validate_pit_constituent_rows(
        pit_rows=pit_rows,
        request_plan=request_plan,
        index_catalog=index_catalog,
    )

    index_catalog_path = artifact_dir / "index_catalog.csv"
    schema_path = artifact_dir / "constituent_schema.csv"
    field_probe_plan_path = artifact_dir / "refinitiv_field_probe_plan.csv"
    request_plan_path = artifact_dir / "request_plan.csv"
    validation_issues_path = artifact_dir / "validation_issues.csv"
    pit_rows_path = artifact_dir / "pit_constituent_rows.parquet" if validation.supplied_rows else None
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    index_catalog.to_csv(index_catalog_path, index=False)
    schema.to_csv(schema_path, index=False)
    field_probe_plan.to_csv(field_probe_plan_path, index=False)
    request_plan.to_csv(request_plan_path, index=False)
    validation.issues.to_csv(validation_issues_path, index=False)
    if pit_rows_path is not None:
        validation.rows.to_parquet(pit_rows_path, index=False)

    expected_indexes = set(index_catalog["index_id"].astype(str))
    expected_requests = set(request_plan["source_request_id"].astype(str))
    catalog_index_ids = index_catalog["index_id"].astype(str).tolist()
    scope_match = bool(
        len(catalog_index_ids) == len(profile.required_index_ids)
        and set(catalog_index_ids) == set(profile.required_index_ids)
    )
    covered_indexes = set(validation.covered_index_ids)
    covered_requests = set(validation.covered_request_ids)
    missing_indexes = sorted(expected_indexes.difference(covered_indexes))
    missing_requests = sorted(expected_requests.difference(covered_requests))
    all_rows_usable = validation.supplied_rows > 0 and validation.usable_rows == validation.supplied_rows
    complete = bool(scope_match and all_rows_usable and not missing_indexes and not missing_requests)

    blockers = []
    if validation.supplied_rows == 0:
        blockers.append("no actual PIT constituent rows supplied")
    if validation.usable_rows < validation.supplied_rows:
        blockers.append("one or more supplied rows failed schema/provenance/leakage validation")
    if missing_indexes:
        blockers.append("target indexes missing source-backed PIT rows")
    if missing_requests:
        blockers.append("logical request windows missing source-backed PIT rows")
    if not scope_match:
        blockers.append("selected index catalog does not exactly match the required scope profile")

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "dataset_id": DATASET_ID,
        "scope": "USA Russell/S&P/Nasdaq/Dow point-in-time index constituents",
        "scope_profile": profile.name,
        "required_index_ids": list(profile.required_index_ids),
        "excluded_index_ids": list(profile.excluded_index_ids),
        "excluded_index_reason": profile.excluded_index_reason,
        "scope_match": scope_match,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "status": "complete" if complete else "incomplete",
        "scorecard_eligible": complete,
        "scaffold_only": validation.supplied_rows == 0,
        "live_refinitiv_requests_executed": False,
        "constituent_history_fabricated": False,
        "candidate_fields_accepted": int(field_probe_plan["accepted_for_pit"].sum()),
        "target_index_families": list(INDEX_FAMILIES),
        "target_index_ids": sorted(expected_indexes),
        "covered_index_ids": sorted(covered_indexes),
        "missing_index_ids": missing_indexes,
        "planned_requests": int(len(request_plan)),
        "covered_requests": int(len(covered_requests.intersection(expected_requests))),
        "missing_request_count": int(len(missing_requests)),
        "missing_request_ids": missing_requests,
        "supplied_pit_rows": validation.supplied_rows,
        "pit_usable_rows": validation.usable_rows,
        "rows_with_known_at_utc": int(pd.to_datetime(validation.rows.get("known_at_utc"), errors="coerce", utc=True).notna().sum())
        if not validation.rows.empty
        else 0,
        "validation_issue_count": int(len(validation.issues)),
        "completion_gates": {
            "actual_rows_supplied": validation.supplied_rows > 0,
            "all_rows_have_valid_known_at_and_provenance": all_rows_usable,
            "all_target_indexes_covered": not missing_indexes,
            "all_request_windows_covered": not missing_requests,
            "catalog_exactly_matches_scope_profile": scope_match,
        },
        "blockers": blockers,
        "refinitiv_field_catalog_path": str(Path(field_catalog_path).expanduser().resolve()) if field_catalog_path else None,
        "refinitiv_field_catalog_sha256": _file_sha256(Path(field_catalog_path).expanduser()) if field_catalog_path else None,
        "generated_at_utc": _now_utc(),
        "artifacts": {
            "index_catalog": str(index_catalog_path.resolve()),
            "constituent_schema": str(schema_path.resolve()),
            "refinitiv_field_probe_plan": str(field_probe_plan_path.resolve()),
            "request_plan": str(request_plan_path.resolve()),
            "validation_issues": str(validation_issues_path.resolve()),
            "pit_constituent_rows": str(pit_rows_path.resolve()) if pit_rows_path else None,
            "summary": str(summary_path.resolve()),
        },
        "warning": (
            "This manifest is not scorecard evidence while status is incomplete. Current constituents, candidate fields, "
            "and plan rows are not constituent history."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_md(manifest, field_probe_plan, request_plan), encoding="utf-8")

    return USAIndexConstituentPlanArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        index_catalog_path=index_catalog_path,
        schema_path=schema_path,
        field_probe_plan_path=field_probe_plan_path,
        request_plan_path=request_plan_path,
        validation_issues_path=validation_issues_path,
        pit_rows_path=pit_rows_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        index_catalog=index_catalog,
        field_probe_plan=field_probe_plan,
        request_plan=request_plan,
        validation=validation,
    )


def _annual_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((chunk_start, chunk_end))
    return chunks


def _issue(
    row_number: int,
    source_record_id: str,
    index_id: str,
    issue_code: str,
    column: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "row_number": row_number,
        "source_record_id": source_record_id,
        "index_id": index_id,
        "severity": "error",
        "issue_code": issue_code,
        "column": column,
        "detail": detail,
    }


def _joined_unique(values: pd.Series) -> str:
    return "|".join(dict.fromkeys(values.dropna().astype(str)))


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_md(manifest: dict[str, Any], field_probe_plan: pd.DataFrame, request_plan: pd.DataFrame) -> str:
    blockers = manifest["blockers"] or ["none"]
    lines = [
        f"# {manifest['run_id']}",
        "",
        "## Status",
        "",
        f"- Manifest status: `{manifest['status']}`",
        f"- Scorecard eligible: `{manifest['scorecard_eligible']}`",
        f"- Supplied PIT rows: `{manifest['supplied_pit_rows']}`",
        f"- PIT-usable rows: `{manifest['pit_usable_rows']}`",
        f"- Logical requests: `{len(request_plan)}`",
        f"- Official Refinitiv fields: `{len(field_probe_plan)}` "
        f"(accepted: `{int(field_probe_plan['accepted_for_pit'].sum())}`)",
        "- Live Refinitiv requests executed: `False`",
        "",
        "## Blockers",
        "",
        *[f"- {blocker}" for blocker in blockers],
        "",
        "## Guardrail",
        "",
        "The dated historical chain is not an accepted anchor because a live historical-date probe returned modern members. ",
        "Use retained current-chain evidence plus complete reverse Joiner/Leaver reconstruction; Russell remains incomplete under the current entitlement.",
        "",
    ]
    return "\n".join(lines)
