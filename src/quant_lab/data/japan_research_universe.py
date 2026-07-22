from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


JAPAN_COUNTRY_CODES = frozenset({"JP", "JPN"})
JAPAN_ORDINARY_SHARE_CATEGORIES = frozenset(
    {"Ordinary Share", "Fully Paid Ordinary Share"}
)
JPX_PRIMARY_EXCHANGE_CODES = frozenset({"TYO"})
JPX_PRIMARY_ORDINARY_SHARE_MARKETS = frozenset(
    {
        "prime_domestic_ordinary_shares",
        "standard_domestic_ordinary_shares",
        "growth_domestic_ordinary_shares",
    }
)

_JPX_MARKET_ALIASES = {
    "プライム（内国株式）": "prime_domestic_ordinary_shares",
    "スタンダード（内国株式）": "standard_domestic_ordinary_shares",
    "グロース（内国株式）": "growth_domestic_ordinary_shares",
    "prime market (domestic stocks)": "prime_domestic_ordinary_shares",
    "prime (domestic stocks)": "prime_domestic_ordinary_shares",
    "prime_domestic_ordinary_shares": "prime_domestic_ordinary_shares",
    "standard market (domestic stocks)": "standard_domestic_ordinary_shares",
    "standard (domestic stocks)": "standard_domestic_ordinary_shares",
    "standard_domestic_ordinary_shares": "standard_domestic_ordinary_shares",
    "growth market (domestic stocks)": "growth_domestic_ordinary_shares",
    "growth (domestic stocks)": "growth_domestic_ordinary_shares",
    "growth_domestic_ordinary_shares": "growth_domestic_ordinary_shares",
}

_APPROVED_KNOWN_AT_COLUMNS = (
    "known_at_utc",
    "public_listing_known_at_utc",
    "source_published_at_utc",
    "publication_timestamp_utc",
    "published_at_utc",
)

_RECONCILIATION_COLUMNS = (
    "source_side",
    "jpx_source_row",
    "jpx_asof_date",
    "jpx_known_at_utc",
    "local_code",
    "jpx_name",
    "jpx_market_category_raw",
    "jpx_market_scope",
    "in_primary_ordinary_share_scope",
    "stable_instrument_id",
    "stable_identifier_type",
    "ric",
    "price_currency",
    "reconciliation_status",
    "unmatched_classification",
)


@dataclass(frozen=True)
class JapanResearchUniverseArtifacts:
    run_id: str
    artifact_dir: Path
    active_symbol_map_path: Path
    reference_symbol_map_path: Path
    stable_instruments_path: Path
    reconciliation_path: Path
    unmatched_path: Path
    quality_metrics_path: Path
    scope_declaration_path: Path
    manifest_path: Path
    manifest: dict[str, object]

    @property
    def active_operating_core_path(self) -> Path:
        return self.active_symbol_map_path

    @property
    def historical_listed_reference_path(self) -> Path:
        return self.reference_symbol_map_path


def load_japan_security_master(path: str | Path) -> pd.DataFrame:
    return _read_table(path)


def load_jpx_listings(path: str | Path) -> pd.DataFrame:
    return normalize_jpx_listings(_read_table(path))


def jpx_primary_ordinary_share_scope() -> pd.DataFrame:
    """Predeclared current JPX equity scope; no market is inferred from vendor data."""

    rows = (
        (
            "prime_domestic_ordinary_shares",
            "プライム（内国株式）",
            "Prime Market domestic ordinary shares",
        ),
        (
            "standard_domestic_ordinary_shares",
            "スタンダード（内国株式）",
            "Standard Market domestic ordinary shares",
        ),
        (
            "growth_domestic_ordinary_shares",
            "グロース（内国株式）",
            "Growth Market domestic ordinary shares",
        ),
    )
    return pd.DataFrame(
        [
            {
                "scope_id": scope_id,
                "jpx_market_category": source_label,
                "description": description,
                "included": True,
                "security_type": "domestic_ordinary_share",
                "primary_exchange_code": "TYO",
                "expected_quote_currency": "JPY",
                "scope_basis": "current_public_jpx_listing_file",
            }
            for scope_id, source_label, description in rows
        ]
    )


def normalize_jpx_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public JPX workbook or an equivalent English-column export."""

    data = listings.copy()
    # Combined exchange-mining exports contain overlapping local ticker codes
    # across countries. Restrict those exports before duplicate classification;
    # native JPX workbooks do not carry these English scope columns.
    if "country" in data.columns:
        data = data.loc[data["country"].astype(str).str.casefold().eq("japan")].copy()
    if "exchange" in data.columns:
        exchange = data["exchange"].astype(str).str.upper()
        data = data.loc[exchange.isin({"JPX", "TSE", "TYO"})].copy()
    code = _coalesce(data, ["コード", "code", "local_code", "ticker", "ticker_symbol"])
    market_raw = _coalesce(
        data,
        ["市場・商品区分", "market_product_category", "market_category", "market_segment"],
    )
    asof_raw = _coalesce(data, ["日付", "asof_date", "jpx_asof_date", "date"])
    known_at_raw = _coalesce(data, list(_APPROVED_KNOWN_AT_COLUMNS))
    market_scope = market_raw.map(_canonical_jpx_market)
    normalized = pd.DataFrame(
        {
            "jpx_source_row": pd.RangeIndex(start=1, stop=len(data) + 1),
            "jpx_asof_date": asof_raw.map(_normalize_date),
            "jpx_known_at_utc": known_at_raw,
            "local_code": code.map(_normalize_local_code),
            "jpx_name": _coalesce(data, ["銘柄名", "name", "security_name", "company_name"]),
            "jpx_market_category_raw": market_raw,
            "jpx_market_scope": market_scope,
            "in_primary_ordinary_share_scope": market_scope.isin(
                JPX_PRIMARY_ORDINARY_SHARE_MARKETS
            ),
            "expected_quote_currency": "JPY",
        }
    )
    return normalized.reset_index(drop=True)


def deduplicate_japan_quote_rows(security_master: pd.DataFrame) -> pd.DataFrame:
    """Collapse vendor quote routes to one stable Japan instrument record."""

    rows = _normalize_japan_security_master(security_master)
    if rows.empty:
        rows["quote_row_count"] = pd.Series(dtype="int64")
        rows["jpx_quote_row_count"] = pd.Series(dtype="int64")
        rows["has_direct_jpx_listing_evidence"] = pd.Series(dtype="bool")
        rows["jpx_first_trade_date"] = pd.Series(dtype="object")
        rows["jpx_last_retire_date"] = pd.Series(dtype="object")
        return rows

    rows["quote_row_count"] = rows.groupby("stable_instrument_id")["ric"].transform(
        "size"
    )
    rows["jpx_quote_row_count"] = (
        rows["is_direct_jpx_quote"]
        .astype(int)
        .groupby(rows["stable_instrument_id"])
        .transform("sum")
    )
    rows["has_direct_jpx_listing_evidence"] = rows["jpx_quote_row_count"].gt(0)

    direct_dates = rows["first_trade_date"].where(rows["is_direct_jpx_quote"])
    retire_dates = rows["retire_date"].where(rows["is_direct_jpx_quote"])
    rows["jpx_first_trade_date"] = direct_dates.groupby(
        rows["stable_instrument_id"]
    ).transform("min")
    rows["jpx_last_retire_date"] = retire_dates.groupby(
        rows["stable_instrument_id"]
    ).transform("max")

    rows["_standard_tse_ric"] = (
        rows["ric"].astype("string").str.match(r"^[^.\^]+\.T$", na=False)
    )
    rows["_jpy_quote"] = rows["price_currency"].eq("JPY")
    rows["_source_order"] = pd.RangeIndex(len(rows))
    ranked = rows.sort_values(
        [
            "stable_instrument_id",
            "is_direct_jpx_quote",
            "is_active",
            "is_primary_ric",
            "is_primary_issue_ric",
            "_standard_tse_ric",
            "_jpy_quote",
            "retire_date",
            "_source_order",
        ],
        ascending=[True, False, False, False, False, False, False, False, True],
        na_position="last",
        kind="stable",
    )
    stable = ranked.drop_duplicates("stable_instrument_id", keep="first").copy()
    stable = stable.drop(columns=["_standard_tse_ric", "_jpy_quote", "_source_order"])
    return stable.sort_values(
        ["has_direct_jpx_listing_evidence", "is_active", "local_code", "ric"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def reconcile_japan_research_universe(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile current public JPX rows to active primary TSE ordinary shares."""

    listings = _ensure_normalized_jpx_listings(jpx_listings)
    stable = deduplicate_japan_quote_rows(security_master)
    candidates = stable.loc[
        stable.get("has_direct_jpx_listing_evidence", False)
        & stable.get("is_active", False)
        & stable.get("is_primary_ric", False)
        & stable.get("is_primary_issue_ric", False)
    ].copy()
    candidate_groups = {
        str(code): group.drop_duplicates("stable_instrument_id")
        for code, group in candidates.loc[candidates["local_code"].notna()].groupby(
            "local_code", sort=False
        )
    }
    public_code_counts = listings.loc[
        listings["in_primary_ordinary_share_scope"], "local_code"
    ].value_counts(dropna=False)
    records: list[dict[str, object]] = []
    matched_ids: set[str] = set()

    for row in listings.to_dict("records"):
        local_code = row.get("local_code")
        base = _public_reconciliation_record(row)
        if not bool(row.get("in_primary_ordinary_share_scope")):
            records.append(
                {
                    **base,
                    "reconciliation_status": "public_out_of_scope",
                    "unmatched_classification": "out_of_scope_market_category",
                }
            )
            continue
        if not local_code:
            records.append(
                {
                    **base,
                    "reconciliation_status": "public_in_scope_unmatched",
                    "unmatched_classification": "missing_public_local_code",
                }
            )
            continue
        if int(public_code_counts.get(local_code, 0)) > 1:
            records.append(
                {
                    **base,
                    "reconciliation_status": "public_in_scope_ambiguous",
                    "unmatched_classification": "duplicate_public_listing_code",
                }
            )
            continue
        matches = candidate_groups.get(str(local_code), pd.DataFrame())
        if matches.empty:
            records.append(
                {
                    **base,
                    "reconciliation_status": "public_in_scope_unmatched",
                    "unmatched_classification": "missing_active_primary_security_master_instrument",
                }
            )
            continue
        if len(matches) != 1:
            records.append(
                {
                    **base,
                    "reconciliation_status": "public_in_scope_ambiguous",
                    "unmatched_classification": "ambiguous_active_primary_security_master_match",
                }
            )
            continue
        match = matches.iloc[0]
        stable_id = str(match["stable_instrument_id"])
        matched_ids.add(stable_id)
        records.append(
            {
                **base,
                "stable_instrument_id": stable_id,
                "stable_identifier_type": match.get("stable_identifier_type"),
                "ric": match.get("ric"),
                "price_currency": match.get("price_currency"),
                "reconciliation_status": "matched",
                "unmatched_classification": "",
            }
        )

    for row in candidates.loc[
        ~candidates["stable_instrument_id"].astype(str).isin(matched_ids)
    ].to_dict("records"):
        records.append(
            {
                "source_side": "security_master",
                "jpx_source_row": pd.NA,
                "jpx_asof_date": pd.NA,
                "jpx_known_at_utc": pd.NA,
                "local_code": row.get("local_code"),
                "jpx_name": pd.NA,
                "jpx_market_category_raw": pd.NA,
                "jpx_market_scope": pd.NA,
                "in_primary_ordinary_share_scope": True,
                "stable_instrument_id": row.get("stable_instrument_id"),
                "stable_identifier_type": row.get("stable_identifier_type"),
                "ric": row.get("ric"),
                "price_currency": row.get("price_currency"),
                "reconciliation_status": "master_active_unmatched",
                "unmatched_classification": "missing_current_public_jpx_listing",
            }
        )

    return pd.DataFrame(records, columns=_RECONCILIATION_COLUMNS)


def build_japan_active_operating_core(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    stable = deduplicate_japan_quote_rows(security_master)
    reconciliation = reconcile_japan_research_universe(security_master, jpx_listings)
    matched = reconciliation.loc[
        reconciliation["reconciliation_status"].eq("matched")
    ].drop_duplicates("stable_instrument_id")
    if matched.empty:
        return _empty_universe_frame(stable)
    selected = stable.merge(
        matched[
            [
                "stable_instrument_id",
                "jpx_asof_date",
                "jpx_known_at_utc",
                "jpx_name",
                "jpx_market_category_raw",
                "jpx_market_scope",
            ]
        ],
        on="stable_instrument_id",
        how="inner",
        validate="one_to_one",
    )
    selected["current_jpx_scope_member"] = True
    selected["research_universe_role"] = "active_jpx_operating_core"
    selected["research_universe_reason"] = (
        "active primary TSE ordinary share reconciled to current public JPX "
        "Prime/Standard/Growth domestic listing"
    )
    selected["point_in_time_membership_claimed"] = False
    return _as_symbol_map(selected, "JPN_JPX_ACTIVE_OPERATING_CORE_V1")


def build_japan_historical_listed_reference(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    """Listed-name denominator candidate retaining inactive direct TSE instruments.

    First/retire dates and current JPX reconciliation are source evidence, not a
    reconstruction of historical JPX market-segment membership.
    """

    stable = deduplicate_japan_quote_rows(security_master)
    reference = stable.loc[stable["has_direct_jpx_listing_evidence"]].copy()
    reconciliation = reconcile_japan_research_universe(security_master, jpx_listings)
    matched = reconciliation.loc[
        reconciliation["reconciliation_status"].eq("matched")
    ].drop_duplicates("stable_instrument_id")
    current_fields = matched.set_index("stable_instrument_id")[
        ["jpx_asof_date", "jpx_known_at_utc", "jpx_name", "jpx_market_scope"]
    ]
    for column in current_fields.columns:
        reference[column] = reference["stable_instrument_id"].map(current_fields[column])
    reference["current_jpx_scope_member"] = reference["stable_instrument_id"].isin(
        set(matched["stable_instrument_id"].dropna().astype(str))
    )
    reference["research_universe_role"] = "historical_jpx_listed_reference_denominator"
    reference["research_universe_reason"] = (
        "active and inactive ordinary-share instruments with direct Tokyo Stock "
        "Exchange quote evidence, deduplicated by stable identifier"
    )
    reference["point_in_time_membership_claimed"] = False
    return _as_symbol_map(reference, "JPN_JPX_HISTORICAL_LISTED_REFERENCE_V1")


def build_japan_listed_reference_core(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    return build_japan_historical_listed_reference(security_master, jpx_listings)


def build_japan_active_core(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    return build_japan_active_operating_core(security_master, jpx_listings)


def japan_research_universe_quality_metrics(
    security_master: pd.DataFrame,
    jpx_listings: pd.DataFrame,
) -> pd.DataFrame:
    listings = _ensure_normalized_jpx_listings(jpx_listings)
    stable = deduplicate_japan_quote_rows(security_master)
    reconciliation = reconcile_japan_research_universe(security_master, listings)
    active = build_japan_active_operating_core(security_master, listings)
    reference = build_japan_historical_listed_reference(security_master, listings)

    public = reconciliation["source_side"].eq("public_jpx")
    in_scope = public & reconciliation["in_primary_ordinary_share_scope"].fillna(False)
    matched = in_scope & reconciliation["reconciliation_status"].eq("matched")
    active_candidates = stable.loc[
        stable.get("has_direct_jpx_listing_evidence", False)
        & stable.get("is_active", False)
        & stable.get("is_primary_ric", False)
        & stable.get("is_primary_issue_ric", False)
    ]
    matched_ids = set(reconciliation.loc[matched, "stable_instrument_id"].dropna().astype(str))

    in_scope_count = int(in_scope.sum())
    active_candidate_count = int(len(active_candidates))
    active_count = int(len(active))
    reference_count = int(len(reference))
    currency = active.get("currency", pd.Series(dtype="object")).astype("string").str.upper()
    listing_known_at = pd.to_datetime(
        reconciliation.loc[in_scope, "jpx_known_at_utc"], errors="coerce", utc=True
    )
    reference_known_at = pd.to_datetime(
        reference.get("security_master_known_at_utc"), errors="coerce", utc=True
    )
    strong_ids = reference.get(
        "stable_identifier_is_strong", pd.Series(False, index=reference.index)
    ).fillna(False)
    public_codes = listings.loc[listings["in_primary_ordinary_share_scope"], "local_code"]

    values = {
        "public_in_scope_listing_match_rate": _ratio(int(matched.sum()), in_scope_count),
        "active_master_public_listing_coverage_rate": _ratio(
            len(matched_ids), active_candidate_count
        ),
        "active_core_currency_present_rate": _ratio(int(currency.notna().sum()), active_count),
        "active_core_jpy_currency_rate": _ratio(int(currency.eq("JPY").sum()), active_count),
        "reference_strong_stable_identifier_rate": _ratio(
            int(strong_ids.sum()), reference_count
        ),
        "public_listing_known_at_rate": _ratio(
            int(listing_known_at.notna().sum()), in_scope_count
        ),
        "reference_security_master_known_at_rate": _ratio(
            int(reference_known_at.notna().sum()), reference_count
        ),
        "public_in_scope_code_unique_rate": _ratio(
            int(public_codes.dropna().nunique()), len(public_codes)
        ),
        "public_in_scope_ambiguous_rows": float(
            (
                in_scope
                & reconciliation["reconciliation_status"].eq(
                    "public_in_scope_ambiguous"
                )
            ).sum()
        ),
        "public_in_scope_unmatched_rows": float(
            (
                in_scope
                & reconciliation["reconciliation_status"].eq(
                    "public_in_scope_unmatched"
                )
            ).sum()
        ),
        "master_active_unmatched_rows": float(
            reconciliation["reconciliation_status"].eq("master_active_unmatched").sum()
        ),
    }
    gates = (
        ("public_in_scope_listing_match_rate", 1.0, "ge"),
        ("active_master_public_listing_coverage_rate", 1.0, "ge"),
        ("active_core_currency_present_rate", 1.0, "ge"),
        ("active_core_jpy_currency_rate", 1.0, "ge"),
        ("reference_strong_stable_identifier_rate", 1.0, "ge"),
        ("public_listing_known_at_rate", 1.0, "ge"),
        ("reference_security_master_known_at_rate", 1.0, "ge"),
        ("public_in_scope_code_unique_rate", 1.0, "ge"),
        ("public_in_scope_ambiguous_rows", 0.0, "le"),
        ("public_in_scope_unmatched_rows", 0.0, "le"),
        ("master_active_unmatched_rows", 0.0, "le"),
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "value": float(values[metric]),
                "threshold": threshold,
                "comparison": comparison,
                "strict_gate": True,
                "passed": (
                    values[metric] >= threshold
                    if comparison == "ge"
                    else values[metric] <= threshold
                ),
            }
            for metric, threshold, comparison in gates
        ]
    )


def write_japan_research_universe_artifacts(
    security_master_path: str | Path,
    jpx_listings_path: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts/research_universes",
) -> JapanResearchUniverseArtifacts:
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    master_source = Path(security_master_path)
    listings_source = Path(jpx_listings_path)
    input_sha256 = {
        "merged_japan_security_master": _sha256(master_source),
        "current_public_jpx_listings": _sha256(listings_source),
    }
    input_sizes = {
        "merged_japan_security_master": master_source.stat().st_size,
        "current_public_jpx_listings": listings_source.stat().st_size,
    }

    security_master = load_japan_security_master(master_source)
    listings = load_jpx_listings(listings_source)
    stable = deduplicate_japan_quote_rows(security_master)
    active = build_japan_active_operating_core(security_master, listings)
    reference = build_japan_historical_listed_reference(security_master, listings)
    reconciliation = reconcile_japan_research_universe(security_master, listings)
    unmatched = reconciliation.loc[
        ~reconciliation["reconciliation_status"].eq("matched")
    ].copy()
    quality = japan_research_universe_quality_metrics(security_master, listings)
    scope = jpx_primary_ordinary_share_scope()

    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    active_path = artifact_dir / "japan_jpx_active_operating_core_v1.csv"
    reference_path = artifact_dir / "japan_jpx_historical_listed_reference_v1.csv"
    stable_path = artifact_dir / "japan_stable_instruments.csv"
    reconciliation_path = artifact_dir / "jpx_security_master_reconciliation.csv"
    unmatched_path = artifact_dir / "jpx_security_master_unmatched.csv"
    quality_path = artifact_dir / "quality_metrics.csv"
    scope_path = artifact_dir / "jpx_primary_ordinary_share_scope.csv"
    manifest_path = artifact_dir / "manifest.json"

    for frame, path in (
        (active, active_path),
        (reference, reference_path),
        (stable, stable_path),
        (reconciliation, reconciliation_path),
        (unmatched, unmatched_path),
        (quality, quality_path),
        (scope, scope_path),
    ):
        _atomic_write_csv(frame, path)

    verified_input_sha256 = {
        "merged_japan_security_master": _sha256(master_source),
        "current_public_jpx_listings": _sha256(listings_source),
    }
    if verified_input_sha256 != input_sha256:
        raise RuntimeError("an input changed while the Japan universe was being built")

    strict_quality_pass = bool(
        not quality.empty
        and quality.loc[quality["strict_gate"], "passed"].astype(bool).all()
    )
    output_paths = {
        "active_operating_core": active_path,
        "historical_listed_reference": reference_path,
        "stable_instruments": stable_path,
        "reconciliation": reconciliation_path,
        "unmatched": unmatched_path,
        "quality_metrics": quality_path,
        "scope_declaration": scope_path,
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "complete" if strict_quality_pass else "quality_blocked",
        "scorecard_scope": "current_jpx_universe_construction_quality",
        "scorecard_eligible": strict_quality_pass,
        "strict_quality_pass": strict_quality_pass,
        "historical_pit_ready": False,
        "historical_backtest_eligible": False,
        "pit_backtest_scorecard_eligible": False,
        "does_not_assert_point_in_time_history": True,
        "pit_warning": (
            "Current JPX listings and security-master first/retire dates do not provide "
            "point-in-time historical membership or source availability history."
        ),
        "universe_ids": {
            "active": "JPN_JPX_ACTIVE_OPERATING_CORE_V1",
            "reference": "JPN_JPX_HISTORICAL_LISTED_REFERENCE_V1",
        },
        "scope": {
            "country": "Japan",
            "country_codes": sorted(JAPAN_COUNTRY_CODES),
            "primary_exchange_codes": sorted(JPX_PRIMARY_EXCHANGE_CODES),
            "ordinary_share_categories": sorted(JAPAN_ORDINARY_SHARE_CATEGORIES),
            "current_public_markets": sorted(JPX_PRIMARY_ORDINARY_SHARE_MARKETS),
            "excluded_examples": [
                "ETF/ETN",
                "REIT and investment funds",
                "PRO Market",
                "foreign shares",
                "preferred and subscription securities",
            ],
        },
        "inputs": {
            "merged_japan_security_master": {
                "path": str(master_source.resolve()),
                "sha256": input_sha256["merged_japan_security_master"],
                "bytes": input_sizes["merged_japan_security_master"],
            },
            "current_public_jpx_listings": {
                "path": str(listings_source.resolve()),
                "sha256": input_sha256["current_public_jpx_listings"],
                "bytes": input_sizes["current_public_jpx_listings"],
            },
        },
        "input_sha256": input_sha256,
        "input_checksums_verified_after_build": True,
        "counts": {
            "security_master_quote_rows": int(len(security_master)),
            "japan_ordinary_share_quote_rows": int(
                stable["quote_row_count"].sum() if not stable.empty else 0
            ),
            "stable_instruments": int(len(stable)),
            "public_jpx_rows": int(len(listings)),
            "public_jpx_in_scope_rows": int(
                listings["in_primary_ordinary_share_scope"].sum()
            ),
            "active_operating_core_rows": int(len(active)),
            "historical_listed_reference_rows": int(len(reference)),
            "unmatched_rows": int(len(unmatched)),
        },
        "reconciliation_status_counts": {
            str(key): int(value)
            for key, value in reconciliation["reconciliation_status"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "unmatched_classification_counts": {
            str(key): int(value)
            for key, value in unmatched["unmatched_classification"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "quality_metrics": {
            str(row.metric): {
                "value": float(row.value),
                "threshold": float(row.threshold),
                "comparison": str(row.comparison),
                "passed": bool(row.passed),
            }
            for row in quality.itertuples(index=False)
        },
        "artifacts": {key: str(path.resolve()) for key, path in output_paths.items()},
        "artifact_sha256": {key: _sha256(path) for key, path in output_paths.items()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return JapanResearchUniverseArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        active_symbol_map_path=active_path,
        reference_symbol_map_path=reference_path,
        stable_instruments_path=stable_path,
        reconciliation_path=reconciliation_path,
        unmatched_path=unmatched_path,
        quality_metrics_path=quality_path,
        scope_declaration_path=scope_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _normalize_japan_security_master(security_master: pd.DataFrame) -> pd.DataFrame:
    if "ric" not in security_master.columns:
        raise ValueError("security master must include a ric column")
    data = security_master.copy()
    country_code = _coalesce(data, ["country_code", "exchange_country"])
    country_name = _coalesce(data, ["country"])
    asset_category = _coalesce(
        data, ["rcs_asset_category_leaf", "asset_category", "security_type"]
    )
    is_japan = country_code.astype("string").str.strip().str.upper().isin(
        JAPAN_COUNTRY_CODES
    ) | country_name.astype("string").str.strip().str.casefold().eq("japan")
    is_ordinary = asset_category.astype("string").str.strip().isin(
        JAPAN_ORDINARY_SHARE_CATEGORIES
    )
    data = data.loc[is_japan & is_ordinary].copy()
    ric = data["ric"].astype("string").str.strip()
    data = data.loc[ric.notna() & ric.ne("")].copy()
    data["ric"] = data["ric"].astype("string").str.strip()
    data["local_code"] = _coalesce(
        data, ["ticker_symbol", "local_code", "ticker"]
    ).map(_normalize_local_code)
    missing_code = data["local_code"].isna()
    data.loc[missing_code, "local_code"] = data.loc[missing_code, "ric"].map(
        _local_code_from_ric
    )
    data["exchange_code"] = _coalesce(data, ["exchange_code"]).map(
        lambda value: _text(value).upper() or pd.NA
    )
    data["is_direct_jpx_quote"] = data["exchange_code"].isin(
        JPX_PRIMARY_EXCHANGE_CODES
    )
    active_bucket = _coalesce(data, ["active_state_bucket"]).astype("string")
    active_fallback = _bool_series(data, "active_flag")
    data["is_active"] = active_bucket.str.strip().str.casefold().eq("active").where(
        active_bucket.notna(), active_fallback
    )
    data["is_primary_ric"] = _bool_series(data, "is_primary_ric")
    data["is_primary_issue_ric"] = _bool_series(data, "is_primary_issue_ric")
    data["price_currency"] = _coalesce(
        data, ["price_currency", "currency", "quote_currency"]
    ).map(_normalize_currency)
    data["security_master_known_at_utc"] = _coalesce(
        data, list(_APPROVED_KNOWN_AT_COLUMNS)
    )
    data["first_trade_date"] = pd.to_datetime(
        _coalesce(data, ["first_trade_date", "listing_date"]), errors="coerce"
    ).dt.date.astype("string")
    data["retire_date"] = pd.to_datetime(
        _coalesce(data, ["retire_date", "delisting_date"]), errors="coerce"
    ).dt.date.astype("string")

    if data.empty:
        data["stable_instrument_id"] = pd.Series(dtype="object")
        data["stable_identifier_type"] = pd.Series(dtype="object")
    else:
        ids = data.apply(_stable_identifier, axis=1, result_type="expand")
        ids.columns = ["stable_instrument_id", "stable_identifier_type"]
        data[["stable_instrument_id", "stable_identifier_type"]] = ids
    data["stable_identifier_is_strong"] = data["stable_identifier_type"].isin(
        {"isin", "instrument_permid"}
    )
    return data.reset_index(drop=True)


def _as_symbol_map(data: pd.DataFrame, universe_id: str) -> pd.DataFrame:
    if data.empty:
        columns = list(data.columns)
        for column in (
            "country",
            "country_code",
            "exchange",
            "ticker",
            "name",
            "currency",
            "refinitiv_ric",
            "refinitiv_symbol_status",
            "research_universe_id",
        ):
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)
    rows = data.copy()
    rows["country"] = "Japan"
    rows["country_code"] = "JPN"
    rows["exchange"] = "TYO"
    rows["ticker"] = rows["local_code"]
    rows["name"] = _coalesce(rows, ["common_name", "jpx_name", "document_title"])
    rows["currency"] = rows["price_currency"]
    rows["refinitiv_ric"] = rows["ric"]
    rows["refinitiv_symbol_status"] = "security_master_discovery_candidate"
    rows["candidate_method"] = "jpx_public_listing_security_master_reconciliation"
    rows["research_universe_id"] = universe_id
    return rows.sort_values(["is_active", "local_code", "refinitiv_ric"], ascending=[False, True, True]).reset_index(
        drop=True
    )


def _empty_universe_frame(stable: pd.DataFrame) -> pd.DataFrame:
    empty = stable.iloc[0:0].copy()
    empty["research_universe_role"] = pd.Series(dtype="object")
    empty["research_universe_reason"] = pd.Series(dtype="object")
    empty["point_in_time_membership_claimed"] = pd.Series(dtype="bool")
    return _as_symbol_map(empty, "JPN_JPX_ACTIVE_OPERATING_CORE_V1")


def _public_reconciliation_record(row: dict[str, object]) -> dict[str, object]:
    return {
        "source_side": "public_jpx",
        "jpx_source_row": row.get("jpx_source_row"),
        "jpx_asof_date": row.get("jpx_asof_date"),
        "jpx_known_at_utc": row.get("jpx_known_at_utc"),
        "local_code": row.get("local_code"),
        "jpx_name": row.get("jpx_name"),
        "jpx_market_category_raw": row.get("jpx_market_category_raw"),
        "jpx_market_scope": row.get("jpx_market_scope"),
        "in_primary_ordinary_share_scope": row.get(
            "in_primary_ordinary_share_scope"
        ),
        "stable_instrument_id": pd.NA,
        "stable_identifier_type": pd.NA,
        "ric": pd.NA,
        "price_currency": pd.NA,
    }


def _ensure_normalized_jpx_listings(listings: pd.DataFrame) -> pd.DataFrame:
    required = {
        "jpx_source_row",
        "jpx_asof_date",
        "local_code",
        "jpx_market_scope",
        "in_primary_ordinary_share_scope",
    }
    return listings.copy() if required.issubset(listings.columns) else normalize_jpx_listings(listings)


def _stable_identifier(row: pd.Series) -> tuple[str, str]:
    isin = _text(row.get("isin")).upper()
    if isin:
        return f"isin:{isin}", "isin"
    instrument_permid = _normalize_identifier(row.get("instrument_permid"))
    if instrument_permid:
        return f"instrument_permid:{instrument_permid}", "instrument_permid"
    sedol = _text(row.get("sedol")).upper()
    if sedol:
        return f"sedol:{sedol}", "sedol"
    return f"ric:{_text(row.get('ric')).upper()}", "ric_fallback"


def _canonical_jpx_market(value: object) -> object:
    text = _text(value)
    if not text:
        return pd.NA
    return _JPX_MARKET_ALIASES.get(text, _JPX_MARKET_ALIASES.get(text.casefold(), "out_of_scope"))


def _normalize_local_code(value: object) -> object:
    text = _text(value).upper()
    if not text:
        return pd.NA
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _local_code_from_ric(value: object) -> object:
    text = _text(value).upper()
    if not text:
        return pd.NA
    return re.split(r"[.\^]", text, maxsplit=1)[0] or pd.NA


def _normalize_identifier(value: object) -> str:
    text = _text(value)
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _normalize_currency(value: object) -> object:
    text = _text(value).upper()
    aliases = {"JAPANESE YEN": "JPY", "YEN": "JPY", "JP YEN": "JPY"}
    return aliases.get(text, text) if text else pd.NA


def _normalize_date(value: object) -> object:
    text = _text(value)
    if not text:
        return pd.NA
    if re.fullmatch(r"\d{8}(?:\.0)?", text):
        text = text[:8]
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else pd.NA


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(source, dtype=object)
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(source, low_memory=False, dtype=object)
        except UnicodeDecodeError:
            return pd.read_csv(source, low_memory=False, dtype=object, encoding="cp932")
    raise ValueError(f"unsupported tabular input: {source}")


def _series(data: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series(default, index=data.index, dtype="object")


def _coalesce(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=data.index, dtype="object")
    for column in columns:
        if column in data.columns:
            values = data[column].copy()
            blank = values.astype("string").str.strip().eq("").fillna(False)
            values = values.mask(blank, pd.NA)
            result = result.where(result.notna(), values)
    return result


def _bool_series(data: pd.DataFrame, column: str) -> pd.Series:
    values = _series(data, column, False)
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
