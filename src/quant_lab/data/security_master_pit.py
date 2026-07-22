from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SecurityMasterPITSeedArtifacts:
    run_id: str
    artifact_dir: Path
    entities_path: Path
    identifier_history_path: Path
    listing_windows_path: Path
    universe_membership_path: Path
    quality_path: Path
    promotion_gates_path: Path
    manifest_path: Path
    summary_path: Path
    entities: pd.DataFrame
    identifier_history: pd.DataFrame
    listing_windows: pd.DataFrame
    universe_membership: pd.DataFrame
    quality: pd.DataFrame
    promotion_gates: pd.DataFrame


def write_security_master_pit_seed_artifacts(
    selected_universe: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    current_snapshot: pd.DataFrame | None = None,
    current_survivor_master: pd.DataFrame | None = None,
) -> SecurityMasterPITSeedArtifacts:
    """Write a PIT security-master scaffold from current-survivor evidence.

    The output is intentionally not production-eligible. It creates the table
    shapes required by the institutional gate while preserving the fact that
    current snapshots cannot solve historical denominator/survivorship bias.
    """

    if selected_universe.empty:
        raise ValueError("selected_universe is empty")
    artifact_dir = Path(artifact_root) / run_id
    quality_dir = artifact_dir / "quality"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    base = _merge_seed_sources(selected_universe, current_snapshot, current_survivor_master)
    asof_date = _infer_asof_date(base)
    source_loaded_at_utc = _now_utc()

    entities = _build_entities(base, source_loaded_at_utc=source_loaded_at_utc)
    identifier_history = _build_identifier_history(base, source_loaded_at_utc=source_loaded_at_utc)
    listing_windows = _build_listing_windows(base, source_loaded_at_utc=source_loaded_at_utc)
    universe_membership = _build_universe_membership(base, asof_date=asof_date, source_loaded_at_utc=source_loaded_at_utc)
    quality = _build_quality(base, asof_date=asof_date)
    promotion_gates = _build_promotion_gates()

    entities_path = artifact_dir / "entities.parquet"
    identifier_history_path = artifact_dir / "identifier_history.parquet"
    listing_windows_path = artifact_dir / "listing_windows.parquet"
    universe_membership_path = artifact_dir / "universe_membership_pit.parquet"
    quality_path = quality_dir / "security_master_pit_quality.csv"
    promotion_gates_path = artifact_dir / "promotion_gates.csv"
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.md"

    entities.to_parquet(entities_path, index=False)
    identifier_history.to_parquet(identifier_history_path, index=False)
    listing_windows.to_parquet(listing_windows_path, index=False)
    universe_membership.to_parquet(universe_membership_path, index=False)
    quality.to_csv(quality_path, index=False)
    promotion_gates.to_csv(promotion_gates_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "current_survivor_seed_only",
        "promotion_state": "current_survivor_seed_only",
        "model_denominator_ready": False,
        "selected_rows": int(len(base)),
        "countries": sorted(base["country"].dropna().astype(str).unique().tolist()) if "country" in base else [],
        "asof_date": str(asof_date),
        "date_min": _date_min_text(base),
        "dead_name_rows": 0,
        "identifier_history_rows": int(len(identifier_history)),
        "listing_window_rows": int(len(listing_windows)),
        "universe_membership_rows": int(len(universe_membership)),
        "generated_at_utc": source_loaded_at_utc,
        "warning": "Current-survivor seed only. Do not use as a production point-in-time backtest denominator.",
        "artifacts": {
            "entities": str(entities_path.resolve()),
            "identifier_history": str(identifier_history_path.resolve()),
            "listing_windows": str(listing_windows_path.resolve()),
            "universe_membership_pit": str(universe_membership_path.resolve()),
            "quality": str(quality_path.resolve()),
            "promotion_gates": str(promotion_gates_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_md(run_id, manifest, promotion_gates), encoding="utf-8")

    return SecurityMasterPITSeedArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        entities_path=entities_path,
        identifier_history_path=identifier_history_path,
        listing_windows_path=listing_windows_path,
        universe_membership_path=universe_membership_path,
        quality_path=quality_path,
        promotion_gates_path=promotion_gates_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        entities=entities,
        identifier_history=identifier_history,
        listing_windows=listing_windows,
        universe_membership=universe_membership,
        quality=quality,
        promotion_gates=promotion_gates,
    )


def _merge_seed_sources(
    selected_universe: pd.DataFrame,
    current_snapshot: pd.DataFrame | None,
    current_survivor_master: pd.DataFrame | None,
) -> pd.DataFrame:
    base = selected_universe.copy()
    if "refinitiv_ric" not in base.columns and "ticker" in base.columns:
        base["refinitiv_ric"] = base["ticker"]
    if "sid" not in base.columns:
        base["sid"] = base["country"].astype(str) + ":" + base["refinitiv_ric"].astype(str)

    snapshot = current_snapshot.copy() if current_snapshot is not None else pd.DataFrame()
    if not snapshot.empty:
        snapshot_key = "refinitiv_ric" if "refinitiv_ric" in snapshot.columns else "requested_ric"
        snapshot = snapshot.drop_duplicates(snapshot_key)
        base = base.merge(
            snapshot,
            left_on="refinitiv_ric",
            right_on=snapshot_key,
            how="left",
            suffixes=("", "_snapshot"),
        )

    survivor = current_survivor_master.copy() if current_survivor_master is not None else pd.DataFrame()
    if not survivor.empty and "ticker" in survivor.columns:
        survivor = survivor.drop_duplicates("ticker")
        base = base.merge(survivor, on="ticker", how="left", suffixes=("", "_survivor"))
    return base


def _build_entities(base: pd.DataFrame, source_loaded_at_utc: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": base["sid"].astype(str),
            "perm_id": _first_available(base, ["Organization PermID", "organization_perm_id", "perm_id"]),
            "issuer_name": _first_available(base, ["name", "name_snapshot", "name_survivor"]),
            "country": _first_available(base, ["country", "country_snapshot", "country_survivor"]),
            "incorporation_country": pd.NA,
            "security_type": "ordinary_equity",
            "primary_exchange": _first_available(base, ["exchange", "exchange_snapshot", "exchange_survivor"]),
            "source_vendor": "refinitiv_current_snapshot_and_l2_seed",
            "entity_source_scope": "current_survivor_seed",
            "source_loaded_at_utc": source_loaded_at_utc,
        }
    )


def _build_identifier_history(base: pd.DataFrame, source_loaded_at_utc: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_specs = (
        ("ric", "refinitiv_ric"),
        ("ticker", "ticker"),
        ("local_symbol", "local_symbol"),
        ("isin", "ISIN"),
        ("isin", "isin"),
        ("sedol", "SEDOL"),
    )
    effective_from = _first_available(base, ["First Trade Date", "first_date", "first_trade_date"])
    for idx, item in base.reset_index(drop=True).iterrows():
        for id_type, column in id_specs:
            if column not in base.columns:
                continue
            value = item.get(column)
            if pd.isna(value) or str(value).strip() == "":
                continue
            rows.append(
                {
                    "entity_id": item["sid"],
                    "sid": item["sid"],
                    "id_type": id_type,
                    "id_value": str(value),
                    "effective_from": effective_from.iloc[idx] if idx < len(effective_from) else pd.NA,
                    "effective_to": pd.NA,
                    "change_type": "current_identifier_seed",
                    "change_reason": "current snapshot only",
                    "source_timestamp_utc": pd.NA,
                    "source_loaded_at_utc": source_loaded_at_utc,
                    "evidence_level": "current_snapshot_seed",
                    "eligible_for_model_denominator": False,
                }
            )
    return pd.DataFrame(rows)


def _build_listing_windows(base: pd.DataFrame, source_loaded_at_utc: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": base["sid"].astype(str),
            "sid": base["sid"].astype(str),
            "ric": _first_available(base, ["refinitiv_ric", "requested_ric"]),
            "ticker": _first_available(base, ["ticker"]),
            "country": _first_available(base, ["country", "country_snapshot", "country_survivor"]),
            "exchange": _first_available(base, ["exchange", "exchange_snapshot", "exchange_survivor"]),
            "currency": _first_available(base, ["currency", "currency_snapshot", "currency_survivor"]),
            "listing_date": _first_available(base, ["First Trade Date", "first_date", "first_trade_date"]),
            "first_trade_date": _first_available(base, ["First Trade Date", "first_date", "first_trade_date"]),
            "effective_from": _first_available(base, ["First Trade Date", "first_date", "first_trade_date"]),
            "effective_to": pd.NA,
            "delisting_date": pd.NA,
            "last_trade_date": _first_available(base, ["last_date"]),
            "listing_status": "active_current_snapshot",
            "delisting_reason": pd.NA,
            "tradable_flag": True,
            "active_flag": True,
            "window_source": "current_survivor_seed",
            "evidence_level": "current_snapshot_seed",
            "eligible_for_model_denominator": False,
            "source_loaded_at_utc": source_loaded_at_utc,
        }
    )


def _build_universe_membership(base: pd.DataFrame, asof_date: str, source_loaded_at_utc: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": asof_date,
            "entity_id": base["sid"].astype(str),
            "sid": base["sid"].astype(str),
            "ric_asof": _first_available(base, ["refinitiv_ric", "requested_ric"]),
            "ticker_asof": _first_available(base, ["ticker"]),
            "country": _first_available(base, ["country", "country_snapshot", "country_survivor"]),
            "exchange": _first_available(base, ["exchange", "exchange_snapshot", "exchange_survivor"]),
            "currency": _first_available(base, ["currency", "currency_snapshot", "currency_survivor"]),
            "is_listed_asof": True,
            "is_investable_asof": False,
            "eligible_flag": False,
            "in_u2_flag": _bool_series(base, "selected_for_u2"),
            "in_u3_flag": _bool_series(base, "selected_for_u3"),
            "exclusion_reason": "current_survivor_seed_only",
            "known_at_utc": pd.NA,
            "membership_source": "current_survivor_seed",
            "source_loaded_at_utc": source_loaded_at_utc,
        }
    )


def _build_quality(base: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    rows = []
    for country, group in base.groupby("country", dropna=False, sort=True):
        active_count = int(len(group))
        rics = _first_available(group, ["refinitiv_ric", "requested_ric"])
        rows.append(
            {
                "country": country,
                "asof_year": int(pd.Timestamp(asof_date).year),
                "active_count": active_count,
                "delisted_count": 0,
                "identifier_change_count": 0,
                "null_ric_rate": float(rics.isna().mean()) if active_count else 1.0,
                "overlap_with_current_survivor_rate": 1.0,
                "dead_name_coverage_rate": 0.0,
                "status": "current_survivor_seed_only",
            }
        )
    return pd.DataFrame(rows)


def _build_promotion_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "current_snapshot_not_promoted",
                "status": "pass",
                "evidence": "seed rows are explicitly ineligible for model denominator",
            },
            {
                "gate_id": "no_current_survivor_only_universe",
                "status": "fail",
                "evidence": "only current-survivor seed rows are available",
            },
            {
                "gate_id": "has_delisted_names_or_policy",
                "status": "fail",
                "evidence": "dead_name_rows=0 and no validated exclusion policy",
            },
            {
                "gate_id": "listing_windows_non_overlapping",
                "status": "partial",
                "evidence": "single current active window per seed security",
            },
            {
                "gate_id": "identifier_history_present",
                "status": "partial",
                "evidence": "current identifiers are present but no historical changes",
            },
        ]
    )


def _summary_md(run_id: str, manifest: dict[str, Any], promotion_gates: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"# Security Master PIT Seed - {run_id}",
            "",
            "This artifact is a scaffold and current-survivor seed. It is not a production PIT denominator.",
            "",
            "## Verdict",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Model denominator ready: `{manifest['model_denominator_ready']}`",
            f"- Selected rows: `{manifest['selected_rows']}`",
            f"- Dead-name rows: `{manifest['dead_name_rows']}`",
            "",
            "## Promotion Gates",
            "",
            promotion_gates.to_markdown(index=False),
            "",
        ]
    )


def _first_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            values = frame[column]
            out = out.where(out.notna(), values)
    return out.reset_index(drop=True)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([False] * len(frame))
    return frame[column].fillna(False).astype(bool).reset_index(drop=True)


def _infer_asof_date(base: pd.DataFrame) -> str:
    for column in ("asof_date", "last_date"):
        if column in base.columns:
            values = pd.to_datetime(base[column], errors="coerce").dropna()
            if not values.empty:
                return str(values.max().date())
    return datetime.now(timezone.utc).date().isoformat()


def _date_min_text(base: pd.DataFrame) -> str:
    values = pd.Series(dtype="datetime64[ns]")
    for column in ("First Trade Date", "first_date", "first_trade_date"):
        if column in base.columns:
            values = pd.concat([values, pd.to_datetime(base[column], errors="coerce").dropna()])
    if values.empty:
        return ""
    return str(values.min().date())


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
