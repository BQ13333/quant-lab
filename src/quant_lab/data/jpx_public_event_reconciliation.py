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
class JPXPublicEventReconciliationArtifacts:
    run_id: str
    artifact_dir: Path
    reconciliation_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_jpx_public_event_reconciliation(
    *,
    run_id: str,
    public_events_dir: str | Path,
    reference_map_csv: str | Path,
    security_master_csv: str | Path | None = None,
    artifact_root: str | Path = "artifacts/jpx_public_event_reconciliation",
) -> JPXPublicEventReconciliationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    public_dir = Path(public_events_dir)
    reference_path = Path(reference_map_csv)
    public_manifest = _read_json(public_dir / "manifest.json")
    if not (
        public_manifest.get("status") == "complete_limited_coverage"
        and public_manifest.get("official_jpx_source") is True
        and public_manifest.get("coverage", {}).get("completed_pages")
        == public_manifest.get("coverage", {}).get("planned_pages")
    ):
        raise ValueError("public JPX archive artifact is not complete")
    for filename, digest in public_manifest.get("sha256", {}).items():
        path = public_dir / filename
        if not path.is_file() or _sha256(path) != str(digest).lower():
            raise ValueError(f"public JPX artifact hash mismatch: {filename}")
    reference = pd.read_csv(reference_path, low_memory=False)
    if not {"local_code", "refinitiv_ric"}.issubset(reference):
        raise ValueError("reference map is missing local_code/refinitiv_ric")
    reference = reference.copy()
    reference["local_code"] = reference["local_code"].astype(str).str.strip().str.upper()
    counts = reference.groupby("local_code")["refinitiv_ric"].nunique()
    ambiguous = set(counts[counts.gt(1)].index)
    unique_reference = reference.loc[~reference["local_code"].isin(ambiguous)].drop_duplicates("local_code")

    events: list[pd.DataFrame] = []
    delisted = pd.read_csv(public_dir / "delisting_events.csv", dtype=str).fillna("")
    if not delisted.empty:
        frame = delisted.rename(columns={"event_date": "official_event_date"}).copy()
        frame["event_type"] = "delisting"
        frame["local_code"] = frame["code"].map(_local_code)
        events.append(frame)
    listed = pd.read_csv(public_dir / "new_listing_events.csv", dtype=str).fillna("")
    if not listed.empty:
        frame = listed.rename(columns={"listing_date": "official_event_date"}).copy()
        frame["event_type"] = "new_listing"
        frame["local_code"] = frame["code"].map(_local_code)
        events.append(frame)
    combined = pd.concat(events, ignore_index=True, sort=False)
    joined = combined.merge(unique_reference, on="local_code", how="left", suffixes=("_official", "_reference"))
    joined["mapping_status"] = "matched"
    joined.loc[joined["local_code"].isin(ambiguous), "mapping_status"] = "ambiguous_reference_code"
    joined.loc[joined["refinitiv_ric"].isna() & ~joined["local_code"].isin(ambiguous), "mapping_status"] = "unmatched_reference_code"
    for date_column in ("retire_date", "first_trade_date"):
        if date_column not in joined:
            joined[date_column] = ""
        joined[date_column] = joined[date_column].astype("object")
    security_path = Path(security_master_csv) if security_master_csv else None
    if security_path is not None:
        security = pd.read_csv(security_path, low_memory=False)
        ticker_column = "ticker_symbol" if "ticker_symbol" in security else "local_code"
        ric_column = "ric" if "ric" in security else "refinitiv_ric"
        if not {ticker_column, ric_column}.issubset(security):
            raise ValueError("security master lacks ticker/RIC columns")
        if "exchange_code" in security:
            security = security.loc[security["exchange_code"].astype(str).eq("TYO")]
        security = security.copy()
        security["local_code"] = security[ticker_column].astype(str).str.strip().str.upper()
        security["_ric"] = security[ric_column].astype(str).str.strip()
        for index in joined.index[~joined["mapping_status"].eq("matched")]:
            row = joined.loc[index]
            candidates = security.loc[security["local_code"].eq(row["local_code"])].copy()
            if candidates.empty:
                continue
            date_column = "retire_date" if row["event_type"] == "delisting" else "first_trade_date"
            if date_column in candidates:
                candidate_dates = pd.to_datetime(candidates[date_column], errors="coerce").dt.date.astype("string")
                exact = candidates.loc[candidate_dates.eq(str(row["official_event_date"]))]
            else:
                exact = candidates.head(0)
            chosen = exact if exact["_ric"].nunique() == 1 else candidates if candidates["_ric"].nunique() == 1 else candidates.head(0)
            if chosen.empty:
                continue
            selected = chosen.iloc[0]
            joined.at[index, "refinitiv_ric"] = selected["_ric"]
            if "retire_date" in selected:
                joined.at[index, "retire_date"] = selected.get("retire_date")
            if "first_trade_date" in selected:
                joined.at[index, "first_trade_date"] = selected.get("first_trade_date")
            joined.at[index, "mapping_status"] = (
                "matched_security_master_exact_event_date" if not exact.empty else "matched_security_master_single_candidate"
            )
    expected_date = pd.Series("", index=joined.index, dtype="object")
    if "retire_date" in joined:
        expected_date.loc[joined["event_type"].eq("delisting")] = joined.loc[
            joined["event_type"].eq("delisting"), "retire_date"
        ].fillna("").astype(str)
    if "first_trade_date" in joined:
        expected_date.loc[joined["event_type"].eq("new_listing")] = joined.loc[
            joined["event_type"].eq("new_listing"), "first_trade_date"
        ].fillna("").astype(str)
    joined["reference_event_date"] = pd.to_datetime(expected_date, errors="coerce").dt.date.astype("string").fillna("")
    official_dates = pd.to_datetime(joined["official_event_date"], errors="coerce")
    reference_dates = pd.to_datetime(joined["reference_event_date"], errors="coerce")
    joined["date_difference_days"] = (reference_dates - official_dates).dt.days
    joined["date_status"] = "missing_reference_date"
    joined.loc[reference_dates.notna() & joined["date_difference_days"].eq(0), "date_status"] = "exact_match"
    joined.loc[reference_dates.notna() & joined["date_difference_days"].ne(0), "date_status"] = "mismatch"

    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    reconciliation_path = root / "event_reconciliation.csv"
    manifest_path = root / "manifest.json"
    _atomic_write_csv(reconciliation_path, joined)
    total = len(joined)
    matched = int(joined["mapping_status"].astype(str).str.startswith("matched").sum())
    exact = int(joined["date_status"].eq("exact_match").sum())
    manifest = {
        "run_id": run_id,
        "artifact_type": "jpx_public_event_refinitiv_cross_validation",
        "status": "complete_with_discrepancies" if exact < total or matched < total else "pass",
        "scorecard_eligible": False,
        "evidence_scope": "recent_official_public_archive_cross_validation",
        "historical_pit_ready": False,
        "counts": {
            "events": total,
            "mapping_matched": matched,
            "mapping_unmatched_or_ambiguous": total - matched,
            "date_exact_match": exact,
            "date_missing_or_mismatch": total - exact,
        },
        "mapping_coverage": matched / total if total else 0.0,
        "date_exact_match_coverage": exact / total if total else 0.0,
        "input_sha256": {
            "public_manifest": _sha256(public_dir / "manifest.json"),
            "reference_map": _sha256(reference_path),
            **({"security_master": _sha256(security_path)} if security_path is not None else {}),
        },
        "output_sha256": _sha256(reconciliation_path),
        "limitations": [
            "public archives do not cover the full required 2008-present period",
            "public event rows lack exact known-at timestamps",
            "this artifact is a validator, not a PIT membership source",
        ],
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(manifest_path, manifest)
    return JPXPublicEventReconciliationArtifacts(run_id, root, reconciliation_path, manifest_path, manifest)


def _local_code(value: str) -> str:
    code = str(value).strip().upper()
    return code[:4] if len(code) == 5 and code.endswith("0") else code


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
