from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SECTOR_FIELDS: tuple[tuple[str, str], ...] = (
    ("trbc_industry_group_code", "TR.TRBCIndustryGroupCode"),
    ("trbc_industry_group", "TR.TRBCIndustryGroup"),
    ("trbc_business_sector_code", "TR.TRBCBusinessSectorCode"),
    ("trbc_business_sector", "TR.TRBCBusinessSector"),
    ("trbc_industry_code", "TR.TRBCIndustryCode"),
    ("trbc_industry", "TR.TRBCIndustry"),
)


def sector_field_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "fundamentals_sector_routing",
                "field_role": role,
                "candidate_field": field,
                "acceptance_status": "source_backed_live_probe",
            }
            for role, field in SECTOR_FIELDS
        ]
    )


def build_sector_routing_profile(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()
    aliases = {
        "ric": ("requested_ric", "Instrument", "instrument", "refinitiv_ric"),
        "trbc_industry_group_code": ("TRBC Industry Group Code",),
        "trbc_industry_group": ("TRBC Industry Group Name",),
        "trbc_business_sector_code": ("TRBC Business Sector Code",),
        "trbc_business_sector": ("TRBC Business Sector Name",),
        "trbc_industry_code": ("TRBC Industry Code",),
        "trbc_industry": ("TRBC Industry Name",),
    }
    profile = pd.DataFrame(index=snapshot.index)
    for target, candidates in aliases.items():
        source = next((column for column in candidates if column in snapshot.columns), None)
        profile[target] = snapshot[source] if source else pd.NA
    profile = profile.dropna(subset=["ric"]).drop_duplicates("ric", keep="last").copy()
    business_code = profile["trbc_business_sector_code"].astype("string").str.replace(r"\.0$", "", regex=True)
    profile["fundamental_route_class"] = "general"
    profile.loc[business_code.eq("5510"), "fundamental_route_class"] = "bank"
    profile.loc[business_code.eq("5530"), "fundamental_route_class"] = "insurance"
    profile["revenue_source_field"] = "TR.Revenue"
    profile.loc[profile["fundamental_route_class"].eq("bank"), "revenue_source_field"] = "TR.BankTotalRevenue"
    profile["profitability_source_field"] = "TR.GrossProfit"
    profile.loc[profile["fundamental_route_class"].eq("bank"), "profitability_source_field"] = "TR.NetInterestIncome"
    profile.loc[profile["fundamental_route_class"].eq("insurance"), "profitability_source_field"] = "TR.OperatingIncome"
    profile["routing_scope"] = "current_classification_only"
    profile["point_in_time_sector_claimed"] = False
    return profile.sort_values("ric").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_sector_snapshot_artifact(
    artifact_dir: str | Path,
    snapshot: pd.DataFrame,
    requested_rics: int,
    run_id: str,
) -> Path:
    directory = Path(artifact_dir)
    source_manifest = directory / "manifest.json"
    snapshot_manifest_path = directory / "snapshot_manifest.json"
    if source_manifest.exists():
        snapshot_manifest_path.write_bytes(source_manifest.read_bytes())

    profile = build_sector_routing_profile(snapshot)
    profile_path = directory / "sector_routing_profile.csv"
    profile.to_csv(profile_path, index=False)
    counts = profile["fundamental_route_class"].value_counts().to_dict() if not profile.empty else {}
    coverage = float(profile["ric"].nunique() / requested_rics) if requested_rics else 0.0
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(directory.resolve()),
        "status": "complete_current_routing" if coverage == 1.0 else "incomplete_current_routing",
        "scope": "current_sector_routing_only",
        "evidence_scope": "current_snapshot",
        "scorecard_eligible": False,
        "partial": True,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "requested_rics": requested_rics,
        "profile_rics": int(profile["ric"].nunique()) if not profile.empty else 0,
        "coverage": coverage,
        "route_class_counts": {str(key): int(value) for key, value in counts.items()},
        "point_in_time_sector_claimed": False,
        "usage_policy": "Routing aid only. Historical sector classifications require independent PIT evidence.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            "sector_routing_profile": _sha256(profile_path),
            "snapshot_manifest": _sha256(snapshot_manifest_path) if snapshot_manifest_path.exists() else "",
        },
        "artifacts": {
            "sector_routing_profile": str(profile_path.resolve()),
            "snapshot_manifest": str(snapshot_manifest_path.resolve()) if snapshot_manifest_path.exists() else "",
        },
    }
    source_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return source_manifest
