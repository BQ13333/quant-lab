from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.security_master_discovery import (
    DEFAULT_DISCOVERY_COUNTRY_CODES,
    DISCOVERY_COUNTRY_NAMES,
    build_security_master_discovery_manifest,
    build_security_master_discovery_tables,
    merge_security_master_discovery_data,
)


def _parse_country_codes(value: str) -> tuple[str, ...]:
    if not value:
        return DEFAULT_DISCOVERY_COUNTRY_CODES
    codes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not codes:
        raise SystemExit("At least one country code is required")
    return tuple(dict.fromkeys(codes))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_components_file(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Components file does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _component_dirs(paths: tuple[str, ...], *, skip_missing_manifest: bool = False) -> list[Path]:
    dirs: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.name == "manifest.json":
            path = path.parent
        if not path.exists():
            raise SystemExit(f"Component artifact does not exist: {path}")
        if not (path / "manifest.json").exists():
            if skip_missing_manifest:
                continue
            raise SystemExit(f"Component artifact has no manifest.json: {path}")
        dirs.append(path)
    if not dirs:
        raise SystemExit("At least one component artifact is required")
    return dirs


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(non_empty, ignore_index=True, sort=False) if non_empty else pd.DataFrame()


def _aggregate_pagination(component_manifests: list[dict[str, Any]], merged_rows: int) -> dict[str, Any]:
    pages_fetched = 0
    stopped_by: str | None = None
    component_states: list[dict[str, Any]] = []
    for manifest in component_manifests:
        pagination = manifest.get("pagination", {})
        pages_fetched += int(pagination.get("pages_fetched", 0) or 0)
        stopped_by = stopped_by or pagination.get("stopped_by")
        component_states.append(
            {
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "rows": manifest.get("rows"),
                "active_rows": manifest.get("active_rows"),
                "inactive_rows": manifest.get("inactive_rows"),
                "stopped_by": pagination.get("stopped_by"),
                "extra_filter": manifest.get("extra_filter"),
                "artifact_dir": manifest.get("artifact_dir"),
            }
        )
    return {
        "top": None,
        "start_skip": 0,
        "max_pages": None,
        "max_rows": None,
        "primary_only": False,
        "extra_filter": "merged_component_artifacts",
        "pages_fetched": pages_fetched,
        "rows_fetched": int(merged_rows),
        "stopped_by": stopped_by,
        "resume": {"merged:components": {"exhausted": stopped_by is None}},
        "component_count": len(component_manifests),
        "components": component_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded Refinitiv Discovery security-master artifacts.")
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--components-file", type=Path)
    parser.add_argument("--skip-missing-manifest", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--countries", default="JPN")
    args = parser.parse_args()

    component_values = list(args.components or [])
    if args.components_file is not None:
        component_values.extend(_read_components_file(args.components_file))
    component_dirs = _component_dirs(tuple(component_values), skip_missing_manifest=args.skip_missing_manifest)
    requested_country_codes = _parse_country_codes(args.countries)
    artifact_dir = Path(args.artifact_root) / args.run_id
    quality_dir = artifact_dir / "quality"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    search_hits = _concat_frames([_read_csv_if_exists(path / "discovery_search_hits.csv") for path in component_dirs])
    enrichment = _concat_frames([_read_csv_if_exists(path / "discovery_enrichment.csv") for path in component_dirs])
    page_log = _concat_frames([_read_csv_if_exists(path / "discovery_page_log.csv") for path in component_dirs])
    manifests = [_read_manifest(path / "manifest.json") for path in component_dirs]

    if not search_hits.empty:
        search_hits = search_hits.drop_duplicates(["ric", "country_code", "active_state_bucket"], keep="first")
        search_hits = search_hits.reset_index(drop=True)
    if not enrichment.empty and "ric" in enrichment.columns:
        enrichment = enrichment.drop_duplicates("ric", keep="first").reset_index(drop=True)

    security_master = merge_security_master_discovery_data(search_hits, enrichment)
    source_loaded_at_utc = datetime.now(timezone.utc).isoformat()
    tables = build_security_master_discovery_tables(
        security_master,
        requested_country_codes=requested_country_codes,
        source_loaded_at_utc=source_loaded_at_utc,
    )

    search_hits_path = artifact_dir / "discovery_search_hits.csv"
    enrichment_path = artifact_dir / "discovery_enrichment.csv"
    page_log_path = artifact_dir / "discovery_page_log.csv"
    security_master_path = artifact_dir / "security_master.csv"
    entities_path = artifact_dir / "entities.parquet"
    identifier_history_path = artifact_dir / "identifier_history.parquet"
    listing_windows_path = artifact_dir / "listing_windows.parquet"
    universe_membership_path = artifact_dir / "universe_membership_pit.parquet"
    quality_path = quality_dir / "security_master_pit_quality.csv"
    manifest_path = artifact_dir / "manifest.json"

    search_hits.to_csv(search_hits_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    page_log.to_csv(page_log_path, index=False)
    tables.security_master.to_csv(security_master_path, index=False)
    tables.entities.to_parquet(entities_path, index=False)
    tables.identifier_history.to_parquet(identifier_history_path, index=False)
    tables.listing_windows.to_parquet(listing_windows_path, index=False)
    tables.universe_membership.to_parquet(universe_membership_path, index=False)
    tables.quality.to_csv(quality_path, index=False)

    pagination_state = _aggregate_pagination(manifests, merged_rows=len(tables.security_master))
    manifest = build_security_master_discovery_manifest(
        run_id=args.run_id,
        artifact_dir=artifact_dir,
        security_master=tables.security_master,
        quality=tables.quality,
        requested_country_codes=requested_country_codes,
        pagination_state=pagination_state,
        source_loaded_at_utc=source_loaded_at_utc,
        artifacts={
            "discovery_search_hits": search_hits_path,
            "discovery_enrichment": enrichment_path,
            "discovery_page_log": page_log_path,
            "security_master": security_master_path,
            "entities": entities_path,
            "identifier_history": identifier_history_path,
            "listing_windows": listing_windows_path,
            "universe_membership_pit": universe_membership_path,
            "quality": quality_path,
        },
    )
    manifest["source_component_run_ids"] = [manifest.get("run_id") for manifest in manifests]
    manifest["source_component_count"] = len(manifests)
    manifest["source_component_artifact_dirs"] = [str(path.resolve()) for path in component_dirs]
    manifest["merge_country_names"] = [DISCOVERY_COUNTRY_NAMES.get(code, code) for code in requested_country_codes]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Run ID: {args.run_id}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Status: {manifest['status']}")
    print(f"Rows: {manifest['rows']}")
    print(f"Active rows: {manifest['active_rows']}")
    print(f"Dead/inactive rows: {manifest['dead_name_rows']}")
    print(f"Manifest: {manifest_path}")
    print(tables.quality.to_string(index=False))


if __name__ == "__main__":
    main()
