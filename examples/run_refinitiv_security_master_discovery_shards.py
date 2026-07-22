from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.security_master_discovery import (
    SUPPORTED_DISCOVERY_COUNTRY_CODES,
    write_refinitiv_security_master_discovery_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


DEFAULT_PREFIXES = tuple("0123456789")


def _normalize_country_code(value: str) -> str:
    aliases = {
        "my": "MYS",
        "malaysia": "MYS",
        "sg": "SGP",
        "singapore": "SGP",
        "id": "IDN",
        "indonesia": "IDN",
        "cn": "CHN",
        "china": "CHN",
        "hk": "HKG",
        "hong kong": "HKG",
        "tw": "TWN",
        "taiwan": "TWN",
        "kr": "KOR",
        "korea": "KOR",
        "south korea": "KOR",
        "jp": "JPN",
        "japan": "JPN",
        "us": "USA",
        "usa": "USA",
        "united states": "USA",
        "th": "THA",
        "thailand": "THA",
    }
    allowed = {code.casefold(): code for code in SUPPORTED_DISCOVERY_COUNTRY_CODES}
    key = value.strip().casefold()
    code = aliases.get(key, allowed.get(key))
    if code is None:
        raise SystemExit(f"Unsupported Refinitiv Discovery country code: {value}")
    return code


def _parse_state_buckets(value: str) -> tuple[str, ...]:
    if not value:
        return ("inactive",)
    allowed = {"active", "inactive"}
    buckets: list[str] = []
    for item in value.split(","):
        bucket = item.strip().casefold()
        if not bucket:
            continue
        if bucket not in allowed:
            raise SystemExit(f"Unsupported state bucket: {item}")
        buckets.append(bucket)
    return tuple(dict.fromkeys(buckets))


def _parse_prefixes(value: str) -> tuple[str, ...]:
    if not value:
        return DEFAULT_PREFIXES
    prefixes = [item.strip() for item in value.split(",") if item.strip()]
    if not prefixes:
        raise SystemExit("At least one prefix is required")
    return tuple(dict.fromkeys(prefixes))


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return token.upper() or "BLANK"


def _quote_filter_literal(value: str) -> str:
    return value.replace("'", "''")


def _prefix_filter(field: str, prefix: str) -> str:
    return f"startswith({field},'{_quote_filter_literal(prefix)}')"


def _load_filter_plan(path: Path) -> list[tuple[str, str]]:
    plan = pd.read_csv(path)
    required = {"shard_id", "extra_filter"}
    missing = required - set(plan.columns)
    if missing:
        raise SystemExit(f"Filter plan is missing columns: {sorted(missing)}")
    rows: list[tuple[str, str]] = []
    for item in plan.itertuples(index=False):
        data = item._asdict()
        shard_id = str(data["shard_id"]).strip()
        extra_filter = str(data["extra_filter"]).strip()
        if not shard_id or not extra_filter:
            continue
        rows.append((shard_id, extra_filter))
    if not rows:
        raise SystemExit("Filter plan has no usable shards")
    return rows


def _all_routes_exhausted(manifest: dict[str, Any]) -> bool:
    resume = manifest.get("pagination", {}).get("resume", {})
    if not resume:
        return False
    return all(bool(route.get("exhausted")) for route in resume.values())


def _summary_row(prefix: str, run_id: str, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    pagination = manifest.get("pagination", {})
    return {
        "prefix": prefix,
        "run_id": run_id,
        "status": manifest.get("status", ""),
        "rows": int(manifest.get("rows", 0)),
        "active_rows": int(manifest.get("active_rows", 0)),
        "inactive_rows": int(manifest.get("inactive_rows", 0)),
        "dead_name_rows": int(manifest.get("dead_name_rows", 0)),
        "pages_fetched": int(pagination.get("pages_fetched", 0)),
        "stopped_by": pagination.get("stopped_by"),
        "routes_exhausted": _all_routes_exhausted(manifest),
        "extra_filter": manifest.get("extra_filter"),
        "manifest_path": str(manifest_path.resolve()),
        "error": "",
    }


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"lseg\.data\..*")
    parser = argparse.ArgumentParser(
        description="Run bounded Refinitiv Discovery security-master shards with one session."
    )
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--state-buckets", default="inactive")
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument("--prefix-field", default="TickerSymbol")
    parser.add_argument("--filter-plan-csv", default=None, help="CSV with shard_id,extra_filter columns.")
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--run-id-prefix", default=None)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--start-skip", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--enrichment-chunk-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--between-shard-sleep", type=float, default=0.0)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    country_code = _normalize_country_code(args.country)
    state_buckets = _parse_state_buckets(args.state_buckets)
    if args.filter_plan_csv:
        shards = _load_filter_plan(Path(args.filter_plan_csv))
    else:
        shards = [(prefix, _prefix_filter(args.prefix_field, prefix)) for prefix in _parse_prefixes(args.prefixes)]
    if args.max_shards is not None:
        if args.max_shards <= 0:
            raise SystemExit("--max-shards must be positive")
        shards = shards[: args.max_shards]
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id_prefix = args.run_id_prefix or f"SECMASTER-DISCOVERY-{country_code}-SHARDED-{run_date}"
    artifact_root = Path(args.artifact_root)
    summary_dir = Path(args.summary_dir) if args.summary_dir else artifact_root / f"{run_id_prefix}-SUMMARY"
    summary_path = summary_dir / "shard_summary.csv"
    summary_manifest_path = summary_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    if args.skip_existing and summary_path.exists():
        rows = pd.read_csv(summary_path).fillna("").to_dict("records")

    completed_shards = {
        str(row.get("prefix"))
        for row in rows
        if str(row.get("error", "")) == "" and str(row.get("status", "")) in {"complete", "partial", "diagnostic"}
    }

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        for shard_id, extra_filter in shards:
            if args.skip_existing and shard_id in completed_shards:
                continue

            shard_run_id = f"{run_id_prefix}-PREFIX-{_safe_token(shard_id)}"
            shard_dir = artifact_root / shard_run_id
            if args.skip_existing and (shard_dir / "manifest.json").exists():
                manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
                rows.append(_summary_row(shard_id, shard_run_id, manifest, shard_dir / "manifest.json"))
                _write_summary(summary_path, rows)
                continue

            try:
                artifact = write_refinitiv_security_master_discovery_artifacts(
                    discovery_client=adapter.ld,
                    get_data_client=adapter.ld,
                    artifact_root=artifact_root,
                    run_id=shard_run_id,
                    country_codes=(country_code,),
                    state_buckets=state_buckets,
                    top=args.top,
                    start_skip=args.start_skip,
                    max_pages=args.max_pages,
                    max_rows=args.max_rows,
                    enrichment_chunk_size=args.enrichment_chunk_size,
                    sleep_seconds=args.sleep_seconds,
                    primary_only=args.primary_only,
                    extra_filter=extra_filter,
                )
                rows.append(_summary_row(shard_id, shard_run_id, artifact.manifest, artifact.manifest_path))
            except Exception as exc:  # pragma: no cover - exercised against live vendor routes.
                rows.append(
                    {
                        "prefix": shard_id,
                        "run_id": shard_run_id,
                        "status": "error",
                        "rows": 0,
                        "active_rows": 0,
                        "inactive_rows": 0,
                        "dead_name_rows": 0,
                        "pages_fetched": 0,
                        "stopped_by": "",
                        "routes_exhausted": False,
                        "extra_filter": extra_filter,
                        "manifest_path": "",
                        "error": repr(exc),
                    }
                )
                if args.stop_on_error:
                    _write_summary(summary_path, rows)
                    raise
            _write_summary(summary_path, rows)
            if args.between_shard_sleep > 0:
                time.sleep(args.between_shard_sleep)
    finally:
        adapter.close()

    summary = pd.DataFrame(rows)
    total_rows = int(summary.get("rows", pd.Series(dtype="int64")).sum()) if not summary.empty else 0
    total_dead = int(summary.get("dead_name_rows", pd.Series(dtype="int64")).sum()) if not summary.empty else 0
    errors = int(summary["status"].eq("error").sum()) if "status" in summary.columns else 0
    capped = int(summary["stopped_by"].fillna("").ne("").sum()) if "stopped_by" in summary.columns else 0
    incomplete = int((~summary["routes_exhausted"].astype(bool)).sum()) if "routes_exhausted" in summary.columns else 0
    summary_manifest = {
        "run_id_prefix": run_id_prefix,
        "country_code": country_code,
        "state_buckets": list(state_buckets),
        "prefix_field": args.prefix_field,
        "filter_plan_csv": str(Path(args.filter_plan_csv).resolve()) if args.filter_plan_csv else None,
        "shards": [shard_id for shard_id, _ in shards],
        "summary_path": str(summary_path.resolve()),
        "total_rows": total_rows,
        "total_dead_name_rows": total_dead,
        "error_shards": errors,
        "capped_shards": capped,
        "non_exhausted_shards": incomplete,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_manifest_path.write_text(json.dumps(summary_manifest, indent=2), encoding="utf-8")

    print(f"Shard run prefix: {run_id_prefix}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {total_rows}")
    print(f"Dead/inactive rows: {total_dead}")
    print(f"Error shards: {errors}")
    print(f"Capped shards: {capped}")
    print(f"Non-exhausted shards: {incomplete}")


if __name__ == "__main__":
    main()
