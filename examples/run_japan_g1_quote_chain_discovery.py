from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g1_quote_chain_discovery import (
    quote_chain_filter,
    write_japan_g1_quote_chain_plan,
)
from quant_lab.data.security_master_discovery import (
    write_refinitiv_security_master_discovery_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


UTC = timezone.utc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded identity-anchored Japan G1 quote-chain Discovery shards.")
    parser.add_argument("--run-id", default="JPN-G1-QUOTE-CHAIN-DISCOVERY-20260712-V1")
    parser.add_argument("--missing-plan", default="artifacts/japan_g1_monthly_materializer/JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1/targeted_missing_code_plan.csv")
    parser.add_argument("--plan-root", default="artifacts/japan_g1_quote_chain_discovery")
    parser.add_argument("--security-master-dir", action="append", dest="security_master_dirs")
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-new-shards", type=int, default=1)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--enrichment-chunk-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--health-symbol", default="7203.T")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.max_new_shards <= 0:
        raise SystemExit("chunk-size and max-new-shards must be positive")

    master_dirs = args.security_master_dirs or _default_security_master_dirs()
    master_paths = [str(Path(value) / "security_master.csv") for value in master_dirs]
    plan_artifact = write_japan_g1_quote_chain_plan(
        run_id=args.run_id,
        missing_plan_path=args.missing_plan,
        security_master_paths=master_paths,
        artifact_root=args.plan_root,
    )
    plan = pd.read_csv(plan_artifact.identity_plan_path, dtype=str).fillna("")
    shards = []
    for number, offset in enumerate(range(0, len(plan), args.chunk_size), start=1):
        rows = plan.iloc[offset : offset + args.chunk_size]
        permids = rows["instrument_permid"].tolist()
        run_id = f"{args.run_id}-SHARD-{number:02d}"
        request = {
            "run_id": run_id,
            "identity_row_ids": rows["identity_row_id"].tolist(),
            "local_codes": rows["local_code"].tolist(),
            "instrument_permids": permids,
            "extra_filter": quote_chain_filter(permids),
            "identity_plan_sha256": _sha256(plan_artifact.identity_plan_path),
            "country_codes": ["JPN"],
            "state_buckets": ["active", "inactive"],
            "top": args.top,
            "max_pages": 1,
        }
        request["request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        request["artifact_dir"] = str((Path(args.artifact_root) / run_id).resolve())
        shards.append(request)
    _atomic_csv(plan_artifact.artifact_dir / "shard_plan.csv", pd.DataFrame(shards))
    pending = [shard for shard in shards if not _completed(shard)]
    _write_state(plan_artifact.artifact_dir, shards, pending, "planned")
    if args.plan_only or not pending:
        print(f"identity_rows={len(plan)}")
        print(f"planned_shards={len(shards)}")
        print(f"completed_shards={len(shards)-len(pending)}")
        print(f"pending_shards={len(pending)}")
        print(f"plan_dir={plan_artifact.artifact_dir}")
        return

    adapter = RefinitivAdapter(session_name=args.session)
    try:
        adapter.open()
        health = adapter.ld.get_history(universe=[args.health_symbol], fields=["TRDPRC_1"], interval="1D")
        if health is None or health.dropna(how="all").empty:
            raise RuntimeError("Refinitiv health probe returned no rows")
        _write_state(plan_artifact.artifact_dir, shards, pending, "vendor_healthy")
        for shard in pending[: args.max_new_shards]:
            request_path = Path(shard["artifact_dir"]) / "quote_chain_request.json"
            _atomic_json(request_path, {**shard, "status": "running", "started_at_utc": datetime.now(UTC).isoformat()})
            try:
                artifact = write_refinitiv_security_master_discovery_artifacts(
                    discovery_client=adapter.ld,
                    get_data_client=adapter.ld,
                    artifact_root=args.artifact_root,
                    run_id=shard["run_id"],
                    country_codes=("JPN",),
                    state_buckets=("active", "inactive"),
                    top=args.top,
                    max_pages=1,
                    enrichment_chunk_size=args.enrichment_chunk_size,
                    sleep_seconds=args.sleep_seconds,
                    extra_filter=shard["extra_filter"],
                )
            except Exception as exc:
                _atomic_json(request_path, {**shard, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000], "failed_at_utc": datetime.now(UTC).isoformat()})
                remaining = [value for value in shards if not _completed(value)]
                _write_state(plan_artifact.artifact_dir, shards, remaining, "stopped_on_error")
                raise
            _atomic_json(request_path, {**shard, "status": "complete", "rows": int(artifact.manifest.get("rows", 0)), "manifest_sha256": _sha256(artifact.manifest_path), "completed_at_utc": datetime.now(UTC).isoformat()})
            remaining = [value for value in shards if not _completed(value)]
            _write_state(plan_artifact.artifact_dir, shards, remaining, "running")
    finally:
        adapter.close()
    remaining = [value for value in shards if not _completed(value)]
    _write_state(plan_artifact.artifact_dir, shards, remaining, "complete" if not remaining else "partial")
    print(f"identity_rows={len(plan)}")
    print(f"completed_shards={len(shards)-len(remaining)}")
    print(f"pending_shards={len(remaining)}")


def _default_security_master_dirs() -> list[str]:
    root = Path("artifacts/data_lake/pit/security_master_history")
    values = [str(root / "SECMASTER-DISCOVERY-JPN-MERGED-20260709")]
    values.extend(str(path) for path in sorted(root.glob("SECMASTER-DISCOVERY-JPN-G1-TARGETED-MISSING-20260712-V1-SHARD-*")) if (path / "security_master.csv").is_file())
    return values


def _completed(shard: dict[str, object]) -> bool:
    root = Path(str(shard["artifact_dir"]))
    try:
        request = json.loads((root / "quote_chain_request.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    output = root / "security_master.csv"
    return bool(
        request.get("status") == "complete"
        and request.get("request_sha256") == shard["request_sha256"]
        and request.get("manifest_sha256") == _sha256(root / "manifest.json")
        and output.is_file()
        and manifest.get("output_sha256", {}).get("security_master") == _sha256(output)
    )


def _write_state(root: Path, shards: list[dict[str, object]], pending: list[dict[str, object]], status: str) -> None:
    _atomic_json(root / "run_state.json", {"status": status, "planned_shards": len(shards), "completed_shards": len(shards)-len(pending), "pending_shards": len(pending), "pending_run_ids": [value["run_id"] for value in pending], "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False}, "updated_at_utc": datetime.now(UTC).isoformat()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


if __name__ == "__main__":
    main()
