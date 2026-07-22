from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.security_master_discovery import (
    write_refinitiv_security_master_discovery_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


UTC = timezone.utc


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"lseg\.data\..*")
    parser = argparse.ArgumentParser(description="Run bounded Refinitiv Discovery shards for unresolved Japan G1 codes.")
    parser.add_argument(
        "--missing-plan",
        default="artifacts/japan_g1_monthly_materializer/JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1/targeted_missing_code_plan.csv",
    )
    parser.add_argument("--run-id-prefix", default="SECMASTER-DISCOVERY-JPN-G1-TARGETED-MISSING-20260712-V1")
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-codes", type=int, default=100)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--enrichment-chunk-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--health-symbol", default="7203.T")
    parser.add_argument(
        "--max-new-shards",
        type=int,
        default=None,
        help="Execute at most this many pending shards while retaining the full immutable plan.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.max_codes <= 0:
        raise SystemExit("chunk-size and max-codes must be positive")
    if args.max_new_shards is not None and args.max_new_shards <= 0:
        raise SystemExit("max-new-shards must be positive when supplied")
    plan = pd.read_csv(Path(args.missing_plan), dtype=str).fillna("")
    codes = sorted(code for code in plan["local_code"].unique() if code)[: args.max_codes]
    if not codes:
        print("No unresolved codes remain.")
        return

    artifact_root = Path(args.artifact_root)
    control_dir = artifact_root / f"{args.run_id_prefix}-CONTROL"
    control_dir.mkdir(parents=True, exist_ok=True)
    input_plan_sha256 = _sha256(Path(args.missing_plan))
    shards = []
    for shard, offset in enumerate(range(0, len(codes), args.chunk_size), start=1):
        selected = codes[offset : offset + args.chunk_size]
        extra_filter = " or ".join(f"TickerSymbol eq '{code}'" for code in selected)
        run_id = f"{args.run_id_prefix}-SHARD-{shard:02d}"
        request = {
            "run_id": run_id,
            "codes": selected,
            "country_codes": ["JPN"],
            "state_buckets": ["active", "inactive"],
            "extra_filter": extra_filter,
            "top": args.top,
            "max_pages": 1,
            "input_plan_sha256": input_plan_sha256,
        }
        request_sha256 = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        shards.append(
            {
                **request,
                "request_sha256": request_sha256,
                "artifact_dir": str((artifact_root / run_id).resolve()),
            }
        )
    _atomic_write_csv(control_dir / "shard_plan.csv", pd.DataFrame(shards))
    pending = [shard for shard in shards if not _completed_shard(shard)]
    _write_control_state(control_dir, input_plan_sha256, shards, pending, "planned")
    if args.plan_only:
        print(f"requested_codes={len(codes)}")
        print(f"planned_shards={len(shards)}")
        print(f"pending_shards={len(pending)}")
        print(f"control_dir={control_dir.resolve()}")
        return
    if not pending:
        print(f"requested_codes={len(codes)}")
        print(f"completed_shards={len(shards)}")
        print("pending_shards=0")
        return

    selected_pending = (
        pending[: args.max_new_shards]
        if args.max_new_shards is not None
        else pending
    )

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    try:
        adapter.open()
    except Exception:
        _write_control_state(control_dir, input_plan_sha256, shards, pending, "blocked_vendor_health")
        raise
    artifacts = []
    try:
        try:
            health = adapter.ld.get_history(
                universe=[args.health_symbol], fields=["TRDPRC_1"], interval="1D"
            )
        except Exception:
            _write_control_state(control_dir, input_plan_sha256, shards, pending, "blocked_vendor_health")
            raise
        if health is None or health.dropna(how="all").empty:
            _write_control_state(control_dir, input_plan_sha256, shards, pending, "blocked_vendor_health")
            raise RuntimeError(f"Refinitiv health probe returned no rows for {args.health_symbol}")
        _write_control_state(control_dir, input_plan_sha256, shards, pending, "vendor_healthy")
        for shard in selected_pending:
            request_path = Path(shard["artifact_dir"]) / "targeted_request.json"
            _atomic_write_json(
                request_path,
                {**shard, "status": "running", "started_at_utc": datetime.now(UTC).isoformat()},
            )
            try:
                artifact = write_refinitiv_security_master_discovery_artifacts(
                    discovery_client=adapter.ld,
                    get_data_client=adapter.ld,
                    artifact_root=artifact_root,
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
                _atomic_write_json(
                    request_path,
                    {
                        **shard,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "failed_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                remaining = [candidate for candidate in shards if not _completed_shard(candidate)]
                _write_control_state(control_dir, input_plan_sha256, shards, remaining, "stopped_on_error")
                raise
            _atomic_write_json(
                request_path,
                {
                    **shard,
                    "status": "complete",
                    "rows": int(artifact.manifest.get("rows", 0)),
                    "manifest_sha256": _sha256(artifact.manifest_path),
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            artifacts.append(artifact)
            remaining = [candidate for candidate in shards if not _completed_shard(candidate)]
            _write_control_state(control_dir, input_plan_sha256, shards, remaining, "running")
    finally:
        adapter.close()
    remaining = [candidate for candidate in shards if not _completed_shard(candidate)]
    _write_control_state(
        control_dir,
        input_plan_sha256,
        shards,
        remaining,
        "complete" if not remaining else "partial",
    )
    print(f"requested_codes={len(codes)}")
    print(f"completed_shards={len(shards) - len(remaining)}")
    print(f"pending_shards={len(remaining)}")
    for artifact in artifacts:
        print(f"{artifact.run_id}: status={artifact.manifest['status']} rows={artifact.manifest['rows']} dir={artifact.artifact_dir.resolve()}")


def _completed_shard(shard: dict[str, object]) -> bool:
    artifact_dir = Path(str(shard["artifact_dir"]))
    request_path = artifact_dir / "targeted_request.json"
    manifest_path = artifact_dir / "manifest.json"
    security_master_path = artifact_dir / "security_master.csv"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if request.get("status") != "complete" or request.get("request_sha256") != shard["request_sha256"]:
        return False
    if not security_master_path.is_file() or manifest.get("artifacts", {}).get("security_master") != str(security_master_path.resolve()):
        return False
    expected = manifest.get("output_sha256", {}).get("security_master")
    return bool(expected) and expected == _sha256(security_master_path)


def _write_control_state(
    control_dir: Path,
    input_plan_sha256: str,
    shards: list[dict[str, object]],
    pending: list[dict[str, object]],
    status: str,
) -> None:
    _atomic_write_json(
        control_dir / "run_state.json",
        {
            "status": status,
            "input_plan_sha256": input_plan_sha256,
            "planned_shards": len(shards),
            "completed_shards": len(shards) - len(pending),
            "pending_shards": len(pending),
            "pending_run_ids": [str(shard["run_id"]) for shard in pending],
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
