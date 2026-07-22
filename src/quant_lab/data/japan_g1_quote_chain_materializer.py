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
PRIMARY_EXCHANGES = {"TYO", "JSD", "OSA"}


@dataclass(frozen=True)
class JapanG1QuoteChainMaterializerArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    candidates_path: Path
    selected_security_master_path: Path
    manifest: dict[str, Any]


def write_japan_g1_quote_chain_materializer(
    *,
    run_id: str,
    identity_plan_dir: str | Path,
    shard_dirs: list[str | Path],
    artifact_root: str | Path = "artifacts/japan_g1_quote_chain_materializer",
) -> JapanG1QuoteChainMaterializerArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    plan_dir = Path(identity_plan_dir).resolve()
    plan_manifest = _read_json(plan_dir / "manifest.json")
    plan_path = plan_dir / "identity_plan.csv"
    if (
        plan_manifest.get("artifact_type") != "japan_g1_quote_chain_identity_plan"
        or plan_manifest.get("output_sha256", {}).get("identity_plan.csv") != _sha256(plan_path)
    ):
        raise ValueError("identity plan contract or hash is invalid")
    plan = pd.read_csv(plan_path, dtype=str).fillna("")

    frames: list[pd.DataFrame] = []
    input_manifest_paths: dict[str, str] = {"identity_plan": str(plan_dir / "manifest.json")}
    input_manifest_hashes: dict[str, str] = {"identity_plan": _sha256(plan_dir / "manifest.json")}
    seen_identity_rows: set[str] = set()
    for index, raw_dir in enumerate(shard_dirs):
        shard_dir = Path(raw_dir).resolve()
        request_path = shard_dir / "quote_chain_request.json"
        manifest_path = shard_dir / "manifest.json"
        output_path = shard_dir / "security_master.csv"
        request = _read_json(request_path)
        manifest = _read_json(manifest_path)
        _validate_shard(
            request=request,
            manifest=manifest,
            request_path=request_path,
            manifest_path=manifest_path,
            output_path=output_path,
            identity_plan_sha256=_sha256(plan_path),
        )
        frame = pd.read_csv(output_path, dtype=str).fillna("")
        frame["quote_chain_shard_run_id"] = request["run_id"]
        frame["quote_chain_request_sha256"] = request["request_sha256"]
        frames.append(frame)
        seen_identity_rows.update(str(value) for value in request["identity_row_ids"])
        key = f"quote_chain_shard_{index:02d}"
        input_manifest_paths[key] = str(manifest_path)
        input_manifest_hashes[key] = _sha256(manifest_path)

    expected_identity_rows = set(plan["identity_row_id"])
    if seen_identity_rows != expected_identity_rows:
        missing = sorted(expected_identity_rows - seen_identity_rows)
        unexpected = sorted(seen_identity_rows - expected_identity_rows)
        raise ValueError(f"quote-chain identity denominator mismatch missing={missing} unexpected={unexpected}")
    quotes = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(
        ["instrument_permid", "ric"], keep="last"
    )
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for identity in plan.to_dict("records"):
        chain = quotes.loc[quotes["instrument_permid"].eq(identity["instrument_permid"])].copy()
        primary_market = chain.loc[chain["exchange_code"].isin(PRIMARY_EXCHANGES)].copy()
        direct = primary_market.loc[
            primary_market["ticker_symbol"].eq(identity["local_code"])
            & primary_market["ric"].str.startswith(f"{identity['local_code']}.")
        ].sort_values(["is_primary_ric", "is_primary_issue_ric", "ric"], ascending=[False, False, True])
        successor = primary_market.loc[
            ~primary_market["ticker_symbol"].eq(identity["local_code"])
            & (
                primary_market["is_primary_ric"].str.lower().eq("true")
                | primary_market["is_primary_issue_ric"].str.lower().eq("true")
            )
        ].sort_values(["is_primary_ric", "is_primary_issue_ric", "ric"], ascending=[False, False, True])
        selected = direct.iloc[0] if not direct.empty else successor.iloc[0] if not successor.empty else None
        route = "direct_tyo_candidate" if not direct.empty else "identity_successor_candidate" if not successor.empty else "unresolved"
        selected_ric = str(selected["ric"]) if selected is not None else ""
        selected_code = str(selected["ticker_symbol"]) if selected is not None else ""
        candidate_rows.append(
            {
                **identity,
                "chain_rows": len(chain),
                "primary_market_rows": len(primary_market),
                "direct_primary_market_rics": "|".join(sorted(set(direct["ric"]))),
                "successor_primary_rics": "|".join(sorted(set(successor["ric"]))),
                "successor_codes": "|".join(sorted(value for value in set(successor["ticker_symbol"]) if value)),
                "selected_ric": selected_ric,
                "selected_code": selected_code,
                "mapping_route": route,
                "identity_tied": selected is not None and str(selected["instrument_permid"]) == identity["instrument_permid"],
                "admission_status": "candidate_requires_independent_event_validation" if selected is not None else "unresolved",
            }
        )
        if selected is not None:
            record = selected.to_dict()
            record.update(
                {
                    "mapping_local_code_override": identity["local_code"],
                    "quote_chain_identity_row_id": identity["identity_row_id"],
                    "quote_chain_mapping_route": route,
                    "quote_chain_admitted": False,
                    "quote_chain_blocker": "independent_official_event_validation_required",
                }
            )
            selected_rows.append(record)

    candidates = pd.DataFrame(candidate_rows)
    selected_master = pd.DataFrame(selected_rows)
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "quote_chain_candidates.csv"
    selected_path = root / "selected_security_master_candidates.csv"
    _atomic_csv(candidates_path, candidates)
    _atomic_csv(selected_path, selected_master)
    unresolved = int(candidates["selected_ric"].eq("").sum())
    counts = candidates["mapping_route"].value_counts().to_dict()
    manifest = {
        "schema_version": "1.0-candidate-only",
        "run_id": run_id,
        "artifact_type": "japan_g1_quote_chain_materializer_candidate",
        "status": "complete_candidate" if not unresolved else "complete_with_blockers",
        "scorecard_eligible": False,
        "independent_validation_required": True,
        "identity_plan_rows": len(plan),
        "unique_quote_rows": len(quotes),
        "selected_candidate_rows": len(selected_master),
        "unresolved_identity_rows": unresolved,
        "mapping_route_counts": {str(key): int(value) for key, value in counts.items()},
        "critical_blockers": ["independent_official_event_validation_required"],
        "input_manifests": input_manifest_paths,
        "input_manifest_sha256": input_manifest_hashes,
        "artifacts": {
            "quote_chain_candidates": str(candidates_path),
            "selected_security_master_candidates": str(selected_path),
        },
        "output_sha256": {
            candidates_path.name: _sha256(candidates_path),
            selected_path.name: _sha256(selected_path),
        },
        "promotion_policy": {
            "alpha": False,
            "historical_backtest": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG1QuoteChainMaterializerArtifacts(
        run_id, root, manifest_path, candidates_path, selected_path, manifest
    )


def _validate_shard(
    *,
    request: dict[str, Any],
    manifest: dict[str, Any],
    request_path: Path,
    manifest_path: Path,
    output_path: Path,
    identity_plan_sha256: str,
) -> None:
    request_payload = {key: value for key, value in request.items() if key not in {"status", "started_at_utc", "completed_at_utc", "rows", "manifest_sha256", "artifact_dir"}}
    recorded_hash = request_payload.pop("request_sha256", "")
    calculated_hash = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    valid = (
        request.get("status") == "complete"
        and recorded_hash == calculated_hash
        and request.get("identity_plan_sha256") == identity_plan_sha256
        and request.get("country_codes") == ["JPN"]
        and request.get("state_buckets") == ["active", "inactive"]
        and request.get("manifest_sha256") == _sha256(manifest_path)
        and output_path.is_file()
        and manifest.get("output_sha256", {}).get("security_master") == _sha256(output_path)
    )
    if not valid:
        raise ValueError(f"quote-chain shard contract invalid: {request_path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
