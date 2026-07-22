from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc
DIRECT_RIC = re.compile(r"^(?P<code>[0-9A-Z]{4})(?:L)?\.(?:T|Q|OS|OJ)(?:\^|$)")


@dataclass(frozen=True)
class JapanG1QuoteChainValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    security_master_path: Path
    transition_plan_path: Path
    manifest: dict[str, Any]


def write_japan_g1_quote_chain_validation(
    *,
    run_id: str,
    materializer_dir: str | Path,
    identity_plan_dir: str | Path,
    shard_dirs: list[str | Path],
    artifact_root: str | Path = "artifacts/japan_g1_quote_chain_validation",
) -> JapanG1QuoteChainValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    materializer = Path(materializer_dir).resolve()
    materializer_manifest = _read_json(materializer / "manifest.json")
    candidate_path = materializer / "quote_chain_candidates.csv"
    selected_path = materializer / "selected_security_master_candidates.csv"
    if not (
        materializer_manifest.get("artifact_type") == "japan_g1_quote_chain_materializer_candidate"
        and materializer_manifest.get("status") == "complete_candidate"
        and materializer_manifest.get("scorecard_eligible") is False
        and materializer_manifest.get("output_sha256", {}).get(candidate_path.name) == _sha256(candidate_path)
        and materializer_manifest.get("output_sha256", {}).get(selected_path.name) == _sha256(selected_path)
    ):
        raise ValueError("quote-chain materializer contract invalid")
    plan_dir = Path(identity_plan_dir).resolve()
    plan_manifest = _read_json(plan_dir / "manifest.json")
    plan_path = plan_dir / "identity_plan.csv"
    if plan_manifest.get("output_sha256", {}).get(plan_path.name) != _sha256(plan_path):
        raise ValueError("identity plan hash invalid")
    plan = pd.read_csv(plan_path, dtype=str).fillna("")
    producer_candidates = pd.read_csv(candidate_path, dtype=str).fillna("")

    frames: list[pd.DataFrame] = []
    input_paths = {
        "materializer": str(materializer / "manifest.json"),
        "identity_plan": str(plan_dir / "manifest.json"),
    }
    input_hashes = {key: _sha256(Path(path)) for key, path in input_paths.items()}
    seen_rows: set[str] = set()
    for index, raw in enumerate(shard_dirs):
        shard = Path(raw).resolve()
        request = _read_json(shard / "quote_chain_request.json")
        manifest_path = shard / "manifest.json"
        manifest = _read_json(manifest_path)
        output = shard / "security_master.csv"
        if not (
            request.get("status") == "complete"
            and request.get("identity_plan_sha256") == _sha256(plan_path)
            and request.get("country_codes") == ["JPN"]
            and request.get("state_buckets") == ["active", "inactive"]
            and request.get("manifest_sha256") == _sha256(manifest_path)
            and manifest.get("output_sha256", {}).get("security_master") == _sha256(output)
        ):
            raise ValueError(f"quote-chain shard evidence invalid: {shard}")
        seen_rows.update(str(value) for value in request.get("identity_row_ids", []))
        frame = pd.read_csv(output, dtype=str).fillna("")
        frame["validated_source_shard"] = request["run_id"]
        frames.append(frame)
        key = f"shard_{index:02d}"
        input_paths[key] = str(manifest_path)
        input_hashes[key] = _sha256(manifest_path)
    if seen_rows != set(plan["identity_row_id"]):
        raise ValueError("quote-chain shard denominator does not equal identity plan")
    quotes = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(
        ["instrument_permid", "ric"], keep="last"
    )

    admitted: list[dict[str, Any]] = []
    admitted_identity_rows = 0
    transitions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    producer_by_id = producer_candidates.set_index("identity_row_id", drop=False)
    for identity in plan.to_dict("records"):
        chain = quotes.loc[quotes["instrument_permid"].eq(identity["instrument_permid"])].copy()
        chain["direct_code"] = chain["ric"].map(
            lambda value: DIRECT_RIC.match(str(value)).group("code") if DIRECT_RIC.match(str(value)) else ""
        )
        direct = chain.loc[
            chain["direct_code"].eq(identity["local_code"])
            & chain["ticker_symbol"].eq(identity["local_code"])
            & chain["exchange_code"].isin({"TYO", "JSD", "OSA"})
        ].copy()
        direct["exchange_rank"] = direct["exchange_code"].map({"TYO": 0, "JSD": 1, "OSA": 2})
        direct["dead_rank"] = direct["ric"].str.contains(r"\^").astype(int)
        direct = direct.sort_values(["exchange_rank", "dead_rank", "ric"])
        producer = producer_by_id.loc[identity["identity_row_id"]]
        if not direct.empty:
            passed = (
                producer["mapping_route"] == "direct_tyo_candidate"
                and producer["selected_ric"] in set(direct["ric"])
                and direct["instrument_permid"].astype(str).eq(identity["instrument_permid"]).all()
            )
            observed = "|".join(sorted(set(direct["ric"])))
            diagnostics.append({"identity_row_id": identity["identity_row_id"], "local_code": identity["local_code"], "check": "direct_route_recomputed", "passed": passed, "observed": observed})
            if not passed:
                raise ValueError(f"direct quote-chain candidate mismatch: {identity['identity_row_id']}")
            admitted_identity_rows += 1
            for _, chosen in direct.iterrows():
                record = chosen.to_dict()
                record.update({"quote_chain_validation_run_id": run_id, "quote_chain_identity_row_id": identity["identity_row_id"], "quote_chain_mapping_route": "independently_validated_direct_code", "mapping_local_code_override": identity["local_code"], "quote_chain_admitted": True})
                admitted.append(record)
        successor_rics = [value for value in str(producer["successor_primary_rics"]).split("|") if value]
        successor_codes = [value for value in str(producer["successor_codes"]).split("|") if value]
        if successor_rics:
            if len(successor_codes) != 1:
                raise ValueError(f"successor code ambiguity: {identity['identity_row_id']}")
            transitions.append({"identity_row_id": identity["identity_row_id"], "local_code": identity["local_code"], "instrument_permid": identity["instrument_permid"], "isins": identity["isins"], "direct_ric_candidates": "|".join(sorted(set(direct["ric"]))) if not direct.empty else "", "candidate_successor_code": successor_codes[0], "candidate_successor_ric": sorted(successor_rics)[0], "status": "pending_official_jpx_event_chain_validation"})
            diagnostics.append({"identity_row_id": identity["identity_row_id"], "local_code": identity["local_code"], "check": "successor_route_quarantined", "passed": True, "observed": "|".join(sorted(successor_rics))})

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    security_path = root / "security_master.csv"
    transition_path = root / "pending_transition_plan.csv"
    diagnostic_path = root / "diagnostics.csv"
    _atomic_csv(security_path, pd.DataFrame(admitted))
    _atomic_csv(transition_path, pd.DataFrame(transitions))
    _atomic_csv(diagnostic_path, pd.DataFrame(diagnostics))
    manifest = {
        "schema_version": "1.0-independent-partial",
        "run_id": run_id,
        "artifact_type": "japan_g1_quote_chain_independent_validation",
        "status": "pass_partial_component",
        "validation_status": "pass",
        "independent_validation": True,
        "scorecard_eligible": False,
        "admitted_direct_identity_rows": admitted_identity_rows,
        "admitted_security_master_rows": len(admitted),
        "pending_transition_rows": len(transitions),
        "critical_blockers": ["successor_routes_require_official_jpx_event_chain_validation"] if transitions else [],
        "input_manifests": input_paths,
        "input_manifest_sha256": input_hashes,
        "artifacts": {
            "security_master": str(security_path),
            "pending_transition_plan": str(transition_path),
            "diagnostics": str(diagnostic_path),
        },
        "output_sha256": {
            "security_master": _sha256(security_path),
            transition_path.name: _sha256(transition_path),
            diagnostic_path.name: _sha256(diagnostic_path),
        },
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanG1QuoteChainValidationArtifacts(run_id, root, manifest_path, security_path, transition_path, manifest)


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
