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
REQUIRED_MASTER_COLUMNS = {
    "ric",
    "ticker_symbol",
    "instrument_permid",
    "isin",
    "exchange_code",
}


@dataclass(frozen=True)
class JapanG1QuoteChainPlanArtifacts:
    run_id: str
    artifact_dir: Path
    identity_plan_path: Path
    unanchored_codes_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_g1_quote_chain_plan(
    *,
    run_id: str,
    missing_plan_path: str | Path,
    security_master_paths: list[str | Path],
    artifact_root: str | Path = "artifacts/japan_g1_quote_chain_discovery",
) -> JapanG1QuoteChainPlanArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    missing_path = Path(missing_plan_path).resolve()
    missing = pd.read_csv(missing_path, dtype=str).fillna("")
    if "local_code" not in missing:
        raise ValueError("missing plan lacks local_code")
    codes = sorted(code for code in missing["local_code"].unique() if code)
    if not codes:
        raise ValueError("missing plan has no unresolved codes")

    source_paths = [Path(path).resolve() for path in security_master_paths]
    if not source_paths:
        raise ValueError("at least one security master is required")
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for index, path in enumerate(source_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype=str).fillna("")
        missing_columns = REQUIRED_MASTER_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(f"security master missing columns: {sorted(missing_columns)}")
        frame = frame.loc[
            frame["ticker_symbol"].isin(codes) & frame["instrument_permid"].ne("")
        ].copy()
        frame["source_index"] = index
        frame["source_path"] = str(path)
        frames.append(frame)
        source_hashes[f"security_master_{index:02d}"] = _sha256(path)

    evidence = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not evidence.empty:
        for (code, permid), group in evidence.groupby(
            ["ticker_symbol", "instrument_permid"], sort=True
        ):
            isins = sorted(value for value in group["isin"].unique() if value)
            payload = f"{code}|{permid}|{'|'.join(isins)}"
            rows.append(
                {
                    "identity_row_id": "JQC-" + hashlib.sha256(payload.encode()).hexdigest()[:16],
                    "local_code": code,
                    "instrument_permid": permid,
                    "isins": "|".join(isins),
                    "evidence_rics": "|".join(sorted(set(group["ric"]))),
                    "evidence_exchange_codes": "|".join(
                        sorted(value for value in set(group["exchange_code"]) if value)
                    ),
                    "source_paths": "|".join(sorted(set(group["source_path"]))),
                }
            )
    plan = pd.DataFrame(
        rows,
        columns=[
            "identity_row_id",
            "local_code",
            "instrument_permid",
            "isins",
            "evidence_rics",
            "evidence_exchange_codes",
            "source_paths",
        ],
    ).sort_values(["local_code", "instrument_permid"]).reset_index(drop=True)
    anchored_codes = set(plan["local_code"]) if not plan.empty else set()
    unanchored = pd.DataFrame(
        {"local_code": sorted(set(codes) - anchored_codes), "reason": "no_instrument_permid_evidence"}
    )

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "identity_plan.csv"
    unanchored_path = root / "unanchored_codes.csv"
    _atomic_csv(plan_path, plan)
    _atomic_csv(unanchored_path, unanchored)
    manifest = {
        "schema_version": "1.0-identity-plan",
        "run_id": run_id,
        "artifact_type": "japan_g1_quote_chain_identity_plan",
        "status": "planned",
        "scorecard_eligible": False,
        "missing_plan": str(missing_path),
        "missing_plan_sha256": _sha256(missing_path),
        "security_master_inputs": [str(path) for path in source_paths],
        "security_master_input_sha256": source_hashes,
        "counts": {
            "unresolved_codes": len(codes),
            "anchored_codes": len(anchored_codes),
            "identity_rows": len(plan),
            "unanchored_codes": len(unanchored),
        },
        "artifacts": {
            "identity_plan": str(plan_path),
            "unanchored_codes": str(unanchored_path),
        },
        "output_sha256": {
            plan_path.name: _sha256(plan_path),
            unanchored_path.name: _sha256(unanchored_path),
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
    return JapanG1QuoteChainPlanArtifacts(
        run_id, root, plan_path, unanchored_path, manifest_path, manifest
    )


def quote_chain_filter(instrument_permids: list[str]) -> str:
    values = sorted({str(value).strip() for value in instrument_permids if str(value).strip()})
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("instrument PermIDs must be non-empty decimal identifiers")
    return " or ".join(f"InstrumentPermID eq '{value}'" for value in values)


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
