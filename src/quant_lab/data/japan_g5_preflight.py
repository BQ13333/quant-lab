from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_lab.agents.japan_data_quality_loop import (
    JapanDataQualityPaths,
    required_japan_hash_inputs,
)


@dataclass(frozen=True)
class JapanG5PreflightArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    immutable_hashes_path: Path
    upstream_readiness_path: Path
    timestamp_audit_path: Path
    hash_coverage_path: Path
    missing_inputs_path: Path
    upstream_readiness: pd.DataFrame
    timestamp_audit: pd.DataFrame
    hash_coverage: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _child(directory: Path | None, name: str) -> Path:
    return directory / name if directory is not None else Path("/__japan_g5_missing__") / name


def _full_artifact(manifest: dict[str, Any]) -> bool:
    return bool(
        str(manifest.get("status", "")).lower() in {"complete", "pass", "accepted_backtest"}
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
        and not any(manifest.get(flag) is True for flag in ("partial", "smoke", "current_survivor"))
    )


def _as_nonnegative_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return max(0, int(numeric))


def _upstream_readiness(paths: JapanDataQualityPaths) -> pd.DataFrame:
    checks = [
        (
            "g1_security_master",
            _child(paths.security_master_dir, "manifest.json"),
            "Full independently validated listing-level security master",
        ),
        (
            "g1_validation",
            _child(paths.g1_validation_dir, "manifest.json"),
            "Independent G1 validation",
        ),
        (
            "g2_ohlcv",
            _child(paths.historical_dir, "historical_run_manifest.json"),
            "Full terminal OHLCV coverage",
        ),
        (
            "g2_corporate_actions",
            _child(paths.corporate_actions_dir, "manifest.json"),
            "Official corporate actions",
        ),
        (
            "g2_liquidity",
            _child(paths.liquidity_dir, "manifest.json"),
            "PIT liquidity and capacity history",
        ),
        (
            "g3_fundamentals_ibes_events",
            _child(paths.fundamentals_ibes_events_dir, "manifest.json"),
            "Full PIT fundamentals, IBES, and earnings history",
        ),
        (
            "g4_calendar_halts_tca",
            _child(paths.fx_calendar_tca_dir, "manifest.json"),
            "Full JPX calendar, halts, market rules, and TCA",
        ),
    ]
    rows = []
    for component, path, requirement in checks:
        manifest = _read_json(path)
        rows.append(
            {
                "component": component,
                "requirement": requirement,
                "manifest_path": str(path.resolve()),
                "manifest_exists": path.exists(),
                "status": str(manifest.get("status", "missing")),
                "scorecard_eligible": manifest.get("scorecard_eligible") is True,
                "ready": _full_artifact(manifest),
            }
        )
    return pd.DataFrame(rows)


def _timestamp_audit(qa_manifest_paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in qa_manifest_paths:
        manifest = _read_json(path)
        known_at = pd.to_numeric(manifest.get("known_at_coverage"), errors="coerce")
        violations = pd.to_numeric(manifest.get("pit_timestamp_violations"), errors="coerce")
        history_rows = _as_nonnegative_int(manifest.get("history_rows"))
        if history_rows == 0:
            history_rows = sum(
                _as_nonnegative_int(manifest.get(key))
                for key in ("consensus_rows", "revision_event_rows", "earnings_event_rows")
            )
        rows.append(
            {
                "artifact": str(path.parent.resolve()),
                "manifest_exists": path.exists(),
                "status": str(manifest.get("status", "missing")),
                "evidence_scope": str(manifest.get("evidence_scope", "")),
                "scorecard_eligible": manifest.get("scorecard_eligible") is True,
                "history_rows": history_rows,
                "known_at_coverage": known_at,
                "pit_timestamp_violations": violations,
                "predecision_timestamp_pass": bool(
                    path.exists()
                    and history_rows > 0
                    and known_at == 1.0
                    and violations == 0
                ),
                "decision_calendar_leakage_pass": False,
            }
        )
    return pd.DataFrame(rows)


def write_japan_g5_preflight_artifacts(
    paths: JapanDataQualityPaths,
    run_id: str,
    qa_manifest_paths: Sequence[str | Path] = (),
) -> JapanG5PreflightArtifacts:
    if paths.leakage_qa_dir is None:
        raise ValueError("Japan leakage QA directory is required")
    artifact_dir = paths.leakage_qa_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    upstream_path = artifact_dir / "upstream_readiness.csv"
    timestamp_path = artifact_dir / "predecision_timestamp_audit.csv"
    hash_coverage_path = artifact_dir / "hash_coverage.csv"
    missing_inputs_path = artifact_dir / "missing_inputs.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    immutable_path = paths.immutable_hashes_path or artifact_dir / "immutable_hashes.json"

    upstream = _upstream_readiness(paths)
    timestamp_audit = _timestamp_audit([Path(path) for path in qa_manifest_paths])
    upstream.to_csv(upstream_path, index=False)
    timestamp_audit.to_csv(timestamp_path, index=False)

    upstream_blockers = upstream.loc[~upstream["ready"], "component"].astype(str).tolist()
    timestamp_pass = bool(
        not timestamp_audit.empty and timestamp_audit["predecision_timestamp_pass"].all()
    )
    blockers = [f"upstream_not_ready:{component}" for component in upstream_blockers]
    blockers.extend(
        [
            "decision_calendar_unavailable_until_admitted_g2_ohlcv",
            "survivorship_audit_unavailable_until_admitted_g1",
            "train_test_overlap_audit_unavailable_until_research_split_is_frozen",
        ]
    )
    if not timestamp_pass:
        blockers.append("predecision_timestamp_qa_incomplete_or_failed")

    summary_path.write_text(
        "\n".join(
            [
                f"# {run_id}",
                "",
                "## Status",
                "- G5 is blocked upstream and receives no scorecard credit.",
                f"- Ready upstream components: `{int(upstream['ready'].sum())}/{len(upstream)}`.",
                f"- Predecision timestamp QA passed: `{timestamp_pass}`.",
                "- Decision-calendar leakage, survivorship, and train/test overlap remain untested.",
                "- Alpha, production, IBKR, and live execution remain blocked.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "blocked_upstream",
        "scope": "japan_g5_preflight_not_decision_calendar_audit",
        "evidence_scope": "pre_admission",
        "scorecard_eligible": False,
        "partial": True,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "decision_rows": 0,
        "decision_entities": 0,
        "features": [],
        "known_at_coverage": float(timestamp_audit["known_at_coverage"].min())
        if timestamp_pass
        else 0.0,
        "total_leakage_violations": -1,
        "future_data_violations": -1,
        "train_test_overlap_rows": -1,
        "survivorship_violations": -1,
        "upstream_components_ready": int(upstream["ready"].sum()),
        "upstream_components_total": len(upstream),
        "predecision_timestamp_pass": timestamp_pass,
        "critical_blockers": blockers,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "upstream_readiness": str(upstream_path.resolve()),
            "predecision_timestamp_audit": str(timestamp_path.resolve()),
            "hash_coverage": str(hash_coverage_path.resolve()),
            "missing_inputs": str(missing_inputs_path.resolve()),
            "summary": str(summary_path.resolve()),
            "immutable_hashes": str(immutable_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    required = required_japan_hash_inputs(paths)
    hash_rows = []
    files: dict[str, str] = {}
    for required_path in required:
        resolved = required_path.resolve()
        exists = resolved.is_file()
        digest = _sha256(resolved) if exists else ""
        hash_rows.append(
            {
                "required_path": str(resolved),
                "exists": exists,
                "sha256": digest,
            }
        )
        if exists:
            files[str(resolved)] = digest
    hash_coverage = pd.DataFrame(hash_rows)
    hash_coverage.to_csv(hash_coverage_path, index=False)
    missing = hash_coverage.loc[~hash_coverage["exists"], ["required_path"]].copy()
    missing["reason"] = "required_gate_input_missing"
    missing.to_csv(missing_inputs_path, index=False)
    coverage_complete = bool(not hash_coverage.empty and hash_coverage["exists"].all())
    inventory = {
        "run_id": f"{run_id}-IMMUTABLE-INVENTORY",
        "status": "pass" if coverage_complete and not blockers else "incomplete",
        "scorecard_eligible": bool(coverage_complete and not blockers),
        "evidence_scope": "full" if coverage_complete and not blockers else "available_gate_inputs_only",
        "algorithm": "sha256",
        "immutable": True,
        "coverage_complete": coverage_complete,
        "required_files": len(hash_coverage),
        "hashed_files": int(hash_coverage["exists"].sum()),
        "missing_files": int((~hash_coverage["exists"]).sum()),
        "files": files,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    immutable_path.parent.mkdir(parents=True, exist_ok=True)
    immutable_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return JapanG5PreflightArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        immutable_hashes_path=immutable_path,
        upstream_readiness_path=upstream_path,
        timestamp_audit_path=timestamp_path,
        hash_coverage_path=hash_coverage_path,
        missing_inputs_path=missing_inputs_path,
        upstream_readiness=upstream,
        timestamp_audit=timestamp_audit,
        hash_coverage=hash_coverage,
    )
