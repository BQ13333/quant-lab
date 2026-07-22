from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_lab.research.malaysia_strategy_registry import malaysia_atomic_cohort


PROTOCOL_ID = "QFORGE20-MYS-PIT-REPLICATION-14-FROZEN-V1"
EXPECTED_STRATEGY_COUNT = 14
MALAYSIA = "Malaysia"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol, repo_root=protocol_path.resolve().parents[1])
    return protocol


def validate_frozen_protocol(protocol: Mapping[str, Any], *, repo_root: Path) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected Malaysia PIT replication protocol identity")
    if protocol.get("market") != MALAYSIA or protocol.get("exchange") != "Bursa Malaysia":
        raise ValueError("PIT replication protocol must remain Malaysia / Bursa only")
    strategies = protocol.get("strategies")
    if not isinstance(strategies, list) or len(strategies) != EXPECTED_STRATEGY_COUNT:
        raise ValueError("PIT replication protocol must contain exactly fourteen strategies")
    if [item.get("sequence") for item in strategies] != list(range(1, 15)):
        raise ValueError("PIT replication strategy order drifted")
    strategy_ids = [str(item.get("strategy_id", "")) for item in strategies]
    if len(set(strategy_ids)) != EXPECTED_STRATEGY_COUNT:
        raise ValueError("PIT replication strategy identities must be unique")

    source_identity = protocol["source_identity"]
    for path_key, hash_key in (
        ("atomic_registry_path", "atomic_registry_sha256"),
        ("next_cohort_protocol_path", "next_cohort_protocol_file_sha256"),
        ("specialist_roster_path", "specialist_roster_sha256"),
    ):
        source_path = repo_root / str(source_identity[path_key])
        if not source_path.is_file() or file_sha256(source_path) != source_identity[hash_key]:
            raise ValueError(f"frozen source identity drifted: {path_key}")

    atomic = malaysia_atomic_cohort()
    for frozen, source in zip(strategies[:10], atomic, strict=True):
        if frozen["strategy_id"] != source.strategy_id:
            raise ValueError("atomic strategy identity drifted")
        if frozen["family"] != source.family or frozen["spec_sha256"] != source.content_hash():
            raise ValueError(f"atomic strategy specification drifted: {source.strategy_id}")

    next_path = repo_root / str(source_identity["next_cohort_protocol_path"])
    next_protocol = json.loads(next_path.read_text(encoding="utf-8"))
    active = {item["strategy_id"]: item for item in next_protocol["hypotheses"]}
    for frozen in strategies[10:]:
        source = active.get(frozen["strategy_id"])
        if source is None:
            raise ValueError(f"next-cohort strategy disappeared: {frozen['strategy_id']}")
        if frozen["family"] != source["mathematical_cluster"]:
            raise ValueError(f"next-cohort family drifted: {frozen['strategy_id']}")
        if frozen["spec_sha256"] != object_sha256(source):
            raise ValueError(f"next-cohort specification drifted: {frozen['strategy_id']}")

    contract = protocol["research_contract"]
    required_false = ("no_parameter_grid", "no_sign_flip", "no_post_hoc_combination")
    if not all(contract.get(item) is True for item in required_false):
        raise ValueError("anti-optimization controls must remain enabled")
    stopping = protocol["stopping_rules"]
    if stopping.get("dry_run_authorized") is not False:
        raise ValueError("dry run must remain unauthorized")
    if stopping.get("live_execution_authorized") is not False:
        raise ValueError("live execution must remain unauthorized")


@dataclass(frozen=True)
class PITCoverageInputs:
    source_hashes_match: bool
    production_adjusted_returns_ready: bool
    historical_u2_membership_reconstructable: bool
    malaysia_listing_rows: int
    malaysia_inactive_rows: int
    malaysia_current_feature_tickers: int
    malaysia_current_adjusted_return_tickers: int
    malaysia_u2_current_tickers: int
    malaysia_u3_current_tickers: int
    inactive_signal_tickers_covered: int
    inactive_adjusted_return_tickers_covered: int
    security_master_promotion_gates_all_pass: bool
    security_master_promotion_statuses: Mapping[str, str]


def evaluate_admissibility(
    inputs: PITCoverageInputs,
    *,
    minimum_dead_name_return_coverage: float,
) -> dict[str, Any]:
    inactive_signal_coverage = (
        inputs.inactive_signal_tickers_covered / inputs.malaysia_inactive_rows
        if inputs.malaysia_inactive_rows
        else 0.0
    )
    inactive_return_coverage = (
        inputs.inactive_adjusted_return_tickers_covered / inputs.malaysia_inactive_rows
        if inputs.malaysia_inactive_rows
        else 0.0
    )
    gates = {
        "all_source_hashes_match": inputs.source_hashes_match,
        "production_adjusted_returns_ready": inputs.production_adjusted_returns_ready,
        "historical_u2_membership_reconstructable": inputs.historical_u2_membership_reconstructable,
        "inactive_or_delisted_names_present": inputs.malaysia_inactive_rows > 0,
        "inactive_or_delisted_names_have_signal_inputs": inputs.inactive_signal_tickers_covered > 0,
        "inactive_or_delisted_names_have_adjusted_returns": (
            inputs.inactive_adjusted_return_tickers_covered > 0
        ),
        "minimum_dead_name_return_coverage": (
            inactive_return_coverage >= minimum_dead_name_return_coverage
        ),
        "security_master_promotion_gates_all_pass": (
            inputs.security_master_promotion_gates_all_pass
        ),
        "no_current_survivor_only_denominator": (
            inputs.historical_u2_membership_reconstructable
            and inactive_signal_coverage > 0.0
            and inactive_return_coverage >= minimum_dead_name_return_coverage
        ),
    }
    failed = [gate_id for gate_id, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "failed_gate_ids": failed,
        "admissible": not failed,
        "coverage": {
            "inactive_signal_input_coverage": inactive_signal_coverage,
            "inactive_adjusted_return_coverage": inactive_return_coverage,
            "minimum_required_inactive_adjusted_return_coverage": minimum_dead_name_return_coverage,
        },
    }


def _receipt_hash_status(protocol: Mapping[str, Any], repo_root: Path) -> tuple[bool, dict[str, Any]]:
    statuses: dict[str, Any] = {}
    all_match = True
    for receipt_id, receipt in protocol["data_receipts"].items():
        path = repo_root / str(receipt["path"])
        exists = path.is_file()
        observed = file_sha256(path) if exists else None
        matches = bool(exists and observed == receipt["sha256"])
        all_match = all_match and matches
        statuses[receipt_id] = {
            "path": str(path),
            "expected_sha256": receipt["sha256"],
            "observed_sha256": observed,
            "matches": matches,
        }
    return all_match, statuses


def audit_pit_replication_inputs(
    protocol: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    validate_frozen_protocol(protocol, repo_root=root)
    source_hashes_match, receipt_status = _receipt_hash_status(protocol, root)
    receipts = protocol["data_receipts"]

    scorecard = json.loads((root / receipts["institutional_scorecard"]["path"]).read_text())
    security_manifest = json.loads(
        (root / receipts["historical_security_master_manifest"]["path"]).read_text()
    )
    adjusted_manifest = json.loads(
        (root / receipts["adjusted_return_manifest"]["path"]).read_text()
    )
    listing = pd.read_parquet(root / receipts["listing_windows"]["path"])
    membership = pd.read_parquet(root / receipts["universe_membership"]["path"])
    adjusted = pd.read_parquet(
        root / receipts["daily_adjusted_returns"]["path"], columns=["ticker"]
    )
    current_master = pd.read_csv(root / receipts["current_security_master"]["path"])
    tiers = pd.read_csv(root / receipts["eligible_universe_tiers"]["path"])
    features = pd.read_parquet(root / receipts["daily_features"]["path"], columns=["ticker", "country"])
    promotion = pd.read_csv(root / receipts["security_master_promotion_gates"]["path"])

    malaysia_listing = listing.loc[listing["country"].eq(MALAYSIA)].copy()
    inactive = malaysia_listing.loc[~malaysia_listing["active_flag"].astype(bool)].copy()
    inactive_rics = set(inactive["ric"].dropna().astype(str))
    malaysia_feature_tickers = set(
        features.loc[features["country"].eq(MALAYSIA), "ticker"].dropna().astype(str)
    )
    malaysia_current_tickers = set(
        current_master.loc[current_master["country"].eq(MALAYSIA), "ticker"].dropna().astype(str)
    )
    adjusted_tickers = set(adjusted["ticker"].dropna().astype(str))
    malaysia_adjusted_tickers = adjusted_tickers & malaysia_current_tickers

    membership_columns = {str(item).lower() for item in membership.columns}
    has_historical_u2_tier = bool(
        membership_columns
        & {"u2_backtest_eligible", "selected_for_u2", "u2_eligible", "universe_tier"}
    )
    membership_dates = pd.to_datetime(membership.get("asof_date"), errors="coerce")
    repeated_time_snapshots = bool(
        membership_dates.notna().any()
        and membership_dates.nunique() > 12
        and membership.duplicated(subset=["ric_asof"], keep=False).any()
    )
    historical_u2_membership_reconstructable = has_historical_u2_tier and repeated_time_snapshots

    promotion_statuses = {
        str(row.gate_id): str(row.status) for row in promotion.itertuples(index=False)
    }
    promotion_all_pass = bool(promotion_statuses) and all(
        status == "pass" for status in promotion_statuses.values()
    )
    inputs = PITCoverageInputs(
        source_hashes_match=source_hashes_match,
        production_adjusted_returns_ready=bool(
            adjusted_manifest.get("production_adjusted_return_ready")
            and adjusted_manifest.get("status") == "pass"
        ),
        historical_u2_membership_reconstructable=historical_u2_membership_reconstructable,
        malaysia_listing_rows=len(malaysia_listing),
        malaysia_inactive_rows=len(inactive),
        malaysia_current_feature_tickers=len(malaysia_feature_tickers),
        malaysia_current_adjusted_return_tickers=len(malaysia_adjusted_tickers),
        malaysia_u2_current_tickers=int(
            tiers.loc[tiers["country"].eq(MALAYSIA), "u2_backtest_eligible"].astype(bool).sum()
        ),
        malaysia_u3_current_tickers=int(
            tiers.loc[tiers["country"].eq(MALAYSIA), "u3_liquid_proxy"].astype(bool).sum()
        ),
        inactive_signal_tickers_covered=len(inactive_rics & malaysia_feature_tickers),
        inactive_adjusted_return_tickers_covered=len(inactive_rics & adjusted_tickers),
        security_master_promotion_gates_all_pass=promotion_all_pass,
        security_master_promotion_statuses=promotion_statuses,
    )
    evaluation = evaluate_admissibility(
        inputs,
        minimum_dead_name_return_coverage=float(
            protocol["admissibility_gates"]["minimum_dead_name_return_coverage"]
        ),
    )
    return {
        "schema_version": 1,
        "workflow_name": protocol["workflow_name"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": object_sha256(protocol),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": MALAYSIA,
        "research_cutoff": protocol["research_cutoff"],
        "institutional_scorecard_claim": {
            "accepted": bool(scorecard.get("accepted")),
            "overall_score": scorecard.get("overall_score"),
            "status": scorecard.get("status"),
        },
        "historical_security_master_claim": {
            "status": security_manifest.get("status"),
            "dead_name_rows_all_three_markets": security_manifest.get("dead_name_rows"),
            "country_coverage_complete": security_manifest.get("country_coverage_complete"),
        },
        "data_counts": {
            "malaysia_listing_window_rows": inputs.malaysia_listing_rows,
            "malaysia_inactive_or_delisted_rows": inputs.malaysia_inactive_rows,
            "malaysia_current_feature_tickers": inputs.malaysia_current_feature_tickers,
            "malaysia_current_adjusted_return_tickers": inputs.malaysia_current_adjusted_return_tickers,
            "malaysia_current_u2_tickers": inputs.malaysia_u2_current_tickers,
            "malaysia_current_u3_tickers": inputs.malaysia_u3_current_tickers,
            "inactive_or_delisted_signal_tickers_covered": inputs.inactive_signal_tickers_covered,
            "inactive_or_delisted_adjusted_return_tickers_covered": (
                inputs.inactive_adjusted_return_tickers_covered
            ),
        },
        "membership_diagnostics": {
            "columns": sorted(membership_columns),
            "has_explicit_historical_u2_tier": has_historical_u2_tier,
            "has_repeated_time_snapshots_by_ric": repeated_time_snapshots,
            "reconstructable": historical_u2_membership_reconstructable,
        },
        "security_master_promotion_statuses": promotion_statuses,
        "receipt_hashes": receipt_status,
        **evaluation,
        "reservation_authorized": evaluation["admissible"],
        "strategy_trials_reserved": False,
        "strategy_computations_started": False,
        "paired_evaluation": {
            "reference_side": "fourteen existing current-survivor research results",
            "replication_side": (
                "ready_for_atomic_reservation"
                if evaluation["admissible"]
                else "not_run_failed_point_in_time_denominator_gate"
            ),
            "status": "ready" if evaluation["admissible"] else "blocked_before_strategy_computation",
            "performance_comparison_created": False,
            "reason": (
                "all predeclared data-admissibility gates passed"
                if evaluation["admissible"]
                else "a current-survivor rerun cannot be represented as a dead-name point-in-time replication"
            ),
        },
        "execution_state": {
            "parameter_optimization_performed": False,
            "post_hoc_combination_search_performed": False,
            "dry_run_started": False,
            "live_execution_authorized": False,
        },
    }


def persist_json(value: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") == rendered:
        return output
    output.write_text(rendered, encoding="utf-8")
    return output
