from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


UTC = timezone.utc
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
PRODUCER_OUTPUTS = (
    "producer_config.json",
    "event_outcomes.csv",
    "state_chains.csv",
    "unresolved_cases.csv",
    "local_ohlcv_routing_audit.csv",
    "missing_price_extraction_plan.csv",
    "evidence_provenance.csv",
    "input_admission.csv",
)


@dataclass(frozen=True)
class JPXReturnToNormalValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def validate_jpx_return_to_normal(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/jpx_return_to_normal_reconstruction_validation",
) -> JPXReturnToNormalValidationArtifacts:
    """Independently validate a reconstruction without importing producer code."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    producer = Path(producer_dir).resolve()
    if not producer.is_dir():
        raise FileNotFoundError(f"producer artifact directory is missing: {producer}")
    root = Path(artifact_root).resolve()
    final_dir = root / run_id
    if final_dir.exists():
        raise FileExistsError(f"immutable validation run already exists: {final_dir}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{run_id}.lock"
    lock_fd = _acquire_lock(lock)
    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=root))
        issues: list[str] = []
        checks: list[dict[str, Any]] = []

        producer_manifest_path = producer / "manifest.json"
        producer_manifest = _read_json(producer_manifest_path, issues, "producer_manifest_invalid")
        _check(
            checks,
            issues,
            "independent_run_id",
            False,
            run_id == producer_manifest.get("run_id"),
            "validator_run_id_matches_producer",
        )
        contract = (
            producer_manifest.get("artifact_type")
            == "jpx_post_2020_return_to_normal_reconstruction"
            and producer_manifest.get("immutable") is True
            and producer_manifest.get("atomic_publish") is True
            and producer_manifest.get("fail_closed") is True
            and producer_manifest.get("scorecard_eligible") is False
            and producer_manifest.get("component_scorecard_eligible") is False
            and producer_manifest.get("full_component_coverage_claimed") is False
            and producer_manifest.get("promotion_policy") == PROMOTION_POLICY
            and producer_manifest.get("independent_validation_required") is True
        )
        _check(checks, issues, "producer_contract", True, contract, "producer_contract_invalid")
        declared_hashes = producer_manifest.get("output_sha256", {})
        hashes_declared = isinstance(declared_hashes, dict) and set(PRODUCER_OUTPUTS).issubset(
            declared_hashes
        )
        _check(
            checks,
            issues,
            "producer_hashes_declared",
            True,
            hashes_declared,
            "producer_output_hashes_missing",
        )
        if isinstance(declared_hashes, dict):
            for filename, digest in sorted(declared_hashes.items()):
                path = producer / filename
                valid = path.is_file() and _sha256(path) == digest
                _check(
                    checks,
                    issues,
                    f"producer_output_hash:{filename}",
                    True,
                    valid,
                    f"producer_output_hash_mismatch:{filename}",
                )

        config = _read_json(producer / "producer_config.json", issues, "producer_config_invalid")
        input_dirs = {
            role: Path(str(path)).resolve()
            for role, path in config.get("inputs", {}).items()
            if str(path)
        }
        required_roles = {"event", "event_validation", "listing", "calendar_validation", "ohlcv"}
        _check(
            checks,
            issues,
            "required_input_roles",
            True,
            required_roles.issubset(input_dirs),
            "required_input_roles_missing",
        )
        input_manifests: dict[str, dict[str, Any]] = {}
        for role, directory in input_dirs.items():
            manifest_path = directory / "manifest.json"
            manifest = _read_json(manifest_path, issues, f"input_manifest_invalid:{role}")
            input_manifests[role] = manifest
            expected = config.get("input_manifest_sha256", {}).get(role)
            valid = manifest_path.is_file() and expected and _sha256(manifest_path) == expected
            _check(
                checks,
                issues,
                f"input_manifest_hash:{role}",
                True,
                valid,
                f"input_manifest_hash_mismatch:{role}",
            )
        _validate_input_contracts(input_manifests, checks, issues)

        provenance = _read_csv(producer / "evidence_provenance.csv", issues, "provenance_invalid")
        provenance_valid = _validate_provenance(provenance, issues)
        _check(
            checks,
            issues,
            "evidence_provenance_hashes",
            len(provenance),
            provenance_valid,
            "evidence_provenance_hash_failure",
        )

        outcomes = _read_csv(
            producer / "event_outcomes.csv", issues, "event_outcomes_invalid", dtype={"security_code": str}
        )
        states = _read_csv(
            producer / "state_chains.csv", issues, "state_chains_invalid", dtype={"security_code": str}
        )
        unresolved = _read_csv(
            producer / "unresolved_cases.csv", issues, "unresolved_cases_invalid", dtype={"security_code": str}
        )
        routing_audit = _read_csv(
            producer / "local_ohlcv_routing_audit.csv",
            issues,
            "local_ohlcv_routing_audit_invalid",
            dtype={"security_code": str},
        )
        extraction_plan = _read_csv(
            producer / "missing_price_extraction_plan.csv",
            issues,
            "missing_price_extraction_plan_invalid",
            dtype={"security_code": str},
        )
        calendar = _load_calendar(input_dirs.get("calendar_validation"), issues)
        windows = _load_windows(input_dirs.get("listing"), issues)
        source_events = _load_source_events(input_dirs.get("event"), issues)
        detail = _load_event_detail(input_dirs.get("event_validation"), issues)
        closing = _load_closing(input_dirs.get("closing_evidence"), issues)

        denominator_issues = _denominator_issues(
            outcomes, states, unresolved, source_events, producer_manifest
        )
        issues.extend(denominator_issues)
        _check(
            checks,
            issues,
            "activation_denominator_and_subsets",
            0,
            len(denominator_issues),
            "activation_denominator_validation_failed",
        )
        case_rows, case_issues = _validate_cases(
            outcomes,
            states,
            calendar,
            windows,
            source_events,
            detail,
            closing,
        )
        issues.extend(case_issues)
        _check(
            checks,
            issues,
            "independent_case_state_validation",
            0,
            len(case_issues),
            "case_state_validation_failed",
        )
        plan_issues = _missing_plan_issues(
            extraction_plan, outcomes, calendar, windows
        )
        issues.extend(plan_issues)
        _check(
            checks,
            issues,
            "missing_price_extraction_plan",
            0,
            len(plan_issues),
            "missing_price_extraction_plan_validation_failed",
        )
        audit_issues = _routing_audit_issues(
            routing_audit, outcomes, producer_manifest
        )
        issues.extend(audit_issues)
        _check(
            checks,
            issues,
            "local_ohlcv_routing_audit",
            0,
            len(audit_issues),
            "local_ohlcv_routing_audit_validation_failed",
        )
        coverage_language_valid = (
            producer_manifest.get("full_component_coverage_claimed") is False
            and producer_manifest.get("scorecard_eligible") is False
            and (
                producer_manifest.get("ohlcv_coverage_complete") is True
                or "partial" in str(producer_manifest.get("coverage_warning", "")).lower()
            )
        )
        _check(
            checks,
            issues,
            "partial_ohlcv_coverage_language",
            True,
            coverage_language_valid,
            "partial_ohlcv_coverage_overclaim",
        )
        count_issues = _manifest_count_issues(
            producer_manifest,
            outcomes,
            states,
            closing,
            routing_audit,
            extraction_plan,
        )
        issues.extend(count_issues)
        _check(
            checks,
            issues,
            "manifest_counts",
            0,
            len(count_issues),
            "producer_manifest_counts_invalid",
        )

        issues = list(dict.fromkeys(value for value in issues if value))
        summary = pd.DataFrame(checks)
        case_detail = pd.DataFrame(case_rows)
        _write_csv(stage / "validation_summary.csv", summary)
        _write_csv(stage / "case_validation.csv", case_detail)
        output_sha256 = {
            "validation_summary.csv": _sha256(stage / "validation_summary.csv"),
            "case_validation.csv": _sha256(stage / "case_validation.csv"),
        }
        manifest = {
            "schema_version": "1.0-independent-atomic-immutable",
            "run_id": run_id,
            "validator_run_id": run_id,
            "producer_run_id": producer_manifest.get("run_id", ""),
            "artifact_type": "jpx_post_2020_return_to_normal_independent_validation",
            "status": "pass_partial_lane" if not issues else "failed_validation",
            "component": "temporary_price_limit_events_return_to_normal",
            "independent_validation": True,
            "immutable": True,
            "atomic_publish": True,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "full_component_coverage_claimed": False,
            "promotion_allowed": False,
            "promotion_policy": dict(PROMOTION_POLICY),
            "validation_issues": issues,
            "counts": {
                "producer_cases": len(outcomes),
                "validated_cases": len(case_detail),
                "case_failures": int(case_detail["validation_pass"].ne(True).sum())
                if not case_detail.empty and "validation_pass" in case_detail
                else len(outcomes),
                "producer_state_rows": len(states),
                "validation_checks": len(summary),
                "failed_checks": int(summary["passed"].ne(True).sum()) if not summary.empty else 1,
            },
            "producer_manifest": str(producer_manifest_path),
            "producer_manifest_sha256": _sha256(producer_manifest_path)
            if producer_manifest_path.is_file()
            else "",
            "artifacts": {
                "validation_summary": "validation_summary.csv",
                "case_validation": "case_validation.csv",
            },
            "output_sha256": output_sha256,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, final_dir)
        stage = None
        return JPXReturnToNormalValidationArtifacts(
            run_id, final_dir, final_dir / "manifest.json", manifest
        )
    except Exception:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


def _validate_input_contracts(
    manifests: dict[str, dict[str, Any]], checks: list[dict[str, Any]], issues: list[str]
) -> None:
    contracts = {
        "event": lambda value: value.get("artifact_type")
        == "jpx_english_exceptional_price_limit_notice_events"
        and value.get("immutable") is True
        and value.get("official_jpx_source") is True,
        "event_validation": lambda value: value.get("artifact_type")
        == "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation"
        and value.get("independent_validation") is True
        and not value.get("validation_issues"),
        "listing": lambda value: value.get("artifact_type")
        == "japan_g1_pit_materializer_producer"
        and value.get("historical_pit_ready") is True
        and not value.get("critical_blockers"),
        "calendar_validation": lambda value: value.get("artifact_type")
        == "japan_exchange_calendar_independent_validation"
        and value.get("official_jpx_calendar") is True
        and not value.get("critical_blockers"),
        "ohlcv": lambda value: value.get("artifact_type")
        == "japan_listing_window_canonical_ohlcv"
        and value.get("quality_passed") is True,
        "closing_evidence": lambda value: value.get("artifact_type")
        == "jpx_closing_quote_order_book_evidence"
        and value.get("immutable") is True
        and value.get("independent_validation") is True
        and not value.get("validation_issues"),
    }
    for role, manifest in manifests.items():
        contract_role = "ohlcv" if role.startswith("ohlcv") else role
        valid = contract_role in contracts and contracts[contract_role](manifest)
        _check(
            checks,
            issues,
            f"input_contract:{role}",
            True,
            valid,
            f"input_contract_invalid:{role}",
        )


def _validate_provenance(frame: pd.DataFrame, issues: list[str]) -> int:
    required = {"input_role", "evidence_type", "path", "sha256", "bytes"}
    if frame.empty or not required.issubset(frame.columns):
        issues.append("evidence_provenance_schema_invalid")
        return 0
    valid = 0
    for row in frame.to_dict("records"):
        path = Path(str(row["path"]))
        try:
            size = int(row["bytes"])
        except (TypeError, ValueError):
            size = -1
        if (
            path.is_file()
            and size == path.stat().st_size
            and str(row["sha256"])
            and _sha256(path) == str(row["sha256"])
        ):
            valid += 1
        else:
            issues.append(f"provenance_hash_mismatch:{path}")
    return valid


def _denominator_issues(
    outcomes: pd.DataFrame,
    states: pd.DataFrame,
    unresolved: pd.DataFrame,
    source_events: pd.DataFrame,
    manifest: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    required_outcomes = {
        "case_id", "security_code", "activation_publication_date", "activation_effective_date",
        "activation_notice_id", "activation_source_url", "activation_source_sha256",
        "outcome_status", "return_normal_date", "trigger_session_date", "unresolved_reason",
    }
    required_states = {"case_id", "state_sequence", "session_date", "state", "security_code"}
    if not required_outcomes.issubset(outcomes.columns):
        return ["event_outcome_schema_invalid"]
    if not required_states.issubset(states.columns):
        return ["state_chain_schema_invalid"]
    if outcomes.empty or outcomes["case_id"].duplicated().any():
        issues.append("case_ids_empty_or_duplicated")
    if states.duplicated(["case_id", "state_sequence"]).any():
        issues.append("state_sequences_duplicated")
    activations = source_events.loc[
        source_events["event_type"].eq("activate")
        & source_events["publication_date"].astype(str).ge("2020-08-03")
    ]
    source_keys = set(
        activations[
            ["notice_id", "security_code", "publication_date", "source_url", "source_sha256"]
        ].astype(str).itertuples(index=False, name=None)
    )
    outcome_keys = set(
        outcomes[
            [
                "activation_notice_id", "security_code", "activation_publication_date",
                "activation_source_url", "activation_source_sha256",
            ]
        ].astype(str).itertuples(index=False, name=None)
    )
    if source_keys != outcome_keys:
        issues.append("activation_key_denominator_mismatch")
    unresolved_ids = set(outcomes.loc[outcomes["outcome_status"].ne("resolved"), "case_id"])
    if unresolved_ids != set(unresolved.get("case_id", pd.Series(dtype=str))):
        issues.append("unresolved_case_subset_mismatch")
    if set(states.get("case_id", pd.Series(dtype=str))) != set(outcomes["case_id"]):
        issues.append("state_case_denominator_mismatch")
    counts = manifest.get("counts", {})
    if counts.get("activation_cases") != len(outcomes):
        issues.append("manifest_activation_case_count_mismatch")
    return issues


def _validate_cases(
    outcomes: pd.DataFrame,
    states: pd.DataFrame,
    calendar: list[str],
    windows: pd.DataFrame,
    source_events: pd.DataFrame,
    detail: pd.DataFrame,
    closing: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    all_issues: list[str] = []
    session_index = {value: index for index, value in enumerate(calendar)}
    source_by_notice = {
        str(row.notice_id): row._asdict() for row in source_events.itertuples(index=False)
    }
    detail_keys = set(
        detail[["publication_date", "event_type", "code", "source_url"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    closing_groups = {
        key: group.to_dict("records")
        for key, group in closing.groupby(["code", "session_date"], sort=False)
    }
    parquet_cache: dict[str, pd.DataFrame] = {}
    for outcome in outcomes.to_dict("records"):
        case_id = str(outcome["case_id"])
        case_issues: list[str] = []
        chain = states.loc[states["case_id"].astype(str).eq(case_id)].sort_values("state_sequence")
        if chain.empty:
            case_issues.append("state_chain_missing")
        else:
            expected_sequence = list(range(1, len(chain) + 1))
            observed_sequence = pd.to_numeric(chain["state_sequence"], errors="coerce").tolist()
            if observed_sequence != expected_sequence:
                case_issues.append("state_sequence_not_contiguous")
        source = source_by_notice.get(str(outcome.get("activation_notice_id", "")))
        if source is None:
            case_issues.append("activation_source_event_missing")
        else:
            event_key = (
                str(source.get("publication_date", "")),
                "BROADENING",
                str(source.get("security_code", "")),
                str(source.get("source_url", "")),
            )
            if event_key not in detail_keys:
                case_issues.append("activation_numeric_validation_missing")
        for state in chain.to_dict("records"):
            _validate_state_source_row(
                state, windows, session_index, parquet_cache, case_issues
            )
        status = str(outcome.get("outcome_status", ""))
        if status == "resolved":
            _validate_resolved_case(
                outcome, chain, calendar, session_index, closing_groups, case_issues
            )
        elif status == "unresolved":
            _validate_unresolved_case(outcome, chain, case_issues)
        else:
            case_issues.append("outcome_status_invalid")
        if case_issues:
            all_issues.extend(f"{case_id}:{issue}" for issue in case_issues)
        rows.append(
            {
                "case_id": case_id,
                "security_code": outcome.get("security_code", ""),
                "outcome_status": status,
                "state_rows": len(chain),
                "validation_pass": not case_issues,
                "validation_issues": "|".join(case_issues),
            }
        )
    return rows, all_issues


def _validate_state_source_row(
    state: dict[str, Any],
    windows: pd.DataFrame,
    session_index: dict[str, int],
    parquet_cache: dict[str, pd.DataFrame],
    issues: list[str],
) -> None:
    session_date = str(state.get("session_date", ""))
    state_name = str(state.get("state", ""))
    source_rejection = state_name in {"unresolved_source_event", "unresolved_calendar"}
    if session_date not in session_index and not source_rejection:
        issues.append("state_date_not_calendar_session")
        return
    if not state_name.startswith("unresolved_source_event") and not state_name.startswith(
        "unresolved_calendar"
    ):
        matching = windows.loc[
            windows["local_code"].eq(str(state.get("security_code", "")))
            & windows["listing_start"].le(session_date)
            & (windows["right_open"] | windows["listing_end"].ge(session_date))
        ]
        if state_name != "unresolved_listing":
            if len(matching) != 1:
                issues.append("state_listing_not_uniquely_admitted")
            elif str(state.get("refinitiv_ric", "")) != str(matching.iloc[0]["refinitiv_ric"]):
                issues.append("state_listing_identity_mismatch")
    source_path = str(state.get("ohlcv_source_path", ""))
    if not source_path:
        return
    path = Path(source_path)
    expected = str(state.get("ohlcv_source_sha256", ""))
    if not path.is_file() or not expected or _sha256(path) != expected:
        issues.append("state_ohlcv_source_hash_mismatch")
        return
    if source_path not in parquet_cache:
        parquet_cache[source_path] = pd.read_parquet(path)
    frame = parquet_cache[source_path]
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    selected = frame.loc[
        frame["ticker"].astype(str).eq(str(state.get("refinitiv_ric", "")))
        & dates.eq(session_date)
    ]
    if len(selected) != 1:
        issues.append("state_ohlcv_row_not_unique")
        return
    row = selected.iloc[0]
    for column in ("open", "high", "low", "close", "volume"):
        if not _same_number(row[column], state.get(column)):
            issues.append(f"state_ohlcv_value_mismatch:{column}")


def _validate_resolved_case(
    outcome: dict[str, Any],
    chain: pd.DataFrame,
    calendar: list[str],
    session_index: dict[str, int],
    closing_groups: dict[tuple[str, str], list[dict[str, Any]]],
    issues: list[str],
) -> None:
    if len(chain) < 2:
        issues.append("resolved_chain_too_short")
        return
    final = chain.iloc[-1]
    trigger = chain.iloc[-2]
    if final["state"] != "return_normal_proven" or trigger["state"] != "cancellation_trigger_non_limit_trade":
        issues.append("resolved_terminal_states_invalid")
    trigger_date = str(outcome.get("trigger_session_date", ""))
    return_date = str(outcome.get("return_normal_date", ""))
    if trigger_date not in session_index:
        issues.append("trigger_date_not_calendar_session")
    else:
        index = session_index[trigger_date]
        expected = calendar[index + 1] if index + 1 < len(calendar) else ""
        if return_date != expected:
            issues.append("return_date_not_next_calendar_session")
    if str(trigger.get("session_date", "")) != trigger_date or str(final.get("session_date", "")) != return_date:
        issues.append("resolved_state_dates_mismatch")
    evidence = str(outcome.get("trigger_evidence", ""))
    if evidence == "positive_volume_non_limit_ohlcv":
        if not _ohlcv_non_limit(trigger):
            issues.append("ohlcv_does_not_prove_non_limit_trade")
    elif evidence == "admitted_closing_trade_evidence":
        key = (str(outcome.get("security_code", "")), trigger_date)
        if not _closing_non_limit(
            closing_groups.get(key, []),
            str(outcome.get("side", "")),
            _number(outcome.get("expanded_boundary_yen")),
        ):
            issues.append("closing_evidence_does_not_prove_non_limit_trade")
    else:
        issues.append("resolved_trigger_evidence_invalid")
    if str(outcome.get("unresolved_reason", "")) not in {"", "nan"}:
        issues.append("resolved_case_has_unresolved_reason")


def _validate_unresolved_case(
    outcome: dict[str, Any], chain: pd.DataFrame, issues: list[str]
) -> None:
    reason = str(outcome.get("unresolved_reason", ""))
    if not reason or reason == "nan":
        issues.append("unresolved_reason_missing")
    if str(outcome.get("return_normal_date", "")) not in {"", "nan"}:
        issues.append("unresolved_case_has_return_date")
    if not chain.empty and chain.iloc[-1]["state"] == "return_normal_proven":
        issues.append("unresolved_case_has_proven_terminal_state")
    if "zero_volume" in reason and not chain.empty:
        if (_number(chain.iloc[-1].get("volume")) or 0) > 0:
            issues.append("zero_volume_reason_not_supported")
    if "boundary_only" in reason and not chain.empty:
        row = chain.iloc[-1]
        if not _ohlcv_boundary_only(row):
            issues.append("boundary_only_reason_not_supported")


def _ohlcv_non_limit(row: pd.Series) -> bool:
    volume = _number(row.get("volume"))
    boundary = _number(row.get("expanded_boundary_yen"))
    values = [_number(row.get(name)) for name in ("open", "high", "low", "close")]
    if volume is None or volume <= 0 or boundary is None or any(value is None for value in values):
        return False
    side = str(row.get("side", ""))
    tolerance = max(1e-9, abs(boundary) * 1e-10)
    low = values[2]
    high = values[1]
    assert low is not None and high is not None
    return (side == "upper" and high <= boundary + tolerance and low < boundary - tolerance) or (
        side == "lower" and low >= boundary - tolerance and high > boundary + tolerance
    )


def _ohlcv_boundary_only(row: pd.Series) -> bool:
    volume = _number(row.get("volume"))
    boundary = _number(row.get("expanded_boundary_yen"))
    values = [_number(row.get(name)) for name in ("open", "high", "low", "close")]
    if volume is None or volume <= 0 or boundary is None or any(value is None for value in values):
        return False
    tolerance = max(1e-9, abs(boundary) * 1e-10)
    return all(abs(float(value) - boundary) <= tolerance for value in values if value is not None)


def _closing_non_limit(
    rows: list[dict[str, Any]], side: str, boundary: float | None
) -> bool:
    if boundary is None:
        return False
    tolerance = max(1e-9, abs(boundary) * 1e-10)
    for row in rows:
        for price_field, volume_field in (
            ("executed_price", "executed_volume"),
            ("closing_auction_price", "closing_auction_volume"),
        ):
            price = _number(row.get(price_field))
            volume = _number(row.get(volume_field))
            if price is None or volume is None or volume <= 0:
                continue
            if (side == "upper" and price < boundary - tolerance) or (
                side == "lower" and price > boundary + tolerance
            ):
                return True
    return False


def _missing_plan_issues(
    plan: pd.DataFrame,
    outcomes: pd.DataFrame,
    calendar: list[str],
    windows: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []
    required = {
        "plan_row_id", "request_key_sha256", "case_id", "security_code",
        "stable_listing_id", "stable_instrument_id", "refinitiv_ric", "exchange_code",
        "listing_window_start", "listing_window_end", "request_start_date", "request_end_date",
        "session_count", "session_dates", "interval", "fields", "adjustment", "currency",
        "writer_owner", "execution_enabled", "network_access_performed", "status",
    }
    missing_cases = set(
        outcomes.loc[
            outcomes["unresolved_reason"].eq("missing_ohlcv_without_closing_evidence"),
            "case_id",
        ].astype(str)
    )
    if not required.issubset(plan.columns):
        return ["missing_price_plan_schema_invalid"]
    if plan["plan_row_id"].duplicated().any() or plan["request_key_sha256"].duplicated().any():
        issues.append("missing_price_plan_keys_duplicated")
    if set(plan["case_id"].astype(str)) != missing_cases:
        issues.append("missing_price_plan_case_denominator_mismatch")
    calendar_set = set(calendar)
    requested_pairs: set[tuple[str, str, str]] = set()
    for row in plan.to_dict("records"):
        case_id = str(row["case_id"])
        dates = [value for value in str(row["session_dates"]).split("|") if value]
        try:
            session_count = int(row["session_count"])
        except (TypeError, ValueError):
            session_count = -1
        if (
            not dates
            or len(dates) != session_count
            or dates != sorted(dates)
            or dates[0] != str(row["request_start_date"])
            or dates[-1] != str(row["request_end_date"])
        ):
            issues.append(f"{case_id}:missing_price_plan_dates_invalid")
            continue
        request_key = "|".join(
            [
                case_id,
                str(row["refinitiv_ric"]),
                dates[0],
                dates[-1],
                str(len(dates)),
            ]
        )
        if hashlib.sha256(request_key.encode("utf-8")).hexdigest() != str(
            row["request_key_sha256"]
        ):
            issues.append(f"{case_id}:missing_price_plan_request_hash_invalid")
        if any(value not in calendar_set for value in dates):
            issues.append(f"{case_id}:missing_price_plan_non_session_date")
        if _truthy(row["execution_enabled"]) or _truthy(row["network_access_performed"]):
            issues.append(f"{case_id}:missing_price_plan_execution_enabled")
        if (
            str(row["writer_owner"]) != "single_vendor_writer"
            or str(row["interval"]) != "1d"
            or str(row["fields"]) != "open|high|low|close|volume"
            or str(row["adjustment"]) != "unadjusted"
            or str(row["currency"]) != "JPY"
        ):
            issues.append(f"{case_id}:missing_price_plan_request_contract_invalid")
        matching = windows.loc[
            windows["local_code"].eq(str(row["security_code"]))
            & windows["refinitiv_ric"].astype(str).eq(str(row["refinitiv_ric"]))
            & windows["listing_start"].le(dates[0])
            & (windows["right_open"] | windows["listing_end"].ge(dates[-1]))
        ]
        if len(matching) != 1:
            issues.append(f"{case_id}:missing_price_plan_outside_admitted_window")
        elif str(matching.iloc[0].get("stable_listing_id", "")) != str(
            row["stable_listing_id"]
        ):
            issues.append(f"{case_id}:missing_price_plan_listing_identity_mismatch")
        for session_date in dates:
            key = (case_id, str(row["refinitiv_ric"]), session_date)
            if key in requested_pairs:
                issues.append(f"{case_id}:missing_price_plan_session_duplicated")
            requested_pairs.add(key)
    return issues


def _routing_audit_issues(
    audit: pd.DataFrame,
    outcomes: pd.DataFrame,
    manifest: dict[str, Any],
) -> list[str]:
    required = {
        "case_id", "security_code", "required_session_date", "admitted_historical_ric",
        "stable_listing_id", "canonical_artifacts_audited", "additional_canonical_resolved",
        "raw_exact_rows_found", "raw_exact_rows_admitted", "raw_same_code_wrong_ric_rows",
        "raw_routing_resolved", "final_status", "audit_decision",
    }
    if not required.issubset(audit.columns):
        return ["local_routing_audit_schema_invalid"]
    issues: list[str] = []
    if audit["case_id"].duplicated().any():
        issues.append("local_routing_audit_case_ids_duplicated")
    counts = manifest.get("counts", {})
    if len(audit) != counts.get("primary_missing_ohlcv_cases"):
        issues.append("local_routing_audit_primary_denominator_mismatch")
    final_status = outcomes.set_index("case_id")["outcome_status"].astype(str).to_dict()
    for row in audit.to_dict("records"):
        case_id = str(row["case_id"])
        if final_status.get(case_id) != str(row["final_status"]):
            issues.append(f"{case_id}:local_routing_audit_final_status_mismatch")
        if int(row["raw_exact_rows_admitted"]) > int(row["raw_exact_rows_found"]):
            issues.append(f"{case_id}:local_routing_audit_raw_counts_invalid")
        if (
            str(row["audit_decision"]) == "unresolved_no_exact_local_ohlcv"
            and str(row["final_status"]) == "resolved"
        ):
            issues.append(f"{case_id}:local_routing_audit_decision_invalid")
    routing = manifest.get("local_ohlcv_routing_audit", {})
    if routing.get("same_code_wrong_ric_rows_admitted") != 0:
        issues.append("same_code_wrong_ric_admission_not_fail_closed")
    return issues


def _manifest_count_issues(
    manifest: dict[str, Any],
    outcomes: pd.DataFrame,
    states: pd.DataFrame,
    closing: pd.DataFrame,
    routing_audit: pd.DataFrame,
    extraction_plan: pd.DataFrame,
) -> list[str]:
    counts = manifest.get("counts", {})
    expected = {
        "activation_cases": len(outcomes),
        "resolved_cases": int(outcomes["outcome_status"].eq("resolved").sum()) if not outcomes.empty else 0,
        "unresolved_cases": int(outcomes["outcome_status"].ne("resolved").sum()) if not outcomes.empty else 0,
        "state_rows": len(states),
        "closing_evidence_rows": len(closing),
        "missing_price_extraction_plan_rows": len(extraction_plan),
        "primary_missing_ohlcv_cases": len(routing_audit),
    }
    return [f"manifest_count_mismatch:{key}" for key, value in expected.items() if counts.get(key) != value]


def _load_calendar(directory: Path | None, issues: list[str]) -> list[str]:
    if directory is None:
        return []
    path = directory / "validated_exchange_calendar.parquet"
    try:
        frame = pd.read_parquet(path)
        values = pd.to_datetime(frame["session_date"], errors="raise").dt.date.astype(str).tolist()
    except Exception:
        issues.append("validated_calendar_unreadable")
        return []
    if len(values) != len(set(values)) or values != sorted(values):
        issues.append("validated_calendar_dates_invalid")
    return values


def _load_windows(directory: Path | None, issues: list[str]) -> pd.DataFrame:
    columns = ["local_code", "listing_start", "listing_end", "right_open", "pit_usable", "refinitiv_ric"]
    if directory is None:
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_parquet(directory / "listing_windows.parquet")
        result = frame.copy()
        result["local_code"] = result["local_code"].astype(str).str.upper()
        result["listing_start"] = pd.to_datetime(result["listing_start"]).dt.date.astype(str)
        result["listing_end"] = pd.to_datetime(result["listing_end"]).dt.date.astype(str)
        result["right_open"] = result["right_open"].map(_truthy)
        result["pit_usable"] = result["pit_usable"].map(_truthy)
        return result.loc[result["pit_usable"]].reset_index(drop=True)
    except Exception:
        issues.append("listing_windows_unreadable")
        return pd.DataFrame(columns=columns)


def _load_source_events(directory: Path | None, issues: list[str]) -> pd.DataFrame:
    if directory is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(directory / "events.csv", dtype={"security_code": str})
    except Exception:
        issues.append("source_events_unreadable")
        return pd.DataFrame()


def _load_event_detail(directory: Path | None, issues: list[str]) -> pd.DataFrame:
    if directory is None:
        return pd.DataFrame(columns=["publication_date", "event_type", "code", "source_url"])
    try:
        return pd.read_csv(directory / "event_validation_detail.csv", dtype={"code": str})
    except Exception:
        issues.append("event_validation_detail_unreadable")
        return pd.DataFrame(columns=["publication_date", "event_type", "code", "source_url"])


def _load_closing(directory: Path | None, issues: list[str]) -> pd.DataFrame:
    columns = ["code", "session_date"]
    if directory is None:
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(directory / "closing_evidence.csv", dtype={"code": str}).fillna("")
        frame["code"] = frame["code"].astype(str).str.upper()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
        return frame
    except Exception:
        issues.append("closing_evidence_unreadable")
        return pd.DataFrame(columns=columns)


def _check(
    rows: list[dict[str, Any]],
    issues: list[str],
    name: str,
    expected: Any,
    observed: Any,
    issue: str,
) -> None:
    passed = _same(expected, observed)
    rows.append(
        {
            "check": name,
            "expected": _display(expected),
            "observed": _display(observed),
            "passed": passed,
        }
    )
    if not passed:
        issues.append(issue)


def _read_json(path: Path, issues: list[str], issue: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        issues.append(issue)
        return {}
    if not isinstance(value, dict):
        issues.append(issue)
        return {}
    return value


def _read_csv(
    path: Path, issues: list[str], issue: str, *, dtype: dict[str, Any] | None = None
) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=dtype, keep_default_na=False)
    except Exception:
        issues.append(issue)
        return pd.DataFrame()


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


def _same_number(left: Any, right: Any) -> bool:
    a = _number(left)
    b = _number(right)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(1e-9, max(abs(a), abs(b)) * 1e-12)


def _display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "pass", "passed", "complete"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {path}") from exc


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate an immutable JPX return-to-normal reconstruction."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-dir", required=True)
    parser.add_argument(
        "--artifact-root", default="artifacts/jpx_return_to_normal_reconstruction_validation"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    result = validate_jpx_return_to_normal(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"manifest": str(result.manifest_path), "status": result.manifest["status"]}, sort_keys=True))
    return 0 if result.manifest["status"] == "pass_partial_lane" else 1


if __name__ == "__main__":
    raise SystemExit(main())
