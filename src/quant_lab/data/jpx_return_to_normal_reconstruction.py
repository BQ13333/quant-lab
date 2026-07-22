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
from typing import Any, Iterable, Sequence

import pandas as pd


UTC = timezone.utc
REGIME_START = "2020-08-03"
EVENT_TYPES = {"activate", "continue"}
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
OUTPUT_NAMES = (
    "producer_config.json",
    "event_outcomes.csv",
    "state_chains.csv",
    "unresolved_cases.csv",
    "local_ohlcv_routing_audit.csv",
    "missing_price_extraction_plan.csv",
    "evidence_provenance.csv",
    "input_admission.csv",
)


class ReturnToNormalInputError(ValueError):
    """Raised when an input is not admissible or has changed in place."""


@dataclass(frozen=True)
class JPXReturnToNormalArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def produce_jpx_return_to_normal(
    *,
    run_id: str,
    event_dir: str | Path,
    event_validation_dir: str | Path,
    listing_dir: str | Path,
    calendar_validation_dir: str | Path,
    ohlcv_dir: str | Path,
    artifact_root: str | Path = "artifacts/jpx_return_to_normal_reconstruction",
    closing_evidence_dir: str | Path | None = None,
    additional_ohlcv_dirs: Sequence[str | Path] = (),
    raw_ohlcv_audit_root: str | Path | None = None,
    raw_historical_dir: str | Path | None = None,
    max_search_sessions: int = 20,
) -> JPXReturnToNormalArtifacts:
    """Reconstruct post-2020 return dates without treating missing data as evidence."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    if max_search_sessions <= 0:
        raise ValueError("max_search_sessions must be positive")

    inputs = {
        "event": Path(event_dir).resolve(),
        "event_validation": Path(event_validation_dir).resolve(),
        "listing": Path(listing_dir).resolve(),
        "calendar_validation": Path(calendar_validation_dir).resolve(),
        "ohlcv": Path(ohlcv_dir).resolve(),
    }
    if closing_evidence_dir is not None:
        inputs["closing_evidence"] = Path(closing_evidence_dir).resolve()
    for index, directory in enumerate(additional_ohlcv_dirs, 1):
        inputs[f"ohlcv_additional_{index:02d}"] = Path(directory).resolve()
    for role, directory in inputs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} artifact directory is missing: {directory}")

    root = Path(artifact_root).resolve()
    final_dir = root / run_id
    if final_dir.exists():
        raise FileExistsError(f"immutable reconstruction run already exists: {final_dir}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{run_id}.lock"
    lock_fd = _acquire_lock(lock)
    stage: Path | None = None
    try:
        if final_dir.exists():
            raise FileExistsError(f"immutable reconstruction run already exists: {final_dir}")
        stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=root))
        admitted, provenance = _admit_inputs(inputs)
        sessions = _load_calendar(admitted["calendar_validation"])
        event_rows, source_event_count, activation_count = _load_events(
            admitted["event"], admitted["event_validation"], sessions
        )
        windows = _load_listing_windows(admitted["listing"])
        closing = _load_closing_evidence(admitted.get("closing_evidence"), provenance)
        primary_ohlcv, primary_provenance = _load_ohlcv(
            admitted["ohlcv"],
            windows,
            event_rows["security_code"].unique(),
            role="ohlcv",
        )
        provenance.extend(primary_provenance)
        primary_outcomes, primary_states = reconstruct_event_states(
            event_rows,
            sessions,
            windows,
            primary_ohlcv,
            closing,
            max_search_sessions=max_search_sessions,
        )
        ohlcv_frames = [primary_ohlcv]
        for role in sorted(value for value in admitted if value.startswith("ohlcv_additional_")):
            frame, frame_provenance = _load_ohlcv(
                admitted[role],
                windows,
                event_rows["security_code"].unique(),
                role=role,
            )
            ohlcv_frames.append(frame)
            provenance.extend(frame_provenance)
        ohlcv = _merge_ohlcv_frames(ohlcv_frames)
        canonical_outcomes, canonical_states = reconstruct_event_states(
            event_rows,
            sessions,
            windows,
            ohlcv,
            closing,
            max_search_sessions=max_search_sessions,
        )
        initial_plan = build_missing_price_extraction_plan(
            canonical_outcomes,
            canonical_states,
            event_rows,
            sessions,
            windows,
            max_search_sessions=max_search_sessions,
        )
        raw_rows, raw_case_audit, raw_provenance = _audit_raw_ohlcv(
            initial_plan,
            raw_ohlcv_audit_root=raw_ohlcv_audit_root,
            raw_historical_dir=raw_historical_dir,
        )
        provenance.extend(raw_provenance)
        if raw_rows.empty:
            outcomes, states = canonical_outcomes, canonical_states
        else:
            ohlcv = _merge_ohlcv_frames([ohlcv, raw_rows])
            outcomes, states = reconstruct_event_states(
                event_rows,
                sessions,
                windows,
                ohlcv,
                closing,
                max_search_sessions=max_search_sessions,
            )
        extraction_plan = build_missing_price_extraction_plan(
            outcomes,
            states,
            event_rows,
            sessions,
            windows,
            max_search_sessions=max_search_sessions,
        )
        routing_audit = _build_routing_audit(
            primary_outcomes,
            primary_states,
            canonical_outcomes,
            outcomes,
            raw_case_audit,
            canonical_source_count=len(ohlcv_frames),
        )
        unresolved = outcomes.loc[outcomes["outcome_status"].ne("resolved")].copy()
        admission = pd.DataFrame(
            [
                {
                    "input_role": role,
                    "artifact_type": data["manifest"].get("artifact_type", ""),
                    "run_id": data["manifest"].get("run_id", ""),
                    "manifest_path": str(data["manifest_path"]),
                    "manifest_sha256": _sha256(data["manifest_path"]),
                    "admitted": True,
                    "admission_issues": "",
                }
                for role, data in admitted.items()
            ]
        )
        config = {
            "schema_version": "1.0",
            "run_id": run_id,
            "inputs": {role: str(path) for role, path in inputs.items()},
            "input_manifest_sha256": {
                role: _sha256(data["manifest_path"]) for role, data in admitted.items()
            },
            "raw_ohlcv_audit_root": str(Path(raw_ohlcv_audit_root).resolve())
            if raw_ohlcv_audit_root is not None
            else "",
            "raw_historical_dir": str(Path(raw_historical_dir).resolve())
            if raw_historical_dir is not None
            else "",
            "max_search_sessions": max_search_sessions,
            "regime_start": REGIME_START,
        }
        _write_json(stage / "producer_config.json", config)
        _write_csv(stage / "event_outcomes.csv", outcomes)
        _write_csv(stage / "state_chains.csv", states)
        _write_csv(stage / "unresolved_cases.csv", unresolved)
        _write_csv(stage / "local_ohlcv_routing_audit.csv", routing_audit)
        _write_csv(stage / "missing_price_extraction_plan.csv", extraction_plan)
        _write_csv(stage / "evidence_provenance.csv", pd.DataFrame(provenance))
        _write_csv(stage / "input_admission.csv", admission)

        output_sha256 = {name: _sha256(stage / name) for name in OUTPUT_NAMES}
        resolved_count = int(outcomes["outcome_status"].eq("resolved").sum())
        ohlcv_manifests = [
            data["manifest"] for role, data in admitted.items() if role.startswith("ohlcv")
        ]
        ohlcv_manifest = admitted["ohlcv"]["manifest"]
        ohlcv_complete = bool(
            ohlcv_manifests
            and all(
                manifest.get("source_extraction_complete")
                and manifest.get("promotion_eligible")
                and manifest.get("evidence_scope") == "full"
                for manifest in ohlcv_manifests
            )
        )
        all_cases_resolved = len(outcomes) == activation_count and resolved_count == activation_count
        status = "complete_post_2020_lane" if all_cases_resolved else "partial_reconstruction"
        manifest = {
            "schema_version": "1.0-atomic-immutable-fail-closed",
            "run_id": run_id,
            "artifact_type": "jpx_post_2020_return_to_normal_reconstruction",
            "status": status,
            "component": "temporary_price_limit_events_return_to_normal",
            "immutable": True,
            "atomic_publish": True,
            "fail_closed": True,
            "official_jpx_calendar": True,
            "official_jpx_event_sources": True,
            "network_accessed": False,
            "refinitiv_calls_performed": False,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "promotion_allowed": False,
            "promotion_policy": dict(PROMOTION_POLICY),
            "regime_scope": "2020-08-03_onward_within_admitted_input_cutoffs",
            "full_component_coverage_claimed": False,
            "post_2020_lane_complete": all_cases_resolved,
            "ohlcv_coverage_complete": ohlcv_complete,
            "ohlcv_coverage_scope": ohlcv_manifest.get("evidence_scope", "unknown"),
            "coverage_warning": (
                "Current OHLCV is partial; resolved cases are event-level proofs only and do not "
                "establish full component coverage."
                if not ohlcv_complete
                else "Full component coverage is not claimed by this post-2020-only lane."
            ),
            "return_rule": (
                "A positive-volume trade away from the broadened boundary triggers return to "
                "normal on the following admitted JPX session."
            ),
            "ambiguity_policy": (
                "Missing, zero-volume, and boundary-only OHLCV never prove a transition; admitted "
                "closing quote/order-book evidence is required to advance those states."
            ),
            "local_ohlcv_routing_audit": {
                "canonical_artifacts_audited": len(ohlcv_frames),
                "raw_audit_performed": raw_ohlcv_audit_root is not None,
                "raw_ohlcv_root": str(Path(raw_ohlcv_audit_root).resolve())
                if raw_ohlcv_audit_root is not None
                else "",
                "historical_identity_routing": "exact_admitted_G1_RIC_and_listing_window_only",
                "same_code_wrong_ric_rows_admitted": 0,
            },
            "counts": {
                "source_post_regime_activate_continue_rows": source_event_count,
                "activation_cases": activation_count,
                "resolved_cases": resolved_count,
                "unresolved_cases": len(unresolved),
                "state_rows": len(states),
                "ohlcv_rows_loaded": len(ohlcv),
                "closing_evidence_rows": len(closing),
                "canonical_ohlcv_artifacts_audited": len(ohlcv_frames),
                "primary_missing_ohlcv_cases": int(
                    primary_outcomes["unresolved_reason"]
                    .eq("missing_ohlcv_without_closing_evidence")
                    .sum()
                ),
                "cases_added_by_additional_canonical_routing": int(
                    len(_newly_resolved_case_ids(primary_outcomes, canonical_outcomes))
                ),
                "raw_exact_rows_admitted": len(raw_rows),
                "cases_added_by_raw_routing": int(
                    len(_newly_resolved_case_ids(canonical_outcomes, outcomes))
                ),
                "missing_price_extraction_plan_rows": len(extraction_plan),
            },
            "unresolved_reason_counts": (
                unresolved["unresolved_reason"].value_counts().sort_index().to_dict()
                if not unresolved.empty
                else {}
            ),
            "inputs": {
                role: {
                    "artifact_dir": str(inputs[role]),
                    "run_id": data["manifest"].get("run_id", ""),
                    "artifact_type": data["manifest"].get("artifact_type", ""),
                    "manifest_sha256": _sha256(data["manifest_path"]),
                }
                for role, data in admitted.items()
            },
            "artifacts": {Path(name).stem: name for name in OUTPUT_NAMES},
            "output_sha256": output_sha256,
            "independent_validation_required": True,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, final_dir)
        stage = None
        return JPXReturnToNormalArtifacts(run_id, final_dir, final_dir / "manifest.json", manifest)
    except Exception:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


def reconstruct_event_states(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    windows: pd.DataFrame,
    ohlcv: pd.DataFrame,
    closing: pd.DataFrame,
    *,
    max_search_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one chain per activation and stop each chain at its first unproven state."""
    session_dates = sessions["session_date"].astype(str).tolist()
    session_index = {value: index for index, value in enumerate(session_dates)}
    price_groups = {
        (str(row.ticker), str(row.session_date)): row._asdict()
        for row in ohlcv.itertuples(index=False)
    }
    closing_groups = {
        key: group.to_dict("records")
        for key, group in closing.groupby(["code", "session_date"], sort=False)
    }
    cases = _event_cases(events)
    outcome_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, 1):
        chain = case["events"]
        first = chain[0]
        case_id = f"JPX-RTN-{case_number:05d}"
        outcome = _base_outcome(case_id, chain)
        invalid = [str(row.get("event_admission_issue", "")) for row in chain if row.get("event_admission_issue")]
        if invalid:
            reason = f"source_event_not_admitted:{invalid[0]}"
            outcome_rows.append({**outcome, "outcome_status": "unresolved", "unresolved_reason": reason})
            state_rows.append(
                _state_row(
                    case_id,
                    1,
                    first,
                    session_date=str(first.get("publication_date", "")),
                    state="unresolved_source_event",
                    reason=reason,
                )
            )
            continue

        effective_dates = {str(row["effective_date"]): row for row in chain}
        anchor = str(chain[-1]["effective_date"])
        start = str(first["effective_date"])
        if start not in session_index or anchor not in session_index:
            reason = "event_effective_date_outside_calendar"
            outcome_rows.append({**outcome, "outcome_status": "unresolved", "unresolved_reason": reason})
            state_rows.append(_state_row(case_id, 1, first, state="unresolved_calendar", reason=reason))
            continue
        end_index = min(len(session_dates), session_index[anchor] + max_search_sessions)
        next_activation = case.get("next_activation_effective_date", "")
        if next_activation in session_index:
            end_index = min(end_index, session_index[next_activation])
        candidate_sessions = session_dates[session_index[start] : end_index]
        resolved = False
        reason = "bounded_search_exhausted_without_proof"
        examined = 0
        ohlcv_rows = 0
        closing_rows = 0
        for sequence, session_date in enumerate(candidate_sessions, 1):
            examined += 1
            event = effective_dates.get(session_date, chain[-1])
            listing, listing_issue = _listing_for(windows, str(first["security_code"]), session_date)
            if listing_issue:
                reason = listing_issue
                state_rows.append(
                    _state_row(case_id, sequence, event, session_date=session_date, state="unresolved_listing", reason=reason)
                )
                break
            ticker = str(listing["refinitiv_ric"])
            price = price_groups.get((ticker, session_date))
            quote_rows = closing_groups.get((str(first["security_code"]), session_date), [])
            ohlcv_rows += int(price is not None)
            closing_rows += len(quote_rows)
            observation, observation_reason = _classify_observation(
                price, quote_rows, str(first["side"]), float(event["expanded_boundary_yen"])
            )
            current_index = session_index[session_date]
            next_session = session_dates[current_index + 1] if current_index + 1 < len(session_dates) else ""
            official_continuation_next = next_session in effective_dates
            if official_continuation_next:
                if observation == "non_limit_trade":
                    reason = "official_continuation_conflicts_with_non_limit_trade"
                    state_rows.append(
                        _state_row(
                            case_id,
                            sequence,
                            event,
                            session_date=session_date,
                            state="unresolved_evidence_conflict",
                            reason=reason,
                            listing=listing,
                            price=price,
                            quote_rows=quote_rows,
                            observation=observation,
                        )
                    )
                    break
                state_rows.append(
                    _state_row(
                        case_id,
                        sequence,
                        event,
                        session_date=session_date,
                        state="expanded_confirmed_by_official_continuation",
                        listing=listing,
                        price=price,
                        quote_rows=quote_rows,
                        observation=observation,
                    )
                )
                continue
            if observation == "non_limit_trade":
                if not next_session:
                    reason = "return_session_outside_calendar"
                    state_rows.append(
                        _state_row(
                            case_id,
                            sequence,
                            event,
                            session_date=session_date,
                            state="unresolved_calendar",
                            reason=reason,
                            listing=listing,
                            price=price,
                            quote_rows=quote_rows,
                            observation=observation,
                        )
                    )
                    break
                return_listing, return_issue = _listing_for(
                    windows, str(first["security_code"]), next_session
                )
                if return_issue:
                    reason = "return_session_not_in_admitted_listing_window"
                    state_rows.append(
                        _state_row(
                            case_id,
                            sequence,
                            event,
                            session_date=session_date,
                            state="cancellation_trigger_non_limit_trade",
                            listing=listing,
                            price=price,
                            quote_rows=quote_rows,
                            observation=observation,
                        )
                    )
                    break
                state_rows.append(
                    _state_row(
                        case_id,
                        sequence,
                        event,
                        session_date=session_date,
                        state="cancellation_trigger_non_limit_trade",
                        listing=listing,
                        price=price,
                        quote_rows=quote_rows,
                        observation=observation,
                    )
                )
                state_rows.append(
                    _state_row(
                        case_id,
                        sequence + 1,
                        event,
                        session_date=next_session,
                        state="return_normal_proven",
                        listing=return_listing,
                        observation="rule_effect_from_prior_non_limit_trade",
                    )
                )
                outcome_rows.append(
                    {
                        **outcome,
                        "outcome_status": "resolved",
                        "return_normal_date": next_session,
                        "trigger_session_date": session_date,
                        "trigger_evidence": (
                            "admitted_closing_trade_evidence"
                            if _classify_closing_rows(
                                quote_rows,
                                str(first["side"]),
                                float(event["expanded_boundary_yen"]),
                            )
                            == "non_limit_trade"
                            else "positive_volume_non_limit_ohlcv"
                        ),
                        "unresolved_reason": "",
                        "sessions_examined": examined,
                        "ohlcv_rows_observed": ohlcv_rows,
                        "closing_evidence_rows_observed": closing_rows,
                    }
                )
                resolved = True
                break
            if observation == "expanded_continuation_proven":
                state_rows.append(
                    _state_row(
                        case_id,
                        sequence,
                        event,
                        session_date=session_date,
                        state="expanded_confirmed_by_closing_evidence",
                        listing=listing,
                        price=price,
                        quote_rows=quote_rows,
                        observation=observation,
                    )
                )
                reason = "closing_state_proven_but_official_continuation_event_missing"
                break
            reason = observation_reason
            state_rows.append(
                _state_row(
                    case_id,
                    sequence,
                    event,
                    session_date=session_date,
                    state="unresolved_observation",
                    reason=reason,
                    listing=listing,
                    price=price,
                    quote_rows=quote_rows,
                    observation=observation,
                )
            )
            break
        if not resolved:
            outcome_rows.append(
                {
                    **outcome,
                    "outcome_status": "unresolved",
                    "unresolved_reason": reason,
                    "sessions_examined": examined,
                    "ohlcv_rows_observed": ohlcv_rows,
                    "closing_evidence_rows_observed": closing_rows,
                }
            )
    outcomes = pd.DataFrame(outcome_rows, columns=_outcome_columns())
    states = pd.DataFrame(state_rows, columns=_state_columns())
    return outcomes, states


def _admit_inputs(inputs: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    admitted: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for role, directory in inputs.items():
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path)
        admitted[role] = {"manifest": manifest, "manifest_path": manifest_path, "directory": directory}
        provenance.append(_provenance(role, "manifest", manifest_path, _sha256(manifest_path)))

    event = admitted["event"]["manifest"]
    _require(
        event.get("artifact_type") == "jpx_english_exceptional_price_limit_notice_events"
        and event.get("immutable") is True
        and event.get("official_jpx_source") is True
        and event.get("status") == "complete_denominator_partial_component",
        "event bundle is not an immutable complete-denominator JPX source",
    )
    _verify_declared_files(admitted["event"], ("events.csv", "event_quality.csv"), provenance)

    validation = admitted["event_validation"]["manifest"]
    _require(
        validation.get("artifact_type")
        == "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation"
        and validation.get("status") == "pass_partial_component"
        and validation.get("independent_validation") is True
        and not validation.get("validation_issues"),
        "event numeric validation is not independently admitted",
    )
    _verify_declared_files(admitted["event_validation"], ("event_validation_detail.csv",), provenance)

    listing = admitted["listing"]["manifest"]
    _require(
        listing.get("artifact_type") == "japan_g1_pit_materializer_producer"
        and listing.get("status") == "complete"
        and listing.get("historical_pit_ready") is True
        and listing.get("official_jpx_source") is True
        and not listing.get("critical_blockers"),
        "listing-window bundle is not admitted",
    )
    _verify_declared_files(admitted["listing"], ("listing_windows.parquet",), provenance)

    calendar = admitted["calendar_validation"]["manifest"]
    _require(
        calendar.get("artifact_type") == "japan_exchange_calendar_independent_validation"
        and calendar.get("status") == "pass_component"
        and calendar.get("independent_validation") is True
        and calendar.get("official_jpx_calendar") is True
        and not calendar.get("critical_blockers"),
        "calendar is not an independently admitted official JPX calendar",
    )
    _verify_declared_files(
        admitted["calendar_validation"], ("validated_exchange_calendar.parquet",), provenance
    )

    for role in sorted(value for value in admitted if value.startswith("ohlcv")):
        ohlcv = admitted[role]["manifest"]
        _require(
            ohlcv.get("artifact_type") == "japan_listing_window_canonical_ohlcv"
            and ohlcv.get("quality_passed") is True,
            f"{role} is not a quality-passed listing-window canonical bundle",
        )
        inventory_path = admitted[role]["directory"] / "hash_inventory.json"
        inventory = _read_json(inventory_path)
        _require(
            inventory.get("algorithm") == "sha256" and inventory.get("immutable") is True,
            f"{role} hash inventory is not immutable SHA-256",
        )
        admitted[role]["inventory"] = inventory
        admitted[role]["inventory_path"] = inventory_path
        provenance.append(
            _provenance(role, "hash_inventory", inventory_path, _sha256(inventory_path))
        )

    if "closing_evidence" in admitted:
        closing = admitted["closing_evidence"]["manifest"]
        _require(
            closing.get("artifact_type") == "jpx_closing_quote_order_book_evidence"
            and closing.get("immutable") is True
            and closing.get("independent_validation") is True
            and not closing.get("validation_issues"),
            "closing evidence is not immutable and independently validated",
        )
        _verify_declared_files(admitted["closing_evidence"], ("closing_evidence.csv",), provenance)
    return admitted, provenance


def _load_calendar(data: dict[str, Any]) -> pd.DataFrame:
    path = data["directory"] / "validated_exchange_calendar.parquet"
    frame = pd.read_parquet(path)
    if "session_date" not in frame or frame.empty:
        raise ReturnToNormalInputError("validated calendar schema is invalid")
    result = frame.copy()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.date.astype(str)
    if result["session_date"].eq("NaT").any() or result["session_date"].duplicated().any():
        raise ReturnToNormalInputError("validated calendar dates are invalid or duplicated")
    return result.sort_values("session_date").reset_index(drop=True)


def _load_events(
    event_data: dict[str, Any], validation_data: dict[str, Any], sessions: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    events = pd.read_csv(event_data["directory"] / "events.csv", dtype={"security_code": str})
    quality = pd.read_csv(event_data["directory"] / "event_quality.csv", dtype={"security_code": str})
    detail = pd.read_csv(
        validation_data["directory"] / "event_validation_detail.csv", dtype={"code": str}
    )
    required_events = {
        "notice_id", "publication_date", "effective_date", "event_type", "security_code",
        "side", "upper_limit_price_yen", "lower_limit_price_yen", "source_url", "source_sha256",
    }
    required_detail = {
        "publication_date", "event_type", "code", "source_url", "resolved_side",
        "validated_base_price_yen", "validated_expanded_range_yen", "validated_normal_range_yen",
        "validated_observed_multiplier", "overlay_applied", "post_overlay_numeric_discrepancy",
        "post_overlay_regime_discrepancy", "expected_multiplier",
    }
    if not required_events.issubset(events.columns) or not required_detail.issubset(detail.columns):
        raise ReturnToNormalInputError("event or validation-detail schema is incomplete")
    source = events.loc[
        events["event_type"].isin(EVENT_TYPES) & events["publication_date"].ge(REGIME_START)
    ].copy()
    source["validation_event_type"] = source["event_type"].map(
        {"activate": "BROADENING", "continue": "CONTINUATION"}
    )
    key_left = ["publication_date", "validation_event_type", "security_code", "source_url"]
    key_right = ["publication_date", "event_type", "code", "source_url"]
    if source.duplicated(key_left).any() or detail.duplicated(key_right).any():
        raise ReturnToNormalInputError("event keys are duplicated")
    merged = source.merge(
        detail,
        left_on=key_left,
        right_on=key_right,
        how="left",
        validate="one_to_one",
        suffixes=("", "_validated"),
    )
    if merged["resolved_side"].isna().any():
        raise ReturnToNormalInputError("post-regime event validation denominator is incomplete")
    quality_keys = ["publication_date", "event_type", "security_code", "source_url"]
    if quality.duplicated(quality_keys).any():
        raise ReturnToNormalInputError("event quality keys are duplicated")
    merged = merged.merge(
        quality[quality_keys + ["validation_pass", "validation_issues"]],
        on=quality_keys,
        how="left",
        validate="one_to_one",
    )
    session_dates = sessions["session_date"].astype(str).tolist()
    session_set = set(session_dates)
    next_session = {
        value: session_dates[index + 1] if index + 1 < len(session_dates) else ""
        for index, value in enumerate(session_dates)
    }
    records = []
    for row in merged.to_dict("records"):
        issues: list[str] = []
        code = str(row.get("security_code", "")).upper()
        publication = str(row.get("publication_date", ""))
        effective = str(row.get("effective_date", ""))
        side = str(row.get("resolved_side", "")).lower()
        if not re.fullmatch(r"[0-9A-Z]{4}", code):
            issues.append("security_code_invalid")
        if publication not in session_set:
            issues.append("publication_not_admitted_session")
        if effective not in session_set:
            issues.append("effective_not_admitted_session")
        elif next_session.get(publication) != effective:
            issues.append("effective_not_next_admitted_session")
        if side not in {"upper", "lower"}:
            issues.append("side_invalid")
        numeric = {
            name: _number(row.get(name))
            for name in (
                "validated_base_price_yen", "validated_expanded_range_yen",
                "validated_normal_range_yen", "validated_observed_multiplier", "expected_multiplier",
            )
        }
        if any(value is None or value <= 0 for value in numeric.values()):
            issues.append("validated_numeric_fields_invalid")
        if numeric["validated_observed_multiplier"] != 4.0 or numeric["expected_multiplier"] != 4.0:
            issues.append("post_regime_multiplier_not_4x")
        if _truthy(row.get("post_overlay_numeric_discrepancy")) or _truthy(
            row.get("post_overlay_regime_discrepancy")
        ):
            issues.append("validated_overlay_discrepancy")
        boundary = _source_or_overlay_boundary(row, side, numeric)
        if boundary is None or boundary <= 0:
            issues.append("expanded_boundary_invalid")
        records.append(
            {
                **row,
                "security_code": code,
                "side": side,
                "expanded_boundary_yen": boundary,
                "event_admission_issue": "|".join(issues),
            }
        )
    result = pd.DataFrame(records)
    activation_count = int(result["event_type"].eq("activate").sum())
    declared_total = int(event_data["manifest"].get("event_type_counts", {}).get("activate", -1))
    if activation_count <= 0 or declared_total < activation_count:
        raise ReturnToNormalInputError("post-regime activation denominator is invalid")
    return result, len(result), activation_count


def _source_or_overlay_boundary(row: dict[str, Any], side: str, numeric: dict[str, float | None]) -> float | None:
    if side not in {"upper", "lower"}:
        return None
    if not _truthy(row.get("overlay_applied")):
        return _number(row.get("upper_limit_price_yen" if side == "upper" else "lower_limit_price_yen"))
    base = numeric["validated_base_price_yen"]
    expanded = numeric["validated_expanded_range_yen"]
    if base is None or expanded is None:
        return None
    target = base + expanded if side == "upper" else max(1.0, base - expanded)
    tick = _standard_tick(target)
    return math.ceil(target / tick) * tick if side == "upper" else math.floor(target / tick) * tick


def _standard_tick(price: float) -> float:
    bands = (
        (3_000, 1), (5_000, 5), (30_000, 10), (50_000, 50), (300_000, 100),
        (500_000, 500), (3_000_000, 1_000), (5_000_000, 5_000),
        (30_000_000, 10_000), (50_000_000, 50_000), (float("inf"), 100_000),
    )
    return float(next(tick for upper, tick in bands if price <= upper))


def _load_listing_windows(data: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(data["directory"] / "listing_windows.parquet")
    required = {
        "stable_listing_id", "stable_instrument_id", "refinitiv_ric", "local_code",
        "exchange_code", "listing_start", "listing_end", "right_open", "pit_usable",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ReturnToNormalInputError("listing-window schema is invalid")
    result = frame.copy()
    result["local_code"] = result["local_code"].astype(str).str.upper()
    result["listing_start"] = pd.to_datetime(result["listing_start"], errors="coerce").dt.date.astype(str)
    result["listing_end"] = pd.to_datetime(result["listing_end"], errors="coerce").dt.date.astype(str)
    result["pit_usable"] = result["pit_usable"].map(_truthy)
    result["right_open"] = result["right_open"].map(_truthy)
    result = result.loc[
        result["pit_usable"] & result["exchange_code"].astype(str).isin({"TYO", "JSD", "OSA"})
    ].copy()
    if result.duplicated(["local_code", "listing_start", "listing_end", "refinitiv_ric"]).any():
        raise ReturnToNormalInputError("admitted listing windows contain duplicate keys")
    return result.reset_index(drop=True)


def _load_ohlcv(
    data: dict[str, Any],
    windows: pd.DataFrame,
    codes: Iterable[str],
    *,
    role: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    directory = data["directory"]
    root = directory / str(data["manifest"].get("artifacts", {}).get("ohlcv_root", "ohlcv"))
    inventory = data["inventory"].get("files", {})
    wanted_codes = {str(value) for value in codes}
    wanted_tickers = set(
        windows.loc[windows["local_code"].isin(wanted_codes), "refinitiv_ric"].astype(str)
    )
    rows: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.parquet")):
        year_match = re.search(r"year_partition=(\d{4})", str(path))
        if year_match and int(year_match.group(1)) < 2020:
            continue
        relative = str(path.relative_to(directory))
        expected = inventory.get(relative)
        if not expected or _sha256(path) != expected:
            raise ReturnToNormalInputError(f"OHLCV partition hash mismatch: {relative}")
        frame = pd.read_parquet(path)
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ReturnToNormalInputError(f"OHLCV partition schema is invalid: {relative}")
        selected = frame.loc[frame["ticker"].astype(str).isin(wanted_tickers), list(required)].copy()
        if selected.empty:
            continue
        selected["_source_path"] = str(path)
        selected["_source_sha256"] = expected
        rows.append(selected)
        provenance.append(_provenance(role, "partition", path, expected))
    columns = [
        "session_date", "ticker", "open", "high", "low", "close", "volume",
        "_source_path", "_source_sha256",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), provenance
    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"date": "session_date"})
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.date.astype(str)
    if result["session_date"].eq("NaT").any() or result.duplicated(["ticker", "session_date"]).any():
        raise ReturnToNormalInputError("selected OHLCV has invalid dates or duplicate ticker/session rows")
    return result[columns].sort_values(["ticker", "session_date"]).reset_index(drop=True), provenance


def _merge_ohlcv_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame.copy() for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=[
                "session_date", "ticker", "open", "high", "low", "close", "volume",
                "_source_path", "_source_sha256",
            ]
        )
    for rank, frame in enumerate(nonempty):
        frame["_source_rank"] = rank
    combined = pd.concat(nonempty, ignore_index=True)
    keys = ["ticker", "session_date"]
    values = ["open", "high", "low", "close", "volume"]
    duplicates = combined.loc[combined.duplicated(keys, keep=False)]
    if not duplicates.empty:
        conflicts = duplicates.groupby(keys, dropna=False)[values].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            first = conflicts.loc[conflicts].index[0]
            raise ReturnToNormalInputError(
                f"accepted OHLCV artifacts conflict for ticker/session: {first}"
            )
    result = (
        combined.sort_values([*keys, "_source_rank", "_source_path"], kind="stable")
        .drop_duplicates(keys, keep="first")
        .drop(columns="_source_rank")
    )
    return result.sort_values(keys).reset_index(drop=True)


def build_missing_price_extraction_plan(
    outcomes: pd.DataFrame,
    states: pd.DataFrame,
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    max_search_sessions: int,
) -> pd.DataFrame:
    """Create exact, listing-window-bounded requests for unresolved missing-price cases."""
    columns = [
        "plan_row_id", "request_key_sha256", "case_id", "security_code",
        "stable_listing_id", "stable_instrument_id", "refinitiv_ric", "exchange_code",
        "listing_window_start", "listing_window_end", "request_start_date", "request_end_date",
        "session_count", "session_dates", "interval", "fields", "adjustment", "currency",
        "writer_owner", "execution_enabled", "network_access_performed", "status",
    ]
    missing = outcomes.loc[
        outcomes["unresolved_reason"].eq("missing_ohlcv_without_closing_evidence")
    ]
    if missing.empty:
        return pd.DataFrame(columns=columns)
    state_by_case = {
        str(row.case_id): row._asdict()
        for row in states.loc[
            states["case_id"].isin(missing["case_id"])
            & states["state"].eq("unresolved_observation")
        ].itertuples(index=False)
    }
    event_cases = {
        f"JPX-RTN-{index:05d}": case
        for index, case in enumerate(_event_cases(events), 1)
    }
    session_dates = sessions["session_date"].astype(str).tolist()
    session_index = {value: index for index, value in enumerate(session_dates)}
    rows: list[dict[str, Any]] = []
    for outcome in missing.sort_values("case_id").to_dict("records"):
        case_id = str(outcome["case_id"])
        state = state_by_case.get(case_id)
        case = event_cases.get(case_id)
        if state is None or case is None:
            raise ReturnToNormalInputError(f"missing-price case lacks state/event chain: {case_id}")
        start = str(state["session_date"])
        anchor = str(case["events"][-1]["effective_date"])
        if start not in session_index or anchor not in session_index:
            raise ReturnToNormalInputError(f"missing-price case is outside admitted calendar: {case_id}")
        end_index = min(len(session_dates), session_index[anchor] + max_search_sessions)
        next_activation = str(case.get("next_activation_effective_date", ""))
        if next_activation in session_index:
            end_index = min(end_index, session_index[next_activation])
        required_dates = session_dates[session_index[start] : end_index]
        segments: list[tuple[dict[str, Any], list[str]]] = []
        for session_date in required_dates:
            listing, issue = _listing_for(windows, str(outcome["security_code"]), session_date)
            if issue:
                break
            identity = (
                str(listing["stable_listing_id"]),
                str(listing["refinitiv_ric"]),
                str(listing["listing_start"]),
                str(listing["listing_end"]),
            )
            if segments:
                previous = segments[-1][0]
                previous_identity = (
                    str(previous["stable_listing_id"]),
                    str(previous["refinitiv_ric"]),
                    str(previous["listing_start"]),
                    str(previous["listing_end"]),
                )
                if identity == previous_identity:
                    segments[-1][1].append(session_date)
                    continue
            segments.append((listing, [session_date]))
        for listing, dates in segments:
            key = "|".join(
                [
                    case_id,
                    str(listing["refinitiv_ric"]),
                    dates[0],
                    dates[-1],
                    str(len(dates)),
                ]
            )
            rows.append(
                {
                    "request_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "case_id": case_id,
                    "security_code": outcome["security_code"],
                    "stable_listing_id": listing["stable_listing_id"],
                    "stable_instrument_id": listing["stable_instrument_id"],
                    "refinitiv_ric": listing["refinitiv_ric"],
                    "exchange_code": listing["exchange_code"],
                    "listing_window_start": listing["listing_start"],
                    "listing_window_end": listing["listing_end"],
                    "request_start_date": dates[0],
                    "request_end_date": dates[-1],
                    "session_count": len(dates),
                    "session_dates": "|".join(dates),
                    "interval": "1d",
                    "fields": "open|high|low|close|volume",
                    "adjustment": "unadjusted",
                    "currency": "JPY",
                    "writer_owner": "single_vendor_writer",
                    "execution_enabled": False,
                    "network_access_performed": False,
                    "status": "pending_exact_missing_price_extraction",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(
        ["request_start_date", "refinitiv_ric", "case_id", "request_end_date"], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "plan_row_id", [f"JPX-RTN-MISSING-{index:05d}" for index in range(1, len(result) + 1)])
    if result["request_key_sha256"].duplicated().any():
        raise ReturnToNormalInputError("missing-price extraction request keys are duplicated")
    return result[columns]


def _audit_raw_ohlcv(
    plan: pd.DataFrame,
    *,
    raw_ohlcv_audit_root: str | Path | None,
    raw_historical_dir: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    raw_columns = [
        "session_date", "ticker", "open", "high", "low", "close", "volume",
        "_source_path", "_source_sha256",
    ]
    audit_columns = [
        "case_id", "planned_exact_ric_sessions", "raw_exact_rows_found",
        "raw_exact_rows_admitted", "raw_same_code_wrong_ric_rows", "raw_audit_status",
    ]
    if raw_ohlcv_audit_root is None and raw_historical_dir is None:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame(columns=audit_columns), []
    if raw_ohlcv_audit_root is None or raw_historical_dir is None:
        raise ValueError("raw OHLCV audit root and historical directory must be provided together")
    raw_root = Path(raw_ohlcv_audit_root).resolve()
    historical = Path(raw_historical_dir).resolve()
    run_manifest_path = historical / "historical_run_manifest.json"
    plan_path = historical / "historical_plan.csv"
    terminal_path = historical / "historical_manifest.csv"
    run_manifest = _read_json(run_manifest_path)
    declared_root = Path(str(run_manifest.get("long_ohlcv_root", ""))).resolve()
    if declared_root != raw_root or not raw_root.is_dir():
        raise ReturnToNormalInputError("raw OHLCV audit root does not match historical run manifest")
    for path in (plan_path, terminal_path):
        if not path.is_file():
            raise ReturnToNormalInputError(f"raw OHLCV lineage file is missing: {path}")
    lineage_plan = pd.read_csv(plan_path, low_memory=False)
    terminal = pd.read_csv(terminal_path, low_memory=False)
    accepted = terminal.loc[terminal["status"].astype(str).str.startswith("completed")].copy()
    accepted_by_batch = {
        str(row.plan_row_id): row._asdict() for row in accepted.itertuples(index=False)
    }
    provenance = [
        _provenance("raw_ohlcv_audit", "historical_run_manifest", run_manifest_path, _sha256(run_manifest_path)),
        _provenance("raw_ohlcv_audit", "historical_plan", plan_path, _sha256(plan_path)),
        _provenance("raw_ohlcv_audit", "historical_terminal", terminal_path, _sha256(terminal_path)),
    ]
    requirements: dict[tuple[str, str], set[str]] = {}
    code_requirements: dict[tuple[str, str], set[str]] = {}
    for row in plan.to_dict("records"):
        for session_date in str(row["session_dates"]).split("|"):
            requirements.setdefault((str(row["refinitiv_ric"]), session_date), set()).add(
                str(row["case_id"])
            )
            code_requirements.setdefault((str(row["security_code"]), session_date), set()).add(
                str(row["case_id"])
            )
    exact_counts = {case_id: 0 for case_id in plan["case_id"].astype(str).unique()}
    admitted_counts = dict(exact_counts)
    wrong_counts = dict(exact_counts)
    raw_rows: list[dict[str, Any]] = []
    required_months = sorted({session_date[:7] for _, session_date in requirements})
    for observation_month in required_months:
        year, month = observation_month.split("-")
        month_dirs = list(
            raw_root.glob(
                f"**/year_partition={int(year)}/month_partition={int(month)}"
            )
        )
        for month_dir in sorted(month_dirs):
            for path in sorted(month_dir.glob("*.parquet")):
                digest = _sha256(path)
                provenance.append(_provenance("raw_ohlcv_audit", "scanned_partition", path, digest))
                frame = pd.read_parquet(path)
                required = {"date", "ticker", "open", "high", "low", "close", "volume", "batch_id"}
                if not required.issubset(frame.columns):
                    raise ReturnToNormalInputError(f"raw OHLCV audit partition schema invalid: {path}")
                dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
                tickers = frame["ticker"].astype(str)
                local_codes = (
                    frame["local_symbol"].astype(str).str.upper()
                    if "local_symbol" in frame
                    else pd.Series("", index=frame.index)
                )
                for index in frame.index:
                    pair = (tickers.loc[index], dates.loc[index])
                    code_pair = (local_codes.loc[index], dates.loc[index])
                    exact_cases = requirements.get(pair, set())
                    wrong_cases = code_requirements.get(code_pair, set()) - exact_cases
                    for case_id in wrong_cases:
                        wrong_counts[case_id] += 1
                    if not exact_cases:
                        continue
                    for case_id in exact_cases:
                        exact_counts[case_id] += 1
                    batch_id = str(frame.at[index, "batch_id"])
                    lineage = accepted_by_batch.get(batch_id)
                    if lineage is None or not _raw_row_matches_lineage(
                        tickers.loc[index], dates.loc[index], lineage
                    ):
                        continue
                    for case_id in exact_cases:
                        admitted_counts[case_id] += 1
                    raw_rows.append(
                        {
                            "session_date": dates.loc[index],
                            "ticker": tickers.loc[index],
                            "open": frame.at[index, "open"],
                            "high": frame.at[index, "high"],
                            "low": frame.at[index, "low"],
                            "close": frame.at[index, "close"],
                            "volume": frame.at[index, "volume"],
                            "_source_path": str(path),
                            "_source_sha256": digest,
                        }
                    )
    raw_frame = _merge_ohlcv_frames([pd.DataFrame(raw_rows, columns=raw_columns)])
    audit_rows = []
    for case_id, group in plan.groupby("case_id", sort=True):
        planned = sum(int(value) for value in group["session_count"])
        audit_rows.append(
            {
                "case_id": case_id,
                "planned_exact_ric_sessions": planned,
                "raw_exact_rows_found": exact_counts.get(str(case_id), 0),
                "raw_exact_rows_admitted": admitted_counts.get(str(case_id), 0),
                "raw_same_code_wrong_ric_rows": wrong_counts.get(str(case_id), 0),
                "raw_audit_status": (
                    "exact_rows_admitted"
                    if admitted_counts.get(str(case_id), 0)
                    else "no_exact_rows_found"
                    if not exact_counts.get(str(case_id), 0)
                    else "exact_rows_rejected_by_lineage"
                ),
            }
        )
    return raw_frame, pd.DataFrame(audit_rows, columns=audit_columns), provenance


def _raw_row_matches_lineage(ticker: str, session_date: str, lineage: dict[str, Any]) -> bool:
    tickers = {value for value in str(lineage.get("tickers", "")).split(",") if value}
    return (
        ticker in tickers
        and str(lineage.get("start_date", "")) <= session_date <= str(lineage.get("end_date", ""))
    )


def _build_routing_audit(
    primary_outcomes: pd.DataFrame,
    primary_states: pd.DataFrame,
    canonical_outcomes: pd.DataFrame,
    final_outcomes: pd.DataFrame,
    raw_case_audit: pd.DataFrame,
    *,
    canonical_source_count: int,
) -> pd.DataFrame:
    columns = [
        "case_id", "security_code", "required_session_date", "admitted_historical_ric",
        "stable_listing_id", "canonical_artifacts_audited", "additional_canonical_resolved",
        "raw_exact_rows_found", "raw_exact_rows_admitted", "raw_same_code_wrong_ric_rows",
        "raw_routing_resolved", "final_status", "audit_decision",
    ]
    missing = primary_outcomes.loc[
        primary_outcomes["unresolved_reason"].eq("missing_ohlcv_without_closing_evidence")
    ]
    if missing.empty:
        return pd.DataFrame(columns=columns)
    state = primary_states.loc[
        primary_states["case_id"].isin(missing["case_id"])
        & primary_states["state"].eq("unresolved_observation")
    ][["case_id", "session_date", "refinitiv_ric", "stable_listing_id"]]
    frame = missing[["case_id", "security_code"]].merge(state, on="case_id", validate="one_to_one")
    canonical_status = canonical_outcomes.set_index("case_id")["outcome_status"].to_dict()
    final_status = final_outcomes.set_index("case_id")["outcome_status"].to_dict()
    raw_by_case = (
        raw_case_audit.set_index("case_id").to_dict("index") if not raw_case_audit.empty else {}
    )
    rows = []
    for row in frame.sort_values("case_id").to_dict("records"):
        case_id = str(row["case_id"])
        raw = raw_by_case.get(case_id, {})
        additional_resolved = canonical_status.get(case_id) == "resolved"
        raw_resolved = canonical_status.get(case_id) != "resolved" and final_status.get(case_id) == "resolved"
        decision = (
            "resolved_by_additional_canonical_exact_ric"
            if additional_resolved
            else "resolved_by_lineage_admitted_raw_exact_ric"
            if raw_resolved
            else "unresolved_no_exact_local_ohlcv"
        )
        rows.append(
            {
                "case_id": case_id,
                "security_code": row["security_code"],
                "required_session_date": row["session_date"],
                "admitted_historical_ric": row["refinitiv_ric"],
                "stable_listing_id": row["stable_listing_id"],
                "canonical_artifacts_audited": canonical_source_count,
                "additional_canonical_resolved": additional_resolved,
                "raw_exact_rows_found": int(raw.get("raw_exact_rows_found", 0)),
                "raw_exact_rows_admitted": int(raw.get("raw_exact_rows_admitted", 0)),
                "raw_same_code_wrong_ric_rows": int(raw.get("raw_same_code_wrong_ric_rows", 0)),
                "raw_routing_resolved": raw_resolved,
                "final_status": final_status.get(case_id, ""),
                "audit_decision": decision,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _newly_resolved_case_ids(before: pd.DataFrame, after: pd.DataFrame) -> set[str]:
    before_unresolved = set(before.loc[before["outcome_status"].ne("resolved"), "case_id"].astype(str))
    after_resolved = set(after.loc[after["outcome_status"].eq("resolved"), "case_id"].astype(str))
    return before_unresolved & after_resolved


def _load_closing_evidence(
    data: dict[str, Any] | None, provenance: list[dict[str, Any]]
) -> pd.DataFrame:
    columns = _closing_columns()
    if data is None:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(data["directory"] / "closing_evidence.csv", dtype={"code": str})
    required = set(columns)
    if not required.issubset(frame.columns):
        raise ReturnToNormalInputError(f"closing evidence missing columns: {sorted(required - set(frame.columns))}")
    result = frame[columns].copy().fillna("")
    result["code"] = result["code"].astype(str).str.upper()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.date.astype(str)
    if result["session_date"].eq("NaT").any() or result.duplicated(["code", "session_date", "source_path"]).any():
        raise ReturnToNormalInputError("closing evidence has invalid or duplicate keys")
    for row in result.to_dict("records"):
        path = Path(str(row["source_path"]))
        digest = str(row["source_sha256"])
        if not path.is_file() or not digest or _sha256(path) != digest:
            raise ReturnToNormalInputError(f"closing evidence source hash mismatch: {path}")
        provenance.append(_provenance("closing_evidence", str(row["evidence_kind"]), path, digest))
    return result


def _event_cases(events: pd.DataFrame) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for code, group in events.sort_values(
        ["security_code", "publication_date", "effective_date", "source_url"]
    ).groupby("security_code", sort=True):
        active: list[dict[str, Any]] | None = None
        code_cases: list[list[dict[str, Any]]] = []
        for row in group.to_dict("records"):
            if row["event_type"] == "activate":
                if active is not None:
                    code_cases.append(active)
                active = [row]
            elif active is None:
                raise ReturnToNormalInputError(f"continuation lacks preceding activation: {code}")
            else:
                active.append(row)
        if active is not None:
            code_cases.append(active)
        for index, chain in enumerate(code_cases):
            next_effective = code_cases[index + 1][0]["effective_date"] if index + 1 < len(code_cases) else ""
            cases.append({"events": chain, "next_activation_effective_date": next_effective})
    return sorted(
        cases,
        key=lambda item: (
            item["events"][0]["publication_date"], item["events"][0]["effective_date"],
            item["events"][0]["security_code"], item["events"][0]["source_url"],
        ),
    )


def _listing_for(windows: pd.DataFrame, code: str, session_date: str) -> tuple[dict[str, Any], str]:
    selected = windows.loc[
        windows["local_code"].eq(code)
        & windows["listing_start"].le(session_date)
        & (windows["right_open"] | windows["listing_end"].ge(session_date))
    ]
    if selected.empty:
        return {}, "no_admitted_listing_window"
    if len(selected) != 1:
        return {}, "ambiguous_admitted_listing_window"
    return selected.iloc[0].to_dict(), ""


def _classify_observation(
    price: dict[str, Any] | None,
    quote_rows: list[dict[str, Any]],
    side: str,
    boundary: float,
) -> tuple[str, str]:
    quote_state = _classify_closing_rows(quote_rows, side, boundary)
    if quote_state == "non_limit_trade":
        return quote_state, ""
    if quote_state == "expanded_continuation_proven":
        return quote_state, ""
    if price is None:
        return "missing_ohlcv", "missing_ohlcv_without_closing_evidence"
    values = [_number(price.get(name)) for name in ("open", "high", "low", "close")]
    volume = _number(price.get("volume"))
    if any(value is None or value <= 0 for value in values):
        return "invalid_ohlcv", "invalid_ohlcv_prices"
    open_, high, low, close = values
    assert open_ is not None and high is not None and low is not None and close is not None
    if low > min(open_, close) or high < max(open_, close) or low > high:
        return "invalid_ohlcv", "invalid_ohlcv_range"
    if volume is None:
        return "missing_volume", "missing_volume_without_closing_evidence"
    if volume <= 0:
        return "zero_volume", "zero_volume_without_closing_evidence"
    tolerance = max(1e-9, abs(boundary) * 1e-10)
    if side == "upper":
        if high > boundary + tolerance:
            return "boundary_breach", "ohlcv_above_expanded_upper_boundary"
        non_limit = low < boundary - tolerance
    else:
        if low < boundary - tolerance:
            return "boundary_breach", "ohlcv_below_expanded_lower_boundary"
        non_limit = high > boundary + tolerance
    if non_limit:
        return "non_limit_trade", ""
    return "boundary_only", "positive_volume_boundary_only_without_closing_evidence"


def _classify_closing_rows(rows: list[dict[str, Any]], side: str, boundary: float) -> str:
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
                return "non_limit_trade"
    for row in rows:
        remaining = _truthy(row.get("boundary_orders_remaining"))
        auction_executed = _truthy(row.get("closing_auction_executed"))
        session_volume = _number(row.get("session_volume"))
        pre_close_volume = _number(row.get("pre_close_volume"))
        auction_price = _number(row.get("closing_auction_price"))
        auction_volume = _number(row.get("closing_auction_volume"))
        book_price = _number(row.get("best_bid_price" if side == "upper" else "best_ask_price"))
        at_boundary = book_price is not None and abs(book_price - boundary) <= tolerance
        condition_one = (
            session_volume == 0 and not auction_executed and remaining and at_boundary
        )
        condition_two = (
            pre_close_volume == 0
            and auction_price is not None
            and abs(auction_price - boundary) <= tolerance
            and auction_volume is not None
            and auction_volume > 0
            and remaining
        )
        if condition_one or condition_two:
            return "expanded_continuation_proven"
    return ""


def _base_outcome(case_id: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
    first = chain[0]
    return {
        "case_id": case_id,
        "security_code": first["security_code"],
        "security_name": first.get("security_name", ""),
        "side": first["side"],
        "activation_publication_date": first["publication_date"],
        "activation_effective_date": first["effective_date"],
        "activation_notice_id": first["notice_id"],
        "activation_source_url": first["source_url"],
        "activation_source_sha256": first["source_sha256"],
        "expanded_boundary_yen": first["expanded_boundary_yen"],
        "continuation_count": len(chain) - 1,
        "chain_notice_ids": "|".join(str(row["notice_id"]) for row in chain),
        "chain_source_sha256": "|".join(str(row["source_sha256"]) for row in chain),
        "outcome_status": "",
        "return_normal_date": "",
        "trigger_session_date": "",
        "trigger_evidence": "",
        "unresolved_reason": "",
        "sessions_examined": 0,
        "ohlcv_rows_observed": 0,
        "closing_evidence_rows_observed": 0,
    }


def _state_row(
    case_id: str,
    sequence: int,
    event: dict[str, Any],
    *,
    session_date: str = "",
    state: str,
    reason: str = "",
    listing: dict[str, Any] | None = None,
    price: dict[str, Any] | None = None,
    quote_rows: list[dict[str, Any]] | None = None,
    observation: str = "",
) -> dict[str, Any]:
    listing = listing or {}
    price = price or {}
    quote_rows = quote_rows or []
    return {
        "case_id": case_id,
        "state_sequence": sequence,
        "session_date": session_date or str(event.get("effective_date", "")),
        "state": state,
        "observation_class": observation,
        "state_reason": reason,
        "security_code": event.get("security_code", ""),
        "side": event.get("side", ""),
        "expanded_boundary_yen": event.get("expanded_boundary_yen", ""),
        "official_event_type": event.get("event_type", ""),
        "official_notice_id": event.get("notice_id", ""),
        "official_source_url": event.get("source_url", ""),
        "official_source_sha256": event.get("source_sha256", ""),
        "stable_listing_id": listing.get("stable_listing_id", ""),
        "stable_instrument_id": listing.get("stable_instrument_id", ""),
        "refinitiv_ric": listing.get("refinitiv_ric", ""),
        "open": price.get("open", ""),
        "high": price.get("high", ""),
        "low": price.get("low", ""),
        "close": price.get("close", ""),
        "volume": price.get("volume", ""),
        "ohlcv_source_path": price.get("_source_path", ""),
        "ohlcv_source_sha256": price.get("_source_sha256", ""),
        "closing_evidence_count": len(quote_rows),
        "closing_evidence_sha256": "|".join(sorted({str(row.get("source_sha256", "")) for row in quote_rows})),
    }


def _verify_declared_files(
    data: dict[str, Any], filenames: Sequence[str], provenance: list[dict[str, Any]]
) -> None:
    hashes = data["manifest"].get("output_sha256", {})
    for filename in filenames:
        path = data["directory"] / filename
        expected = hashes.get(filename)
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ReturnToNormalInputError(f"declared output hash mismatch: {path}")
        provenance.append(_provenance("input", filename, path, expected))


def _provenance(role: str, evidence_type: str, path: Path, digest: str) -> dict[str, Any]:
    return {
        "input_role": role,
        "evidence_type": evidence_type,
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _closing_columns() -> list[str]:
    return [
        "code", "session_date", "evidence_kind", "executed_price", "executed_volume",
        "session_volume", "pre_close_volume", "closing_auction_price", "closing_auction_volume",
        "closing_auction_executed", "boundary_orders_remaining", "best_bid_price",
        "best_bid_size", "best_ask_price", "best_ask_size", "source_path", "source_sha256",
    ]


def _outcome_columns() -> list[str]:
    return [
        "case_id", "security_code", "security_name", "side", "activation_publication_date",
        "activation_effective_date", "activation_notice_id", "activation_source_url",
        "activation_source_sha256", "expanded_boundary_yen", "continuation_count",
        "chain_notice_ids", "chain_source_sha256", "outcome_status", "return_normal_date",
        "trigger_session_date", "trigger_evidence", "unresolved_reason", "sessions_examined",
        "ohlcv_rows_observed", "closing_evidence_rows_observed",
    ]


def _state_columns() -> list[str]:
    return [
        "case_id", "state_sequence", "session_date", "state", "observation_class",
        "state_reason", "security_code", "side", "expanded_boundary_yen",
        "official_event_type", "official_notice_id", "official_source_url",
        "official_source_sha256", "stable_listing_id", "stable_instrument_id",
        "refinitiv_ric", "open", "high", "low", "close", "volume", "ohlcv_source_path",
        "ohlcv_source_sha256", "closing_evidence_count", "closing_evidence_sha256",
    ]


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReturnToNormalInputError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReturnToNormalInputError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ReturnToNormalInputError(f"JSON artifact is not an object: {path}")
    return value


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
        description="Produce an immutable fail-closed JPX post-2020 return-to-normal reconstruction."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--event-validation-dir", required=True)
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--calendar-validation-dir", required=True)
    parser.add_argument("--ohlcv-dir", required=True)
    parser.add_argument("--additional-ohlcv-dir", action="append", default=[])
    parser.add_argument("--raw-ohlcv-audit-root")
    parser.add_argument("--raw-historical-dir")
    parser.add_argument("--closing-evidence-dir")
    parser.add_argument("--artifact-root", default="artifacts/jpx_return_to_normal_reconstruction")
    parser.add_argument("--max-search-sessions", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    result = produce_jpx_return_to_normal(
        run_id=args.run_id,
        event_dir=args.event_dir,
        event_validation_dir=args.event_validation_dir,
        listing_dir=args.listing_dir,
        calendar_validation_dir=args.calendar_validation_dir,
        ohlcv_dir=args.ohlcv_dir,
        closing_evidence_dir=args.closing_evidence_dir,
        additional_ohlcv_dirs=args.additional_ohlcv_dir,
        raw_ohlcv_audit_root=args.raw_ohlcv_audit_root,
        raw_historical_dir=args.raw_historical_dir,
        artifact_root=args.artifact_root,
        max_search_sessions=args.max_search_sessions,
    )
    print(json.dumps({"manifest": str(result.manifest_path), "status": result.manifest["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
