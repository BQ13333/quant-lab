from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_lab.data.jpx_monthly_quotations import conservative_quotation_known_at


UTC = timezone.utc
VERSION = "2.0-monthly-official"
PRIMARY_EXCHANGES = {"TYO", "JSD", "OSA"}


@dataclass(frozen=True)
class JapanG1MonthlyMaterializerArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    missing_plan_path: Path
    manifest: dict[str, Any]


def write_japan_g1_monthly_materializer(
    *,
    run_id: str,
    monthly_qa_dir: str | Path,
    security_master_dirs: Iterable[str | Path],
    artifact_root: str | Path = "artifacts/japan_g1_monthly_materializer",
) -> JapanG1MonthlyMaterializerArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    monthly_dir = Path(monthly_qa_dir)
    master_dirs = [Path(path) for path in security_master_dirs]
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(artifact_dir)
    monthly_manifest_path = monthly_dir / "manifest.json"
    monthly_manifest = _read_json(monthly_manifest_path)
    blockers = _monthly_source_blockers(monthly_manifest)
    input_manifests = {"monthly_membership_qa": str(monthly_manifest_path.resolve())}
    input_hashes = {
        "monthly_membership_qa": _sha256(monthly_manifest_path)
        if monthly_manifest_path.is_file()
        else ""
    }
    master_frames = []
    for index, directory in enumerate(master_dirs):
        name = f"security_master_{index:02d}_{directory.name}"
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path)
        security_path = _security_master_path(directory, manifest)
        input_manifests[name] = str(manifest_path.resolve())
        input_hashes[name] = _sha256(manifest_path) if manifest_path.is_file() else ""
        if not manifest_path.is_file() or not security_path.is_file():
            blockers.append(f"{name}:security_master_input_missing")
            continue
        targeted_blockers = _targeted_shard_blockers(directory, manifest)
        if targeted_blockers:
            blockers.extend(f"{name}:{blocker}" for blocker in targeted_blockers)
            continue
        quote_chain_blockers = _validated_quote_chain_blockers(directory, manifest, security_path)
        if quote_chain_blockers:
            blockers.extend(f"{name}:{blocker}" for blocker in quote_chain_blockers)
            continue
        frame = pd.read_csv(security_path, dtype=str, low_memory=False).fillna("")
        frame["_source_artifact"] = directory.name
        master_frames.append(frame)
    if not master_frames:
        blockers.append("no_security_master_rows")

    monthly_path = Path(str(monthly_manifest.get("artifacts", {}).get("monthly_membership", "")))
    if blockers or not monthly_path.is_file():
        manifest = _blocked_manifest(
            run_id, artifact_dir, blockers or ["monthly_membership_missing"], input_manifests, input_hashes
        )
        _atomic_write_json(paths["manifest"], manifest)
        _atomic_write_csv(paths["missing_plan"], pd.DataFrame(columns=_missing_columns()))
        return JapanG1MonthlyMaterializerArtifacts(
            run_id, artifact_dir, paths["manifest"], paths["missing_plan"], manifest
        )

    membership = pd.read_parquet(monthly_path)
    blockers.extend(_monthly_data_blockers(membership))
    if blockers:
        manifest = _blocked_manifest(
            run_id, artifact_dir, blockers, input_manifests, input_hashes
        )
        _atomic_write_json(paths["manifest"], manifest)
        _atomic_write_csv(paths["missing_plan"], pd.DataFrame(columns=_missing_columns()))
        return JapanG1MonthlyMaterializerArtifacts(
            run_id, artifact_dir, paths["manifest"], paths["missing_plan"], manifest
        )
    security_master = _normalize_security_master(pd.concat(master_frames, ignore_index=True))
    segments = _reconstruct_segments(membership)
    reconciled, candidate_audit = _map_segments(segments, security_master)
    missing = reconciled.loc[
        ~reconciled["mapping_status"].eq("matched"), _missing_columns()
    ].copy()
    _atomic_write_csv(paths["missing_plan"], missing)
    _atomic_write_csv(paths["candidate_audit"], candidate_audit)

    reconciled["currency"] = "JPY"
    reconciled["pit_usable"] = reconciled["mapping_status"].eq("matched")
    reconciled["blocker_reason"] = ""
    reconciled.loc[~reconciled["pit_usable"], "blocker_reason"] = reconciled.loc[
        ~reconciled["pit_usable"], "mapping_status"
    ]
    matched = reconciled.loc[reconciled["pit_usable"]].copy()
    entities = _entities(matched)
    listing_windows = _listing_windows(matched)
    identifier_history = _identifier_history(matched)
    monthly_pit = _mapped_monthly_membership(membership, matched)
    reference = _symbol_map(reconciled)
    active = reference.loc[
        reference["right_open"].astype(bool) & reference["pit_usable"].astype(bool)
    ].copy()

    _atomic_write_parquet(paths["entities"], entities)
    _atomic_write_parquet(paths["listing_windows"], listing_windows)
    _atomic_write_parquet(paths["identifier_history"], identifier_history)
    _atomic_write_parquet(paths["universe_membership"], monthly_pit)
    _atomic_write_csv(paths["active_candidate"], active)
    _atomic_write_csv(paths["reference_candidate"], reference)
    _atomic_write_csv(paths["reconciliation"], reconciled)

    overlap_rows = _overlap_count(reference.loc[reference["pit_usable"]])
    unresolved = int((~reconciled["pit_usable"]).sum())
    monthly_unmapped = (
        int((~monthly_pit["identifier_mapped"]).sum())
        if not monthly_pit.empty
        else len(membership)
    )
    all_blockers = []
    if unresolved:
        all_blockers.append(f"unresolved_or_ambiguous_listing_windows={unresolved}")
    if monthly_unmapped:
        all_blockers.append(f"unmapped_monthly_membership_rows={monthly_unmapped}")
    if overlap_rows:
        all_blockers.append(f"overlapping_canonical_ric_windows={overlap_rows}")
    output_files = {
        "entities.parquet": paths["entities"],
        "listing_windows.parquet": paths["listing_windows"],
        "identifier_history.parquet": paths["identifier_history"],
        "universe_membership_pit.parquet": paths["universe_membership"],
        "active_symbol_map_candidate.csv": paths["active_candidate"],
        "reference_symbol_map_candidate.csv": paths["reference_candidate"],
        "reconciliation.csv": paths["reconciliation"],
    }
    manifest = {
        "schema_version": VERSION,
        "run_id": run_id,
        "producer_run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_type": "japan_g1_pit_materializer_producer",
        "status": "complete" if not all_blockers else "complete_with_blockers",
        "scorecard_eligible": False,
        "independent_audit": False,
        "historical_pit_ready": not all_blockers,
        "evidence_scope": "producer_candidate_only",
        "official_jpx_source": True,
        "source_method": "official_20_year_monthly_TSE_JASDAQ_OSE_membership_plus_LSEG_identifier_mapping",
        "known_at_policy": {
            "membership": "official monthly observation known conservatively on the 21st calendar day of the following month at 00:00 JST",
            "window_start": "first included monthly observation known_at",
            "window_end": "first subsequent monthly publication proving absence or venue-regime transition",
            "right_open": "final-source-month windows remain open and have null window_end_known_at",
        },
        "counts": {
            "monthly_membership_rows": len(membership),
            "candidate_segments": len(reconciled),
            "matched_segments": int(reconciled["pit_usable"].sum()),
            "unresolved_segments": unresolved,
            "mapped_monthly_rows": int(monthly_pit["identifier_mapped"].sum()) if not monthly_pit.empty else 0,
            "identifier_mapped_monthly_rows": int(monthly_pit["identifier_mapped"].sum()) if not monthly_pit.empty else 0,
            "tradable_monthly_rows": int(monthly_pit["pit_usable"].sum()) if not monthly_pit.empty else 0,
            "lifecycle_excluded_monthly_rows": int(monthly_pit["lifecycle_excluded"].sum()) if not monthly_pit.empty else 0,
            "unmapped_monthly_rows": monthly_unmapped,
            "active_candidate_rows": len(active),
            "reference_candidate_rows": len(reference),
            "security_master_input_rows": len(security_master),
            "overlap_rows": overlap_rows,
        },
        "critical_blockers": all_blockers,
        "input_manifests": input_manifests,
        "input_manifest_sha256": input_hashes,
        "input_data_sha256": {
            "monthly_membership.parquet": _sha256(monthly_path),
            **{
                f"security_master_{index:02d}.csv": _sha256(_security_master_path(directory, _read_json(directory / "manifest.json")))
                for index, directory in enumerate(master_dirs)
                if _security_master_path(directory, _read_json(directory / "manifest.json")).is_file()
            },
        },
        "output_sha256": {name: _sha256(path) for name, path in output_files.items()},
        "diagnostic_sha256": {
            "targeted_missing_code_plan.csv": _sha256(paths["missing_plan"]),
            "candidate_audit.csv": _sha256(paths["candidate_audit"]),
        },
        "artifacts": {
            "entities": str(paths["entities"].resolve()),
            "listing_windows": str(paths["listing_windows"].resolve()),
            "identifier_history": str(paths["identifier_history"].resolve()),
            "universe_membership": str(paths["universe_membership"].resolve()),
            "active_candidate": str(paths["active_candidate"].resolve()),
            "reference_candidate": str(paths["reference_candidate"].resolve()),
            "reconciliation": str(paths["reconciliation"].resolve()),
            "missing_plan": str(paths["missing_plan"].resolve()),
            "candidate_audit": str(paths["candidate_audit"].resolve()),
        },
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(paths["manifest"], manifest)
    return JapanG1MonthlyMaterializerArtifacts(
        run_id, artifact_dir, paths["manifest"], paths["missing_plan"], manifest
    )


def _monthly_source_blockers(manifest: dict[str, Any]) -> list[str]:
    if not (
        manifest.get("status") == "pass_candidate_source"
        and manifest.get("independent_validation") is True
        and int(manifest.get("planned_months", 0)) == 240
        and int(manifest.get("passed_months", 0)) == 240
        and int(manifest.get("anomalies", -1)) == 0
        and manifest.get("evidence_scope") == "full_20_year_monthly_tse_ose_union"
    ):
        return ["combined_monthly_membership_not_independently_complete"]
    for filename, digest in manifest.get("output_sha256", {}).items():
        path = Path(str(manifest.get("artifacts", {}).get(filename.rsplit(".", 1)[0], "")))
        if filename == "monthly_membership.parquet":
            path = Path(str(manifest.get("artifacts", {}).get("monthly_membership", "")))
        if not path.is_file() or _sha256(path) != str(digest):
            return [f"combined_monthly_output_hash_failure:{filename}"]
    return []


def _monthly_data_blockers(frame: pd.DataFrame) -> list[str]:
    required = {"observation_month", "code", "source_families", "known_at_utc", "currency", "pit_usable"}
    if frame.empty or not required.issubset(frame.columns):
        return ["combined_monthly_membership_schema_or_rows_invalid"]
    periods = pd.PeriodIndex(frame["observation_month"], freq="M")
    expected = pd.period_range("2006-07", "2026-06", freq="M")
    if list(sorted(periods.unique())) != list(expected):
        return ["combined_monthly_membership_240_month_denominator_invalid"]
    if frame.duplicated(["observation_month", "code"]).any():
        return ["combined_monthly_membership_duplicate_keys"]
    if not frame["currency"].astype(str).eq("JPY").all():
        return ["combined_monthly_membership_currency_invalid"]
    if not frame["pit_usable"].astype(str).str.lower().isin({"true", "1", "yes"}).all():
        return ["combined_monthly_membership_contains_unusable_rows"]
    if pd.to_datetime(frame["known_at_utc"], errors="coerce", utc=True).isna().any():
        return ["combined_monthly_membership_known_at_invalid"]
    return []


def _security_master_path(directory: Path, manifest: dict[str, Any]) -> Path:
    declared = manifest.get("artifacts", {}).get("security_master") if isinstance(manifest.get("artifacts"), dict) else None
    return Path(str(declared)) if declared else directory / "security_master.csv"


def _targeted_shard_blockers(directory: Path, manifest: dict[str, Any]) -> list[str]:
    if "SECMASTER-DISCOVERY-JPN-G1-TARGETED-MISSING-" not in directory.name or "-SHARD-" not in directory.name:
        return []
    request_path = directory / "targeted_request.json"
    request = _read_json(request_path)
    security_path = _security_master_path(directory, manifest)
    blockers = []
    if request.get("status") != "complete":
        blockers.append("targeted_request_not_complete")
    if request.get("run_id") != directory.name or manifest.get("run_id") != directory.name:
        blockers.append("targeted_run_id_mismatch")
    request_payload = {
        key: request.get(key)
        for key in (
            "run_id",
            "codes",
            "country_codes",
            "state_buckets",
            "extra_filter",
            "top",
            "max_pages",
            "input_plan_sha256",
        )
    }
    request_digest = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if request.get("request_sha256") != request_digest:
        blockers.append("targeted_request_hash_mismatch")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or request.get("manifest_sha256") != _sha256(manifest_path):
        blockers.append("targeted_manifest_hash_mismatch")
    if request.get("country_codes") != ["JPN"] or manifest.get("requested_country_codes") != ["JPN"]:
        blockers.append("targeted_country_scope_invalid")
    if request.get("state_buckets") != ["active", "inactive"]:
        blockers.append("targeted_state_scope_invalid")
    if request.get("extra_filter") != manifest.get("extra_filter"):
        blockers.append("targeted_filter_mismatch")
    expected_security_hash = manifest.get("output_sha256", {}).get("security_master")
    if not security_path.is_file() or not expected_security_hash or _sha256(security_path) != expected_security_hash:
        blockers.append("targeted_security_master_hash_mismatch")
    codes = request.get("codes")
    if not isinstance(codes, list) or not codes or any(not re.fullmatch(r"[0-9A-Z]{4}", str(code)) for code in codes):
        blockers.append("targeted_code_scope_invalid")
    return blockers


def _validated_quote_chain_blockers(
    directory: Path, manifest: dict[str, Any], security_path: Path
) -> list[str]:
    artifact_type = manifest.get("artifact_type")
    if artifact_type == "japan_g1_corporate_transition_independent_validation":
        return _validated_transition_blockers(directory, manifest, security_path)
    if artifact_type != "japan_g1_quote_chain_independent_validation":
        return []
    blockers: list[str] = []
    if not (
        manifest.get("status") == "pass_partial_component"
        and manifest.get("validation_status") == "pass"
        and manifest.get("independent_validation") is True
        and manifest.get("scorecard_eligible") is False
    ):
        blockers.append("quote_chain_validation_contract_invalid")
    expected = manifest.get("output_sha256", {}).get("security_master")
    if not security_path.is_file() or not expected or _sha256(security_path) != expected:
        blockers.append("quote_chain_security_master_hash_mismatch")
        return blockers
    frame = pd.read_csv(security_path, dtype=str).fillna("")
    required = {
        "ric",
        "ticker_symbol",
        "exchange_code",
        "instrument_permid",
        "quote_chain_mapping_route",
        "quote_chain_admitted",
    }
    if frame.empty or not required.issubset(frame.columns):
        blockers.append("quote_chain_security_master_schema_invalid")
        return blockers
    admitted = frame["quote_chain_admitted"].str.lower().isin({"true", "1", "yes"})
    direct = frame["quote_chain_mapping_route"].eq("independently_validated_direct_code")
    ric_codes = frame["ric"].map(_code_from_primary_ric)
    valid = (
        admitted
        & direct
        & frame["exchange_code"].isin(PRIMARY_EXCHANGES)
        & ric_codes.eq(frame["ticker_symbol"])
        & frame["instrument_permid"].ne("")
    )
    if not valid.all():
        blockers.append("quote_chain_contains_unadmitted_or_successor_rows")
    return blockers


def _validated_transition_blockers(
    directory: Path, manifest: dict[str, Any], security_path: Path
) -> list[str]:
    blockers: list[str] = []
    counts = manifest.get("counts", {})
    if not (
        manifest.get("status") == "pass_component"
        and manifest.get("validation_status") == "pass"
        and manifest.get("independent_validation") is True
        and manifest.get("clean_room_evidence_checks") is True
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("scorecard_eligible") is False
        and not manifest.get("critical_blockers")
        and int(counts.get("transition_denominator", 0)) == 66
        and int(counts.get("validated_transition_rows", 0)) == 66
        and int(counts.get("mapped_transition_rows", 0)) == 66
        and int(counts.get("validated_successor_boundary_rows", 0)) == 66
        and int(counts.get("direct_security_master_rows", 0)) == 32
        and int(counts.get("combined_security_master_rows", 0)) == 164
    ):
        blockers.append("transition_validation_contract_invalid")
    expected = manifest.get("output_sha256", {}).get("security_master")
    if not security_path.is_file() or not expected or _sha256(security_path) != expected:
        blockers.append("transition_security_master_hash_mismatch")
        return blockers
    for key, path_text in manifest.get("input_manifests", {}).items():
        path = Path(str(path_text))
        digest = manifest.get("input_manifest_sha256", {}).get(key, "")
        if not path.is_file() or not digest or _sha256(path) != digest:
            blockers.append(f"transition_input_lineage_mismatch:{key}")
    frame = pd.read_csv(security_path, dtype=str).fillna("")
    required = {
        "ric",
        "ticker_symbol",
        "exchange_code",
        "instrument_permid",
        "quote_chain_mapping_route",
        "quote_chain_admitted",
        "mapping_local_code_override",
    }
    if len(frame) != 164 or not required.issubset(frame.columns):
        blockers.append("transition_security_master_schema_or_denominator_invalid")
        return blockers
    admitted = frame["quote_chain_admitted"].str.lower().isin({"true", "1", "yes"})
    direct = frame["quote_chain_mapping_route"].eq("independently_validated_direct_code")
    transition = frame["quote_chain_mapping_route"].eq(
        "independently_validated_official_transition_alias"
    )
    boundary = frame["quote_chain_mapping_route"].eq(
        "independently_validated_official_successor_boundary"
    )
    ric_codes = frame["ric"].map(_code_from_primary_ric)
    direct_valid = (
        admitted
        & direct
        & frame["exchange_code"].isin(PRIMARY_EXCHANGES)
        & ric_codes.eq(frame["ticker_symbol"])
        & frame["instrument_permid"].ne("")
    )
    transition_admitted = frame.get(
        "transition_admitted", pd.Series("", index=frame.index)
    ).str.lower().isin({"true", "1", "yes"})
    transition_valid = (
        admitted
        & transition
        & transition_admitted
        & frame["exchange_code"].isin(PRIMARY_EXCHANGES)
        & ric_codes.eq(frame["ticker_symbol"])
        & frame["mapping_local_code_override"].str.fullmatch(r"[0-9A-Z]{4}")
        & frame.get("transition_old_code", pd.Series("", index=frame.index)).eq(
            frame["mapping_local_code_override"]
        )
        & frame.get("transition_successor_code", pd.Series("", index=frame.index)).eq(
            frame["ticker_symbol"]
        )
        & frame.get("transition_old_event_date", pd.Series("", index=frame.index)).ne("")
        & frame.get("transition_successor_listing_date", pd.Series("", index=frame.index)).ne("")
        & frame["instrument_permid"].ne("")
    )
    boundary_valid = (
        admitted
        & boundary
        & transition_admitted
        & frame["exchange_code"].isin(PRIMARY_EXCHANGES)
        & ric_codes.eq(frame["ticker_symbol"])
        & frame["mapping_local_code_override"].eq(frame["ticker_symbol"])
        & frame.get("transition_successor_code", pd.Series("", index=frame.index)).eq(
            frame["ticker_symbol"]
        )
        & frame.get("transition_successor_listing_date", pd.Series("", index=frame.index)).eq(
            frame.get("first_trade_date", pd.Series("", index=frame.index))
        )
        & frame["instrument_permid"].ne("")
    )
    if int(direct.sum()) != 32 or not direct_valid.loc[direct].all():
        blockers.append("transition_artifact_direct_rows_invalid")
    if int(transition.sum()) != 66 or not transition_valid.loc[transition].all():
        blockers.append("transition_artifact_alias_rows_invalid")
    if int(boundary.sum()) != 66 or not boundary_valid.loc[boundary].all():
        blockers.append("transition_artifact_successor_boundary_rows_invalid")
    if not (direct | transition | boundary).all():
        blockers.append("transition_artifact_contains_unknown_route")
    return blockers


def _normalize_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ric", "ticker_symbol", "exchange_code", "rcs_asset_category_leaf"}
    if not required.issubset(frame.columns):
        raise ValueError(f"security master missing columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    for column in ("ric", "ticker_symbol", "exchange_code", "rcs_asset_category_leaf", "isin", "sedol", "perm_id", "instrument_permid", "is_primary_ric", "is_primary_issue_ric"):
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    result["first_trade_date"] = pd.to_datetime(result.get("first_trade_date"), errors="coerce")
    result["retire_date"] = pd.to_datetime(result.get("retire_date"), errors="coerce")
    result["mapping_local_code"] = result["ticker_symbol"].where(
        result["ticker_symbol"].str.fullmatch(r"[0-9A-Z]{4}"), ""
    )
    derived = result["ric"].map(_code_from_primary_ric)
    result.loc[result["mapping_local_code"].eq(""), "mapping_local_code"] = derived
    if "mapping_local_code_override" in result.columns:
        override = result["mapping_local_code_override"].fillna("").astype(str).str.strip()
        valid_override = override.str.fullmatch(r"[0-9A-Z]{4}")
        result.loc[valid_override, "mapping_local_code"] = override.loc[valid_override]
    result = result.loc[
        result["exchange_code"].isin(PRIMARY_EXCHANGES)
        & result["rcs_asset_category_leaf"].str.contains("Ordinary Share", case=False, na=False)
        & result["ric"].ne("")
        & result["mapping_local_code"].ne("")
    ].copy()
    return result.drop_duplicates(["ric", "mapping_local_code"], keep="last").reset_index(drop=True)


def _reconstruct_segments(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"observation_month", "code", "source_families", "known_at_utc"}
    if membership.empty or not required.issubset(membership.columns):
        raise ValueError("monthly membership is empty or missing required columns")
    data = membership.copy()
    data["period"] = pd.PeriodIndex(data["observation_month"], freq="M")
    rows = []
    final_period = data["period"].max()
    for code, group in data.groupby("code", sort=True):
        group = group.sort_values("period")
        run = []
        previous = None
        family = None
        ordinal = 0
        for record in group.to_dict("records"):
            period = record["period"]
            if run and (period.ordinal != previous.ordinal + 1 or record["source_families"] != family):
                ordinal += 1
                rows.append(_segment_row(str(code), run, ordinal, final_period))
                run = []
            run.append(record)
            previous = period
            family = record["source_families"]
        if run:
            ordinal += 1
            rows.append(_segment_row(str(code), run, ordinal, final_period))
    return pd.DataFrame(rows)


def _code_from_primary_ric(ric: str) -> str:
    value = str(ric).strip().upper()
    direct = re.match(r"^([0-9A-Z]{4})(?:L)?\.(?:T|Q|OS|OJ)(?:\^|$)", value)
    if direct:
        return direct.group(1)
    odd = re.match(r"^O([0-9A-Z]{4})O\.(?:T|Q|OS|OJ)(?:\^|$)", value)
    return odd.group(1) if odd else ""


def _segment_row(code: str, records: list[dict[str, Any]], ordinal: int, final_period: pd.Period) -> dict[str, Any]:
    start = records[0]["period"]
    end = records[-1]["period"]
    right_open = end == final_period
    end_known_at = "" if right_open else conservative_quotation_known_at(end + 1)
    return {
        "stable_listing_id": f"JPX:{code}:{start}:{ordinal}",
        "local_code": code,
        "listing_start": start.start_time.date().isoformat(),
        "listing_end": end.end_time.date().isoformat(),
        "known_at": str(records[0]["known_at_utc"]),
        "window_end_known_at": end_known_at,
        "right_open": right_open,
        "source_families": str(records[0]["source_families"]),
        "source_artifacts": str(records[0].get("source_artifacts", "")),
        "source_months": len(records),
        "mapping_status": "unmapped",
    }


def _preferred_exchanges(families: str) -> list[str]:
    values = set(str(families).split("|"))
    if "TSE" in values and (
        "JASDAQ" in values or "OSE_MAIN" in values or "OSE_COMBINED" in values
    ):
        return ["TYO", "OSA", "JSD"]
    if "TSE" in values:
        return ["TYO"]
    if "JASDAQ" in values:
        return ["JSD", "OSA", "TYO"]
    if "OSE_COMBINED" in values:
        return ["OSA", "JSD", "TYO"]
    return ["OSA", "JSD", "TYO"]


def _map_segments(segments: pd.DataFrame, security_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_rows = []
    audit_rows = []
    by_code = {code: group.copy() for code, group in security_master.groupby("mapping_local_code")}
    for segment in segments.to_dict("records"):
        candidates = by_code.get(str(segment["local_code"]), pd.DataFrame(columns=security_master.columns)).copy()
        start = pd.Timestamp(segment["listing_start"])
        end = pd.Timestamp(segment["listing_end"])
        if not candidates.empty:
            candidates = candidates.loc[
                (candidates["first_trade_date"].isna() | candidates["first_trade_date"].le(end))
                & (candidates["retire_date"].isna() | candidates["retire_date"].ge(start))
            ].copy()
        preferences = _preferred_exchanges(str(segment["source_families"]))
        if not candidates.empty:
            candidates = candidates.loc[candidates["exchange_code"].isin(preferences)].copy()
        if not candidates.empty:
            code = str(segment["local_code"])
            candidates["exchange_rank"] = candidates["exchange_code"].map(
                {exchange: rank for rank, exchange in enumerate(preferences)}
            ).fillna(9)
            candidates["canonical_ric_rank"] = candidates["ric"].map(
                lambda ric: 0 if str(ric).startswith(f"{code}.") else 1 if str(ric).startswith(f"{code}L.") else 2
            )
            candidates["primary_rank"] = (
                ~candidates["is_primary_ric"].str.lower().eq("true")
            ).astype(int) + (~candidates["is_primary_issue_ric"].str.lower().eq("true")).astype(int)
            candidates["lifecycle_rank"] = 0
            if not bool(segment.get("right_open", False)):
                retirement_gap = (candidates["retire_date"] - end).abs().dt.days
                candidates["lifecycle_rank"] = (~(
                    candidates["retire_date"].notna() & retirement_gap.le(45)
                )).astype(int)
            candidates = candidates.sort_values(
                ["canonical_ric_rank", "exchange_rank", "primary_rank", "lifecycle_rank", "ric"]
            )
            top = candidates.iloc[0]
            finalists = candidates.loc[
                candidates["canonical_ric_rank"].eq(top["canonical_ric_rank"])
                & candidates["exchange_rank"].eq(top["exchange_rank"])
                & candidates["primary_rank"].eq(top["primary_rank"])
                & candidates["lifecycle_rank"].eq(top["lifecycle_rank"])
            ].copy()
        else:
            finalists = candidates
        for candidate in candidates.to_dict("records"):
            audit_rows.append(
                {
                    "stable_listing_id": segment["stable_listing_id"],
                    "local_code": segment["local_code"],
                    "listing_start": segment["listing_start"],
                    "listing_end": segment["listing_end"],
                    "source_families": segment["source_families"],
                    "candidate_ric": candidate.get("ric", ""),
                    "exchange_code": candidate.get("exchange_code", ""),
                    "exchange_rank": candidate.get("exchange_rank", ""),
                    "canonical_ric_rank": candidate.get("canonical_ric_rank", ""),
                    "primary_rank": candidate.get("primary_rank", ""),
                    "lifecycle_rank": candidate.get("lifecycle_rank", ""),
                }
            )
        output = dict(segment)
        unique_rics = finalists["ric"].nunique() if not finalists.empty else 0
        if unique_rics == 1:
            chosen = finalists.iloc[0]
            exact_start = start
            exact_end = end
            if pd.notna(chosen["first_trade_date"]):
                exact_start = max(exact_start, chosen["first_trade_date"])
            if pd.notna(chosen["retire_date"]):
                exact_end = min(exact_end, chosen["retire_date"])
            output.update(
                {
                    "refinitiv_ric": chosen["ric"],
                    "exchange_code": chosen["exchange_code"],
                    "isin": chosen.get("isin", ""),
                    "sedol": chosen.get("sedol", ""),
                    "perm_id": chosen.get("perm_id", ""),
                    "instrument_permid": chosen.get("instrument_permid", ""),
                    "stable_instrument_id": _stable_instrument_id(chosen),
                    "preferred_exchange_code": preferences[0],
                    "mapping_route": (
                        "exact_primary_exchange"
                        if chosen["exchange_code"] == preferences[0]
                        else "successor_canonical_exchange"
                    ),
                    "mapping_status": "matched",
                    "candidate_count": len(candidates),
                    "monthly_observed_listing_start": segment["listing_start"],
                    "monthly_observed_listing_end": segment["listing_end"],
                    "listing_start": exact_start.date().isoformat(),
                    "listing_end": exact_end.date().isoformat(),
                }
            )
        else:
            output.update(
                {
                    "refinitiv_ric": "",
                    "exchange_code": "",
                    "isin": "",
                    "sedol": "",
                    "perm_id": "",
                    "instrument_permid": "",
                    "stable_instrument_id": "",
                    "preferred_exchange_code": preferences[0],
                    "mapping_route": "",
                    "mapping_status": "unresolved_no_candidate" if unique_rics == 0 else "ambiguous_best_candidate",
                    "candidate_count": len(candidates),
                }
            )
        mapped_rows.append(output)
    return pd.DataFrame(mapped_rows), pd.DataFrame(audit_rows)


def _stable_instrument_id(row: pd.Series) -> str:
    for label, field in (("isin", "isin"), ("permid", "perm_id"), ("instrument_permid", "instrument_permid"), ("ric", "ric")):
        value = str(row.get(field, "")).strip()
        if value and value.lower() != "nan":
            return f"{label}:{value}"
    return ""


def _entities(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["stable_listing_id", "stable_instrument_id", "local_code", "refinitiv_ric", "exchange_code", "isin", "sedol", "perm_id"]].drop_duplicates("stable_listing_id")


def _listing_windows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stable_listing_id", "stable_instrument_id", "refinitiv_ric", "local_code", "exchange_code", "listing_start", "listing_end", "known_at", "window_end_known_at", "right_open", "source_families", "currency", "pit_usable"]
    return frame.reindex(columns=columns).reset_index(drop=True)


def _identifier_history(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        for kind, field in (("local_code", "local_code"), ("ric", "refinitiv_ric"), ("isin", "isin"), ("sedol", "sedol")):
            value = str(row.get(field, "")).strip()
            if not value or value.lower() == "nan":
                continue
            rows.append(
                {
                    "stable_listing_id": row["stable_listing_id"],
                    "identifier_type": kind,
                    "identifier_value": value,
                    "valid_from": row["listing_start"],
                    "valid_to": row["listing_end"],
                    "known_at": row["known_at"],
                    "window_end_known_at": row["window_end_known_at"],
                }
            )
    return pd.DataFrame(rows)


def _mapped_monthly_membership(membership: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    lookup_rows = []
    for row in matched.to_dict("records"):
        observed_start = row.get("monthly_observed_listing_start") or row["listing_start"]
        observed_end = row.get("monthly_observed_listing_end") or row["listing_end"]
        for period in pd.period_range(pd.Timestamp(observed_start), pd.Timestamp(observed_end), freq="M"):
            lookup_rows.append(
                {
                    "observation_month": str(period),
                    "code": row["local_code"],
                    "stable_listing_id": row["stable_listing_id"],
                    "stable_instrument_id": row["stable_instrument_id"],
                    "refinitiv_ric": row["refinitiv_ric"],
                    "exchange_code": row["exchange_code"],
                    "listing_valid_from": row["listing_start"],
                    "listing_valid_to": row["listing_end"],
                }
            )
    lookup = pd.DataFrame(
        lookup_rows,
        columns=[
            "observation_month",
            "code",
            "stable_listing_id",
            "stable_instrument_id",
            "refinitiv_ric",
            "exchange_code",
            "listing_valid_from",
            "listing_valid_to",
        ],
    )
    result = membership.merge(lookup, on=["observation_month", "code"], how="left", validate="one_to_one")
    result["identifier_mapped"] = result["refinitiv_ric"].notna()
    month_end = pd.Series(
        pd.PeriodIndex(result["observation_month"], freq="M").end_time.normalize(),
        index=result.index,
    )
    valid_from = pd.to_datetime(result["listing_valid_from"], errors="coerce")
    valid_to = pd.to_datetime(result["listing_valid_to"], errors="coerce")
    result["lifecycle_excluded"] = result["identifier_mapped"] & (
        month_end.lt(valid_from) | month_end.gt(valid_to)
    )
    result["pit_usable"] = (
        result["identifier_mapped"]
        & result["known_at_utc"].notna()
        & ~result["lifecycle_excluded"]
    )
    result["lifecycle_exclusion_reason"] = ""
    result.loc[
        result["lifecycle_excluded"], "lifecycle_exclusion_reason"
    ] = "official_monthly_observation_outside_exact_listing_window"
    result["membership_source"] = "official_TSE_JASDAQ_OSE_monthly_quotations"
    return result


def _symbol_map(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stable_listing_id", "stable_instrument_id", "local_code", "refinitiv_ric", "exchange_code", "preferred_exchange_code", "mapping_route", "listing_start", "listing_end", "known_at", "window_end_known_at", "right_open", "source_families", "currency", "mapping_status", "candidate_count", "pit_usable", "blocker_reason"]
    return frame.reindex(columns=columns).reset_index(drop=True)


def _overlap_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    dated = frame.assign(
        _start=pd.to_datetime(frame["listing_start"], errors="coerce"),
        _end=pd.to_datetime(frame["listing_end"], errors="coerce"),
    ).sort_values(["refinitiv_ric", "_start", "_end"])
    previous = dated.groupby("refinitiv_ric", sort=False)["_end"].shift()
    return int(dated["_start"].le(previous).fillna(False).sum())


def _missing_columns() -> list[str]:
    return ["stable_listing_id", "local_code", "listing_start", "listing_end", "source_families", "mapping_status", "candidate_count"]


def _paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "manifest.json",
        "entities": root / "entities.parquet",
        "listing_windows": root / "listing_windows.parquet",
        "identifier_history": root / "identifier_history.parquet",
        "universe_membership": root / "universe_membership_pit.parquet",
        "active_candidate": root / "active_symbol_map_candidate.csv",
        "reference_candidate": root / "reference_symbol_map_candidate.csv",
        "reconciliation": root / "reconciliation.csv",
        "missing_plan": root / "targeted_missing_code_plan.csv",
        "candidate_audit": root / "candidate_audit.csv",
    }


def _blocked_manifest(run_id: str, root: Path, blockers: list[str], input_paths: dict[str, str], input_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": VERSION,
        "run_id": run_id,
        "producer_run_id": run_id,
        "artifact_dir": str(root.resolve()),
        "artifact_type": "japan_g1_pit_materializer_producer",
        "status": "blocked_upstream",
        "scorecard_eligible": False,
        "independent_audit": False,
        "historical_pit_ready": False,
        "evidence_scope": "blocked_no_candidate_output",
        "critical_blockers": blockers,
        "input_manifests": input_paths,
        "input_manifest_sha256": input_hashes,
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode())


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
