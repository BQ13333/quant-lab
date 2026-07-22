from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


DEFAULT_JAPAN_G1_SOURCE_DIR = Path(
    "artifacts/research_universes/JPX-UNIVERSE-RECONCILIATION-V2-20260711"
)
DEFAULT_JAPAN_G1_QA_ARTIFACT_ROOT = Path("artifacts/japan_g1_reconciliation_qa")

RECONCILIATION_FILE = "jpx_security_master_reconciliation.csv"
UNMATCHED_FILE = "jpx_security_master_unmatched.csv"
STABLE_INSTRUMENTS_FILE = "japan_stable_instruments.csv"

ACTION_CATEGORIES: tuple[str, ...] = (
    "ticker_normalization",
    "duplicate_quote_listing",
    "out_of_scope_market",
    "alpha_codes",
    "future_retired_status",
    "stale_active_flag",
    "missing_primary_ric",
    "truly_unresolved",
)

RESOLVABILITY_RANK = {
    "ticker_normalization": 1,
    "duplicate_quote_listing": 2,
    "out_of_scope_market": 3,
    "alpha_codes": 4,
    "future_retired_status": 5,
    "stale_active_flag": 6,
    "missing_primary_ric": 7,
    "truly_unresolved": 8,
}

_CATEGORY_ACTION = {
    "ticker_normalization": (
        "Validate the candidate code against the listing class and stable identifier; "
        "do not map an issuer ordinary share onto a distinct preferred/class security."
    ),
    "duplicate_quote_listing": (
        "Resolve duplicate Tokyo quote/listing evidence by stable identifier and confirm "
        "which record owns the current public listing."
    ),
    "out_of_scope_market": (
        "Confirm the official JPX market/security type, then record an explicit scope "
        "decision before excluding the row."
    ),
    "alpha_codes": (
        "Validate the alphanumeric JPX code and market segment against a current official "
        "listing source with alpha-code support."
    ),
    "future_retired_status": (
        "Validate first/retire dates and effective status with current listing evidence; "
        "do not infer historical membership."
    ),
    "stale_active_flag": (
        "Refresh the security-master active flag and retirement evidence, preserving the "
        "conflicting source rows."
    ),
    "missing_primary_ric": (
        "Obtain or validate the current primary issue RIC for the exact stable instrument."
    ),
    "truly_unresolved": (
        "Research the exact stable identifier, official market segment, and current listing "
        "status; keep the denominator difference open until source-backed resolution."
    ),
}

_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\bREIT\b|REAL ESTATE INVESTMENT|INFRASTRUCTURE|\bIF ORD\b|"
    r"BANK OF JAPAN|PREFERRED|PREFERENCE|SUBSCRIPTION|投資法人|優先|種類株式",
    flags=re.IGNORECASE,
)

_RECONCILIATION_REQUIRED = {
    "source_side",
    "local_code",
    "stable_instrument_id",
    "ric",
    "reconciliation_status",
}
_STABLE_REQUIRED = {
    "local_code",
    "stable_instrument_id",
    "ric",
    "is_active",
    "is_primary_ric",
}


@dataclass(frozen=True)
class JapanG1ReconciliationQAArtifacts:
    run_id: str
    artifact_dir: Path
    classified_rows_path: Path
    category_counts_path: Path
    agent_next_queue_path: Path
    source_checksums_path: Path
    checksums_path: Path
    manifest_path: Path
    classified_rows: pd.DataFrame
    category_counts: pd.DataFrame
    agent_next_queue: pd.DataFrame
    manifest: dict[str, object]


def default_japan_g1_reconciliation_qa_run_id(
    source_dir: str | Path = DEFAULT_JAPAN_G1_SOURCE_DIR,
) -> str:
    return f"{Path(source_dir).name}-G1-DENOMINATOR-DIFFERENCE-QA"


def classify_japan_g1_denominator_differences(
    reconciliation: pd.DataFrame,
    unmatched: pd.DataFrame,
    stable_instruments: pd.DataFrame,
    *,
    as_of_date: str | date | datetime | None = None,
) -> pd.DataFrame:
    """Classify the public/master denominator difference without waiving any row."""

    _require_columns(reconciliation, _RECONCILIATION_REQUIRED, "reconciliation")
    _require_columns(unmatched, _RECONCILIATION_REQUIRED, "unmatched")
    _require_columns(stable_instruments, _STABLE_REQUIRED, "stable instruments")
    _verify_unmatched_is_reconciliation_subset(reconciliation, unmatched)

    target = unmatched.loc[
        unmatched["reconciliation_status"].astype(str).isin(
            {"public_in_scope_unmatched", "master_active_unmatched"}
        )
    ].copy()
    target["source_unmatched_row_number"] = target.index.to_series().astype(int) + 2
    target = target.reset_index(drop=True)
    stable = stable_instruments.copy().reset_index(drop=True)
    stable["_stable_source_row_number"] = stable.index + 2
    reference_date = _as_date(as_of_date)

    stable_by_id = {
        str(key): group.copy()
        for key, group in stable.loc[
            stable["stable_instrument_id"].map(_text).ne("")
        ].groupby(stable["stable_instrument_id"].map(_text), sort=False)
    }
    stable_by_code = {
        str(key): group.copy()
        for key, group in stable.loc[
            stable["local_code"].map(_normalize_code).ne("")
        ].groupby(stable["local_code"].map(_normalize_code), sort=False)
    }

    records: list[dict[str, object]] = []
    for row in target.to_dict("records"):
        source_side = _text(row.get("source_side"))
        local_code = _normalize_code(row.get("local_code"))
        stable_id = _text(row.get("stable_instrument_id"))
        exact = stable_by_id.get(stable_id, pd.DataFrame()) if stable_id else pd.DataFrame()
        if exact.empty and local_code:
            exact = stable_by_code.get(local_code, pd.DataFrame())

        normalized_code = _ticker_normalization_candidate(local_code, stable_by_code)
        normalized = (
            stable_by_code.get(normalized_code, pd.DataFrame())
            if normalized_code
            else pd.DataFrame()
        )
        evidence = exact if not exact.empty else normalized
        category, reason = _classify_row(
            row=row,
            exact=exact,
            normalized=normalized,
            normalized_code=normalized_code,
            stable_by_code=stable_by_code,
            reference_date=reference_date,
        )
        if category == "duplicate_quote_listing" and local_code in stable_by_code:
            evidence = stable_by_code[local_code]
        elif category == "ticker_normalization" and evidence.empty:
            ric_code = _normalize_code(_ric_local_code(row.get("ric")))
            evidence = stable_by_code.get(ric_code, evidence)
        qa_row_id = _qa_row_id(row)
        record = {
            "qa_row_id": qa_row_id,
            **row,
            "action_category": category,
            "resolvability_rank": RESOLVABILITY_RANK[category],
            "classification_reason": reason,
            "recommended_action": _CATEGORY_ACTION[category],
            "resolution_state": "open",
            "requires_source_backed_resolution": True,
            "auto_waived": False,
            "point_in_time_claimed": False,
            "ticker_normalization_candidate": normalized_code,
            **_evidence_fields(evidence),
            "source_row_evidence_json": _records_json([row]),
            "stable_row_evidence_json": _records_json(evidence.to_dict("records")),
        }
        records.append(record)

    classified = pd.DataFrame(records)
    if classified.empty:
        raise ValueError("no public-unmatched or active-master-only rows were found")
    if classified["qa_row_id"].duplicated().any():
        raise ValueError("QA row identifiers are not unique")
    if not classified["action_category"].isin(ACTION_CATEGORIES).all():
        raise AssertionError("an unsupported Japan G1 action category was emitted")
    return classified


def build_agent_next_queue(classified_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "qa_row_id",
        "source_side",
        "jpx_source_row",
        "local_code",
        "jpx_name",
        "stable_instrument_id",
        "ric",
        "action_category",
        "resolvability_rank",
        "classification_reason",
        "recommended_action",
        "ticker_normalization_candidate",
        "evidence_candidate_rics",
        "evidence_retire_dates",
        "resolution_state",
        "requires_source_backed_resolution",
        "auto_waived",
        "point_in_time_claimed",
    ]
    queue = classified_rows.reindex(columns=columns).copy()
    queue["_source_order"] = queue["source_side"].map(
        {"public_jpx": 0, "security_master": 1}
    ).fillna(2)
    queue["_row_order"] = pd.to_numeric(
        classified_rows.get("source_unmatched_row_number"), errors="coerce"
    )
    queue = queue.sort_values(
        ["resolvability_rank", "_source_order", "_row_order", "local_code"],
        kind="stable",
        na_position="last",
    ).drop(columns=["_source_order", "_row_order"])
    queue.insert(0, "agent_priority", range(1, len(queue) + 1))
    return queue.reset_index(drop=True)


def write_japan_g1_reconciliation_qa_artifacts(
    *,
    source_dir: str | Path = DEFAULT_JAPAN_G1_SOURCE_DIR,
    artifact_root: str | Path = DEFAULT_JAPAN_G1_QA_ARTIFACT_ROOT,
    run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> JapanG1ReconciliationQAArtifacts:
    """Build and atomically publish an immutable denominator-difference QA run."""

    source = Path(source_dir)
    source_paths = {
        "reconciliation": source / RECONCILIATION_FILE,
        "unmatched": source / UNMATCHED_FILE,
        "stable_instruments": source / STABLE_INSTRUMENTS_FILE,
    }
    for path in source_paths.values():
        _require_source(path)

    resolved_run_id = run_id or default_japan_g1_reconciliation_qa_run_id(source)
    if not resolved_run_id.strip() or Path(resolved_run_id).name != resolved_run_id:
        raise ValueError("run_id must be a non-empty path-safe name")

    root = Path(artifact_root)
    final_dir = root / resolved_run_id
    if final_dir.exists():
        raise FileExistsError(f"immutable artifact directory already exists: {final_dir}")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{resolved_run_id}.lock"
    lock_fd = _acquire_lock(lock_path)
    stage_dir: Path | None = None
    try:
        if final_dir.exists():
            raise FileExistsError(f"immutable artifact directory already exists: {final_dir}")
        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}.tmp-", dir=root)
        )
        source_hashes = {key: _sha256(path) for key, path in source_paths.items()}
        source_manifest = _read_json(source / "manifest.json")
        manifest_verification = _verify_source_manifest_hashes(
            source_manifest, source_hashes
        )

        reconciliation = _read_csv(source_paths["reconciliation"])
        unmatched = _read_csv(source_paths["unmatched"])
        stable = _read_csv(source_paths["stable_instruments"])
        as_of_date = _source_as_of_date(reconciliation, source_manifest)
        classified = classify_japan_g1_denominator_differences(
            reconciliation,
            unmatched,
            stable,
            as_of_date=as_of_date,
        )
        queue = build_agent_next_queue(classified)
        category_counts = _category_counts(classified)
        source_checksums = pd.DataFrame(
            [
                {
                    "source_key": key,
                    "file_name": path.name,
                    "sha256": source_hashes[key],
                    "bytes": path.stat().st_size,
                    "manifest_checksum_present": manifest_verification[key][0],
                    "manifest_checksum_verified": manifest_verification[key][1],
                }
                for key, path in source_paths.items()
            ]
        )

        stage_paths = {
            "classified_rows": stage_dir / "classified_denominator_differences.csv",
            "category_counts": stage_dir / "category_counts.csv",
            "agent_next_queue": stage_dir / "agent_next_queue.csv",
            "source_checksums": stage_dir / "source_checksums.csv",
        }
        _write_csv(classified, stage_paths["classified_rows"])
        _write_csv(category_counts, stage_paths["category_counts"])
        _write_csv(queue, stage_paths["agent_next_queue"])
        _write_csv(source_checksums, stage_paths["source_checksums"])

        output_hashes = {key: _sha256(path) for key, path in stage_paths.items()}
        checksums_path = stage_dir / "checksums.sha256"
        _write_text(
            checksums_path,
            "".join(
                f"{output_hashes[key]}  {stage_paths[key].name}\n"
                for key in sorted(stage_paths)
            ),
        )

        current_source_hashes = {
            key: _sha256(path) for key, path in source_paths.items()
        }
        if current_source_hashes != source_hashes:
            raise RuntimeError("Japan G1 reconciliation QA source checksum changed during build")

        public_count = int(
            classified["reconciliation_status"].eq("public_in_scope_unmatched").sum()
        )
        master_count = int(
            classified["reconciliation_status"].eq("master_active_unmatched").sum()
        )
        excluded_count = int(len(unmatched) - len(classified))
        category_count_map = {
            str(row.action_category): int(row.row_count)
            for row in category_counts.itertuples(index=False)
        }
        generated_at = (clock or (lambda: datetime.now(timezone.utc)))()
        manifest: dict[str, object] = {
            "schema_version": "1.0",
            "artifact_type": "japan_g1_denominator_difference_qa",
            "run_id": resolved_run_id,
            "artifact_dir": str(final_dir.resolve()),
            "status": (
                "qa_complete_with_unresolved"
                if category_count_map["truly_unresolved"]
                else "qa_complete_requires_actions"
            ),
            "immutable": True,
            "atomic_publish": True,
            "classification_as_of_date": as_of_date.isoformat() if as_of_date else None,
            "counts": {
                "source_unmatched_rows": int(len(unmatched)),
                "public_in_scope_unmatched_rows": public_count,
                "active_master_only_rows": master_count,
                "target_denominator_difference_rows": int(len(classified)),
                "classified_rows": int(len(classified)),
                "excluded_preclassified_rows": excluded_count,
                "agent_next_queue_rows": int(len(queue)),
                "truly_unresolved_rows": category_count_map["truly_unresolved"],
                "auto_waived_rows": int(classified["auto_waived"].astype(bool).sum()),
            },
            "category_counts": category_count_map,
            "all_target_rows_classified": bool(len(classified) == public_count + master_count),
            "all_rows_require_source_backed_resolution": bool(
                classified["requires_source_backed_resolution"].astype(bool).all()
            ),
            "no_unresolved_rows_auto_waived": bool(
                not classified.loc[
                    classified["action_category"].eq("truly_unresolved"), "auto_waived"
                ].astype(bool).any()
            ),
            "does_not_assert_point_in_time": True,
            "historical_pit_ready": False,
            "historical_backtest_eligible": False,
            "pit_warning": (
                "This QA uses current reconciliation and security-master evidence only; "
                "classification does not establish historical point-in-time membership."
            ),
            "source_dir": str(source.resolve()),
            "sources": {
                key: {
                    "path": str(path.resolve()),
                    "sha256": source_hashes[key],
                    "bytes": path.stat().st_size,
                    "manifest_checksum_present": manifest_verification[key][0],
                    "manifest_checksum_verified": manifest_verification[key][1],
                }
                for key, path in source_paths.items()
            },
            "source_checksums_verified_after_build": True,
            "artifacts": {
                key: str((final_dir / path.name).resolve())
                for key, path in stage_paths.items()
            }
            | {"checksums": str((final_dir / checksums_path.name).resolve())},
            "artifact_sha256": output_hashes,
            "generated_at_utc": _as_utc_iso(generated_at),
        }
        manifest_path = stage_dir / "manifest.json"
        _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if final_dir.exists():
            raise FileExistsError(f"immutable artifact directory already exists: {final_dir}")
        os.replace(stage_dir, final_dir)
        stage_dir = None

        return JapanG1ReconciliationQAArtifacts(
            run_id=resolved_run_id,
            artifact_dir=final_dir,
            classified_rows_path=final_dir / stage_paths["classified_rows"].name,
            category_counts_path=final_dir / stage_paths["category_counts"].name,
            agent_next_queue_path=final_dir / stage_paths["agent_next_queue"].name,
            source_checksums_path=final_dir / stage_paths["source_checksums"].name,
            checksums_path=final_dir / checksums_path.name,
            manifest_path=final_dir / manifest_path.name,
            classified_rows=classified.reset_index(drop=True),
            category_counts=category_counts.reset_index(drop=True),
            agent_next_queue=queue.reset_index(drop=True),
            manifest=manifest,
        )
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


run_japan_g1_reconciliation_qa = write_japan_g1_reconciliation_qa_artifacts


def _classify_row(
    *,
    row: dict[str, object],
    exact: pd.DataFrame,
    normalized: pd.DataFrame,
    normalized_code: str,
    stable_by_code: dict[str, pd.DataFrame],
    reference_date: date | None,
) -> tuple[str, str]:
    source_side = _text(row.get("source_side"))
    local_code = _normalize_code(row.get("local_code"))
    if not normalized.empty:
        return (
            "ticker_normalization",
            f"public code {local_code} has no exact stable row; candidate base code "
            f"{normalized_code} has {len(normalized)} stable row(s)",
        )

    evidence = exact
    if source_side == "security_master":
        ric_code = _normalize_code(_ric_local_code(row.get("ric")))
        if ric_code and local_code and ric_code != local_code and ric_code in stable_by_code:
            return (
                "ticker_normalization",
                f"security-master local code {local_code} differs from RIC code {ric_code}",
            )
        duplicate_ids = 0
        if local_code and local_code in stable_by_code:
            duplicate_ids = stable_by_code[local_code]["stable_instrument_id"].map(_text).nunique()
        jpx_quotes = _numeric_max(evidence, "jpx_quote_row_count")
        if duplicate_ids > 1 or jpx_quotes > 1:
            return (
                "duplicate_quote_listing",
                f"local code {local_code} has {duplicate_ids} stable instrument(s) and "
                f"maximum {jpx_quotes:g} direct JPX quote row(s)",
            )

        retire_dates = _dates(evidence, "retire_date") | _dates(
            evidence, "jpx_last_retire_date"
        )
        if retire_dates:
            if reference_date and any(value <= reference_date for value in retire_dates):
                return (
                    "stale_active_flag",
                    "active-master-only row has retirement evidence effective on or before "
                    f"{reference_date.isoformat()}",
                )
            return (
                "future_retired_status",
                "active-master-only row carries scheduled or otherwise unresolved retirement evidence",
            )

        if not evidence.empty and not _bool_any(evidence, "is_active"):
            return (
                "stale_active_flag",
                "reconciliation labels the row active but stable evidence has no active quote",
            )
        if re.search(r"[A-Z]", local_code):
            return (
                "alpha_codes",
                f"master-only JPX code {local_code} contains alphabetic code characters",
            )
        if _looks_out_of_scope(row, evidence):
            return (
                "out_of_scope_market",
                "name or identifier evidence indicates a fund, infrastructure, foreign, "
                "preferred, or other excluded listing type",
            )
        if not _text(row.get("ric")) or (
            not evidence.empty and not _bool_any(evidence, "is_primary_ric")
        ):
            return (
                "missing_primary_ric",
                "the exact stable instrument lacks a validated active primary RIC",
            )
        return (
            "truly_unresolved",
            "active primary master row is absent from the public denominator and available "
            "evidence does not establish market segment or retirement status",
        )

    if not exact.empty:
        retire_dates = _dates(exact, "retire_date") | _dates(
            exact, "jpx_last_retire_date"
        )
        if retire_dates or not _bool_any(exact, "is_active"):
            return (
                "future_retired_status",
                "current public code has only inactive or retirement-bearing stable evidence",
            )
        if not _bool_any(exact, "is_primary_ric"):
            return (
                "missing_primary_ric",
                "current public code has stable evidence but no validated primary RIC",
            )
        if _stable_id_count(exact) > 1 or _numeric_max(exact, "jpx_quote_row_count") > 1:
            return (
                "duplicate_quote_listing",
                "current public code has multiple stable instruments or direct JPX quote rows",
            )
    if _looks_out_of_scope(row, exact):
        return (
            "out_of_scope_market",
            "public row name indicates a preferred, fund, or other excluded listing type",
        )
    if re.search(r"[A-Z]", local_code):
        return (
            "alpha_codes",
            f"public unmatched JPX code {local_code} contains alphabetic code characters",
        )
    return (
        "truly_unresolved",
        "public in-scope row has no exact stable-instrument evidence or validated primary RIC",
    )


def _ticker_normalization_candidate(
    local_code: str, stable_by_code: dict[str, pd.DataFrame]
) -> str:
    if re.fullmatch(r"\d{5}", local_code):
        base = local_code[:4]
        if base in stable_by_code and local_code not in stable_by_code:
            return base
    return ""


def _looks_out_of_scope(row: dict[str, object], evidence: pd.DataFrame) -> bool:
    text_parts = [
        _text(row.get("jpx_name")),
        _text(row.get("jpx_market_category_raw")),
    ]
    for column in ("common_name", "document_title", "jpx_name"):
        if column in evidence.columns:
            text_parts.extend(evidence[column].map(_text).tolist())
    if _OUT_OF_SCOPE_PATTERN.search(" ".join(text_parts)):
        return True
    if "isin" in evidence.columns:
        isins = [value.upper() for value in evidence["isin"].map(_text) if value]
        if isins and all(not value.startswith("JP") for value in isins):
            return True
    return False


def _evidence_fields(evidence: pd.DataFrame) -> dict[str, object]:
    return {
        "evidence_stable_row_count": int(len(evidence)),
        "evidence_stable_source_rows": _joined(evidence, "_stable_source_row_number"),
        "evidence_candidate_stable_ids": _joined(evidence, "stable_instrument_id"),
        "evidence_candidate_rics": _joined(evidence, "ric"),
        "evidence_candidate_local_codes": _joined(evidence, "local_code"),
        "evidence_active_states": _joined(evidence, "is_active"),
        "evidence_primary_ric_states": _joined(evidence, "is_primary_ric"),
        "evidence_primary_issue_ric_states": _joined(evidence, "is_primary_issue_ric"),
        "evidence_first_trade_dates": _joined(evidence, "first_trade_date"),
        "evidence_retire_dates": _joined(evidence, "retire_date"),
        "evidence_jpx_last_retire_dates": _joined(evidence, "jpx_last_retire_date"),
        "evidence_quote_row_counts": _joined(evidence, "quote_row_count"),
        "evidence_jpx_quote_row_counts": _joined(evidence, "jpx_quote_row_count"),
        "evidence_names": _joined(evidence, "common_name"),
    }


def _category_counts(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in ACTION_CATEGORIES:
        selected = classified["action_category"].eq(category)
        rows.append(
            {
                "action_category": category,
                "resolvability_rank": RESOLVABILITY_RANK[category],
                "row_count": int(selected.sum()),
                "public_in_scope_unmatched_rows": int(
                    (selected & classified["source_side"].eq("public_jpx")).sum()
                ),
                "active_master_only_rows": int(
                    (selected & classified["source_side"].eq("security_master")).sum()
                ),
                "auto_waived_rows": int(
                    classified.loc[selected, "auto_waived"].astype(bool).sum()
                ),
                "recommended_action": _CATEGORY_ACTION[category],
            }
        )
    return pd.DataFrame(rows)


def _verify_unmatched_is_reconciliation_subset(
    reconciliation: pd.DataFrame, unmatched: pd.DataFrame
) -> None:
    columns = [column for column in unmatched.columns if column in reconciliation.columns]
    reconciliation_rows = Counter(_row_signature(row, columns) for row in reconciliation.to_dict("records"))
    unmatched_rows = Counter(_row_signature(row, columns) for row in unmatched.to_dict("records"))
    missing = unmatched_rows - reconciliation_rows
    if missing:
        raise ValueError("unmatched rows are not an evidence-preserving subset of reconciliation")


def _verify_source_manifest_hashes(
    manifest: dict[str, object], source_hashes: dict[str, str]
) -> dict[str, tuple[bool, bool]]:
    artifact_hashes = manifest.get("artifact_sha256", {})
    if not isinstance(artifact_hashes, dict):
        artifact_hashes = {}
    result: dict[str, tuple[bool, bool]] = {}
    for key, actual in source_hashes.items():
        expected = artifact_hashes.get(key)
        present = isinstance(expected, str) and bool(expected)
        verified = bool(present and expected == actual)
        if present and not verified:
            raise RuntimeError(f"source manifest checksum mismatch for {key}")
        result[key] = (present, verified)
    return result


def _source_as_of_date(
    reconciliation: pd.DataFrame, manifest: dict[str, object]
) -> date | None:
    if "jpx_asof_date" in reconciliation.columns:
        parsed = pd.to_datetime(reconciliation["jpx_asof_date"], errors="coerce")
        if parsed.notna().any():
            return parsed.max().date()
    generated_at = manifest.get("generated_at_utc")
    return _as_date(generated_at)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=object, keep_default_na=False, low_memory=False)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required Japan G1 reconciliation source is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"required Japan G1 reconciliation source is empty: {path}")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"another writer owns the Japan G1 QA lock: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _write_text(path, frame.to_csv(index=False))


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qa_row_id(row: dict[str, object]) -> str:
    keys = (
        "source_side",
        "jpx_source_row",
        "local_code",
        "stable_instrument_id",
        "ric",
        "reconciliation_status",
    )
    payload = json.dumps([_text(row.get(key)) for key in keys], separators=(",", ":"))
    return f"jpn-g1-qa:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _row_signature(row: dict[str, object], columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(_text(row.get(column)) for column in columns)


def _records_json(records: list[dict[str, object]]) -> str:
    normalized = [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in records
    ]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> object:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _joined(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    values = [_text(value) for value in frame[column] if _text(value)]
    return "|".join(dict.fromkeys(values))


def _stable_id_count(frame: pd.DataFrame) -> int:
    if frame.empty or "stable_instrument_id" not in frame.columns:
        return 0
    return int(frame["stable_instrument_id"].map(_text).replace("", pd.NA).nunique())


def _numeric_max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.max()) if values.notna().any() else 0.0


def _bool_any(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    return bool(frame[column].map(_as_bool).any())


def _dates(frame: pd.DataFrame, column: str) -> set[date]:
    if frame.empty or column not in frame.columns:
        return set()
    return {value for value in frame[column].map(_as_date) if value is not None}


def _as_bool(value: object) -> bool:
    return _text(value).casefold() in {"true", "1", "yes", "y"}


def _as_date(value: object) -> date | None:
    if value is None or _text(value) == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed.date() if pd.notna(parsed) else None


def _normalize_code(value: object) -> str:
    text = _text(value).upper()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _ric_local_code(value: object) -> str:
    return re.split(r"[.\^]", _text(value).upper(), maxsplit=1)[0]


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _as_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()
