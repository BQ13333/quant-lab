from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pypdf import PdfReader


UTC = timezone.utc
VERSION = "1.4-official-monthly-pdf"
_COMMON_SHARE = re.compile(
    r"(?P<code>[1-9][0-9A-Z]{3})[ \t\u3000]*(?P<name>[^\n]{1,100}?)[ \t\u3000]*普通株(?:式)?"
)
_COMBINED_COMMON_SHARE = re.compile(
    r"(?m)^(?:19|20)\d{2}/\d{2}\s+(?P<code>[1-9][0-9A-Z]{3})\s*(?P<name>[^\n]{1,100}?)\s*普通株(?:式)?"
)
_UNIT = (
    r"(?:10,?000|5,?000|3,?000|2,?000|1,?000|"
    r"500|200|1\s*0\s*0|5\s*0|1\s*0|5|1)"
)


@dataclass(frozen=True)
class JapanMonthlyMicrostructureArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_monthly_microstructure(
    *,
    run_id: str,
    source_dirs: Iterable[str | Path],
    combined_membership_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_monthly_microstructure",
    max_files: int = 100,
    seed_run_dir: str | Path | None = None,
) -> JapanMonthlyMicrostructureArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if max_files <= 0:
        raise ValueError("max_files must be positive")
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        paths = _paths(root)
        plan = _load_or_build_plan(paths["plan"], paths["plan_hash"], source_dirs)
        if seed_run_dir is not None and not paths["state"].is_file():
            seeded = _seed_state(plan, Path(seed_run_dir), root / "file_outputs")
            _atomic_write_csv(paths["state"], seeded)
        state = _read_csv(paths["state"], _state_columns())
        complete_ids = set(state.loc[state["status"].eq("completed"), "plan_id"])
        pending = plan.loc[~plan["plan_id"].isin(complete_ids)].head(max_files)
        state_rows = state.to_dict("records")
        for item in pending.to_dict("records"):
            state_rows = [row for row in state_rows if row.get("plan_id") != item["plan_id"]]
            state_rows.append(_process_file(item, root / "file_outputs"))
            _atomic_write_csv(paths["state"], pd.DataFrame(state_rows, columns=_state_columns()))
        state = _read_csv(paths["state"], _state_columns())
        manifest = _publish_aggregate(
            run_id=run_id,
            root=root,
            paths=paths,
            plan=plan,
            state=state,
            combined_membership_dir=Path(combined_membership_dir),
        )
        _atomic_write_json(paths["manifest"], manifest)
        return JapanMonthlyMicrostructureArtifacts(run_id, root, paths["manifest"], manifest)
    finally:
        lock.rmdir()


def parse_tse_lot_size_text(text: str) -> pd.DataFrame:
    combined = bool(re.search(r"(?m)^(?:19|20)\d{2}/\d{2}\s+[1-9][0-9A-Z]{3}\s", text))
    pattern = _COMBINED_COMMON_SHARE if combined else _COMMON_SHARE
    rows = []
    for code, block in _record_blocks(text, pattern):
        if combined:
            match = re.search(rf"(?P<unit>{_UNIT})\s+(?:\d|[－-])", block[:1200])
            route = "tse_combined_row"
        else:
            match = re.search(
                rf"(?m)^\s*(?:(?:1|2)\s*)?(?:[^\d\n]{{0,24}})?"
                rf"(?:貸借|信用|整|監|審|確)?\s*(?P<unit>{_UNIT})\s*(?:\d|[－-])",
                block[:1200],
            )
            route = "tse_legacy_block"
        if match:
            rows.append({"code": code, "trading_unit": _unit(match.group("unit")), "parser_route": route})
    return _parsed_frame(rows)


def parse_ose_lot_size_text(text: str) -> pd.DataFrame:
    rows = []
    if "Closing Price and Trading Volume of Issues" in text:
        for match in re.finditer(
            rf"(?m)^\s*(?P<unit>{_UNIT})\s+\d{{1,2}}(?:/\d{{1,2}})?\s+"
            rf"(?P<code>[1-9][0-9A-Z]{{3}})\s+",
            text,
        ):
            rows.append(
                {
                    "code": match.group("code").upper(),
                    "trading_unit": _unit(match.group("unit")),
                    "parser_route": "jasdaq_legacy_row",
                }
            )
    for code, block in _record_blocks(text, _COMMON_SHARE):
        match = re.search(
            rf"(?m)^\s*[*#]?\s*(?P<unit>{_UNIT})\s*(?:[□◇]\s*)?(?:\d{{2}}|[-－])",
            block[:1200],
        )
        if match:
            rows.append(
                {
                    "code": code,
                    "trading_unit": _unit(match.group("unit")),
                    "parser_route": "ose_block",
                }
            )
    return _parsed_frame(rows)


def _record_blocks(text: str, pattern: re.Pattern[str]):
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        yield match.group("code").upper(), text[match.end() : end]


def _parsed_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["code", "trading_unit", "parser_route"])
    return frame.drop_duplicates().reset_index(drop=True)


def _unit(value: str) -> int:
    return int(re.sub(r"[\s,]", "", str(value)))


def _load_or_build_plan(plan_path: Path, hash_path: Path, source_dirs: Iterable[str | Path]) -> pd.DataFrame:
    if plan_path.is_file():
        expected = hash_path.read_text(encoding="utf-8").strip() if hash_path.is_file() else ""
        if len(expected) != 64 or _sha256(plan_path) != expected:
            raise ValueError("monthly microstructure plan hash mismatch")
        plan = pd.read_csv(plan_path, dtype=str).fillna("")
        if "parser_version" not in plan or not plan["parser_version"].eq(VERSION).all():
            raise ValueError("monthly microstructure plan parser version mismatch")
        return plan
    rows = []
    for directory_value in source_dirs:
        directory = Path(directory_value)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"source manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"complete", "complete_candidate_source"}:
            raise ValueError(f"source is not complete: {directory.name}")
        status = pd.read_csv(directory / "file_status.csv", dtype=str).fillna("")
        if not status["status"].eq("completed").all():
            raise ValueError(f"source has incomplete files: {directory.name}")
        parser_kind = "tse" if "jpx_monthly_quotations" in directory.parts else "ose"
        source_family = (
            "TSE" if parser_kind == "tse" else "OSE_MAIN" if "ose_main_monthly_quotations" in directory.parts else "JASDAQ_OSE"
        )
        for item in status.to_dict("records"):
            raw_path = Path(item["raw_path"])
            normalized_path = Path(item["normalized_path"])
            if _sha256(raw_path) != item["sha256"] or _sha256(normalized_path) != item["normalized_sha256"]:
                raise ValueError(f"source hash mismatch: {item['request_id']}")
            yyyymm = re.search(r"((?:19|20)\d{4})", str(item["request_id"]))
            if not yyyymm:
                raise ValueError(f"cannot derive observation month: {item['request_id']}")
            rows.append(
                {
                    "plan_id": f"{directory.name}:{item['request_id']}",
                    "source_artifact": directory.name,
                    "source_manifest": str(manifest_path.resolve()),
                    "source_manifest_sha256": _sha256(manifest_path),
                    "source_family": source_family,
                    "parser_kind": parser_kind,
                    "parser_version": VERSION,
                    "request_id": item["request_id"],
                    "observation_month": f"{yyyymm.group(1)[:4]}-{yyyymm.group(1)[4:]}",
                    "raw_path": str(raw_path.resolve()),
                    "raw_sha256": item["sha256"],
                    "normalized_path": str(normalized_path.resolve()),
                    "normalized_sha256": item["normalized_sha256"],
                }
            )
    plan = pd.DataFrame(rows).sort_values(["observation_month", "source_family", "request_id"])
    if plan["plan_id"].duplicated().any():
        raise ValueError("monthly microstructure plan IDs are not unique")
    _atomic_write_csv(plan_path, plan)
    _atomic_write_bytes(hash_path, (_sha256(plan_path) + "\n").encode())
    return plan


def _process_file(item: dict[str, str], output_root: Path) -> dict[str, Any]:
    observed = datetime.now(UTC).isoformat()
    raw_path = Path(item["raw_path"])
    normalized_path = Path(item["normalized_path"])
    if _sha256(raw_path) != item["raw_sha256"] or _sha256(normalized_path) != item["normalized_sha256"]:
        return _state_row(item, "failed", observed, "input_hash_mismatch")
    try:
        expected = pd.read_parquet(normalized_path)[["code"]].astype(str).drop_duplicates()
        logging.getLogger("pypdf").setLevel(logging.CRITICAL)
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw_path.read_bytes())).pages)
        parsed = parse_tse_lot_size_text(text) if item["parser_kind"] == "tse" else parse_ose_lot_size_text(text)
        conflicts = parsed.groupby("code")["trading_unit"].nunique().gt(1)
        conflict_codes = set(conflicts[conflicts].index)
        parsed = parsed.loc[~parsed["code"].isin(conflict_codes)].drop_duplicates("code")
        output = expected.merge(parsed, on="code", how="left", validate="one_to_one")
        output.insert(0, "observation_month", item["observation_month"])
        output["source_family"] = item["source_family"]
        output["source_artifact"] = item["source_artifact"]
        output["request_id"] = item["request_id"]
        output["raw_sha256"] = item["raw_sha256"]
        output["unit_observed"] = output["trading_unit"].notna()
        output_path = output_root / f"{hashlib.sha256(item['plan_id'].encode()).hexdigest()}.parquet"
        _atomic_write_parquet(output_path, output)
        return {
            **_state_row(item, "completed", observed, ""),
            "expected_rows": len(expected),
            "parsed_rows": int(output["unit_observed"].sum()),
            "coverage": float(output["unit_observed"].mean()) if len(output) else 1.0,
            "conflict_codes": len(conflict_codes),
            "output_path": str(output_path.resolve()),
            "output_sha256": _sha256(output_path),
            "lineage": "parsed_current_version",
        }
    except Exception as exc:
        return _state_row(item, "failed", observed, f"{type(exc).__name__}: {exc}")


def _state_row(item: dict[str, str], status: str, observed: str, error: str) -> dict[str, Any]:
    return {
        "plan_id": item["plan_id"],
        "status": status,
        "observed_at_utc": observed,
        "expected_rows": 0,
        "parsed_rows": 0,
        "coverage": 0.0,
        "conflict_codes": 0,
        "output_path": "",
        "output_sha256": "",
        "error": error,
        "parser_version": VERSION,
        "lineage": "",
    }


def _seed_state(plan: pd.DataFrame, seed_root: Path, output_root: Path) -> pd.DataFrame:
    seed_plan_path = seed_root / "file_plan.csv"
    seed_state_path = seed_root / "file_state.csv"
    if not seed_plan_path.is_file() or not seed_state_path.is_file():
        raise FileNotFoundError(f"seed plan/state missing: {seed_root}")
    seed_plan = pd.read_csv(seed_plan_path, dtype=str).fillna("").set_index("plan_id")
    seed_state = pd.read_csv(seed_state_path, dtype=str).fillna("")
    current = plan.set_index("plan_id")
    rows = []
    output_root.mkdir(parents=True, exist_ok=True)
    for row in seed_state.to_dict("records"):
        plan_id = row["plan_id"]
        if (
            row.get("status") != "completed"
            or float(row.get("coverage") or 0) != 1.0
            or int(float(row.get("conflict_codes") or 0)) != 0
            or plan_id not in seed_plan.index
            or plan_id not in current.index
        ):
            continue
        old_plan = seed_plan.loc[plan_id]
        new_plan = current.loc[plan_id]
        if any(old_plan[field] != new_plan[field] for field in ("raw_sha256", "normalized_sha256")):
            continue
        source = Path(row["output_path"])
        if not source.is_file() or _sha256(source) != row["output_sha256"]:
            continue
        target = output_root / source.name
        shutil.copy2(source, target)
        rows.append(
            {
                **row,
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "output_path": str(target.resolve()),
                "output_sha256": _sha256(target),
                "parser_version": VERSION,
                "lineage": f"hash_verified_seed:{seed_root.name}",
            }
        )
    return pd.DataFrame(rows, columns=_state_columns())


def _publish_aggregate(
    *,
    run_id: str,
    root: Path,
    paths: dict[str, Path],
    plan: pd.DataFrame,
    state: pd.DataFrame,
    combined_membership_dir: Path,
) -> dict[str, Any]:
    completed = state.loc[state["status"].eq("completed")]
    frames = []
    for row in completed.to_dict("records"):
        output_path = Path(row["output_path"])
        if not output_path.is_file() or _sha256(output_path) != row["output_sha256"]:
            raise ValueError(f"microstructure output hash mismatch: {row['plan_id']}")
        frames.append(pd.read_parquet(output_path))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["observation_month", "code", "trading_unit", "source_family", "unit_observed"]
    )
    _atomic_write_parquet(paths["raw_observations"], raw)
    valid = raw.loc[raw["unit_observed"].astype(bool) & raw["trading_unit"].notna()].copy()
    by_source = valid.groupby(["observation_month", "code", "source_family"], as_index=False).agg(
        trading_unit=("trading_unit", "first"),
        unit_variants=("trading_unit", "nunique"),
        source_observations=("source_family", "size"),
    )
    conflicts = by_source.loc[by_source["unit_variants"].gt(1)].copy()
    direct = by_source.loc[by_source["unit_variants"].eq(1)].copy().rename(
        columns={"source_family": "preferred_source_family"}
    )
    venue_variants = valid.groupby(["observation_month", "code"], as_index=False).agg(
        unit_variants=("trading_unit", "nunique"),
        source_families=("source_family", lambda values: "|".join(sorted(set(map(str, values))))),
        observed_units=("trading_unit", lambda values: "|".join(map(str, sorted(set(map(int, values))))))
    )
    venue_variants = venue_variants.loc[venue_variants["unit_variants"].gt(1)].copy()
    membership_manifest = json.loads((combined_membership_dir / "manifest.json").read_text(encoding="utf-8"))
    membership_path = Path(membership_manifest["artifacts"]["monthly_membership"])
    expected_membership_hash = membership_manifest.get("output_sha256", {}).get("monthly_membership.parquet")
    if expected_membership_hash != _sha256(membership_path):
        raise ValueError("combined membership hash mismatch")
    membership = pd.read_parquet(membership_path)[
        ["observation_month", "code", "known_at_utc", "source_families"]
    ].copy()
    membership["preferred_source_family"] = membership["source_families"].map(_preferred_unit_source)
    history = membership.merge(
        direct[["observation_month", "code", "preferred_source_family", "trading_unit", "source_observations"]],
        on=["observation_month", "code", "preferred_source_family"],
        how="left",
        validate="one_to_one",
    )
    conflict_keys = set(zip(conflicts["observation_month"], conflicts["code"], conflicts["source_family"]))
    history["unit_mapping_status"] = "direct_official_monthly_pdf"
    history.loc[history["trading_unit"].isna(), "unit_mapping_status"] = "unresolved_parser_gap"
    if conflict_keys:
        mask = pd.Series(
            list(zip(history["observation_month"], history["code"], history["preferred_source_family"])),
            index=history.index,
        ).isin(conflict_keys)
        history.loc[mask, "unit_mapping_status"] = "conflicting_official_sources"
    history["pit_usable"] = history["trading_unit"].notna() & history["known_at_utc"].notna()
    unresolved = history.loc[~history["pit_usable"]].copy()
    _atomic_write_parquet(paths["history"], history)
    _atomic_write_csv(paths["unresolved"], unresolved)
    _atomic_write_csv(paths["conflicts"], conflicts)
    _atomic_write_csv(paths["venue_variants"], venue_variants)
    _atomic_write_csv(paths["file_quality"], state)
    planned = len(plan)
    completed_files = int(state["status"].eq("completed").sum())
    failed_files = int(state["status"].eq("failed").sum())
    source_missing_rows = int(
        (pd.to_numeric(state["expected_rows"], errors="coerce").fillna(0)
         - pd.to_numeric(state["parsed_rows"], errors="coerce").fillna(0)).sum()
    )
    parser_conflict_codes = int(pd.to_numeric(state["conflict_codes"], errors="coerce").fillna(0).sum())
    coverage = float(history["pit_usable"].mean()) if len(history) else 0.0
    complete = (
        planned > 0 and completed_files == planned and failed_files == 0
        and source_missing_rows == 0 and parser_conflict_codes == 0
        and not len(unresolved) and not len(conflicts)
    )
    output_paths = [
        paths["raw_observations"], paths["history"], paths["unresolved"],
        paths["conflicts"], paths["venue_variants"], paths["file_quality"]
    ]
    return {
        "schema_version": VERSION,
        "run_id": run_id,
        "artifact_type": "japan_official_monthly_microstructure_producer",
        "status": "complete_candidate" if complete else "in_progress" if completed_files < planned else "complete_with_blockers",
        "scorecard_eligible": False,
        "independent_validation_required": True,
        "official_jpx_source": True,
        "historical_lot_size_candidate": complete,
        "counts": {
            "planned_files": planned,
            "completed_files": completed_files,
            "failed_files": failed_files,
            "raw_source_rows": len(raw),
            "membership_rows": len(history),
            "mapped_membership_rows": int(history["pit_usable"].sum()),
            "unresolved_membership_rows": len(unresolved),
            "conflict_rows": len(conflicts),
            "venue_specific_variant_rows": len(venue_variants),
            "source_parser_missing_rows": source_missing_rows,
            "source_parser_conflict_codes": parser_conflict_codes,
        },
        "coverage": coverage,
        "critical_blockers": [] if complete else [
            f"pending_or_failed_files={planned - completed_files + failed_files}",
            f"unresolved_membership_rows={len(unresolved)}",
            f"conflicting_code_month_rows={len(conflicts)}",
            f"source_parser_missing_rows={source_missing_rows}",
            f"source_parser_conflict_codes={parser_conflict_codes}",
        ],
        "input_plan_sha256": _sha256(paths["plan"]),
        "combined_membership_manifest": str((combined_membership_dir / "manifest.json").resolve()),
        "combined_membership_sha256": _sha256(membership_path),
        "artifacts": {path.stem: str(path.resolve()) for path in output_paths},
        "output_sha256": {path.name: _sha256(path) for path in output_paths},
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "plan": root / "file_plan.csv",
        "plan_hash": root / "file_plan.sha256",
        "state": root / "file_state.csv",
        "file_quality": root / "file_quality.csv",
        "raw_observations": root / "raw_lot_size_observations.parquet",
        "history": root / "monthly_lot_size_history.parquet",
        "unresolved": root / "unresolved_lot_sizes.csv",
        "conflicts": root / "lot_size_conflicts.csv",
        "venue_variants": root / "venue_specific_lot_size_variants.csv",
        "manifest": root / "manifest.json",
    }


def _state_columns() -> list[str]:
    return [
        "plan_id", "status", "observed_at_utc", "expected_rows", "parsed_rows",
        "coverage", "conflict_codes", "output_path", "output_sha256", "error",
        "parser_version", "lineage",
    ]


def _preferred_unit_source(source_families: str) -> str:
    families = set(str(source_families).split("|"))
    if "TSE" in families:
        return "TSE"
    if families.intersection({"JASDAQ", "OSE_COMBINED"}):
        return "JASDAQ_OSE"
    if "OSE_MAIN" in families:
        return "OSE_MAIN"
    return ""


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame[columns]


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
