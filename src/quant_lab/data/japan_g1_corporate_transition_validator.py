from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


UTC = timezone.utc
PRIMARY_EXCHANGES = {"TYO", "JSD", "OSA"}
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}


@dataclass(frozen=True)
class JapanG1CorporateTransitionValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    security_master_path: Path
    ledger_path: Path
    manifest: dict[str, Any]


def write_japan_g1_corporate_transition_validation(
    *,
    run_id: str,
    transition_evidence_dir: str | Path,
    direct_validation_dir: str | Path,
    quote_chain_shard_dirs: Iterable[str | Path],
    successor_vendor_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g1_corporate_transition_validation",
    expected_transition_rows: int = 66,
) -> JapanG1CorporateTransitionValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    evidence = Path(transition_evidence_dir).resolve()
    direct = Path(direct_validation_dir).resolve()
    successor_vendor = Path(successor_vendor_dir).resolve()
    evidence_manifest_path = evidence / "manifest.json"
    evidence_manifest = _read_json(evidence_manifest_path)
    candidates_path = evidence / "admitted_transition_candidates.csv"
    reconciliation_path = evidence / "transition_reconciliation.csv"
    page_state_path = evidence / "page_state.csv"
    _require_output_hashes(
        evidence,
        evidence_manifest,
        (candidates_path.name, reconciliation_path.name, page_state_path.name),
    )
    counts = evidence_manifest.get("counts", {})
    if not (
        evidence_manifest.get("artifact_type")
        == "japan_g1_official_corporate_transition_evidence_candidate"
        and evidence_manifest.get("status") == "complete_candidate"
        and evidence_manifest.get("denominator_valid") is True
        and evidence_manifest.get("scorecard_eligible") is False
        and evidence_manifest.get("independent_validation_required") is True
        and int(counts.get("transition_denominator", 0)) == expected_transition_rows
        and int(counts.get("event_segments_admitted_candidate", 0))
        == expected_transition_rows
        and int(counts.get("unresolved_candidates", -1)) == 0
    ):
        raise ValueError("transition evidence producer contract invalid")

    candidates = pd.read_csv(candidates_path, dtype=str).fillna("")
    reconciliation = pd.read_csv(reconciliation_path, dtype=str).fillna("")
    if len(candidates) != expected_transition_rows or not candidates.equals(reconciliation):
        raise ValueError("candidate/reconciliation denominator or row equality failed")
    page_state = pd.read_csv(page_state_path, dtype=str).fillna("")
    source_text = _validate_and_read_official_pages(page_state)
    delistings = _load_verified_delistings(evidence_manifest)
    ledger = independently_validate_transition_rows(
        candidates,
        delistings=delistings,
        source_text_by_url=source_text,
    )

    direct_manifest_path = direct / "manifest.json"
    direct_manifest = _read_json(direct_manifest_path)
    direct_path = direct / "security_master.csv"
    if not (
        direct_manifest.get("artifact_type")
        == "japan_g1_quote_chain_independent_validation"
        and direct_manifest.get("status") == "pass_partial_component"
        and direct_manifest.get("independent_validation") is True
        and direct_manifest.get("output_sha256", {}).get("security_master")
        == _sha256(direct_path)
    ):
        raise ValueError("direct quote-chain validation contract invalid")
    direct_rows = pd.read_csv(direct_path, dtype=str).fillna("")

    vendor_frames: list[pd.DataFrame] = []
    lineage_paths: dict[str, str] = {
        "transition_evidence": str(evidence_manifest_path),
        "direct_quote_chain_validation": str(direct_manifest_path),
    }
    lineage_hashes: dict[str, str] = {
        key: _sha256(Path(value)) for key, value in lineage_paths.items()
    }
    for index, raw in enumerate([*quote_chain_shard_dirs, successor_vendor]):
        directory = Path(raw).resolve()
        frame, manifest_path = _verified_vendor_shard(directory)
        vendor_frames.append(frame)
        key = f"vendor_shard_{index:02d}"
        lineage_paths[key] = str(manifest_path)
        lineage_hashes[key] = _sha256(manifest_path)
    vendor_rows = pd.concat(vendor_frames, ignore_index=True, sort=False).drop_duplicates(
        ["ric", "ticker_symbol", "instrument_permid"], keep="last"
    )

    aliases: list[dict[str, Any]] = []
    successor_boundaries: list[dict[str, Any]] = []
    mapping_issues: list[str] = []
    for row in ledger.to_dict("records"):
        chosen = _select_successor_vendor_row(vendor_rows, row)
        if chosen is None:
            mapping_issues.append(f"successor_vendor_route_missing:{row['old_code']}->{row['successor_code']}")
            continue
        alias_start = _alias_start_after_direct_route(direct_rows, str(row["old_code"]))
        record = chosen.to_dict()
        record.update(
            {
                "mapping_local_code_override": row["old_code"],
                "first_trade_date": alias_start,
                "retire_date": row["old_event_date"],
                "quote_chain_validation_run_id": run_id,
                "quote_chain_identity_row_id": row["identity_row_id"],
                "quote_chain_mapping_route": "independently_validated_official_transition_alias",
                "quote_chain_admitted": True,
                "transition_validation_run_id": run_id,
                "transition_evidence_class": row["evidence_class"],
                "transition_old_code": row["old_code"],
                "transition_successor_code": row["successor_code"],
                "transition_old_event_date": row["old_event_date"],
                "transition_successor_listing_date": row["official_listing_date"],
                "transition_admitted": True,
            }
        )
        aliases.append(record)
        boundary = chosen.to_dict()
        boundary.update(
            {
                "mapping_local_code_override": row["successor_code"],
                "first_trade_date": row["official_listing_date"],
                "retire_date": row.get("successor_later_delisting_date", "")
                or chosen.get("retire_date", ""),
                "quote_chain_validation_run_id": run_id,
                "quote_chain_identity_row_id": row["identity_row_id"],
                "quote_chain_mapping_route": "independently_validated_official_successor_boundary",
                "quote_chain_admitted": True,
                "transition_validation_run_id": run_id,
                "transition_evidence_class": row["evidence_class"],
                "transition_old_code": row["old_code"],
                "transition_successor_code": row["successor_code"],
                "transition_old_event_date": row["old_event_date"],
                "transition_successor_listing_date": row["official_listing_date"],
                "transition_admitted": True,
            }
        )
        successor_boundaries.append(boundary)

    blockers = list(ledger.loc[~ledger["validation_pass"], "validation_issue"])
    blockers.extend(mapping_issues)
    if len(ledger) != expected_transition_rows:
        blockers.append(f"validated_transition_denominator={len(ledger)}/{expected_transition_rows}")
    if len(aliases) != expected_transition_rows:
        blockers.append(f"mapped_transition_denominator={len(aliases)}/{expected_transition_rows}")
    if len(successor_boundaries) != expected_transition_rows:
        blockers.append(
            f"successor_boundary_denominator={len(successor_boundaries)}/{expected_transition_rows}"
        )
    blockers = list(dict.fromkeys(value for value in blockers if value))

    aliases_frame = pd.DataFrame(aliases)
    boundaries_frame = pd.DataFrame(successor_boundaries)
    security_master = pd.concat(
        [direct_rows, aliases_frame, boundaries_frame], ignore_index=True, sort=False
    )
    if not security_master.empty:
        override = security_master.get(
            "mapping_local_code_override", pd.Series("", index=security_master.index)
        ).fillna("").astype(str)
        security_master["_dedupe_code"] = override
        security_master.loc[security_master["_dedupe_code"].eq(""), "_dedupe_code"] = security_master.loc[
            security_master["_dedupe_code"].eq(""), "ticker_symbol"
        ].astype(str)
        security_master = security_master.drop_duplicates(
            ["ric", "_dedupe_code"], keep="last"
        ).drop(columns="_dedupe_code")

    root = (Path(artifact_root) / run_id).resolve()
    if root in {evidence, direct, successor_vendor}:
        raise ValueError("validation output must differ from input evidence")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        ledger_path = root / "validated_transition_ledger.csv"
        alias_path = root / "validated_transition_aliases.csv"
        boundary_path = root / "validated_successor_boundaries.csv"
        security_path = root / "security_master.csv"
        _atomic_csv(ledger_path, ledger)
        _atomic_csv(alias_path, aliases_frame)
        _atomic_csv(boundary_path, boundaries_frame)
        _atomic_csv(security_path, security_master)
        passed = not blockers
        manifest = {
            "schema_version": "1.0-independent-official-transition",
            "run_id": run_id,
            "artifact_type": "japan_g1_corporate_transition_independent_validation",
            "status": "pass_component" if passed else "failed_validation",
            "validation_status": "pass" if passed else "fail",
            "independent_validation": True,
            "clean_room_evidence_checks": True,
            "component": "security_master_corporate_transitions",
            "component_scorecard_eligible": passed,
            "scorecard_eligible": False,
            "counts": {
                "transition_denominator": expected_transition_rows,
                "validated_transition_rows": int(ledger["validation_pass"].sum()),
                "mapped_transition_rows": len(aliases_frame),
                "validated_successor_boundary_rows": len(boundaries_frame),
                "direct_security_master_rows": len(direct_rows),
                "combined_security_master_rows": len(security_master),
                "official_source_pages": len(page_state),
                "verified_vendor_rows": len(vendor_rows),
            },
            "critical_blockers": blockers,
            "input_manifests": lineage_paths,
            "input_manifest_sha256": lineage_hashes,
            "artifacts": {
                "validated_transition_ledger": str(ledger_path),
                "validated_transition_aliases": str(alias_path),
                "validated_successor_boundaries": str(boundary_path),
                "security_master": str(security_path),
            },
            "output_sha256": {
                ledger_path.name: _sha256(ledger_path),
                alias_path.name: _sha256(alias_path),
                boundary_path.name: _sha256(boundary_path),
                "security_master": _sha256(security_path),
            },
            "promotion_policy": PROMOTION_POLICY,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanG1CorporateTransitionValidationArtifacts(
            run_id, root, manifest_path, security_path, ledger_path, manifest
        )
    finally:
        lock.rmdir()


def independently_validate_transition_rows(
    frame: pd.DataFrame,
    *,
    delistings: pd.DataFrame,
    source_text_by_url: dict[str, str],
) -> pd.DataFrame:
    required = {
        "evidence_class",
        "identity_row_id",
        "old_code",
        "old_issue_name",
        "old_event_date",
        "old_event_reason",
        "old_event_source_url",
        "successor_code",
        "successor_issue_name",
        "successor_primary_rics",
        "official_listing_date",
        "listing_source_url",
        "gap_days",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"transition schema invalid: {sorted(required - set(frame.columns))}")
    if frame["old_code"].duplicated().any() or frame["successor_code"].duplicated().any():
        raise ValueError("transition codes are not one-to-one")
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        issues: list[str] = []
        old_code = str(row["old_code"])
        successor_code = str(row["successor_code"])
        if not re.fullmatch(r"[0-9A-Z]{4}", old_code) or not re.fullmatch(
            r"[0-9A-Z]{4}", successor_code
        ):
            issues.append("malformed_transition_code")
        try:
            old_date = date.fromisoformat(str(row["old_event_date"]))
            listing_date = date.fromisoformat(str(row["official_listing_date"]))
            gap = (listing_date - old_date).days
        except ValueError:
            gap = -1
        if gap < 2 or gap > 6 or str(row["gap_days"]) != str(gap):
            issues.append("transition_gap_not_exactly_2_to_6_days")

        evidence_class = str(row["evidence_class"])
        if evidence_class == "tpm_official":
            if old_code != "2336" or successor_code != "9243":
                issues.append("tpm_identity_mismatch")
            if not _source_proves_row(source_text_by_url, row, require_old=True):
                issues.append("tpm_raw_sources_do_not_prove_transition")
        elif old_code == "3398":
            if successor_code != "3549" or not _source_proves_row(
                source_text_by_url, row, require_old=True
            ):
                issues.append("supplemental_raw_source_does_not_prove_transition")
        else:
            matches = delistings.loc[
                delistings["code"].astype(str).eq(old_code)
                & delistings["event_date"].astype(str).eq(str(row["old_event_date"]))
                & delistings["issue_name"].astype(str).eq(str(row["old_issue_name"]))
                & delistings["reason"].astype(str).eq(str(row["old_event_reason"]))
                & delistings["source_url"].astype(str).eq(str(row["old_event_source_url"]))
            ]
            if len(matches) != 1:
                issues.append("official_delisting_event_not_unique_exact")
            target = _reason_target(str(row["old_event_reason"]))
            if old_code == "6187":
                bridge_text = source_text_by_url.get(
                    str(row.get("name_bridge_source_url", "")), ""
                )
                bridge_valid = bool(
                    bridge_text
                    and old_code in bridge_text
                    and successor_code in bridge_text
                    and _canonical_name(target) in _canonical_name(bridge_text)
                    and _canonical_name(str(row["successor_issue_name"]))
                    in _canonical_name(bridge_text)
                    and _text_mentions_iso_date(
                        bridge_text, str(row["official_listing_date"])
                    )
                )
                if not bridge_valid:
                    issues.append("successor_name_change_not_exactly_supported_by_bridge_source")
            elif not target or _canonical_name(target) != _canonical_name(
                str(row["successor_issue_name"])
            ):
                issues.append("successor_name_not_exactly_supported_by_delisting_reason")
            if not _source_proves_row(source_text_by_url, row, require_old=False):
                issues.append("official_listing_raw_source_does_not_prove_successor")
        if evidence_class == "identity_anchored":
            candidate_rics = [
                value for value in str(row["successor_primary_rics"]).split("|") if value
            ]
            if not candidate_rics or not all(
                _ric_code(value) == successor_code for value in candidate_rics
            ):
                issues.append("identity_successor_ric_code_mismatch")
        elif str(row["successor_primary_rics"]).strip():
            issues.append("unanchored_transition_contains_unvalidated_successor_ric")
        output = dict(row)
        output.update(
            {
                "independently_recomputed_gap_days": gap,
                "validation_pass": not issues,
                "validation_issue": "|".join(issues),
            }
        )
        rows.append(output)
    return pd.DataFrame(rows).sort_values(["evidence_class", "old_code"]).reset_index(drop=True)


def _source_proves_row(
    source_text_by_url: dict[str, str], row: dict[str, Any], *, require_old: bool
) -> bool:
    listing_url = str(row["listing_source_url"])
    listing_text = source_text_by_url.get(listing_url, "")
    if not listing_text:
        return False
    successor_name = _canonical_name(str(row["successor_issue_name"]))
    listing_canonical = _canonical_name(listing_text)
    passed = (
        str(row["successor_code"]) in listing_text
        and bool(successor_name)
        and successor_name in listing_canonical
        and _text_mentions_iso_date(listing_text, str(row["official_listing_date"]))
    )
    if not require_old:
        return passed
    old_text = source_text_by_url.get(str(row["old_event_source_url"]), listing_text)
    old_canonical = _canonical_name(old_text)
    return bool(
        passed
        and str(row["old_code"]) in old_text
        and _canonical_name(str(row["old_issue_name"])) in old_canonical
        and _text_mentions_iso_date(old_text, str(row["old_event_date"]))
    )


def _validate_and_read_official_pages(page_state: pd.DataFrame) -> dict[str, str]:
    required = {"url", "terminal_status", "raw_path", "sha256", "content_type"}
    if page_state.empty or not required.issubset(page_state.columns):
        raise ValueError("official page-state schema invalid")
    if page_state["url"].duplicated().any():
        raise ValueError("official page-state contains duplicate URLs")
    output: dict[str, str] = {}
    for row in page_state.to_dict("records"):
        path = Path(str(row["raw_path"]))
        if not (
            row["terminal_status"] == "success"
            and str(row["url"]).startswith("https://www.jpx.co.jp/")
            and path.is_file()
            and _sha256(path) == str(row["sha256"])
        ):
            raise ValueError(f"official raw-page contract invalid: {row['url']}")
        body = path.read_bytes()
        if path.suffix.lower() == ".pdf" or "pdf" in str(row["content_type"]).lower():
            text = " ".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(body)).pages)
        else:
            text = " ".join(BeautifulSoup(body, "html.parser").stripped_strings)
        normalized_text = " ".join(text.split())
        normalized_text = re.sub(r"\b(20)\s+(\d{2})\b", r"\1\2", normalized_text)
        output[str(row["url"])] = normalized_text
    return output


def _load_verified_delistings(evidence_manifest: dict[str, Any]) -> pd.DataFrame:
    manifest_path = Path(str(evidence_manifest.get("input_manifests", {}).get("delisting_events", "")))
    expected_manifest_hash = evidence_manifest.get("input_manifest_sha256", {}).get(
        "delisting_events", ""
    )
    if not manifest_path.is_file() or _sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("delisting manifest lineage mismatch")
    manifest = _read_json(manifest_path)
    path = manifest_path.parent / "delisting_events.csv"
    expected = manifest.get("sha256", {}).get(path.name, "")
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ValueError("delisting event hash mismatch")
    return pd.read_csv(path, dtype=str).fillna("")


def _verified_vendor_shard(directory: Path) -> tuple[pd.DataFrame, Path]:
    manifest_path = directory / "manifest.json"
    request_paths = [directory / "quote_chain_request.json", directory / "targeted_request.json"]
    request_path = next((path for path in request_paths if path.is_file()), None)
    manifest = _read_json(manifest_path)
    request = _read_json(request_path) if request_path else {}
    output = directory / "security_master.csv"
    if not (
        request_path is not None
        and request.get("status") == "complete"
        and request.get("run_id") == directory.name
        and request.get("country_codes") == ["JPN"]
        and request.get("state_buckets") == ["active", "inactive"]
        and request.get("manifest_sha256") == _sha256(manifest_path)
        and manifest.get("output_sha256", {}).get("security_master") == _sha256(output)
    ):
        raise ValueError(f"vendor shard contract invalid: {directory}")
    frame = pd.read_csv(output, dtype=str).fillna("")
    required = {
        "ric",
        "ticker_symbol",
        "exchange_code",
        "instrument_permid",
        "rcs_asset_category_leaf",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"vendor shard schema invalid: {directory}")
    return frame, manifest_path


def _select_successor_vendor_row(
    vendor: pd.DataFrame, transition: dict[str, Any]
) -> pd.Series | None:
    code = str(transition["successor_code"])
    declared = [
        value for value in str(transition.get("successor_primary_rics", "")).split("|") if value
    ]
    candidates = vendor.loc[
        vendor["ticker_symbol"].astype(str).eq(code)
        & vendor["exchange_code"].isin(PRIMARY_EXCHANGES)
        & vendor["rcs_asset_category_leaf"].str.contains("Ordinary Share", case=False, na=False)
    ].copy()
    if declared:
        candidates = candidates.loc[candidates["ric"].isin(declared)]
    if candidates.empty:
        return None
    candidates["canonical_rank"] = candidates["ric"].map(
        lambda value: 0 if _ric_code(value) == code else 1
    )
    candidates["primary_rank"] = ~candidates.get(
        "is_primary_ric", pd.Series("", index=candidates.index)
    ).astype(str).str.lower().eq("true")
    candidates["dead_rank"] = candidates["ric"].str.contains(r"\^").astype(int)
    candidates = candidates.sort_values(
        ["canonical_rank", "primary_rank", "dead_rank", "ric"]
    )
    return candidates.iloc[0]


def _alias_start_after_direct_route(direct_rows: pd.DataFrame, old_code: str) -> str:
    if direct_rows.empty or "mapping_local_code_override" not in direct_rows.columns:
        return ""
    rows = direct_rows.loc[
        direct_rows["mapping_local_code_override"].astype(str).eq(str(old_code))
    ]
    retirements = pd.to_datetime(rows.get("retire_date"), errors="coerce").dropna()
    if retirements.empty:
        return ""
    return (retirements.max().date() + timedelta(days=1)).isoformat()


def _reason_target(reason: str) -> str:
    value = re.sub(
        r"\s*\(\s*(?:Stock Transfer|Stock Swap)\s*\)\s*$",
        "",
        str(reason),
        flags=re.I,
    )
    match = re.fullmatch(
        r"(?:Becoming a wholly owned subsidiary of|Merging with)\s+(.+)",
        value,
        re.I,
    )
    return match.group(1).strip() if match else ""


def _canonical_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(str(value))).upper()
    normalized = re.sub(r"\*\*", "", normalized)
    normalized = re.sub(
        r"\s*\(\s*(?:STOCK\s+TRANSFER|STOCK\s+SWAP)\s*\)\s*$", "", normalized
    )
    normalized = re.sub(r"\.?\s*STOCKS?$", "", normalized)
    return "".join(character for character in normalized if character.isalnum())


def _english_dates(text: str) -> set[str]:
    pattern = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\.?\s+(\d{1,2}),?\s+(20\d{2})\b",
        re.I,
    )
    result: set[str] = set()
    for month_text, day_text, year_text in pattern.findall(str(text)):
        try:
            month = datetime.strptime(month_text[:3].title(), "%b").month
            result.add(date(int(year_text), month, int(day_text)).isoformat())
        except ValueError:
            continue
    return result


def _text_mentions_iso_date(text: str, iso_date: str) -> bool:
    try:
        expected = date.fromisoformat(str(iso_date))
    except ValueError:
        return False
    if expected.isoformat() in _english_dates(text):
        return True
    month = expected.strftime("%B")
    short_month = expected.strftime("%b")
    day = expected.day
    explicit = re.search(
        rf"\b(?:{month}|{short_month})\.?\s+{day},?\s+{expected.year}\b",
        str(text),
        re.I,
    )
    if explicit:
        return True
    contextual = re.search(
        rf"\b(?:{month}|{short_month})\.?\s+{day}\b(?!\s*,?\s*\d{{4}})",
        str(text),
        re.I,
    )
    return bool(contextual and re.search(rf"\b{expected.year}\b", str(text)))


def _ric_code(value: str) -> str:
    match = re.match(r"^([0-9A-Z]{4})(?:L)?\.(?:T|Q|OS|OJ)(?:\^|$)", str(value).upper())
    return match.group(1) if match else ""


def _require_output_hashes(root: Path, manifest: dict[str, Any], names: Iterable[str]) -> None:
    for name in names:
        path = root / name
        expected = manifest.get("output_sha256", {}).get(name, "")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ValueError(f"transition evidence output hash mismatch: {name}")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
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
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
