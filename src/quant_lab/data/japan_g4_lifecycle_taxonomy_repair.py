from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from quant_lab.data.japan_g4_halt_report_events import (
    EVENT_ID_VERSION,
    SOURCE_RECORD_VERSION,
    source_record_sha256,
)


VERSION = "1.0-bounded-v14-lifecycle-taxonomy-repair"
WARRANT_MARKER = "新株予約権"
FOREIGN_SHARE_MARKER = "<外国株>"
NOTICE_FOREIGN_SHARE_MARKER = "市場区分:マザーズ(外国株)"
NOTICE_URL = "https://www.jpx.co.jp/news/1021/20150302-11.html"
NOTICE_EVENT_ID = "765af2e977f955916788a27d"
TARGET_CLASS_BY_CODE = {
    "77089": "warrant",
    "45639": "warrant",
    "76389": "warrant",
    "35989": "warrant",
    "38109": "warrant",
    "8634": "foreign_share",
    "9399": "foreign_share",
    "5412": "foreign_share",
    "3827": "foreign_share",
}
TARGET_MARKER_BY_CLASS = {
    "warrant": WARRANT_MARKER,
    "foreign_share": FOREIGN_SHARE_MARKER,
}

SOURCE_COLUMNS = (
    "event_id",
    "report_month",
    "channel",
    "event_type",
    "event_date",
    "raw_code",
    "normalized_local_code",
    "instrument_class",
    "issue_name",
    "halt_start_printed",
    "halt_end_printed",
    "source_url",
    "source_sha256",
    "source_page",
    "source_row_ordinal",
    "source_record_text",
    "source_record_sha256",
    "copied_source_path",
    "taxonomy_evidence_route",
    "taxonomy_evidence_url",
    "taxonomy_evidence_path",
    "taxonomy_evidence_sha256",
)
LIFECYCLE_REQUIRED_COLUMNS = SOURCE_COLUMNS[: SOURCE_COLUMNS.index("copied_source_path")]
REPAIR_COLUMNS = (
    "event_id",
    "source_record_sha256",
    "report_month",
    "channel",
    "event_date",
    "raw_code",
    "normalized_local_code",
    "source_instrument_class",
    "reclassified_instrument_class",
    "required_source_marker",
    "taxonomy_evidence_route",
    "taxonomy_evidence_url",
    "taxonomy_evidence_path",
    "taxonomy_evidence_sha256",
    "source_sha256",
    "source_page",
    "source_row_ordinal",
    "source_record_text",
    "taxonomy_disposition",
    "in_research_scope_after",
    "mapping_created",
    "timestamp_created",
    "zero_event_created",
    "classification_inferred",
)


@dataclass(frozen=True)
class LifecycleTaxonomyRepairArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    repairs_path: Path
    manifest: dict[str, Any]


def run_japan_g4_lifecycle_taxonomy_repair(
    *,
    run_id: str,
    source_packet_dirs: Sequence[str | Path],
    official_notice_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_lifecycle_taxonomy_repair",
) -> LifecycleTaxonomyRepairArtifacts:
    _validate_run_id(run_id)
    packets = tuple(Path(path).resolve() for path in source_packet_dirs)
    if not packets:
        raise ValueError("at least one copied lifecycle producer packet is required")

    candidates: list[tuple[dict[str, str], Path]] = []
    for packet in packets:
        events_path = packet / "exact_events.csv"
        if not events_path.is_file():
            raise ValueError(f"copied lifecycle exact_events.csv missing: {packet}")
        events = pd.read_csv(events_path, dtype=str).fillna("")
        missing = set(LIFECYCLE_REQUIRED_COLUMNS) - set(events.columns)
        if missing:
            raise ValueError(f"copied lifecycle schema missing: {sorted(missing)}")
        for row in events.loc[events["raw_code"].isin(TARGET_CLASS_BY_CODE)].to_dict("records"):
            candidates.append(({str(key): str(value) for key, value in row.items()}, packet))

    observed_codes = [row["raw_code"] for row, _ in candidates]
    if sorted(observed_codes) != sorted(TARGET_CLASS_BY_CODE):
        raise ValueError("taxonomy repair requires exactly one retained row for each target code")
    if len({row["event_id"] for row, _ in candidates}) != len(TARGET_CLASS_BY_CODE):
        raise ValueError("taxonomy repair event IDs are not unique")
    if len({row["source_record_sha256"] for row, _ in candidates}) != len(TARGET_CLASS_BY_CODE):
        raise ValueError("taxonomy repair source-record hashes are not unique")

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once taxonomy repair artifact exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        raw_dir = staging / "raw" / "pdfs"
        raw_dir.mkdir(parents=True)
        source_rows: list[dict[str, Any]] = []
        repair_rows: list[dict[str, Any]] = []
        copied_by_identity: dict[tuple[str, str], str] = {}
        source_lineage: list[dict[str, Any]] = []
        notice_source = Path(official_notice_path).resolve()
        notice_body = notice_source.read_bytes()
        notice_sha = hashlib.sha256(notice_body).hexdigest()
        _validate_3827_notice(notice_body)
        notice_dir = staging / "raw" / "notices"
        notice_dir.mkdir(parents=True)
        notice_relative = "raw/notices/20150302-11.html"
        shutil.copyfile(notice_source, staging / notice_relative)

        for row, packet in sorted(candidates, key=lambda item: item[0]["event_id"]):
            target_class = TARGET_CLASS_BY_CODE[row["raw_code"]]
            notice_backed = row["raw_code"] == "3827"
            marker = NOTICE_FOREIGN_SHARE_MARKER if notice_backed else TARGET_MARKER_BY_CLASS[target_class]
            _validate_source_row(row, None if notice_backed else marker)
            if notice_backed:
                _bind_3827_notice_to_event(row, notice_body)

            source_path = _safe_packet_source_path(packet, row)
            body = source_path.read_bytes()
            observed_sha = hashlib.sha256(body).hexdigest()
            if observed_sha != row["source_sha256"]:
                raise ValueError(f"official PDF hash mismatch: {row['event_id']}")
            _require_marker_on_page(body, row, None if notice_backed else marker)

            identity = (row["report_month"], observed_sha)
            copied_name = copied_by_identity.get(identity)
            if copied_name is None:
                copied_name = f"{row['report_month']}-{observed_sha[:12]}.pdf"
                shutil.copyfile(source_path, raw_dir / copied_name)
                copied_by_identity[identity] = copied_name
                source_lineage.append(
                    {
                        "report_month": row["report_month"],
                        "source_url": row["source_url"],
                        "source_sha256": observed_sha,
                        "copied_source_path": f"raw/pdfs/{copied_name}",
                        "source_packet_exact_events_sha256": _sha(packet / "exact_events.csv"),
                    }
                )

            source_row = {column: row[column] for column in LIFECYCLE_REQUIRED_COLUMNS}
            source_row["copied_source_path"] = f"raw/pdfs/{copied_name}"
            source_row["taxonomy_evidence_route"] = (
                "separate_official_jpx_delisting_notice" if notice_backed else "retained_lifecycle_source_record"
            )
            source_row["taxonomy_evidence_url"] = NOTICE_URL if notice_backed else row["source_url"]
            source_row["taxonomy_evidence_path"] = notice_relative if notice_backed else f"raw/pdfs/{copied_name}"
            source_row["taxonomy_evidence_sha256"] = notice_sha if notice_backed else row["source_sha256"]
            source_rows.append(source_row)
            repair_rows.append(
                {
                    "event_id": row["event_id"],
                    "source_record_sha256": row["source_record_sha256"],
                    "report_month": row["report_month"],
                    "channel": row["channel"],
                    "event_date": row["event_date"],
                    "raw_code": row["raw_code"],
                    "normalized_local_code": row["normalized_local_code"],
                    "source_instrument_class": row["instrument_class"],
                    "reclassified_instrument_class": target_class,
                    "required_source_marker": marker,
                    "taxonomy_evidence_route": source_row["taxonomy_evidence_route"],
                    "taxonomy_evidence_url": source_row["taxonomy_evidence_url"],
                    "taxonomy_evidence_path": source_row["taxonomy_evidence_path"],
                    "taxonomy_evidence_sha256": source_row["taxonomy_evidence_sha256"],
                    "source_sha256": row["source_sha256"],
                    "source_page": row["source_page"],
                    "source_row_ordinal": row["source_row_ordinal"],
                    "source_record_text": row["source_record_text"],
                    "taxonomy_disposition": f"quarantined_{target_class}",
                    "in_research_scope_after": False,
                    "mapping_created": False,
                    "timestamp_created": False,
                    "zero_event_created": False,
                    "classification_inferred": False,
                }
            )

        repairs = pd.DataFrame(repair_rows, columns=REPAIR_COLUMNS)
        sources = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
        repairs.to_csv(staging / "taxonomy_repairs.csv", index=False)
        sources.to_csv(staging / "source_records.csv", index=False)
        pd.DataFrame(source_lineage).sort_values(["report_month", "source_sha256"]).to_csv(
            staging / "source_lineage.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "event_id": NOTICE_EVENT_ID,
                    "raw_code": "3827",
                    "event_date": "2015-03-26",
                    "source_url": NOTICE_URL,
                    "source_path": notice_relative,
                    "source_sha256": notice_sha,
                    "required_market_division": "Mothers (Foreign Stocks)",
                    "classification_basis": "explicit official notice market division",
                }
            ]
        ).to_csv(staging / "notice_lineage.csv", index=False)

        inventory = _inventory(staging)
        inventory_path = staging / "artifact_hashes.json"
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": VERSION,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "bounded_taxonomy_repair_candidate_pending_independent_validation",
            "source_contract": "V14 copied lifecycle exact rows plus retained official JPX PDF bytes",
            "repair_rows": len(repairs),
            "warrant_rows": int(repairs["reclassified_instrument_class"].eq("warrant").sum()),
            "foreign_share_rows": int(repairs["reclassified_instrument_class"].eq("foreign_share").sum()),
            "unique_event_ids": int(repairs["event_id"].nunique()),
            "unique_source_record_hashes": int(repairs["source_record_sha256"].nunique()),
            "official_pdf_count": len(copied_by_identity),
            "official_notice_count": 1,
            "official_notice_sha256": notice_sha,
            "notice_backed_foreign_share_rows": 1,
            "classification_inferred_rows": 0,
            "mappings_created": 0,
            "timestamps_created": 0,
            "zero_events_created": 0,
            "research_scope_rows_after": 0,
            "source_semantics_checked_with": "pypdf page text",
            "independent_validation_required": True,
            "refinitiv_used": False,
            "scorecard_eligible": False,
            "write_once": True,
            "artifact_hashes_sha256": _sha(inventory_path),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return LifecycleTaxonomyRepairArtifacts(
        run_id=run_id,
        artifact_dir=root,
        manifest_path=root / "manifest.json",
        repairs_path=root / "taxonomy_repairs.csv",
        manifest=manifest,
    )


def verify_japan_g4_lifecycle_taxonomy_repair(root: str | Path) -> list[str]:
    path = Path(root).resolve()
    issues = _verify_inventory(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        repairs = pd.read_csv(path / "taxonomy_repairs.csv", dtype=str).fillna("")
        if manifest.get("status") != "bounded_taxonomy_repair_candidate_pending_independent_validation":
            issues.append("producer_status_mismatch")
        if len(repairs) != 9 or set(repairs.get("raw_code", [])) != set(TARGET_CLASS_BY_CODE):
            issues.append("repair_row_set_mismatch")
        if repairs.get("event_id", pd.Series(dtype=str)).nunique() != 9:
            issues.append("event_id_uniqueness_failure")
        for field in (
            "in_research_scope_after",
            "mapping_created",
            "timestamp_created",
            "zero_event_created",
            "classification_inferred",
        ):
            if repairs.get(field, pd.Series(dtype=str)).str.casefold().ne("false").any():
                issues.append(f"forbidden_true_flag:{field}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        issues.append(f"artifact_unreadable:{type(exc).__name__}")
    return sorted(set(issues))


def _validate_source_row(row: dict[str, str], marker: str | None) -> None:
    if row["instrument_class"] != "equity":
        raise ValueError(f"source row is not the V14 equity misclassification: {row['event_id']}")
    if marker is not None and marker not in _canonical(row["source_record_text"]):
        raise ValueError(f"required Japanese source marker missing: {row['event_id']}")
    identity = {
        "version": SOURCE_RECORD_VERSION,
        "report_month": row["report_month"],
        "channel": row["channel"],
        "source_sha256": row["source_sha256"],
        "source_page": int(row["source_page"]),
        "source_row_ordinal": int(row["source_row_ordinal"]),
        "event_date": row["event_date"],
        "raw_code": row["raw_code"],
        "normalized_local_code": row["normalized_local_code"],
        "instrument_class": row["instrument_class"],
        "issue_name": row["issue_name"],
        "halt_start_printed": row["halt_start_printed"],
        "halt_end_printed": row["halt_end_printed"],
    }
    observed_record = source_record_sha256(identity)
    if observed_record != row["source_record_sha256"]:
        raise ValueError(f"source-record identity hash mismatch: {row['event_id']}")
    observed_event = hashlib.sha256(
        f"{EVENT_ID_VERSION}|{observed_record}".encode("utf-8")
    ).hexdigest()[:24]
    if observed_event != row["event_id"]:
        raise ValueError(f"event identity mismatch: {row['event_id']}")


def _safe_packet_source_path(packet: Path, row: dict[str, str]) -> Path:
    relative = Path(row.get("source_path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe lifecycle source path: {relative}")
    source = (packet / relative).resolve()
    if packet not in source.parents or not source.is_file():
        raise ValueError(f"lifecycle source PDF missing: {relative}")
    if source.suffix.casefold() != ".pdf" or "jpx.co.jp/" not in row["source_url"]:
        raise ValueError(f"source is not an official JPX PDF: {row['event_id']}")
    return source


def _require_marker_on_page(body: bytes, row: dict[str, str], marker: str | None) -> None:
    pages = PdfReader(io.BytesIO(body)).pages
    page_number = int(row["source_page"])
    if page_number < 1 or page_number > len(pages):
        raise ValueError(f"source page outside official PDF: {row['event_id']}")
    page_text = _canonical(pages[page_number - 1].extract_text() or "")
    if (marker is not None and marker not in page_text) or _dense_canonical(row["source_record_text"]) not in _dense_canonical(page_text):
        raise ValueError(f"official PDF does not contain exact retained source row: {row['event_id']}")


def _validate_3827_notice(body: bytes) -> None:
    text = _canonical(BeautifulSoup(body, "html.parser").get_text(" ", strip=True))
    required = (
        "コード:3827",
        NOTICE_FOREIGN_SHARE_MARKER,
        "上場廃止日 2015年3月26日(木)",
        "ジャパンインベスト・グループ・ピー・エル・シー",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError(f"official 3827 notice semantics missing: {missing}")


def _bind_3827_notice_to_event(row: dict[str, str], body: bytes) -> None:
    if (
        row["event_id"] != NOTICE_EVENT_ID
        or row["raw_code"] != "3827"
        or row["event_date"] != "2015-03-26"
        or not row["issue_name"].startswith("ジャパンインベスト・グループ・ピー・エル・シー")
    ):
        raise ValueError("official 3827 notice does not bind to the exact V14 event")
    _validate_3827_notice(body)


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).replace("～", "~").replace("〜", "~")
    return re.sub(r"\s+", " ", normalized).strip()


def _dense_canonical(value: str) -> str:
    return re.sub(r"\s+", "", _canonical(value))


def _verify_inventory(root: Path) -> list[str]:
    issues: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        inventory_path = root / "artifact_hashes.json"
        if _sha(inventory_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("artifact_hashes_digest_mismatch")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for relative, expected in inventory.items():
            item = Path(str(relative))
            if item.is_absolute() or ".." in item.parts:
                issues.append("unsafe_inventory_path")
            elif not (root / item).is_file() or _sha(root / item) != expected:
                issues.append(f"artifact_hash_mismatch:{relative}")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        expected = set(inventory) | {"manifest.json", "artifact_hashes.json"}
        if actual != expected:
            issues.append("artifact_file_denominator_mismatch")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"inventory_unreadable:{type(exc).__name__}")
    return issues


def _inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
