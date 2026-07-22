from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader


HALT_CHANNEL = "halt_suspension_resumption"
SOURCE_RECORD_VERSION = "jpx-g4-report-row-v1"
EVENT_ID_VERSION = "jpx-g4-report-event-v1"
NORMALIZED_DISPOSITION = "normalized_wrapped_time_range"
APPENDIX_DISPOSITION = "out_of_instrument_scope_appendix_reference"
AGGREGATE_DISPOSITION = "out_of_instrument_scope_market_aggregate"

REPAIR_COLUMNS = (
    "report_event_id",
    "source_record_sha256",
    "report_month",
    "channel",
    "event_date",
    "raw_code",
    "normalized_local_code",
    "instrument_class",
    "issue_name",
    "halt_start_printed",
    "halt_end_printed",
    "repaired_halt_start_printed",
    "repaired_halt_end_printed",
    "repaired_halt_start_jst",
    "repaired_halt_end_jst",
    "source_sha256",
    "source_page",
    "source_row_ordinal",
    "source_record_text",
    "normalization_disposition",
    "in_research_scope_after",
    "local_code_invented",
    "source_semantics",
)

UNCHANGED_DATE_GRANULAR_ROWS = (
    ("0d9186d176b3143423592d2d944a37d4d0aeed2b3a9ccb1b55a6b59591f0e933", "2009-03", "1872"),
    ("f99c1c447719d8b48c72bb4ded73c3a4bb00a5309190672b41911bad295f6fac", "2009-03", "2652"),
    ("9ec1a725d8151951c3fc31118006121b4b0ca4cb43e1d6666cafe1e48db8ff05", "2009-05", "3064"),
    ("dfd2a1082666f7594a9bc767f6768bd0318a65c845dfd6cc2579256c8e147764", "2009-06", "8685"),
    ("77573ca15b2947706190a7f966c5cadcb38644b8c03e8934ea922675a1fc4099", "2009-06", "8918"),
    ("0e72753d609648f1f4d55d20444c4e85b5b473143963b7914be8a03e13ec8b20", "2009-06", "4350"),
    ("3a7811e2c32df9c56bd7dedf2156b29b8c4b88fd79c32e527e7bb0f77513e3b7", "2009-06", "6730"),
    ("3adf70ee2a80f5f4015fe6ef0486590e059e6a62cb86aff532cc5629a4bfb634", "2009-07", "8685"),
    ("e1c67c1d6907b99445c296c4ea6b1b4e67696c7b5133e7df9d1f608b5559c933", "2009-07", "8918"),
    ("a83b12d3a770b90cf647b455b554caa31872942ef9605295f0f4710d74f0f086", "2011-04", "8710"),
    ("9422ecbd27c1c716ab29eb47cbcc5f98198188c318c30b8773fa328cb014dc91", "2011-05", "8710"),
)


@dataclass(frozen=True)
class HaltNormalizationRepair:
    report_event_id: str
    source_record_sha256: str
    report_month: str
    event_date: str
    raw_code: str
    normalized_local_code: str
    instrument_class: str
    issue_name: str
    halt_start_printed: str
    halt_end_printed: str
    repaired_halt_start_printed: str
    repaired_halt_end_printed: str
    source_sha256: str
    source_page: int
    source_row_ordinal: int
    source_record_text: str
    normalization_disposition: str
    source_semantics: str
    page_marker: str

    @property
    def repaired_halt_start_jst(self) -> str:
        if not self.repaired_halt_start_printed:
            return ""
        return f"{self.event_date}T{self.repaired_halt_start_printed}:00+09:00"

    @property
    def repaired_halt_end_jst(self) -> str:
        if not self.repaired_halt_end_printed:
            return ""
        return f"{self.event_date}T{self.repaired_halt_end_printed}:00+09:00"

    @property
    def in_research_scope_after(self) -> bool:
        return self.normalization_disposition == NORMALIZED_DISPOSITION

    def validate_source_row(self, source: dict[str, Any]) -> None:
        expected = {
            "event_id": self.report_event_id,
            "source_record_sha256": self.source_record_sha256,
            "report_month": self.report_month,
            "channel": HALT_CHANNEL,
            "event_date": self.event_date,
            "raw_code": self.raw_code,
            "normalized_local_code": self.normalized_local_code,
            "instrument_class": self.instrument_class,
            "issue_name": self.issue_name,
            "halt_start_printed": self.halt_start_printed,
            "halt_end_printed": self.halt_end_printed,
            "source_sha256": self.source_sha256,
            "source_page": str(self.source_page),
            "source_row_ordinal": str(self.source_row_ordinal),
            "source_record_text": self.source_record_text,
        }
        mismatches = [
            field
            for field, value in expected.items()
            if str(source.get(field, "")) != str(value)
        ]
        if mismatches:
            raise ValueError(
                "halt normalization repair source-row contract mismatch: "
                f"{self.source_record_sha256}:{'|'.join(mismatches)}"
            )

    def source_identity(self) -> dict[str, Any]:
        return {
            "version": SOURCE_RECORD_VERSION,
            "report_month": self.report_month,
            "channel": HALT_CHANNEL,
            "source_sha256": self.source_sha256,
            "source_page": self.source_page,
            "source_row_ordinal": self.source_row_ordinal,
            "event_date": self.event_date,
            "raw_code": self.raw_code,
            "normalized_local_code": self.normalized_local_code,
            "instrument_class": self.instrument_class,
            "issue_name": self.issue_name,
            "halt_start_printed": self.halt_start_printed,
            "halt_end_printed": self.halt_end_printed,
        }

    def artifact_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("page_marker")
        row.update(
            {
                "channel": HALT_CHANNEL,
                "repaired_halt_start_jst": self.repaired_halt_start_jst,
                "repaired_halt_end_jst": self.repaired_halt_end_jst,
                "in_research_scope_after": self.in_research_scope_after,
                "local_code_invented": False,
            }
        )
        return {column: row[column] for column in REPAIR_COLUMNS}


@dataclass(frozen=True)
class HaltNormalizationRepairArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    repairs_path: Path
    manifest: dict[str, Any]


def _repair(
    event_id: str,
    record_sha: str,
    month: str,
    event_date: str,
    raw_code: str,
    issue_name: str,
    start: str,
    end: str,
    source_sha: str,
    page: int,
    ordinal: int,
    record_text: str,
    disposition: str,
    semantics: str,
    marker: str,
    *,
    repaired_start: str = "",
    repaired_end: str = "",
) -> HaltNormalizationRepair:
    return HaltNormalizationRepair(
        event_id,
        record_sha,
        month,
        event_date,
        raw_code,
        raw_code if raw_code.isascii() and raw_code.isdigit() else "",
        "equity",
        issue_name,
        start,
        end,
        repaired_start,
        repaired_end,
        source_sha,
        page,
        ordinal,
        record_text,
        disposition,
        semantics,
        marker,
    )


_S12 = "c86c868a01e18e2afc17b5bf15f408e8c838a3ff2397cf18a81f9cb9581f9720"
_S13 = "e7c63b7e0ff65dc6a4ea5ad8a6650c8b6b1306f79454092549c4c42589ba8958"
_S20 = "2ff5a87bcc558b574ad38b4ea7c1745f3692b038473812942fb130784bdd4725"

HALT_NORMALIZATION_REPAIRS = (
    _repair("507b4c0a64f374a5de1553e7", "ada193784ba67fd41f728c7098099d7f6812fccd2511de606fcfd1b07e030a2a", "2012-02", "2012-02-02", "", "別紙参照", "09:00", "12:29", _S12, 2, 1, "市場第一部 2012.02.02 別紙参照 9:00~12:29", APPENDIX_DISPOSITION, "market-segment appendix pointer, not an instrument row", "市場第一部 2012.02.02 別紙参照 9:00~12:29"),
    _repair("df626b8c698ff7dee593f49b", "bb8f1feae007a54e895b8288974f76c29d83f2dca885d7d57bd4fcee26627f93", "2012-02", "2012-02-02", "", "別紙参照", "09:00", "12:29", _S12, 2, 7, "市場第二部 2012.02.02 別紙参照 9:00~12:29", APPENDIX_DISPOSITION, "market-segment appendix pointer, not an instrument row", "市場第二部 2012.02.02 別紙参照 9:00~12:29"),
    _repair("bd3442ebae0389b88a24165b", "4020ded60e62598386caf2bb0cff162ef3146cc1eb8a40788109dc584b7b80c6", "2012-02", "2012-02-02", "", "別紙参照", "09:00", "12:29", _S12, 2, 10, "マザーズ 2012.02.02 別紙参照 9:00~12:29", APPENDIX_DISPOSITION, "market-segment appendix pointer, not an instrument row", "マザーズ 2012.02.02 別紙参照 9:00~12:29"),
    _repair("eecf102a06628b373a85518a", "312e13eac18d061acd5b4c0c29bfdd936df8427366413b95047196c397142d06", "2013-09", "2013-09-30", "4585", "UMNファーマ UMN Pharma Inc.", "", "", _S13, 4, 5, "2013.09.30 4585 UMNファーマ UMN Pharma Inc.", NORMALIZED_DISPOSITION, "time range is wrapped onto the immediately following line", "2013.09.30 4585 UMNファーマ UMN Pharma Inc. 17:10~17:25", repaired_start="17:10", repaired_end="17:25"),
    _repair("a43676147845ea79c549f75f", "e065ec4bf5338c23e7a88978d1506d1824a996357adef56bcee658c521da515f", "2020-10", "2020-10-01", "全銘柄", "All Listed Issues", "08:56", "17:30", _S20, 5, 1, "市場第一部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "市場第一部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30"),
    _repair("3704202495019130830d2705", "7f6b032500b86c6c9fe21a6cd2334e3944aed821b86cf87fee36b0e56a950c23", "2020-10", "2020-10-01", "全銘柄", "All Listed Issues", "08:56", "17:30", _S20, 5, 12, "市場第二部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "市場第二部 2020.10.01 全銘柄 All Listed Issues 8:56~17:30"),
    _repair("6c2603fbbcfa2be20c62bb4c", "9632e9aea346d6a1f3be71f7808bd77a5dce11ea4d7d45b6a5983e7b407767de", "2020-10", "2020-10-01", "全銘柄", "All Listed Issues", "08:56", "17:30", _S20, 5, 13, "マザーズ 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "マザーズ 2020.10.01 全銘柄 All Listed Issues 8:56~17:30"),
    _repair("bbf5abd3849655ef8ae3cde6", "1e6e6be36d3283105355050d055aa2f610d7e41207a00324f05e626e2a8156ea", "2020-10", "2020-10-01", "全銘柄", "全銘柄", "08:56", "17:30", _S20, 5, 16, "TOKYO PRO Market 2020.10.01 全銘柄 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "TOKYO PRO Market 2020.10.01 全銘柄 8:56~17:30"),
    _repair("97891b11479f24eb72938b8a", "31f193cf7ab03e5941aded3b1292d489cdc11202234c72351fa0d0081f6c8ae7", "2020-10", "2020-10-01", "全銘柄", "All Listed Issues", "08:56", "17:30", _S20, 5, 17, "JASDAQスタンダード 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "JASDAQスタンダード 2020.10.01 全銘柄 All Listed Issues 8:56~17:30"),
    _repair("c4d38b30cdae1b7962b1de16", "218f01717eaf43e821f5fa50618973c65066cb0b52c91c216097af8c05898723", "2020-10", "2020-10-01", "全銘柄", "All Listed Issues", "08:56", "17:30", _S20, 5, 21, "JASDAQグロース 2020.10.01 全銘柄 All Listed Issues 8:56~17:30", AGGREGATE_DISPOSITION, "all listed issues market-segment aggregate, not one instrument", "JASDAQグロース 2020.10.01 全銘柄 All Listed Issues 8:56~17:30"),
)

_REPAIR_BY_RECORD = {repair.source_record_sha256: repair for repair in HALT_NORMALIZATION_REPAIRS}


def normalization_repair_for(source: dict[str, Any]) -> HaltNormalizationRepair | None:
    repair = _REPAIR_BY_RECORD.get(str(source.get("source_record_sha256", "")))
    if repair is not None:
        repair.validate_source_row(source)
    return repair


def run_halt_report_normalization_repair(
    *,
    run_id: str,
    source_paths: dict[str, str | Path],
    artifact_root: str | Path = "artifacts/japan_g4_halt_report_normalization_repair",
) -> HaltNormalizationRepairArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    expected_months = {repair.report_month for repair in HALT_NORMALIZATION_REPAIRS}
    if set(source_paths) != expected_months:
        raise ValueError("normalization repair requires exactly 2012-02, 2013-09, and 2020-10 sources")

    root = Path(artifact_root).resolve() / run_id
    if root.exists():
        raise FileExistsError(f"write-once normalization repair artifact exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        raw = staging / "raw"
        raw.mkdir()
        source_rows: list[dict[str, Any]] = []
        for month in sorted(expected_months):
            source = Path(source_paths[month]).resolve()
            body = source.read_bytes()
            observed = hashlib.sha256(body).hexdigest()
            repairs = [item for item in HALT_NORMALIZATION_REPAIRS if item.report_month == month]
            expected = {item.source_sha256 for item in repairs}
            if expected != {observed}:
                raise ValueError(f"retained source hash mismatch: {month}")
            pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages]
            for repair in repairs:
                if repair.source_page > len(pages) or _canonical(repair.page_marker) not in _canonical(pages[repair.source_page - 1]):
                    raise ValueError(f"retained source semantics mismatch: {repair.source_record_sha256}")
                if _record_sha(repair.source_identity()) != repair.source_record_sha256:
                    raise ValueError(f"source-record identity mismatch: {repair.source_record_sha256}")
                event_id = hashlib.sha256(
                    f"{EVENT_ID_VERSION}|{repair.source_record_sha256}".encode("utf-8")
                ).hexdigest()[:24]
                if event_id != repair.report_event_id:
                    raise ValueError(f"event identity mismatch: {repair.report_event_id}")
            target = raw / f"{month}.pdf"
            shutil.copyfile(source, target)
            if _sha(target) != observed:
                raise ValueError(f"retained source copy hash mismatch: {month}")
            source_rows.append(
                {
                    "report_month": month,
                    "source_path": str(source),
                    "retained_path": str(target.relative_to(staging)),
                    "expected_sha256": observed,
                    "observed_sha256": _sha(target),
                    "hash_match": True,
                    "semantic_rows_verified": len(repairs),
                }
            )

        repairs = pd.DataFrame(
            [repair.artifact_row() for repair in HALT_NORMALIZATION_REPAIRS],
            columns=REPAIR_COLUMNS,
        )
        repairs_path = staging / "normalization_repairs.csv"
        repairs.to_csv(repairs_path, index=False)
        pd.DataFrame(source_rows).to_csv(staging / "source_hash_validation.csv", index=False)
        pd.DataFrame(
            [
                {
                    "source_record_sha256": digest,
                    "report_month": month,
                    "local_code": code,
                    "normalization_status": "missing_halt_start",
                    "contract_action": "unchanged",
                }
                for digest, month, code in UNCHANGED_DATE_GRANULAR_ROWS
            ]
        ).to_csv(staging / "unchanged_date_granular_contract.csv", index=False)

        inventory = _inventory(staging)
        inventory_path = staging / "artifact_hashes.json"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "1.0-bounded-exact-record-repair",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "bounded_normalization_repair_candidate_pending_independent_validation",
            "repair_rows": len(repairs),
            "wrapped_time_range_rows": int(repairs["normalization_disposition"].eq(NORMALIZED_DISPOSITION).sum()),
            "appendix_reference_rows": int(repairs["normalization_disposition"].eq(APPENDIX_DISPOSITION).sum()),
            "market_aggregate_rows": int(repairs["normalization_disposition"].eq(AGGREGATE_DISPOSITION).sum()),
            "instrument_events_created": 1,
            "evidence_only_rows": 9,
            "invented_local_codes": 0,
            "unchanged_date_granular_rows": len(UNCHANGED_DATE_GRANULAR_ROWS),
            "source_hashes_exact": True,
            "source_semantics_checked_with": "pypdf sequential page text",
            "refinitiv_used": False,
            "primary_halt_v8_modified": False,
            "scorecard_eligible": False,
            "independent_validation_required": True,
            "write_once": True,
            "artifact_hashes_sha256": _sha(inventory_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return HaltNormalizationRepairArtifacts(
        run_id,
        root,
        root / "manifest.json",
        root / "normalization_repairs.csv",
        manifest,
    )


def verify_halt_report_normalization_repair(root: str | Path) -> list[str]:
    path = Path(root).resolve()
    issues: list[str] = []
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        inventory_path = path / "artifact_hashes.json"
        if _sha(inventory_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("artifact_hashes_digest_mismatch")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for relative, expected in inventory.items():
            item = Path(str(relative))
            if item.is_absolute() or ".." in item.parts:
                issues.append("unsafe_inventory_path")
            elif not (path / item).is_file() or _sha(path / item) != expected:
                issues.append(f"artifact_hash_mismatch:{relative}")
        expected_files = set(inventory) | {"manifest.json", "artifact_hashes.json"}
        actual_files = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
        if actual_files != expected_files:
            issues.append("artifact_file_denominator_mismatch")
        repairs = pd.read_csv(path / "normalization_repairs.csv", dtype=str).fillna("")
        if len(repairs) != 10 or set(repairs["source_record_sha256"]) != set(_REPAIR_BY_RECORD):
            issues.append("repair_record_set_mismatch")
        if repairs["local_code_invented"].str.casefold().ne("false").any():
            issues.append("local_code_invention_claimed")
        if int(manifest.get("unchanged_date_granular_rows", -1)) != 11:
            issues.append("date_granular_contract_count_mismatch")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        issues.append(f"artifact_unreadable:{type(exc).__name__}")
    return sorted(set(issues))


def _record_sha(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("～", "~").replace("〜", "~")
    return re.sub(r"\s+", " ", normalized).strip()


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
