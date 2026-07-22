from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from pypdf import PdfReader

from quant_lab.data.jpx_monthly_quotations import conservative_quotation_known_at


UTC = timezone.utc
BASE = "https://www.jpx.co.jp/files/ose/markets/data/equities/price"
USER_AGENT = "quant-lab-ose-monthly-quotations/1.0"
_OFFICIAL_MIGRATED_REPAIRS = {
    "200906-ose": f"{BASE}/KM200906t.pdf",
}
_COMMON_SHARE = re.compile(
    r"(?P<code>[1-9][0-9A-Z]{3})\s*(?P<name>[^\n]{1,100}?)\s*普通株(?:式)?"
)
_LEGACY_ROW = re.compile(
    r"(?m)^\s*[0-9][0-9,]*\s+(?:[1-9]|1[0-2])\s+"
    r"(?P<code>[1-9][0-9A-Z]{3})\s+(?P<name>\S.{0,100}?)"
    r"(?=\s+(?:[-|]|[0-9][0-9,]*(?:\.\d+)?))"
)


Fetcher = Callable[[str], tuple[int, bytes, Mapping[str, str]]]
PdfTextExtractor = Callable[[bytes], str]


@dataclass(frozen=True)
class OSEMonthlyQuotationArtifacts:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    status_path: Path
    quality_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def ose_monthly_quotation_url(month: pd.Period | str) -> str:
    period = month if isinstance(month, pd.Period) else pd.Period(month, freq="M")
    yyyymm = period.strftime("%Y%m")
    if period <= pd.Period("2010-09", freq="M"):
        return f"{BASE}/KM{yyyymm}.pdf"
    return f"{BASE}/k{yyyymm[2:]}.pdf"


def build_ose_monthly_quotation_plan(
    *, start_month: str = "2007-12", end_month: str = "2013-06"
) -> pd.DataFrame:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must not precede start_month")
    if end > pd.Period("2013-06", freq="M"):
        raise ValueError("OSE cash-equity membership source ends before the July 2013 TSE integration")
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        yyyymm = month.strftime("%Y%m")
        rows.append(
            {
                "request_id": f"{yyyymm}-ose",
                "observation_month": str(month),
                "yyyymm": yyyymm,
                "membership_effective_month_end": month.end_time.date().isoformat(),
                "known_at_utc": conservative_quotation_known_at(month),
                "known_at_policy": "official_monthly_publication_lagged_to_21st_calendar_day_00_jst",
                "source_url": ose_monthly_quotation_url(month),
                "source_official_jpx": True,
                "source_exchange": "Osaka Securities Exchange",
            }
        )
    return pd.DataFrame(rows)


def parse_ose_monthly_quotation_text(
    text: str,
    *,
    observation_month: str,
    known_at_utc: str,
    source_url: str,
    source_sha256: str,
) -> pd.DataFrame:
    period = pd.Period(observation_month, freq="M")
    source_exchange = (
        "JASDAQ Securities Exchange"
        if period <= pd.Period("2010-09", freq="M")
        else "Osaka Securities Exchange combined cash markets"
    )
    modern = list(_COMMON_SHARE.finditer(text))
    layout = "ordinary_share_label" if modern else "legacy_stock_quotation_row"
    matches = modern if layout == "ordinary_share_label" else list(_LEGACY_ROW.finditer(text))
    rows = []
    for match in matches:
        name = " ".join(match.group("name").split())
        if layout == "legacy_stock_quotation_row":
            name = _repair_legacy_text(name)
        rows.append(
            {
                "observation_month": observation_month,
                "membership_effective_month_end": pd.Period(
                    observation_month, freq="M"
                ).end_time.date().isoformat(),
                "known_at_utc": known_at_utc,
                "known_at_policy": "official_monthly_publication_lagged_to_21st_calendar_day_00_jst",
                "code": match.group("code").upper(),
                "issue_name_japanese": name,
                "source_exchange": source_exchange,
                "parser_layout": layout,
                "currency": "JPY",
                "membership_observed": True,
                "pit_usable": True,
                "source_url": source_url,
                "source_sha256": source_sha256,
            }
        )
    columns = [
        "observation_month",
        "membership_effective_month_end",
        "known_at_utc",
        "known_at_policy",
        "code",
        "issue_name_japanese",
        "source_exchange",
        "parser_layout",
        "currency",
        "membership_observed",
        "pit_usable",
        "source_url",
        "source_sha256",
    ]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["observation_month", "code"]
    )


def run_ose_monthly_quotations(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/ose_monthly_quotations",
    start_month: str = "2007-12",
    end_month: str = "2013-06",
    max_files: int = 12,
    pacing_seconds: float = 0.1,
    retries: int = 3,
    fetcher: Fetcher | None = None,
    pdf_text_extractor: PdfTextExtractor | None = None,
) -> OSEMonthlyQuotationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    if max_files <= 0 or retries < 0 or pacing_seconds < 0:
        raise ValueError("bounds must be non-negative and max_files positive")
    fetch = fetcher or _fetch
    extract_text = pdf_text_extractor or _extract_pdf_text
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "request_plan.csv"
    plan_hash_path = root / "request_plan.sha256"
    status_path = root / "file_status.csv"
    quality_path = root / "file_quality.csv"
    manifest_path = root / "manifest.json"
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc

    try:
        if plan_path.is_file():
            if not plan_hash_path.is_file() or plan_hash_path.read_text().strip() != _sha256(plan_path):
                raise RuntimeError("immutable OSE request plan hash mismatch")
            plan = pd.read_csv(plan_path, dtype=str).fillna("")
        else:
            plan = build_ose_monthly_quotation_plan(
                start_month=start_month, end_month=end_month
            )
            _atomic_write_csv(plan_path, plan)
            _atomic_write_text(plan_hash_path, _sha256(plan_path) + "\n")

        status = _read_csv(status_path, _status_columns())
        quality = _read_csv(quality_path, _quality_columns())
        wrong_repair = status["request_id"].eq("200906-ose") & status["source_url"].str.contains(
            "000k0906.pdf", na=False
        )
        if wrong_repair.any():
            status = status.loc[~wrong_repair].copy()
            quality = quality.loc[~quality["request_id"].eq("200906-ose")].copy()
            _atomic_write_csv(status_path, status)
            _atomic_write_csv(quality_path, quality)
        completed = set(status.loc[status["status"].eq("completed"), "request_id"])
        previous = status.drop_duplicates("request_id", keep="last").set_index("request_id")
        pending = plan.loc[~plan["request_id"].isin(completed)].copy()
        prior_status = previous["status"].to_dict() if not previous.empty else {}
        pending["_retry_priority"] = pending["request_id"].map(prior_status).fillna("").ne("").astype(int)
        pending = pending.sort_values(["_retry_priority", "yyyymm"]).head(max_files)
        status_rows = status.to_dict("records")
        quality_rows = quality.to_dict("records")

        for item in pending.to_dict("records"):
            request_id = str(item["request_id"])
            effective_source_url = str(item["source_url"])
            body = b""
            http_status = 0
            attempts = 0
            error = ""
            for attempt in range(1, retries + 2):
                attempts = attempt
                try:
                    http_status, body, _ = fetch(effective_source_url)
                    if http_status == 200 or (http_status != 429 and not 500 <= http_status <= 599):
                        break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if attempt <= retries:
                    time.sleep(min(2 ** (attempt - 1), 30))
            if http_status == 404 and request_id in _OFFICIAL_MIGRATED_REPAIRS:
                try:
                    effective_source_url = _OFFICIAL_MIGRATED_REPAIRS[request_id]
                    http_status, body, _ = fetch(effective_source_url)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    http_status = 422

            raw_path = root / "raw" / f"{item['yyyymm']}-ose.pdf"
            normalized_path = root / "normalized" / f"month={str(item['observation_month'])}.parquet"
            terminal = "failed"
            rows = 0
            duplicates = 0
            layout = ""
            digest = ""
            if http_status == 200 and body.startswith(b"%PDF"):
                _atomic_write_bytes(raw_path, body)
                digest = _sha256(raw_path)
                try:
                    frame = parse_ose_monthly_quotation_text(
                        extract_text(body),
                        observation_month=str(item["observation_month"]),
                        known_at_utc=str(item["known_at_utc"]),
                        source_url=effective_source_url,
                        source_sha256=digest,
                    )
                    rows = len(frame)
                    duplicates = int(frame.duplicated(["observation_month", "code"]).sum())
                    layout = str(frame["parser_layout"].iloc[0]) if rows else ""
                    if rows >= 500 and duplicates == 0:
                        _atomic_write_parquet(normalized_path, frame)
                        terminal = "completed"
                    else:
                        error = "normalized_quality_failed"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            elif http_status == 200:
                error = "response_is_not_pdf"
            elif not error:
                error = f"http_{http_status}"

            status_rows = [r for r in status_rows if r.get("request_id") != request_id]
            status_rows.append(
                {
                    "request_id": request_id,
                    "status": terminal,
                    "http_status": http_status,
                    "attempts": attempts,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "raw_path": str(raw_path.resolve()) if raw_path.is_file() else "",
                    "normalized_path": str(normalized_path.resolve()) if normalized_path.is_file() else "",
                    "normalized_sha256": _sha256(normalized_path) if normalized_path.is_file() else "",
                    "sha256": digest,
                    "source_url": effective_source_url,
                    "error": error,
                }
            )
            quality_rows = [r for r in quality_rows if r.get("request_id") != request_id]
            quality_rows.append(
                {
                    "request_id": request_id,
                    "normalized_rows": rows,
                    "duplicate_rows": duplicates,
                    "parser_layout": layout,
                    "quality_pass": terminal == "completed",
                }
            )
            _atomic_write_csv(status_path, pd.DataFrame(status_rows, columns=_status_columns()))
            _atomic_write_csv(quality_path, pd.DataFrame(quality_rows, columns=_quality_columns()))
            if pacing_seconds:
                time.sleep(pacing_seconds)

        status = _read_csv(status_path, _status_columns())
        quality = _read_csv(quality_path, _quality_columns())
        completed_count = int(status["status"].eq("completed").sum())
        normalized_rows = int(pd.to_numeric(quality["normalized_rows"], errors="coerce").fillna(0).sum())
        complete = completed_count == len(plan) and bool(quality["quality_pass"].astype(str).str.lower().eq("true").all())
        manifest = {
            "run_id": run_id,
            "artifact_type": "official_jasdaq_then_ose_combined_monthly_cash_equity_membership_source",
            "status": "complete_candidate_source" if complete else "in_progress",
            "scorecard_eligible": False,
            "evidence_scope": "full_premerger_ose_monthly" if complete else "bounded_partial",
            "publisher": "JASDAQ Securities Exchange / Osaka Securities Exchange / Japan Exchange Group",
            "planned_files": len(plan),
            "completed_files": completed_count,
            "normalized_rows": normalized_rows,
            "source_start_month": str(plan["observation_month"].min()),
            "source_end_month": str(plan["observation_month"].max()),
            "request_plan_sha256": _sha256(plan_path),
            "artifacts": {
                "request_plan": str(plan_path.resolve()),
                "file_status": str(status_path.resolve()),
                "file_quality": str(quality_path.resolve()),
            },
            "promotion_policy": {"alpha": False, "production": False, "ibkr": False, "live_trading": False},
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return OSEMonthlyQuotationArtifacts(
            run_id, root, plan_path, status_path, quality_path, manifest_path, manifest
        )
    finally:
        lock.rmdir()


def _repair_legacy_text(value: str) -> str:
    try:
        return value.encode("latin1").decode("cp932")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _fetch(url: str) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _extract_pdf_text(payload: bytes) -> str:
    reader = PdfReader(io.BytesIO(payload))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _status_columns() -> list[str]:
    return ["request_id", "status", "http_status", "attempts", "observed_at_utc", "raw_path", "normalized_path", "normalized_sha256", "sha256", "source_url", "error"]


def _quality_columns() -> list[str]:
    return ["request_id", "normalized_rows", "duplicate_rows", "parser_layout", "quality_pass"]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype=str).fillna("").reindex(columns=columns, fill_value="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        handle.write(payload)
        temp = Path(handle.name)
    os.replace(temp, path)
