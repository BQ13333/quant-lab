from __future__ import annotations

import hashlib
import io
import json
import os
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

from quant_lab.data.jpx_monthly_quotations import (
    conservative_quotation_known_at,
    parse_monthly_quotation_text,
)


UTC = timezone.utc
BASE = "https://www.jpx.co.jp/files/ose/markets/data/equities/price"
USER_AGENT = "quant-lab-ose-main-monthly-quotations/1.0"
Fetcher = Callable[[str], tuple[int, bytes, Mapping[str, str]]]
PdfTextExtractor = Callable[[bytes], str]


@dataclass(frozen=True)
class OSEMainMonthlyQuotationArtifacts:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    status_path: Path
    quality_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def build_ose_main_monthly_quotation_plan(
    *, start_month: str = "2006-07", end_month: str = "2010-09"
) -> pd.DataFrame:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must not precede start_month")
    if end > pd.Period("2010-09", freq="M"):
        raise ValueError("separate OSE main-market files end at the October 2010 cash-market integration")
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        yyyymm = month.strftime("%Y%m")
        for band in range(1, 10):
            rows.append(
                {
                    "request_id": f"{yyyymm}-ose-main-{band}",
                    "observation_month": str(month),
                    "yyyymm": yyyymm,
                    "code_band": band,
                    "membership_effective_month_end": month.end_time.date().isoformat(),
                    "known_at_utc": conservative_quotation_known_at(month),
                    "known_at_policy": "official_monthly_publication_lagged_to_21st_calendar_day_00_jst",
                    "source_url": f"{BASE}/{band}000k{yyyymm[2:]}.pdf",
                    "source_official_jpx": True,
                    "source_exchange": "Osaka Securities Exchange main market",
                }
            )
    return pd.DataFrame(rows)


def run_ose_main_monthly_quotations(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/ose_main_monthly_quotations",
    start_month: str = "2006-07",
    end_month: str = "2010-09",
    max_files: int = 45,
    pacing_seconds: float = 0.05,
    retries: int = 3,
    fetcher: Fetcher | None = None,
    pdf_text_extractor: PdfTextExtractor | None = None,
) -> OSEMainMonthlyQuotationArtifacts:
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
                raise RuntimeError("immutable OSE main request plan hash mismatch")
            plan = pd.read_csv(plan_path, dtype=str).fillna("")
        else:
            plan = build_ose_main_monthly_quotation_plan(
                start_month=start_month, end_month=end_month
            )
            _atomic_write_csv(plan_path, plan)
            _atomic_write_text(plan_hash_path, _sha256(plan_path) + "\n")
        status = _read_csv(status_path, _status_columns())
        quality = _read_csv(quality_path, _quality_columns())
        completed = set(status.loc[status["status"].eq("completed"), "request_id"])
        previous = status.drop_duplicates("request_id", keep="last").set_index("request_id")
        prior_status = previous["status"].to_dict() if not previous.empty else {}
        pending = plan.loc[~plan["request_id"].isin(completed)].copy()
        pending["_retry_priority"] = pending["request_id"].map(prior_status).fillna("").ne("").astype(int)
        pending = pending.sort_values(["_retry_priority", "yyyymm", "code_band"]).head(max_files)
        status_rows = status.to_dict("records")
        quality_rows = quality.to_dict("records")
        for item in pending.to_dict("records"):
            request_id = str(item["request_id"])
            body = b""
            http_status = 0
            attempts = 0
            error = ""
            for attempt in range(1, retries + 2):
                attempts = attempt
                try:
                    http_status, body, _ = fetch(str(item["source_url"]))
                    if http_status == 200 or (http_status != 429 and not 500 <= http_status <= 599):
                        break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if attempt <= retries:
                    time.sleep(min(2 ** (attempt - 1), 30))
            raw_path = root / "raw" / str(item["yyyymm"]) / f"band-{item['code_band']}.pdf"
            normalized_path = root / "normalized" / f"month={item['observation_month']}" / f"band={item['code_band']}.parquet"
            terminal = "failed"
            rows = 0
            duplicates = 0
            range_violations = 0
            digest = ""
            if http_status == 200 and body.startswith(b"%PDF"):
                _atomic_write_bytes(raw_path, body)
                digest = _sha256(raw_path)
                try:
                    frame = parse_monthly_quotation_text(
                        extract_text(body),
                        observation_month=str(item["observation_month"]),
                        band=str(item["code_band"]),
                        known_at_utc=str(item["known_at_utc"]),
                        source_url=str(item["source_url"]),
                        source_sha256=digest,
                    )
                    frame["source_exchange"] = "Osaka Securities Exchange main market"
                    rows = len(frame)
                    duplicates = int(frame.duplicated(["observation_month", "code"]).sum())
                    range_violations = int((frame["code"].str[0] != str(item["code_band"])).sum()) if rows else 0
                    if rows >= 1 and duplicates == 0 and range_violations == 0:
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
            status_rows = [row for row in status_rows if row.get("request_id") != request_id]
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
                    "source_url": str(item["source_url"]),
                    "error": error,
                }
            )
            quality_rows = [row for row in quality_rows if row.get("request_id") != request_id]
            quality_rows.append(
                {
                    "request_id": request_id,
                    "normalized_rows": rows,
                    "duplicate_rows": duplicates,
                    "code_range_violations": range_violations,
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
            "artifact_type": "official_ose_main_market_monthly_cash_equity_membership_source",
            "status": "complete_candidate_source" if complete else "in_progress",
            "scorecard_eligible": False,
            "evidence_scope": "full_preintegration_ose_main_monthly" if complete else "bounded_partial",
            "publisher": "Osaka Securities Exchange / Japan Exchange Group",
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
        return OSEMainMonthlyQuotationArtifacts(
            run_id, root, plan_path, status_path, quality_path, manifest_path, manifest
        )
    finally:
        lock.rmdir()


def _fetch(url: str) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _extract_pdf_text(payload: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(payload)).pages)


def _status_columns() -> list[str]:
    return ["request_id", "status", "http_status", "attempts", "observed_at_utc", "raw_path", "normalized_path", "normalized_sha256", "sha256", "source_url", "error"]


def _quality_columns() -> list[str]:
    return ["request_id", "normalized_rows", "duplicate_rows", "code_range_violations", "quality_pass"]


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
