from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


UTC = timezone.utc
JST = timezone(pd.Timedelta(hours=9))
BASE = "https://www.jpx.co.jp"
USER_AGENT = "quant-lab-jpx-monthly-quotations/1.0"
EARLY_TEMPLATE = (
    BASE
    + "/files/tse/english/market/data/price/{year}/{yyyymm}/st{band}_{yyyymm}.pdf"
)
_COMMON_SHARE = re.compile(
    r"(?P<code>[1-9][0-9A-Z]{3})\s*(?P<name>[^\n]{1,100}?)\s*普通株(?:式)?"
)
_COMBINED_COMMON_SHARE = re.compile(
    r"(?m)^(?:19|20)\d{2}/\d{2}\s+(?P<code>[1-9][0-9A-Z]{3})\s*(?P<name>[^\n]{1,100}?)\s*普通株(?:式)?"
)
_2015_HASHES = {
    1: "b5b4pj00000033an",
    2: "b5b4pj000000apmi",
    3: "b5b4pj000000i1i1",
    4: "b5b4pj000000k2ej",
    5: "b5b4pj000000lfcl",
    6: "b5b4pj000000n04m",
    7: "b5b4pj000000omxg",
    8: "b5b4pj000000pw0s",
    9: "b5b4pj000000r0f0",
    10: "b5b4pj000000segp",
    11: "b5b4pj000000u0gt",
    12: "b5b4pj000000v7jm",
}


Fetcher = Callable[[str], tuple[int, bytes, Mapping[str, str]]]
PdfTextExtractor = Callable[[bytes], str]


@dataclass(frozen=True)
class JPXMonthlyQuotationArtifacts:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    status_path: Path
    quality_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def quotation_archive_page(year: int, current_year: int = 2026) -> str:
    if year > current_year or year < current_year - 10:
        raise ValueError("JPX public archive selector covers the current year and ten prior years")
    if year == current_year:
        return f"{BASE}/english/markets/statistics-equities/price/index.html"
    offset = current_year - year
    return f"{BASE}/english/markets/statistics-equities/price/00-archives-{offset:02d}.html"


def discover_quotation_urls(
    *,
    start_year: int = 2016,
    end_year: int = 2026,
    fetcher: Fetcher | None = None,
) -> dict[tuple[str, int], str]:
    if end_year < start_year:
        raise ValueError("end_year must not precede start_year")
    fetch = fetcher or _fetch
    urls: dict[tuple[str, int], str] = {}
    for year in range(start_year, end_year + 1):
        page_url = quotation_archive_page(year, end_year)
        status, body, _ = fetch(page_url)
        if status != 200:
            raise RuntimeError(f"JPX quotation archive discovery failed for {year}: HTTP {status}")
        soup = BeautifulSoup(body, "html.parser")
        for link in soup.find_all("a", href=True):
            href = str(link["href"]).strip()
            banded = re.search(r"/st([1-9])_((?:19|20)\d{4})\.pdf(?:\?.*)?$", href)
            combined = re.search(r"/st_((?:19|20)\d{4})\.pdf(?:\?.*)?$", href)
            if not banded and not combined:
                continue
            band = int(banded.group(1)) if banded else 0
            yyyymm = banded.group(2) if banded else combined.group(1)
            if int(yyyymm[:4]) == year:
                urls[(yyyymm, band)] = urllib.parse.urljoin(page_url, href)
    return urls


def build_monthly_quotation_plan(
    *,
    start_month: str = "2007-12",
    end_month: str = "2026-06",
    discovered_urls: Mapping[tuple[str, int], str] | None = None,
) -> pd.DataFrame:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must not precede start_month")
    discovered = dict(discovered_urls or {})
    rows: list[dict[str, Any]] = []
    for month in pd.period_range(start, end, freq="M"):
        yyyymm = month.strftime("%Y%m")
        known_at = conservative_quotation_known_at(month)
        bands = [0] if month.year >= 2020 else list(range(1, 10))
        for band in bands:
            if month.year <= 2014:
                url = EARLY_TEMPLATE.format(year=month.year, yyyymm=yyyymm, band=band)
                discovery_method = "official_tse_deterministic_archive_path"
            elif month.year == 2015:
                token = _2015_HASHES[month.month]
                url = (
                    f"{BASE}/english/markets/statistics-equities/price/"
                    f"{token}-att/st{band}_{yyyymm}.pdf"
                )
                discovery_method = "official_jpx_migrated_archive_path"
            else:
                url = discovered.get((yyyymm, band), "")
                discovery_method = "official_jpx_archive_page_discovery"
            band_label: int | str = "all" if band == 0 else band
            rows.append(
                {
                    "request_id": f"{yyyymm}-st{band if band else ''}",
                    "observation_month": str(month),
                    "yyyymm": yyyymm,
                    "year": month.year,
                    "month": month.month,
                    "code_band": band_label,
                    "membership_effective_month_end": month.end_time.date().isoformat(),
                    "known_at_utc": known_at,
                    "known_at_policy": "official_7th_business_day_schedule_lagged_to_21st_calendar_day_00_jst",
                    "source_url": url,
                    "discovery_method": discovery_method,
                    "source_official_jpx": True,
                }
            )
    return pd.DataFrame(rows)


def conservative_quotation_known_at(month: pd.Period | str) -> str:
    period = month if isinstance(month, pd.Period) else pd.Period(month, freq="M")
    next_month = period + 1
    local = datetime(next_month.year, next_month.month, 21, tzinfo=JST)
    return local.astimezone(UTC).isoformat()


def parse_monthly_quotation_text(
    text: str,
    *,
    observation_month: str,
    band: int | str,
    known_at_utc: str,
    source_url: str,
    source_sha256: str,
) -> pd.DataFrame:
    rows = []
    all_codes = str(band).lower() == "all" or int(band) == 0
    pattern = _COMBINED_COMMON_SHARE if all_codes else _COMMON_SHARE
    for match in pattern.finditer(text):
        code = match.group("code").upper()
        if not all_codes and int(code[0]) != int(band):
            continue
        rows.append(
            {
                "observation_month": observation_month,
                "membership_effective_month_end": pd.Period(observation_month, freq="M").end_time.date().isoformat(),
                "known_at_utc": known_at_utc,
                "known_at_policy": "official_7th_business_day_schedule_lagged_to_21st_calendar_day_00_jst",
                "code": code,
                "issue_name_japanese": " ".join(match.group("name").split()),
                "code_band": "all" if all_codes else int(band),
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
        "code_band",
        "currency",
        "membership_observed",
        "pit_usable",
        "source_url",
        "source_sha256",
    ]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(["observation_month", "code"])


def run_jpx_monthly_quotations(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/jpx_monthly_quotations",
    start_month: str = "2007-12",
    end_month: str = "2026-06",
    max_files: int = 9,
    pacing_seconds: float = 0.2,
    retries: int = 3,
    fetcher: Fetcher | None = None,
    pdf_text_extractor: PdfTextExtractor | None = None,
) -> JPXMonthlyQuotationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    if max_files <= 0 or retries < 0 or pacing_seconds < 0:
        raise ValueError("bounds must be non-negative and max_files positive")
    fetch = fetcher or _fetch
    extract_text = pdf_text_extractor or _extract_pdf_text
    root = Path(artifact_root) / run_id
    raw_root = root / "raw"
    normalized_root = root / "normalized"
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "request_plan.csv"
    plan_hash_path = root / "request_plan.sha256"
    status_path = root / "file_status.csv"
    quality_path = root / "file_quality.csv"
    manifest_path = root / "manifest.json"
    lock_path = root / ".writer.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock_path}") from exc

    try:
        if plan_path.is_file():
            _verify_staged_plan(plan_path, plan_hash_path)
            plan = pd.read_csv(plan_path, dtype=str).fillna("")
        else:
            end = pd.Period(end_month, freq="M")
            discovered: dict[tuple[str, int], str] = {}
            if end.year >= 2016:
                discovered = discover_quotation_urls(
                    start_year=2016,
                    end_year=end.year,
                    fetcher=fetch,
                )
            plan = build_monthly_quotation_plan(
                start_month=start_month,
                end_month=end_month,
                discovered_urls=discovered,
            )
            _atomic_write_csv(plan_path, plan)
            _atomic_write_text(plan_hash_path, _sha256(plan_path) + "\n")

        status = _read_csv(status_path, _status_columns())
        status, hashes_changed = _backfill_normalized_hashes(status)
        if hashes_changed:
            _atomic_write_csv(status_path, status)
        quality = _read_csv(quality_path, _quality_columns())
        status, quality, invalidated = _invalidate_low_denominator_combined(plan, status, quality)
        if invalidated:
            _atomic_write_csv(status_path, status)
            _atomic_write_csv(quality_path, quality)
        fallback_path = root / "fallback_routes.csv"
        fallback = _read_csv(
            fallback_path,
            ["request_id", "archival_url", "original_publisher_url", "capture_timestamp", "expected_sha256", "route_type", "status"],
        )
        fallback_map = {
            str(row["request_id"]): row
            for row in fallback.loc[fallback["status"].eq("validated")].to_dict("records")
        }
        completed_ids = set(status.loc[status["status"].eq("completed"), "request_id"])
        prior_status = status.drop_duplicates("request_id", keep="last").set_index("request_id")["status"].to_dict()
        prior_rows = {
            str(row["request_id"]): row
            for row in status.drop_duplicates("request_id", keep="last").to_dict("records")
        }
        pending = plan.loc[~plan["request_id"].isin(completed_ids)].copy()
        pending["_prior_status"] = pending["request_id"].map(prior_status).fillna("")
        pending["_retry_priority"] = pending["_prior_status"].ne("").astype(int)
        pending = pending.sort_values(["_retry_priority", "year", "month", "code_band"]).head(max_files)
        status_rows = status.to_dict("records")
        quality_rows = quality.to_dict("records")
        for item in pending.to_dict("records"):
            request_id = str(item["request_id"])
            source_url = str(item["source_url"])
            effective_source_url = source_url
            source_route = "official_jpx_direct"
            observed_at = datetime.now(UTC).isoformat()
            http_status = 0
            body = b""
            error = ""
            attempts = 0
            terminal = "failed"
            rows = 0
            duplicates = 0
            range_violations = 0
            digest = ""
            raw_path = raw_root / str(item["year"]) / f"{item['yyyymm']}-st{item['code_band']}.pdf"
            band_value = str(item["code_band"])
            normalized_path = (
                normalized_root
                / f"year={item['year']}"
                / f"month={int(item['month']):02d}"
                / f"band={band_value}.parquet"
            )
            if not source_url:
                error = "missing_official_source_url"
            else:
                prior = prior_rows.get(request_id, {})
                cached_raw = Path(str(prior.get("raw_path", "")))
                cached_sha = str(prior.get("sha256", ""))
                if (
                    str(prior.get("error", "")) == "combined_parser_denominator_below_1000"
                    and cached_raw.is_file()
                    and len(cached_sha) == 64
                    and _sha256(cached_raw) == cached_sha
                ):
                    body = cached_raw.read_bytes()
                    http_status = 200
                    effective_source_url = str(prior.get("effective_source_url", "")) or source_url
                    source_route = "cached_raw_reparse_after_parser_hardening"
                else:
                    for attempt in range(1, retries + 2):
                        attempts = attempt
                        try:
                            http_status, body, _ = fetch(source_url)
                            if http_status == 200 or (http_status != 429 and not 500 <= http_status <= 599):
                                break
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                        if attempt <= retries:
                            time.sleep(min(2 ** (attempt - 1), 30))
                fallback_row = fallback_map.get(request_id)
                if http_status == 404 and fallback_row:
                    effective_source_url = str(fallback_row["archival_url"])
                    source_route = str(fallback_row.get("route_type", "")) or "internet_archive_mirror_of_official_tse_pdf"
                    try:
                        http_status, body, _ = fetch(effective_source_url)
                        expected_fallback_sha = str(fallback_row["expected_sha256"])
                        if http_status == 200 and expected_fallback_sha and hashlib.sha256(body).hexdigest() != expected_fallback_sha:
                            error = "archival_fallback_sha256_mismatch"
                            http_status = 422
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
            if http_status == 200 and body.startswith(b"%PDF"):
                _atomic_write_bytes(raw_path, body)
                digest = _sha256(raw_path)
                try:
                    frame = parse_monthly_quotation_text(
                        extract_text(body),
                        observation_month=str(item["observation_month"]),
                        band=band_value,
                        known_at_utc=str(item["known_at_utc"]),
                        source_url=effective_source_url,
                        source_sha256=digest,
                    )
                    rows = len(frame)
                    duplicates = int(frame.duplicated(["observation_month", "code"]).sum())
                    range_violations = (
                        0
                        if band_value == "all" or not rows
                        else int((frame["code"].str[0].astype(int) != int(band_value)).sum())
                    )
                    minimum_rows = 1000 if band_value == "all" else 1
                    if rows >= minimum_rows and duplicates == 0 and range_violations == 0:
                        _atomic_write_parquet(normalized_path, frame)
                        terminal = "completed"
                    else:
                        error = "normalized_quality_failed"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            elif http_status == 200:
                error = "response_is_not_pdf"
            status_rows = [row for row in status_rows if row.get("request_id") != request_id]
            status_rows.append(
                {
                    "request_id": request_id,
                    "status": terminal,
                    "http_status": http_status,
                    "attempts": attempts,
                    "observed_at_utc": observed_at,
                    "raw_path": str(raw_path.resolve()) if raw_path.is_file() else "",
                    "normalized_path": str(normalized_path.resolve()) if normalized_path.is_file() else "",
                    "normalized_sha256": _sha256(normalized_path) if normalized_path.is_file() else "",
                    "sha256": digest,
                    "effective_source_url": effective_source_url,
                    "source_route": source_route,
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
        planned = len(plan)
        complete = int(status["status"].eq("completed").sum())
        missing_urls = int(plan["source_url"].astype(str).str.strip().eq("").sum())
        full_complete = planned > 0 and complete == planned and missing_urls == 0
        manifest = {
            "run_id": run_id,
            "artifact_type": "official_jpx_monthly_quotation_pit_membership_source",
            "status": "complete_candidate_source" if full_complete else "in_progress",
            "official_jpx_source": True,
            "source_schedule": "approximately 09:00 JST on the seventh business day of the following month",
            "known_at_policy": "lagged conservatively to the 21st calendar day of the following month at 00:00 JST",
            "evidence_scope": "full_monthly_snapshots" if full_complete else "bounded_partial",
            "scorecard_eligible": False,
            "historical_pit_candidate": full_complete,
            "independent_validation_required": True,
            "plan_immutable": True,
            "request_plan_sha256": _sha256(plan_path),
            "planned_files": planned,
            "completed_files": complete,
            "failed_files": int(status["status"].eq("failed").sum()),
            "missing_source_urls": missing_urls,
            "validated_fallback_routes": len(fallback_map),
            "fallback_routes_sha256": _sha256(fallback_path) if fallback_path.is_file() else "",
            "normalized_rows": int(pd.to_numeric(quality["normalized_rows"], errors="coerce").fillna(0).sum()),
            "coverage": {
                "start_month": start_month,
                "end_month": end_month,
                "pre_2020_code_bands": list(range(1, 10)),
                "post_2020_combined_monthly_file": True,
            },
            "promotion_policy": {"alpha": False, "production": False, "ibkr": False, "live_trading": False},
            "artifacts": {
                "request_plan": str(plan_path.resolve()),
                "file_status": str(status_path.resolve()),
                "file_quality": str(quality_path.resolve()),
                "raw_root": str(raw_root.resolve()),
                "normalized_root": str(normalized_root.resolve()),
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(manifest_path, manifest)
        return JPXMonthlyQuotationArtifacts(
            run_id=run_id,
            artifact_dir=root,
            plan_path=plan_path,
            status_path=status_path,
            quality_path=quality_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    finally:
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def _extract_pdf_text(body: bytes) -> str:
    reader = PdfReader(io.BytesIO(body))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _fetch(url: str) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _verify_staged_plan(plan_path: Path, hash_path: Path) -> None:
    if not hash_path.is_file():
        raise ValueError("immutable request plan checksum is missing")
    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    if len(expected) != 64 or _sha256(plan_path) != expected:
        raise ValueError("immutable request plan checksum mismatch")


def _status_columns() -> list[str]:
    return [
        "request_id",
        "status",
        "http_status",
        "attempts",
        "observed_at_utc",
        "raw_path",
        "normalized_path",
        "normalized_sha256",
        "sha256",
        "effective_source_url",
        "source_route",
        "error",
    ]


def _quality_columns() -> list[str]:
    return ["request_id", "normalized_rows", "duplicate_rows", "code_range_violations", "quality_pass"]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame[columns]


def _backfill_normalized_hashes(status: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = status.copy()
    changed = False
    for index, row in result.loc[result["status"].eq("completed")].iterrows():
        path = Path(str(row["normalized_path"]))
        if not path.is_file():
            continue
        digest = _sha256(path)
        if str(row["normalized_sha256"]) != digest:
            result.at[index, "normalized_sha256"] = digest
            changed = True
    return result, changed


def _invalidate_low_denominator_combined(
    plan: pd.DataFrame,
    status: pd.DataFrame,
    quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    result_status = status.copy()
    result_quality = quality.copy()
    combined_ids = set(plan.loc[plan["code_band"].astype(str).eq("all"), "request_id"])
    low_ids = set(
        result_quality.loc[
            result_quality["request_id"].isin(combined_ids)
            & pd.to_numeric(result_quality["normalized_rows"], errors="coerce").fillna(0).lt(1000),
            "request_id",
        ]
    )
    mask = result_status["request_id"].isin(low_ids) & result_status["status"].eq("completed")
    result_status.loc[mask, "status"] = "failed"
    result_status.loc[mask, "error"] = "combined_parser_denominator_below_1000"
    result_quality.loc[result_quality["request_id"].isin(low_ids), "quality_pass"] = False
    return result_status, result_quality, int(mask.sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


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
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
