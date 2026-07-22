from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
CDX_API = "https://web.archive.org/cdx/search/cdx"
Fetcher = Callable[[str], tuple[int, bytes, Mapping[str, str]]]


def build_jpx_archival_fallbacks(
    *,
    source_dir: str | Path,
    max_routes: int = 50,
    pacing_seconds: float = 0.1,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    source = Path(source_dir)
    plan_path = source / "request_plan.csv"
    status_path = source / "file_status.csv"
    output_path = source / "fallback_routes.csv"
    manifest_path = source / "fallback_manifest.json"
    plan = pd.read_csv(plan_path, dtype=str).fillna("")
    status = pd.read_csv(status_path, dtype=str).fillna("")
    failed = status.loc[status["status"].eq("failed") & status["http_status"].eq("404"), ["request_id"]]
    pending = failed.merge(plan[["request_id", "source_url"]], on="request_id", how="left").head(max_routes)
    existing = (
        pd.read_csv(output_path, dtype=str).fillna("")
        if output_path.is_file()
        else pd.DataFrame(columns=_columns())
    )
    for column in _columns():
        if column not in existing:
            existing[column] = ""
    existing = existing[_columns()]
    fetch = fetcher or _fetch
    rows = existing.to_dict("records")
    done = set(existing.loc[existing["status"].eq("validated"), "request_id"])
    page_cache: dict[str, tuple[str, str, bytes]] = {}
    for item in pending.loc[~pending["request_id"].isin(done)].to_dict("records"):
        request_id = str(item["request_id"])
        publisher_url = ""
        capture_timestamp = ""
        archival_url = ""
        digest = ""
        terminal = "failed"
        error = ""
        route_type = ""
        try:
            repair_url = _current_jpx_repair_url(request_id)
            if repair_url:
                pdf_status, pdf_body, _ = fetch(repair_url)
                if pdf_status == 200 and pdf_body.startswith(b"%PDF"):
                    publisher_url = repair_url
                    archival_url = repair_url
                    digest = hashlib.sha256(pdf_body).hexdigest()
                    terminal = "validated"
                    route_type = "official_jpx_current_migrated_repair"
                else:
                    raise ValueError(f"current_jpx_repair_http_{pdf_status}")
            else:
                route_type = "internet_archive_mirror_of_official_tse_pdf"
            yyyymm = request_id[:6]
            band = request_id.rsplit("st", 1)[-1]
            if not repair_url and yyyymm not in page_cache:
                page_url = f"http://www.tse.or.jp/english/market/data/price/{yyyymm}.html"
                cdx_url = CDX_API + "?" + urllib.parse.urlencode(
                    {
                        "url": page_url,
                        "output": "json",
                        "filter": "statuscode:200",
                        "fl": "timestamp,original",
                        "collapse": "digest",
                    }
                )
                cdx_status, cdx_body, _ = fetch(cdx_url)
                payload = json.loads(cdx_body.decode("utf-8")) if cdx_status == 200 else []
                captures = payload[1:] if isinstance(payload, list) and len(payload) > 1 else []
                if not captures:
                    raise ValueError("no_archival_month_page")
                page_timestamp, original_page = captures[0][:2]
                archived_page = f"https://web.archive.org/web/{page_timestamp}id_/{original_page}"
                page_status, page_body, _ = fetch(archived_page)
                if page_status != 200:
                    raise ValueError(f"archival_month_page_http_{page_status}")
                page_cache[yyyymm] = (page_timestamp, original_page, page_body)
            if not repair_url:
                capture_timestamp, original_page, page_body = page_cache[yyyymm]
                target_pattern = re.compile(rf"^st{re.escape(band)}_{yyyymm}(?:t\d*)?\.pdf$")
                soup = BeautifulSoup(page_body, "html.parser")
                matches = [
                    urllib.parse.urljoin(original_page, str(link["href"]))
                    for link in soup.find_all("a", href=True)
                    if target_pattern.fullmatch(str(link["href"]).split("/")[-1])
                ]
                if matches:
                    publisher_url = matches[0]
                    archival_url = f"https://web.archive.org/web/{capture_timestamp}id_/{publisher_url}"
                    pdf_status, pdf_body, _ = fetch(archival_url)
                    if pdf_status == 200 and pdf_body.startswith(b"%PDF"):
                        digest = hashlib.sha256(pdf_body).hexdigest()
                        terminal = "validated"
                    else:
                        error = f"archival_pdf_invalid_http_{pdf_status}"
                else:
                    error = "official_pdf_link_missing_from_archived_month_page"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows = [row for row in rows if row.get("request_id") != request_id]
        rows.append(
            {
                "request_id": request_id,
                "primary_url": str(item["source_url"]),
                "original_publisher_url": publisher_url,
                "capture_timestamp": capture_timestamp,
                "archival_url": archival_url,
                "expected_sha256": digest,
                "route_type": route_type,
                "status": terminal,
                "error": error,
            }
        )
        _atomic_write_csv(output_path, pd.DataFrame(rows, columns=_columns()))
        if pacing_seconds:
            time.sleep(pacing_seconds)
    result = pd.read_csv(output_path, dtype=str).fillna("") if output_path.is_file() else pd.DataFrame(columns=_columns())
    manifest = {
        "artifact_type": "jpx_official_pdf_archival_fallback_registry",
        "status": "complete_for_current_404s" if len(result.loc[result["status"].eq("validated")]) == len(failed) else "incomplete",
        "scorecard_eligible": False,
        "publisher": "Tokyo Stock Exchange",
        "transport": "Internet Archive archival mirror",
        "failed_404_routes": len(failed),
        "validated_routes": int(result["status"].eq("validated").sum()),
        "failed_routes": int(result["status"].eq("failed").sum()),
        "fallback_routes_sha256": _sha256(output_path) if output_path.is_file() else "",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _fetch(url: str) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-jpx-archive-fallback/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _columns() -> list[str]:
    return ["request_id", "primary_url", "original_publisher_url", "capture_timestamp", "archival_url", "expected_sha256", "route_type", "status", "error"]


def _current_jpx_repair_url(request_id: str) -> str:
    if request_id not in {"201312-st2", "201312-st3", "201312-st4", "201312-st9"}:
        return ""
    band = request_id.rsplit("st", 1)[-1]
    return (
        "https://www.jpx.co.jp/files/tse/market/data/price/"
        f"b7gje60000047579-att/st{band}_201312.pdf"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        handle.write(payload)
        temp = Path(handle.name)
    os.replace(temp, path)
