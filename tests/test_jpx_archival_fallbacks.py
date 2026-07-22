from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.jpx_archival_fallbacks import build_jpx_archival_fallbacks


def test_fallback_registry_validates_archived_official_pdf(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    url = "https://www.jpx.co.jp/files/tse/english/market/data/price/2013/201301/st2_201301.pdf"
    pd.DataFrame([{"request_id": "201301-st2", "source_url": url}]).to_csv(
        source / "request_plan.csv", index=False
    )
    pd.DataFrame([{"request_id": "201301-st2", "status": "failed", "http_status": "404"}]).to_csv(
        source / "file_status.csv", index=False
    )

    def fetcher(request_url: str):
        if "cdx/search" in request_url:
            return 200, json.dumps([["timestamp", "original"], ["20130201", "http://www.tse.or.jp/english/market/data/price/201301.html"]]).encode(), {}
        if request_url.endswith("201301.html"):
            return 200, b'<a href="/english/market/data/price/hash-att/st2_201301.pdf">pdf</a>', {}
        return 200, b"%PDF archived official fixture", {}

    manifest = build_jpx_archival_fallbacks(
        source_dir=source, max_routes=1, pacing_seconds=0, fetcher=fetcher
    )
    routes = pd.read_csv(source / "fallback_routes.csv")
    assert manifest["status"] == "complete_for_current_404s"
    assert routes.iloc[0]["status"] == "validated"
    assert len(routes.iloc[0]["expected_sha256"]) == 64
