from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_pre2013_price_limit_archive import (
    validate_japan_pre2013_price_limit_archive,
    write_japan_pre2013_price_limit_archive,
)


def _page(detail_id: int) -> bytes:
    notice = detail_id in {637, 640}
    if detail_id == 637:
        title = "Expansion of daily price limits (preliminary notice) -Alpha-"
        body = (
            "The trading conditions of Alpha (Code: 1234) have fallen under the criteria. "
            "The upper daily price limit may be expanded on the next business day (August 6)."
        )
    elif detail_id == 640:
        title = "Expansion of daily price limits (Daily price limit will be expanded) -Alpha-"
        body = (
            "Alpha (Code: 1234) will expand only the upper daily price limit on the next "
            "business day (August 7). Upper daily price limit: 600 yen "
            "(Lower limit will remain 300 yen as per normal) Base price: 1,200 yen; "
            "Upper daily price limit: 1,800 yen; Lower daily price limit: 900 yen"
        )
    else:
        title = f"Ordinary market notice {detail_id}"
        body = "Not an exceptional price-limit notice."
    assert notice == ("daily price limits" in title)
    return f"""<!doctype html>
<html><head><title>{title} | Japan Exchange Group</title></head>
<body><div id="readArea"><h1><span class="date-news">Aug. 05, 2013 <em>TSE</em></span></h1>
<p class="component-text">{body}</p></div></body></html>""".encode()


def _japanese_page(detail_id: int) -> bytes:
    if detail_id == 826:
        title = "制限値幅の拡大について（拡大予告）-アルファ-"
        body = "アルファ 株式（コード 1234）の制限値幅を拡大する可能性があります。"
    elif detail_id == 830:
        title = "制限値幅の拡大について（制限値幅を拡大します）-アルファ-"
        body = "アルファ 株式（コード 1234）の制限値幅を拡大します。"
    else:
        title = f"通常のマーケットニュース {detail_id}"
        body = "制限値幅の臨時措置ではありません。"
    return f"""<!doctype html>
<html><head><meta charset="UTF-8"><title>{title} | 日本取引所グループ</title></head>
<body><div id="readArea"><h1><span class="date-news">2013/08/05 <em>東証</em></span></h1>
<p class="txt">{body}</p></div></body></html>""".encode()


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    if url.endswith("/english/news/news.js"):
        return 200, b"var start_year = 2022; var start_month = 1; var baseYear = parseInt(2026,10); var baseMonth = parseInt(7,10); if (36 < month_between) return '';", {}
    if url.endswith("news_search_info.json"):
        return 200, b'{"business":{"kind":{"x":{"2024/09/02":"1"}}}}', {}
    if url.endswith("news_search_info_static.json"):
        return 200, b'{"business":{"kind":{"x":{"2023/05/01":"1"}}}}', {}
    if url.endswith("/english/news/index.html") or url.endswith("/news/index.html"):
        return 200, b"<html><title>Market News</title></html>", {}
    if "news_ym_201309.json" in url:
        return 200, json.dumps({"items": [{"title": "Control", "url": "/english/news/detail/detail_674.html"}]}).encode(), {}
    if "static-archives-201309.html" in url:
        return 200, b'<select class="backnumber"><option selected>Sep. 2013</option></select><a href="/english/news/detail/detail_674.html">Control</a>', {}
    if "/news/detail/detail_868.html" in url:
        return 200, _japanese_page(868).replace(b"2013/08/05", b"2013/09/02"), {}
    if "/news/detail/detail_822.html" in url:
        return 404, b"official 404", {}
    if "/news/detail/detail_" in url and "/english/" not in url:
        detail_id = int(url.rsplit("detail_", 1)[1].split(".", 1)[0])
        return 200, _japanese_page(detail_id), {}
    if "/english/news/detail/detail_24" in url or "/english/news/detail/detail_239" in url:
        detail_id = int(url.rsplit("detail_", 1)[1].split(".", 1)[0])
        return (200, _page(detail_id), {}) if detail_id == 24102 else (404, b"official 404", {})
    if "detail_674.html" in url:
        return 200, _page(674).replace(b"Aug. 05, 2013", b"Sep. 02, 2013"), {}
    if "detail_633.html" in url:
        return 404, b"official 404", {}
    if "detail_" in url:
        detail_id = int(url.rsplit("detail_", 1)[1].split(".", 1)[0])
        return 200, _page(detail_id), {}
    return 404, b"official 404", {}


def test_producer_retains_numerator_but_never_invents_denominator(tmp_path: Path) -> None:
    result = write_japan_pre2013_price_limit_archive(
        run_id="TEST-PRE2013",
        artifact_root=tmp_path,
        local_artifact_root=tmp_path / "local",
        fetcher=_fetch,
        max_workers=3,
    )

    coverage = pd.read_csv(result.artifact_dir / "coverage_months.csv", dtype=str, keep_default_na=False)
    notices = pd.read_csv(result.artifact_dir / "migrated_notices.csv", dtype=str)
    completeness = json.loads((result.artifact_dir / "completeness.json").read_text())
    vendor = pd.read_csv(result.artifact_dir / "vendor_route_plan.csv", dtype=str)
    reconciliation = pd.read_csv(
        result.artifact_dir / "bilingual_notice_reconciliation.csv", dtype=str
    )
    completion_plan = pd.read_csv(result.artifact_dir / "completion_plan.csv", dtype=str)

    assert len(coverage) == 86
    assert coverage.iloc[0]["month"] == "2006-07"
    assert coverage.iloc[-1]["month"] == "2013-08"
    assert coverage["official_notice_denominator"].eq("").all()
    assert coverage["official_denominator_proven"].eq("False").all()
    assert len(notices) == 2
    assert completeness["notice_denominator"]["value"] is None
    assert completeness["notice_denominator"]["missing_notice_count"] is None
    assert completeness["exact_unresolved_denominator"]["monthly_partitions_unsatisfied"] == 86
    assert completeness["completion_claimed"] is False
    assert completeness["observed_public_numerator"]["bilingual_matched_notice_pages"] == 2
    assert reconciliation["reconciliation_status"].eq("matched").all()
    assert completion_plan["query_start"].eq("2006-07-01T00:00:00+09:00").all()
    assert completion_plan.loc[
        completion_plan["requirement_id"].eq("JPLA-COMP-004"), "execution_status"
    ].eq("producer_executed_independent_recheck_required").all()
    assert completion_plan.loc[
        ~completion_plan["requirement_id"].eq("JPLA-COMP-004"), "execution_status"
    ].eq("required_not_executed").all()
    assert vendor["live_call_performed"].eq("False").all()
    assert validate_japan_pre2013_price_limit_archive(result.artifact_dir)["valid"]


def test_validator_rejects_tampered_raw_source(tmp_path: Path) -> None:
    result = write_japan_pre2013_price_limit_archive(
        run_id="TEST-TAMPER",
        artifact_root=tmp_path,
        local_artifact_root=tmp_path / "local",
        fetcher=_fetch,
        max_workers=2,
    )
    attempts = pd.read_csv(result.artifact_dir / "source_attempts.csv", dtype=str)
    raw = result.artifact_dir / attempts.iloc[0]["raw_path"]
    raw.write_bytes(b"tampered")

    validation = validate_japan_pre2013_price_limit_archive(result.artifact_dir)
    assert not validation["valid"]
    assert any("hash" in issue for issue in validation["issues"])


def test_validator_rejects_fabricated_completion(tmp_path: Path) -> None:
    result = write_japan_pre2013_price_limit_archive(
        run_id="TEST-FABRICATED-PASS",
        artifact_root=tmp_path,
        local_artifact_root=tmp_path / "local",
        fetcher=_fetch,
        max_workers=2,
    )
    completeness_path = result.artifact_dir / "completeness.json"
    completeness = json.loads(completeness_path.read_text())
    completeness["completion_claimed"] = True
    completeness_path.write_text(json.dumps(completeness), encoding="utf-8")

    validation = validate_japan_pre2013_price_limit_archive(result.artifact_dir)
    assert not validation["valid"]
    assert "completeness_claim_must_be_false" in validation["issues"]
