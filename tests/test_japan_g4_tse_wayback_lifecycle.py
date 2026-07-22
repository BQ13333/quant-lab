from __future__ import annotations

import json
import base64
import hashlib
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

import quant_lab.data.japan_g4_tse_wayback_lifecycle as lifecycle

from quant_lab.data.japan_g4_tse_wayback_lifecycle import (
    FetchResponse,
    _parse_retained_report_events,
    build_acquisition_plan,
    exact_lifecycle_pdf_url,
    run_japan_g4_tse_wayback_lifecycle,
    verify_wayback_lifecycle_artifact,
)


MONTH = "2012-01"
PAGE_URL = "http://www.tse.or.jp/english/market/data/geppo/201201.html"
PDF_URL = (
    "http://www.tse.or.jp/english/market/data/geppo/"
    "b7gje6000000vuvc-att/20_idou1201.pdf"
)
HTML_TIMESTAMP = "20120222082642"
PDF_TIMESTAMP = "20120404082407"
HTML = (
    b'<html><a href="/english/market/data/geppo/'
    b'b7gje6000000vuvc-att/20_idou1201.pdf">report</a>'
    b'<a href="/english/market/data/geppo/x-att/16_allse1201.pdf">other</a></html>'
)
PDF = b"%PDF-1.4 mocked retained bytes"
PAGES = [
    """20 Changes in Listed Companies and Issues, Etc. (Jan. 2012)
Delisting
1st Section 2012.01.17 2738 Example Corporation
Suspension of Trading
1st Section 2012.01.19 8809 Example Building Co.,Ltd. 16:00～16:25
21 Ex-New, Ex-Rights, Etc. (Jan. 2012)
2012.01.30 9999 Must Not Count
"""
]

COPIED_MONTH = "2015-01"
COPIED_PARENT_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
    "00-archives-10.html"
)
COPIED_PARENT_TIMESTAMP = "20250503084309"
COPIED_PDF_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/"
    "b5b4pj0000004mgs-att/17_idou1501.pdf"
)
COPIED_PARENT_HTML = (
    f'<html><a href="{COPIED_PDF_URL}">Jan. 2015 report</a></html>'.encode()
)
COPIED_PDF = b"%PDF-1.7 copied official JPX bytes"
COPIED_PAGES = [
    """20 Changes in Listed Companies and Issues, Etc. (Jan. 2015)
Delisting
1st Section 2015.01.05 1111 Copied Delisting Corporation
Suspension of Trading
1st Section 2015.01.06 2222 Copied Halt Corporation 10:00～10:15
JGB Futures 2015.01.07 Long-term JGB Futures All Contracts 11:00～11:10
21 Ex-New, Ex-Rights, Etc. (Jan. 2015)
"""
]

APRIL_2013_HALT_SECTION = [
    """20 Changes in Listed Companies and Issues, Etc. (Apr. 2013)
Suspension of Trading
市場第一部 2013.04.12 3606 レナウン RENOWN INCORPORATED 14:10～16:15
1st Section 2013.04.12 8255 原信ナルスホールディングス HARASHIN NARUS Holdings Co.,Ltd. 15:00～15:15
2013.04.15 7541 メガネトップ MEGANE TOP CO.,LTD. 12:00～12:25
2013.04.22 7003 三井造船 Mitsui Engineering & Shipbuilding Co.,Ltd. 8:20～9:35
2013.04.22 7012 川崎重工業 Kawasaki Heavy Industries,Ltd. 8:20～9:35
2013.04.23 7728 日本電産トーソク NIDEC TOSOK CORPORATION 15:20～15:35
2013.04.23 7756 日本電産コパル NIDEC COPAL CORPORATION 15:20～15:35
2013.04.26 5457 住友鋼管 Sumitomo Pipe & Tube Co.,Ltd. 15:30～15:45
2013.04.26 8002 丸紅 Marubeni Corporation 15:01～15:15
2013.04.26 9430 ＮＥＣモバイリング NEC Mobiling, Ltd. 15:01～15:25
2013.04.26 9810 日鐵商事 NIPPON STEEL TRADING CO.,LTD. 15:21～15:35
2013.04.26 9938 住金物産 Sumikin Bussan Corporation 15:21～15:35
市場第二部 2013.04.05 3644 １ｓｔホールディングス 1st Holdings,Inc. 16:00～16:25
2nd Section
有価証券オプション 2013.04.22 7012 川崎重工業 Kawasaki Heavy Industries,Ltd. 8:20～9:35
Individual Options 2013.04.26 8002 丸紅 Marubeni Corporation 15:01～15:15
Temporary Trading Halt (Circuit Breakers)
国債先物 2013.04.05 長期国債先物取引 全限月 13:08～13:18:44
JGB Futures
2013.04.05 ミニ長期国債先物取引 全限月 13:08～13:18:44
2013.04.05 長期国債先物取引 全限月 13:29～13:39:20
2013.04.05 ミニ長期国債先物取引 全限月 13:29～13:39:20
2013.04.08 長期国債先物取引 全限月 12:35～12:45
2013.04.08 ミニ長期国債先物取引 全限月 12:35～12:45
2013.04.10 長期国債先物取引 全限月 16:40～16:50
2013.04.10 ミニ長期国債先物取引 全限月 16:40～16:50
2013.04.12 長期国債先物取引 全限月 15:35～15:45
2013.04.12 ミニ長期国債先物取引 全限月 15:35～15:45
国債先物オプション 2013.04.05 国債先物オプション取引 全限月 13:08～13:18:44
Options on JGB Futures
2013.04.05 国債先物オプション取引 全限月 13:29～13:39:20
2013.04.08 国債先物オプション取引 全限月 12:35～12:45
2013.04.10 国債先物オプション取引 全限月 16:40～16:50
2013.04.12 国債先物オプション取引 全限月 15:35～15:45
21 Ex-New, Ex-Rights, Etc. (Apr. 2013)
"""
]

MAY_2013_HALT_SECTION = [
    """20 Changes in Listed Companies and Issues, Etc. (May 2013)
Suspension of Trading
市場第一部 2013.05.10 4208 宇部興産 Ube Industries,Ltd. 15:15～15:30
1st Section 2013.05.10 4711 東急コミュニティー TOKYU COMMUNITY CORP. 16:00～16:15
2013.05.10 8815 東急不動産 TOKYU LAND CORPORATION 16:00～16:15
2013.05.10 8879 東急リバブル TOKYU LIVABLE,INC. 16:00～16:15
2013.05.13 6773 パイオニア PIONEER CORPORATION 15:07～15:30
2013.05.17 9037 ハマキョウレックス HAMAKYOREX CO.,LTD. 15:01～15:15
市場第二部 2013.05.10 3010 価値開発 KACHIKAIHATSU CO.LTD. 16:55～17:30
2nd Section 2013.05.10 5390 宇部マテリアルズ Ube Material Industries,Ltd. 15:15～15:30
2013.05.15 7474 ジー・ネットワークス G.networks CO.,LTD. 17:00～17:15
2013.05.30 6443 東洋製作所 TOYO ENGINEERING WORKS,LTD. 15:00～15:25
マザーズ 2013.05.17 3726 フェヴリナホールディングス Favorina Holdings Co.,Ltd. 17:01～17:15
Mothers
Temporary Trading Halt (Circuit Breakers)
国債先物 2013.05.10 長期国債先物取引 全限月 12:39～12:49
JGB Futures
2013.05.10 ミニ長期国債先物取引 全限月 12:39～12:49
2013.05.13 長期国債先物取引 全限月 13:04～13:51
2013.05.13 ミニ長期国債先物取引 全限月 13:04～13:51
2013.05.23 長期国債先物取引 全限月 8:53～9:03
2013.05.23 ミニ長期国債先物取引 全限月 8:53～9:03
国債先物オプション 2013.05.10 国債先物オプション取引 全限月 12:39～12:49
Options on JGB Futures
2013.05.13 国債先物オプション取引 全限月 13:04～13:51
2013.05.23 国債先物オプション取引 全限月 8:53～9:03
21 Ex-New, Ex-Rights, Etc. (May 2013)
"""
]


def _cdx(timestamp: str, original: str, mimetype: str, digest: str) -> bytes:
    return json.dumps(
        [
            ["timestamp", "original", "statuscode", "digest", "length", "mimetype"],
            [timestamp, original, "200", digest, "123", mimetype],
        ]
    ).encode()


def fixture_transport(url: str) -> FetchResponse:
    if "/cdx/search/cdx" in url:
        original = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["url"][0]
        if original.endswith("201201.html"):
            body = _cdx(HTML_TIMESTAMP, PAGE_URL, "text/html", "HTMLDIGEST")
        elif original.endswith("20_idou1201.pdf"):
            body = _cdx(PDF_TIMESTAMP, PDF_URL, "application/pdf", "PDFDIGEST")
        else:
            body = json.dumps([["timestamp", "original", "statuscode", "digest", "length", "mimetype"]]).encode()
        return FetchResponse(200, body, {"content-type": "application/json"}, url)
    if f"/{HTML_TIMESTAMP}id_/" in url:
        return FetchResponse(200, HTML, {"memento-datetime": "Wed, 22 Feb 2012 08:26:42 GMT"}, url)
    if f"/{PDF_TIMESTAMP}id_/" in url:
        return FetchResponse(200, PDF, {"memento-datetime": "Wed, 04 Apr 2012 08:24:07 GMT"}, url)
    raise AssertionError(f"unexpected URL: {url}")


def _cdx_digest(body: bytes) -> str:
    return base64.b32encode(hashlib.sha1(body).digest()).decode("ascii").rstrip("=")


def copied_fixture_transport(url: str) -> FetchResponse:
    if "/cdx/search/cdx" in url:
        original = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["url"][0]
        assert original == COPIED_PARENT_URL
        body = _cdx(
            COPIED_PARENT_TIMESTAMP,
            COPIED_PARENT_URL,
            "text/html",
            _cdx_digest(COPIED_PARENT_HTML),
        )
        return FetchResponse(200, body, {"content-type": "application/json"}, url)
    if f"/{COPIED_PARENT_TIMESTAMP}id_/" in url:
        return FetchResponse(
            200,
            COPIED_PARENT_HTML,
            {
                "content-type": "text/html",
                "content-length": str(len(COPIED_PARENT_HTML)),
                "memento-datetime": "Sat, 03 May 2025 08:43:09 GMT",
            },
            url,
        )
    if url == COPIED_PDF_URL:
        return FetchResponse(
            200,
            COPIED_PDF,
            {
                "content-type": "application/pdf",
                "content-length": str(len(COPIED_PDF)),
                "etag": '"copied-fixture"',
            },
            url,
        )
    raise AssertionError(f"unexpected copied-route URL: {url}")


def test_plan_encodes_bounded_priority_order() -> None:
    plan = build_acquisition_plan()
    assert len(plan) == 114
    assert plan.iloc[0]["report_month"] == "2012-01"
    assert plan.iloc[47]["report_month"] == "2015-12"
    assert plan.iloc[48]["report_month"] == "2006-07"
    assert plan.iloc[-1]["report_month"] == "2009-12"


def test_copied_registry_is_exactly_the_audited_16_month_scope() -> None:
    assert lifecycle.COPIED_FILE_ROUTE_ENABLED is True
    expected_tokens = {
        "2014-09": "b7gje60000026ao9-att/17_idou1409.pdf",
        "2014-10": "b7gje60000027r55-att/17_idou1410.pdf",
        "2014-11": "b7gje600000294jl-att/17_idou1411.pdf",
        "2014-12": "b7gje6000002aeyz-att/17_idou1412.pdf",
        "2015-01": "b5b4pj0000004mgs-att/17_idou1501.pdf",
        "2015-02": "b5b4pj000000fnkm-att/17_idou1502.pdf",
        "2015-03": "b5b4pj000000idmr-att/17_idou1503.pdf",
        "2015-04": "b5b4pj000000kmok-att/17_idou1504.pdf",
        "2015-05": "b5b4pj000000m7co-att/17_idou1505.pdf",
        "2015-06": "b5b4pj000000nd3i-att/17_idou1506.pdf",
        "2015-07": "b5b4pj000000oz0h-att/17_idou1507.pdf",
        "2015-08": "b5b4pj000000qc9y-att/17_idou1508.pdf",
        "2015-09": "b5b4pj000000rgwu-att/17_idou1509.pdf",
        "2015-10": "b5b4pj000000t1f1-att/17_idou1510.pdf",
        "2015-11": "b5b4pj000000u9pb-att/17_idou1511.pdf",
        "2015-12": "b5b4pj000000vj5c-att/17_idou1512.pdf",
    }
    assert set(lifecycle.OFFICIAL_JPX_COPIED_FILE_REGISTRY) == set(expected_tokens)
    for month, token in expected_tokens.items():
        route = lifecycle.OFFICIAL_JPX_COPIED_FILE_REGISTRY[month]
        assert route["official_pdf_url"].endswith(token)
        if month.startswith("2014"):
            assert route["archived_parent_lineage_strength"] == lifecycle.WEAKER_ARCHIVED_MONTH_LINEAGE
            assert route["parent_link_url"].endswith(month.replace("-", "") + ".html")
        else:
            assert route["archived_parent_lineage_strength"] == lifecycle.DIRECT_ARCHIVED_PDF_LINEAGE
            assert route["parent_link_url"] == route["official_pdf_url"]


def test_copied_route_preserves_live_bytes_headers_claims_and_non_equity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "COPIED_FILE_ROUTE_ENABLED", True)
    result = run_japan_g4_tse_wayback_lifecycle(
        run_id="COPIED-2015-01",
        months=[COPIED_MONTH],
        artifact_root=tmp_path / "producer",
        max_months=1,
        sleep_seconds=0,
        retries=0,
        transport=copied_fixture_transport,
        page_extractor=lambda body: COPIED_PAGES,
    )
    assert result.manifest["status"] == "bounded_acquisition_complete"
    assert result.manifest["route_modes"] == [lifecycle.COPIED_FILE_ROUTE]
    assert result.manifest["live_official_pdf_response_preserved"] is True
    assert result.manifest["exact_nonempty_claim_count"] == 2
    assert result.manifest["exact_event_count"] == 3
    inventory = pd.read_csv(result.artifact_dir / "source_inventory.csv", dtype=str)
    assert inventory.loc[0, "pdf_sha256"] == hashlib.sha256(COPIED_PDF).hexdigest()
    assert json.loads(inventory.loc[0, "pdf_http_headers"])["etag"] == '"copied-fixture"'
    assert inventory.loc[0, "archived_parent_lineage_strength"] == lifecycle.DIRECT_ARCHIVED_PDF_LINEAGE
    events = pd.read_csv(result.artifact_dir / "exact_events.csv", dtype=str)
    assert events["instrument_class"].eq("jgb_future").sum() == 1
    assert (result.artifact_dir / "raw/pdfs/2015-01.pdf").read_bytes() == COPIED_PDF
    assert verify_wayback_lifecycle_artifact(result.artifact_dir) == []


def test_exact_link_accepts_only_one_official_month_contract() -> None:
    assert exact_lifecycle_pdf_url(HTML, PAGE_URL, MONTH) == PDF_URL
    wrong = HTML + b'<a href="/english/market/data/geppo/x-att/20_idou1202.pdf">wrong month</a>'
    assert exact_lifecycle_pdf_url(wrong, PAGE_URL, MONTH) == PDF_URL
    duplicate = HTML + b'<a href="/english/market/data/geppo/y-att/17_idou1201.pdf">duplicate</a>'
    try:
        exact_lifecycle_pdf_url(duplicate, PAGE_URL, MONTH)
    except RuntimeError as exc:
        assert "found 2" in str(exc)
    else:
        raise AssertionError("ambiguous lifecycle links must fail closed")


def test_mocked_acquisition_publishes_only_nonempty_exact_claims(tmp_path: Path) -> None:
    result = run_japan_g4_tse_wayback_lifecycle(
        run_id="MOCK-2012-01",
        months=[MONTH],
        artifact_root=tmp_path / "producer",
        max_months=1,
        sleep_seconds=0,
        retries=0,
        transport=fixture_transport,
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "bounded_acquisition_complete"
    assert result.manifest["acquired_exact_pdf_months"] == [MONTH]
    assert result.manifest["exact_nonempty_claim_count"] == 2
    assert result.manifest["exact_event_count"] == 2
    assert result.manifest["zero_event_claims_published"] == 0
    assert result.manifest["refinitiv_used"] is False
    inventory = pd.read_csv(result.artifact_dir / "source_inventory.csv", dtype=str)
    assert inventory.loc[0, "official_pdf_url"] == PDF_URL
    assert inventory.loc[0, "html_cdx_timestamp"] == HTML_TIMESTAMP
    assert inventory.loc[0, "pdf_cdx_timestamp"] == PDF_TIMESTAMP
    assert inventory.loc[0, "pdf_magic"] == "%PDF-"
    claims = pd.read_csv(result.artifact_dir / "exact_nonempty_claims.csv")
    assert set(claims["event_count"]) == {1}
    assert not claims["zero_event_asserted"].astype(bool).any()
    assert verify_wayback_lifecycle_artifact(result.artifact_dir) == []


def test_missing_pdf_capture_is_quarantined_and_never_zero(tmp_path: Path) -> None:
    def missing_pdf(url: str) -> FetchResponse:
        response = fixture_transport(url)
        if "/cdx/search/cdx" in url:
            original = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["url"][0]
            if original.endswith("20_idou1201.pdf"):
                response = FetchResponse(
                    200,
                    json.dumps([["timestamp", "original", "statuscode", "digest", "length", "mimetype"]]).encode(),
                    {},
                    url,
                )
        return response

    result = run_japan_g4_tse_wayback_lifecycle(
        run_id="NO-PDF",
        months=[MONTH],
        artifact_root=tmp_path / "producer",
        max_months=1,
        sleep_seconds=0,
        retries=0,
        transport=missing_pdf,
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "bounded_acquisition_with_failures"
    assert result.manifest["exact_nonempty_claim_count"] == 0
    failures = pd.read_csv(result.artifact_dir / "failures_quarantine.csv")
    assert failures["quarantined"].astype(bool).all()
    assert not failures["zero_inferred"].astype(bool).any()
    completion = pd.read_csv(result.artifact_dir / "completion_plan.csv")
    rows = completion.loc[completion["report_month"].eq(MONTH)]
    assert rows["completion_status"].eq("capture_or_parse_failed").all()


def test_retained_parser_narrowly_adapts_all_derivatives_label() -> None:
    pages = [
        """20 Changes in Listed Companies and Issues, Etc. (Aug. 2012)
Suspension of Trading
1st Section 2012.08.03 3258 Example Equity 15:07～16:15
全派生商品 2012.08.07 全派生商品（ToSTNeT取引を除く） the Derivatives Market 9:22～10:54
the Derivatives Market
Individual Options 2012.08.09 6758 Example Option 15:00～15:15
21 Ex-New, Ex-Rights, Etc. (Aug. 2012)
"""
    ]
    rows = _parse_retained_report_events(
        pages,
        report_month="2012-08",
        channel="halt_suspension_resumption",
        source_url=PDF_URL,
        source_path="raw/pdfs/2012-08.pdf",
        source_sha256="a" * 64,
    )
    assert len(rows) == 3
    assert rows[1]["instrument_class"] == "all_derivatives"
    assert rows[1]["source_record_text"].startswith("全派生商品 2012.08.07")


def test_exact_2013_april_may_halt_sections_restore_event_denominator() -> None:
    april = _parse_retained_report_events(
        APRIL_2013_HALT_SECTION,
        report_month="2013-04",
        channel="halt_suspension_resumption",
        source_url=PDF_URL,
        source_path="raw/pdfs/2013-04.pdf",
        source_sha256="4" * 64,
    )
    may = _parse_retained_report_events(
        MAY_2013_HALT_SECTION,
        report_month="2013-05",
        channel="halt_suspension_resumption",
        source_url=PDF_URL,
        source_path="raw/pdfs/2013-05.pdf",
        source_sha256="5" * 64,
    )

    assert len(april) == 30
    assert len(may) == 20
    assert sum(row["instrument_class"] == "jgb_future" for row in april) == 10
    assert sum(row["instrument_class"] == "jgb_future_option" for row in april) == 5
    assert sum(row["instrument_class"] == "jgb_future" for row in may) == 6
    assert sum(row["instrument_class"] == "jgb_future_option" for row in may) == 3
    assert 230 + len(april) + len(may) == 280
    assert all(not row["known_at"] and not row["ric"] for row in april + may)
    assert all(not row["admissible_to_shared_halt_history"] for row in april + may)


def test_mixed_supported_and_unsupported_rows_are_preserved_and_quarantined() -> None:
    pages = [
        """20 Changes in Listed Companies and Issues, Etc. (Jun. 2013)
Suspension of Trading
市場第一部 2013.06.03 1001 Supported Equity 10:00～10:10
商品先物 2013.06.04 Unsupported Future 10:00～10:10
2013.06.05 Continued Unsupported Future 10:00～10:10
国債先物 2013.06.06 長期国債先物取引 全限月 10:00～10:10
市場第二部 2013.06.07 1002 Supported Equity Again 10:00～10:10
21 Ex-New, Ex-Rights, Etc. (Jun. 2013)
"""
    ]
    rows = _parse_retained_report_events(
        pages,
        report_month="2013-06",
        channel="halt_suspension_resumption",
        source_url=PDF_URL,
        source_path="raw/pdfs/2013-06.pdf",
        source_sha256="6" * 64,
    )

    assert len(rows) == 5
    assert [row["instrument_class"] for row in rows] == [
        "equity",
        "quarantined_unknown_instrument",
        "quarantined_unknown_instrument",
        "jgb_future",
        "equity",
    ]
    assert all(
        row["mapping_status"] == "quarantined_unrecognized_instrument_class"
        for row in rows[1:3]
    )
    assert all(not row["known_at"] and not row["ric"] for row in rows)
    assert all(not row["admissible_to_shared_halt_history"] for row in rows)


def test_negative_same_day_halt_interval_is_preserved_but_quarantined() -> None:
    pages = [
        """20 Changes in Listed Companies and Issues, Etc. (Jun. 2013)
Suspension of Trading
国債先物 2013.06.06 長期国債先物取引 全限月 16:00～15:59
21 Ex-New, Ex-Rights, Etc. (Jun. 2013)
"""
    ]
    rows = _parse_retained_report_events(
        pages,
        report_month="2013-06",
        channel="halt_suspension_resumption",
        source_url=PDF_URL,
        source_path="raw/pdfs/2013-06.pdf",
        source_sha256="7" * 64,
    )

    assert len(rows) == 1
    assert rows[0]["halt_start_printed"] == "16:00"
    assert rows[0]["halt_end_printed"] == "15:59"
    assert rows[0]["timestamp_status"] == "quarantined_negative_same_day_halt_interval"
    assert rows[0]["admissible_to_shared_halt_history"] is False
    assert rows[0]["known_at"] == ""


def test_interrupted_run_resumes_without_mutating_retained_raw_bytes(tmp_path: Path) -> None:
    calls = 0

    def interrupted(url: str) -> FetchResponse:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return fixture_transport(url)

    kwargs = {
        "run_id": "RESUME",
        "months": [MONTH],
        "artifact_root": tmp_path / "producer",
        "max_months": 1,
        "sleep_seconds": 0,
        "retries": 0,
        "page_extractor": lambda body: PAGES,
    }
    try:
        run_japan_g4_tse_wayback_lifecycle(transport=interrupted, **kwargs)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("fixture interruption was not raised")
    page_path = tmp_path / "producer" / "RESUME" / "raw" / "pages" / f"{MONTH}.html"
    retained = page_path.read_bytes()
    result = run_japan_g4_tse_wayback_lifecycle(transport=fixture_transport, **kwargs)
    assert result.manifest["status"] == "bounded_acquisition_complete"
    assert page_path.read_bytes() == retained == HTML
