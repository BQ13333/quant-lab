from __future__ import annotations

import hashlib
from quant_lab.data.ose_monthly_quotations import (
    build_ose_monthly_quotation_plan,
    ose_monthly_quotation_url,
    parse_ose_monthly_quotation_text,
    run_ose_monthly_quotations,
)


def test_ose_url_switch_and_plan_bounds() -> None:
    assert ose_monthly_quotation_url("2010-09").endswith("/KM201009.pdf")
    assert ose_monthly_quotation_url("2010-10").endswith("/k1010.pdf")
    plan = build_ose_monthly_quotation_plan(start_month="2013-05", end_month="2013-06")
    assert plan["request_id"].tolist() == ["201305-ose", "201306-ose"]
    assert plan["known_at_utc"].str.endswith("+00:00").all()


def test_legacy_and_modern_parsers() -> None:
    digest = hashlib.sha256(b"source").hexdigest()
    legacy = parse_ose_monthly_quotation_text(
        "100 5 1376 Company 650 1/4\n1,000 3 1380 Other 600 1/5\n",
        observation_month="2007-12",
        known_at_utc="2008-01-20T15:00:00+00:00",
        source_url="https://example.test/legacy.pdf",
        source_sha256=digest,
    )
    assert legacy["code"].tolist() == ["1376", "1380"]
    assert set(legacy["parser_layout"]) == {"legacy_stock_quotation_row"}

    modern_text = "\n".join(f"{1000 + index} Name {index} 普通株式" for index in range(500))
    modern = parse_ose_monthly_quotation_text(
        modern_text,
        observation_month="2012-01",
        known_at_utc="2012-02-20T15:00:00+00:00",
        source_url="https://example.test/modern.pdf",
        source_sha256=digest,
    )
    assert len(modern) == 500
    assert set(modern["parser_layout"]) == {"ordinary_share_label"}


def test_resumable_runner_requires_month_denominator(tmp_path) -> None:
    rows = "\n".join(f"100 5 {1000 + index} Company{index} 650 1/4" for index in range(500))

    def fetcher(url: str):
        return 200, b"%PDF fake", {}

    result = run_ose_monthly_quotations(
        run_id="OSE",
        artifact_root=tmp_path,
        start_month="2013-06",
        end_month="2013-06",
        max_files=1,
        fetcher=fetcher,
        pdf_text_extractor=lambda _: rows,
        pacing_seconds=0,
    )
    assert result.manifest["status"] == "complete_candidate_source"
    assert result.manifest["completed_files"] == 1


def test_runner_uses_official_migrated_repair(tmp_path) -> None:
    rows = "\n".join(f"100 5 {1000 + index} Company{index} 650 1/4" for index in range(500))
    calls = []

    def fetcher(url: str):
        calls.append(url)
        return (200, b"%PDF fake", {}) if url.endswith("KM200906t.pdf") else (404, b"", {})

    result = run_ose_monthly_quotations(
        run_id="REPAIR",
        artifact_root=tmp_path,
        start_month="2009-06",
        end_month="2009-06",
        max_files=1,
        fetcher=fetcher,
        pdf_text_extractor=lambda _: rows,
        pacing_seconds=0,
    )
    assert result.manifest["status"] == "complete_candidate_source"
    assert calls[-1].endswith("KM200906t.pdf")
