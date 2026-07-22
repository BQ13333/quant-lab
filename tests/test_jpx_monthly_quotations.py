from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from quant_lab.data.jpx_monthly_quotations import (
    build_monthly_quotation_plan,
    conservative_quotation_known_at,
    parse_monthly_quotation_text,
    run_jpx_monthly_quotations,
)


def test_early_plan_is_complete_immutable_shape_and_conservatively_lagged() -> None:
    plan = build_monthly_quotation_plan(start_month="2007-12", end_month="2008-01")
    assert len(plan) == 18
    assert plan["source_url"].str.startswith("https://www.jpx.co.jp/files/tse/english/").all()
    assert plan["request_id"].is_unique
    assert plan.iloc[0]["known_at_utc"] == "2008-01-20T15:00:00+00:00"
    assert conservative_quotation_known_at("2008-01") == "2008-02-20T15:00:00+00:00"


def test_parser_keeps_only_common_shares_in_the_requested_code_band() -> None:
    text = """
水産・農林業1301 極洋 普通株式
Construction 130A Example Alpha 普通株
ETF 1302 Example ETF 受益証券
Retail 2301 Wrong Band 普通株式
"""
    result = parse_monthly_quotation_text(
        text,
        observation_month="2024-01",
        band=1,
        known_at_utc="2024-02-20T15:00:00+00:00",
        source_url="https://www.jpx.co.jp/example.pdf",
        source_sha256="a" * 64,
    )
    assert result["code"].tolist() == ["1301", "130A"]
    assert result["pit_usable"].all()
    assert result["currency"].eq("JPY").all()


def test_combined_post_2020_file_keeps_all_code_bands() -> None:
    discovered = {
        ("202001", 0): "https://www.jpx.co.jp/english/markets/statistics-equities/price/example/st_202001.pdf"
    }
    plan = build_monthly_quotation_plan(
        start_month="2020-01", end_month="2020-01", discovered_urls=discovered
    )
    assert len(plan) == 1
    assert plan.iloc[0]["code_band"] == "all"
    result = parse_monthly_quotation_text(
        "2020/01 1301 One 普通株式 Company One\n2020/01 9301 Nine 普通株式 Company Nine",
        observation_month="2020-01",
        band="all",
        known_at_utc="2020-02-20T15:00:00+00:00",
        source_url=discovered[("202001", 0)],
        source_sha256="b" * 64,
    )
    assert result["code"].tolist() == ["1301", "9301"]


def test_runner_is_bounded_resumable_and_partitioned(tmp_path: Path) -> None:
    def fetcher(url: str):
        return 200, b"%PDF fake fixture", {}

    def extractor(body: bytes) -> str:
        assert body.startswith(b"%PDF")
        band = calls.pop(0)
        return f"Industry {band}301 Fixture {band} 普通株式"

    calls = list(range(1, 10))
    first = run_jpx_monthly_quotations(
        run_id="TEST",
        artifact_root=tmp_path,
        start_month="2007-12",
        end_month="2007-12",
        max_files=2,
        pacing_seconds=0,
        retries=0,
        fetcher=fetcher,
        pdf_text_extractor=extractor,
    )
    assert first.manifest["completed_files"] == 2
    assert first.manifest["planned_files"] == 9
    assert first.manifest["scorecard_eligible"] is False
    assert len(list((first.artifact_dir / "normalized").rglob("*.parquet"))) == 2

    second = run_jpx_monthly_quotations(
        run_id="TEST",
        artifact_root=tmp_path,
        start_month="2007-12",
        end_month="2007-12",
        max_files=2,
        pacing_seconds=0,
        retries=0,
        fetcher=fetcher,
        pdf_text_extractor=extractor,
    )
    assert second.manifest["completed_files"] == 4
    assert len(list((second.artifact_dir / "normalized").rglob("*.parquet"))) == 4
    assert not (second.artifact_dir / ".writer.lock").exists()


def test_never_attempted_files_are_prioritized_over_failed_retries(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(url: str):
        calls.append(url)
        return 200, b"%PDF fixture", {}

    source = tmp_path / "TEST"
    source.mkdir()
    plan = build_monthly_quotation_plan(start_month="2007-12", end_month="2007-12")
    plan.to_csv(source / "request_plan.csv", index=False)
    (source / "request_plan.sha256").write_text(
        hashlib.sha256((source / "request_plan.csv").read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "request_id": "200712-st1",
                "status": "failed",
                "http_status": "404",
            }
        ]
    ).to_csv(source / "file_status.csv", index=False)

    run_jpx_monthly_quotations(
        run_id="TEST",
        artifact_root=tmp_path,
        start_month="2007-12",
        end_month="2007-12",
        max_files=1,
        pacing_seconds=0,
        retries=0,
        fetcher=fetcher,
        pdf_text_extractor=lambda body: "Industry 2301 Fixture 普通株式",
    )

    assert calls[0].endswith("st2_200712.pdf")
