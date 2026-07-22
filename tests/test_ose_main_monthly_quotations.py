from __future__ import annotations

from quant_lab.data.ose_main_monthly_quotations import (
    build_ose_main_monthly_quotation_plan,
    run_ose_main_monthly_quotations,
)


def test_plan_is_banded_and_stops_before_integration() -> None:
    plan = build_ose_main_monthly_quotation_plan(start_month="2010-09", end_month="2010-09")
    assert len(plan) == 9
    assert plan["source_url"].iloc[0].endswith("/1000k1009.pdf")
    assert plan["source_url"].iloc[-1].endswith("/9000k1009.pdf")


def test_bounded_runner_completes_all_bands(tmp_path) -> None:
    def fetcher(url: str):
        return 200, b"%PDF fake", {}

    def text(_: bytes) -> str:
        return "1234 Name 普通株式"

    result = run_ose_main_monthly_quotations(
        run_id="OSE-MAIN",
        artifact_root=tmp_path,
        start_month="2010-09",
        end_month="2010-09",
        max_files=9,
        pacing_seconds=0,
        fetcher=fetcher,
        pdf_text_extractor=text,
    )
    assert result.manifest["completed_files"] == 1
    assert result.manifest["status"] == "in_progress"
