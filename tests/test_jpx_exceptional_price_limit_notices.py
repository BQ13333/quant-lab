from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.jpx_exceptional_price_limit_notices import (
    ImmutableArtifactError,
    POST_2020_BLOCKER,
    PRE_2013_BLOCKER,
    build_notice_plan,
    classify_notice_title,
    main,
    parse_archived_notice_corpus,
    parse_notice_page,
    produce_jpx_exceptional_price_limits,
    sha256_file,
    validate_event,
)


FIXTURES = Path(__file__).parent / "fixtures" / "jpx_exceptional_price_limits"
ACTIVATE_URL = "https://www.jpx.co.jp/english/news/1030/20260630-01.html"
CORRECTION_URL = "https://www.jpx.co.jp/english/news/1030/20170328-01.html"


def test_title_classifier_uses_six_event_taxonomy() -> None:
    assert classify_notice_title("Broadening of Daily Price Limits (Preliminary Notice) : Alpha") == "preliminary"
    assert classify_notice_title("Daily Price Limits to be Broadened : Alpha") == "activate"
    assert classify_notice_title("Daily Price Limits Continue to be Broadened : Alpha") == "continue"
    assert classify_notice_title("Announced Broadening of Daily Price Limits to be Cancelled: Alpha") == "cancel"
    assert classify_notice_title("Daily Price Limits to be Returned to Normal : Alpha") == "return"
    assert classify_notice_title("Daily Price Limits to be Broadened : Alpha【Revised】") == "correction"
    assert classify_notice_title("Daily Price Limits to be Broadened (ETF/ETN) : Alpha") == ""
    assert classify_notice_title("Correction of Broadening of Daily Price Limits (Preliminary Notice) : Alpha") == ""


def test_parser_uses_read_area_br_lines_nfkc_and_one_row_per_security() -> None:
    rows = parse_notice_page(
        (FIXTURES / "activate_multi.html").read_bytes(),
        event_type="activate",
        archive_publication_date="2026-06-30",
        source_url=ACTIVATE_URL,
        title="Daily Price Limits to be Broadened : 2 issues",
    )
    assert [(row["security_code"], row["side"]) for row in rows] == [
        ("584A", "upper"),
        ("7815", "lower"),
    ]
    assert {row["effective_date"] for row in rows} == {"2026-07-01"}
    assert rows[0]["upper_limit_price_yen"] == 6_580
    assert rows[0]["observed_multiplier"] == 4.0
    quality = [validate_event(row, ["2026-06-30", "2026-07-01"]) for row in rows]
    assert all(row["validation_pass"] for row in quality)
    assert all(row["upper_tick_valid"] and row["lower_tick_valid"] for row in quality)


def test_strict_validation_flags_numeric_tick_and_effective_session_failures() -> None:
    event = parse_notice_page(
        (FIXTURES / "activate_multi.html").read_bytes(),
        event_type="activate",
        archive_publication_date="2026-06-30",
        source_url=ACTIVATE_URL,
        title="Daily Price Limits to be Broadened : 2 issues",
    )[0]
    event["expanded_range_yen"] = 600
    event["observed_multiplier"] = 600 / 700
    event["upper_limit_price_yen"] = 6_579
    event["effective_date"] = "2026-07-02"
    quality = validate_event(event, ["2026-06-30", "2026-07-01"])
    assert not quality["validation_pass"]
    assert set(quality["validation_issues"].split("|")) >= {
        "range_multiplier_invalid",
        "upper_limit_tick_rounding_invalid",
        "effective_date_not_session",
        "effective_session_not_expected",
    }


def test_revised_activation_is_a_correction_event() -> None:
    row = parse_notice_page(
        (FIXTURES / "correction.html").read_bytes(),
        event_type="correction",
        archive_publication_date="2017-03-28",
        source_url=CORRECTION_URL,
        title="Daily Price Limits to be Broadened : SAWAFUJI ELECTRIC CO.,LTD.【Revised】",
    )[0]
    assert row["security_code"] == "6901"
    assert row["event_type"] == "correction"
    assert row["observed_multiplier"] == 2.0
    assert validate_event(row, ["2017-03-28", "2017-03-29"])["validation_pass"]


@pytest.mark.parametrize(
    ("fixture", "publication", "url", "code", "effective", "sessions"),
    [
        (
            "activate_missing_effective_parenthesis.html",
            "2024-03-27",
            "https://www.jpx.co.jp/english/news/1030/20240327-01.html",
            "6938",
            "2024-03-28",
            ["2024-03-27", "2024-03-28"],
        ),
        (
            "activate_valid_effective_date_in_note.html",
            "2025-04-30",
            "https://www.jpx.co.jp/english/news/1030/20250430-03.html",
            "260A",
            "2025-05-01",
            ["2025-04-30", "2025-05-01"],
        ),
    ],
)
def test_parser_recovers_unambiguous_effective_date_variants(
    fixture: str,
    publication: str,
    url: str,
    code: str,
    effective: str,
    sessions: list[str],
) -> None:
    row = parse_notice_page(
        (FIXTURES / fixture).read_bytes(),
        event_type="activate",
        archive_publication_date=publication,
        source_url=url,
        title="Daily Price Limits to be Broadened : 1 issue",
    )[0]

    assert row["security_code"] == code
    assert row["effective_date"] == effective
    assert validate_event(row, sessions)["validation_pass"]


def test_resumable_producer_finalizes_immutably_and_verifies_hashes(tmp_path: Path) -> None:
    archive_manifest, sessions = _source_fixture(tmp_path)
    bodies = {
        ACTIVATE_URL: (FIXTURES / "activate_multi.html").read_bytes(),
        CORRECTION_URL: (FIXTURES / "correction.html").read_bytes(),
    }
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
        calls.append(url)
        return 200, bodies[url], {"Content-Type": "text/html"}

    kwargs = {
        "run_id": "fixture-run",
        "archive_manifest_path": archive_manifest,
        "session_calendar_path": sessions,
        "artifact_root": tmp_path / "artifacts",
        "expected_archive_sources": 1,
        "expected_notice_pages": 2,
        "expected_event_rows": 3,
        "sleep_seconds": 0,
    }
    partial = produce_jpx_exceptional_price_limits(**kwargs, max_requests=1, fetcher=fetch)
    assert partial.manifest["status"] == "in_progress"
    assert partial.manifest["counts"]["completed_notice_pages"] == 1

    complete = produce_jpx_exceptional_price_limits(**kwargs, fetcher=fetch)
    assert complete.manifest["immutable"] is True
    assert complete.manifest["counts"]["completed_notice_pages"] == 2
    assert complete.manifest["counts"]["parsed_event_rows"] == 3
    assert complete.manifest["remaining_completion_blockers"] == [PRE_2013_BLOCKER, POST_2020_BLOCKER]
    assert complete.manifest["expected_months"] == 1
    assert complete.manifest["expected_notice_pages"] == 2
    assert complete.manifest["denominator_reconciliation"] == {
        "expected_months": 1,
        "observed_month_sources": 1,
        "month_source_denominator_exact": True,
        "expected_notice_pages": 2,
        "planned_notice_pages": 2,
        "completed_notice_pages": 2,
        "notice_plan_denominator_exact": True,
        "completed_page_denominator_exact": True,
        "page_state_exact_plan": True,
    }
    assert complete.manifest["pre_2013_complete"] is False
    assert complete.manifest["post_2020_removal_complete"] is False
    assert complete.manifest["bounded_public_window_scorecard_credit"] is False
    assert complete.manifest["scorecard_eligible"] is False
    assert len(complete.manifest["per_page_sha256"]) == 2
    assert (complete.artifact_dir / "parse_quarantine.csv").read_text(encoding="utf-8").splitlines() == [
        "notice_id,url,raw_path,sha256,error"
    ]
    assert calls.count(CORRECTION_URL) + calls.count(ACTIVATE_URL) == 2

    offline_events, offline_quality, offline_quarantine = parse_archived_notice_corpus(
        archive_manifest_path=archive_manifest,
        page_state_path=complete.artifact_dir / "page_state.csv",
        session_calendar_path=sessions,
        expected_archive_sources=1,
        expected_notice_pages=2,
    )
    assert len(offline_events) == 3
    assert offline_quality["validation_pass"].all()
    assert offline_quarantine.empty

    def no_fetch(_: str) -> tuple[int, bytes, dict[str, str]]:
        raise AssertionError("a finalized run must not access the network")

    resumed = produce_jpx_exceptional_price_limits(**kwargs, fetcher=no_fetch)
    assert resumed.manifest == complete.manifest

    raw_path = Path(pd.read_csv(complete.artifact_dir / "page_state.csv").iloc[0]["raw_path"])
    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")
    with pytest.raises(ImmutableArtifactError, match="retained notice hash mismatch"):
        produce_jpx_exceptional_price_limits(**kwargs, fetcher=no_fetch)


def test_cli_executes_with_injected_fixture_fetcher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_manifest, sessions = _source_fixture(tmp_path)
    bodies = {
        ACTIVATE_URL: (FIXTURES / "activate_multi.html").read_bytes(),
        CORRECTION_URL: (FIXTURES / "correction.html").read_bytes(),
    }

    assert main(
        [
            "--run-id",
            "cli-run",
            "--archive-manifest",
            str(archive_manifest),
            "--session-calendar",
            str(sessions),
            "--artifact-root",
            str(tmp_path / "cli-artifacts"),
            "--expected-archive-sources",
            "1",
            "--expected-notice-pages",
            "2",
            "--expected-event-rows",
            "3",
            "--sleep-seconds",
            "0",
        ],
        fetcher=lambda url: (200, bodies[url], {"Content-Type": "text/html"}),
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete_denominator_partial_component"
    assert Path(output["manifest"]).is_file()


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    monthly = source / "news_ym_fixture.json"
    shutil.copyfile(FIXTURES / "news_ym_fixture.json", monthly)
    archive_manifest = source / "archive_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_id": "news_ym_fixture",
                "url": "https://www.jpx.co.jp/english/news/news_ym_fixture.json",
                "http_status": 200,
                "content_type": "application/json",
                "raw_path": str(monthly),
                "sha256": sha256_file(monthly),
                "bytes": monthly.stat().st_size,
            }
        ]
    ).to_csv(archive_manifest, index=False)
    sessions = source / "sessions.csv"
    shutil.copyfile(FIXTURES / "sessions.csv", sessions)
    plan, inventory = build_notice_plan(
        archive_manifest, expected_archive_sources=1, expected_notice_pages=2
    )
    assert len(plan) == 2 and len(inventory) == 1
    return archive_manifest, sessions
