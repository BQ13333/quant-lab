from __future__ import annotations

from quant_lab.data.japan_g4_halt_normalized_events import (
    DELISTING_CHANNEL,
    HALT_CHANNEL,
    conservative_report_known_at,
    parse_explicit_report_events,
)


def _parse(text: str, channel: str):
    return parse_explicit_report_events(
        text,
        report_month="2023-03",
        channel=channel,
        source_url="https://www.jpx.co.jp/report.pdf",
        source_path="/tmp/report.pdf",
        source_sha256="a" * 64,
    )


def test_multiday_halt_is_one_event_with_exact_interval() -> None:
    rows = _parse(
        "Changes in Listed Companies and Issues, Etc. (Mar. 2023)\n"
        "Suspension of Trading\n"
        "Bonds 2023.03.20 00450799 Credit Suisse Group AG 12:30～2023/3/22 12:29\n",
        HALT_CHANNEL,
    )
    assert len(rows) == 1
    assert rows[0]["local_code"] == "00450799"
    assert rows[0]["event_time_semantics"] == "multi_day_interval"
    assert rows[0]["event_start_jst"] == "2023-03-20T12:30:00+09:00"
    assert rows[0]["interval_end_jst"] == "2023-03-22T12:29:00+09:00"


def test_appendix_date_is_not_inferred_as_halt_event() -> None:
    rows = _parse(
        "Changes in Listed Companies and Issues, Etc. (Mar. 2023)\n"
        "Suspension of Trading\n"
        "2023.03.20 1234 Actual Issue 10:00～10:15\n"
        "変更日：2023.03.27 変更後単位 100株\n",
        HALT_CHANNEL,
    )
    assert len(rows) == 1
    assert rows[0]["local_code"] == "1234"


def test_delisting_and_conservative_known_at_are_date_only() -> None:
    rows = _parse(
        "Changes in Listed Companies and Issues, Etc. (Mar. 2023)\n"
        "Delisting\n1st Section 2023.03.28 5678 Example Company\nAssignment to the 1st Section\n",
        DELISTING_CHANNEL,
    )
    assert len(rows) == 1
    assert rows[0]["event_time_semantics"] == "date_only_reported_suspension"
    assert rows[0]["issue_name"] == "Example Company"
    assert rows[0]["admissible_to_shared_halt_history"] is False
    assert conservative_report_known_at("2023-03") == "2023-04-21T00:00:00+00:00"
