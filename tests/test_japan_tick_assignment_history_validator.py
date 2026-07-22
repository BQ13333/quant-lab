from __future__ import annotations

import pandas as pd

from quant_lab.data.japan_tick_assignment_history_validator import (
    _independent_event_parse,
    _independent_replay,
)


def test_independent_parser_reconstructs_notice_event_keys(tmp_path) -> None:
    raw = tmp_path / "notice.html"
    raw.write_text(
        "<p>Stocks that will newly be subject to narrow tick sizes</p>"
        "<table><tr><th>Code</th><th>Name</th></tr><tr><td>285A</td><td>Alpha</td></tr></table>"
        "<p>Stocks that will no longer be subject to narrow tick sizes</p>"
        "<table><tr><th>Code</th><th>Name</th></tr><tr><td>8303</td><td>Beta</td></tr></table>",
        encoding="utf-8",
    )
    notices = pd.DataFrame(
        [
            {
                "url": "https://www.jpx.co.jp/english/news/test.html",
                "effective_date": "2023-09-07",
                "raw_path": str(raw),
            }
        ]
    )
    assert _independent_event_parse(notices) == {
        ("2023-09-07", "ADD", "285A", "https://www.jpx.co.jp/english/news/test.html"),
        ("2023-09-07", "REMOVE", "8303", "https://www.jpx.co.jp/english/news/test.html"),
    }


def test_independent_replay_requires_exact_checkpoint_sets() -> None:
    checkpoints = pd.DataFrame(
        [
            {"effective_date": "2014-01-14", "index_scope": "TOPIX100", "code": "1111"},
            {"effective_date": "2014-07-22", "index_scope": "TOPIX100", "code": "2222"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "effective_date": "2014-07-22",
                "action": "REMOVE",
                "code": "1111",
            },
            {
                "effective_date": "2014-07-22",
                "action": "ADD",
                "code": "2222",
            },
        ]
    )
    reconciliation, state = _independent_replay(events, checkpoints)
    assert reconciliation["exact_set_match"].tolist() == [True, True]
    assert state == {"2222"}
