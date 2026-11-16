"""Placeholder: US-only modules were moved out of this repo on 16 Nov 2026."""

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Old US-only modules and tests were moved out of the main tree on "
        "16 Nov 2026 (Asia/Kuala_Lumpur) after Ben said the old US data is "
        "no longer needed and was a distraction. They were archived, not "
        "deleted, at /workspace/quant_house/_archive_not_in_use/"
        "quant-lab_us_modules/. Shared fee/calendar utilities "
        "(usa_tca_calibration, usa_exchange_calendar_tca) stay because Japan "
        "TCA imports them."
    )
)


def test_usa_only_modules_archived_and_not_in_main_tree() -> None:
    assert False, "should be skipped"
