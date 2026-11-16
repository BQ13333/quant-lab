from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.point_in_time import (
    PITFeatureSpec,
    assert_no_future_feature_leakage,
    build_point_in_time_event_features,
    point_in_time_asof_join,
    validate_pit_events,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["US:AAA", "US:AAA", "US:AAA", "US:BBB"],
            "event_type": ["earnings", "analyst_revision", "analyst_revision", "earnings"],
            "event_time_utc": [
                "2024-02-01T00:00:00Z",
                "2024-02-05T14:00:00Z",
                "2024-02-10T14:00:00Z",
                "2024-02-01T00:00:00Z",
            ],
            "available_time_utc": [
                "2024-02-01T21:05:00Z",
                "2024-02-05T14:05:00Z",
                "2024-02-10T14:05:00Z",
                "2024-02-01T13:00:00Z",
            ],
            "eps_surprise": [0.12, pd.NA, pd.NA, -0.04],
            "revision_pct": [pd.NA, 0.03, -0.01, pd.NA],
        }
    )


def test_validate_pit_events_requires_available_time():
    events = _events().drop(columns=["available_time_utc"])

    with pytest.raises(ValueError, match="available_time_utc"):
        validate_pit_events(events)


def test_point_in_time_asof_join_does_not_see_after_close_release_early():
    decisions = pd.DataFrame(
        {
            "sid": ["US:AAA", "US:AAA"],
            "decision_time_utc": ["2024-02-01T20:30:00Z", "2024-02-02T14:30:00Z"],
        }
    )
    earnings = _events()[_events()["event_type"] == "earnings"]

    joined = point_in_time_asof_join(
        decisions,
        earnings,
        value_cols=["eps_surprise"],
    )

    assert pd.isna(joined.loc[0, "eps_surprise"])
    assert joined.loc[1, "eps_surprise"] == 0.12
    assert joined.loc[1, "source_available_time_utc"] == pd.Timestamp("2024-02-01 21:05:00")


def test_point_in_time_asof_join_respects_minimum_lag():
    decisions = pd.DataFrame(
        {
            "sid": ["US:AAA", "US:AAA"],
            "decision_time_utc": ["2024-02-05T14:30:00Z", "2024-02-05T15:10:00Z"],
        }
    )
    revisions = _events()[_events()["event_type"] == "analyst_revision"]

    joined = point_in_time_asof_join(
        decisions,
        revisions,
        value_cols=["revision_pct"],
        lag=pd.Timedelta(hours=1),
    )

    assert pd.isna(joined.loc[0, "revision_pct"])
    assert joined.loc[1, "revision_pct"] == 0.03


def test_build_point_in_time_event_features_uses_only_available_events():
    decisions = pd.DataFrame(
        {
            "sid": ["US:AAA", "US:AAA", "US:AAA"],
            "decision_time_utc": [
                "2024-02-02T14:30:00Z",
                "2024-02-09T14:30:00Z",
                "2024-02-12T14:00:00Z",
            ],
        }
    )
    specs = [
        PITFeatureSpec(
            name="last_eps_surprise",
            event_type="earnings",
            value_col="eps_surprise",
            agg="last",
        ),
        PITFeatureSpec(
            name="revision_count_7d",
            event_type="analyst_revision",
            value_col="revision_pct",
            lookback=pd.Timedelta(days=7),
            agg="count",
        ),
        PITFeatureSpec(
            name="revision_sum_7d",
            event_type="analyst_revision",
            value_col="revision_pct",
            lookback=pd.Timedelta(days=7),
            agg="sum",
        ),
    ]

    features = build_point_in_time_event_features(decisions, _events(), specs)

    assert features["last_eps_surprise"].tolist() == [0.12, 0.12, 0.12]
    assert features["revision_count_7d"].tolist() == [0.0, 1.0, 2.0]
    assert pd.isna(features.loc[0, "revision_sum_7d"])
    assert features.loc[1, "revision_sum_7d"] == 0.03
    assert features.loc[2, "revision_sum_7d"] == pytest.approx(0.02)


def test_assert_no_future_feature_leakage_raises():
    features = pd.DataFrame(
        {
            "decision_time_utc": ["2024-02-01T20:30:00Z"],
            "source_available_time_utc": ["2024-02-01T21:05:00Z"],
            "feature": [1.0],
        }
    )

    with pytest.raises(ValueError, match="point-in-time leakage"):
        assert_no_future_feature_leakage(features)


def test_mixed_timestamp_units_same_result():
    decisions=pd.DataFrame({'sid':['a'],'decision_time_utc':pd.to_datetime(['2024-01-02']).as_unit('us')})
    events=pd.DataFrame({'sid':['a'],'available_time_utc':pd.to_datetime(['2024-01-01']).as_unit('ns'),'value':[3]})
    mixed=point_in_time_asof_join(decisions,events)
    decisions['decision_time_utc']=decisions.decision_time_utc.astype('datetime64[ns]')
    pd.testing.assert_frame_equal(mixed,point_in_time_asof_join(decisions,events))
