from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


Aggregation = Literal["last", "count", "sum", "mean", "max", "min"]


@dataclass(frozen=True)
class AvailabilityLagPolicy:
    """Lag applied between source availability and strategy use."""

    min_lag: pd.Timedelta = pd.Timedelta(0)
    decision_time_col: str = "decision_time_utc"
    available_time_col: str = "available_time_utc"


@dataclass(frozen=True)
class PITFeatureSpec:
    """Point-in-time event feature definition."""

    name: str
    value_col: str
    event_type: str | None = None
    event_type_col: str = "event_type"
    lookback: pd.Timedelta | None = None
    agg: Aggregation = "last"


def _to_utc_naive(values: pd.Series | pd.Index | list) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(converted, pd.Series):
        return converted.dt.tz_localize(None).astype("datetime64[ns]")
    if isinstance(converted, pd.DatetimeIndex):
        return pd.Series(converted.tz_localize(None).astype("datetime64[ns]"), index=getattr(values, "index", None))
    return pd.Series(converted)


def normalize_pit_times(
    frame: pd.DataFrame,
    time_cols: tuple[str, ...] = ("event_time_utc", "release_time_utc", "available_time_utc", "decision_time_utc"),
) -> pd.DataFrame:
    """Normalize known-at timestamps to timezone-free UTC pandas timestamps."""

    data = frame.copy()
    for col in time_cols:
        if col in data.columns:
            data[col] = _to_utc_naive(data[col])
    return data


def validate_pit_events(
    events: pd.DataFrame,
    sid_col: str = "sid",
    available_time_col: str = "available_time_utc",
    event_time_col: str | None = "event_time_utc",
) -> pd.DataFrame:
    """Validate and normalize an event stream before it can become features."""

    missing = [col for col in (sid_col, available_time_col) if col not in events.columns]
    if missing:
        raise ValueError(f"missing required point-in-time columns: {missing}")

    data = normalize_pit_times(events, time_cols=tuple(col for col in (available_time_col, event_time_col or "") if col))
    if data[available_time_col].isna().any():
        raise ValueError("available_time_utc must be populated for every event")
    if data[sid_col].isna().any():
        raise ValueError("sid must be populated for every event")
    if event_time_col and event_time_col in data.columns:
        bad = data[event_time_col].notna() & data[available_time_col].lt(data[event_time_col] - pd.Timedelta(days=3660))
        if bad.any():
            raise ValueError("available_time_utc is implausibly earlier than event_time_utc")
    return data


def assert_no_future_feature_leakage(
    features: pd.DataFrame,
    decision_time_col: str = "decision_time_utc",
    source_available_time_col: str = "source_available_time_utc",
    lag: pd.Timedelta = pd.Timedelta(0),
) -> None:
    """Raise if any feature row used source data unavailable at decision time."""

    missing = [col for col in (decision_time_col, source_available_time_col) if col not in features.columns]
    if missing:
        raise ValueError(f"missing required feature audit columns: {missing}")
    data = normalize_pit_times(features, time_cols=(decision_time_col, source_available_time_col))
    violations = data[source_available_time_col].notna() & data[source_available_time_col].gt(data[decision_time_col] - lag)
    if violations.any():
        examples = data.loc[violations, [decision_time_col, source_available_time_col]].head(5).to_dict("records")
        raise ValueError(f"point-in-time leakage detected: {examples}")


def point_in_time_asof_join(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    sid_col: str = "sid",
    decision_time_col: str = "decision_time_utc",
    available_time_col: str = "available_time_utc",
    value_cols: list[str] | None = None,
    lag: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    """Join each decision row to the latest event available before that decision."""

    missing_decision = [col for col in (sid_col, decision_time_col) if col not in decisions.columns]
    if missing_decision:
        raise ValueError(f"missing required decision columns: {missing_decision}")

    right = validate_pit_events(events, sid_col=sid_col, available_time_col=available_time_col)
    left = normalize_pit_times(decisions, time_cols=(decision_time_col,))
    if left[decision_time_col].isna().any():
        raise ValueError("decision_time_utc must be populated for every decision")

    value_cols = value_cols or [
        col
        for col in right.columns
        if col not in {sid_col, available_time_col, "event_time_utc", "release_time_utc", "event_type"}
    ]
    keep_cols = [sid_col, available_time_col, *[col for col in value_cols if col in right.columns]]
    right = right[keep_cols].sort_values([sid_col, available_time_col]).reset_index(drop=True)
    left = left.copy()
    left["_effective_decision_time"] = left[decision_time_col] - lag
    left["_original_order"] = range(len(left))

    joined_parts = []
    for sid, left_group in left.groupby(sid_col, dropna=False, sort=False):
        right_group = right[right[sid_col] == sid]
        if right_group.empty:
            joined = left_group.copy()
            for col in value_cols:
                joined[col] = pd.NA
            joined["source_available_time_utc"] = pd.NaT
            joined_parts.append(joined)
            continue
        joined = pd.merge_asof(
            left_group.sort_values("_effective_decision_time"),
            right_group.sort_values(available_time_col),
            left_on="_effective_decision_time",
            right_on=available_time_col,
            direction="backward",
            suffixes=("", "_event"),
        )
        event_sid_col = f"{sid_col}_event"
        if event_sid_col in joined.columns:
            joined = joined.drop(columns=[event_sid_col])
        joined["source_available_time_utc"] = joined[available_time_col]
        joined_parts.append(joined)

    if not joined_parts:
        return left.drop(columns=["_effective_decision_time", "_original_order"])
    result = pd.concat(joined_parts, ignore_index=True).sort_values("_original_order")
    drop_cols = ["_effective_decision_time", "_original_order"]
    if available_time_col in result.columns and available_time_col != "source_available_time_utc":
        drop_cols.append(available_time_col)
    return result.drop(columns=[col for col in drop_cols if col in result.columns]).reset_index(drop=True)


def build_point_in_time_event_features(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    specs: list[PITFeatureSpec],
    sid_col: str = "sid",
    decision_time_col: str = "decision_time_utc",
    available_time_col: str = "available_time_utc",
    lag: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    """Build event features using only source rows available before each decision."""

    if not specs:
        raise ValueError("at least one PITFeatureSpec is required")

    normalized_events = validate_pit_events(events, sid_col=sid_col, available_time_col=available_time_col)
    normalized_decisions = normalize_pit_times(decisions, time_cols=(decision_time_col,))
    rows = []
    for decision in normalized_decisions.itertuples(index=False):
        decision_dict = decision._asdict()
        sid = decision_dict[sid_col]
        decision_time = decision_dict[decision_time_col]
        cutoff = decision_time - lag
        base = {sid_col: sid, decision_time_col: decision_time}
        source_times = []

        sid_events = normalized_events[
            (normalized_events[sid_col] == sid) & normalized_events[available_time_col].le(cutoff)
        ]
        for spec in specs:
            spec_events = sid_events
            if spec.event_type is not None:
                if spec.event_type_col not in spec_events.columns:
                    raise ValueError(f"missing event type column for spec {spec.name}: {spec.event_type_col}")
                spec_events = spec_events[spec_events[spec.event_type_col] == spec.event_type]
            if spec.lookback is not None:
                spec_events = spec_events[spec_events[available_time_col].ge(cutoff - spec.lookback)]
            if spec.value_col not in spec_events.columns and spec.agg != "count":
                raise ValueError(f"missing value column for spec {spec.name}: {spec.value_col}")

            if spec_events.empty:
                base[spec.name] = 0.0 if spec.agg == "count" else pd.NA
                continue
            source_times.append(spec_events[available_time_col].max())
            if spec.agg == "last":
                base[spec.name] = spec_events.sort_values(available_time_col)[spec.value_col].iloc[-1]
            elif spec.agg == "count":
                base[spec.name] = float(len(spec_events))
            else:
                values = pd.to_numeric(spec_events[spec.value_col], errors="coerce")
                if spec.agg == "sum":
                    base[spec.name] = values.sum()
                elif spec.agg == "mean":
                    base[spec.name] = values.mean()
                elif spec.agg == "max":
                    base[spec.name] = values.max()
                elif spec.agg == "min":
                    base[spec.name] = values.min()
                else:
                    raise ValueError(f"unsupported aggregation: {spec.agg}")

        base["source_available_time_utc"] = max(source_times) if source_times else pd.NaT
        rows.append(base)

    result = pd.DataFrame(rows)
    assert_no_future_feature_leakage(
        result,
        decision_time_col=decision_time_col,
        source_available_time_col="source_available_time_utc",
        lag=lag,
    )
    return result
