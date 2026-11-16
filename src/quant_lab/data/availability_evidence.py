"""Immutable publication vintages; corroboration is exploration only."""
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class Grade(str, Enum):
    PRIMARY='PRIMARY'
    CAPTURED='CAPTURED'
    CORROBORATED='CORROBORATED'
    UNGRADED='UNGRADED'

@dataclass(frozen=True)
class Record:
    value: float
    grade: Grade
    published_at: object = None
    captured_at: object = None
    vintage_id: str = ''

    def __post_init__(self):
        import math
        if not isinstance(self.value,(float,int)) or not math.isfinite(self.value):
            raise ValueError('finite vintage value required')
        if not isinstance(self.grade,Grade) or not self.vintage_id:
            raise ValueError('grade and vintage required')


def available_at(record, calendar):
    time=record.published_at if record.grade == Grade.PRIMARY else record.captured_at if record.grade == Grade.CAPTURED else None
    if time is None: return pd.NaT
    time=pd.Timestamp(time)
    if pd.isna(time) or time.tzinfo is None: raise ValueError('observed timezone-aware publication/capture required')
    sessions=pd.DatetimeIndex(calendar)
    if sessions.tz is None or not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError('ordered timezone-aware decision calendar required')
    following=sessions[sessions > time]
    return following[0] if len(following) else pd.NaT


def append_vintage(records, record):
    if any(r.vintage_id == record.vintage_id for r in records):
        raise ValueError('vintage is append-only')
    return (*records,record)


def admission_view(panel, calendar):
    """Record cells indexed by decision time -> masked values and audit times."""
    if panel.attrs.get('exploration_only'): raise ValueError('exploration cannot enter admission')
    values=pd.DataFrame(float('nan'),index=panel.index,columns=panel.columns)
    times=pd.DataFrame(index=panel.index,columns=panel.columns,dtype=object)
    for day in panel.index:
        for col in panel.columns:
            r=panel.at[day,col]
            if not isinstance(r,Record): continue
            at=available_at(r,calendar)
            if pd.notna(at) and at <= day:
                values.at[day,col]=r.value
                times.at[day,col]=at
    values.attrs['availability_grade_verified']=True
    return values,times


def exploration_view(panel, *, ledger, reservation):
    trial=ledger.reserve_trial(**reservation)
    result=panel.map(lambda r:r.value if isinstance(r,Record) and r.grade != Grade.UNGRADED else float('nan'))
    result.attrs.update(exploration_only=True,trial_id=trial.trial_id)
    return result
