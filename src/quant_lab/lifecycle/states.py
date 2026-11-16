"""Typed, JSON-safe observations and registered period schedules."""
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from typing import Generic, TypeVar, TypeAlias, Mapping
import numpy as np

T = TypeVar('T')

class Permission(str, Enum):
    OK = 'OK'
    BLOCK = 'BLOCK'
    HALT = 'HALT'

class Lifecycle(str, Enum):
    CANDIDATE = 'CANDIDATE'
    VALIDATED = 'VALIDATED'
    PAPER = 'PAPER'
    RAMP = 'RAMP'
    KEEP = 'KEEP'
    REDUCE = 'REDUCE'
    PAUSE = 'PAUSE'
    RETIRED = 'RETIRED'

@dataclass(frozen=True)
class Value(Generic[T]):
    value: T

@dataclass(frozen=True)
class NotApplicable:
    pass

@dataclass(frozen=True)
class NoObservationExpected:
    pass

@dataclass(frozen=True)
class Missing:
    pass

Observed: TypeAlias = Value[T] | NotApplicable | NoObservationExpected | Missing

class Expectation(str, Enum):
    DUE = 'DUE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    NO_OBSERVATION_EXPECTED = 'NO_OBSERVATION_EXPECTED'

@dataclass(frozen=True)
class Schedule:
    """Immutable name/expectation pairs; unknown or omitted inputs fail closed."""
    fields: tuple[tuple[str, Expectation], ...]

    def __post_init__(self) -> None:
        if len(dict(self.fields)) != len(self.fields) or any(not isinstance(e, Expectation) for _, e in self.fields):
            raise ValueError('invalid schedule')


def validate_inputs(schedule: Schedule, inputs: Mapping[str, Observed]) -> Permission:
    if set(inputs) != dict(schedule.fields).keys():
        return Permission.BLOCK
    for key, expected in schedule.fields:
        item = inputs[key]
        if isinstance(item, Value):
            if expected != Expectation.DUE:
                return Permission.BLOCK
            try:
                if not np.isfinite(item.value).all():
                    return Permission.BLOCK
            except (TypeError, ValueError):
                return Permission.BLOCK
        elif not ((expected == Expectation.NOT_APPLICABLE and isinstance(item, NotApplicable)) or
                  (expected == Expectation.NO_OBSERVATION_EXPECTED and isinstance(item, NoObservationExpected))):
            return Permission.BLOCK
    return Permission.OK


@dataclass(frozen=True)
class ScheduleCalendar:
    # (field, cadence in months; zero means not applicable, first due month)
    cadence: tuple[tuple[str, int, int], ...] = (
        ('ic', 1, 1), ('drawdown', 1, 1), ('cost', 0, 1),
        ('crowd', 0, 1), ('capacity', 1, 1), ('mechanism_broken', 1, 1), ('halt', 1, 1))
    no_observation_months: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        if not isinstance(self.cadence, tuple) or any(not isinstance(x, tuple) for x in self.cadence) or not isinstance(self.no_observation_months, tuple) or any(not isinstance(x, tuple) or not isinstance(x[1], tuple) for x in self.no_observation_months):
            raise ValueError('calendar must be deeply immutable')
        names = [x[0] for x in self.cadence]
        if len(set(names)) != len(names) or any(type(c) is not int or c < 0 or type(f) is not int or f < 1 for _, c, f in self.cadence):
            raise ValueError('invalid calendar')
        if len(dict(self.no_observation_months)) != len(self.no_observation_months):
            raise ValueError('duplicate exception')
        for name, months in self.no_observation_months:
            if name not in names or any(type(t) is not int or t < 1 for t in months):
                raise ValueError('invalid calendar exception')

    def schedule_for(self, t):
        if type(t) is not int or t < 1:
            raise ValueError('invalid month')
        exceptions = dict(self.no_observation_months)
        return Schedule(tuple((name, Expectation.NO_OBSERVATION_EXPECTED if t in exceptions.get(name, ())
            else Expectation.DUE if cadence and t >= first and (t-first) % cadence == 0
            else Expectation.NOT_APPLICABLE) for name, cadence, first in self.cadence))

    def sha256(self):
        import hashlib
        return hashlib.sha256(to_json(self).encode()).hexdigest()


def to_json(value: object) -> str:
    def encode(x):
        if isinstance(x, Enum):
            return {'type': type(x).__name__, 'value': x.value}
        if is_dataclass(x):
            return {'type': type(x).__name__, 'fields': {f.name: encode(getattr(x, f.name)) for f in fields(x)}}
        if isinstance(x, tuple):
            return {'type': 'tuple', 'items': [encode(v) for v in x]}
        if isinstance(x, list):
            return [encode(v) for v in x]
        if isinstance(x, dict):
            return {'type': 'dict', 'items': [[k, encode(v)] for k, v in x.items()]}
        return x
    return json.dumps({'schema_version': 1, 'payload': encode(value)}, allow_nan=False, sort_keys=True)


def from_json(payload: str) -> object:
    from .monitor import MonitorState, Policy
    classes = {c.__name__: c for c in (Value, Missing, NotApplicable, NoObservationExpected,
        Schedule, ScheduleCalendar, MonitorState, Policy, Permission, Lifecycle, Expectation)}
    def decode(x):
        if isinstance(x, list):
            return [decode(v) for v in x]
        if not isinstance(x, dict):
            return x
        kind = x.get('type')
        if kind == 'tuple':
            return tuple(decode(v) for v in x['items'])
        if kind == 'dict':
            return {k: decode(v) for k, v in x['items']}
        if kind not in classes:
            raise ValueError('unknown type tag')
        cls = classes[kind]
        return cls(x['value']) if issubclass(cls, Enum) else cls(**{k: decode(v) for k,v in x['fields'].items()})
    try:
        raw = json.loads(payload)
        if type(raw['schema_version']) is not int or raw['schema_version'] != 1 or set(raw) != {'schema_version', 'payload'}:
            raise ValueError('unknown schema version')
        return decode(raw['payload'])
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid serialized state') from exc
