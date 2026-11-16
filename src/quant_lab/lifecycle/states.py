"""Typed, JSON-safe observations and registered period schedules."""
from dataclasses import dataclass
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
            try:
                if not np.isfinite(item.value).all():
                    return Permission.BLOCK
            except (TypeError, ValueError):
                return Permission.BLOCK
        elif not ((expected == Expectation.NOT_APPLICABLE and isinstance(item, NotApplicable)) or
                  (expected == Expectation.NO_OBSERVATION_EXPECTED and isinstance(item, NoObservationExpected))):
            return Permission.BLOCK
    return Permission.OK


def to_json(value: object) -> str:
    """Encode observations, schedules, enums and dictionaries with explicit tags."""
    def encode(x: object) -> object:
        if isinstance(x, Enum):
            return {'type': type(x).__name__, 'value': x.value}
        if isinstance(x, Value):
            return {'type': 'Value', 'value': encode(x.value)}
        if isinstance(x, (Missing, NotApplicable, NoObservationExpected)):
            return {'type': type(x).__name__}
        if isinstance(x, Schedule):
            return {'type': 'Schedule', 'fields': [[k, encode(v)] for k, v in x.fields]}
        if isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}
        return x
    return json.dumps(encode(value), allow_nan=False, sort_keys=True)


def from_json(payload: str) -> object:
    def decode(x: dict) -> object:
        kind = x.get('type')
        if kind in ('Permission', 'Lifecycle', 'Expectation'):
            return globals()[kind](x['value'])
        if kind == 'Value':
            return Value(x['value'])
        if kind in ('Missing', 'NotApplicable', 'NoObservationExpected'):
            return globals()[kind]()
        if kind == 'Schedule':
            return Schedule(tuple((k, v) for k, v in x['fields']))
        return x
    return json.loads(payload, object_hook=decode)
