"""Immutable JSON values with explicit, detached wire projections.

Frozen models alone do not freeze their nested dicts/lists. Boundary owners use
these values internally and return fresh JSON containers to outside consumers.
This module contains no scientific or approval policy.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, eq=False)
class FrozenMapping(Mapping[str, Any]):
    _values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo: dict) -> FrozenMapping:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return thaw_payload(self) == thaw_payload(other)


def freeze_payload(value: Any) -> Any:
    """Copy a JSON value into immutable containers; reject non-JSON objects."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("frozen_payload_keys_must_be_strings")
        return FrozenMapping(
            MappingProxyType({key: freeze_payload(item) for key, item in value.items()})
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_payload(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("frozen_payload_requires_finite_json_values")


def thaw_payload(value: Any) -> Any:
    """Return fresh mutable JSON containers, never references into a snapshot."""
    if isinstance(value, Mapping):
        return {key: thaw_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_payload(item) for item in value]
    return value
