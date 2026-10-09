"""How each stated exposure grouping reaches the rows, decided by the host.

A grouping (:mod:`.exposure_group_spec`) becomes a closed-domain variable a
plan can name only when every rule can be read from a column of the input as
it is stated: the summary of the concept over the stated window, in the unit
the threshold is written in, at a threshold the concept can take on both of
its sides.  The population owner judges each rule exactly as it judges a
population measurement (``population_compile.read_threshold``); this owner
names what it finds in its own stable codes.  A grouping waits for an
extraction when one that holds the stated summary would let the host read it;
any other finding leaves it not applied.

Nothing here reads a row.  An applied grouping is declared here, with the
columns it reads and a digest of everything that decides its levels, and is
derived when the study's data are prepared.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

from ..schema import ResearchContext
from .exposure_group_spec import (
    EXPOSURE_GROUP_SPEC_SCHEMA_VERSION,
    ExposureGroupings,
    ExposureGroupSpec,
    GroupRule,
    grouping_labels,
    grouping_levels,
    grouping_rules,
)
from .population_compile import read_threshold

EXPOSURE_GROUP_COMPILE_SCHEMA_VERSION = "easyicu.exposure_group_compile/1"

Disposition = Literal["applied", "requires_extraction", "not_applied"]

#: Why a grouping waits for an extraction that would hold what reads it.
#: Stable: a published code never changes.
REQUIRES_EXTRACTION_REASONS = (
    "exposure_group_concept_not_in_export",
    "exposure_group_source_column_unavailable",
)
#: Why a grouping is not applied.  Stable: a published code never changes.
NOT_APPLIED_REASONS = (
    "exposure_group_concept_unavailable",
    "exposure_group_source_column_unreadable",
    "exposure_group_unit_mismatch",
    "exposure_group_unit_unrecorded",
    "exposure_group_threshold_outside_domain",
    "exposure_group_name_taken",
)

#: Each finding of the population owner, in this owner's code.
_FROM_POPULATION: Mapping[str, str] = MappingProxyType(
    {
        "population_concept_not_in_export": "exposure_group_concept_not_in_export",
        "population_concept_unavailable": "exposure_group_concept_unavailable",
        "population_identifier_column": "exposure_group_concept_unavailable",
        "population_column_unresolved": "exposure_group_source_column_unavailable",
        "population_window_unreadable": "exposure_group_source_column_unavailable",
        "population_whole_stay_in_finite_window": (
            "exposure_group_source_column_unavailable"
        ),
        "population_event_time_not_hours": "exposure_group_source_column_unreadable",
        "population_condition_column_not_status": (
            "exposure_group_source_column_unreadable"
        ),
        "population_unit_mismatch": "exposure_group_unit_mismatch",
        "population_unit_unrecorded": "exposure_group_unit_unrecorded",
        "population_threshold_outside_domain": (
            "exposure_group_threshold_outside_domain"
        ),
    }
)


def grouping_variable_name(grouping: ExposureGroupSpec) -> str:
    """The derived variable's name: the concept and the grouping's id."""

    return f"{grouping.concept}_group_{grouping.id}"


@dataclass(frozen=True)
class CompiledGrouping:
    """One grouping's disposition and, when applied, the variable it declares."""

    grouping: ExposureGroupSpec
    disposition: Disposition
    reason: Optional[str]
    detail: str
    #: The column each summary is read from, when every rule can be read.
    source_columns: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def applied(self) -> bool:
        return self.disposition == "applied"

    @property
    def variable(self) -> Optional[str]:
        return grouping_variable_name(self.grouping) if self.applied else None

    def derivation(self) -> dict[str, Any]:
        """Everything that decides which level a stay is given."""

        grouping = self.grouping
        return {
            "concept": grouping.concept,
            "window": (
                None
                if grouping.window is None
                else grouping.window.model_dump(mode="json")
            ),
            "scale": grouping.scale,
            "groups": [
                {
                    "id": group.id,
                    "rule": (
                        group.rule
                        if group.rule == "otherwise"
                        else group.rule.model_dump(mode="json")
                    ),
                }
                for group in grouping.groups
            ],
            "unmeasured": grouping.unmeasured.handling,
            "source_columns": dict(sorted(self.source_columns.items())),
        }

    def derivation_sha256(self) -> str:
        return _digest(self.derivation())

    def record(self) -> dict[str, Any]:
        return {
            "id": self.grouping.id,
            "disposition": self.disposition,
            "reason": self.reason,
            "detail": self.detail,
            "variable": self.variable,
            "levels": list(grouping_levels(self.grouping)),
            "labels": dict(grouping_labels(self.grouping)),
            "reference": self.grouping.reference,
            "source": self.grouping.source,
            "quote": self.grouping.quote,
            "derivation": self.derivation(),
            "derivation_sha256": self.derivation_sha256(),
        }


@dataclass(frozen=True)
class CompiledGroupings:
    """Every grouping's disposition."""

    groupings: tuple[CompiledGrouping, ...]

    @property
    def applied(self) -> tuple[CompiledGrouping, ...]:
        return tuple(item for item in self.groupings if item.applied)

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": EXPOSURE_GROUP_COMPILE_SCHEMA_VERSION,
            "spec_schema_version": EXPOSURE_GROUP_SPEC_SCHEMA_VERSION,
            "groupings": [item.record() for item in self.groupings],
        }

    def sha256(self) -> str:
        return _digest(self.record())


def compile_exposure_groupings(
    groupings: ExposureGroupings, context: ResearchContext
) -> CompiledGroupings:
    """Decide how each stated grouping reaches the rows of ``context``."""

    names = {str(variable.name) for variable in context.variables}
    return CompiledGroupings(
        groupings=tuple(_compile(item, context, names) for item in groupings.groupings)
    )


class _Unread(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _compile(
    grouping: ExposureGroupSpec, context: ResearchContext, names: set[str]
) -> CompiledGrouping:
    try:
        name = grouping_variable_name(grouping)
        if name in names:
            raise _Unread(
                "exposure_group_name_taken",
                f"The input already holds a variable named {name!r}.",
            )
        columns: dict[str, str] = {}
        for rule in grouping_rules(grouping):
            columns[rule.summary] = _read(grouping, rule, context)
    except _Unread as found:
        return CompiledGrouping(
            grouping=grouping,
            disposition=(
                "requires_extraction"
                if found.reason in REQUIRES_EXTRACTION_REASONS
                else "not_applied"
            ),
            reason=found.reason,
            detail=found.detail,
        )
    read = ", ".join(
        f"{summary} from {column!r}" for summary, column in columns.items()
    )
    return CompiledGrouping(
        grouping=grouping,
        disposition="applied",
        reason=None,
        detail=f"{len(grouping_levels(grouping))} levels; reads {read}.",
        source_columns=MappingProxyType(columns),
    )


def _read(
    grouping: ExposureGroupSpec, rule: GroupRule, context: ResearchContext
) -> str:
    reading = read_threshold(
        context,
        concept=grouping.concept,
        summary=rule.summary,
        window=grouping.window,
        op=rule.op,
        value=rule.value,
        unit=rule.unit,
    )
    if reading.reason is not None:
        code = _FROM_POPULATION.get(reading.reason)
        if code is None:
            raise ValueError(f"no exposure grouping code for {reading.reason!r}")
        if code == "exposure_group_concept_not_in_export":
            raise _Unread(
                code,
                f"This input does not hold {grouping.concept!r}; an extraction that "
                f"holds it would let the host derive {grouping.id}.",
            )
        raise _Unread(code, reading.detail)
    assert reading.column is not None
    _inside_domain(grouping, rule, reading.column, context)
    return reading.column


def _inside_domain(
    grouping: ExposureGroupSpec, rule: GroupRule, column: str, context: ResearchContext
) -> None:
    """A threshold the concept's declared range holds on both of its sides."""

    variable = next(
        (item for item in context.variables if str(item.name) == column), None
    )
    bounds = list(getattr(variable, "valid_range", None) or ())
    if len(bounds) != 2 or not all(_is_number(bound) for bound in bounds):
        return
    low, high = (float(bound) for bound in bounds)
    if not low < rule.value < high:
        raise _Unread(
            "exposure_group_threshold_outside_domain",
            f"{grouping.id} compares the {rule.summary} of {grouping.concept!r} with "
            f"{rule.value:g}, but {column!r} declares values from {low:g} to "
            f"{high:g}, so the threshold cannot separate its stays.",
        )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "EXPOSURE_GROUP_COMPILE_SCHEMA_VERSION",
    "NOT_APPLIED_REASONS",
    "REQUIRES_EXTRACTION_REASONS",
    "CompiledGrouping",
    "CompiledGroupings",
    "Disposition",
    "compile_exposure_groupings",
    "grouping_variable_name",
]
