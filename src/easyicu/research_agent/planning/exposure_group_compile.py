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
derived when the study's data are prepared.  What a grouping can read is
listed here too (:func:`grouping_sources`): each concept's numeric summaries
over a window counted from ICU admission, and each numeric value recorded
once per stay, as the host materialized them.

An applied grouping's reference and primary contrast are part of what it
declares (:func:`group_contrast`), as the level codes of its variable: the
grouping record, the staged cohort's receipt and an input's planned column
carry them, and the context builder states them on the context the plan is
made on, so a template that fixes them reads the study's
(:func:`exposure_group_contrast`) instead of choosing its own.

An input planned on metadata alone names each concept once and holds no row.
There a grouping is listed and compiled against the columns the prepared
data will hold (:func:`planned_on_metadata`), so the derivation it records
is the one a run on those data compiles.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..cohort.materializer import EVENT_SUMMARIES, summary_column_name
from ..contracts.exposure_group_rules import (
    EXPOSURE_GROUP_CONTRASTS_KEY,
    GroupContrast,
)
from ..research_context.materialization_window import (
    context_column_windows,
    window_anchor,
)
from ..research_context.stay_events import column_kind, stay_outcome_columns
from ..schema import ResearchContext
from .cohort_identity import cohort_identity_columns
from .exposure_group_spec import (
    EXPOSURE_GROUP_SPEC_SCHEMA_VERSION,
    UNMEASURED_GROUP_ID,
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
    "exposure_group_concept_without_values",
)

#: Each finding of the population owner, in this owner's code.
_FROM_POPULATION: Mapping[str, str] = MappingProxyType(
    {
        "population_concept_not_in_export": "exposure_group_concept_not_in_export",
        "population_concept_unavailable": "exposure_group_concept_unavailable",
        "population_concept_without_values": "exposure_group_concept_without_values",
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


#: Each typed representation of a numeric summary over a window, by summary.
_WINDOW_SUMMARY_TRANSFORMS: Mapping[str, str] = MappingProxyType(
    {f"window_numeric_{summary}": summary for summary in sorted(EVENT_SUMMARIES)}
)
#: The typed representation of a value recorded once per stay.
_ONE_VALUE_TRANSFORM = "stay_level_unique_value"
_NUMERIC_DTYPES = ("int", "uint", "float", "double", "decimal")
#: What a metadata-only planning input records it is.
_METADATA_ONLY_STAGE = "metadata_only_planning"
#: The kinds of value the host materializes once per stay, as the concept.
_ONE_VALUE_KINDS = frozenset({"admission", "stay_level"})


@dataclass(frozen=True)
class GroupingSource:
    """One concept's values a grouping can read over one window.

    ``window`` is ``[start_hours, end_hours)`` after ICU admission, and
    ``None`` for a value recorded once per stay, whose one summary is
    ``value``.
    """

    concept: str
    summaries: tuple[str, ...]
    window: Optional[tuple[float, float]]
    unit: Optional[str]
    valid_range: Optional[tuple[float, float]]

    def line(self) -> str:
        if self.window is None:
            read = "one value per stay (summary value, no window)"
        else:
            start, end = self.window
            read = (
                f"{', '.join(self.summaries)} over hours [{start:g}, {end:g}) "
                "after ICU admission"
            )
        unit = f"unit {self.unit}" if self.unit else "no unit recorded"
        line = f"- {self.concept}: {read}; {unit}"
        if self.valid_range is not None:
            low, high = self.valid_range
            line += f"; declared values from {low:g} to {high:g}"
        return line


def planned_on_metadata(context: ResearchContext) -> bool:
    """Whether ``context`` is an input planned on metadata alone, with no row."""

    provenance = getattr(context.cohort, "provenance", None) or {}
    return provenance.get("evidence_stage") == _METADATA_ONLY_STAGE


def _as_prepared(context: ResearchContext) -> ResearchContext:
    """``context`` as the prepared data will hold it, when planned on metadata.

    A metadata-only input names a concept once.  Its data are prepared over
    the window the host materializes it over (``context_column_windows``): a
    concept measured over time becomes its first, maximum, mean and minimum
    there, named as the materializer names them
    (``cohort.materializer.summary_column_name``) and typed as
    ``intake.materialized_metadata`` types them, and a value recorded once per
    stay stays the concept's own column.  A column
    the input already names as one such summary is read as that summary.  Any
    other context is returned as it is.
    """

    if not planned_on_metadata(context):
        return context
    windows = context_column_windows(context)
    outcomes = stay_outcome_columns(context)
    excluded = set(cohort_identity_columns(context)) | set(outcomes)
    prepared: dict[str, Any] = {}
    for variable in context.variables:
        name = str(variable.name)
        concept = str(variable.source_concept or "").strip() or name
        numeric = str(variable.dtype or "").casefold().startswith(_NUMERIC_DTYPES)
        window = windows.get(name)
        if name in excluded or not numeric or variable.unit_normalization:
            prepared.setdefault(name, variable)
        elif (
            window is not None
            and window.source == "host"
            and window.anchor == window_anchor("icu_admission")
            and window.start_hours is not None
            and window.end_hours is not None
        ):
            summaries = (
                tuple(_WINDOW_SUMMARY_TRANSFORMS.values())
                if name == concept
                else tuple(
                    summary
                    for summary in _WINDOW_SUMMARY_TRANSFORMS.values()
                    if name == summary_column_name(concept, summary)
                )
            )
            if not summaries:
                prepared.setdefault(name, variable)
            for summary in summaries:
                column = summary_column_name(concept, summary)
                prepared.setdefault(
                    column,
                    variable.model_copy(
                        update={
                            "name": column,
                            "source_concept": concept,
                            "unit_normalization": f"window_numeric_{summary}",
                            "analysis_window": window.label,
                            "analysis_window_role": "outer_observation_window",
                        }
                    ),
                )
        elif (
            name == concept
            and column_kind(variable, column=name, concept=concept, outcomes=outcomes)
            in _ONE_VALUE_KINDS
        ):
            prepared.setdefault(
                name,
                variable.model_copy(
                    update={"unit_normalization": _ONE_VALUE_TRANSFORM}
                ),
            )
        else:
            prepared.setdefault(name, variable)
    return context.model_copy(update={"variables": list(prepared.values())})


def grouping_sources(context: ResearchContext) -> tuple[GroupingSource, ...]:
    """The numeric values of ``context`` a grouping can read, by concept and window.

    A column is listed when the host materialized it as a numeric summary of
    its concept over a window counted from ICU admission, or as a numeric
    value recorded once per stay.  An identifier and a stay's outcome are
    not values an exposure is formed from.  A context planned on metadata is
    read as its prepared data will hold it.
    """

    context = _as_prepared(context)
    windows = context_column_windows(context)
    excluded = set(cohort_identity_columns(context)) | set(
        stay_outcome_columns(context)
    )
    found: dict[tuple[str, Optional[tuple[float, float]]], dict[str, Any]] = {}
    for variable in context.variables:
        name = str(variable.name)
        if name in excluded:
            continue
        transform = str(variable.unit_normalization or "")
        window: Optional[tuple[float, float]]
        if transform in _WINDOW_SUMMARY_TRANSFORMS:
            column_window = windows.get(name)
            if (
                column_window is None
                or column_window.anchor != window_anchor("icu_admission")
                or column_window.start_hours is None
                or column_window.end_hours is None
            ):
                continue
            summary = _WINDOW_SUMMARY_TRANSFORMS[transform]
            window = (column_window.start_hours, column_window.end_hours)
        elif transform == _ONE_VALUE_TRANSFORM and str(
            variable.dtype or ""
        ).casefold().startswith(_NUMERIC_DTYPES):
            summary, window = "value", None
        else:
            continue
        concept = str(variable.source_concept or "").strip() or name
        entry = found.setdefault(
            (concept, window),
            {"summaries": set(), "unit": variable.unit, "range": None},
        )
        entry["summaries"].add(summary)
        bounds = list(variable.valid_range or ())
        if len(bounds) == 2 and all(_is_number(bound) for bound in bounds):
            entry["range"] = (float(bounds[0]), float(bounds[1]))
    return tuple(
        GroupingSource(
            concept=concept,
            summaries=tuple(sorted(entry["summaries"])),
            window=window,
            unit=entry["unit"],
            valid_range=entry["range"],
        )
        for (concept, window), entry in sorted(
            found.items(), key=lambda item: (item[0][0], item[0][1] or (-1.0, -1.0))
        )
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
            "contrast": self.grouping.contrast,
            # The groups the primary estimate compares, as level codes.
            "compared": group_contrast(self).compared() if self.applied else None,
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


_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CandidateExposureGroupings(BaseModel):
    """The groupings an accepted candidate plan applied, for a run that follows it.

    ``stated`` is every grouping the candidate's study stated, none when it
    stated none; each was applied, or the candidate would have stopped.
    ``derivations`` holds each one's derivation digest, by grouping id, and
    ``record_sha256`` is the digest of the candidate run's grouping record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_sha256: _Sha256
    stated: ExposureGroupings
    derivations: dict[str, _Sha256]

    @model_validator(mode="after")
    def _each_stated_grouping_applied(self) -> "CandidateExposureGroupings":
        if set(self.derivations) != {item.id for item in self.stated.groupings}:
            raise ValueError(
                "a candidate carries the digest of each grouping it stated"
            )
        return self


def group_contrast(item: CompiledGrouping) -> GroupContrast:
    """The applied grouping's reference and primary contrast, as its level codes.

    Unstated, the reference is the first group in code order and the contrast
    the last group other than the reference; the unmeasured level is neither.
    """

    spec = item.grouping
    levels = grouping_levels(spec)
    codes = {level: index for index, level in enumerate(levels, start=1)}
    groups = [level for level in levels if level != UNMEASURED_GROUP_ID]
    reference = spec.reference or groups[0]
    contrast = spec.contrast or next(
        level for level in reversed(groups) if level != reference
    )
    return GroupContrast(
        variable=str(item.variable),
        reference=codes[reference],
        contrast=codes[contrast],
    )


def exposure_group_contrast(
    context: ResearchContext, variable: str
) -> Optional[GroupContrast]:
    """The reference and contrast ``context`` states for ``variable``, if grouped.

    The context builder states them from the grouping's sealed declaration.
    """

    records = context.cohort.provenance.get(EXPOSURE_GROUP_CONTRASTS_KEY) or ()
    for record in records:
        if isinstance(record, Mapping) and record.get("variable") == variable:
            return GroupContrast(
                variable=variable,
                reference=int(record["reference"]),
                contrast=int(record["contrast"]),
            )
    return None


def compile_exposure_groupings(
    groupings: ExposureGroupings, context: ResearchContext
) -> CompiledGroupings:
    """Decide how each stated grouping reaches the rows of ``context``.

    A context planned on metadata is read as its prepared data will hold it.
    """

    names = {str(variable.name) for variable in context.variables}
    context = _as_prepared(context)
    names |= {str(variable.name) for variable in context.variables}
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
    "CandidateExposureGroupings",
    "CompiledGrouping",
    "CompiledGroupings",
    "Disposition",
    "GroupingSource",
    "compile_exposure_groupings",
    "exposure_group_contrast",
    "group_contrast",
    "grouping_sources",
    "grouping_variable_name",
    "planned_on_metadata",
]
