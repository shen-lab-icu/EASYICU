"""An exposure a study forms by grouping one measured value, as the Planner states it.

A study that compares stays by where a measurement falls -- hypoglycaemia,
hyperglycaemia only and normoglycaemia in the first 24 h; the classes of body
mass index -- needs a closed-domain exposure that no export column holds.  The
Planner states it as typed rules over one concept's summary, never as code:
each group names a threshold rule or ``otherwise``, and the host decides
which column a summary is read from and derives the group column
(:mod:`.exposure_group_compile`).  Nothing here reads a row.

The groups are matched in the order they are listed: a stay belongs to the
first group whose rule it meets, and ``otherwise`` takes every measured stay
no earlier group took.  Listing order is a priority, not a scale.  An ordinal
grouping numbers its groups along its scale, ``g1`` lowest, so the level codes
sort in scale order wherever levels are sorted by name.  A stay with no
measurement of the concept in the window meets no rule and is never
``otherwise``: ``unmeasured`` says whether such stays form their own group,
which is not a level of the scale, or leave the study.

Each group is a level the analysis models by name, so a reference is a group
the study states, never whichever group comes first, and so is the group the
primary estimate compares with it (``contrast``).  A nominal grouping has no
order, so nothing may read it as a trend.

A window is hours after ICU admission, ``[start_hours, end_hours)``.  A value
recorded once per stay, such as body mass index, has none, and its rules read
that one value (``value``).  A threshold is in the unit the study states it
in; the host refuses a unit the column does not record instead of converting.

A study may form up to :data:`MAX_EXPOSURE_GROUPINGS` groupings, each with its
own id, compiled into its own variable: a primary grouping beside a finer
secondary one of the same value, for instance.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .population_spec import CriterionSource, SpecWindow, quote_written_in

EXPOSURE_GROUP_SPEC_SCHEMA_VERSION = "easyicu.exposure_group_spec/1"
MAX_EXPOSURE_GROUPINGS = 3
MAX_EXPOSURE_GROUPS = 6
#: The level of the stays with no measurement, when they form their own group.
UNMEASURED_GROUP_ID = "gU"

ExposureScale = Literal["nominal", "ordinal"]
#: The summaries the host materializes over a window, and a once-per-stay value.
GroupSummary = Literal["min", "max", "mean", "first", "value"]
GroupOp = Literal["<", "<=", ">", ">="]

#: The summaries of one window that lie between its minimum and maximum.
_BETWEEN = ("first", "mean")
_NEGATION = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}


class GroupRule(BaseModel):
    """The summary of the value meets a threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: GroupSummary
    op: GroupOp
    value: float = Field(allow_inf_nan=False)
    #: The threshold's unit when the statement gives one.
    unit: Optional[str] = Field(default=None, min_length=1, max_length=32)


class ExposureGroup(BaseModel):
    """One level: the stays meeting ``rule`` that no earlier group took."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^g[1-6]$")
    label: str = Field(min_length=1, max_length=60)
    rule: Union[GroupRule, Literal["otherwise"]]


class UnmeasuredOwnGroup(BaseModel):
    """Stays without a measurement form the level ``gU``, outside any scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handling: Literal["own_group"]
    label: str = Field(min_length=1, max_length=60)


class UnmeasuredExcluded(BaseModel):
    """Stays without a measurement leave the study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handling: Literal["exclude"]


Unmeasured = Annotated[
    Union[UnmeasuredOwnGroup, UnmeasuredExcluded], Field(discriminator="handling")
]


class ExposureGroupSpec(BaseModel):
    """One grouping of one concept's value into named levels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Unique within the study; the derived variable is named after it.
    id: str = Field(pattern=r"^x[1-3]$")
    concept: str = Field(min_length=1, max_length=128)
    window: Optional[SpecWindow]
    scale: ExposureScale
    groups: list[ExposureGroup] = Field(min_length=2, max_length=MAX_EXPOSURE_GROUPS)
    unmeasured: Unmeasured
    #: The group the others are compared with.
    reference: Optional[str] = Field(default=None, pattern=r"^g[1-6]$")
    #: The group whose comparison with the reference is the primary estimate.
    contrast: Optional[str] = Field(default=None, pattern=r"^g[1-6]$")
    #: The words the grouping is stated in, verbatim.
    quote: str = Field(min_length=2, max_length=160)
    source: CriterionSource

    @field_validator("concept")
    @classmethod
    def _named(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("an exposure grouping names its concept")
        return cleaned

    @model_validator(mode="after")
    def _consistent(self) -> "ExposureGroupSpec":
        ids = [group.id for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("exposure group ids must be unique")
        labels = [group.label.strip().casefold() for group in self.groups]
        if isinstance(self.unmeasured, UnmeasuredOwnGroup):
            labels.append(self.unmeasured.label.strip().casefold())
        if len(labels) != len(set(labels)):
            raise ValueError("exposure group labels must be distinct")
        if any(group.rule == "otherwise" for group in self.groups[:-1]):
            raise ValueError(
                "otherwise takes the measured stays no group before it took, so "
                "it is the last group and there is one"
            )
        summaries = {rule.summary for rule in grouping_rules(self)}
        if self.window is None and summaries - {"value"}:
            raise ValueError(
                "a grouping without a window reads a value recorded once per stay: "
                "write its rules with summary value, or state the window"
            )
        if self.window is not None and "value" in summaries:
            raise ValueError(
                "a windowed grouping summarizes the window: write each rule's "
                "summary as min, max, mean or first"
            )
        units = {rule.unit for rule in grouping_rules(self)}
        if len(units) > 1:
            raise ValueError(
                "exposure_group_units_differ: state every threshold of one grouping "
                "in one unit"
            )
        if self.reference is not None and self.reference not in ids:
            raise ValueError("the reference names one of the groups")
        if self.contrast is not None and (
            self.reference is None
            or self.contrast not in ids
            or self.contrast == self.reference
        ):
            raise ValueError(
                "the contrast names another group than the reference, which it "
                "is compared with"
            )
        _reachable(self)
        if self.scale == "ordinal":
            _numbered_along_the_scale(self)
        return self


def grouping_rules(spec: ExposureGroupSpec) -> list[GroupRule]:
    """The groups' threshold rules in matching order, ``otherwise`` aside."""

    return [group.rule for group in spec.groups if isinstance(group.rule, GroupRule)]


def _negated(rule: GroupRule) -> GroupRule:
    return rule.model_copy(update={"op": _NEGATION[rule.op]})


def _bound_below(op: str) -> bool:
    return op in {">", ">="}


def _bounds(
    rules: Sequence[GroupRule],
) -> tuple[dict[str, tuple[float, bool]], dict[str, tuple[float, bool]]]:
    """Each summary's tightest bound from below and from above, closed or not."""

    low: dict[str, tuple[float, bool]] = {}
    high: dict[str, tuple[float, bool]] = {}
    for rule in rules:
        closed = rule.op in {"<=", ">="}
        if _bound_below(rule.op):
            current = low.get(rule.summary, (-math.inf, True))
            if (rule.value, not closed) > (current[0], not current[1]):
                low[rule.summary] = (rule.value, closed)
        else:
            current = high.get(rule.summary, (math.inf, True))
            if (rule.value, closed) < (current[0], current[1]):
                high[rule.summary] = (rule.value, closed)
    return low, high


def _satisfiable(rules: Sequence[GroupRule]) -> bool:
    """Whether one measured stay can meet every rule at once.

    The summaries of one window are ordered: its minimum is at most its first
    and mean values, and each of those is at most its maximum.  Every
    rule bounds one summary from one side, so the rules are met together
    exactly when each summary's bounds leave room and no summary is bounded
    below past where a summary at or above it is bounded above.
    """

    low, high = _bounds(rules)

    def room(below: str, above: str) -> bool:
        start, start_closed = low.get(below, (-math.inf, True))
        end, end_closed = high.get(above, (math.inf, True))
        return start < end or (start == end and start_closed and end_closed)

    names = set(low) | set(high)
    if any(not room(name, name) for name in names):
        return False
    pairs = [("min", name) for name in (*_BETWEEN, "max")]
    pairs += [(name, "max") for name in _BETWEEN]
    return all(room(below, above) for below, above in pairs)


def _reachable(spec: ExposureGroupSpec) -> None:
    earlier: list[GroupRule] = []
    for group in spec.groups:
        if group.rule == "otherwise":
            if not _satisfiable(earlier):
                raise ValueError(
                    f"exposure_group_rule_unreachable: every measured stay meets a "
                    f"group before {group.id}, so otherwise takes none"
                )
            return
        if not _satisfiable([*earlier, group.rule]):
            raise ValueError(
                f"exposure_group_rule_unreachable: no measured stay can reach "
                f"{group.id}, since a group before it takes every stay its rule meets"
            )
        earlier.append(_negated(group.rule))
    if _satisfiable(earlier):
        raise ValueError(
            "exposure_group_rules_not_exhaustive: a measured stay can meet no "
            "group; end with an otherwise group or cover the remaining values"
        )


def _numbered_along_the_scale(spec: ExposureGroupSpec) -> None:
    """An ordinal grouping reads one summary, and its ids ascend along it.

    Over one summary every group holds one interval of values: its rule
    without the rules of the groups before it.  ``g1`` is the lowest, so each
    group's interval ends where the next group's begins, or below it.
    """

    summaries = {rule.summary for rule in grouping_rules(spec)}
    if len(summaries) != 1:
        raise ValueError(
            "an ordinal grouping reads one summary, so that its groups lie along "
            "one scale: group a mixture of summaries as nominal"
        )
    summary = next(iter(summaries))
    intervals: dict[str, tuple[float, float]] = {}
    earlier: list[GroupRule] = []
    for group in spec.groups:
        rules = list(earlier)
        if isinstance(group.rule, GroupRule):
            rules.append(group.rule)
            earlier.append(_negated(group.rule))
        low, high = _bounds(rules)
        intervals[group.id] = (
            low.get(summary, (-math.inf, True))[0],
            high.get(summary, (math.inf, True))[0],
        )
    ordered = sorted(intervals)
    for below, above in zip(ordered, ordered[1:]):
        if intervals[below][1] > intervals[above][0]:
            raise ValueError(
                f"an ordinal grouping numbers its groups along its scale, g1 "
                f"lowest, but {below} holds values above {above}'s: renumber them"
            )


class ExposureGroupings(BaseModel):
    """Every grouping the study forms; none when it forms none."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groupings: list[ExposureGroupSpec] = Field(
        default_factory=list, max_length=MAX_EXPOSURE_GROUPINGS
    )

    @model_validator(mode="after")
    def _distinct(self) -> "ExposureGroupings":
        ids = [item.id for item in self.groupings]
        if len(ids) != len(set(ids)):
            raise ValueError("exposure grouping ids must be unique")
        stated = [
            json.dumps(
                item.model_dump(mode="json", exclude={"id", "quote", "source"}),
                sort_keys=True,
            )
            for item in self.groupings
        ]
        if len(stated) != len(set(stated)):
            raise ValueError("two exposure groupings state the same groups")
        return self


#: The owner's errors kept for refused groupings.
_MAX_SPEC_ERRORS = 20


class ExposureGroupingsRefused(ValueError):
    """The owner refuses stated groupings; ``errors`` locate why, without their input."""

    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        located = "; ".join(
            f"{item['loc'] or '<groupings>'}: {item['msg']}" for item in errors
        )
        super().__init__(f"the exposure groupings are refused: {located}")


def read_stated_exposure_groupings(raw: Any) -> ExposureGroupings:
    """The groupings as stated, read strictly; none when nothing is stated."""

    if raw is None:
        return ExposureGroupings()
    try:
        return ExposureGroupings.model_validate(raw)
    except ValidationError as exc:
        raise ExposureGroupingsRefused(
            [
                {
                    "loc": ".".join(str(part) for part in error["loc"]),
                    "type": error["type"],
                    "msg": error["msg"],
                }
                for error in exc.errors(include_url=False, include_input=False)
            ][:_MAX_SPEC_ERRORS]
        ) from exc


def unquoted_groupings(
    groupings: ExposureGroupings, study_texts: Sequence[str]
) -> tuple[ExposureGroupSpec, ...]:
    """The groupings citing the study's words whose quote is not written in them."""

    return tuple(
        item
        for item in groupings.groupings
        if not quote_written_in(item.quote, item.source, study_texts)
    )


def grouping_levels(spec: ExposureGroupSpec) -> tuple[str, ...]:
    """The level codes of the grouping's variable, the scale's first."""

    levels = tuple(sorted(group.id for group in spec.groups))
    if isinstance(spec.unmeasured, UnmeasuredOwnGroup):
        return (*levels, UNMEASURED_GROUP_ID)
    return levels


def grouping_labels(spec: ExposureGroupSpec) -> Mapping[str, str]:
    """Each level code's reader-facing label."""

    labels = {group.id: group.label for group in spec.groups}
    if isinstance(spec.unmeasured, UnmeasuredOwnGroup):
        labels[UNMEASURED_GROUP_ID] = spec.unmeasured.label
    return labels


__all__ = [
    "EXPOSURE_GROUP_SPEC_SCHEMA_VERSION",
    "MAX_EXPOSURE_GROUPINGS",
    "MAX_EXPOSURE_GROUPS",
    "UNMEASURED_GROUP_ID",
    "ExposureGroup",
    "ExposureGroupSpec",
    "ExposureGroupings",
    "ExposureGroupingsRefused",
    "ExposureScale",
    "GroupOp",
    "GroupRule",
    "GroupSummary",
    "Unmeasured",
    "UnmeasuredExcluded",
    "UnmeasuredOwnGroup",
    "grouping_labels",
    "grouping_levels",
    "grouping_rules",
    "read_stated_exposure_groupings",
    "unquoted_groupings",
]
