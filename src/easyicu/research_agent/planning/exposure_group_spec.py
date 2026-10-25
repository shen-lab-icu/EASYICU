"""An exposure a study forms by grouping one measured value, as the Planner states it.

A study that compares stays by where a measurement falls -- hypoglycaemia,
hyperglycaemia only and normoglycaemia in the first 24 h; the classes of body
mass index -- needs a closed-domain exposure that no export column holds.  The
Planner states it as typed rules over one concept's summary, never as code:
each group names a threshold rule or ``otherwise``, and the host decides
which column a summary is read from and derives the group column
(:mod:`.exposure_group_compile`).  Nothing here reads a row.

The compiler requires disjoint rules: a stay belongs to its unique matching
group, and ``otherwise`` takes every measured stay no explicit rule took.
An ordinal
grouping's groups are numbered along its scale, ``g1`` lowest -- the owner
renumbers stated ids that do not follow it, and the reference and contrast
follow their groups -- and its level codes follow its scale; a nominal
grouping's codes follow the order its groups are stated (:func:`grouping_levels`).  A stay with no
measurement of the concept in the window meets no rule and is never
``otherwise``: ``unmeasured`` says whether such stays form their own group,
which is not a level of any scale, or leave the study.  So an ordinal
grouping's unmeasured stays leave it: its levels are its scale.

Each group is a level the analysis models by name, so a reference is a group
the study states, and so is the group the primary estimate compares with it
(``contrast``). Unstated, templates propose the first and last group in code
order; the compiled plan supplies the displayed comparison. Neither side may
be the unmeasured level. A nominal grouping has no order, so
nothing may read it as a trend.

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
import re
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..contracts.exposure_group_rules import (
    ExposureGroupPartitionError,
    GroupInterval,
    partition_intervals,
    threshold_interval,
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
    """One level: the stays uniquely meeting ``rule``."""

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

    @model_validator(mode="before")
    @classmethod
    def _ordinal_ids_along_the_scale(cls, data: Any) -> Any:
        return _otherwise_named_by_its_values(_renumbered_along_the_scale(data))

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
            if isinstance(self.unmeasured, UnmeasuredOwnGroup):
                raise ValueError(
                    "exposure_group_ordinal_unmeasured_own_group: the unmeasured "
                    "stays of an ordinal grouping are on no level of its scale; "
                    "let them leave the study, or group them as nominal"
                )
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
    # One-dimensional partition checks belong to the compiler, so refusals are
    # recorded and the structured retry receives the exact conflicting groups.
    if len({rule.summary for rule in grouping_rules(spec)}) == 1:
        return
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


def grouping_partition(spec: ExposureGroupSpec) -> tuple[GroupInterval, ...]:
    """Prove the partition and retain only exclusions stated in its quoted words.

    The Planner shape is unchanged. A between band is the complement of two
    disjoint outer rules; an explicit exclusion is quoted as an interval or a
    one-sided inequality, never inferred from unmeasured handling.
    """

    try:
        intervals, gaps = partition_intervals([group.model_dump() for group in spec.groups])
    except ExposureGroupPartitionError as exc:
        labels = ", ".join(f"{group.id} ({group.label})" for group in spec.groups if group.id in exc.groups)
        raise ExposureGroupPartitionError(exc.reason, f"{spec.id}: {labels}; {exc}", exc.groups) from exc
    for gap in gaps:
        if not _stated_exclusion(spec.quote, gap) or spec.source not in {"question", "study_wording"}:
            raise ExposureGroupPartitionError(
                "exposure_group_rules_not_exhaustive",
                f"{spec.id}: no group takes {gap.text()} of {gap.summary}; "
                "cover it or quote the study's explicit exclusion with its exact bounds",
            )
    for group in spec.groups:
        actual = intervals.get(group.id)
        if actual is None:
            continue
        named = _label_interval(group.label, actual.summary, actual.unit)
        if named is not None and named != actual:
            raise ExposureGroupPartitionError(
                "exposure_group_label_rule_mismatch",
                f"{spec.id}: {group.id} label {group.label!r} means {named.text()}, "
                f"but its rule takes {actual.text()}; make label and rule agree",
                (group.id,),
            )
    return gaps


_NUMBER = r"-?\d+(?:\.\d+)?"
_BRACKET_RANGE = re.compile(rf"([\[(])\s*({_NUMBER}|-inf)\s*,\s*({_NUMBER}|inf)\s*([\])])")


def _label_interval(label: str, summary: str, unit: Optional[str]) -> Optional[GroupInterval]:
    """Read unambiguous numeric labels; free-text category names remain names."""

    text = label.strip().casefold()
    bracket = _BRACKET_RANGE.search(text)
    if bracket:
        opening, lower, upper, closing = bracket.groups()
        return GroupInterval(summary, unit, float(lower), float(upper), opening == "[", closing == "]")
    between = re.search(rf"({_NUMBER})\s*(?:–|—|-|to|至)\s*<\s*({_NUMBER})", text)
    if between:
        return GroupInterval(summary, unit, float(between[1]), float(between[2]), True, False)
    patterns = (
        (rf"(?:<=|≤)\s*({_NUMBER})|({_NUMBER})\s*(?:岁|years?)?\s*及以下", "<="),
        (rf"(?:>=|≥|at least)\s*({_NUMBER})|({_NUMBER})\s*(?:岁|years?)?\s*(?:及以上|以上|(?:or|and) (?:above|more|older))", ">="),
        (rf"(?:<|below|under|less than)\s*({_NUMBER})|({_NUMBER})\s*(?:岁|years?)?\s*以下", "<"),
        (rf"(?:>|above|over|greater than)\s*({_NUMBER})|({_NUMBER})\s*(?:岁|years?)?\s*以上且不含", ">"),
    )
    # Compound bounds are not half-lines (for example "≥2, <4 mmol/L").
    matches = []
    for pattern, op in patterns:
        found = re.search(pattern, text)
        if found:
            matches.append((found, op))
    # Prefer inclusive spellings over their contained '<' / 'below' fragments.
    chosen = []
    for found, op in matches:
        if any(other.start() <= found.start() and other.end() >= found.end() for other, _ in chosen):
            continue
        chosen.append((found, op))
    if not chosen:
        return None
    bounds = [threshold_interval({"summary": summary, "unit": unit, "op": op,
                                  "value": next(value for value in found.groups() if value is not None)})
              for found, op in chosen]
    lower = max((bound.lower, not bound.lower_inclusive) for bound in bounds)
    upper = min((bound.upper, bound.upper_inclusive) for bound in bounds)
    return GroupInterval(summary, unit, lower[0], upper[0], not lower[1], upper[1])


def _stated_exclusion(quote: str, gap: GroupInterval) -> bool:
    # Require a positive exclusion clause, so "do not exclude" cannot authorize
    # removing values. Unsupported prose is returned to the Planner to clarify.
    clauses = re.split(r"[;；。\n]", quote.casefold())
    for clause in clauses:
        match = re.match(r"\s*(?:exclude|excluding|排除)\s*(?:values?\s*)?(.+)", clause)
        if match and _label_interval(match[1], gap.summary, gap.unit) == gap:
            return True
    return False


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
    intervals = _intervals_along(spec.groups, next(iter(summaries)))
    ordered = sorted(intervals)
    for below, above in zip(ordered, ordered[1:]):
        if intervals[below][1] > intervals[above][0]:
            raise ValueError(
                f"an ordinal grouping numbers its groups along its scale, g1 "
                f"lowest, but {below} holds values above {above}'s: renumber them"
            )


def _intervals_along(
    groups: Sequence[ExposureGroup], summary: str
) -> dict[str, tuple[float, float]]:
    """Each group's interval of ``summary``: its rule without the earlier groups'."""

    intervals: dict[str, tuple[float, float]] = {}
    earlier: list[GroupRule] = []
    for group in groups:
        rules = list(earlier)
        if isinstance(group.rule, GroupRule):
            rules.append(group.rule)
            earlier.append(_negated(group.rule))
        low, high = _bounds(rules)
        intervals[group.id] = (
            low.get(summary, (-math.inf, True))[0],
            high.get(summary, (math.inf, True))[0],
        )
    return intervals


def _renumbered_along_the_scale(data: Any) -> Any:
    """A stated ordinal grouping with its ids numbered along its scale.

    The ids only name the groups: a group holds the values its rule takes in
    the order the groups are listed, under its label.  So the owner numbers
    an ordinal grouping's groups itself -- the stated ids, lowest to the group
    holding the lowest values -- and the reference and contrast follow the
    groups they name.  The rules, their order and the labels are unchanged.
    A grouping the owner cannot read this way is returned as stated, and
    validation refuses it with its reason.
    """

    if not isinstance(data, Mapping) or data.get("scale") != "ordinal":
        return data
    stated = data.get("groups")
    if not isinstance(stated, list):
        return data
    try:
        groups = [ExposureGroup.model_validate(item) for item in stated]
    except ValidationError:
        return data
    ids = [group.id for group in groups]
    summaries = {
        group.rule.summary for group in groups if isinstance(group.rule, GroupRule)
    }
    if len(ids) != len(set(ids)) or len(summaries) != 1:
        return data
    intervals = _intervals_along(groups, next(iter(summaries)))
    along = sorted(groups, key=lambda group: intervals[group.id])
    named = dict(zip((group.id for group in along), sorted(ids)))
    if all(old == new for old, new in named.items()):
        return data
    return {
        **data,
        "groups": [{**group.model_dump(), "id": named[group.id]} for group in groups],
        **{
            key: named.get(data[key], data[key])
            for key in ("reference", "contrast")
            if isinstance(data.get(key), str)
        },
    }


def _interval_label(rules: Sequence[GroupRule], summary: str) -> str:
    low, high = _bounds(rules)
    parts = []
    if summary in low:
        value, closed = low[summary]
        parts.append(f"{'≥' if closed else '>'}{value:g}")
    if summary in high:
        value, closed = high[summary]
        parts.append(f"{'≤' if closed else '<'}{value:g}")
    units = {rule.unit for rule in rules if rule.unit}
    unit = f" {next(iter(units))}" if len(units) == 1 else ""
    return f"{', '.join(parts)}{unit}" if parts else ""


def _otherwise_named_by_its_values(data: Any) -> Any:
    """A stated grouping whose ``otherwise`` group is labelled by its values.

    The label is what the reader sees; ``otherwise`` is the rule's keyword,
    not a name (pilot1 round 2, 10-22, A6: 「<2 mmol/L / otherwise / ≥4
    mmol/L」 on a Chinese plan card).  When the groups read one summary, the
    owner names that group by the values it holds, the earlier groups' rules
    negated: 「≥2, <4 mmol/L」.  Its rule, place and id are unchanged.
    """

    if not isinstance(data, Mapping):
        return data
    stated = data.get("groups")
    if not isinstance(stated, list):
        return data
    try:
        groups = [ExposureGroup.model_validate(item) for item in stated]
    except ValidationError:
        return data
    rules = [group.rule for group in groups if isinstance(group.rule, GroupRule)]
    summaries = {rule.summary for rule in rules}
    last = groups[-1] if groups else None
    if (
        last is None
        or last.rule != "otherwise"
        or last.label.strip().casefold() != "otherwise"
        or len(summaries) != 1
    ):
        return data
    label = _interval_label(
        [_negated(rule) for rule in rules],
        next(iter(summaries)),
    )
    if not label or label.casefold() in {group.label.strip().casefold() for group in groups}:
        return data
    return {
        **data,
        "groups": [
            *(group.model_dump() for group in groups[:-1]),
            {**last.model_dump(), "label": label},
        ],
    }


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
    """The grouping's level ids in the order of their codes, ``1`` first.

    A nominal grouping's groups take their codes in the order the study
    states them; an ordinal grouping's along its scale, which its ids are
    numbered along.  The unmeasured level, when it is one, is last.
    """

    stated = tuple(group.id for group in spec.groups)
    levels = tuple(sorted(stated)) if spec.scale == "ordinal" else stated
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
    "grouping_partition",
    "read_stated_exposure_groupings",
    "unquoted_groupings",
]
