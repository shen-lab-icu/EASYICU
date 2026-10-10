"""Which level of an exposure grouping each stay takes, from its materialized values.

Owner
-----
A study may form its exposure by grouping one measured value
(``planning/exposure_group_spec.py``).  The host compiles the stated rules
into the parameters a derivation receipt records
(``planning/exposure_group_compile.CompiledGrouping.derivation``).  This module
is the one evaluation of those parameters over the columns they name: the host
derives a run's group column with it, and the materialized-metadata validator
recomputes that column with it, so the column a run stages and the column its
authority vouches for are computed by the same code.  It reads the parameters
as a plain mapping and imports no planning owner, so no host module joins the
execution kernel through it; it reads no file.

The groups are matched in their stated order: a measured stay takes the first
group whose threshold its summary meets, and ``otherwise`` takes every measured
stay no earlier group took.  A stay is measured when every summary the rules
read holds a value.  An unmeasured stay is never ``otherwise``: it takes the
unmeasured level when unmeasured stays form their own group, and no level when
they leave the study.

A column holds each level as an integer code, because a study's declared
domain is read from its descriptor before any row exists, and a descriptor
declares integer levels only: the groups are ``1`` to ``k``, and the
unmeasured level is ``k + 1``.  Labels are for readers.  The column's
transform says whether its codes lie along a scale: an ordinal grouping's
are ``1`` (lowest) to ``k`` along it, its ids numbered along it, and since
the unmeasured level is on no scale, an ordinal grouping's unmeasured stays
leave the study.  A nominal grouping's codes follow the order its groups are
stated, so code ``1`` is the group the study names first.

An input planned on metadata alone holds no rows, so no level is derived on
it: a grouping's column is declared there (:class:`PlannedGroupColumn`) with
the codes it will hold when the study's data are prepared.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

#: The transform of a nominal grouping's column of level codes.
EXPOSURE_GROUP_TRANSFORM_ID = "exposure_group"
#: The transform of an ordinal grouping's column: its codes lie along its scale.
EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID = "exposure_group_ordinal"
EXPOSURE_GROUP_TRANSFORMS = frozenset(
    {EXPOSURE_GROUP_TRANSFORM_ID, EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID}
)
#: The level of the stays with no measurement, when they form their own group.
UNMEASURED_GROUP_ID = "gU"
#: Every key of a grouping's derivation parameters.
GROUPING_PARAMETER_KEYS = frozenset(
    {"concept", "window", "scale", "groups", "unmeasured", "source_columns"}
)
#: Every key of a group column's declaration on an input planned on metadata.
PLANNED_GROUP_COLUMN_KEYS = frozenset(
    {
        "variable",
        "transform",
        "levels",
        "concept",
        "window",
        "description",
        "compared",
        "groupings_record_sha256",
    }
)
#: Where a research context states each grouped exposure's reference and
#: primary contrast, as level codes.  The context builder projects them from
#: the grouping's sealed declaration -- a staged cohort's receipt, or an
#: input's planned column -- so no other owner states them.
EXPOSURE_GROUP_CONTRASTS_KEY = "exposure_group_contrasts"
#: The keys of a grouping's comparison, as level codes.
GROUP_COMPARISON_KEYS = frozenset({"reference", "contrast"})

_GROUP_ID = re.compile(r"^g[1-6]$")
_SUMMARIES = frozenset({"min", "max", "mean", "first", "value"})
_COMPARE = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}
_SCALES = frozenset({"nominal", "ordinal"})
_UNMEASURED = frozenset({"own_group", "exclude"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: Two to six groups, and the unmeasured level.
_MAX_LEVELS = 7


class ExposureGroupRuleError(ValueError):
    """The parameters, or the columns they read, do not decide each stay's level."""


@dataclass(frozen=True)
class GroupContrast:
    """A grouped exposure's reference and primary contrast, as level codes.

    Both are groups -- the ones the study states or, unstated, the first and
    the last group in code order -- and the unmeasured level is neither.
    """

    variable: str
    reference: int
    contrast: int

    def compared(self) -> dict[str, int]:
        return {"reference": self.reference, "contrast": self.contrast}

    def record(self) -> dict[str, Any]:
        return {"variable": self.variable, **self.compared()}


def read_group_contrast(raw: Any, *, variable: str, groups: int) -> GroupContrast:
    """The comparison a declaration states, read strictly.

    ``groups`` is how many codes name groups: ``1`` to ``groups``, the
    unmeasured level, when it is one, after them.
    """

    if not isinstance(raw, Mapping) or set(raw) != GROUP_COMPARISON_KEYS:
        raise ExposureGroupRuleError(
            "a grouping's comparison holds its reference and its contrast"
        )
    reference, contrast = raw["reference"], raw["contrast"]
    if (
        any(
            isinstance(code, bool)
            or not isinstance(code, int)
            or not 1 <= code <= groups
            for code in (reference, contrast)
        )
        or reference == contrast
    ):
        raise ExposureGroupRuleError(
            "a grouping compares two of its groups, never its unmeasured level"
        )
    return GroupContrast(variable=variable, reference=reference, contrast=contrast)


@dataclass(frozen=True)
class GroupThreshold:
    """A summary of the value meets a threshold."""

    summary: str
    op: str
    value: float


@dataclass(frozen=True)
class GroupingRules:
    """One grouping's derivation parameters, read strictly."""

    concept: str
    scale: str
    #: Each group's id and threshold in matching order; ``None`` is otherwise.
    groups: tuple[tuple[str, Optional[GroupThreshold]], ...]
    unmeasured: str
    #: The column each summary the rules read is taken from.
    source_columns: Mapping[str, str]

    @property
    def levels(self) -> tuple[str, ...]:
        """The level ids in the order of their codes; ``gU`` last when it is one.

        A nominal grouping's groups in the order they are stated, an ordinal
        grouping's along its scale, which its ids are numbered along.
        """

        stated = tuple(group_id for group_id, _rule in self.groups)
        ids = tuple(sorted(stated)) if self.scale == "ordinal" else stated
        if self.unmeasured == "own_group":
            return (*ids, UNMEASURED_GROUP_ID)
        return ids

    @property
    def codes(self) -> Mapping[str, int]:
        """Each level id's code in the column: its place among the levels."""

        return MappingProxyType(
            {level: index for index, level in enumerate(self.levels, start=1)}
        )

    @property
    def read_columns(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.source_columns.values())))

    @property
    def transform_id(self) -> str:
        if self.scale == "ordinal":
            return EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID
        return EXPOSURE_GROUP_TRANSFORM_ID


def read_grouping_rules(parameters: Mapping[str, Any]) -> GroupingRules:
    """The parameters a receipt records, or :class:`ExposureGroupRuleError`."""

    if not isinstance(parameters, Mapping):
        raise ExposureGroupRuleError("a grouping's parameters are an object")
    if set(parameters) != GROUPING_PARAMETER_KEYS:
        raise ExposureGroupRuleError(
            "a grouping's parameters hold exactly "
            + ", ".join(sorted(GROUPING_PARAMETER_KEYS))
        )
    concept = parameters["concept"]
    if not isinstance(concept, str) or not concept.strip():
        raise ExposureGroupRuleError("a grouping names its concept")
    scale = parameters["scale"]
    if scale not in _SCALES:
        raise ExposureGroupRuleError(f"unknown grouping scale {scale!r}")
    unmeasured = parameters["unmeasured"]
    if unmeasured not in _UNMEASURED:
        raise ExposureGroupRuleError(f"unknown unmeasured handling {unmeasured!r}")
    if scale == "ordinal" and unmeasured != "exclude":
        raise ExposureGroupRuleError(
            "an ordinal grouping's unmeasured stays leave the study: the "
            "unmeasured level lies on no scale"
        )
    _window(parameters["window"])
    columns = _source_columns(parameters["source_columns"])
    groups = _groups(parameters["groups"], columns)
    read = {rule.summary for _group_id, rule in groups if rule is not None}
    if read != set(columns):
        raise ExposureGroupRuleError(
            "a grouping names a source column for exactly the summaries its rules read"
        )
    return GroupingRules(
        concept=concept,
        scale=scale,
        groups=groups,
        unmeasured=unmeasured,
        source_columns=MappingProxyType(dict(sorted(columns.items()))),
    )


def _window(raw: Any) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping) or set(raw) != {"start_hours", "end_hours"}:
        raise ExposureGroupRuleError(
            "a grouping's window holds start_hours and end_hours"
        )
    start, end = raw["start_hours"], raw["end_hours"]
    if not all(_is_number(bound) for bound in (start, end)) or not start < end:
        raise ExposureGroupRuleError("a grouping's window ends after it starts")


def _source_columns(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ExposureGroupRuleError("a grouping names the columns its rules read")
    columns: dict[str, str] = {}
    for summary, column in raw.items():
        if summary not in _SUMMARIES:
            raise ExposureGroupRuleError(f"unknown summary {summary!r}")
        if not isinstance(column, str) or not column.strip():
            raise ExposureGroupRuleError(f"the {summary} summary names no column")
        columns[summary] = column
    return columns


def _groups(
    raw: Any, columns: Mapping[str, str]
) -> tuple[tuple[str, Optional[GroupThreshold]], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ExposureGroupRuleError("a grouping lists its groups")
    if not 2 <= len(raw) <= 6:
        raise ExposureGroupRuleError("a grouping has two to six groups")
    groups: list[tuple[str, Optional[GroupThreshold]]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"id", "rule"}:
            raise ExposureGroupRuleError("a group holds its id and its rule")
        group_id = item["id"]
        if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
            raise ExposureGroupRuleError(f"unknown group id {group_id!r}")
        rule = item["rule"]
        if rule == "otherwise":
            if index != len(raw) - 1:
                raise ExposureGroupRuleError("otherwise is the last group")
            groups.append((group_id, None))
            continue
        groups.append((group_id, _threshold(rule, columns)))
    ids = [group_id for group_id, _rule in groups]
    if len(ids) != len(set(ids)):
        raise ExposureGroupRuleError("group ids are unique")
    return tuple(groups)


def _threshold(raw: Any, columns: Mapping[str, str]) -> GroupThreshold:
    if (
        not isinstance(raw, Mapping)
        or not {"summary", "op", "value"} <= set(raw)
        or set(raw) - {"summary", "op", "value", "unit"}
    ):
        raise ExposureGroupRuleError("a rule holds a summary, an op and a value")
    summary, op, value = raw["summary"], raw["op"], raw["value"]
    if summary not in columns:
        raise ExposureGroupRuleError(f"a rule reads {summary!r}, which names no column")
    if op not in _COMPARE:
        raise ExposureGroupRuleError(f"unknown comparison {op!r}")
    if not _is_number(value):
        raise ExposureGroupRuleError("a rule's threshold is a finite number")
    return GroupThreshold(summary=summary, op=op, value=float(value))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def grouping_level_description(rules: GroupingRules, labels: Mapping[str, str]) -> str:
    """What a grouping's column holds, for its readers: each code and its label."""

    described = "; ".join(
        f"{rules.codes[level]} = {labels[level]}" for level in rules.levels
    )
    return (
        f"{rules.scale.capitalize()} groups of {rules.concept} stated by the "
        f"study, as level codes: {described}"
    )


@dataclass(frozen=True)
class PlannedGroupColumn:
    """A grouping's column declared on an input planned on metadata alone.

    The input holds no rows, so the column holds no value: the declaration
    says what it will hold when the study's data are prepared -- codes ``1``
    to ``levels`` of the grouping of ``concept`` over ``window``, its
    ``transform`` saying whether they lie along a scale -- the groups its
    primary estimate compares, and the digest of the record that stated the
    grouping.
    """

    variable: str
    transform: str
    levels: int
    concept: str
    window: Optional[tuple[float, float]]
    description: str
    contrast: GroupContrast
    groupings_record_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "transform": self.transform,
            "levels": self.levels,
            "concept": self.concept,
            "window": (
                None
                if self.window is None
                else {"start_hours": self.window[0], "end_hours": self.window[1]}
            ),
            "description": self.description,
            "compared": self.contrast.compared(),
            "groupings_record_sha256": self.groupings_record_sha256,
        }


def planned_group_column(
    parameters: Mapping[str, Any],
    *,
    variable: str,
    labels: Mapping[str, str],
    compared: Mapping[str, Any],
    groupings_record_sha256: str,
) -> PlannedGroupColumn:
    """The declaration of the column a grouping's derivation ``parameters`` form.

    ``compared`` is the grouping's reference and primary contrast, as the
    level codes of two of its groups.
    """

    rules = read_grouping_rules(parameters)
    read_group_contrast(compared, variable=variable, groups=len(rules.groups))
    window = parameters["window"]
    return _planned(
        {
            "variable": variable,
            "transform": rules.transform_id,
            "levels": len(rules.levels),
            "concept": rules.concept,
            "window": window,
            "description": grouping_level_description(rules, labels),
            "compared": dict(compared),
            "groupings_record_sha256": groupings_record_sha256,
        }
    )


def read_planned_group_columns(raw: Any) -> tuple[PlannedGroupColumn, ...]:
    """The group columns an input planned on metadata declares, read strictly.

    One record states them all, so they carry one record digest.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ExposureGroupRuleError("planned group columns are a non-empty list")
    columns = tuple(_planned(item) for item in raw)
    if len({item.variable for item in columns}) != len(columns):
        raise ExposureGroupRuleError("each planned group column is declared once")
    if len({item.groupings_record_sha256 for item in columns}) != 1:
        raise ExposureGroupRuleError("one grouping record states every planned column")
    return columns


def _planned(raw: Any) -> PlannedGroupColumn:
    if not isinstance(raw, Mapping) or set(raw) != PLANNED_GROUP_COLUMN_KEYS:
        raise ExposureGroupRuleError(
            "a planned group column holds exactly "
            + ", ".join(sorted(PLANNED_GROUP_COLUMN_KEYS))
        )
    texts = [raw[key] for key in ("variable", "concept", "description")]
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ExposureGroupRuleError("a planned group column names its variable")
    if raw["transform"] not in EXPOSURE_GROUP_TRANSFORMS:
        raise ExposureGroupRuleError(f"unknown group transform {raw['transform']!r}")
    levels = raw["levels"]
    if (
        isinstance(levels, bool)
        or not isinstance(levels, int)
        or not 2 <= levels <= _MAX_LEVELS
    ):
        raise ExposureGroupRuleError("a planned group column has two to seven levels")
    digest = raw["groupings_record_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ExposureGroupRuleError("a planned group column names its record digest")
    window = raw["window"]
    _window(window)
    return PlannedGroupColumn(
        variable=raw["variable"],
        transform=raw["transform"],
        levels=levels,
        concept=raw["concept"],
        window=(
            None
            if window is None
            else (float(window["start_hours"]), float(window["end_hours"]))
        ),
        description=raw["description"],
        # Its creation checked that both codes name groups; read back, each
        # names one of its levels.
        contrast=read_group_contrast(
            raw["compared"], variable=raw["variable"], groups=levels
        ),
        groupings_record_sha256=digest,
    )


def evaluate_grouping(rules: GroupingRules, table: pa.Table) -> pa.Array:
    """Each row's level code; null for an unmeasured stay that leaves the study."""

    values = {
        summary: _numbers(table, column)
        for summary, column in rules.source_columns.items()
    }
    measured = np.ones(table.num_rows, dtype=bool)
    for array in values.values():
        measured &= ~np.isnan(array)
    codes = rules.codes
    levels = np.zeros(table.num_rows, dtype=np.int64)
    unassigned = measured.copy()
    for group_id, rule in rules.groups:
        if rule is None:
            taken = unassigned
        else:
            with np.errstate(invalid="ignore"):
                taken = unassigned & _COMPARE[rule.op](values[rule.summary], rule.value)
        levels[taken] = codes[group_id]
        unassigned &= ~taken
    if unassigned.any():
        raise ExposureGroupRuleError(
            f"{int(unassigned.sum())} measured stays meet no group of the grouping"
        )
    if rules.unmeasured == "own_group":
        levels[~measured] = codes[UNMEASURED_GROUP_ID]
        return pa.array(levels, type=pa.int64())
    return pa.array(levels, type=pa.int64(), mask=~measured)


def _numbers(table: pa.Table, column: str) -> np.ndarray:
    """The column's values as floats; a missing value is NaN."""

    if column not in table.column_names:
        raise ExposureGroupRuleError(f"the cohort holds no column {column!r}")
    values = table.column(column)
    kind = values.type
    if not (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_decimal(kind)
    ):
        raise ExposureGroupRuleError(
            f"{column!r} holds {kind}, which no threshold reads"
        )
    floats = pc.cast(values, pa.float64()).combine_chunks()
    return np.asarray(floats.to_numpy(zero_copy_only=False), dtype=np.float64)


__all__ = [
    "EXPOSURE_GROUP_CONTRASTS_KEY",
    "EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID",
    "EXPOSURE_GROUP_TRANSFORMS",
    "EXPOSURE_GROUP_TRANSFORM_ID",
    "GROUPING_PARAMETER_KEYS",
    "GROUP_COMPARISON_KEYS",
    "PLANNED_GROUP_COLUMN_KEYS",
    "UNMEASURED_GROUP_ID",
    "ExposureGroupRuleError",
    "GroupContrast",
    "GroupThreshold",
    "GroupingRules",
    "PlannedGroupColumn",
    "evaluate_grouping",
    "grouping_level_description",
    "planned_group_column",
    "read_group_contrast",
    "read_grouping_rules",
    "read_planned_group_columns",
]
