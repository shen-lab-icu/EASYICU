"""The names a grouped exposure's levels go by in figures, tables and the draft.

A grouping's column holds level codes.  Each level is shown under one English
reader name, which the plan states as its display label ``<column>=<code>``
when it is compiled, so the figures, the tables and the manuscript name a
level alike and the plan a reviewer approves shows the names it seals.

A name is the study's own words for the group when every level of the
grouping has words a reader sentence can carry -- English, short, no markup
and no number the claim binder would have to trace (``reader_claim_labels``)
-- and no other grouping uses the same words.  Otherwise every level of the
grouping goes by a neutral name: ``"<Grouping> group <code>"``, an ordinal
grouping's first and last level saying ``(lowest)`` and ``(highest)`` as the
grouping's scale numbers them, and the unmeasured level ``"<Grouping> not
measured"``, where ``<Grouping>`` is the grouping's name for readers
(``grouping_reader_labels``: its concept's name, with its window or as its
column when another grouping shares it).  Nothing here gives a group a
meaning its record does not.

The levels are read from the context: the context builder states them from
the grouping's sealed declaration (``exposure_group_labels``).  These keys
are the host's: a label the Planner wrote for one of them is replaced.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    UNMEASURED_GROUP_ID,
    ExposureGroupRuleError,
    grouping_reader_labels,
    read_level_labels,
)
from ..reporting.manuscript_labels import reader_claim_labels
from ..schema import ResearchContext, ValidationFinding

#: The longest of the study's words a reader name carries.
_MAX_NAME = 40


class ExposureGroupNamesUnreadable(ValueError):
    """The context states its groupings' levels in a shape no name is read from."""


def exposure_group_reader_names(context: ResearchContext) -> dict[str, str]:
    """Each grouped level's reader name, keyed ``<column>=<code>``, and each
    grouping column's own name, keyed ``<column>``; empty without a grouping.

    Raises :class:`ExposureGroupNamesUnreadable` when the context states its
    levels in another shape.
    """

    groupings = _groupings(context)
    grouping_names = grouping_reader_labels(groupings)
    words = {item["variable"]: _study_words(item) for item in groupings}
    used: dict[str, int] = {}
    for named in words.values():
        for word in set(named or ()):
            used[word] = used.get(word, 0) + 1
    names: dict[str, str] = {}
    for item in groupings:
        variable = item["variable"]
        concept = grouping_names[variable]
        own = words[variable]
        if own is not None and all(used[word] == 1 for word in own):
            levels = {level["code"]: level["label"].strip() for level in item["levels"]}
        else:
            levels = _neutral(item, concept)
        for code, name in levels.items():
            names[f"{variable}={code}"] = name
        names[variable] = f"{concept} group"
    return names


def exposure_group_display_labels(
    context: ResearchContext, planner_labels: Mapping[str, str] | Sequence[Any]
) -> dict[str, str]:
    """``planner_labels`` with every grouped level named by the host.

    ``planner_labels`` is a mapping, or the outline's label items (each with
    a ``key`` and a ``value``).

    A label the Planner wrote for a level of a grouping, or for a code the
    grouping does not have, is replaced or dropped; a grouping column the
    Planner named keeps its name.  A context whose levels cannot be read
    names none (:func:`exposure_group_label_findings` reports it).
    """

    if not isinstance(planner_labels, Mapping):
        planner_labels = {item.key: item.value for item in planner_labels}
    try:
        names = exposure_group_reader_names(context)
    except ExposureGroupNamesUnreadable:
        return dict(planner_labels)
    grouped = {key for key in names if "=" not in key}
    labels = {
        key: value
        for key, value in planner_labels.items()
        if key.partition("=")[0] not in grouped or "=" not in key
    }
    for key, name in names.items():
        if "=" in key or key not in labels:
            labels[key] = name
    return labels


def replaced_planner_labels(
    context: ResearchContext, planner_labels: Mapping[str, str]
) -> tuple[str, ...]:
    """The Planner's level labels of a grouping the host's names replaced."""

    try:
        names = exposure_group_reader_names(context)
    except ExposureGroupNamesUnreadable:
        return ()
    grouped = {key for key in names if "=" not in key}
    return tuple(
        sorted(
            key
            for key, value in planner_labels.items()
            if "=" in key
            and key.partition("=")[0] in grouped
            and names.get(key) != value
        )
    )


def exposure_group_label_findings(
    context: ResearchContext, planner_labels: Mapping[str, str] | Sequence[Any]
) -> list[ValidationFinding]:
    """What the host's names did to the plan's labels, as warnings.

    A context whose grouped levels cannot be read leaves the plan showing
    their codes; a level label the Planner wrote is replaced by the host's.
    """

    if not isinstance(planner_labels, Mapping):
        planner_labels = {item.key: item.value for item in planner_labels}
    try:
        exposure_group_reader_names(context)
    except ExposureGroupNamesUnreadable as exc:
        return [
            ValidationFinding(
                validator="exposure_group_reader_names",
                severity="warning",
                message=(
                    "The context states its exposure groupings' levels in a shape "
                    "no reader name is read from, so the plan shows their codes."
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason_code": "exposure_group_reader_names_unreadable",
                    "error": str(exc)[:300],
                },
            )
        ]
    replaced = replaced_planner_labels(context, planner_labels)
    if not replaced:
        return []
    return [
        ValidationFinding(
            validator="exposure_group_reader_names",
            severity="warning",
            message=(
                "The plan names each level of a grouped exposure as its grouping "
                "record does; the outline's labels for "
                + ", ".join(replaced)
                + " were replaced."
            ),
            evidence_ids=["analysis_plan"],
            detail={
                "reason_code": "exposure_group_planner_labels_replaced",
                "keys": list(replaced),
            },
        )
    ]


def _groupings(context: ResearchContext) -> list[dict[str, Any]]:
    raw = context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ExposureGroupNamesUnreadable("the context's grouped levels are not a list")
    groupings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) - {"parameters"} != {
            "variable",
            "concept",
            "scale",
            "levels",
        }:
            raise ExposureGroupNamesUnreadable(
                "a grouping's levels hold its variable, concept, scale, levels "
                "and, when it states them, its parameters"
            )
        variable, concept, scale = item["variable"], item["concept"], item["scale"]
        if (
            not isinstance(variable, str)
            or not variable
            or "=" in variable
            or variable in seen
            or not isinstance(concept, str)
            or not concept
            or scale not in ("nominal", "ordinal")
        ):
            raise ExposureGroupNamesUnreadable("a grouping names its column once")
        levels = item["levels"]
        try:
            read = read_level_labels(
                levels, levels=len(levels) if isinstance(levels, list) else -1
            )
        except ExposureGroupRuleError as exc:
            raise ExposureGroupNamesUnreadable(str(exc)) from exc
        if len(read) < 2 or (
            scale == "ordinal" and read[-1]["group"] == UNMEASURED_GROUP_ID
        ):
            raise ExposureGroupNamesUnreadable("a grouping's levels form no grouping")
        seen.add(variable)
        groupings.append(
            {
                "variable": variable,
                "concept": concept,
                "scale": scale,
                "levels": read,
                **(
                    {"parameters": item["parameters"]}
                    if isinstance(item.get("parameters"), Mapping)
                    else {}
                ),
            }
        )
    return groupings


def _study_words(item: Mapping[str, Any]) -> Optional[tuple[str, ...]]:
    """The study's words for each level, compared caseless, when a reader
    sentence can carry every one of them; ``None`` otherwise."""

    words = [level["label"].strip() for level in item["levels"]]
    keys = {f"level_{index}": word for index, word in enumerate(words)}
    carried = reader_claim_labels(None, keys)
    if len(carried) != len(keys) or any(len(word) > _MAX_NAME for word in words):
        return None
    folded = tuple(" ".join(word.split()).casefold() for word in words)
    return folded if len(set(folded)) == len(folded) else None


def _neutral(item: Mapping[str, Any], concept: str) -> dict[int, str]:
    groups = [level for level in item["levels"] if level["group"] != UNMEASURED_GROUP_ID]
    names: dict[int, str] = {}
    for level in item["levels"]:
        code = level["code"]
        if level["group"] == UNMEASURED_GROUP_ID:
            names[code] = f"{concept} not measured"
            continue
        name = f"{concept} group {code}"
        if item["scale"] == "ordinal" and code == groups[0]["code"]:
            name += " (lowest)"
        elif item["scale"] == "ordinal" and code == groups[-1]["code"]:
            name += " (highest)"
        names[code] = name
    return names


__all__ = [
    "ExposureGroupNamesUnreadable",
    "exposure_group_display_labels",
    "exposure_group_label_findings",
    "exposure_group_reader_names",
    "replaced_planner_labels",
]
