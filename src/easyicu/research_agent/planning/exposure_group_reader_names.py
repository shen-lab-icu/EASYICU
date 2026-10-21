"""The names a grouped exposure's levels go by in figures, tables and the draft.

A grouping's column holds level codes.  Each level is shown under one English
reader name, which the plan states as its display label ``<column>=<code>``
when it is compiled, so the figures and the tables name a level alike and the
plan a reviewer approves seals the names they show.

A name is the study's own words for the group when every level of the
grouping has words a reader sentence can carry -- English, short, no markup
and no number the claim binder would have to trace (``reader_claim_labels``)
-- and no other grouping uses the same words.  Otherwise every level of the
grouping goes by the rule the host forms it with, written in English from the
grouping's sealed derivation (``grouping_rule_texts``) with its comparison
signs in words: "Minimum below 70 mg/dL", "Neither minimum below 70 mg/dL
nor maximum above 180 mg/dL", "No value recorded" -- the words the draft's
host sentences name such a level by.  A level is never named by its place
("<Grouping> group 2"): nothing here gives a group a meaning its record does
not, and no name hides the rule that forms it.

A rule states its thresholds, and a threshold the numeric binder traces
(180, 2.5) cannot stand in a prose sentence without provenance.  The Writer
is shown the study's own words for such a grouping's levels instead
(:func:`writer_level_labels`), and a host claim sentence names a level only
by a name that needs none (``reader_claim_labels``).

The plan reads the levels from the context: the context builder states them
from the grouping's sealed declaration (``exposure_group_labels``).  These
keys are the host's: a label the Planner wrote for one of them is replaced.
Whoever names the levels of a run's grouping record calls
:func:`grouping_level_reader_names` with the same levels and derivations, so
the run's readers (``orchestration.exposure_group_labels``, and the web
projections built on it) name a level as its figures do.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..authority.evidence_store import prose_requires_numeric_provenance
from ..contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    GROUPING_RULE_WORDING_VERSION,
    UNMEASURED_GROUP_ID,
    ExposureGroupRuleError,
    grouping_reader_labels,
    grouping_rule_texts,
    read_level_labels,
    reads_as_markup,
)
from ..reporting.manuscript_labels import reader_claim_labels
from ..schema import ResearchContext, ValidationFinding

#: The longest of the study's words a reader name carries.
_MAX_NAME = 40
#: The longest rule a reader name carries: within the plan's display-label
#: cap and the web projection's, so no reader shows a shortened rule.
_MAX_RULE_NAME = 240
#: A rule's comparison signs as a reader sentence words them, the two-glyph
#: forms first.
_SIGN_WORDS = (("≤", "at most"), ("≥", "at least"), ("<", "below"), (">", "above"))
#: What the context states of each grouping, and what it may state beside.
_GROUPING_KEYS = frozenset({"variable", "concept", "scale", "levels"})
_OPTIONAL_GROUPING_KEYS = frozenset({"parameters", "rule_wording_version"})


class ExposureGroupNamesUnreadable(ValueError):
    """The context states its groupings' levels in a shape no name is read from."""


def rule_reader_name(rule_text: str) -> str:
    """A level's rule as its reader name: comparison signs in words, the
    first letter capitalised ("minimum < 70 mg/dL" is "Minimum below 70 mg/dL")."""

    text = " ".join(str(rule_text).split())
    for sign, word in _SIGN_WORDS:
        text = text.replace(sign, word)
    return text[:1].upper() + text[1:]


def grouping_level_reader_names(
    groupings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, str]]:
    """Each grouping's English reader name for each of its level codes, keyed
    by the grouping's column.

    Each of ``groupings`` holds its column (``variable``), its levels in code
    order (each a ``code``, a ``group`` id and the study's ``label``, empty
    when the record states none) and, when it states them, its derivation
    ``parameters``.  A level goes unnamed only when its grouping's words are
    not carried and its rule cannot be written from ``parameters``, would
    read as markup or would be longer than a reader name can be.
    """

    words = {item["variable"]: _study_words(item["levels"]) for item in groupings}
    used: dict[str, int] = {}
    for named in words.values():
        for word in set(named or ()):
            used[word] = used.get(word, 0) + 1
    names: dict[str, dict[int, str]] = {}
    for item in groupings:
        own = words[item["variable"]]
        if own is not None and all(used[word] == 1 for word in own):
            names[item["variable"]] = {
                level["code"]: level["label"].strip() for level in item["levels"]
            }
        else:
            names[item["variable"]] = _rule_names(item)
    return names


def exposure_group_reader_names(context: ResearchContext) -> dict[str, str]:
    """Each grouped level's reader name, keyed ``<column>=<code>``, and each
    grouping column's own name, keyed ``<column>``; empty without a grouping.

    Raises :class:`ExposureGroupNamesUnreadable` when the context states its
    levels in another shape, or a level can be named neither by the study's
    words nor by its rule.
    """

    groupings = _groupings(context)
    grouping_names = grouping_reader_labels(groupings)
    level_names = grouping_level_reader_names(groupings)
    names: dict[str, str] = {}
    for item in groupings:
        variable = item["variable"]
        named = level_names[variable]
        if set(named) != {level["code"] for level in item["levels"]}:
            raise ExposureGroupNamesUnreadable(
                "a grouping whose words no reader sentence carries states no "
                "rule each of its levels is named by"
            )
        for code, name in named.items():
            names[f"{variable}={code}"] = name
        names[variable] = f"{grouping_names[variable]} group"
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


def writer_level_labels(
    context: Optional[ResearchContext], labels: Mapping[str, str]
) -> dict[str, str]:
    """``labels`` as the Writer is shown them.

    When the name of a grouped level would carry a number the binder traces
    into a prose sentence -- its rule's thresholds -- every level of that
    grouping goes by the study's own words for it, which the Writer renders in
    the manuscript's language; a level whose words carry one too, or read as
    markup, is left out, so the Writer cites it as ``level <code>``.  Every
    other label is shown as it is.
    """

    try:
        groupings = _groupings(context) if context is not None else []
    except ExposureGroupNamesUnreadable:
        return dict(labels)
    words: dict[str, str] = {}
    for item in groupings:
        keys = {
            f"{item['variable']}={level['code']}": " ".join(level["label"].split())
            for level in item["levels"]
        }
        if any(prose_requires_numeric_provenance(labels.get(key, "")) for key in keys):
            words.update(keys)
    shown: dict[str, str] = {}
    for key, value in labels.items():
        if key not in words:
            shown[key] = value
            continue
        own = words[key]
        if own and not prose_requires_numeric_provenance(own) and not reads_as_markup(own):
            shown[key] = own
    return shown


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
        if (
            not isinstance(item, Mapping)
            or not _GROUPING_KEYS <= set(item)
            or set(item) - _GROUPING_KEYS - _OPTIONAL_GROUPING_KEYS
            or item.get("rule_wording_version", 1) not in (1, GROUPING_RULE_WORDING_VERSION)
        ):
            raise ExposureGroupNamesUnreadable(
                "a grouping's levels hold its variable, concept, scale, levels "
                "and, when it states them, its parameters and their rule wording"
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


def _study_words(levels: Sequence[Mapping[str, Any]]) -> Optional[tuple[str, ...]]:
    """The study's words for each level, compared caseless, when a reader
    sentence can carry every one of them; ``None`` otherwise."""

    words = [
        level["label"].strip() if isinstance(level.get("label"), str) else ""
        for level in levels
    ]
    keys = {f"level_{index}": word for index, word in enumerate(words)}
    carried = reader_claim_labels(None, keys)
    if len(carried) != len(keys) or any(len(word) > _MAX_NAME for word in words):
        return None
    folded = tuple(" ".join(word.split()).casefold() for word in words)
    return folded if len(set(folded)) == len(folded) else None


def _rule_names(item: Mapping[str, Any]) -> dict[int, str]:
    """Each level's rule as its name, from the grouping's derivation."""

    parameters = item.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    try:
        rules = grouping_rule_texts(parameters, include_measure=False)
    except (ExposureGroupRuleError, KeyError, TypeError):
        return {}
    names: dict[int, str] = {}
    for level in item["levels"]:
        text = rules.get(level["group"])
        name = rule_reader_name(text) if text else ""
        if name and not reads_as_markup(name) and len(name) <= _MAX_RULE_NAME:
            names[level["code"]] = name
    return names


__all__ = [
    "ExposureGroupNamesUnreadable",
    "exposure_group_display_labels",
    "exposure_group_label_findings",
    "exposure_group_reader_names",
    "grouping_level_reader_names",
    "replaced_planner_labels",
    "rule_reader_name",
    "writer_level_labels",
]
