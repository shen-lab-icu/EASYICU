"""A run's exposure groupings, as the plan card and the result readers name them.

A study can group a concept's value into named levels (an
``ExposureGroupSpec``); the analysis models each level by a code.  The host
reads each applied grouping from the run's grouping record with the study's
own label for each code and an English rule it writes from the record
(``orchestration.exposure_group_labels.recorded_exposure_group_labels``:
``[]`` when the run made no grouping, ``None`` when its record cannot be
read).  The plan card and the result readers then name the levels in the
study's words instead of their codes.

This owner keeps a record only when every row is well formed and within the
groupings' caps, and adds the grouped concept's names from the concept
dictionary.  It writes no label: a grouping whose record carries none is
``codes_only`` and is shown by its codes alone.  A record that cannot be read,
or one with a malformed row, is ``unavailable``: the reader says the record
cannot be read instead of showing a plan without groups, and no code is given
a meaning.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

MAX_GROUPINGS = 3
#: Six groups along the scale, and the unmeasured stays' own level.
MAX_LEVELS = 7
UNREADABLE_REASON = "exposure_group_record_unreadable"
_SCALES = ("ordinal", "nominal")
_STATUSES = ("labelled", "codes_only")


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _code(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _level(row: Any, *, labelled: bool) -> Optional[Dict[str, Any]]:
    if not isinstance(row, Mapping) or _code(row.get("code")) is None:
        return None
    return {
        "code": row["code"],
        "group": _text(row.get("group"), 16),
        "label": _text(row.get("label"), 80) if labelled else "",
        "rule": _text(row.get("rule"), 240),
        "unmeasured": row.get("unmeasured") is True,
    }


def _grouping(row: Any) -> Optional[Dict[str, Any]]:
    from easyicu.concept.catalog import CONCEPT_DICTIONARY

    if not isinstance(row, Mapping):
        return None
    scale = row.get("scale")
    status = row.get("status")
    raw_levels = row.get("levels")
    variable = _text(row.get("variable"), 160)
    if (
        not variable
        or scale not in _SCALES
        or status not in _STATUSES
        or not isinstance(raw_levels, (list, tuple))
        or not 2 <= len(raw_levels) <= MAX_LEVELS
    ):
        return None
    labelled = status == "labelled"
    levels = [_level(level, labelled=labelled) for level in raw_levels]
    if any(level is None for level in levels):
        return None
    if labelled and any(not level["label"] for level in levels):
        return None
    codes = [level["code"] for level in levels]
    if len(set(codes)) != len(codes):
        return None
    compared = {
        "reference": _code(row.get("reference")),
        "contrast": _code(row.get("contrast")),
    }
    measured = {level["code"] for level in levels if not level["unmeasured"]}
    if any(code is not None and code not in measured for code in compared.values()):
        return None
    concept = _text(row.get("concept"), 128)
    names = CONCEPT_DICTIONARY.get(concept) or ()
    return {
        "variable": variable,
        "concept": concept,
        "concept_label_en": _text(names[0], 80) if len(names) > 0 else None,
        "concept_label_zh": _text(names[1], 80) if len(names) > 1 else None,
        "scale": scale,
        "status": status,
        "levels": levels,
        **compared,
    }


def project_exposure_groups(rows: Any) -> Optional[Dict[str, Any]]:
    """The readers' view of a run's groupings; ``None`` when it made none.

    ``rows`` is the grouping owner's return: ``[]`` (no grouping), ``None``
    (a record that cannot be read) or the groupings' rows.
    """

    if isinstance(rows, (list, tuple)) and not rows:
        return None
    groupings = (
        [_grouping(row) for row in rows]
        if isinstance(rows, (list, tuple)) and len(rows) <= MAX_GROUPINGS
        else [None]
    )
    variables = [row["variable"] for row in groupings if row is not None]
    if any(row is None for row in groupings) or len(set(variables)) != len(variables):
        return {"status": "unavailable", "reason_code": UNREADABLE_REASON, "groupings": []}
    return {"status": "shown", "reason_code": None, "groupings": groupings}


__all__ = [
    "MAX_GROUPINGS",
    "MAX_LEVELS",
    "UNREADABLE_REASON",
    "project_exposure_groups",
]
