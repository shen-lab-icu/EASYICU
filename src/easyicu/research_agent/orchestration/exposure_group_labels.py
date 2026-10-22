"""What a grouped exposure's level codes mean to its readers.

A grouping's column holds level codes.  Whoever shows them -- the plan a
reviewer approves, a result table, a preview -- names each code by the
study's own words for its group (the label the grouping record holds) and by
the rule the host forms the group with, written in English from the record's
derivation by the grouping contract (``grouping_rule_texts``), never by a
model.  A level without a label is shown as its code alone: nothing here
gives a code a meaning the record does not.

Each level also carries its English reader name (``reader_name``): the name
its figures and tables show, written by the plan's own naming owner
(``planning.exposure_group_reader_names``) from the same words and
derivations, so a reader that shows an English page names a level as the
run's figures do.  ``None`` when the level can be named neither way.

Labels and rules are read from the run's grouping record with its digest.
The caller supplies the reviewed/current plan for contrast direction; without
one, only explicit study choices are marked. A default is never a study choice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ..contracts.exposure_group_rules import (
    UNMEASURED_GROUP_ID,
    ExposureGroupRuleError,
    grouping_rule_texts,
    read_group_contrast,
    read_grouping_rules,
)
from ..planning.exposure_group_compile import planned_exposure_contrasts
from ..planning.exposure_group_reader_names import grouping_level_reader_names
from ..schema import AnalysisPlan
from .exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    EXPOSURE_GROUPINGS_RECORD_SCHEMA,
)

_MAX_RECORD_BYTES = 2 * 1024 * 1024


class ExposureGroupLabelError(ValueError):
    """A grouping record whose levels cannot be named."""


def exposure_group_label_rows(
    record: Mapping[str, Any], *, record_sha256: str,
    plan: AnalysisPlan | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Each applied grouping of ``record``: its level codes and what each names.

    ``record_sha256`` is the digest of the record's bytes, which every row
    carries.  A record that asked nothing has no row.
    """

    if record.get("schema_version") != EXPOSURE_GROUPINGS_RECORD_SCHEMA:
        raise ExposureGroupLabelError("not a grouping record")
    if record.get("not_asked") is not None:
        return []
    compiled = record.get("compiled")
    groupings = compiled.get("groupings") if isinstance(compiled, Mapping) else None
    if not isinstance(groupings, list):
        raise ExposureGroupLabelError("a grouping record lists its groupings")
    applied = [
        item
        for item in groupings
        if isinstance(item, Mapping) and item.get("disposition") == "applied"
    ]
    pairs = planned_exposure_contrasts(
        plan or {}, variables=[item.get("variable") for item in applied],
    )
    rows = [
        _row(item, record_sha256=record_sha256, pair=pairs.get(item.get("variable")))
        for item in applied
    ]
    names = grouping_level_reader_names([
        {
            "variable": row["variable"],
            "levels": [
                {**level, "label": level["label"] or ""} for level in row["levels"]
            ],
            "parameters": item.get("derivation"),
        }
        for row, item in zip(rows, applied)
    ])
    for row in rows:
        named = names[row["variable"]]
        for level in row["levels"]:
            level["reader_name"] = named.get(level["code"])
    return rows


def recorded_exposure_group_labels(
    run_dir: Path, *, plan: AnalysisPlan | Mapping[str, Any] | None = None,
) -> Optional[list[dict[str, Any]]]:
    """The label rows of the grouping record a run wrote.

    ``[]`` when the run wrote none: it planned no grouping.  ``None`` when its
    record cannot be read, which a reader reports instead of showing no
    grouping.
    """

    path = Path(run_dir) / EXPOSURE_GROUPINGS_FILENAME
    if not path.exists() and not path.is_symlink():
        return []
    try:
        if path.is_symlink():
            raise ExposureGroupLabelError("the grouping record is a link")
        raw = path.read_bytes()
        if len(raw) > _MAX_RECORD_BYTES:
            raise ExposureGroupLabelError("the grouping record is too large")
        record = json.loads(raw)
        if not isinstance(record, Mapping):
            raise ExposureGroupLabelError("the grouping record is not an object")
        return exposure_group_label_rows(
            record, record_sha256=hashlib.sha256(raw).hexdigest(), plan=plan,
        )
    except (OSError, ValueError):
        return None


def _row(
    item: Mapping[str, Any], *, record_sha256: str,
    pair: tuple[str, str] | None,
) -> dict[str, Any]:
    derivation = item.get("derivation")
    try:
        rules = read_grouping_rules(derivation)
        written = grouping_rule_texts(derivation)
    except ExposureGroupRuleError as exc:
        raise ExposureGroupLabelError(str(exc)) from exc
    labels = item.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    variable = item.get("variable")
    if not isinstance(variable, str) or not variable:
        raise ExposureGroupLabelError("an applied grouping names its column")
    read_group_contrast(item.get("compared"), variable=variable, groups=len(rules.groups))
    # The preplan default is a proposal, not a study choice. The reviewed
    # executable plan supplies both sides when it actually reports a contrast.
    measured_groups = {group for group, _ in rules.groups}
    if any(
        item.get(key) is not None and (
            not isinstance(item[key], str) or item[key] not in measured_groups
        )
        for key in ("reference", "contrast")
    ):
        raise ExposureGroupLabelError("a stated comparison names a declared group")
    chosen = {key: rules.codes[item[key]] if item.get(key) else None
              for key in ("reference", "contrast")}
    if pair is not None:
        measured = {str(rules.codes[group]) for group in measured_groups}
        if not set(pair) <= measured or pair[0] == pair[1]:
            raise ExposureGroupLabelError("the plan compares two measured groups")
        for index, key in enumerate(("reference", "contrast")):
            if chosen[key] is not None and str(chosen[key]) != pair[index]:
                raise ExposureGroupLabelError("the plan changes a stated grouped contrast")
            chosen[key] = int(pair[index])
    levels = [
        {
            "code": rules.codes[level],
            "group": level,
            "label": _label(labels.get(level)),
            "rule": written[level],
            "unmeasured": level == UNMEASURED_GROUP_ID,
        }
        for level in rules.levels
    ]
    return {
        "variable": variable,
        "concept": rules.concept,
        "scale": rules.scale,
        "status": (
            "labelled"
            if all(level["label"] is not None for level in levels)
            else "codes_only"
        ),
        "levels": levels,
        **chosen,
        # The measured column each summary its rules read is taken from: what
        # a study prepares to form these groups again.
        "source_columns": dict(rules.source_columns),
        "groupings_record_sha256": record_sha256,
    }


def _label(raw: Any) -> Optional[str]:
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


__all__ = [
    "ExposureGroupLabelError",
    "exposure_group_label_rows",
    "recorded_exposure_group_labels",
]
