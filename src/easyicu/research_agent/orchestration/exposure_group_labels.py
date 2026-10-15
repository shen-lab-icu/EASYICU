"""What a grouped exposure's level codes mean to its readers.

A grouping's column holds level codes.  Whoever shows them -- the plan a
reviewer approves, a result table, a preview -- names each code by the
study's own words for its group (the label the grouping record holds) and by
the rule the host forms the group with, written in English from the record's
derivation by the grouping contract (``grouping_rule_texts``), never by a
model.  A level without a label is shown as its code alone: nothing here
gives a code a meaning the record does not.

The rows are read from the run's grouping record (``exposure_groupings.json``)
each time, with its digest, so every reader shows the record the run sealed.
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
    read_grouping_rules,
)
from .exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    EXPOSURE_GROUPINGS_RECORD_SCHEMA,
)

_MAX_RECORD_BYTES = 2 * 1024 * 1024


class ExposureGroupLabelError(ValueError):
    """A grouping record whose levels cannot be named."""


def exposure_group_label_rows(
    record: Mapping[str, Any], *, record_sha256: str
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
    return [
        _row(item, record_sha256=record_sha256)
        for item in groupings
        if isinstance(item, Mapping) and item.get("disposition") == "applied"
    ]


def recorded_exposure_group_labels(run_dir: Path) -> Optional[list[dict[str, Any]]]:
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
            record, record_sha256=hashlib.sha256(raw).hexdigest()
        )
    except (OSError, ValueError):
        return None


def _row(item: Mapping[str, Any], *, record_sha256: str) -> dict[str, Any]:
    derivation = item.get("derivation")
    try:
        rules = read_grouping_rules(derivation)
        written = grouping_rule_texts(derivation)
    except ExposureGroupRuleError as exc:
        raise ExposureGroupLabelError(str(exc)) from exc
    labels = item.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    compared = item.get("compared")
    if not isinstance(compared, Mapping) or set(compared) != {"reference", "contrast"}:
        raise ExposureGroupLabelError("an applied grouping states its comparison")
    variable = item.get("variable")
    if not isinstance(variable, str) or not variable:
        raise ExposureGroupLabelError("an applied grouping names its column")
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
        "reference": int(compared["reference"]),
        "contrast": int(compared["contrast"]),
        "groupings_record_sha256": record_sha256,
    }


def _label(raw: Any) -> Optional[str]:
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


__all__ = [
    "ExposureGroupLabelError",
    "exposure_group_label_rows",
    "recorded_exposure_group_labels",
]
