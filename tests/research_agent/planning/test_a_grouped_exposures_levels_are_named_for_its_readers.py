"""A grouped exposure's level codes are named for whoever shows them.

The column a grouping forms holds codes.  A reader names each code by the
study's own words for its group, held in the run's grouping record, and by
the rule the host forms it with, written from the record's derivation.  A
level the record gives no words is its code alone, and a record that cannot
be read is reported, never shown as no grouping.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.concept import catalog as concept_catalog
from easyicu.research_agent.orchestration.exposure_group_labels import (
    exposure_group_label_rows,
    recorded_exposure_group_labels,
)
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    EXPOSURE_GROUPINGS_RECORD_SCHEMA,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    compile_exposure_groupings,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    read_stated_exposure_groupings,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
)

_DAY = {"start_hours": 0, "end_hours": 24}


def _rule(summary: str, op: str, value: float, unit: str | None = None) -> dict:
    return {
        "summary": summary,
        "op": op,
        "value": value,
        **({"unit": unit} if unit else {}),
    }


def _glucose(**changes: Any) -> dict:
    return {
        "id": "x1",
        "concept": "glu",
        "window": _DAY,
        "scale": "nominal",
        "groups": [
            {"id": "g1", "label": "low", "rule": _rule("min", "<", 70, "mg/dL")},
            {"id": "g2", "label": "high only", "rule": _rule("max", ">", 180, "mg/dL")},
            {"id": "g3", "label": "in range", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "own_group", "label": "not measured"},
        "quote": "glucose below 70 or above 180",
        "source": "question",
        **changes,
    }


def _bmi() -> dict:
    return {
        "id": "x2",
        "concept": "bmi",
        "window": None,
        "scale": "ordinal",
        # Matched in this order; the codes follow the scale.
        "groups": [
            {"id": "g1", "label": "under 18.5", "rule": _rule("value", "<", 18.5)},
            {"id": "g3", "label": "30 or more", "rule": _rule("value", ">=", 30)},
            {"id": "g2", "label": "18.5 to 29.9", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "exclude"},
        "reference": "g2",
        "contrast": "g3",
        "quote": "body mass index category",
        "source": "outline",
    }


def _lab(name: str, summary: str) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=VariableRole.LAB,
        dtype="float64",
        unit="mg/dL",
        source_concept="glu",
        unit_normalization=f"window_numeric_{summary}",
        analysis_window="icu_admission[0,24]h",
        valid_range=[0, 1000],
    )


_BMI = ConceptDescriptor(
    name="bmi",
    role=VariableRole.DEMOGRAPHIC,
    dtype="float64",
    unit="kg/m2",
    source_concept="bmi",
    unit_normalization="stay_level_unique_value",
    valid_range=[8, 70],
)


def _context() -> ResearchContext:
    base = _planner_context()
    constraints = json.dumps(
        {
            "materialization_window": {
                "role": "outer_observation_window",
                "anchor": "ICU admission",
                "hours": 24.0,
            }
        }
    )
    return base.model_copy(
        update={
            "variables": [
                *base.variables,
                _lab("glu_min", "min"),
                _lab("glu_max", "max"),
                _BMI,
            ],
            "data_constraints": constraints,
        }
    )


def _record(*specs: dict) -> dict:
    stated = read_stated_exposure_groupings({"groupings": list(specs)})
    compiled = compile_exposure_groupings(stated, _context())
    return {
        "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
        "exposure_grouping_enabled": True,
        "planner": None,
        "compiled": compiled.record(),
        "compiled_sha256": compiled.sha256(),
    }


def _write(run_dir: Path, record: Any) -> str:
    raw = (record if isinstance(record, str) else json.dumps(record)).encode("utf-8")
    (run_dir / EXPOSURE_GROUPINGS_FILENAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_each_code_is_named_by_the_studys_words_and_the_hosts_rule(
    tmp_path: Path,
) -> None:
    digest = _write(tmp_path, _record(_glucose(), _bmi()))
    glucose = "Glucose over 0-24 h after ICU admission"
    bmi = f"{concept_catalog.CONCEPT_DICTIONARY['bmi'][0]}, recorded once per stay"

    nominal, ordinal = recorded_exposure_group_labels(tmp_path)

    assert nominal == {
        "variable": "glu_group_x1",
        "concept": "glu",
        "scale": "nominal",
        "status": "labelled",
        "levels": [
            {
                "code": 1,
                "group": "g1",
                "label": "low",
                "rule": f"{glucose}: minimum < 70 mg/dL",
                "unmeasured": False,
            },
            {
                "code": 2,
                "group": "g2",
                "label": "high only",
                "rule": f"{glucose}: maximum > 180 mg/dL, and none of minimum < 70 mg/dL",
                "unmeasured": False,
            },
            {
                "code": 3,
                "group": "g3",
                "label": "in range",
                "rule": f"{glucose}: none of minimum < 70 mg/dL; maximum > 180 mg/dL",
                "unmeasured": False,
            },
            {
                "code": 4,
                "group": "gU",
                "label": "not measured",
                "rule": f"{glucose}: no value recorded",
                "unmeasured": True,
            },
        ],
        # Unstated: the first group and the last other one, never unmeasured.
        "reference": 1,
        "contrast": 3,
        "groupings_record_sha256": digest,
    }
    # An ordinal grouping's codes follow its scale; each rule follows the
    # order the groups are matched in.
    assert [(row["code"], row["group"], row["rule"]) for row in ordinal["levels"]] == [
        (1, "g1", f"{bmi}: value < 18.5"),
        (2, "g2", f"{bmi}: none of value < 18.5; value ≥ 30"),
        (3, "g3", f"{bmi}: value ≥ 30, and none of value < 18.5"),
    ]
    assert (ordinal["scale"], ordinal["reference"], ordinal["contrast"]) == (
        "ordinal",
        2,
        3,
    )


def test_a_level_without_words_is_its_code_alone(tmp_path: Path) -> None:
    record = _record(_glucose())
    (grouping,) = record["compiled"]["groupings"]
    del grouping["labels"]["g2"]
    _write(tmp_path, record)

    (row,) = recorded_exposure_group_labels(tmp_path)

    assert row["status"] == "codes_only"
    assert [level["label"] for level in row["levels"]] == [
        "low",
        None,
        "in range",
        "not measured",
    ]


def test_only_a_grouping_the_host_applied_is_named(tmp_path: Path) -> None:
    record = _record(_glucose(), _bmi())
    record["compiled"]["groupings"][0]["disposition"] = "not_applied"
    _write(tmp_path, record)

    assert [row["variable"] for row in recorded_exposure_group_labels(tmp_path)] == [
        "bmi_group_x2"
    ]


def test_a_run_that_planned_no_grouping_names_none(tmp_path: Path) -> None:
    assert recorded_exposure_group_labels(tmp_path) == []

    _write(
        tmp_path,
        {
            "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
            "exposure_grouping_enabled": True,
            "not_asked": "capability_review_first",
        },
    )
    assert recorded_exposure_group_labels(tmp_path) == []


def test_a_record_that_cannot_be_read_is_reported_not_taken_for_none(
    tmp_path: Path,
) -> None:
    record = _record(_glucose())

    for unreadable in (
        "{not json",
        "[]",
        {**record, "schema_version": "easyicu.exposure_groupings_record/0"},
        {**record, "compiled": {"groupings": "x1"}},
    ):
        _write(tmp_path, unreadable)
        assert recorded_exposure_group_labels(tmp_path) is None

    broken = json.loads(json.dumps(record))
    broken["compiled"]["groupings"][0]["compared"] = None
    _write(tmp_path, broken)
    assert recorded_exposure_group_labels(tmp_path) is None
    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).unlink()
    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).mkdir()
    assert recorded_exposure_group_labels(tmp_path) is None
    # A link is not the record the run wrote, even to a readable one.
    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write(elsewhere, record)
    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).symlink_to(
        elsewhere / EXPOSURE_GROUPINGS_FILENAME
    )
    assert recorded_exposure_group_labels(tmp_path) is None


def test_the_rows_are_read_from_the_record_alone() -> None:
    record = _record(_glucose())

    rows = exposure_group_label_rows(record, record_sha256="a" * 64)

    assert [row["groupings_record_sha256"] for row in rows] == ["a" * 64]
    assert rows == exposure_group_label_rows(
        json.loads(json.dumps(record)), record_sha256="a" * 64
    )
