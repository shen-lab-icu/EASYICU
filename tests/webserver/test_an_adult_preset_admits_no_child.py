"""An adult cohort preset applies the adult age floor.

``adult_first`` always applied age 18 and over. ``adult_all``, the same
population without the one-stay restriction, normalized its minimum age to 0
unless a caller stated it. Copilot saves that preset when a researcher asks
for every adult ICU stay, so the study executed every age under a label that
says adults. The cohort owner now applies the floor for both adult presets.
An export recorded under the old rule may hold children: Data Extraction
reports it as executed under an earlier rule, and the launch refuses it until
the study's cohort is extracted again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import dataio, primary_cohort
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.extraction_handoff import (
    bound_export_mismatches,
    compile_study_cohort,
)
from easyicu.webserver.research_launch_scientific import (
    _require_export_holds_study_cohort,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_OUTDATED = "registered_export_cohort_execution_outdated"


def test_an_adult_preset_admits_no_child() -> None:
    execute = primary_cohort.normalize_execution_cohort

    every_adult_stay = execute({"preset": "adult_all"})
    assert every_adult_stay["age_min"] == 18
    assert every_adult_stay["exclude_readmissions"] is False
    # A stated minimum below the floor is raised to it; a higher one stays.
    assert execute({"preset": "adult_all", "age_min": 0})["age_min"] == 18
    assert execute({"preset": "adult_all", "age_min": 16})["age_min"] == 18
    assert execute({"preset": "adult_all", "age_min": 65})["age_min"] == 65
    # The first-stay preset keeps its floor and its restriction.
    first_stay = execute({"preset": "adult_first", "age_min": 0})
    assert (first_stay["age_min"], first_stay["exclude_readmissions"]) == (18, True)
    # Other presets invent no floor.
    assert execute({"preset": "all_icu"})["age_min"] == 0
    assert execute({"preset": "sepsis3"})["age_min"] == 0

    scope = primary_cohort.normalize_primary_cohort_scope({"preset": "adult_all"})
    assert scope.admission_eligibility["minimum_age_years"] == 18
    assert (
        scope.admission_eligibility["repeated_admission_policy"]
        == "all_icu_admissions"
    )
    # Data Extraction executes, and records, the same floor.
    assert dataio.normalize_export_cohort_contract({"preset": "adult_all"})["age_min"] == 18
    # A confirmed eligibility option replaces either adult preset by its own
    # explicit fields.
    for preset in sorted(primary_cohort.ADULT_COHORT_PRESETS):
        applied = cohort_eligibility.apply_option_to_cohort(
            {"preset": preset}, "no_eligibility_filter"
        )
        assert applied["preset"] == "all_icu", preset


def _recorded(preset: str, age_min: Any) -> dict[str, Any]:
    """A contract as an export manifest records it, with the age it executed."""

    contract = dataio.normalize_export_cohort_contract({"preset": "all_icu"})
    contract["preset"] = preset
    if age_min is None:
        contract.pop("age_min")
    else:
        contract["age_min"] = age_min
    return contract


@pytest.mark.parametrize(
    ("preset", "age_min", "current"),
    [
        ("adult_all", 0, False),
        ("adult_all", 16, False),
        ("adult_all", None, False),
        ("adult_all", 18, True),
        ("adult_all", 65, True),
        ("adult_first", 18, True),
        ("all_icu", 0, True),
    ],
)
def test_an_export_recorded_below_the_adult_floor_is_outdated(
    preset: str, age_min: Any, current: bool
) -> None:
    contract = _recorded(preset, age_min)
    manifest = {
        "cohort_contract": contract,
        "cohort_execution": dataio.export_cohort_execution(contract),
    }

    assert dataio.export_cohort_execution_current(manifest) is current


def _write_export(export_path: Path, contract: dict[str, Any]) -> None:
    export_path.mkdir(parents=True)
    manifest = {
        "schema_version": "easyicu_native_export_v2",
        "database": "eicu",
        "data_path": str(export_path.parent / "raw"),
        "format": "parquet",
        "files": [{"file": "demographics.parquet", "module": "demographics"}],
        "cohort_contract": contract,
        "cohort_execution": dataio.export_cohort_execution(contract),
    }
    (export_path / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_a_study_on_an_export_without_the_floor_extracts_its_cohort_again(
    tmp_path: Path,
) -> None:
    def study(export_path: Path) -> dict[str, Any]:
        return {
            "id": "study-adults",
            "revision": 1,
            "question": "Is admission lactate associated with ICU mortality in adults?",
            "data_source": {"path": str(export_path), "database": "eicu"},
            "cohort": {"preset": "adult_all"},
        }

    before = tmp_path / "before"
    _write_export(before, _recorded("adult_all", 0))
    assert bound_export_mismatches(
        study(before), json.loads((before / "_manifest.json").read_text())
    ) == (_OUTDATED,)
    with pytest.raises(ResearchPipelineRunError) as caught:
        _require_export_holds_study_cohort(study(before), str(before))
    assert caught.value.code == "research_pipeline_export_cohort_mismatch"
    assert caught.value.details == {"mismatch_codes": [_OUTDATED]}

    again = tmp_path / "again"
    _write_export(again, compile_study_cohort(study(again)))
    _require_export_holds_study_cohort(study(again), str(again))
