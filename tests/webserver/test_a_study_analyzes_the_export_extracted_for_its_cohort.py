"""A study analyzes the export that was extracted for the cohort it states.

Data Extraction records in each export manifest the cohort contract the export
was extracted for and the rule that executed it.  A study whose cohort changed
after extraction, or that is bound to an export prepared for another cohort,
must not plan or run on those rows as if they were its population.  The
Copilot extraction handoff owns the answer; the reuse decision and the research
launch read it, and the launch refuses before anything is spent.  A resumed run
keeps the package its sealed plan bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import dataio
from easyicu.webserver import research_pipeline_run_preparation as preparation
from easyicu.webserver.pi_copilot.extraction_handoff import (
    bound_export_mismatches,
    compile_registered_export_handoff,
    compile_study_cohort,
)
from easyicu.webserver.research_launch_scientific import (
    _require_export_holds_study_cohort,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_COHORT_MISMATCH = "registered_export_cohort_mismatch"
_EXECUTION_OUTDATED = "registered_export_cohort_execution_outdated"


def _study(export_path: Path, *, hours: int = 48, **cohort: object) -> dict:
    return {
        "id": "study-lactate",
        "revision": 3,
        "title": "Lactate clearance and hospital mortality",
        "question": "Is early lactate clearance associated with hospital mortality?",
        "data_source": {"path": str(export_path), "database": "miiv"},
        "cohort": {"preset": "all_icu", "age_min": 18, **cohort},
        "modules": ["demographics", "outcome"],
        "time_window": {"observation_hours": hours, "anchor": "ICU admission"},
        "export_format": "parquet",
    }


def _write_export(
    export_path: Path, contract: dict | None, *, record: bool = True
) -> dict:
    raw_path = export_path.parent / f"{export_path.name}_raw"
    raw_path.mkdir(parents=True)
    export_path.mkdir(parents=True)
    manifest: dict = {
        "schema_version": "easyicu_native_export_v2",
        "database": "miiv",
        "data_path": str(raw_path),
        "format": "parquet",
        "files": [
            {"file": "demographics.parquet", "module": "demographics"},
            {"file": "outcome.parquet", "module": "outcome"},
        ],
    }
    if contract is not None:
        manifest["cohort_contract"] = contract
    if contract is not None and record:
        manifest["cohort_execution"] = dataio.export_cohort_execution(contract)
    (export_path / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _extracted_for(study: dict, export_path: Path, *, record: bool = True) -> dict:
    return _write_export(export_path, compile_study_cohort(study), record=record)


def test_an_export_extracted_for_the_study_holds_its_rows(tmp_path: Path) -> None:
    study = _study(tmp_path / "export")
    manifest = _extracted_for(study, tmp_path / "export")

    assert bound_export_mismatches(study, manifest) == ()


@pytest.mark.parametrize(
    "changed",
    [
        {"preset": "sepsis3"},
        {"preset": "icd", "include_diagnoses": ["A41"]},
        {"age_min": 65},
        {"exclude_readmissions": True},
        {"min_icu_los_hours": 24},
    ],
    ids=["concept_population", "diagnosis_population", "age", "first_stay", "stay_length"],
)
def test_a_cohort_changed_after_extraction_is_not_the_exports(
    tmp_path: Path, changed: dict
) -> None:
    export_path = tmp_path / "export"
    manifest = _extracted_for(_study(export_path), export_path)

    assert bound_export_mismatches(_study(export_path, **changed), manifest) == (
        _COHORT_MISMATCH,
    )


def test_a_window_decides_the_rows_only_of_a_concept_population(tmp_path: Path) -> None:
    plain = _extracted_for(_study(tmp_path / "plain", hours=48), tmp_path / "plain")
    concept = _extracted_for(
        _study(tmp_path / "concept", hours=48, preset="sepsis3"), tmp_path / "concept"
    )

    # The window bounds neither rows nor scores; it decides who enters a
    # concept-derived cohort, so only there does a changed window change rows.
    assert bound_export_mismatches(_study(tmp_path / "plain", hours=24), plain) == ()
    assert bound_export_mismatches(
        _study(tmp_path / "concept", hours=24, preset="sepsis3"), concept
    ) == (_COHORT_MISMATCH,)


def test_a_sepsis_definition_differs_by_what_it_executes(tmp_path: Path) -> None:
    export_path = tmp_path / "export"
    manifest = _extracted_for(_study(export_path, preset="sepsis3"), export_path)

    relabelled = _study(
        export_path, preset="sepsis3", sepsis_definition={"runtime_profile": "site_label"}
    )
    any_event = _study(
        export_path, preset="sepsis3", sepsis_definition={"sofa_increase": {"si_window": "any"}}
    )

    assert bound_export_mismatches(relabelled, manifest) == ()
    assert bound_export_mismatches(any_event, manifest) == (_COHORT_MISMATCH,)


def test_an_export_from_before_the_execution_record_agrees_only_where_rows_did(
    tmp_path: Path,
) -> None:
    scored = _study(tmp_path / "scored", hours=48)
    scored_manifest = _extracted_for(scored, tmp_path / "scored", record=False)
    concept = _study(tmp_path / "concept", hours=24, preset="sepsis3")
    concept_manifest = _extracted_for(concept, tmp_path / "concept", record=False)
    plain = _study(tmp_path / "plain", hours=24)
    plain_manifest = _extracted_for(plain, tmp_path / "plain", record=False)

    # Scores over a 48 h window and a concept cohort decided at any time are
    # rows the current rule does not produce; a 24 h plain export is.
    assert bound_export_mismatches(scored, scored_manifest) == (_EXECUTION_OUTDATED,)
    assert bound_export_mismatches(concept, concept_manifest) == (_EXECUTION_OUTDATED,)
    assert bound_export_mismatches(plain, plain_manifest) == ()


def test_the_reuse_decision_reads_the_same_answer(tmp_path: Path) -> None:
    export_path = tmp_path / "export"
    manifest = _extracted_for(_study(export_path), export_path, record=False)
    changed = _study(export_path, preset="sepsis3")

    handoff = compile_registered_export_handoff(
        changed, {"id": "source-1", "path": str(export_path), "database": "miiv"}
    )

    assert handoff.reusable is False
    assert handoff.mismatch_codes == bound_export_mismatches(changed, manifest)
    assert handoff.mismatch_codes == (_COHORT_MISMATCH, _EXECUTION_OUTDATED)


def test_the_launch_refuses_rows_extracted_for_another_cohort(tmp_path: Path) -> None:
    export_path = tmp_path / "export"
    _extracted_for(_study(export_path), export_path)

    _require_export_holds_study_cohort(_study(export_path), str(export_path))
    with pytest.raises(ResearchPipelineRunError) as caught:
        _require_export_holds_study_cohort(
            _study(export_path, preset="sepsis3"), str(export_path)
        )

    assert caught.value.code == "research_pipeline_export_cohort_mismatch"
    assert caught.value.details == {"mismatch_codes": [_COHORT_MISMATCH]}
    assert "easyicu_start_extraction" in str(caught.value)


def test_a_package_recording_no_population_serves_what_the_run_reapplies(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "export"
    _write_export(export_path, None)

    # Typed age and stay bounds are left to the run's predicates: a family
    # template re-applies them, a progressive plan if its Planner writes them.
    _require_export_holds_study_cohort(
        _study(export_path, age_min=65, min_icu_los_hours=24), str(export_path)
    )
    for population in (
        {"preset": "sepsis3"},
        {"preset": "icd", "include_diagnoses": ["A41"]},
    ):
        with pytest.raises(ResearchPipelineRunError) as caught:
            _require_export_holds_study_cohort(
                _study(export_path, **population), str(export_path)
            )
        assert caught.value.details == {
            "mismatch_codes": ["registered_export_cohort_unrecorded"]
        }


def test_a_study_local_prepared_cohort_is_the_studys_whole_input(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "easyicu_export_manifest.json").write_text(
        json.dumps({"entry_mode": "study_local_prepared_cohort"}), encoding="utf-8"
    )

    _require_export_holds_study_cohort(
        _study(prepared, preset="sepsis3"), str(prepared)
    )


def test_a_folder_without_a_prepared_manifest_is_left_to_package_validation(
    tmp_path: Path,
) -> None:
    raw_database = tmp_path / "raw_database"
    raw_database.mkdir()
    (raw_database / "icustays.csv").write_text("stay_id\n", encoding="utf-8")

    _require_export_holds_study_cohort(_study(raw_database), str(raw_database))
    _require_export_holds_study_cohort(
        _study(tmp_path / "missing"), str(tmp_path / "missing")
    )


def test_an_unexecutable_study_cohort_is_refused_by_name(tmp_path: Path) -> None:
    export_path = tmp_path / "export"
    _extracted_for(_study(export_path), export_path)

    with pytest.raises(ResearchPipelineRunError) as caught:
        _require_export_holds_study_cohort(
            _study(export_path, preset="icd"), str(export_path)
        )

    assert caught.value.code == "research_pipeline_export_cohort_invalid"
    assert caught.value.details.get("reason_code")


class _Reached(Exception):
    """The launch passed the cohort check and reached its next stage."""


def _request(export_path: Path, study: dict, **resume: str) -> preparation.ResearchPipelineLaunchRequest:
    return preparation.ResearchPipelineLaunchRequest(
        export_path=str(export_path),
        study_context=study,
        project_root=str(export_path.parent / "workspace"),
        provider={"provider": "openai"},
        provider_environment={"OPENAI_API_KEY": "test"},
        credential_source="pi_verified",
        literature_search_authorized=False,
        plan_revision_source_run_id="",
        execution_resume_source_run_id=resume.get("execution", ""),
        development_resume_source_job_id=resume.get("development", ""),
        budget_mode="planner_canary",
        runner_image=None,
    )


@pytest.fixture
def next_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stop(*_args: object, **_kwargs: object) -> None:
        raise _Reached

    monkeypatch.setattr(preparation, "_neutral_materialization_scope", _stop)
    monkeypatch.delenv(
        "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID", raising=False
    )


def test_a_new_plan_on_another_cohorts_export_stops_before_its_scope(
    tmp_path: Path, next_stage: None
) -> None:
    export_path = tmp_path / "export"
    _extracted_for(_study(export_path), export_path)

    with pytest.raises(_Reached):
        preparation._prepare_scientific_launch(_request(export_path, _study(export_path)))
    with pytest.raises(ResearchPipelineRunError) as caught:
        preparation._prepare_scientific_launch(
            _request(export_path, _study(export_path, age_min=65))
        )

    assert caught.value.code == "research_pipeline_export_cohort_mismatch"


@pytest.mark.parametrize("resume", [{"execution": "run_sealed"}, {"development": "job_sealed"}])
def test_a_resumed_run_keeps_the_package_its_plan_bound(
    tmp_path: Path, next_stage: None, resume: dict
) -> None:
    export_path = tmp_path / "export"
    _extracted_for(_study(export_path), export_path, record=False)

    with pytest.raises(_Reached):
        preparation._prepare_scientific_launch(
            _request(export_path, _study(export_path, age_min=65), **resume)
        )


def test_a_development_resume_named_by_the_environment_keeps_its_package(
    tmp_path: Path, next_stage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "export"
    _extracted_for(_study(export_path), export_path)
    monkeypatch.setenv("EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID", "job_sealed")

    with pytest.raises(_Reached):
        preparation._prepare_scientific_launch(
            _request(export_path, _study(export_path, age_min=65))
        )
