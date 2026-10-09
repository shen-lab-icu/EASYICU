"""A study's approved target trial binds the signed emulation suite.

The ``target_trial_design`` section is the host's: a browser or model write
cannot carry it, the host's design write carries no approval, and only the
approval write -- the researcher's click -- adds one, minted for that study
and that record.  A study that states no trial keeps every digest it had; a
stated or approved trial moves the scientific digest; a turn restored from
an earlier snapshot clears the trial, which is set up and approved again.

At run start the trial projection reads the approved record and the
universe's schema and column metadata, and signs the suite: times, labels,
onset columns and endpoint from the stated trial; the onset window the
materializer read; the confounders the record carries, with a window
measurement keeping its unmeasured state; stays resampled, or patients when
the run binds their grouping.  A trial not yet approved routes nothing; a
study of another family, an extraction without the trial's windows or
columns, a label the suite cannot print, an approval minted for another
study, and a second sealed design are each refused with their own code.  The
run's own stop when the approved trial no longer compiles on its data names
its reason in the failure record.  Synthetic export rows only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.planning.target_trial_configuration import (
    TARGET_TRIAL_CONFIRMATION_OWNER,
    TargetTrialConfirmationError,
    load_target_trial_design,
    target_trial_approval_event_id,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.pi_copilot.workflow import gate_detail_projection
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
    signed_projection,
)
from tests.support.target_trial import (
    STUDY_ID,
    compiled_target_trial,
    target_trial_design,
)
from tests.support.target_trial_export import (
    ONSET_WINDOWS,
    acquire_target_trial,
    export_context,
    export_trial_population,
    export_trial_spec,
    target_trial_export,
)

_SCIENCE = "a" * 64


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )


@pytest.fixture(scope="module")
def extraction(tmp_path_factory):
    """The synthetic export, acquired with and without the trial's onset window.

    Without it every column, the onsets included, is read over the hours
    before time zero.  Covariates read past time zero are the run's
    recompile's to refuse: this adapter checks only what it reads.
    """

    root = tmp_path_factory.mktemp("target_trial_export")
    export = target_trial_export(root / "export")
    trial = acquire_target_trial(export, root / "trial", onset_windows=ONSET_WINDOWS)
    one_window = acquire_target_trial(export, root / "one_window")
    return trial, one_window


def _design(universe: Path, **kwargs) -> dict:
    population = export_trial_population()
    compiled = compiled_target_trial(
        export_context(universe),
        spec=kwargs.pop("spec", None) or export_trial_spec(),
        population=population,
    )
    assert compiled.approvable
    return target_trial_design(compiled=compiled, population=population, **kwargs)


def _study(design: dict, *, family: str = "causal_inference", study_id: str = STUDY_ID):
    return {
        "id": study_id,
        "analysis_design": {"analysis_family": family},
        "target_trial_design": design,
        "cohort": {},
    }


def _project(study: dict, universe: Path, *, specs=(), dependence=None):
    return compile_web_scientific_runtime_projection(
        study=study,
        sensitivity_specs=list(specs),
        primary_exposure=None,
        primary_exposure_source=None,
        target_outcome="mort_28d",
        declared_covariates=[],
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256=_SCIENCE,
        literature_citation_keys=(),
        direct_comparator_literature_keys=(),
        dependence=dependence,
    )


def _refused(
    study: dict, universe: Path, **kwargs
) -> WebScientificRuntimeProjectionError:
    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        _project(study, universe, **kwargs)
    return caught.value


# -- the section ----------------------------------------------------------------


def test_the_section_is_written_only_by_the_host() -> None:
    created = context_store.upsert_context({"id": STUDY_ID, "question": "q"})
    design = target_trial_design(approved=False)

    for write in (
        lambda: context_store.upsert_context(
            {"id": STUDY_ID, "target_trial_design": design}
        ),
        lambda: context_store.validate_context_update(
            {"id": STUDY_ID, "target_trial_design": design}
        ),
    ):
        with pytest.raises(context_store.StudyContextError) as caught:
            write()
        assert caught.value.detail["error"] == "study_target_trial_design_server_owned"

    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.bind_target_trial_design(
            STUDY_ID, target_trial_design(), expected_revision=created["revision"]
        )
    assert caught.value.detail["error"] == "target_trial_approval_click_only"

    written = context_store.bind_target_trial_design(
        STUDY_ID, design, expected_revision=created["revision"]
    )
    # Kept in its canonical form: not yet approved.
    assert written["target_trial_design"] == {**design, "approval": None}

    # The click is on the record the study keeps, and confirms every line.
    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.record_target_trial_approval(
            STUDY_ID,
            confirmed_compile_sha256="d" * 64,
            n_lines_confirmed=design["confirmation_lines"],
            expected_revision=written["revision"],
        )
    assert caught.value.detail["error"] == "target_trial_approval_record_mismatch"
    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.record_target_trial_approval(
            STUDY_ID,
            confirmed_compile_sha256=design["compile_sha256"],
            n_lines_confirmed=design["confirmation_lines"] - 1,
            expected_revision=written["revision"],
        )
    assert caught.value.detail["error"] == "target_trial_design_invalid"

    approved = context_store.record_target_trial_approval(
        STUDY_ID,
        confirmed_compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        expected_revision=written["revision"],
    )
    approval = approved["target_trial_design"]["approval"]
    assert approval["approval_event_id"] == target_trial_approval_event_id(
        study_id=STUDY_ID,
        compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        confirmed_at=approval["confirmed_at"],
    )
    assert (
        context_store.get_context(STUDY_ID)["target_trial_design"]["approval"]
        == approval
    )


def test_the_digest_moves_only_with_a_stated_trial() -> None:
    created = context_store.upsert_context({"id": STUDY_ID, "question": "q"})
    before = context_store.scientific_configuration_sha256(created)
    design = target_trial_design(approved=False)

    stated = context_store.bind_target_trial_design(
        STUDY_ID, design, expected_revision=created["revision"]
    )
    approved = context_store.record_target_trial_approval(
        STUDY_ID,
        confirmed_compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        expected_revision=stated["revision"],
    )
    cleared = context_store.bind_target_trial_design(
        STUDY_ID, {}, expected_revision=approved["revision"]
    )

    digests = [
        context_store.scientific_configuration_sha256(row)
        for row in (stated, approved, cleared)
    ]
    assert len({before, *digests[:2]}) == 3
    assert digests[2] == before
    assert cleared["target_trial_design"] == {}


def test_a_restored_turn_clears_the_trial() -> None:
    created = context_store.upsert_context({"id": STUDY_ID, "question": "q"})
    design = target_trial_design(approved=False)
    stated = context_store.bind_target_trial_design(
        STUDY_ID, design, expected_revision=created["revision"]
    )
    approved = context_store.record_target_trial_approval(
        STUDY_ID,
        confirmed_compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        expected_revision=stated["revision"],
    )

    restored = context_store.restore_turn_configuration_snapshot(
        STUDY_ID,
        {
            "schema_version": "easyicu.pi-turn-study-snapshot/1",
            "study_context_id": STUDY_ID,
            "source_revision": created["revision"],
            "configuration": {
                "question": "q",
                # A replayed receipt mints nothing.
                "target_trial_design": approved["target_trial_design"],
            },
        },
        expected_revision=approved["revision"],
    )

    assert restored["target_trial_design"] == {}


def test_a_stored_section_a_later_contract_refuses_stays_inspectable(tmp_path) -> None:
    context_store.upsert_context({"id": STUDY_ID, "question": "q"})
    raw = json.loads(context_store._CONFIG_PATH.read_text(encoding="utf-8"))
    stale = {**target_trial_design(), "schema_version": "easyicu.target_trial_design/0"}
    raw["contexts"][0]["target_trial_design"] = stale
    context_store._CONFIG_PATH.write_text(json.dumps(raw), encoding="utf-8")

    stored = context_store.get_context(STUDY_ID)

    assert stored["target_trial_design"] == stale
    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.scientific_configuration_sha256(stored)
    assert caught.value.detail["error"] == "target_trial_design_invalid"


# -- the projection --------------------------------------------------------------


def test_an_approved_trial_binds_the_signed_suite(extraction) -> None:
    universe, _ = extraction
    design = _design(universe)
    projection = _project(_study(design), universe)
    authority = projection.authority
    approval = design["approval"]

    assert authority["authority_kind"] == "target_trial_suite"
    assert authority["protocol_content_sha256"] == _SCIENCE
    assert authority["target_trial_compile_sha256"] == design["compile_sha256"]
    assert (
        authority["target_trial_compile_confirmation_lines"]
        == (design["confirmation_lines"])
    )
    assert authority["confirmation"] == {
        "confirmed_by": "researcher",
        "approval_event_id": approval["approval_event_id"],
        "confirmed_compile_sha256": design["compile_sha256"],
        "n_lines_confirmed": approval["n_lines_confirmed"],
    }
    assert authority["treatment_onset_columns"] == [
        "vaso_ind_onset_time",
        "other_vaso_onset_time",
    ]
    assert authority["treatment_onset_window_hours"] == [0.0, 12.0]
    assert (authority["time_zero_hours"], authority["grace_period_hours"]) == (6, 6)
    assert (
        authority["event_column"],
        authority["followup_time_column"],
        authority["endpoint_horizon_days"],
    ) == ("mort_28d", "followup_days_28d", 28)
    assert (authority["unit_id_column"], authority["resampling_unit"]) == (
        "stay_id",
        "icu_stay",
    )
    assert [
        (item["column"], item["label"], item["coding"], item["unmeasured_state"])
        for item in authority["covariates"]
    ] == [
        ("age", "Age", "continuous", False),
        ("lact_max", "Lactate, highest before time zero", "continuous", True),
    ]
    assert (authority["initiate_label"], authority["defer_label"]) == (
        "Early start",
        "No early start",
    )
    assert authority["evidence_ceiling"] == "analysis_only"
    assert projection.analysis_only_execution is False
    assert (
        projection.projection_sha256
        == signed_projection(
            authority, scientific_configuration_sha256=_SCIENCE
        ).projection_sha256
    )
    assert projection.bound_target_trial == (
        load_target_trial_design(design, study_id=STUDY_ID)
        .confirmed()
        .model_dump(mode="json")
    )


def test_a_patient_grouping_resamples_patients(extraction) -> None:
    universe, _ = extraction
    grouping = PlannedDependenceRequirement(
        group_source="stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )

    authority = _project(
        _study(_design(universe)), universe, dependence=grouping
    ).authority

    assert (
        authority["resampling_unit"],
        authority["patient_group_column"],
        authority["patient_group_derivation"],
        authority["patient_group_delimiter"],
    ) == ("patient", "stay_id", "prefix_before_delimiter", ":s")


def test_a_trial_not_yet_approved_routes_nothing(extraction) -> None:
    universe, _ = extraction

    assert _project(_study(_design(universe, approved=False)), universe) is None
    assert _project(_study({}), universe) is None


def test_a_trial_in_a_study_of_another_family_is_refused(extraction) -> None:
    universe, _ = extraction

    error = _refused(_study(_design(universe), family="association"), universe)

    assert error.code == "target_trial_family_mismatch"


def test_an_extraction_without_the_trial_s_windows_is_refused(extraction) -> None:
    universe, one_window = extraction

    error = _refused(_study(_design(universe)), one_window)

    assert error.code == "target_trial_materialization_mismatch"
    assert error.details["onset_column"] == "vaso_ind_onset_time"
    assert error.details["covariate_window"] == {"start_hours": 0, "end_hours": 6}
    assert error.details["treatment_onset_window"] == {
        "start_hours": 0,
        "end_hours": 12,
    }


def test_a_label_the_suite_cannot_print_is_refused(extraction) -> None:
    universe, _ = extraction
    spec = export_trial_spec().model_copy(
        update={
            "strategies": export_trial_spec().strategies.model_copy(
                update={"initiate_label": "Start within 6 h"}
            )
        }
    )

    error = _refused(_study(_design(universe, spec=spec)), universe)

    assert error.code == "target_trial_configuration_invalid"
    assert error.details["fields"] == ["target_trial_suite.initiate_label"]


def test_an_approval_minted_for_another_study_is_refused(extraction) -> None:
    universe, _ = extraction

    error = _refused(_study(_design(universe), study_id="study_other0001"), universe)

    assert error.code == "target_trial_configuration_invalid"
    assert error.details["reason_code"] == "target_trial_approval_event_mismatch"


def test_a_trial_and_another_sealed_design_are_refused_together(extraction) -> None:
    universe, _ = extraction
    landmark = SimpleNamespace(spec_id="sens_landmark", strategy="landmark")

    error = _refused(_study(_design(universe)), universe, specs=[landmark])

    assert error.code == "research_pipeline_conflicting_sealed_suites"
    assert error.details["spec_ids"] == ["sens_landmark"]


# -- the run's stop ----------------------------------------------------------------


def test_a_drifted_trial_names_its_stop_in_the_failure_record(tmp_path) -> None:
    stop = TargetTrialConfirmationError(
        "target_trial_compile_drifted", "private detail"
    )

    assert agent_pipeline_runs._safe_pipeline_typed_failure(stop) == {
        "owner": TARGET_TRIAL_CONFIRMATION_OWNER,
        "reason_code": "target_trial_compile_drifted",
    }
    code = agent_pipeline_runs._pipeline_failure_code(stop)
    assert code == "research_pipeline_progressive_compile_failed"
    assert (
        "approve it again"
        in agent_pipeline_runs._progressive_compile_failure_message(stop)
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": STUDY_ID},
        provider={},
        exc=stop,
        code=code,
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate_detail_projection(gate["detail"])["gate_detail_code"] == (
        "target_trial_compile_drifted"
    )
    assert "private detail" not in json.dumps(gate)
    # The owner's allowlist: another code under its name does not cross.
    forged = TargetTrialConfirmationError("target_trial_anything", "x")
    assert agent_pipeline_runs._safe_pipeline_typed_failure(forged) == {}
    # The local owner name is the planning owner's.
    assert agent_pipeline_runs._TARGET_TRIAL_CONFIRMATION_OWNER == (
        TARGET_TRIAL_CONFIRMATION_OWNER
    )


def test_a_causal_plan_without_a_confirmed_trial_says_what_to_do() -> None:
    stop = ProgressivePlanCompileError(
        "tte_trial_not_confirmed", "x", path="analysis_type"
    )

    message = agent_pipeline_runs._progressive_compile_failure_message(stop)

    assert message.startswith("This study has no confirmed target trial")
    assert "approve it on its confirmation card" in message
