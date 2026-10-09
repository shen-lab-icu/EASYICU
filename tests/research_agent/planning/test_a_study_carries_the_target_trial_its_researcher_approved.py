"""A study carries the target trial its researcher approved, and a run binds it.

The study setup states the trial and its population; the host compiles them
and keeps the record, its digest and the lines its card lists.  The click on
the card adds the approval, whose event id the host mints for that study.
Each count and digest comes from its own owner and must agree: the section
refuses a record that is not the one its digest names, an approval of another
record or of another number of lines, an approval of a record the host cannot
approve, and an approval minted for another study.

A run binds the approved trial before planning: it compiles the trial on the
context the run built and requires the approved record.  A changed onset
window, a confounder the data no longer carries, another database, or a spec
or population changed after the approval each stop the run with
``target_trial_compile_drifted`` under the confirmation owner, before any
model is called.  Synthetic contexts only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.reviewed_requirements import (
    bind_reviewed_requirements,
)
from easyicu.research_agent.planning.target_trial_configuration import (
    TARGET_TRIAL_CONFIRMATION_OWNER,
    ConfirmedTargetTrial,
    TargetTrialConfirmationError,
    TargetTrialDesignError,
    bind_confirmed_target_trial,
    load_target_trial_design,
    normalize_target_trial_design,
    target_trial_approval_event_id,
)
from easyicu.research_agent.planning.population_spec import PopulationSpec
from tests.support.target_trial import (
    CONFIRMED_AT,
    STUDY_ID,
    compiled_target_trial,
    target_trial_context,
    target_trial_design,
    target_trial_population,
    target_trial_spec,
)


def _confirmed(**changes) -> dict:
    design = load_target_trial_design(target_trial_design(), study_id=STUDY_ID)
    confirmed = design.confirmed().model_dump(mode="json")
    confirmed.update(changes)
    return confirmed


def _refused(
    design: dict, *, study_id: str | None = STUDY_ID
) -> TargetTrialDesignError:
    with pytest.raises(TargetTrialDesignError) as caught:
        load_target_trial_design(design, study_id=study_id)
    return caught.value


# -- the section ----------------------------------------------------------------


def test_the_section_is_kept_as_the_host_compiled_it() -> None:
    design = target_trial_design()

    assert normalize_target_trial_design(design, study_id=STUDY_ID) == design
    assert normalize_target_trial_design({}, study_id=STUDY_ID) == {}
    assert normalize_target_trial_design(None, study_id=STUDY_ID) == {}
    assert load_target_trial_design({}, study_id=STUDY_ID) is None
    # Closed: no field outside the contract, none of its own left out.
    assert _refused({**design, "notes": "x"}).code == "target_trial_design_invalid"
    without_version = {k: v for k, v in design.items() if k != "schema_version"}
    assert _refused(without_version).code == "target_trial_design_invalid"
    assert _refused(["not", "an", "object"]).field == "target_trial_design"


def test_the_record_kept_is_the_one_its_digest_names() -> None:
    design = target_trial_design(approved=False)
    record = design["compile_record"]

    edited = {**record, "protocol": [*record["protocol"][:-1]]}
    assert _refused({**design, "compile_record": edited}).code == (
        "target_trial_design_invalid"
    )
    # A spec the record was not compiled from.
    other_spec = target_trial_spec(
        strategies={"initiate_label": "Prompt start"}
    ).model_dump(mode="json")
    assert (
        _refused({**design, "spec": other_spec}).code == "target_trial_design_invalid"
    )
    # The lines the card lists are the record's, not a number of their own.
    lines = design["confirmation_lines"]
    assert _refused({**design, "confirmation_lines": lines + 1}).code == (
        "target_trial_design_invalid"
    )


def test_an_approval_is_of_the_record_kept_and_every_line_it_lists() -> None:
    design = target_trial_design()
    approval = design["approval"]

    for change in (
        {"confirmed_compile_sha256": "d" * 64},
        {"n_lines_confirmed": approval["n_lines_confirmed"] - 1},
    ):
        error = _refused({**design, "approval": {**approval, **change}}, study_id=None)
        assert error.code == "target_trial_design_invalid"


def test_a_record_the_host_cannot_approve_carries_no_approval() -> None:
    # Onsets read from a later hour: the treatment waits for an extraction.
    waiting = compiled_target_trial(
        target_trial_context(onset_window="icu_admission[2,12]h")
    )
    assert not waiting.approvable

    unapproved = target_trial_design(compiled=waiting, approved=False)
    assert load_target_trial_design(unapproved, study_id=STUDY_ID).approval is None
    error = _refused(target_trial_design(compiled=waiting))
    assert error.code == "target_trial_design_invalid"


def test_an_approval_carries_the_event_id_minted_for_its_study() -> None:
    design = target_trial_design()
    approval = design["approval"]

    assert approval["approval_event_id"] == target_trial_approval_event_id(
        study_id=STUDY_ID,
        compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        confirmed_at=CONFIRMED_AT,
    )
    assert approval["approval_event_id"].startswith("approval:target-trial:")
    assert load_target_trial_design(design, study_id=STUDY_ID).approval is not None
    # Carried over to another study, it is not that study's click.
    error = _refused(design, study_id="study_other0001")
    assert error.code == "target_trial_approval_event_mismatch"
    assert error.field == "target_trial_design.approval.approval_event_id"
    # Every coordinate of the click enters the id.
    assert (
        len(
            {
                target_trial_approval_event_id(
                    study_id=study,
                    compile_sha256=sha,
                    n_lines_confirmed=n,
                    confirmed_at=at,
                )
                for study, sha, n, at in (
                    (STUDY_ID, "a" * 64, 5, CONFIRMED_AT),
                    ("study_other0001", "a" * 64, 5, CONFIRMED_AT),
                    (STUDY_ID, "b" * 64, 5, CONFIRMED_AT),
                    (STUDY_ID, "a" * 64, 4, CONFIRMED_AT),
                    (STUDY_ID, "a" * 64, 5, "2026-10-09T14:00:01Z"),
                )
            }
        )
        == 5
    )
    # The host's clock writes the time, to the second, in UTC.
    timed = {**approval, "confirmed_at": "2026-10-09 14:00"}
    assert _refused({**design, "approval": timed}).code == "target_trial_design_invalid"


def test_a_trial_without_an_approval_binds_nothing() -> None:
    design = load_target_trial_design(
        target_trial_design(approved=False), study_id=STUDY_ID
    )

    assert design.confirmed() is None
    with pytest.raises(ValidationError):
        ConfirmedTargetTrial.model_validate({**_confirmed(), "approval": None})


# -- the run's binding -----------------------------------------------------------


def test_a_run_binds_the_approved_trial_on_its_own_context() -> None:
    context = target_trial_context()
    before = context.model_dump(mode="json")

    assert bind_confirmed_target_trial(context, _confirmed()) is context
    assert context.model_dump(mode="json") == before
    assert bind_confirmed_target_trial(context, None) is context


@pytest.mark.parametrize(
    "context",
    [
        # The onsets read from a later hour.
        target_trial_context(onset_window="icu_admission[2,12]h"),
        # A confounder summarized past time zero, as a first-day extraction
        # reads it.
        target_trial_context(covariate_window="icu_admission[0,24]h"),
        # A confounder the data no longer holds.
        target_trial_context(without=("map_min",)),
        # The indication's status gone: the population compiles otherwise.
        target_trial_context(without=("shock",)),
    ],
    ids=["onset_window", "covariate_window", "confounder", "indication"],
)
def test_a_context_the_trial_no_longer_compiles_to_stops_the_run(context) -> None:
    with pytest.raises(TargetTrialConfirmationError) as caught:
        bind_confirmed_target_trial(context, _confirmed())

    assert caught.value.reason_code == "target_trial_compile_drifted"
    assert caught.value.easyicu_safe_diagnostic == {
        "owner": TARGET_TRIAL_CONFIRMATION_OWNER,
        "reason_code": "target_trial_compile_drifted",
    }


def test_another_database_stops_the_run() -> None:
    context = target_trial_context()
    elsewhere = context.model_copy(
        update={"cohort": context.cohort.model_copy(update={"database": "eicu"})}
    )

    with pytest.raises(TargetTrialConfirmationError) as caught:
        bind_confirmed_target_trial(elsewhere, _confirmed())

    assert caught.value.reason_code == "target_trial_compile_drifted"


def test_a_spec_or_population_changed_after_the_approval_stops_the_run() -> None:
    adults_only = PopulationSpec.model_validate(
        {
            "criteria": [
                item
                for item in target_trial_population().model_dump(mode="json")[
                    "criteria"
                ]
                if item["id"] == "c2"
            ]
        }
    )
    later = target_trial_spec(time_zero={"hours_after_icu_admission": 4})

    for changed in (
        {"population_spec": adults_only.model_dump(mode="json")},
        {"spec": later.model_dump(mode="json")},
    ):
        with pytest.raises(TargetTrialConfirmationError):
            bind_confirmed_target_trial(target_trial_context(), _confirmed(**changed))


# -- the run's configuration -----------------------------------------------------


def test_the_run_configuration_carries_the_approved_trial(tmp_path) -> None:
    payload = _confirmed()

    assert (
        "bound_target_trial" not in PipelineConfig(workdir=tmp_path).canonical_payload()
    )
    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, bound_target_trial=payload)
    config = PipelineConfig(
        workdir=tmp_path, bound_target_trial=payload, require_human_plan_review=True
    )
    assert config.canonical_payload()["bound_target_trial"] == payload
    with pytest.raises(ValidationError):
        PipelineConfig(
            workdir=tmp_path,
            bound_target_trial={**payload, "approval": None},
            require_human_plan_review=True,
        )

    # The frozen configuration binds as it was handed in, before any other set.
    context = target_trial_context()
    assert bind_reviewed_requirements(context, config) == context
    assert bind_reviewed_requirements(context, config, restoring=True) == context
    with pytest.raises(TargetTrialConfirmationError):
        bind_reviewed_requirements(target_trial_context(without=("map_min",)), config)
