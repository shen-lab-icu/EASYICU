"""A causal question records its design before anything is planned.

One owner (``webserver.causal_trial_design``) decides, from the question the
study states, the causal design the host records: the study setup writes it
on the turn the study has a question and no design, with the question's words
it rests on, and the launch never plans such a study from keywords.  The
design follows the bound source's patient grouping.  A study bound to a
database v1 emulates no trial in, and a treatment the v1 trial emulation does
not register, are capability gaps on the first turn and at the launch.  A stored reading that no longer explains the design is cleared.
Synthetic studies only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
from easyicu.webserver import source_identity_authority
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.causal_trial_design import (
    STUDY_CAUSAL_TRIAL_DESIGN_UNSET,
    TARGET_TRIAL_TREATMENT_NOT_REGISTERED,
    causal_design_for,
    question_causal_design,
    registered_treatment_concepts,
    stop_unset_causal_design,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)
from easyicu.webserver.target_trial_card import target_trial_next_action
from easyicu.webserver.target_trial_setup import (
    TARGET_TRIAL_COMPILE_STOPS,
    TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
    target_trial_database_gap,
)

_NOREPINEPHRINE = (
    "以入 ICU 后第 6 小时为时间零点，比较 6 小时内开始去甲肾上腺素与这 6 小时内不开始的 28 天死亡。"
)
_ALBUMIN = (
    "以入 ICU 后第 6 小时为时间零点，比较 24 小时内开始静脉输注白蛋白与这 24 小时内不开始的 28 天死亡。"
)
_STAYS = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "bootstrap",
}


def _bound(tmp_path, database: str = "miiv") -> dict[str, Any]:
    return {"data_source": {"path": str(tmp_path / "export"), "database": database}}


def _grouping(monkeypatch: pytest.MonkeyPatch, binding: Any) -> None:
    monkeypatch.setattr(
        source_identity_authority,
        "resolve_study_patient_grouping",
        lambda **_kwargs: binding,
    )


def test_a_source_that_groups_patients_resamples_patients(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, SimpleNamespace(output_identity_column="patient_stay_id"))

    assert causal_design_for(_bound(tmp_path)) == {**_STAYS, "cluster_unit": "patient"}


def test_a_source_without_patient_groups_resamples_stays(tmp_path, monkeypatch) -> None:
    # The trial compile decides whether stays alone may be resampled.
    _grouping(monkeypatch, None)

    assert causal_design_for(_bound(tmp_path)) == _STAYS


def test_the_registry_decides_which_treatments_v1_reads() -> None:
    registered = registered_treatment_concepts()

    assert {"vaso_ind", "other_vaso", "norepi_rate", "norepi_equiv", "milrinone"} <= registered
    assert not {"albumin_iv", "abx", "rrt", "packed_rbc", "vent_ind"} & registered
    # Only the catalog group of the registered concepts lends its members.
    lending = [
        name for name, members in CONCEPT_GROUPS_INTERNAL.items()
        if {"vaso_ind", "other_vaso"} & set(members)
    ]
    assert lending == ["vasopressors"]


def test_an_unregistered_treatment_is_a_gap_stating_what_v1_registers() -> None:
    decision = question_causal_design(_ALBUMIN, {})

    assert decision is not None and decision.design is None
    gap = decision.gap
    assert gap["code"] == TARGET_TRIAL_TREATMENT_NOT_REGISTERED
    assert gap["treatment_concepts"] == [
        {"id": "albumin_iv", "label_en": "Albumin IV", "label_zh": "静脉白蛋白"}
    ]
    classes = {item["id"]: item for item in gap["supported_classes"]}
    assert set(classes) == {"inotrope", "vasoactive", "vasopressor"}
    assert classes["vasoactive"]["label_zh"] == "血管活性药"
    assert {"id": "angiotensin_ii", "label_en": "angiotensin II"} in classes["vasoactive"]["agents"]


def _current(**fields: Any) -> dict[str, Any]:
    return {"id": "study-causal-reading", "revision": 3, "active_job_id": None, **fields}


def _update(monkeypatch, current: dict[str, Any], proposal: dict[str, Any], message: str):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw))
        or {**current, **raw, "revision": current["revision"] + 1},
    )
    session = PiSessionRecord(
        session_id="pi-causal-reading",
        binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"]
        ),
    )
    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session, user_message=message, allowed_actions={"configure"}
        ),
    )
    return result, writes


@pytest.mark.parametrize("grouped", [True, False])
def test_saving_a_trial_shaped_question_records_the_causal_design(
    tmp_path, monkeypatch, grouped: bool
) -> None:
    _grouping(monkeypatch, SimpleNamespace(output_identity_column="patient_stay_id") if grouped else None)
    current = _current(**_bound(tmp_path), analysis_design={})

    result, writes = _update(monkeypatch, current, {"question": _NOREPINEPHRINE}, _NOREPINEPHRINE)

    assert result["code"] == "study_context_updated"
    design = {**_STAYS, **({"cluster_unit": "patient"} if grouped else {})}
    assert writes[-1]["analysis_design"] == design
    reading = writes[-1]["causal_trial_reading"]
    assert reading["source"] == "question_trial_shape"
    assert all(item["evidence"] in _NOREPINEPHRINE for item in reading["elements"])
    assert result["details"]["causal_trial_reading"] == reading
    assert "easyicu_state_target_trial" in result["summary"]
    # The workflow's next step is the trial statement.
    study = {**current, **writes[-1]}
    assert target_trial_next_action(study, None) == "target_trial_statement_needed"


def test_a_design_the_study_or_the_turn_states_is_kept(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    association = {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    current = _current(**_bound(tmp_path), analysis_design=dict(association))

    result, writes = _update(monkeypatch, current, {"question": _NOREPINEPHRINE}, _NOREPINEPHRINE)

    assert result["code"] == "study_context_updated"
    assert "causal_trial_reading" not in writes[-1]
    assert writes[-1].get("analysis_design", association)["analysis_family"] == "association_study"


def test_an_unregistered_treatment_records_no_design_and_says_why(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    current = _current(**_bound(tmp_path), analysis_design={})

    result, writes = _update(monkeypatch, current, {"question": _ALBUMIN}, _ALBUMIN)

    assert result["code"] == "study_context_updated"
    assert not writes[-1].get("analysis_design")
    assert "causal_trial_reading" not in writes[-1]
    gap = result["details"]["causal_trial_gap"]
    assert gap["code"] == TARGET_TRIAL_TREATMENT_NOT_REGISTERED
    assert "not planned as an association" in result["summary"]
    assert "norepinephrine" in result["summary"]


def test_a_database_v1_emulates_no_trial_in_is_a_gap_on_the_first_turn(
    tmp_path, monkeypatch
) -> None:
    current = _current(**_bound(tmp_path, "eicu"), analysis_design={})

    result, writes = _update(monkeypatch, current, {"question": _NOREPINEPHRINE}, _NOREPINEPHRINE)

    assert result["code"] == "study_context_updated"
    assert not writes[-1].get("analysis_design")
    assert "causal_trial_reading" not in writes[-1]
    # The trial setup's own refusal, by the same stable code.
    assert TARGET_TRIAL_DATABASE_OUT_OF_SCOPE == "target_trial_database_out_of_scope"
    assert TARGET_TRIAL_DATABASE_OUT_OF_SCOPE in TARGET_TRIAL_COMPILE_STOPS
    assert result["details"]["causal_trial_gap"] == {
        "code": TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
        "database": "eicu",
        "supported_databases": ["miiv"],
    }
    assert "not in eicu" in result["summary"]
    assert "not planned as an association" in result["summary"]


@pytest.mark.parametrize(
    ("bound", "out_of_scope"),
    [("MIMIC-IV", None), ("mimic-iv", None), ("eICU-CRD", "eicu"), ("hirid", "hirid")],
)
def test_the_bound_database_is_read_by_its_registry_key(bound: str, out_of_scope) -> None:
    # A display name or alias is the database the launch reads, not another one.
    gap = target_trial_database_gap({"data_source": {"database": bound}})

    assert (gap or {}).get("database") == out_of_scope


def _launch(monkeypatch, study: dict[str, Any], question: str):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store,
        "upsert_context",
        lambda raw, **kwargs: writes.append({**raw, **kwargs}) or dict(raw),
    )
    try:
        stop_unset_causal_design(study, question=question, database="miiv")
    except ResearchPipelineRunError as exc:
        return exc, writes
    return None, writes


def test_the_launch_records_the_design_and_returns_to_the_trial(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path), "analysis_design": {}}

    stopped, writes = _launch(monkeypatch, study, _NOREPINEPHRINE)

    assert stopped is not None and stopped.code == STUDY_CAUSAL_TRIAL_DESIGN_UNSET
    assert stopped.details["next_action"] == "target_trial_statement_needed"
    assert stopped.details["causal_trial_reading"]["source"] == "question_trial_shape"
    assert stopped.details["design_recorded"] is True
    (write,) = writes
    assert write["analysis_design"] == _STAYS
    assert write["expected_revision"] == 5 and write["_server_causal_trial_reading_write"]


def test_a_new_run_of_the_study_stops_before_it_is_planned(tmp_path, monkeypatch) -> None:
    # The launch's own preparation runs the guard, before the export is read.
    _grouping(monkeypatch, None)
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store,
        "upsert_context",
        lambda raw, **kwargs: writes.append(dict(raw)) or dict(raw),
    )
    request = ResearchPipelineLaunchRequest(
        export_path=str(tmp_path / "export"),
        study_context={
            "id": "study-launch",
            "revision": 5,
            "question": _NOREPINEPHRINE,
            **_bound(tmp_path),
            "analysis_design": {},
        },
        project_root=str(tmp_path / "workspace"),
        provider={"provider": "openai"},
        provider_environment={"OPENAI_API_KEY": "test"},
        credential_source="pi_verified",
        literature_search_authorized=False,
        plan_revision_source_run_id="",
        execution_resume_source_run_id="",
        development_resume_source_job_id="",
        budget_mode="planner_canary",
        runner_image=None,
    )

    with pytest.raises(ResearchPipelineRunError) as stopped:
        prepare_research_pipeline_run(request)

    assert stopped.value.code == STUDY_CAUSAL_TRIAL_DESIGN_UNSET
    assert [write["analysis_design"] for write in writes] == [_STAYS]


def test_the_launch_refuses_an_unregistered_treatment_without_writing(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path), "analysis_design": {}}

    stopped, writes = _launch(monkeypatch, study, _ALBUMIN)

    assert stopped is not None and stopped.code == TARGET_TRIAL_TREATMENT_NOT_REGISTERED
    assert stopped.details["causal_trial_gap"]["treatment_concepts"][0]["id"] == "albumin_iv"
    assert writes == []


def test_the_launch_refuses_a_trial_in_a_database_v1_emulates_none_in(
    tmp_path, monkeypatch
) -> None:
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path, "eicu"), "analysis_design": {}}

    stopped, writes = _launch(monkeypatch, study, _NOREPINEPHRINE)

    assert stopped is not None and stopped.code == TARGET_TRIAL_DATABASE_OUT_OF_SCOPE
    assert stopped.details["causal_trial_gap"]["database"] == "eicu"
    assert "causal_trial_reading" in stopped.details
    assert writes == []


def test_a_question_only_the_planner_reads_as_causal_still_stops(tmp_path, monkeypatch) -> None:
    # The planner's keyword routing reads it as causal; the reader does not,
    # so nothing is recorded, and the launch still returns to the trial.
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path), "analysis_design": {}}

    stopped, writes = _launch(
        monkeypatch, study, "Is lactate associated with mortality? Report covariate balance."
    )

    assert stopped is not None and stopped.code == STUDY_CAUSAL_TRIAL_DESIGN_UNSET
    assert "causal_trial_reading" not in stopped.details
    assert stopped.details["next_action"] == "target_trial_statement_needed"
    assert writes == []


@pytest.mark.parametrize(
    ("design", "question"),
    [
        ({}, "前 24 小时最高乳酸与 28 天死亡的关联"),
        (
            {"analysis_family": "association_study", "analysis_unit": "icu_stay",
             "variance_estimator": "model_based"},
            _NOREPINEPHRINE,
        ),
    ],
)
def test_the_launch_plans_a_study_its_question_does_not_make_causal(
    tmp_path, monkeypatch, design, question
) -> None:
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path), "analysis_design": design}

    stopped, writes = _launch(monkeypatch, study, question)

    assert stopped is None and writes == []


def test_a_reading_that_no_longer_explains_the_design_is_cleared() -> None:
    reading = question_causal_design(_NOREPINEPHRINE, {}).reading.record()
    saved = context_store.upsert_context(
        {
            "id": "study-stale-reading",
            "question": _NOREPINEPHRINE,
            "analysis_design": dict(_STAYS),
            "causal_trial_reading": reading,
        },
        _server_causal_trial_reading_write=True,
    )
    assert saved["causal_trial_reading"] == reading

    # A client cannot write it.
    with pytest.raises(context_store.StudyContextError) as refused:
        context_store.upsert_context(
            {"id": "study-stale-reading", "causal_trial_reading": reading}
        )
    assert refused.value.detail["error"] == "study_causal_trial_reading_server_owned"

    # A new question no longer holds its words: the design stays, the reading goes.
    changed = context_store.upsert_context(
        {"id": "study-stale-reading", "question": "早开始去甲肾上腺素是否降低 28 天死亡？"}
    )
    assert changed["analysis_design"]["analysis_family"] == "causal_inference"
    assert changed["causal_trial_reading"] is None
