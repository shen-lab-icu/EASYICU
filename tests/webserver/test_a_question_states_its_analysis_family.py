"""A question that states its analysis family in its own words is planned on it.

One owner (``webserver.study_family_design``) decides, from the study's
question (``study_family_reading``), the design the host records when the
study states none: a question that asks whether two things are related is an
association study, one that asks how common something is, with no adjusted,
causal or predictive estimate, is descriptive epidemiology.  The study setup
records it with the words it rests on, and the launch records it before it
plans, so the planner's keyword routing never guesses the family.  A design
the study or the turn states is kept; a question that reads another family is
recorded beside it as a conflict.  Synthetic questions and studies only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import source_identity_authority
from easyicu.webserver import study_contexts as context_store
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
from easyicu.webserver.study_family_design import (
    STUDY_FAMILY_READING_FIELD,
    family_reading_conflict,
    record_unset_family_design,
)
from easyicu.webserver.study_family_reading import study_family_reading

_ASSOCIATION_ZH = "在成人 ICU 住院中，校正年龄和性别后，入 ICU 时的乳酸是否与 28 天死亡相关？"
_DESCRIPTIVE_EN = (
    "Among adult ICU stays, what proportion had a serum sodium below 130 mmol/L in the "
    "first 24 hours, and how did hospital mortality differ between those stays and the rest?"
)


@pytest.mark.parametrize(
    ("question", "family", "evidence"),
    [
        ("乳酸是否与 28 天死亡相关？", "association_study", "是否与 28 天死亡相关"),
        ("血钠与院内死亡是否有关？", "association_study", "与院内死亡是否有关"),
        ("入 ICU 时 BMI 与院内死亡的关系", "association_study", "与院内死亡的关系"),
        ("脓毒症患者院内死亡的危险因素", "association_study", "的危险因素"),
        (
            "Is admission lactate associated with hospital mortality?",
            "association_study",
            "Is admission lactate associated with",
        ),
        (
            "Examine the association between body mass index and mortality.",
            "association_study",
            "association between body mass index and",
        ),
        ("What are the risk factors for delirium?", "association_study", "risk factors for"),
        ("入 ICU 后 24 小时内低钠血症的比例是多少？", "descriptive_epidemiology", "比例是多少"),
        ("有多大比例的住院在首日接受了机械通气？", "descriptive_epidemiology", "多大比例"),
        ("成人 ICU 住院中急性肾损伤的发生率", "descriptive_epidemiology", "发生率"),
        (
            "What proportion of adult stays received vasopressors on day one?",
            "descriptive_epidemiology",
            "What proportion",
        ),
        ("How common is hypoglycemia in the first day?", "descriptive_epidemiology", "How common"),
    ],
)
def test_a_question_is_read_by_the_family_its_words_state(
    question: str, family: str, evidence: str
) -> None:
    reading = study_family_reading(question)

    assert reading is not None and reading.family == family
    assert [item.evidence for item in reading.elements] == [evidence]
    for item in reading.elements:
        assert question[item.start : item.end] == item.evidence


def test_a_relationship_the_question_asks_to_describe_is_a_description() -> None:
    # As the study setup's descriptive default reads a prevalence question
    # that also describes a relationship.
    question = "估计低钠血症的患病率，并描述低钠血症与院内死亡的关系"

    reading = study_family_reading(question)

    assert reading is not None and reading.family == "descriptive_epidemiology"
    assert [item.evidence for item in reading.elements] == ["患病率"]
    assert study_family_reading("Describe the relationship between BMI and mortality.") is None


def test_a_relationship_makes_an_association_whatever_proportion_it_also_asks() -> None:
    question = "低钠血症的比例是多少？低钠血症是否与院内死亡相关？"

    reading = study_family_reading(question)

    assert reading is not None and reading.family == "association_study"
    assert [item.element for item in reading.elements] == ["proportion", "relationship"]


@pytest.mark.parametrize(
    "question",
    [
        # A noun qualified as related asks nothing; a declined association.
        "列出与本研究相关的变量",
        "不研究乳酸与死亡的关系，只描述乳酸的分布",
        "Do not estimate any association between lactate and death; describe lactate.",
        # The data themselves.
        "检查与本研究相关的变量在首日的缺失率和单位",
        "What proportion of stays have missing lactate values in the first day?",
        "Audit the completeness of the first-day laboratory panel.",
        # A proportion with an adjusted, causal or predictive estimate.
        "What proportion of patients died, adjusted for age and severity?",
        "校正年龄后的死亡比例是多少？",
        "Predict hospital mortality from first-day data and report the proportion of deaths.",
        # Groups compared, no proportion asked; nothing at all.
        "How did hospital mortality differ between septic and non-septic stays?",
        "",
    ],
)
def test_a_question_that_states_no_family_is_read_as_none(question: str) -> None:
    assert study_family_reading(question) is None


def _bound(tmp_path) -> dict[str, Any]:
    return {"data_source": {"path": str(tmp_path / "export"), "database": "miiv"}}


def _grouping(monkeypatch: pytest.MonkeyPatch, binding: Any) -> None:
    monkeypatch.setattr(
        source_identity_authority,
        "resolve_study_patient_grouping",
        lambda **_kwargs: binding,
    )


def _current(**fields: Any) -> dict[str, Any]:
    return {"id": "study-family-reading", "revision": 3, "active_job_id": None, **fields}


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
        session_id="pi-family-reading",
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
def test_saving_a_question_records_the_family_it_states(tmp_path, monkeypatch, grouped) -> None:
    _grouping(monkeypatch, SimpleNamespace(output_identity_column="patient_stay_id") if grouped else None)
    current = _current(**_bound(tmp_path), analysis_design={})

    result, writes = _update(monkeypatch, current, {"question": _ASSOCIATION_ZH}, _ASSOCIATION_ZH)

    assert result["code"] == "study_context_updated"
    assert writes[-1]["analysis_design"] == (
        {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        }
        if grouped
        else {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        }
    )
    reading = writes[-1][STUDY_FAMILY_READING_FIELD]
    assert reading["family"] == "association_study" and "conflict" not in reading
    assert all(item["evidence"] in _ASSOCIATION_ZH for item in reading["elements"])
    assert result["details"][STUDY_FAMILY_READING_FIELD] == reading
    assert "do not ask the researcher to choose an analysis family" in result["summary"]


def test_a_trial_shaped_question_is_left_to_the_causal_reading(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    question = (
        "以入 ICU 后第 6 小时为时间零点，比较 6 小时内开始去甲肾上腺素与这 6 小时内不开始的 "
        "28 天死亡，以及两组死亡的比例是多少？"
    )
    current = _current(**_bound(tmp_path), analysis_design={})

    result, writes = _update(monkeypatch, current, {"question": question}, question)

    assert writes[-1]["analysis_design"]["analysis_family"] == "causal_inference"
    assert STUDY_FAMILY_READING_FIELD not in writes[-1]
    assert STUDY_FAMILY_READING_FIELD not in result["details"]


def test_a_stated_design_is_kept_and_a_conflict_is_shown(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    descriptive = {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    current = _current(**_bound(tmp_path), analysis_design=dict(descriptive))

    result, writes = _update(monkeypatch, current, {"question": _ASSOCIATION_ZH}, _ASSOCIATION_ZH)

    assert writes[-1].get("analysis_design", descriptive) == descriptive
    conflict = writes[-1][STUDY_FAMILY_READING_FIELD]
    assert conflict["family"] == "association_study"
    assert conflict["conflict"] == {"design_family": "descriptive_epidemiology"}
    assert "the design is kept" in result["summary"]

    # A design that agrees with the words shows no conflict; a causal one is R1's.
    assert family_reading_conflict(_ASSOCIATION_ZH, {"analysis_family": "association_study"}) is None
    assert family_reading_conflict(_ASSOCIATION_ZH, {"analysis_family": "causal_inference"}) is None


def test_the_reading_is_the_hosts_and_is_cleared_when_it_says_nothing() -> None:
    reading = study_family_reading(_ASSOCIATION_ZH).record()
    design = {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    saved = context_store.upsert_context(
        {
            "id": "study-family-stale",
            "question": _ASSOCIATION_ZH,
            "analysis_design": dict(design),
            STUDY_FAMILY_READING_FIELD: reading,
        },
        _server_study_family_reading_write=True,
    )
    assert saved[STUDY_FAMILY_READING_FIELD] == reading
    # Read back from the store, the reading stays.
    assert context_store.get_context("study-family-stale")[STUDY_FAMILY_READING_FIELD] == reading

    with pytest.raises(context_store.StudyContextError) as refused:
        context_store.upsert_context({"id": "study-family-stale", STUDY_FAMILY_READING_FIELD: reading})
    assert refused.value.detail["error"] == "study_family_reading_server_owned"

    # The researcher states another family: the reading no longer explains it.
    changed = context_store.upsert_context(
        {"id": "study-family-stale", "analysis_design": {**design, "analysis_family": "descriptive_epidemiology"}}
    )
    assert changed["analysis_design"]["analysis_family"] == "descriptive_epidemiology"
    assert changed[STUDY_FAMILY_READING_FIELD] is None


def test_a_conflict_is_cleared_once_the_design_follows_the_words() -> None:
    reading = study_family_reading(_ASSOCIATION_ZH)
    design = {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    context_store.upsert_context(
        {
            "id": "study-family-conflict",
            "question": _ASSOCIATION_ZH,
            "analysis_design": dict(design),
            STUDY_FAMILY_READING_FIELD: reading.record(design_family="descriptive_epidemiology"),
        },
        _server_study_family_reading_write=True,
    )

    agreed = context_store.upsert_context(
        {"id": "study-family-conflict", "analysis_design": {**design, "analysis_family": "association_study"}}
    )

    assert agreed[STUDY_FAMILY_READING_FIELD] is None


def _launch_writes(monkeypatch) -> list[dict[str, Any]]:
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store,
        "upsert_context",
        lambda raw, **kwargs: writes.append({**raw, **kwargs})
        or {**raw, "revision": 6},
    )
    return writes


def test_the_launch_records_the_family_before_it_plans(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, None)
    writes = _launch_writes(monkeypatch)
    study = {"id": "study-launch", "revision": 5, **_bound(tmp_path), "analysis_design": {}}

    launched = record_unset_family_design(study, question=_DESCRIPTIVE_EN)

    (write,) = writes
    assert write["analysis_design"]["analysis_family"] == "descriptive_epidemiology"
    assert write["expected_revision"] == 5 and write["_server_study_family_reading_write"]
    assert launched["analysis_design"] == write["analysis_design"]
    assert launched[STUDY_FAMILY_READING_FIELD]["family"] == "descriptive_epidemiology"

    # A study that states a design, or a question that states no family, is untouched.
    stated = {**study, "analysis_design": {"analysis_family": "association_study"}}
    assert record_unset_family_design(stated, question=_DESCRIPTIVE_EN) is stated
    assert record_unset_family_design(study, question="Show the cohort.") is study
    assert len(writes) == 1


def test_a_new_run_records_the_family_in_its_own_preparation(tmp_path, monkeypatch) -> None:
    # The launch's own preparation records the design before anything else
    # reads the study, whatever stops the launch later.
    _grouping(monkeypatch, None)
    writes = _launch_writes(monkeypatch)
    request = ResearchPipelineLaunchRequest(
        export_path=str(tmp_path / "export"),
        study_context={
            "id": "study-launch",
            "revision": 5,
            "question": _ASSOCIATION_ZH,
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

    try:
        prepare_research_pipeline_run(request)
    except ResearchPipelineRunError:
        pass

    assert [write["analysis_design"]["analysis_family"] for write in writes] == ["association_study"]
