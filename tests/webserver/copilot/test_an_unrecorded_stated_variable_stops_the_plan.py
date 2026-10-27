"""A stated variable the conversation could not record stays owed and stops the plan.

pilot1 (10-26): the question said 「调整年龄和性别后」, the conversation saved
「调整年龄」 and 「调整性别」, and the second quote is not the question's text.
Sex was not recorded, nothing asked for it again, and the plan was adjusted for
age alone.  Synthetic fixtures only; no rows are read.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.projections import (
    project_study_context, project_study_setup_receipt,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.unread_stated_variables import (
    STATED_VARIABLE_UNRECORDED, UNREAD_STATED_VARIABLES_FIELD,
    pending_unread, stop_unrecorded_stated_variables,
)
from tests.support.node import run_node
from .stated_variable_fixtures import _update

QUESTION = "在 eICU demo 中，入 ICU 24 小时内的最高乳酸与院内死亡之间，调整年龄和性别后是否呈非线性关系？"
SEX_OWED = {
    "field": "stated_adjustments", "variable": "性别",
    "reason": "a stated variable must quote the question's own variable and words",
}


def _bound(question: str = QUESTION, **fields):
    return {
        "id": "adjustment-study", "revision": 2, "question": question,
        "data_source": {"path": "/synthetic/source", "database": "miiv"}, **fields,
    }


@pytest.mark.parametrize("field,question,words,variables,concepts", [
    ("stated_adjustments", QUESTION, "调整年龄和性别", ("年龄", "性别"), ("age", "sex")),
    ("stated_adjustments", "In ICU stays, is peak lactate associated with death, adjusted for age and sex?",
     "adjusted for age and sex", ("age", "sex"), ("age", "sex")),
    ("stated_adjustments", "校正年龄、性别和试验评分后，指标与结局的关系如何？",
     "校正年龄、性别和试验评分", ("年龄", "性别", "试验评分"), ("age", "sex", None)),
    ("stated_subgroups", "按年龄和性别分层后，指标与结局的关系如何？",
     "按年龄和性别分层", ("年龄", "性别"), ("age", "sex")),
])
def test_a_coordinated_clause_records_each_variable_it_names(
    monkeypatch, field, question, words, variables, concepts,
):
    result, writes = _update(
        monkeypatch, field=field, question=question, concepts=("age", "sex"),
        current=_bound(question), statements=[{"variable": name, "words": words} for name in variables],
    )
    assert result["status"] == "ok"
    saved = writes[0][field]
    assert [(item["variable"], item["words"], item["concept"]) for item in saved] == [
        (name, words, concept) for name, concept in zip(variables, concepts)
    ]
    assert f"{field}_unread" not in result["details"]
    assert UNREAD_STATED_VARIABLES_FIELD not in writes[0]


def test_a_rewritten_quote_stays_owed_and_the_tool_asks_for_the_whole_clause(monkeypatch):
    """The pilot1 save, replayed: one item quoted, one rewritten per variable."""

    result, writes = _update(monkeypatch, question=QUESTION, concepts=("age", "sex"),
                             current=_bound(), statements=[
                                 {"variable": "年龄", "words": "调整年龄"},
                                 {"variable": "性别", "words": "调整性别"},
                             ])
    assert result["status"] == "ok"
    assert [item["variable"] for item in writes[0]["stated_adjustments"]] == ["年龄"]
    assert writes[0][UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert result["details"][UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert "whole clause" in result["summary"] and "never rewrite the clause" in result["summary"]
    assert "study_stated_variable_unrecorded" in result["summary"]
    with pytest.raises(ResearchPipelineRunError) as stop:
        stop_unrecorded_stated_variables(writes[0])
    assert stop.value.code == STATED_VARIABLE_UNRECORDED
    assert stop.value.details["stated_variable"] == "性别"
    assert stop.value.details["unrecorded"] == [SEX_OWED]
    assert "adjust for “性别”" in str(stop.value)


def test_a_later_turn_that_omits_the_owed_variable_keeps_it_owed(monkeypatch):
    saved = _bound(stated_adjustments=[{
        "variable": "年龄", "words": "调整年龄", "concept": "age", "source_status": "available",
    }], **{UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]})
    result, writes = _update(monkeypatch, question=QUESTION, concepts=("age", "sex"), current=saved,
                             statements=[{"variable": "年龄", "words": "调整年龄"}])
    assert result["status"] == "ok"
    assert UNREAD_STATED_VARIABLES_FIELD not in writes[0]  # unchanged, still stored
    assert result["details"][UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert "Not yet recorded" in result["summary"]


def test_restating_the_owed_variable_with_the_whole_clause_settles_it(monkeypatch):
    saved = _bound(stated_adjustments=[{
        "variable": "年龄", "words": "调整年龄", "concept": "age", "source_status": "available",
    }], **{UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]})
    result, writes = _update(monkeypatch, question=QUESTION, concepts=("age", "sex"), current=saved,
                             statements=[{"variable": "性别", "words": "调整年龄和性别"}])
    assert result["status"] == "ok"
    assert [item["concept"] for item in writes[0]["stated_adjustments"]] == ["age", "sex"]
    assert writes[0][UNREAD_STATED_VARIABLES_FIELD] == []
    assert UNREAD_STATED_VARIABLES_FIELD not in result["details"]
    stop_unrecorded_stated_variables({**saved, **writes[0]})


@pytest.mark.parametrize("statement", [
    {"variable": "体重", "words": "调整体重"},  # the question does not name it
    {"variable": "性别", "words": "调整性别"},  # the question negates it (below)
])
def test_a_reading_of_no_question_variable_is_only_reported(monkeypatch, statement):
    names = {"年龄": ("age",), "性别": ("sex",), "体重": ("weight",)}
    monkeypatch.setattr(
        "easyicu.webserver.study_intent.catalog_named_concepts",
        lambda text: names.get(text, ("age",) if text == QUESTION else ()),
    )
    result, writes = _update(monkeypatch, question=QUESTION, concepts=("age", "sex"),
                             current=_bound(), statements=[statement])
    assert result["status"] == "ok"
    assert result["details"]["stated_adjustments_unread"][0]["variable"] == statement["variable"]
    assert UNREAD_STATED_VARIABLES_FIELD not in writes[0]


def test_an_owed_variable_lapses_when_the_question_stops_naming_it():
    owed = _bound(**{UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]})
    assert pending_unread(owed) == [SEX_OWED]
    assert pending_unread({**owed, "question": "调整年龄后，最高乳酸与院内死亡是否呈非线性关系？"}) == []
    # Recorded under its concept by other words of the question settles it too.
    assert pending_unread({**owed, "stated_adjustments": [{
        "variable": "性别", "words": "调整年龄和性别", "concept": "sex", "source_status": "available",
    }]}) == []
    # A recording in the other field does not.
    assert pending_unread({**owed, "stated_subgroups": [{
        "variable": "性别", "words": "调整年龄和性别", "concept": "sex", "source_status": "available",
    }]}) == [SEX_OWED]


@pytest.mark.parametrize("roster,selection,stops", [
    (["age", "sex"], "exact", False),  # the exact roster binds sex in the primary model
    (["age"], "exact", True),
    (["age", "sex"], "planner_selectable", True),  # the stated subset would be the whole roster
])
def test_an_exact_roster_that_binds_the_owed_adjustment_keeps_planning_open(roster, selection, stops):
    study = _bound(covariates=roster, covariate_selection=selection, **(
        {"covariate_authority": "user"} if selection == "exact" else {}
    ), **{UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]})
    if stops:
        with pytest.raises(ResearchPipelineRunError) as stop:
            stop_unrecorded_stated_variables(study)
        assert stop.value.details["stated_variable"] == "性别"
    else:
        stop_unrecorded_stated_variables(study)
    # The record stays owed either way: a later roster change brings the stop back.
    assert pending_unread(study) == [SEX_OWED]


def test_the_store_keeps_the_owed_record_server_owned_and_digest_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    age = {"variable": "年龄", "words": "调整年龄", "concept": "age", "source_status": "available"}
    raw = {"id": "stored-owed", "question": QUESTION}
    original = study_contexts.upsert_context({**raw, "stated_adjustments": [age]},
                                             _server_question_wording_write=True)
    with pytest.raises(study_contexts.StudyContextError) as refused:
        study_contexts.upsert_context({**raw, UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]})
    assert refused.value.detail["error"] == "study_stated_variables_unread_server_owned"
    with pytest.raises(study_contexts.StudyContextError) as invalid:
        study_contexts.upsert_context({**raw, UNREAD_STATED_VARIABLES_FIELD: [{**SEX_OWED, "field": "covariates"}]},
                                      _server_question_wording_write=True)
    assert invalid.value.detail["error"] == "study_stated_variables_unread_invalid"
    owed = study_contexts.upsert_context({**raw, UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]},
                                         _server_question_wording_write=True)
    assert study_contexts.get_context(owed["id"])[UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert (study_contexts.scientific_configuration_sha256(original)
            != study_contexts.scientific_configuration_sha256(owed))
    # A study that never owed anything keeps the digest it had before the field existed.
    assert study_contexts.scientific_configuration_sha256({**original, UNREAD_STATED_VARIABLES_FIELD: []}) == (
        study_contexts.scientific_configuration_sha256(original))
    receipt = project_study_setup_receipt(owed)
    assert receipt.configuration[UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert project_study_context(owed)[UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert UNREAD_STATED_VARIABLES_FIELD not in project_study_setup_receipt(original).configuration
    assert UNREAD_STATED_VARIABLES_FIELD not in project_study_context(original)

    recorded = study_contexts.upsert_context({**raw, "stated_adjustments": [age, {
        "variable": "性别", "words": "调整年龄和性别", "concept": "sex", "source_status": "available",
    }]}, _server_question_wording_write=True)
    assert recorded[UNREAD_STATED_VARIABLES_FIELD] == []
    # Rewinding to the turn that still owed it owes it again; a turn that owed nothing clears it.
    rewound = study_contexts.restore_turn_configuration_snapshot(
        owed["id"], {"schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": owed["id"],
                     "source_revision": owed["revision"], "configuration": receipt.configuration},
        expected_revision=recorded["revision"],
    )
    assert rewound[UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    assert [item["variable"] for item in rewound["stated_adjustments"]] == ["年龄"]
    cleared = study_contexts.restore_turn_configuration_snapshot(
        owed["id"], {"schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": owed["id"],
                     "source_revision": original["revision"],
                     "configuration": project_study_setup_receipt(original).configuration},
        expected_revision=rewound["revision"],
    )
    assert cleared[UNREAD_STATED_VARIABLES_FIELD] == []
    owed_again = study_contexts.upsert_context({**raw, UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]},
                                               _server_question_wording_write=True)
    assert owed_again[UNREAD_STATED_VARIABLES_FIELD] == [SEX_OWED]
    # The researcher removes it from the question: nothing is owed any more.
    renamed = study_contexts.upsert_context(
        {**raw, "question": "调整年龄后，最高乳酸与院内死亡是否呈非线性关系？"},
    )
    assert renamed[UNREAD_STATED_VARIABLES_FIELD] == []
    assert [item["variable"] for item in renamed["stated_adjustments"]] == ["年龄"]


def _launch(study):
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = preparation.ResearchPipelineLaunchRequest(
        export_path="/typed/demo", study_context=study, project_root="/server/workspace",
        provider={"provider": "openai"}, provider_environment={"OPENAI_API_KEY": "test"},
        credential_source="pi_verified", literature_search_authorized=False,
        plan_revision_source_run_id="", execution_resume_source_run_id="",
        development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None,
    )
    return preparation._prepare_scientific_launch(request)


def test_a_plan_stops_before_planning_while_a_stated_variable_is_owed(monkeypatch):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(concepts=[SimpleNamespace(concept_id="age"),
                                                SimpleNamespace(concept_id="sex")]),
    )
    question = "调整年龄和性别后估计指标与结局的关系。"
    age = {"variable": "年龄", "words": "调整年龄和性别", "concept": "age", "source_status": "unbound"}
    study = {"id": "study-1", "question": question, "data_source": {"database": "miiv"},
             "stated_adjustments": [age], UNREAD_STATED_VARIABLES_FIELD: [SEX_OWED]}
    with pytest.raises(ResearchPipelineRunError) as stop:
        _launch(study)
    assert stop.value.code == STATED_VARIABLE_UNRECORDED
    assert stop.value.details["stated_variable"] == "性别"
    sex = {**age, "variable": "性别", "concept": "sex"}
    result = _launch({**study, "stated_adjustments": [age, sex]})
    assert result.metadata_planning_coordinates["required_adjustment_concepts"] == ["age", "sex"]


@pytest.mark.parametrize("language,words", [
    ("en", "was not recorded from it, so planning stopped before any model was called"),
    ("zh", "没有记录下来，因此规划在调用模型前已停止"),
])
def test_the_stop_reads_in_the_researchers_language(language, words):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the error-text owner")
    script = r"""
global.window = {EU_HTML: {esc: String}, EasyICU: {guidedPi: {declare: (_name, owner) => {global.owner = owner;}}}};
require(process.argv[1]);
const zh = process.argv[2] === 'zh';
const text = global.owner.create({tr: (en, cn) => (zh ? cn : en), staticPreview: () => false});
process.stdout.write(JSON.stringify([
  text.runFailureText('study_stated_variable_unrecorded'),
  text.errorText({code: 'study_stated_variable_unrecorded', message: 'The question asks to adjust for'}),
]));
"""
    owner = Path(study_contexts.__file__).parent / "static/js/screens-guided-pi-error-text.js"
    run, error = json.loads(run_node(node, script, str(owner), language, check=True).stdout)
    assert words in run and run == error
