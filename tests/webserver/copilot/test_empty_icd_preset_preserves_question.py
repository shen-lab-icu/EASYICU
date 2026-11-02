"""An incomplete ICD proposal must never alter the saved population."""

import copy

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiSessionRecord, ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.study_context_update import update_study_context


@pytest.fixture
def make_context(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")

    def create(cohort, question):
        study = study_contexts.upsert_context({"title": "Synthetic study", "cohort": cohort})
        session = PiSessionRecord(
            session_id="synthetic",
            binding=AuthorityBinding(study_context_id=study["id"], study_revision=study["revision"]),
        )
        ctx = ToolExecutionContext(session=session, user_message=question, allowed_actions=("configure",))
        return study, ctx

    return create


@pytest.mark.parametrize(
    ("prior", "proposal", "question"),
    [
        ({}, {"preset": "icd"}, "纳入 ICD-10 K92.2 消化道出血患者，报告院内死亡率。"),
        ({}, {"preset": "adult_all", "icd_enabled": True}, "纳入 ICD 患者"),
        ({"preset": "adult_all", "icd_enabled": True, "icd_include": "K92.2"},
         {"preset": "icd", "icd_enabled": False}, "请生成研究计划"),
        ({"preset": "icd", "icd_enabled": True, "icd_include": "K92.2"},
         {"preset": "icd", "icd_include": ""}, "请继续生成计划"),
    ],
)
def test_empty_icd_filter_preserves_entire_study_and_grant(make_context, prior, proposal, question):
    study, context = make_context(prior, question)
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    result = update_study_context(context, {
        "question": question,
        "purpose": "描述性研究",
        "cohort": proposal,
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "patient",
            "variance_estimator": "none_counts_only",
        },
    })
    assert result["status"] == "blocked"
    assert result["code"] == "study_cohort_icd_codes_required"
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"
    assert context.user_message == question


def test_user_requested_valid_icd_codes_are_not_blocked_as_empty(make_context):
    question = "纳入 ICD-10 K92.2 消化道出血患者，报告院内死亡率。"
    study, context = make_context({}, question)
    result = update_study_context(context, {
        "question": question,
        "cohort": {"preset": "icd", "icd_include": "K92.2"},
    })
    assert result["code"] == "study_context_updated"
    saved = study_contexts.get_context(study["id"])
    assert saved["cohort"]["icd_include"] == "K92.2"


@pytest.mark.parametrize(("cohort", "code"), [
    ({"preset": "adult_all", "icd_enabled": "invalid"}, "invalid_cohort_boolean"),
    ({"preset": "adult_all", "exclude_readmissions": True,
      "first_icu_stay_per_hospitalization": True}, "conflicting_first_icu_stay_scope"),
])
def test_other_executable_cohort_contracts_remain_strict(make_context, cohort, code):
    from easyicu.webserver.primary_cohort import PrimaryCohortContractError

    study, context = make_context({}, "请纳入指定人群")
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    with pytest.raises(PrimaryCohortContractError, match=code):
        update_study_context(context, {"cohort": cohort})
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"


def test_legacy_preset_cannot_bypass_first_stay_conflict_before_update_or_grant(make_context):
    from easyicu.webserver.primary_cohort import PrimaryCohortContractError

    prior = {"preset": "legacy_unrecognized", "label": "Preserve this label"}
    study, context = make_context(prior, "研究 ICU 患者")
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    with pytest.raises(PrimaryCohortContractError, match="conflicting_first_icu_stay_scope"):
        update_study_context(context, {"cohort": {
            "exclude_readmissions": True, "first_icu_stay_per_hospitalization": True,
        }})
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"


def test_legacy_preset_incomplete_icd_axis_is_blocked_before_update_or_grant(make_context):
    prior = {"preset": "legacy_unrecognized", "label": "Preserve this label"}
    study, context = make_context(prior, "纳入 ICD 患者")
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    result = update_study_context(context, {"cohort": {"icd_enabled": True}})
    assert result["code"] == "study_cohort_icd_codes_required"
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"


def test_legacy_preset_update_preserves_enabled_icd_codes(make_context):
    prior = {"preset": "legacy_unrecognized", "icd_enabled": True, "icd_include": "K92.2"}
    study, context = make_context(prior, "保留 ICD K92.2 人群")
    result = update_study_context(context, {"cohort": {"label": "复核标签"}})
    assert result["code"] == "study_context_updated"
    saved = study_contexts.get_context(study["id"])["cohort"]
    assert saved["preset"] == prior["preset"]
    assert saved["icd_enabled"] is True
    assert saved["icd_include"] == "K92.2"


@pytest.mark.parametrize(("prior", "proposal"), [
    ({"preset": "legacy_unrecognized", "icd_enabled": True, "icd_include": "K92.2"},
     {"icd_include": "K25.0"}),
    ({"preset": "legacy_unrecognized"}, {"icd_enabled": True, "icd_include": "K92.2"}),
    ({"preset": "legacy_unrecognized"}, {"age_min": 18}),
])
def test_legacy_preset_never_changes_population_axes_or_consumes_grant(make_context, prior, proposal):
    study, context = make_context(prior, "继续研究")
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    result = update_study_context(context, {"cohort": proposal})
    assert result["code"] == "unsupported_cohort_preset"
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"
