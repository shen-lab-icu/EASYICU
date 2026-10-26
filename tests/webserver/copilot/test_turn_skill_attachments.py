"""Synthetic contracts for host-owned turn attachments; no ASGI transport."""

import json
import time

import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.extensions.skill_attachments import attachment_payload
from easyicu.extensions.method_attachments import builtin_attachment
from easyicu.webserver import agent_pipeline_runs, jobs, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord
from easyicu.webserver.pi_copilot.projections import project_study_setup_receipt
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.skill_attachments import validate_attachments, load_attached_users
from easyicu.webserver.question_family_check import FAMILY_DISAGREEMENT, family_choice_texts, researcher_choice
from easyicu.webserver.study_family_scope import study_family_database_gap
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway, allow_unrelated_message_test, study_state as study_state,
)

from easyicu.research_agent.method_skills import METHOD_SKILL_REGISTRY_VERSION
from easyicu.webserver.routes.pi_copilot import PiMessageRequest


BUILTIN = {
    "source": "builtin",
    "id": "adjusted-exposure-outcome-study",
    "version": METHOD_SKILL_REGISTRY_VERSION,
}
REAL_UPSERT = study_contexts.upsert_context
REAL_GET_CONTEXT = study_contexts.get_context


def test_a_turn_carries_a_typed_skill_without_task_starter_prose():
    request = PiMessageRequest(
        project_id="synthetic-project",
        message="Is the synthetic exposure associated with the outcome?",
        attached_skills=[BUILTIN],
    )
    assert request.attached_skills[0].id == BUILTIN["id"]


def _install(registry, body="Use concise sentences.", **kwargs):
    return registry.install_skill(
        "---\nname: synthetic-helper\ndescription: Synthetic helper\n---\n" + body,
        **kwargs,
    )


@pytest.fixture
def registry(tmp_path):
    return ExtensionRegistry(tmp_path / "extensions")


def _validate(registry, value, activation=None):
    return validate_attachments(value, activation=activation or registry.snapshot(), registry=registry)


@pytest.mark.parametrize("value,code", [
    ([{**BUILTIN, "id": "unknown-synthetic-method"}], "skill_attachment_unknown"),
    ([{**BUILTIN, "version": "old"}], "skill_attachment_stale"),
    ([{**BUILTIN, "digest": "0" * 64}], "skill_attachment_stale"),
    ([BUILTIN, BUILTIN], "skill_attachment_duplicate"),
    ([BUILTIN, {**BUILTIN, "id": "cohort-characterization-table-one"}], "skill_attachment_family_conflict"),
    ([BUILTIN] * 9, "skill_attachment_invalid"),
    ([{**BUILTIN, "capability_id": "forged"}], "skill_attachment_invalid"),
    ([{"source": "user", "id": "missing", "digest": "0" * 64}], "skill_attachment_unknown"),
    ([{"source": "user", "id": "bad", "digest": "wrong"}], "skill_attachment_invalid"),
])
def test_invalid_attachments_are_refused_as_a_whole(registry, value, code):
    with pytest.raises(PiCopilotError) as error:
        _validate(registry, value)
    assert error.value.code == code


def test_disabled_builtin_is_refused(registry, monkeypatch):
    from easyicu.webserver.pi_copilot import skill_attachments
    monkeypatch.setattr(skill_attachments.capabilities, "capability_settings", lambda: {"science_skills_enabled": False})
    with pytest.raises(PiCopilotError) as error:
        _validate(registry, [BUILTIN])
    assert error.value.code == "skill_attachment_disabled"


@pytest.mark.parametrize("kwargs", [{"enabled": False}, {"stages": ["writing"]}])
def test_inactive_or_wrong_stage_user_skill_is_refused(registry, kwargs):
    installed = _install(registry, **kwargs)
    with pytest.raises(PiCopilotError) as error:
        _validate(registry, [{"source": "user", "id": installed["name"], "digest": installed["digest"]}])
    assert error.value.code == "skill_attachment_disabled"


def test_user_skill_loads_the_frozen_digest_even_after_registry_edits(registry):
    original = _install(registry)
    frozen = registry.snapshot()
    newer = _install(registry, "A different synthetic instruction.")
    ref = {"source": "user", "id": original["name"], "digest": original["digest"]}
    refs = _validate(registry, [ref], frozen)
    record = PiSessionRecord(session_id="synthetic", extension_activation=frozen)
    loaded = load_attached_users(record, refs, registry)
    assert loaded[0]["digest"] == original["digest"]
    assert loaded[0]["instructions"] == "Use concise sentences."
    assert loaded[0]["authority"] == "user_advisory_only"
    with pytest.raises(PiCopilotError) as error:
        _validate(registry, [{**ref, "digest": newer["digest"]}], frozen)
    assert error.value.code == "skill_attachment_stale"


def test_a_tampered_frozen_object_is_refused(registry):
    original = _install(registry)
    frozen = registry.snapshot()
    (registry.skill_objects / f"{original['digest']}.md").write_text(
        "---\nname: synthetic-helper\ndescription: Synthetic helper\n---\nTampered body."
    )
    with pytest.raises(PiCopilotError) as error:
        _validate(registry, [{"source": "user", "id": original["name"], "digest": original["digest"]}], frozen)
    assert error.value.code == "extension_skill_object_mismatch"


def test_explicit_user_source_cannot_load_a_shell_skill_with_the_same_name(registry):
    skill = registry.install_skill("---\nname: web-prototype\ndescription: Synthetic helper\n---\nUse concise synthetic labels.")
    frozen = registry.snapshot()
    refs = _validate(registry, [{"source": "user", "id": "web-prototype", "digest": skill["digest"]}], frozen)
    loaded = load_attached_users(PiSessionRecord(session_id="synthetic", extension_activation=frozen), refs, registry)
    assert loaded[0]["instructions"] == "Use concise synthetic labels."
    assert loaded[0]["digest"] == skill["digest"]


@pytest.fixture
def saved_study(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    return study_contexts.upsert_context({
        "question": "Is synthetic exposure associated with synthetic outcome?",
        "title": "Synthetic study",
    })


def _attach(study, refs=None):
    return study_contexts.upsert_context(
        {"id": study["id"], "attached_skills": refs or [builtin_attachment(BUILTIN["id"])]},
        expected_revision=study["revision"], _server_skill_attachments_write=True,
    )


def test_builtin_workflow_pins_the_planner_request_and_setup_receipt(saved_study):
    saved = _attach(saved_study)
    preferences = agent_pipeline_runs._research_user_preferences(saved)
    assert preferences["inferred_analysis_family"] == "association_study"
    constraints = json.loads(preferences["data_constraints"])
    assert constraints["method_skill_constraints"] == [{
        "id": BUILTIN["id"], "analysis_family": "association_study",
        "capability_id": "association_adjusted_v1", "action_ids": ["association.adjusted_association"],
    }]
    assert constraints["attached_skills"][0]["digest"] == builtin_attachment(BUILTIN["id"])["digest"]
    assert project_study_setup_receipt(saved).configuration["attached_skills"] == constraints["attached_skills"]
    assert saved["question"] == saved_study["question"]
    assert study_contexts.get_context(saved["id"])["attached_skills"] == saved["attached_skills"]
    assert study_contexts.scientific_configuration_sha256(saved) != study_contexts.scientific_configuration_sha256(saved_study)
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext, UserPreferences
    from easyicu.research_agent.agents.progressive_prompt_contracts import outline_study_anchors

    context = ResearchContext(
        research_question=saved["question"],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_patients=2, n_stays=2),
        variables=[], user_preferences=UserPreferences(**preferences),
    )
    outbound = outline_study_anchors(context)
    assert "association_adjusted_v1" in outbound
    assert "association.adjusted_association" in outbound
    assert constraints["attached_skills"][0]["digest"] in outbound


@pytest.mark.parametrize("choice", ["association_study", "descriptive_epidemiology"])
def test_question_conflict_uses_existing_disagreement_and_choice(saved_study, choice):
    original = study_contexts.upsert_context({"id": saved_study["id"], "question": "Describe the prevalence of a synthetic outcome."})
    saved = _attach(original)
    assert saved["analysis_design"] == {}
    gap = study_family_database_gap(saved, question=saved["question"], database="miiv")
    assert gap["code"] == FAMILY_DISAGREEMENT
    offered = dict(family_choice_texts(gap, "en"))
    chosen = researcher_choice(saved["question_family_check"], saved["question"], offered[choice])
    settled = study_contexts.upsert_context(
        {"id": saved["id"], "question_family_check": chosen}, _server_study_family_reading_write=True,
    )
    assert settled["analysis_design"]["analysis_family"] == choice
    assert study_family_database_gap(settled, question=settled["question"], database="miiv") is None
    constraints = json.loads(agent_pipeline_runs._research_user_preferences(settled)["data_constraints"])
    assert bool(constraints["method_skill_constraints"]) == (choice == "association_study")
    assert constraints["attached_skills"]  # The choice does not erase provenance.
    overwritten = study_contexts.upsert_context(
        {"id": settled["id"], "question_family_check": saved["question_family_check"]},
        _server_study_family_reading_write=True,
    )
    assert overwritten["question_family_check"] == chosen
    assert overwritten["analysis_design"]["analysis_family"] == choice


def test_a_model_cannot_remove_the_host_pin(saved_study):
    saved = _attach(saved_study)
    overwritten = study_contexts.upsert_context({
        "id": saved["id"], "analysis_design": {
            "analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
        },
    })
    assert overwritten["analysis_design"]["analysis_family"] == "association_study"
    with pytest.raises(study_contexts.StudyContextError) as error:
        study_contexts.upsert_context({"id": saved["id"], "attached_skills": []})
    assert error.value.detail["error"] == "skill_attachment_server_owned"


def test_an_attachment_also_compares_the_checked_conversation_reading(saved_study):
    from easyicu.webserver.question_family_check import check_question_family

    question = "Synthetic research question."
    original = study_contexts.upsert_context({
        "id": saved_study["id"], "question": question,
        "question_family_check": check_question_family({"family": "prediction_model", "words": question}, question),
    }, _server_study_family_reading_write=True)
    saved = _attach(original)
    assert saved["question_family_check"]["host_family"] == "prediction_model"
    assert saved["question_family_check"]["status"] == "host_disagrees"
    assert saved["analysis_design"] == {}


def test_regeneration_restores_the_original_attachment_and_family_choice_state(saved_study):
    saved_study = study_contexts.upsert_context({"id": saved_study["id"], "question": "Describe the prevalence of a synthetic outcome."})
    original = _attach(saved_study)
    snapshot = {
        "schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": original["id"],
        "source_revision": original["revision"], "configuration": project_study_setup_receipt(original).configuration,
    }
    chosen = researcher_choice(original["question_family_check"], original["question"], "Treat it as a descriptive study (proportions, rates or distributions)")
    later = study_contexts.upsert_context({"id": original["id"], "question_family_check": chosen}, _server_study_family_reading_write=True)
    restored = study_contexts.restore_turn_configuration_snapshot(original["id"], snapshot, expected_revision=later["revision"])
    assert restored["attached_skills"] == original["attached_skills"]
    assert restored["question_family_check"] == original["question_family_check"]
    assert restored["analysis_design"] == {}


@pytest.fixture
def service(tmp_path, registry, study_state):
    instance = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway(), extension_registry=registry)
    yield instance
    instance.close()


def _session(service):
    return service.create_session(project_id="synthetic-project", external_llm_opt_in=True)["session"]


def _send(service, session_id, **kwargs):
    allow_unrelated_message_test(service, session_id)
    result = service.send_message(session_id, project_id="synthetic-project", message="Explain the synthetic setup.", **kwargs)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = jobs.MANAGER.get(result["job_id"])
        if job.status in {"done", "failed", "cancelled"}:
            assert job.status == "done", job.error
            return
        time.sleep(0.01)
    pytest.fail("synthetic turn did not finish")


def test_project_defaults_are_frozen_at_creation_and_drafts_can_detach(service):
    old = _session(service)
    service.project_default_skills(project_id="synthetic-project", attached_skills=[BUILTIN])
    fresh = _session(service)
    service.project_default_skills(project_id="synthetic-project", attached_skills=[])
    later = _session(service)
    assert old["draft_attached_skills"] == []
    assert fresh["default_attached_skills"][0]["id"] == BUILTIN["id"]
    assert later["draft_attached_skills"] == []
    assert service._get_record(fresh["session_id"]).draft_attached_skills
    service.set_draft_skills(fresh["session_id"], project_id="synthetic-project", attached_skills=[])
    assert not service._get_record(fresh["session_id"]).draft_attached_skills
    assert service._get_record(fresh["session_id"]).default_attached_skills
    legacy = PiSessionRecord(session_id="legacy")
    assert not legacy.default_attached_skills and not legacy.draft_attached_skills


def test_attached_user_skill_reaches_one_turn_and_its_receipts(service, registry):
    installed = _install(registry)
    ref = {"source": "user", "id": installed["name"], "digest": installed["digest"]}
    _session(service)  # Initialize the project before its defaults are edited.
    service.project_default_skills(project_id="synthetic-project", attached_skills=[ref])
    session = _session(service)
    service.project_default_skills(project_id="synthetic-project", attached_skills=[])
    _install(registry, "New revision should not reach this conversation.")
    _send(service, session["session_id"])
    prompt = next(p for m, p, _ in service.gateway.calls if m == "session.prompt")
    assert prompt["attached_skill_context"][0]["instructions"] == "Use concise sentences."
    assert prompt["attached_skill_context"][0]["digest"] == ref["digest"]
    assert prompt["message"] == "Explain the synthetic setup."
    record = service._get_record(session["session_id"])
    assert attachment_payload(record.last_attached_skills) == [ref]
    assert not record.draft_attached_skills
    replay = service.replay_store.snapshot(session_id=session["session_id"], project_id="synthetic-project")
    assert replay["turns"][0]["attached_skills"] == [ref]
    _send(service, session["session_id"])
    prompts = [p for m, p, _ in service.gateway.calls if m == "session.prompt"]
    assert "attached_skill_context" not in prompts[-1]
    assert service.replay_store.attachments_for_turn(session_id=session["session_id"], project_id="synthetic-project", turn_index=0) == [ref]


def test_attachable_catalog_is_bounded_and_uses_frozen_user_identities(service, registry):
    original = _install(registry)
    session = _session(service)
    _install(registry, "A new version.")
    catalog = service.attachable_skills(session["session_id"], project_id="synthetic-project")
    assert len(catalog["items"]) < 128
    assert catalog["max_attached_skills"] == 8
    row = next(row for row in catalog["items"] if row["source"] == "user")
    assert row["digest"] == original["digest"]
    assert {"id", "title", "title_zh", "category", "enabled"} <= row.keys()
    with pytest.raises(PiCopilotError):
        service.set_draft_skills(session["session_id"], project_id="foreign", attached_skills=[BUILTIN])


def test_draft_api_and_turn_apply_the_builtin_through_the_real_study_store(service, monkeypatch, tmp_path):
    from easyicu.webserver.routes import skill_attachments as routes

    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "real-study-store.json")
    monkeypatch.setattr(study_contexts, "get_context", REAL_GET_CONTEXT)
    monkeypatch.setattr(study_contexts, "upsert_context", REAL_UPSERT)
    study = REAL_UPSERT({"id": "study-test", "question": "Is synthetic exposure associated with the synthetic outcome?"})
    monkeypatch.setattr(service, "_resolve_project_context", lambda **kwargs: study)
    service.project_store.bind("synthetic-project", study["id"])
    session = _session(service)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    result = routes.put_draft_skills(session["session_id"], routes.DraftSkillsRequest(project_id="synthetic-project", attached_skills=[BUILTIN]))
    assert result["attached_skills"][0]["digest"]
    _send(service, session["session_id"])
    saved = REAL_GET_CONTEXT(study["id"])
    preferences = agent_pipeline_runs._research_user_preferences(saved)
    assert preferences["inferred_analysis_family"] == "association_study"
    assert json.loads(preferences["data_constraints"])["attached_skills"] == result["attached_skills"]
    assert service.gateway.tool_contexts[-1].session.binding.study_revision == saved["revision"]
    assert not service._get_record(session["session_id"]).draft_attached_skills


def test_invalid_draft_is_atomic_and_a_busy_turn_cannot_be_edited(service):
    session = _session(service)
    service.set_draft_skills(session["session_id"], project_id="synthetic-project", attached_skills=[BUILTIN])
    for refs, expected in (([{**BUILTIN, "version": "stale"}], "skill_attachment_stale"), ([], "pi_session_busy")):
        if not refs:
            service._busy_sessions.add(session["session_id"])
        with pytest.raises(PiCopilotError) as error:
            service.set_draft_skills(session["session_id"], project_id="synthetic-project", attached_skills=refs)
        assert error.value.code == expected
        assert service._get_record(session["session_id"]).draft_attached_skills
    service._busy_sessions.clear()


def test_default_skills_survive_project_archiving_and_reopening(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import project_authority

    monkeypatch.setattr(project_authority, "_MAX_PROJECTS", 1)
    store = project_authority.ProjectAuthorityStore(tmp_path / "projects.json")
    store.bind("first", "study-first")
    store.set_default_skills("first", [BUILTIN])
    store.bind("second", "study-second")
    assert store.binding("first").default_skills[0].id == BUILTIN["id"]
    store.set_default_skills("first", [])
    store.bind("first", "study-first")
    assert not store.binding("first").default_skills
