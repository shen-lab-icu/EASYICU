"""Synthetic mention admission and reading contracts, without ASGI transport."""

import json
import time

import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.webserver import agent_pipeline_runs, jobs, sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.projections import project_study_setup_receipt
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes.pi_copilot import PiMessageRequest
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway, study_state as study_state,
)

REAL_GET_CONTEXT = study_contexts.get_context
REAL_UPSERT = study_contexts.upsert_context


@pytest.fixture
def service(tmp_path, study_state):
    instance = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway(),
        extension_registry=ExtensionRegistry(tmp_path / "extensions"),
    )
    yield instance
    instance.close()


def session(service):
    return service.create_session(project_id="synthetic-project", external_llm_opt_in=True)["session"]["session_id"]


def resolve(service, sid, **kwargs):
    return service.resolve_mention(sid, project_id="synthetic-project", **kwargs)


def test_turn_request_accepts_a_typed_concept_without_rewriting_the_question():
    request = PiMessageRequest(
        project_id="synthetic-project", message="Compare these two measurements.",
        mentions=[{"kind": "concept", "id": "lact", "digest": "a" * 64, "role": "exposure"}],
    )
    assert request.mentions[0].id == "lact"
    assert request.message == "Compare these two measurements."


def test_resolved_concept_carries_definition_and_database_availability(service):
    item = resolve(service, session(service), kind="concept", id="lact", role="exposure")
    assert item["reference"]["id"] == "lact"
    assert len(item["reference"]["digest"]) == 64
    assert item["object"]["definition"]["description"]
    assert item["object"]["availability"]["miiv"]["concept"] == "lact"


@pytest.mark.parametrize("identity,code", [("unknown_synthetic_concept", "mention_unknown"), ("lact", "mention_stale")])
def test_unknown_or_stale_reference_is_refused_before_draft_changes(service, identity, code):
    sid = session(service)
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[
            {"kind": "concept", "id": identity, "digest": "0" * 64, "role": "exposure"},
        ])
    assert error.value.code == code


def test_concept_pins_reading_setup_receipt_and_planner_provenance(service, tmp_path, monkeypatch):
    item = resolve(service, session(service), kind="concept", id="lact", role="exposure")
    # A real study store, isolated from the service fixture's in-memory study.
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(study_contexts, "get_context", REAL_GET_CONTEXT)
    monkeypatch.setattr(study_contexts, "upsert_context", REAL_UPSERT)
    saved = study_contexts.upsert_context({
        "question": "Is the synthetic measurement associated with the outcome?",
        "mentioned_objects": [item],
    }, _server_skill_attachments_write=True)
    assert saved["primary_exposure"] == "lact"
    changed = study_contexts.upsert_context({"id": saved["id"], "primary_exposure": "crea"})
    assert changed["primary_exposure"] == "lact"
    assert project_study_setup_receipt(changed).configuration["mentioned_objects"] == [item]
    constraints = json.loads(agent_pipeline_runs._research_user_preferences(changed)["data_constraints"])
    assert constraints["mentions"] == [item["reference"]]


@pytest.fixture
def host(service, tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "real-studies.json")
    monkeypatch.setattr(study_contexts, "get_context", REAL_GET_CONTEXT)
    monkeypatch.setattr(study_contexts, "upsert_context", REAL_UPSERT)
    study = REAL_UPSERT({"id": "study-test", "question": "Is a synthetic measurement associated with an outcome?"})
    monkeypatch.setattr(service, "_resolve_project_context", lambda **kwargs: REAL_GET_CONTEXT(study["id"]))
    service.project_store.bind("synthetic-project", study["id"])
    return service, session(service), study["id"]


def send(service, sid, **kwargs):
    result = service.send_message(sid, project_id="synthetic-project", message="Explain the synthetic setup.", **kwargs)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = jobs.MANAGER.get(result["job_id"])
        if job.status in {"done", "failed", "cancelled"}:
            assert job.status == "done", job.error
            return
        time.sleep(0.01)
    pytest.fail("synthetic turn did not finish")


def test_draft_turn_and_replay_keep_exact_mentions_and_clear_only_the_sent_draft(host):
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="lact", role="exposure")
    service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[item["reference"]])
    send(service, sid)
    saved = REAL_GET_CONTEXT(study_id)
    assert saved["execution_concepts"]["primary_exposure"] == "lact"
    assert saved["mentioned_objects"] == [item]
    record = service._get_record(sid)
    assert record.last_mentions[0].id == "lact" and record.draft_mentions == []
    replay = service.replay_store.snapshot(session_id=sid, project_id="synthetic-project")
    assert replay["turns"][0]["mentioned_objects"] == [item]
    assert service.replay_store.mentions_for_turn(session_id=sid, project_id="synthetic-project", turn_index=0) == [item["reference"]]
    assert service.gateway.tool_contexts[-1].session.binding.study_revision == saved["revision"]
    send(service, sid)
    assert service._get_record(sid).last_mentions == []
    assert REAL_GET_CONTEXT(study_id)["mentioned_objects"] == [item]


@pytest.mark.parametrize("role,field", [("exposure", "primary_exposure"), ("outcome", "outcome"), ("covariate", "covariates")])
def test_explicit_concept_roles_survive_a_contrary_model_reading(host, role, field):
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="lact", role=role)
    send(service, sid, mentions=[item["reference"]])
    changed = REAL_UPSERT({"id": study_id, field: ["crea"] if role == "covariate" else "crea",
                          "execution_concepts": {field: ["crea"] if role == "covariate" else "crea"}})
    actual = changed["execution_concepts"][field]
    assert "lact" in actual if role == "covariate" else actual == "lact"


def test_an_unassigned_unavailable_concept_uses_the_existing_capability_refusal(host):
    from easyicu.webserver.study_family_scope import study_family_database_gap
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="d_dimer")
    assert item["object"]["availability"]["eicu"]["available"] is False
    send(service, sid, mentions=[item["reference"]])
    saved = REAL_GET_CONTEXT(study_id)
    assert not saved["primary_exposure"] and not saved["outcome"]
    gap = study_family_database_gap(saved, question=saved["question"], database="eicu")
    assert gap["code"] == "study_family_database_out_of_scope"
    assert {"capability": "concept:d_dimer", "code": "required_concept_unmapped"}.items() <= gap["missing_capabilities"][0].items()
    saved = REAL_UPSERT({"id": study_id, "data_source": {"database": "eicu"}, "analysis_design": {
        "analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
    }})
    constraints = json.loads(agent_pipeline_runs._research_user_preferences(saved)["data_constraints"])
    assert "d_dimer" in constraints["database_study_requirements"]["concepts"]


def test_mortality_meaning_changes_remain_an_unapplied_substitution(host):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import assess_study
    from easyicu.webserver.research_capability_scope import study_requirements
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="death", role="outcome")
    send(service, sid, mentions=[item["reference"]])
    saved = REAL_GET_CONTEXT(study_id)
    decision = assess_study(research_capability_profile("hirid"), study_requirements(saved, family="descriptive_epidemiology"))
    assert decision.status == "substitution"
    assert decision.substitutions[0].reason_code == "mortality_meaning_differs"
    assert saved["outcome"] == "death"


@pytest.mark.parametrize("source", ["database", "registered_source"])
def test_database_mentions_need_no_data_handling_declaration(host, monkeypatch, tmp_path, source):
    service, sid, study_id = host
    registry = {"sources": [{"id": "synthetic-source", "ok": True, "database": "miiv", "path": str(tmp_path / "synthetic-export")} ]}
    monkeypatch.setattr(sources, "load_registry", lambda: registry)
    item = resolve(service, sid, kind="database", id="miiv" if source == "database" else "synthetic-source", source=source)
    # A declaration is recorded, not required (owner decision 2026-10-27).
    monkeypatch.setattr(service, "_data_handling_for_session", lambda record: "unspecified")
    send(service, sid, mentions=[item["reference"]])
    assert any(method == "session.prompt" for method, _, _ in service.gateway.calls)


def test_a_registered_source_mention_binds_only_that_registration(host, monkeypatch, tmp_path):
    service, sid, study_id = host
    selected = {"id": "synthetic-source", "ok": True, "database": "miiv", "path": str(tmp_path / "synthetic-export"), "label": "Synthetic export"}
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [selected]})
    item = resolve(service, sid, kind="database", id=selected["id"], source="registered_source")
    assert str(tmp_path) not in json.dumps(item)
    send(service, sid, mentions=[item["reference"]])
    assert REAL_GET_CONTEXT(study_id)["data_source"]["path"] == selected["path"]
    assert service._get_record(sid).data_source_authorization.status == "confirmed"
    with pytest.raises(study_contexts.StudyContextError) as error:
        REAL_UPSERT({"id": study_id, "data_source": {"database": "miiv", "path": str(tmp_path / "other-export")}})
    assert error.value.detail["error"] == "mention_source_conflict"


def test_a_database_id_guides_source_selection_and_refuses_a_different_database(host):
    service, sid, study_id = host
    item = resolve(service, sid, kind="database", id="miiv")
    send(service, sid, mentions=[item["reference"]])
    saved = REAL_GET_CONTEXT(study_id)
    assert saved["data_source"]["database"] == "miiv"
    assert not saved["data_source"]["path"]
    with pytest.raises(study_contexts.StudyContextError) as error:
        REAL_UPSERT({"id": study_id, "data_source": {"database": "eicu"}})
    assert error.value.detail["error"] == "mention_database_conflict"
    assert error.value.detail["field"] == "mentioned_objects"


def test_source_registration_changes_refuse_the_old_digest(host, monkeypatch, tmp_path):
    service, sid, _ = host
    row = {"id": "synthetic-source", "ok": True, "database": "miiv", "path": str(tmp_path / "synthetic-export")}
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [row]})
    item = resolve(service, sid, kind="database", id=row["id"], source="registered_source")
    row["path"] = str(tmp_path / "replacement")
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[item["reference"]])
    assert error.value.code == "mention_stale"


def test_skill_mentions_use_sk2_loading_and_freeze_the_original_user_skill(host):
    service, _, _ = host
    installed = service.extension_registry.install_skill("---\nname: synthetic-note\ndescription: Synthetic note\n---\nUse short sentences.")
    sid = session(service)
    item = resolve(service, sid, kind="skill", id="synthetic-note", source="user")
    assert item["reference"]["attachment"]["digest"] == installed["digest"]
    service.extension_registry.install_skill("---\nname: synthetic-note\ndescription: Synthetic note\n---\nLater instructions.")
    send(service, sid, mentions=[item["reference"]])
    prompt = next(params for method, params, _ in service.gateway.calls if method == "session.prompt")
    assert prompt["attached_skill_context"][0]["instructions"] == "Use short sentences."
    replay = service.replay_store.snapshot(session_id=sid, project_id="synthetic-project")
    assert replay["turns"][0]["attached_skills"] == [item["reference"]["attachment"]]


@pytest.mark.parametrize("kind", ["database", "concept", "file", "skill"])
def test_picker_is_bounded_and_project_scoped(service, kind):
    sid = session(service)
    result = service.search_mentions(sid, project_id="synthetic-project", kind=kind, limit=1)
    assert len(result["items"]) <= 1 and result["limit"] == 1
    with pytest.raises(PiCopilotError):
        service.search_mentions(sid, project_id="another-project", kind=kind)
    with pytest.raises(PiCopilotError) as error:
        service.search_mentions(sid, project_id="synthetic-project", kind=kind, limit=41)
    assert error.value.code == "mention_search_invalid"


def test_mentions_cannot_be_forged_through_model_study_updates(host):
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="lact")
    with pytest.raises(study_contexts.StudyContextError) as error:
        REAL_UPSERT({"id": study_id, "mentioned_objects": [item]})
    assert error.value.detail["error"] == "mention_server_owned"


@pytest.mark.parametrize("damage,code", [("duplicate", "mention_duplicate"), ("role", "mention_role_conflict"), ("limit", "mention_invalid")])
def test_invalid_batches_preserve_the_entire_existing_draft(host, damage, code):
    service, sid, _ = host
    item = resolve(service, sid, kind="concept", id="lact", role="exposure")["reference"]
    service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[item])
    other = resolve(service, sid, kind="concept", id="crea", role="exposure")["reference"]
    value = [item, other] if damage == "role" else [item] * (17 if damage == "limit" else 2)
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=value)
    assert error.value.code == code
    assert service._get_record(sid).draft_mentions[0].id == "lact"


def test_busy_and_queued_turns_cannot_lose_draft_mentions(host):
    service, sid, _ = host
    item = resolve(service, sid, kind="concept", id="lact")["reference"]
    service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[item])
    service._busy_sessions.add(sid)
    try:
        with pytest.raises(PiCopilotError) as error:
            service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[])
        assert error.value.code == "pi_session_busy"
        with pytest.raises(PiCopilotError) as error:
            service.send_message(sid, project_id="synthetic-project", message="Explain this.", message_origin="typed", queue_if_busy=True)
        assert error.value.code == "pi_followup_plain_message_required"
        assert service._get_record(sid).draft_mentions[0].id == "lact"
    finally:
        service._busy_sessions.clear()


def test_a_file_mention_uses_project_material_bytes_and_keeps_contents_turn_local(host):
    service, sid, study_id = host
    material = service.workspace.write_file("synthetic-project", "notes/reading.md", "Synthetic study note.")
    picker = service.search_mentions(sid, project_id="synthetic-project", kind="file", query="reading")
    item = resolve(service, sid, kind="file", id=picker["items"][0]["id"])
    assert item["object"]["sha256"] == material["sha256"]
    assert item["object"]["scientific_evidence"] is False
    send(service, sid, mentions=[item["reference"]])
    prompt = next(params for method, params, _ in service.gateway.calls if method == "session.prompt")
    assert prompt["mentioned_objects"][0]["content"]["text"] == "Synthetic study note."
    assert "Synthetic study note." not in json.dumps(REAL_GET_CONTEXT(study_id))
    replay = service.replay_store.snapshot(session_id=sid, project_id="synthetic-project")
    assert replay["turns"][0]["mentioned_objects"] == [item]


def test_material_changes_refuse_an_old_reference_on_send_and_regeneration(host):
    service, sid, _ = host
    material = service.workspace.write_file("synthetic-project", "note.md", "Original synthetic note.")
    picker = service.search_mentions(sid, project_id="synthetic-project", kind="file")
    item = resolve(service, sid, kind="file", id=picker["items"][0]["id"])
    send(service, sid, mentions=[item["reference"]])
    service.workspace.edit_file("synthetic-project", "note.md", old_text="Original", new_text="Changed", expected_sha256=material["sha256"])
    with pytest.raises(PiCopilotError) as error:
        service._prepare_turn_mentions(service._get_record(sid), [], {"turn_index": 0})
    assert error.value.code == "mention_stale"
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[item["reference"]])
    assert error.value.code == "mention_stale"


def test_a_file_from_another_project_is_unknown_and_data_files_are_not_offered(host):
    from easyicu.webserver.mention_objects import file_id
    service, sid, _ = host
    service.workspace.write_file("foreign-project", "notes.md", "Foreign synthetic note.")
    service.workspace.write_file("synthetic-project", "notes.md", "Local synthetic note.")
    service.workspace.write_file("synthetic-project", "data/notes.md", "Synthetic data directory.")
    service.workspace.write_file("synthetic-project", "table.csv", "category,count\nA,2\n")
    assert [row["label"] for row in service.search_mentions(sid, project_id="synthetic-project", kind="file")["items"]] == ["notes.md"]
    with pytest.raises(PiCopilotError) as error:
        resolve(service, sid, kind="file", id=file_id("foreign-project", "notes.md"))
    assert error.value.code == "mention_unknown"
    with pytest.raises(PiCopilotError) as error:
        resolve(service, sid, kind="file", id=file_id("synthetic-project", "table.csv"))
    assert error.value.code == "tabular_read_unavailable_when_hosted"


def test_a_later_workspace_read_cannot_change_a_pinned_file(host):
    from easyicu.extensions.skill_attachments import parse_mentions
    from easyicu.webserver.pi_copilot.contracts import ToolExecutionContext
    from easyicu.webserver.pi_copilot.tools import _read_project_file
    service, sid, _ = host
    written = service.workspace.write_file("synthetic-project", "note.md", "Original synthetic note.")
    picker = service.search_mentions(sid, project_id="synthetic-project", kind="file")
    item = resolve(service, sid, kind="file", id=picker["items"][0]["id"])
    record = service._get_record(sid).model_copy(update={"agent_mode": "workspace", "last_mentions": parse_mentions([item["reference"]])})
    service.workspace.edit_file("synthetic-project", "note.md", old_text="Original", new_text="Changed", expected_sha256=written["sha256"])
    with pytest.raises(PiCopilotError) as error:
        _read_project_file(ToolExecutionContext(session=record, workspace=service.workspace), {"file": "note.md"})
    assert error.value.code == "mention_stale"


def test_a_later_turn_can_replace_one_pinned_role_and_regeneration_recovers_the_original(host):
    service, sid, study_id = host
    first = resolve(service, sid, kind="concept", id="lact", role="exposure")
    second = resolve(service, sid, kind="concept", id="crea", role="exposure")
    send(service, sid, mentions=[first["reference"]])
    original = REAL_GET_CONTEXT(study_id)
    snapshot = {"schema_version": "easyicu.pi-turn-study-snapshot/1", "study_context_id": study_id,
                "source_revision": original["revision"], "configuration": project_study_setup_receipt(original).configuration}
    send(service, sid, mentions=[second["reference"]])
    assert REAL_GET_CONTEXT(study_id)["primary_exposure"] == "crea"
    regenerated = service._prepare_turn_mentions(service._get_record(sid), [second["reference"]], {"turn_index": 0})
    assert regenerated == [first]
    restored = study_contexts.restore_turn_configuration_snapshot(study_id, snapshot, expected_revision=REAL_GET_CONTEXT(study_id)["revision"])
    assert restored["primary_exposure"] == "lact"
    assert restored["mentioned_objects"] == [first]


def test_resolved_metadata_cannot_be_tampered_with_even_by_a_study_snapshot(host):
    service, sid, study_id = host
    item = resolve(service, sid, kind="concept", id="lact")
    item["object"]["definition"]["description"] = "Forged synthetic meaning."
    with pytest.raises(study_contexts.StudyContextError) as error:
        REAL_UPSERT({"id": study_id, "mentioned_objects": [item]}, _server_skill_attachments_write=True)
    assert error.value.detail["error"] == "mention_stale"


def test_direct_routes_reach_the_scoped_owners_and_return_stable_errors(host, monkeypatch):
    from fastapi import HTTPException
    from easyicu.webserver.routes import mentions as routes
    service, sid, _ = host
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    picker = routes.search_mentions(sid, "synthetic-project", "concept", "lact", 1)
    assert picker["items"][0]["id"] == "lact"
    item = routes.resolve_mention(sid, routes.ResolveMentionRequest(project_id="synthetic-project", kind="concept", id="lact"))
    result = routes.put_draft_mentions(sid, routes.DraftMentionsRequest(project_id="synthetic-project", mentions=[item["reference"]]))
    assert result["mentioned_objects"] == [item]
    with pytest.raises(HTTPException) as error:
        routes.resolve_mention(sid, routes.ResolveMentionRequest(project_id="synthetic-project", kind="concept", id="missing_synthetic"))
    assert error.value.detail == {"error": "mention_unknown"}


@pytest.mark.parametrize("query,limit", [("x" * 121, 1), ("", 0), ("", True)])
def test_picker_rejects_invalid_search_bounds(service, query, limit):
    with pytest.raises(PiCopilotError) as error:
        service.search_mentions(session(service), project_id="synthetic-project", kind="concept", query=query, limit=limit)
    assert error.value.code == "mention_search_invalid"


@pytest.mark.parametrize("selector", [
    {"kind": "concept", "id": "../lact"},
    {"kind": "concept", "id": "lact", "source": "database"},
    {"kind": "database", "id": "miiv", "role": "exposure"},
])
def test_resolve_does_not_ignore_invalid_coordinates(service, selector):
    with pytest.raises(PiCopilotError) as error:
        resolve(service, session(service), **selector)
    assert error.value.code == "mention_invalid"


def test_two_database_pins_cannot_be_admitted_in_the_same_turn(host):
    service, sid, _ = host
    refs = [resolve(service, sid, kind="database", id=database)["reference"] for database in ("miiv", "eicu")]
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=refs)
    assert error.value.code == "mention_database_conflict"


def test_skill_mentions_and_legacy_attachments_share_one_budget(host):
    from easyicu.extensions.method_attachments import builtin_attachment
    service, sid, _ = host
    item = resolve(service, sid, kind="skill", source="builtin", id="adjusted-exposure-outcome-study")
    same = builtin_attachment("adjusted-exposure-outcome-study")
    send(service, sid, mentions=[item["reference"]], attached_skills=[same])
    assert len(service._get_record(sid).last_attached_skills) == 1


def test_unknown_mention_precedes_source_mutation_in_the_same_message(host, monkeypatch, tmp_path):
    service, sid, study_id = host
    source = {"id": "synthetic-source", "ok": True, "database": "miiv", "path": str(tmp_path / "synthetic-export")}
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [source]})
    before = REAL_GET_CONTEXT(study_id)
    with pytest.raises(PiCopilotError) as error:
        service.send_message(sid, project_id="synthetic-project", message=f"Use {source['path']}",
                             mentions=[{"kind": "concept", "id": "unknown_synthetic", "digest": "a" * 64}])
    assert error.value.code == "mention_unknown"
    assert REAL_GET_CONTEXT(study_id) == before


def test_an_active_study_refuses_mentions_before_submission(host):
    service, sid, study_id = host
    current = REAL_UPSERT({"id": study_id, "active_job_id": "synthetic-running-job"})
    record = service._get_record(sid)
    record.binding = service._binding_for_context(current)
    service._save_record(record)
    ref = resolve(service, sid, kind="concept", id="lact")["reference"]
    with pytest.raises(PiCopilotError) as error:
        service.send_message(sid, project_id="synthetic-project", message="Explain this synthetic measurement.", mentions=[ref])
    assert error.value.code == "study_context_active_job_conflict"
    assert REAL_GET_CONTEXT(study_id) == current


@pytest.mark.parametrize("field", ["description", "target"])
def test_reference_digest_changes_with_catalogue_definition_metadata(service, monkeypatch, field):
    from easyicu import resources
    sid = session(service)
    original = resolve(service, sid, kind="concept", id="lact")
    dictionary = resources.load_dictionary(include_sofa2=True)
    from copy import deepcopy
    synthetic = deepcopy(dictionary)
    setattr(synthetic.get("lact"), field, "Changed synthetic definition.")
    monkeypatch.setattr(resources, "load_dictionary", lambda **kwargs: synthetic)
    newer = resolve(service, sid, kind="concept", id="lact")
    assert original["reference"]["digest"] != newer["reference"]["digest"]
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[original["reference"]])
    assert error.value.code == "mention_stale"


def test_legacy_skill_and_mention_lists_cannot_each_claim_a_separate_skill_budget(host):
    service, _, _ = host
    refs = []
    for index in range(8):
        installed = service.extension_registry.install_skill(f"---\nname: synthetic-note-{index}\ndescription: Synthetic note\n---\nUse short sentences.")
        refs.append({"source": "user", "id": installed["name"], "digest": installed["digest"]})
    sid = session(service)
    from easyicu.extensions.method_attachments import builtin_attachment
    extra = builtin_attachment("adjusted-exposure-outcome-study")
    with pytest.raises(PiCopilotError) as error:
        service.send_message(sid, project_id="synthetic-project", message="Explain this synthetic study.",
                             attached_skills=refs, mentions=[{"kind": "skill", "attachment": extra}])
    assert error.value.code == "skill_attachment_invalid"


def test_combined_file_context_is_bounded_before_any_turn_or_study_mutation(host):
    service, sid, study_id = host
    for index in range(16):
        service.workspace.write_file("synthetic-project", f"note-{index}.md", "é" * 1900)
    picks = service.search_mentions(sid, project_id="synthetic-project", kind="file")
    refs = [resolve(service, sid, kind="file", id=pick["id"])["reference"] for pick in picks["items"]]
    before = REAL_GET_CONTEXT(study_id)
    with pytest.raises(PiCopilotError) as error:
        service.send_message(sid, project_id="synthetic-project", message="Explain these synthetic notes.", mentions=refs)
    assert error.value.code == "mention_context_too_large"
    assert REAL_GET_CONTEXT(study_id) == before


@pytest.mark.parametrize("skill_id", ["six-database-cohort-extraction", "clinical-concept-definitions", "sample-size-and-power"])
def test_a_catalog_skill_without_an_attachment_capability_is_not_offered_or_admitted(service, skill_id):
    from easyicu.extensions.method_attachments import builtin_attachment
    sid = session(service)
    # These catalogue-only packages currently have no sk2 method capability;
    # their absence must be a stable refusal, never an AssertionError/500.
    with pytest.raises(PiCopilotError) as error:
        service.set_draft_mentions(sid, project_id="synthetic-project", mentions=[
            {"kind": "skill", "attachment": builtin_attachment(skill_id)},
        ])
    assert error.value.code == "skill_attachment_disabled"
    picker = service.search_mentions(sid, project_id="synthetic-project", kind="skill", query=skill_id)
    assert picker["items"] == []


def test_run_binding_retains_mentions_and_a_candidate_changes_scientific_identity(host, tmp_path):
    service, sid, study_id = host
    before = REAL_GET_CONTEXT(study_id)
    item = resolve(service, sid, kind="concept", id="lact")
    send(service, sid, mentions=[item["reference"]])
    saved = REAL_GET_CONTEXT(study_id)
    assert study_contexts.scientific_configuration_sha256(saved) != study_contexts.scientific_configuration_sha256(before)
    export = str(tmp_path / "synthetic-export")
    saved = REAL_UPSERT({"id": study_id, "data_source": {"database": "miiv", "path": export}})
    binding = study_contexts.build_agent_context_binding(saved, export_path=export)
    assert binding["informational"]["mentioned_objects"] == [item]
    assert REAL_GET_CONTEXT(study_id)["mentioned_objects"] == [item]


def test_a_long_project_file_uses_an_explicit_excerpt_with_its_full_digest(host):
    service, sid, _ = host
    body = "Synthetic project note. " * 200
    written = service.workspace.write_file("synthetic-project", "long-note.md", body)
    picker = service.search_mentions(sid, project_id="synthetic-project", kind="file")
    item = resolve(service, sid, kind="file", id=picker["items"][0]["id"])
    send(service, sid, mentions=[item["reference"]])
    prompt = next(params for method, params, _ in service.gateway.calls if method == "session.prompt")
    material = prompt["mentioned_objects"][0]["content"]
    assert material["text"] == body[:2000]
    assert material["truncated"] is True
    assert material["sha256"] == written["sha256"]


def test_concept_search_offers_the_entry_first_and_keeps_bookkeeping_to_its_exact_name(service):
    sid = session(service)

    def ids(query, limit=40):
        return [row["id"] for row in service.search_mentions(
            sid, project_id="synthetic-project", kind="concept", query=query, limit=limit)["items"]]

    found = ids("aki")
    assert found[0] == "aki_stage_reference"
    assert "aki_reference" in found and "aki_stage_source_native" in found
    assert not {"aki_reference_status", "aki_source_native_authority", "kdigo_aki"} & set(found)
    # A bookkeeping column or loader entry point answers its exact name.
    assert ids("aki_reference_status") == ["aki_reference_status"]
    assert "kdigo_aki" in ids("kdigo_aki")
    # The client shows Chinese names, so a Chinese query finds the concept.
    assert "lact" in ids("乳酸")
    assert "aki_stage_rrt_reference" in ids("肾脏替代治疗")
    assert all("_terms" not in row for row in service.search_mentions(
        sid, project_id="synthetic-project", kind="concept", query="乳酸")["items"])
