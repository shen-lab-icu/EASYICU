"""Conversation quick answers use synthetic data and direct host owners only."""

import json

import pandas as pd
import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver import sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.tool_catalog import TOOL_ARGUMENTS, TOOL_MODEL_SCHEMAS
from easyicu.webserver.pi_copilot.tools import execute_tool
from tests.support.native_outcome_export import typed_native_export
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


def reading(question="What is the median of the first-day peak lactate?", **updates):
    return {"question": question, "intent": "describe", "population": {"criteria": []},
            "measure": {"concept": "lact", "summary": "max", "statistic": "median",
                        "window": {"start_hours": 0, "end_hours": 24}}, **updates}


@pytest.fixture
def host(tmp_path, monkeypatch, request):
    database = getattr(request, "param", "eicu_demo")
    export = typed_native_export(
        tmp_path / "synthetic", database=database,
        longitudinal=pd.DataFrame({"stay_id": range(60), "charttime": [1.0] * 60,
                                   "lact": [float(i % 15 + 1) / 2 for i in range(60)]}),
        longitudinal_concepts=["lact"],
        statics=pd.DataFrame({"stay_id": range(60), "age": [45] * 60, "sex": ["Male"] * 30 + ["Female"] * 30}),
        static_concepts=["age", "sex"],
        outcome=pd.DataFrame({"stay_id": range(60), "charttime": [0.0] * 60, "icu_death": [False] * 30 + [True] * 30}),
        outcome_concepts=["icu_death"],
    )
    registry = {"sources": [{"id": "src_synthetic", "database": database, "path": str(export), "ok": True}]}
    monkeypatch.setattr(sources, "load_registry", lambda: registry)
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    study = study_contexts.upsert_context({"id": "study-quick", "question": "",
                                         "data_source": {"database": database, "path": str(export)}})
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway(),
                              extension_registry=ExtensionRegistry(tmp_path / "extensions"))
    service.project_store.bind("project-quick", study["id"])
    record = PiSessionRecord(session_id="pi-quick", project_id="project-quick", external_llm_opt_in=True,
                             credentialed_data_in_history=False,
                             binding=AuthorityBinding(study_context_id=study["id"], study_revision=study["revision"]))
    service._save_record(record)
    yield service, record, registry
    service.close()


def submit(host, request=None, *, message=None, job_id="message-quick"):
    service, record, _ = host
    request = request or reading()
    service.replay_store.start_turn(session_id=record.session_id, project_id=record.project_id,
                                    job_id=job_id, allowed_actions=[], user_message=message or request["question"])
    context = ToolExecutionContext(
        session=record, user_message=message or request["question"],
        data_handling=service._data_handling_for_session(record),
        quick_answer_submit=lambda ctx, read: service._submit_quick_answer(ctx, read, message_job_id=job_id),
        authority_validator=lambda binding: service._binding_stale_details(binding, project_id=record.project_id),
    )
    result = execute_tool("easyicu_quick_answer", {"reading": request}, context)
    service.replay_store.finish_turn(session_id=record.session_id, project_id=record.project_id,
                                     job_id=job_id, status="done")
    return result["details"]["quick_answer"]


def approve(host, attachment, **overrides):
    service, record, _ = host
    return service.approve_quick_answer(record.session_id, **{
        "project_id": record.project_id, "quick_answer_id": attachment["quick_answer_id"],
        "card_sha256": attachment["proposal"]["card"]["sha256"], **overrides,
    })["quick_answer"]


def test_quick_tool_accepts_intent_only():
    arguments = TOOL_ARGUMENTS["easyicu_quick_answer"]
    assert arguments.model == ("reading",)
    assert arguments.host == ()
    assert arguments.required == ("reading",)

    from easyicu.webserver.pi_copilot.quick_answer_store import QuickQuestionInput

    assert TOOL_MODEL_SCHEMAS["easyicu_quick_answer"] == QuickQuestionInput.model_json_schema()


def test_tool_card_click_answer_binds_every_number_and_reloads(host):
    service, record, _ = host
    attachment = submit(host)
    assert attachment["proposal"]["status"] == "approval_required", attachment
    assert attachment["answer"] is None
    answer = approve(host, attachment)["answer"]
    assert answer["status"] == "ok", answer
    assert next(row["value"] for row in answer["rows"] if row["statistic"] == "median") == 4
    directory = service._quick_store().path(record.session_id, attachment["quick_answer_id"]).parent
    evidence = EvidenceStore(directory / "execution")
    for row, sentence in zip(answer["rows"], answer["sentences"], strict=True):
        coordinate = row["evidence"]
        assert sentence["evidence"] == coordinate
        claim = next(c for c in evidence.numeric_claims() if c.source_field == coordinate["source_field"])
        assert claim.canonical == row["value"]
        assert evidence.get(coordinate["evidence_id"]).sha256 == coordinate["sha256"]
    assert approve(host, attachment)["answer"] == answer
    restarted = PiCopilotService(store_path=service.store_path, gateway=FakeGateway(),
                                 extension_registry=service.extension_registry)
    try:
        public = restarted.get_session(record.session_id, project_id=record.project_id)["session"]
        assert public["quick_answers"][0]["answer"] == answer
        assert public["quick_answers"][0]["message_job_id"] == "message-quick"
        assert str(directory) not in json.dumps(public["quick_answers"])
        assert len([row for row in public["conversation_replay"]["turns"] if row.get("action_code") == "approve_quick_answer"]) == 1
    finally:
        restarted.close()


@pytest.mark.parametrize("question,reason", [
    ("Is the measurement associated after adjusting for age?", "quick_requires_adjusted_study"),
    ("How many are still alive?", "quick_requires_survival_study"),
])
def test_simple_wording_cannot_skip_full_study_and_handoff_keeps_reading(host, question, reason):
    service, record, _ = host
    attachment = submit(host, reading(question))
    assert attachment["proposal"]["status"] == "full_study"
    assert attachment["proposal"]["reason"] == reason
    result = service.expand_quick_answer(record.session_id, project_id=record.project_id,
                                         quick_answer_id=attachment["quick_answer_id"])
    saved = study_contexts.get_context(result["study_context_id"])
    assert saved["question"] == question
    assert saved["quick_answer_handoff"] == result["expand"]
    from easyicu.webserver.agent_pipeline_runs import _research_user_preferences

    constraints = json.loads(_research_user_preferences(saved)["data_constraints"])
    assert constraints["quick_answer_handoff"]["reading"]["measure"] == attachment["proposal"]["expand"]["reading"]["measure"]


def test_stale_digest_never_executes(host):
    service, record, _ = host
    attachment = submit(host)
    with pytest.raises(PiCopilotError, match="quick_approval_mismatch"):
        approve(host, attachment, card_sha256="0" * 64)
    assert not (service._quick_store().path(record.session_id, attachment["quick_answer_id"]).parent / "execution").exists()


def test_a_proportion_uses_only_the_host_evidence(host):
    attachment = submit(host, reading("What fraction of stays is male?", measure={
        "concept": "sex", "statistic": "proportion", "level": "Male",
    }))
    answer = approve(host, attachment)["answer"]
    assert answer["status"] == "ok", answer["reason"]
    values = {row["statistic"]: row["value"] for row in answer["rows"]}
    assert (values["numerator"], values["denominator"], values["percentage"]) == (30, 60, 50)


def test_small_cells_are_not_persisted_as_an_answer(host):
    attachment = submit(host, reading("How many stays have the first-day peak lactate over seven?", population={"criteria": [{
        "id": "c1", "kind": "measurement", "source": "question", "role": "include", "quote": "lactate over seven",
        "concept": "lact", "op": ">", "value": 7, "summary": "max",
        "window": {"start_hours": 0, "end_hours": 24},
    }]}))
    answer = approve(host, attachment)["answer"]
    assert answer["reason"] == "quick_small_cells_suppressed"
    assert not answer["rows"] and not answer["sentences"]


def test_chat_yes_is_not_approval_and_reader_cannot_replace_the_question(host):
    service, record, _ = host
    first = submit(host)
    second = submit(host, message="yes", job_id="message-yes")
    assert second["proposal"]["status"] == "full_study"
    assert second["proposal"]["expand"]["reading"]["question"] == "yes"
    assert service._quick_store().read(record.session_id, record.project_id, first["quick_answer_id"])["answer"] is None
    with pytest.raises(PiCopilotError, match="unregistered"):
        execute_tool("easyicu_approve_quick_answer", {"card_sha256": first["proposal"]["card"]["sha256"]},
                     ToolExecutionContext(session=record, user_message="yes", allowed_actions=["run", "resume"]))


def test_a_credentialed_source_needs_no_declaration_for_a_quick_answer(host, monkeypatch):
    service, record, registry = host
    source = registry["sources"][0]
    source["database"] = "miiv"
    study = study_contexts.upsert_context({"id": record.binding.study_context_id,
                                         "data_source": {"path": source["path"], "database": "miiv"}})
    record.binding = service._binding_for_context(study)
    service._save_record(record)
    service.gateway.environ["EASYICU_PI_DATA_HANDLING"] = "unspecified"
    # No declaration is asked (owner decision 2026-10-27); the quick path still
    # needs a verified model connection before it reads the source.
    with pytest.raises(PiCopilotError) as failure:
        submit(host)
    assert failure.value.code == "pi_provider_research_configuration_unverified"
    assert not list(service._quick_store().root.glob("*/qa_*"))


@pytest.mark.parametrize("change", ["revision", "branch", "source"])
def test_card_is_bound_to_current_study_message_and_registration(host, change):
    service, record, registry = host
    attachment = submit(host)
    if change == "revision":
        study_contexts.upsert_context({"id": record.binding.study_context_id, "title": "Changed"})
    elif change == "branch":
        service.replay_store.supersede_from_turn_index(session_id=record.session_id, project_id=record.project_id, turn_index=0)
    else:
        registry["sources"][0]["id"] = "src_other"
    with pytest.raises(PiCopilotError):
        approve(host, attachment)


@pytest.mark.parametrize("host", ["hirid"], indirect=True)
def test_source_substitution_requires_acceptance_on_the_same_click(host, monkeypatch):
    from easyicu.research_agent.providers.mocks import MockLLMClient

    monkeypatch.setattr(host[0], "_quick_provider", lambda *_args: MockLLMClient())
    attachment = submit(host, reading("What proportion died in hospital?", measure={
        "concept": "hospital_death", "statistic": "proportion", "level": 1,
    }))
    assert attachment["proposal"]["status"] == "approval_required", attachment["proposal"]["reason"]
    assert attachment["proposal"]["card"]["capability"]["status"] == "substitution"
    with pytest.raises(PiCopilotError, match="quick_substitution_approval_required"):
        approve(host, attachment)
    answer = approve(host, attachment, accept_substitutions=True)["answer"]
    assert answer["status"] == "ok", answer["reason"]
    assert answer["expand"]["effective_reading"]["measure"]["concept"] == "icu_death"
    with pytest.raises(PiCopilotError, match="quick_approval_mismatch"):
        approve(host, attachment)


@pytest.mark.parametrize("tamper", ["answer", "evidence"])
def test_reload_rechecks_evidence_instead_of_trusting_stored_numbers(host, tamper):
    service, record, _ = host
    attachment = submit(host)
    approve(host, attachment)
    store = service._quick_store()
    row = store.read(record.session_id, record.project_id, attachment["quick_answer_id"])
    if tamper == "answer":
        row["answer"]["rows"][0]["value"] = 999
        store.write(row)
    else:
        directory = store.path(record.session_id, attachment["quick_answer_id"]).parent / "execution"
        evidence = EvidenceStore(directory)
        evidence_row = evidence.get(row["answer"]["rows"][0]["evidence"]["evidence_id"])
        (directory / evidence_row.relative_path).write_text("{}")
    projected = service._quick_attachments(record, [{"job_id": "message-quick"}])[0]["answer"]
    assert projected["reason"] == "quick_evidence_binding_failed"
    assert projected["rows"] == [] and projected["sentences"] == []


def test_changed_prepared_bytes_are_not_rebound_at_approval(host):
    service, record, _ = host
    attachment = submit(host)
    row = service._quick_store().read(record.session_id, record.project_id, attachment["quick_answer_id"])
    from pathlib import Path

    path = Path(row["prepared"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    answer = approve(host, attachment)["answer"]
    assert answer["reason"] == "quick_source_changed" and not answer["rows"]


def test_client_cannot_write_or_reuse_a_handoff_for_a_different_question(host):
    service, record, _ = host
    attachment = submit(host, reading("How many are still alive?"))
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.upsert_context({"id": record.binding.study_context_id,
                                       "quick_answer_handoff": attachment["proposal"]["expand"]})
    service.expand_quick_answer(record.session_id, project_id=record.project_id, quick_answer_id=attachment["quick_answer_id"])
    changed = study_contexts.upsert_context({"id": record.binding.study_context_id, "question": "Describe another synthetic measurement."})
    assert changed["quick_answer_handoff"] is None


def test_a_changed_handoff_reading_invalidates_scientific_configuration(host):
    service, record, _ = host
    attachment = submit(host, reading("How many are still alive?"))
    service.expand_quick_answer(record.session_id, project_id=record.project_id, quick_answer_id=attachment["quick_answer_id"])
    study = study_contexts.get_context(record.binding.study_context_id)
    before = study_contexts.scientific_configuration_sha256(study)
    study["quick_answer_handoff"]["reading"]["measure"]["statistic"] = "mean"
    assert study_contexts.scientific_configuration_sha256(study) != before


def test_click_routes_accept_only_typed_host_coordinates(host, monkeypatch):
    from pydantic import ValidationError
    from easyicu.webserver.routes import pi_copilot as routes

    service, record, _ = host
    attachment = submit(host)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    request = {"project_id": record.project_id, "card_sha256": attachment["proposal"]["card"]["sha256"]}
    for extra in ({"conversation_text": "yes"}, {"accept_substitutions": "yes"}, {"source": "invented"}):
        with pytest.raises(ValidationError):
            routes.PiQuickApprovalRequest(**request, **extra)
    result = routes.post_pi_quick_answer_approval(record.session_id, attachment["quick_answer_id"], routes.PiQuickApprovalRequest(**request))
    assert result["quick_answer"]["answer"]["status"] == "ok"
    result = routes.post_pi_quick_answer_expand(record.session_id, attachment["quick_answer_id"], routes.PiQuickExpandRequest(project_id=record.project_id))
    assert study_contexts.get_context(result["study_context_id"])["quick_answer_handoff"] == result["expand"]


def test_tool_rejects_host_fields_even_inside_the_reading(host):
    _, record, _ = host
    with pytest.raises(PiCopilotError, match="quick_reading_invalid"):
        execute_tool("easyicu_quick_answer", {"reading": {**reading(), "card_sha256": "a" * 64}}, ToolExecutionContext(session=record))


def test_preparation_budget_stops_before_allocating_any_rows(host, monkeypatch):
    from easyicu.webserver.quick_answer import preparation

    monkeypatch.setattr(preparation, "MAX_PREPARATION_BYTES", 1)
    monkeypatch.setattr(preparation, "acquire_universe_for_question", lambda **_kw: pytest.fail("allocated rows"))
    attachment = submit(host)
    assert attachment["proposal"]["reason"] == "quick_prepared_source_too_large"
    assert attachment["proposal"]["card"] is None


COMPARE = "What is the median first-day peak lactate, and does it differ between men and women?"
FIRST_STAYS = "In each patient's first ICU stay, what is the median first-day peak lactate, and does it differ by sex?"


def verified_identity(monkeypatch, tmp_path, patients, *, first_stays=None):
    """Hand the quick path a synthetic grouping (and first-stay coordinate) as the source owner would."""
    import hashlib

    from easyicu.research_agent.acquisition.first_icu_stay import FirstIcuStayBinding
    from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
    from easyicu.webserver import source_identity_authority

    def written(name, frame):
        path = tmp_path / name
        frame.to_parquet(path, index=False)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    path, digest = written("grouping.parquet", pd.DataFrame({"stay_id": range(60), "patient_key": patients}))
    grouping = PatientGroupingBinding(
        mapping_path=path, mapping_sha256=digest, mapping_stay_column="stay_id", mapping_patient_column="patient_key",
        authority_coordinates={"schema_version": "easyicu.patient_grouping_runtime_authority/1",
                               "authority_ref": "synthetic/patient_grouping", "mapping_sha256": digest,
                               "grouping_derivation": "prefix_before_:s", "provider_visible_values": False},
    )
    calls = []

    def resolve(binding):
        def resolver(**kwargs):
            calls.append(kwargs)
            return binding
        return resolver

    monkeypatch.setattr(source_identity_authority, "resolve_study_patient_grouping", resolve(grouping))
    first = None
    if first_stays is not None:
        path, digest = written("first.parquet", pd.DataFrame({"stay_id": range(60), "first_icu_stay": first_stays}))
        first = FirstIcuStayBinding(coordinate_path=path, coordinate_sha256=digest,
                                    authority_coordinates={"authority_ref": "synthetic/first_icu_stay"})
    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay", resolve(first))
    return calls


def test_a_comparison_reads_patients_from_the_sources_verified_grouping(host, monkeypatch, tmp_path):
    service, record, registry = host
    calls = verified_identity(monkeypatch, tmp_path, range(60))
    attachment = submit(host, reading(COMPARE, intent="compare", grouping=["sex"]))
    assert attachment["proposal"]["status"] == "approval_required", attachment["proposal"]["reason"]
    assert calls == [{"export_path": registry["sources"][0]["path"], "database": "eicu_demo"}]
    prepared = service._quick_store().read(record.session_id, record.project_id, attachment["quick_answer_id"])["prepared"]
    assert (prepared["unit_column"], prepared["patient_column"]) == ("patient_stay_id", "patient_stay_id")
    answer = approve(host, attachment)["answer"]
    assert answer["status"] == "ok", answer["reason"]
    assert [row["group"] for row in answer["rows"] if row["statistic"].startswith("p_value:")] == ["Comparison"]


def test_repeated_patients_stop_the_comparison_and_their_first_stays_answer_it(host, monkeypatch, tmp_path):
    verified_identity(monkeypatch, tmp_path, [stay // 2 for stay in range(60)],
                      first_stays=[stay % 2 == 0 for stay in range(60)])
    attachment = submit(host, reading(COMPARE, intent="compare", grouping=["sex"]))
    assert attachment["proposal"]["status"] == "approval_required", attachment["proposal"]["reason"]
    answer = approve(host, attachment)["answer"]
    assert answer["reason"] == "quick_repeated_units_require_full_study" and not answer["rows"]
    first = submit(host, reading(FIRST_STAYS, intent="compare", grouping=["sex"], population={"criteria": [{
        "id": "c1", "kind": "first_icu_stay", "source": "question", "role": "include",
        "quote": "each patient's first ICU stay",
    }]}), job_id="message-first")
    assert first["proposal"]["status"] == "approval_required", first["proposal"]["reason"]
    answer = approve(host, first)["answer"]
    assert answer["status"] == "ok", answer["reason"]
    assert {row["group"]: row["value"] for row in answer["rows"] if row["statistic"] == "cohort_n"} == {
        "Overall": 30, "Male": 15, "Female": 15,
    }


def test_without_a_verified_grouping_a_comparison_keeps_its_refusal(host):
    attachment = submit(host, reading(COMPARE, intent="compare", grouping=["sex"]))
    assert attachment["proposal"]["reason"] == "quick_independent_units_unverified"
    assert attachment["proposal"]["card"] is None


def test_cross_session_click_does_not_find_another_sessions_card(host):
    service, record, _ = host
    attachment = submit(host)
    other = record.model_copy(update={"session_id": "pi-other"})
    service._save_record(other)
    with pytest.raises(PiCopilotError, match="quick_answer_not_found"):
        service.approve_quick_answer(other.session_id, project_id=record.project_id,
                                     quick_answer_id=attachment["quick_answer_id"],
                                     card_sha256=attachment["proposal"]["card"]["sha256"])


def test_source_consent_is_rechecked_on_the_click(host):
    service, record, _ = host
    attachment = submit(host)
    record.data_source_authorization.status = "pending"
    service._save_record(record)
    with pytest.raises(PiCopilotError, match="pi_session_data_source_confirmation_required"):
        approve(host, attachment)


def test_quick_tool_finishes_without_another_model_answer():
    from pathlib import Path
    from tests.support.node import run_node

    app = Path(__file__).resolve().parents[3] / "src/easyicu/webserver/pi_copilot/node_app"
    if not (app / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    program = f"""
      import assert from 'node:assert/strict';
      import {{hostPostToolFinalization}} from {json.dumps((app / 'src/post-tool-finalization.mjs').as_uri())};
      for (const status of ['approval_required', 'stopped', 'full_study']) {{
        const stream = hostPostToolFinalization({{api: 'test', provider: 'test', id: 'test'}}, {{messages: [{{
          role: 'toolResult', toolName: 'easyicu_quick_answer', isError: false,
          details: {{details: {{quick_answer: {{proposal: {{status}}}}}}}},
        }}]}}, 'zh');
        assert.ok(stream);
        for await (const event of stream) {{ void event; }}
        const answer = await stream.result();
        assert.equal(answer.stopReason, 'stop');
        assert.equal(answer.content[0].text, '');
      }}
    """
    run_node("node", program, module=True, cwd=app, timeout=30, check=True)
