"""Exact public-demo recordings travel through the authorized provider boundary."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE

from easyicu.research_agent.providers.client_trust import (
    authorized_complete,
    provider_authorization_manifest,
    require_run_clearance,
)
from easyicu.research_agent.providers.protocol import LLMMessage, StructuredOutputRequest


@contextmanager
def public_scope(tmp_path, monkeypatch, **overrides):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.webserver import demo_source_storage

    # Source-owner verification is exercised separately; no patient data is read.
    monkeypatch.setattr(demo_source_storage, "egress_source_database", lambda path, database: database)
    options = {"export_path": tmp_path, "database": "eicu_demo",
               "profile_ref": PUBLIC_DEMO_ACCEPTANCE.ref}
    options.update(overrides)
    with public_demo_replay_scope(**options) as scope:
        yield scope


def test_replay_allowlist_tracks_the_current_profile_and_rejects_its_frozen_predecessor(tmp_path, monkeypatch):
    from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE_2026_10_25
    from easyicu.research_agent.providers.client_trust import ACCEPTANCE_PROFILE_REF
    from easyicu.research_agent.providers.protocol import ReplayError

    assert ACCEPTANCE_PROFILE_REF == PUBLIC_DEMO_ACCEPTANCE.ref
    with pytest.raises(ReplayError, match="model_replay_profile_forbidden"):
        with public_scope(tmp_path, monkeypatch, profile_ref=PUBLIC_DEMO_ACCEPTANCE_2026_10_25.ref):
            pytest.fail("a frozen profile must not authorize current replay")


def recorded(scope, **options):
    from easyicu.research_agent.providers.protocol import (
        ModelIdentity, RecordedResponse, ReplayCassette, request_digest,
    )
    model = ModelIdentity(provider="synthetic", model="synthetic-model")
    messages = [LLMMessage("user", "Describe an invented blue square.")]
    key = request_digest(model, messages, **options)
    cassette = ReplayCassette.create(source=scope.source, model=model,
                                    entries=(RecordedResponse(request_sha256=key, response="blue"),))
    return cassette, messages, key


def test_exact_hit_and_receipt(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, _ = recorded(scope)
        client = ReplayLLMClient(cassette)
        assert authorized_complete(client, messages) == "blue"
        receipt = provider_authorization_manifest(client)["clients"][0]
        assert receipt["provider_kind"] == "replay"
        assert receipt["cassette_sha256"] == cassette.cassette_sha256
        assert receipt["reportable"] is False
        assert receipt["publishable"] is False


def test_family_cassette_matches_the_whole_request_without_a_response_digest(tmp_path, monkeypatch):
    from easyicu.research_agent.agents.family_spec_planner import (
        family_spec_structured_output_request, family_spec_user_prompt, parse_family_plan_spec,
    )
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import (
        ModelIdentity, RecordedResponse, ReplayCassette, ReplayError, request_digest,
    )
    from tests.research_agent.planning.family_spec_fixtures import (
        _descriptive_context, _descriptive_payload, _request,
    )

    request = _request(_descriptive_context(), cohort_mode="all_input_rows")
    changed = type(request).model_validate({
        **request.model_dump(mode="json"),
        "allowed_literature_citation_keys": [*request.allowed_literature_citation_keys, "method_beta"],
    })
    assert request.request_sha256 != changed.request_sha256
    messages = [LLMMessage("user", family_spec_user_prompt(request, variable_descriptions={}))]
    other_messages = [LLMMessage("user", family_spec_user_prompt(changed, variable_descriptions={}))]
    schema = family_spec_structured_output_request(request)
    assert schema == family_spec_structured_output_request(changed)
    payload = _descriptive_payload(request, baseline_variables=[])
    payload.pop("request_sha256")
    model = ModelIdentity(provider="synthetic", model="synthetic-model")
    with public_scope(tmp_path, monkeypatch) as scope:
        cassette = ReplayCassette.create(
            source=scope.source, model=model,
            entries=(RecordedResponse(
                request_sha256=request_digest(model, messages, structured_output=schema),
                response=json.dumps(payload),
            ),),
        )
        client = ReplayLLMClient(cassette)
        raw = authorized_complete(client, messages, structured_output=schema)
        assert parse_family_plan_spec(raw, request).request_sha256 == request.request_sha256
        with pytest.raises(ReplayError, match="model_replay_request_missing"):
            authorized_complete(client, other_messages, structured_output=schema)


@pytest.mark.parametrize("change", ["message", "temperature", "max_tokens", "schema", "model"])
def test_miss_names_the_exact_digest_and_never_falls_back(tmp_path, monkeypatch, change):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError, request_digest

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, original = recorded(scope)
        options = {}
        if change == "message":
            messages = [LLMMessage("user", "Describe an invented red square.")]
        elif change == "schema":
            options["structured_output"] = StructuredOutputRequest.from_schema(name="square", schema={"type": "object"})
        elif change == "model":
            cassette = type(cassette).create(source=scope.source,
                model=cassette.model.model_copy(update={"model": "another-model"}), entries=cassette.entries)
        else:
            options[change] = 17 if change == "max_tokens" else 0.5
        expected = request_digest(cassette.model, messages, **options)
        assert expected != original
        with pytest.raises(ReplayError) as caught:
            authorized_complete(ReplayLLMClient(cassette), messages, **options)
        assert caught.value.reason_code == "model_replay_request_missing"
        assert caught.value.request_sha256 == expected
        assert expected in str(caught.value)


@pytest.mark.parametrize("options,reason", [
    ({"database": "eicu"}, "model_replay_source_forbidden"),
    ({"database": "unregistered"}, "model_replay_source_forbidden"),
    ({"profile_ref": "npj_dm/2026-07-19"}, "model_replay_profile_forbidden"),
])
def test_scope_refuses_other_sources_and_profiles(tmp_path, monkeypatch, options, reason):
    from easyicu.research_agent.providers.protocol import ReplayError

    with pytest.raises(ReplayError, match=reason), public_scope(tmp_path, monkeypatch, **options):
        pytest.fail("forbidden scope admitted")


def test_hosted_and_out_of_scope_delivery_are_refused(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, _ = recorded(scope)
        client = ReplayLLMClient(cassette)
        monkeypatch.setenv("EASYICU_HOSTED", "1")
        with pytest.raises(ReplayError, match="model_replay_hosted_forbidden"):
            authorized_complete(client, messages)
        monkeypatch.delenv("EASYICU_HOSTED")
        with pytest.raises(ReplayError, match="model_replay_source_forbidden"):
            require_run_clearance("eicu", None, client)
    with pytest.raises(ReplayError, match="model_replay_scope_required"):
        authorized_complete(client, messages)


@pytest.mark.parametrize("setting", ["EASYICU_HOSTED", "EASYICU_DEPLOYMENT"])
def test_hosted_scope_is_refused_before_source_access(tmp_path, monkeypatch, setting):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.research_agent.providers.protocol import ReplayError

    monkeypatch.setenv(setting, "1" if setting == "EASYICU_HOSTED" else "hosted")
    with pytest.raises(ReplayError, match="model_replay_hosted_forbidden"):
        with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                      profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
            pytest.fail("hosted scope admitted")


def test_source_owner_must_verify_demo_manifest(tmp_path):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError

    with pytest.raises(ProviderConfigurationError, match="public_demo_manifest_invalid"):
        with public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                     profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
            pytest.fail("a label is not source authority")


def test_cassette_digest_detects_tampering(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.protocol import ReplayCassette, ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        path = tmp_path / "cassette.json"
        cassette.write(path)
        assert ReplayCassette.read(path) == cassette
        raw = json.loads(path.read_text())
        raw["entries"][0]["response"] = "altered"
        path.write_text(json.dumps(raw))
        with pytest.raises(ReplayError, match="model_replay_cassette_digest_mismatch"):
            ReplayCassette.read(path)


def test_recording_excludes_prompts_and_scrubs_response_secrets(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import RecordingLLMClient
    from easyicu.research_agent.providers.protocol import ModelIdentity
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    secret = "sk-" + "x" * 32
    with public_scope(tmp_path, monkeypatch):
        inner = ScriptedMockLLMClient([json.dumps({"answer": "blue", "api_key": secret,
            "headers": {"custom": "secret-header"}, "base_url": "https://private.invalid",
            "account_id": "account-123"})])
        recorder = RecordingLLMClient(inner, model=ModelIdentity(provider="synthetic", model="synthetic-model"))
        answer = authorized_complete(recorder, [LLMMessage("user", "private prompt " + secret)])
        cassette = recorder.cassette()
        text = cassette.model_dump_json()
        for forbidden in (secret, "secret-header", "private.invalid", "account-123", "private prompt"):
            assert forbidden not in text
        assert cassette.entries[0].response == answer
        assert json.loads(answer)["answer"] == "blue"


def test_replay_cannot_be_wrapped_with_a_fallback_provider(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError
    from easyicu.research_agent.providers.llm import FallbackLLMClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, _ = recorded(scope)
        fallback = ScriptedMockLLMClient(["must not answer"])
        client = FallbackLLMClient(ReplayLLMClient(cassette), fallback)
        with pytest.raises(ReplayError, match="model_replay_mixed_provider_forbidden"):
            authorized_complete(client, messages)
        assert fallback.calls == []


def test_schema_order_is_canonical_and_seed_is_bound(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.protocol import request_digest

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, key = recorded(scope)
        first = StructuredOutputRequest.from_schema(name="shape", schema={"type": "object", "properties": {"side": {"type": "number"}}})
        second = StructuredOutputRequest.from_schema(name="shape", schema={"properties": {"side": {"type": "number"}}, "type": "object"})
        assert request_digest(cassette.model, messages, structured_output=first) == request_digest(cassette.model, messages, structured_output=second)
        assert request_digest(cassette.model, messages, seed=7) != key
        assert request_digest(cassette.model, messages, seed=7) != request_digest(cassette.model, messages, seed=8)
        assert request_digest(cassette.model, messages, top_p=0.5) != key


@pytest.mark.parametrize("field,value", [("schema_version", "unknown/2"), ("api_key", "secret"), ("cassette_sha256", "invalid")])
def test_cassette_format_is_closed(tmp_path, monkeypatch, field, value):
    from easyicu.research_agent.providers.protocol import ReplayCassette, ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        payload = cassette.model_dump(mode="json")
        payload[field] = value
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ReplayError, match="model_replay_cassette_invalid"):
            ReplayCassette.read(path)


def test_duplicate_keys_and_conflicting_live_answers_are_refused(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import RecordingLLMClient
    from easyicu.research_agent.providers.protocol import ReplayCassette, ReplayError
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, _ = recorded(scope)
        with pytest.raises(ReplayError, match="model_replay_duplicate_request"):
            ReplayCassette.create(source=cassette.source, model=cassette.model, entries=cassette.entries * 2)
        recorder = RecordingLLMClient(ScriptedMockLLMClient(["blue", "red"]), model=cassette.model)
        assert authorized_complete(recorder, messages) == "blue"
        with pytest.raises(ReplayError, match="model_replay_request_conflict"):
            authorized_complete(recorder, messages)


def test_source_bound_cassette_cannot_be_relabelled(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        changed = type(cassette).create(source=scope.source.model_copy(update={"version": "another"}), model=cassette.model)
        with pytest.raises(ReplayError, match="model_replay_source_mismatch"):
            ReplayLLMClient(changed)


def test_only_the_exact_adapter_constructor_can_register_trust(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.client_trust import _register_replay_client, ProviderConfigurationError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        class Impostor(ReplayLLMClient):
            pass
        with pytest.raises(ProviderConfigurationError):
            Impostor(cassette)
        with pytest.raises(ProviderConfigurationError):
            _register_replay_client(ReplayLLMClient(cassette))


@pytest.mark.parametrize("field,value,reason", [
    ("submission_profile_name", "production", "model_replay_profile_forbidden"),
    ("development_diagnostic", False, "model_replay_run_policy_forbidden"),
    ("require_human_plan_review", False, "model_replay_run_policy_forbidden"),
    ("generate_manuscript", True, "model_replay_run_policy_forbidden"),
])
def test_run_and_resume_cannot_move_replay_into_study_authority(tmp_path, monkeypatch, field, value, reason):
    from types import SimpleNamespace
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError
    from easyicu.research_agent.orchestration.instance_lifecycle import pipeline_instance_lifecycle

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        config = dict(submission_profile_name="public_demo_acceptance_dev", submission_profile_version=PUBLIC_DEMO_ACCEPTANCE.version,
            development_diagnostic=True, require_human_plan_review=True, generate_manuscript=False)
        config[field] = value
        pipeline = SimpleNamespace(_llm=ReplayLLMClient(cassette), _config=SimpleNamespace(**config))
        for operation in ("run", "resume"):
            @pipeline_instance_lifecycle(operation)
            def forbidden(self):
                pytest.fail("dependent operation ran")
            with pytest.raises(ReplayError, match=reason):
                forbidden(pipeline)


def test_replay_never_has_paper_execution_authority_even_in_docker(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.execution_identity import ExecutionIdentity
    from easyicu.research_agent.providers.clients import ReplayLLMClient

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        identity = ExecutionIdentity.create(submission_profile_name="ordinary_study", submission_profile_version="1",
            runner="docker", runner_image_digest="sha256:" + "b" * 64, network_policy="none",
            llm_seed=0, input_authority_sha256="c" * 64, provider_client=ReplayLLMClient(cassette),
            code_version={"git_sha": "d" * 40, "git_dirty": False})
        assert identity.paper_eligible is False


def test_credentialed_mimic_is_refused_but_pinned_public_release_is_admitted(tmp_path, monkeypatch):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope
    from easyicu.research_agent.providers.protocol import ReplayError
    from easyicu.webserver import demo_source_storage as storage
    from easyicu.webserver.demo_source_contracts import SOURCES

    options = dict(export_path=tmp_path, database="miiv", profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref)
    monkeypatch.setattr(storage, "_verified_public_demo", lambda path: None)
    with pytest.raises(ReplayError, match="model_replay_source_forbidden"):
        with public_demo_replay_scope(**options):
            pytest.fail("credentialed source admitted")
    source = next(source for source in SOURCES if source.database == "miiv")
    monkeypatch.setattr(storage, "_verified_public_demo", lambda path: source)
    with public_demo_replay_scope(**options) as scope:
        assert scope.source.database == "miiv"


def test_recorder_requires_an_authorized_inner_provider(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import RecordingLLMClient
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        with pytest.raises(ProviderConfigurationError):
            RecordingLLMClient(object(), model=cassette.model)


def test_only_the_public_source_owner_may_activate_a_scope(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.client_trust import _bind_public_demo_replay_scope, _reset_public_demo_replay_scope
    from easyicu.research_agent.providers.protocol import ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        token = None
        try:
            with pytest.raises(ReplayError, match="model_replay_source_authority_required"):
                token = _bind_public_demo_replay_scope(scope)
        finally:
            if token is not None:
                _reset_public_demo_replay_scope(token)


def test_playback_preserves_finish_reason_without_claiming_live_transport(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient, current_provider_call_receipt
    from easyicu.research_agent.providers.protocol import RecordedResponse

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, key = recorded(scope)
        cassette = type(cassette).create(source=cassette.source, model=cassette.model,
            entries=(RecordedResponse(request_sha256=key, response="incomplete", finish_reason="length"),))
        assert authorized_complete(ReplayLLMClient(cassette), messages) == "incomplete"
        assert current_provider_call_receipt().finish_reason == "length"
        assert current_provider_call_receipt().transport_attempts == 0


@pytest.mark.parametrize("response", [
    '```json\n{"password": "opaque-credential", "answer": "blue"}\n```',
    'base_url="https://private.invalid"; account_id="private-account"',
    "api_key='opaque-credential'",
    'password="opaque-credential"',
    '{"accessToken":"opaque-credential", "privateKey":"opaque-credential"}',
    'project_id="private-account"',
])
def test_scrubbing_also_covers_quoted_fields_outside_plain_json(response):
    from easyicu.research_agent.providers.protocol import scrub_response

    cleaned = scrub_response(response)
    for secret in ("opaque-credential", "private.invalid", "private-account"):
        assert secret not in cleaned
    assert scrub_response(cleaned) == cleaned


def test_scrubbing_preserves_scientific_keys_while_removing_credential_keys():
    from easyicu.research_agent.providers.protocol import scrub_response

    payload = {"key": "term", "input_key": "column", "citation_key": "reference",
               "starter_template_key": "template", "private_key": "private-value", "api_key": "api-value"}
    cleaned = json.loads(scrub_response(json.dumps(payload)))
    assert {key: cleaned[key] for key in ("key", "input_key", "citation_key", "starter_template_key")} == {
        key: payload[key] for key in ("key", "input_key", "citation_key", "starter_template_key")}
    assert "private-value" not in json.dumps(cleaned)
    assert "api-value" not in json.dumps(cleaned)


def test_message_role_and_model_wire_settings_are_request_authority(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.protocol import request_digest

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, key = recorded(scope)
        assert request_digest(cassette.model, [LLMMessage("system", messages[0].content)]) != key
        for changes in ({"provider": "another"}, {"supports_strict_json_schema": True},
                        {"extra_body_json": '{"enable_thinking":true}'}):
            assert request_digest(cassette.model.model_copy(update=changes), messages) != key


def test_valid_digest_does_not_admit_a_secret_bearing_cassette(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.protocol import RecordedResponse, ReplayCassette, ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, key = recorded(scope)
        with pytest.raises(ReplayError, match="model_replay_cassette_contains_secrets"):
            ReplayCassette.create(source=scope.source, model=cassette.model,
                entries=(RecordedResponse(request_sha256=key, response='{"password":"private-value"}'),))


def test_duplicate_approvals_and_incomplete_reuse_grants_are_refused(tmp_path, monkeypatch):
    from pydantic import ValidationError
    from easyicu.research_agent.providers.protocol import RecordedApproval, ReplayCassette, ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        grant = RecordedApproval(review_id="review-" + "a" * 16, authority_sha256="a" * 64, decision="approved")
        with pytest.raises(ReplayError, match="model_replay_duplicate_approval"):
            ReplayCassette.create(source=scope.source, model=cassette.model, approvals=(grant, grant))
        with pytest.raises(ValidationError):
            RecordedApproval.model_validate({**grant.model_dump(), "acceptance_reuse_authorized": True})


def test_replay_revalidates_a_model_copy_and_the_active_export_scope(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import ReplayLLMClient
    from easyicu.research_agent.providers.protocol import ReplayError

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, messages, _ = recorded(scope)
        with pytest.raises(ReplayError, match="model_replay_cassette_digest_mismatch"):
            ReplayLLMClient(cassette.model_copy(update={"cassette_sha256": "f" * 64}))
        client = ReplayLLMClient(cassette)
        with public_scope(tmp_path, monkeypatch, export_path=tmp_path / "other"):
            with pytest.raises(ReplayError, match="model_replay_source_mismatch"):
                authorized_complete(client, messages)


def test_recorder_requires_the_exact_transport_model_and_captures_finish_reason(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from easyicu.research_agent.providers.clients import OpenAIClient, RecordingLLMClient, recording_model_identity
    from easyicu.research_agent.providers.protocol import ReplayError

    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="invented", refusal=None),
                                                        finish_reason="length")], usage=None)
    transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: transport))
    with public_scope(tmp_path, monkeypatch):
        inner = OpenAIClient(model="synthetic-transport", api_key="test-placeholder", base_url="http://127.0.0.1:8787/v1",
                             request_timeout=1, max_retries=0)
        model = recording_model_identity(inner)
        with pytest.raises(ReplayError, match="model_replay_recording_model_mismatch"):
            RecordingLLMClient(inner, model=model.model_copy(update={"model": "wrong-model"}))
        assert calls == []
        recorder = RecordingLLMClient(inner, model=model)
        assert authorized_complete(recorder, [LLMMessage("user", "An invented shape.")]) == "invented"
        assert recorder.cassette().entries[0].finish_reason == "length"
        assert len(calls) == 1


def test_cassette_size_is_bounded_and_writing_never_overwrites(tmp_path, monkeypatch):
    from easyicu.research_agent.providers import protocol

    with public_scope(tmp_path, monkeypatch) as scope:
        cassette, _, _ = recorded(scope)
        path = tmp_path / "cassette.json"
        cassette.write(path)
        with pytest.raises(FileExistsError):
            cassette.write(path)
        monkeypatch.setattr(protocol, "MAX_CASSETTE_BYTES", 10)
        with pytest.raises(protocol.ReplayError, match="model_replay_cassette_too_large"):
            protocol.ReplayCassette.read(path)


@pytest.mark.parametrize("change", [
    {"model": "https://private.invalid/model"},
    {"model": "api_key=private-value"},
    {"extra_body_json": '{"unrecognized_wire_option": true}'},
    {"extra_body_json": '{"reasoning": {"api_key": "private-value"}}'},
])
def test_recorded_model_coordinates_exclude_transport_secrets_and_unknown_settings(change):
    from pydantic import ValidationError
    from easyicu.research_agent.providers.protocol import ModelIdentity

    with pytest.raises(ValidationError):
        ModelIdentity(**{"provider": "synthetic", "model": "synthetic-model", **change})
