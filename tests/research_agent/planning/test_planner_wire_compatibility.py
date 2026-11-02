"""The v2 planner wire is smaller without changing its accepted plans."""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from easyicu.research_agent.agents.plan_payload import planner_structured_output_request
from easyicu.research_agent.planning.method_literature import METHOD_CARDS
from easyicu.research_agent.providers.protocol import StructuredOutputRequest
from easyicu.research_agent.providers.strict_json_schema import StrictJsonSchemaError, compact_planner_wire_schema

KEYS = ["vincent_sofa_1996", "singer_sepsis3_2016", "strobe_2007", "record_2015", "suissa_immortal_time_2008", "anderson_landmark_1983", "durrleman_splines_1989", "sterne_missing_data_2009", "ricu_2023", "johnson_mimiciv_2023"]


def _step(schema):
    value = schema["properties"]["steps"]["items"]
    while "$ref" in value:
        value = schema["$defs"][value["$ref"].removeprefix("#/$defs/")]
    return value


@pytest.mark.parametrize("keys", [None, [], ["strobe_2007"], KEYS, list(reversed(KEYS))])
def test_legacy_schema_is_stable_and_wire_is_versioned_and_deterministic(keys):
    request = planner_structured_output_request(keys)
    original = json.loads(request.schema_json)
    wire = json.loads(request.transport_schema_json)
    assert request.wire_schema_version == "easyicu.planner-wire/2"
    assert wire == compact_planner_wire_schema(original)
    assert request.to_openai_response_format()["json_schema"]["schema"] == wire
    assert request.payload_bytes == len(request.canonical_payload_json.encode())
    assert set(_step(original)["properties"]) == set(_step(wire)["properties"])
    assert planner_structured_output_request(keys) == request
    legacy = StructuredOutputRequest.from_schema(name=request.name, schema=original)
    assert legacy.schema_json == request.schema_json
    assert legacy.authority_sha256 != request.authority_sha256
    if keys is None:
        assert request.payload_bytes < 30_000
    if keys == KEYS:
        assert request.payload_bytes < 30_100


@pytest.mark.parametrize("mutation", ["legacy_with_wire", "wrong_wire", "missing_wire", "unknown_version"])
def test_public_construction_rejects_inconsistent_transport_before_dispatch(mutation):
    from dataclasses import replace

    request = planner_structured_output_request(KEYS)
    changes = {
        "legacy_with_wire": {"wire_schema_version": "legacy/1"},
        "wrong_wire": {"wire_schema_json": request.schema_json},
        "missing_wire": {"wire_schema_json": None},
        "unknown_version": {"wire_schema_version": "unknown/3"},
    }[mutation]
    dispatched = []
    with pytest.raises(ValueError):
        candidate = replace(request, **changes)
        dispatched.append((candidate.payload_bytes, candidate.transport_schema_json))
    assert not dispatched
    with pytest.raises(ValueError):
        StructuredOutputRequest(**{**request.__dict__, **changes})


def test_legacy_direct_constructor_retains_existing_semantics():
    schema = '{"type": "object", "properties": {}}'
    request = StructuredOutputRequest(name="legacy", schema_json=schema)
    assert request.schema_json == request.transport_schema_json == schema
    assert request.to_openai_response_format()["json_schema"]["schema"] == json.loads(schema)


@pytest.mark.parametrize("name", ["", "bad name", " padded ", "x" * 65])
def test_wire_direct_constructor_rejects_names_factory_would_not_preserve(name):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(planner_structured_output_request(), name=name)


@pytest.mark.parametrize("strict", ["yes", 1, None])
def test_wire_direct_constructor_requires_worker_compatible_strict_bool(strict):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(planner_structured_output_request(), strict=strict)


def test_wire_direct_constructor_rejects_nonobject_root_like_factory():
    schema = '{"$defs":{},"type":"string"}'
    with pytest.raises(ValueError):
        StructuredOutputRequest(name="synthetic", schema_json=schema,
                                wire_schema_json=schema, wire_schema_version="easyicu.planner-wire/2")


def test_all_source_and_design_element_pairs_match_wire_contract():
    request = planner_structured_output_request(KEYS)
    schemas = [json.loads(request.schema_json), json.loads(request.transport_schema_json)]
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
    values = [Draft202012Validator({**_step(schema)["properties"]["literature_design_bindings"]["items"], "$defs": schema["$defs"]}) for schema in schemas]
    roster = [Draft202012Validator({**_step(schema)["properties"]["literature_citation_keys"]["items"], "$defs": schema["$defs"]}) for schema in schemas]
    elements = {"adjustment", "dependence", "estimand", "exposure", "missing_data", "outcome", "population", "reporting", "robustness", "time_zero"}
    authority = {}
    for card in METHOD_CARDS:
        authority.setdefault(card.source_key, set()).update(card.design_elements)
    for key in [*KEYS, "unselected_source"]:
        assert [validator.is_valid(key) for validator in roster] == [key in KEYS] * 2
        for element in elements | {"unknown"}:
            candidate = {"citation_key": key, "design_elements": [element], "application": "valid application", "divergence": None}
            expected = key in KEYS and element in authority.get(key, elements)
            assert [validator.is_valid(candidate) for validator in values] == [expected] * 2


def test_literal_ref_objects_are_data_not_schema_pointers():
    literal = {"$ref": "https://example.invalid/not-a-schema", "properties": {"a": 1}}
    schema = {"type": "object", "additionalProperties": False, "required": ["a", "b", "c"], "properties": {
        "a": {"enum": [literal]}, "b": {"const": literal}, "c": {"type": "string", "default": literal}}, "$defs": {}}
    before = json.loads(json.dumps(schema))
    wire = compact_planner_wire_schema(schema)
    assert schema == before
    assert wire["properties"]["a"]["const"] == literal
    assert wire["properties"]["b"]["const"] == literal
    assert wire["properties"]["c"]["default"] == literal


@pytest.mark.parametrize("reference", ["#/$defs/a/properties/x", "#/$defs/a%2Fproperties", "https://example.invalid/s", "#/$defs/missing"])
def test_unsupported_or_missing_reference_fails_closed(reference):
    schema = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"$ref": reference}}, "$defs": {"a": {"type": "string"}}}
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema(schema)


def test_scoped_or_recursive_reference_fails_closed():
    scoped = {"$id": "https://example.invalid/s", "type": "object", "additionalProperties": False, "required": [], "properties": {}, "$defs": {}}
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema(scoped)
    cyclic = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"$ref": "#/$defs/a"}}, "$defs": {"a": {"$ref": "#/$defs/a"}}}
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema(cyclic)


def test_openai_transport_receives_wire_not_legacy(monkeypatch, ra):
    from types import SimpleNamespace
    from tests.support.provider_openai import _mock_transport_client
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient
    calls = []
    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'), finish_reason='stop')], usage=None)
    client = _mock_transport_client(monkeypatch, OpenAIClient, model='synthetic', completions=Completions(), supports_strict_json_schema=True)
    request = planner_structured_output_request(KEYS)
    client.complete([LLMMessage(role='user', content='synthetic')], structured_output=request)
    assert calls[0]['response_format'] == request.to_openai_response_format()
    assert calls[0]['response_format']['json_schema']['schema'] == json.loads(request.transport_schema_json)
    assert calls[0]['response_format']['json_schema']['schema'] != json.loads(request.schema_json)


def test_anthropic_transport_receives_wire_not_legacy(monkeypatch):
    from tests.support.provider_anthropic import _build_client, _install_fake_sdk, _response
    pytest.importorskip("anthropic")
    from easyicu.research_agent.providers.protocol import LLMMessage
    _, messages = _install_fake_sdk(monkeypatch, [_response()])
    client = _build_client(strict=True)
    request = planner_structured_output_request(KEYS)
    client.complete_with_usage([LLMMessage(role='user', content='synthetic')], structured_output=request)
    assert messages.calls[0]['output_config']['format']['schema'] == json.loads(request.transport_schema_json)
    assert messages.calls[0]['output_config']['format']['schema'] != json.loads(request.schema_json)


def test_codex_app_server_receives_wire_not_legacy(monkeypatch, tmp_path):
    from tests.support.provider_app_server import _authorized_client, _FakeRuntime
    from easyicu.research_agent.providers import codex_app_server
    from easyicu.research_agent.providers.protocol import LLMMessage
    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, 'CodexAppServerRuntime', _FakeRuntime)
    request = planner_structured_output_request(KEYS)
    _authorized_client(tmp_path).complete_with_usage([LLMMessage(role='user', content='synthetic')], structured_output=request)
    sent = _FakeRuntime.instances[0].calls[2][1]['outputSchema']
    assert sent == json.loads(request.transport_schema_json)
    assert sent != json.loads(request.schema_json)


def test_codex_cli_receives_wire_not_legacy(monkeypatch):
    import shutil
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from tests.support.provider_cli import _authorized_client
    from easyicu.research_agent.providers.protocol import LLMMessage
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/' + name)
    captured = {}
    def fake_run(argv, **kwargs):
        captured['schema'] = json.loads(Path(argv[argv.index('--output-schema') + 1]).read_text())
        return SimpleNamespace(returncode=0, stdout='{}', stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    request = planner_structured_output_request(KEYS)
    _authorized_client('codex').complete([LLMMessage(role='user', content='synthetic')], structured_output=request)
    assert captured['schema'] == json.loads(request.transport_schema_json)
    assert captured['schema'] != json.loads(request.schema_json)


def test_supervised_metadata_worker_sends_and_accounts_exact_wire(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_concurrency as slots, provider_adapter
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient

    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'id':'fake','object':'chat.completion','created':1,'model':'synthetic',
                'choices':[{'index':0,'message':{'role':'assistant','content':'{}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':3,'completion_tokens':1,'total_tokens':4}}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        monkeypatch.setattr(slots.state_paths, 'state_root', lambda: tmp_path / 'state')
        env = {'OPENAI_API_KEY':'synthetic-secret','OPENAI_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','OPENAI_MODEL':'synthetic','EASYICU_DISABLE_PROVIDER_ENV_FILE':'1','EASYICU_PROVIDER_DATA_HANDLING':'user_accepted_external'}
        provider = {'provider':'openai','external':True}
        client, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10, supports_strict_json_schema=True)
        ledger = ProviderHardStopLedger(path=tmp_path/'budget.json',task_ids=('task',),limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),batch_id='task',declaration_sha256='a'*64)
        task = ledger.start_task('task')
        supervised = MetadataProviderClient(inner=client,task=task,provider=provider,environment=env,public=public,timeout=10)
        request = planner_structured_output_request(KEYS)
        from dataclasses import replace
        with pytest.raises(ValueError):
            HardStopClient(supervised, role='planner', task=task).complete(
                [LLMMessage(role='user', content='synthetic metadata')], max_tokens=32,
                structured_output=replace(request, wire_schema_version='legacy/1'))
        assert not seen and not ledger.snapshot()['tasks'][0]['calls']
        result = HardStopClient(supervised,role='planner',task=task).complete([LLMMessage(role='user',content='synthetic metadata')],max_tokens=32,structured_output=request)
        assert json.loads(result) == {}
        assert seen[0]['response_format'] == request.to_openai_response_format()
        assert seen[0]['response_format']['json_schema']['schema'] != json.loads(request.schema_json)
        call = ledger.snapshot()['tasks'][0]['calls'][0]
        assert call['state'] == 'completed'
        assert call['prompt_payload_bytes'] == len('synthetic metadata'.encode()) + request.payload_bytes
    finally:
        server.shutdown(); thread.join(timeout=5); server.server_close()

@pytest.mark.parametrize('name', ['a~1b', 'a~0b', 'a/b', 'a%2Fb'])
def test_pointer_escapes_and_names_fail_closed(name):
    schema = {'type':'object','additionalProperties':False,'required':['x'],
              'properties':{'x':{'$ref':'#/$defs/'+name}},
              '$defs':{name:{'type':'string'},'a/b':{'type':'integer'}}}
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema(schema)


def test_explicit_legacy_dialect_and_unknown_keyword_fail_closed():
    base = {'type':'object','additionalProperties':False,'required':['x'],
            'properties':{'x':{'enum':['fixed']}},'$defs':{}}
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema({'$schema':'http://json-schema.org/draft-04/schema#', **base})
    with pytest.raises(StrictJsonSchemaError):
        compact_planner_wire_schema({**base,'unknownFutureKeyword':{'const':'unexpected'}})


@pytest.mark.parametrize('mutation', ['wire', 'version', 'legacy', 'legacy_with_wire'])
def test_worker_rejects_schema_or_version_tampering_before_provider(monkeypatch, mutation):
    import io
    from types import SimpleNamespace
    from pathlib import Path
    from easyicu.webserver import metadata_provider_worker as worker, provider_adapter

    request = planner_structured_output_request(KEYS)
    envelope = {'provider':{'provider':'synthetic'}, 'environment':{'EASYICU_DISABLE_PROVIDER_ENV_FILE':'1'},
        'timeout':10, 'provider_identity':{}, 'messages':[{'role':'user','content':'synthetic'}],
        'max_tokens':32,'temperature':0,'strict_json_schema_enabled':True,
        'structured_output': {'name':request.name,'schema_json':request.schema_json,
                              'wire_schema_json':request.transport_schema_json,
                              'wire_schema_version':request.wire_schema_version,'strict':True}}
    if mutation == 'wire':
        envelope['structured_output']['wire_schema_json'] = request.schema_json
    elif mutation == 'version':
        envelope['structured_output']['wire_schema_version'] = 'unknown/3'
    elif mutation == 'legacy_with_wire':
        envelope['structured_output']['wire_schema_version'] = 'legacy/1'
    else:
        envelope['structured_output']['schema_json'] += ' '
    sent = []
    fake = SimpleNamespace(complete_with_usage=lambda *args,**kwargs: sent.append(True))
    monkeypatch.setattr(provider_adapter, 'build_research_agent_provider_client', lambda *a,**k: (fake, {'transport_max_attempts':1}))
    monkeypatch.setattr(worker, 'verify_host_worker', lambda *a: 'synthetic')
    monkeypatch.setattr(worker, 'identity', lambda: 'synthetic')
    exits = []
    monkeypatch.setattr(worker, 'complete_host_worker_cleanup', lambda *a, **k: exits.append(k['exitcode']))
    class WorkerParked(Exception):
        pass
    monkeypatch.setattr(worker.time, 'sleep', lambda _seconds: (_ for _ in ()).throw(WorkerParked()))
    input_stream = SimpleNamespace(buffer=io.BytesIO(json.dumps(envelope).encode()))
    output_stream = SimpleNamespace(buffer=io.BytesIO())
    with monkeypatch.context() as patch:
        patch.setattr(worker.sys, 'stdin', input_stream)
        patch.setattr(worker.sys, 'stdout', output_stream)
        with pytest.raises(WorkerParked):
            worker._worker(Path('/private/tmp'), 'nonce', 'synthetic')
    result = json.loads(output_stream.buffer.getvalue())
    assert exits == [1] and result['ok'] is False and not sent
