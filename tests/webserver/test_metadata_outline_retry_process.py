"""Actual SIGKILL during metadata outline retry; loopback Provider only."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


_CHILD = r'''
import sys
from pathlib import Path
import hashlib, json
import pandas as pd
import easyicu.research_agent.providers.llm
from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.webserver import agent_review_recovery, metadata_provider_client, provider_adapter, study_contexts
from easyicu.research_agent.orchestration.config import PipelineConfig
from tests.research_agent.planning.progressive_planner_fixtures import _context as outline_context
from tests.research_agent.planning.family_spec_fixtures import _context as family_context, ALLOWED_CITATIONS, DIRECT_COMPARATORS
root = Path(sys.argv[1]); port = int(sys.argv[2]); stage = sys.argv[3]; database = sys.argv[4]; root.mkdir(exist_ok=True)
source = root / "source"; source.mkdir()
study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(source), "database": database}}
digest = study_contexts.scientific_configuration_sha256(study)
wrapper = root / "study-a" / "run_original"; wrapper.mkdir(parents=True)
run_dir = wrapper / "pipeline" / "run_inner"; run_dir.mkdir(parents=True)
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
provider = {"provider": "openai", "external": True}
client, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=30)
book = ProviderHardStopLedger(path=wrapper / ".runtime" / "provider_hard_stop_ledger.json", task_ids=("task",),
    limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"), batch_id="task", declaration_sha256=digest)
task = book.start_task("task")
config = PipelineConfig(workdir=wrapper / "pipeline", runner_kind="docker")
seed = agent_review_recovery.WebReviewRecoverySeed.create(wrapper_dir=str(wrapper), study=study,
    scientific_configuration_sha256=digest, provider_meta=provider, provider_public=public,
    credential_source="pi_verified", budget_mode="planner_canary", prepared_package_binding=None,
    pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest(),
    acquisition_projection={}, hard_stop_ledger_path=str(book.path), hard_stop_task_id="task",
    hard_stop_declaration_sha256=digest, created_at=1.0)
agent_review_recovery.put_recovery_seed(seed)
(run_dir / "preplan_literature_bundle.json").write_text('{"research_question":"synthetic","citations":[]}')
inputs = wrapper / "pipeline_input"; inputs.mkdir()
selected = ["age"]
pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "age": pd.Series(dtype="float64")}).to_parquet(inputs / "planner_catalog.parquet", index=False)
(inputs / "planner_catalog_receipt.json").write_text(json.dumps({
    "schema_version": "easyicu.metadata-only-planning-catalog/1", "database": database, "row_identity_column": "stay_id",
    "selected_concepts": selected, "selected_concepts_sha256": hashlib.sha256(b'["age"]').hexdigest(),
    "patient_rows_read": False, "patient_rows_written": False, "observed_feasibility_claims": False,
    "execution_authorized": False, "operationalized_columns": [], "planning_target_outcome": None, "planning_endpoint": None,
}))

supervised = metadata_provider_client.MetadataProviderClient(inner=client, task=task, provider=provider,
    environment=env, public=public, timeout=30)
llm = HardStopClient(supervised, role="planner", task=task)
if stage == "family_spec":
    context = family_context(exact=False)
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": database})})
    options = {"planner_strategy": FAMILY_SPEC_STRATEGY, "allowed_literature_citation_keys": ALLOWED_CITATIONS,
        "direct_comparator_literature_keys": DIRECT_COMPARATORS, "comparison_literature_keys": DIRECT_COMPARATORS,
        "enforce_article_contract": True, "article_contract_context": context, "planning_contract_context": "",
        "required_primary_cohort_selection_mode": "predicate_filtered"}
else:
    context = outline_context()
    options = {"stop_after_outline": True}
if stage == "foundation":
    options = {}
def checkpoint_callback(checkpoint):
    (run_dir / f"progressive_planner_checkpoint_{checkpoint.sequence:03d}.json").write_text(checkpoint.model_dump_json(indent=2))
ProgressivePlannerAgent(llm).run(context, **options,
    checkpoint_callback=checkpoint_callback if stage == "foundation" else None,
    resume_dependency_context={"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"},
    outline_retry_journal={"task": task, "run_dir": run_dir, "literature_sha256": "e" * 64})
'''

_FOUNDATION_CONTINUE = r'''
import os, signal, sys
from pathlib import Path
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, TaskProviderHardStop
from easyicu.research_agent.planning.progressive_artifacts import ProgressivePlannerCheckpoint, ProgressivePlannerCheckpointRecorder
from easyicu.research_agent.planning import outline_retry_journal as journal
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.webserver import metadata_provider_client, provider_adapter, study_contexts
from tests.research_agent.planning.progressive_planner_fixtures import _context
from tests.research_agent.planning.test_progressive_planner_checkpoints import _RecordingEvidence
root, port, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
wrapper = root / "study-a" / "run_original"
study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(root / "source"), "database": "miiv"}}
digest = study_contexts.scientific_configuration_sha256(study)
book = ProviderHardStopLedger(path=wrapper / ".runtime" / "provider_hard_stop_ledger.json", task_ids=("task",),
    limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"), batch_id="task",
    declaration_sha256=digest, resume_existing=True, require_existing=True)
task = TaskProviderHardStop(book, "task")
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
provider = {"provider": "openai", "external": True}
client, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=30)
guarded = metadata_provider_client.MetadataProviderClient(inner=client, task=task, provider=provider,
    environment=env, public=public, timeout=30)
llm = HardStopClient(guarded, role="planner", task=task)
original = wrapper / "pipeline" / "run_inner"
source = original if mode == "interrupt" else root / "continued"
target = root / ("continued" if mode == "interrupt" else "finished")
target.mkdir()
checkpoint_path = source / "progressive_planner_checkpoint_000.json"
checkpoint = ProgressivePlannerCheckpoint.model_validate_json(checkpoint_path.read_text())
recorder = ProgressivePlannerCheckpointRecorder(run_dir=target, evidence=_RecordingEvidence(),
    prompt_pack_version="v1", source_chain=(checkpoint,))
if mode == "interrupt":
    old_complete = journal.OutlineRetryJournal.complete
    def before_transport(self, messages, **kwargs):
        if self._data["binding"]["stage"] == "foundation":
            (root / "continuation_ready").write_text("ready")
            signal.pause()
        return old_complete(self, messages, **kwargs)
    journal.OutlineRetryJournal.complete = before_transport
class FoundationAccepted(Exception): pass
def capture(checkpoint):
    if checkpoint.stage == "foundation": raise FoundationAccepted()
    recorder.record(checkpoint)
try:
    ProgressivePlannerAgent(llm).run(_context(), checkpoint_callback=capture, resume_checkpoint=checkpoint,
        resume_outline_validated_callback=recorder.persist_validated_outline,
        resume_dependency_context={"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"},
        outline_retry_journal={"task": task, "run_dir": target, "literature_sha256": "e" * 64,
                               "accepted_checkpoint_path": checkpoint_path})
except FoundationAccepted:
    (root / "foundation_accepted").write_text("accepted")
'''

_FACTORY_CHILD = r'''
import json, sys
from pathlib import Path
import pandas as pd
from easyicu.webserver import agent_pipeline_runs, durable_research, jobs, research_auto_resume, sources
root, port = Path(sys.argv[1]), int(sys.argv[2]); root.mkdir(exist_ok=True)
export = root / "source"; export.mkdir()
pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(export / "demographics.parquet", index=False)
(export / "_manifest.json").write_text(json.dumps({"database": "miiv", "format": "parquet",
    "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
    "feature_definitions": {"included": False}, "files": [{"file": "demographics.parquet",
    "module": "demographics", "concepts": 1, "concept_ids": ["age"], "rows": 1}]}))
assert sources.register_source(str(export), label="synthetic", crossdb=False)["ok"]
study = {"id": "study-a", "revision": 1, "question": "Describe the age distribution of adult ICU patients.",
         "data_source": {"path": str(export), "database": "miiv"}}
service = durable_research.DurableResearchService(root / "durable", manager=jobs.MANAGER,
    dispatcher_factory=lambda *_args, **_kwargs: object())
durable_research._SERVICE = service
descriptor = research_auto_resume.prepare_descriptor(study=study, project_root=root, credential_source="pi_verified",
    budget_mode="planner_canary", external_llm_opt_in=True)
service.store.prepare("original", descriptor)
service.store.activate("original")
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
runner = agent_pipeline_runs.make_research_pipeline_run_runner(export_path=str(export), study_context=study,
    project_root=str(root), provider={"provider": "openai", "external": True},
    provider_environment=env, credential_source="pi_verified", budget_mode="planner_canary")
class Job:
    id = "original"
    cancel_requested = False
    def emit(self, event): pass
    def is_cancel_requested(self): return False
runner(Job())
'''

_FACTORY_RECOVER = r'''
import signal, sys
from pathlib import Path
from easyicu.webserver import agent_review_recovery, durable_research, jobs, research_auto_resume, settings, study_contexts
root, port, source_job, attempt = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
wrapper = root / "study-a" / ("run_" + source_job)
seed = agent_review_recovery.load_recovery_seed(wrapper)
if seed is None: raise RuntimeError("missing continuation seed")
study = seed.study
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
settings.load_settings = lambda: {"ai_enabled": True}
study_contexts.get_context = lambda _id: study
research_auto_resume._provider_environment = lambda _seed: env
service = durable_research.DurableResearchService(root / "durable", manager=jobs.MANAGER,
    dispatcher_factory=lambda *_args, **_kwargs: object())
durable_research._SERVICE = service
descriptor = service.store.get("original")["payload"]
assert Path(descriptor["provider_binding"]["wrapper_dir"]) == wrapper
descriptor["_resume_attempt"] = attempt
if attempt == 2:
    from easyicu.research_agent.planning import outline_retry_journal
    old_complete = outline_retry_journal.OutlineRetryJournal.complete
    def interrupt_before_transport(self, messages, **kwargs):
        (root / "factory_continuation_ready").write_text("ready")
        signal.pause()
        return old_complete(self, messages, **kwargs)
    outline_retry_journal.OutlineRetryJournal.complete = interrupt_before_transport
class Job:
    id = "original"
    cancel_requested = False
    def emit(self, event): pass
    def is_cancel_requested(self): return False
job = Job()
research_auto_resume.recover_runner(descriptor, job)(job)
'''


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
def test_production_factory_creates_own_guarded_metadata_request(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver import metadata_provider_concurrency as slots

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
    arrived, release = threading.Event(), threading.Event()
    arrived_again, release_again = threading.Event(), threading.Event()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(json.loads(request))
            (tmp_path / f"factory_request_{len(requests)}.json").write_bytes(request)
            if len(requests) == 3:
                arrived.set(); release.wait(45)
            elif len(requests) >= 4:
                arrived_again.set(); release_again.wait(45)
            content = json.dumps({"selected_concepts": ["age"], "inclusion_exclusion": ["age >= 18"], "rationale": "Age is the requested measurement."} if len(requests) == 1 else {"groupings": []})
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            try:
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
        def log_message(self, *_args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    root = tmp_path / "run"
    child = subprocess.Popen([sys.executable, "-c", _FACTORY_CHILD, str(root), str(server.server_port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    children = [child]
    try:
        deadline = time.monotonic() + 60
        while not arrived.is_set() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not arrived.is_set():
            pytest.fail("real factory did not reach Provider: " + (child.stderr.read() if child.poll() is not None else "timed out"))
        child.kill(); child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        ledger_path = root / "study-a" / "run_original" / ".runtime" / "provider_hard_stop_ledger.json"
        first = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert [row["role"] for row in first] == ["acquisition", "planner", "planner"]
        assert first[-1]["state"] == "in_progress"
        held = slots.get(first[-1]["attempt_id"])
        release.set()
        assert_host_execution_quiescent(Path(held["guard_path"]), allow_missing=False, wait_timeout=30)
        child = subprocess.Popen([sys.executable, "-c", _FACTORY_RECOVER, str(root), str(server.server_port), "original", "2"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        deadline = time.monotonic() + 60
        while not (root / "factory_continuation_ready").exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not (root / "factory_continuation_ready").exists():
            pytest.fail("real factory continuation did not reach retry journal: " + (child.stderr.read() if child.poll() is not None else "timed out"))
        child.kill(); child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        second = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert second == first
        child = subprocess.Popen([sys.executable, "-c", _FACTORY_RECOVER, str(root), str(server.server_port), "original-restart-2", "3"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        # Reaching a fourth request is enough to prove the second continuation
        # bound the real factory's journal and original budget again.
        deadline = time.monotonic() + 60
        while len(requests) < 4 and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if len(requests) < 4:
            pytest.fail("second real factory continuation did not reach Provider: " + (child.stderr.read() if child.poll() is not None else "timed out"))
        assert requests[2]["messages"] == requests[3]["messages"]
        third = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert len(third) == len(first) + 1 and third[:len(first)] == first
        assert third[-1]["state"] == "in_progress"
        child.kill(); child.wait(timeout=10)
        release_again.set()
        final_guard = slots.get(third[-1]["attempt_id"])
        assert_host_execution_quiescent(Path(final_guard["guard_path"]), allow_missing=False, wait_timeout=30)
    finally:
        release.set(); release_again.set()
        for child in children:
            if child.poll() is None: child.kill(); child.wait(timeout=5)
        server.shutdown(); thread.join(timeout=5); server.server_close()


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
def test_foundation_survives_two_host_sigkills_without_resetting_budget(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.research_agent.planning.outline_retry_journal import FILE_NAME, load_retry_journal
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver import metadata_provider_concurrency as slots
    from tests.research_agent.planning.progressive_planner_fixtures import _outline_payload, _foundation_payload

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
    arrived, release = threading.Event(), threading.Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request["messages"])
            if len(calls) == 2:
                arrived.set()
                release.wait(45)
            content = json.dumps(_outline_payload() if len(calls) == 1 else _foundation_payload())
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_args): pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    root = tmp_path / "run"
    children = []
    try:
        child = subprocess.Popen([sys.executable, "-c", _CHILD, str(root), str(server.server_port), "foundation", "miiv"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        assert arrived.wait(60), "foundation request was not reached"
        child.kill(); child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        wrapper = root / "study-a" / "run_original"
        ledger_path = wrapper / ".runtime" / "provider_hard_stop_ledger.json"
        before = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert [row["state"] for row in before] == ["completed", "in_progress"]
        held = slots.get(before[-1]["attempt_id"])
        release.set()
        assert_host_execution_quiescent(Path(held["guard_path"]), allow_missing=False, wait_timeout=30)
        child = subprocess.Popen([sys.executable, "-c", _FOUNDATION_CONTINUE, str(root), str(server.server_port), "interrupt"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        deadline = time.monotonic() + 45
        while not (root / "continuation_ready").exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not (root / "continuation_ready").exists():
            pytest.fail("continuation did not reach the pre-transport interruption: " + (child.stderr.read() if child.poll() is not None else "timed out"))
        child.kill(); child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        continued = root / "continued"
        assert (continued / "progressive_planner_checkpoint_000.json").exists()
        journal = load_retry_journal(continued / ".runtime" / FILE_NAME)
        assert journal["expected_retry_guarded"] and journal["abandoned"] == 1
        child = subprocess.Popen([sys.executable, "-c", _FOUNDATION_CONTINUE, str(root), str(server.server_port), "finish"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        _, stderr = child.communicate(timeout=45)
        assert child.returncode == 0, stderr
        assert (root / "foundation_accepted").exists()
        assert len(calls) == 3 and calls[1] == calls[2]
        after = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert len(after) == 3 and after[:2] == before
    finally:
        release.set()
        for child in children:
            if child.poll() is None: child.kill(); child.wait(timeout=5)
        server.shutdown(); thread.join(timeout=5); server.server_close()


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
@pytest.mark.parametrize(("stage", "database"), [("outline", "miiv"), ("family_spec", "eicu")])
def test_outline_retry_survives_real_host_sigkill_with_original_budget(tmp_path, monkeypatch, stage, database):
    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, load_provider_hard_stop_ledger
    from easyicu.research_agent.planning.outline_retry_journal import FILE_NAME, load_retry_journal
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_client, metadata_provider_concurrency as slots, provider_adapter, research_auto_resume, settings, study_contexts
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
    from tests.research_agent.planning.progressive_planner_fixtures import _context as outline_context, _outline_payload
    from tests.research_agent.planning.family_spec_fixtures import (
        _context as family_context, _request, _spec_payload, ALLOWED_CITATIONS, DIRECT_COMPARATORS,
    )
    if stage == "family_spec":
        context = family_context(exact=False)
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": database})})
        options = {"planner_strategy": FAMILY_SPEC_STRATEGY, "allowed_literature_citation_keys": ALLOWED_CITATIONS,
                   "direct_comparator_literature_keys": DIRECT_COMPARATORS, "comparison_literature_keys": DIRECT_COMPARATORS,
                   "enforce_article_contract": True, "article_contract_context": context, "planning_contract_context": "",
                   "required_primary_cohort_selection_mode": "predicate_filtered"}
        accepted = _spec_payload(_request(context))
    else:
        context, options, accepted = outline_context(), {"stop_after_outline": True}, _outline_payload()

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    arrived, release = threading.Event(), threading.Event()
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request["messages"])
            if len(calls) == 2:
                arrived.set()
                release.wait(45)
            content = '{"bad":"not-an-outline"}' if len(calls) == 1 else json.dumps(accepted)
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "run"
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(root), str(server.server_port), stage, database],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        assert arrived.wait(60), "second fake Provider request was not reached"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        wrapper = root / "study-a" / "run_original"
        ledger_path = wrapper / ".runtime" / "provider_hard_stop_ledger.json"
        before = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]
        assert [row["state"] for row in before["calls"]] == ["completed", "in_progress"]
        held = slots.get(before["calls"][-1]["attempt_id"])
        assert held is not None
        release.set()
        assert_host_execution_quiescent(Path(held["guard_path"]), allow_missing=False, wait_timeout=30)
        env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
               "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
               "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        provider = {"provider": "openai", "external": True}
        client, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=30)
        study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(root / "source"), "database": database}}
        digest = study_contexts.scientific_configuration_sha256(study)
        seed = SimpleNamespace(provider_public=public, hard_stop_ledger_path=str(ledger_path), hard_stop_task_id="task",
                               budget_mode="planner_canary", hard_stop_declaration_sha256=digest)
        with pytest.raises(ResearchPipelineRunError, match="original authority"):
            research_auto_resume._verify_guarded_planner_inflight(seed, before)
        monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
        research_auto_resume._verify_guarded_planner_inflight(seed, before)
        # Exercise the actual Web recovery admission and checkpoint owners.
        # The continuation factory below is a narrow runner shim; it invokes
        # the real guarded planner without touching patient data.
        monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
        monkeypatch.setattr(study_contexts, "get_context", lambda _id: study)
        monkeypatch.setattr(research_auto_resume, "_provider_environment", lambda _seed: env)
        captured = {}
        source = wrapper / "pipeline" / "run_inner" / ".runtime" / FILE_NAME
        def make_runner(**kwargs):
            captured.update(kwargs)
            def run(job):
                original_task = job.provider_recovery.task
                resumed_client, resumed_public = provider_adapter.build_research_agent_provider_client(
                    provider, environ=env, request_timeout=30)
                guarded = metadata_provider_client.MetadataProviderClient(inner=resumed_client, task=original_task, provider=provider,
                    environment=env, public=resumed_public, timeout=30)
                return ProgressivePlannerAgent(HardStopClient(guarded, role="planner", task=original_task)).run(
                    context, **options,
                    resume_dependency_context={"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"},
                    outline_retry_journal={"task": original_task, "run_dir": root / "continued", "source_path": source,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "literature_sha256": "e" * 64})
            return run
        monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", make_runner)
        descriptor = research_auto_resume.prepare_descriptor(study=study, project_root=root, credential_source="pi_verified",
            budget_mode="planner_canary", external_llm_opt_in=True)
        descriptor["provider_binding"] = {"wrapper_dir": str(wrapper), "hard_stop_ledger_path": str(ledger_path),
            "hard_stop_task_id": "task", "hard_stop_declaration_sha256": digest}
        descriptor["_resume_attempt"] = 2
        job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda _event: None)
        result = research_auto_resume.recover_runner(descriptor, job)(job)
        assert captured["development_resume_source_job_id"] == "original"
        assert result.steps
        assert len(calls) == 3 and calls[1] == calls[2]
        after = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert len(after) == 3 and after[0] == before["calls"][0] and after[1] == before["calls"][1]
        assert slots.get(after[1]["attempt_id"]) == held
        journal = load_retry_journal(root / "continued" / ".runtime" / FILE_NAME)
        assert journal["abandoned"] == 1 and journal["binding"]["stage"] == stage
    finally:
        release.set()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
