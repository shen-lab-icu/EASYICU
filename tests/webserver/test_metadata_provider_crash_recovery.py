"""Guarded fake request across a real parent SIGKILL; no external Provider."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


_PARENT = r'''
import json, sys
from pathlib import Path
import easyicu.research_agent.providers.llm
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.webserver import metadata_provider_client, provider_adapter, research_auto_resume, research_early_resume, study_contexts

root = Path(sys.argv[1]); port = int(sys.argv[2]); role = sys.argv[3]; source = root / "source"; source.mkdir()
study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(source), "database": "miiv"}}
digest = study_contexts.scientific_configuration_sha256(study)
wrapper = root / "study-a" / "run_original"; wrapper.mkdir(parents=True)
source_binding = {"database": "miiv", "export_path": str(source), "source_identity": {"registration_id": "r"},
                  "catalog_sha256": "a" * 64, "without_values": []}
research_early_resume.metadata_binding = lambda *args: dict(source_binding)
environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
               "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
               "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
provider = {"provider": "openai", "external": True}
client, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=25)
ledger = ProviderHardStopLedger(path=wrapper / ".runtime" / "provider_hard_stop_ledger.json", task_ids=("task",),
    limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"), batch_id="task", declaration_sha256=digest)
task = ledger.start_task("task")
research_early_resume.create_seed(wrapper=wrapper, study_id="study-a", scientific_sha256=digest,
    credential_source="pi_verified", budget_mode="planner_canary", provider_public=public,
    database="miiv", export_path=str(source), runner_image=None, original_source_identity=source_binding["source_identity"],
    hard_stop_ledger_path=str(ledger.path), hard_stop_task_id="task", hard_stop_declaration_sha256=digest)
descriptor = research_auto_resume.prepare_descriptor(study=study, project_root=root, credential_source="pi_verified",
    budget_mode="planner_canary", external_llm_opt_in=True)
descriptor["provider_binding"] = {"wrapper_dir": str(wrapper), "hard_stop_ledger_path": str(ledger.path),
    "hard_stop_task_id": "task", "hard_stop_declaration_sha256": digest}
descriptor["_resume_attempt"] = 2
(root / "descriptor.json").write_text(json.dumps(descriptor))
supervised = metadata_provider_client.MetadataProviderClient(inner=client, task=task, provider=provider,
    environment=environment, public=public, timeout=25)
HardStopClient(supervised, role=role, task=task).complete([LLMMessage(role="user", content="synthetic metadata")], max_tokens=8)
'''


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL test requires POSIX")
@pytest.mark.parametrize("role", ["acquisition", "planner", "provider_recovery_probe"])
def test_inflight_fake_crash_retains_unknown_then_restarts_with_provisioned_slot(tmp_path, monkeypatch, role):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_concurrency as slots, provider_adapter, research_auto_resume, research_early_resume, settings, study_contexts
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    first_entered, release_first = threading.Event(), threading.Event()
    count = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            count.append(1)
            if len(count) == 1:
                first_entered.set()
                release_first.wait(timeout=30)
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "1")
    child = subprocess.Popen([sys.executable, "-c", _PARENT, str(tmp_path), str(server.server_port), role],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy())
    try:
        assert first_entered.wait(timeout=60), f"fake Provider was not reached; launcher_exit={child.poll()}"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        descriptor = json.loads((tmp_path / "descriptor.json").read_text())
        ledger_path = Path(descriptor["provider_binding"]["hard_stop_ledger_path"])
        before = load_provider_hard_stop_ledger(ledger_path)
        old_call = before["tasks"][0]["calls"][0]
        assert old_call["state"] == "in_progress"
        held = slots.get(old_call["attempt_id"])
        assert held is not None
        assert_host_execution_quiescent(Path(held["guard_path"]), allow_missing=False, wait_timeout=15)
        source = str(tmp_path / "source")
        study = {"id": "study-a", "question": "synthetic", "data_source": {"path": source, "database": "miiv"}}
        binding = {"database": "miiv", "export_path": source, "source_identity": {"registration_id": "r"},
                   "catalog_sha256": "a" * 64, "without_values": []}
        monkeypatch.setattr(research_early_resume, "metadata_binding", lambda *args: dict(binding))
        monkeypatch.setattr(study_contexts, "get_context", lambda _id: study)
        monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
        environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        monkeypatch.setattr(research_auto_resume, "_provider_environment", lambda _seed: environment)
        job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda _event: None)
        seed = SimpleNamespace(provider_public=research_early_resume.load_seed(Path(descriptor["provider_binding"]["wrapper_dir"]))["provider_public"],
                               hard_stop_ledger_path=str(ledger_path), hard_stop_task_id="task", budget_mode="planner_canary",
                               hard_stop_declaration_sha256=descriptor["scientific_configuration_sha256"])
        def check():
            if role != "acquisition":
                research_auto_resume._verify_guarded_planner_inflight(seed, load_provider_hard_stop_ledger(ledger_path)["tasks"][0])
            else:
                research_auto_resume.recover_runner(descriptor, job)
        with pytest.raises(ResearchPipelineRunError) as refused:
            check()
        assert refused.value.code == "research_auto_resume_metadata_provider_concurrency_unavailable"
        assert len(load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]) == 1

        # Only the isolated fake server is provisioned to hold both requests.
        monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
        def make(**kwargs):
            def run(bound_job):
                task = bound_job.provider_recovery.task
                client, public = provider_adapter.build_research_agent_provider_client(kwargs["provider"], environ=environment, request_timeout=10)
                supervised = MetadataProviderClient(inner=client, task=task, provider=kwargs["provider"],
                    environment=environment, public=public, timeout=10)
                return {"answer": HardStopClient(supervised, role=role, task=task).complete(
                    [LLMMessage(role="user", content="synthetic metadata")], max_tokens=8)}
            return run
        monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", make)
        if role != "acquisition":
            check()
            task = research_auto_resume._original_task(seed)
            client, public = provider_adapter.build_research_agent_provider_client({"provider": "openai", "external": True}, environ=environment, request_timeout=10)
            supervised = MetadataProviderClient(inner=client, task=task, provider={"provider": "openai", "external": True}, environment=environment, public=public, timeout=10)
            assert HardStopClient(supervised, role=role, task=task).complete([LLMMessage(role="user", content="synthetic metadata")], max_tokens=8) == "ok"
        else:
            assert research_auto_resume.recover_runner(descriptor, job)(job) == {"answer": "ok"}
        calls = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert len(calls) == 2 and calls[0] == old_call and calls[1]["state"] == "completed"
        assert slots.get(old_call["attempt_id"]) == held
        assert len(count) == 2
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        release_first.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
