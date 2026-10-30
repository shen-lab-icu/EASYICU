"""Production DBOS plus the Web recovery factory after process death.

This is synthetic source data and a loopback provider, not a research run.
"""
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

import pytest

pytest.importorskip("dbos")

_START = r'''
import json, sys, time
from pathlib import Path
import pandas as pd
from easyicu.webserver import agent_pipeline_runs, durable_research, jobs, research_auto_resume, sources, study_contexts
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
study_contexts.clear_active_job_if = lambda *args, **kwargs: {}
manager = jobs.JobManager(); manager.enable_persistence(root / "jobs.sqlite3")
service = durable_research.DurableResearchService(root / "durable", manager=manager)
durable_research._SERVICE = service; service.launch()
descriptor = research_auto_resume.prepare_descriptor(study=study, project_root=root, credential_source="pi_verified",
    budget_mode="planner_canary", external_llm_opt_in=True)
job = service.reserve(descriptor)
(root / "job-id.txt").write_text(job.id)
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
runner = agent_pipeline_runs.make_research_pipeline_run_runner(export_path=str(export), study_context=study,
    project_root=str(root), provider={"provider": "openai", "external": True},
    provider_environment=env, credential_source="pi_verified", budget_mode="planner_canary")
service.activate(job.id, runner)
while True:
    if job.status != "running": raise RuntimeError("unexpected terminal: " + str(job.error))
    time.sleep(.05)
'''

_RESTART = r'''
import sys, time
from pathlib import Path
from easyicu.webserver import agent_review_recovery, durable_research, jobs, research_auto_resume, settings, study_contexts
root, port = Path(sys.argv[1]), int(sys.argv[2]); identity = (root / "job-id.txt").read_text()
wrapper = root / "study-a" / ("run_" + identity)
seed = agent_review_recovery.load_recovery_seed(wrapper)
if seed is None: raise RuntimeError("missing recovery seed")
env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
settings.load_settings = lambda: {"ai_enabled": True}
study_contexts.get_context = lambda _id: seed.study
study_contexts.clear_active_job_if = lambda *args, **kwargs: {}
research_auto_resume._provider_environment = lambda _seed: env
manager = jobs.JobManager(); manager.enable_persistence(root / "jobs.sqlite3")
service = durable_research.DurableResearchService(root / "durable", manager=manager)
durable_research._SERVICE = service; service.launch()
while True:
    job = manager.get(identity)
    command = service.store.get(identity)
    if job and job.status != "running":
        raise RuntimeError("recovery stopped: " + str(job.error) + " command status="
                           + str(command.get("status")) + " reason=" + str(command.get("reason")))
    time.sleep(.05)
'''


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
def test_dbos_restarts_production_web_factory_with_original_budget(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
    from easyicu.webserver import metadata_provider_concurrency as slots

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
    requests = []
    first_inflight, second_inflight = threading.Event(), threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if len(requests) == 3:
                first_inflight.set(); release.wait(60)
            elif len(requests) >= 4:
                second_inflight.set(); release.wait(60)
            content = ({"selected_concepts": ["age"], "inclusion_exclusion": ["age >= 18"],
                        "rationale": "Age is the requested measurement."} if len(requests) == 1
                       else {"groupings": []})
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant",
                "content": json.dumps(content)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            try:
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    root = tmp_path / "run"
    children = []
    try:
        child = subprocess.Popen([sys.executable, "-c", _START, str(root), str(server.server_port)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        deadline = time.monotonic() + 90
        while not first_inflight.is_set() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not first_inflight.is_set():
            pytest.fail("initial DBOS request not reached: " + (child.stderr.read() if child.poll() is not None else "timeout"))
        child.kill(); child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        identity = (root / "job-id.txt").read_text()
        ledger_path = root / "study-a" / ("run_" + identity) / ".runtime" / "provider_hard_stop_ledger.json"
        first = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert first[-1]["state"] == "in_progress"
        guard = slots.get(first[-1]["attempt_id"])
        assert_host_execution_quiescent(Path(guard["guard_path"]), allow_missing=False, wait_timeout=30)
        child = subprocess.Popen([sys.executable, "-c", _RESTART, str(root), str(server.server_port)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        children.append(child)
        deadline = time.monotonic() + 90
        while not second_inflight.is_set() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not second_inflight.is_set():
            pytest.fail("DBOS recovery request not reached: " + (child.stderr.read() if child.poll() is not None else "timeout"))
        assert requests[2]["messages"] == requests[3]["messages"]
        current = load_provider_hard_stop_ledger(ledger_path)["tasks"][0]["calls"]
        assert current[:len(first)] == first and len(current) == len(first) + 1
        assert current[-1]["state"] == "in_progress"
        child.kill(); child.wait(timeout=10)
        guard = slots.get(current[-1]["attempt_id"])
        assert_host_execution_quiescent(Path(guard["guard_path"]), allow_missing=False, wait_timeout=30)
    finally:
        release.set()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
