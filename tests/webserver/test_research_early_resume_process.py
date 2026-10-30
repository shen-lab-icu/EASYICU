"""Actual host death at the pre-call boundary, with a fake provider only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


_CHILD = r'''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.webserver import (agent_pipeline_runs, provider_adapter, provider_gate, research_auto_resume as recovery,
    research_early_resume as early, settings, study_contexts, deployment_lease)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

root = Path(sys.argv[1]); mode = sys.argv[2]
deployment_lease.acquire_single_process_lease(root / "web.lock")
source = root / "source"; source.mkdir(exist_ok=True)
study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(source)}}
digest = study_contexts.scientific_configuration_sha256(study)
wrapper = root / "study-a" / "run_original"; wrapper.mkdir(parents=True, exist_ok=True)
binding = {"database": "sic", "export_path": str(source), "source_identity": {"registration_id": "r"},
    "catalog_sha256": "a" * 64, "without_values": []}
early.metadata_binding = lambda *args: dict(binding)
public = {"provider": "openai", "model": "synthetic", "credential_fingerprint": "b" * 64}
ledger_path = wrapper / ".runtime" / "provider_hard_stop_ledger.json"
descriptor = recovery.prepare_descriptor(study=study, project_root=root, credential_source="pi_verified",
    budget_mode="planner_canary", external_llm_opt_in=True)
descriptor["provider_binding"] = {"wrapper_dir": str(wrapper), "hard_stop_ledger_path": str(ledger_path),
    "hard_stop_task_id": "web-original", "hard_stop_declaration_sha256": digest}
descriptor["_resume_attempt"] = 2
job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda event: None)

if mode in {"before", "inflight"}:
    book = ProviderHardStopLedger(path=ledger_path, task_ids=("web-original",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="web-original", declaration_sha256=digest)
    book.start_task("web-original")
    early.create_seed(wrapper=wrapper, study_id="study-a", scientific_sha256=digest,
        credential_source="pi_verified", budget_mode="planner_canary", provider_public=public,
        database="sic", export_path=str(source), runner_image=None,
        original_source_identity=binding["source_identity"], hard_stop_ledger_path=str(ledger_path),
        hard_stop_task_id="web-original", hard_stop_declaration_sha256=digest)
    if mode == "inflight":
        book.reserve_transport_attempt(task_id="web-original", role="acquisition", model="synthetic",
            messages=[], max_tokens=4, prior_attempt_id=None)
    (root / "ready").write_text(mode)
    while True: time.sleep(1)

study_contexts.get_context = lambda key: study
settings.load_settings = lambda: {"ai_enabled": True}
recovery._provider_environment = lambda seed: {}
provider_gate.resolve_provider_gate = lambda **kwargs: {"provider": "openai", "external": True}
provider_adapter.build_research_agent_provider_client = lambda *a, **kw: (object(), dict(public))
def make(**kwargs):
    def run(job):
        task = job.provider_recovery.task
        return {"job_id": job.id, "ledger_path": str(task.ledger.path),
            "calls": len(task.ledger.snapshot()["tasks"][0]["calls"])}
    return run
agent_pipeline_runs.make_research_pipeline_run_runner = make
try:
    value = recovery.recover_runner(descriptor, job)(job)
    print(json.dumps({"status": "resumed", **value}))
except ResearchPipelineRunError as exc:
    print(json.dumps({"status": "refused", "code": exc.code}))
'''


@pytest.mark.parametrize("mode", ["before", "inflight"])
def test_sigkill_recovery_preserves_the_original_reservation(tmp_path, mode):
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path), mode], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready").exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "ready").exists(), child.communicate(timeout=1) if child.poll() is not None else "child not ready"
        # A second live Web owner must be denied the same deployment lease.
        contender = subprocess.run(
            [sys.executable, "-c", "from pathlib import Path; import sys; from easyicu.webserver.deployment_lease import acquire_single_process_lease; acquire_single_process_lease(Path(sys.argv[1]))", str(tmp_path / "web.lock")],
            capture_output=True, text=True, timeout=30,
        )
        assert contender.returncode != 0 and "exactly one server worker" in contender.stderr
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        assert child.returncode == -signal.SIGKILL
        result = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), "resume"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout.strip())
        if mode == "before":
            assert receipt == {"status": "resumed", "job_id": "original-restart-2",
                "ledger_path": str(tmp_path / "study-a" / "run_original" / ".runtime" / "provider_hard_stop_ledger.json"), "calls": 0}
        else:
            assert receipt == {"status": "refused", "code": "research_auto_resume_early_provider_attempt_already_reserved"}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
