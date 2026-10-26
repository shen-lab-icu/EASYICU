"""Recovered lifecycle metadata cannot stand in for a scientific result."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from easyicu.webserver import jobs
from easyicu.webserver.jobs import Job
from easyicu.webserver.routes.jobs import jobs_events
from tests.support.node import run_node


def _events(job: Job, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    monkeypatch.setattr(jobs, "MANAGER", SimpleNamespace(get=lambda job_id: job))

    async def collect():
        response = await jobs_events(job.id)
        return [json.loads(chunk.removeprefix("data: ")) async for chunk in response.body_iterator]

    return asyncio.run(collect())


def test_sse_preserves_recovered_omission_reason_and_timestamp(monkeypatch):
    original = Job("done-job", "agent-run")
    original.finish("done", {"human_review_pending": True})
    recovered = Job.from_lifecycle_receipt(original.lifecycle_receipt())
    event = _events(recovered, monkeypatch)[-1]
    assert event["type"] == "end"
    assert event["status"] == "done"
    assert event["result"] is None
    assert event["recovered"] is True
    assert event["result_omitted"] is True
    assert event["reason_code"] == "job_receipt_restored"
    assert event["timestamp"] == recovered.snapshot()["events"][-1]["timestamp"]


def test_live_oversized_sse_end_still_rehydrates_its_owned_result(monkeypatch):
    job = Job("live-job", "agent-run", max_event_bytes=512)
    result = {"human_review_pending": True, "summary": "synthetic " * 100}
    job.finish("done", result)
    assert job.snapshot()["events"][-1]["result_omitted"] is True
    event = _events(job, monkeypatch)[-1]
    assert event["result"] == result
    assert not event.get("result_omitted")
    assert not event.get("recovered")


def test_child_job_owner_marks_restored_results_unverified_and_does_not_continue():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source_dir = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js"
    program = f"""
      import assert from 'node:assert/strict';
      import fs from 'node:fs';
      import vm from 'node:vm';
      const modules = new Map();
      const guidedPi = {{
        declare: (name, owner) => modules.set(name, owner),
        require: (name) => modules.get(name), jobProgress: (job) => job.progress || [],
      }};
      const context = vm.createContext({{window:{{EasyICU:{{guidedPi}}}}}});
      for (const name of ['screens-guided-pi-replay.js', 'screens-guided-pi-childjob.js']) {{
        vm.runInContext(fs.readFileSync({json.dumps(str(source_dir))} + '/' + name, 'utf8'), context);
      }}
      const messages = [];
      let activeId = '', stream = null, continued = 0, refreshed = 0;
      const host = {{
        tr:(en, zh) => en, messages:() => messages, render:() => {{}}, api:() => ({{}}),
        childJobId:() => activeId, setChildJobId:(id) => {{activeId = id;}},
        childSource:() => stream, setChildSource:(value) => {{stream = value;}},
        session:() => ({{session_id:'synthetic'}}), sessionIsStale:() => false,
        rebind:async () => {{}}, loadWorkflow:async () => {{refreshed++;}},
        refreshSession:async () => {{}}, archiveChildJob:async () => {{}},
        continueSystemOwnedPlanProgression:async () => {{continued++; return false;}},
        upsertActivityStep:(activity, step) => {{
          const at = activity.steps.findIndex(row => row.id === step.id);
          if (at < 0) activity.steps.push(step); else activity.steps[at] = step;
        }},
        activity:{{pipelineTime:() => 1000, finishPipelineSteps:() => {{}}}},
      }};
      const owner = modules.get('childJob').create(host);
      activeId = 'recovered';
      owner.handleChildJobEvent('recovered', 'agent-run', {{type:'end', status:'done',
        result:null, recovered:true, result_omitted:true, timestamp:'2026-01-01T00:00:00Z'}});
      await new Promise(resolve => setTimeout(resolve, 0));
      assert.equal(continued, 0);
      assert.equal(refreshed, 1);
      const recovered = messages.find(row => row.childJobId === 'recovered');
      assert.equal(recovered.status, 'blocked');
      assert.match(recovered.displayTitle, /need verification/);
      assert.match(recovered.steps.find(row => row.id === 'pipeline-terminal').label, /verify the plan/);
      owner.hydrateProjectedJob({{present:true, job_id:'hydrated', kind:'agent-run', status:'done',
        recovered:true, result_omitted:true, created_at_epoch:1, finished_at_epoch:2, progress:[]}});
      const hydrated = messages.find(row => row.childJobId === 'hydrated');
      assert.equal(hydrated.status, 'blocked');
      assert.match(hydrated.displayTitle, /need verification/);
      activeId = 'live';
      owner.handleChildJobEvent('live', 'agent-run', {{type:'end', status:'done', result:{{human_review_pending:true}}}});
      await new Promise(resolve => setTimeout(resolve, 0));
      assert.equal(continued, 1, 'Existing live result progression remains available');
      const live = messages.find(row => row.childJobId === 'live');
      assert.match(live.displayTitle, /ready for review/);
    """
    result = run_node(node, program, module=True, timeout=20)
    assert result.returncode == 0, result.stderr
