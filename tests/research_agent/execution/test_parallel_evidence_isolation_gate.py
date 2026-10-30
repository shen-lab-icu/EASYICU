"""Concurrent workers must not observe a peer's unfinished scientific evidence."""

import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution import phase
from easyicu.research_agent.execution.run_coordination import RunCoordinator


@pytest.mark.integration
def test_unisolated_workers_are_serialized_and_the_reason_is_recorded(
    ra, synthetic_cohort, tmp_path, monkeypatch,
):
    """Exercise the actual execution path with the isolation owner unavailable.

    This synthetic diagnostic removes the other preflight reasons only; the
    isolation owner remains unchanged and must independently prevent parallel
    dispatch. No patient or external model data are involved.
    """

    calls = []
    original_sequential = RunCoordinator.run_sequential

    def sequential(self, **kwargs):
        calls.append("sequential")
        return original_sequential(self, **kwargs)

    def parallel(self, **kwargs):
        raise AssertionError("unisolated workers must not be dispatched together")

    monkeypatch.setattr(phase, "_step_emit_execution_preflight", lambda **kwargs: (False, False))
    monkeypatch.setattr(RunCoordinator, "run_sequential", sequential)
    monkeypatch.setattr(RunCoordinator, "run_parallel", parallel)
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path), llm=ra.MockLLMClient(), max_concurrent_steps=2,
        enable_replanning=False, enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    result = pipeline.run(
        question="Describe the synthetic ICU cohort", cohort=synthetic_cohort, database="miiv",
    )
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert calls == ["sequential"]
    assert any(
        finding.get("validator") == "parallel_evidence_isolation"
        and finding.get("detail", {}).get("reason") == "parallel_evidence_isolation_unavailable"
        for finding in manifest["findings"]
    )
