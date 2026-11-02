"""No-network synthetic manual resume probe against the real pipeline entry."""
import json
import socket

import pytest



def test_manual_resume_sealed_plan_under_engine_drift(ra, tmp_path, monkeypatch):
    from easyicu.research_agent.agents.planner import PlannerAgent
    from easyicu.research_agent.authority.run_input import engine_code_sha256
    from tests.support import resume_capsule as fixture_owner

    def no_network(*args, **kwargs):
        raise AssertionError("external networking is forbidden in the synthetic probe")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    actual_current = engine_code_sha256()
    original_builder = fixture_owner.build_environment_identity
    def old_source_environment(*, llm_signature):
        result = original_builder(llm_signature=llm_signature)
        return {**result, "engine_code_sha256": "7927010ce8cd43f03c7f8b584bd6704dc5ec15d2bf11e68f7ca7494a37553a64"}
    monkeypatch.setattr(fixture_owner, "build_environment_identity", old_source_environment)
    pipeline, run_dir, kwargs = fixture_owner._write_capsule_resume_fixture(ra, tmp_path)
    assert pipeline._planner_strategy == "monolithic_v1"
    before = (run_dir / "analysis_plan.json").read_bytes()
    from easyicu.research_agent.agents.plan_payload import planner_structured_output_request
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    old_schema_sha = "6359fcaf4e8f4f03cde5ee30663a2e287d84eb792f97eecfc48783b465630629"
    assert planner_structured_output_request([]).authority_sha256 != old_schema_sha
    metrics = run_dir / "planner_prompt_metrics.json"
    metrics.write_text(json.dumps({"schema_version": "easyicu.planner_prompt_metrics/1",
                                   "structured_output_authority_sha256": old_schema_sha}))
    EvidenceStore(run_dir).register_file(kind="log", description="Prior synthetic planner request metrics",
        source_path=metrics, evidence_id="planner_prompt_metrics", producer="planner", generation_mode="llm")
    def forbid_planner(*args, **kwargs):
        raise AssertionError("manual resume unexpectedly re-entered classic Planner")
    monkeypatch.setattr(PlannerAgent, "run", forbid_planner)
    original_generate = pipeline._generate_or_resume_plan
    observed = {}
    def stop_after_real_generation(**kwargs):
        result = original_generate(**kwargs)
        observed.update(reused=result.reused_prior_plan, mode=result.plan_generation_mode,
                        steps=[item.step_id for item in result.plan.steps])
        raise RuntimeError("synthetic stop after real plan selection")
    monkeypatch.setattr(pipeline, "_generate_or_resume_plan", stop_after_real_generation)
    with pytest.raises(RuntimeError, match="synthetic stop after real plan selection"):
        pipeline.run(**kwargs)
    receipt = json.loads(next(run_dir.glob("resume_environment_receipt_*.json")).read_text())
    assert receipt["environment_drift"] is True
    assert "engine_code_sha256" in receipt["changed_fields"]
    assert receipt["current_environment"]["engine_code_sha256"] == actual_current
    assert observed == {"reused": True, "mode": "resumed", "steps": ["01_summary"]}
    assert (run_dir / "analysis_plan.json").read_bytes() == before
    assert json.loads(metrics.read_text())["structured_output_authority_sha256"] == old_schema_sha


def test_manual_resume_missing_sealed_plan_never_replans(ra, tmp_path, monkeypatch):
    from easyicu.research_agent.agents.planner import PlannerAgent
    from tests.support import resume_capsule as fixture_owner

    def forbidden(*args, **kwargs):
        raise AssertionError("missing plan must not initiate network or Planner")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    pipeline, run_dir, kwargs = fixture_owner._write_capsule_resume_fixture(ra, tmp_path)
    (run_dir / "analysis_plan.json").unlink()
    sealed = run_dir / "evidence" / "analysis_plan__analysis_plan.json"
    assert sealed.is_file()
    sealed.unlink()
    monkeypatch.setattr(PlannerAgent, "run", forbidden)
    with pytest.raises((ValueError, RuntimeError)) as error:
        pipeline.run(**kwargs)
    assert "plan" in str(error.value).lower() or "evidence" in str(error.value).lower()
    print("missing sealed plan:", type(error.value).__name__, str(error.value))
