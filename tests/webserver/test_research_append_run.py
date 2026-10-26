"""Result additions run once, in a new run, over immutable synthetic inputs."""

import pytest
import json
from types import SimpleNamespace

from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.webserver.research_plan_revision import PlanRevisionGuard
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.support.result_append import approvable_review, source_plan, addition, completed_source


def test_append_run_admits_only_its_new_steps():
    source = source_plan()
    plan = source.model_copy(deep=True, update={"steps": [addition()], "revision": source.revision + 1})
    guard = PlanRevisionGuard.for_append(
        plan=source.model_dump(mode="json"), review=approvable_review(source),
        expected_plan=plan.model_dump(mode="json"), action="sensitivity", append_run=True,
    )
    PipelineServices(plan_revision_guard=guard).validate_plan_revision(plan=plan, context=None)
    with pytest.raises(ResearchPipelineRunError):
        guard.validate_plan(plan=source, context=None)
    replay = source.model_copy(deep=True, update={"revision": source.revision + 1})
    with pytest.raises(ResearchPipelineRunError, match="distinct source"):
        PlanRevisionGuard.for_append(plan=source.model_dump(mode="json"), review=approvable_review(source),
                                     expected_plan=replay.model_dump(mode="json"), action="sensitivity", append_run=True)


@pytest.mark.parametrize("kind", ["sensitivity", "exposure", "subgroup", "article"])
def test_append_kind_runs_from_sealed_source_to_result(tmp_path, kind, monkeypatch):
    from easyicu.webserver.research_append_source import seal_completed_source, read_json
    from easyicu.webserver.research_append_run import append_run_for_id
    from easyicu.webserver import agent_pipeline_runs, study_contexts
    from easyicu.webserver.routes import research_append as route

    wrapper, run, source_plan, _, baseline = completed_source(tmp_path)
    source = seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
                                   baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})
    before = {str(path.relative_to(wrapper)): path.read_bytes() for path in wrapper.rglob("*") if path.is_file()}
    attention = []
    service = SimpleNamespace(_research_run_row=lambda *args: {"project_dir": str(wrapper), "study_id": "study"},
                              rebind_session=lambda *args, **kwargs: None,
        host_action_session=lambda *args, **kwargs: None,
        replay_store=SimpleNamespace(attention=SimpleNamespace(record=attention.extend)))
    monkeypatch.setattr(route, "get_pi_copilot_service", lambda: service)
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    response = route.post_append_run("conversation", route.AppendRequest(
        project_id="project", run_id="source", kind=kind, expected_source_sha256=source.digest))
    child = append_run_for_id(response["run_id"])
    assert agent_pipeline_runs.pending_review(child.checkpoint.run_id)["is_append_run"]
    assert [(row.run_id, row.kind, row.reason_code) for row in attention] == [
        (child.checkpoint.run_id, "needs_review", "human_plan_review_required")]
    assert child.plan.steps and not ({s.step_id for s in child.plan.steps} & {s.step_id for s in source_plan.steps})
    assert len(child.plan.steps) == {"sensitivity": 1, "exposure": 3, "subgroup": 1, "article": 2}[kind]
    widened = child.plan.model_copy(deep=True, update={"steps": [*child.plan.steps, addition()]})
    with pytest.raises(ValueError, match="roster changed"):
        child.authority.validate_plan(widened)
    from easyicu.research_agent.canonical_json import canonical_sha256
    body = child.authority.model_dump(mode="json")
    body["source_run_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        type(child.authority).model_validate_json(json.dumps(body))
    body = child.authority.model_dump(mode="json")
    body["append_plan_sha256"] = "b" * 64
    body["execution_contract_sha256"] = canonical_sha256({k: v for k, v in body.items() if k != "execution_contract_sha256"})
    with pytest.raises(ValueError, match="typed owner"):
        type(child.authority).model_validate_json(json.dumps(body))
    with pytest.raises(ValueError, match="roster drifted"):
        baseline.validate_plan(child.plan)
    from easyicu.webserver.agent_runs import read_run_record
    pending = read_run_record(str(child.wrapper))
    assert pending.ok, pending
    agent_pipeline_runs.resume_research_pipeline(
        run_id=child.checkpoint.run_id, study_context_id="study", decision="approved", reviewer="synthetic",
        note="", job=SimpleNamespace(id="append-test", cancel_requested=False, emit=lambda _: None),
        current_study_context={"id": "study"})
    result = read_json(child.run_dir / "append_result.json")
    assert result["execution_complete"] is True
    from easyicu.webserver.pi_copilot.projections import project_run_outcome
    completed = read_run_record(str(child.wrapper))
    assert completed.ok, completed
    public_run = project_run_outcome(completed.to_dict())
    assert public_run["analysis_results_available"]
    assert not public_run["numeric_verified"]
    assert {str(path.relative_to(wrapper)): path.read_bytes() for path in wrapper.rglob("*") if path.is_file()} == before
    assert source_plan.design_selection.execution_tier == "first_result"
    if kind == "article":
        assert child.plan.design_selection.execution_tier == "article"
        assert any("figure:append_article_figure" in step["output_files"] for step in result["steps"])
    if kind == "subgroup":
        assert result["steps"][0]["interaction_tests"]
        assert result["steps"][0]["stratum_estimates"]
        assert all(row["subgroup"] for row in result["steps"][0]["analysis_rows"])
        public = read_json(child.wrapper / "result_tables.json")["append"]
        assert len(public["strata"]) == 2
        assert all(row["interaction_p_value"] == result["steps"][0]["interaction_tests"][row["subgroup"]]
                   for row in public["strata"])
    if kind == "exposure":
        assert result["steps"][-1]["interpretation"] == "different_exposure_definition"
        assert "table:append_exposure_absolute_risk" in result["steps"][-1]["output_files"]
