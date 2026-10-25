"""The sealed run tier, never study notes, owns the literature obligation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.orchestration import scientific_plan_review_gate as gate
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.planning.design_selection import ResearchDesignSelection
from easyicu.research_agent.planning.literature_design_authority import LiteratureDesignAuthorityError
from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext


def context():
    return ResearchContext(
        research_question="Association of a synthetic marker with a synthetic binary outcome.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=2),
        variables=[],
    )


def plan(tier="first_result"):
    candidates = [dict(
        design_id=name, analysis_type="association_study", estimand="Synthetic association",
        time_zero="Synthetic baseline", observation_window="Fixed synthetic window",
        primary_method=method, required_variables=["synthetic_marker", "synthetic_outcome"],
        assumptions=["Baseline precedes outcome"], novelty_positioning="Unassessed",
        figure_role="Result brief", supports="Association", cannot_prove="Causation",
        disposition=disposition, decision_reason="Follows the synthetic question",
    ) for name, method, disposition in [
        ("primary", "logistic regression", "selected"), ("alternative", "stratified regression", "rejected"),
    ]]
    return AnalysisPlan(
        research_question=context().research_question, analysis_type="association_study", steps=[],
        design_selection=ResearchDesignSelection(execution_tier=tier, candidates=candidates),
    )


def _preplan(tmp_path, monkeypatch, *, tier="first_result", live=False, strict=False, skill=None,
             resume_state=None, evidence=None):
    config = PipelineConfig(
        workdir=tmp_path, execution_tier=tier, enable_pubmed=live,
        require_human_plan_review=True, require_literature_design_authority=strict,
        planner_strategy="progressive_v2",
    )
    runtime = SimpleNamespace(
        _config=config, _enable_literature=True, _enable_pubmed=live,
        _pubmed_email=None, _pubmed_api_key=None, _enable_tavily=False,
        _tavily_api_key=None, _tavily_retmax=3, _tavily_include_domains=(),
        _bound_preplan_literature=None, _development_resume_reuse_bound_literature=False,
        _enable_hypothesis_generator=True, _hypothesis_generator_top_k=3,
        _finalise_aborted=Mock(return_value=SimpleNamespace()),
        _llm_signature=lambda _: "synthetic", _iter_mock_clients=lambda _: iter(()),
    )
    prepare = Mock(return_value=LiteratureBundle(research_question=context().research_question, citations=[]))
    hypotheses = Mock(return_value=SimpleNamespace(
        candidates=[], to_json=lambda: {"candidates": []}, to_markdown=lambda: "No synthetic candidates.",
    ))
    monkeypatch.setattr(pipeline_module, "prepare_preplan_literature", prepare)
    monkeypatch.setattr(pipeline_module, "generate_hypotheses", hypotheses)
    study = context().model_copy(update={
        "notes": "Synthetic notes request literature and novelty positioning.",
        "execution_tier": "article" if tier == "first_result" else "first_result",
    })
    events, findings = [], []
    result = pipeline_module._run_preplan_literature_and_hypothesis(
        runtime, skill_obj=skill,
        emit_progress=lambda stage, message, **kw: events.append(dict(stage=stage, message=message, **kw)),
        run_id="synthetic_run", agent_context=study, run_dir=tmp_path,
        evidence=evidence if evidence is not None else EvidenceStore(tmp_path), findings=findings,
        context=study, context_path=tmp_path / "context.json", llm=None, resume_state=resume_state,
    )
    return result, prepare, hypotheses, events, findings, runtime


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("skill", [None, "synthetic_fixed_skill"])
def test_first_result_continues_without_any_literature_or_hypothesis_call(tmp_path, monkeypatch, live, strict, skill):
    result, prepare, hypotheses, events, findings, _ = _preplan(
        tmp_path, monkeypatch, live=live, strict=strict, skill=skill,
    )
    prepare.assert_not_called()
    hypotheses.assert_not_called()
    assert result[0] is None  # Planning continues; no provider, abort or worker to await.
    assert result[2:] == ([], [], None)  # No citation or comparison authority fabricated.
    assert not findings
    assert events == [{
        "stage": "hypothesis", "message": "Literature is deferred to a full article run.",
        "status": "skipped", "run_id": "synthetic_run",
    }]
    assert not (tmp_path / "hypothesis_blueprint.json").exists()
    assert not (tmp_path / "preplan_literature_bundle.json").exists()


def test_appended_article_run_searches_the_same_study_and_keeps_preplan_gate(tmp_path, monkeypatch):
    first = tmp_path / "first"
    article = tmp_path / "article"
    first.mkdir()
    article.mkdir()
    _, first_call, _, _, _, _ = _preplan(first, monkeypatch)
    first_call.assert_not_called()
    result, prepare, _, events, findings, runtime = _preplan(
        article, monkeypatch, tier="article", strict=True,
    )
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["enable_pubmed"] is False
    assert result[0] is not None  # Missing reviewed literature still stops an article.
    assert any(f.validator == "literature_design_authority" and f.severity == "error" for f in findings)
    runtime._finalise_aborted.assert_called_once()
    assert events[-1]["status"] == "error"


def test_article_cannot_bypass_strict_literature_with_a_fixed_skill(tmp_path, monkeypatch):
    with pytest.raises(LiteratureDesignAuthorityError, match="cannot be bypassed"):
        _preplan(tmp_path, monkeypatch, tier="article", strict=True, skill=object())


def test_first_result_resume_without_a_literature_bundle_still_skips(tmp_path, monkeypatch):
    result, prepare, hypotheses, _, findings, _ = _preplan(
        tmp_path, monkeypatch, resume_state={}, live=True, strict=True,
    )
    assert result[0] is None and result[2:] == ([], [], None)
    prepare.assert_not_called()
    hypotheses.assert_not_called()
    assert not findings


@pytest.mark.parametrize("corrupt", [False, "digest", "question"])
def test_older_first_result_resume_keeps_its_verified_literature_binding(tmp_path, monkeypatch, corrupt):
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    bundle = LiteratureBundle(
        research_question=context().research_question,
        citations=[{"key": key, "title": "Synthetic comparison", "year": "2025"}
                   for key in ("synthetic_included", "synthetic_excluded")],
        screening_decisions=[{
            "citation_key": key, "source": "bound_search", "disposition": disposition,
            "evidence_role": "direct_comparator", "rationale": "Synthetic screening decision",
        } for key, disposition in [("synthetic_included", "include"), ("synthetic_excluded", "exclude")]],
    )
    if corrupt == "question":
        bundle = bundle.model_copy(update={"research_question": "A different synthetic question."})
    source = tmp_path / "preplan_literature_bundle.json"
    source.write_text(bundle.model_dump_json())
    evidence = EvidenceStore(tmp_path)
    evidence.register_file(kind="log", description="Synthetic prior literature", source_path=source,
                           evidence_id="preplan_literature_bundle")
    study, proposed = context(), plan()
    approved = build_plan_scientific_review(context=study, plan=proposed, literature=bundle)
    gate.persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=approved)
    if corrupt:
        if corrupt == "digest":
            record = evidence.get("preplan_literature_bundle")
            (tmp_path / record.relative_path).write_text('{}')
        with pytest.raises(ValueError, match="question-matched" if corrupt == "question" else "digest-verified"):
            _preplan(tmp_path, monkeypatch, resume_state={}, evidence=evidence)
        return
    fresh, _, _, _, _, _ = _preplan(tmp_path, monkeypatch, evidence=evidence)
    assert fresh[-1] is None  # A new run never adopts an older bundle implicitly.
    result, prepare, hypotheses, _, findings, _ = _preplan(
        tmp_path, monkeypatch, resume_state={}, evidence=evidence,
    )
    assert result[-1] == bundle
    assert result[2:4] == (["synthetic_included"], ["synthetic_included"])
    current = build_plan_scientific_review(context=study, plan=proposed, literature=result[-1])
    resumed, _ = gate.persist_or_validate_scientific_plan_review(
        run_dir=tmp_path, evidence=evidence, current_review=current, reuse_existing_review=True,
    )
    assert resumed == approved
    prepare.assert_not_called()
    hypotheses.assert_not_called()
    assert not findings


def test_offline_article_still_runs_hypothesis_generation(tmp_path, monkeypatch):
    _, prepare, hypotheses, events, _, _ = _preplan(tmp_path, monkeypatch, tier="article")
    prepare.assert_called_once()
    hypotheses.assert_called_once()
    assert prepare.call_args.kwargs["enable_pubmed"] is False
    assert (tmp_path / "hypothesis_blueprint.json").is_file()
    assert (tmp_path / "hypothesis_candidates.json").is_file()
    assert events[0]["message"] == "Building pre-plan literature and hypothesis blueprint."


def test_first_result_planning_instructions_withhold_literature_claims():
    from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions

    instructions = first_result_instructions()
    assert "Do not claim novelty, literature comparisons or citation support" in instructions
    assert "deferred to an article run" in instructions


def test_first_result_workflow_completes_without_literature(tmp_path, monkeypatch):
    from easyicu.research_agent.orchestration.workflow import WorkflowCompleted, build_pipeline_workflow
    from easyicu.research_agent.reporting.write_phase import run_write_phase

    calls = []

    def prepare():
        result, literature_call, _, _, _, _ = _preplan(tmp_path, monkeypatch, live=True, strict=True)
        literature_call.assert_not_called()
        assert result[0] is None
        calls.append("plan")
        return SimpleNamespace(plan=plan(), aborted_result=None)

    def execute(prepared):
        calls.append("execute")
        return SimpleNamespace(plan=prepared.plan)

    def write(prepared, executed):
        calls.append("brief")
        return run_write_phase(
            SimpleNamespace(_config=SimpleNamespace(generate_manuscript=True)),
            plan_result=prepared, execute_result=executed,
            run_dir=tmp_path, run_id="synthetic", stop_after_analysis=False,
            manuscript_title=None, manuscript_authors=None, run_language="en",
            emit_progress=lambda *a, **kw: pytest.fail("article writer entered"), force_writer_probe=True,
        )

    workflow = build_pipeline_workflow(
        plan_invoker=prepare, execute_invoker=execute, write_invoker=write,
        finalise_invoker=lambda _plan, _executed, written: written.bound_path,
    )
    outcome = workflow.start()
    assert isinstance(outcome, WorkflowCompleted)
    assert workflow.state == "completed"
    assert outcome.final_result.is_file()
    assert calls == ["plan", "execute", "brief"]


@pytest.mark.parametrize("tier", ["first_result", "article"])
def test_postplan_gates_read_the_sealed_plan_not_the_profile(tier):
    findings = []
    config = SimpleNamespace(
        execution_tier="article" if tier == "first_result" else "first_result",
        require_literature_design_authority=True, require_literature_retrieval_evidence=True,
    )
    gate.append_required_literature_findings(config, findings, plan(tier), None)
    assert {f.validator for f in findings} == (
        set() if tier == "first_result" else {"literature_design_authority", "literature_retrieval_gate"}
    )


@pytest.mark.parametrize("design,retrieval", [(False, False), (True, False), (False, True)])
def test_article_preserves_each_evidence_gate_setting(design, retrieval):
    findings = []
    config = SimpleNamespace(require_literature_design_authority=design, require_literature_retrieval_evidence=retrieval)
    gate.append_required_literature_findings(config, findings, plan("article"), None)
    assert {f.validator for f in findings} == (
        ({"literature_design_authority"} if design else set())
        | ({"literature_retrieval_gate"} if retrieval else set())
    )


@pytest.mark.parametrize("tier", ["first_result", "article"])
def test_plan_finalization_applies_the_tier_before_literature_gates(tmp_path, tier):
    import inspect

    import pyarrow as pa
    import pyarrow.parquet as pq

    from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding

    proposed = plan(tier)
    saved = tmp_path / "analysis_plan.json"
    saved.write_text(proposed.model_dump_json())
    cohort = tmp_path / "synthetic.parquet"
    pq.write_table(pa.table({"synthetic_marker": [0, 1], "synthetic_outcome": [1, 0]}), cohort)
    config = PipelineConfig(
        workdir=tmp_path, execution_tier=tier, require_human_plan_review=True,
        require_literature_design_authority=True, require_literature_retrieval_evidence=True,
        planner_strategy="progressive_v2",
    )
    runtime = SimpleNamespace(
        _config=config, _services=PipelineServices(),
        _scientific_runtime_authorities=ScientificRuntimeAuthorities(None, None),
        _required_primary_cohort_selection_mode=None,
    )
    method = pipeline_module.ResearchAgentPipeline._validate_and_persist_plan
    arguments = {name: None for name in inspect.signature(method).parameters if name != "self"}
    arguments.update(
        generation=SimpleNamespace(
            plan=proposed, proposed_plan=proposed, reused_prior_plan=True, reused_plan_path=saved,
            migrated_plan_path=None, plan_generation_mode="resumed", used_mock_llm=True, planner_prompt_metrics=None,
        ),
        context=context(), agent_context=context(), context_path=tmp_path / "context.json",
        cohort_path=cohort, run_dir=tmp_path, run_id="synthetic", findings=[],
        evidence=EvidenceStore(tmp_path), know_how_binding=PlannerKnowHowBinding(),
        llm_signature="synthetic", prompt_version="synthetic", prompt_files=(),
        allowed_literature_citation_keys=[], direct_comparator_literature_keys=[],
        emit_progress=lambda *a, **kw: None,
    )
    result = method(runtime, **arguments)
    assert result.plan.design_selection.execution_tier == tier
    literature_errors = {f.validator for f in result.findings if f.validator.startswith("literature_")}
    assert literature_errors == (
        set() if tier == "first_result" else {"literature_design_authority", "literature_retrieval_gate"}
    )
    assert result.evidence.get("scientific_plan_review") is not None
