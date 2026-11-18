"""A plan can only leave the repair loop through its governed review boundary."""

import json

import pytest

from easyicu.research_agent.authority.plan_review import HumanReviewRequest
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QuestionRequirement,
    judge_question_requirements,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


NOT_COVERED = "question_requirement_not_covered"
GAP = "question_requirement_capability_gap"


def requirement(quote, *, disposition="not_covered", requirement_id="r1", **extra):
    return {
        "id": requirement_id,
        "quote": quote,
        "disposition": disposition,
        "verified_by_host": disposition in {"covered", "not_covered"},
        **extra,
    }


def run_script(tmp_path, *script):
    from easyicu.research_agent.orchestration.plan_repair import (
        check_plan_requirements_for_repair,
        planning_repair_prompt,
        run_plan_with_repairs,
    )

    prompts, notices = [], []
    responses = iter(script)

    def run():
        prompts.append(planning_repair_prompt())
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        rows, reason = response
        (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).write_text(
            json.dumps({"judged": rows}), encoding="utf-8"
        )
        request = HumanReviewRequest.create(
            kind="scientific_stop", summary=reason,
            authority_sha256="a" * 64, payload={"reason": reason},
        )
        check_plan_requirements_for_repair(tmp_path, (request,))
        return reason

    def classify(error):
        if isinstance(error, ProgressivePlanCompileError):
            return error.easyicu_safe_diagnostic
        return {}

    def result():
        return run_plan_with_repairs(run, check_failure=classify, notify=notices.append)

    return result, prompts, notices


def host_check(code):
    return ProgressivePlanCompileError(code, "synthetic validator detail", path="steps[1].inputs")


def test_host_check_then_repaired_plan(tmp_path):
    result, prompts, notices = run_script(
        tmp_path, host_check("progressive_unknown_variable"), ([], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert len(notices) == 1
    assert prompts[0] == ""
    assert "progressive_unknown_variable" in prompts[1]
    assert "steps[1].inputs" in prompts[1]


@pytest.mark.parametrize("quote", ["Report estimates by laboratory shift", "按化验班次报告估计值"])
def test_coverage_failure_then_repaired_plan(tmp_path, quote):
    result, prompts, notices = run_script(
        tmp_path, ([requirement(quote)], NOT_COVERED),
        ([requirement(quote, disposition="covered")], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert len(notices) == 1
    assert '"requirement_id":"r1"' in prompts[1]
    assert quote in prompts[1]


def test_available_capability_claim_is_returned_with_host_verdict(tmp_path):
    quote = "Summarize admission body temperature"
    claim = requirement(
        quote, disposition="capability_gap",
        gap={"requirement": "design_element_unsupported", "concept": "temp"},
        gap_verification="unverified", gap_fact="the study offers the registered concept 'temp'",
    )
    result, prompts, notices = run_script(
        tmp_path, ([claim], GAP),
        ([requirement(quote, disposition="covered")], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert len(notices) == 1
    assert '"host_verdict":"unverified"' in prompts[1]
    assert claim["gap_fact"] in prompts[1]


def test_verified_gap_is_not_replanned(tmp_path):
    gap = requirement(
        "Validate the model in an independent source",
        disposition="capability_gap", gap_verification="verified",
        verified_by_host=True, gap_fact="this study uses one source",
    )
    result, prompts, notices = run_script(tmp_path, ([gap], GAP))
    assert result() == (GAP, None)
    assert prompts == [""]
    assert notices == []


def test_repeat_ignores_row_order_and_planner_id_changes(tmp_path):
    first = requirement("Summarize laboratory shift", requirement_id="r1")
    second = requirement("Summarize collection location", requirement_id="r2")
    result, prompts, notices = run_script(
        tmp_path, ([first, second], NOT_COVERED),
        ([{**second, "id": "r1"}, {**first, "id": "r2"}], NOT_COVERED),
    )
    reason, stop = result()
    assert reason == NOT_COVERED
    assert stop == {"repairs": 1, "limit": 2, "outcome": "repeated_failure"}
    assert len(prompts) == 2 and len(notices) == 1


def test_distinct_checks_exhaust_budget_and_mark_the_terminal_receipt(tmp_path):
    checks = [host_check(f"progressive_fixture_check_{number}") for number in range(3)]
    result, prompts, notices = run_script(tmp_path, *checks)
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value is checks[-1]
    assert error.value.easyicu_plan_repair == {"repairs": 2, "limit": 2, "outcome": "budget_exhausted"}
    assert len(prompts) == 3 and len(notices) == 2
    assert checks[1].reason_code in prompts[2]


def test_dropping_an_unanswered_requirement_cannot_unlock_approval(tmp_path):
    first = requirement("Summarize laboratory shift")
    result, prompts, notices = run_script(
        tmp_path, ([first], NOT_COVERED), ([], "operator_plan_approval_required"),
    )
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value.reason_code == "progressive_question_requirement_repair_omitted"
    assert error.value.easyicu_plan_repair["outcome"] == "repeated_failure"
    assert len(prompts) == 2 and len(notices) == 1


@pytest.mark.parametrize("quote", ["Include a sensitivity analysis", "请提供敏感性分析"])
@pytest.mark.parametrize("include_owner", [True, False])
def test_host_delivery_depends_on_the_current_plan_owner(quote, include_owner):
    item = QuestionRequirement(id="r1", kind="analysis", quote=quote, coverage="plan")
    step = AnalysisStep(
        step_id="robustness", intent="Check uncertainty",
        planned_analysis_role="sensitivity", method="sensitivity_analysis", inputs=["signal"],
        scientific_action_id="association.robustness_panel", expected_outputs=["table:sensitivity"],
    )
    plan = AnalysisPlan(
        research_question=quote, analysis_type="association", steps=[step] if include_owner else [],
    )
    (judged,) = judge_question_requirements(
        (item,), plan=plan, family_template=False, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("No gap was declared"), question=quote,
    )
    assert judged.disposition == ("covered" if include_owner else "not_covered")
    if include_owner:
        assert judged.delivery["source"] == "host_verified"


@pytest.mark.parametrize("quote", [
    "Include a literature-informed design, data and cohort description, reproducible analysis, "
    "missing data handling, repeated stay handling, analysis diagnostics, and sensitivity analyses.",
    "请提供基于文献的研究设计、数据和人群描述、可复现的分析、缺失数据处理、重复住院处理、分析诊断和敏感性分析。",
])
def test_generic_deliverables_use_their_typed_owners(quote):
    from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
    from easyicu.research_agent.contracts.runtime_dependence import RuntimeDependenceContract
    from tests.research_agent.planning.scientific_review_fixtures import (
        _absolute_risk_distribution_step,
        _traditional_table_one_step,
    )

    primary = _absolute_risk_distribution_step().model_copy(update={
        "scientific_action_id": "descriptive.descriptive_summary",
        "literature_citation_keys": ["synthetic_evidence"],
        "runtime_dependence_contract": RuntimeDependenceContract(
            owner_ref="scientific_runtime_contract:" + "a" * 64,
            dependence=PlannedDependenceRequirement(group_source="patient", group_derivation="identity"),
        ),
    })
    audit = AnalysisStep(
        step_id="diagnostics", intent="Audit measurement completeness", planned_analysis_role="auxiliary",
        method="measurement_audit", expected_outputs=["table:measurement_audit"],
        measurement_audit_spec={"products": [{"product_id": "measurement_audit", "audit": "measurement_missingness"}]},
    )
    sensitivity = AnalysisStep(
        step_id="sensitivity", intent="Check a prespecified alternative", planned_analysis_role="sensitivity",
        scientific_action_id="descriptive.descriptive_summary", expected_outputs=["table:sensitivity"],
    )
    plan = AnalysisPlan(
        research_question=quote, analysis_type="descriptive_epidemiology",
        steps=[primary, _traditional_table_one_step(), audit, sensitivity],
    )
    item = QuestionRequirement(id="r1", kind="analysis", quote=quote, coverage="plan")
    (row,) = judge_question_requirements(
        (item,), plan=plan, family_template=False, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("No gap was declared"), question=quote,
    )
    assert row.disposition == "covered"
    assert row.delivery["owner_step_ids_by_kind"] == {
        "literature_grounded_design": [primary.step_id],
        "population_description": ["table_one"],
        "reproducible_analysis": [primary.step_id],
        "missing_data_handling": [primary.step_id],
        "repeated_admission_handling": [primary.step_id],
        "diagnostics": [audit.step_id],
        "sensitivity_analyses": [sensitivity.step_id],
    }


def test_repair_uses_uncovered_clauses_and_preserves_explicit_exposure_choice(tmp_path):
    quote = "Provide diagnostics and sensitivity analyses"
    first = requirement(quote, clauses=[
        {"quote": "diagnostics", "disposition": "not_covered"},
        {"quote": "sensitivity analyses", "disposition": "not_covered"},
    ])
    second = {**first, "clauses": [
        {"quote": "diagnostics", "disposition": "covered"},
        {"quote": "sensitivity analyses", "disposition": "not_covered"},
    ]}
    choice = requirement("Compare the two laboratory markers", requirement_id="r2", exposure_choice={"options": ["analyse_both"]})
    result, prompts, notices = run_script(
        tmp_path, ([first, choice], NOT_COVERED), ([second, choice], NOT_COVERED),
        ([requirement(quote, disposition="covered"), choice], NOT_COVERED),
    )
    assert result() == (NOT_COVERED, None)
    assert len(notices) == 2
    assert '"uncovered_clauses":["sensitivity analyses"]' in prompts[-1]
    assert choice["quote"] not in prompts[-1]


def test_source_owner_contradicts_a_gap_for_a_registered_available_concept():
    from easyicu.research_agent.planning.capability_gap import check_capability_gap
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    quote = "Estimate admission body temperature"
    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
    )
    item = QuestionRequirement(
        id="r1", kind="analysis", quote=quote, coverage="capability_gap",
        gap={"requirement": "design_element_unsupported", "element": "analysis",
             "concept": "temp", "detail": "The draft claims the source cannot supply this measurement."},
    )
    (row,) = judge_question_requirements(
        [item], plan=AnalysisPlan(research_question=quote, steps=[]), family_template=False,
        relatives=lambda name: frozenset({name}),
        check_gap=lambda gap: check_capability_gap(gap, context=context),
    )
    assert row.disposition == "not_covered"
    assert row.verified_by_host
    assert row.verification == "unverified"
    assert "bound source offers registered concept" in row.fact


@pytest.mark.parametrize("verdict", ["unverified", "unverifiable", "verified"])
def test_outline_gap_is_repairable_only_without_host_verification(verdict):
    from easyicu.research_agent.planning.capability_gap import CapabilityGapCheck, capability_gap_stop
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check

    outline = ProgressivePlanOutline(
        analysis_type="association", cohort_objective="Summarize a synthetic measurement.", steps=[],
        rationale="The candidate claims the analysis is unsupported.",
        capability_gap="design_element_unsupported", capability_gap_concept="synthetic_signal",
    )
    fact = "Synthetic bound-source verdict"
    check = CapabilityGapCheck(verdict, fact)
    if verdict == "unverified":
        with pytest.raises(ProgressivePlanCompileError) as error:
            capability_gap_stop(outline, check)
        stop = error.value
        assert not hasattr(stop, "easyicu_planning_capability_gap")
    else:
        stop = capability_gap_stop(outline, check)

    feedback = _repairable_plan_check(stop)
    if verdict == "verified":
        assert feedback == {}
    else:
        assert feedback["capability_gap"]["verification"] == verdict
        assert feedback["capability_gap"]["fact"] == fact
        assert "rationale" not in feedback["capability_gap"]


def test_unknown_delivery_cannot_be_approved(tmp_path):
    from types import SimpleNamespace

    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    quote = "Send an editable slide deck to an archive"
    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
    )
    plan = AnalysisPlan(research_question=quote, steps=[])
    facts = SimpleNamespace(family_template=True, question_requirements=[
        QuestionRequirement(id="r1", kind="delivery", quote=quote, coverage="plan"),
    ])
    findings = question_requirement_outcome(context=context, plan=plan, facts=facts, run_dir=tmp_path)
    (request,) = human_review_requests_for_plan(
        findings=findings, plan=plan, require_plan_review=True,
        evidence=SimpleNamespace(root=tmp_path, records=lambda: ()),
    )
    assert request.payload["reason"] == NOT_COVERED
    assert request.payload["approval_allowed"] is False


def test_real_progressive_request_contains_typed_feedback(tmp_path):
    from easyicu.research_agent.agents.progressive_planner import (
        FAMILY_SPEC_STRATEGY,
        ProgressivePlannerAgent,
    )
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.plan_repair import run_plan_with_repairs
    from easyicu.research_agent.orchestration.progressive_planning import run_pipeline_progressive_planner
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check
    from tests.research_agent.planning.family_spec_fixtures import (
        ALLOWED_CITATIONS,
        DIRECT_COMPARATORS,
        _descriptive_context,
        _descriptive_payload,
        _request,
    )

    context = _descriptive_context().model_copy(update={
        "research_question": "Describe the distribution of the endpoint by the study marker. Provide diagnostics.",
    })
    spec = _descriptive_payload(_request(context, cohort_mode="all_input_rows"), baseline_variables=[])
    spec["question_requirements"] = [{
        "id": "r1", "kind": "analysis", "quote": "Provide diagnostics", "coverage": "plan",
    }]
    llm = ScriptedMockLLMClient([json.dumps(spec), json.dumps(spec)])
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"synthetic metadata catalog")
    run_dirs = []

    def run():
        run_dir = tmp_path / f"candidate-{len(run_dirs)}"
        run_dirs.append(run_dir)
        evidence = EvidenceStore(run_dir)
        findings = []
        output = run_pipeline_progressive_planner(
            planner=ProgressivePlannerAgent(llm), context=context, run_dir=run_dir, evidence=evidence,
            prompt_pack_version="test-v1", resume_checkpoint_path=None, resume_checkpoint_sha256=None,
            stop_after_outline=False, cohort_path=cohort, llm_signature="mock:test",
            planner_kwargs={
                "planner_strategy": FAMILY_SPEC_STRATEGY,
                "allowed_literature_citation_keys": ALLOWED_CITATIONS,
                "direct_comparator_literature_keys": DIRECT_COMPARATORS,
                "comparison_literature_keys": DIRECT_COMPARATORS,
                "enforce_article_contract": True,
                "article_contract_context": context,
                "planning_contract_context": "",
            },
            preplan_literature=None, required_primary_cohort_selection_mode="all_input_rows",
            know_how_binding=PlannerKnowHowBinding(), planning_contract_context="",
            finding_sink=findings.append,
        )
        if len(run_dirs) == 1:
            raise host_check("progressive_unknown_variable")
        return human_review_requests_for_plan(
            findings=findings, plan=output.plan, evidence=evidence, require_plan_review=True,
        )

    requests, receipt = run_plan_with_repairs(run, check_failure=_repairable_plan_check, notify=lambda _: None)
    assert receipt is None
    assert [request.payload["reason"] for request in requests] == ["operator_plan_approval_required"]
    assert len(llm.calls) == 2
    assert "progressive_unknown_variable" not in str(llm.calls[0])
    assert "progressive_unknown_variable" in str(llm.calls[1])
    assert "steps[1].inputs" in str(llm.calls[1])


def test_terminal_repair_diagnostic_projects_only_safe_receipt_fields(tmp_path):
    from easyicu.webserver.agent_pipeline_runs import (
        _pipeline_failure_code,
        _record_pipeline_failure,
        _repairable_plan_check,
    )
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection

    error = host_check("progressive_unknown_variable")
    error.easyicu_plan_repair = {"repairs": 1, "limit": 2, "outcome": "repeated_failure"}
    check = _repairable_plan_check(error)
    assert check["path"] == "steps[1].inputs"
    assert "message" not in check
    diagnostic = _record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "study-synthetic"}, provider={},
        exc=error, code=_pipeline_failure_code(error), execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    public = gate_detail_projection(gate["detail"])
    assert public["gate_detail_code"] == "progressive_unknown_variable"
    assert public["plan_repair"] == error.easyicu_plan_repair
    assert [action["label_key"] for action in gate["next_actions"]] == ["stop_next_actions.edit_plan"]
    assert "synthetic validator detail" not in (tmp_path / diagnostic).read_text()


def test_review_rechecks_a_removed_host_deliverable_owner(tmp_path):
    from types import SimpleNamespace

    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    quote = "可复现分析及敏感性分析"
    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
    )
    primary = AnalysisStep(
        step_id="primary", intent="Estimate the association of the synthetic series",
        planned_analysis_role="primary", scientific_action_id="association.adjusted_association",
        expected_outputs=["table:primary"],
    )
    sensitivity = AnalysisStep(
        step_id="sensitivity", intent="Check the stability of the estimate",
        planned_analysis_role="sensitivity", scientific_action_id="association.robustness_panel",
        expected_outputs=["table:stability"],
    )
    plan = AnalysisPlan(research_question=quote, analysis_type="association", steps=[primary, sensitivity])
    facts = SimpleNamespace(family_template=True, question_requirements=[
        QuestionRequirement(
            id="r1", kind="analysis", quote=quote, coverage="plan", clauses=["可复现分析", "敏感性分析"],
        ),
    ])
    findings = question_requirement_outcome(context=context, plan=plan, facts=facts, run_dir=tmp_path)
    assert findings == []
    evidence = SimpleNamespace(root=tmp_path, records=lambda: ())
    approved_candidate = human_review_requests_for_plan(
        findings=findings, plan=plan, evidence=evidence, require_plan_review=True,
    )
    assert [row.payload["reason"] for row in approved_candidate] == ["operator_plan_approval_required"]
    (uncovered,) = human_review_requests_for_plan(
        findings=findings, plan=plan.model_copy(update={"steps": [primary]}),
        evidence=evidence, require_plan_review=True,
    )
    assert uncovered.payload["reason"] == NOT_COVERED
    assert uncovered.payload["approval_allowed"] is False
    review = json.loads((tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).read_text())
    assert [clause["disposition"] for clause in review["judged"][0]["clauses"]] == ["covered", "not_covered"]


def test_one_web_child_retries_with_the_same_source_and_provider_task(tmp_path, monkeypatch):
    import pandas as pd

    from easyicu.research_agent import ResearchAgentPipeline
    from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
    from easyicu.research_agent.orchestration.plan_repair import planning_repair_prompt
    from easyicu.research_agent.orchestration.workflow import HumanReviewPending
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver import agent_pipeline_runs, provider_adapter
    from tests.webserver.copilot.research_workflow_fixtures import _write_pipeline_export

    study = {
        "id": "study-synthetic-repair", "revision": 1,
        "question": "Describe missingness and report the count of documented discharge outcomes.",
        "outcome": "In-hospital death", "execution_concepts": {"outcome": "death"},
    }
    export = _write_pipeline_export(tmp_path / "export")
    study["data_source"] = {"path": str(export), "database": "miiv"}
    client = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["age", "death"],
        "inclusion_exclusion": [],
        "rationale": "Describe missingness without inferring individual outcomes.",
    })])
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client",
                        lambda *_args, **_kwargs: (client, {"provider": "mock", "model": "synthetic-repair"}))
    monkeypatch.setattr(agent_pipeline_runs, "_metadata_only_planning_catalog",
                        lambda *, database, **_kwargs: build_database_capability_catalog(database))
    attempts, tasks, projections = [], [], []

    class CandidatePipeline:
        def run(self, **arguments):
            assert arguments["question"] == study["question"]
            assert pd.read_parquet(arguments["cohort"]).empty
            attempts.append((arguments["cohort"], planning_repair_prompt()))
            if len(attempts) == 1:
                raise host_check("progressive_unknown_variable")
            return HumanReviewPending(
                run_id="run_synthetic_repaired", thread_id="thread_synthetic_repaired",
                run_dir=str(tmp_path / "candidate"),
                requests=(HumanReviewRequest.create(
                    kind="scientific_stop", summary="Review the candidate plan.",
                    authority_sha256="c" * 64, payload={"reason": "operator_plan_approval_required"},
                ),),
            )

    def create_pipeline(config, *, services):
        assert config.require_human_plan_review
        tasks.append(services.provider_hard_stop)
        return CandidatePipeline()

    def project(**kwargs):
        projections.append(kwargs)
        return {"run_id": kwargs["pending"].run_id}

    monkeypatch.setattr(ResearchAgentPipeline, "from_config", create_pipeline)
    monkeypatch.setattr(agent_pipeline_runs, "_write_projection", project)
    job_events = []

    class Job:
        id = "job-synthetic-plan-repair"
        cancel_requested = False

        def emit(self, event):
            job_events.append(event)

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export), study_context=study, project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "synthetic-key"}, credential_source="pi_verified",
        budget_mode="planner_canary",
    )
    assert runner(Job()) == {"run_id": "run_synthetic_repaired"}
    assert len(attempts) == 2 and attempts[0][0] == attempts[1][0]
    assert "steps[1].inputs" in attempts[1][1]
    assert tasks[0] is tasks[1] and tasks[0] is not None
    assert len(projections) == 1
    assert projections[0]["pending"].requests[0].payload["reason"] == "operator_plan_approval_required"
    assert [event["current"] for event in job_events if event.get("retry_phase") == "rejected"] == [1]
