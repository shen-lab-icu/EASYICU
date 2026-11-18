"""A plan can only leave the repair loop through its governed review boundary."""

import json

import pytest

from easyicu.research_agent.authority.plan_review import HumanReviewRequest
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QUESTION_REQUIREMENTS_FILENAME,
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


def run_script(tmp_path, *script, check_failure=None):
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
        return run_plan_with_repairs(run, check_failure=check_failure or classify, notify=notices.append)

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


@pytest.mark.parametrize("other", [
    {"kind": "estimand", "effect_measure": "odds_ratio", "disposition": "covered"},
    {"kind": "definition", "disposition": "definition_only"},
])
def test_an_uncovered_row_survives_a_shared_quote(tmp_path, other):
    quote = "Report the measures of association for the laboratory marker"
    measure = {"effect_measure": "risk_ratio"} if other["kind"] == "estimand" else {}
    missing = requirement(quote, **measure)
    sibling = requirement(quote.upper(), requirement_id="r2", **other)
    result, prompts, notices = run_script(
        tmp_path, ([missing, sibling], NOT_COVERED),
        ([{**missing, "disposition": "covered"}, sibling], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert notices == [1]
    assert '"requirement_id":"r1"' in prompts[1]


@pytest.mark.parametrize("same_identity", [False, True])
def test_omitting_one_of_two_host_verified_rows_cannot_unlock_approval(tmp_path, same_identity):
    quote = "Summarize afternoon recovery and graph the estimates"
    analysis = {"kind": "analysis"}
    delivery = {"kind": "delivery", "deliverable_kinds": ["figures"]}
    covered = requirement(quote, disposition="covered", **(delivery if same_identity else analysis))
    missing = requirement(quote.upper(), requirement_id="r2", **delivery)
    result, prompts, notices = run_script(
        tmp_path, ([covered, missing], NOT_COVERED),
        ([{**covered, "id": "renumbered"}], "operator_plan_approval_required"),
    )
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value.reason_code == "progressive_question_requirement_repair_omitted"
    assert error.value.easyicu_plan_repair["outcome"] == "repeated_failure"
    assert len(prompts) == 2 and notices == [1]
    assert '"requirement_id":"r2"' in prompts[1]


@pytest.mark.parametrize("same_identity", [False, True])
def test_resolving_one_host_verified_sibling_is_repair_progress(tmp_path, same_identity):
    quote = "Summarize afternoon recovery and graph the estimates"
    delivery = {"kind": "delivery", "deliverable_kinds": ["figures"]}
    first = requirement(quote, **(delivery if same_identity else {"kind": "analysis"}))
    second = requirement(quote.upper(), requirement_id="r2", **delivery)
    fixed_first = {**first, "disposition": "covered", "id": "renumbered"}
    result, prompts, notices = run_script(
        tmp_path, ([first, second], NOT_COVERED),
        ([second, fixed_first], NOT_COVERED),
        ([fixed_first, {**second, "disposition": "covered"}], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert len(prompts) == 3 and notices == [1, 2]
    latest = json.loads(prompts[-1].split("\n", 1)[1])[-1]["requirements"]
    assert [row["requirement_id"] for row in latest] == ["r2"]


def test_an_unverified_claim_cannot_replace_previously_verified_requirement(tmp_path):
    quote = "Describe whether temperature measurements were delayed"
    covered = requirement(quote, kind="analysis", disposition="covered")
    missing = requirement(quote, kind="analysis", requirement_id="r2")
    claim = requirement(quote, kind="analysis", requirement_id="r3", disposition="capability_gap")
    result, _, notices = run_script(
        tmp_path, ([covered, missing], NOT_COVERED), ([covered, claim], GAP), ([covered, claim], GAP),
    )
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value.reason_code == "progressive_question_requirement_repair_omitted"
    assert notices == [1, 2]


def test_a_verified_row_satisfies_prior_feedback_in_any_row_order(tmp_path):
    quote = "Report the change in recovery probability"
    definition = requirement(quote, disposition="definition_only", requirement_id="r1", kind="definition")
    missing = requirement(quote, requirement_id="r2", kind="analysis")
    result, _, notices = run_script(
        tmp_path, ([definition, missing], NOT_COVERED),
        ([{**missing, "disposition": "covered"}, definition], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert notices == [1]


def test_removing_one_unrequested_effect_measure_is_progress(tmp_path):
    quote = "Summarize recovery for the study population"
    reason = "question_effect_measure_unrequested"
    extra_odds = requirement("", reason_code=reason, effect_measure="odds_ratio")
    extra_risk = requirement("", reason_code=reason, effect_measure="risk_ratio", requirement_id="r2")
    covered = requirement(quote, disposition="covered", requirement_id="r3")
    result, prompts, notices = run_script(
        tmp_path, ([covered, extra_odds, extra_risk], NOT_COVERED),
        ([covered, extra_risk], NOT_COVERED), ([covered], "operator_plan_approval_required"),
    )
    assert result() == ("operator_plan_approval_required", None)
    assert notices == [1, 2]
    assert all(f'"effect_measure":"{row["effect_measure"]}"' in prompts[1] for row in (extra_odds, extra_risk))


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


@pytest.mark.parametrize("family_template", [True, False])
@pytest.mark.parametrize("include_figure", [False, True])
def test_delivery_requires_a_figure_owner_in_the_reviewed_plan(include_figure, family_template, tmp_path):
    from types import SimpleNamespace

    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    quote = "Render a plot of the study estimates."
    item = QuestionRequirement(
        id="r1", kind="delivery", quote=quote, coverage="plan", deliverable_kinds=["figures"],
    )
    table = AnalysisStep(step_id="estimates", intent="Estimate the effect", expected_outputs=["table:estimates"])
    figure = AnalysisStep(
        step_id="plot", intent="Plot estimates", method="visualization",
        expected_outputs=["figure:estimates"], inputs=["table:estimates"],
        figure_panels=[{
            "panel_id": "estimates", "figure_output": "figure:estimates", "article_role": "primary_estimand",
            "chart_type": "forest", "source_products": ["table:estimates"],
        }],
    )
    draft = AnalysisPlan(research_question=quote, steps=[table, figure])
    reviewed = draft if include_figure else draft.model_copy(update={"steps": [table]})
    (judged,) = judge_question_requirements(
        (item,), plan=reviewed, family_template=family_template, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("No gap was declared"),
    )
    assert judged.disposition == ("covered" if include_figure else "not_covered")
    if include_figure:
        assert judged.delivery["owner_step_ids_by_kind"] == {"figures": [figure.step_id]}

    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
    )
    findings = question_requirement_outcome(
        context=context, plan=draft, run_dir=tmp_path,
        facts=SimpleNamespace(family_template=family_template, question_requirements=[item]),
    )
    (request,) = human_review_requests_for_plan(
        findings=findings, plan=reviewed, require_plan_review=True,
        evidence=SimpleNamespace(root=tmp_path, records=lambda: ()),
    )
    assert request.payload["reason"] == ("operator_plan_approval_required" if include_figure else NOT_COVERED)


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
        ([{**second, "disposition": "covered", "clauses": [
            {**clause, "disposition": "covered"} for clause in second["clauses"]
        ]}, choice], NOT_COVERED),
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


def test_a_wrapped_terminal_receipt_stops_the_repair_loop(tmp_path):
    omitted = host_check("progressive_question_requirement_repair_omitted")
    omitted.easyicu_plan_repair = {"repairs": 1, "limit": 2, "outcome": "repeated_failure"}
    wrapped = RuntimeError("Could not review the synthetic recovery plan")
    wrapped.__cause__ = omitted

    from easyicu.research_agent.orchestration.plan_repair import planning_repair_prompt, run_plan_with_repairs
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check

    calls, notices = [], []

    def run():
        calls.append(planning_repair_prompt())
        raise wrapped

    with pytest.raises(RuntimeError) as error:
        run_plan_with_repairs(run, check_failure=_repairable_plan_check, notify=notices.append)
    assert error.value is wrapped
    assert calls == [""]
    assert notices == []
    assert not hasattr(wrapped, "easyicu_plan_repair")


@pytest.mark.parametrize("chain_link", [None, "__cause__", "__context__"])
def test_terminal_repair_diagnostic_projects_only_safe_receipt_fields(tmp_path, chain_link):
    from easyicu.webserver.agent_pipeline_runs import (
        _pipeline_failure_code,
        _record_pipeline_failure,
        _repairable_plan_check,
    )
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection

    error = host_check("progressive_unknown_variable")
    receipt = {"repairs": 1, "limit": 2, "outcome": "repeated_failure"}
    error.easyicu_plan_repair = {**receipt, "untrusted": "synthetic provider text"}
    if chain_link:
        wrapped = RuntimeError("Could not compile the synthetic study plan")
        setattr(wrapped, chain_link, error)
        error = wrapped
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
    assert gate["detail"]["plan_repair"] == receipt
    assert json.loads((tmp_path / diagnostic).read_text())["plan_repair"] == receipt
    assert public["plan_repair"] == receipt
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


@pytest.mark.parametrize("edit_deliverables", [False, True])
def test_one_web_child_retries_with_the_same_source_and_provider_task(tmp_path, monkeypatch, edit_deliverables):
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
    if edit_deliverables:
        study["plan_deliverables"] = {
            "requested": ["references"], "check": [], "confirmed": True, "source": "user_edit",
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
        assert config.recovery_payload()["bound_plan_deliverables"] == study.get("plan_deliverables")
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


def test_a_real_pipeline_projects_the_terminal_repair_and_only_its_checkpoint(tmp_path, monkeypatch):
    from pathlib import Path

    import pandas as pd

    from easyicu.research_agent import PipelineConfig, PipelineServices, ResearchAgentPipeline
    from easyicu.research_agent.agents import family_spec_planner
    from easyicu.research_agent.orchestration.human_review_checkpoint import checkpoint_path
    from easyicu.research_agent.orchestration.plan_repair import run_plan_with_repairs
    from easyicu.research_agent.orchestration.workflow import HumanReviewPending
    from easyicu.research_agent.planning.requirement_typing import delivery
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check, _write_projection
    from easyicu.webserver.agent_runs import read_run_review
    from easyicu.webserver.pi_copilot.projections import project_run_outcome
    from tests.support.pipeline_contracts import disable_article_contract

    from easyicu.research_agent.orchestration import progressive_planning

    disable_article_contract(monkeypatch)
    plan_progressively = progressive_planning.run_pipeline_progressive_planner

    def plan_without_article_requirements(**kwargs):
        kwargs["planner_kwargs"]["enforce_article_contract"] = False
        return plan_progressively(**kwargs)

    monkeypatch.setattr(progressive_planning, "run_pipeline_progressive_planner", plan_without_article_requirements)
    quote = "Give me a typeset copy of the results"
    question = f"Describe the distribution of hospital mortality by sex. {quote}."
    bind_delivery = delivery.bind_requested_host_delivery

    def omit_delivery_owner(plan, **kwargs):
        bound = bind_delivery(plan, **kwargs)
        assert [step.host_delivery_kind for step in bound.steps if step.host_delivery_kind] == ["manuscript_pdf"]
        return bound.model_copy(update={"steps": bound.execution_steps})

    monkeypatch.setattr(delivery, "bind_requested_host_delivery", omit_delivery_owner)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": range(1, 13), "sex": [0, 1] * 6, "death": [0, 0, 1, 1] * 3,
    }).to_parquet(cohort, index=False)
    client = ScriptedMockLLMClient([])
    build_request = family_spec_planner.build_family_spec_request
    labels = {
        "sex": "Sex documented at admission",
        "sex=0": "Recorded sex category zero",
        "sex=1": "Recorded sex category one",
        "death": "Observed hospital mortality",
    }

    def script_response(*args, **kwargs):
        request = build_request(*args, **kwargs)
        client.responses.append(json.dumps({
            "schema_version": "easyicu.family_plan_spec/1",
            "family_id": request.family_id,
            "request_sha256": request.request_sha256,
            "adjustment_set": [],
            "baseline_variables": [],
            "distribution_missing_policies": {
                "denominator_policy": "all_declared_rows",
                "missing_exposure_policy": "exclude_from_denominator",
                "missing_outcome_policy": "fail_closed",
            },
            "reader_display_labels": [
                {"key": key, "value": labels[key]}
                for key in (*request.required_reader_label_keys, *request.level_label_keys)
            ],
            "comparator_applications": [],
            "roster_decision_note": "Describe documented outcomes for the two source levels.",
            "question_requirements": [{
                "id": "r1", "kind": "delivery", "quote": quote, "coverage": "plan",
                "deliverable_kinds": ["manuscript_pdf"],
            }],
        }))
        return request

    monkeypatch.setattr(family_spec_planner, "build_family_spec_request", script_response)
    workdir = tmp_path / "pipelines"
    config = PipelineConfig(
        workdir=workdir, planner_strategy="family_spec_v1", planner_only=True,
        require_human_plan_review=True, enable_literature=False, enable_memory=False,
    )
    services = PipelineServices(llm=client)
    notices = []

    def run():
        return ResearchAgentPipeline.from_config(config, services=services).run(
            question=question, cohort=cohort, database="miiv",
            target_outcome="death", primary_exposure="sex",
            user_preferences={
                "inferred_analysis_family": "descriptive_epidemiology",
                "covariate_selection": "planner_selectable",
            },
        )

    pending, receipt = run_plan_with_repairs(run, check_failure=_repairable_plan_check, notify=notices.append)
    assert isinstance(pending, HumanReviewPending)
    assert notices == [1]
    assert len(client.calls) == 2
    assert quote in str(client.calls[-1]) and NOT_COVERED in str(client.calls[-1])

    run_dir = Path(pending.run_dir)
    discarded, = set(run_dir.parent.glob("*/" + QUESTION_REQUIREMENTS_REVIEW_FILENAME)) - {
        run_dir / QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    }
    assert not checkpoint_path(discarded.parent).exists()
    assert checkpoint_path(run_dir).is_file()
    assert NOT_COVERED in {request.payload["reason"] for request in pending.requests}
    assert all(request.payload["approval_allowed"] is False for request in pending.requests)

    wrapper = tmp_path / "web-run"
    result = _write_projection(
        wrapper_dir=wrapper, study={"id": "study-synthetic", "question": question},
        provider={"provider": "mock"}, acquisition=None,
        run_dir=run_dir, pending=pending, plan_repair=receipt,
    )
    assert result["human_review_pending"]
    assert result["gate"]["detail"]["plan_repair"] == receipt
    projected = project_run_outcome(read_run_review(str(wrapper)))
    assert projected["run_id"] == pending.run_id
    assert projected["plan_repair"] == {"repairs": 1, "limit": 2, "outcome": "repeated_failure"}
    assert "source_run_manifest.json" in {row["artifact"] for row in projected["artifact_refs"]}


@pytest.mark.parametrize("kind,quote", [
    ("article_preview", "Provide a browser preview"),
    ("manuscript_pdf", "Export a manuscript PDF"),
    ("references", "提供参考文献清单"),
    ("result_tables", "请提供结果表"),
    ("figures", "Include the figures"),
])
def test_a_delivery_requirement_with_no_step_stays_not_covered(kind, quote):
    item = QuestionRequirement(id="r1", kind="delivery", quote=quote, coverage="plan", deliverable_kinds=[kind])
    plan = AnalysisPlan(research_question=quote, steps=[])
    (judged,) = judge_question_requirements(
        (item,), plan=plan, family_template=True, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("no scientific gap was declared"),
    )
    assert judged.disposition == "not_covered" and judged.verified_by_host
    assert judged.delivery is None
    assert judged.row()["deliverable_kinds"] == [kind]


def test_an_unanswered_methods_clause_cannot_disappear_from_its_row(tmp_path):
    quote = "State the units and the observation window for the measurements"
    units = {"quote": "the units", "disposition": "covered"}
    window = {"quote": "the observation window", "disposition": "not_covered"}
    row = requirement(quote, kind="analysis", clauses=[units, window])
    result, _, notices = run_script(
        tmp_path, ([row], NOT_COVERED),
        ([{**row, "disposition": "covered", "clauses": [units]}], "operator_plan_approval_required"),
    )
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value.reason_code == "progressive_question_requirement_repair_omitted"
    assert notices == [1]


@pytest.mark.parametrize("draft_question, user_question, expected", [
    ("Include a PDF alongside the ward summary", "Summarize recovery by ward.", []),
    ("Summarize recovery by ward", "Summarize recovery by ward. Provide a PDF.", ["manuscript_pdf"]),
])
def test_host_delivery_uses_the_users_question_over_a_planner_rephrasing(draft_question, user_question, expected):
    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery

    # Simulate the legacy transport, which can return its own question text.
    plan = AnalysisPlan(research_question=draft_question, steps=[])
    bound = bind_requested_host_delivery(plan, question=user_question)
    assert [step.host_delivery_kind for step in bound.steps] == expected


@pytest.mark.parametrize("quote", ["Export an online preview.", "Please provide a browser preview."])
def test_a_gap_error_can_be_repaired_by_a_verified_delivery_owner(tmp_path, quote):
    from types import SimpleNamespace

    from easyicu.research_agent.agents.family_spec_planner import check_question_requirement_gaps
    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check

    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
    )
    claim = QuestionRequirement(
        id="r1", kind="analysis", quote=quote, coverage="capability_gap",
        gap={"requirement": "design_element_unsupported", "concept": "temp",
             "element": "analysis", "detail": "The draft claims body temperature is unavailable."},
    )
    with pytest.raises(ProgressivePlanCompileError) as caught:
        check_question_requirement_gaps(SimpleNamespace(question_requirements=[claim]), context=context)
    assert caught.value.reason_code == "progressive_capability_gap_claim_unverified"
    fixed = QuestionRequirement(id="r2", kind="delivery", quote=quote, coverage="plan")
    plan = bind_requested_host_delivery(AnalysisPlan(research_question=quote, steps=[]))
    (judged,) = judge_question_requirements(
        [fixed], plan=plan, family_template=True, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("the repaired delivery does not assert a gap"),
    )
    assert judged.disposition == "covered"
    assert judged.delivery["owner_step_ids_by_kind"] == {"article_preview": [plan.steps[0].step_id]}
    reason = "operator_plan_approval_required"
    result, prompts, notices = run_script(
        tmp_path, caught.value, ([judged.row()], reason), ([judged.row()], reason),
        check_failure=_repairable_plan_check,
    )
    assert result() == (reason, None)
    assert '"deliverable_kinds":["article_preview"]' in prompts[1]
    assert notices == [1]


def test_an_unverified_gap_can_gain_clauses_when_the_repair_covers_them(tmp_path):
    quote = "State the sampling interval and the measurement units"
    claim = requirement(quote, kind="analysis", disposition="capability_gap", gap_verification="unverified")
    fixed = requirement(quote, kind="analysis", disposition="covered", clauses=[
        {"quote": "the sampling interval", "disposition": "covered"},
        {"quote": "the measurement units", "disposition": "covered"},
    ])
    result, _, notices = run_script(tmp_path, ([claim], GAP), ([fixed], "operator_plan_approval_required"))
    assert result() == ("operator_plan_approval_required", None)
    assert notices == [1]


@pytest.mark.parametrize("quote, prior_kinds", [
    ("Include body temperature in the online preview.", ()),
    ("Show admission temperature in the browser preview.", ()),
    ("Please provide the browser preview.", ("article_preview",)),
    ("请提供网页文章预览。", ("article_preview",)),
])
def test_a_gap_error_can_be_repaired_by_binding_its_concept(tmp_path, quote, prior_kinds):
    from types import SimpleNamespace

    from easyicu.research_agent.agents.family_spec_planner import check_question_requirement_gaps
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
    from easyicu.webserver.agent_pipeline_runs import _repairable_plan_check

    context = ResearchContext(
        research_question=quote, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
    )
    claim = QuestionRequirement(
        id="r1", kind="analysis", quote=quote, coverage="capability_gap",
        deliverable_kinds=[],
        gap={"requirement": "design_element_unsupported", "concept": "temp",
             "element": "analysis", "detail": "The draft claims body temperature is unavailable."},
    )
    with pytest.raises(ProgressivePlanCompileError) as caught:
        check_question_requirement_gaps(SimpleNamespace(question_requirements=[claim]), context=context)
    assert caught.value.reason_code == "progressive_capability_gap_claim_unverified"
    assert caught.value.easyicu_question_requirement_gap["deliverable_kinds"] == prior_kinds
    fixed = requirement(quote, requirement_id="r2", kind="analysis", concepts=["temp"], disposition="covered")
    approved = "operator_plan_approval_required"
    result, prompts, notices = run_script(
        tmp_path, caught.value, ([fixed], approved), ([fixed], approved), check_failure=_repairable_plan_check,
    )
    assert result() == (approved, None)
    assert ('"deliverable_kinds":["article_preview"]' in prompts[1]) == bool(prior_kinds)
    assert notices == [1]


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("question", [
    "Compare 28-day mortality by lactate, and do not provide a PDF.",
    "Compare overnight readmission by age group, and do not provide a PDF.",
])
def test_a_delivery_exclusion_cannot_answer_bundled_science(tmp_path, question, selected):
    from types import SimpleNamespace

    from easyicu.research_agent.contracts.product_files import PlanDeliverables
    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.planning.requirement_typing.delivery import bind_confirmed_host_delivery
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    context = ResearchContext(
        research_question=question, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
    )
    item = QuestionRequirement(
        id="r1", kind="delivery", quote=question, coverage="plan", deliverable_kinds=["manuscript_pdf"],
    )
    plan = bind_confirmed_host_delivery(
        AnalysisPlan(research_question=question, steps=[]),
        PlanDeliverables(requested=("manuscript_pdf",) if selected else (), confirmed=True, source="user_edit"),
    )
    findings = question_requirement_outcome(
        context=context, plan=plan,
        facts=SimpleNamespace(family_template=True, question_requirements=[item]), run_dir=tmp_path,
    )
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    assert record["coverage"]["complete"] is False
    assert record["judged"][0]["disposition"] == "not_covered"
    assert any(finding.detail.get("reason") == NOT_COVERED for finding in findings)


@pytest.mark.parametrize("disposition", ["covered", "not_requested"])
def test_repair_cannot_relabel_an_unverified_scientific_gap_as_delivery(tmp_path, disposition):
    quote = "Compare spline-adjusted overnight readmission"
    prior = requirement(quote, kind="analysis", disposition="capability_gap",
                        verified_by_host=False, gap_verification="unverified")
    disguised = requirement(quote, kind="delivery", deliverable_kinds=["manuscript_pdf"], disposition=disposition, verified_by_host=True)
    result, _, notices = run_script(
        tmp_path, ([prior], NOT_COVERED), ([disguised], "operator_plan_approval_required"),
    )
    with pytest.raises(ProgressivePlanCompileError) as error:
        result()
    assert error.value.reason_code == "progressive_question_requirement_repair_omitted"
    assert notices == [1]


@pytest.mark.parametrize("delivery_quote, kinds, disposition", [
    ("Deliver result tables or figures", ["result_tables", "figures"], "needs_clarification"),
    ("Export a PDF or include an online preview", ["manuscript_pdf", "article_preview"], "needs_clarification"),
])
def test_original_question_survives_plan_paraphrase_and_approval_refuses_clarification(
    tmp_path, delivery_quote, kinds, disposition,
):
    from types import SimpleNamespace

    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery

    question = f"Compare morning recovery. {delivery_quote}."
    context = ResearchContext(
        research_question=question, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
    )
    item = QuestionRequirement(
        id="r1", kind="delivery", quote=delivery_quote, coverage="plan", deliverable_kinds=kinds,
    )
    plan = bind_requested_host_delivery(AnalysisPlan(research_question="Compare morning recovery.", steps=[]), question=question)
    findings = question_requirement_outcome(
        context=context, plan=plan,
        facts=SimpleNamespace(family_template=True, question_requirements=[item]), run_dir=tmp_path,
    )
    requests = human_review_requests_for_plan(
        findings=findings, plan=plan, evidence=SimpleNamespace(root=tmp_path, records=lambda: ()),
        require_plan_review=True,
    )
    assert requests
    assert all(request.payload["approval_allowed"] is False for request in requests)
    record_path = tmp_path / QUESTION_REQUIREMENTS_FILENAME
    assert json.loads(record_path.read_text())["question"] == question
    reviewed = json.loads((tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).read_text())
    assert reviewed["question"] == question
    assert reviewed["judged"][0]["disposition"] == disposition

    unedited = plan.model_copy(update={"research_question": question})
    question_requirement_outcome(
        context=context, plan=unedited,
        facts=SimpleNamespace(family_template=True, question_requirements=[item]), run_dir=tmp_path,
    )
    assert json.loads(record_path.read_text())["question"] == question


@pytest.mark.parametrize("kind", ["article_preview", "manuscript_pdf", "references"])
def test_a_confirmed_selection_without_its_owner_blocks_approval_even_without_a_planner_row(kind):
    from easyicu.research_agent.contracts.product_files import PlanDeliverables
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan

    plan = AnalysisPlan(
        research_question="Summarize the duration of overnight observation.",
        steps=[], deliverables=PlanDeliverables(requested=(kind,), confirmed=True, source="user_edit"),
    )
    (request,) = human_review_requests_for_plan(findings=[], plan=plan, require_plan_review=True)
    assert request.payload["reason"] == NOT_COVERED
    assert request.payload["approval_allowed"] is False
    assert request.payload["deliverable_kinds"] == [kind]
