from __future__ import annotations

from easyicu.research_agent.agentic_coder import AgenticCoderAgent
from easyicu.research_agent.agents import CoderAgent
from easyicu.research_agent.llm import LLMMessage
from easyicu.research_agent.plan_utils import effect_output_authorized
from easyicu.research_agent.schema import PlannedModelRequirement


class _RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[LLMMessage] = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.messages = list(messages)
        return "import os\n"


def _context(ra):
    return ra.ResearchContext(
        research_question="Describe an ICU cohort.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[],
    )


def test_coder_prompt_forbids_figure_when_not_declared(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="summary",
        intent="Compute a summary table.",
        inputs=["x"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )

    CoderAgent(llm).run(context=_context(ra), step=step)

    prompt = llm.messages[-1].content
    assert "DECLARED OUTPUT SCOPE (binding)" in prompt
    assert "declares no figure product" in prompt
    assert "Do not render, save, or register figures" in prompt


def test_coder_prompt_allows_only_declared_figure_products(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="render",
        intent="Render the declared figure.",
        inputs=["artifact:summary"],
        expected_outputs=["figure:summary"],
        method="descriptive_summary",
    )

    CoderAgent(llm).run(context=_context(ra), step=step)

    prompt = llm.messages[-1].content
    assert "Figure rendering is allowed only for the explicitly declared" in prompt
    assert "declares no figure product" not in prompt


def test_coder_repair_requires_standard_helper_after_sparse_event_diagnosis(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="event_definition",
        intent="Construct the Agent-selected binary event exposure.",
        inputs=["event_n", "event_measured", "event_max"],
        expected_outputs=["table:event_definition"],
        method="prespecified_binary_event_definition",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="import pandas as pd\n",
        run_log=(
            "Binary event reconciliation accepts representative value 0 on "
            "reconciled positive rows."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED SPARSE-EVENT REPAIR (binding)" in prompt
    assert "methods.source_status.reconcile_binary_event_presence" in prompt
    assert "Do not replace those columns" in prompt


def _ordinary_run_repair_and_agentic_prompts(*, ra, step):  # noqa: ANN001
    context = _context(ra)
    llm = _RecordingLLM()
    coder = CoderAgent(llm)
    coder.run(context=context, step=step)
    run_prompt = llm.messages[-1].content
    coder.repair(
        context=context,
        step=step,
        code="import os\n",
        run_log="synthetic failure",
        attempt=1,
    )
    repair_prompt = llm.messages[-1].content
    agentic_prompt = AgenticCoderAgent(coder)._build_prompt(context, step)
    return run_prompt, repair_prompt, agentic_prompt


def test_all_coder_paths_fail_close_effect_scope_for_non_effect_owner(ra):
    step = ra.AnalysisStep(
        step_id="summary",
        intent="Describe the cohort without fitting an effect model.",
        inputs=["x"],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )

    assert effect_output_authorized(step) is False
    prompts = _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step)
    for prompt in prompts:
        assert "effect_output_authorized: false" in prompt
        assert "reference-group contrasts" in prompt
        assert "nested step_summary fields" in prompt
        assert "p-values for any such undeclared effect contrast or interaction" in prompt
        assert "Descriptive counts, denominators, rates, absolute summaries" in prompt
        assert "inferred analysis family is context only" in prompt
    run_prompt, repair_prompt, _agentic_prompt = prompts
    assert "Pick methods and figures this family calls for" not in run_prompt
    assert "family label does not authorize another method" in run_prompt
    assert "family label cannot add or replace a scientific product" in repair_prompt


def test_all_coder_paths_keep_declared_effect_owner_authorized(ra):
    step = ra.AnalysisStep(
        step_id="primary_effect",
        intent="Estimate the declared adjusted association.",
        inputs=["exposure", "outcome"],
        expected_outputs=["statistic:adjusted_or"],
        method="adjusted_logistic_regression with prespecified covariates",
    )

    assert effect_output_authorized(step) is True
    prompts = _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step)
    for prompt in prompts:
        assert "effect_output_authorized: true" in prompt
        assert "Effect authorization does not widen scope" in prompt
        assert "effect_output_authorized=false" not in prompt


def test_effect_capable_method_without_effect_product_remains_fail_closed(ra):
    for expected_output in (
        "table:cohort_summary",
        "table:adjusted_association_input_audit",
        "figure:primary_effect",
        "log:odds_ratio",
    ):
        step = ra.AnalysisStep(
            step_id="model_diagnostics",
            intent="Audit a prespecified model input table.",
            inputs=["exposure", "outcome"],
            expected_outputs=[expected_output],
            method="adjusted_logistic_regression",
        )

        assert effect_output_authorized(step) is False, expected_output
        for prompt in _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step):
            assert "effect_output_authorized: false" in prompt


def test_closed_adjusted_association_alias_is_an_effect_product(ra):
    step = ra.AnalysisStep(
        step_id="primary_effect",
        intent="Estimate the declared adjusted association.",
        inputs=["exposure", "outcome"],
        expected_outputs=["table:primary_adjusted_association"],
        method="logistic_regression",
    )

    assert effect_output_authorized(step) is True


def test_typed_model_requirement_roster_also_authorizes_effect_output(ra):
    requirement = PlannedModelRequirement(
        requirement_id="primary_source_aware",
        outcome="mortality",
        outcome_type="binary",
        method_family="logistic_regression",
        exposure_source="primary_measurement",
        analysis_role="primary",
        analysis_set="source_aware",
        required_for_step_success=True,
    )
    step = ra.AnalysisStep(
        step_id="adjusted_models",
        intent="Fit the planner-owned adjusted model roster.",
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        model_requirements=[requirement],
    )

    assert effect_output_authorized(step) is True
    assert "effect_output_authorized: true" in _ordinary_run_repair_and_agentic_prompts(
        ra=ra,
        step=step,
    )[0]


def test_prespecified_robustness_refit_prompt_has_effect_authority(ra):
    step = ra.AnalysisStep(
        step_id="locked_robustness_refits",
        intent="Refit the primary estimand across planner-locked specifications.",
        expected_outputs=[
            "table:robustness_grid",
            "table:sensitivity_specification_matrix",
        ],
        method="prespecified_robustness_analysis",
    )

    assert effect_output_authorized(step) is True
    for prompt in _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step):
        assert "effect_output_authorized: true" in prompt


def test_coder_repair_prompt_forbids_helper_result_name_shadowing(ra):
    step = ra.AnalysisStep(
        step_id="repair_runtime_failure",
        intent="Repair a generated analysis without changing its method.",
        expected_outputs=["table:result"],
        method="descriptive_summary",
    )

    _run_prompt, repair_prompt, _agentic_prompt = (
        _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step)
    )

    assert "same name as a helper function called in that scope" in repair_prompt
    assert "never write `audit = audit(...)`" in repair_prompt
    assert "UnboundLocalError" in repair_prompt
    assert "Use a distinct result name" in repair_prompt


def test_coder_prompt_binds_typed_inputs_to_resolved_manifest(ra):
    step = ra.AnalysisStep(
        step_id="consume",
        intent="Consume the declared upstream table.",
        inputs=["table:scaling_summary"],
        expected_outputs=["table:result"],
        method="descriptive_summary",
    )

    prompts = _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step)
    for prompt in prompts:
        assert "TYPED INPUT BINDING (binding)" in prompt
        assert "EASYICU_RESOLVED_INPUTS_JSON" in prompt
        assert "Do not glob EASYICU_EVIDENCE_DIR" in prompt
        assert "reconstruct a declared upstream product" in prompt
        assert "one input_bindings row per typed input" in prompt
        assert "for each loaded tabular input, its row_count" in prompt
        assert "every shared non-key column" in prompt
        assert "The host repeats that key-and-value comparison" in prompt


def test_runtime_only_builds_visualization_request_for_figure_step(ra):
    context = _context(ra)
    supervisor = ra.RuntimeSupervisor()
    state = supervisor.bootstrap_state(run_id="run", context=context)
    table_step = ra.AnalysisStep(
        step_id="summary",
        intent="Compute a summary table.",
        expected_outputs=["table:summary"],
    )
    figure_step = ra.AnalysisStep(
        step_id="render",
        intent="Render the summary.",
        expected_outputs=["figure:summary"],
    )

    table_state = supervisor.prepare_step_state(
        state=state, context=context, step=table_step, evidence_refs=[]
    )
    figure_state = supervisor.prepare_step_state(
        state=state, context=context, step=figure_step, evidence_refs=[]
    )

    assert table_state.visualization_request is None
    assert figure_state.visualization_request is not None
