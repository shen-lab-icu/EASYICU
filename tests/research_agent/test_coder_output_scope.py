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


def test_coder_context_exposes_registered_source_concept_metadata(ra):
    llm = _RecordingLLM()
    context = ra.ResearchContext(
        research_question="Describe a binary intervention.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="treatment_first",
                description="Registered treatment event indicator",
                role="intervention",
                dtype="float64",
                source_concept="treatment_event",
                observed_domain={
                    "n_unique": 1,
                    "is_constant": True,
                    "is_binary": True,
                    "min": 1.0,
                    "max": 1.0,
                },
            )
        ],
    )
    step = ra.AnalysisStep(
        step_id="event_definition",
        intent="Construct the selected binary event exposure.",
        inputs=["treatment_first"],
        expected_outputs=["table:event_definition"],
        method="prespecified_binary_event_definition",
    )

    CoderAgent(llm).run(context=context, step=step)

    prompt = llm.messages[-1].content
    assert "source_concept=treatment_event" in prompt
    assert "description='Registered treatment event indicator'" in prompt
    assert "role=intervention" in prompt
    assert "observed=CONSTANT(single value; no variation to model)" in prompt


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
    assert "do not add a second missingness/binary filter" in prompt
    assert "permitted missingness in the raw representative column" in prompt
    assert "Never publish a completed exposure artefact" in prompt
    assert "bind the selected base `source_concept`" in prompt
    assert "explicit event/indicator metadata" in prompt
    assert "record that binding" in prompt
    assert "Never hard-code `indicator_semantics`" in prompt


def test_coder_sparse_event_repair_surfaces_referenced_context_metadata(ra):
    llm = _RecordingLLM()
    context = ra.ResearchContext(
        research_question="Define the selected treatment event.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="treatment_first",
                description="Registered treatment event indicator",
                role="intervention",
                dtype="float64",
                source_concept="treatment_event",
                observed_domain={"is_binary": True, "n_unique": 1},
            ),
            ra.ConceptDescriptor(
                name="unreferenced_first",
                description="Another event indicator",
                role="intervention",
                dtype="float64",
                source_concept="other_event",
            ),
        ],
    )
    step = ra.AnalysisStep(
        step_id="event_definition",
        intent="Construct the Agent-selected binary event exposure.",
        inputs=["treatment_first"],
        expected_outputs=["table:event_definition"],
        method="prespecified_binary_event_definition",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="value = frame['treatment_first']\n",
        run_log="Binary event presence lacks representative value reconciliation.",
    )

    prompt = llm.messages[-1].content
    assert "Authoritative ResearchContext metadata" in prompt
    metadata_line = next(
        line
        for line in prompt.splitlines()
        if "Authoritative ResearchContext metadata" in line
    )
    assert '"name": "treatment_first"' in metadata_line
    assert '"source_concept": "treatment_event"' in metadata_line
    assert '"role": "intervention"' in metadata_line
    assert '"description": "Registered treatment event indicator"' in metadata_line
    assert "unreferenced_first" not in metadata_line


def test_coder_repair_preserves_standard_helper_across_later_traceback(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="event_definition",
        intent="Construct the Agent-selected binary event exposure.",
        inputs=["event_n", "event_measured", "event_max"],
        expected_outputs=["table:event_definition"],
        method="prespecified_binary_event_definition",
    )
    code = (
        "from easyicu.research_agent.methods.source_status import "
        "reconcile_binary_event_presence\n"
        "result = reconcile_binary_event_presence(frame, "
        "count_column='event_n', measured_column='event_measured', "
        "representative_column='event_max')\n"
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code=code,
        run_log="TypeError: cannot convert the series to int",
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED SPARSE-EVENT REPAIR (binding)" in prompt
    assert "Do not replace those columns" in prompt
    assert "`BinaryEventPresenceResult` dataclass, NOT a dictionary" in prompt
    assert "`helper_result.values`" in prompt
    assert "never require `isinstance(helper_result, dict)`" in prompt
    assert "Its three column arguments are keyword-only" in prompt
    assert "count_column=count_col" in prompt


def test_coder_repair_separates_provenance_audit_from_value_selection(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="table_one",
        intent="Describe the locked cohort.",
        inputs=["gcs_first", "gcs_measured", "gcs_n"],
        expected_outputs=["table:table_one"],
        method="descriptive_statistics",
    )
    code = "valid = frame['gcs_first'].notna() & frame['gcs_measured'].eq(1)\n"

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code=code,
        run_log="GCS reporting is incorrectly gated on measured and count columns being present.",
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED PROVENANCE/VALUE-SELECTION REPAIR" in prompt
    assert "sole basis for its descriptive non-missing denominator" in prompt
    assert "Do not require either companion" in prompt
    assert "fail the entire completed step" in prompt


def test_coder_repair_fail_closes_nonterminating_provenance_audit(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="adjusted_model",
        intent="Fit the planner-owned adjusted model.",
        inputs=["artifact:quality_checked_analysis_data"],
        expected_outputs=["table:adjusted_model"],
        method="regression",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="audit = provenance_audit(raw_frame)\nmodel.fit(analysis_frame)\n",
        run_log=(
            "A measurement-provenance audit records invalid or discordant pairs "
            "but does not fail the completed step before outputs can be published."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED PROVENANCE/VALUE-SELECTION REPAIR" in prompt
    assert "same authoritative typed working frame used by the model" in prompt
    assert "instead of a hard-coded column list" in prompt
    assert "raise before model fitting or output registration" in prompt
    assert "treat an empty checks collection as failure" in prompt
    assert "Do not rely only on an `any(...)` or `all(...)` generator" in prompt


def test_coder_repair_requires_bidirectional_provenance_pairs(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Audit the planner-owned working data.",
        inputs=["artifact:quality_checked_analysis_data"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="measured = [c for c in frame if c.endswith('_measured')]\n",
        run_log=(
            "DETAIL: {\"reason\": "
            "\"provenance_pair_scan_not_bidirectional\"}"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED PROVENANCE/VALUE-SELECTION REPAIR" in prompt
    assert "from both `*_measured` and `*_n` columns" in prompt
    assert "never scan in only one direction" in prompt


def test_coder_repair_consumes_authoritative_exposure_definition(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run diagnostics for the planner-owned exposure.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code=(
            "exposure_definition = typed.get("
            "'artifact:primary_exposure_definition')\n"
            "exposure_col = 'candidate_event_max'\n"
        ),
        run_log=(
            "DETAIL: {\"reason\": \"authoritative_primary_exposure_unused\"}"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED AUTHORITATIVE-EXPOSURE BINDING REPAIR" in prompt
    assert "Reuse the script's existing typed definition resolver" in prompt
    assert "do not leave that resolver unused" in prompt
    assert "Do not substitute a hard-coded column" in prompt


def test_coder_repair_replaces_undefined_helper_without_stub(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Render planner-owned diagnostics.",
        inputs=["table:diagnostics"],
        expected_outputs=["figure:diagnostics"],
        method="visualization",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="result = missing_renderer(source)\n",
        run_log=(
            "DETAIL: {\"reason\": \"undefined_helper_call\", "
            "\"calls\": [{\"name\": \"missing_renderer\", \"line\": 1}]}"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED UNDEFINED-HELPER REPAIR" in prompt
    assert "Prefer calling an already defined equivalent helper" in prompt
    assert "Never insert a stub, no-op" in prompt


def test_coder_repair_binds_finalized_tabular_exposure_product(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run diagnostics for the planner-owned exposure.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="definition = typed['artifact:primary_exposure_definition']\n",
        run_log=(
            "RuntimeError: primary_exposure_definition has no registered "
            "executable exposure column available in the analysis data"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt
    assert "row-aligned finalized exposure table" in prompt
    assert "exact planner-selected `ResearchContext.primary_exposure`" in prompt
    assert 'fact is: "selected_first"' in prompt
    assert "Do not repeat raw-event reconciliation" in prompt
    assert "Before any integer/boolean cast" in prompt
    assert "non-missing, finite, and exactly in {0, 1}" in prompt
    assert "Verify row alignment using the artifact's stable row key" in prompt
    assert "retain the separate bidirectional count/measured provenance audit" in prompt


def test_coder_repair_validates_finalized_binary_exposure_before_cast(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run diagnostics for the planner-owned exposure.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="treatment = exposure_table['selected_first'].astype(int)\n",
        run_log=(
            "Row-aligned exposure-table branch bypasses binary-event validation "
            "and provenance reconciliation."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt
    assert "never let a fractional value be truncated" in prompt
    assert "without redefining it" in prompt


def test_coder_repair_separates_finalized_table_from_raw_event_definition(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run diagnostics for the planner-owned exposure.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="helper_result = reconcile_binary_event_presence(frame)\n",
        run_log=(
            "The finalized exposure-table branch bypasses registered exposure "
            "metadata validation before applying the sparse binary-event exception."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt
    assert "Do not fabricate source-concept, role, indicator-semantics" in prompt
    assert "do not invoke the sparse binary-event reconciliation" in prompt
    assert "Only a separate raw-definition mapping branch" in prompt


def test_coder_repair_completes_declared_assignment_product(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="assignment",
        intent="Fit the Planner-owned assignment model.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["artifact:assignment_model"],
        method="confounder_selection_and_propensity_model",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="assignment_models = []\n",
        run_log=(
            "DETAIL: {\"kind\": \"assignment_model_unfitted\"}; "
            "assignment model artifact but registered no successfully fitted "
            "assignment model"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED ASSIGNMENT-PRODUCT COMPLETION REPAIR" in prompt
    assert "empty/all-missing table" in prompt
    assert "Do not invent a substitute exposure" in prompt
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt


def test_coder_repair_consumes_registered_assignment_product_columns(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="balance",
        intent="Diagnose positivity for the Planner-owned assignment models.",
        inputs=["artifact:assignment_model"],
        expected_outputs=["table:balance"],
        method="positivity_and_balance_diagnostics",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="raise RuntimeError('missing propensity')\n",
        run_log="Registered propensity-score column is unavailable",
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED ASSIGNMENT-PRODUCT BINDING REPAIR" in prompt
    assert "product_contract" in prompt
    assert "exact propensity_score_column" in prompt
    assert "do not scan arbitrary numeric columns" in prompt
    assert "do not" in prompt and "refit an assignment model" in prompt


def test_coder_repair_preserves_typed_dataframe_artifact(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="assignment",
        intent="Fit the Planner-owned assignment model.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["artifact:assignment_model"],
        method="confounder_selection_and_propensity_model",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="exposure_definition = {}\n",
        run_log=(
            "The finalized primary exposure artifact is discarded before "
            "exposure resolution."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt
    assert "never coerce a DataFrame to `{}`, `[]`, `None`, or text" in prompt


def test_coder_repair_removes_constructed_exposure_fallback(ra):
    llm = _RecordingLLM()
    context = _context(ra).model_copy(update={"primary_exposure": "selected_first"})
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run diagnostics for the planner-owned exposure.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )

    CoderAgent(llm).repair(
        context=context,
        step=step,
        code="try:\n    resolved = bind(definition)\nexcept:\n    resolved = {}\n",
        run_log=(
            "DETAIL: {\"reason\": "
            "\"authoritative_primary_exposure_fallback\"}"
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED AUTHORITATIVE-EXPOSURE BINDING REPAIR" in prompt
    assert "Do not catch a binding failure and construct replacement" in prompt
    assert "DIAGNOSED TABULAR AUTHORITATIVE-EXPOSURE REPAIR" in prompt
    assert "fail closed if the exact planner-selected column is absent" in prompt


def test_coder_repair_removes_untraceable_figure_audit_columns(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="descriptive_figure",
        intent="Render the declared figure from upstream values.",
        inputs=["table:descriptive"],
        expected_outputs=["figure:descriptive"],
        method="visualization",
    )
    code = "source_data['count_integer'] = source_data['count'].mod(1).eq(0)\n"

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code=code,
        run_log=(
            "These source-data value columns were not verified against any "
            "row-aligned upstream value vector: ['count_integer']; one verified "
            "column cannot authenticate another renamed or transformed value."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED FIGURE SOURCE-DATA TRACE REPAIR" in prompt
    assert "remove unplotted derived numeric/boolean audit fields" in prompt
    assert "Keep such checks internal" in prompt


def test_coder_repair_fail_closes_partial_structural_accounting_figure(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="cohort_accounting_figure",
        intent="Render the declared cohort-accounting figure.",
        inputs=["table:cohort_accounting"],
        expected_outputs=["figure:cohort_accounting"],
        method="visualization",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="valid_rows = frame.loc[valid_mask].copy()\n",
        run_log=(
            "Renders a partial cohort flow after excluding invalid source rows."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED STRUCTURAL-ACCOUNTING FIGURE REPAIR" in prompt
    assert "Validate every required label, count, denominator" in prompt
    assert "keep figure_files empty" in prompt
    assert "do not change the cohort" in prompt


def test_coder_repair_removes_arbitrary_figure_column_fallback(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="registered_table_figure",
        intent="Render the declared table without changing its schema.",
        inputs=["table:registered_summary"],
        expected_outputs=["figure:registered_summary"],
        method="visualization",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="count_col = next(c for c in frame if is_numeric(frame[c]))\n",
        run_log=(
            "Column discovery can silently bind the figure to an unintended "
            "numeric column. The helper falls back to the first frame column "
            "with any numeric value when no named count candidate is present."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED FIGURE SCHEMA-BINDING REPAIR" in prompt
    assert "remove every fallback that chooses the first numeric column" in prompt
    assert "explicit semantic candidate names" in prompt
    assert "keep figure_files empty" in prompt


def test_coder_repair_preserves_ordinal_covariate_without_linearizing_it(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="assignment_model",
        intent="Fit the prespecified treatment assignment model.",
        inputs=["artifact:analysis_data"],
        expected_outputs=["table:assignment_scores"],
        method="propensity_score_logistic_regression",
    )

    CoderAgent(llm).repair(
        context=_context(ra),
        step=step,
        code="import pandas as pd\n",
        run_log=(
            "The script passes an ordinal score as a numeric covariate, "
            "imposing a continuous linear effect on an ordinal variable."
        ),
    )

    prompt = llm.messages[-1].content
    assert "DIAGNOSED ORDINAL-COVARIATE REPAIR (binding)" in prompt
    assert "preserve the Agent-selected covariate" in prompt
    assert "observed ordered levels explicitly" in prompt
    assert "Record the chosen encoding and reference" in prompt
    assert "Do not drop the covariate" in prompt


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
        assert "manifest['planner_declared_inputs']" in prompt
        assert "exact Planner-owned consumer scope" in prompt
        assert "producer product's semantics" in prompt
        assert "only eligible raw-variable or column coordinates" in prompt
        assert "Do not discover them by scanning the full ResearchContext" in prompt
        assert "manifest['context']" in prompt
        assert "immutable Agent-produced ResearchContext" in prompt
        assert "do not copy prompt literals" in prompt
        assert "product_contract" in prompt
        assert "successful producer's step summary" in prompt
        assert "do not recover them from DataFrame.attrs" in prompt
        assert "Do not glob EASYICU_EVIDENCE_DIR" in prompt
        assert "reconstruct a declared upstream product" in prompt
        assert "one input_bindings row per typed input" in prompt
        assert "for each loaded tabular input, its row_count" in prompt
        assert "every shared non-key column" in prompt
        assert "The host repeats that key-and-value comparison" in prompt


def test_coder_prompts_bind_untyped_only_inputs_to_planner_scope(ra):
    step = ra.AnalysisStep(
        step_id="consume_raw_columns",
        intent="Validate the Planner-declared raw inputs.",
        inputs=["selected_first", "selected_measured"],
        expected_outputs=["table:result"],
        method="descriptive_summary",
    )

    prompts = _ordinary_run_repair_and_agentic_prompts(ra=ra, step=step)
    for prompt in prompts:
        assert "TYPED INPUT BINDING (binding)" in prompt
        assert "applies even when the step declares only untyped" in prompt
        assert "manifest['planner_declared_inputs']" in prompt
        assert "['selected_first', 'selected_measured']" in prompt
        assert "Exact typed inputs for this step: []" in prompt


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
