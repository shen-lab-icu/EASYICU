from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.core import (
    AnalyzerAgent,
    ReportingPromptBudgetError,
    WriterAgent,
)
from easyicu.research_agent.agents.reporting import (
    _project_writer_evidence_digest,
    _project_writer_literature_digest,
    _group_writer_numeric_citations,
)
from easyicu.research_agent.research_context.outbound import format_outbound_safe_context
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate the association between exposure and death.",
        cohort=CohortDescriptor(
            cohort_name="reporting_scope",
            database="miiv",
            n_patients=100,
            n_stays=100,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(
                name="stay_id",
                dtype="object",
                role=VariableRole.ID,
            ),
            ConceptDescriptor(
                name="exposure",
                dtype="float64",
                role=VariableRole.INTERVENTION,
            ),
            ConceptDescriptor(
                name="death",
                dtype="int64",
                role=VariableRole.OUTCOME,
            ),
        ],
        primary_exposure="exposure",
        target_outcome="death",
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="01_primary",
        intent="Estimate the prespecified association.",
        inputs=["exposure", "death"],
        expected_outputs=["table:adjusted_estimates"],
        method="adjusted_association_models",
    )


def test_analyzer_oversize_fails_before_provider_call() -> None:
    llm = PatternScriptedMockLLMClient([], default="unused")

    with pytest.raises(ReportingPromptBudgetError, match="Analyzer"):
        AnalyzerAgent(llm).run(
            context=_context(),
            step=_step(),
            step_summary={"estimate": 1.0},
            evidence_ids=["e" * 50_000],
        )

    assert llm.calls == []


@pytest.mark.parametrize("text", ["", "  ", "(analyzer failed: invalid response)"])
def test_analyzer_rejects_empty_or_failed_completion(text):
    llm = PatternScriptedMockLLMClient([], default=text)
    with pytest.raises(ValueError, match="empty or failed interpretation"):
        AnalyzerAgent(llm).run(
            context=_context(), step=_step(), step_summary={"estimate": 1.0},
            evidence_ids=["result"],
        )
    assert len(llm.calls) == 1


def test_writer_oversize_fails_before_provider_call() -> None:
    llm = PatternScriptedMockLLMClient([], default="unused")

    with pytest.raises(ReportingPromptBudgetError, match="Writer"):
        WriterAgent(llm)._call_section(
            section_name="Results",
            instruction="Write one evidence-bound sentence.",
            context=_context(),
            evidence_ids=["primary_result"],
            evidence_digest="x" * 70_000,
        )

    assert llm.calls == []


def test_writer_repair_reserves_feedback_without_shortening_original_inputs() -> None:
    from easyicu.research_agent.agents.core import _coder_prompt_payload_bytes

    llm = PatternScriptedMockLLMClient([], default="## Results\n\nComplete.")
    writer = WriterAgent(llm)
    kwargs = dict(section_name="Results", instruction="Write the complete section.",
                  context=_context(), evidence_ids=["primary_result"])
    marker = "Exact source fact and reference: {evidence:primary_result}"
    writer._call_section(**kwargs, evidence_digest=marker)
    base_bytes = _coder_prompt_payload_bytes(llm.calls[-1][0])
    digest = marker + "x" * (64_000 - base_bytes - 128)
    feedback = "Fix the rejected section without changing the source facts. " * 40
    writer._call_section(**kwargs, evidence_digest=digest, repair_feedback=feedback)
    messages = llm.calls[-1][0]
    assert digest in messages[1].content
    assert messages[-1].content == feedback
    assert 64_000 < _coder_prompt_payload_bytes(messages) <= 72_000

    before = len(llm.calls)
    with pytest.raises(ReportingPromptBudgetError, match="repair feedback"):
        writer._call_section(**kwargs, evidence_digest=marker, repair_feedback="x" * 8_001)
    assert len(llm.calls) == before


def test_writer_non_result_section_uses_role_scoped_evidence_projection() -> None:
    digest = (
        "RUN_CONTEXT\n"
        + "x" * 20_000
        + "\n## EXECUTED METHOD BOUNDARY\n"
        + "methods-only" * 300
        + "\n## host-authorized scientific claims\n{claim:step.result}\n"
        + "\n## numeric citation authority\n{evidence:primary_result}\n"
        + "\n## secondary numbers\n"
        + "y" * 70_000
    )
    projected = _project_writer_evidence_digest("Methods", digest)

    assert "RUN_CONTEXT" not in projected
    assert "x" * 20_000 not in projected
    assert "## EXECUTED METHOD BOUNDARY" in projected
    assert "methods-only" in projected
    assert "{claim:step.result}" in projected
    assert "## numeric citation authority" in projected
    assert "## secondary numbers" not in projected
    results_projection = _project_writer_evidence_digest("Results", digest)
    assert "RUN_CONTEXT" not in results_projection
    assert "x" * 20_000 not in results_projection
    assert "## EXECUTED METHOD BOUNDARY" in results_projection
    assert "methods-only" in results_projection
    assert "{claim:step.result}" in results_projection
    assert "## numeric citation authority" in results_projection
    assert "## secondary numbers" in results_projection

    llm = PatternScriptedMockLLMClient([], default="## Methods\n\nComplete.")
    WriterAgent(llm)._call_section(
        section_name="Methods",
        instruction="Write one evidence-bound sentence.",
        context=_context(),
        evidence_ids=["primary_result"],
        evidence_digest=digest,
    )

    assert len(llm.calls) == 1
    assert "RUN_CONTEXT" not in llm.calls[0][0][1].content
    assert "## secondary numbers" not in llm.calls[0][0][1].content


def test_writer_projection_removes_only_redundant_uncited_preamble() -> None:
    digest = (
        "RUN_CONTEXT\n"
        + "redundant study coordinates " * 50
        + "\n## EXECUTED METHOD BOUNDARY\n"
        + "Writer instruction: cite the supplied owner.\n"
        + "Preserve the fitted model. {claim:step.result}\n"
        + "## numeric citation authority\n"
        + "odds_ratio=1.24; cite={evidence:primary_result}\n"
        + "## secondary numbers\n"
        + "sensitivity_odds_ratio=1.18; cite={evidence:sensitivity_result}\n"
    )

    projected = _project_writer_evidence_digest("Results", digest)

    assert "RUN_CONTEXT" not in projected
    assert "redundant study coordinates" not in projected
    assert "Writer instruction:" not in projected
    assert projected.startswith("## EXECUTED METHOD BOUNDARY")
    assert "{claim:step.result}" in projected
    assert "{evidence:primary_result}" in projected
    assert "{evidence:sensitivity_result}" in projected


def test_writer_results_excludes_prior_study_digest() -> None:
    literature_digest = (
        "direct_comparator: [@prior_study]\n"
        "method:functional_form: [@spline_method]\n"
    )

    assert _project_writer_literature_digest("Results", literature_digest) == ""
    assert (
        _project_writer_literature_digest("Introduction", literature_digest)
        == literature_digest
    )
    assert (
        _project_writer_literature_digest("Methods", literature_digest)
        == literature_digest
    )
    assert (
        _project_writer_literature_digest("Discussion", literature_digest)
        == literature_digest
    )

    llm = PatternScriptedMockLLMClient([], default="## Results\n\nComplete.")
    WriterAgent(llm)._call_section(
        section_name="Results",
        instruction="Write one evidence-bound sentence.",
        context=_context(),
        evidence_ids=["primary_result"],
        evidence_digest="{evidence:primary_result}",
        literature_digest=literature_digest,
    )

    assert "[@prior_study]" not in llm.calls[0][0][1].content
    assert "(none required for this section)" in llm.calls[0][0][1].content


def test_numeric_citation_grouping_preserves_values_and_owner_boundaries() -> None:
    digest = (
        "## secondary numbers\n- primary\n"
        "  n=231 (canonical=231.0); cite={evidence:primary}\n"
        "  risk=12.3 (canonical=12.345); cite={evidence:primary}\n"
        "  p=0.04 (canonical=0.04); cite={evidence:sensitivity}\n"
        "- alternative\n"
        "  n=209 (canonical=209.0); cite={evidence:primary}\n"
        "  note without numeric authority\n"
    )
    projected = _group_writer_numeric_citations(digest)
    assert "Citation for every value in this block: {evidence:primary}\n" in projected
    assert "    n=231 (canonical=231.0)\n" in projected
    assert "    risk=12.3 (canonical=12.345)\n" in projected
    assert "p=0.04 (canonical=0.04); cite={evidence:sensitivity}" in projected
    assert "- alternative\n  n=209 (canonical=209.0); cite={evidence:primary}" in projected
    assert projected.endswith("  note without numeric authority\n")


def test_writer_context_preserves_definitions_without_source_profile_numbers() -> None:
    from easyicu.research_agent.schema import MissingnessProfile

    context = _context()
    variable = context.variables[1]
    context.variables[2].description = "Death recorded during the index hospitalization."
    variable.unit = "mmol/L"
    variable.analysis_window = "first 24 hours after ICU admission"
    variable.missingness_semantics = "Not measured differs from structural non-applicability."
    variable.observed_domain = {"shape": "continuous", "n_unique": 61}
    variable.missingness = MissingnessProfile(
        fraction_missing=0.2, n_missing=20, n_total=100,
    )
    before = context.model_dump_json()
    full = json.loads(format_outbound_safe_context(context))
    projected = json.loads(format_outbound_safe_context(
        context, include_exploratory_profiles=False,
    ))
    assert context.model_dump_json() == before
    for original, row in zip(full["variables"], projected["variables"], strict=True):
        assert row == {k: v for k, v in original.items()
                       if k not in {"observed_shape", "missingness", "aggregation_hint"}}
    llm = PatternScriptedMockLLMClient([], default="## Results\n\nComplete.")
    digest = "n=80 {evidence:analysis}; missing=20 {evidence:measurement_audit}"
    WriterAgent(llm)._call_section(
        section_name="Results", instruction="Describe the evidence.",
        context=context, evidence_ids=["analysis", "measurement_audit"], evidence_digest=digest,
    )
    prompt = llm.calls[0][0][1].content
    assert digest in prompt
    for text in (context.variables[2].description, variable.unit, variable.analysis_window,
                 variable.missingness_semantics):
        assert text in prompt
    assert '"n_missing"' not in prompt
    assert '"observed_shape"' not in prompt
