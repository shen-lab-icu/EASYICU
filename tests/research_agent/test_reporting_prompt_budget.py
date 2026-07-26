from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import (
    AnalyzerAgent,
    ReportingPromptBudgetError,
    WriterAgent,
)
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
