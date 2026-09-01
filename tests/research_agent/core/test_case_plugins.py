"""Tests for the case-plugin infrastructure.

These exercise the deliberately *opt-in* nature of case plugins: a default
``ResearchAgentPipeline()`` carries no plugins, and an empty registry returns
``None`` from every hook so the pipeline can be wired through it without any
case-specific contamination. No case-specific plugins are bundled with the
package — users supply their own.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_default_pipeline_has_empty_plugin_registry(ra, tmp_path: Path):
    """A pipeline constructed without ``case_plugin_registry`` should
    end up with an empty registry, i.e. no case-specific bias.
    """
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    registry = pipeline._case_plugin_registry  # type: ignore[attr-defined]

    from easyicu.research_agent.fallback import CasePluginRegistry

    assert isinstance(registry, CasePluginRegistry)
    assert len(registry) == 0
    assert registry.names() == []


def test_registry_returns_none_for_unrecognised_research_question(ra):
    """An empty registry returns ``None`` for every hook — proving the
    pipeline can be wired through it without any fallback contamination.
    """
    from easyicu.research_agent.fallback import CasePluginRegistry

    registry = CasePluginRegistry()
    cohort = ra.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1)
    context = ra.ResearchContext(
        research_question="Unrelated question",
        cohort=cohort,
        variables=[],
    )
    step = ra.AnalysisStep(
        step_id="t01_unrelated",
        intent="something",
        inputs=[],
        expected_outputs=[],
    )

    assert registry.fallback_code(context=context, step=step) is None
    assert registry.repair_code(context=context, step=step, code="", run_log="") is None
    assert registry.v15_task_template(context=context, task_key="lactate") is None
    assert registry.column_aliases(context=context) == {}
