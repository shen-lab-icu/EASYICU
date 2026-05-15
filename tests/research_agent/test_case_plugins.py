"""Tests for the case-plugin infrastructure.

These exercise the deliberately *opt-in* nature of case plugins: a default
``ResearchAgentPipeline()`` carries no plugins, and constructing one with
the bundled ``lactate_map_vaso`` plugin adds the historical deterministic
fallbacks back in.
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


def test_pipeline_accepts_explicit_plugin_registry(ra, tmp_path: Path, lactate_plugin_registry):
    """When the caller supplies a registry, the pipeline keeps it and
    exposes the plugin names through the registry's introspection API.
    """
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        case_plugin_registry=lactate_plugin_registry,
    )

    registry = pipeline._case_plugin_registry  # type: ignore[attr-defined]
    assert registry is lactate_plugin_registry
    assert "lactate_map_vaso" in registry.names()


def test_lactate_plugin_satisfies_case_plugin_protocol(ra):
    """The bundled plugin must satisfy the runtime-checkable
    :class:`CasePlugin` protocol.
    """
    from easyicu.research_agent.case_plugins.lactate_map_vaso import plugin
    from easyicu.research_agent.fallback import CasePlugin

    assert isinstance(plugin, CasePlugin)
    assert plugin.name == "lactate_map_vaso"


def test_lactate_plugin_v15_task_template_emits_lactate_code(ra):
    """The plugin's ``v15_task_template("lactate")`` hook must produce
    a runnable Python string that references the lactate variable.
    """
    from easyicu.research_agent.case_plugins.lactate_map_vaso import plugin

    code = plugin.v15_task_template("lactate")
    assert code is not None
    assert "lactate_max_24h" in code
    assert "death" in code


def test_lactate_plugin_column_aliases_include_canonical_keys(ra):
    """The plugin must expose canonical → alias mappings for the
    columns the original hardcoded pipeline assumed.
    """
    from easyicu.research_agent.case_plugins.lactate_map_vaso import plugin

    aliases = plugin.column_aliases()
    assert "lactate" in aliases
    assert "lactate_max_24h" in aliases["lactate"]
    assert "vasopressor" in aliases
    assert "vaso_any_24h" in aliases["vasopressor"]


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
