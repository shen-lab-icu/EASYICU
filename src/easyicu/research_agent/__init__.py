"""easyicu.research_agent - Traceable, ICU-aware analysis agent layer.

This sub-package extends EasyICU from "data extraction and visualisation"
to "data extraction → analysis → manuscript scaffold" while keeping every
generated number, figure and sentence linked back to:

* the EasyICU concept dictionary entry that produced it,
* the cohort configuration that selected the rows,
* the deterministic Python script that ran the computation, and
* the raw run log captured during execution.

Design goals
------------
1. **ICU-aware by construction** — the agent receives a structured
   ``ResearchContext`` describing variable types, aggregation rules,
   time-window semantics, missingness profile and known pitfalls
   (e.g. "do not take the mean of an ordinal SOFA component").
2. **Traceable end-to-end** — every artefact (table, figure, statistic,
   manuscript sentence) is registered in an ``EvidenceStore`` with a
   SHA-256 hash and a pointer to the script, inputs and run log that
   produced it. Manuscript scaffolding can only cite registered
   evidence ids.
3. **Deterministic compute, agentic planning** — the LLM proposes
   plans and writes code, but the executed code is plain Python in a
   subprocess sandbox. Numbers are never produced by the LLM directly.
4. **No hard external dependency on heavy frameworks** — no LangGraph,
   no Docker. The default loop is a small sequential state machine
   that is easy to read, debug and cite.

Public API
----------
``ResearchAgentPipeline`` is the single end-to-end entry point::

    from easyicu.research_agent import ResearchAgentPipeline

    pipeline = ResearchAgentPipeline(workdir="./research_output")
    result = pipeline.run(
        question="Is admission SOFA score associated with ICU mortality?",
        cohort_parquet="cohort.parquet",
        concept_names=["sofa", "death", "los_icu", "age", "sex"],
        database="miiv",
    )
    print(result.report_path)        # results_report.md
    print(result.manuscript_path)    # manuscript_scaffold.md
    print(result.manifest_path)      # manifest.json

The pipeline is also exposed through the ``easyicu-research-agent``
console script (see :mod:`easyicu.research_agent.cli`).

This module is intentionally optional: importing it has no side
effects on the rest of EasyICU and the heavy imports (LLM clients,
subprocess sandbox) are lazy so a failed import in one component does
not break the rest.
"""

from __future__ import annotations

__all__ = [
    # Schemas
    "ResearchContext",
    "ConceptDescriptor",
    "CohortDescriptor",
    "TimeWindow",
    "AnalysisStep",
    "AnalysisPlan",
    "EvidenceRecord",
    "AnalysisManifest",
    "PipelineResult",
    # Context builder
    "build_research_context",
    "build_naive_research_context",
    "retrieve_context_variables",
    "build_retrieved_research_context",
    "build_lactate_map_vaso_research_context",
    "build_lactate_map_vaso_context_ablation_table",
    "context_information_summary",
    "write_research_context",
    # ICU rules
    "ICU_RULES",
    "VariableKind",
    # LLM
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "LLMRouter",
    # Agents
    "PlannerAgent",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "LiteratureAgent",
    # Validators / runtime components
    "CodeRunner",
    "DockerRunner",
    "RunResult",
    "CohortAuditor",
    "StatisticalValidator",
    "ConceptUsageAuditor",
    "LLMConceptAuditor",
    "VisualQAAuditor",
    "VLMVisualQAAdapter",
    "FigureContract",
    "PanelSpec",
    "make_figure_contract",
    "audit_figure_contract",
    "apply_publication_style",
    "save_publication_figure",
    "audit_publication_exports",
    "EvidenceStore",
    "EasyICUCasePackage",
    "index_export_package",
    "read_exported_concept",
    "build_lactate_map_vaso_cohort_from_export",
    "ReplicationTarget",
    "LACTATE_MAP_VASO_EXPORT_GROUPS",
    "LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS",
    "discover_easyicu_exports",
    "export_lactate_map_vaso_concepts_from_easyicu",
    "shock_strata",
    "summarize_lactate_map_vaso_cohort",
    "run_lactate_map_vaso_replication",
    # Skills (M4-inspired)
    "ClinicalSkill",
    "register_skill",
    "get_skill",
    "list_skills",
    # Memory (HealthFlow-inspired)
    "RunMemory",
    # LaTeX (OpenLens-inspired)
    "scaffold_to_latex",
    "latex_template_preamble",
    # BibTeX export (T3.4)
    "render_bibtex",
    "render_thebibliography_block",
    # Literature (OpenLens-inspired)
    "CitationRecord",
    "LiteratureBundle",
    "PubMedLiteratureClient",
    "TavilyLiteratureClient",
    # MCP server (M4-inspired)
    "mcp_dispatch",
    "MCP_TOOLS",
    "MCP_TOOL_SCHEMAS",
    # Cost tracking (T3.2)
    "CostMeter",
    "MeteredClient",
    "CostRecord",
    # Pipeline
    "ResearchAgentPipeline",
]

# Schemas are dependency-free, safe to import eagerly.
from .schema import (
    ResearchContext,
    ConceptDescriptor,
    CohortDescriptor,
    TimeWindow,
    AnalysisStep,
    AnalysisPlan,
    EvidenceRecord,
    AnalysisManifest,
    PipelineResult,
    CostRecord,
)
from .icu_rules import ICU_RULES, VariableKind


def __getattr__(name: str):
    """Lazy import of heavier components.

    Keeps ``import easyicu.research_agent`` cheap so that simply having
    the module installed does not pull in pandas-heavy code paths or
    optional LLM SDKs unless the user actually uses them.
    """
    if name in {
        "build_research_context",
        "build_naive_research_context",
        "retrieve_context_variables",
        "build_retrieved_research_context",
    }:
        from . import context as _context
        return getattr(_context, name)
    if name in {
        "build_lactate_map_vaso_research_context",
        "build_lactate_map_vaso_context_ablation_table",
        "context_information_summary",
        "write_research_context",
    }:
        from . import case_contexts as _case_contexts
        return getattr(_case_contexts, name)
    if name in {"LLMClient", "MockLLMClient", "OpenAIClient", "LLMRouter"}:
        from . import llm as _llm
        return getattr(_llm, name)
    if name in {"PlannerAgent", "CoderAgent", "AnalyzerAgent", "WriterAgent"}:
        from . import agents as _agents
        return getattr(_agents, name)
    if name == "LiteratureAgent":
        from .literature import LiteratureAgent
        return LiteratureAgent
    if name in {
        "CitationRecord",
        "LiteratureBundle",
        "PubMedLiteratureClient",
        "TavilyLiteratureClient",
    }:
        from . import literature as _lit
        return getattr(_lit, name)
    if name in {"CodeRunner", "DockerRunner", "RunResult"}:
        from . import runner as _runner
        return getattr(_runner, name)
    if name in {
        "CohortAuditor",
        "StatisticalValidator",
        "ConceptUsageAuditor",
        "LLMConceptAuditor",
    }:
        from . import validators as _validators
        return getattr(_validators, name)
    if name in {"VisualQAAuditor", "VLMVisualQAAdapter"}:
        from . import visual_qa as _visual_qa
        return getattr(_visual_qa, name)
    if name in {
        "FigureContract",
        "PanelSpec",
        "make_figure_contract",
        "audit_figure_contract",
        "apply_publication_style",
        "save_publication_figure",
        "audit_publication_exports",
    }:
        from . import publication_figures as _pubfig
        return getattr(_pubfig, name)
    if name == "EvidenceStore":
        from .evidence import EvidenceStore
        return EvidenceStore
    if name in {
        "EasyICUCasePackage",
        "index_export_package",
        "read_exported_concept",
        "build_lactate_map_vaso_cohort_from_export",
    }:
        from . import easyicu_case_builder as _case_builder
        return getattr(_case_builder, name)
    if name in {
        "ReplicationTarget",
        "LACTATE_MAP_VASO_EXPORT_GROUPS",
        "LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS",
        "discover_easyicu_exports",
        "export_lactate_map_vaso_concepts_from_easyicu",
        "shock_strata",
        "summarize_lactate_map_vaso_cohort",
        "run_lactate_map_vaso_replication",
    }:
        from . import replication as _replication
        return getattr(_replication, name)
    if name in {"ClinicalSkill", "register_skill", "get_skill", "list_skills"}:
        from . import skills as _skills
        return getattr(_skills, name)
    if name == "RunMemory":
        from .memory import RunMemory
        return RunMemory
    if name in {"scaffold_to_latex", "latex_template_preamble"}:
        from . import latex as _latex
        return getattr(_latex, name)
    if name in {"render_bibtex", "render_thebibliography_block"}:
        from . import bibtex as _bibtex
        return getattr(_bibtex, name)
    if name in {"CostMeter", "MeteredClient"}:
        from . import cost as _cost
        return getattr(_cost, name)
    if name in {"mcp_dispatch", "MCP_TOOLS", "MCP_TOOL_SCHEMAS"}:
        from . import mcp_server as _mcp
        mapping = {
            "mcp_dispatch": _mcp.dispatch,
            "MCP_TOOLS": _mcp.TOOLS,
            "MCP_TOOL_SCHEMAS": _mcp.TOOL_SCHEMAS,
        }
        return mapping[name]
    if name == "ResearchAgentPipeline":
        from .pipeline import ResearchAgentPipeline
        return ResearchAgentPipeline
    raise AttributeError(f"module 'easyicu.research_agent' has no attribute {name!r}")
