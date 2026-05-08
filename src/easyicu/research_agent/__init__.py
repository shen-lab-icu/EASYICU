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

    from easyicu.research_agent import ResearchAgentPipeline, MockLLMClient

    pipeline = ResearchAgentPipeline(
        workdir="./research_output",
        llm=MockLLMClient(),  # tests/demo only; use a real client for research runs
    )
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

import os

from .step_summary import step_summary

os_environ = os.environ

__all__ = [
    # Schemas
    "ResearchContext",
    "ConceptDescriptor",
    "CohortDescriptor",
    "TimeWindow",
    "TemporalConstraint",
    "AnalysisStep",
    "AnalysisPlan",
    "EvidenceRef",
    "ConceptRef",
    "ClinicalSemanticsResolution",
    "DataExtractionRequest",
    "DataExtractionResult",
    "StatisticalAnalysisRequest",
    "StatisticalAnalysisResult",
    "VisualizationRequest",
    "VisualizationResult",
    "ManuscriptDraftPacket",
    "HypothesisBlueprint",
    "CritiqueReport",
    "ReflectionMemoryEntry",
    "AgentRuntimeState",
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
    # Architecture / temporal semantics / experiment specs
    "SystemLayer",
    "AgentRole",
    "ArchitectureProfile",
    "default_architecture_profile",
    "architecture_profile_markdown",
    "ConceptValidationLayer",
    "TemporalAlignmentEngine",
    "ICUEpisodeResolver",
    "EpisodeResolution",
    "TimeWindowSemanticParser",
    "ConceptDatabaseAvailability",
    "cross_database_concept_availability",
    "default_public_databases",
    "explain_concept_availability",
    "hypothesis_cross_database_feasibility",
    "ExperimentSpec",
    "CohortInputSpec",
    "RuntimeSpec",
    "load_experiment_spec",
    "dump_experiment_spec",
    "build_pipeline_from_spec",
    # LLM
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "LLMRouter",
    # Agents
    "PlannerAgent",
    "ReplannerAgent",
    "ClinicalSemanticsAgent",
    "DataExtractionAgent",
    "StatisticalAnalysisAgent",
    "VisualizationAgent",
    "ManuscriptAgent",
    "CriticAgent",
    "RuntimeSupervisor",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "LiteratureAgent",
    "HypothesisBlueprintAgent",
    # Validators / runtime components
    "CodeRunner",
    "DockerRunner",
    "RunResult",
    "CohortAuditor",
    "StatisticalValidator",
    "ClinicalConstraintValidator",
    "StatisticalGuard",
    "ConceptUsageAuditor",
    "LLMConceptAuditor",
    "VisualQAAuditor",
    "VLMVisualQAAdapter",
    "PublicationFigureSkill",
    "PublicationFigureSkillResult",
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
    # Skills
    "ClinicalSkill",
    "register_skill",
    "get_skill",
    "list_skills",
    "AnalysisTypeSpec",
    "get_analysis_type",
    "list_analysis_types",
    "infer_analysis_type",
    "planner_analysis_type_guide",
    "analysis_type_catalog_markdown",
    "ICUAgentBenchMetricSpec",
    "ICUAgentBenchTask",
    "ICUAgentBenchSuite",
    "ICUAgentBenchTaskResult",
    "ICUAgentBenchReport",
    "default_icu_agent_bench_suite",
    "icu_agent_bench_markdown",
    # Memory
    "RunMemory",
    "StrategyCard",
    # LaTeX
    "scaffold_to_latex",
    "latex_template_preamble",
    # BibTeX export (T3.4)
    "render_bibtex",
    "render_thebibliography_block",
    # Literature
    "CitationRecord",
    "LiteratureBundle",
    "PubMedLiteratureClient",
    "TavilyLiteratureClient",
    "render_hypothesis_blueprint_for_prompt",
    # MCP server
    "mcp_dispatch",
    "MCP_TOOLS",
    "MCP_TOOL_SCHEMAS",
    # Cost tracking (T3.2)
    "CostMeter",
    "MeteredClient",
    "CostRecord",
    "AuditEvent",
    "AuditLogger",
    "WorkflowGraph",
    "WorkflowNode",
    "WorkflowEdge",
    "ExecutionReplayBundle",
    "build_workflow_graph",
    "render_workflow_graph_mermaid",
    "build_execution_replay",
    # Pipeline
    "ResearchAgentPipeline",
    # Prompt pack provenance
    "PROMPT_PACK_VERSION",
    "prompt_pack_files",
    # Compatibility exports used by generated scripts
    "step_summary",
    "os_environ",
]

# Schemas are dependency-free, safe to import eagerly.
from .schema import (
    ResearchContext,
    ConceptDescriptor,
    CohortDescriptor,
    TimeWindow,
    TemporalConstraint,
    AnalysisStep,
    AnalysisPlan,
    EvidenceRef,
    ConceptRef,
    ClinicalSemanticsResolution,
    DataExtractionRequest,
    DataExtractionResult,
    StatisticalAnalysisRequest,
    StatisticalAnalysisResult,
    VisualizationRequest,
    VisualizationResult,
    ManuscriptDraftPacket,
    HypothesisBlueprint,
    CritiqueReport,
    ReflectionMemoryEntry,
    AgentRuntimeState,
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
        "SystemLayer",
        "AgentRole",
        "ArchitectureProfile",
        "default_architecture_profile",
        "architecture_profile_markdown",
    }:
        from . import architecture as _architecture

        return getattr(_architecture, name)
    if name in {
        "ConceptValidationLayer",
        "TemporalAlignmentEngine",
        "ICUEpisodeResolver",
        "EpisodeResolution",
        "TimeWindowSemanticParser",
    }:
        from . import temporal_semantics as _temporal

        return getattr(_temporal, name)
    if name in {
        "ConceptDatabaseAvailability",
        "cross_database_concept_availability",
        "default_public_databases",
        "explain_concept_availability",
        "hypothesis_cross_database_feasibility",
    }:
        from . import concept_availability as _availability

        return getattr(_availability, name)
    if name in {
        "ExperimentSpec",
        "CohortInputSpec",
        "RuntimeSpec",
        "load_experiment_spec",
        "dump_experiment_spec",
        "build_pipeline_from_spec",
    }:
        from . import experiment_spec as _spec

        return getattr(_spec, name)
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
    if name in {
        "PlannerAgent",
        "ReplannerAgent",
        "ClinicalSemanticsAgent",
        "DataExtractionAgent",
        "StatisticalAnalysisAgent",
        "VisualizationAgent",
        "ManuscriptAgent",
        "CriticAgent",
        "RuntimeSupervisor",
        "CoderAgent",
        "AnalyzerAgent",
        "WriterAgent",
    }:
        from . import agents as _agents

        return getattr(_agents, name)
    if name in {"PROMPT_PACK_VERSION", "prompt_pack_files"}:
        from . import prompts as _prompts

        return getattr(_prompts, name)
    if name in {"LiteratureAgent", "HypothesisBlueprintAgent"}:
        from . import literature as _lit

        return getattr(_lit, name)
    if name in {
        "CitationRecord",
        "LiteratureBundle",
        "PubMedLiteratureClient",
        "TavilyLiteratureClient",
        "render_hypothesis_blueprint_for_prompt",
    }:
        from . import literature as _lit

        return getattr(_lit, name)
    if name in {"CodeRunner", "DockerRunner", "RunResult"}:
        from . import runner as _runner

        return getattr(_runner, name)
    if name in {
        "CohortAuditor",
        "StatisticalValidator",
        "ClinicalConstraintValidator",
        "StatisticalGuard",
        "ConceptUsageAuditor",
        "LLMConceptAuditor",
    }:
        from . import validators as _validators

        return getattr(_validators, name)
    if name in {
        "AuditEvent",
        "AuditLogger",
        "WorkflowGraph",
        "WorkflowNode",
        "WorkflowEdge",
        "ExecutionReplayBundle",
        "build_workflow_graph",
        "render_workflow_graph_mermaid",
        "build_execution_replay",
    }:
        from . import runtime_artifacts as _runtime_artifacts

        return getattr(_runtime_artifacts, name)
    if name in {"VisualQAAuditor", "VLMVisualQAAdapter"}:
        from . import visual_qa as _visual_qa

        return getattr(_visual_qa, name)
    if name in {"PublicationFigureSkill", "PublicationFigureSkillResult"}:
        from . import figure_skill as _figure_skill

        return getattr(_figure_skill, name)
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
    if name in {
        "AnalysisTypeSpec",
        "get_analysis_type",
        "list_analysis_types",
        "infer_analysis_type",
        "planner_analysis_type_guide",
        "analysis_type_catalog_markdown",
    }:
        from . import analysis_types as _analysis_types

        return getattr(_analysis_types, name)
    if name in {
        "ICUAgentBenchMetricSpec",
        "ICUAgentBenchTask",
        "ICUAgentBenchSuite",
        "ICUAgentBenchTaskResult",
        "ICUAgentBenchReport",
        "default_icu_agent_bench_suite",
        "icu_agent_bench_markdown",
    }:
        from . import icu_agent_bench as _icu_agent_bench

        return getattr(_icu_agent_bench, name)
    if name in {"RunMemory", "StrategyCard"}:
        from . import memory as _memory

        return getattr(_memory, name)
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
