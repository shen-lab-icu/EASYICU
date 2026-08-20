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

    from easyicu.research_agent import (
        MockLLMClient,
        PipelineConfig,
        PipelineServices,
        ResearchAgentPipeline,
    )

    pipeline = ResearchAgentPipeline(
        config=PipelineConfig(workdir="./research_output"),
        services=PipelineServices(
            llm=MockLLMClient(),  # tests/demo only; use a real client for research runs
        ),
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

__all__ = [
    # Schemas
    "ResearchContext",
    "ClinicalDefinitionReference",
    "ConceptDescriptor",
    "EndpointSpec",
    "FixedWindowTrajectoryMetadata",
    "ClusterSelectionCandidate",
    "ClusterSelectionManifest",
    "CohortDescriptor",
    "TimeWindow",
    "TemporalConstraint",
    "PlannedAnalysisRole",
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
    "PaperClaimRecord",
    "PaperProfile",
    "PaperReplicationSpec",
    "PaperResultLedger",
    "ReplicationDeviationItem",
    "ReplicationDeviationReport",
    "ProbeSummary",
    "StepRecord",
    "KnowHowCard",
    "KnowHowCitation",
    "KnowHowHit",
    "KnowHowRegistry",
    "KnowHowIntegrityError",
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
    "detect_overadjustment",
    "composite_constituents",
    "is_derived_exposure",
    "overadjustment_caution",
    "detect_outcome_as_predictor",
    "outcome_leakage_caution",
    "treatment_mediator_caution",
    "concept_methodology_profile",
    "concept_methodology_tag",
    "ConceptMethodologyProfile",
    "COMPOSITE_EXPOSURE_CONSTITUENTS",
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
    "RealDataConceptFeasibility",
    "CandidateAlreadyRegisteredError",
    "CandidateNotExecutableError",
    "CandidateNotRegisteredError",
    "CandidateRegistryEntry",
    "DiscoveryCandidateRecord",
    "DiscoveryTriageResult",
    "ExecutableHypothesisCandidate",
    "IDEA_EXTRACTION_SYSTEM_PROMPT",
    "IDEA_MINING_SNAPSHOT_SCHEMA_VERSION",
    "IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION",
    "IdeaMiningCandidateTriageRecord",
    "IdeaMiningDryRunResult",
    "IdeaExtractionError",
    "IdeaMiningError",
    "IdeaMiningFeasibilityRecord",
    "IdeaMiningYieldReport",
    "IdeaCandidateRegistry",
    "LiteratureIdeaCandidate",
    "NoveltyLabel",
    "NonExecutableCandidateError",
    "OutcomeDeterminability",
    "OutcomeDeterminabilityStatus",
    "PriorArtAssessment",
    "PriorArtQueryRecord",
    "PriorArtSearchHit",
    "IdeaRegistryError",
    "SelectionStatus",
    "SourceAdapterLevel",
    "SourceMaterial",
    "SourceSnapshotItem",
    "SourceSnapshotManifest",
    "assess_prior_art_for_candidates",
    "assess_prior_art_for_idea",
    "build_idea_extraction_messages",
    "build_discovery_candidate_records",
    "build_prior_art_queries",
    "cross_database_concept_availability",
    "default_public_databases",
    "explain_concept_availability",
    "extract_literature_ideas",
    "fetch_source_materials_from_scope",
    "freeze_source_snapshot",
    "hypothesis_cross_database_feasibility",
    "map_literature_idea_to_executable_candidate",
    "real_data_concept_feasibility",
    "render_discovery_report",
    "run_idea_mining_dry_run",
    "LiteratureFunnelResult",
    "LiteratureFunnelRoute",
    "LiteratureFunnelSpec",
    "build_literature_funnel_queries",
    "extract_gap_sections_from_text",
    "fetch_literature_funnel_corpus",
    "fetch_literature_funnel_source_materials",
    "IDEA_QUALITY_EVAL_SCHEMA_VERSION",
    "IdeaQualityEvalItem",
    "IdeaQualityEvalSet",
    "IdeaQualityExpectedLabel",
    "IdeaQualityPrediction",
    "IdeaQualityScorecard",
    "default_idea_quality_eval_path",
    "load_idea_quality_eval_set",
    "score_idea_quality_predictions",
    "summarize_idea_quality_eval_set",
    # Idea-mining literature scope (discovery lever 1)
    "JOURNAL_PRESETS",
    "LiteratureScopeSpec",
    "build_pubmed_query_from_scope",
    "resolve_journals",
    "resolve_year_range",
    "DISCOVERY_HANDOFF_SCHEMA_VERSION",
    "DISCOVERY_PACKAGE_SCHEMA_VERSION",
    "DiscoveryHandoffPacket",
    "DiscoveryManuscriptPackageAssessment",
    "ManuscriptFigureInventoryItem",
    "build_handoff_from_row",
    "load_discovery_ledger",
    "select_discovery_row",
    "render_discovery_story_figure",
    "validate_discovery_manuscript_package",
    "write_discovery_package_assessment",
    "write_handoff_packet",
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
    "AnthropicMessagesClient",
    "CLIAgentLLMClient",
    "LLMRouter",
    "build_llm_client",
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
    "FigureContractQualityValidator",
    "FigureSourceDataValidator",
    "ClinicalConstraintValidator",
    "StatisticalGuard",
    "ConceptUsageAuditor",
    "LLMConceptAuditor",
    "ReplicationDesignAuditor",
    "ReplicationResultComparator",
    "PublicationClaimAuditor",
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
    "audit_publication_exports_json",
    "EvidenceStore",
    "EvidenceEnforcementMode",
    "EvidenceEnforcementError",
    "ScientificClaim",
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
    "load_paper_source",
    "parse_paper_profile",
    "build_paper_replication_spec",
    "build_paper_result_ledger",
    "collect_easyicu_metrics",
    "compare_paper_to_easyicu",
    "compare_metric_values",
    "render_replication_report",
    "render_deviation_report",
    "render_showcase_manuscript",
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
    "TaskCategory",
    "ICUAgentBenchMetricSpec",
    "ICUAgentBenchNumericBound",
    "ICUAgentBenchGoldAnswer",
    "ICUAgentBenchTask",
    "ICUAgentBenchSuite",
    "ICUAgentBenchTaskResult",
    "ICUAgentBenchReport",
    "default_icu_agent_bench_suite",
    "grade_bench_task",
    "aggregate_bench_report",
    "TaskMetricDelta",
    "BenchABComparison",
    "compare_bench_reports",
    "icu_agent_bench_markdown",
    # Memory
    "RunMemory",
    "StrategyCard",
    "MemoryScoreBreakdown",
    "MemoryRetrievalAuditEntry",
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
    # Reproducibility envelope (O20)
    "ReproEnvelope",
    "ReproRecordingClient",
    "build_environment_snapshot",
    "envelope_role_resolver",
    # Multiple-testing correction (O22)
    "MultipleTestingReport",
    "PValueRecord",
    "build_multiple_testing_report",
    # Causal audit (O18)
    "CausalAuditReport",
    "CausalLanguageHit",
    "EffectLabel",
    "label_effects",
    "run_causal_audit",
    "scan_manuscript_for_causal_language",
    # Reporting checklist (O16)
    "ChecklistItem",
    "ChecklistReport",
    "build_strobe_checklist",
    "build_tripod_ai_checklist",
    "build_internal_phenotype_checklist",
    "choose_checklist",
    "checklist_names_for_kind",
    # Reviewer round (O15)
    "ReviewerComment",
    "ReviewerCritique",
    "ReviewerReport",
    "run_reviewer_round",
    # Provenance (O27)
    "ProvenanceBundle",
    "SourceFileRecord",
    "build_provenance_bundle",
    "hash_sources",
    # Sensitivity (O23): E-value + negative-control
    "EValueResult",
    "NegativeControlResult",
    "compute_e_value",
    "run_negative_control_check",
    # Notebook + lockfile (O26)
    "NotebookStep",
    "build_notebook",
    "build_requirements_lockfile",
    "write_notebook",
    # Missing data (O25)
    "MICEImputationResult",
    "TippingPointResult",
    "mice_impute",
    "tipping_point_analysis",
    # Fairness / subgroup (O24)
    "SubgroupAnalysisResult",
    "SubgroupEstimate",
    "run_subgroup_analysis",
    # Hypothesis generator (O17)
    "HypothesisCandidate",
    "HypothesisFeasibilitySignal",
    "HypothesisGeneratorResult",
    "LITERATURE_SATURATION_SIGNAL_STATEMENT",
    "generate_hypotheses",
    # Analysis-pattern auditor (generic ICU footguns)
    "AnalysisPatternAuditor",
    # PDF render
    "PDFRenderResult",
    "render_pdf_for_run",
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
    "PipelineConfig",
    "PipelineServices",
    "SubmissionProfile",
    "NPJ_DM_2026_05",
    "NPJ_DM_2026_06",
    "NPJ_DM_2026_07",
    "NPJ_DM_2026_07_16",
    "NPJ_DM_2026_07_17",
    "NPJ_DM_2026_07_18",
    "NPJ_DM_2026_07_19",
    "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_16",
    "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_17",
    "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19",
    "E1_REVIEWED_DEMO_2026_08_17",
    "E1_REVIEWED_DEMO_2026_08_19",
    "NPJ_DM_2026_07_21_KNOW_HOW",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV",
    "DEFAULT_SUBMISSION_PROFILE_REF",
    "SUBMISSION_PROFILE_REGISTRY",
    "get_submission_profile",
    # Prompt pack provenance
    "PROMPT_PACK_VERSION",
    "prompt_pack_files",
]


def __getattr__(name: str):
    """Lazy import of heavier components.

    Keeps ``import easyicu.research_agent`` cheap so that simply having
    the module installed does not pull in pandas-heavy code paths or
    optional LLM SDKs unless the user actually uses them.
    """
    if name in {
        "ResearchContext",
        "ClinicalDefinitionReference",
        "ConceptDescriptor",
        "EndpointSpec",
        "FixedWindowTrajectoryMetadata",
        "ClusterSelectionCandidate",
        "ClusterSelectionManifest",
        "CohortDescriptor",
        "TimeWindow",
        "TemporalConstraint",
        "PlannedAnalysisRole",
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
        "CostRecord",
        "PaperClaimRecord",
        "PaperProfile",
        "PaperReplicationSpec",
        "PaperResultLedger",
        "ReplicationDeviationItem",
        "ReplicationDeviationReport",
        "ProbeSummary",
        "StepRecord",
    }:
        from . import schema as _schema

        return getattr(_schema, name)
    if name in {
        "COMPOSITE_EXPOSURE_CONSTITUENTS",
        "ICU_RULES",
        "ConceptMethodologyProfile",
        "VariableKind",
        "composite_constituents",
        "concept_methodology_profile",
        "concept_methodology_tag",
        "detect_outcome_as_predictor",
        "detect_overadjustment",
        "is_derived_exposure",
        "outcome_leakage_caution",
        "overadjustment_caution",
        "treatment_mediator_caution",
    }:
        from . import icu_rules as _icu_rules

        return getattr(_icu_rules, name)
    if name in {
        "KnowHowCard",
        "KnowHowCitation",
        "KnowHowHit",
        "KnowHowRegistry",
        "KnowHowIntegrityError",
    }:
        from . import know_how as _know_how

        return getattr(_know_how, name)
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
        from .research_context import temporal_semantics as _temporal

        return getattr(_temporal, name)
    if name in {
        "ConceptDatabaseAvailability",
        "RealDataConceptFeasibility",
        "cross_database_concept_availability",
        "default_public_databases",
        "explain_concept_availability",
        "hypothesis_cross_database_feasibility",
        "real_data_concept_feasibility",
    }:
        from . import concept_availability as _availability

        return getattr(_availability, name)
    if name in {
        "CandidateAlreadyRegisteredError",
        "CandidateNotExecutableError",
        "CandidateNotRegisteredError",
        "CandidateRegistryEntry",
        "IdeaCandidateRegistry",
        "IdeaRegistryError",
        "SelectionStatus",
    }:
        from .discovery import idea_registry as _idea_registry

        return getattr(_idea_registry, name)
    if name in {
        "DISCOVERY_REPORT_SCHEMA_VERSION",
        "DiscoveryCandidateRecord",
        "DiscoveryTriageResult",
        "ExecutableHypothesisCandidate",
        "IDEA_EXTRACTION_SYSTEM_PROMPT",
        "IDEA_MINING_SNAPSHOT_SCHEMA_VERSION",
        "IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION",
        "IdeaMiningCandidateTriageRecord",
        "IdeaMiningDryRunResult",
        "IdeaExtractionError",
        "IdeaMiningError",
        "IdeaMiningFeasibilityRecord",
        "IdeaMiningYieldReport",
        "LiteratureIdeaCandidate",
        "NoveltyLabel",
        "NonExecutableCandidateError",
        "OutcomeDeterminability",
        "OutcomeDeterminabilityStatus",
        "PriorArtAssessment",
        "PriorArtQueryRecord",
        "PriorArtSearchHit",
        "SourceAdapterLevel",
        "SourceMaterial",
        "SourceSnapshotItem",
        "SourceSnapshotManifest",
        "assess_prior_art_for_candidates",
        "assess_prior_art_for_idea",
        "build_idea_extraction_messages",
        "build_discovery_candidate_records",
        "build_prior_art_queries",
        "extract_literature_ideas",
        "fetch_source_materials_from_scope",
        "freeze_source_snapshot",
        "map_literature_idea_to_executable_candidate",
        "render_discovery_report",
        "run_idea_mining_dry_run",
    }:
        from .discovery import idea_mining as _idea_mining

        return getattr(_idea_mining, name)
    if name in {
        "JOURNAL_PRESETS",
        "LiteratureScopeSpec",
        "build_pubmed_query_from_scope",
        "resolve_journals",
        "resolve_year_range",
    }:
        from .discovery import idea_scope as _idea_scope

        return getattr(_idea_scope, name)
    if name in {
        "LiteratureFunnelResult",
        "LiteratureFunnelRoute",
        "LiteratureFunnelSpec",
        "build_literature_funnel_queries",
        "extract_gap_sections_from_text",
        "fetch_literature_funnel_corpus",
        "fetch_literature_funnel_source_materials",
    }:
        from .discovery import idea_mining_funnel as _idea_mining_funnel

        return getattr(_idea_mining_funnel, name)
    if name in {
        "IDEA_QUALITY_EVAL_SCHEMA_VERSION",
        "IdeaQualityEvalItem",
        "IdeaQualityEvalSet",
        "IdeaQualityExpectedLabel",
        "IdeaQualityPrediction",
        "IdeaQualityScorecard",
        "default_idea_quality_eval_path",
        "load_idea_quality_eval_set",
        "score_idea_quality_predictions",
        "summarize_idea_quality_eval_set",
    }:
        from .discovery import idea_mining_eval as _idea_mining_eval

        return getattr(_idea_mining_eval, name)
    if name in {
        "DISCOVERY_HANDOFF_SCHEMA_VERSION",
        "DiscoveryHandoffPacket",
        "build_handoff_from_row",
        "load_discovery_ledger",
        "select_discovery_row",
        "write_handoff_packet",
    }:
        from .discovery import discovery_handoff as _discovery_handoff

        return getattr(_discovery_handoff, name)
    if name in {
        "DISCOVERY_PACKAGE_SCHEMA_VERSION",
        "DiscoveryManuscriptPackageAssessment",
        "ManuscriptFigureInventoryItem",
        "validate_discovery_manuscript_package",
        "write_discovery_package_assessment",
    }:
        from .discovery import discovery_package as _discovery_package

        return getattr(_discovery_package, name)
    if name in {"render_discovery_story_figure"}:
        from .discovery import discovery_story_figure as _discovery_story_figure

        return getattr(_discovery_story_figure, name)
    if name in {
        "ExperimentSpec",
        "CohortInputSpec",
        "RuntimeSpec",
        "load_experiment_spec",
        "dump_experiment_spec",
        "build_pipeline_from_spec",
    }:
        from .orchestration import experiment_spec as _spec

        return getattr(_spec, name)
    if name in {
        "build_research_context",
        "build_naive_research_context",
        "retrieve_context_variables",
        "build_retrieved_research_context",
    }:
        from .research_context import builder as _context

        return getattr(_context, name)
    if name in {
        "build_lactate_map_vaso_research_context",
        "build_lactate_map_vaso_context_ablation_table",
        "context_information_summary",
        "write_research_context",
    }:
        from .case_plugins import contexts as _case_contexts

        return getattr(_case_contexts, name)
    if name == "LLMClient":
        from .providers import protocol as _protocol

        return getattr(_protocol, name)
    if name == "MockLLMClient":
        from .providers import mocks as _mocks

        return getattr(_mocks, name)
    if name in {
        "OpenAIClient",
        "AnthropicMessagesClient",
        "CLIAgentLLMClient",
        "LLMRouter",
        "build_llm_client",
    }:
        from .providers import llm as _llm

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
        from .providers import prompts as _prompts

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
    if name == "RunResult":
        from .contracts.runtime import RunResult

        return RunResult
    if name in {"CodeRunner", "DockerRunner"}:
        from .execution import runner as _runner

        return getattr(_runner, name)
    if name in {
        "CohortAuditor",
        "StatisticalValidator",
        "FigureContractQualityValidator",
        "FigureSourceDataValidator",
        "ClinicalConstraintValidator",
        "StatisticalGuard",
        "ConceptUsageAuditor",
        "LLMConceptAuditor",
        "ReplicationDesignAuditor",
        "ReplicationResultComparator",
        "PublicationClaimAuditor",
    }:
        from .audits import validators as _validators

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
        from .authority import runtime_artifacts as _runtime_artifacts

        return getattr(_runtime_artifacts, name)
    if name in {"VisualQAAuditor", "VLMVisualQAAdapter"}:
        from .gates import visual_qa as _visual_qa

        return getattr(_visual_qa, name)
    if name in {"PublicationFigureSkill", "PublicationFigureSkillResult"}:
        from .figures import skill as _figure_skill

        return getattr(_figure_skill, name)
    if name in {
        "FigureContract",
        "PanelSpec",
        "make_figure_contract",
        "audit_figure_contract",
        "apply_publication_style",
        "save_publication_figure",
        "audit_publication_exports",
        "audit_publication_exports_json",
    }:
        from .figures import publication as _pubfig

        return getattr(_pubfig, name)
    if name in {
        "EvidenceStore",
        "EvidenceEnforcementMode",
        "EvidenceEnforcementError",
    }:
        from .authority import evidence_store as _evidence

        return getattr(_evidence, name)
    if name == "ScientificClaim":
        from .authority.scientific_claims import ScientificClaim

        return ScientificClaim
    if name in {
        "EasyICUCasePackage",
        "index_export_package",
        "read_exported_concept",
        "build_lactate_map_vaso_cohort_from_export",
    }:
        from .case_plugins import builder as _case_builder

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
    if name in {
        "load_paper_source",
        "parse_paper_profile",
        "build_paper_replication_spec",
        "build_paper_result_ledger",
        "collect_easyicu_metrics",
        "compare_paper_to_easyicu",
        "compare_metric_values",
        "render_replication_report",
        "render_deviation_report",
        "render_showcase_manuscript",
    }:
        from .replication import paper as _paper_replication

        return getattr(_paper_replication, name)
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
        from .planning import analysis_types as _analysis_types

        return getattr(_analysis_types, name)
    if name in {
        "TaskCategory",
        "ICUAgentBenchMetricSpec",
        "ICUAgentBenchNumericBound",
        "ICUAgentBenchGoldAnswer",
        "ICUAgentBenchTask",
        "ICUAgentBenchSuite",
        "ICUAgentBenchTaskResult",
        "ICUAgentBenchReport",
        "default_icu_agent_bench_suite",
        "grade_bench_task",
        "aggregate_bench_report",
        "TaskMetricDelta",
        "BenchABComparison",
        "compare_bench_reports",
        "icu_agent_bench_markdown",
    }:
        from . import icu_agent_bench as _icu_agent_bench

        return getattr(_icu_agent_bench, name)
    if name in {
        "RunMemory",
        "StrategyCard",
        "MemoryScoreBreakdown",
        "MemoryRetrievalAuditEntry",
    }:
        from .learning import memory as _memory

        return getattr(_memory, name)
    if name in {"scaffold_to_latex", "latex_template_preamble"}:
        from .reporting import latex as _latex

        return getattr(_latex, name)
    if name in {"render_bibtex", "render_thebibliography_block"}:
        from .reporting import bibtex as _bibtex

        return getattr(_bibtex, name)
    if name in {"CostMeter", "MeteredClient"}:
        from .providers import cost as _cost

        return getattr(_cost, name)
    if name in {
        "ReproEnvelope",
        "ReproRecordingClient",
        "build_environment_snapshot",
        "envelope_role_resolver",
    }:
        from .replication import envelope as _repro

        return getattr(_repro, name)
    if name in {
        "MultipleTestingReport",
        "PValueRecord",
        "build_multiple_testing_report",
    }:
        from .methods import multiple_testing as _mt

        return getattr(_mt, name)
    if name in {
        "CausalAuditReport",
        "CausalLanguageHit",
        "EffectLabel",
        "label_effects",
        "run_causal_audit",
        "scan_manuscript_for_causal_language",
    }:
        from .review import causal_audit as _ca

        return getattr(_ca, name)
    if name in {
        "ChecklistItem",
        "ChecklistReport",
        "build_strobe_checklist",
        "build_tripod_ai_checklist",
        "build_internal_phenotype_checklist",
        "choose_checklist",
        "checklist_names_for_kind",
    }:
        from .reporting import reporting_checklist as _rc

        return getattr(_rc, name)
    if name in {
        "ReviewerComment",
        "ReviewerCritique",
        "ReviewerReport",
        "run_reviewer_round",
    }:
        from .reporting import reviewer as _rv

        return getattr(_rv, name)
    if name in {
        "ProvenanceBundle",
        "SourceFileRecord",
        "build_provenance_bundle",
        "hash_sources",
    }:
        from .authority import provenance as _prov

        return getattr(_prov, name)
    if name in {
        "EValueResult",
        "NegativeControlResult",
        "compute_e_value",
        "run_negative_control_check",
    }:
        from .methods import sensitivity as _sens

        return getattr(_sens, name)
    if name in {
        "NotebookStep",
        "build_notebook",
        "build_requirements_lockfile",
        "write_notebook",
    }:
        from .replication import notebook as _repro_art

        return getattr(_repro_art, name)
    if name in {
        "MICEImputationResult",
        "TippingPointResult",
        "mice_impute",
        "tipping_point_analysis",
    }:
        from .methods import missing_data as _md

        return getattr(_md, name)
    if name in {
        "SubgroupAnalysisResult",
        "SubgroupEstimate",
        "run_subgroup_analysis",
    }:
        from .methods import fairness as _fair

        return getattr(_fair, name)
    if name in {
        "HypothesisCandidate",
        "HypothesisFeasibilitySignal",
        "HypothesisGeneratorResult",
        "LITERATURE_SATURATION_SIGNAL_STATEMENT",
        "generate_hypotheses",
    }:
        from .discovery import hypothesis_generator as _hg

        return getattr(_hg, name)
    if name == "AnalysisPatternAuditor":
        from .audits.patterns import AnalysisPatternAuditor

        return AnalysisPatternAuditor
    if name in {"PDFRenderResult", "render_pdf_for_run"}:
        from .reporting import pdf_render as _pdf

        return getattr(_pdf, name)
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
    if name == "PipelineConfig":
        from .orchestration.config import PipelineConfig

        return PipelineConfig
    if name == "PipelineServices":
        from .orchestration.services import PipelineServices

        return PipelineServices
    if name in {
        "SubmissionProfile",
        "NPJ_DM_2026_05",
        "NPJ_DM_2026_06",
        "NPJ_DM_2026_07",
        "NPJ_DM_2026_07_16",
        "NPJ_DM_2026_07_17",
        "NPJ_DM_2026_07_18",
        "NPJ_DM_2026_07_19",
        "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_16",
        "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_17",
        "E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19",
        "E1_REVIEWED_DEMO_2026_08_17",
        "E1_REVIEWED_DEMO_2026_08_19",
        "NPJ_DM_2026_07_21_KNOW_HOW",
        "NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV",
        "NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV",
        "NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV",
        "DEFAULT_SUBMISSION_PROFILE_REF",
        "SUBMISSION_PROFILE_REGISTRY",
        "get_submission_profile",
    }:
        from .orchestration import profiles as _profiles

        return getattr(_profiles, name)
    raise AttributeError(f"module 'easyicu.research_agent' has no attribute {name!r}")
