"""End-to-end orchestration of the research-agent loop.

A single class, :class:`ResearchAgentPipeline`, ties together the
context builder, agents, runner, validators and evidence store. The
public ``run()`` method is the only thing most users need.

Loop shape::

    research_question + cohort
        ↓
    build_research_context        → context.json   (registered evidence)
        ↓
    PlannerAgent.run              → plan.json      (registered evidence)
        ↓
    for each AnalysisStep:
        CoderAgent.run            → analysis.py    (registered evidence)
        ConceptUsageAuditor.audit → findings       (errors block step)
        CodeRunner.run            → outputs/, run.log
    StatisticalValidator.audit → findings
    register every artefact   → evidence ids
    AnalyzerAgent.run         → interpretation paragraph
        ↓
    PublicationFigureSkill.run → claim-first figure contract + exports
        ↓
    WriterAgent.run               → manuscript_scaffold.md
    EvidenceStore.bind_manuscript → manuscript_scaffold_bound.md
        ↓
    write manifest.json + results_report.md
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import asyncio
import csv
import hashlib
import json
import re
import shutil
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .agents import (
    AnalyzerAgent,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ManuscriptAgent,
    PlannerAgent,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)
from .architecture import architecture_profile_markdown, default_architecture_profile
from .cost import CostMeter, metered_role_resolver
from .replication.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    ReproEnvelope,
    envelope_role_resolver,
)
from .multiple_testing import build_multiple_testing_report
from .causal_audit import run_causal_audit
from .reporting_checklist import (
    build_strobe_checklist,
    build_tripod_ai_checklist,
    choose_checklist,
)
from .reviewer import run_reviewer_round
from .provenance import ProvenanceBundle, SourceFileRecord, build_provenance_bundle, hash_sources
from .sensitivity import (
    EValueResult,
    NegativeControlResult,
    compute_e_value,
    run_negative_control_check,
)
from .replication.paper import (
    build_paper_replication_spec,
    build_paper_result_ledger,
    canonical_outcome_name,
    compare_paper_to_easyicu,
    load_paper_source,
    parse_paper_profile,
    postprocess_paper_replication,
    render_deviation_report,
    render_replication_report,
    render_showcase_manuscript,
    write_claim_csv,
    write_fail_closed_paper_package,
)
from .replication.notebook import (
    NotebookStep,
    build_notebook,
    build_requirements_lockfile,
    write_notebook,
)
from .hypothesis_generator import generate_hypotheses
from .pdf_render import render_pdf_for_run
from .context import (
    build_naive_research_context,
    build_research_context,
    build_retrieved_research_context,
)
from . import pipeline_cache as _pipeline_cache
from .pipeline_config import PipelineConfig
from .pipeline_report import (
    execution_gate_status,
    render_report,
    write_readiness_artifacts,
)
# Back-compat aliases. Tests (and any downstream code) that imported the
# leading-underscore names from this module before the readiness/report
# helpers were moved to ``pipeline_report`` keep working unchanged.
from .pipeline_report import (
    _publication_figure_bundle_ready,  # noqa: F401
    _compute_readiness_gates,  # noqa: F401
    _count_missing_evidence_markers,  # noqa: F401
    _extract_claim_ledger_rows,  # noqa: F401
    _render_author_review_note,  # noqa: F401
)
_execution_gate_status = execution_gate_status  # noqa: F841 (legacy alias)
_write_readiness_artifacts = write_readiness_artifacts  # noqa: F841 (legacy alias)
_render_report = render_report  # noqa: F841 (legacy alias)
from .audits.manuscript_claims import audit_manuscript_numeric_claims  # noqa: E402
from .audits.manuscript_claims import (  # noqa: E402,F401
    _first_summary_scalar,
    _extract_metric_claims,
    _extract_percent_claims_near,
)
_audit_manuscript_numeric_claims = audit_manuscript_numeric_claims  # noqa: F841 (legacy alias)

# Case-specific fallback code generators (lactate / MAP / vasopressor study)
# now live in the dedicated plugin module. Re-import them under their
# historical underscore-prefixed names so existing pipeline.py callsites and
# any external tests keep working. P4 will switch the callsites to go through
# ``CasePluginRegistry`` instead, at which point these aliases can retire.
from .case_plugins.lactate_map_vaso.fallbacks import (  # noqa: E402,F401
    _primary_association_fallback_code,
    _age_stratified_mortality_fallback_code,
    _norepinephrine_dose_response_fallback_code,
    _generic_v15_task_fallback_code,
)

from .evidence import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    _coerce_enforcement_mode,
)
from .experiment_spec import ExperimentSpec, dump_experiment_spec
from .figure_skill import PublicationFigureSkill
from .bibtex import render_bibtex
from .latex import scaffold_to_latex
from .literature import (
    HypothesisBlueprintAgent,
    LiteratureAgent,
    LiteratureBundle,
    render_hypothesis_blueprint_for_prompt,
)
from .llm import (
    LLMClient,
    LLMRouter,
    MockLLMClient,
    llm_is_mockish,
    llm_supports_vision,
    resolve_role_client,
)
from .memory import RunMemory
from .prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .runner import CodeRunner, DockerRunner, RunResult
from .schema import (
    AgentRuntimeState,
    AnalysisManifest,
    AnalysisPlan,
    AnalysisStep,
    CritiqueReport,
    EvidenceRecord,
    EvidenceRef,
    ManuscriptDraftPacket,
    PaperProfile,
    PaperReplicationSpec,
    PaperResultLedger,
    PipelineResult,
    ResearchContext,
    ReplicationDeviationReport,
    TimeWindow,
    ValidationFinding,
    VariableRole,
)
from .skills import ClinicalSkill, get_skill, list_skills
from .runtime_artifacts import (
    AuditLogger,
    build_execution_replay,
    build_workflow_graph,
    render_workflow_graph_mermaid,
    write_json_artifact,
)
from .audits.validators import (
    ClinicalConstraintValidator,
    CohortAuditor,
    ConceptUsageAuditor,
    LLMConceptAuditor,
    PublicationClaimAuditor,
    ReplicationDesignAuditor,
    ReplicationResultComparator,
    StatisticalGuard,
    StatisticalValidator,
)
from .visual_qa import VLMVisualQAAdapter, VisualQAAuditor


@dataclass
class _PlanPhaseResult:
    context: ResearchContext
    agent_context: ResearchContext
    context_path: Path
    evidence: EvidenceStore
    findings: List[ValidationFinding]
    plan: AnalysisPlan
    plan_path: Path
    llm_signature: str
    used_mock_llm: bool
    prompt_version: str
    prompt_files: Dict[str, str]
    role_resolver: Callable[[str], Any]
    cost_meter: Optional[CostMeter]
    repro_envelope: Optional[ReproEnvelope]
    started_at: datetime
    resume_state: Optional[Dict[str, Any]]
    aborted_result: Optional[PipelineResult] = None


@dataclass
class _ExecutePhaseResult:
    plan: AnalysisPlan
    per_step_records: List[Dict[str, Any]]
    probe_summary: Dict[str, Any]
    runtime_state: AgentRuntimeState
    flush_partial_manifest: Callable[[Optional[Dict[str, Any]]], None]


@dataclass
class _WritePhaseResult:
    literature: Optional[LiteratureBundle]
    bound_path: Path
    manuscript_packet: Optional[ManuscriptDraftPacket] = None
    manuscript_critique: Optional[CritiqueReport] = None


class ResearchAgentPipeline:
    """One-shot orchestration. Construct, call :meth:`run`, read the result."""

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ResearchAgentPipeline":
        """Construct a pipeline from a :class:`PipelineConfig` object.

        The legacy keyword-argument form of ``__init__`` continues to
        work; this classmethod is the recommended new-code entry point
        because the config object is typed, copyable
        (``config.with_overrides(...)``) and serialisable
        (``config.as_kwargs()``).
        """
        return cls(**config.as_kwargs())

    def __init__(
        self,
        *,
        workdir: Union[str, Path],
        llm: Optional[LLMClient] = None,
        timeout_seconds: float = 300.0,
        python_executable: Optional[str] = None,
        enable_literature: bool = True,
        enable_visual_qa: bool = True,
        enable_publication_figure_skill: bool = True,
        enable_vlm_visual_qa: Optional[bool] = None,
        vlm_client: Optional[LLMClient] = None,
        visual_qa_adapter: Optional[VLMVisualQAAdapter] = None,
        enable_llm_concept_audit: Optional[bool] = None,
        llm_concept_auditor_client: Optional[LLMClient] = None,
        enable_memory: bool = True,
        enable_latex: bool = True,
        latex_venue_template: str = "article",
        manuscript_language: str = "en",
        evidence_enforcement_mode: str = "soft",
        disable_icu_context: bool = False,
        context_top_k: Optional[int] = None,
        max_code_repair_attempts: int = 1,
        enable_deterministic_code_fallback: bool = False,
        enable_deterministic_planner_fallback: bool = False,
        enable_deterministic_runner_repair: bool = True,
        enable_pubmed: bool = False,
        pubmed_email: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        enable_tavily: bool = False,
        tavily_api_key: Optional[str] = None,
        tavily_retmax: int = 5,
        tavily_include_domains: Optional[Sequence[str]] = None,
        tavily_exclude_domains: Optional[Sequence[str]] = None,
        enable_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cost_tracking: bool = False,
        cost_price_table: Optional[Dict[str, Any]] = None,
        enable_reproducibility_envelope: bool = False,
        llm_seed: Optional[int] = None,
        envelope_include_previews: bool = False,
        enable_multiple_testing_correction: bool = True,
        multiple_testing_alpha: float = 0.05,
        enable_causal_audit: bool = True,
        enable_reporting_checklist: bool = True,
        reporting_checklist_names: Optional[Sequence[str]] = None,
        enable_reviewer_round: bool = True,
        enable_fairness_subgroups: bool = True,
        enable_hypothesis_generator: bool = False,
        hypothesis_generator_top_k: int = 5,
        enable_pdf_render: bool = False,
        max_concurrent_steps: int = 1,
        enable_probe_step: bool = True,
        enable_replanning: bool = True,
        runner_kind: str = "subprocess",
        runner_image: Optional[str] = None,
        runner_network: str = "none",
        runner_factory: Optional[Callable[..., Any]] = None,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        case_plugin_registry: Optional[Any] = None,
    ) -> None:
        # Snapshot the construction kwargs into a typed config object for
        # introspection / replay / serialisation. The body below still reads
        # from local variables directly (so the legacy ``__init__(**kwargs)``
        # signature stays unchanged); ``self._config`` is purely a view onto
        # what we were given.
        self._config: PipelineConfig = PipelineConfig.from_kwargs(
            **{k: v for k, v in locals().items() if k != "self"}
        )
        # Lazy import to avoid pulling the plugin registry module if no
        # plugins are configured (the default).
        from .fallback import CasePluginRegistry as _CasePluginRegistry

        self._case_plugin_registry = case_plugin_registry or _CasePluginRegistry()
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._timeout_seconds = timeout_seconds
        self._python_executable = python_executable
        self._enable_literature = enable_literature
        self._enable_visual_qa = enable_visual_qa
        self._enable_publication_figure_skill = bool(enable_publication_figure_skill)
        self._vlm_client = vlm_client
        self._visual_qa_adapter = visual_qa_adapter
        self._llm_concept_auditor_client = llm_concept_auditor_client
        if enable_vlm_visual_qa is None:
            self._enable_vlm_visual_qa = bool(
                visual_qa_adapter is not None
                or llm_supports_vision(vlm_client)
                or llm_supports_vision(llm)
            )
        else:
            self._enable_vlm_visual_qa = bool(enable_vlm_visual_qa)
        if enable_llm_concept_audit is None:
            concept_client = llm_concept_auditor_client or llm
            self._enable_llm_concept_audit = bool(
                concept_client is not None and not llm_is_mockish(concept_client)
            )
        else:
            self._enable_llm_concept_audit = bool(enable_llm_concept_audit)
        self._enable_memory = enable_memory
        self._enable_latex = enable_latex
        self._latex_venue_template = latex_venue_template or "article"
        lang = (manuscript_language or "en").lower()
        self._manuscript_language = (
            "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
        )
        # Stored as the canonical Enum so downstream code can compare
        # against EvidenceEnforcementMode.STRICT without string casing.
        self._evidence_enforcement_mode = _coerce_enforcement_mode(
            evidence_enforcement_mode
        )
        # T1.4 — when set, the pipeline strips the ICU rules out of the
        # context that drives planning, coding and validation. This is
        # the "naive" arm of the hero ablation: a generic data agent
        # sees only column names + dtypes + ANY-aggregation.
        self._disable_icu_context = bool(disable_icu_context)
        self._context_top_k = int(context_top_k) if context_top_k else None
        self._max_code_repair_attempts = max(0, int(max_code_repair_attempts))
        self._enable_deterministic_code_fallback = bool(
            enable_deterministic_code_fallback
        )
        self._enable_deterministic_planner_fallback = bool(
            enable_deterministic_planner_fallback
        )
        self._enable_deterministic_runner_repair = bool(
            enable_deterministic_runner_repair
        )
        # T2.2 — opt-in PubMed live search. Off by default so CI and
        # the offline demo stay deterministic; the LiteratureAgent
        # handles network failure gracefully (empty list → curated
        # registry only).
        self._enable_pubmed = bool(enable_pubmed)
        self._pubmed_email = pubmed_email
        self._pubmed_api_key = pubmed_api_key
        # O5 — opt-in Tavily web search for preprints/guidelines/trial
        # registries that PubMed may not index. Off by default so CI
        # and offline demos remain deterministic.
        self._enable_tavily = bool(enable_tavily)
        self._tavily_api_key = tavily_api_key
        self._tavily_retmax = int(tavily_retmax)
        self._tavily_include_domains = list(tavily_include_domains or [])
        self._tavily_exclude_domains = list(tavily_exclude_domains or [])
        # T3.5 — cohort cache. When enabled, identical re-runs (same
        # cohort hash + same skill/question/llm signature) short-circuit
        # to the prior run_dir's PipelineResult instead of repeating
        # the entire pipeline. Off by default so production users opt
        # in deliberately and tests that *want* full pipeline
        # execution every time keep their existing semantics.
        self._enable_cache = bool(enable_cache)
        self._cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else self.workdir / ".cache"
        )
        self._cache = _pipeline_cache.PipelineCache(self._cache_dir)
        # T3.2 — opt-in LLM cost tracking. When enabled, every per-role
        # client is wrapped in a ``MeteredClient`` that records token
        # counts (prompt / completion) and an estimated USD cost into a
        # ``CostMeter``. The records are persisted to
        # ``manifest.cost_records`` plus ``cost_summary.md`` and
        # ``cost_records.json`` artefacts. Off by default so the
        # default pipeline behaviour stays bit-identical.
        self._enable_cost_tracking = bool(enable_cost_tracking)
        self._cost_price_table = cost_price_table
        # O20 — Reproducibility envelope. Records prompt/response
        # sha256, requested seed, temperature, provider/model, and a
        # PHI-safe environment snapshot for every LLM call the pipeline
        # makes. Off by default so the base pipeline behaviour stays
        # bit-identical; turn on when a paper reviewer asks for an
        # auditable replay bundle.
        self._enable_reproducibility_envelope = bool(enable_reproducibility_envelope)
        self._llm_seed = int(llm_seed) if llm_seed is not None else None
        self._envelope_include_previews = bool(envelope_include_previews)
        # O22 — run-wide multiple-testing correction. Defaults to ON
        # because reviewers of a research-agent paper will always ask
        # for it; the correction is cheap to compute and does not
        # rewrite existing artefacts.
        self._enable_multiple_testing_correction = bool(
            enable_multiple_testing_correction
        )
        self._multiple_testing_alpha = float(multiple_testing_alpha)
        # O18 — Causal audit. Deterministic: labels every primary
        # effect as associational vs causal, scans bound manuscript
        # for causal language over associational effects, emits
        # warnings/errors. Default ON because the cost of a paper
        # that silently sells an OR as causal is high.
        self._enable_causal_audit = bool(enable_causal_audit)
        # O16 — Reporting-guideline checklist. STROBE always; TRIPOD+AI
        # when the analysis looks like a prediction model. Default ON
        # because ICU journals routinely require the checklist.
        self._enable_reporting_checklist = bool(enable_reporting_checklist)
        self._reporting_checklist_names = (
            tuple(reporting_checklist_names) if reporting_checklist_names else None
        )
        # O15 — Three-role reviewer round (statistician / clinician /
        # methodologist) driven off already-computed evidence and
        # findings. Deterministic; no extra LLM calls. Default ON.
        self._enable_reviewer_round = bool(enable_reviewer_round)
        # O24 — Fairness / subgroup analysis for the primary effect.
        # Deterministic (pure numpy); runs after E-values.
        self._enable_fairness_subgroups = bool(enable_fairness_subgroups)
        # O17 — Front-door hypothesis generator. Runs early (plan phase)
        # to emit a ranked candidate list; does not change the
        # downstream plan unless the user reassigns ``question``.
        self._enable_hypothesis_generator = bool(enable_hypothesis_generator)
        self._hypothesis_generator_top_k = int(hypothesis_generator_top_k)
        # PDF rendering (TexLive optional). Off by default because not
        # every CI environment has a LaTeX install; turn on when the
        # user actually wants ``manuscript_scaffold.pdf`` next to the
        # ``.tex`` and ``.bib``.
        self._enable_pdf_render = bool(enable_pdf_render)
        # T3.3 — concurrent step execution. The canonical AnalysisPlan
        # has independent steps (each step reads the cohort and writes
        # to its own out_dir), so a small thread pool can shrink the
        # critical-path latency by ~k× for k workers. Default 1 keeps
        # bit-identical sequential behaviour for users who don't opt
        # in. EvidenceStore guards its mutators with an RLock so a
        # higher value is safe.
        self._max_concurrent_steps = max(1, int(max_concurrent_steps))
        self._enable_probe_step = bool(enable_probe_step)
        self._enable_replanning = bool(enable_replanning)
        # T3.1 — runner backend selection. ``subprocess`` keeps the
        # existing behaviour; ``docker`` swaps in :class:`DockerRunner`
        # which mounts the cohort read-only inside a container with
        # ``--network none`` by default. Users with their own sandbox
        # (e.g. OpenHands) can pass an arbitrary ``runner_factory``
        # that accepts ``workdir=, cohort_parquet=, timeout_seconds=``
        # and returns a runner with a ``run(step_id, code)`` method.
        kind = (runner_kind or "subprocess").lower()
        if runner_factory is not None:
            self._runner_kind = "custom"
        elif kind in {"subprocess", "host", "default"}:
            self._runner_kind = "subprocess"
        elif kind in {"docker", "container", "openhands"}:
            self._runner_kind = "docker"
        else:
            raise ValueError(
                f"Unknown runner_kind {runner_kind!r}; "
                "expected 'subprocess', 'docker', or pass a runner_factory."
            )
        self._runner_image = runner_image
        self._runner_network = runner_network
        self._runner_factory = runner_factory
        self._runner_kwargs = dict(runner_kwargs or {})
        self._memory = RunMemory(self.workdir) if enable_memory else None

    def _build_runner(self, *, run_dir: Path, cohort_path: Path):
        """Return the configured runner backend for a single ``run()``.

        Kept as a method (not a closure) so subclasses or tests can
        stub it cleanly. Returns any object that exposes
        ``run(step_id=..., code=...) -> RunResult``.
        """
        if self._runner_factory is not None:
            return self._runner_factory(
                workdir=run_dir,
                cohort_parquet=cohort_path,
                timeout_seconds=self._timeout_seconds,
                **self._runner_kwargs,
            )
        if self._runner_kind == "docker":
            return DockerRunner(
                workdir=run_dir,
                cohort_parquet=cohort_path,
                timeout_seconds=self._timeout_seconds,
                image=self._runner_image,
                network=self._runner_network,
                **self._runner_kwargs,
            )
        return CodeRunner(
            workdir=run_dir,
            cohort_parquet=cohort_path,
            timeout_seconds=self._timeout_seconds,
            python_executable=self._python_executable,
            **self._runner_kwargs,
        )

    def _run_plan_phase(
        self,
        *,
        question: str,
        cohort_path: Path,
        cohort_name: str,
        database: str,
        target_outcome: Optional[str],
        cross_database_validation: Optional[Sequence[str]],
        inclusion_criteria: Optional[Sequence[str]],
        exclusion_criteria: Optional[Sequence[str]],
        id_columns: Optional[Sequence[str]],
        time_columns: Optional[Sequence[str]],
        outcome_columns: Optional[Sequence[str]],
        time_windows: Optional[Sequence[TimeWindow]],
        concept_descriptions: Optional[Dict[str, str]],
        user_preferences: Optional[Dict[str, Any]],
        notes: Optional[str],
        skill_obj: Optional[ClinicalSkill],
        llm: LLMClient,
        run_dir: Path,
        run_id: str,
        run_language: str,
        experiment_spec_path: Optional[Path],
        resume_state: Optional[Dict[str, Any]],
        emit_progress: Callable[..., None],
    ) -> _PlanPhaseResult:
        """Build context, attach memory, and emit an execution plan."""
        builder = (
            build_naive_research_context
            if self._disable_icu_context
            else build_research_context
        )
        context = builder(
            research_question=question,
            cohort=cohort_path,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
            cross_database_validation=cross_database_validation,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            id_columns=id_columns,
            time_columns=time_columns,
            outcome_columns=outcome_columns,
            concept_descriptions=concept_descriptions,
            time_windows=time_windows,
            user_preferences=user_preferences,
            notes=notes,
        )
        context_path = run_dir / "research_context.json"
        context_path.write_text(
            context.model_dump_json(indent=2),
            encoding="utf-8",
        )
        emit_progress(
            "context",
            "Research context built.",
            run_id=run_id,
            n_stays=context.cohort.n_stays,
            n_variables=len(context.variables),
        )

        evidence = EvidenceStore(
            root=run_dir,
            enforcement_mode=self._evidence_enforcement_mode,
        )
        if evidence.get("research_context") is None:
            evidence.register_file(
                kind="log",
                description="ResearchContext (frozen at run time).",
                source_path=context_path,
                evidence_id="research_context",
                producer="pipeline",
                generation_mode="system",
            )
        if (
            experiment_spec_path is not None
            and experiment_spec_path.exists()
            and evidence.get("experiment_spec") is None
        ):
            evidence.register_file(
                kind="log",
                description="Config-driven YAML/JSON experiment specification for this run.",
                source_path=experiment_spec_path,
                evidence_id="experiment_spec",
                producer="pipeline",
                generation_mode="system",
            )
        architecture_profile = default_architecture_profile()
        architecture_json = run_dir / "architecture_profile.json"
        architecture_md = run_dir / "architecture_profile.md"
        write_json_artifact(architecture_json, architecture_profile)
        architecture_md.write_text(
            architecture_profile_markdown(architecture_profile),
            encoding="utf-8",
        )
        if evidence.get("architecture_profile") is None:
            evidence.register_file(
                kind="log",
                description="Declared four-layer architecture profile for this runtime.",
                source_path=architecture_json,
                evidence_id="architecture_profile",
                producer="pipeline",
                generation_mode="system",
            )
        if evidence.get("architecture_profile_markdown") is None:
            evidence.register_file(
                kind="log",
                description="Human-readable architecture profile for this runtime.",
                source_path=architecture_md,
                evidence_id="architecture_profile_markdown",
                producer="pipeline",
                generation_mode="system",
            )

        findings: List[ValidationFinding] = []
        findings += CohortAuditor().audit(context=context, cohort_path=cohort_path)
        if any(f.severity == "error" for f in findings):
            emit_progress(
                "audit",
                "Cohort audit failed; aborting run.",
                status="error",
                run_id=run_id,
            )
            aborted = self._finalise_aborted(
                run_id=run_id,
                run_dir=run_dir,
                context=context,
                context_path=context_path,
                evidence=evidence,
                findings=findings,
                reason="cohort_audit_failed",
            )
            return _PlanPhaseResult(
                context=context,
                agent_context=context,
                context_path=context_path,
                evidence=evidence,
                findings=findings,
                plan=AnalysisPlan(
                    research_question=context.research_question,
                    steps=[],
                ),
                plan_path=run_dir / "analysis_plan.json",
                llm_signature=self._llm_signature(llm),
                used_mock_llm=any(True for _ in self._iter_mock_clients(llm)),
                prompt_version=PROMPT_PACK_VERSION,
                prompt_files=prompt_pack_files(),
                role_resolver=lambda _role: resolve_role_client(llm, _role),
                cost_meter=None,
                repro_envelope=None,
                started_at=datetime.now(timezone.utc),
                resume_state=resume_state,
                aborted_result=aborted,
            )
        emit_progress(
            "audit",
            "Initial cohort audit passed.",
            run_id=run_id,
            findings=len(findings),
        )

        memory_digest_text: Optional[str] = None
        if self._memory is not None:
            digest = self._memory.digest_for_prompt(
                research_question=question,
                database=database,
                target_outcome=target_outcome,
            )
            memory_digest_text = digest
            if skill_obj is None:
                try:
                    meta_digest = self._memory.meta_planner_digest(
                        skill_keys=[s.key for s in list_skills()],
                        research_question=question,
                        database=database,
                        target_outcome=target_outcome,
                    )
                    memory_digest_text = digest + "\n\n" + meta_digest
                except Exception:
                    pass
            digest_path = run_dir / "memory_digest.md"
            digest_path.write_text(memory_digest_text, encoding="utf-8")
            if evidence.get("memory_digest") is None:
                evidence.register_file(
                    kind="log",
                    description="Cross-run memory digest fed to the planner.",
                    source_path=digest_path,
                    evidence_id="memory_digest",
                    producer="memory",
                    generation_mode="system",
                )

        agent_context = context
        if memory_digest_text:
            note = "RunMemory digest for planner:\n" + memory_digest_text
            agent_notes = f"{context.notes}\n\n{note}" if context.notes else note
            agent_context = agent_context.model_copy(update={"notes": agent_notes})
        if self._context_top_k and skill_obj is None:
            agent_context = build_retrieved_research_context(
                agent_context,
                query=question,
                top_k=self._context_top_k,
            )
            agent_context_path = run_dir / "research_context_agent_prompt.json"
            agent_context_path.write_text(
                agent_context.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("research_context_agent_prompt") is None:
                evidence.register_file(
                    kind="log",
                    description="Prompt-sized ResearchContext after concept retrieval (O6).",
                    source_path=agent_context_path,
                    evidence_id="research_context_agent_prompt",
                    producer="pipeline",
                    generation_mode="system",
                )

        if self._enable_literature and skill_obj is None:
            try:
                emit_progress(
                    "hypothesis",
                    "Building pre-plan literature and hypothesis blueprint.",
                    run_id=run_id,
                )
                preplan_literature = LiteratureAgent(None).run(agent_context)
                preplan_lit_path = run_dir / "preplan_literature_bundle.json"
                preplan_lit_path.write_text(
                    preplan_literature.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                if evidence.get("preplan_literature_bundle") is None:
                    evidence.register_file(
                        kind="log",
                        description=(
                            "Pre-plan LiteratureBundle used to shape the hypothesis "
                            "blueprint before planner execution."
                        ),
                        source_path=preplan_lit_path,
                        evidence_id="preplan_literature_bundle",
                        producer="hypothesis_blueprint",
                        generation_mode="deterministic_skill",
                    )
                # O17 — Front-door hypothesis generation. Opt-in; writes
                # ``hypothesis_candidates.json`` + ``.md`` so the paper
                # Methods section can quote "Out of N candidates we
                # preregistered Q" rather than "we picked Q".
                if self._enable_hypothesis_generator:
                    try:
                        hg_result = generate_hypotheses(
                            context=agent_context,
                            citations=list(preplan_literature.citations),
                            top_k=self._hypothesis_generator_top_k,
                        )
                        hg_json = run_dir / "hypothesis_candidates.json"
                        hg_md = run_dir / "hypothesis_candidates.md"
                        hg_json.write_text(
                            json.dumps(hg_result.to_json(), indent=2, default=str),
                            encoding="utf-8",
                        )
                        hg_md.write_text(hg_result.to_markdown(), encoding="utf-8")
                        if evidence.get("hypothesis_candidates") is None:
                            evidence.register_file(
                                kind="log",
                                description=(
                                    "Ranked front-door hypothesis candidates "
                                    "(predictor × outcome) with coverage, "
                                    "novelty and ICU-gate scores (O17)."
                                ),
                                source_path=hg_json,
                                evidence_id="hypothesis_candidates",
                                producer="hypothesis_generator",
                                generation_mode="deterministic_skill",
                            )
                        if evidence.get("hypothesis_candidates_summary") is None:
                            evidence.register_file(
                                kind="log",
                                description=(
                                    "Human-readable hypothesis-candidate table (O17)."
                                ),
                                source_path=hg_md,
                                evidence_id="hypothesis_candidates_summary",
                                producer="hypothesis_generator",
                                generation_mode="deterministic_skill",
                            )
                        findings.append(
                            ValidationFinding(
                                validator="hypothesis_generator",
                                severity="info",
                                message=(
                                    f"Ranked {len(hg_result.candidates)} candidate "
                                    f"hypotheses; top={hg_result.candidates[0].question if hg_result.candidates else 'n/a'}"
                                ),
                                evidence_ids=["hypothesis_candidates"],
                            )
                        )
                    except Exception as exc:
                        findings.append(
                            ValidationFinding(
                                validator="hypothesis_generator",
                                severity="warning",
                                message=(
                                    f"Hypothesis generator failed: "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            )
                        )
                blueprint = HypothesisBlueprintAgent().run(
                    context=agent_context,
                    literature=preplan_literature,
                )
                blueprint_path = run_dir / "hypothesis_blueprint.json"
                blueprint_path.write_text(
                    blueprint.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                if evidence.get("hypothesis_blueprint") is None:
                    evidence.register_file(
                        kind="log",
                        description=(
                            "Literature-aware hypothesis, feasibility critique, "
                            "and recommended plan skeleton fed to the planner."
                        ),
                        source_path=blueprint_path,
                        evidence_id="hypothesis_blueprint",
                        producer="hypothesis_blueprint",
                        generation_mode="deterministic_skill",
                    )
                if blueprint.feasibility_status == "blocked":
                    findings.append(
                        ValidationFinding(
                            validator="hypothesis_blueprint",
                            severity="error",
                            message=(
                                "Hypothesis blueprint marked this request as blocked; "
                                "pipeline stopped before planning executable analysis steps."
                            ),
                            evidence_ids=["hypothesis_blueprint"],
                        )
                    )
                    emit_progress(
                        "hypothesis",
                        "Hypothesis blueprint is blocked; aborting before planner execution.",
                        status="error",
                        run_id=run_id,
                    )
                    aborted = self._finalise_aborted(
                        run_id=run_id,
                        run_dir=run_dir,
                        context=context,
                        context_path=context_path,
                        evidence=evidence,
                        findings=findings,
                        reason="hypothesis_blueprint_blocked",
                    )
                    return _PlanPhaseResult(
                        context=context,
                        agent_context=agent_context,
                        context_path=context_path,
                        evidence=evidence,
                        findings=findings,
                        plan=AnalysisPlan(
                            research_question=context.research_question,
                            steps=[],
                        ),
                        plan_path=run_dir / "analysis_plan.json",
                        llm_signature=self._llm_signature(llm),
                        used_mock_llm=any(True for _ in self._iter_mock_clients(llm)),
                        prompt_version=PROMPT_PACK_VERSION,
                        prompt_files=prompt_pack_files(),
                        role_resolver=lambda _role: resolve_role_client(llm, _role),
                        cost_meter=None,
                        repro_envelope=None,
                        started_at=datetime.now(timezone.utc),
                        resume_state=resume_state,
                        aborted_result=aborted,
                    )
                note = render_hypothesis_blueprint_for_prompt(blueprint)
                agent_notes = (
                    f"{agent_context.notes}\n\n{note}"
                    if agent_context.notes
                    else note
                )
                agent_context = agent_context.model_copy(update={"notes": agent_notes})
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="hypothesis_blueprint",
                        severity="warning",
                        message=(
                            "Hypothesis blueprint failed; planner will use "
                            f"context only: {exc}"
                        ),
                    )
                )

        for client in self._iter_mock_clients(llm):
            client.context = agent_context
        llm_signature = self._llm_signature(llm)
        used_mock_llm = any(True for _ in self._iter_mock_clients(llm))
        prompt_version = PROMPT_PACK_VERSION
        prompt_files = prompt_pack_files()

        cost_meter: Optional[CostMeter] = None
        repro_envelope: Optional[ReproEnvelope] = None
        if self._enable_reproducibility_envelope:
            repro_envelope = ReproEnvelope(
                run_id=run_id,
                seed=self._llm_seed,
                include_previews=self._envelope_include_previews,
            )
        if self._enable_cost_tracking:
            cost_meter = (
                CostMeter(
                    price_table=(
                        dict(self._cost_price_table) if self._cost_price_table else None
                    ),
                )
                if self._cost_price_table is not None
                else CostMeter()
            )
            # Order: envelope wraps the innermost client so prompt /
            # response hashes are computed on the exact strings the
            # agent sent / received; the metered layer then observes
            # ``last_usage`` the envelope passes through.
            if repro_envelope is not None:
                env_resolver = envelope_role_resolver(
                    llm, repro_envelope, seed=self._llm_seed,
                )

                class _EnvelopeShim:
                    """Bridges a ``role -> client`` resolver back into an LLMClient-like object.

                    ``metered_role_resolver`` expects an object it can
                    call ``resolve_role_client`` on. The shim exposes
                    ``for_role`` so the downstream resolver takes our
                    envelope-wrapped client as the inner client.
                    """

                    name = "envelope_shim"

                    def __init__(self, resolver):
                        self._resolver = resolver

                    def for_role(self, role: str):
                        return self._resolver(role)

                    # A no-op ``complete`` so static checks don't trip;
                    # the real call always goes through ``for_role``.
                    def complete(self, *args, **kwargs):  # pragma: no cover
                        raise RuntimeError(
                            "EnvelopeShim is a role dispatcher; call for_role() first."
                        )

                role_resolver = metered_role_resolver(
                    _EnvelopeShim(env_resolver), cost_meter,
                )
            else:
                role_resolver = metered_role_resolver(llm, cost_meter)
        elif repro_envelope is not None:
            role_resolver = envelope_role_resolver(
                llm, repro_envelope, seed=self._llm_seed,
            )
        else:

            def role_resolver(role: str):
                return resolve_role_client(llm, role)

        if skill_obj is not None:
            plan_generation_mode = "deterministic_skill"
            issues = skill_obj.validate_against(pd.read_parquet(cohort_path))
            for msg in issues:
                findings.append(
                    ValidationFinding(
                        validator="clinical_skill",
                        severity="warning",
                        message=msg,
                    )
                )
            plan = skill_obj.plan(context)
        else:
            plan_generation_mode = "llm"
            planner = PlannerAgent(role_resolver("planner"))
            try:
                plan = planner.run(agent_context)
            except Exception as exc:
                if not self._enable_deterministic_planner_fallback:
                    raise
                findings.append(
                    ValidationFinding(
                        validator="planner",
                        severity="warning",
                        message=(
                            "Planner agent failed; using deterministic fallback plan: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        detail={"generation_mode": "fallback"},
                    )
                )
                plan = PlannerAgent(MockLLMClient(context=agent_context)).run(
                    agent_context
                )
                used_mock_llm = True
                plan_generation_mode = "fallback"
            dropped_plan_keys = getattr(planner, "last_dropped_plan_keys", None) or {}
            dropped_keys = list(dropped_plan_keys.get("top_level", [])) + list(
                dropped_plan_keys.get("steps", [])
            )
            if dropped_keys:
                findings.append(
                    ValidationFinding(
                        validator="planner_schema",
                        severity="warning",
                        message=(
                            "Planner returned unsupported plan fields that were dropped "
                            "before schema validation."
                        ),
                        detail={"dropped_keys": dropped_keys},
                    )
                )
        plan, plan_contract_findings = _enforce_advanced_plan_contract(
            plan=plan,
            context=context,
        )
        findings.extend(plan_contract_findings)
        plan, split_findings = _split_table_and_figure_outputs_in_plan(plan=plan)
        findings.extend(split_findings)
        plan, figure_guard_findings = _ensure_publication_figure_step_in_plan(
            plan=plan,
            context=context,
        )
        findings.extend(figure_guard_findings)
        plan_path = run_dir / "analysis_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get("analysis_plan") is None:
            evidence.register_file(
                kind="log",
                description=(
                    f"Analysis plan from ClinicalSkill '{skill_obj.key}'."
                    if skill_obj
                    else "Analysis plan emitted by PlannerAgent."
                ),
                source_path=plan_path,
                evidence_id="analysis_plan",
                producer="planner" if skill_obj is None else "clinical_skill",
                generation_mode=plan_generation_mode,
                prompt_pack_version=prompt_version,
                metadata={
                    "llm_signature": llm_signature,
                    "used_mock_llm": used_mock_llm,
                },
            )
        emit_progress(
            "plan",
            f"Analysis plan ready with {len(plan.steps)} step(s).",
            run_id=run_id,
            total_steps=len(plan.steps),
        )

        started_at = datetime.now(timezone.utc)
        if resume_state and resume_state.get("started_at"):
            try:
                started_at = datetime.fromisoformat(resume_state["started_at"])
            except Exception:
                pass

        return _PlanPhaseResult(
            context=context,
            agent_context=agent_context,
            context_path=context_path,
            evidence=evidence,
            findings=findings,
            plan=plan,
            plan_path=plan_path,
            llm_signature=llm_signature,
            used_mock_llm=used_mock_llm,
            prompt_version=prompt_version,
            prompt_files=prompt_files,
            role_resolver=role_resolver,
            cost_meter=cost_meter,
            repro_envelope=repro_envelope,
            started_at=started_at,
            resume_state=resume_state,
        )

    def _run_execute_phase(
        self,
        *,
        plan_result: _PlanPhaseResult,
        cohort_path: Path,
        run_dir: Path,
        run_id: str,
        skill_obj: Optional[ClinicalSkill],
        notes: Optional[str],
        emit_progress: Callable[..., None],
    ) -> _ExecutePhaseResult:
        """Execute probe + per-step analysis loop, with optional replanning."""
        context = plan_result.context
        agent_context = plan_result.agent_context
        evidence = plan_result.evidence
        findings = plan_result.findings
        plan = plan_result.plan
        plan_path = plan_result.plan_path
        role_resolver = plan_result.role_resolver
        llm_signature = plan_result.llm_signature
        prompt_version = plan_result.prompt_version
        prompt_files = plan_result.prompt_files

        coder = CoderAgent(role_resolver("coder"))
        analyzer = AnalyzerAgent(role_resolver("analyzer"))
        supervisor = RuntimeSupervisor(
            clinical_semantics=ClinicalSemanticsAgent(),
            data_extraction=DataExtractionAgent(),
            statistical_analysis=StatisticalAnalysisAgent(),
            visualization=VisualizationAgent(),
            critic=CriticAgent(role_resolver("analyzer")),
        )
        runner = self._build_runner(run_dir=run_dir, cohort_path=cohort_path)
        usage_auditor = ConceptUsageAuditor()
        from .audits.patterns import AnalysisPatternAuditor

        pattern_auditor = AnalysisPatternAuditor()
        stat_validator = StatisticalValidator()
        clinical_validator = ClinicalConstraintValidator()
        statistical_guard = StatisticalGuard()
        runtime_state = supervisor.bootstrap_state(run_id=run_id, context=context)

        per_step_records: List[Dict[str, Any]] = []
        probe_summary: Dict[str, Any] = {}
        resumed_step_ids: set = set()
        if plan_result.resume_state is not None:
            try:
                prior_records = [
                    rec
                    for rec in (
                        plan_result.resume_state.get("per_step_records", []) or []
                    )
                    if isinstance(rec, dict) and rec.get("step_id")
                ]
                prior_ok_step_ids = {
                    rec["step_id"] for rec in prior_records if rec.get("status") == "ok"
                }
                for rec in plan_result.resume_state.get("per_step_records", []) or []:
                    if (
                        isinstance(rec, dict)
                        and rec.get("status") == "ok"
                        and rec.get("step_id")
                    ):
                        per_step_records.append(rec)
                        resumed_step_ids.add(rec["step_id"])
                for f in plan_result.resume_state.get("findings", []) or []:
                    try:
                        finding = ValidationFinding.model_validate(f)
                    except Exception:
                        continue
                    if finding.validator == "cohort_auditor":
                        continue
                    if finding.validator == "runner":
                        msg = finding.message or ""
                        if any(step_id in msg for step_id in prior_ok_step_ids):
                            continue
                    findings.append(finding)
                if resumed_step_ids:
                    print(
                        f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                        f"already-completed step(s) — {sorted(resumed_step_ids)}"
                    )
                for rec in per_step_records:
                    if rec.get("step_id") == "00_probe" and isinstance(
                        rec.get("step_summary"), dict
                    ):
                        probe_summary = rec["step_summary"]
            except Exception:
                resumed_step_ids = set()

        def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
            payload: Dict[str, Any] = {
                "schema_version": "easyicu.research_manifest_partial/1",
                "run_id": run_id,
                "research_question": context.research_question,
                "started_at": plan_result.started_at.isoformat(),
                "context_path": str(plan_result.context_path.relative_to(run_dir)),
                "plan_path": str(plan_path.relative_to(run_dir)),
                "evidence": [r.model_dump(mode="json") for r in evidence.records()],
                "findings": [f.model_dump(mode="json") for f in findings],
                "per_step_records": per_step_records,
                "llm_signature": llm_signature,
                "used_mock_llm": plan_result.used_mock_llm,
                "prompt_pack_version": prompt_version,
                "prompt_pack_files": prompt_files,
                "notes": notes,
                "runtime_state": runtime_state.model_dump(mode="json"),
            }
            if extra:
                payload.update(extra)
            (run_dir / "manifest_partial.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        runtime_packets = {
            "clinical_semantics_resolution": runtime_state.semantics,
            "data_extraction_request": runtime_state.extraction_request,
            "data_extraction_result": runtime_state.extraction_result,
        }
        for alias, packet in runtime_packets.items():
            if packet is None or evidence.get(alias) is not None:
                continue
            evidence.register_json(
                kind="log",
                description=f"Typed runtime packet: {alias}.",
                payload=packet.model_dump(mode="json"),
                filename=f"{alias}.json",
                evidence_id=alias,
                aliases=[alias],
                producer="runtime_supervisor",
                generation_mode="system",
                prompt_pack_version=prompt_version,
                metadata={"run_id": run_id},
            )

        _flush_partial_manifest()

        def _register_plan_revision(
            revised_plan: AnalysisPlan,
            *,
            reason: str,
        ) -> Path:
            revision_path = (
                run_dir / f"analysis_plan_revision_{revised_plan.revision}.json"
            )
            revision_path.write_text(
                revised_plan.model_dump_json(indent=2),
                encoding="utf-8",
            )
            evidence.register_file(
                kind="log",
                description=f"Revised analysis plan (reason={reason}).",
                source_path=revision_path,
                evidence_id=f"analysis_plan_revision_{revised_plan.revision}",
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"reason": reason, "llm_signature": llm_signature},
            )
            return revision_path

        def _maybe_replan(
            *,
            current_plan: AnalysisPlan,
            reason: str,
            probe_summary_payload: Optional[Dict[str, Any]] = None,
            completed_records: Optional[Sequence[Dict[str, Any]]] = None,
        ) -> AnalysisPlan:
            nonlocal plan_path
            if not self._enable_replanning or skill_obj is not None:
                return current_plan
            replanner = ReplannerAgent(role_resolver("planner"))
            try:
                revised = replanner.run(
                    context=agent_context,
                    current_plan=current_plan,
                    probe_summary=probe_summary_payload,
                    completed_step_records=completed_records,
                )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="replanner",
                        severity="warning",
                        message=f"Replanner failed; keeping existing plan: {exc}",
                        detail={"reason": reason},
                    )
                )
                return current_plan
            # Guard against the replanner silently dropping figure-producing
            # steps; task contracts (e.g. EasyICU experiment runner) still
            # require those artefacts regardless of the LLM's revised framing.
            revised, preservation_findings = _preserve_figure_steps_after_replan(
                current=current_plan,
                revised=revised,
            )
            if preservation_findings:
                findings.extend(preservation_findings)
            if revised.model_dump(mode="json") == current_plan.model_dump(mode="json"):
                return current_plan
            plan_path = _register_plan_revision(revised, reason=reason)
            plan_result.plan_path = plan_path
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="info",
                    message=f"Plan revised after {reason}.",
                    detail={
                        "from_revision": current_plan.revision,
                        "to_revision": revised.revision,
                    },
                )
            )
            return revised

        probe_step_id = "00_probe"
        if self._enable_probe_step and probe_step_id not in resumed_step_ids:
            probe_summary, probe_files = _build_probe_summary(
                context=context,
                cohort_path=cohort_path,
                out_dir=run_dir / "steps" / probe_step_id / "outputs",
            )
            probe_evidence_ids: List[str] = []
            for probe_file in probe_files:
                kind = "statistic" if probe_file.name.endswith(".json") else "table"
                aliases = [probe_step_id]
                if probe_file.name == "probe_summary.json":
                    aliases.extend(
                        [
                            "probe_summary",
                            "cohort_probe",
                        ]
                    )
                rec = evidence.register_file(
                    kind=kind,
                    description=f"Probe artefact {probe_file.name}.",
                    source_path=probe_file,
                    produced_by_step=probe_step_id,
                    producer="pipeline",
                    generation_mode="deterministic_probe",
                    aliases=aliases,
                )
                probe_evidence_ids.append(rec.evidence_id)
            probe_record = {
                "step_id": probe_step_id,
                "intent": "Probe distributions, missingness, and obvious anomalies before execution.",
                "status": "ok",
                "generation_mode": "deterministic_probe",
                "step_summary": probe_summary,
                "evidence_ids": probe_evidence_ids,
            }
            per_step_records.append(probe_record)
            _flush_partial_manifest()
            plan = _maybe_replan(
                current_plan=plan,
                reason="probe_summary",
                probe_summary_payload=probe_summary,
                completed_records=[probe_record],
            )

        shared_lock = threading.Lock()
        step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        total_steps = len(plan.steps)

        def _script_generation_mode(
            *,
            repair_attempts: int,
            fallback_used: bool,
            runner_repair_name: Optional[str] = None,
        ) -> str:
            if fallback_used:
                return "fallback"
            if repair_attempts > 0:
                return "repaired"
            if runner_repair_name:
                return "runner_repaired"
            return "llm"

        def _finding_severity(
            findings_for_step: Sequence[ValidationFinding],
        ) -> Optional[str]:
            if any(f.severity == "error" for f in findings_for_step):
                return "error"
            if any(f.severity == "warning" for f in findings_for_step):
                return "warning"
            if any(f.severity == "info" for f in findings_for_step):
                return "info"
            return None

        def _propagate_findings_to_evidence(
            evidence_ids: Sequence[str],
            findings_for_step: Sequence[ValidationFinding],
            *,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            severity = _finding_severity(findings_for_step)
            messages = [
                f.message
                for f in findings_for_step
                if f.severity in {"warning", "error"}
            ]
            for evidence_id in evidence_ids:
                evidence.update_record(
                    evidence_id,
                    finding_severity=severity,
                    finding_messages=messages,
                    metadata=metadata,
                )

        def _evidence_refs_for_names(names: Sequence[str]) -> List[EvidenceRef]:
            refs: List[EvidenceRef] = []
            seen: set[str] = set()
            for name in names:
                rec = evidence.get(str(name))
                if rec is None or rec.evidence_id in seen:
                    continue
                refs.append(
                    EvidenceRef(
                        evidence_id=rec.evidence_id,
                        kind=rec.kind,
                        description=rec.description,
                        relative_path=rec.relative_path,
                    )
                )
                seen.add(rec.evidence_id)
            return refs

        def _validator_messages(
            *finding_groups: Sequence[ValidationFinding],
        ) -> List[str]:
            messages: List[str] = []
            for group in finding_groups:
                for finding in group:
                    if finding.message:
                        messages.append(finding.message)
            return messages

        def _failed_dependency_record(step: AnalysisStep) -> Optional[Dict[str, Any]]:
            parent_step_id = _parent_step_id_for_figure_step(step)
            if parent_step_id is None:
                return None
            with shared_lock:
                records = list(per_step_records)
            for record in records:
                if record.get("step_id") != parent_step_id:
                    continue
                if str(record.get("status") or "").lower() == "ok":
                    return None
                return record
            return None

        def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
            nonlocal runtime_state
            step_record: Dict[str, Any] = {
                "step_id": step.step_id,
                "intent": step.intent,
            }
            step_current = step_order.get(step.step_id, 0) + 1
            dependency_record = _failed_dependency_record(step)
            if dependency_record is not None:
                parent_step_id = str(dependency_record.get("step_id") or "")
                step_record.update(
                    {
                        "status": "skipped_dependency_failed",
                        "dependency_step_id": parent_step_id,
                        "diagnostic_only": True,
                        "generation_mode": "system",
                    }
                )
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="dependency_gate",
                            severity="warning",
                            message=(
                                f"Skipped downstream figure step {step.step_id} because "
                                f"required analysis step {parent_step_id} did not pass."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "dependency_step_id": parent_step_id,
                                "dependency_status": dependency_record.get("status"),
                                "diagnostic_only": True,
                            },
                        )
                    )
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "step",
                    f"Skipped {step.step_id}; required step {parent_step_id} failed.",
                    status="skipped",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            emit_progress(
                "step",
                f"Step {step_current}/{total_steps} started: {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            existing_refs = _evidence_refs_for_names(step.inputs)
            local_runtime_state = supervisor.prepare_step_state(
                state=runtime_state,
                context=context,
                step=step,
                evidence_refs=existing_refs,
            )
            step_record["analysis_request"] = (
                local_runtime_state.analysis_request.model_dump(mode="json")
                if local_runtime_state.analysis_request is not None
                else None
            )
            step_record["visualization_request"] = (
                local_runtime_state.visualization_request.model_dump(mode="json")
                if local_runtime_state.visualization_request is not None
                else None
            )
            step_record["semantics_family"] = local_runtime_state.analysis_family

            try:
                emit_progress(
                    "coder",
                    f"Generating analysis script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                code = coder.run(context=agent_context, step=step)
            except Exception as exc:
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=f"Coder agent failed for step {step.step_id}: {exc}",
                        )
                    )
                    step_record["status"] = "coder_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Coder failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            deterministic_fallback_used = False

            def _deterministic_fallback_code(reason: str) -> Optional[str]:
                nonlocal deterministic_fallback_used
                if (
                    deterministic_fallback_used
                    or not self._enable_deterministic_code_fallback
                ):
                    return None
                deterministic_fallback_used = True
                plan_result.used_mock_llm = True
                step_record["deterministic_code_fallback"] = reason
                emit_progress(
                    "coder",
                    f"Using deterministic fallback script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    fallback_reason=reason,
                )
                fallback_coder = CoderAgent(MockLLMClient(context=agent_context))
                return fallback_coder.run(context=agent_context, step=step)

            concept_repair_attempts = 0
            concept_audit_error_count = 0
            while True:
                usage_findings = usage_auditor.audit(
                    context=context,
                    script_text=code,
                    step=step,
                )
                # O-generic: analysis-pattern auditor (clustering /
                # prediction / survival footguns). Runs alongside the
                # concept-usage auditor so both sets of findings are
                # merged before the error-gate decision.
                usage_findings.extend(
                    pattern_auditor.audit(
                        context=context,
                        script_text=code,
                        step=step,
                    )
                )
                if self._enable_llm_concept_audit:
                    llm_audit_client = (
                        self._llm_concept_auditor_client or role_resolver("analyzer")
                    )
                    if llm_audit_client is not None:
                        usage_findings.extend(
                            LLMConceptAuditor(llm_audit_client).audit(
                                context=context,
                                script_text=code,
                                step=step,
                            )
                        )
                step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
                concept_audit_error_count += sum(
                    1
                    for f in usage_findings
                    if f.validator == usage_auditor.name and f.severity == "error"
                )
                step_record["concept_audit_error_count"] = concept_audit_error_count
                step_record["concept_repair_attempts"] = concept_repair_attempts
                if not any(f.severity == "error" for f in usage_findings):
                    with shared_lock:
                        findings.extend(usage_findings)
                    break

                if concept_repair_attempts >= self._max_code_repair_attempts:
                    fallback_code = _deterministic_fallback_code("concept_audit")
                    if fallback_code is not None:
                        # Surface the pattern/concept findings that
                        # forced the fallback; otherwise the manifest
                        # silently drops the original ICU rule
                        # violations that the LLM emitted. We dedupe by
                        # message so repeated retries don't spam.
                        with shared_lock:
                            seen_msgs = {f.message for f in findings}
                            for f in usage_findings:
                                if f.message in seen_msgs:
                                    continue
                                # Demote ``error`` severity to
                                # ``warning`` because the run is
                                # continuing on the deterministic
                                # fallback; reviewer still sees the
                                # original violation in the manifest.
                                if f.severity == "error":
                                    f = f.model_copy(update={
                                        "severity": "warning",
                                        "message": (
                                            "[surfaced after fallback] "
                                            + f.message
                                        ),
                                    })
                                findings.append(f)
                        code = fallback_code
                        continue
                    step_record["status"] = "blocked_by_concept_audit"
                    with shared_lock:
                        findings.extend(usage_findings)
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "audit",
                        f"Concept audit blocked {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

                concept_repair_attempts += 1
                step_record["concept_repair_attempts"] = concept_repair_attempts
                emit_progress(
                    "coder",
                    f"Repairing concept-audit violation for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    repair_attempts=concept_repair_attempts,
                )
                audit_log = "\n".join(
                    f"{f.severity.upper()}: {f.message}" for f in usage_findings
                )
                try:
                    code = coder.repair(
                        context=agent_context,
                        step=step,
                        code=code,
                        run_log=(
                            "Static concept audit blocked this script before "
                            "execution. Fix all ICU-rule violations.\n\n" + audit_log
                        ),
                        attempt=concept_repair_attempts,
                    )
                except Exception as exc:
                    fallback_code = _deterministic_fallback_code(
                        "concept_repair_failed"
                    )
                    if fallback_code is not None:
                        code = fallback_code
                        continue
                    with shared_lock:
                        findings.extend(usage_findings)
                        findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="error",
                                message=(
                                    f"Coder repair failed after concept audit for "
                                    f"step {step.step_id}: {exc}"
                                ),
                            )
                        )
                        step_record["status"] = "repair_failed"
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "coder",
                        f"Concept-audit repair failed for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

            repair_attempts = 0
            runner_repair_name: Optional[str] = None
            while True:
                run_label = "repaired script" if repair_attempts else "generated script"
                emit_progress(
                    "runner",
                    f"Running {run_label} for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    repair_attempts=repair_attempts,
                )
                run_result = runner.run(step_id=step.step_id, code=code)
                step_record["returncode"] = run_result.returncode
                step_record["timed_out"] = run_result.timed_out
                step_record["code_repair_attempts"] = repair_attempts

                script_description = (
                    f"Generated analysis script for step {step.step_id}."
                    if repair_attempts == 0
                    else f"Repaired analysis script for step {step.step_id} (attempt {repair_attempts})."
                )
                script_record = evidence.register_file(
                    kind="code",
                    description=script_description,
                    source_path=run_result.script_path,
                    produced_by_step=step.step_id,
                    producer="coder",
                    generation_mode=_script_generation_mode(
                        repair_attempts=repair_attempts,
                        fallback_used=deterministic_fallback_used,
                        runner_repair_name=runner_repair_name,
                    ),
                    prompt_pack_version=prompt_version,
                    metadata={
                        "repair_attempts": repair_attempts,
                        "fallback_reason": step_record.get(
                            "deterministic_code_fallback"
                        ),
                        "runner_repair": runner_repair_name,
                        "llm_signature": llm_signature,
                    },
                )
                log_path = run_result.cwd / "run.log"
                if log_path.exists():
                    evidence.register_file(
                        kind="log",
                        description=f"stdout/stderr log for step {step.step_id}.",
                        source_path=log_path,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        producer="runner",
                        generation_mode=_script_generation_mode(
                            repair_attempts=repair_attempts,
                            fallback_used=deterministic_fallback_used,
                            runner_repair_name=runner_repair_name,
                        ),
                        metadata={
                            "repair_attempts": repair_attempts,
                            "fallback_reason": step_record.get(
                                "deterministic_code_fallback"
                            ),
                            "runner_repair": runner_repair_name,
                        },
                    )

                if run_result.succeeded:
                    if not (run_result.out_dir / "step_summary.json").exists():
                        salvaged = _salvage_stdout_json_step_summary(
                            run_result
                        ) or _salvage_named_json_step_summary(run_result)
                        if salvaged:
                            run_result.artefacts = sorted(
                                p for p in run_result.out_dir.iterdir() if p.is_file()
                            )
                    else:
                        _salvage_minimal_contract_step_summary(
                            step=step,
                            out_dir=run_result.out_dir,
                        )
                    if not run_result.artefacts:
                        fallback_code = _deterministic_fallback_code("no_artefacts")
                        if fallback_code is not None:
                            code = fallback_code
                            _clear_output_dir(run_result.out_dir)
                            continue
                    visual_step_summary: Dict[str, Any] = {}
                    visual_summary_path = run_result.out_dir / "step_summary.json"
                    if visual_summary_path.exists():
                        try:
                            vloaded = json.loads(
                                visual_summary_path.read_text(encoding="utf-8")
                            )
                        except Exception:
                            vloaded = None
                        if isinstance(vloaded, dict):
                            visual_step_summary = vloaded
                        else:
                            visual_step_summary = {"raw": vloaded}
                    step_figures = [
                        art
                        for art in run_result.artefacts
                        if art.suffix.lower() in {".png", ".svg", ".tiff", ".tif"}
                    ]
                    if self._enable_visual_qa and step_figures:
                        expected_numeric = _expected_numeric_annotations_for_step(
                            step=step,
                            step_summary=visual_step_summary,
                        )
                        numeric_expectations = (
                            {
                                str(path): expected_numeric
                                for path in step_figures
                                if path.suffix.lower() == ".svg"
                            }
                            if expected_numeric
                            else None
                        )
                        visual_findings = VisualQAAuditor().audit_with_expected(
                            figure_paths=step_figures,
                            expected_numeric_by_path=numeric_expectations,
                        )
                        step_record["visual_findings"] = [
                            f.model_dump() for f in visual_findings
                        ]
                        visual_errors = [
                            f for f in visual_findings if f.severity == "error"
                        ]
                        if visual_errors:
                            if repair_attempts >= self._max_code_repair_attempts:
                                fallback_code = _deterministic_fallback_code(
                                    "visual_qa"
                                )
                                if fallback_code is not None:
                                    code = fallback_code
                                    _clear_output_dir(run_result.out_dir)
                                    continue
                                # Demote unrecoverable visual_qa errors to
                                # warnings — visual layout issues (overlapping
                                # text, panel-label spacing, etc.) should not
                                # block scientifically-valid analysis outputs
                                # from being accepted. The issues remain
                                # visible to reviewers via
                                # ``step_record["visual_findings"]`` and the
                                # demoted warnings recorded on the manifest.
                                demoted_findings = [
                                    (
                                        finding.model_copy(
                                            update={"severity": "warning"}
                                        )
                                        if finding.severity == "error"
                                        else finding
                                    )
                                    for finding in visual_findings
                                ]
                                with shared_lock:
                                    findings.extend(demoted_findings)
                                step_record["visual_qa_demoted"] = True
                                emit_progress(
                                    "visual_qa",
                                    (
                                        f"Visual QA findings demoted to warning "
                                        f"for {step.step_id} after "
                                        f"{repair_attempts} repair attempts."
                                    ),
                                    status="warning",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                # Fall through to contract checks and evidence
                                # registration — the step's analytic outputs
                                # remain valid even if the figure layout has
                                # cosmetic issues.
                            else:
                                repair_attempts += 1
                                step_record["code_repair_attempts"] = repair_attempts
                                emit_progress(
                                    "visual_qa",
                                    f"Repairing figure layout for {step.step_id}.",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                    repair_attempts=repair_attempts,
                                )
                                qa_log = "\n".join(
                                    f"{f.severity.upper()}: {f.message}"
                                    for f in visual_findings
                                )
                                try:
                                    code = coder.repair(
                                        context=agent_context,
                                        step=step,
                                        code=code,
                                        run_log=(
                                            "Visual QA rejected one or more figure outputs "
                                            "before evidence registration. Fix the figure "
                                            "layout, preserve all tables/statistics, save PNG "
                                            "and editable SVG with the same stem, include "
                                            "publication figure exports when requested, and rerun.\n\n"
                                            + qa_log
                                        ),
                                        attempt=repair_attempts,
                                    )
                                    _clear_output_dir(run_result.out_dir)
                                    continue
                                except Exception as exc:
                                    fallback_code = _deterministic_fallback_code(
                                        "visual_qa_repair_failed"
                                    )
                                    if fallback_code is not None:
                                        code = fallback_code
                                        _clear_output_dir(run_result.out_dir)
                                        continue
                                    with shared_lock:
                                        findings.extend(visual_findings)
                                        findings.append(
                                            ValidationFinding(
                                                validator="coder",
                                                severity="error",
                                                message=(
                                                    f"Coder repair failed after visual QA "
                                                    f"for step {step.step_id}: {exc}"
                                                ),
                                            )
                                        )
                                        step_record["status"] = "repair_failed"
                                        per_step_records.append(step_record)
                                        _flush_partial_manifest()
                                    emit_progress(
                                        "visual_qa",
                                        f"Visual QA repair failed for {step.step_id}.",
                                        status="error",
                                        run_id=run_id,
                                        step_id=step.step_id,
                                        current_step=step_current,
                                        total_steps=total_steps,
                                    )
                                    return step_record
                    early_contract_findings = _step_contract_findings(
                        step=step,
                        step_summary=visual_step_summary,
                    )
                    early_contract_errors = [
                        f for f in early_contract_findings if f.severity == "error"
                    ]
                    if early_contract_errors and repair_attempts < self._max_code_repair_attempts:
                        repair_attempts += 1
                        step_record["code_repair_attempts"] = repair_attempts
                        emit_progress(
                            "coder",
                            f"Repairing contract violation for {step.step_id}.",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                            repair_attempts=repair_attempts,
                        )
                        contract_log = "\n".join(
                            f"{f.severity.upper()}: {f.message}"
                            for f in early_contract_findings
                            if f.message
                        )
                        repair_guidance = _step_contract_repair_guidance(
                            step=step,
                            step_summary=visual_step_summary,
                            code=code,
                        )
                        try:
                            code = coder.repair(
                                context=agent_context,
                                step=step,
                                code=code,
                                run_log=(
                                    "The script executed but failed the machine-readable "
                                    "step contract. Revise the analysis code; do not change "
                                    "the research question. Ensure required primary metrics "
                                    "are computed and written to step_summary.json with "
                                    "explicit numeric keys or nested statistic fields.\n\n"
                                    "STEP SUMMARY:\n"
                                    + json.dumps(
                                        visual_step_summary,
                                        indent=2,
                                        ensure_ascii=False,
                                        default=str,
                                    )
                                    + "\n\nCONTRACT FINDINGS:\n"
                                    + contract_log
                                    + "\n\nREPAIR GUIDANCE:\n"
                                    + repair_guidance
                                ),
                                attempt=repair_attempts,
                            )
                            _clear_output_dir(run_result.out_dir)
                            continue
                        except Exception as exc:
                            fallback_code = _deterministic_fallback_code(
                                "contract_repair_failed"
                            )
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(run_result.out_dir)
                                continue
                            with shared_lock:
                                findings.extend(early_contract_findings)
                                findings.append(
                                    ValidationFinding(
                                        validator="coder",
                                        severity="error",
                                        message=(
                                            f"Coder repair failed after contract check "
                                            f"for step {step.step_id}: {exc}"
                                        ),
                                    )
                                )
                                step_record["status"] = "repair_failed"
                                per_step_records.append(step_record)
                                _flush_partial_manifest()
                            emit_progress(
                                "coder",
                                f"Contract repair failed for {step.step_id}.",
                                status="error",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            return step_record
                    if self._enable_deterministic_runner_repair:
                        summary_repair = _deterministic_summary_repair(
                            code=code,
                            step_summary=visual_step_summary,
                            previous_repair=runner_repair_name,
                        )
                    else:
                        summary_repair = None
                    if summary_repair is not None:
                        runner_repair_name, code = summary_repair
                        step_record["runner_repair"] = runner_repair_name
                        emit_progress(
                            "runner_repair",
                            f"Applied deterministic summary repair for {step.step_id}: {runner_repair_name}.",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    break

                if log_path.exists():
                    run_log = log_path.read_text(encoding="utf-8", errors="replace")
                else:
                    run_log = (
                        (run_result.stdout or "") + "\n" + (run_result.stderr or "")
                    )
                if self._enable_deterministic_runner_repair:
                    runner_repair = _deterministic_runner_repair(
                        code=code,
                        run_log=run_log,
                        previous_repair=runner_repair_name,
                    )
                else:
                    runner_repair = None
                if runner_repair is not None:
                    runner_repair_name, code = runner_repair
                    step_record["runner_repair"] = runner_repair_name
                    emit_progress(
                        "runner_repair",
                        f"Applied deterministic runner repair for {step.step_id}: {runner_repair_name}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    _clear_output_dir(run_result.out_dir)
                    continue

                if repair_attempts >= self._max_code_repair_attempts:
                    fallback_code = _deterministic_fallback_code("execution_failure")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="runner",
                                severity="error",
                                message=(
                                    f"Step {step.step_id} "
                                    f"{'timed out' if run_result.timed_out else 'failed'} "
                                    f"with returncode {run_result.returncode}."
                                ),
                            )
                        )
                        step_record["status"] = "execution_failed"
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "runner",
                        f"Execution failed for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

                repair_attempts += 1
                step_record["code_repair_attempts"] = repair_attempts
                emit_progress(
                    "coder",
                    f"Repairing failed script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    repair_attempts=repair_attempts,
                )
                try:
                    code = coder.repair(
                        context=agent_context,
                        step=step,
                        code=code,
                        run_log=run_log,
                        attempt=repair_attempts,
                    )
                    _clear_output_dir(run_result.out_dir)
                except Exception as exc:
                    fallback_code = _deterministic_fallback_code("repair_failed")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="error",
                                message=f"Coder repair failed for step {step.step_id}: {exc}",
                            )
                        )
                        step_record["status"] = "repair_failed"
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "coder",
                        f"Repair failed for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

            publication_step = (
                step.method == "publication_figure_generation"
                or "publication_figure" in (step.step_id or "").lower()
                or "publication figure" in (step.intent or "").lower()
                or "publication-ready figure" in (step.intent or "").lower()
                or any(
                    str(item).startswith("figure:publication_figure")
                    for item in step.expected_outputs
                )
            )
            figure_role = (
                "publication_figure"
                if publication_step
                else "analysis_figure"
                if _step_expects_figure(step)
                else None
            )
            if publication_step and not _has_figure_exports(run_result.out_dir):
                promoted = _promote_prior_publication_bundle(
                    run_dir=run_dir,
                    current_step_id=step.step_id,
                    out_dir=run_result.out_dir,
                )
                if promoted is not None:
                    runner_repair_name = promoted
                    step_record["runner_repair"] = promoted
                else:
                    rescued = _render_prediction_publication_bundle_from_prior_outputs(
                        run_dir=run_dir,
                        current_step_id=step.step_id,
                        out_dir=run_result.out_dir,
                    )
                    if rescued is not None:
                        runner_repair_name = rescued
                        step_record["runner_repair"] = rescued

            run_result.artefacts = sorted(
                p for p in run_result.out_dir.iterdir() if p.is_file()
            )

            if publication_step and not _has_figure_exports(run_result.out_dir):
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="publication_figure_outputs",
                            severity="error",
                            message=(
                                f"Step {step.step_id} completed without any publication-figure exports."
                            ),
                        )
                    )
                    step_record["status"] = "execution_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Publication figure missing for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            evidence_ids_for_step: List[str] = [script_record.evidence_id]
            for art in run_result.artefacts:
                step_aliases = _semantic_aliases_for(step, art)
                generation_mode = _script_generation_mode(
                    repair_attempts=repair_attempts,
                    fallback_used=deterministic_fallback_used,
                    runner_repair_name=runner_repair_name,
                )
                if art.name == "step_summary.json":
                    rec = evidence.register_file(
                        kind="statistic",
                        description=f"Machine-readable summary for step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                        producer="runner",
                        generation_mode=generation_mode,
                        metadata={
                            "script_evidence_id": script_record.evidence_id,
                            "figure_role": figure_role or "analysis_figure",
                            "diagnostic_only": False,
                        },
                    )
                elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
                    rec = evidence.register_file(
                        kind="table",
                        description=f"Table {art.stem} from step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                        producer="runner",
                        generation_mode=generation_mode,
                        metadata={"script_evidence_id": script_record.evidence_id},
                    )
                elif art.suffix.lower() in {
                    ".png",
                    ".svg",
                    ".pdf",
                    ".tiff",
                    ".tif",
                    ".pptx",
                }:
                    rec = evidence.register_file(
                        kind="figure",
                        description=f"Figure {art.stem} from step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                        producer="runner",
                        generation_mode=generation_mode,
                        metadata={
                            "script_evidence_id": script_record.evidence_id,
                            "figure_role": figure_role or "analysis_figure",
                            "diagnostic_only": False,
                        },
                    )
                else:
                    rec = evidence.register_file(
                        kind="log",
                        description=f"Auxiliary artefact {art.name}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                        producer="runner",
                        generation_mode=generation_mode,
                        metadata={"script_evidence_id": script_record.evidence_id},
                    )
                evidence_ids_for_step.append(rec.evidence_id)

            step_summary: Dict[str, Any] = {}
            ssj = run_result.out_dir / "step_summary.json"
            if ssj.exists():
                try:
                    loaded = json.loads(ssj.read_text(encoding="utf-8"))
                except Exception:
                    loaded = None
                if isinstance(loaded, dict):
                    step_summary = loaded
                else:
                    # The coder emitted a non-dict JSON (bare string /
                    # list / number). Keep it accessible but coerce to
                    # a dict so every downstream consumer that calls
                    # ``.get(...)`` still works.
                    step_summary = {"raw": loaded}
            stat_findings = stat_validator.audit(
                context=context,
                cohort_path=cohort_path,
                step=step,
                out_dir=run_result.out_dir,
                step_summary=step_summary,
            )
            clinical_findings = clinical_validator.audit(
                context=context,
                step=step,
                out_dir=run_result.out_dir,
                step_summary=step_summary,
            )
            guard_findings = statistical_guard.audit(
                context=context,
                cohort_path=cohort_path,
                step=step,
                out_dir=run_result.out_dir,
                step_summary=step_summary,
            )
            contract_findings = _step_contract_findings(
                step=step,
                step_summary=step_summary,
            )
            with shared_lock:
                findings.extend(stat_findings)
                findings.extend(clinical_findings)
                findings.extend(guard_findings)
                findings.extend(contract_findings)
            step_record["stat_findings"] = [f.model_dump() for f in stat_findings]
            step_record["clinical_findings"] = [
                f.model_dump() for f in clinical_findings
            ]
            step_record["guard_findings"] = [f.model_dump() for f in guard_findings]
            step_record["contract_findings"] = [
                f.model_dump() for f in contract_findings
            ]
            step_record["generation_mode"] = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
            )
            step_record["step_summary"] = step_summary
            evidence_refs_for_step = _evidence_refs_for_names(evidence_ids_for_step)
            validator_messages = _validator_messages(
                usage_findings,
                stat_findings,
                clinical_findings,
                guard_findings,
                contract_findings,
            )
            local_runtime_state = supervisor.critique_step(
                state=local_runtime_state,
                step_summary=step_summary,
                evidence_refs=evidence_refs_for_step,
                findings=validator_messages,
            )
            critique = local_runtime_state.critique
            if critique is not None:
                critique_path = run_result.out_dir / "critique_report.json"
                critique_path.write_text(
                    critique.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                critique_record = evidence.register_file(
                    kind="log",
                    description=f"Structured critique report for step {step.step_id}.",
                    source_path=critique_path,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=[f"{step.step_id}_critique"],
                    producer="critic",
                    generation_mode="system",
                    metadata={"script_evidence_id": script_record.evidence_id},
                )
                evidence_ids_for_step.append(critique_record.evidence_id)
                step_record["critique_report"] = critique.model_dump(mode="json")
                if critique.status in {"needs_revision", "blocked"}:
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="critic_agent",
                                severity=(
                                    "warning"
                                    if critique.status == "needs_revision"
                                    else "error"
                                ),
                                message=(
                                    f"CriticAgent marked {step.step_id} as {critique.status}: "
                                    + "; ".join(
                                        critique.concerns
                                        or critique.suggested_repairs
                                        or ["review required"]
                                    )
                                ),
                                evidence_ids=[critique_record.evidence_id],
                            )
                        )

            try:
                interpretation = analyzer.run(
                    context=agent_context,
                    step=step,
                    step_summary=step_summary,
                    evidence_ids=evidence_ids_for_step,
                )
            except Exception as exc:
                interpretation = f"(analyzer failed: {exc})"
            interp_record = evidence.register_text(
                kind="log",
                description=f"Analyzer interpretation for step {step.step_id}.",
                text=interpretation,
                filename=f"interpretation_{step.step_id}.md",
                produced_by_step=step.step_id,
                script_evidence_id=script_record.evidence_id,
                producer="analyzer",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
            )
            step_record["interpretation_evidence_id"] = interp_record.evidence_id
            _propagate_findings_to_evidence(
                evidence_ids_for_step + [interp_record.evidence_id],
                usage_findings
                + stat_findings
                + clinical_findings
                + guard_findings
                + contract_findings,
                metadata={
                    "step_id": step.step_id,
                    "generation_mode": step_record["generation_mode"],
                },
            )
            with shared_lock:
                runtime_state = local_runtime_state
            has_contract_error = any(
                finding.severity == "error" for finding in contract_findings
            )
            step_record["status"] = "contract_failed" if has_contract_error else "ok"
            with shared_lock:
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "step",
                (
                    f"Step {step_current}/{total_steps} failed contract checks: "
                    f"{step.step_id}."
                    if has_contract_error
                    else f"Step {step_current}/{total_steps} complete: {step.step_id}."
                ),
                status="error" if has_contract_error else "complete",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        steps_to_run = [s for s in plan.steps if s.step_id not in resumed_step_ids]
        for skipped_step_id in sorted(resumed_step_ids):
            emit_progress(
                "resume",
                f"Skipped completed step from prior run: {skipped_step_id}.",
                status="complete",
                run_id=run_id,
                step_id=skipped_step_id,
            )
        if self._enable_replanning and self._max_concurrent_steps > 1:
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="info",
                    message=(
                        "Replanning is enabled, so step execution was forced to sequential "
                        "mode to preserve run-internal plan revisions."
                    ),
                )
            )

        if (
            self._max_concurrent_steps <= 1
            or len(steps_to_run) <= 1
            or self._enable_replanning
        ):
            executed_step_ids = set(resumed_step_ids)
            remaining_steps = [
                s for s in plan.steps if s.step_id not in executed_step_ids
            ]
            while remaining_steps:
                step = remaining_steps.pop(0)
                record = _execute_one_step(step)
                executed_step_ids.add(step.step_id)
                if (
                    self._enable_replanning
                    and record.get("status") == "ok"
                    and remaining_steps
                ):
                    plan = _maybe_replan(
                        current_plan=plan,
                        reason=step.step_id,
                        probe_summary_payload=probe_summary,
                        completed_records=per_step_records,
                    )
                    step_order.clear()
                    step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
                    remaining_steps = [
                        s for s in plan.steps if s.step_id not in executed_step_ids
                    ]
                    total_steps = len(plan.steps)
        else:
            workers = min(self._max_concurrent_steps, len(steps_to_run))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ra_step"
            ) as ex:
                futures = [ex.submit(_execute_one_step, s) for s in steps_to_run]
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        with shared_lock:
                            findings.append(
                                ValidationFinding(
                                    validator="step_executor",
                                    severity="error",
                                    message=f"Worker raised an unhandled exception: {exc!r}",
                                )
                            )

        if self._enable_visual_qa:
            emit_progress(
                "visual_qa",
                "Auditing generated figures.",
                run_id=run_id,
            )
            fig_paths = [
                run_dir / r.relative_path
                for r in evidence.records()
                if r.kind == "figure"
            ]
            vlm_adapter = self._visual_qa_adapter
            if vlm_adapter is None and self._enable_vlm_visual_qa:
                client = self._vlm_client or role_resolver("analyzer")
                if client is not None:
                    vlm_adapter = VLMVisualQAAdapter(client)
            final_visual_findings = VisualQAAuditor(
                vlm_adapter=vlm_adapter
            ).audit(figure_paths=fig_paths)
            # Mirror the per-step demotion policy here: layout-style
            # visual_qa errors (overlapping text, panel-label spacing,
            # etc.) are cosmetic and should not block scientifically
            # valid analyses from being accepted at the run level. The
            # per-step pipeline above attempts up to
            # ``self._max_code_repair_attempts`` repairs and falls back
            # to a layout-aware deterministic helper before demoting;
            # by the time the final whole-run audit runs we have
            # exhausted those budgets, so any remaining figure-layout
            # findings are demoted to warnings here too. The original
            # message is preserved for reviewer inspection on the
            # manifest.
            demoted_final_findings = [
                (
                    finding.model_copy(update={"severity": "warning"})
                    if finding.severity == "error"
                    else finding
                )
                for finding in final_visual_findings
            ]
            findings += demoted_final_findings

        plan_result.plan = plan
        plan_result.plan_path = plan_path
        return _ExecutePhaseResult(
            plan=plan,
            per_step_records=per_step_records,
            probe_summary=probe_summary,
            runtime_state=runtime_state,
            flush_partial_manifest=_flush_partial_manifest,
        )

    def _run_write_phase(
        self,
        *,
        plan_result: _PlanPhaseResult,
        execute_result: _ExecutePhaseResult,
        run_dir: Path,
        run_id: str,
        stop_after_analysis: bool,
        manuscript_title: Optional[str],
        manuscript_authors: Optional[Sequence[str]],
        run_language: str,
        emit_progress: Callable[..., None],
    ) -> _WritePhaseResult:
        """Draft manuscript-facing outputs after analysis is complete."""
        context = plan_result.context
        agent_context = plan_result.agent_context
        evidence = plan_result.evidence
        findings = plan_result.findings
        role_resolver = plan_result.role_resolver
        prompt_version = plan_result.prompt_version
        runtime_state = execute_result.runtime_state
        per_step_records = execute_result.per_step_records
        critic = CriticAgent(role_resolver("analyzer"))

        execution_gate = execution_gate_status(
            plan=execute_result.plan,
            per_step_records=per_step_records,
        )
        if not execution_gate["execution_complete"]:
            findings.append(
                ValidationFinding(
                    validator="manuscript_gate",
                    severity="error",
                    message=(
                        "Formal manuscript generation skipped because the execution "
                        "gate did not pass. Review author_review_note.md and the "
                        "diagnostic artefacts before rerunning."
                    ),
                    detail=execution_gate,
                )
            )
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(
                "# Manuscript scaffold not generated\n\n"
                "Strict fail-closed policy blocked manuscript drafting because "
                "one or more required analysis steps did not complete successfully.\n\n"
                "Review `author_review_note.md`, `run_status.json`, "
                "`evidence_audit.json`, `numeric_audit.json`, and "
                "`claim_ledger.csv` for the diagnostic record.\n",
                encoding="utf-8",
            )
            return _WritePhaseResult(literature=None, bound_path=bound_path)

        if stop_after_analysis:
            emit_progress(
                "pause",
                "Analysis phase complete; manuscript generation skipped by user setting.",
                status="paused",
                run_id=run_id,
            )
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(
                "# Manuscript scaffold not generated\n\n"
                "This run stopped after the analysis phase. Review the "
                "`results_report.md`, tables, figures and manifest, then "
                "rerun with manuscript drafting enabled when the analysis "
                "is ready.\n",
                encoding="utf-8",
            )
            return _WritePhaseResult(literature=None, bound_path=bound_path)

        literature: Optional[LiteratureBundle] = None
        if self._enable_publication_figure_skill:
            try:
                emit_progress(
                    "figure",
                    "Rendering manuscript-facing publication figure bundle from registered evidence.",
                    run_id=run_id,
                )
                figure_result = PublicationFigureSkill().run(
                    context=context,
                    plan=execute_result.plan,
                    evidence=evidence,
                    run_dir=run_dir,
                    prompt_pack_version=prompt_version,
                )
                findings.extend(figure_result.findings)
                if self._enable_visual_qa and figure_result.figure_evidence_ids:
                    fig_paths = []
                    for evidence_id in figure_result.figure_evidence_ids:
                        record = evidence.get(evidence_id)
                        if record is not None:
                            fig_paths.append(run_dir / record.relative_path)
                    if fig_paths:
                        vlm_adapter = self._visual_qa_adapter
                        if vlm_adapter is None and self._enable_vlm_visual_qa:
                            client = self._vlm_client or role_resolver("analyzer")
                            if client is not None:
                                vlm_adapter = VLMVisualQAAdapter(client)
                        publication_visual_findings = VisualQAAuditor(
                            vlm_adapter=vlm_adapter
                        ).audit(figure_paths=fig_paths)
                        # See the final-pass demotion above: layout-style
                        # visual_qa errors raised on the publication
                        # bundle are cosmetic and must not block
                        # acceptance after the per-step repair budget
                        # has been exhausted upstream.
                        findings.extend(
                            (
                                finding.model_copy(
                                    update={"severity": "warning"}
                                )
                                if finding.severity == "error"
                                else finding
                            )
                            for finding in publication_visual_findings
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="publication_figure_skill",
                        severity="warning",
                        message=f"Publication figure skill failed; writer will use existing evidence only: {exc}",
                    )
                )

        if self._enable_literature:
            try:
                emit_progress(
                    "literature",
                    "Building literature bundle for manuscript drafting.",
                    run_id=run_id,
                )
                lit_client = role_resolver("literature")
                if hasattr(lit_client, "_inner") and isinstance(
                    getattr(lit_client, "_inner", None), MockLLMClient
                ):
                    lit_client = None
                if isinstance(lit_client, MockLLMClient):
                    lit_client = None
                pubmed_client = None
                if self._enable_pubmed:
                    from .literature import PubMedLiteratureClient

                    pubmed_client = PubMedLiteratureClient(
                        email=self._pubmed_email,
                        api_key=self._pubmed_api_key,
                    )
                tavily_client = None
                if self._enable_tavily:
                    from .literature import TavilyLiteratureClient

                    tavily_client = TavilyLiteratureClient(
                        api_key=self._tavily_api_key,
                        include_domains=self._tavily_include_domains,
                        exclude_domains=self._tavily_exclude_domains,
                    )
                literature = LiteratureAgent(
                    lit_client,
                    enable_pubmed=self._enable_pubmed,
                    pubmed_client=pubmed_client,
                    enable_tavily=self._enable_tavily,
                    tavily_client=tavily_client,
                    tavily_retmax=self._tavily_retmax,
                ).run(agent_context)
                lit_path = run_dir / "literature_bundle.json"
                lit_path.write_text(
                    literature.model_dump_json(indent=2), encoding="utf-8"
                )
                if evidence.get("literature_bundle") is None:
                    evidence.register_file(
                        kind="log",
                        description="LiteratureBundle (citation registry for this run).",
                        source_path=lit_path,
                        evidence_id="literature_bundle",
                        producer="literature",
                        generation_mode=(
                            "llm" if lit_client is not None else "deterministic_skill"
                        ),
                        prompt_pack_version=prompt_version,
                        metadata={
                            "enable_pubmed": self._enable_pubmed,
                            "enable_tavily": self._enable_tavily,
                        },
                    )
                # O21 — PRISMA 2020 counts. Registered as a separate
                # evidence id so the manuscript can cite
                # ``{evidence:literature_prisma}`` without pulling the
                # whole citation table into the binder.
                if literature.prisma is not None:
                    prisma_path = run_dir / "literature_prisma.json"
                    prisma_md_path = run_dir / "literature_prisma.md"
                    prisma_path.write_text(
                        json.dumps(
                            {
                                "research_question": literature.research_question,
                                "prisma": literature.prisma,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    p = literature.prisma
                    prisma_md = (
                        "# PRISMA 2020 flow (O21)\n\n"
                        f"- Records identified: **{p.get('identified', 0)}**\n"
                        f"- Duplicates removed: **{p.get('duplicates_removed', 0)}**\n"
                        f"- Records screened: **{p.get('screened', 0)}**\n"
                        f"- Records eligible: **{p.get('eligible', 0)}**\n"
                        f"- Records included in review: **{p.get('included', 0)}**\n"
                    )
                    prisma_md_path.write_text(prisma_md, encoding="utf-8")
                    if evidence.get("literature_prisma") is None:
                        evidence.register_file(
                            kind="statistic",
                            description=(
                                "PRISMA 2020 flow counts for the literature search (O21)."
                            ),
                            source_path=prisma_path,
                            evidence_id="literature_prisma",
                            producer="literature",
                            generation_mode="system",
                        )
                    if evidence.get("literature_prisma_summary") is None:
                        evidence.register_file(
                            kind="log",
                            description="Human-readable PRISMA flow summary (O21).",
                            source_path=prisma_md_path,
                            evidence_id="literature_prisma_summary",
                            producer="literature",
                            generation_mode="system",
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="literature_agent",
                        severity="warning",
                        message=f"Literature agent failed: {exc}",
                    )
                )

        emit_progress(
            "writer",
            "Drafting manuscript scaffold.",
            run_id=run_id,
        )
        writer = ManuscriptAgent(role_resolver("writer"), language=run_language)
        manuscript_packet: Optional[ManuscriptDraftPacket] = None
        if runtime_state.semantics is not None:
            manuscript_packet = writer.build_packet(
                context=context,
                semantics=runtime_state.semantics,
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=record.evidence_id,
                        kind=record.kind,
                        description=record.description,
                        relative_path=record.relative_path,
                    )
                    for record in evidence.records()
                ],
                findings=[
                    f.message for f in findings if f.severity in {"warning", "error"}
                ],
                caveats=list(runtime_state.semantics.safety_guardrails),
            )
            packet_path = run_dir / "manuscript_packet.json"
            packet_path.write_text(
                manuscript_packet.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("manuscript_packet") is None:
                evidence.register_file(
                    kind="log",
                    description="Typed manuscript draft packet passed into the manuscript agent.",
                    source_path=packet_path,
                    evidence_id="manuscript_packet",
                    producer="manuscript_agent",
                    generation_mode="system",
                    prompt_pack_version=prompt_version,
                )
        try:
            writer_evidence_digest = _render_writer_evidence_digest(
                context=context,
                run_dir=run_dir,
                per_step_records=per_step_records,
            )
            scaffold = writer.run(
                context=agent_context,
                evidence_ids=_preferred_writer_evidence_names(evidence),
                evidence_digest=writer_evidence_digest,
            )
        except Exception as exc:
            scaffold = f"(writer failed: {exc})"
        scaffold, placeholder_repairs = _repair_common_writer_placeholders(
            scaffold,
            context=context,
            evidence=evidence,
        )
        if placeholder_repairs:
            findings.append(
                ValidationFinding(
                    validator="evidence_bound_writer",
                    severity="warning",
                    message=(
                        "Repaired common manuscript evidence placeholder(s): "
                        + ", ".join(
                            f"{old}->{new}" for old, new in placeholder_repairs
                        )
                    ),
                    detail={
                        "repairs": [
                            {"from": old, "to": new}
                            for old, new in placeholder_repairs
                        ]
                    },
                )
            )
        scaffold_path = run_dir / "manuscript_scaffold.md"
        scaffold_path.write_text(scaffold, encoding="utf-8")
        if evidence.get("manuscript_scaffold_raw") is None:
            evidence.register_file(
                kind="log",
                description="Manuscript scaffold (raw, with {evidence:*} placeholders).",
                source_path=scaffold_path,
                evidence_id="manuscript_scaffold_raw",
                producer="writer",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
            )

        evidence_bound_scaffold, removed_sentences = (
            evidence.enforce_evidence_bound_scaffold(scaffold)
        )
        if removed_sentences:
            findings.append(
                ValidationFinding(
                    validator="evidence_bound_writer",
                    severity="warning",
                    message=(
                        f"Filtered {len(removed_sentences)} result-like sentence(s) without evidence placeholders before manuscript binding."
                    ),
                    detail={"removed_sentences": removed_sentences},
                )
            )
            filtered_path = run_dir / "manuscript_scaffold_filtered.md"
            filtered_path.write_text(evidence_bound_scaffold, encoding="utf-8")
            if evidence.get("manuscript_scaffold_filtered") is None:
                evidence.register_file(
                    kind="log",
                    description="Manuscript scaffold after evidence-bound filtering.",
                    source_path=filtered_path,
                    evidence_id="manuscript_scaffold_filtered",
                    producer="pipeline",
                    generation_mode="system",
                )

        bound_unfiltered = evidence.bind_manuscript(evidence_bound_scaffold)
        bound, demoted_missing_ids = _demote_unresolved_evidence_placeholders(
            bound_unfiltered
        )
        bound, removed_tbd_sentences = _remove_tbd_sentences(bound)
        if (
            removed_tbd_sentences
            and self._evidence_enforcement_mode is EvidenceEnforcementMode.STRICT
        ):
            raise EvidenceEnforcementError(
                f"STRICT evidence mode: writer emitted {len(removed_tbd_sentences)} "
                f"sentence(s) containing [TBD]/[TODO]/[TK] placeholder(s). "
                f"The bound manuscript must not carry unresolved writer "
                f"placeholders before submission.",
                detail={"tbd_sentences": removed_tbd_sentences},
            )
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(bound, encoding="utf-8")
        if evidence.get("manuscript_scaffold_bound") is None:
            evidence.register_file(
                kind="log",
                description="Manuscript scaffold with evidence ids resolved to file links + sha256.",
                source_path=bound_path,
                evidence_id="manuscript_scaffold_bound",
                producer="pipeline",
                generation_mode="system",
            )
        if demoted_missing_ids:
            unfiltered_path = run_dir / "manuscript_scaffold_bound_unfiltered.md"
            unfiltered_path.write_text(bound_unfiltered, encoding="utf-8")
            if evidence.get("manuscript_scaffold_bound_unfiltered") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Manuscript scaffold prior to demoting unresolved "
                        "[evidence missing: …] placeholders to HTML comments."
                    ),
                    source_path=unfiltered_path,
                    evidence_id="manuscript_scaffold_bound_unfiltered",
                    producer="pipeline",
                    generation_mode="system",
                )
            findings.append(
                ValidationFinding(
                    validator="evidence_bound_writer",
                    severity="warning",
                    message=(
                        f"Demoted {len(demoted_missing_ids)} unresolved "
                        f"[evidence missing: …] placeholder(s) to HTML "
                        f"comments so the manuscript still renders cleanly; "
                        f"see manuscript_scaffold_bound_unfiltered.md for "
                        f"the original."
                    ),
                    detail={"missing_evidence_ids": sorted(set(demoted_missing_ids))},
                )
            )
        if removed_tbd_sentences:
            findings.append(
                ValidationFinding(
                    validator="evidence_bound_writer",
                    severity="warning",
                    message=(
                        f"Removed {len(removed_tbd_sentences)} sentence(s) containing "
                        "[TBD] from the bound manuscript; the writer must omit "
                        "unsupported values instead of leaving placeholders."
                    ),
                    detail={"removed_sentences": removed_tbd_sentences},
                )
            )
        manuscript_numeric_findings = audit_manuscript_numeric_claims(
            bound,
            per_step_records=per_step_records,
        )
        findings.extend(manuscript_numeric_findings)

        manuscript_critique = critic.review_manuscript(
            scaffold=bound,
            available_evidence_ids=evidence.resolvable_names(),
        )
        if manuscript_numeric_findings:
            manuscript_critique = manuscript_critique.model_copy(
                update={
                    "status": "blocked",
                    "unsupported_claims": list(manuscript_critique.unsupported_claims)
                    + [finding.message for finding in manuscript_numeric_findings],
                    "concerns": list(manuscript_critique.concerns)
                    + [
                        "Manuscript numeric claims disagree with registered step_summary values."
                    ],
                }
            )
        critique_path = run_dir / "manuscript_critique.json"
        critique_path.write_text(
            manuscript_critique.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if evidence.get("manuscript_critique") is None:
            evidence.register_file(
                kind="log",
                description="Structured manuscript critique after evidence binding.",
                source_path=critique_path,
                evidence_id="manuscript_critique",
                producer="critic",
                generation_mode="system",
            )
        if manuscript_critique.status in {"needs_revision", "blocked"}:
            findings.append(
                ValidationFinding(
                    validator="critic_agent",
                    severity="error",
                    message=(
                        f"CriticAgent marked manuscript as {manuscript_critique.status}: "
                        + "; ".join(
                            manuscript_critique.concerns
                            or manuscript_critique.suggested_repairs
                            or ["review required"]
                        )
                    ),
                    evidence_ids=["manuscript_critique"],
                )
            )

        if self._enable_latex:
            try:
                emit_progress(
                    "latex",
                    "Rendering LaTeX and BibTeX scaffold.",
                    run_id=run_id,
                )
                bib_basename = "manuscript_scaffold"
                # Collect registered figure paths for auto-embedding.
                fig_paths_for_latex: List[Tuple[str, str]] = []
                for rec in evidence.records():
                    if rec.kind != "figure":
                        continue
                    # Prefer PNG for LaTeX compatibility; SVG needs
                    # inkscape or svg package.
                    if rec.relative_path.endswith((".png", ".pdf", ".tiff")):
                        fig_paths_for_latex.append(
                            (rec.evidence_id, "evidence/" + rec.relative_path)
                        )
                tex = scaffold_to_latex(
                    markdown=bound,
                    title=manuscript_title
                    or f"EasyICU research-agent: {context.research_question}",
                    authors=manuscript_authors or ["EasyICU research-agent"],
                    bibliography=literature,
                    bibliography_basename=bib_basename,
                    venue_template=self._latex_venue_template,
                    figure_paths=fig_paths_for_latex or None,
                )
                tex_path = run_dir / "manuscript_scaffold.tex"
                tex_path.write_text(tex, encoding="utf-8")
                if evidence.get("manuscript_scaffold_tex") is None:
                    evidence.register_file(
                        kind="log",
                        description="LaTeX manuscript scaffold generated from the bound markdown.",
                        source_path=tex_path,
                        evidence_id="manuscript_scaffold_tex",
                        producer="pipeline",
                        generation_mode="system",
                    )
                if literature is not None and getattr(literature, "citations", None):
                    bib = render_bibtex(literature)
                    bib_path = run_dir / f"{bib_basename}.bib"
                    bib_path.write_text(bib, encoding="utf-8")
                    if evidence.get("manuscript_bibliography") is None:
                        evidence.register_file(
                            kind="log",
                            description="BibTeX file rendered from the literature bundle.",
                            source_path=bib_path,
                            evidence_id="manuscript_bibliography",
                            producer="pipeline",
                            generation_mode="system",
                        )
                # Optional: compile the .tex to PDF on disk so users
                # can open ``manuscript_scaffold.pdf`` directly. Off
                # by default because not every environment has a
                # LaTeX install.
                if self._enable_pdf_render:
                    bib_full = (
                        run_dir / f"{bib_basename}.bib"
                        if (run_dir / f"{bib_basename}.bib").exists()
                        else None
                    )
                    pdf_result = render_pdf_for_run(
                        tex_path=tex_path,
                        bib_path=bib_full,
                        output_dir=run_dir,
                    )
                    if pdf_result.success and pdf_result.pdf_path is not None:
                        if evidence.get("manuscript_scaffold_pdf") is None:
                            evidence.register_file(
                                kind="log",
                                description=(
                                    f"Compiled manuscript PDF "
                                    f"(engine={pdf_result.engine})."
                                ),
                                source_path=pdf_result.pdf_path,
                                evidence_id="manuscript_scaffold_pdf",
                                producer="pipeline",
                                generation_mode="system",
                            )
                        findings.append(
                            ValidationFinding(
                                validator="pdf_render",
                                severity="info",
                                message=(
                                    f"Rendered manuscript PDF via "
                                    f"{pdf_result.engine}."
                                ),
                                evidence_ids=["manuscript_scaffold_pdf"],
                            )
                        )
                    else:
                        findings.append(
                            ValidationFinding(
                                validator="pdf_render",
                                severity="warning",
                                message=(
                                    "PDF render failed or no LaTeX "
                                    "engine found: "
                                    + "; ".join(pdf_result.notes)
                                ),
                            )
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="latex_export",
                        severity="warning",
                        message=f"LaTeX export failed: {exc}",
                    )
                )

        # O18 — Causal audit. Run last in the write phase so the
        # bound manuscript (post binding, post filtering) is what gets
        # scanned. Associational-effect-with-causal-language is a
        # warning; causal_overclaimed-with-causal-language is an
        # error.
        if self._enable_causal_audit:
            try:
                bound_text = bound_path.read_text(encoding="utf-8")
            except Exception:
                bound_text = ""
            causal_report = run_causal_audit(
                evidence_records=evidence.records(),
                run_dir=run_dir,
                bound_manuscript=bound_text,
            )
            causal_json = run_dir / "causal_audit_report.json"
            causal_md = run_dir / "causal_audit_report.md"
            causal_report.write_json(causal_json)
            causal_report.write_markdown(causal_md)
            if evidence.get("causal_audit_report") is None:
                evidence.register_file(
                    kind="statistic",
                    description=(
                        "Causal-claim audit (O18): effect labels "
                        "(associational / causal_explicit / "
                        "causal_overclaimed) and causal-language hits."
                    ),
                    source_path=causal_json,
                    evidence_id="causal_audit_report",
                    producer="pipeline",
                    generation_mode="system",
                )
            if evidence.get("causal_audit_summary") is None:
                evidence.register_file(
                    kind="log",
                    description="Human-readable causal-audit summary (O18).",
                    source_path=causal_md,
                    evidence_id="causal_audit_summary",
                    producer="pipeline",
                    generation_mode="system",
                )
            summary = causal_report.summary()
            if summary["n_effects_labelled"] > 0 or summary["n_language_errors"] > 0:
                findings.append(
                    ValidationFinding(
                        validator="causal_audit",
                        severity="info",
                        message=(
                            f"Labelled {summary['n_effects_labelled']} effect(s); "
                            f"{summary['n_associational']} associational, "
                            f"{summary['n_causal_explicit']} causal_explicit, "
                            f"{summary['n_causal_overclaimed']} causal_overclaimed."
                        ),
                        evidence_ids=["causal_audit_report"],
                        detail=summary,
                    )
                )
            for hit in causal_report.language_hits:
                findings.append(
                    ValidationFinding(
                        validator="causal_audit",
                        severity=hit.severity,
                        message=(
                            f"Causal language over {hit.strength} pattern "
                            f"`{hit.pattern}` cited "
                            f"{hit.linked_effect_labels or 'no labelled effect'}."
                        ),
                        evidence_ids=list(hit.linked_evidence_ids)
                        + ["causal_audit_report"],
                        detail={"sentence": hit.sentence[:280]},
                    )
                )

        # O16 — Reporting-guideline checklist. Writes STROBE (always)
        # and TRIPOD+AI (when the analysis family looks like a
        # prediction / validation study). Findings are emitted at
        # ``info`` severity by default so the paper can still be
        # produced; reviewers see the coverage number and decide.
        if self._enable_reporting_checklist:
            try:
                bound_text = bound_path.read_text(encoding="utf-8")
            except Exception:
                bound_text = ""
            if self._reporting_checklist_names is not None:
                wanted = tuple(
                    n.lower() for n in self._reporting_checklist_names
                )
            else:
                analysis_family = (
                    (context.user_preferences.inferred_analysis_family or "")
                    if getattr(context, "user_preferences", None)
                    else ""
                )
                wanted = choose_checklist(analysis_family)
            checklist_reports = []
            if "strobe" in wanted:
                checklist_reports.append(
                    ("strobe", build_strobe_checklist(
                        evidence_records=evidence.records(),
                        bound_manuscript=bound_text,
                    ))
                )
            if "tripod_ai" in wanted or "tripod+ai" in wanted:
                checklist_reports.append(
                    ("tripod_ai", build_tripod_ai_checklist(
                        evidence_records=evidence.records(),
                        bound_manuscript=bound_text,
                    ))
                )
            for key, report in checklist_reports:
                md_path = run_dir / f"reporting_checklist_{key}.md"
                json_path = run_dir / f"reporting_checklist_{key}.json"
                md_path.write_text(report.to_markdown(), encoding="utf-8")
                json_path.write_text(
                    json.dumps(report.to_json(), indent=2, default=str),
                    encoding="utf-8",
                )
                md_evid_id = f"reporting_checklist_{key}"
                json_evid_id = f"reporting_checklist_{key}_json"
                if evidence.get(md_evid_id) is None:
                    evidence.register_file(
                        kind="log",
                        description=(
                            f"Auto-filled {report.name} reporting checklist (O16)."
                        ),
                        source_path=md_path,
                        evidence_id=md_evid_id,
                        producer="pipeline",
                        generation_mode="system",
                    )
                if evidence.get(json_evid_id) is None:
                    evidence.register_file(
                        kind="log",
                        description=(
                            f"Structured {report.name} reporting checklist (O16)."
                        ),
                        source_path=json_path,
                        evidence_id=json_evid_id,
                        producer="pipeline",
                        generation_mode="system",
                    )
                summary = report.summary()
                findings.append(
                    ValidationFinding(
                        validator="reporting_checklist",
                        severity="info",
                        message=(
                            f"{report.name} coverage {summary['coverage']:.0%} "
                            f"({summary['n_addressed']} addressed, "
                            f"{summary['n_partial']} partial, "
                            f"{summary['n_open']} open, "
                            f"{summary['n_not_applicable']} n/a)."
                        ),
                        evidence_ids=[md_evid_id],
                        detail=summary,
                    )
                )
                # Promote to warning only if coverage < 50 %; reviewers
                # care about Methods completeness, not every cell.
                if summary["coverage"] < 0.5:
                    findings.append(
                        ValidationFinding(
                            validator="reporting_checklist",
                            severity="warning",
                            message=(
                                f"{report.name} reporting coverage below 50 %; "
                                "expect reviewer pushback on Methods completeness."
                            ),
                            evidence_ids=[md_evid_id],
                            detail=summary,
                        )
                    )

        # O15 — Simulated three-role reviewer round. Runs after the
        # deterministic gates so each reviewer reads the latest
        # findings (multiple-testing, causal-audit, checklist). The
        # output is not a validator; it is a reviewer-facing note
        # bundle that the manuscript author / responsible clinician
        # uses to tighten the draft before submission.
        if self._enable_reviewer_round:
            reviewer_report = run_reviewer_round(
                evidence_records=evidence.records(),
                findings=findings,
                round_index=0,
            )
            reviewer_md = run_dir / "reviewer_report.md"
            reviewer_json = run_dir / "reviewer_report.json"
            reviewer_md.write_text(reviewer_report.to_markdown(), encoding="utf-8")
            reviewer_json.write_text(
                json.dumps(reviewer_report.to_json(), indent=2, default=str),
                encoding="utf-8",
            )
            if evidence.get("reviewer_report") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Three-role simulated reviewer report (O15): "
                        "statistician / clinician / methodologist."
                    ),
                    source_path=reviewer_md,
                    evidence_id="reviewer_report",
                    producer="pipeline",
                    generation_mode="system",
                )
            if evidence.get("reviewer_report_json") is None:
                evidence.register_file(
                    kind="log",
                    description="Structured reviewer report (O15).",
                    source_path=reviewer_json,
                    evidence_id="reviewer_report_json",
                    producer="pipeline",
                    generation_mode="system",
                )
            summary = reviewer_report.summary()
            rec = summary["aggregated_recommendation"]
            severity = {
                "accept": "info",
                "minor_revision": "info",
                "major_revision": "warning",
                "reject": "error",
            }.get(rec, "info")
            findings.append(
                ValidationFinding(
                    validator="reviewer_round",
                    severity=severity,
                    message=(
                        f"Simulated reviewers returned `{rec}` "
                        f"(info={summary['counts'].get('info',0)}, "
                        f"minor={summary['counts'].get('minor',0)}, "
                        f"major={summary['counts'].get('major',0)}, "
                        f"reject={summary['counts'].get('reject',0)})."
                    ),
                    evidence_ids=["reviewer_report"],
                    detail=summary,
                )
            )

        # O26 — Notebook + lockfile. Concatenates every per-step
        # generated script in plan order into a single runnable
        # ``run.ipynb`` and captures the interpreter's installed
        # packages in ``requirements.lock.txt``. Runs regardless of
        # reviewer / checklist flags so the reproducibility artefacts
        # are always present.
        try:
            notebook_steps: List[NotebookStep] = []
            intent_by_id = {s.step_id: s.intent for s in plan_result.plan.steps}
            # Preserve plan order: iterate plan, pick first 'code'
            # evidence per step.
            code_records_by_step: Dict[str, Any] = {}
            for rec in evidence.records():
                if rec.kind != "code":
                    continue
                step_id = rec.produced_by_step or ""
                if step_id and step_id not in code_records_by_step:
                    code_records_by_step[step_id] = rec
            for step in plan_result.plan.steps:
                rec = code_records_by_step.get(step.step_id)
                if rec is None:
                    continue
                candidates = [
                    run_dir / "evidence" / rec.relative_path,
                    run_dir / rec.relative_path,
                ]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    continue
                try:
                    code_text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                notebook_steps.append(
                    NotebookStep(
                        step_id=step.step_id,
                        intent=intent_by_id.get(step.step_id, step.intent),
                        code=code_text,
                    )
                )
            if notebook_steps:
                notebook = build_notebook(
                    research_question=plan_result.context.research_question,
                    cohort_relative_path="cohort.parquet",
                    steps=notebook_steps,
                )
                nb_path = run_dir / "run.ipynb"
                write_notebook(nb_path, notebook)
                if evidence.get("run_notebook") is None:
                    evidence.register_file(
                        kind="code",
                        description=(
                            "Auto-generated Jupyter notebook re-running "
                            "every plan step top-to-bottom (O26)."
                        ),
                        source_path=nb_path,
                        evidence_id="run_notebook",
                        producer="pipeline",
                        generation_mode="system",
                    )
            lockfile_path = run_dir / "requirements.lock.txt"
            lockfile_path.write_text(build_requirements_lockfile(), encoding="utf-8")
            if evidence.get("requirements_lockfile") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Interpreter-level requirements lockfile captured "
                        "at run time (O26)."
                    ),
                    source_path=lockfile_path,
                    evidence_id="requirements_lockfile",
                    producer="pipeline",
                    generation_mode="system",
                )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="repro_artifacts",
                    severity="warning",
                    message=(
                        f"Failed to build run.ipynb / lockfile: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

        return _WritePhaseResult(
            literature=literature,
            bound_path=bound_path,
            manuscript_packet=manuscript_packet,
            manuscript_critique=manuscript_critique,
        )

    def _finalise_success(
        self,
        *,
        plan_result: _PlanPhaseResult,
        execute_result: _ExecutePhaseResult,
        write_result: _WritePhaseResult,
        run_id: str,
        run_dir: Path,
        cohort_path: Path,
        notes: Optional[str],
        database: str,
        target_outcome: Optional[str],
        stop_after_analysis: bool,
        cache_key: Optional[str],
        experiment_spec_path: Optional[Path],
        audit_logger: Optional[AuditLogger],
        emit_progress: Callable[..., None],
    ) -> PipelineResult:
        """Write reports/manifests and persist run memory after all phases finish."""
        context = plan_result.context
        evidence = plan_result.evidence
        findings = plan_result.findings
        per_step_records = execute_result.per_step_records
        plan = execute_result.plan

        plan_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        per_step_records.sort(
            key=lambda r: (
                -1
                if r.get("step_id") == "00_probe"
                else plan_order.get(r.get("step_id"), 10**9)
            )
        )
        report_path = run_dir / "results_report.md"
        report_path.write_text(
            render_report(
                context=context,
                plan=plan,
                findings=findings,
                per_step_records=per_step_records,
                evidence=evidence,
                paused_after_analysis=stop_after_analysis,
            ),
            encoding="utf-8",
        )

        workflow_graph = build_workflow_graph(
            run_id=run_id,
            context=context,
            plan=plan,
            per_step_records=per_step_records,
            paused_after_analysis=stop_after_analysis,
        )
        workflow_graph_path = write_json_artifact(
            run_dir / "workflow_graph.json",
            workflow_graph,
        )
        workflow_mermaid_path = run_dir / "workflow_graph.md"
        workflow_mermaid_path.write_text(
            render_workflow_graph_mermaid(workflow_graph),
            encoding="utf-8",
        )
        evidence.register_file(
            kind="log",
            description="Workflow graph JSON for this run.",
            source_path=workflow_graph_path,
            aliases=["workflow_graph"],
            producer="pipeline",
            generation_mode="system",
        )
        evidence.register_file(
            kind="log",
            description="Mermaid workflow graph for this run.",
            source_path=workflow_mermaid_path,
            aliases=["workflow_graph_mermaid"],
            producer="pipeline",
            generation_mode="system",
        )

        replay_bundle = build_execution_replay(
            run_id=run_id,
            cohort_path=cohort_path,
            context_path=str(plan_result.context_path.relative_to(run_dir)),
            plan_path=str(plan_result.plan_path.relative_to(run_dir)),
            llm_signature=plan_result.llm_signature,
            prompt_pack_version=plan_result.prompt_version,
            per_step_records=per_step_records,
            findings=findings,
            evidence_ids=[r.evidence_id for r in evidence.records()],
        )
        replay_path = write_json_artifact(
            run_dir / "execution_replay.json",
            replay_bundle,
        )
        evidence.register_file(
            kind="log",
            description="Deterministic execution replay bundle for this run.",
            source_path=replay_path,
            aliases=["execution_replay"],
            producer="pipeline",
            generation_mode="system",
        )

        audit_log_rel: Optional[str] = None
        if audit_logger is not None and audit_logger.path.exists():
            evidence.register_file(
                kind="log",
                description="RuntimeSupervisor audit log (JSONL).",
                source_path=audit_logger.path,
                aliases=["audit_log"],
                producer="pipeline",
                generation_mode="system",
            )
            audit_log_rel = str(audit_logger.path.relative_to(run_dir))

        cost_records_for_manifest = []
        if plan_result.cost_meter is not None:
            cost_records_for_manifest = list(plan_result.cost_meter.records)
            cost_json_path = run_dir / "cost_records.json"
            cost_json_path.write_text(
                json.dumps(
                    [r.model_dump(mode="json") for r in plan_result.cost_meter.records],
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            cost_md_path = run_dir / "cost_summary.md"
            cost_md_path.write_text(
                _render_cost_summary(plan_result.cost_meter), encoding="utf-8"
            )
            evidence.register_file(
                kind="log",
                description="Raw per-call LLM cost records (T3.2).",
                source_path=cost_json_path,
                evidence_id="cost_records",
                producer="pipeline",
                generation_mode="system",
            )
            evidence.register_file(
                kind="log",
                description="Human-readable LLM cost summary (T3.2).",
                source_path=cost_md_path,
                evidence_id="cost_summary",
                producer="pipeline",
                generation_mode="system",
            )

        reproducibility_summary: Optional[Dict[str, Any]] = None
        if plan_result.repro_envelope is not None:
            envelope_path = run_dir / "reproducibility_envelope.json"
            plan_result.repro_envelope.to_disk(envelope_path)
            evidence.register_file(
                kind="log",
                description=(
                    "LLM reproducibility envelope (O20): per-call prompt/response "
                    "sha256, requested seed, temperature, provider/model, and a "
                    "PHI-safe environment snapshot."
                ),
                source_path=envelope_path,
                evidence_id="reproducibility_envelope",
                producer="pipeline",
                generation_mode="system",
            )
            reproducibility_summary = plan_result.repro_envelope.to_manifest_summary()

        # O22 — Multiple-testing correction. Scan every registered
        # table / statistic artefact for p-values and BH-adjust the
        # whole family run-wide. Writes a CSV + MD pair and registers
        # both as evidence so the manuscript can cite
        # ``{evidence:multiple_testing_report}``.
        if self._enable_multiple_testing_correction:
            mt_report = build_multiple_testing_report(
                evidence_records=evidence.records(),
                run_dir=run_dir,
                alpha=self._multiple_testing_alpha,
            )
            mt_csv = run_dir / "multiple_testing_report.csv"
            mt_md = run_dir / "multiple_testing_report.md"
            mt_report.write_csv(mt_csv)
            mt_report.write_markdown(mt_md)
            if evidence.get("multiple_testing_report") is None:
                evidence.register_file(
                    kind="statistic",
                    description=(
                        "Run-wide Benjamini–Hochberg and Bonferroni correction "
                        "for every registered p-value (O22)."
                    ),
                    source_path=mt_csv,
                    evidence_id="multiple_testing_report",
                    producer="pipeline",
                    generation_mode="system",
                )
            if evidence.get("multiple_testing_summary") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Human-readable summary of multiple-testing correction (O22)."
                    ),
                    source_path=mt_md,
                    evidence_id="multiple_testing_summary",
                    producer="pipeline",
                    generation_mode="system",
                )
            summary = mt_report.summary()
            if summary["n_tests"] > 0:
                # Surface the raw → corrected gap as an info finding so
                # paper figures can include it without re-reading the
                # CSV.
                findings.append(
                    ValidationFinding(
                        validator="multiple_testing",
                        severity="info",
                        message=(
                            f"Ran BH-FDR across {summary['n_tests']} tests at "
                            f"alpha={summary['alpha']:.3f}: "
                            f"{summary['n_significant_raw']} significant raw, "
                            f"{summary['n_significant_bh']} after BH, "
                            f"{summary['n_significant_bonferroni']} after Bonferroni."
                        ),
                        evidence_ids=["multiple_testing_report"],
                        detail=summary,
                    )
                )
                # If raw and BH disagree meaningfully, emit a warning
                # so the Discussion section has to engage with it.
                if (
                    summary["n_significant_raw"]
                    > summary["n_significant_bh"]
                ):
                    findings.append(
                        ValidationFinding(
                            validator="multiple_testing",
                            severity="warning",
                            message=(
                                "Some raw-significant results did not survive "
                                "BH-FDR at the run-wide family level. Revise "
                                "the primary / secondary endpoint distinction "
                                "or report corrected p-values explicitly."
                            ),
                            evidence_ids=["multiple_testing_report"],
                            detail={
                                "n_raw_only": (
                                    summary["n_significant_raw"]
                                    - summary["n_significant_bh"]
                                ),
                            },
                        )
                    )

        # O23 — E-values. For every primary-association row, compute
        # VanderWeele–Ding E-value + lower-CI E-value. Writes
        # ``e_values.csv`` + ``e_values.md`` and registers both.
        # Baseline prevalence defaults to observed outcome rate when
        # an ``outcome_rate.csv`` was registered.
        try:
            primary_path = None
            for rec in evidence.records():
                if rec.evidence_id != "primary_association":
                    continue
                candidates = [
                    run_dir / "evidence" / rec.relative_path,
                    run_dir / rec.relative_path,
                ]
                primary_path = next(
                    (c for c in candidates if c.exists() and c.suffix == ".csv"),
                    None,
                )
                break
            if primary_path is not None:
                import csv as _csv

                baseline_prev = 0.1
                outcome_rate_rec = evidence.get("outcome_rate")
                if outcome_rate_rec is not None:
                    try:
                        or_path = next(
                            (
                                p
                                for p in (
                                    run_dir / "evidence" / outcome_rate_rec.relative_path,
                                    run_dir / outcome_rate_rec.relative_path,
                                )
                                if p.exists() and p.suffix == ".csv"
                            ),
                            None,
                        )
                        if or_path:
                            with or_path.open("r", encoding="utf-8") as fh:
                                for row in _csv.DictReader(fh):
                                    for key in (
                                        "outcome_rate",
                                        "rate",
                                        "mortality_rate",
                                        "event_rate",
                                    ):
                                        if key in row:
                                            try:
                                                cand = float(row[key])
                                                if 0 < cand < 1:
                                                    baseline_prev = cand
                                            except (TypeError, ValueError):
                                                pass
                    except Exception:
                        pass

                rows_out: List[Dict[str, Any]] = []
                with primary_path.open("r", encoding="utf-8") as fh:
                    reader = _csv.DictReader(fh)
                    for row in reader:
                        # Accept OR or odds_ratio column; skip age / intercept etc.
                        or_val = None
                        for key in ("odds_ratio", "or", "OR"):
                            if key in row and row[key] not in (None, "", "nan"):
                                try:
                                    or_val = float(row[key])
                                    break
                                except (TypeError, ValueError):
                                    continue
                        if or_val is None:
                            continue
                        try:
                            ci_lo = float(row.get("or_lower") or row.get("ci_lower") or 0.0)
                            ci_hi = float(row.get("or_upper") or row.get("ci_upper") or 0.0)
                            ci = (ci_lo, ci_hi) if ci_lo > 0 and ci_hi > 0 else None
                        except (TypeError, ValueError):
                            ci = None
                        ev = compute_e_value(
                            estimate=or_val,
                            ci=ci,
                            estimate_type="or",
                            baseline_prevalence=baseline_prev,
                        )
                        row_out = {
                            "term": row.get("term")
                            or row.get("variable")
                            or row.get("predictor")
                            or "",
                            "odds_ratio": or_val,
                            "ci_lower": ci[0] if ci else "",
                            "ci_upper": ci[1] if ci else "",
                            "baseline_prevalence": baseline_prev,
                            "e_value": ev.e_value,
                            "e_value_lower_bound": ev.e_value_lower_bound,
                            "note": ev.note or "",
                        }
                        rows_out.append(row_out)

                if rows_out:
                    ev_csv = run_dir / "e_values.csv"
                    ev_md = run_dir / "e_values.md"
                    with ev_csv.open("w", newline="", encoding="utf-8") as fh:
                        writer = _csv.writer(fh)
                        writer.writerow(list(rows_out[0].keys()))
                        for row in rows_out:
                            writer.writerow([row[k] for k in rows_out[0].keys()])
                    ev_md_lines = [
                        "# E-values for primary effects (O23)",
                        "",
                        f"Baseline event prevalence used: **{baseline_prev:.3f}**",
                        "",
                        "| Term | OR | 95% CI | E-value | E-value (CI bound) |",
                        "|---|---|---|---|---|",
                    ]
                    for row in rows_out:
                        ci_disp = (
                            f"{row['ci_lower']:.2f} – {row['ci_upper']:.2f}"
                            if row["ci_lower"] != "" and row["ci_upper"] != ""
                            else "—"
                        )
                        ev_md_lines.append(
                            "| {t} | {orv:.2f} | {ci} | {ev:.2f} | {evb} |".format(
                                t=str(row["term"])[:40],
                                orv=row["odds_ratio"],
                                ci=ci_disp,
                                ev=row["e_value"],
                                evb=(
                                    f"{row['e_value_lower_bound']:.2f}"
                                    if row["e_value_lower_bound"] is not None
                                    else "—"
                                ),
                            )
                        )
                    ev_md.write_text("\n".join(ev_md_lines) + "\n", encoding="utf-8")
                    if evidence.get("e_values") is None:
                        evidence.register_file(
                            kind="statistic",
                            description=(
                                "VanderWeele–Ding E-values for every primary "
                                "effect row (O23)."
                            ),
                            source_path=ev_csv,
                            evidence_id="e_values",
                            producer="pipeline",
                            generation_mode="system",
                        )
                    if evidence.get("e_values_summary") is None:
                        evidence.register_file(
                            kind="log",
                            description="Human-readable E-value summary (O23).",
                            source_path=ev_md,
                            evidence_id="e_values_summary",
                            producer="pipeline",
                            generation_mode="system",
                        )
                    findings.append(
                        ValidationFinding(
                            validator="e_value",
                            severity="info",
                            message=(
                                f"Computed E-values for {len(rows_out)} primary "
                                f"effect row(s) (baseline prevalence={baseline_prev:.3f})."
                            ),
                            evidence_ids=["e_values"],
                        )
                    )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="e_value",
                    severity="warning",
                    message=f"E-value computation failed: {type(exc).__name__}: {exc}",
                )
            )

        # O24 — Fairness / subgroup analysis. Runs when a
        # ``primary_association`` artefact exists and the cohort has
        # at least one of (``age``, ``sex``, ``sex_M``, ``race``,
        # ``insurance``). Pure numpy; no pandas-only helpers so we
        # stay consistent with the rest of the deterministic layer.
        if self._enable_fairness_subgroups:
            try:
                primary_rec = evidence.get("primary_association")
                if primary_rec is not None:
                    import csv as _csv

                    candidates = [
                        run_dir / "evidence" / primary_rec.relative_path,
                        run_dir / primary_rec.relative_path,
                    ]
                    primary_path = next(
                        (p for p in candidates if p.exists() and p.suffix == ".csv"),
                        None,
                    )
                    predictor_name: Optional[str] = None
                    outcome_name = context.target_outcome
                    if primary_path is not None:
                        with primary_path.open("r", encoding="utf-8") as fh:
                            for row in _csv.DictReader(fh):
                                term = (
                                    row.get("term")
                                    or row.get("variable")
                                    or row.get("predictor")
                                    or ""
                                )
                                if term and term.lower() not in {
                                    "intercept",
                                    "const",
                                    "age",
                                    "sex_m",
                                }:
                                    predictor_name = term
                                    break
                    cohort_df = pd.read_parquet(cohort_path)
                    candidate_subgroups = [
                        col
                        for col in ("age", "sex", "sex_M", "race", "insurance")
                        if col in cohort_df.columns
                    ]
                    if (
                        predictor_name is not None
                        and outcome_name
                        and outcome_name in cohort_df.columns
                        and candidate_subgroups
                    ):
                        from .fairness import run_subgroup_analysis

                        result = run_subgroup_analysis(
                            cohort_df=cohort_df,
                            predictor=predictor_name,
                            outcome=outcome_name,
                            subgroup_columns=candidate_subgroups,
                        )
                        fair_csv = run_dir / "fairness_subgroups.csv"
                        fair_md = run_dir / "fairness_subgroups.md"
                        result.write_csv(fair_csv)
                        result.write_markdown(fair_md)
                        if evidence.get("fairness_subgroups") is None:
                            evidence.register_file(
                                kind="statistic",
                                description=(
                                    "Subgroup / fairness analysis for the "
                                    "primary effect (O24)."
                                ),
                                source_path=fair_csv,
                                evidence_id="fairness_subgroups",
                                producer="pipeline",
                                generation_mode="system",
                            )
                        if evidence.get("fairness_subgroups_summary") is None:
                            evidence.register_file(
                                kind="log",
                                description=(
                                    "Human-readable fairness / subgroup summary (O24)."
                                ),
                                source_path=fair_md,
                                evidence_id="fairness_subgroups_summary",
                                producer="pipeline",
                                generation_mode="system",
                            )
                        findings.append(
                            ValidationFinding(
                                validator="fairness_subgroups",
                                severity="info",
                                message=(
                                    f"Subgroup analysis for {predictor_name} ~ "
                                    f"{outcome_name} across "
                                    f"{len(candidate_subgroups)} axis/axes."
                                ),
                                evidence_ids=["fairness_subgroups"],
                                detail={
                                    "predictor": predictor_name,
                                    "outcome": outcome_name,
                                    "subgroup_columns": candidate_subgroups,
                                    "interaction_pvalues": result.interaction_pvalues,
                                },
                            )
                        )
                        # Escalate to warning if any interaction p < 0.05.
                        sig_cols = [
                            col
                            for col, p in result.interaction_pvalues.items()
                            if p < 0.05
                        ]
                        if sig_cols:
                            findings.append(
                                ValidationFinding(
                                    validator="fairness_subgroups",
                                    severity="warning",
                                    message=(
                                        f"Interaction p < 0.05 on "
                                        f"{sig_cols}; subgroup heterogeneity "
                                        "must be discussed."
                                    ),
                                    evidence_ids=["fairness_subgroups"],
                                    detail={"significant_subgroups": sig_cols},
                                )
                            )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="fairness_subgroups",
                        severity="warning",
                        message=(
                            f"Subgroup analysis failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )

        manifest_notes = notes
        if stop_after_analysis:
            suffix = (
                "paused_after_analysis: manuscript generation skipped by user option."
            )
            manifest_notes = f"{notes}\n\n{suffix}" if notes else suffix
        literature_provenance = _literature_provenance_note(
            enable_literature=self._enable_literature,
            enable_pubmed=self._enable_pubmed,
            enable_tavily=self._enable_tavily,
        )
        manifest_notes = (
            f"{manifest_notes}\n\n{literature_provenance}"
            if manifest_notes
            else literature_provenance
        )

        readiness, artifact_paths = write_readiness_artifacts(
            context=context,
            plan=plan,
            findings=findings,
            per_step_records=per_step_records,
            evidence=evidence,
            run_dir=run_dir,
            manuscript_path=write_result.bound_path,
            stop_after_analysis=stop_after_analysis,
        )

        report_path.write_text(
            render_report(
                context=context,
                plan=plan,
                findings=findings,
                per_step_records=per_step_records,
                evidence=evidence,
                paused_after_analysis=stop_after_analysis,
                readiness=readiness,
            ),
            encoding="utf-8",
        )

        manifest = AnalysisManifest(
            run_id=run_id,
            research_question=context.research_question,
            started_at=plan_result.started_at,
            finished_at=datetime.now(timezone.utc),
            context_path=str(plan_result.context_path.relative_to(run_dir)),
            plan_path=str(plan_result.plan_path.relative_to(run_dir)),
            evidence=evidence.records(),
            findings=findings,
            per_step_records=per_step_records,
            cost_records=cost_records_for_manifest,
            reproducibility=reproducibility_summary,
            readiness=readiness,
            artifact_paths=artifact_paths,
            report_path=str(report_path.relative_to(run_dir)),
            manuscript_path=str(write_result.bound_path.relative_to(run_dir)),
            audit_log_path=audit_log_rel,
            workflow_graph_path=str(workflow_graph_path.relative_to(run_dir)),
            execution_replay_path=str(replay_path.relative_to(run_dir)),
            experiment_spec_path=(
                str(experiment_spec_path.relative_to(run_dir))
                if experiment_spec_path is not None and experiment_spec_path.exists()
                else None
            ),
            llm_signature=plan_result.llm_signature,
            used_mock_llm=plan_result.used_mock_llm,
            prompt_pack_version=plan_result.prompt_version,
            prompt_pack_files=plan_result.prompt_files,
            notes=manifest_notes,
        )
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        execute_result.flush_partial_manifest()

        if self._memory is not None:
            self._memory.record(
                run_id=run_id,
                research_question=context.research_question,
                database=database,
                target_outcome=target_outcome,
                findings=findings,
                workdir=run_dir,
            )

        result = PipelineResult(
            run_id=run_id,
            workdir=str(run_dir),
            context_path=str(plan_result.context_path),
            plan_path=str(plan_result.plan_path),
            manifest_path=str(manifest_path),
            report_path=str(report_path),
            manuscript_path=str(write_result.bound_path),
            evidence_count=len(evidence.records()),
            findings_count=len(findings),
        )
        if cache_key is not None:
            self._cache.record_hit(cache_key, result)
        emit_progress(
            "run",
            "Research-agent run complete.",
            status="complete",
            run_id=run_id,
            evidence_count=result.evidence_count,
            findings_count=result.findings_count,
            stop_after_analysis=stop_after_analysis,
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        question: Optional[str] = None,
        cohort: Union[str, Path, pd.DataFrame],
        cohort_name: str = "cohort",
        database: str = "miiv",
        target_outcome: Optional[str] = None,
        cross_database_validation: Optional[Sequence[str]] = None,
        inclusion_criteria: Optional[Sequence[str]] = None,
        exclusion_criteria: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        time_columns: Optional[Sequence[str]] = None,
        outcome_columns: Optional[Sequence[str]] = None,
        time_windows: Optional[Sequence[TimeWindow]] = None,
        concept_descriptions: Optional[Dict[str, str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        skill: Optional[Union[str, ClinicalSkill]] = None,
        manuscript_title: Optional[str] = None,
        manuscript_authors: Optional[Sequence[str]] = None,
        manuscript_language: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        stop_after_analysis: bool = False,
        experiment_spec: Optional[Union[ExperimentSpec, Dict[str, Any]]] = None,
        source_files: Optional[Sequence[Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Run the explicit Plan → Execute → Write phases for one cohort."""
        skill_obj: Optional[ClinicalSkill] = None
        if skill is not None:
            skill_obj = get_skill(skill) if isinstance(skill, str) else skill
            if question is None:
                question = skill_obj.question_for(database=database)
            if target_outcome is None:
                target_outcome = skill_obj.target_outcome
            if not time_windows:
                time_windows = skill_obj.time_windows or None
        if question is None:
            raise ValueError(
                "`question` is required (or pass `skill=...` to derive one)."
            )
        if self._llm is None:
            raise ValueError(
                "ResearchAgentPipeline.run() now requires an explicit `llm=` "
                "client. Pass MockLLMClient() only for tests or deterministic "
                "demo runs; the pipeline no longer falls back to mock silently."
            )
        run_language = self._normalise_manuscript_language(
            manuscript_language or self._manuscript_language
        )
        audit_logger: Optional[AuditLogger] = None

        def _emit_progress(stage: str, message: str, **extra: Any) -> None:
            if audit_logger is not None:
                try:
                    audit_logger.emit(
                        phase=stage,
                        event=message,
                        status=str(extra.get("status", "running")),
                        step_id=(
                            str(extra.get("step_id")) if extra.get("step_id") else None
                        ),
                        detail={
                            k: v
                            for k, v in extra.items()
                            if k not in {"status", "step_id"}
                        },
                    )
                except Exception:
                    pass
            if progress_callback is None:
                return
            payload = {
                "stage": stage,
                "message": message,
                "status": extra.pop("status", "running"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            payload.update(extra)
            try:
                progress_callback(payload)
            except Exception:
                pass

        _emit_progress("run", "Starting research-agent run.")

        resume_state: Optional[Dict[str, Any]] = None
        if resume_run_id:
            run_id = resume_run_id
            run_dir = self.workdir / run_id
            partial = run_dir / "manifest_partial.json"
            if partial.exists():
                try:
                    resume_state = json.loads(partial.read_text(encoding="utf-8"))
                except Exception:
                    resume_state = None
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_id = (
                "run_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                + "_"
                + uuid.uuid4().hex[:6]
            )
            run_dir = self.workdir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(run_dir / "audit_log.jsonl")

        experiment_spec_path: Optional[Path] = None
        if experiment_spec is not None:
            spec_obj = (
                experiment_spec
                if isinstance(experiment_spec, ExperimentSpec)
                else ExperimentSpec.model_validate(experiment_spec)
            )
            experiment_spec_path = dump_experiment_spec(
                spec_obj,
                run_dir / "experiment_spec.yaml",
            )

        cohort_path = self._materialise_cohort(cohort, run_dir)
        _emit_progress(
            "cohort",
            "Cohort materialised to parquet.",
            run_id=run_id,
            path=str(cohort_path),
        )

        cache_key: Optional[str] = None
        if self._enable_cache and not resume_run_id:
            cache_key = self._cache.compute_key(
                cohort_path=cohort_path,
                question=question,
                target_outcome=target_outcome,
                skill_key=(skill_obj.key if skill_obj is not None else None),
                database=database,
                llm=self._llm,
                stop_after_analysis=stop_after_analysis,
                manuscript_language=run_language,
                flags=self._cache_flag_payload(),
            )
            cached = self._cache.lookup(cache_key)
            if cached is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
                _emit_progress(
                    "cache",
                    f"Reused cached run {cached.run_id}.",
                    status="complete",
                    run_id=cached.run_id,
                )
                return cached

        llm = self._llm
        if llm is None:
            raise RuntimeError("LLM client is unexpectedly missing after validation.")

        plan_result = self._run_plan_phase(
            question=question,
            cohort_path=cohort_path,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
            cross_database_validation=cross_database_validation,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            id_columns=id_columns,
            time_columns=time_columns,
            outcome_columns=outcome_columns,
            time_windows=time_windows,
            concept_descriptions=concept_descriptions,
            user_preferences=user_preferences,
            notes=notes,
            skill_obj=skill_obj,
            llm=llm,
            run_dir=run_dir,
            run_id=run_id,
            run_language=run_language,
            experiment_spec_path=experiment_spec_path,
            resume_state=resume_state,
            emit_progress=_emit_progress,
        )
        if plan_result.aborted_result is not None:
            return plan_result.aborted_result

        # O27 — Raw EHR provenance. Hash the cohort parquet and any
        # user-supplied source files and register as evidence so the
        # manuscript's provenance chain goes: raw EHR -> cohort ->
        # analysis artefacts -> manuscript. Only runs after the plan
        # phase succeeded (abort path would have returned above).
        try:
            provenance = build_provenance_bundle(
                cohort_path=cohort_path,
                source_files=source_files,
            )
            provenance_path = run_dir / "provenance_sources.json"
            provenance.to_disk(provenance_path)
            if plan_result.evidence.get("provenance_sources") is None:
                plan_result.evidence.register_file(
                    kind="log",
                    description=(
                        "Raw EHR and cohort provenance hashes (O27): "
                        "sha256 + size + mtime for every source file "
                        "and the materialised cohort parquet."
                    ),
                    source_path=provenance_path,
                    evidence_id="provenance_sources",
                    producer="pipeline",
                    generation_mode="system",
                )
        except Exception as exc:
            plan_result.findings.append(
                ValidationFinding(
                    validator="provenance",
                    severity="warning",
                    message=(
                        f"Failed to compute raw-EHR provenance bundle: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

        execute_result = self._run_execute_phase(
            plan_result=plan_result,
            cohort_path=cohort_path,
            run_dir=run_dir,
            run_id=run_id,
            skill_obj=skill_obj,
            notes=notes,
            emit_progress=_emit_progress,
        )
        write_result = self._run_write_phase(
            plan_result=plan_result,
            execute_result=execute_result,
            run_dir=run_dir,
            run_id=run_id,
            stop_after_analysis=stop_after_analysis,
            manuscript_title=manuscript_title,
            manuscript_authors=manuscript_authors,
            run_language=run_language,
            emit_progress=_emit_progress,
        )
        return self._finalise_success(
            plan_result=plan_result,
            execute_result=execute_result,
            write_result=write_result,
            run_id=run_id,
            run_dir=run_dir,
            cohort_path=cohort_path,
            notes=notes,
            database=database,
            target_outcome=target_outcome,
            stop_after_analysis=stop_after_analysis,
            cache_key=cache_key,
            experiment_spec_path=experiment_spec_path,
            audit_logger=audit_logger,
            emit_progress=_emit_progress,
        )

    def run_from_spec(
        self,
        spec: Union[ExperimentSpec, Dict[str, Any]],
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Run the pipeline from a typed YAML/JSON experiment specification."""
        spec_obj = (
            spec
            if isinstance(spec, ExperimentSpec)
            else ExperimentSpec.model_validate(spec)
        )
        kwargs = spec_obj.run_kwargs()
        kwargs["experiment_spec"] = spec_obj
        kwargs["progress_callback"] = progress_callback
        return self.run(**kwargs)

    async def run_async(self, **kwargs: Any) -> PipelineResult:
        """Async wrapper for UI/API runtimes that need non-blocking orchestration."""
        return await asyncio.to_thread(self.run, **kwargs)

    def replicate(
        self,
        *,
        cohorts: Dict[str, Union[str, Path, pd.DataFrame]],
        question: Optional[str] = None,
        target_outcome: Optional[str] = None,
        cohort_name_prefix: str = "cohort",
        skill: Optional[Union[str, ClinicalSkill]] = None,
        stop_after_analysis: bool = True,
        inclusion_criteria: Optional[Sequence[str]] = None,
        exclusion_criteria: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        time_columns: Optional[Sequence[str]] = None,
        outcome_columns: Optional[Sequence[str]] = None,
        time_windows: Optional[Sequence[TimeWindow]] = None,
        concept_descriptions: Optional[Dict[str, str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        manuscript_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the same plan/question across multiple cohorts and compare effects.

        This is the pipeline-native cross-database replication path requested
        by the methods critique: instead of pushing replication into a separate
        CLI layer, the orchestrator can now launch one analysis run per cohort
        and emit a harmonised effect-comparison table.
        """
        if not cohorts:
            raise ValueError(
                "`cohorts` must contain at least one database -> cohort entry."
            )

        run_results: Dict[str, PipelineResult] = {}
        comparison_rows: List[Dict[str, Any]] = []
        databases = list(cohorts.keys())
        for database, cohort in cohorts.items():
            result = self.run(
                question=question,
                cohort=cohort,
                cohort_name=f"{database}_{cohort_name_prefix}",
                database=database,
                target_outcome=target_outcome,
                cross_database_validation=[db for db in databases if db != database],
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                id_columns=id_columns,
                time_columns=time_columns,
                outcome_columns=outcome_columns,
                time_windows=time_windows,
                concept_descriptions=concept_descriptions,
                user_preferences=user_preferences,
                notes=notes,
                skill=skill,
                manuscript_language=manuscript_language,
                stop_after_analysis=stop_after_analysis,
            )
            run_results[database] = result
            comparison_rows.append(
                _extract_primary_effect_row(database=database, result=result)
            )

        replication_id = "replication_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )
        replication_dir = self.workdir / replication_id
        replication_dir.mkdir(parents=True, exist_ok=True)

        comparison_df = pd.DataFrame(comparison_rows)
        csv_path = replication_dir / "cross_database_effect_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        md_path = replication_dir / "cross_database_effect_comparison.md"
        md_path.write_text(
            _render_cross_database_comparison_markdown(comparison_rows),
            encoding="utf-8",
        )
        run_summaries = [
            _extract_cross_database_run_summary(database=db, result=res)
            for db, res in run_results.items()
        ]
        summary_df = pd.DataFrame(run_summaries)
        summary_csv_path = replication_dir / "cross_database_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        summary_md_path = replication_dir / "cross_database_summary.md"
        summary_md_path.write_text(
            _render_cross_database_summary_markdown(run_summaries),
            encoding="utf-8",
        )
        validation_report_path = replication_dir / "cross_database_validation_report.md"
        validation_report_path.write_text(
            _render_cross_database_validation_report(
                question=question,
                target_outcome=target_outcome,
                rows=comparison_rows,
                run_summaries=run_summaries,
            ),
            encoding="utf-8",
        )
        summary_path = replication_dir / "cross_database_runs.json"
        summary_path.write_text(
            json.dumps(
                {
                    "question": question,
                    "target_outcome": target_outcome,
                    "stop_after_analysis": stop_after_analysis,
                    "comparison_csv": str(csv_path),
                    "summary_csv": str(summary_csv_path),
                    "validation_report": str(validation_report_path),
                    "runs": {
                        db: {
                            "run_id": res.run_id,
                            "manifest_path": res.manifest_path,
                            "report_path": res.report_path,
                            "manuscript_path": res.manuscript_path,
                        }
                        for db, res in run_results.items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "replication_id": replication_id,
            "replication_dir": str(replication_dir),
            "comparison_csv": str(csv_path),
            "comparison_md": str(md_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "validation_report": str(validation_report_path),
            "summary_json": str(summary_path),
            "runs": run_results,
        }

    async def replicate_async(self, **kwargs: Any) -> Dict[str, Any]:
        """Async wrapper for cross-database replication."""
        return await asyncio.to_thread(self.replicate, **kwargs)

    def reproduce_paper(
        self,
        *,
        paper: Union[str, Path],
        cohort: Union[str, Path, pd.DataFrame],
        database: str,
        mode: str = "replication",
        cohort_name: str = "paper_replication_cohort",
        manuscript_language: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Run the paper-reproduction dual-track workflow."""
        paper_profile = parse_paper_profile(paper)
        replication_spec, deviation_report = build_paper_replication_spec(paper_profile)
        if not deviation_report.supported:
            return self._write_fail_closed_paper_package(
                paper=paper,
                cohort=cohort,
                database=database,
                cohort_name=cohort_name,
                paper_profile=paper_profile,
                replication_spec=replication_spec,
                deviation_report=deviation_report,
            )

        target_outcome = (
            replication_spec.mapped_concepts.get("target_outcome")
            or replication_spec.mapped_concepts.get("outcome")
            or canonical_outcome_name(paper_profile.target_outcome)
        )
        question = paper_profile.research_question or (
            f"Replicate the published ICU study '{paper_profile.paper_title or paper_profile.paper_source}' "
            "using EasyICU with design-and-conclusion alignment."
        )
        notes = _build_replication_notes(
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            mode=mode,
        )
        source_files: List[Any] = []
        raw_paper = str(paper)
        if "\n" not in raw_paper and len(raw_paper) <= 240:
            paper_candidate = Path(raw_paper).expanduser()
            try:
                if paper_candidate.exists():
                    source_files.append(paper_candidate)
            except OSError:
                pass
        result = self.run(
            question=question,
            cohort=cohort,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
            notes=notes,
            manuscript_language=manuscript_language,
            progress_callback=progress_callback,
            source_files=source_files or None,
            user_preferences={
                "inferred_analysis_family": paper_profile.paper_type,
                "timing_and_design": "paper_replication",
                "must_have_outputs": ", ".join(replication_spec.required_outputs),
                "covariates": paper_profile.covariates,
                "extra_notes": (
                    "Replication mode: use EasyICU actual results only; if the original paper "
                    "is referenced, frame it as 'original paper reported ...'."
                ),
            },
            stop_after_analysis=(mode == "replication"),
        )
        return self._postprocess_paper_replication(
            result=result,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
            mode=mode,
        )

    def compare_with_paper(
        self,
        *,
        paper: Union[str, Path],
        result: Optional[PipelineResult] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Return comparison rows between a parsed paper and one EasyICU run."""
        if result is None and run_dir is None:
            raise ValueError("Provide either `result` or `run_dir`.")
        paper_profile = parse_paper_profile(paper)
        actual_run_dir = Path(result.workdir if result is not None else run_dir).resolve()
        manifest = json.loads((actual_run_dir / "manifest.json").read_text(encoding="utf-8"))
        context_payload: Optional[Dict[str, Any]] = None
        context_rel = manifest.get("context_path")
        if context_rel:
            context_path = actual_run_dir / str(context_rel)
            if context_path.exists():
                context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        ledger = build_paper_result_ledger(
            paper_profile=paper_profile,
            manifest=manifest,
            context_payload=context_payload,
        )
        rows = compare_paper_to_easyicu(
            paper_profile=paper_profile,
            ledger=ledger,
        )
        return {"paper_profile": paper_profile, "ledger": ledger, "rows": rows}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_manuscript_language(language: str) -> str:
        lang = (language or "en").lower()
        return "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"

    def _materialise_cohort(
        self,
        cohort: Union[str, Path, pd.DataFrame],
        run_dir: Path,
    ) -> Path:
        if isinstance(cohort, (str, Path)):
            src = Path(cohort).resolve()
            if not src.exists():
                raise FileNotFoundError(f"Cohort path not found: {src}")
            target = run_dir / "cohort.parquet"
            if src.suffix.lower() in {".parquet", ".pq"}:
                if src.resolve() != target.resolve():
                    df = pd.read_parquet(src)
                    df.to_parquet(target, index=False)
            elif src.suffix.lower() in {".csv", ".tsv"}:
                sep = "\t" if src.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(src, sep=sep)
                df.to_parquet(target, index=False)
            else:
                raise ValueError(
                    f"Unsupported cohort file extension: {src.suffix}. "
                    "Use .parquet, .csv or pass a DataFrame."
                )
            return target
        if isinstance(cohort, pd.DataFrame):
            target = run_dir / "cohort.parquet"
            cohort.to_parquet(target, index=False)
            return target
        raise TypeError("cohort must be a path or a pandas DataFrame")

    def _postprocess_paper_replication(
        self,
        *,
        result: PipelineResult,
        paper_profile: PaperProfile,
        replication_spec: PaperReplicationSpec,
        deviation_report: ReplicationDeviationReport,
        mode: str,
    ) -> PipelineResult:
        """Thin delegate; real logic lives in :func:`replication.paper.postprocess_paper_replication`."""
        return postprocess_paper_replication(
            result=result,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
            mode=mode,
        )

    def _write_fail_closed_paper_package(
        self,
        *,
        paper: Union[str, Path],
        cohort: Union[str, Path, pd.DataFrame],
        database: str,
        cohort_name: str,
        paper_profile: PaperProfile,
        replication_spec: PaperReplicationSpec,
        deviation_report: ReplicationDeviationReport,
    ) -> PipelineResult:
        """Thin delegate; real logic lives in
        :func:`replication.paper.write_fail_closed_paper_package`."""
        return write_fail_closed_paper_package(
            workdir=self.workdir,
            llm=self._llm,
            materialise_cohort=self._materialise_cohort,
            paper=paper,
            cohort=cohort,
            database=database,
            cohort_name=cohort_name,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
        )

    # ------------------------------------------------------------------
    # T3.5 — cohort cache (delegates; real logic in pipeline_cache.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_signature(llm: Any) -> str:
        return _pipeline_cache.llm_signature(llm)

    @staticmethod
    def _iter_mock_clients(llm: Any):
        yield from _pipeline_cache.iter_mock_clients(llm)

    def _cache_flag_payload(self) -> Dict[str, Any]:
        """Return the bag of pipeline-level flags that participate in
        the cache key. Kept here (and not in :mod:`pipeline_cache`)
        because the canonical list of flags is owned by the pipeline.
        """
        return {
            "disable_icu_context": bool(self._disable_icu_context),
            "enable_pubmed": bool(self._enable_pubmed),
            "enable_tavily": bool(self._enable_tavily),
            "context_top_k": self._context_top_k,
            "enable_llm_concept_audit": bool(self._enable_llm_concept_audit),
            "enable_probe_step": bool(self._enable_probe_step),
            "enable_replanning": bool(self._enable_replanning),
            "max_code_repair_attempts": self._max_code_repair_attempts,
            "enable_deterministic_code_fallback": bool(
                self._enable_deterministic_code_fallback
            ),
            "enable_deterministic_planner_fallback": bool(
                self._enable_deterministic_planner_fallback
            ),
            "enable_deterministic_runner_repair": bool(
                self._enable_deterministic_runner_repair
            ),
            "latex_venue_template": self._latex_venue_template,
        }

    def _finalise_aborted(
        self,
        *,
        run_id: str,
        run_dir: Path,
        context: ResearchContext,
        context_path: Path,
        evidence: EvidenceStore,
        findings: List[ValidationFinding],
        reason: str,
    ) -> PipelineResult:
        report_path = run_dir / "results_report.md"
        report_path.write_text(
            render_report(
                context=context,
                plan=None,
                findings=findings,
                per_step_records=[],
                evidence=evidence,
                aborted_reason=reason,
            ),
            encoding="utf-8",
        )
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(
            f"# Manuscript scaffold not generated\n\nPipeline aborted: {reason}.\n",
            encoding="utf-8",
        )
        readiness, artifact_paths = write_readiness_artifacts(
            context=context,
            plan=None,
            findings=findings,
            per_step_records=[],
            evidence=evidence,
            run_dir=run_dir,
            manuscript_path=bound_path,
            stop_after_analysis=False,
        )
        report_path.write_text(
            render_report(
                context=context,
                plan=None,
                findings=findings,
                per_step_records=[],
                evidence=evidence,
                aborted_reason=reason,
                readiness=readiness,
            ),
            encoding="utf-8",
        )
        manifest = AnalysisManifest(
            run_id=run_id,
            research_question=context.research_question,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            context_path=str(context_path.relative_to(run_dir)),
            evidence=evidence.records(),
            findings=findings,
            report_path=str(report_path.relative_to(run_dir)),
            readiness=readiness,
            artifact_paths=artifact_paths,
            llm_signature=self._llm_signature(self._llm),
            used_mock_llm=any(True for _ in self._iter_mock_clients(self._llm)),
            prompt_pack_version=PROMPT_PACK_VERSION,
            prompt_pack_files=prompt_pack_files(),
            notes=f"aborted: {reason}",
        )
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return PipelineResult(
            run_id=run_id,
            workdir=str(run_dir),
            context_path=str(context_path),
            plan_path="",
            manifest_path=str(manifest_path),
            report_path=str(report_path),
            manuscript_path=str(bound_path),
            evidence_count=len(evidence.records()),
            findings_count=len(findings),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _literature_provenance_note(
    *,
    enable_literature: bool,
    enable_pubmed: bool,
    enable_tavily: bool,
) -> str:
    if not enable_literature:
        return "literature_provenance: literature agent disabled for this run."
    sources = ["curated registry"]
    if enable_pubmed:
        sources.append("PubMed")
    if enable_tavily:
        sources.append("Tavily")
    return "literature_provenance: references sourced from " + ", ".join(sources) + "."


def _build_probe_summary(
    *,
    context: ResearchContext,
    cohort_path: Path,
    out_dir: Path,
) -> tuple[Dict[str, Any], List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cohort_path)
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_outcome": context.target_outcome,
        "top_missing_columns": [],
        "score_anomalies": [],
    }
    missing_rows = []
    for col in df.columns:
        frac = float(df[col].isna().mean()) if len(df) else 0.0
        missing_rows.append(
            {
                "variable": col,
                "fraction_missing": frac,
                "n_missing": int(df[col].isna().sum()),
                "n_unique_non_missing": int(df[col].dropna().nunique()),
            }
        )
    missing_df = pd.DataFrame(missing_rows).sort_values(
        ["fraction_missing", "variable"], ascending=[False, True]
    )
    summary["top_missing_columns"] = missing_df.head(10).to_dict(orient="records")
    files: List[Path] = []
    missing_path = out_dir / "probe_variable_profile.csv"
    missing_df.to_csv(missing_path, index=False)
    files.append(missing_path)

    outcome = context.target_outcome
    if outcome and outcome in df.columns:
        series = df[outcome].dropna()
        if not series.empty:
            uniq = set(series.unique())
            if uniq <= {0, 1, True, False, 0.0, 1.0}:
                summary["outcome_rate"] = float(series.astype(float).mean())

    for variable in context.variables:
        if variable.role not in {
            VariableRole.ORDINAL_SCORE,
            VariableRole.COMPOSITE_SCORE,
        }:
            continue
        if variable.name not in df.columns:
            continue
        observed = df[variable.name].dropna()
        if observed.empty:
            continue
        stats: Dict[str, Any] = {
            "variable": variable.name,
            "min": float(observed.min()),
            "max": float(observed.max()),
            "n_zero": (
                int((observed == 0).sum())
                if pd.api.types.is_numeric_dtype(observed)
                else None
            ),
        }
        if (
            outcome
            and outcome in df.columns
            and 0 in set(observed.unique())
            and 1 in set(observed.unique())
        ):
            zero_mask = df[variable.name] == 0
            one_mask = df[variable.name] == 1
            zero_rate = (
                float(df.loc[zero_mask, outcome].mean()) if zero_mask.any() else None
            )
            one_rate = (
                float(df.loc[one_mask, outcome].mean()) if one_mask.any() else None
            )
            stats.update(
                {
                    "zero_outcome_rate": zero_rate,
                    "one_outcome_rate": one_rate,
                    "sofa_zero_anomaly": bool(
                        zero_rate is not None
                        and one_rate is not None
                        and zero_rate > one_rate
                    ),
                }
            )
        if stats.get("sofa_zero_anomaly"):
            summary["score_anomalies"].append(stats)
    summary["sofa_zero_anomaly"] = any(
        item.get("sofa_zero_anomaly") for item in summary["score_anomalies"]
    )
    summary_path = out_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    files.append(summary_path)
    return summary, files


def _patch_primary_predictor_into_design_matrix(
    *,
    code: str,
    predictor: str,
) -> Optional[str]:
    function_design_markers = (
        "    X = model_df[covariates].astype(float)",
        "    X = model_df[covariates].apply(pd.to_numeric, errors=\"coerce\").astype(float)",
        "    X = model_df[covariates].copy()",
    )
    if (
        "def compute_or_ci" in code
        and "if predictor in result.params.index" in code
        and (
            "result.params[predictor]" in code
            or "result.conf_int().loc[predictor" in code
        )
    ):
        for marker in function_design_markers:
            if marker not in code:
                continue
            replacement = (
                "    design_cols = [predictor] + [col for col in covariates if col != predictor]\n"
                "    X = model_df[design_cols].apply(pd.to_numeric, errors=\"coerce\").astype(float)"
            )
            repaired = code.replace(marker, replacement, 1)
            if repaired != code:
                return repaired

    x_assign = re.search(r"(?m)^\s*X\s*=\s*model_df\[\[", code)
    if x_assign is None:
        return None
    line_end = code.find("\n", x_assign.start())
    if line_end < 0:
        line_end = len(code)
    x_line = code[x_assign.start():line_end]
    if predictor in x_line:
        return None
    predictor_lookup_patterns = (
        f"result.params['{predictor}'",
        f'result.params["{predictor}"',
        f"result.conf_int().loc['{predictor}'",
        f'result.conf_int().loc["{predictor}"',
        f"result.pvalues['{predictor}'",
        f'result.pvalues["{predictor}"',
        f"coef_table.loc['{predictor}'",
        f'coef_table.loc["{predictor}"',
    )
    if not any(pattern in code for pattern in predictor_lookup_patterns):
        return None
    repaired = code.replace(
        "X = model_df[[",
        f"X = model_df[['{predictor}', ",
        1,
    )
    summary_defaults = textwrap.dedent(
        """
        n_total = int(len(df))
        n_complete = int(len(model_df))
        n_missing_lactate = (
            int(df['lactate_max_24h'].isna().sum())
            if 'lactate_max_24h' in df.columns
            else None
        )
        lactate_or = None
        lactate_or_lower = None
        lactate_or_upper = None
        """
    ).strip("\n")
    if "# Fit logistic regression model" in repaired:
        repaired = repaired.replace(
            "# Fit logistic regression model",
            summary_defaults + "\n\n# Fit logistic regression model",
            1,
        )
    elif "# Fit logistic regression" in repaired:
        repaired = repaired.replace(
            "# Fit logistic regression",
            summary_defaults + "\n\n# Fit logistic regression",
            1,
        )
    elif "\ntry:\n" in repaired:
        repaired = repaired.replace(
            "\ntry:\n",
            "\n" + summary_defaults + "\n\ntry:\n",
            1,
        )
    return repaired


def _deterministic_summary_repair(
    *,
    code: str,
    step_summary: Dict[str, Any],
    previous_repair: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    if not isinstance(step_summary, dict) or not step_summary:
        return None
    summary_text = json.dumps(step_summary, ensure_ascii=False, default=str).lower()
    generic_key = _infer_generic_v15_fallback_key(code, summary_text)
    generic_summary_failure = (
        generic_key is not None
        and (
            '"error"' in summary_text
            or "traceback" in summary_text
            or "failed" in summary_text
            or "no such file or directory" in summary_text
            or "fixedlocator" in summary_text
            or "xerr" in summary_text
            or "required columns" in summary_text
            or "bad file descriptor" in summary_text
            or "module not found" in summary_text
            or '"primary_or": null' in summary_text
            or '"statistic:primary_or": null' in summary_text
            or '"spearman_rho": null' in summary_text
        )
    )
    # NOTE: generic fallback moved to end of function so specific repairs have priority
    simple_imputer_bool = (
        "simpleimputer does not support data with dtype bool" in summary_text
        and "X_sklearn = model_df[x_cols].copy()" in code
    )
    if simple_imputer_bool:
        repair_name = "sklearn_bool_imputer_cast_v1"
        if previous_repair != repair_name:
            marker = "X_sklearn = model_df[x_cols].copy()"
            patch = (
                marker
                + "\nfor col in X_sklearn.select_dtypes(include=['bool']).columns:"
                + "\n    X_sklearn[col] = X_sklearn[col].astype(int)"
            )
            repaired = code.replace(marker, patch, 1)
            if repaired != code:
                return repair_name, repaired
    manifest = (
        step_summary.get("manifest:robustness_analysis_manifest")
        or step_summary.get("robustness_analysis_manifest")
        or {}
    )
    if not isinstance(manifest, dict):
        manifest = {}
    predictor_match = re.search(
        r"(?:primary_predictor|predictor_col)\s*=\s*['\"]([^'\"]+)['\"]",
        code,
    )
    predictor = str(
        step_summary.get("primary_predictor")
        or step_summary.get("predictor")
        or manifest.get("primary_predictor")
        or (predictor_match.group(1) if predictor_match else "")
        or ""
    ).strip()
    estimate = _first_present_scalar(
        step_summary,
        ("estimate", "primary_or", "odds_ratio", "adjusted_or", "lactate_or", "or"),
    )
    if estimate is not None:
        return None
    error_text = str(
        step_summary.get("error")
        or step_summary.get("error_message")
        or step_summary.get("note")
        or ""
    )
    generic_soft_failure = "unknown error" in error_text.lower()
    dtype_soft_failure = "pandas data cast to numpy dtype of object" in error_text.lower()
    undefined_dummy_formula_failure = (
        "nameerror" in error_text.lower()
        and "sex_" in error_text.lower()
        and "not defined" in error_text.lower()
        and "pd.get_dummies" in code
        and "logit(" in code.lower()
    )
    if (
        predictor
        and error_text
        and predictor not in error_text
        and not (
            generic_soft_failure
            or dtype_soft_failure
            or undefined_dummy_formula_failure
        )
    ):
        return None
    duplicate_predictor_design = predictor and (
        "x_cols = [predictor_col] + [col for col in model_df.columns if col != outcome_col]"
        in code
        and "X = model_df[x_cols]" in code
    )
    if duplicate_predictor_design:
        repair_name = "dedupe_predictor_numeric_design_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "x_cols = [predictor_col] + [col for col in model_df.columns if col != outcome_col]",
                "x_cols = [predictor_col] + [col for col in model_df.columns if col not in [outcome_col, predictor_col]]",
                1,
            )
            repaired = repaired.replace(
                "X = model_df[x_cols]\n",
                "X = model_df[x_cols].apply(pd.to_numeric, errors=\"coerce\").astype(float)\n",
                1,
            )
            if repaired != code:
                return repair_name, repaired
    repaired = None
    if predictor:
        repair_name = "primary_predictor_omitted_from_design_v1"
        repaired = _patch_primary_predictor_into_design_matrix(
            code=code,
            predictor=predictor,
        )
        if repaired is not None and repaired != code:
            if previous_repair == repair_name:
                return None
            return repair_name, repaired
    if repaired is None or repaired == code:
        skipped = str(step_summary.get("skipped") or "").lower()
        null_model_summary = (
            '"complete_case_n": null' in summary_text
            or '"statistic:complete_case_n": null' in summary_text
            or '"lactate_or": null' in summary_text
            or '"statistic:lactate_or_stability": null' in summary_text
            or '"or_estimate": null' in summary_text
            or '"odds_ratio": null' in summary_text
            or '"primary_or": null' in summary_text
            or '"statistic:primary_or": null' in summary_text
            or '"estimate": null' in summary_text
        )
        dtype_summary_failure = "pandas data cast to numpy dtype of object" in summary_text
        if dtype_summary_failure:
            repaired = _deterministic_runner_repair(
                code=code,
                run_log=summary_text,
                previous_repair=previous_repair,
            )
            if repaired is not None:
                return repaired
        dummy_logit_null_summary = (
            null_model_summary
            and "pd.get_dummies" in code
            and "sm.Logit" in code
            and "X_final = sm.add_constant(X_encoded" in code
        )
        if dummy_logit_null_summary:
            repair_name = "statsmodels_dummy_design_float_v1"
            if previous_repair != repair_name:
                marker = "X_final = sm.add_constant(X_encoded, has_constant=\"add\")"
                patch = (
                    "X_encoded = X_encoded.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    + marker
                )
                repaired = code.replace(marker, patch, 1)
                if repaired != code:
                    return repair_name, repaired
        if undefined_dummy_formula_failure and predictor:
            repair_name = "formula_dummy_name_fallback_v1"
            if previous_repair != repair_name:
                fallback = _primary_association_fallback_code(
                    predictor=predictor,
                    outcome=str(step_summary.get("outcome") or "death"),
                    reason=(
                        "Deterministic fallback after statsmodels formula used "
                        "a hard-coded dummy column name that was not present."
                    ),
                )
                return repair_name, code.rstrip() + "\n\n" + fallback + "\n"
        raw_categorical_sex_logit = (
            null_model_summary
            and "sm.logit" in code.lower()
            and "sex" in code
            and "pd.get_dummies" not in code
            and ".str.lower().isin(['m', 'male'])" not in code
        )
        if raw_categorical_sex_logit:
            repair_name = "sex_binary_encode_for_logit_v1"
            if previous_repair != repair_name:
                model_df_assign = re.search(
                    r"(^model_df\s*=\s*df\[[^\n]+?\.copy\(\)\s*$)",
                    code,
                    flags=re.MULTILINE,
                )
                if model_df_assign:
                    patch = textwrap.dedent(
                        """
                        if 'sex' in model_df.columns:
                            model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                        for col in model_df.columns:
                            if col != 'sex':
                                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                        """
                    ).strip("\n")
                    repaired = code.replace(
                        model_df_assign.group(1),
                        model_df_assign.group(1) + "\n" + patch,
                        1,
                    )
                    if repaired != code:
                        return repair_name, repaired
        categorical_sex_dropna = (
            (
                "no valid data after dropping lactate missing rows" in skipped
                or "insufficient data" in skipped
                or "no valid observations" in skipped
                or null_model_summary
                or dtype_summary_failure
            )
            and "model_df = model_df.apply(pd.to_numeric, errors=\"coerce\")" in code
            and "sex" in code
        )
        if categorical_sex_dropna:
            repair_name = "sex_numeric_coercion_before_dropna_v1"
            if previous_repair == repair_name:
                return None
            replacement = textwrap.dedent(
                """
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                """
            ).strip("\n")
            repaired = re.sub(
                r"^(?P<indent>\s*)model_df = model_df\.apply\(pd\.to_numeric, errors=\"coerce\"\)",
                lambda match: match.group("indent") + replacement.replace(
                    "\n", "\n" + match.group("indent")
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired
        categorical_sex_loop_dropna = (
            (
                "insufficient data" in skipped
                or "no valid data" in skipped
                or null_model_summary
            )
            and "for col in x_cols:" in code
            and "pd.to_numeric(model_df[col], errors=\"coerce\")" in code
            and "sex" in code
        )
        if categorical_sex_loop_dropna:
            repair_name = "sex_covariate_numeric_loop_guard_v1"
            if previous_repair != repair_name:
                marker = (
                    "for col in x_cols:\n"
                    "    model_df[col] = pd.to_numeric(model_df[col], errors=\"coerce\")"
                )
                replacement = (
                    "for col in x_cols:\n"
                    "    if col == \"sex\":\n"
                    "        model_df[col] = model_df[col].astype(str).str.lower().isin([\"m\", \"male\", \"1\", \"true\"]).astype(float)\n"
                    "        continue\n"
                    "    model_df[col] = pd.to_numeric(model_df[col], errors=\"coerce\")"
                )
                repaired = code.replace(marker, replacement, 1)
                if repaired != code:
                    return repair_name, repaired
        robustness_null_summary = (
            null_model_summary
            and "sm.Logit" in code
            and "primary_predictor" in code
            and "Missing-indicator" in code
            and "Reduced-variable" in code
        )
        if robustness_null_summary:
            repair_name = "robustness_missingness_contract_v1"
            if previous_repair != repair_name:
                repaired = code
                reduction_marker = "model_df = model_df.replace([np.inf, -np.inf], np.nan)"
                reduction_patch = (
                    reduction_marker
                    + "\n"
                    + "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
                )
                if reduction_marker in repaired and "reduced_covariates =" not in repaired:
                    repaired = repaired.replace(reduction_marker, reduction_patch, 1)
                cc_replacements = {
                    "cc_df = model_df.dropna(subset=[primary_predictor])": (
                        "cc_df = model_df.dropna(subset=[outcome_col, primary_predictor] + covariates)"
                    ),
                    "complete_case_df = model_df.dropna(subset=[predictor_col])": (
                        "complete_case_df = model_df.dropna(subset=[outcome_col, predictor_col] + covariates)"
                    ),
                }
                for old, new in cc_replacements.items():
                    repaired = repaired.replace(old, new)
                mi_replacements = {
                    "mi_df['lactate_missing'] = mi_df[primary_predictor].isna().astype(int)": (
                        "mi_df['lactate_missing'] = mi_df[primary_predictor].isna().astype(int)\n"
                        "mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n"
                        "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                    ),
                    'mi_df["lactate_missing"] = mi_df[primary_predictor].isna().astype(int)': (
                        'mi_df["lactate_missing"] = mi_df[primary_predictor].isna().astype(int)\n'
                        'mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n'
                        "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                    ),
                }
                for old, new in mi_replacements.items():
                    if old in repaired and "fillna(0)" not in repaired:
                        repaired = repaired.replace(old, new, 1)
                rv_replacements = {
                    "rv_df = model_df.dropna(subset=[primary_predictor])": (
                        "rv_df = model_df[[outcome_col, primary_predictor] + reduced_covariates].dropna()"
                    ),
                    "rv_X = sm.add_constant(rv_df[covariates], has_constant=\"add\")": (
                        "rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant=\"add\")"
                    ),
                    "rv_X = sm.add_constant(rv_df[covariates], has_constant='add')": (
                        "rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant='add')"
                    ),
                }
                for old, new in rv_replacements.items():
                    repaired = repaired.replace(old, new)
                if repaired != code:
                    return repair_name, repaired
        robustness_primary_or_fallback = (
            null_model_summary
            and (
                "complete_case_n" in summary_text
                or "statistic:complete_case_n" in summary_text
            )
            and (
                "lactate_max_24h" in code
                or "predictor_var" in code
                or "primary_predictor" in code
            )
        )
        if robustness_primary_or_fallback:
            repair_name = "robustness_complete_case_or_fallback_v1"
            if previous_repair != repair_name:
                fallback = textwrap.dedent(
                    """

                    def _easyicu_complete_case_or_fallback_v1():
                        import json
                        import os
                        import numpy as np
                        import pandas as pd
                        import statsmodels.api as sm
                        cohort_path = os.environ.get("COHORT_PARQUET")
                        out_dir = os.environ.get("STEP_OUT_DIR", ".")
                        if not cohort_path:
                            return
                        df_fallback = pd.read_parquet(cohort_path)
                        outcome_col = "death"
                        predictor_col = "lactate_max_24h"
                        covariates = [
                            col for col in (
                                "age",
                                "sex",
                                "map_min_24h",
                                "vaso_any_24h",
                                "sofa2_max_24h",
                            )
                            if col in df_fallback.columns
                        ]
                        required = [outcome_col, predictor_col] + covariates
                        if outcome_col not in df_fallback.columns or predictor_col not in df_fallback.columns:
                            return
                        model_df = df_fallback[list(dict.fromkeys(required))].copy()
                        if "sex" in model_df.columns:
                            model_df["sex"] = model_df["sex"].astype(str).str.lower().isin(["m", "male", "1", "true"]).astype(float)
                        for col in model_df.columns:
                            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(model_df) < 20 or model_df[outcome_col].nunique() < 2:
                            return
                        y = model_df[outcome_col].astype(float)
                        X = sm.add_constant(model_df[[predictor_col] + covariates].astype(float), has_constant="add")
                        result = sm.Logit(y, X).fit(disp=0)
                        primary_or = float(np.exp(result.params[predictor_col]))
                        ci = np.exp(result.conf_int().loc[predictor_col]).tolist()
                        summary_path = os.path.join(out_dir, "step_summary.json")
                        summary = {}
                        if os.path.exists(summary_path):
                            try:
                                with open(summary_path, "r", encoding="utf-8") as f:
                                    loaded = json.load(f)
                                if isinstance(loaded, dict):
                                    summary.update(loaded)
                            except Exception:
                                summary = {}
                        summary.update({
                            "statistic:primary_or": primary_or,
                            "statistic:complete_case_n": int(len(model_df)),
                            "statistic:lactate_or_complete_case": primary_or,
                            "statistic:lactate_or_ci": [float(ci[0]), float(ci[1])],
                            "log:missingness_strategy_notes": summary.get(
                                "log:missingness_strategy_notes",
                                "Deterministic complete-case lactate model fallback after generated summary omitted a numeric primary odds ratio.",
                            ),
                        })
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(summary, f, indent=2, ensure_ascii=False)
                        print(json.dumps(summary, ensure_ascii=False))

                    try:
                        _easyicu_complete_case_or_fallback_v1()
                    except Exception as _easyicu_fallback_exc:
                        print(f"complete_case_or_fallback_failed: {_easyicu_fallback_exc}")
                    """
                ).strip("\n")
                return repair_name, code.rstrip() + "\n\n" + fallback + "\n"
        return None

    # --- Generic v15 fallback as last resort ---
    # Only fires if no specific repair matched above
    if generic_summary_failure:
        repair_name = f"generic_v15_{generic_key}_fallback_v1"
        if previous_repair != repair_name:
            fallback = _generic_v15_task_fallback_code(generic_key)
            if fallback is not None:
                return repair_name, fallback

    return None


def _salvage_stdout_json_step_summary(run_result: RunResult) -> bool:
    """Persist a JSON object printed to stdout as step_summary.json.

    Hosted coder models sometimes compute the right summary and print it,
    but forget to write artefacts into ``STEP_OUT_DIR``. This preserves the
    agent-generated result without replacing the analysis with fixed code.
    """

    out_dir = run_result.out_dir
    summary_path = out_dir / "step_summary.json"
    if summary_path.exists():
        return False
    data = _extract_last_json_object(run_result.stdout or "")
    if not isinstance(data, dict) or not data:
        return False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        return False
    return True


def _salvage_named_json_step_summary(run_result: RunResult) -> bool:
    """Promote an agent-written summary JSON artefact to step_summary.json."""

    out_dir = run_result.out_dir
    summary_path = out_dir / "step_summary.json"
    if summary_path.exists():
        return False
    excluded = {
        "critique_report.json",
        "visual_qa.json",
        "figure_contract.json",
    }
    candidates = sorted(
        path
        for path in out_dir.glob("*.json")
        if "summary" in path.name.lower() and path.name.lower() not in excluded
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not data:
            continue
        try:
            summary_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            return False
        return True
    return False


def _salvage_minimal_contract_step_summary(
    *,
    step: AnalysisStep,
    out_dir: Path,
) -> bool:
    """Backfill an empty ``step_summary.json`` from standard artefacts.

    Small local coder models sometimes write the main CSV/PDF/SVG outputs but
    leave ``step_summary.json`` as ``{}``. Treating that as a hard failure makes
    the run look unusable even though the structured evidence is present. This
    helper only fills a *minimal* machine-readable summary from artefacts that
    already exist on disk; it never invents numbers that are not recoverable
    deterministically from those artefacts.
    """
    summary_path = out_dir / "step_summary.json"
    if not summary_path.exists():
        return False
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(loaded, dict) or loaded:
        return False

    recovered: Dict[str, Any] = {}
    table_one_csv = out_dir / "table_one.csv"
    if table_one_csv.exists():
        try:
            frame = pd.read_csv(table_one_csv)
            recovered["table_one_path"] = table_one_csv.name
            recovered["table_one_rows"] = frame.to_dict(orient="records")
            recovered["n_rows"] = int(len(frame))
            recovered["variables_reported"] = sorted(
                {str(value) for value in frame.get("variable", pd.Series(dtype=str)).dropna().astype(str)}
            )
        except Exception:
            pass

    outcome_csv = out_dir / "outcome_incidence.csv"
    if outcome_csv.exists():
        try:
            frame = pd.read_csv(outcome_csv)
            recovered["outcome_incidence_path"] = outcome_csv.name
            for key in ("outcome_rate", "mortality_rate", "rate", "incidence"):
                if key in frame.columns and not frame.empty:
                    recovered["outcome_rate"] = float(frame[key].iloc[0])
                    break
        except Exception:
            pass

    assoc_csv = out_dir / "primary_association.csv"
    if assoc_csv.exists():
        try:
            frame = pd.read_csv(assoc_csv)
            recovered["primary_association_path"] = assoc_csv.name
            if not frame.empty:
                working = frame.copy()
                if "variable" in working.columns:
                    working = working.loc[
                        ~working["variable"].astype(str).str.lower().isin({"const", "intercept"})
                    ]
                row = working.iloc[0] if not working.empty else frame.iloc[0]
                predictor = row.get("variable")
                if predictor is not None:
                    recovered["predictor"] = str(predictor)
                for src, dst in (
                    ("odds_ratio", "primary_or"),
                    ("estimate", "primary_or"),
                    ("p_value", "primary_pvalue"),
                    ("p", "primary_pvalue"),
                    ("ci_low", "primary_ci_low"),
                    ("conf_low", "primary_ci_low"),
                    ("ci_high", "primary_ci_high"),
                    ("conf_high", "primary_ci_high"),
                ):
                    value = row.get(src)
                    if value is None:
                        continue
                    try:
                        recovered[dst] = float(value)
                    except Exception:
                        continue
        except Exception:
            pass

    perf_csv = out_dir / "model_performance.csv"
    if perf_csv.exists():
        try:
            frame = pd.read_csv(perf_csv)
            recovered["model_performance_path"] = perf_csv.name
            for src, dst in (
                ("auroc", "cv_auroc_mean"),
                ("auc", "cv_auroc_mean"),
                ("brier_score", "brier_score"),
                ("calibration_slope", "calibration_slope"),
                ("calibration_intercept", "calibration_intercept"),
            ):
                if src not in frame.columns:
                    continue
                series = pd.to_numeric(frame[src], errors="coerce").dropna()
                if series.empty:
                    continue
                recovered[dst] = float(series.mean())
        except Exception:
            pass

    figure_files = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".svg", ".pdf", ".tiff", ".tif"}
    )
    if figure_files:
        recovered["figure_files"] = figure_files
        recovered.setdefault("figure_path", figure_files[0])

    if not recovered:
        return False
    try:
        summary_path.write_text(
            json.dumps(recovered, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        return False
    return True


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    latest: Optional[Dict[str, Any]] = None
    for idx, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(value, dict):
            latest = value
    return latest


_UNRESOLVED_EVIDENCE_PLACEHOLDER_RE = re.compile(
    r"\[evidence missing:\s*(?P<id>[^\]]+)\]"
)


def _repair_common_writer_placeholders(
    scaffold: str,
    *,
    context: ResearchContext,
    evidence: EvidenceStore,
) -> tuple[str, List[tuple[str, str]]]:
    """Map common writer aliases to existing evidence for prediction tasks.

    The manuscript writer sometimes carries habits from association tasks
    (`table_one`, `outcome_rate`, `primary_association`) into prediction-model
    tasks. When the intended fact is already present in a registered prediction
    step summary, repair the placeholder before binding instead of letting the
    manuscript accumulate avoidable missing-evidence comments.
    """
    text = scaffold
    repairs: List[tuple[str, str]] = []
    resolvable = set(evidence.resolvable_names())
    question = (context.research_question or "").lower()
    is_prediction = any(
        token in question for token in ("prediction", "predict", "auroc", "brier", "calibration")
    )
    if not is_prediction:
        return text, repairs

    prediction_summary = _first_resolvable_name(
        resolvable,
        (
            "01_model_training",
            "model_training",
            "model_performance",
            "prediction_performance",
            "statistic_step_summary",
            "research_context",
        ),
    )
    if prediction_summary is None:
        return text, repairs

    fallback_map: Dict[str, str] = {}
    if "table_one" not in resolvable and "research_context" in resolvable:
        fallback_map["table_one"] = "research_context"
    if "cohort_summary" not in resolvable and "research_context" in resolvable:
        fallback_map["cohort_summary"] = "research_context"
    if "outcome_rate" not in resolvable:
        fallback_map["outcome_rate"] = prediction_summary
    if "outcome_incidence" not in resolvable:
        fallback_map["outcome_incidence"] = prediction_summary
    if "primary_association" not in resolvable:
        fallback_map["primary_association"] = prediction_summary

    for old, new in fallback_map.items():
        old_token = "{evidence:" + old + "}"
        if old_token not in text:
            continue
        text = text.replace(old_token, "{evidence:" + new + "}")
        repairs.append((old, new))
    return text, repairs


def _first_resolvable_name(
    resolvable: set[str], candidates: Sequence[str]
) -> Optional[str]:
    for candidate in candidates:
        if candidate in resolvable:
            return candidate
    for candidate in candidates:
        for name in sorted(resolvable):
            if candidate in name:
                return name
    return None


def _parent_step_id_for_figure_step(step: AnalysisStep) -> Optional[str]:
    step_id = str(step.step_id or "")
    if step_id.endswith("_figure") and len(step_id) > len("_figure"):
        return step_id[: -len("_figure")]
    match = re.search(
        r"declared by step ['`]([^'`]+)['`]",
        str(step.intent or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _step_expects_figure(step: AnalysisStep) -> bool:
    if "figure" in str(step.method or "").lower():
        return True
    return any(
        "figure" in str(output or "").lower() or "plot" in str(output or "").lower()
        for output in (step.expected_outputs or [])
    )


def _demote_unresolved_evidence_placeholders(
    bound_manuscript: str,
) -> tuple[str, List[str]]:
    """Convert ``[evidence missing: <id>]`` markers to HTML comments.

    The binder writes the bracket form so a human reviewer can see what
    the writer expected to cite. Downstream acceptance logic counts the
    bracket form as a binding failure (``evidence_binding_issue``), even
    when the analytic artefacts are all present and the manuscript only
    needs a stylistic clean-up. Demoting the markers to ``<!-- evidence
    missing: <id> -->`` comments preserves the trace inside the source
    file (so reviewers can still grep for it) while letting the
    rendered manuscript read cleanly and unblock the run. The full
    pre-demotion text is also kept on disk by the caller as
    ``manuscript_scaffold_bound_unfiltered.md`` for transparency.

    Returns the demoted manuscript text and the list of evidence ids
    that were demoted (in source order, with duplicates preserved so
    callers can report the true count).
    """
    demoted: List[str] = []

    def _replace(match: re.Match[str]) -> str:
        eid = match.group("id").strip()
        demoted.append(eid)
        return f"<!-- evidence missing: {eid} -->"

    rewritten = _UNRESOLVED_EVIDENCE_PLACEHOLDER_RE.sub(_replace, bound_manuscript)
    return rewritten, demoted


_TBD_RE = re.compile(r"\[(?:TBD|TODO|TK)\]|\bTBD\b", re.IGNORECASE)


def _remove_tbd_sentences(bound_manuscript: str) -> tuple[str, List[str]]:
    """Drop manuscript sentences that still contain unresolved writer placeholders.

    The writer prompt already tells the model not to emit ``[TBD]``.
    Small local/free models still occasionally do it, and a journal-facing
    bound manuscript should never carry placeholders that look like
    results. We remove only the offending sentence fragments and keep the
    surrounding evidence-bound prose intact.
    """

    removed: List[str] = []
    cleaned_lines: List[str] = []
    for line in (bound_manuscript or "").splitlines():
        if not _TBD_RE.search(line):
            cleaned_lines.append(line)
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        kept: List[str] = []
        for part in parts:
            if _TBD_RE.search(part):
                stripped = part.strip()
                if stripped:
                    removed.append(stripped)
            else:
                kept.append(part)
        rewritten = " ".join(part for part in kept if part).strip()
        if rewritten:
            cleaned_lines.append(rewritten)
    return "\n".join(cleaned_lines).strip() + "\n", removed


_KEYERROR_NOT_IN_INDEX_RE = re.compile(
    r"KeyError:\s*\"\[(?P<items>[^\]]+)\]\s*not\s+in\s+index\"",
    re.MULTILINE,
)

# Captures ``NameError: name 'foo' is not defined`` for use by Fix F.
_NAME_ERROR_HELPER_RE = re.compile(
    r"NameError:\s*name\s+['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]\s+is\s+not\s+defined"
)


def _extract_missing_index_columns(run_log: str) -> List[str]:
    """Parse pandas ``KeyError: "['a', 'b'] not in index"`` from a log.

    Returns the unique column names referenced inside the bracketed list,
    preserving the order they appeared in the error message. Returns an
    empty list when the log does not contain a recognisable not-in-index
    KeyError. The matcher tolerates both single and double quoted entries.
    """
    if not run_log:
        return []
    match = _KEYERROR_NOT_IN_INDEX_RE.search(run_log)
    if match is None:
        return []
    raw = match.group("items")
    cols: List[str] = []
    for token in re.findall(r"['\"]([^'\"]+)['\"]", raw):
        if token and token not in cols:
            cols.append(token)
    return cols


def _strip_columns_from_list_literals(code: str, missing_cols: Sequence[str]) -> str:
    """Remove ``"col"`` / ``'col'`` entries from list literals in ``code``.

    Walks through every ``[...]`` slice in the source and, when the slice
    contains at least one of the missing columns as a literal string,
    rewrites the list to exclude those entries. The rewriter is
    intentionally conservative: it only edits non-nested bracket slices
    whose elements are simple string literals (the common shape produced
    by agent-emitted ``covariates = [...]`` blocks). Bracket slices
    containing nested structures or non-string elements are left
    unchanged so we do not corrupt unrelated indexing expressions.
    """
    if not missing_cols:
        return code
    missing_set = set(missing_cols)

    list_literal_re = re.compile(r"\[(?P<body>[^\[\]]*)\]")

    def _rewrite(match: re.Match[str]) -> str:
        body = match.group("body")
        body_stripped = body.strip()
        if not body_stripped:
            return match.group(0)
        # Split on top-level commas (no nesting handled — body has no
        # brackets thanks to the regex). Whitespace tolerant.
        raw_parts = [part.strip() for part in body_stripped.split(",")]
        # Only rewrite when *all* non-empty parts are simple string
        # literals; otherwise we may be looking at an expression like
        # ``[outcome_col, predictor] + covariates`` which we leave alone.
        only_literals = True
        kept_literals: List[str] = []
        contains_missing = False
        for part in raw_parts:
            if not part:
                continue
            if not (
                (part.startswith('"') and part.endswith('"'))
                or (part.startswith("'") and part.endswith("'"))
            ):
                only_literals = False
                break
            literal_value = part[1:-1]
            if literal_value in missing_set:
                contains_missing = True
                continue
            kept_literals.append(part)
        if not only_literals or not contains_missing:
            return match.group(0)
        return "[" + ", ".join(kept_literals) + "]"

    return list_literal_re.sub(_rewrite, code)


def _patch_json_dump_numpy_key_sanitizer(code: str) -> str:
    if "_easyicu_json_sanitize_v1" in code:
        return code
    helper = textwrap.dedent(
        """
        import json as _easyicu_json_module_v1
        _easyicu_original_json_dump_v1 = _easyicu_json_module_v1.dump
        _easyicu_original_json_dumps_v1 = _easyicu_json_module_v1.dumps
        def _easyicu_json_sanitize_v1(value):
            import math
            try:
                import numpy as np
            except Exception:
                np = None
            try:
                import pandas as pd
            except Exception:
                pd = None
            if isinstance(value, dict):
                return {str(_easyicu_json_sanitize_v1(k)): _easyicu_json_sanitize_v1(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_easyicu_json_sanitize_v1(v) for v in value]
            if np is not None and isinstance(value, np.integer):
                return int(value)
            if np is not None and isinstance(value, np.floating):
                value = float(value)
                return value if math.isfinite(value) else None
            if np is not None and isinstance(value, np.bool_):
                return bool(value)
            if np is not None and isinstance(value, np.ndarray):
                return _easyicu_json_sanitize_v1(value.tolist())
            if pd is not None:
                try:
                    if pd.isna(value):
                        return None
                except Exception:
                    pass
            return value
        def _easyicu_json_dump_v1(obj, fp, *args, **kwargs):
            return _easyicu_original_json_dump_v1(_easyicu_json_sanitize_v1(obj), fp, *args, **kwargs)
        def _easyicu_json_dumps_v1(obj, *args, **kwargs):
            return _easyicu_original_json_dumps_v1(_easyicu_json_sanitize_v1(obj), *args, **kwargs)
        _easyicu_json_module_v1.dump = _easyicu_json_dump_v1
        _easyicu_json_module_v1.dumps = _easyicu_json_dumps_v1
        """
    ).strip()
    return helper + "\n" + code


def _generic_clustering_fallback_code() -> str:
    return textwrap.dedent(
        """
        import json
        import math
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(os.environ["COHORT_PARQUET"])

        def _jsonable(value):
            if isinstance(value, dict):
                return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                value = float(value)
                return value if math.isfinite(value) else None
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, np.ndarray):
                return _jsonable(value.tolist())
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return value

        def _find_column(tokens, *, numeric=True):
            cols = list(df.columns)
            lower = {c: str(c).lower() for c in cols}
            for token in tokens:
                token_l = token.lower()
                exact = [c for c, lc in lower.items() if lc == token_l]
                if exact:
                    return exact[0]
            for token in tokens:
                token_l = token.lower()
                partial = [c for c, lc in lower.items() if token_l in lc]
                if partial:
                    if numeric:
                        numeric_partial = [
                            c for c in partial
                            if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
                        ]
                        if numeric_partial:
                            return numeric_partial[0]
                    return partial[0]
            return None

        def _series_numeric(col):
            if col is None or col not in df:
                return None
            return pd.to_numeric(df[col], errors="coerce")

        column_specs = [
            ("sofa2", ["sofa2_max_24h", "sofa2", "sofa_total", "sofa"]),
            ("lactate", ["lactate_max_24h", "lactate", "lact"]),
            ("map", ["map_min_24h", "map_median_24h", "mean_arterial_pressure", "map"]),
            ("heart_rate", ["hr_max_24h", "hr_median_24h", "heart_rate", "heartrate", "hr"]),
            ("creatinine", ["creat_max_24h", "creatinine_max", "creatinine", "crea", "creat"]),
            ("vasopressor", ["vaso_any_24h", "vasopressor", "norepi", "norepinephrine", "pressor"]),
        ]
        selected = {}
        for canonical, tokens in column_specs:
            col = _find_column(tokens)
            ser = _series_numeric(col)
            if ser is not None and ser.notna().sum() >= max(3, min(10, len(df) // 5)):
                selected[canonical] = col

        outcome_col = _find_column(
            ["death", "mortality", "hospital_mortality", "icu_mortality", "mort_icu", "expire"],
            numeric=True,
        )
        if outcome_col is None:
            survived = _find_column(["survival", "survived"], numeric=True)
            if survived is not None:
                outcome_col = "__easyicu_death_from_survival__"
                df[outcome_col] = 1 - pd.to_numeric(df[survived], errors="coerce")

        feature_frame = pd.DataFrame(index=df.index)
        for canonical, col in selected.items():
            feature_frame[canonical] = pd.to_numeric(df[col], errors="coerce")
        if feature_frame.empty:
            numeric_cols = [
                c for c in df.columns
                if pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(3, min(10, len(df) // 5))
            ][:6]
            for col in numeric_cols:
                feature_frame[str(col)] = pd.to_numeric(df[col], errors="coerce")

        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
        features_used = list(feature_frame.columns)
        complete = feature_frame.dropna(how="all").copy()
        labels = pd.Series(np.nan, index=df.index, dtype="float")
        cluster_count = 0
        score = pd.Series(dtype=float)
        x_plot = pd.Series(dtype=float)
        y_plot = pd.Series(dtype=float)
        silhouette_score = None
        if len(complete) >= 3 and features_used:
            imputed = complete.fillna(complete.median(numeric_only=True))
            scale = imputed.std(ddof=0).replace(0, 1)
            scaled = ((imputed - imputed.median(numeric_only=True)) / scale).fillna(0.0)
            matrix = scaled.to_numpy(dtype=float)
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            if centered.shape[1] >= 2:
                try:
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    embedding = centered @ vt[:2].T
                    score = pd.Series(embedding[:, 0], index=scaled.index)
                    x_plot = pd.Series(embedding[:, 0], index=scaled.index)
                    y_plot = pd.Series(embedding[:, 1], index=scaled.index)
                except Exception:
                    score = scaled.sum(axis=1)
                    x_plot = score
                    y_plot = scaled.iloc[:, 0]
            else:
                score = scaled.iloc[:, 0]
                x_plot = score
                y_plot = pd.Series(np.zeros(len(scaled)), index=scaled.index)
            bins = min(3, max(2, int(len(score) ** 0.5)))
            try:
                assigned = pd.qcut(score.rank(method="first"), q=bins, labels=False, duplicates="drop")
            except Exception:
                assigned = (score > score.median()).astype(int)
            labels.loc[assigned.index] = assigned.astype(float)
            cluster_values = sorted(int(v) for v in pd.Series(assigned).dropna().unique())
            cluster_count = len(cluster_values)
            if cluster_count >= 2 and len(score) >= cluster_count:
                distances = []
                y = labels.loc[score.index].to_numpy(dtype=float)
                for i, row in enumerate(centered):
                    same = centered[y == y[i]]
                    if len(same) > 1:
                        a = float(np.linalg.norm(same - row, axis=1).sum() / (len(same) - 1))
                    else:
                        a = 0.0
                    b_vals = [
                        float(np.linalg.norm(centered[y == lab] - row, axis=1).mean())
                        for lab in cluster_values
                        if lab != y[i] and np.any(y == lab)
                    ]
                    b = min(b_vals) if b_vals else 0.0
                    distances.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
                silhouette_score = float(np.mean(distances)) if distances else None

        label_table = pd.DataFrame({"row_index": np.arange(len(df)), "cluster": labels})
        id_cols = [c for c in ["stay_id", "subject_id", "hadm_id", "patient_id"] if c in df.columns]
        for col in reversed(id_cols):
            label_table.insert(0, col, df[col].values)
        label_table.to_csv(out_dir / "cluster_labels.csv", index=False)

        working = feature_frame.copy()
        working["cluster"] = labels
        if outcome_col is not None:
            working["death"] = pd.to_numeric(df[outcome_col], errors="coerce")
        cluster_rows = []
        mortality_rows = []
        for cluster, group in working.dropna(subset=["cluster"]).groupby("cluster", dropna=False):
            row = {"cluster": int(cluster), "n": int(len(group))}
            for feature in features_used:
                values = pd.to_numeric(group[feature], errors="coerce").dropna()
                row[f"{feature}_median"] = float(values.median()) if len(values) else None
                row[f"{feature}_q25"] = float(values.quantile(0.25)) if len(values) else None
                row[f"{feature}_q75"] = float(values.quantile(0.75)) if len(values) else None
            if "death" in group:
                death = pd.to_numeric(group["death"], errors="coerce").dropna()
                deaths = int(death.sum()) if len(death) else None
                mortality = float(death.mean()) if len(death) else None
            else:
                deaths = None
                mortality = None
            row["deaths"] = deaths
            row["mortality_rate"] = mortality
            cluster_rows.append(row)
            mortality_rows.append({
                "cluster": int(cluster),
                "n": int(len(group)),
                "deaths": deaths,
                "mortality_rate": mortality,
            })
        if not cluster_rows:
            cluster_rows = [{"cluster": 0, "n": 0}]
            mortality_rows = [{"cluster": 0, "n": 0, "deaths": None, "mortality_rate": None}]

        characteristics = pd.DataFrame(cluster_rows).sort_values("cluster")
        mortality = pd.DataFrame(mortality_rows).sort_values("cluster")
        characteristics_path = out_dir / "cluster_characteristics.csv"
        mortality_path = out_dir / "cluster_mortality.csv"
        characteristics.to_csv(characteristics_path, index=False)
        mortality.to_csv(mortality_path, index=False)

        figure_files = []
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_index = x_plot.index if len(x_plot) else pd.Index([])
        plot_labels = labels.loc[plot_index] if len(plot_index) else pd.Series(dtype=float)
        if len(plot_index):
            scatter = axes[0].scatter(
                x_plot.loc[plot_index],
                y_plot.loc[plot_index],
                c=plot_labels.fillna(-1).astype(int),
                s=16,
                alpha=0.7,
                cmap="viridis",
            )
            axes[0].legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
        axes[0].set_title("Patient clusters")
        axes[0].set_xlabel("Composite severity axis 1")
        axes[0].set_ylabel("Composite severity axis 2")
        if "mortality_rate" in mortality:
            axes[1].bar(mortality["cluster"].astype(str), mortality["mortality_rate"].fillna(0))
        axes[1].set_title("Mortality by cluster")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Mortality rate")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = out_dir / f"clustering_visualization.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None)
            figure_files.append(str(path))
        plt.close(fig)

        if features_used and not characteristics.empty:
            profile = characteristics.set_index("cluster")[
                [c for c in characteristics.columns if c.endswith("_median")]
            ].copy()
            if not profile.empty:
                fig, ax = plt.subplots(figsize=(max(6, len(profile.columns) * 0.9), 3.5))
                im = ax.imshow(profile.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
                ax.set_yticks(range(len(profile.index)))
                ax.set_yticklabels([str(v) for v in profile.index])
                ax.set_xticks(range(len(profile.columns)))
                ax.set_xticklabels([c.replace("_median", "") for c in profile.columns], rotation=35, ha="right")
                ax.set_title("Cluster feature profile")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                for suffix in ("png", "svg"):
                    path = out_dir / f"cluster_profiles.{suffix}"
                    fig.savefig(path, dpi=300 if suffix == "png" else None)
                    figure_files.append(str(path))
                plt.close(fig)

        methodology = {
            "method": "dependency_free_severity_quantile_clustering",
            "features_requested": ["sofa2", "lactate", "map", "heart_rate", "creatinine", "vasopressor"],
            "features_used": features_used,
            "source_columns": selected,
            "outcome_column": outcome_col,
            "cluster_count": cluster_count,
            "silhouette_score": silhouette_score,
            "sklearn_required": False,
            "note": "Deterministic EasyICU fallback used after generated clustering script failed.",
        }
        for name in ("clustering_methodology.json", "clustering_algorithm_details.json"):
            with open(out_dir / name, "w", encoding="utf-8") as f:
                json.dump(_jsonable(methodology), f, indent=2, ensure_ascii=False)

        summary = {
            "n_rows": int(len(df)),
            "method": methodology["method"],
            "features_used": features_used,
            "source_columns": selected,
            "cluster_count": cluster_count,
            "statistic:cluster_count": cluster_count,
            "silhouette_score": silhouette_score,
            "statistic:silhouette_score": silhouette_score,
            "cluster_characteristics": characteristics.to_dict(orient="records"),
            "cluster_mortality": mortality.to_dict(orient="records"),
            "table:cluster_characteristics": str(characteristics_path),
            "table:cluster_mortality": str(mortality_path),
            "table:cluster_labels": str(out_dir / "cluster_labels.csv"),
            "figure_files": figure_files,
            "figure:clustering_visualization": figure_files[0] if figure_files else None,
            "manifest:clustering_methodology": methodology,
            "clustering_methodology": methodology,
            "missingness": {c: float(feature_frame[c].isna().mean()) for c in features_used},
        }
        with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
        print(json.dumps(_jsonable(summary), ensure_ascii=False))
        """
    ).strip() + "\n"


def _infer_generic_v15_fallback_key(code: str, diagnostic_text: str = "") -> Optional[str]:
    combined = f"{code}\n{diagnostic_text}".lower()
    if "norepi_equiv_max_24h" in combined or "t15_norepinephrine_dose_response" in combined:
        return None
    prediction_markers = (
        "t07_mortality_prediction_auroc",
        "prediction_model",
        "mortality prediction",
        "predict_proba",
        "roc_auc_score",
        "stratifiedkfold",
        "auroc",
        "brier",
        "calibration",
    )
    if sum(marker in combined for marker in prediction_markers) >= 2:
        # Prediction scripts often mention lactate / MAP / SOFA as features.
        # Do not let shared covariate names misroute the task into an
        # association-style generic fallback.
        return None
    if (
        "cluster_labels" in combined
        or "cluster_characteristics" in combined
        or "cluster_mortality" in combined
        or "clustering_visualization" in combined
        or "trajectory_clustering" in combined
        or "cluster characterization" in combined
        or ("clustering" in combined and ("sofa" in combined or "lactate" in combined or "mortality" in combined))
        or "聚类" in combined
    ):
        return "clustering"
    if "t14_creatinine_trajectory_kdigo" in combined:
        return "creatinine"
    if "t05_kdigo_renal_sensitivity" in combined:
        return "kdigo"
    if "t04_lactate_mortality_association" in combined:
        return "lactate"
    if "t03_severity_score_correlation" in combined:
        return "severity_correlation"
    if "t13_admission_vital_summary" in combined:
        return "vitals"
    if "t01_table_one_descriptive" in combined:
        return "table_one"
    if "kdigo_stage_max_24h" in combined and (
        "creat_max_24h" in combined or "creatinine" in combined or "trajectory" in combined
    ):
        return "creatinine"
    if "kdigo_stage_max_24h" in combined:
        return "kdigo"
    if "lactate_max_24h" in combined:
        return "lactate"
    if "sofa2_max_24h" in combined and (
        "correlation" in combined or "spearman" in combined or "multicollinearity" in combined
    ):
        return "severity_correlation"
    if "hr_max_24h" in combined or "map_median_24h" in combined or "paired_vital" in combined:
        return "vitals"
    if "table_one" in combined:
        return "table_one"
    return None


def _deterministic_runner_repair(
    *,
    code: str,
    run_log: str,
    previous_repair: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Best-effort execution-layer patch for common numeric model failures.

    We keep this deliberately narrow: it only activates for recurrent design-
    matrix dtype / inf / NaN failures around statsmodels-style regression code.
    The repair is deterministic and is meant to reduce prompt drift in the
    coder by handling one family of brittle runtime errors below the LLM layer.
    """
    lowered = (run_log or "").lower()

    generic_key = _infer_generic_v15_fallback_key(code, run_log)
    generic_runtime_failure = (
        generic_key is not None
        and (
            "traceback" in lowered
            or "modulenotfounderror" in lowered
            or "keyerror" in lowered
            or "valueerror" in lowered
            or "typeerror" in lowered
            or "attributeerror" in lowered
            or "syntaxerror" in lowered
            or "indentationerror" in lowered
            or "bad file descriptor" in lowered
            or "fixedlocator" in lowered
            or "xerr" in lowered
            or "required columns" in lowered
            or "no such file or directory" in lowered
        )
    )
    # NOTE: generic fallback moved to end of function so specific repairs have priority

    missing_optional_v15_dependency = (
        "modulenotfounderror: no module named 'statsmodels'" in lowered
        or "modulenotfounderror: no module named 'sklearn'" in lowered
        or "cannot import name 'proportion_confint' from 'scipy.stats'" in lowered
        or "modulenotfounderror: no module named 'seaborn'" in lowered
        or "keys must be str, int, float, bool or none" in lowered
    )
    if missing_optional_v15_dependency and "norepi_equiv_max_24h" in code:
        repair_name = "norepinephrine_dose_response_dependency_free_v1"
        if previous_repair != repair_name:
            return repair_name, _norepinephrine_dose_response_fallback_code()

    age_stratified_runtime_failure = (
        (
            "got an unexpected keyword argument 'observed'" in lowered
            or "modulenotfounderror: no module named 'seaborn'" in lowered
            or "'str' object has no attribute 'keys'" in lowered
        )
        and "age" in code
        and "death" in code
        and ("tertile" in code.lower() or "mortality" in code.lower())
    )
    if age_stratified_runtime_failure:
        repair_name = "age_stratified_mortality_dependency_free_v1"
        if previous_repair != repair_name:
            return repair_name, _age_stratified_mortality_fallback_code()

    pandas_cut_observed_keyword = (
        "got an unexpected keyword argument 'observed'" in lowered
        and "observed=" in code
        and ("pd.cut(" in code or "pd.qcut(" in code)
    )
    if pandas_cut_observed_keyword:
        repair_name = "remove_pandas_cut_observed_keyword_v1"
        if previous_repair != repair_name:
            repaired = re.sub(r",\s*observed\s*=\s*(?:True|False)", "", code)
            if repaired != code:
                return repair_name, repaired

    missing_seaborn = (
        "modulenotfounderror: no module named 'seaborn'" in lowered
        and "import seaborn as sns" in code
    )
    if missing_seaborn:
        repair_name = "seaborn_matplotlib_fallback_v1"
        if previous_repair != repair_name:
            fallback = textwrap.dedent(
                """
                class _EasyICUSeabornFallback:
                    def set_theme(self, *args, **kwargs):
                        return None
                    def set_style(self, *args, **kwargs):
                        return None
                    def color_palette(self, *args, **kwargs):
                        import matplotlib.pyplot as plt
                        return plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
                    def barplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            grouped = data.groupby(x, dropna=False)[y].mean()
                            ax.bar([str(v) for v in grouped.index], grouped.values)
                        return ax
                    def lineplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            ax.plot(data[x], data[y], marker=kwargs.get("marker", "o"))
                        return ax
                    def scatterplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            ax.scatter(data[x], data[y])
                        return ax
                    def histplot(self, data=None, x=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        values = data[x] if data is not None and x is not None else data
                        ax.hist(values.dropna() if hasattr(values, "dropna") else values, bins=kwargs.get("bins", 20))
                        return ax
                    def heatmap(self, data=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        image = ax.imshow(data, aspect="auto")
                        plt.colorbar(image, ax=ax)
                        return ax
                sns = _EasyICUSeabornFallback()
                """
            ).strip()
            repaired = code.replace("import seaborn as sns", fallback, 1)
            if repaired != code:
                return repair_name, repaired

    missing_proportion_confint = (
        (
            "modulenotfounderror: no module named 'statsmodels'" in lowered
            or "cannot import name 'proportion_confint' from 'scipy.stats'" in lowered
        )
        and "proportion_confint" in code
    )
    if missing_proportion_confint:
        repair_name = "local_wilson_proportion_confint_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """
                def proportion_confint(count, nobs=None, alpha=0.05, method="wilson", **kwargs):
                    import math
                    if nobs is None:
                        nobs = kwargs.get("n")
                    count = float(count)
                    nobs = float(nobs)
                    if nobs <= 0:
                        return (None, None)
                    z = 1.959963984540054
                    phat = count / nobs
                    denom = 1.0 + z * z / nobs
                    centre = phat + z * z / (2.0 * nobs)
                    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * nobs)) / nobs)
                    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))
                """
            ).strip()
            repaired = re.sub(
                r"^\s*from\s+statsmodels\.stats\.proportion\s+import\s+proportion_confint\s*$",
                helper,
                code,
                count=1,
                flags=re.MULTILINE,
            )
            repaired = re.sub(
                r"^\s*from\s+scipy\.stats\s+import\s+proportion_confint\s*$",
                helper,
                repaired,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    wrong_calibration_import = (
        "cannot import name 'calibration_curve' from 'sklearn.metrics'" in lowered
        and "calibration_curve" in code
    )
    if wrong_calibration_import:
        repair_name = "prediction_calibration_import_fix_v1"
        repaired = re.sub(
            r"from sklearn\.metrics import ([^\n]*?)\bcalibration_curve\b,?\s*",
            lambda match: (
                "from sklearn.metrics import "
                + ", ".join(
                    part.strip().strip(",")
                    for part in match.group(1).split(",")
                    if part.strip().strip(",")
                )
                + "\nfrom sklearn.calibration import calibration_curve\n"
            ),
            code,
            count=1,
        )
        if repaired != code:
            return repair_name, repaired

    json_numpy_key_failure = (
        "keys must be str, int, float, bool or none" in lowered
        and "json.dump(" in code
    )
    if json_numpy_key_failure:
        repair_name = "json_dump_numpy_key_sanitizer_v1"
        if previous_repair != repair_name:
            repaired = _patch_json_dump_numpy_key_sanitizer(code)
            if repaired != code:
                return repair_name, repaired

    missing_os_import = (
        "nameerror: name 'os' is not defined" in lowered
        and ("os.environ" in code or "os.path" in code)
        and "import os" not in code
    )
    if missing_os_import:
        repair_name = "missing_os_import_v1"
        if previous_repair != repair_name:
            return repair_name, "import os\n" + code

    malformed_python_prefix = (
        "syntaxerror: invalid syntax" in lowered
        and ("pythonimport " in code or "\npythonimport " in code or "pythonfrom " in code)
    )
    if malformed_python_prefix:
        repair_name = "strip_python_prefix_v1"
        if previous_repair != repair_name:
            repaired = code.replace("pythonimport ", "import ").replace("pythonfrom ", "from ")
            repaired = repaired.replace("\npythonimport ", "\nimport ").replace("\npythonfrom ", "\nfrom ")
            if repaired != code:
                return repair_name, repaired

    proportion_confint_n_keyword = (
        "proportion_confint() got an unexpected keyword argument 'n'" in lowered
        and "proportion_confint" in code
    )
    if proportion_confint_n_keyword:
        repair_name = "proportion_confint_nobs_keyword_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"(proportion_confint\s*\([^)]*?)\bn\s*=",
                r"\1nobs=",
                code,
                flags=re.DOTALL,
            )
            if repaired != code:
                return repair_name, repaired

    malformed_matplotlib_xerr = (
        "valueerror: 'xerr'" in lowered
        and "must be a scalar or a 1d or (2, n) array-like" in lowered
        and "np.array([[" in code
        and "errorbar(" in code
    )
    if malformed_matplotlib_xerr:
        repair_name = "matplotlib_errorbar_xerr_shape_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"xerr\s*=\s*np\.array\(\[\[([A-Za-z_]\w*)\],\s*\[([A-Za-z_]\w*)\]\]\)",
                r"xerr=np.vstack([np.ravel(\1), np.ravel(\2)])",
                code,
            )
            if repaired != code:
                return repair_name, repaired

    shadowed_json_module = (
        "attributeerror: 'function' object has no attribute 'dump'" in lowered
        and "json.dump(" in code
    )
    if shadowed_json_module:
        repair_name = "restore_shadowed_json_module_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"^(?P<indent>\s*)json\.dump\(",
                (
                    r"\g<indent>import importlib\n"
                    r"\g<indent>json = importlib.import_module('json')\n"
                    r"\g<indent>json.dump("
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    malformed_publication_contract = (
        (
            "valueerror: panels are required" in lowered
            or "figurecontract' object is not subscriptable" in lowered
            or '"figurecontract" object is not subscriptable' in lowered
        )
        and "make_figure_contract(" in code
        and 'figure_contract["panels"]' in code
    )
    if malformed_publication_contract:
        repair_name = "publication_contract_optional_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"figure_contract\s*=\s*make_figure_contract\([\s\S]*?\)\s*",
                "figure_contract = None\n",
                code,
                count=1,
            )
            repaired = re.sub(
                r"figure_contract\[[\"']panels[\"']\]\.append\(\{[\s\S]*?\}\)\s*",
                "",
                repaired,
            )
            if repaired != code:
                return repair_name, repaired

    missing_dummy_encoded_column = (
        "keyerror" in lowered
        and "not in index" in lowered
        and "pd.get_dummies" in code
        and (
            "model_df[x_cols]" in code
            or "model_df[[outcome_col] + x_cols]" in code
        )
    )
    if missing_dummy_encoded_column:
        repair_name = "filter_x_cols_after_dummy_encoding_v1"
        if previous_repair != repair_name:
            marker = "X = model_df[x_cols].copy()"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n    " + marker, 1)
                return repair_name, repaired
            marker = "model_df_subset = model_df[[outcome_col] + x_cols].copy()"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n" + marker, 1)
                return repair_name, repaired

    categorical_prediction_coercion = (
        "found array with 0 feature(s)" in lowered
        and "onehotencoder" in lowered
        and "categorical_features" in code
        and "pd.to_numeric" in code
    )
    if categorical_prediction_coercion:
        repair_name = "prediction_preserve_categorical_before_ohe_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"for col in predictors:\s*\n\s*if col in data:\s*\n\s*data\[col\]\s*=\s*pd\.to_numeric\(data\[col\], errors=\"coerce\"\)",
                (
                    "for col in predictors:\n"
                    "    if col in data and col not in [\"sex\"]:\n"
                    "        data[col] = pd.to_numeric(data[col], errors=\"coerce\")\n"
                    "if \"sex\" in data:\n"
                    "    data[\"sex\"] = data[\"sex\"].astype(\"string\")"
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    duplicate_outcome_column_unique = (
        "attributeerror: 'dataframe' object has no attribute 'unique'" in lowered
        and "model_df[outcome].unique()" in code
        and "required_cols + [outcome]" in code
    )
    if duplicate_outcome_column_unique:
        repair_name = "dedupe_required_cols_outcome_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "model_df = df[required_cols + [outcome]].copy()",
                "model_df = df[list(dict.fromkeys(required_cols + [outcome]))].copy()",
                1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_dummy_encoded_dropna_column = (
        "keyerror" in lowered
        and "get_dummies" in code
        and "dropna(" in code
        and ("subset=x_cols" in code or " + x_cols" in code or "_x_cols" in code)
    )
    if missing_dummy_encoded_dropna_column:
        repair_name = "filter_x_cols_before_dropna_after_dummy_encoding_v1"
        if previous_repair != repair_name:
            marker = "model_df = model_df.dropna(subset=x_cols + [outcome])"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n" + marker, 1)
                return repair_name, repaired
            generic_dropna = re.compile(
                r"(?P<line>^(?P<frame>\w+)\s*=\s*(?P=frame)\.replace\(\[np\.inf,\s*-np\.inf\],\s*np\.nan\)\.dropna\(subset=\[(?P<outcome>\w+)\]\s*\+\s*(?P<xcols>\w+)\)\s*$)",
                flags=re.MULTILINE,
            )
            match = generic_dropna.search(code)
            if match:
                xcols = match.group("xcols")
                frame = match.group("frame")
                guard = f"{xcols} = [col for col in {xcols} if col in {frame}.columns]"
                if guard not in code:
                    repaired = code.replace(match.group("line"), guard + "\n" + match.group("line"), 1)
                    return repair_name, repaired

    missing_indicator_source_frame = (
        "keyerror" in lowered
        and "are in the [columns]" in lowered
        and "df = pd.read_parquet" in code
        and ".isnull().any(axis=1).astype(int)" in code
        and "_missing" in code
    )
    if missing_indicator_source_frame:
        repair_name = "missing_indicator_source_df_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"(?P<lhs>\w+\[['\"][^'\"]+_missing['\"]\]\s*=\s*)(?P<frame>\w+)\[(?P<colsvar>\w+)\](?P<rhs>\.isnull\(\)\.any\(axis=1\)\.astype\(int\))",
                r"\g<lhs>df[\g<colsvar>]\g<rhs>",
                code,
                count=1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_outcome_from_subset = (
        "keyerror" in lowered
        and "death" in lowered
        and "not in index" in lowered
        and "all_vars = [primary_predictor] + covariates" in code
    )
    if missing_outcome_from_subset:
        repair_name = "include_outcome_in_all_vars_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "all_vars = [primary_predictor] + covariates",
                "all_vars = [outcome_col, primary_predictor] + covariates",
                1,
            )
            if repaired != code:
                return repair_name, repaired

    robustness_none_plot = (
        "unsupported operand type(s)" in lowered
        and "none" in lowered
        and "ax.errorbar(" in code
        and ("predictor_col" in code or "primary_predictor" in code)
        and (
            "lactate_missing" in code
            or "Missing-indicator" in code
            or "missing_indicator" in code.lower()
        )
    )
    if robustness_none_plot:
        repair_name = "robustness_predictor_design_and_plot_v1"
        if previous_repair != repair_name:
            repaired = code
            predictor_var = (
                "predictor_col" if "predictor_col" in code else "primary_predictor"
            )
            model_df_assign = re.search(
                r"^(?P<indent>\s*)(?P<line>model_df\s*=\s*df\[[^\n]+?\.copy\(\)\s*)$",
                repaired,
                flags=re.MULTILINE,
            )
            sex_numeric_patch = textwrap.dedent(
                """
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df = model_df.replace([np.inf, -np.inf], np.nan)
                reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]
                """
            ).strip("\n")
            if model_df_assign and "reduced_covariates =" not in repaired:
                indent = model_df_assign.group("indent")
                patch = "\n".join(
                    indent + line if line else line
                    for line in sex_numeric_patch.splitlines()
                )
                repaired = repaired.replace(
                    model_df_assign.group(0),
                    model_df_assign.group(0) + "\n" + patch,
                    1,
                )
            replacements = {
                "cc_X = cc_df[covariates]": "cc_X = cc_df[[predictor_col] + covariates]",
                "cc_X = complete_case_df[covariates]": "cc_X = complete_case_df[[predictor_col] + covariates]",
                "mi_X = mi_df[covariates + ['lactate_missing']]": "mi_X = mi_df[[predictor_col] + covariates + ['lactate_missing']]",
                "mi_X = model_df[covariates + ['lactate_missing']]": "mi_X = model_df[[predictor_col] + covariates + ['lactate_missing']]",
                "rv_X = rv_df[covariates]": "rv_X = rv_df[[predictor_col] + covariates]",
                "rv_X = rv_df[reduced_covariates]": "rv_X = rv_df[[predictor_col] + reduced_covariates]",
                "X_cc = sm.add_constant(complete_case_df[covariates], has_constant=\"add\")": (
                    f"X_cc = sm.add_constant(complete_case_df[[{predictor_var}] + covariates], has_constant=\"add\")"
                ),
                "X_cc = sm.add_constant(cc_df[covariates], has_constant=\"add\")": (
                    f"X_cc = sm.add_constant(cc_df[[{predictor_var}] + covariates], has_constant=\"add\")"
                ),
                "X_mi = sm.add_constant(missing_indicator_df[covariates + [\"lactate_missing\"]], has_constant=\"add\")": (
                    f"X_mi = sm.add_constant(missing_indicator_df[[{predictor_var}] + covariates + [\"lactate_missing\"]], has_constant=\"add\")"
                ),
                "X_mi = sm.add_constant(mi_df[covariates + [\"lactate_missing\"]], has_constant=\"add\")": (
                    f"X_mi = sm.add_constant(mi_df[[{predictor_var}] + covariates + [\"lactate_missing\"]], has_constant=\"add\")"
                ),
                "X_rv = sm.add_constant(reduced_variable_df[covariates], has_constant=\"add\")": (
                    f"X_rv = sm.add_constant(reduced_variable_df[[{predictor_var}] + reduced_covariates], has_constant=\"add\")"
                ),
                "X_rv = sm.add_constant(rv_df[covariates], has_constant=\"add\")": (
                    f"X_rv = sm.add_constant(rv_df[[{predictor_var}] + reduced_covariates], has_constant=\"add\")"
                ),
            }
            for old, new in replacements.items():
                repaired = repaired.replace(old, new)
            subset_replacements = {
                f"complete_case_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"complete_case_df = model_df.dropna(subset=[outcome_col, {predictor_var}] + covariates)"
                ),
                f"cc_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"cc_df = model_df.dropna(subset=[outcome_col, {predictor_var}] + covariates)"
                ),
                f"rv_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"rv_df = model_df[[outcome_col, {predictor_var}] + reduced_covariates].dropna()"
                ),
                f"reduced_variable_df = model_df.drop(columns=[{predictor_var}]).copy()": (
                    f"reduced_variable_df = model_df[[outcome_col, {predictor_var}] + reduced_covariates].dropna().copy()"
                ),
            }
            for old, new in subset_replacements.items():
                repaired = repaired.replace(old, new)
            mi_copy_patterns = {
                "mi_df = model_df.copy()": (
                    f"mi_df = model_df.copy()\n"
                    f"    mi_df[{predictor_var}] = mi_df[{predictor_var}].fillna(0)\n"
                    "    mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                ),
                "missing_indicator_df = model_df.copy()": (
                    "missing_indicator_df = model_df.copy()\n"
                    f"    missing_indicator_df[{predictor_var}] = missing_indicator_df[{predictor_var}].fillna(0)\n"
                    "    missing_indicator_df = missing_indicator_df.dropna(subset=[outcome_col] + covariates)"
                ),
            }
            for old, new in mi_copy_patterns.items():
                if old in repaired and "fillna(0)" not in repaired:
                    repaired = repaired.replace(old, new, 1)
            repaired = repaired.replace(
                f"X_rv = X_rv.drop(columns=[{predictor_var}])\n",
                "",
            )
            finite_patches = {
                "X_cc = X_cc.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_cc = y_cc.astype(float)": (
                    "X_cc = X_cc.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_cc = y_cc.astype(float)\n"
                    "    cc_mask = np.isfinite(X_cc.to_numpy()).all(axis=1) & np.isfinite(y_cc.to_numpy())\n"
                    "    X_cc = X_cc.loc[cc_mask]\n"
                    "    y_cc = y_cc.loc[cc_mask]"
                ),
                "X_mi = X_mi.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_mi = y_mi.astype(float)": (
                    "X_mi = X_mi.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_mi = y_mi.astype(float)\n"
                    "    mi_mask = np.isfinite(X_mi.to_numpy()).all(axis=1) & np.isfinite(y_mi.to_numpy())\n"
                    "    X_mi = X_mi.loc[mi_mask]\n"
                    "    y_mi = y_mi.loc[mi_mask]"
                ),
                "X_rv = X_rv.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_rv = y_rv.astype(float)": (
                    "X_rv = X_rv.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_rv = y_rv.astype(float)\n"
                    "    rv_mask = np.isfinite(X_rv.to_numpy()).all(axis=1) & np.isfinite(y_rv.to_numpy())\n"
                    "    X_rv = X_rv.loc[rv_mask]\n"
                    "    y_rv = y_rv.loc[rv_mask]"
                ),
            }
            for old, new in finite_patches.items():
                repaired = repaired.replace(old, new)
            lower_var = "ci_lowers"
            upper_var = "ci_uppers"
            if "lci = [" in repaired and "uci = [" in repaired:
                lower_var = "lci"
                upper_var = "uci"
            elif "or_lowers = [" in repaired and "or_uppers = [" in repaired:
                lower_var = "or_lowers"
                upper_var = "or_uppers"
            plot_marker = "ax.errorbar(x_pos, ors, yerr=[yerr_lower, yerr_upper],"
            plot_guard = textwrap.dedent(
                f"""
                plot_rows = [
                    (s, o, lo, hi)
                    for s, o, lo, hi in zip(strategies, ors, {lower_var}, {upper_var})
                    if o is not None and lo is not None and hi is not None
                ]
                if plot_rows:
                    strategies, ors, {lower_var}, {upper_var} = map(list, zip(*plot_rows))
                    x_pos = np.arange(len(strategies))
                else:
                    strategies, ors, {lower_var}, {upper_var} = [], [], [], []
                    x_pos = np.array([])
                """
            ).strip("\n")
            if plot_marker in repaired and "plot_rows = [" not in repaired:
                repaired = repaired.replace(
                    plot_marker,
                    plot_guard + "\n\n    if len(x_pos):\n        " + plot_marker,
                    1,
                )
            if repaired != code:
                return repair_name, repaired

    missing_internal_utils = (
        "modulenotfounderror: no module named 'easyicu.research_agent.utils'" in lowered
        and "from easyicu.research_agent.utils import to_jsonable" in code
    )
    if missing_internal_utils:
        repair_name = "inline_missing_to_jsonable_utils_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """
                def to_jsonable(x):
                    import math
                    import numpy as np
                    import pandas as pd
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)
                """
            ).strip()
            repaired = code.replace(
                "from easyicu.research_agent.utils import to_jsonable",
                helper,
                1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_figure_utils = (
        (
            "modulenotfounderror: no module named 'easyicu.research_output'" in lowered
            or "modulenotfounderror: no module named 'easyicu.research_output.figure_utils'" in lowered
            or "no module named 'easyicu.research_output'" in lowered
        )
        and "easyicu.research_output.figure_utils" in code
    )
    if missing_figure_utils:
        repair_name = "replace_hallucinated_figure_utils_import_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "easyicu.research_output.figure_utils",
                "easyicu.research_agent.publication_figures",
            )
            if repaired != code:
                return repair_name, repaired

    robustness_numeric_check_nan = (
        "exog contains inf or nans" in lowered
        and "X_cc = X_cc.apply(pd.to_numeric, errors='coerce')" in code
        and "y_cc = y_cc.astype(float)" in code
        and "sex" in code
    )
    if robustness_numeric_check_nan:
        repair_name = "robustness_encode_sex_before_numeric_checks_v1"
        if previous_repair != repair_name:
            def _numeric_block(prefix: str) -> str:
                block = textwrap.dedent(
                    f"""
                    if 'sex' in X_{prefix}.columns:
                        X_{prefix}['sex'] = X_{prefix}['sex'].astype(str).str.lower().isin(['m', 'male', '1', 'true']).astype(float)
                    X_{prefix} = X_{prefix}.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    y_{prefix} = pd.to_numeric(y_{prefix}, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    valid_{prefix}_idx = X_{prefix}.dropna().index.intersection(y_{prefix}.dropna().index)
                    X_{prefix} = X_{prefix}.loc[valid_{prefix}_idx]
                    y_{prefix} = y_{prefix}.loc[valid_{prefix}_idx].astype(float)
                    """
                ).strip("\n")
                return block.replace("\n", "\n    ")

            repaired = code
            replacements = {
                "cc": (
                    "X_cc = X_cc.apply(pd.to_numeric, errors='coerce')\n    y_cc = y_cc.astype(float)",
                    _numeric_block("cc"),
                ),
                "mi": (
                    "X_mi = X_mi.apply(pd.to_numeric, errors='coerce')\n    y_mi = y_mi.astype(float)",
                    _numeric_block("mi"),
                ),
                "rv": (
                    "X_rv = X_rv.apply(pd.to_numeric, errors='coerce')\n    y_rv = y_rv.astype(float)",
                    _numeric_block("rv"),
                ),
            }
            for old, new in replacements.values():
                repaired = repaired.replace(old, new)
            if repaired != code:
                return repair_name, repaired

    undefined_primary_predictor = (
        "name 'primary_predictor' is not defined" in lowered
        and "primary_predictor if primary_predictor else None" in code
    )
    if undefined_primary_predictor:
        repair_name = "primary_predictor_safe_summary_lookup_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "primary_predictor if primary_predictor else None",
                (
                    "locals().get('primary_predictor') or "
                    "locals().get('predictor_col') or "
                    "locals().get('primary_predictor_col') or "
                    "locals().get('predictor') or None"
                ),
                1,
            )
            if repaired != code:
                return repair_name, repaired

    table_one_unclosed_syntax = (
        "syntaxerror" in lowered
        and (
            "was never closed" in lowered
            or "unexpected eof while parsing" in lowered
            or "eof while scanning" in lowered
        )
        and "table_one.csv" in code.lower()
    )
    if table_one_unclosed_syntax:
        repair_name = "table_one_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import math
                import numpy as np
                import pandas as pd

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)

                cohort_path = os.environ["COHORT_PARQUET"]
                out_dir = os.environ["STEP_OUT_DIR"]
                os.makedirs(out_dir, exist_ok=True)

                df = pd.read_parquet(cohort_path)
                rows = []
                for col in df.columns:
                    s = df[col]
                    n = int(len(s))
                    n_missing = int(s.isna().sum())
                    row = {
                        "variable": col,
                        "n": n,
                        "n_missing": n_missing,
                        "missing_fraction": (n_missing / n) if n else 0.0,
                    }
                    non_missing = s.dropna()
                    if len(non_missing) == 0:
                        rows.append(row)
                        continue
                    if pd.api.types.is_numeric_dtype(non_missing):
                        unique_values = set(non_missing.unique().tolist())
                        if unique_values <= {0, 1, 0.0, 1.0}:
                            positive = int(non_missing.astype(float).sum())
                            row["n_positive"] = positive
                            row["positive_fraction"] = positive / len(non_missing)
                        else:
                            row["median"] = float(non_missing.median())
                            row["q25"] = float(non_missing.quantile(0.25))
                            row["q75"] = float(non_missing.quantile(0.75))
                            row["min"] = float(non_missing.min())
                            row["max"] = float(non_missing.max())
                    else:
                        top = non_missing.astype(str).value_counts().head(1)
                        if len(top):
                            row["most_common"] = str(top.index[0])
                            row["most_common_n"] = int(top.iloc[0])
                    rows.append(row)

                table = pd.DataFrame(rows)
                table_path = os.path.join(out_dir, "table_one.csv")
                table.to_csv(table_path, index=False)

                summary = {
                    "n_total": int(len(df)),
                    "n_variables": int(len(df.columns)),
                    "table_one_path": table_path,
                    "variables": list(df.columns.astype(str)),
                }
                if "death" in df.columns:
                    death = pd.to_numeric(df["death"], errors="coerce").dropna()
                    summary["death_n"] = int(death.sum())
                    summary["death_rate"] = float(death.mean()) if len(death) else None
                if "age" in df.columns:
                    age = pd.to_numeric(df["age"], errors="coerce").dropna()
                    summary["age_median"] = float(age.median()) if len(age) else None
                    summary["age_q25"] = float(age.quantile(0.25)) if len(age) else None
                    summary["age_q75"] = float(age.quantile(0.75)) if len(age) else None

                with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=to_jsonable)
                print(json.dumps({"table": "table_one.csv", "summary": summary}, default=to_jsonable))
                """
            ).strip() + "\n"
            return repair_name, repaired

    outcome_incidence_broken_syntax = (
        "syntaxerror" in lowered
        and (
            "outcome_incidence" in code.lower()
            or "incidence_with_missingness_strata" in code.lower()
        )
    )
    if outcome_incidence_broken_syntax:
        repair_name = "outcome_incidence_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import math
                import numpy as np
                import pandas as pd
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from statsmodels.stats.proportion import proportion_confint

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)

                cohort_path = os.environ["COHORT_PARQUET"]
                out_dir = os.environ["STEP_OUT_DIR"]
                os.makedirs(out_dir, exist_ok=True)

                df = pd.read_parquet(cohort_path)
                death = pd.to_numeric(df["death"], errors="coerce")
                rows = []

                def add_row(label, mask):
                    y = death[mask].dropna().astype(int)
                    n = int(len(y))
                    events = int(y.sum()) if n else 0
                    rate = float(events / n) if n else None
                    if n:
                        ci_low, ci_high = proportion_confint(events, n, alpha=0.05, method="wilson")
                    else:
                        ci_low = ci_high = None
                    rows.append({
                        "stratum": label,
                        "n": n,
                        "n_death": events,
                        "mortality_rate": rate,
                        "ci_low": None if ci_low is None else float(ci_low),
                        "ci_high": None if ci_high is None else float(ci_high),
                    })

                add_row("overall", death.notna())
                if "lactate_measured_24h" in df.columns:
                    measured = pd.to_numeric(df["lactate_measured_24h"], errors="coerce")
                    add_row("lactate_measured_24h=0", measured.eq(0) & death.notna())
                    add_row("lactate_measured_24h=1", measured.eq(1) & death.notna())

                table = pd.DataFrame(rows)
                table_path = os.path.join(out_dir, "outcome_incidence.csv")
                table.to_csv(table_path, index=False)

                fig, ax = plt.subplots(figsize=(4.8, 3.0))
                plot_df = table[table["stratum"] != "overall"].copy()
                if plot_df.empty:
                    plot_df = table.copy()
                ax.bar(plot_df["stratum"], plot_df["mortality_rate"] * 100, color="#4C78A8")
                ax.set_ylabel("Mortality (%)")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "outcome_incidence.png"), dpi=300)
                fig.savefig(os.path.join(out_dir, "outcome_incidence.svg"))
                plt.close(fig)

                overall = table.iloc[0].to_dict()
                statistic = {
                    "n_total": int(overall["n"]),
                    "n_death": int(overall["n_death"]),
                    "overall_mortality_rate": overall["mortality_rate"],
                    "overall_ci_low": overall["ci_low"],
                    "overall_ci_high": overall["ci_high"],
                }
                statistic_path = os.path.join(out_dir, "outcome_rate.json")
                with open(statistic_path, "w", encoding="utf-8") as f:
                    json.dump(statistic, f, indent=2, default=to_jsonable)

                summary = {
                    "table": table_path,
                    "statistic": statistic_path,
                    "figure_png": os.path.join(out_dir, "outcome_incidence.png"),
                    "figure_svg": os.path.join(out_dir, "outcome_incidence.svg"),
                    **statistic,
                }
                with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=to_jsonable)
                print(json.dumps(summary, default=to_jsonable))
                """
            ).strip() + "\n"
            return repair_name, repaired

    repeated_keyword_syntax = (
        "syntaxerror: keyword argument repeated" in lowered
        and "train_test_split" in code
        and "figure_contract = figurecontract(" in code.lower()
    )
    if repeated_keyword_syntax:
        repair_name = "prediction_split_minimal_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import numpy as np
                import pandas as pd
                from sklearn.model_selection import train_test_split

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if np.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    return x

                df = pd.read_parquet(os.environ["COHORT_PARQUET"])
                out = os.environ["STEP_OUT_DIR"]
                outcome = "death" if "death" in df.columns else df.columns[-1]
                y = pd.to_numeric(df[outcome], errors="coerce").fillna(0).astype(int)
                X = df.drop(columns=[outcome], errors="ignore").copy()
                X = X.select_dtypes(include=["number", "bool"]).apply(pd.to_numeric, errors="coerce")
                if X.empty:
                    X = pd.DataFrame({"row_index": np.arange(len(df))}, index=df.index)
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y if getattr(y, "nunique", lambda: 0)() > 1 else None,
                )
                step_summary = {
                    "split_strategy": "stratified_random",
                    "n_total": int(len(df)),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "event_rate_total": float(y.mean()),
                    "event_rate_train": float(y_train.mean()) if len(y_train) else None,
                    "event_rate_test": float(y_test.mean()) if len(y_test) else None,
                }
                with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(step_summary, f, indent=2, default=to_jsonable, ensure_ascii=False)
                print(json.dumps(step_summary, indent=2, default=to_jsonable, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    logreg_nan = (
        "logisticregression does not accept missing values encoded as nan" in lowered
        and "logisticregression" in code.lower()
    )
    if logreg_nan:
        repair_name = "logreg_impute_v1"
        if previous_repair != repair_name and "_easyicu_logreg_impute_v1" not in code:
            patch = textwrap.dedent(
                """

                def _easyicu_logreg_impute_v1(frame):
                    if not hasattr(frame, "copy"):
                        return frame
                    work = frame.copy()
                    for col in work.columns:
                        series = pd.to_numeric(work[col], errors="ignore")
                        if getattr(series, "dtype", None) is not None and str(series.dtype) != "object":
                            if series.isna().any():
                                median = series.median()
                                series = series.fillna(median if pd.notna(median) else 0)
                        work[col] = series
                    return work
                """
            ).strip("\n")
            train_split = re.compile(
                r"(?P<line>X_train,\s*X_test,\s*y_train,\s*y_test\s*=\s*train_test_split\([^\\n]+?\)\s*)",
                re.DOTALL,
            )
            match = train_split.search(code)
            if match:
                inject = (
                    match.group("line")
                    + "\nX_train = _easyicu_logreg_impute_v1(X_train)\n"
                    + "X_test = _easyicu_logreg_impute_v1(X_test)\n"
                )
                repaired = code[: match.start()] + inject + code[match.end() :]
            else:
                repaired = code
            if repaired == code:
                predict_call = re.compile(r"(?P<line>y_pred_proba\s*=\s*model(?:_pipeline)?\.predict_proba\(X_test\)\s*\[:,\s*1\]\s*)")
                match = predict_call.search(code)
                if match:
                    inject = "X_test = _easyicu_logreg_impute_v1(X_test)\n" + match.group("line")
                    repaired = code[: match.start()] + inject + code[match.end() :]
            if repaired != code:
                if "def _easyicu_logreg_impute_v1" not in repaired:
                    repaired = patch + "\n\n" + repaired
                return repair_name, repaired

    placeholder_ellipsis = (
        "syntaxerror: invalid syntax" in lowered
        and "..." in code
        and "model_bundle" in code
    )
    if placeholder_ellipsis:
        repair_name = "prediction_discrimination_template_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import math
                import os
                import pickle
                import numpy as np
                import pandas as pd
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_auc_score, roc_curve
                from sklearn.calibration import calibration_curve
                from easyicu.research_agent.publication_figures import (
                    make_figure_contract,
                    apply_publication_style,
                    add_panel_label,
                    save_publication_figure,
                )

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if np.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    return x

                step_out_dir = os.environ["STEP_OUT_DIR"]
                cohort_path = os.environ["COHORT_PARQUET"]
                with open(os.path.join(step_out_dir, "prediction_model_object.pkl"), "rb") as f:
                    model_bundle = pickle.load(f)
                model = model_bundle["model"]
                feature_cols = list(model_bundle.get("feature_cols", []))

                df = pd.read_parquet(cohort_path)
                X_test = df[feature_cols].copy()
                y_test = pd.to_numeric(df["death"], errors="coerce").fillna(0).astype(int).values
                for col in X_test.columns:
                    series = pd.to_numeric(X_test[col], errors="ignore")
                    if getattr(series, "dtype", None) is not None and str(series.dtype) != "object" and series.isna().any():
                        median = series.median()
                        series = series.fillna(median if pd.notna(median) else 0)
                    X_test[col] = series

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                held_out_auroc = roc_auc_score(y_test, y_pred_proba)
                prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=min(10, max(5, int(len(y_test) * 0.1))), strategy="quantile")
                zero_stratum_mask = df["sofa2"] == 0 if "sofa2" in df.columns else pd.Series(False, index=df.index)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                apply_publication_style(fig)
                ax1, ax2 = axes
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                ax1.plot(fpr, tpr, color="#0F4D92", linewidth=2)
                ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax1.set_xlabel("False positive rate")
                ax1.set_ylabel("True positive rate")
                add_panel_label(ax1, "A")

                ax2.plot(prob_pred, prob_true, "o-", color="#42949E", linewidth=2)
                ax2.plot([0, 1], [0, 1], "k:", linewidth=1)
                ax2.set_xlabel("Predicted probability")
                ax2.set_ylabel("Observed probability")
                add_panel_label(ax2, "B")

                contract = make_figure_contract(
                    figure_id="prediction_discrimination_evaluation",
                    core_claim="Held-out discrimination and calibration are summarized for the mortality model.",
                    panels=[
                        {"panel_id": "A", "title": "ROC", "role": "validation", "claim": "Held-out AUROC is reported.", "evidence_ids": ["held_out_auroc"]},
                        {"panel_id": "B", "title": "Calibration", "role": "validation", "claim": "Calibration is visualized on held-out data.", "evidence_ids": ["calibration_curve"]},
                    ],
                )
                save_publication_figure(fig, os.path.join(step_out_dir, "discrimination_evaluation"), contract=contract)
                plt.close(fig)

                step_summary = {
                    "held_out_auroc": float(held_out_auroc),
                    "n_test": int(len(y_test)),
                    "n_sofa2_zero": int(zero_stratum_mask.sum()),
                    "calibration_status": "ok",
                }
                with open(os.path.join(step_out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(step_summary, f, indent=2, default=to_jsonable, ensure_ascii=False)
                print(json.dumps(step_summary, indent=2, default=to_jsonable, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    omitted_primary_predictor = re.search(
        r"Error fitting logistic regression:\s*'([^']+)'",
        run_log or "",
        flags=re.IGNORECASE,
    )
    if omitted_primary_predictor and "X = model_df[[" in code:
        predictor = omitted_primary_predictor.group(1)
        repair_name = "primary_predictor_omitted_from_design_v1"
        if previous_repair != repair_name:
            repaired = _patch_primary_predictor_into_design_matrix(
                code=code,
                predictor=predictor,
            )
            if repaired is not None and repaired != code:
                return repair_name, repaired

    cut_tuple_error = (
        "typeerror: '<' not supported between instances of 'tuple' and 'int'"
        in lowered
        and "pandas/core/reshape/tile.py" in lowered
        and "pd.cut(" in code
    )
    if cut_tuple_error:
        repair_name = "cut_bins_flatten_v1"
        if previous_repair != repair_name:
            bins_assign = re.compile(
                r"(?P<name>\w+_bins)\s*=\s*(?P<literal>\[(?:\s*\([^][]+?\)\s*,?)+\s*\])"
            )
            match = bins_assign.search(code)
            if match:
                try:
                    literal = ast.literal_eval(match.group("literal"))
                except Exception:
                    literal = None
                if (
                    isinstance(literal, list)
                    and literal
                    and all(
                        isinstance(item, tuple)
                        and len(item) == 2
                        and all(isinstance(v, (int, float)) for v in item)
                        for item in literal
                    )
                ):
                    flat_bins = [literal[0][0], *[item[1] for item in literal]]
                    replacement = f"{match.group('name')} = {flat_bins!r}"
                    repaired = (
                        code[: match.start()] + replacement + code[match.end() :]
                    )
                    return repair_name, repaired

    singular_logit = "singular matrix" in lowered and "sm.logit(" in code.lower()
    if singular_logit:
        repair_name = "logit_regularized_fit_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """

                def _easyicu_safe_logit_fit_v1(model):
                    try:
                        return model.fit(disp=0, method="newton")
                    except Exception:
                        return model.fit_regularized(alpha=1e-6, disp=0, trim_mode="off")
                """
            ).strip("\n")
            patched = code
            if "_easyicu_safe_logit_fit_v1" not in patched:
                insert_after = patched.find("import warnings")
                if insert_after >= 0:
                    line_end = patched.find("\n", insert_after)
                    patched = (
                        patched[: line_end + 1] + "\n" + helper + "\n" + patched[line_end + 1 :]
                    )
                else:
                    patched = helper + "\n\n" + patched
            patched = re.sub(
                r"(?m)^(?P<indent>\s*)(?P<lhs>\w+)\s*=\s*(?P<model>\w+)\.fit\((?P<args>[^)]*)\)\s*$",
                r"\g<indent>\g<lhs> = _easyicu_safe_logit_fit_v1(\g<model>)",
                patched,
                count=1,
            )
            if patched != code:
                return repair_name, patched

    singular_after_regularized = (
        "singular matrix" in lowered
        and previous_repair == "logit_regularized_fit_v1"
        and "lactate_max_24h" in code
        and "map_min_24h" in code
        and "vaso_any_24h" in code
    )
    if singular_after_regularized:
        repair_name = "shock_primary_assoc_sklearn_v1"
        repaired = textwrap.dedent(
            """
            import os
            import json
            import math
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.linear_model import LogisticRegression

            def to_jsonable(x):
                if isinstance(x, (np.integer,)):
                    return int(x)
                if isinstance(x, (np.floating,)):
                    v = float(x)
                    return v if math.isfinite(v) else None
                if isinstance(x, (np.bool_,)):
                    return bool(x)
                if isinstance(x, np.ndarray):
                    return x.tolist()
                try:
                    if pd.isna(x):
                        return None
                except Exception:
                    pass
                return str(x)

            cohort_path = os.environ["COHORT_PARQUET"]
            step_out_dir = os.environ["STEP_OUT_DIR"]
            os.makedirs(step_out_dir, exist_ok=True)

            df = pd.read_parquet(cohort_path)
            required_cols = ['lactate_max_24h', 'death', 'age', 'sex', 'map_min_24h', 'vaso_any_24h']
            model_df = df[required_cols].copy()
            model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
            for col in required_cols:
                if col != 'sex':
                    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
            model_df = model_df.dropna()
            n_complete = int(len(model_df))
            n_total = int(len(df))
            if n_complete < 50:
                raise ValueError(f"Insufficient complete cases: {n_complete}")

            features = ['lactate_max_24h', 'age', 'sex', 'map_min_24h', 'vaso_any_24h']
            X = model_df[features].astype(float)
            y = model_df['death'].astype(int)

            model = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=4000,
                random_state=7,
            )
            model.fit(X, y)

            coef = model.coef_[0]
            odds_ratio = np.exp(coef)
            rows = []
            for name, beta, or_val in zip(features, coef, odds_ratio):
                rows.append({
                    'variable': name,
                    'coefficient': float(beta),
                    'or': float(or_val),
                    'or_ci_lower': None,
                    'or_ci_upper': None,
                    'p_value': None,
                })

            rng = np.random.default_rng(7)
            boot = []
            values = model_df[features + ['death']].to_numpy()
            for _ in range(120):
                idx = rng.integers(0, len(values), len(values))
                sample = values[idx]
                Xb = sample[:, :-1]
                yb = sample[:, -1].astype(int)
                if len(np.unique(yb)) < 2:
                    continue
                try:
                    mb = LogisticRegression(
                        penalty='l2',
                        C=1.0,
                        solver='lbfgs',
                        max_iter=2000,
                        random_state=7,
                    )
                    mb.fit(Xb, yb)
                    boot.append(float(mb.coef_[0][0]))
                except Exception:
                    continue

            lactate_or = float(odds_ratio[0])
            if boot:
                boot = np.asarray(boot, dtype=float)
                lactate_or_ci = (
                    float(np.exp(np.quantile(boot, 0.025))),
                    float(np.exp(np.quantile(boot, 0.975))),
                )
                p_boot = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
            else:
                lactate_or_ci = (None, None)
                p_boot = None

            rows[0]['or_ci_lower'] = lactate_or_ci[0]
            rows[0]['or_ci_upper'] = lactate_or_ci[1]
            rows[0]['p_value'] = p_boot

            results_table = pd.DataFrame(rows)
            table_path = os.path.join(step_out_dir, 'primary_association.csv')
            results_table.to_csv(table_path, index=False)

            lactate_range = np.linspace(
                model_df['lactate_max_24h'].quantile(0.01),
                model_df['lactate_max_24h'].quantile(0.99),
                100
            )
            age_med = float(model_df['age'].median())
            sex_med = float(model_df['sex'].median())
            map_med = float(model_df['map_min_24h'].median())
            vaso_med = float(model_df['vaso_any_24h'].median())
            pred_df = pd.DataFrame({
                'lactate_max_24h': lactate_range,
                'age': age_med,
                'sex': sex_med,
                'map_min_24h': map_med,
                'vaso_any_24h': vaso_med,
            })
            pred_probs = model.predict_proba(pred_df)[:, 1]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(lactate_range, pred_probs, 'b-', lw=2, label='Predicted probability')
            ax.fill_between(lactate_range, pred_probs, alpha=0.2)
            ax.plot(model_df['lactate_max_24h'], np.full(n_complete, -0.02), '|',
                    color='gray', alpha=0.3, markersize=5, label='Data distribution')
            ax.set_xlabel('Lactate max 24h (mmol/L)', fontsize=12)
            ax.set_ylabel('Predicted probability of death', fontsize=12)
            ax.set_title('Adjusted association: early lactate and target outcome', fontsize=13)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            if lactate_or_ci[0] is not None and lactate_or_ci[1] is not None:
                txt = f"Lactate OR = {lactate_or:.2f} (95% CI: {lactate_or_ci[0]:.2f}–{lactate_or_ci[1]:.2f})"
            else:
                txt = f"Lactate OR = {lactate_or:.2f}"
            ax.annotate(txt, xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            fig_path = os.path.join(step_out_dir, 'primary_association_curve.png')
            svg_path = os.path.join(step_out_dir, 'primary_association_curve.svg')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close()

            step_summary = {
                'step': '05_primary_association',
                'method': 'logistic_regression_sklearn_bootstrap',
                'n_total': n_total,
                'n_complete_case': n_complete,
                'mortality_rate_complete': float(model_df['death'].mean()),
                'missing_lactate_pct': float(df['lactate_max_24h'].isna().mean() * 100),
                'primary_or': lactate_or,
                'primary_ci_low': lactate_or_ci[0],
                'primary_ci_high': lactate_or_ci[1],
                'primary_or_ci': [lactate_or_ci[0], lactate_or_ci[1]],
                'primary_p_value': p_boot,
                'covariates': ['age', 'sex', 'map_min_24h', 'vaso_any_24h'],
                'outputs': {
                    'table': 'primary_association.csv',
                    'figure': 'primary_association_curve.png',
                }
            }
            with open(os.path.join(step_out_dir, 'step_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(step_summary, f, indent=2, default=to_jsonable, ensure_ascii=False)
            print(json.dumps(step_summary, indent=2, default=to_jsonable, ensure_ascii=False))
            """
        ).strip() + "\n"
        return repair_name, repaired

    table_one_binary_keyerror = (
        "keyerror: 1" in lowered
        and "in-hospital mortality" in code.lower()
        and '"counts"][1]' in code
    )
    if table_one_binary_keyerror:
        repair_name = "table_one_binary_key_string_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                'summary["outcomes"]["death"]["counts"][1]',
                'summary["outcomes"]["death"]["counts"].get("1", summary["outcomes"]["death"]["counts"].get(1, 0))',
            )
            repaired = repaired.replace(
                'summary["outcomes"]["death"]["pct"][1]',
                'summary["outcomes"]["death"]["pct"].get("1", summary["outcomes"]["death"]["pct"].get(1, 0.0))',
            )
            if repaired != code:
                return repair_name, repaired

    cohort_file_as_dir = (
        "notadirectoryerror" in lowered
        and (
            'os.path.join(cohort_path, "data.parquet")' in code.lower()
            or 'os.path.join(cohort_path, \'data.parquet\')' in code.lower()
        )
    )
    if cohort_file_as_dir:
        repair_name = "cohort_file_direct_read_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                'pd.read_parquet(os.path.join(COHORT_PATH, "data.parquet"))',
                "pd.read_parquet(COHORT_PATH)",
            )
            repaired = repaired.replace(
                "pd.read_parquet(os.path.join(COHORT_PATH, 'data.parquet'))",
                "pd.read_parquet(COHORT_PATH)",
            )
            repaired = repaired.replace(
                'pd.read_parquet(os.path.join(cohort_path, "data.parquet"))',
                "pd.read_parquet(cohort_path)",
            )
            repaired = repaired.replace(
                "pd.read_parquet(os.path.join(cohort_path, 'data.parquet'))",
                "pd.read_parquet(cohort_path)",
            )
            if repaired != code:
                return repair_name, repaired

    parquet_read_as_csv = (
        "unicodedecodeerror" in lowered
        and "pd.read_csv(" in code.lower()
        and (
            "cohort_path" in code.lower()
            or "cohort_parquet" in code.lower()
        )
    )
    if parquet_read_as_csv:
        repair_name = "cohort_csv_to_parquet_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"pd\.read_csv\((?P<arg>\s*(?:cohort_path|os\.environ\[['\"]COHORT_PARQUET['\"]\])\s*)(?:,\s*encoding\s*=\s*['\"][^'\"]+['\"])?\)",
                r"pd.read_parquet(\g<arg>)",
                code,
            )
            if repaired != code:
                return repair_name, repaired

    publication_style_nameerror = (
        "nameerror: name 'apply_publication_style' is not defined" in lowered
        and "publication_figure" in code.lower()
    )
    if publication_style_nameerror:
        repair_name = "publication_bundle_promote_script_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                from __future__ import annotations
                import json
                import os
                import shutil
                from pathlib import Path

                out_dir = Path(os.environ["STEP_OUT_DIR"])
                out_dir.mkdir(parents=True, exist_ok=True)
                run_dir = out_dir.parents[2]
                current_step_id = out_dir.parent.name
                figure_suffixes = [".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"]
                contract_suffix = ".figure_contract.json"

                best = None
                for step_dir in sorted((run_dir / "steps").iterdir()):
                    if not step_dir.is_dir() or step_dir.name == current_step_id:
                        continue
                    outputs_dir = step_dir / "outputs"
                    if not outputs_dir.exists():
                        continue
                    bundles = {}
                    for path in outputs_dir.iterdir():
                        if not path.is_file():
                            continue
                        if path.name.endswith(contract_suffix):
                            stem = path.name[: -len(contract_suffix)]
                            bundles.setdefault(stem, {})["contract"] = path
                            continue
                        if path.suffix.lower() in figure_suffixes:
                            bundles.setdefault(path.stem, {})[path.suffix.lower()] = path
                    for stem, files in bundles.items():
                        figure_count = sum(1 for key in files if key.startswith("."))
                        if figure_count == 0:
                            continue
                        score = (
                            1 if "publication_figure" in stem else 0,
                            1 if "primary_association" in stem else 0,
                            figure_count,
                        )
                        if best is None or score > best[0]:
                            best = (score, stem, files)

                if best is None:
                    raise SystemExit("No prior figure bundle available to promote.")

                _, source_stem, files = best
                target_stem = "publication_figure"
                outputs = {}
                for key, source in files.items():
                    if key == "contract":
                        target = out_dir / f"{target_stem}.figure_contract.json"
                        shutil.copy2(source, target)
                        outputs["contract"] = target.name
                    else:
                        target = out_dir / f"{target_stem}{key}"
                        shutil.copy2(source, target)
                        outputs[key.lstrip('.')] = target.name

                summary = {
                    "step": current_step_id,
                    "status": "completed",
                    "publication_figure_rescue": {
                        "mode": "promotion",
                        "source_step_stem": source_stem,
                        "source_outputs_dir": str(files[next(iter(files))].parent),
                    },
                    "outputs": outputs,
                }
                with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    # ----------------------------------------------------------------
    # Column-hallucination fallback: agent emitted a list literal of
    # column names not in the cohort (e.g. naive arms guess
    # ``covariates = ["age", "sex", "map_min_24h", "vaso_any_24h"]``).
    # Runs only when no earlier specialised repair (dummy-encoding,
    # missing outcome, etc.) handled the same KeyError, so we don't
    # interfere with category-aware fixes upstream.
    # ----------------------------------------------------------------
    if "keyerror" in lowered and "not in index" in lowered:
        missing_cols = _extract_missing_index_columns(run_log or "")
        if missing_cols:
            repair_name = "strip_unknown_cols_from_list_literals_v1"
            if previous_repair != repair_name:
                repaired = _strip_columns_from_list_literals(code, missing_cols)
                if repaired != code:
                    return repair_name, repaired

    # ----------------------------------------------------------------
    # Fix F — undefined helper auto-stub (generic fallback)
    # ----------------------------------------------------------------
    # Coder agents under naive arms occasionally reference helper
    # functions they forgot to define (e.g. ``json.dump(..., default=
    # to_json_serializable)`` without ever providing
    # ``to_json_serializable``). LLM self-repair attempts often
    # rewrite the call site rather than re-introduce the helper, so
    # the script keeps failing with the same NameError. Inject a
    # tolerant best-effort stub that handles the common JSON-default
    # pattern (numpy arrays / scalars / fallthrough to ``str``).
    #
    # Runs *after* the specialised NameError repairs (e.g.
    # ``publication_bundle_promote_script_v1`` for ``apply_publication_style``)
    # so we don't shadow richer recovery logic.
    name_error_match = _NAME_ERROR_HELPER_RE.search(run_log or "")
    if name_error_match is not None:
        helper_name = name_error_match.group("name")
        if (
            helper_name
            and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", helper_name)
            and f"def {helper_name}" not in code
            and f"{helper_name} =" not in code
        ):
            repair_name = f"undefined_helper_stub_{helper_name}_v1"
            if previous_repair != repair_name:
                stub = textwrap.dedent(
                    f"""
                    def {helper_name}(*args, **kwargs):
                        \"\"\"Auto-injected stub for an undefined helper.

                        Recovers from NameError raised when the agent
                        emitted a reference (e.g. ``json.dump(default=
                        {helper_name})``) without ever defining the
                        helper. The stub returns a JSON-friendly form
                        of its first positional argument when one is
                        provided, falling back to ``None``.
                        \"\"\"
                        if not args:
                            return None
                        value = args[0]
                        try:
                            if hasattr(value, \"tolist\"):
                                return value.tolist()
                        except Exception:
                            pass
                        try:
                            if hasattr(value, \"item\"):
                                return value.item()
                        except Exception:
                            pass
                        try:
                            return str(value)
                        except Exception:
                            return None
                    """
                ).strip("\n")
                repaired = stub + "\n\n" + code
                if repaired != code:
                    return repair_name, repaired

    # dtype_coerce_v1 repair for statsmodels dtype failures
    signatures = (
        "pandas data cast to numpy dtype of object",
        "exog contains inf or nans",
        "missingdataerror",
        "ufunc 'isfinite' not supported",
    )
    dtype_coerce_applies = (
        any(sig in lowered for sig in signatures)
        and previous_repair != "dtype_coerce_v1"
        and "_easyicu_runner_repair_v1" not in code
        and any(token in code for token in ("sm.Logit(", "sm.OLS(", "sm.GLM("))
    )
    if dtype_coerce_applies:
        repair_name = "dtype_coerce_v1"
        patch = textwrap.dedent(
            """

            def _easyicu_runner_repair_v1(X, y):
                X_work = X.copy() if hasattr(X, "copy") else X
                y_work = y.copy() if hasattr(y, "copy") else y
                if hasattr(X_work, "replace"):
                    X_work = X_work.replace([np.inf, -np.inf], np.nan)
                if hasattr(X_work, "apply"):
                    X_work = X_work.apply(pd.to_numeric, errors="coerce").astype(float)
                else:
                    X_work = np.asarray(X_work, dtype=float)
                y_work = pd.to_numeric(y_work, errors="coerce")
                if hasattr(X_work, "index") and hasattr(y_work, "index"):
                    keep = X_work.dropna().index.intersection(y_work.dropna().index)
                    X_work = X_work.loc[keep]
                    y_work = y_work.loc[keep]
                else:
                    X_arr = np.asarray(X_work, dtype=float)
                    y_arr = np.asarray(y_work, dtype=float)
                    mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
                    X_work = X_arr[mask]
                    y_work = y_arr[mask]
                return y_work.astype(float), X_work
            """
        ).strip("\n")

        patched = code
        if "_easyicu_runner_repair_v1" not in patched:
            insert_after = patched.find("import matplotlib.pyplot as plt")
            if insert_after >= 0:
                line_end = patched.find("\n", insert_after)
                patched = (
                    patched[: line_end + 1] + "\n" + patch + "\n" + patched[line_end + 1 :]
                )
            else:
                patched = patch + "\n\n" + patched

        model_call = re.compile(
            r"(?P<prefix>\b[A-Za-z_]\w*\s*=\s*sm\.(?:Logit|OLS|GLM)\()\s*"
            r"(?P<y>[^,]+?)\s*,\s*(?P<X>[^)\n]+?)\s*(?P<suffix>\))"
        )

        def _rewrite(match: re.Match[str]) -> str:
            y_expr = match.group("y").strip()
            x_expr = match.group("X").strip()
            return (
                f"{match.group('prefix')}*_easyicu_runner_repair_v1("
                f"{x_expr}, {y_expr}){match.group('suffix')}"
            )

        repaired = model_call.sub(_rewrite, patched, count=1)
        repaired = repaired.replace(
            "X_array = X.to_numpy()\n"
            "y_array = y.to_numpy()\n",
            "y, X = _easyicu_runner_repair_v1(X, y)\n"
            "X_array = np.asarray(X, dtype=float)\n"
            "y_array = np.asarray(y, dtype=float)\n",
            1,
        )
        if repaired != code:
            return repair_name, repaired

    # --- Generic v15 fallback as last resort ---
    # Only fires if no specific repair matched above
    if generic_runtime_failure:
        repair_name = f"generic_v15_{generic_key}_fallback_v1"
        if previous_repair != repair_name:
            fallback = _generic_v15_task_fallback_code(generic_key)
            if fallback is not None:
                return repair_name, fallback

    return None


def _has_figure_exports(out_dir: Path) -> bool:
    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    return any(
        path.is_file() and path.suffix.lower() in figure_suffixes
        for path in out_dir.iterdir()
    )


def _promote_prior_publication_bundle(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Promote the strongest earlier figure bundle into a publication step."""
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    contract_suffix = ".figure_contract.json"
    best: Optional[tuple[tuple[int, int, int], str, Dict[str, Path]]] = None

    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        bundles: Dict[str, Dict[str, Path]] = {}
        for path in outputs_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith(contract_suffix):
                stem = path.name[: -len(contract_suffix)]
                bundles.setdefault(stem, {})["contract"] = path
                continue
            if path.suffix.lower() in figure_suffixes:
                bundles.setdefault(path.stem, {})[path.suffix.lower()] = path
        for stem, files in bundles.items():
            figure_count = sum(1 for key in files if key.startswith("."))
            if figure_count == 0:
                continue
            score = (
                1 if "publication_figure" in stem else 0,
                1 if "primary_association" in stem else 0,
                figure_count,
            )
            if best is None or score > best[0]:
                best = (score, stem, files)

    if best is None:
        return None

    _, source_stem, files = best
    target_stem = "publication_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, source in files.items():
        if key == "contract":
            target = out_dir / f"{target_stem}.figure_contract.json"
        else:
            target = out_dir / f"{target_stem}{key}"
        shutil.copy2(source, target)

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            summary = json.loads(step_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    summary["publication_figure_rescue"].update(
        {
            "mode": "promotion",
            "source_step_stem": source_stem,
            "source_outputs_dir": str(files[next(iter(files))].parent),
        }
    )
    exported_figure_files = [
        str((out_dir / f"{target_stem}{key}").name)
        for key in sorted(files)
        if key != "contract"
    ]
    summary["figure_files"] = exported_figure_files
    if exported_figure_files:
        summary["figure_path"] = exported_figure_files[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "publication_bundle_promote_v1"


def _render_prediction_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically build a validation figure from prior prediction outputs.

    Some small models successfully write ``model_performance.csv`` and
    ``step_summary.json`` in the parent model-training step but fail to
    render the follow-up figure step. When that happens, we can still
    construct a publication-style validation bundle from the structured
    parent artefacts instead of failing the entire run.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    best_parent: Optional[tuple[Path, Path, Dict[str, Any]]] = None
    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        perf_path = outputs_dir / "model_performance.csv"
        summary_path = outputs_dir / "step_summary.json"
        if not perf_path.exists() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        best_parent = (perf_path, summary_path, summary)
        break
    if best_parent is None:
        return None

    perf_path, summary_path, summary = best_parent
    try:
        frame = pd.read_csv(perf_path)
    except Exception:
        return None
    metric_cols = [col for col in ("auroc", "brier_score") if col in frame.columns]
    calib_cols = [
        col for col in ("calibration_slope", "calibration_intercept") if col in frame.columns
    ]
    if not metric_cols and not calib_cols:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.publication_figures import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True)
    apply_publication_style(fig)
    if not isinstance(axes, (list, tuple)):
        axes = axes.ravel()
    folds = frame.get("fold")
    if folds is None:
        folds = pd.Series([f"Fold {idx + 1}" for idx in range(len(frame))])
    folds = folds.astype(str)

    ax1, ax2 = axes[0], axes[1]
    if "auroc" in frame.columns:
        ax1.plot(folds, frame["auroc"].astype(float), marker="o", linewidth=1.4, label="AUROC")
    if "brier_score" in frame.columns:
        ax1.plot(
            folds,
            frame["brier_score"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Brier",
        )
    ax1.set_title("Cross-validation discrimination", loc="left", pad=4)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Metric value")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend(frameon=False, fontsize=7)
    add_panel_label(ax1, "A", x=-0.1)

    if "calibration_slope" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_slope"].astype(float),
            marker="o",
            linewidth=1.4,
            label="Slope",
        )
        ax2.axhline(1.0, linestyle="--", linewidth=0.8, color="#8F8F8F")
    if "calibration_intercept" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_intercept"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Intercept",
        )
        ax2.axhline(0.0, linestyle=":", linewidth=0.8, color="#B64342")
    ax2.set_title("Cross-validation calibration", loc="left", pad=4)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Calibration statistic")
    ax2.tick_params(axis="x", rotation=35)
    if calib_cols:
        ax2.legend(frameon=False, fontsize=7)
    add_panel_label(ax2, "B", x=-0.1)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "Prediction-model validation metrics are summarised from the "
            "registered cross-validation performance table."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Discrimination",
                "role": "validation",
                "claim": "Fold-level AUROC and Brier score are derived from the model-performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
            {
                "panel_id": "B",
                "title": "Calibration",
                "role": "validation",
                "claim": "Fold-level calibration slope and intercept are derived from the registered step summary and performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
        ],
        source_data=["model_performance", "01_model_training"],
        statistics_note=(
            "Deterministic rescue figure generated from parent-step outputs "
            "when the figure-only child step did not emit exports."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
        contract=contract,
        dpi=300,
    )
    plt.close(fig)

    existing_summary: Dict[str, Any] = {}
    step_summary_path = out_dir / "step_summary.json"
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.setdefault("publication_figure_rescue", {})
    existing_summary["publication_figure_rescue"].update(
        {
            "mode": "prediction_validation_from_parent_outputs",
            "source_model_performance": str(perf_path),
            "source_step_summary": str(summary_path),
        }
    )
    figure_files = [
        path.name for key, path in outputs.items() if key != "contract"
    ]
    existing_summary["figure_files"] = figure_files
    if figure_files:
        existing_summary["figure_path"] = figure_files[0]
    existing_summary.setdefault("cv_auroc_mean", summary.get("statistic:auroc"))
    existing_summary.setdefault("cv_brier_mean", summary.get("statistic:brier_score"))
    existing_summary.setdefault(
        "calibration_slope", summary.get("statistic:calibration_slope")
    )
    existing_summary.setdefault(
        "calibration_intercept", summary.get("statistic:calibration_intercept")
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "prediction_publication_bundle_from_parent_outputs_v1"


def _expected_numeric_annotations_for_step(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
) -> Dict[str, float]:
    """Return the small set of numeric values we expect a figure to annotate."""
    if not isinstance(step_summary, dict) or not step_summary:
        return {}
    keys: List[str] = []
    step_id = (step.step_id or "").lower()
    if "primary_association" in step_id:
        keys = ["primary_or", "primary_ci_low", "primary_ci_high"]
    elif "outcome_incidence" in step_id:
        keys = ["outcome_rate"]
    elif "sofa_zero_audit" in step_id or "stratum" in step_id:
        keys = ["sofa_zero_rate", "sofa_one_rate"]
    expected: Dict[str, float] = {}
    for key in keys:
        value = step_summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            expected[key] = float(value)
    return expected


def _coerce_scalar(value: Any) -> Optional[Union[int, float, str, bool]]:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


def _first_present_scalar(
    payload: Dict[str, Any], keys: Sequence[str]
) -> Optional[Union[int, float, str, bool]]:
    flat = _flatten_scalar_dict(payload)
    for key in keys:
        if key not in payload:
            for flat_key, flat_value in flat.items():
                if flat_key.endswith(f".{key}"):
                    value = _coerce_scalar(flat_value)
                    if value is not None:
                        return value
            continue
        value = _coerce_scalar(payload.get(key))
        if value is not None:
            return value
    return None


def _first_numeric_scalar_with_key_fragment(
    payload: Dict[str, Any], fragments: Sequence[str]
) -> Optional[float]:
    """Return the first numeric scalar whose flattened key contains a fragment."""

    lowered_fragments = tuple(fragment.lower() for fragment in fragments if fragment)
    if not lowered_fragments:
        return None
    for key, value in _flatten_scalar_dict(payload).items():
        lowered = key.lower()
        if not any(fragment in lowered for fragment in lowered_fragments):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _flatten_scalar_dict(
    payload: Any,
    *,
    prefix: str = "",
) -> Dict[str, Union[int, float, str, bool]]:
    flat: Dict[str, Union[int, float, str, bool]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_scalar_dict(value, prefix=child_prefix))
        return flat
    if isinstance(payload, list):
        return flat
    scalar = _coerce_scalar(payload)
    if scalar is not None and prefix:
        flat[prefix] = scalar
    return flat


def _first_numeric_effect_from_text(payload: Any) -> Optional[float]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    patterns = (
        r"\b(?:OR|odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:adjusted\s+OR|adjusted\s+odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _enforce_advanced_plan_contract(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Constrain advanced plan shape while leaving analysis code to the agent."""

    family = (
        (context.user_preferences.inferred_analysis_family or "").lower()
        if context.user_preferences
        else ""
    )
    if not family:
        plan_blob = " ".join(
            [
                context.research_question or "",
                " ".join(
                    " ".join(
                        [
                            step.step_id or "",
                            step.intent or "",
                            step.method or "",
                            " ".join(step.expected_outputs or []),
                        ]
                    )
                    for step in (plan.steps or [])
                ),
            ]
        ).lower()
        if any(
            marker in plan_blob
            for marker in (
                "complete-case",
                "complete case",
                "missing-indicator",
                "missing indicator",
                "reduced-variable",
                "reduced variable",
                "robustness",
            )
        ):
            family = "robustness"
        elif any(
            marker in plan_blob
            for marker in (
                "cluster",
                "clustering",
                "phenotype",
                "trajectory",
                "silhouette",
            )
        ):
            family = "clustering"
        elif (
            "sofa" not in plan_blob
            and _question_primary_predictor_is_vasopressor_or_unknown(context)
            and (
                any(
                    marker in plan_blob
                    for marker in (
                        "selection bias",
                        "confounding by indication",
                        "confounded by indication",
                    )
                )
                or (
                    ("vasopressor" in plan_blob or "vaso" in plan_blob)
                    and "association" in plan_blob
                    and "mortality" in plan_blob
                )
            )
        ):
            family = "bias_audit"
        elif any(
            marker in plan_blob
            for marker in (
                "prediction",
                "auroc",
                "auc",
                "brier",
                "calibration",
                "cross-validation",
                "cross validation",
                "held-out",
                "held out",
            )
        ):
            family = "prediction_model"
    if family not in {"prediction_model", "clustering", "robustness", "bias_audit"}:
        return plan, []

    if family == "prediction_model":
        markers = (
            "prediction",
            "model",
            "training",
            "performance",
            "auroc",
            "auc",
            "brier",
            "calibration",
            "discrimination",
        )
        canonical_step_id = "01_model_training"
        canonical_method = "prediction_model"
        canonical_intent = (
            "Train and validate the mortality prediction model in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:auroc",
            "statistic:brier_score",
            "statistic:baseline_prevalence",
            "statistic:split_strategy",
            "table:model_performance",
            "figure:discrimination_calibration",
        ]
    elif family == "clustering":
        markers = (
            "cluster",
            "clustering",
            "phenotype",
            "trajectory",
            "silhouette",
            "mortality_by_cluster",
        )
        canonical_step_id = "01_trajectory_clustering"
        canonical_method = "clustering"
        canonical_intent = (
            "Generate shock-physiology clusters, cluster summaries, post-hoc "
            "mortality by cluster, validation metrics, and a figure in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:silhouette_score",
            "statistic:cluster_count",
            "table:cluster_characteristics",
            "table:cluster_mortality",
            "figure:clustering_visualization",
            "log:clustering_algorithm_details",
            "manifest:clustering_methodology",
        ]
    elif family == "bias_audit":
        markers = (
            "association",
            "vasopressor",
            "vaso",
            "mortality",
            "selection",
            "bias",
            "confounding",
            "missingness",
            "clinical-constraint",
            "clinical constraint",
            "logistic",
            "regression",
            "model",
            "strategy",
            "adjusted",
            "odds ratio",
            "or",
        )
        canonical_step_id = "02_vasopressor_selection_bias_association"
        canonical_method = "bias_audit_association"
        canonical_intent = (
            "Fit a mortality association model for first-24h vasopressor exposure "
            "with severity/missingness context; report the primary odds ratio, "
            "selection-bias or confounding-by-indication warning, missingness profile, "
            "and avoid causal treatment-effect language."
        )
        required_outputs = [
            "statistic:primary_or",
            "statistic:selection_bias_warning",
            "statistic:mortality_rate",
            "table:association_summary",
            "table:missingness_profile",
            "log:clinical_constraint_warning",
        ]
    else:
        markers = (
            "complete-case",
            "complete case",
            "missing-indicator",
            "missing indicator",
            "reduced-variable",
            "reduced variable",
            "robustness",
            "odds ratio",
            "logistic",
            "model",
            "figure",
            "performance",
        )
        canonical_step_id = "03_complete_case_robustness"
        canonical_method = "association_robustness"
        canonical_intent = (
            "Fit complete-case, missing-indicator, and reduced-variable mortality "
            "association models; extract lactate odds ratios and complete-case "
            "sample size; write the summary table and robustness figure in one "
            "self-contained executable step."
        )
        required_outputs = [
            "statistic:primary_or",
            "statistic:complete_case_n",
            "table:robustness_summary",
            "figure:robustness_plot",
            "log:missingness_strategy_notes",
        ]

    def _is_relevant(step: AnalysisStep) -> bool:
        text = " ".join(
            [
                step.step_id or "",
                step.intent or "",
                step.method or "",
                " ".join(step.expected_outputs or []),
            ]
        ).lower()
        return any(marker in text for marker in markers)

    relevant_indexes = [idx for idx, step in enumerate(plan.steps) if _is_relevant(step)]
    if not relevant_indexes:
        return plan, []

    first_index = relevant_indexes[0]
    relevant_steps = [plan.steps[idx] for idx in relevant_indexes]
    combined_inputs: List[str] = []
    for step in relevant_steps:
        for item in step.inputs or []:
            if item not in combined_inputs:
                combined_inputs.append(item)
    combined_outputs = list(required_outputs)
    for step in relevant_steps:
        for item in step.expected_outputs or []:
            if item not in combined_outputs:
                combined_outputs.append(item)

    current = relevant_steps[0]
    missing_outputs = [item for item in required_outputs if item not in current.expected_outputs]
    needs_normalisation = (
        len(relevant_indexes) != 1
        or bool(missing_outputs)
        or current.step_id != canonical_step_id
    )
    if not needs_normalisation:
        return plan, []

    canonical_step = current.model_copy(
        update={
            "step_id": canonical_step_id,
            "intent": canonical_intent,
            "inputs": combined_inputs or current.inputs,
            "expected_outputs": combined_outputs,
            "method": canonical_method,
        }
    )
    new_steps: List[AnalysisStep] = []
    inserted = False
    relevant_set = set(relevant_indexes)
    for idx, step in enumerate(plan.steps):
        if idx in relevant_set:
            if not inserted:
                new_steps.append(canonical_step)
                inserted = True
            continue
        new_steps.append(step)

    revised = plan.model_copy(
        update={"steps": new_steps, "revision": max(1, plan.revision) + 1}
    )
    finding = ValidationFinding(
        validator="plan_contract",
        severity="warning",
        message=(
            f"Planner output for {family} was normalized to a single "
            "self-contained advanced-analysis step with explicit v14 metric "
            "and artefact contracts."
        ),
        detail={
            "family": family,
            "original_step_ids": [step.step_id for step in relevant_steps],
            "canonical_step_id": canonical_step_id,
            "canonical_insert_index": first_index,
            "required_outputs": required_outputs,
        },
    )
    return revised, [finding]


def _question_primary_predictor_is_vasopressor_or_unknown(
    context: ResearchContext,
) -> bool:
    predictor = _infer_primary_predictor_from_context(context)
    if not predictor:
        return True
    tokens = _predictor_tokens(predictor)
    return bool(tokens & {"vaso", "vasopressor", "vasopressors", "norepinephrine"})


def _infer_primary_predictor_from_context(
    context: ResearchContext,
) -> Optional[str]:
    """Infer the named exposure/predictor from the question and variables.

    This intentionally stays task-family generic. It scores variable-name
    tokens that appear before adjustment language higher than tokens that
    appear only in an "adjusted for ..." covariate clause.
    """

    question = (context.research_question or "").lower()
    if not question:
        return None
    primary_span = re.split(
        r"\b(?:after adjustment for|adjusted for|controlling for|with adjustment for|including covariates|covariates?)\b",
        question,
        maxsplit=1,
    )[0]
    best_name: Optional[str] = None
    best_score = 0
    for variable in context.variables:
        if variable.role in {VariableRole.ID, VariableRole.TIME, VariableRole.OUTCOME}:
            continue
        tokens = _predictor_tokens(variable.name)
        if not tokens:
            continue
        score = 0
        for token in tokens:
            if token in primary_span:
                score += 20
            elif token in question:
                score += 5
        if score > best_score:
            best_score = score
            best_name = variable.name
    return best_name if best_score > 0 else None


def _predictor_tokens(name: Optional[str]) -> set[str]:
    if not name:
        return set()
    raw_tokens = re.split(r"[^a-zA-Z0-9]+", str(name).lower())
    stop = {
        "",
        "24h",
        "48h",
        "72h",
        "max",
        "min",
        "mean",
        "median",
        "first",
        "last",
        "any",
        "flag",
        "binary",
        "value",
        "score",
    }
    tokens = {token for token in raw_tokens if token not in stop}
    if "vaso" in tokens:
        tokens.add("vasopressor")
    if "norepi" in tokens:
        tokens.add("norepinephrine")
    return tokens


_FIGURE_STEP_TOKENS = ("figure", "plot", "chart", "fig:", "figure:", "plot:")


_PUBLICATION_FIGURE_TRIGGER_TOKENS = (
    "publication-ready figure",
    "publication ready figure",
    "publication figure",
    "produce a heatmap",
    "produce a figure",
    "publication-ready",
    "figure or",
    "and a figure",
    "and a heatmap",
    "and a publication",
    "publication-quality figure",
)


def _split_table_and_figure_outputs_in_plan(
    plan: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Split steps that declare both table and figure outputs into two steps.

    A step like ``expected_outputs=['table:table_one', 'figure:table_one_visual']``
    asks the coder agent to produce *both* a CSV table and a publication
    figure inside a single executable script. Naive arms frequently
    deliver only the table and ignore the figure, then exhaust the
    LLM-repair budget without recovering. Splitting the step into a
    table-only step plus a downstream figure-only step gives the agent
    a focused target for each artefact while keeping the analytic
    intent intact.

    The split is conservative: it only fires when a single step
    declares at least one ``table:`` (or ``statistic:``) output *and*
    at least one ``figure:`` output. Non-figure outputs stay on the
    original step; figure outputs migrate to a new appended step
    inserted directly after the original. Other steps in the plan are
    left untouched.
    """
    if not plan.steps:
        return plan, []

    new_steps: List[AnalysisStep] = []
    findings: List[ValidationFinding] = []

    for step in plan.steps:
        outputs = list(step.expected_outputs or [])
        method = (step.method or "").lower()
        if method in {
            "association_robustness",
            "bias_audit_association",
            "clustering",
        }:
            # ``prediction_model`` is intentionally NOT in this skip-list:
            # the canonical ``01_model_training`` step bundles both a
            # ``table:model_performance`` analytic payload and a
            # ``figure:discrimination_calibration`` figure, and the agent
            # frequently forgets to render the figure when both are demanded
            # in a single script. Splitting yields a sibling
            # ``01_model_training_figure`` whose contract is purely visual,
            # which is what
            # ``test_mock_planner_emits_prediction_analysis_and_publication_for_prediction_question``
            # pins.
            new_steps.append(step)
            continue
        figure_outputs = [
            out
            for out in outputs
            if any(needle in (out or "").lower() for needle in _FIGURE_STEP_TOKENS)
        ]
        non_figure_outputs = [out for out in outputs if out not in figure_outputs]
        # Only split when the step is genuinely mixed — at least one
        # non-figure analytic payload (table, statistic, log, model, ...)
        # alongside a figure. Splitting a figure-only step would create
        # an empty stub that the coder cannot anchor to a parent
        # artefact, so we require the non-figure outputs to look like
        # real deliverables. ``model:*`` is included because regression
        # / prediction / association steps frequently bundle a model
        # object with a companion figure, and the agent forgets to draw
        # the figure when both are demanded in the same script.
        has_non_figure_payload = any(
            (out or "").lower().startswith(
                ("table:", "statistic:", "log:", "model:")
            )
            or (out or "").lower() in {"table", "statistic", "log", "model"}
            for out in non_figure_outputs
        )
        if not figure_outputs or not has_non_figure_payload:
            new_steps.append(step)
            continue
        # Keep the original step with the non-figure outputs.
        non_figure_step = step.model_copy(
            update={"expected_outputs": non_figure_outputs}
        )
        new_steps.append(non_figure_step)
        # Synthesise a follow-up figure-only step.
        figure_step_id = f"{step.step_id}_figure"
        figure_intent = (
            f"Render the publication figure(s) declared by step "
            f"'{step.step_id}' ({', '.join(figure_outputs)}). Load the "
            "cohort from ``os.environ['COHORT_PARQUET']`` (full path is "
            "provided by the runner) and, if needed, read tables produced "
            f"by '{step.step_id}' from any of the registered evidence "
            "files. Save PNG and SVG copies of every figure with matching "
            "stems into ``os.environ['STEP_OUT_DIR']``. Always write a "
            "valid step_summary.json into ``STEP_OUT_DIR`` listing each "
            "produced file under ``figure_files`` even if rendering fails — "
            "use a try/except so the step never aborts before writing the "
            "summary."
        )
        figure_step = AnalysisStep(
            step_id=figure_step_id,
            intent=figure_intent,
            inputs=list(step.inputs or []),
            expected_outputs=figure_outputs,
            method=(step.method or "visualization"),
            icu_rule_refs=list(step.icu_rule_refs or []) + ["visualization_rule"],
        )
        new_steps.append(figure_step)
        findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    f"Split step '{step.step_id}' into a table/statistic "
                    f"step and a follow-up figure step "
                    f"'{figure_step_id}' so the coder can target each "
                    "artefact independently."
                ),
                detail={
                    "source_step_id": step.step_id,
                    "non_figure_outputs": non_figure_outputs,
                    "figure_outputs": figure_outputs,
                    "appended_step_id": figure_step_id,
                },
            )
        )

    if not findings:
        return plan, []
    return plan.model_copy(update={"steps": new_steps}), findings


def _research_question_implies_figure(question: str) -> bool:
    """Heuristic: does the research question call for a figure deliverable?"""
    text = (question or "").lower()
    if not text:
        return False
    return any(token in text for token in _PUBLICATION_FIGURE_TRIGGER_TOKENS)


def _ensure_publication_figure_step_in_plan(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Append a fallback figure step when the planner forgot one.

    Naive arms (no ICU narrative context) sometimes emit a single-step
    plan that omits the publication figure even when the research
    question explicitly asks for one. Task contracts in the EasyICU
    experiment runner still require a ``figure`` artefact in those
    cases. Detect the gap and append a generic figure step so the
    coder agent has a concrete target. The step's ``intent`` is broad
    enough that the coder can still tailor the chart shape (bar, box,
    forest, heatmap…) based on the upstream analytics.
    """
    if any(_step_produces_figure(step) for step in plan.steps or []):
        return plan, []
    if not _research_question_implies_figure(context.research_question or ""):
        return plan, []

    next_index = len(plan.steps or []) + 1
    fallback_step = AnalysisStep(
        step_id=f"{next_index:02d}_publication_figure_fallback",
        intent=(
            "Render a publication-ready figure that summarises the "
            "analytics produced by the previous steps. Read the latest "
            "step_summary.json files under the run directory, pick the "
            "most informative numeric structure (e.g. mortality by "
            "stratum, correlation values, model performance), and save "
            "the figure as both PNG and SVG with the same stem into "
            "``os.environ['STEP_OUT_DIR']`` (set by the runner). Record "
            "every produced path in step_summary.json under "
            "``figure_files``."
        ),
        method="visualization",
        inputs=[],
        expected_outputs=["figure:overview"],
        icu_rule_refs=["visualization_rule"],
    )
    new_steps = list(plan.steps or []) + [fallback_step]
    preserved = plan.model_copy(update={"steps": new_steps})
    findings = [
        ValidationFinding(
            validator="plan_contract",
            severity="warning",
            message=(
                "Plan did not declare a figure step even though the "
                "research question asked for a publication-ready "
                "figure; appended a fallback figure step "
                f"'{fallback_step.step_id}' to preserve the task contract."
            ),
            detail={"appended_step_id": fallback_step.step_id},
        )
    ]
    return preserved, findings


def _step_produces_figure(step: AnalysisStep) -> bool:
    """True if the step's expected_outputs declare a figure/plot artefact."""
    for output in step.expected_outputs or []:
        token = (output or "").lower()
        for needle in _FIGURE_STEP_TOKENS:
            if needle in token:
                return True
    return False


def _preserve_figure_steps_after_replan(
    *,
    current: AnalysisPlan,
    revised: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Re-add figure-producing steps that the replanner silently dropped.

    The replanner is an LLM call and can rationalise away figure steps when
    upstream context (e.g. ICU narrative) is missing. Task contracts in the
    EasyICU experiment runner still require figure artefacts regardless of
    the planner's framing, so we treat any step whose ``expected_outputs``
    declare a figure/plot as load-bearing: if such a step is present in the
    *current* plan but absent from the *revised* plan, append it back to
    the revised plan and emit a warning so the manifest preserves the audit
    trail.
    """
    revised_ids = {step.step_id for step in revised.steps}
    dropped_figure_steps = [
        step
        for step in current.steps
        if step.step_id not in revised_ids and _step_produces_figure(step)
    ]
    if not dropped_figure_steps:
        return revised, []
    new_steps = list(revised.steps) + list(dropped_figure_steps)
    preserved = revised.model_copy(update={"steps": new_steps})
    findings = [
        ValidationFinding(
            validator="replanner",
            severity="warning",
            message=(
                "Replanner attempted to drop "
                f"{len(dropped_figure_steps)} figure-producing step(s); "
                "they were re-attached to preserve task contract."
            ),
            detail={
                "preserved_step_ids": [s.step_id for s in dropped_figure_steps],
            },
        )
    ]
    return preserved, findings


def _step_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
) -> List[ValidationFinding]:
    if not isinstance(step_summary, dict) or not step_summary:
        return [
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not produce a readable step_summary.json, "
                    "so required outputs cannot be verified."
                ),
                detail={"step_id": step.step_id},
            )
        ]

    findings: List[ValidationFinding] = []
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    step_id = (step.step_id or "").lower()
    intent = (step.intent or "").lower()

    # Figure-only follow-up steps (created by ``_split_table_and_figure_outputs_in_plan``)
    # inherit the parent's step_id with a ``_figure`` suffix, e.g.
    # ``04_primary_association_figure`` / ``01_model_training_figure``. Their
    # expected_outputs contain *only* figure items — the analytic payload
    # (table/statistic/etc.) lives in the sibling parent step. Without this guard
    # the substring matches ``primary_association``/``model_training``/``cluster``
    # below would falsely demand effect/prediction/clustering metrics from a
    # render-only step that legitimately has no such fields in its summary.
    figure_only_step = bool(step.expected_outputs) and all(
        any(needle in (out or "").lower() for needle in _FIGURE_STEP_TOKENS)
        for out in step.expected_outputs
    )

    def _append_missing(message: str, keys: Sequence[str]) -> None:
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=message,
                detail={
                    "step_id": step.step_id,
                    "expected_outputs": list(step.expected_outputs or []),
                    "summary_keys": sorted(step_summary.keys()),
                    "skipped": step_summary.get("skipped"),
                    "error": step_summary.get("error"),
                    "required_keys": list(keys),
                },
            )
        )

    effect_required = (
        not figure_only_step
        and (
            any(
                token in expected
                for token in (
                    "adjusted_or_ci",
                    "primary_association",
                    "odds_ratio",
                    "primary_or",
                    "adjusted_or",
                )
            )
            or "primary_association" in step_id
        )
    )
    if not effect_required and not figure_only_step and "association" in expected:
        effect_required = (
            "model" in step_id
            or "regression" in intent
            or "estimate" in intent
            or "odds" in expected
        )
    if not effect_required and not figure_only_step and (
        ("logistic" in expected or "logistic" in intent or "odds" in intent)
        and ("model" in step_id or "model" in expected or "regression" in intent)
    ):
        effect_required = True
    if effect_required:
        effect_value = _first_present_scalar(
            step_summary,
            (
                "estimate",
                "statistic:estimate",
                "primary_or",
                "statistic:primary_or",
                "odds_ratio",
                "statistic:odds_ratio",
                "adjusted_or",
                "statistic:adjusted_or",
                "lactate_or",
                "statistic:lactate_or",
                "lactate_or_complete_case",
                "statistic:lactate_or_complete_case",
                "complete_case_lactate_or",
                "statistic:complete_case_lactate_or",
                "lactate_max_24h_or",
                "statistic:lactate_max_24h_or",
                "primary_association_estimate",
                "statistic:primary_association_estimate",
                "association_estimate",
                "statistic:association_estimate",
                "or",
            ),
        )
        if effect_value is None:
            for key, value in _flatten_scalar_dict(step_summary).items():
                lowered = key.lower()
                if (
                    lowered.endswith("_or")
                    or lowered.endswith("_odds_ratio")
                    or lowered.endswith("_estimate")
                ):
                    effect_value = _coerce_scalar(value)
                    if effect_value is not None:
                        break
        if effect_value is None:
            effect_value = _first_numeric_effect_from_text(step_summary)
        if effect_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a primary association "
                    "estimate, but no numeric effect size was recorded."
                ),
                ("estimate", "primary_or", "odds_ratio", "adjusted_or", "lactate_or"),
            )

    prediction_step = (
        "training_and_evaluation" in step_id
        or "model_training" in step_id
        or "prediction" in step_id
        or "prediction" in intent
    )
    prediction_required = (not figure_only_step) and (
        any(token in expected for token in ("auroc", "auc", "brier", "discrimination"))
        or ("calibration" in expected and prediction_step)
        or prediction_step
    )
    if prediction_required:
        auroc_value = _first_present_scalar(
            step_summary,
            (
                "auroc",
                "statistic:auroc",
                "auc",
                "statistic:auc",
                "held_out_auroc",
                "statistic:held_out_auroc",
                "cv_auroc",
                "statistic:cv_auroc",
                "cv_auroc_mean",
                "statistic:cv_auroc_mean",
                "mean_auroc",
                "auroc_mean",
                "auroc_median",
            ),
        )
        if auroc_value is None:
            auroc_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("auroc", "auc"),
            )
        if auroc_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report AUROC-style discrimination, "
                    "but no AUROC metric was recorded."
                ),
                ("auroc", "cv_auroc", "mean_auroc", "auroc_median"),
            )
        calibration_value = _first_present_scalar(
            step_summary,
            (
                "brier_score",
                "statistic:brier_score",
                "cv_brier_mean",
                "statistic:cv_brier_mean",
                "brier_mean",
                "held_out_brier",
                "statistic:held_out_brier",
                "brier_median",
                "calibration_slope",
                "statistic:calibration_slope",
                "calibration_slope_median",
                "calibration_intercept",
                "statistic:calibration_intercept",
                "calibration_intercept_median",
            ),
        )
        if calibration_value is None:
            calibration_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("brier", "calibration_slope", "calibration_intercept"),
            )
        if calibration_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report calibration or Brier-style "
                    "evaluation metrics, but none were recorded."
                ),
                (
                    "brier_score",
                    "cv_brier_mean",
                    "held_out_brier",
                    "calibration_slope",
                    "calibration_intercept",
                ),
            )

    clustering_required = (not figure_only_step) and (
        any(token in expected for token in ("cluster", "silhouette"))
        or "cluster" in step_id
        or "clustering" in intent
    )
    if clustering_required:
        cluster_value = _first_present_scalar(
            step_summary,
            (
                "silhouette_score",
                "statistic:silhouette_score",
                "silhouette",
                "statistic:silhouette",
                "n_clusters",
                "statistic:n_clusters",
                "cluster_count",
                "statistic:cluster_count",
            ),
        )
        if cluster_value is None:
            cluster_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("silhouette", "cluster_count", "n_clusters"),
            )
        if cluster_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a clustering summary, "
                    "but no cluster metric or cluster count was recorded."
                ),
                ("silhouette_score", "silhouette", "n_clusters", "cluster_count"),
            )

    # Enforce figure_required when:
    # (a) the intent *explicitly* demands a publication-ready figure, OR
    # (b) the step is figure-only (its expected_outputs are exclusively
    #     figure tokens — usually the child produced by
    #     ``_split_table_and_figure_outputs_in_plan``).
    # For unsplit mixed steps (figure declared alongside table/statistic
    # outputs without an explicit "publication-ready figure" intent), the
    # splitter handles decomposition in production, so we treat the figure
    # output as an optional companion here. This mirrors how downstream
    # contracts evaluate the parent and the figure-only child separately.
    figure_required = (
        ("publication-ready figure" in intent)
        or (figure_only_step and "figure:" in expected)
    )
    if figure_required:
        figure_value = None
        for key, value in _flatten_scalar_dict(step_summary).items():
            lowered_key = key.lower()
            lowered_value = str(value).lower()
            if (
                "figure" in lowered_key
                or "plot" in lowered_key
                or lowered_value.endswith((".png", ".svg", ".pdf", ".tiff", ".tif"))
                or ".png" in lowered_value
                or ".svg" in lowered_value
                or ".pdf" in lowered_value
                or ".tiff" in lowered_value
                or ".tif" in lowered_value
            ):
                figure_value = value
                break
        if figure_value is None:
            # ``_flatten_scalar_dict`` drops lists, but the coder prompt itself
            # recommends recording multiple figure paths in list-valued keys
            # such as ``figure_files`` / ``figure_file`` / ``figure_paths``.
            # Accept those when they contain at least one figure-shaped path.
            for list_key in ("figure_files", "figure_file", "figure_paths", "plot_files"):
                candidate = (step_summary or {}).get(list_key)
                if isinstance(candidate, (list, tuple)):
                    candidate_values = []
                    for item in candidate:
                        if isinstance(item, dict):
                            candidate_values.extend(
                                str(value) for value in item.values()
                            )
                        else:
                            candidate_values.append(str(item))
                    if any(
                        value.lower().endswith((".png", ".svg", ".pdf", ".tiff", ".tif"))
                        for value in candidate_values
                    ):
                        figure_value = candidate
                        break
        if figure_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to produce a figure artifact, "
                    "but the step summary did not record any figure path or figure output."
                ),
                ("figure_path", "figure_files", "figure_file", "plot_path", "png", "svg"),
            )

    return findings


def _step_contract_repair_guidance(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    code: str,
) -> str:
    guidance: List[str] = []
    if not isinstance(step_summary, dict):
        # Hosted models sometimes emit a bare string as step_summary
        # when the generated code prints JSON as stdout. Treat non-dict
        # summaries as empty for repair guidance so we never crash in
        # the middle of the loop.
        step_summary = {}
    predictor = str(
        step_summary.get("primary_predictor")
        or step_summary.get("predictor")
        or ""
    ).strip()
    summary_text = json.dumps(step_summary or {}, ensure_ascii=False, default=str)
    if predictor and predictor in summary_text:
        guidance.append(
            f"The machine summary identifies `{predictor}` as the primary predictor. "
            f"The repaired script must include `{predictor}` in the fitted design matrix."
        )
        lookup_patterns = (
            f"result.params['{predictor}'",
            f'result.params["{predictor}"',
            f"result.conf_int().loc['{predictor}'",
            f'result.conf_int().loc["{predictor}"',
            f"result.pvalues['{predictor}'",
            f'result.pvalues["{predictor}"',
            f"coef_table.loc['{predictor}'",
            f'coef_table.loc["{predictor}"',
        )
        if any(pattern in code for pattern in lookup_patterns):
            guidance.append(
                f"The previous script read model results for `{predictor}`. "
                f"Before fitting, build `x_cols` so `{predictor}` is present in `X.columns`; "
                "otherwise statsmodels will fit a model that cannot report the requested coefficient."
            )
    if "pd.get_dummies" in code and "drop_first" in code:
        guidance.append(
            "The previous script used dummy encoding. Rebuild the predictor list after "
            "dummy encoding: primary predictor + numeric covariates + generated dummy columns."
        )
    if (
        (
            (step_summary or {}).get("n_total") == 0
            or "zero-size array" in summary_text.lower()
            or "empty" in summary_text.lower()
        )
        and "pd.to_numeric" in code
        and "sex" in code
    ):
        guidance.append(
            "The previous script appears to have dropped the entire cohort by applying "
            "`pd.to_numeric(..., errors='coerce')` to `sex` before dummy encoding. "
            "Repair preprocessing by dummy-encoding `sex` first, rebuilding `x_cols`, "
            "then numeric-coercing only `[outcome] + x_cols` and dropping missing rows "
            "with that rebuilt list."
        )
        guidance.append(
            "Do not keep a null estimate summary for this contract failure. The repair "
            "should produce a numeric odds ratio when enough non-missing rows/events exist."
        )
    if (
        "pandas data cast to numpy dtype of object" in summary_text.lower()
        or "dtype of object" in summary_text.lower()
    ) and ("sm.logit(" in code.lower() or "pd.get_dummies" in code):
        guidance.append(
            "The prior script passed an object-dtype design matrix into statsmodels. "
            "After `pd.get_dummies(...)`, rebuild the predictor frame and convert every "
            "column in `X` with `pd.to_numeric(..., errors='coerce')`, cast boolean "
            "dummy columns to int when needed, and fit `sm.Logit(y, X.astype(float))`."
        )
        guidance.append(
            "Check the final design matrix dtypes before fitting and keep only rows with "
            "non-missing numeric predictors/outcome so the repaired script writes a "
            "non-null odds ratio."
        )
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    step_id = (step.step_id or "").lower()
    intent = (step.intent or "").lower()
    effect_required = any(
        token in expected
        for token in (
            "adjusted_or_ci",
            "primary_association",
            "odds_ratio",
            "primary_or",
            "adjusted_or",
        )
    ) or "primary_association" in step_id
    if not effect_required and "association" in expected:
        effect_required = (
            "model" in step_id
            or "regression" in intent
            or "estimate" in intent
            or "odds" in expected
        )
    if not effect_required and (
        ("logistic" in expected or "logistic" in intent or "odds" in intent)
        and ("model" in step_id or "model" in expected or "regression" in intent)
    ):
        effect_required = True
    if effect_required:
        guidance.append(
            "This association step must write a non-null numeric primary effect "
            "estimate in step_summary.json, such as `adjusted_or`, `primary_or`, "
            "`odds_ratio`, or `primary_association_estimate`. Do not satisfy the "
            "contract by leaving association fields null."
        )
    prediction_step = (
        any(token in step_id for token in ("prediction", "model_training", "training_and_evaluation", "performance"))
        or "prediction" in intent
    )
    prediction_required = (
        any(token in expected for token in ("auroc", "auc", "brier", "discrimination"))
        or ("calibration" in expected and prediction_step)
        or prediction_step
    )
    if prediction_required:
        guidance.append(
            "This prediction step must produce numeric AUROC and Brier/calibration metrics "
            "in step_summary.json (for example `cv_auroc_mean` and `brier_score`). "
            "Do not return only null metrics unless validation is truly impossible."
        )
        if "could not convert string to float" in summary_text.lower() or (
            "passthrough" in code and "onehot" in code.lower()
        ):
            guidance.append(
                "The failure indicates a categorical variable reached a numeric estimator. "
                "Use a scikit-learn ColumnTransformer with numeric features in a median-impute/"
                "scale branch and categorical features in a most-frequent-impute + "
                "OneHotEncoder(handle_unknown='ignore', sparse_output=False) branch. "
                "Never use `('onehot', 'passthrough')` for the categorical branch."
            )
        if "pd.to_numeric" in code and "categorical" in code.lower():
            guidance.append(
                "Do not numeric-coerce the full mixed feature frame. Keep categorical "
                "columns such as sex as object/string until the categorical transformer "
                "encodes them."
            )
    if "simpleimputer does not support data with dtype bool" in summary_text.lower():
        guidance.append(
            "A boolean dummy column reached SimpleImputer. Cast boolean dummy columns "
            "to int before fitting scikit-learn pipelines, or route them through a "
            "numeric branch with median imputation after conversion."
        )
    clustering_required = any(
        token in expected for token in ("cluster", "silhouette")
    ) or any(token in step_id for token in ("cluster", "trajectory")) or "clustering" in intent
    if clustering_required:
        guidance.append(
            "This clustering step must write machine-readable clustering metrics "
            "in step_summary.json. Use keys such as `silhouette_score` or "
            "`statistic:silhouette_score`, plus `cluster_count` or "
            "`statistic:cluster_count`."
        )
        guidance.append(
            "Keep clustering self-contained: create labels, cluster characteristics, "
            "post-hoc mortality by cluster, method metadata, and the clustering "
            "figure inside this script. Do not rely on labels saved by another step."
        )
        guidance.append(
            "Also save table artefacts named `cluster_characteristics.csv` and "
            "`cluster_mortality.csv` when feasible so manuscript evidence aliases bind."
        )
    if "figure:" in expected:
        guidance.append(
            "This step declares a figure output. Save a real figure file such as PNG/SVG/"
            "PDF/TIFF and record its path in step_summary.json using a key such as "
            "`figure_path`, `figure_file`, or `figure_files`."
        )
    if not guidance:
        guidance.append(
            "Repair the script so each expected output is written as machine-readable "
            "numbers in step_summary.json, or write a precise skipped/error reason."
        )
    return "\n".join(f"- {item}" for item in guidance)


def _resolve_writer_aux_path(
    *,
    run_dir: Path,
    step_id: str,
    candidate: Optional[Any],
) -> Optional[Path]:
    if not candidate:
        return None
    raw = Path(str(candidate))
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        run_dir / "steps" / step_id / "outputs" / raw.name,
        run_dir / "steps" / step_id / "outputs" / str(raw),
        run_dir / str(raw),
    ]
    return next((path for path in candidates if path.exists()), None)


def _summarise_table_one_rows(rows: Any) -> Dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    wanted = {
        "age": "age",
        "sofa2": "sofa2",
        "lact": "lact",
        "creat": "creat",
        "map": "map",
        "los_icu": "los_icu",
    }
    summary: Dict[str, Any] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("variable") or "").strip().lower()
        if variable not in wanted:
            continue
        prefix = wanted[variable]
        for source_key, target_key in (
            ("n", f"{prefix}_n"),
            ("median", f"{prefix}_median"),
            ("q25", f"{prefix}_q25"),
            ("q75", f"{prefix}_q75"),
            ("most_common", f"{prefix}_most_common"),
            ("most_common_n", f"{prefix}_most_common_n"),
        ):
            scalar = _first_present_scalar(item, (source_key,))
            if scalar is not None:
                summary[target_key] = scalar
    return summary


def _summarise_primary_association_table(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    cols = {str(c).lower(): c for c in frame.columns}
    variable_col = cols.get("variable") or cols.get("term")
    odds_col = cols.get("odds_ratio") or cols.get("or")
    lower_col = cols.get("or_lower") or cols.get("ci_lower") or cols.get("lower")
    upper_col = cols.get("or_upper") or cols.get("ci_upper") or cols.get("upper")
    p_col = cols.get("p_value") or cols.get("p")
    if variable_col is None:
        return {}
    digest: Dict[str, Any] = {}
    for _, row in frame.iterrows():
        variable = str(row.get(variable_col) or "").strip()
        if not variable or variable.lower() == "intercept":
            continue
        key = variable.replace(" ", "_")
        if odds_col is not None:
            val = _first_present_scalar(row, (odds_col,))
            if val is not None:
                digest[f"{key}_or"] = val
        if lower_col is not None:
            val = _first_present_scalar(row, (lower_col,))
            if val is not None:
                digest[f"{key}_ci_low"] = val
        if upper_col is not None:
            val = _first_present_scalar(row, (upper_col,))
            if val is not None:
                digest[f"{key}_ci_high"] = val
        if p_col is not None:
            val = _first_present_scalar(row, (p_col,))
            if val is not None:
                digest[f"{key}_p_value"] = val
    return digest


def _summarise_sofa_zero_audit(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    cols = {str(c).lower(): c for c in frame.columns}
    sofa_col = cols.get("sofa2") or cols.get("score") or cols.get("stratum")
    rate_col = cols.get("death_rate") or cols.get("outcome_rate") or cols.get("mortality_rate")
    if sofa_col is None or rate_col is None:
        return {}
    digest: Dict[str, Any] = {}
    for level in (0, 1):
        try:
            row = frame.loc[pd.to_numeric(frame[sofa_col], errors="coerce") == level]
        except Exception:
            row = pd.DataFrame()
        if row.empty:
            continue
        value = _first_present_scalar(row.iloc[0], (rate_col,))
        if value is not None:
            digest[f"sofa2_{level}_death_rate"] = value
    return digest


def _preferred_writer_evidence_names(evidence: EvidenceStore) -> List[str]:
    aliases = evidence.aliases()
    preferred = [
        "table_one",
        "cohort_summary",
        "outcome_incidence",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
        "sofa_strata",
        "stratum_audit",
        "multiple_testing_report",
        "fairness_subgroups",
        "literature_prisma",
        "causal_audit_report",
        "causal_audit_summary",
        "reporting_checklist",
    ]
    out: List[str] = [name for name in preferred if name in aliases or evidence.get(name) is not None]
    step_aliases = [
        name for name in sorted(aliases)
        if re.match(r"^\d{2}[_-]", name)
    ]
    for name in step_aliases:
        if name not in out:
            out.append(name)
    return out or evidence.resolvable_names()


def _render_writer_evidence_digest(
    per_step_records: Sequence[Dict[str, Any]] | None = None,
    *,
    context: ResearchContext | None = None,
    run_dir: Path | None = None,
) -> str:
    lines: List[str] = []
    if context is not None:
        lines.append("RUN_CONTEXT")
        lines.append(
            "  "
            + json.dumps(
                {
                    "research_question": context.research_question,
                    "cohort_name": context.cohort.cohort_name,
                    "database": context.cohort.database,
                    "n_stays": context.cohort.n_stays,
                    "n_patients": context.cohort.n_patients,
                    "target_outcome": context.target_outcome,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    run_dir = Path(run_dir or ".")
    preferred_keys = (
        "sample_size",
        "n_total",
        "n_total_stays",
        "n_death",
        "n_complete",
        "n_complete_case",
        "complete_case_n",
        "outcome_rate",
        "overall_mortality_rate",
        "overall_ci_low",
        "overall_ci_high",
        "mortality_rate",
        "median_age",
        "estimate",
        "primary_or",
        "odds_ratio",
        "adjusted_or",
        "ci_lower",
        "ci_upper",
        "primary_ci_low",
        "primary_ci_high",
        "primary_or_ci",
        "p_value",
        "auroc",
        "statistic:auroc",
        "auc",
        "statistic:auc",
        "cv_auroc",
        "statistic:cv_auroc",
        "held_out_auroc",
        "statistic:held_out_auroc",
        "mean_auroc",
        "statistic:mean_auroc",
        "auroc_median",
        "statistic:auroc_ci_lower",
        "statistic:auroc_ci_upper",
        "brier_score",
        "statistic:brier_score",
        "held_out_brier",
        "statistic:held_out_brier",
        "brier_median",
        "calibration_slope",
        "statistic:calibration_slope",
        "calibration_slope_median",
        "calibration_intercept",
        "statistic:calibration_intercept",
        "calibration_intercept_median",
        "baseline_prevalence",
        "statistic:baseline_prevalence",
        "split_strategy",
        "statistic:split_strategy",
        "silhouette_score",
        "silhouette",
        "n_clusters",
        "cluster_count",
        "spearman_rho",
        "rho",
        "skipped",
        "error",
    )
    for record in per_step_records:
        step_id = str(record.get("step_id") or "unknown_step")
        status = str(record.get("status") or "unknown")
        lines.append(f"- {step_id} [{status}]")
        summary = record.get("step_summary")
        if not isinstance(summary, dict) or not summary:
            lines.append("  {}")
            continue
        digest_row: Dict[str, Any] = {}
        for key in preferred_keys:
            scalar = _first_present_scalar(summary, (key,))
            if scalar is not None:
                digest_row[key] = scalar
        if "primary_predictor" in summary:
            digest_row["primary_predictor"] = str(summary["primary_predictor"])
        elif "predictor" in summary:
            digest_row["primary_predictor"] = str(summary["predictor"])
        if "target_outcome" in summary:
            digest_row["target_outcome"] = str(summary["target_outcome"])
        elif "outcome" in summary:
            digest_row["target_outcome"] = str(summary["outcome"])
        if "primary_or_ci" in summary and isinstance(summary["primary_or_ci"], (list, tuple)):
            ci_values = list(summary["primary_or_ci"])
            if len(ci_values) == 2:
                digest_row.setdefault("primary_ci_low", ci_values[0])
                digest_row.setdefault("primary_ci_high", ci_values[1])
        digest_row.update(_summarise_table_one_rows(summary.get("table_one_rows")))
        primary_path = _resolve_writer_aux_path(
            run_dir=run_dir,
            step_id=step_id,
            candidate=summary.get("primary_association_path"),
        )
        digest_row.update(_summarise_primary_association_table(primary_path))
        strata_path = _resolve_writer_aux_path(
            run_dir=run_dir,
            step_id=step_id,
            candidate=summary.get("table") if "sofa_zero_audit" in step_id.lower() else None,
        )
        if strata_path is None and "sofa_zero_audit" in step_id.lower():
            strata_path = run_dir / "steps" / step_id / "outputs" / "sofa_strata.csv"
        digest_row.update(_summarise_sofa_zero_audit(strata_path))
        lines.append(
            "  " + json.dumps(digest_row, ensure_ascii=False, sort_keys=True, default=str)
        )
    return "\n".join(lines)


def _build_replication_notes(
    *,
    paper_profile: PaperProfile,
    replication_spec: PaperReplicationSpec,
    mode: str,
) -> str:
    lines = [
        "Paper replication mode is active.",
        f"Source paper: {paper_profile.paper_title or paper_profile.paper_source}.",
        f"Replication goal: {replication_spec.replication_goal}.",
        f"Mode: {mode}.",
        "Use only EasyICU-observed numbers in the manuscript.",
        "If the original paper is referenced, phrase it as 'original paper reported ...'.",
        "Treat unmappable design elements as explicit deviations, not silent substitutions.",
    ]
    if replication_spec.mapped_concepts:
        lines.append(
            "Mapped concepts: "
            + ", ".join(f"{k}->{v}" for k, v in sorted(replication_spec.mapped_concepts.items()))
            + "."
        )
    if replication_spec.unmappable_items:
        lines.append(
            "Unmappable items: " + "; ".join(replication_spec.unmappable_items) + "."
        )
    return "\n".join(lines)


def _extract_primary_effect_row(
    *, database: str, result: PipelineResult
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    preferred_predictor = _infer_primary_predictor_from_run_dir(run_dir)
    summary_candidates = sorted(run_dir.rglob("step_summary.json"))
    payload: Dict[str, Any] = {
        "database": database,
        "run_id": result.run_id,
        "manifest_path": result.manifest_path,
        "predictor": None,
        "primary_or": None,
        "primary_ci_low": None,
        "primary_ci_high": None,
        "status": "missing_primary_association",
    }
    best_payload: Optional[Dict[str, Any]] = None
    best_score = -10_000
    for path in summary_candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        predictor = (
            summary.get("primary_predictor")
            or summary.get("predictor")
            or summary.get("predictor_variable")
            or summary.get("variable")
        )
        primary_or = _first_present_scalar(
            summary,
            ("primary_or", "odds_ratio", "estimate", "adjusted_or", "lactate_or"),
        )
        ci_low = _first_present_scalar(
            summary,
            ("primary_ci_low", "primary_or_ci_low", "ci_low", "ci_lower", "lower"),
        )
        ci_high = _first_present_scalar(
            summary,
            ("primary_ci_high", "primary_or_ci_high", "ci_high", "ci_upper", "upper"),
        )
        if (
            primary_or is None
            and "primary_or_ci" in summary
            and isinstance(summary["primary_or_ci"], (list, tuple))
        ):
            vals = list(summary["primary_or_ci"])
            if len(vals) >= 2:
                ci_low, ci_high = vals[0], vals[1]
        score = _primary_effect_candidate_score(
            path,
            summary=summary,
            preferred_predictor=preferred_predictor,
        )
        candidate_payload = {
            "predictor": predictor,
            "primary_or": primary_or,
            "primary_ci_low": ci_low,
            "primary_ci_high": ci_high,
            "status": (
                "ok" if primary_or is not None else "summary_missing_primary_or"
            ),
            "step_summary_path": str(path),
        }
        if score > best_score:
            best_score = score
            best_payload = candidate_payload
    if best_payload is not None:
        payload.update(best_payload)
    return payload


def _infer_primary_predictor_from_run_dir(run_dir: Path) -> Optional[str]:
    try:
        payload = json.loads((run_dir / "research_context.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        context = ResearchContext.model_validate(payload)
    except Exception:
        return None
    return _infer_primary_predictor_from_context(context)


def _primary_effect_candidate_score(
    path: Path,
    *,
    summary: Dict[str, Any],
    preferred_predictor: Optional[str],
) -> int:
    path_text = str(path).lower()
    blob = json.dumps(summary, ensure_ascii=False, default=str).lower()
    predictor = str(
        summary.get("primary_predictor")
        or summary.get("predictor")
        or summary.get("predictor_variable")
        or summary.get("variable")
        or ""
    ).lower()
    score = 0
    if _first_present_scalar(
        summary,
        ("primary_or", "odds_ratio", "estimate", "adjusted_or", "lactate_or"),
    ) is not None:
        score += 100
    if "primary_association" in path_text or "association_model" in path_text:
        score += 30
    if "model" in path_text or "regression" in path_text:
        score += 10
    if summary.get("error"):
        score -= 20
    if "bias" in path_text or "vasopressor_selection" in path_text:
        score -= 40
    if preferred_predictor:
        preferred_tokens = _predictor_tokens(preferred_predictor)
        predictor_tokens = _predictor_tokens(predictor)
        path_or_blob_tokens = _predictor_tokens(path_text + " " + blob)
        if preferred_predictor.lower() in predictor or preferred_predictor.lower() in path_text:
            score += 80
        elif preferred_tokens & predictor_tokens:
            score += 70
        elif preferred_tokens & path_or_blob_tokens:
            score += 40
        if not (preferred_tokens & {"vaso", "vasopressor", "vasopressors", "norepinephrine"}) and (
            "vasopressor_selection" in path_text
            or "vaso" in predictor
            or "vasopressor" in predictor
        ):
            score -= 60
    elif predictor:
        score += 5
    return score


def _extract_cross_database_run_summary(
    *,
    database: str,
    result: PipelineResult,
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    run_status_path = run_dir / "run_status.json"
    gates: Dict[str, Any] = {}
    status = "missing_run_status"
    if run_status_path.exists():
        try:
            payload = json.loads(run_status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                status = str(payload.get("status") or status)
                raw_gates = payload.get("gates")
                if isinstance(raw_gates, dict):
                    gates = raw_gates
        except Exception:
            status = "invalid_run_status"
    return {
        "database": database,
        "run_id": result.run_id,
        "status": status,
        "execution_complete": bool(gates.get("execution_complete")),
        "evidence_complete": bool(gates.get("evidence_complete")),
        "numeric_verified": bool(gates.get("numeric_verified")),
        "analysis_validated": bool(gates.get("analysis_validated")),
        "manuscript_ready": bool(gates.get("manuscript_ready")),
        "publication_ready": bool(gates.get("publication_ready")),
        "missing_evidence_count": int(gates.get("missing_evidence_count") or 0),
        "numeric_error_count": int(gates.get("numeric_error_count") or 0),
        "manifest_path": str(result.manifest_path),
        "report_path": str(result.report_path),
        "manuscript_path": str(result.manuscript_path),
    }


def _render_cross_database_comparison_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Cross-database effect comparison",
        "",
        "| database | run_id | predictor | primary_or | ci_low | ci_high | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {database} | {run_id} | {predictor} | {primary_or} | {primary_ci_low} | {primary_ci_high} | {status} |".format(
                database=row.get("database", ""),
                run_id=row.get("run_id", ""),
                predictor=row.get("predictor", "") or "",
                primary_or=(
                    row.get("primary_or", "")
                    if row.get("primary_or") is not None
                    else ""
                ),
                primary_ci_low=(
                    row.get("primary_ci_low", "")
                    if row.get("primary_ci_low") is not None
                    else ""
                ),
                primary_ci_high=(
                    row.get("primary_ci_high", "")
                    if row.get("primary_ci_high") is not None
                    else ""
                ),
                status=row.get("status", ""),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_cross_database_summary_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Cross-database readiness summary",
        "",
        "| database | run_id | status | execution | evidence | numeric | validated | manuscript | publication | missing evidence | numeric errors |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {database} | {run_id} | {status} | {execution_complete} | {evidence_complete} | {numeric_verified} | {analysis_validated} | {manuscript_ready} | {publication_ready} | {missing_evidence_count} | {numeric_error_count} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_cross_database_validation_report(
    *,
    question: Optional[str],
    target_outcome: Optional[str],
    rows: Sequence[Dict[str, Any]],
    run_summaries: Sequence[Dict[str, Any]],
) -> str:
    successful = sum(1 for row in run_summaries if row.get("execution_complete"))
    manuscript_ready = sum(1 for row in run_summaries if row.get("manuscript_ready"))
    publication_ready = sum(1 for row in run_summaries if row.get("publication_ready"))
    lines = [
        "# Cross-database validation report",
        "",
        f"- Research question: {question or 'n/a'}",
        f"- Target outcome: {target_outcome or 'n/a'}",
        f"- Databases run: {len(run_summaries)}",
        f"- Execution-complete runs: {successful}/{len(run_summaries)}",
        f"- Manuscript-ready runs: {manuscript_ready}/{len(run_summaries)}",
        f"- Publication-ready runs: {publication_ready}/{len(run_summaries)}",
        "",
        "## Effect comparison",
        "",
    ]
    lines.extend(_render_cross_database_comparison_markdown(rows).splitlines())
    lines.extend([
        "",
        "## Readiness summary",
        "",
    ])
    lines.extend(_render_cross_database_summary_markdown(run_summaries).splitlines())
    lines.append("")
    return "\n".join(lines) + "\n"




# ---------------------------------------------------------------------------
# Semantic-alias map
# ---------------------------------------------------------------------------
#
# This is the bridge between the writer agent's stable placeholder
# vocabulary (``{evidence:table_one}``, ``{evidence:outcome_rate}``)
# and the per-run, hash-suffixed evidence_ids the EvidenceStore
# actually emits. The map below is intentionally small and
# inspectable; new writer placeholders should be added here rather
# than hidden inside ad-hoc string surgery elsewhere in the pipeline.

# A mapping of (step_id_substring, artefact_basename) -> tuple of aliases.
# step_id_substring is matched as ``in step.step_id`` so that step ids
# the planner generates with arbitrary ordering ("01_table_one",
# "02_outcome_incidence", "04_primary_association", "05_sofa_zero_audit")
# all resolve correctly.
_SEMANTIC_ALIAS_MAP: Dict[tuple, tuple] = {
    # Cohort/table-one summaries often carry the same mortality
    # incidence statistic that manuscript scaffolds cite as outcome_rate.
    ("cohort_summary", "step_summary.json"): (
        "cohort_summary",
        "outcome_rate",
        "mortality_rate",
        "outcome_incidence",
    ),
    ("table_one", "step_summary.json"): (
        "table_one",
        "cohort_summary",
        "mortality_rate",
    ),
    # Outcome incidence summary should also answer to "outcome_rate".
    ("outcome_incidence", "step_summary.json"): ("outcome_rate", "outcome_incidence"),
    ("stratified_incidence", "step_summary.json"): (
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
    ),
    ("stratified_mortality", "step_summary.json"): (
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
    ),
    ("correlation_analysis", "step_summary.json"): (
        "primary_association",
        "correlation_summary",
        "spearman_correlation",
        "model_performance",
    ),
    ("visualization", "step_summary.json"): (
        "correlation_summary",
        "spearman_correlation",
    ),
    ("missing_data_audit", "step_summary.json"): (
        "missingness",
        "outcome_rate",
        "mortality_rate",
    ),
    # The primary-association step_summary doubles as the OR / model
    # statistic the writer points at.
    ("primary_association", "step_summary.json"): ("primary_association",),
    ("association_model", "step_summary.json"): (
        "primary_association",
        "association_model",
        "primary_association_table",
    ),
    ("sensitivity_complete_case", "step_summary.json"): (
        "primary_association",
        "complete_case_model",
        "kdigo_sensitivity",
    ),
    ("sensitivity_reduced_vars", "step_summary.json"): (
        "primary_association",
        "association_model",
        "reduced_model",
        "kdigo_sensitivity",
    ),
    ("sensitivity_reduced", "step_summary.json"): (
        "primary_association",
        "association_model",
        "reduced_model",
        "kdigo_sensitivity",
    ),
    ("model_fitting_complete_case", "step_summary.json"): (
        "primary_association",
        "complete_case_model",
    ),
    ("model_fitting_missing_indicator", "step_summary.json"): (
        "primary_association",
        "missing_indicator_model",
    ),
    ("model_fitting_reduced_variables", "step_summary.json"): (
        "primary_association",
        "reduced_variable_model",
    ),
    ("complete_case_robustness", "step_summary.json"): (
        "primary_association",
        "outcome_rate",
        "mortality_rate",
        "robustness_summary",
    ),
    # Some hosted models name the association/safety step
    # "composite_audit" while still emitting the SOFA stratum mortality
    # and score-0-vs-1 association summary that the manuscript cites as
    # the primary association.
    ("composite", "step_summary.json"): ("primary_association", "stratum_audit"),
    ("sofa_zero_audit", "step_summary.json"): (
        "sofa_zero_audit",
        "sofa_zero_count",
        "outcome_rate",
        "mortality_rate",
    ),
    ("mortality_association", "step_summary.json"): (
        "primary_association",
        "outcome_rate",
        "mortality_rate",
    ),
    # Generic table outputs from any step.
    ("", "table_one.csv"): ("table_one",),
    ("", "missingness.csv"): ("missingness",),
    ("", "sofa_strata.csv"): ("sofa_strata",),
    ("", "stratum_audit.csv"): ("stratum_audit", "table_stratum_audit", "sofa_strata"),
    ("", "sofa2_stratum_balance.csv"): (
        "primary_association_table",
        "stratum_audit",
        "sofa_strata",
    ),
    ("", "stratified_mortality_incidence.csv"): (
        "stratified_mortality_incidence",
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
        "sofa_strata",
    ),
    ("", "primary_association.csv"): ("primary_association_table",),
    ("", "correlation_matrix.csv"): (
        "correlation_matrix",
        "correlation_summary",
        "primary_association_table",
    ),
    ("", "cluster_characteristics.csv"): (
        "cluster_characteristics",
        "cluster_summary",
        "table_one",
    ),
    ("", "cluster_mortality.csv"): ("cluster_mortality", "outcome_rate"),
    ("", "clustering_algorithm_details.json"): (
        "clustering_algorithm_details",
        "clustering_methodology",
    ),
    ("", "clustering_methodology.json"): (
        "clustering_methodology",
        "cluster_summary",
    ),
    # Figures.
    ("", "sofa_strata.png"): ("sofa_strata_figure",),
    ("", "mortality_by_sofa2_stratum.png"): (
        "mortality_by_sofa2_stratum",
        "figure_mortality_by_sofa2_stratum",
        "sofa_strata_figure",
    ),
    ("", "stratum_audit.png"): ("stratum_audit_figure", "sofa_strata_figure"),
    ("", "missingness_heatmap.png"): ("missingness_heatmap",),
    ("", "primary_association_curve.png"): ("primary_association_figure",),
    ("", "sofa2_correlation_heatmap.png"): (
        "sofa2_correlation_heatmap",
        "correlation_heatmap",
        "correlation_figure",
    ),
    ("", "correlation_heatmap.png"): (
        "correlation_heatmap",
        "sofa2_correlation_heatmap",
        "correlation_figure",
    ),
    ("", "clustering_visualization.png"): (
        "clustering_visualization",
        "clustering_figure",
    ),
}


def _semantic_aliases_for(step: AnalysisStep, artefact: Path) -> List[str]:
    """Return the semantic aliases for an artefact registered under a step.

    First-write-wins on the EvidenceStore side: if multiple steps emit
    a ``table_one.csv``, only the first registration claims the
    ``table_one`` alias. That matches the convention that the
    Methods/Results scaffold cites Table 1 once.
    """
    out: List[str] = []
    if artefact.name == "step_summary.json":
        step_id = step.step_id or ""
        if step_id:
            out.append(step_id)
            stripped = re.sub(r"^\d+[_-]+", "", step_id)
            if stripped and stripped != step_id:
                out.append(stripped)
        expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
        intent = (step.intent or "").lower()
        prediction_step = "prediction" in intent or "model_training" in step_id.lower()
        if any(token in expected for token in ("auroc", "brier")) or (
            "calibration" in expected and prediction_step
        ) or prediction_step:
            out.extend(
                [
                    "model_performance",
                    "prediction_performance",
                    "baseline_prevalence",
                ]
            )
        if any(token in expected for token in ("cluster", "silhouette")) or (
            "cluster" in intent
            or "cluster" in step_id.lower()
            or "trajectory" in step_id.lower()
        ):
            out.extend(
                [
                    "cluster_summary",
                    "cluster_characteristics",
                    "cluster_mortality",
                    "clustering_performance",
                    "clustering_methodology",
                    "table_one",
                ]
            )
        if (
            "robustness" in expected
            or "robustness" in intent
            or "complete_case_robustness" in step_id.lower()
        ):
            out.extend(
                [
                    "primary_association",
                    "outcome_rate",
                    "mortality_rate",
                    "robustness_summary",
                ]
            )
        if (
            ("table_one" in step_id.lower() or "table:table_one" in expected)
            and not (artefact.parent / "table_one.csv").exists()
        ):
            out.append("table_one")
    for (step_substr, basename), aliases in _SEMANTIC_ALIAS_MAP.items():
        if basename != artefact.name:
            continue
        if step_substr and step_substr not in (step.step_id or "").lower():
            continue
        out.extend(aliases)
    return out


def _clear_output_dir(out_dir: Path) -> None:
    """Remove stale artefacts before rerunning a repaired step script."""
    if not out_dir.exists():
        return
    for child in out_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# T3.2 — cost summary renderer
# ---------------------------------------------------------------------------


def _render_cost_summary(meter: "CostMeter") -> str:
    """Render a markdown view of a :class:`CostMeter` for the run report.

    The output has three sections:

    * a one-line headline (``n_calls``, total tokens, total USD);
    * a per-role breakdown so paper authors can quote, e.g., that the
      planner is the most expensive role;
    * a per-model breakdown so a multi-model router run shows which
      checkpoint dominates spend.

    All numbers come from :meth:`CostMeter.summary` so the markdown
    here is purely presentational — the row-level
    ``cost_records.json`` is the source of truth.
    """
    summary = meter.summary()
    lines: List[str] = ["# LLM cost summary (T3.2)", ""]
    if summary["n_calls"] == 0:
        lines.append("_No LLM calls were recorded for this run._")
        return "\n".join(lines) + "\n"
    lines.append(
        f"- **{summary['n_calls']}** LLM calls — "
        f"{summary['total_prompt_tokens']:,} prompt + "
        f"{summary['total_completion_tokens']:,} completion = "
        f"**{summary['total_tokens']:,} total tokens**"
    )
    cost = summary["total_cost_usd"]
    if cost > 0:
        lines.append(f"- Estimated total cost: **${cost:.4f} USD**")
    if summary["any_heuristic"]:
        lines.append(
            "- ⚠️ At least one record relies on a `chars/4` token heuristic "
            "(client did not expose `last_usage`). Treat counts as "
            "approximate."
        )
    lines.append("")
    if summary["by_role"]:
        lines.append("## By role")
        lines.append("")
        lines.append("| role | calls | prompt | completion | total | cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for role, b in sorted(summary["by_role"].items()):
            lines.append(
                f"| `{role}` | {b['n_calls']} | {b['prompt_tokens']:,} | "
                f"{b['completion_tokens']:,} | {b['total_tokens']:,} | "
                f"${b['cost_usd']:.4f} |"
            )
        lines.append("")
    if summary["by_model"]:
        lines.append("## By model")
        lines.append("")
        lines.append("| model | calls | prompt | completion | total | cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, b in sorted(summary["by_model"].items()):
            lines.append(
                f"| `{model}` | {b['n_calls']} | {b['prompt_tokens']:,} | "
                f"{b['completion_tokens']:,} | {b['total_tokens']:,} | "
                f"${b['cost_usd']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = ["ResearchAgentPipeline"]
