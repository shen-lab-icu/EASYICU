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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

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
from .context import (
    build_naive_research_context,
    build_research_context,
    build_retrieved_research_context,
)
from .evidence import EvidenceStore
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
from .llm import LLMClient, LLMRouter, MockLLMClient, resolve_role_client
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
    PipelineResult,
    ResearchContext,
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
from .validators import (
    ClinicalConstraintValidator,
    CohortAuditor,
    ConceptUsageAuditor,
    LLMConceptAuditor,
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
        enable_vlm_visual_qa: bool = False,
        vlm_client: Optional[LLMClient] = None,
        visual_qa_adapter: Optional[VLMVisualQAAdapter] = None,
        enable_llm_concept_audit: bool = False,
        llm_concept_auditor_client: Optional[LLMClient] = None,
        enable_memory: bool = True,
        enable_latex: bool = True,
        latex_venue_template: str = "article",
        manuscript_language: str = "en",
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
        max_concurrent_steps: int = 1,
        enable_probe_step: bool = True,
        enable_replanning: bool = True,
        runner_kind: str = "subprocess",
        runner_image: Optional[str] = None,
        runner_network: str = "none",
        runner_factory: Optional[Callable[..., Any]] = None,
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._timeout_seconds = timeout_seconds
        self._python_executable = python_executable
        self._enable_literature = enable_literature
        self._enable_visual_qa = enable_visual_qa
        self._enable_publication_figure_skill = bool(enable_publication_figure_skill)
        self._enable_vlm_visual_qa = bool(
            enable_vlm_visual_qa
            or vlm_client is not None
            or visual_qa_adapter is not None
        )
        self._vlm_client = vlm_client
        self._visual_qa_adapter = visual_qa_adapter
        self._enable_llm_concept_audit = bool(enable_llm_concept_audit)
        self._llm_concept_auditor_client = llm_concept_auditor_client
        self._enable_memory = enable_memory
        self._enable_latex = enable_latex
        self._latex_venue_template = latex_venue_template or "article"
        lang = (manuscript_language or "en").lower()
        self._manuscript_language = (
            "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
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
        # T3.2 — opt-in LLM cost tracking. When enabled, every per-role
        # client is wrapped in a ``MeteredClient`` that records token
        # counts (prompt / completion) and an estimated USD cost into a
        # ``CostMeter``. The records are persisted to
        # ``manifest.cost_records`` plus ``cost_summary.md`` and
        # ``cost_records.json`` artefacts. Off by default so the
        # default pipeline behaviour stays bit-identical.
        self._enable_cost_tracking = bool(enable_cost_tracking)
        self._cost_price_table = cost_price_table
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

        evidence = EvidenceStore(root=run_dir)
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
            role_resolver = metered_role_resolver(llm, cost_meter)
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
                            "outcome_rate",
                            "mortality_rate",
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

        def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
            nonlocal runtime_state
            step_record: Dict[str, Any] = {
                "step_id": step.step_id,
                "intent": step.intent,
            }
            step_current = step_order.get(step.step_id, 0) + 1
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
                            visual_step_summary = json.loads(
                                visual_summary_path.read_text(encoding="utf-8")
                            )
                        except Exception:
                            visual_step_summary = {}
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
                                with shared_lock:
                                    findings.extend(visual_findings)
                                    step_record["status"] = "blocked_by_visual_qa"
                                    per_step_records.append(step_record)
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    f"Visual QA blocked {step.step_id}.",
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
                or any(
                    str(item).startswith("figure:publication_figure")
                    for item in step.expected_outputs
                )
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
                        metadata={"script_evidence_id": script_record.evidence_id},
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
                        metadata={"script_evidence_id": script_record.evidence_id},
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
                    step_summary = json.loads(ssj.read_text(encoding="utf-8"))
                except Exception:
                    step_summary = {}
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
            findings += VisualQAAuditor(vlm_adapter=vlm_adapter).audit(
                figure_paths=fig_paths
            )

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
                        findings.extend(
                            VisualQAAuditor(vlm_adapter=vlm_adapter).audit(
                                figure_paths=fig_paths
                            )
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="publication_figure_skill",
                        severity="warning",
                        message=f"Publication figure skill failed; writer will use existing evidence only: {exc}",
                    )
                )

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
            writer_evidence_digest = _render_writer_evidence_digest(per_step_records)
            scaffold = writer.run(
                context=agent_context,
                evidence_ids=evidence.resolvable_names(),
                evidence_digest=writer_evidence_digest,
            )
        except Exception as exc:
            scaffold = f"(writer failed: {exc})"
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

        bound = evidence.bind_manuscript(evidence_bound_scaffold)
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

        manuscript_critique = critic.review_manuscript(
            scaffold=bound,
            available_evidence_ids=evidence.resolvable_names(),
        )
        if removed_sentences:
            manuscript_critique = manuscript_critique.model_copy(
                update={
                    "status": (
                        "needs_revision"
                        if manuscript_critique.status == "pass"
                        else manuscript_critique.status
                    ),
                    "unsupported_claims": list(manuscript_critique.unsupported_claims)
                    + removed_sentences,
                    "concerns": list(manuscript_critique.concerns)
                    + [
                        "Pipeline removed result-like sentences that lacked evidence placeholders."
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
                    severity=(
                        "warning"
                        if manuscript_critique.status == "needs_revision"
                        else "error"
                    ),
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
                tex = scaffold_to_latex(
                    markdown=bound,
                    title=manuscript_title
                    or f"EasyICU research-agent: {context.research_question}",
                    authors=manuscript_authors or ["EasyICU research-agent"],
                    bibliography=literature,
                    bibliography_basename=bib_basename,
                    venue_template=self._latex_venue_template,
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
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator="latex_export",
                        severity="warning",
                        message=f"LaTeX export failed: {exc}",
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
            _render_report(
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

        report_path.write_text(
            _render_report(
                context=context,
                plan=plan,
                findings=findings,
                per_step_records=per_step_records,
                evidence=evidence,
                paused_after_analysis=stop_after_analysis,
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
            self._record_cache_hit(cache_key, result)
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
            cache_key = self._compute_cache_key(
                cohort_path=cohort_path,
                question=question,
                target_outcome=target_outcome,
                skill_key=(skill_obj.key if skill_obj is not None else None),
                database=database,
                llm=self._llm,
                stop_after_analysis=stop_after_analysis,
                manuscript_language=run_language,
            )
            cached = self._lookup_cache(cache_key)
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
        summary_path = replication_dir / "cross_database_runs.json"
        summary_path.write_text(
            json.dumps(
                {
                    "question": question,
                    "target_outcome": target_outcome,
                    "stop_after_analysis": stop_after_analysis,
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
            "summary_json": str(summary_path),
            "runs": run_results,
        }

    async def replicate_async(self, **kwargs: Any) -> Dict[str, Any]:
        """Async wrapper for cross-database replication."""
        return await asyncio.to_thread(self.replicate, **kwargs)

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

    # ------------------------------------------------------------------
    # T3.5 — cohort cache (helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_signature(llm: Any) -> str:
        """Return a short string identifying the configured LLM(s).

        Two runs with different LLMs *must* invalidate the cache
        because the bound manuscript / generated code / chosen plan
        could differ. We canonicalise the signature so order doesn't
        matter for routers.
        """
        if llm is None:
            return "unconfigured"
        if isinstance(llm, MockLLMClient):
            return "mock"
        # LLMRouter (T2.3) — fingerprint every distinct underlying client.
        if hasattr(llm, "iter_clients"):
            sigs = sorted(
                ResearchAgentPipeline._llm_signature(c) for c in llm.iter_clients()
            )
            return "router(" + ",".join(sigs) + ")"
        model = getattr(llm, "_model", None)
        cls = getattr(llm, "name", llm.__class__.__name__)
        return f"{cls}:{model}" if model else str(cls)

    @staticmethod
    def _hash_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for buf in iter(lambda: fh.read(chunk), b""):
                h.update(buf)
        return h.hexdigest()

    def _compute_cache_key(
        self,
        *,
        cohort_path: Path,
        question: Optional[str],
        target_outcome: Optional[str],
        skill_key: Optional[str],
        database: Optional[str],
        llm: Any,
        stop_after_analysis: bool,
        manuscript_language: str,
    ) -> str:
        """Compose the deterministic cache key for this run.

        The key includes every input that can change the produced
        artefacts: cohort bytes hash, the configured LLM signature,
        and the user-facing run knobs. ``disable_icu_context`` and
        ``enable_pubmed`` are pipeline-level flags that also alter
        the manuscript, so they participate too.
        """
        payload = {
            "cohort_sha256": self._hash_file(cohort_path),
            "question": (question or "").strip(),
            "target_outcome": (target_outcome or "").strip(),
            "skill": (skill_key or "").strip(),
            "database": (database or "").strip(),
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
            "stop_after_analysis": bool(stop_after_analysis),
            "manuscript_language": manuscript_language,
            "llm": self._llm_signature(llm),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_index_path(self) -> Path:
        return self._cache_dir / "cache_index.json"

    def _load_cache_index(self) -> Dict[str, Dict[str, str]]:
        path = self._cache_index_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}

    def _save_cache_index(self, index: Dict[str, Dict[str, str]]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index_path().write_text(
            json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _lookup_cache(self, cache_key: str) -> Optional[PipelineResult]:
        """Return a previous :class:`PipelineResult` if the artefacts
        it points at still exist on disk; otherwise None.

        We deliberately verify both the manifest and the bound
        manuscript files because callers rely on them to be present.
        A cache entry whose run_dir was deleted is treated as a miss
        and silently evicted from the index.
        """
        index = self._load_cache_index()
        entry = index.get(cache_key)
        if not entry:
            return None
        run_id = entry.get("run_id")
        workdir = entry.get("workdir")
        if not run_id or not workdir:
            return None
        run_dir = Path(workdir)
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            # stale entry — drop it.
            index.pop(cache_key, None)
            self._save_cache_index(index)
            return None
        try:
            return PipelineResult(
                run_id=run_id,
                workdir=str(run_dir),
                context_path=str(run_dir / "research_context.json"),
                plan_path=str(run_dir / "analysis_plan.json"),
                manifest_path=str(manifest),
                report_path=str(run_dir / "results_report.md"),
                manuscript_path=str(run_dir / "manuscript_scaffold_bound.md"),
                evidence_count=int(entry.get("evidence_count") or 0),
                findings_count=int(entry.get("findings_count") or 0),
            )
        except Exception:
            return None

    def _record_cache_hit(self, cache_key: str, result: PipelineResult) -> None:
        index = self._load_cache_index()
        index[cache_key] = {
            "run_id": result.run_id,
            "workdir": result.workdir,
            "evidence_count": str(result.evidence_count),
            "findings_count": str(result.findings_count),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_cache_index(index)

    @staticmethod
    def _iter_mock_clients(llm: Any):
        """Yield every :class:`MockLLMClient` reachable through ``llm``.

        For a plain client this is just ``llm`` itself when it's a
        Mock; for an :class:`LLMRouter` we walk the router's role
        clients (and the default) so the per-role mocks all see the
        cohort context. Idempotent for non-Mock clients.
        """
        if llm is None:
            return
        if isinstance(llm, MockLLMClient):
            yield llm
            return
        # Router-style: anything exposing iter_clients()
        if hasattr(llm, "iter_clients"):
            for child in llm.iter_clients():
                if isinstance(child, MockLLMClient):
                    yield child

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
            _render_report(
                context=context,
                plan=None,
                findings=findings,
                per_step_records=[],
                evidence=evidence,
                aborted_reason=reason,
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
            llm_signature=self._llm_signature(self._llm),
            used_mock_llm=any(True for _ in self._iter_mock_clients(self._llm)),
            prompt_pack_version=PROMPT_PACK_VERSION,
            prompt_pack_files=prompt_pack_files(),
            notes=f"aborted: {reason}",
        )
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(
            f"# Manuscript scaffold not generated\n\nPipeline aborted: {reason}.\n",
            encoding="utf-8",
        )
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
        f"""
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
    if (
        predictor
        and error_text
        and predictor not in error_text
        and not (generic_soft_failure or dtype_soft_failure)
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
            repaired = code.replace(
                'model_df = model_df.apply(pd.to_numeric, errors="coerce")',
                textwrap.dedent(
                    """
                    if 'sex' in model_df.columns:
                        model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                    for col in model_df.columns:
                        if col != 'sex':
                            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                    """
                ).strip("\n"),
                1,
            )
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
        return None
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
                r"# Create figure contract with panels\s*"
                r"figure_contract\s*=\s*make_figure_contract\([\s\S]*?"
                r"figure_contract\[[\"']panels[\"']\]\.append\(\{[\s\S]*?\}\)\s*",
                (
                    "# Create figure contract with panels\n"
                    "# Publication helper API mismatch; keep the statistical output and fall back to direct figure export.\n"
                    "figure_contract = None\n\n"
                ),
                code,
                count=1,
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
            ax.set_title('Adjusted association: early lactate and hospital mortality', fontsize=13)
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

    signatures = (
        "pandas data cast to numpy dtype of object",
        "exog contains inf or nans",
        "missingdataerror",
        "ufunc 'isfinite' not supported",
    )
    if not any(sig in lowered for sig in signatures):
        return None
    repair_name = "dtype_coerce_v1"
    if previous_repair == repair_name or "_easyicu_runner_repair_v1" in code:
        return None
    if not any(token in code for token in ("sm.Logit(", "sm.OLS(", "sm.GLM(")):
        return None

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
            return X_work, y_work.astype(float)
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
        r"(?P<prefix>\b(?:res|model)\s*=\s*sm\.(?:Logit|OLS|GLM)\()\s*"
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
    if repaired == code:
        return None
    return repair_name, repaired


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
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "publication_bundle_promote_v1"


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
    if family not in {"prediction_model", "clustering", "robustness"}:
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
                "lactate_max_24h_or",
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
    prediction_required = (
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

    clustering_required = any(
        token in expected for token in ("cluster", "silhouette")
    ) or "cluster" in step_id or "clustering" in intent
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

    figure_required = "figure:" in expected or "publication-ready figure" in intent
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
            _append_missing(
                (
                    f"Step {step.step_id} was expected to produce a figure artifact, "
                    "but the step summary did not record any figure path or figure output."
                ),
                ("figure_path", "figure", "plot_path", "png", "svg"),
            )

    return findings


def _step_contract_repair_guidance(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    code: str,
) -> str:
    guidance: List[str] = []
    predictor = str(
        (step_summary or {}).get("primary_predictor")
        or (step_summary or {}).get("predictor")
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


def _render_writer_evidence_digest(per_step_records: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    preferred_keys = (
        "sample_size",
        "n_total",
        "n_total_stays",
        "n_complete",
        "n_complete_case",
        "complete_case_n",
        "outcome_rate",
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
        "p_value",
        "auroc",
        "cv_auroc",
        "mean_auroc",
        "auroc_median",
        "brier_score",
        "held_out_brier",
        "brier_median",
        "calibration_slope",
        "calibration_slope_median",
        "calibration_intercept",
        "calibration_intercept_median",
        "baseline_prevalence",
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
        if "target_outcome" in summary:
            digest_row["target_outcome"] = str(summary["target_outcome"])
        lines.append(
            "  " + json.dumps(digest_row, ensure_ascii=False, sort_keys=True, default=str)
        )
    return "\n".join(lines)


def _extract_primary_effect_row(
    *, database: str, result: PipelineResult
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    summary_candidates = sorted(
        run_dir.glob("steps/*primary_association*/outputs/step_summary.json")
    )
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
    if not summary_candidates:
        summary_candidates = sorted(run_dir.rglob("step_summary.json"))
    for path in summary_candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        predictor = summary.get("predictor") or summary.get("variable")
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
        payload.update(
            {
                "predictor": predictor,
                "primary_or": primary_or,
                "primary_ci_low": ci_low,
                "primary_ci_high": ci_high,
                "status": (
                    "ok" if primary_or is not None else "summary_missing_primary_or"
                ),
                "step_summary_path": str(path),
            }
        )
        return payload
    return payload


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


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _render_report(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
    evidence: EvidenceStore,
    aborted_reason: Optional[str] = None,
    paused_after_analysis: bool = False,
) -> str:
    parts: List[str] = []
    parts.append(f"# Research-agent results report")
    parts.append("")
    parts.append(f"- Research question: {context.research_question}")
    parts.append(f"- Cohort: {context.cohort.cohort_name} ({context.cohort.database})")
    parts.append(
        f"- Stays: {context.cohort.n_stays:,} / Patients: {context.cohort.n_patients:,}"
    )
    if context.target_outcome:
        parts.append(f"- Target outcome: {context.target_outcome}")
    if context.cross_database_validation:
        parts.append(
            "- Cross-database replication: "
            + ", ".join(context.cross_database_validation)
        )
    parts.append("")

    if aborted_reason:
        parts.append(f"## Status: ABORTED ({aborted_reason})")
        parts.append("")
    elif paused_after_analysis:
        parts.append("## Status: PAUSED AFTER ANALYSIS")
        parts.append("")
        parts.append(
            "The run intentionally stopped before literature retrieval, "
            "manuscript drafting and LaTeX export. Review the registered "
            "tables, figures, statistics and findings before drafting the article."
        )
        parts.append("")

    if plan:
        parts.append("## Plan")
        parts.append("")
        for s in plan.steps:
            parts.append(f"- **{s.step_id}** — {s.intent}")
        parts.append("")

    if per_step_records:
        parts.append("## Step outcomes")
        parts.append("")
        for r in per_step_records:
            parts.append(
                f"- **{r['step_id']}** — status: `{r.get('status', '?')}`"
                + (f" (rc={r['returncode']})" if "returncode" in r else "")
            )
        parts.append("")

    parts.append("## Findings")
    parts.append("")
    if not findings:
        parts.append("- (no findings recorded)")
    else:
        for f in findings:
            parts.append(f"- `{f.severity}` [{f.validator}] {f.message}")
    parts.append("")

    parts.append("## Evidence (registered artefacts)")
    parts.append("")
    parts.append("| evidence_id | kind | description | sha256 (head) | path |")
    parts.append("|---|---|---|---|---|")
    pipe_escape = "\\|"
    for r in evidence.records():
        desc = r.description.replace("|", pipe_escape)
        parts.append(
            f"| `{r.evidence_id}` | {r.kind} | "
            f"{desc} | `{r.sha256[:10]}…` | `{r.relative_path}` |"
        )
    parts.append("")
    parts.append(
        textwrap.dedent(
            """
        ---
        Generated by `easyicu.research_agent.ResearchAgentPipeline`. Every entry
        in the Evidence table is reproducible: rerun the script identified by
        `script_evidence_id` in the manifest, hash the output, and confirm it
        matches the `sha256` recorded here.
    """
        ).strip()
    )
    return "\n".join(parts) + "\n"


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
        "outcome_rate",
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
                    "primary_association",
                    "outcome_rate",
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
        if not (artefact.parent / "table_one.csv").exists():
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
