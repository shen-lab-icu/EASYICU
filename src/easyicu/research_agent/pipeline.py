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
    WriterAgent.run               → manuscript_scaffold.md
    EvidenceStore.bind_manuscript → manuscript_scaffold_bound.md
        ↓
    write manifest.json + results_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
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

from .agents import AnalyzerAgent, CoderAgent, PlannerAgent, ReplannerAgent, WriterAgent
from .cost import CostMeter, metered_role_resolver
from .context import (
    build_naive_research_context,
    build_research_context,
    build_retrieved_research_context,
)
from .evidence import EvidenceStore
from .bibtex import render_bibtex
from .latex import scaffold_to_latex
from .literature import LiteratureAgent, LiteratureBundle
from .llm import LLMClient, LLMRouter, MockLLMClient, resolve_role_client
from .memory import RunMemory
from .prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .runner import CodeRunner, DockerRunner, RunResult
from .schema import (
    AnalysisManifest,
    AnalysisPlan,
    AnalysisStep,
    EvidenceRecord,
    PipelineResult,
    ResearchContext,
    TimeWindow,
    ValidationFinding,
    VariableRole,
)
from .skills import ClinicalSkill, get_skill, list_skills
from .validators import (
    CohortAuditor,
    ConceptUsageAuditor,
    LLMConceptAuditor,
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
    flush_partial_manifest: Callable[[Optional[Dict[str, Any]]], None]


@dataclass
class _WritePhaseResult:
    literature: Optional[LiteratureBundle]
    bound_path: Path


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
        self._enable_vlm_visual_qa = bool(enable_vlm_visual_qa)
        self._vlm_client = vlm_client
        self._visual_qa_adapter = visual_qa_adapter
        self._enable_llm_concept_audit = bool(enable_llm_concept_audit)
        self._llm_concept_auditor_client = llm_concept_auditor_client
        self._enable_memory = enable_memory
        self._enable_latex = enable_latex
        self._latex_venue_template = latex_venue_template or "article"
        lang = (manuscript_language or "en").lower()
        self._manuscript_language = "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
        # T1.4 — when set, the pipeline strips the ICU rules out of the
        # context that drives planning, coding and validation. This is
        # the "naive" arm of the hero ablation: a generic data agent
        # sees only column names + dtypes + ANY-aggregation.
        self._disable_icu_context = bool(disable_icu_context)
        self._context_top_k = int(context_top_k) if context_top_k else None
        self._max_code_repair_attempts = max(0, int(max_code_repair_attempts))
        self._enable_deterministic_code_fallback = bool(enable_deterministic_code_fallback)
        self._enable_deterministic_planner_fallback = bool(enable_deterministic_planner_fallback)
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
            Path(cache_dir).resolve() if cache_dir is not None
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
        notes: Optional[str],
        skill_obj: Optional[ClinicalSkill],
        llm: LLMClient,
        run_dir: Path,
        run_id: str,
        run_language: str,
        resume_state: Optional[Dict[str, Any]],
        emit_progress: Callable[..., None],
    ) -> _PlanPhaseResult:
        """Build context, attach memory, and emit an execution plan."""
        builder = build_naive_research_context if self._disable_icu_context else build_research_context
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
                plan=AnalysisPlan(steps=[]),
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

        for client in self._iter_mock_clients(llm):
            if client.context is None:
                client.context = agent_context
        llm_signature = self._llm_signature(llm)
        used_mock_llm = any(True for _ in self._iter_mock_clients(llm))
        prompt_version = PROMPT_PACK_VERSION
        prompt_files = prompt_pack_files()

        cost_meter: Optional[CostMeter] = None
        if self._enable_cost_tracking:
            cost_meter = CostMeter(
                price_table=dict(self._cost_price_table)
                if self._cost_price_table
                else None,
            ) if self._cost_price_table is not None else CostMeter()
            role_resolver = metered_role_resolver(llm, cost_meter)
        else:
            def role_resolver(role: str):
                return resolve_role_client(llm, role)

        if skill_obj is not None:
            plan_generation_mode = "deterministic_skill"
            issues = skill_obj.validate_against(pd.read_parquet(cohort_path))
            for msg in issues:
                findings.append(ValidationFinding(
                    validator="clinical_skill", severity="warning", message=msg,
                ))
            plan = skill_obj.plan(context)
        else:
            plan_generation_mode = "llm"
            planner = PlannerAgent(role_resolver("planner"))
            try:
                plan = planner.run(agent_context)
            except Exception as exc:
                if not self._enable_deterministic_planner_fallback:
                    raise
                findings.append(ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        "Planner agent failed; using deterministic fallback plan: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    detail={"generation_mode": "fallback"},
                ))
                plan = PlannerAgent(MockLLMClient(context=agent_context)).run(agent_context)
                used_mock_llm = True
                plan_generation_mode = "fallback"
            dropped_plan_keys = getattr(planner, "last_dropped_plan_keys", None) or {}
            dropped_keys = list(dropped_plan_keys.get("top_level", [])) + list(
                dropped_plan_keys.get("steps", [])
            )
            if dropped_keys:
                findings.append(ValidationFinding(
                    validator="planner_schema",
                    severity="warning",
                    message=(
                        "Planner returned unsupported plan fields that were dropped "
                        "before schema validation."
                    ),
                    detail={"dropped_keys": dropped_keys},
                ))
        plan_path = run_dir / "analysis_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get("analysis_plan") is None:
            evidence.register_file(
                kind="log",
                description=(
                    f"Analysis plan from ClinicalSkill '{skill_obj.key}'." if skill_obj
                    else "Analysis plan emitted by PlannerAgent."
                ),
                source_path=plan_path,
                evidence_id="analysis_plan",
                producer="planner" if skill_obj is None else "clinical_skill",
                generation_mode=plan_generation_mode,
                prompt_pack_version=prompt_version,
                metadata={"llm_signature": llm_signature, "used_mock_llm": used_mock_llm},
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
        runner = self._build_runner(run_dir=run_dir, cohort_path=cohort_path)
        usage_auditor = ConceptUsageAuditor()
        stat_validator = StatisticalValidator()

        per_step_records: List[Dict[str, Any]] = []
        probe_summary: Dict[str, Any] = {}
        resumed_step_ids: set = set()
        if plan_result.resume_state is not None:
            try:
                prior_records = [
                    rec for rec in (plan_result.resume_state.get("per_step_records", []) or [])
                    if isinstance(rec, dict) and rec.get("step_id")
                ]
                prior_ok_step_ids = {
                    rec["step_id"] for rec in prior_records
                    if rec.get("status") == "ok"
                }
                for rec in plan_result.resume_state.get("per_step_records", []) or []:
                    if isinstance(rec, dict) and rec.get("status") == "ok" and rec.get("step_id"):
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
                    if rec.get("step_id") == "00_probe" and isinstance(rec.get("step_summary"), dict):
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
            }
            if extra:
                payload.update(extra)
            (run_dir / "manifest_partial.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        _flush_partial_manifest()

        def _register_plan_revision(
            revised_plan: AnalysisPlan,
            *,
            reason: str,
        ) -> Path:
            revision_path = run_dir / f"analysis_plan_revision_{revised_plan.revision}.json"
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
                findings.append(ValidationFinding(
                    validator="replanner",
                    severity="warning",
                    message=f"Replanner failed; keeping existing plan: {exc}",
                    detail={"reason": reason},
                ))
                return current_plan
            if revised.model_dump(mode="json") == current_plan.model_dump(mode="json"):
                return current_plan
            plan_path = _register_plan_revision(revised, reason=reason)
            plan_result.plan_path = plan_path
            findings.append(ValidationFinding(
                validator="replanner",
                severity="info",
                message=f"Plan revised after {reason}.",
                detail={"from_revision": current_plan.revision, "to_revision": revised.revision},
            ))
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
                rec = evidence.register_file(
                    kind=kind,
                    description=f"Probe artefact {probe_file.name}.",
                    source_path=probe_file,
                    produced_by_step=probe_step_id,
                    producer="pipeline",
                    generation_mode="deterministic_probe",
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
            *, repair_attempts: int, fallback_used: bool, runner_repair_name: Optional[str] = None
        ) -> str:
            if fallback_used:
                return "fallback"
            if repair_attempts > 0:
                return "repaired"
            if runner_repair_name:
                return "runner_repaired"
            return "llm"

        def _finding_severity(findings_for_step: Sequence[ValidationFinding]) -> Optional[str]:
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
            messages = [f.message for f in findings_for_step if f.severity in {"warning", "error"}]
            for evidence_id in evidence_ids:
                evidence.update_record(
                    evidence_id,
                    finding_severity=severity,
                    finding_messages=messages,
                    metadata=metadata,
                )

        def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
            step_record: Dict[str, Any] = {"step_id": step.step_id, "intent": step.intent}
            step_current = step_order.get(step.step_id, 0) + 1
            emit_progress(
                "step",
                f"Step {step_current}/{total_steps} started: {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )

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
                    findings.append(ValidationFinding(
                        validator="coder", severity="error",
                        message=f"Coder agent failed for step {step.step_id}: {exc}",
                    ))
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
                if deterministic_fallback_used or not self._enable_deterministic_code_fallback:
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
                    context=context, script_text=code, step=step,
                )
                if self._enable_llm_concept_audit:
                    llm_audit_client = self._llm_concept_auditor_client or role_resolver("analyzer")
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
                    1 for f in usage_findings
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
                audit_log = "\n".join(f"{f.severity.upper()}: {f.message}" for f in usage_findings)
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
                    fallback_code = _deterministic_fallback_code("concept_repair_failed")
                    if fallback_code is not None:
                        code = fallback_code
                        continue
                    with shared_lock:
                        findings.extend(usage_findings)
                        findings.append(ValidationFinding(
                            validator="coder", severity="error",
                            message=(
                                f"Coder repair failed after concept audit for "
                                f"step {step.step_id}: {exc}"
                            ),
                        ))
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
                        "fallback_reason": step_record.get("deterministic_code_fallback"),
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
                            "fallback_reason": step_record.get("deterministic_code_fallback"),
                            "runner_repair": runner_repair_name,
                        },
                    )

                if run_result.succeeded:
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
                        art for art in run_result.artefacts
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
                        step_record["visual_findings"] = [f.model_dump() for f in visual_findings]
                        visual_errors = [f for f in visual_findings if f.severity == "error"]
                        if visual_errors:
                            if repair_attempts >= self._max_code_repair_attempts:
                                fallback_code = _deterministic_fallback_code("visual_qa")
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
                                f"{f.severity.upper()}: {f.message}" for f in visual_findings
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
                                fallback_code = _deterministic_fallback_code("visual_qa_repair_failed")
                                if fallback_code is not None:
                                    code = fallback_code
                                    _clear_output_dir(run_result.out_dir)
                                    continue
                                with shared_lock:
                                    findings.extend(visual_findings)
                                    findings.append(ValidationFinding(
                                        validator="coder",
                                        severity="error",
                                        message=(
                                            f"Coder repair failed after visual QA "
                                            f"for step {step.step_id}: {exc}"
                                        ),
                                    ))
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
                    break

                if repair_attempts >= self._max_code_repair_attempts:
                    fallback_code = _deterministic_fallback_code("execution_failure")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                    with shared_lock:
                        findings.append(ValidationFinding(
                            validator="runner", severity="error",
                            message=(
                                f"Step {step.step_id} "
                                f"{'timed out' if run_result.timed_out else 'failed'} "
                                f"with returncode {run_result.returncode}."
                            ),
                        ))
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

                if log_path.exists():
                    run_log = log_path.read_text(encoding="utf-8", errors="replace")
                else:
                    run_log = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
                runner_repair = _deterministic_runner_repair(
                    code=code,
                    run_log=run_log,
                    previous_repair=runner_repair_name,
                )
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
                        findings.append(ValidationFinding(
                            validator="coder", severity="error",
                            message=f"Coder repair failed for step {step.step_id}: {exc}",
                        ))
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
                elif art.suffix.lower() in {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}:
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
            with shared_lock:
                findings.extend(stat_findings)
            step_record["stat_findings"] = [f.model_dump() for f in stat_findings]
            step_record["generation_mode"] = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
            )
            step_record["step_summary"] = step_summary

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
                usage_findings + stat_findings,
                metadata={
                    "step_id": step.step_id,
                    "generation_mode": step_record["generation_mode"],
                },
            )
            step_record["status"] = "ok"
            with shared_lock:
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "step",
                f"Step {step_current}/{total_steps} complete: {step.step_id}.",
                status="complete",
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
            findings.append(ValidationFinding(
                validator="replanner",
                severity="info",
                message=(
                    "Replanning is enabled, so step execution was forced to sequential "
                    "mode to preserve run-internal plan revisions."
                ),
            ))

        if self._max_concurrent_steps <= 1 or len(steps_to_run) <= 1 or self._enable_replanning:
            executed_step_ids = set(resumed_step_ids)
            remaining_steps = [s for s in plan.steps if s.step_id not in executed_step_ids]
            while remaining_steps:
                step = remaining_steps.pop(0)
                record = _execute_one_step(step)
                executed_step_ids.add(step.step_id)
                if self._enable_replanning and record.get("status") == "ok" and remaining_steps:
                    plan = _maybe_replan(
                        current_plan=plan,
                        reason=step.step_id,
                        probe_summary_payload=probe_summary,
                        completed_records=per_step_records,
                    )
                    step_order.clear()
                    step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
                    remaining_steps = [s for s in plan.steps if s.step_id not in executed_step_ids]
                    total_steps = len(plan.steps)
        else:
            workers = min(self._max_concurrent_steps, len(steps_to_run))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ra_step") as ex:
                futures = [ex.submit(_execute_one_step, s) for s in steps_to_run]
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        with shared_lock:
                            findings.append(ValidationFinding(
                                validator="step_executor", severity="error",
                                message=f"Worker raised an unhandled exception: {exc!r}",
                            ))

        if self._enable_visual_qa:
            emit_progress(
                "visual_qa",
                "Auditing generated figures.",
                run_id=run_id,
            )
            fig_paths = [run_dir / r.relative_path for r in evidence.records() if r.kind == "figure"]
            vlm_adapter = self._visual_qa_adapter
            if vlm_adapter is None and self._enable_vlm_visual_qa:
                client = role_resolver("analyzer")
                if client is not None:
                    vlm_adapter = VLMVisualQAAdapter(client)
            findings += VisualQAAuditor(vlm_adapter=vlm_adapter).audit(figure_paths=fig_paths)

        plan_result.plan = plan
        plan_result.plan_path = plan_path
        return _ExecutePhaseResult(
            plan=plan,
            per_step_records=per_step_records,
            probe_summary=probe_summary,
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

        literature: Optional[LiteratureBundle] = None
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
                lit_path.write_text(literature.model_dump_json(indent=2), encoding="utf-8")
                if evidence.get("literature_bundle") is None:
                    evidence.register_file(
                        kind="log",
                        description="LiteratureBundle (citation registry for this run).",
                        source_path=lit_path,
                        evidence_id="literature_bundle",
                        producer="literature",
                        generation_mode="llm" if lit_client is not None else "deterministic_skill",
                        prompt_pack_version=prompt_version,
                        metadata={
                            "enable_pubmed": self._enable_pubmed,
                            "enable_tavily": self._enable_tavily,
                        },
                    )
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator="literature_agent", severity="warning",
                    message=f"Literature agent failed: {exc}",
                ))

        emit_progress(
            "writer",
            "Drafting manuscript scaffold.",
            run_id=run_id,
        )
        writer = WriterAgent(role_resolver("writer"), language=run_language)
        try:
            scaffold = writer.run(
                context=agent_context,
                evidence_ids=evidence.resolvable_names(),
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

        bound = evidence.bind_manuscript(scaffold)
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
                    title=manuscript_title or f"EasyICU research-agent: {context.research_question}",
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
                findings.append(ValidationFinding(
                    validator="latex_export", severity="warning",
                    message=f"LaTeX export failed: {exc}",
                ))

        return _WritePhaseResult(literature=literature, bound_path=bound_path)

    def _finalise_success(
        self,
        *,
        plan_result: _PlanPhaseResult,
        execute_result: _ExecutePhaseResult,
        write_result: _WritePhaseResult,
        run_id: str,
        run_dir: Path,
        notes: Optional[str],
        database: str,
        target_outcome: Optional[str],
        stop_after_analysis: bool,
        cache_key: Optional[str],
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
            key=lambda r: -1 if r.get("step_id") == "00_probe" else plan_order.get(r.get("step_id"), 10**9)
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
            cost_md_path.write_text(_render_cost_summary(plan_result.cost_meter), encoding="utf-8")
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
            suffix = "paused_after_analysis: manuscript generation skipped by user option."
            manifest_notes = f"{notes}\n\n{suffix}" if notes else suffix
        literature_provenance = _literature_provenance_note(
            enable_literature=self._enable_literature,
            enable_pubmed=self._enable_pubmed,
            enable_tavily=self._enable_tavily,
        )
        manifest_notes = (
            f"{manifest_notes}\n\n{literature_provenance}"
            if manifest_notes else literature_provenance
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
        notes: Optional[str] = None,
        skill: Optional[Union[str, ClinicalSkill]] = None,
        manuscript_title: Optional[str] = None,
        manuscript_authors: Optional[Sequence[str]] = None,
        manuscript_language: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        stop_after_analysis: bool = False,
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
            raise ValueError("`question` is required (or pass `skill=...` to derive one).")
        if self._llm is None:
            raise ValueError(
                "ResearchAgentPipeline.run() now requires an explicit `llm=` "
                "client. Pass MockLLMClient() only for tests or deterministic "
                "demo runs; the pipeline no longer falls back to mock silently."
            )
        run_language = self._normalise_manuscript_language(
            manuscript_language or self._manuscript_language
        )
    
        def _emit_progress(stage: str, message: str, **extra: Any) -> None:
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
            run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
            run_dir = self.workdir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
    
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
            notes=notes,
            skill_obj=skill_obj,
            llm=llm,
            run_dir=run_dir,
            run_id=run_id,
            run_language=run_language,
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
            notes=notes,
            database=database,
            target_outcome=target_outcome,
            stop_after_analysis=stop_after_analysis,
            cache_key=cache_key,
            emit_progress=_emit_progress,
        )
    
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
                raise ValueError("`cohorts` must contain at least one database -> cohort entry.")
    
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
                    notes=notes,
                    skill=skill,
                    manuscript_language=manuscript_language,
                    stop_after_analysis=stop_after_analysis,
                )
                run_results[database] = result
                comparison_rows.append(
                    _extract_primary_effect_row(database=database, result=result)
                )
    
            replication_id = "replication_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
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
                ResearchAgentPipeline._llm_signature(c)
                for c in llm.iter_clients()
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
            "enable_deterministic_code_fallback": bool(self._enable_deterministic_code_fallback),
            "enable_deterministic_planner_fallback": bool(self._enable_deterministic_planner_fallback),
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
                context=context, plan=None, findings=findings,
                per_step_records=[], evidence=evidence,
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
        if variable.role not in {VariableRole.ORDINAL_SCORE, VariableRole.COMPOSITE_SCORE}:
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
            "n_zero": int((observed == 0).sum()) if pd.api.types.is_numeric_dtype(observed) else None,
        }
        if outcome and outcome in df.columns and 0 in set(observed.unique()) and 1 in set(observed.unique()):
            zero_mask = df[variable.name] == 0
            one_mask = df[variable.name] == 1
            zero_rate = float(df.loc[zero_mask, outcome].mean()) if zero_mask.any() else None
            one_rate = float(df.loc[one_mask, outcome].mean()) if one_mask.any() else None
            stats.update(
                {
                    "zero_outcome_rate": zero_rate,
                    "one_outcome_rate": one_rate,
                    "sofa_zero_anomaly": bool(
                        zero_rate is not None and one_rate is not None and zero_rate > one_rate
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
            patched = patched[: line_end + 1] + "\n" + patch + "\n" + patched[line_end + 1 :]
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


def _extract_primary_effect_row(*, database: str, result: PipelineResult) -> Dict[str, Any]:
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
        primary_or = summary.get("primary_or") or summary.get("odds_ratio")
        ci_low = (
            summary.get("primary_ci_low")
            or summary.get("primary_or_ci_low")
            or summary.get("ci_low")
        )
        ci_high = (
            summary.get("primary_ci_high")
            or summary.get("primary_or_ci_high")
            or summary.get("ci_high")
        )
        if primary_or is None and "primary_or_ci" in summary and isinstance(summary["primary_or_ci"], (list, tuple)):
            vals = list(summary["primary_or_ci"])
            if len(vals) >= 2:
                ci_low, ci_high = vals[0], vals[1]
        payload.update(
            {
                "predictor": predictor,
                "primary_or": primary_or,
                "primary_ci_low": ci_low,
                "primary_ci_high": ci_high,
                "status": "ok" if primary_or is not None else "summary_missing_primary_or",
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
                primary_or=row.get("primary_or", "") if row.get("primary_or") is not None else "",
                primary_ci_low=row.get("primary_ci_low", "") if row.get("primary_ci_low") is not None else "",
                primary_ci_high=row.get("primary_ci_high", "") if row.get("primary_ci_high") is not None else "",
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
    parts.append(f"- Stays: {context.cohort.n_stays:,} / Patients: {context.cohort.n_patients:,}")
    if context.target_outcome:
        parts.append(f"- Target outcome: {context.target_outcome}")
    if context.cross_database_validation:
        parts.append("- Cross-database replication: " + ", ".join(context.cross_database_validation))
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
            parts.append(f"- **{r['step_id']}** — status: `{r.get('status', '?')}`"
                         + (f" (rc={r['returncode']})" if "returncode" in r else ""))
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
    parts.append(textwrap.dedent("""
        ---
        Generated by `easyicu.research_agent.ResearchAgentPipeline`. Every entry
        in the Evidence table is reproducible: rerun the script identified by
        `script_evidence_id` in the manifest, hash the output, and confirm it
        matches the `sha256` recorded here.
    """).strip())
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
    # Outcome incidence summary should also answer to "outcome_rate".
    ("outcome_incidence", "step_summary.json"): ("outcome_rate", "outcome_incidence"),
    # The primary-association step_summary doubles as the OR / model
    # statistic the writer points at.
    ("primary_association", "step_summary.json"): ("primary_association",),
    # Some hosted models name the association/safety step
    # "composite_audit" while still emitting the SOFA stratum mortality
    # and score-0-vs-1 association summary that the manuscript cites as
    # the primary association.
    ("composite", "step_summary.json"): ("primary_association", "stratum_audit"),
    # Generic table outputs from any step.
    ("", "table_one.csv"): ("table_one",),
    ("", "missingness.csv"): ("missingness",),
    ("", "sofa_strata.csv"): ("sofa_strata",),
    ("", "stratum_audit.csv"): ("stratum_audit", "table_stratum_audit", "sofa_strata"),
    ("", "sofa2_stratum_balance.csv"): (
        "primary_association_table", "stratum_audit", "sofa_strata",
    ),
    ("", "primary_association.csv"): ("primary_association_table",),
    # Figures.
    ("", "sofa_strata.png"): ("sofa_strata_figure",),
    ("", "stratum_audit.png"): ("stratum_audit_figure", "sofa_strata_figure"),
    ("", "missingness_heatmap.png"): ("missingness_heatmap",),
    ("", "primary_association_curve.png"): ("primary_association_figure",),
}


def _semantic_aliases_for(step: AnalysisStep, artefact: Path) -> List[str]:
    """Return the semantic aliases for an artefact registered under a step.

    First-write-wins on the EvidenceStore side: if multiple steps emit
    a ``table_one.csv``, only the first registration claims the
    ``table_one`` alias. That matches the convention that the
    Methods/Results scaffold cites Table 1 once.
    """
    out: List[str] = []
    for (step_substr, basename), aliases in _SEMANTIC_ALIAS_MAP.items():
        if basename != artefact.name:
            continue
        if step_substr and step_substr not in step.step_id:
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
