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

import hashlib
import json
import shutil
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from .agents import AnalyzerAgent, CoderAgent, PlannerAgent, WriterAgent
from .cost import CostMeter, metered_role_resolver
from .context import build_naive_research_context, build_research_context
from .evidence import EvidenceStore
from .bibtex import render_bibtex
from .latex import scaffold_to_latex
from .literature import LiteratureAgent, LiteratureBundle
from .llm import LLMClient, LLMRouter, MockLLMClient, resolve_role_client
from .memory import RunMemory
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
)
from .skills import ClinicalSkill, get_skill
from .validators import CohortAuditor, ConceptUsageAuditor, StatisticalValidator
from .visual_qa import VisualQAAuditor


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
        enable_memory: bool = True,
        enable_latex: bool = True,
        disable_icu_context: bool = False,
        enable_pubmed: bool = False,
        pubmed_email: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        enable_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cost_tracking: bool = False,
        cost_price_table: Optional[Dict[str, Any]] = None,
        max_concurrent_steps: int = 1,
        runner_kind: str = "subprocess",
        runner_image: Optional[str] = None,
        runner_network: str = "none",
        runner_factory: Optional[Callable[..., Any]] = None,
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        # llm is set to a context-bound mock client at run() time if None.
        self._llm = llm
        self._timeout_seconds = timeout_seconds
        self._python_executable = python_executable
        self._enable_literature = enable_literature
        self._enable_visual_qa = enable_visual_qa
        self._enable_memory = enable_memory
        self._enable_latex = enable_latex
        # T1.4 — when set, the pipeline strips the ICU rules out of the
        # context that drives planning, coding and validation. This is
        # the "naive" arm of the hero ablation: a generic data agent
        # sees only column names + dtypes + ANY-aggregation.
        self._disable_icu_context = bool(disable_icu_context)
        # T2.2 — opt-in PubMed live search. Off by default so CI and
        # the offline demo stay deterministic; the LiteratureAgent
        # handles network failure gracefully (empty list → curated
        # registry only).
        self._enable_pubmed = bool(enable_pubmed)
        self._pubmed_email = pubmed_email
        self._pubmed_api_key = pubmed_api_key
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
        resume_run_id: Optional[str] = None,
        stop_after_analysis: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        # Resolve skill (M4-style) and let it fill in missing parameters.
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

        def _emit_progress(stage: str, message: str, **extra: Any) -> None:
            """Best-effort progress callback for web / CLI frontends.

            The pipeline must stay usable in batch jobs, so a buggy UI
            callback is never allowed to fail the scientific run.
            """
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

        # T2.4 — resume mode: reuse the existing run directory and skip
        # any step the prior run already completed at status="ok". The
        # generated run_id is preserved across resumes so manifests
        # are addressable by a stable identifier.
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

        # 1) Materialise cohort to parquet so every step gets a stable path.
        cohort_path = self._materialise_cohort(cohort, run_dir)
        _emit_progress(
            "cohort",
            "Cohort materialised to parquet.",
            run_id=run_id,
            path=str(cohort_path),
        )

        # 1b) T3.5 — cohort cache. Short-circuit identical re-runs by
        #     reading the prior evidence store. Skipped when resuming
        #     a partial run (the resume contract already addresses
        #     "do not re-execute completed steps") and when caching
        #     is disabled (default).
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
            )
            cached = self._lookup_cache(cache_key)
            if cached is not None:
                # The previous run is still on disk — return its
                # PipelineResult and discard the freshly-created
                # ``run_dir`` so we don't leak empty directories.
                shutil.rmtree(run_dir, ignore_errors=True)
                _emit_progress(
                    "cache",
                    f"Reused cached run {cached.run_id}.",
                    status="complete",
                    run_id=cached.run_id,
                )
                return cached

        # 2) Build context. T1.4 — the ablation flag swaps in a naive
        #    builder that emits only column name + dtype + ANY-agg
        #    (the kind of context a generic data agent would synthesise),
        #    so the rest of the pipeline can be compared head-to-head.
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
        _emit_progress(
            "context",
            "Research context built.",
            run_id=run_id,
            n_stays=context.cohort.n_stays,
            n_variables=len(context.variables),
        )

        # 3) Build evidence store and register the context.
        evidence = EvidenceStore(root=run_dir)
        evidence.register_file(
            kind="log",
            description="ResearchContext (frozen at run time).",
            source_path=context_path,
            evidence_id="research_context",
        )

        # 4) Initial cohort audit.
        findings: List[ValidationFinding] = []
        findings += CohortAuditor().audit(context=context, cohort_path=cohort_path)
        if any(f.severity == "error" for f in findings):
            _emit_progress(
                "audit",
                "Cohort audit failed; aborting run.",
                status="error",
                run_id=run_id,
            )
            return self._finalise_aborted(
                run_id=run_id, run_dir=run_dir, context=context,
                context_path=context_path, evidence=evidence,
                findings=findings, reason="cohort_audit_failed",
            )
        _emit_progress(
            "audit",
            "Initial cohort audit passed.",
            run_id=run_id,
            findings=len(findings),
        )

        # 4b) Memory digest (HealthFlow-style) — feed past lessons to the planner.
        if self._memory is not None:
            digest = self._memory.digest_for_prompt(
                research_question=question,
                database=database,
                target_outcome=target_outcome,
            )
            digest_path = run_dir / "memory_digest.md"
            digest_path.write_text(digest, encoding="utf-8")
            evidence.register_file(
                kind="log",
                description="Cross-run memory digest fed to the planner.",
                source_path=digest_path,
                evidence_id="memory_digest",
            )

        # 5) Plan — either from a ClinicalSkill (deterministic) or PlannerAgent.
        llm = self._llm or MockLLMClient(context=context)
        # T2.3 — bind ``ResearchContext`` onto every MockLLMClient
        # reachable through the supplied client/router so the canned
        # responses use the cohort that's actually being analysed.
        for client in self._iter_mock_clients(llm):
            if client.context is None:
                client.context = context

        # T3.2 — set up a per-role resolver. Without tracking it is the
        # plain ``resolve_role_client`` (preserving previous behaviour);
        # with tracking it returns a ``MeteredClient`` that records
        # token usage and estimated cost into ``cost_meter``.
        cost_meter: Optional[CostMeter] = None
        if self._enable_cost_tracking:
            cost_meter = CostMeter(
                price_table=dict(self._cost_price_table)
                if self._cost_price_table
                else None,  # type: ignore[arg-type]
            ) if self._cost_price_table is not None else CostMeter()
            role_resolver = metered_role_resolver(llm, cost_meter)
        else:
            def role_resolver(role: str):
                return resolve_role_client(llm, role)

        if skill_obj is not None:
            issues = skill_obj.validate_against(pd.read_parquet(cohort_path))
            for msg in issues:
                findings.append(ValidationFinding(
                    validator="clinical_skill", severity="warning", message=msg,
                ))
            plan = skill_obj.plan(context)
        else:
            planner = PlannerAgent(role_resolver("planner"))
            plan = planner.run(context)
        plan_path = run_dir / "analysis_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        evidence.register_file(
            kind="log",
            description=(
                f"Analysis plan from ClinicalSkill '{skill_obj.key}'." if skill_obj
                else "Analysis plan emitted by PlannerAgent."
            ),
            source_path=plan_path,
            evidence_id="analysis_plan",
        )
        _emit_progress(
            "plan",
            f"Analysis plan ready with {len(plan.steps)} step(s).",
            run_id=run_id,
            total_steps=len(plan.steps),
        )

        # 6) Step loop. Each agent uses its role-bound LLM client (T2.3),
        #    optionally wrapped in a meter (T3.2).
        coder = CoderAgent(role_resolver("coder"))
        analyzer = AnalyzerAgent(role_resolver("analyzer"))
        runner = self._build_runner(run_dir=run_dir, cohort_path=cohort_path)
        usage_auditor = ConceptUsageAuditor()
        stat_validator = StatisticalValidator()

        per_step_records: List[Dict[str, Any]] = []
        # Resume: replay any already-completed step_records and findings
        # so the partial manifest carries forward and the loop can
        # ``continue`` past steps with status="ok".
        resumed_step_ids: set = set()
        if resume_state is not None:
            try:
                prior_records = [
                    rec for rec in (resume_state.get("per_step_records", []) or [])
                    if isinstance(rec, dict) and rec.get("step_id")
                ]
                prior_ok_step_ids = {
                    rec["step_id"] for rec in prior_records
                    if rec.get("status") == "ok"
                }
                for rec in resume_state.get("per_step_records", []) or []:
                    if isinstance(rec, dict) and rec.get("status") == "ok" and rec.get("step_id"):
                        per_step_records.append(rec)
                        resumed_step_ids.add(rec["step_id"])
                for f in resume_state.get("findings", []) or []:
                    try:
                        finding = ValidationFinding.model_validate(f)
                    except Exception:
                        # Skip ill-formed prior findings rather than
                        # discard the rest of the resume state.
                        continue
                    if finding.validator == "cohort_auditor":
                        # Cohort findings are recomputed at the top of
                        # every resume. Replaying them creates noisy
                        # duplicates in the final report.
                        continue
                    if finding.validator == "runner":
                        # A resume can successfully rerun a previously
                        # failed step. Drop stale runner errors for any
                        # step that now has an ok record in the partial
                        # manifest.
                        msg = finding.message or ""
                        if any(step_id in msg for step_id in prior_ok_step_ids):
                            continue
                    findings.append(finding)
                if resumed_step_ids:
                    print(
                        f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                        f"already-completed step(s) — {sorted(resumed_step_ids)}"
                    )
            except Exception:
                # Defensive: any parse failure → start the loop fresh.
                resumed_step_ids = set()

        # Capture started_at so resumed runs preserve the original start
        # time across the resume boundary (matches the manifest contract).
        started_at = datetime.now(timezone.utc)
        if resume_state and resume_state.get("started_at"):
            try:
                started_at = datetime.fromisoformat(resume_state["started_at"])
            except Exception:
                pass

        def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
            """Write `manifest_partial.json` after every step (T2.4).

            The partial manifest is the resume sentinel: if the process
            crashes mid-loop, this file is what tells the next invocation
            which steps it can skip.
            """
            payload: Dict[str, Any] = {
                "schema_version": "easyicu.research_manifest_partial/1",
                "run_id": run_id,
                "research_question": context.research_question,
                "started_at": started_at.isoformat(),
                "context_path": str(context_path.relative_to(run_dir)),
                "plan_path": str(plan_path.relative_to(run_dir)),
                "evidence": [r.model_dump(mode="json") for r in evidence.records()],
                "findings": [f.model_dump(mode="json") for f in findings],
                "per_step_records": per_step_records,
                "notes": notes,
            }
            if extra:
                payload.update(extra)
            (run_dir / "manifest_partial.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        # Snapshot before the loop so a crash on the very first step
        # still leaves a partial manifest.
        _flush_partial_manifest()

        # T3.3 — guard the cross-step shared state (findings list,
        # per_step_records list, _flush_partial_manifest write) when
        # workers run in parallel. EvidenceStore already guards itself
        # internally so it does not need a separate lock here.
        shared_lock = threading.Lock()
        step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        total_steps = len(plan.steps)

        def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
            """Run all per-step phases for a single :class:`AnalysisStep`.

            Returns the step_record dict; appends per-step findings
            (coder error, usage audit, runner exit, statistical audit)
            and per-step evidence records to the shared lists under
            ``shared_lock`` so concurrent workers stay consistent.

            Importantly, every shared-state mutation (findings,
            per_step_records, _flush_partial_manifest) is serialised
            via ``shared_lock``; EvidenceStore.register_* mutators run
            *outside* the lock because they're already thread-safe and
            holding two locks would risk deadlock.
            """
            step_record: Dict[str, Any] = {"step_id": step.step_id, "intent": step.intent}
            step_current = step_order.get(step.step_id, 0) + 1
            _emit_progress(
                "step",
                f"Step {step_current}/{total_steps} started: {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )

            # 6a) Code generation.
            try:
                _emit_progress(
                    "coder",
                    f"Generating analysis script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                code = coder.run(context=context, step=step)
            except Exception as exc:
                with shared_lock:
                    findings.append(ValidationFinding(
                        validator="coder", severity="error",
                        message=f"Coder agent failed for step {step.step_id}: {exc}",
                    ))
                    step_record["status"] = "coder_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                _emit_progress(
                    "coder",
                    f"Coder failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            # 6b) Static audit before running.
            usage_findings = usage_auditor.audit(
                context=context, script_text=code, step=step,
            )
            with shared_lock:
                findings.extend(usage_findings)
            step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
            if any(f.severity == "error" for f in usage_findings):
                # Block this step but let other workers keep going.
                step_record["status"] = "blocked_by_concept_audit"
                with shared_lock:
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                _emit_progress(
                    "audit",
                    f"Concept audit blocked {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            # 6c) Execute.
            _emit_progress(
                "runner",
                f"Running generated script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            run_result = runner.run(step_id=step.step_id, code=code)
            step_record["returncode"] = run_result.returncode
            step_record["timed_out"] = run_result.timed_out

            # Register the script + run log immediately so they're visible
            # even on failure.
            script_record = evidence.register_file(
                kind="code",
                description=f"Generated analysis script for step {step.step_id}.",
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
            )
            log_path = run_result.cwd / "run.log"
            if log_path.exists():
                evidence.register_file(
                    kind="log",
                    description=f"stdout/stderr log for step {step.step_id}.",
                    source_path=log_path,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                )

            if not run_result.succeeded:
                with shared_lock:
                    findings.append(ValidationFinding(
                        validator="runner", severity="error",
                        message=(
                            f"Step {step.step_id} {'timed out' if run_result.timed_out else 'failed'} "
                            f"with returncode {run_result.returncode}."
                        ),
                    ))
                    step_record["status"] = "execution_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                _emit_progress(
                    "runner",
                    f"Execution failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            # 6d) Register all artefacts. Aliases give the writer stable,
            #     human-readable names so manuscript placeholders resolve
            #     without baking hashed evidence_ids into prompts.
            evidence_ids_for_step: List[str] = [script_record.evidence_id]
            for art in run_result.artefacts:
                step_aliases = _semantic_aliases_for(step, art)
                if art.name == "step_summary.json":
                    rec = evidence.register_file(
                        kind="statistic",
                        description=f"Machine-readable summary for step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                    )
                elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
                    rec = evidence.register_file(
                        kind="table",
                        description=f"Table {art.stem} from step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                    )
                elif art.suffix.lower() in {".png", ".svg", ".pdf"}:
                    rec = evidence.register_file(
                        kind="figure",
                        description=f"Figure {art.stem} from step {step.step_id}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                    )
                else:
                    rec = evidence.register_file(
                        kind="log",
                        description=f"Auxiliary artefact {art.name}.",
                        source_path=art,
                        produced_by_step=step.step_id,
                        script_evidence_id=script_record.evidence_id,
                        aliases=step_aliases,
                    )
                evidence_ids_for_step.append(rec.evidence_id)

            # 6e) Statistical validator.
            step_summary: Dict[str, Any] = {}
            ssj = run_result.out_dir / "step_summary.json"
            if ssj.exists():
                try:
                    step_summary = json.loads(ssj.read_text(encoding="utf-8"))
                except Exception:
                    step_summary = {}
            stat_findings = stat_validator.audit(
                context=context, cohort_path=cohort_path,
                step=step, out_dir=run_result.out_dir,
                step_summary=step_summary,
            )
            with shared_lock:
                findings.extend(stat_findings)
            step_record["stat_findings"] = [f.model_dump() for f in stat_findings]

            # 6f) Analyzer interpretation.
            try:
                interpretation = analyzer.run(
                    context=context, step=step,
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
            )
            step_record["interpretation_evidence_id"] = interp_record.evidence_id
            step_record["status"] = "ok"
            with shared_lock:
                per_step_records.append(step_record)
                _flush_partial_manifest()
            _emit_progress(
                "step",
                f"Step {step_current}/{total_steps} complete: {step.step_id}.",
                status="complete",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        # T3.3 — sequential vs concurrent execution. With
        # ``max_concurrent_steps == 1`` the loop runs in the calling
        # thread (no executor overhead, exact pre-T3.3 behaviour).
        # Above 1, a ThreadPoolExecutor schedules workers up to the
        # cap. Output ordering of ``per_step_records`` reflects
        # completion order; the final report renderer sorts by plan
        # order so paper output stays deterministic.
        steps_to_run = [s for s in plan.steps if s.step_id not in resumed_step_ids]
        for skipped_step_id in sorted(resumed_step_ids):
            _emit_progress(
                "resume",
                f"Skipped completed step from prior run: {skipped_step_id}.",
                status="complete",
                run_id=run_id,
                step_id=skipped_step_id,
            )
        if self._max_concurrent_steps <= 1 or len(steps_to_run) <= 1:
            for step in steps_to_run:
                _execute_one_step(step)
        else:
            workers = min(self._max_concurrent_steps, len(steps_to_run))
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="ra_step") as ex:
                futures = [ex.submit(_execute_one_step, s) for s in steps_to_run]
                for fut in as_completed(futures):
                    # Surface any unexpected exception that escaped the
                    # per-step try/except blocks. We don't abort the
                    # whole pipeline — other workers may still be
                    # producing valid evidence — but we make sure the
                    # crash is visible in the run findings.
                    exc = fut.exception()
                    if exc is not None:
                        with shared_lock:
                            findings.append(ValidationFinding(
                                validator="step_executor", severity="error",
                                message=f"Worker raised an unhandled exception: {exc!r}",
                            ))

        # 6b) Visual QA over registered figures (OpenLens-style). This
        #     still belongs to the analysis phase because it validates
        #     the figures the user reviews before approving a manuscript.
        if self._enable_visual_qa:
            _emit_progress(
                "visual_qa",
                "Auditing generated figures.",
                run_id=run_id,
            )
            fig_paths = [
                run_dir / r.relative_path
                for r in evidence.records() if r.kind == "figure"
            ]
            findings += VisualQAAuditor().audit(figure_paths=fig_paths)

        literature: Optional[LiteratureBundle] = None
        if stop_after_analysis:
            _emit_progress(
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
        elif self._enable_literature:
            # Literature bundle (OpenLens-style) — registered as evidence so
            # the writer can cite it the same way it cites tables.
            try:
                _emit_progress(
                    "literature",
                    "Building literature bundle for manuscript drafting.",
                    run_id=run_id,
                )
                # T2.3 — route to the literature-role client (or
                # ``None`` so LiteratureAgent uses its offline curated
                # registry when the configured client is a Mock).
                lit_client = role_resolver("literature")
                # Unwrap MeteredClient → Mock so the agent sees the real
                # mock and can short-circuit network access.
                if hasattr(lit_client, "_inner") and isinstance(
                    getattr(lit_client, "_inner", None), MockLLMClient
                ):
                    lit_client = None
                if isinstance(lit_client, MockLLMClient):
                    lit_client = None
                # T2.2 — opt-in PubMed live search; supplied client
                # carries the user's NCBI etiquette fields.
                pubmed_client = None
                if self._enable_pubmed:
                    from .literature import PubMedLiteratureClient
                    pubmed_client = PubMedLiteratureClient(
                        email=self._pubmed_email,
                        api_key=self._pubmed_api_key,
                    )
                literature = LiteratureAgent(
                    lit_client,
                    enable_pubmed=self._enable_pubmed,
                    pubmed_client=pubmed_client,
                ).run(context)
                lit_path = run_dir / "literature_bundle.json"
                lit_path.write_text(literature.model_dump_json(indent=2), encoding="utf-8")
                evidence.register_file(
                    kind="log",
                    description="LiteratureBundle (citation registry for this run).",
                    source_path=lit_path,
                    evidence_id="literature_bundle",
                )
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator="literature_agent", severity="warning",
                    message=f"Literature agent failed: {exc}",
                ))

        if not stop_after_analysis:
            # 7) Manuscript scaffold.
            # The writer is given both the raw evidence_ids and the
            # semantic aliases so the LLM may pick whichever feels more
            # natural in prose; the binder accepts either form.
            _emit_progress(
                "writer",
                "Drafting manuscript scaffold.",
                run_id=run_id,
            )
            writer = WriterAgent(role_resolver("writer"))
            try:
                scaffold = writer.run(
                    context=context, evidence_ids=evidence.resolvable_names(),
                )
            except Exception as exc:
                scaffold = f"(writer failed: {exc})"
            scaffold_path = run_dir / "manuscript_scaffold.md"
            scaffold_path.write_text(scaffold, encoding="utf-8")
            evidence.register_file(
                kind="log",
                description="Manuscript scaffold (raw, with {evidence:*} placeholders).",
                source_path=scaffold_path,
                evidence_id="manuscript_scaffold_raw",
            )

            bound = evidence.bind_manuscript(scaffold)
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(bound, encoding="utf-8")
            evidence.register_file(
                kind="log",
                description="Manuscript scaffold with evidence ids resolved to file links + sha256.",
                source_path=bound_path,
                evidence_id="manuscript_scaffold_bound",
            )

            # 7b) LaTeX export (OpenLens-style) — uses the bound markdown.
            if self._enable_latex:
                try:
                    _emit_progress(
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
                    )
                    tex_path = run_dir / f"{bib_basename}.tex"
                    tex_path.write_text(tex, encoding="utf-8")
                    evidence.register_file(
                        kind="log",
                        description="LaTeX scaffold (Discussion left as red TODO).",
                        source_path=tex_path,
                        evidence_id="manuscript_scaffold_tex",
                    )
                    # T3.4 — emit a real ``.bib`` alongside the ``.tex``
                    # whenever the run produced a literature bundle. The
                    # manuscript already contains a ``\bibliography{…}``
                    # pointing at this file, so a single ``pdflatex /
                    # bibtex / pdflatex / pdflatex`` cycle yields a fully
                    # cited PDF.
                    bib_text = render_bibtex(literature)
                    if bib_text:
                        bib_path = run_dir / f"{bib_basename}.bib"
                        bib_path.write_text(bib_text, encoding="utf-8")
                        evidence.register_file(
                            kind="log",
                            description="BibTeX bibliography for the LaTeX scaffold.",
                            source_path=bib_path,
                            evidence_id="manuscript_scaffold_bib",
                        )
                except Exception as exc:
                    findings.append(ValidationFinding(
                        validator="latex_writer", severity="warning",
                        message=f"LaTeX export failed: {exc}",
                    ))

        # 8) Results report. T3.3 — under concurrent execution
        #    ``per_step_records`` is in completion order; sort by plan
        #    order before rendering so paper output stays deterministic
        #    regardless of which worker happened to finish first.
        plan_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        per_step_records.sort(key=lambda r: plan_order.get(r.get("step_id"), 10**9))
        report_path = run_dir / "results_report.md"
        report_path.write_text(
            _render_report(
                context=context, plan=plan, findings=findings,
                per_step_records=per_step_records, evidence=evidence,
                paused_after_analysis=stop_after_analysis,
            ),
            encoding="utf-8",
        )

        # 8b) T3.2 — write a per-run cost report and persist the records
        #     into the manifest. ``cost_records.json`` is the raw row-
        #     level log; ``cost_summary.md`` is the human-readable view
        #     that paper authors can paste into a supplementary table.
        cost_records_for_manifest = []
        if cost_meter is not None:
            cost_records_for_manifest = list(cost_meter.records)
            cost_json_path = run_dir / "cost_records.json"
            cost_json_path.write_text(
                json.dumps(
                    [r.model_dump(mode="json") for r in cost_meter.records],
                    indent=2, ensure_ascii=False, default=str,
                ),
                encoding="utf-8",
            )
            cost_md_path = run_dir / "cost_summary.md"
            cost_md_path.write_text(_render_cost_summary(cost_meter), encoding="utf-8")
            evidence.register_file(
                kind="log",
                description="Raw per-call LLM cost records (T3.2).",
                source_path=cost_json_path,
                evidence_id="cost_records",
            )
            evidence.register_file(
                kind="log",
                description="Human-readable LLM cost summary (T3.2).",
                source_path=cost_md_path,
                evidence_id="cost_summary",
            )

        # 9) Manifest. ``started_at`` is captured from the resume state
        #    when present so the final manifest correctly reflects the
        #    original start time across a resume boundary (T2.4).
        manifest_notes = notes
        if stop_after_analysis:
            suffix = "paused_after_analysis: manuscript generation skipped by user option."
            manifest_notes = f"{notes}\n\n{suffix}" if notes else suffix

        manifest = AnalysisManifest(
            run_id=run_id,
            research_question=context.research_question,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            context_path=str(context_path.relative_to(run_dir)),
            plan_path=str(plan_path.relative_to(run_dir)),
            evidence=evidence.records(),
            findings=findings,
            cost_records=cost_records_for_manifest,
            report_path=str(report_path.relative_to(run_dir)),
            manuscript_path=str(bound_path.relative_to(run_dir)),
            notes=manifest_notes,
        )
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        # One last partial-manifest flush so the on-disk partial state
        # is consistent with the final manifest (audit-friendly).
        _flush_partial_manifest()

        # Persist into RunMemory so the next run can learn from this one.
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
            context_path=str(context_path),
            plan_path=str(plan_path),
            manifest_path=str(manifest_path),
            report_path=str(report_path),
            manuscript_path=str(bound_path),
            evidence_count=len(evidence.records()),
            findings_count=len(findings),
        )
        # T3.5 — record this run's outputs so a future identical
        # invocation can short-circuit. We only cache on a clean
        # finish; aborted runs are written through `_finalise_aborted`
        # which deliberately does not touch the cache.
        if cache_key is not None:
            self._record_cache_hit(cache_key, result)
        _emit_progress(
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
    # Helpers
    # ------------------------------------------------------------------

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
        if llm is None or isinstance(llm, MockLLMClient):
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
            "stop_after_analysis": bool(stop_after_analysis),
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
    # Generic table outputs from any step.
    ("", "table_one.csv"): ("table_one",),
    ("", "missingness.csv"): ("missingness",),
    ("", "sofa_strata.csv"): ("sofa_strata",),
    ("", "primary_association.csv"): ("primary_association_table",),
    # Figures.
    ("", "sofa_strata.png"): ("sofa_strata_figure",),
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
