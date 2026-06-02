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
import asyncio
import csv
import json
import logging
import math
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

logger = logging.getLogger(__name__)

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
from .contracts import _ExecutePhaseResult, _PlanPhaseResult, _WritePhaseResult
from .concept_dict_audit import (
    assert_dict_matches as assert_concept_dict_matches,
    write_concept_dict_fingerprint,
)
from .cohort_schema import ensure_cohort_definition, write_locked_cohort_definition
from .robustness_panel import ensure_robustness_specs, write_locked_robustness_specs
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

from .evidence import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    _coerce_enforcement_mode,
)
from .experience import (
    ExperienceBank,
    ExperienceRecord,
    mine_experience_from_run,
)
from .manuscript_post import (
    bind_numeric_values,
    _demote_unresolved_evidence_placeholders,
    _first_resolvable_name,
    _remove_tbd_sentences,
    _repair_common_writer_placeholders,
)
from .summary_repair import (
    _extract_last_json_object,
    _salvage_minimal_contract_step_summary,
    _salvage_named_json_step_summary,
    _salvage_stdout_json_step_summary,
)
from .code_repair import (
    _KEYERROR_NOT_IN_INDEX_RE,
    _NAME_ERROR_HELPER_RE,
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    _extract_missing_index_columns,
    _generic_clustering_fallback_code,
    _infer_generic_v15_fallback_key,
    _patch_json_dump_numpy_key_sanitizer,
    _patch_primary_predictor_into_design_matrix,
    _strip_columns_from_list_literals,
)
from .scalar_utils import (
    _coerce_scalar,
    _expected_numeric_annotations_for_step,
    _first_numeric_effect_from_text,
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)
from .pipeline_cross_db import (
    _build_replication_notes,
    _extract_cross_database_run_summary,
    _literature_provenance_note,
    _render_cross_database_comparison_markdown,
    _render_cross_database_summary_markdown,
    _render_cross_database_validation_report,
)
from .pipeline_primary_effect import (
    _extract_primary_effect_row,
    _infer_primary_predictor_from_run_dir,
    _primary_effect_candidate_score,
)
from .pipeline_writer_aux import (
    _preferred_writer_evidence_names,
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
    _resolve_writer_aux_path,
    _summarise_primary_association_table,
    _summarise_sofa_zero_audit,
    _summarise_table_one_rows,
)
from .plan_utils import (
    _ensure_publication_figure_step_in_plan,
    _enforce_advanced_plan_contract,
    _infer_primary_predictor_from_context,
    _parent_step_id_for_figure_step,
    _predictor_tokens,
    _preserve_figure_steps_after_replan,
    _question_primary_predictor_is_vasopressor_or_unknown,
    _research_question_implies_figure,
    _split_table_and_figure_outputs_in_plan,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_expects_figure,
    _step_produces_figure,
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
    dedupe_findings,
)
from .visual_qa import VLMVisualQAAdapter, VisualQAAuditor


from .pipeline_package import (
    _concept_dictionary_manifest_fields,  # noqa: F401
    _render_cost_summary,  # noqa: F401
)


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
        submission_profile_name: Optional[str] = None,
        submission_profile_version: Optional[str] = None,
        submission_profile_locked_at: Optional[str] = None,
        expected_concept_dict_sha: Optional[str] = None,
        expected_sofa2_dict_sha: Optional[str] = None,
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
        max_total_steps: int = 12,
        max_numeric_claims_per_step: int = 100,
        writer_digest_widened: bool = False,
        writer_digest_secondary_cap_per_step: int = 20,
        enable_experience_bank: bool = False,
        experience_bank_path: Optional[Union[str, Path]] = None,
        experience_bank_top_k: int = 5,
        experience_bank_min_similarity: float = 0.2,
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
        self._submission_profile_name = submission_profile_name
        self._submission_profile_version = submission_profile_version
        self._submission_profile_locked_at = submission_profile_locked_at
        self._expected_concept_dict_sha = expected_concept_dict_sha
        self._expected_sofa2_dict_sha = expected_sofa2_dict_sha
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
        # 0 / None means "no cap". Anything positive enforces the cap in
        # the replanner overflow guard (see pipeline_execute.py).
        self._max_total_steps = (
            int(max_total_steps) if max_total_steps and max_total_steps > 0 else 0
        )
        self._max_numeric_claims_per_step = (
            int(max_numeric_claims_per_step)
            if max_numeric_claims_per_step and max_numeric_claims_per_step > 0
            else 0
        )
        # Phase-1 writer-digest widening (default off). When True, the
        # writer's evidence digest is augmented with a "secondary
        # numbers" block enumerating NumericClaim entries outside the
        # ``WRITER_DIGEST_PREFERRED_KEYS`` primary subset. The binder
        # already accepts these; the flag controls only what the
        # writer SEES. See pipeline_writer_aux._render_writer_evidence_digest_v2.
        self._writer_digest_widened = bool(writer_digest_widened)
        self._writer_digest_secondary_cap_per_step = max(
            0, int(writer_digest_secondary_cap_per_step)
        )
        # Phase-1 cross-run experience bank (default off). When True,
        # the planner sees retrieved hints in its context block and
        # the post-run reflector mines + writes back records via the
        # deterministic helper in experience.py. The bank is read in
        # ``_retrieve_experience_hints`` and written in
        # ``_reflect_and_persist_experience``.
        self._enable_experience_bank = bool(enable_experience_bank)
        self._experience_bank_path: Optional[Path] = (
            Path(experience_bank_path) if experience_bank_path else None
        )
        self._experience_bank_top_k = max(0, int(experience_bank_top_k))
        self._experience_bank_min_similarity = float(experience_bank_min_similarity)
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
        concept_fingerprint_path = run_dir / "concept_dict_fingerprint.json"
        concept_fingerprint = write_concept_dict_fingerprint(concept_fingerprint_path)
        assert_concept_dict_matches(
            concept_fingerprint,
            expected_concept_dict_sha=self._expected_concept_dict_sha,
            expected_sofa2_dict_sha=self._expected_sofa2_dict_sha,
            mode="strict",
        )
        if evidence.get("concept_dict_fingerprint") is None:
            evidence.register_file(
                kind="log",
                description="SHA-256 fingerprint for EasyICU concept dictionaries.",
                source_path=concept_fingerprint_path,
                evidence_id="concept_dict_fingerprint",
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

        experience_digest_text: Optional[str] = None
        experience_hits = self.retrieve_experience_hints(
            research_question=question,
            database=database,
        )
        if experience_hits:
            bank = self._experience_bank()
            bank_record_count = len(bank.records()) if bank is not None else len(experience_hits)
            retired_count = max(0, bank_record_count - len(experience_hits))
            lines = [
                "# Experience Hints",
                "",
                "Only deterministic, audit-safe experience buckets are shown: "
                "concept_usage_hint and failure_counter_example. These are hints, "
                "not ICU rules; cohort definitions and clinical rules still come "
                "from the concept dictionary and validators.",
                "",
                f"Selected {len(experience_hits)} card(s); withheld/retired "
                f"{retired_count} card(s) below the retrieval threshold or top-k.",
            ]
            for idx, (record, score) in enumerate(experience_hits, start=1):
                lines.extend(
                    [
                        "",
                        f"## Card {idx}: {record.kind}",
                        f"- score: {score:.3f}",
                        f"- database: {record.database}",
                        f"- cohort: {record.cohort_name}",
                        f"- produced_by: {record.producer_run_id or 'unknown'}",
                        f"- reason: lexical overlap with the current question"
                        + (
                            " plus same-database boost"
                            if database and record.database == database
                            else ""
                        ),
                        f"- summary: {record.summary}",
                    ]
                )
            experience_digest_text = "\n".join(lines) + "\n"
            experience_path = run_dir / "experience_hints.md"
            experience_path.write_text(experience_digest_text, encoding="utf-8")
            if evidence.get("experience_hints") is None:
                evidence.register_file(
                    kind="log",
                    description="Audit log of cross-run experience cards fed to the planner.",
                    source_path=experience_path,
                    evidence_id="experience_hints",
                    producer="experience_bank",
                    generation_mode="system",
                )

        agent_context = context
        planner_notes: List[str] = []
        if memory_digest_text:
            planner_notes.append("RunMemory digest for planner:\n" + memory_digest_text)
        if experience_digest_text:
            planner_notes.append(
                "Experience hints for planner:\n" + experience_digest_text
            )
        if planner_notes:
            note = "\n\n".join(planner_notes)
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

        # C1 (pilot 20260515 fix): cap initial plan size for the same
        # reason the replanner is capped (see pipeline_execute.py).
        # Truncation happens AFTER figure-step guard so a publication
        # figure step is not accidentally dropped.
        cap = self._max_total_steps
        if cap > 0 and len(plan.steps) > cap:
            dropped = [s.step_id for s in plan.steps[cap:]]
            plan = plan.model_copy(update={"steps": list(plan.steps[:cap])})
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        f"Initial plan had {len(dropped) + cap} steps; "
                        f"truncated to max_total_steps={cap}. Dropped: "
                        f"{', '.join(dropped[:6])}"
                        + (" ..." if len(dropped) > 6 else "")
                    ),
                    detail={"dropped_step_ids": dropped, "cap": cap},
                )
            )
        plan = ensure_cohort_definition(plan)
        plan = ensure_robustness_specs(plan)
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
        write_locked_cohort_definition(
            run_dir=run_dir,
            plan=plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
        )
        write_locked_robustness_specs(
            run_dir=run_dir,
            plan=plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
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
    ) -> "_ExecutePhaseResult":
        """Delegate to :mod:`pipeline_execute`.

        The 1500-line loop body is in :mod:`pipeline_execute` so this
        file does not have to host both the orchestration shell and the
        execute-phase guts. Late-imported to keep ``import pipeline``
        free of a cycle.
        """
        from .pipeline_execute import run_execute_phase
        return run_execute_phase(
            self,
            plan_result=plan_result,
            cohort_path=cohort_path,
            run_dir=run_dir,
            run_id=run_id,
            skill_obj=skill_obj,
            notes=notes,
            emit_progress=emit_progress,
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
        force_writer_probe: bool = False,
    ) -> _WritePhaseResult:
        """Delegate to :mod:`pipeline_write`."""
        from .pipeline_write import run_write_phase

        return run_write_phase(
            self,
            plan_result=plan_result,
            execute_result=execute_result,
            run_dir=run_dir,
            run_id=run_id,
            stop_after_analysis=stop_after_analysis,
            manuscript_title=manuscript_title,
            manuscript_authors=manuscript_authors,
            run_language=run_language,
            emit_progress=emit_progress,
            force_writer_probe=force_writer_probe,
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
        """Delegate to :mod:`pipeline_package`."""
        from .pipeline_package import finalise_success

        return finalise_success(
            self,
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
            emit_progress=emit_progress,
        )

    # ------------------------------------------------------------------
    # Cross-run experience bank (Phase-1, Commit 3 — opt-in)
    # ------------------------------------------------------------------

    def _experience_bank(self) -> Optional[ExperienceBank]:
        """Lazy-construct the ExperienceBank from the configured path.

        Returns ``None`` when the feature is disabled or no path was
        configured. Callers must treat the bank as optional.
        """
        if not self._enable_experience_bank or self._experience_bank_path is None:
            return None
        return ExperienceBank(path=self._experience_bank_path)

    def retrieve_experience_hints(
        self,
        *,
        research_question: str,
        database: Optional[str] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        """Top-k experience records most lexically similar to the question.

        Returns an empty list when the bank is disabled, missing, or
        contains no records above the configured similarity floor.
        Surface the returned ``summary`` lines verbatim in the planner
        prompt — the bank guarantees they are self-contained
        sentences. Callers that need the raw records (e.g. for
        debugging) can read ``ExperienceBank(path).records()``
        directly.
        """
        bank = self._experience_bank()
        if bank is None:
            return []
        return bank.retrieve(
            research_question=research_question,
            database=database,
            top_k=self._experience_bank_top_k,
            min_similarity=self._experience_bank_min_similarity,
        )

    def reflect_and_persist_experience(
        self,
        *,
        run_dir: Path,
        context: ResearchContext,
        database: str,
        cohort_name: str,
    ) -> List[ExperienceRecord]:
        """Mine experience records from a completed run and write them back.

        Reads ``run_status.json`` from ``run_dir`` for the gates +
        findings + superseded-error partition. Returns the records
        that were registered (after dedup against the existing bank);
        an empty list when the feature is disabled or the run dir
        does not yet have a run_status.

        Idempotent on ``(kind, summary)``: re-running over the same
        run_dir does not duplicate records, only refreshes
        ``produced_at`` / ``producer_run_id``.
        """
        bank = self._experience_bank()
        if bank is None:
            return []
        run_status_path = Path(run_dir) / "run_status.json"
        if not run_status_path.exists():
            return []
        try:
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "experience-bank: run_status.json at %s could not be read: %s",
                run_status_path, exc,
            )
            return []
        gates = run_status.get("gates") or run_status.get("readiness_gates") or {}
        findings = run_status.get("findings") or []
        superseded_errors = (
            gates.get("superseded_errors")
            or run_status.get("superseded_errors")
            or []
        )
        plan_step_ids = []
        for step in run_status.get("plan_steps") or []:
            sid = step.get("step_id") if isinstance(step, dict) else None
            if sid:
                plan_step_ids.append(str(sid))
        if not plan_step_ids:
            # Fall back to per_step_records if plan_steps not surfaced.
            for record in run_status.get("per_step_records") or []:
                sid = record.get("step_id") if isinstance(record, dict) else None
                if sid:
                    plan_step_ids.append(str(sid))
        records = mine_experience_from_run(
            research_question=context.research_question,
            database=database,
            cohort_name=cohort_name,
            gates=gates,
            findings=findings,
            superseded_errors=superseded_errors,
            plan_step_ids=plan_step_ids,
            producer_run_id=Path(run_dir).name,
        )
        bank.extend(records)
        return records

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
        force_writer_probe: bool = False,
        _use_graph: bool = False,
    ) -> PipelineResult:
        """Run the explicit Plan → Execute → Write phases for one cohort.

        Pass ``_use_graph=True`` (or call :meth:`run_with_graph`) to
        dispatch the phases through the opt-in langgraph wrapper in
        :mod:`easyicu.research_agent.graph`. Behaviour is identical
        either way; the graph path is a PoC for future branching /
        checkpointing work.
        """
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

        def _plan_invoker():
            return self._run_plan_phase(
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

        def _provenance_hook(plan_result):
            # O27 — Raw EHR provenance. Hash the cohort parquet and any
            # user-supplied source files and register as evidence so the
            # manuscript's provenance chain goes: raw EHR -> cohort ->
            # analysis artefacts -> manuscript. Only runs after the plan
            # phase succeeded.
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

        def _execute_invoker(plan_result):
            return self._run_execute_phase(
                plan_result=plan_result,
                cohort_path=cohort_path,
                run_dir=run_dir,
                run_id=run_id,
                skill_obj=skill_obj,
                notes=notes,
                emit_progress=_emit_progress,
            )

        def _write_invoker(plan_result, execute_result):
            return self._run_write_phase(
                plan_result=plan_result,
                execute_result=execute_result,
                run_dir=run_dir,
                run_id=run_id,
                stop_after_analysis=stop_after_analysis,
                manuscript_title=manuscript_title,
                manuscript_authors=manuscript_authors,
                run_language=run_language,
                emit_progress=_emit_progress,
                force_writer_probe=force_writer_probe,
            )

        def _finalise_invoker(plan_result, execute_result, write_result):
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

        if _use_graph:
            from .graph import build_pipeline_graph

            graph = build_pipeline_graph(
                plan_invoker=_plan_invoker,
                execute_invoker=_execute_invoker,
                write_invoker=_write_invoker,
                finalise_invoker=_finalise_invoker,
                provenance_hook=_provenance_hook,
            )
            final_state = graph.invoke({})
            return final_state["final_result"]

        plan_result = _plan_invoker()
        if plan_result.aborted_result is not None:
            return plan_result.aborted_result
        _provenance_hook(plan_result)
        execute_result = _execute_invoker(plan_result)
        write_result = _write_invoker(plan_result, execute_result)
        return _finalise_invoker(plan_result, execute_result, write_result)

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

    def run_with_graph(self, **kwargs: Any) -> PipelineResult:
        """Opt-in PoC: dispatch the phases through a langgraph StateGraph.

        Requires the ``agentic`` extra (``pip install easyicu[agentic]``).
        Behaviour is identical to :meth:`run`; see
        :mod:`easyicu.research_agent.graph` for the wrapper design.
        """
        kwargs.pop("_use_graph", None)
        return self.run(_use_graph=True, **kwargs)

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
        """Delegate to :mod:`pipeline_package`."""
        from .pipeline_package import finalise_aborted

        return finalise_aborted(
            self,
            run_id=run_id,
            run_dir=run_dir,
            context=context,
            context_path=context_path,
            evidence=evidence,
            findings=findings,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _has_figure_exports(out_dir: Path) -> bool:
    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    return any(
        path.is_file() and path.suffix.lower() in figure_suffixes
        for path in out_dir.iterdir()
    )


def _promote_sibling_figure_exports(*, out_dir: Path) -> Optional[str]:
    """Promote figure files written beside ``outputs/`` into ``outputs/``.

    Some generated scripts treat ``STEP_OUT_DIR`` as a filename stem and
    write ``outputs.svg`` / ``outputs.png`` beside the canonical
    ``outputs/`` directory. The execution contract only registers files inside
    ``outputs/``, so normalize that common layout before declaring the
    publication figure missing.
    """
    parent = out_dir.parent
    source_stem = out_dir.name
    figure_suffixes = (".pdf", ".png", ".svg", ".tiff", ".tif", ".pptx")
    figure_sources = [
        parent / f"{source_stem}{suffix}"
        for suffix in figure_suffixes
        if (parent / f"{source_stem}{suffix}").is_file()
    ]
    if not figure_sources:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    target_stem = "publication_figure"
    exported_figure_files: List[str] = []
    for source in figure_sources:
        target = out_dir / f"{target_stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        exported_figure_files.append(target.name)

    contract_source = parent / f"{source_stem}.figure_contract.json"
    if contract_source.is_file():
        shutil.copy2(contract_source, out_dir / f"{target_stem}.figure_contract.json")

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    summary["publication_figure_rescue"].update(
        {
            "mode": "sibling_outputs_stem",
            "source_stem": source_stem,
            "source_dir": str(parent),
        }
    )
    summary["figure_files"] = sorted(exported_figure_files)
    if exported_figure_files:
        summary["figure_path"] = sorted(exported_figure_files)[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "sibling_figure_exports_promote_v1"


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
            out.extend(["outcome_rate", "mortality_rate", "robustness_summary"])
            if _step_summary_has_primary_effect(artefact):
                out.append("primary_association")
        if (
            ("table_one" in step_id.lower() or "table:table_one" in expected)
            and not (artefact.parent / "table_one.csv").exists()
        ):
            out.append("table_one")
        if "report" in step_id.lower() and _step_summary_has_any_key(
            artefact,
            ("mortality_rate", "outcome_rate", "death_rate", "icu_mortality_rate"),
        ):
            out.extend(
                [
                    "outcome_rate",
                    "mortality_rate",
                    "cohort_summary",
                ]
            )
    for (step_substr, basename), aliases in _SEMANTIC_ALIAS_MAP.items():
        if basename != artefact.name:
            continue
        if step_substr and step_substr not in (step.step_id or "").lower():
            continue
        out.extend(aliases)
    return out


def _step_summary_has_primary_effect(path: Path) -> bool:
    """Return True when an existing step_summary has a finite primary effect.

    Robustness/missingness steps frequently mention "robustness" in their
    intent but are not themselves primary association artefacts. The
    ``primary_association`` alias should therefore be attached only when the
    registered summary actually contains an effect estimate. For synthetic
    tests that pass a non-existent path, preserve the historical aliasing
    behaviour by returning True.
    """

    if path.name != "step_summary.json" or not path.exists():
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    def _finite(value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    direct_keys = {
        "primary_or",
        "odds_ratio",
        "adjusted_or",
        "estimate",
        "primary_association_estimate",
        "primary_point_estimate",
    }
    for key in direct_keys:
        value = payload.get(key)
        if _finite(value):
            return True
        if isinstance(value, dict) and any(_finite(v) for v in value.values()):
            return True
    for value in payload.values():
        if isinstance(value, dict):
            association = value.get("primary_association")
            if isinstance(association, dict) and any(
                _finite(association.get(k))
                for k in ("odds_ratio", "or", "estimate", "adjusted_or")
            ):
                return True
    return False


def _step_summary_has_any_key(path: Path, keys: Sequence[str]) -> bool:
    """Return True when an existing step_summary contains any requested key."""

    if path.name != "step_summary.json" or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in keys)


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


__all__ = ["ResearchAgentPipeline"]
