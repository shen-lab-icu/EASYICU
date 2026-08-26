"""Typed, immutable configuration for :class:`ResearchAgentPipeline`.

``PipelineConfig`` is the sole declarative source for pipeline behavior:

* IDEs and type-checkers can autocomplete / validate construction;
* tests can build a baseline config and override only what they care
  about (``config = PipelineConfig.default().with_overrides(...)``);
* configuration can be loaded from YAML / TOML via
  :meth:`PipelineConfig.from_kwargs`, with unknown or misspelled keys
  rejected instead of silently ignored;
* live collaborators stay out of serialization and are injected separately
  through :class:`~easyicu.research_agent.orchestration.services.PipelineServices`.

The historical flat ``ResearchAgentPipeline(workdir=..., ...)`` call remains
as a deprecation adapter. New code should construct ``PipelineConfig`` and
``PipelineServices`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Union

from ..authority.secret_redaction import is_sensitive_key, string_contains_secret
from ..planning.cohort_contract import CohortSelectionMode
from ..providers.prompt_budget import DEFAULT_MAX_PROMPT_TOKENS


class PipelineConfigRecoveryError(ValueError):
    """A pipeline configuration cannot be persisted for exact recovery."""


_RECOVERY_BLOCKED_FIELDS = frozenset(
    {
        "pubmed_api_key",
        "tavily_api_key",
        # Runner kwargs are deliberately opaque and may carry headers,
        # connection strings, mounts, or other live-host details.  A host that
        # needs durable review recovery must keep them out of the declarative
        # checkpoint rather than guessing which nested values are safe.
        "runner_kwargs",
    }
)

_RECOVERY_SECRET_KEY_PARTS = (
    "api_key",
    "authorization",
    "client_secret",
    "connection_string",
    "cookie",
    "database_url",
    "dsn",
    "password",
    "private_key",
    "proxy_authorization",
    "refresh_token",
    "access_token",
)


def _recovery_key_is_secret(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _RECOVERY_SECRET_KEY_PARTS)


def _render_recovery_value(value: Any, *, path: str) -> Any:
    """Render one config value losslessly or reject unsafe opaque state.

    This is intentionally distinct from :meth:`canonical_payload`.  The
    canonical payload is a provenance projection and hashes values below any
    key named ``key``; that is correct for broad log redaction but not
    reversible (literature citation keys are ordinary scientific identities).
    Durable recovery instead preserves schema-owned identities and refuses
    actual credential-shaped values outright.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if string_contains_secret(value):
            raise PipelineConfigRecoveryError(
                f"pipeline_config_recovery_secret_value:{path}"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        rendered: Dict[str, Any] = {}
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key)
            item_path = f"{path}.{key}" if path else key
            if _recovery_key_is_secret(key) and item not in (None, "", (), [], {}):
                raise PipelineConfigRecoveryError(
                    f"pipeline_config_recovery_secret_field:{item_path}"
                )
            rendered[key] = _render_recovery_value(item, path=item_path)
        return rendered
    if isinstance(value, (list, tuple)):
        return [
            _render_recovery_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        rendered = [
            _render_recovery_value(item, path=f"{path}[]") for item in value
        ]
        return sorted(
            rendered,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    raise PipelineConfigRecoveryError(
        "pipeline_config_recovery_unsupported_value:"
        f"{path}:{type(value).__module__}.{type(value).__qualname__}"
    )


def step_provider_call_entitlement(
    *,
    max_code_repair_attempts: int,
    max_step_llm_repair_attempts: int,
    llm_concept_audit_enabled: bool,
) -> int:
    """Return the provider calls one step may legitimately spend.

    The arithmetic lived only in a comment beside ``max_step_provider_calls``,
    which is how the three numbers drifted apart before: each is edited for its
    own reason, and nothing recomputes their sum. This is that sum, in one
    place, so a change to any term is felt by the other two.

    The generation term was a literal ``1`` while ``coder_generation.py``
    declares ``MAX_INITIAL_GENERATION_ATTEMPTS = 2`` -- "at most one audited
    regeneration" -- so the host's own documented happy path could spend two
    where this sum funded one. That is the same drift this function exists to
    stop, in its own first term.
    """

    from ..agents.coder_generation import MAX_INITIAL_GENERATION_ATTEMPTS

    return (
        MAX_INITIAL_GENERATION_ATTEMPTS  # initial generation, per its own policy
        + max(0, int(max_code_repair_attempts))
        + max(0, int(max_step_llm_repair_attempts))
        # execution/phase.py reserves the final call for the concept audit
        # (``reserved_final_category``) only when that auditor is enabled.
        + (1 if llm_concept_audit_enabled else 0)
    )


def assert_step_provider_budget_funds_its_repairs(
    *,
    max_step_provider_calls: int,
    max_code_repair_attempts: int,
    max_step_llm_repair_attempts: int,
    llm_concept_audit_enabled: bool,
    allow_underfunded: bool = False,
) -> None:
    """Refuse a budget that cannot pay for the repairs the same config promises.

    A step that exhausts its provider budget mid-repair fails, and it fails the
    way a scientifically broken step fails — so the run reports an analysis
    problem that is really an accounting one. Catching it at construction is
    the only point where the two are still distinguishable.

    Deliberate under-funding stays available through ``allow_underfunded``;
    what is refused is under-funding nobody decided on.
    """

    from ..agents.coder_generation import MAX_INITIAL_GENERATION_ATTEMPTS

    granted = max(0, int(max_step_provider_calls))
    entitled = step_provider_call_entitlement(
        max_code_repair_attempts=max_code_repair_attempts,
        max_step_llm_repair_attempts=max_step_llm_repair_attempts,
        llm_concept_audit_enabled=llm_concept_audit_enabled,
    )
    if allow_underfunded or granted >= entitled:
        return
    raise ValueError(
        f"max_step_provider_calls={granted} cannot fund the repair policy this "
        f"configuration declares: {MAX_INITIAL_GENERATION_ATTEMPTS} initial "
        f"generation attempt(s) + "
        f"{max(0, int(max_code_repair_attempts))} code repairs + "
        f"{max(0, int(max_step_llm_repair_attempts))} LLM repairs"
        + (" + 1 reserved concept audit" if llm_concept_audit_enabled else "")
        + f" = {entitled} calls. Raise max_step_provider_calls to at least "
        f"{entitled}, lower the repair attempts to match, or pass "
        "allow_underfunded_step_provider_calls=True to declare that the "
        "shortfall is intended."
    )


def _deep_freeze(value: Any) -> Any:
    """Return an immutable view of a plain data container, recursively.

    Only the exact builtin containers are converted. A subclass (a pydantic
    model, a ``defaultdict`` a caller relies on, a numpy array) keeps its
    identity and behaviour: this exists to stop a shared ``runner_kwargs``
    dict being edited after the config was hashed, not to re-type the
    collaborators a run was handed.
    """

    if type(value) is dict:
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    if type(value) is set:
        return frozenset(_deep_freeze(item) for item in value)
    if type(value) is tuple:
        frozen = tuple(_deep_freeze(item) for item in value)
        return frozen if frozen != value else value
    return value


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable declarative settings for one pipeline.

    Defaults intentionally match the legacy keyword adapter so both
    construction paths produce identical behavior.

    The dataclass is frozen: the config is shared across the planner,
    execution and reporting layers and is hashed into the run's authority
    digest, so a post-construction mutation would leave the recorded
    configuration disagreeing with the one that actually ran (and, on resume,
    reload a different config than the checkpoint was taken under). Use
    :meth:`with_overrides` to derive a changed copy.

    ``frozen=True`` alone binds only the field *references*, so a caller
    holding the same ``runner_kwargs`` dict could still mutate the config after
    it was hashed into the run authority. :meth:`__post_init__` therefore also
    freezes the plain data containers: every ``dict`` becomes a read-only
    mapping and every ``list``/``set`` a tuple/frozenset, recursively.
    Provider clients, factories, adapters, review control planes, and plugin
    registries belong in ``PipelineServices`` instead.
    """

    # --- required -------------------------------------------------------
    workdir: Union[str, Path]

    # --- core runtime ---------------------------------------------------
    # One generated-code attempt. The former 300 s was below the honest cost of
    # the analyses this agent is asked to run — a Cox fit plus PH diagnostics on
    # a six-figure cohort, a bootstrap stability sweep, a propensity match — so
    # those steps could never pass on this path however the script was written.
    # Timeouts are now terminal (see execution/failure_classification.py), which
    # removes the retry multiplication that used to spend the whole repair
    # budget re-running the same overlong computation; per-step worst-case wall
    # clock is lower at 900 s once than it was at 300 s repeated.
    timeout_seconds: float = 900.0
    # Registered deterministic standards can execute a planner-owned workload
    # (for example, a fixed resampling design) that is intentionally much
    # longer than one generated-code attempt. Keep that bounded workload on a
    # separate wall-clock budget so raising it never gives ordinary coder
    # scripts additional runtime.
    standard_executor_timeout_seconds: float = 3_600.0
    python_executable: Optional[str] = None

    # --- feature toggles -------------------------------------------------
    enable_literature: bool = True
    enable_visual_qa: bool = True
    # Nature Figure is the public skill contract over the existing claim-first,
    # code-backed publication renderer. Keep the legacy field name as the
    # stable API used by CLI and benchmark callers.
    enable_publication_figure_skill: bool = True
    # Nature Writing is an independent publication skill so a host can unplug
    # its prose policy without weakening evidence/numeric audits.
    enable_nature_writing_skill: bool = True
    # Optional host-compiled, content-digest-bound user extension snapshot.
    # Only the writing advisory is consumed by the pipeline; MCP tools remain
    # in the receipt and are never promoted into scientific evidence here.
    extension_activation: Optional[Dict[str, Any]] = None
    enable_vlm_visual_qa: Optional[bool] = None
    # Uploading a rendered figure is a separate decision from authorizing the
    # provider: the image can carry per-patient marks, small-cell strata or
    # local paths that the text outbound projection would have stripped.
    allow_external_figure_upload: bool = False
    enable_llm_concept_audit: Optional[bool] = None
    enable_memory: bool = True
    enable_latex: bool = True
    # Interactive hosts can require an explicit, digest-bound operator review
    # of every generated plan even when no scientific validator emitted an
    # error.  The default remains non-interactive for CLI/benchmark callers;
    # the Guided Web Copilot enables this because its product contract is
    # plan -> user confirmation -> execution.
    require_human_plan_review: bool = False
    # Opt-in next-stage contract: reviewed comparator full text/supplements
    # must shape all seven design dimensions before Provider planning, and the
    # selected design must record its exact adopt/adapt/diverge decisions.
    require_literature_design_authority: bool = False
    # A diagnostic Planner-only run may persist and expose the exact review
    # checkpoint, but no caller may resume it into Execute.
    planner_only: bool = False
    # Formal interactive runs must not spend execution budget on a capability
    # whose registered scientific ceiling is analysis-only. Diagnostic callers
    # may leave this disabled and retain the honest lower claim ceiling.
    require_reportable_scientific_capability: bool = False

    # --- evidence enforcement -------------------------------------------
    # "soft" (default): unsupported sentences are filtered and unresolved
    # placeholders are demoted to comments; warnings surface in findings.
    # "strict": EvidenceStore raises EvidenceEnforcementError, aborting the
    # run, so CI / final submission cannot ship a silently repaired manuscript.
    evidence_enforcement_mode: str = "soft"
    latex_venue_template: str = "article"
    # Interactive previews are deliberately marked as drafts. CLI and formal
    # benchmark callers keep the historical unmarked scaffold unless their
    # host opts in explicitly.
    latex_draft_watermark: bool = False
    manuscript_language: str = "en"

    # --- context / ablation knobs ---------------------------------------
    # Historical untyped ablation only. Typed materialized exports require the
    # ICU-aware ResearchContext v2 authority path and fail closed at run entry.
    disable_icu_context: bool = False
    context_top_k: Optional[int] = None

    # --- code-repair / determinism --------------------------------------
    # 3, not 1: cheap/flaky hosted models (e.g. deepseek-v4-flash) repeatedly
    # emit syntactically or semantically broken analysis code (SyntaxError,
    # AttributeError on a renamed column). A single repair attempt routinely
    # ran out before the model produced runnable code, fail-closing an
    # otherwise valid analysis. This budget is per failure-class (success-path
    # contract/visual repairs and runtime-crash repairs each get their own),
    # so genuinely broken steps still terminate.
    max_code_repair_attempts: int = 3
    # Cross-gate stop-loss. Concept, contract, visual, and runtime repair used
    # to each receive the per-class budget above, multiplying model calls for
    # one local defect. Keep legacy per-class limits for compatibility, but cap
    # actual LLM repair calls across the whole step.
    max_step_llm_repair_attempts: int = 2
    # Real coder/concept-audit provider attempts share one crash-safe budget,
    # including initial generation, transport/fallback retries, compatibility
    # repair, patch, and full-rewrite fallback. Successful first-pass steps do
    # not spend the headroom. Full rewrites and transport retries still consume
    # the same monotonic stop-loss rather than receiving a hidden budget.
    #
    # The floor is the sum of what the step is entitled to spend:
    #   1 generation
    # + max_code_repair_attempts       (3)
    # + max_step_llm_repair_attempts   (2)
    # + 1 reserved final concept audit, when the LLM concept audit is enabled
    #     (execution/phase.py reserves the last call for that category)
    #   = 7
    # At exactly 7 the step is entitled to spend every call it is granted, so a
    # single structured-retry on a malformed response silently costs a repair
    # attempt instead of costing headroom. Two spare calls keep transport noise
    # from being charged to the science.
    max_step_provider_calls: int = 9
    # Deliberately running a step on less than its repair policy costs is a
    # legitimate choice — exercising the stop-loss, or capping spend on a
    # throwaway run. It must be a choice. Without this flag the shortfall is
    # invisible until a step fails, and a step that was never funded to finish
    # is indistinguishable in the record from one whose science failed.
    allow_underfunded_step_provider_calls: bool = False
    # How large one outbound prompt is expected to grow, in provider-metered
    # tokens. This is a *local design ceiling*, not the model's context window:
    # nothing in this package knows that window, and the current provider does
    # not report one, so nothing here guesses it. The default clears the
    # largest prompt this system has ever actually produced (26,040 tokens) and
    # stays far below any current model, so it catches a runaway projection
    # without being the routine binding constraint. Raise it when a payload is
    # legitimately larger; the change lands in the run authority digest because
    # this config is hashed into it. See providers/prompt_budget.py.
    max_prompt_tokens_per_call: int = DEFAULT_MAX_PROMPT_TOKENS
    enable_deterministic_code_fallback: bool = False
    enable_deterministic_planner_fallback: bool = False
    # The legacy Planner emits the complete executable DAG in one response.
    # Progressive v2 emits a compact scientific skeleton and lets the host
    # compile exact products, levels, consumption contracts, and methods.
    planner_strategy: Literal["monolithic_v1", "progressive_v2"] = (
        "monolithic_v1"
    )
    # Explicit non-paper replay of one dependency-bound Progressive Planner
    # prefix. Both fields are required together; the terminal file SHA binds
    # the selected append-only chain before any provider call is made.
    development_progressive_resume_checkpoint_path: Optional[
        Union[str, Path]
    ] = None
    development_progressive_resume_checkpoint_sha256: Optional[str] = None
    # Explicit non-paper execution of one previously locked AnalysisPlan.  The
    # exact JSON bytes are digest-bound, then revalidated and shaped by the
    # current host before any step runs.  This is deliberately separate from
    # Progressive checkpoint replay: a changed Planner/compiler may invalidate
    # its prompt dependency while the already selected scientific plan remains
    # a valid execution input.
    development_locked_analysis_plan_path: Optional[Union[str, Path]] = None
    development_locked_analysis_plan_sha256: Optional[str] = None
    # Development canaries stop before another expensive Planner request once
    # any one of these exact-run limits is exhausted. Formal profiles cannot
    # enable this partial-run checkpointing envelope.
    development_planner_efficiency_max_calls: Optional[int] = None
    development_planner_efficiency_max_reported_tokens: Optional[int] = None
    development_planner_efficiency_max_wall_seconds: Optional[float] = None
    enable_deterministic_runner_repair: bool = True
    # --- literature search backends -------------------------------------
    enable_pubmed: bool = False
    pubmed_email: Optional[str] = None
    pubmed_api_key: Optional[str] = None
    # Optional digest-verified literature metadata supplied by an outer host
    # that already performed an explicitly authorized search.  The complete
    # payload is frozen and hashed into run authority; the pipeline validates
    # it as a LiteratureBundle before exposing any citation key to Planner.
    bound_preplan_literature: Optional[Dict[str, Any]] = None
    # Optional host-compiled instructions from an exact prior scientific plan
    # review. This is an immutable string rather than a mutable old plan: the
    # new run remains fresh, the value is hashed into run configuration, and
    # the host must prove the StudyContext digest has not changed before
    # supplying it. Only plan-owned findings may appear here.
    bound_plan_revision_contract: Optional[str] = None
    enable_tavily: bool = False
    tavily_api_key: Optional[str] = None
    tavily_retmax: int = 5
    tavily_include_domains: Optional[Sequence[str]] = None
    tavily_exclude_domains: Optional[Sequence[str]] = None

    # --- cohort cache (delegated to PipelineCache) -----------------------
    enable_cache: bool = False
    cache_dir: Optional[Union[str, Path]] = None

    # --- cost / reproducibility ------------------------------------------
    enable_cost_tracking: bool = False
    cost_price_table: Optional[Dict[str, Any]] = None
    # Optional outer stop-loss. These eight fields are all-or-none and are
    # enforced by a live ``TaskProviderHardStop`` service. They live in the
    # declarative config as well so run identity/checkpoints cannot omit the
    # exact limits or price assumptions used for a paid benchmark.
    max_provider_attempts_per_run: Optional[int] = None
    max_provider_attempts_per_batch: Optional[int] = None
    max_total_tokens_per_run: Optional[int] = None
    max_total_tokens_per_batch: Optional[int] = None
    max_estimated_cost_usd_per_batch: Optional[float] = None
    max_wall_clock_seconds_per_task: Optional[float] = None
    provider_input_cost_usd_per_million_tokens: Optional[float] = None
    provider_output_cost_usd_per_million_tokens: Optional[float] = None
    enable_reproducibility_envelope: bool = False
    llm_seed: Optional[int] = None
    execution_data_seed: Optional[int] = None
    execution_input_authority_sha256: Optional[str] = None
    envelope_include_previews: bool = False
    submission_profile_name: Optional[str] = None
    submission_profile_version: Optional[str] = None
    submission_profile_locked_at: Optional[str] = None
    expected_concept_dict_sha: Optional[str] = None
    expected_sofa2_dict_sha: Optional[str] = None

    # --- statistical safeguards -----------------------------------------
    enable_multiple_testing_correction: bool = True
    multiple_testing_alpha: float = 0.05
    enable_causal_audit: bool = True
    enable_reporting_checklist: bool = True
    reporting_checklist_names: Optional[Sequence[str]] = None
    # Authoritative benchmark task kind used for kind-specific reporting
    # checks. Optional for non-benchmark pipeline runs.
    task_kind: Optional[str] = None
    # Optional caller-owned population contract. The shared pipeline enforces
    # the generic typed mode; the caller decides which mode a task requires.
    required_primary_cohort_selection_mode: Optional[CohortSelectionMode] = None
    # Optional caller-reviewed trajectory execution projection. It remains
    # case-neutral in the shared engine: the caller supplies exact concepts and
    # numerical rules, while plan validation and deterministic executors bind
    # them to this immutable, run-hashed configuration.
    trajectory_scientific_runtime_authority: Optional[Dict[str, Any]] = None
    # Optional caller-reviewed current-run authority for non-trajectory cases.
    # It is mutually exclusive with the trajectory authority and is consumed
    # by a typed plan gate plus its deterministic owner runner.
    current_case_scientific_runtime_authority: Optional[Dict[str, Any]] = None
    scientific_runtime_projection_sha256: Optional[str] = None
    enable_reviewer_round: bool = True
    enable_fairness_subgroups: bool = False
    enable_hypothesis_generator: bool = False
    hypothesis_generator_top_k: int = 5
    enable_pdf_render: bool = False

    # --- execution shape -------------------------------------------------
    max_concurrent_steps: int = 1
    # Non-paper development acceleration. The full cohort is used for context,
    # planning, locked cohort selection, and QC; only then is the selected
    # analysis cohort deterministically sampled for execution. ``None`` keeps
    # the full cohort. Submission profiles must never enable this option.
    development_sample_size: Optional[int] = None
    development_sample_seed: int = 20260719
    # Explicitly non-paper Canonical9 development lane. This permits a resumed
    # diagnostic run to retain an auditable multi-image runtime lineage while
    # framework fixes are tested step-by-step. Submission profiles must never
    # enable it, and paper-facing runs still require one immutable runtime.
    development_diagnostic: bool = False
    enable_probe_step: bool = True
    enable_replanning: bool = True
    # Hard cap on plan size after any replanner revision. The replanner
    # can still revise existing steps in place; it just may not push the
    # total count above this. Set to 0 / None to disable (legacy behaviour).
    # Pilot run 20260515 saw the planner expand a simple SOFA-2 association to
    # 30 steps with 13 revisions before being killed at step 20; this cap is
    # the guard against that, and 16 is still far below where that run went.
    #
    # It was 12, which covered probe + cohort summary + 2-3 primary models +
    # 2-3 sensitivities + figure + interpretation — an association task. It did
    # not cover the harder families: a task declaring four required products
    # (prediction, survival, causal, trajectory each do) also needs cohort
    # definition, missingness, the primary model and robustness before those
    # four, and truncation is silent — it drops steps with a warning and the
    # run still completes and still scores. A cap that binds in normal
    # operation quietly shrinks the science instead of reporting a limit, so
    # the guard is set where only a runaway reaches it.
    max_total_steps: int = 16
    # --- replanning convergence guards (2026-06-11) ---------------------
    # The replanner runs after the probe, after an agent-authored progressive
    # step, or when a clean deterministic step explicitly requests revision. A
    # verbose model can return cosmetically-different but substantively
    # identical plans, each costing a full LLM call: the E1 20260611 real
    # run produced revisions 4-6 carrying an identical step DAG, and the
    # run was killed mid-step-7 before finishing. These two guards stop the
    # churn without removing genuine adaptivity.
    #   * ``max_consecutive_noop_replans`` — stop invoking the replanner
    #     once it returns this many revisions in a row whose substantive
    #     step DAG (step_id + method + expected_outputs) is unchanged.
    #   * ``max_replans`` — hard backstop on the total number of
    #     *substantive* revisions in a run. When the run reaches this cap
    #     without the plan converging it fails closed to ``diagnostic_only``
    #     (a runaway replan loop must not launder a manuscript). Default 6
    #     preserves legitimate repair headroom — a real run rarely needs
    #     more than a handful of substantive revisions — while killing the
    #     pathological churn (the E3 20260706 run replanned 9× over ~50 min
    #     and still failed). ``stabilization_mode`` tightens this to 3 so a
    #     fast primary-only iteration fails closed sooner.
    # 0 disables either guard (legacy behaviour).
    max_consecutive_noop_replans: int = 2
    max_replans: int = 6
    # Stabilization / primary-only iteration mode. When True the effective
    # replan budget is tightened to 3 (fail closed faster while debugging a
    # case), leaving full runs on the default budget of 6.
    stabilization_mode: bool = False
    # Hard cap on numeric-claim leaves registered per single step.
    # Prevents one step that dumps a full interaction matrix into
    # step_summary.json from creating hundreds of footnotes when its
    # numbers are referenced in the manuscript. Pilot run 20260515
    # had a step with 295 claims; 100 covers any realistic clinical
    # analysis without truncating real result quantities.
    max_numeric_claims_per_step: int = 100
    # --- writer namespace breadth ---------------------------------------
    # When False (default), the writer's evidence digest is the curated
    # ``preferred_keys`` tuple in reporting.writer_evidence. When True, the
    # digest is augmented with a "secondary numbers" block that
    # enumerates every NumericClaim that isn't already covered by the
    # primary block, capped per step. The binder
    # (``bind_numeric_values()``) already accepts numbers outside
    # preferred_keys via the full claim registry; this flag controls
    # only what the writer SEES, not what the binder ACCEPTS. The cap
    # below prevents prompt-bloat on steps that registered a large
    # number of leaves under ``max_numeric_claims_per_step``.
    #
    # Background: Phase-0 baseline comparison (May 2026) noted that
    # data-to-paper exposes every numeric in every source artefact to
    # the writer via auto-generated hypertargets; our writer was being
    # given the narrower ``preferred_keys`` subset. Widening parity-
    # tests as the "primary" subset, plus a "secondary" block, is the
    # Phase-1 step toward the wider writer namespace.
    writer_digest_widened: bool = False
    writer_digest_secondary_cap_per_step: int = 20

    # --- cross-run experience bank --------------------------------------
    # Phase-1 widening (Commit 3, May 2026). When enabled, the pipeline
    # reads the experience bank at planner-start (retrieving up to
    # ``experience_bank_top_k`` records lexically similar to the
    # current research question) and writes back any
    # ExperienceRecords mined from this run at completion. The bank
    # is a single JSONL file shared across all runs that point at
    # the same ``experience_bank_path`` — so multiple parallel runs
    # against the same path will see each other's hints once their
    # respective ``add`` calls return.
    #
    # The mined records are produced by a deterministic reflector
    # (no LLM call): see
    # ``easyicu.research_agent.learning.experience.mine_experience_from_run``.
    # The bank is opt-in because (i) it changes the planner's input
    # surface and (ii) the npj DM submission run does not depend on
    # experience-bank behaviour.
    enable_experience_bank: bool = False
    experience_bank_path: Optional[Union[str, Path]] = None
    experience_bank_top_k: int = 5
    experience_bank_min_similarity: float = 0.2
    enable_know_how: bool = False
    allow_curated_mvp_know_how: bool = False
    enable_coder_resources: bool = False
    enable_reviewed_memory: bool = False
    reviewed_memory_namespaces: Sequence[str] = ()
    enable_capability_workflow: bool = False
    expected_runner_image_digest: Optional[str] = None
    capability_request: Optional[Dict[str, Any]] = None
    capability_approval: Optional[Dict[str, Any]] = None
    capability_activation: Optional[Dict[str, Any]] = None
    know_how_paths: Sequence[Union[str, Path]] = ()
    know_how_top_k: int = 3
    know_how_min_score: float = 0.15

    # --- code runner ----------------------------------------------------
    runner_kind: str = "auto"
    runner_image: Optional[str] = None
    runner_network: str = "none"
    host_runner_authorized: bool = False
    runner_kwargs: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "PipelineConfig":
        """Build a config from keyword arguments.

        Unknown keys raise :class:`TypeError` so misspelled YAML, TOML, or
        argparse options cannot silently fall back to a pipeline default.
        """
        return cls(**kwargs)

    def __post_init__(self) -> None:
        for field_def in fields(self):
            value = getattr(self, field_def.name)
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, field_def.name, frozen)
        if (
            self.require_reportable_scientific_capability
            and not self.require_human_plan_review
        ):
            raise ValueError(
                "require_reportable_scientific_capability requires "
                "require_human_plan_review so the pre-execution gate cannot be skipped"
            )
        if self.require_literature_design_authority:
            if not self.enable_literature:
                raise ValueError(
                    "require_literature_design_authority requires enable_literature"
                )
            if not self.require_human_plan_review:
                raise ValueError(
                    "require_literature_design_authority requires "
                    "require_human_plan_review"
                )
            if self.planner_strategy != "progressive_v2":
                raise ValueError(
                    "require_literature_design_authority requires progressive_v2"
                )
        if self.planner_strategy not in {"monolithic_v1", "progressive_v2"}:
            raise ValueError(
                "planner_strategy must be 'monolithic_v1' or 'progressive_v2'"
            )
        from .profiles import (
            is_paper_facing_profile,
            require_profile_planner_strategy,
            require_profile_pubmed_setting,
            require_profile_literature_design_authority_setting,
        )

        require_profile_planner_strategy(
            name=self.submission_profile_name,
            version=self.submission_profile_version,
            planner_strategy=self.planner_strategy,
        )
        require_profile_pubmed_setting(
            name=self.submission_profile_name,
            version=self.submission_profile_version,
            enabled=self.enable_pubmed,
        )
        require_profile_literature_design_authority_setting(
            name=self.submission_profile_name,
            version=self.submission_profile_version,
            enabled=self.require_literature_design_authority,
        )
        progressive_resume_values = (
            self.development_progressive_resume_checkpoint_path,
            self.development_progressive_resume_checkpoint_sha256,
        )
        if any(value is not None for value in progressive_resume_values):
            if any(value is None for value in progressive_resume_values):
                raise ValueError(
                    "development progressive resume checkpoint path and SHA-256 "
                    "must be configured together"
                )
            profile_is_development_only = bool(
                self.submission_profile_name
                and not is_paper_facing_profile(self.submission_profile_name)
            )
            if not self.development_diagnostic and not profile_is_development_only:
                raise ValueError(
                    "development progressive resume requires either "
                    "development_diagnostic=True or a registered development-only "
                    "profile"
                )
            if self.planner_strategy != "progressive_v2":
                raise ValueError(
                    "development progressive resume requires "
                    "planner_strategy='progressive_v2'"
                )
            if self.enable_deterministic_planner_fallback:
                raise ValueError(
                    "development progressive resume cannot silently replace a "
                    "failed or rejected checkpoint with a fallback plan"
                )
            if is_paper_facing_profile(self.submission_profile_name):
                raise ValueError(
                    "development progressive resume cannot be combined with a "
                    "paper-facing submission profile"
                )
            resume_digest = str(
                self.development_progressive_resume_checkpoint_sha256 or ""
            ).strip()
            if len(resume_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in resume_digest
            ):
                raise ValueError(
                    "development progressive resume checkpoint SHA-256 is invalid"
                )
        locked_plan_values = (
            self.development_locked_analysis_plan_path,
            self.development_locked_analysis_plan_sha256,
        )
        if any(value is not None for value in locked_plan_values):
            if any(value is None for value in locked_plan_values):
                raise ValueError(
                    "development locked analysis plan path and SHA-256 must be "
                    "configured together"
                )
            if not self.development_diagnostic:
                raise ValueError(
                    "development locked analysis plan requires "
                    "development_diagnostic=True"
                )
            if is_paper_facing_profile(self.submission_profile_name):
                raise ValueError(
                    "development locked analysis plan cannot be combined with "
                    "a paper-facing submission profile"
                )
            if any(value is not None for value in progressive_resume_values):
                raise ValueError(
                    "development locked analysis plan and Progressive checkpoint "
                    "resume are mutually exclusive"
                )
            locked_digest = str(
                self.development_locked_analysis_plan_sha256 or ""
            ).strip()
            if len(locked_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in locked_digest
            ):
                raise ValueError(
                    "development locked analysis plan SHA-256 is invalid"
                )
        planner_efficiency_values = (
            self.development_planner_efficiency_max_calls,
            self.development_planner_efficiency_max_reported_tokens,
            self.development_planner_efficiency_max_wall_seconds,
        )
        if any(value is not None for value in planner_efficiency_values):
            if any(value is None for value in planner_efficiency_values):
                raise ValueError(
                    "development Planner efficiency limits must be configured "
                    "together"
                )
            profile_is_development_only = bool(
                self.submission_profile_name
                and not is_paper_facing_profile(self.submission_profile_name)
            )
            if not self.development_diagnostic and not profile_is_development_only:
                raise ValueError(
                    "development Planner efficiency limits require either "
                    "development_diagnostic=True or a registered development-only "
                    "profile"
                )
            if self.planner_strategy != "progressive_v2":
                raise ValueError(
                    "development Planner efficiency limits require "
                    "planner_strategy='progressive_v2'"
                )
            if is_paper_facing_profile(self.submission_profile_name):
                raise ValueError(
                    "development Planner efficiency limits cannot be combined "
                    "with a paper-facing submission profile"
                )
            from ..providers.efficiency_budget import PlannerEfficiencyLimits

            PlannerEfficiencyLimits(
                max_calls=int(
                    self.development_planner_efficiency_max_calls or 0
                ),
                max_reported_tokens=int(
                    self.development_planner_efficiency_max_reported_tokens or 0
                ),
                max_wall_seconds=float(
                    self.development_planner_efficiency_max_wall_seconds or 0.0
                ),
            )
        assert_step_provider_budget_funds_its_repairs(
            max_step_provider_calls=self.max_step_provider_calls,
            max_code_repair_attempts=self.max_code_repair_attempts,
            max_step_llm_repair_attempts=self.max_step_llm_repair_attempts,
            # `None` means "decide from the client", which a declarative config
            # cannot see. Count the reserved audit call only when it was asked
            # for outright; the pipeline re-checks with the resolved flag, so
            # the stricter number is applied by the layer that knows it.
            llm_concept_audit_enabled=self.enable_llm_concept_audit is True,
            allow_underfunded=self.allow_underfunded_step_provider_calls,
        )
        hard_stop_values = {
            "max_provider_attempts_per_run": self.max_provider_attempts_per_run,
            "max_provider_attempts_per_batch": self.max_provider_attempts_per_batch,
            "max_total_tokens_per_run": self.max_total_tokens_per_run,
            "max_total_tokens_per_batch": self.max_total_tokens_per_batch,
            "max_estimated_cost_usd_per_batch": (self.max_estimated_cost_usd_per_batch),
            "max_wall_clock_seconds_per_task": self.max_wall_clock_seconds_per_task,
            "input_cost_usd_per_million_tokens": (
                self.provider_input_cost_usd_per_million_tokens
            ),
            "output_cost_usd_per_million_tokens": (
                self.provider_output_cost_usd_per_million_tokens
            ),
        }
        if any(value is not None for value in hard_stop_values.values()):
            if any(value is None for value in hard_stop_values.values()):
                raise ValueError(
                    "Provider hard-stop configuration is all-or-none; declare "
                    "run/batch attempts, tokens, cost, wall clock, and both prices"
                )
            from ..authority.provider_hard_stop import ProviderHardStopLimits

            ProviderHardStopLimits(**hard_stop_values)  # type: ignore[arg-type]
        if self.required_primary_cohort_selection_mode not in {
            None,
            "predicate_filtered",
            "all_input_rows",
        }:
            raise ValueError(
                "required_primary_cohort_selection_mode must be "
                "'predicate_filtered', 'all_input_rows', or None"
            )
        authority_count = sum(
            value is not None
            for value in (
                self.trajectory_scientific_runtime_authority,
                self.current_case_scientific_runtime_authority,
            )
        )
        if authority_count > 1:
            raise ValueError(
                "trajectory and current-case scientific authorities are mutually "
                "exclusive"
            )
        if (authority_count == 0) != (
            self.scientific_runtime_projection_sha256 is None
        ):
            raise ValueError(
                "scientific authority and runtime projection digest must be "
                "configured together"
            )
        if self.trajectory_scientific_runtime_authority is not None:
            from ..trajectory.scientific_runtime_authority import (
                load_trajectory_scientific_runtime_authority,
            )

            load_trajectory_scientific_runtime_authority(
                self.trajectory_scientific_runtime_authority
            )
        if self.current_case_scientific_runtime_authority is not None:
            from ..authority.current_case_scientific_runtime import (
                load_current_case_scientific_runtime_authority,
            )

            load_current_case_scientific_runtime_authority(
                self.current_case_scientific_runtime_authority
            )
        if authority_count:
            digest = str(self.scientific_runtime_projection_sha256 or "")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "scientific_runtime_projection_sha256 must be a SHA-256 digest"
                )

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        """Return a new :class:`PipelineConfig` with the given fields
        replaced. Equivalent to ``dataclasses.replace``.
        """
        return replace(self, **overrides)

    def _field_values(self) -> Dict[str, Any]:
        """Return every field by *reference*.

        Deliberately not :func:`dataclasses.asdict`, which recursively copies
        values and needlessly changes immutable mapping wrappers.
        """

        return {f.name: getattr(self, f.name) for f in fields(self)}

    def as_kwargs(self) -> Dict[str, Any]:
        """Return the flat declarative settings as a plain dictionary."""
        return self._field_values()

    def canonical_payload(self) -> Dict[str, Any]:
        """Return a JSON-safe rendering of every field.

        Secret-bearing string fields are replaced by a digest: the payload is
        written into run provenance, and an API key in a manifest is a leak
        even though the key is genuinely part of the configuration.
        """

        def _render(value: Any, *, key: str = "") -> Any:
            if (
                isinstance(value, str)
                and value
                and (is_sensitive_key(key) or string_contains_secret(value))
            ):
                return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {
                    str(k): _render(v, key=str(k)) for k, v in sorted(value.items())
                }
            if isinstance(value, (list, tuple, set, frozenset)):
                return [_render(v, key=key) for v in value]
            return f"<{type(value).__module__}.{type(value).__qualname__}>"

        return {
            key: _render(value, key=key)
            for key, value in sorted(self._field_values().items())
        }

    def recovery_payload(self) -> Dict[str, Any]:
        """Return a lossless, credential-free payload for durable resume.

        ``canonical_payload`` is intentionally one-way: it hashes anything
        below a broadly sensitive key name so provenance can be written even
        for opaque caller mappings.  Reconstructing a live configuration from
        that projection changes ordinary schema-owned identities such as a
        literature citation ``key``.  Recovery therefore has its own contract:
        preserve all supported declarative values exactly, and refuse fields
        that may carry credentials or opaque runner state.
        """

        values = self._field_values()
        for field_name in sorted(_RECOVERY_BLOCKED_FIELDS):
            value = values.get(field_name)
            if value not in (None, "", (), [], {}):
                raise PipelineConfigRecoveryError(
                    f"pipeline_config_recovery_field_not_persistable:{field_name}"
                )
        return {
            key: _render_recovery_value(value, path=key)
            for key, value in sorted(values.items())
        }

    @classmethod
    def from_recovery_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_digest: str,
    ) -> "PipelineConfig":
        """Reconstruct and bind one persisted config to its original digest."""

        if not isinstance(payload, Mapping):
            raise PipelineConfigRecoveryError(
                "pipeline_config_recovery_payload_not_mapping"
            )
        digest = str(expected_digest or "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise PipelineConfigRecoveryError(
                "pipeline_config_recovery_digest_invalid"
            )
        try:
            config = cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise PipelineConfigRecoveryError(
                "pipeline_config_recovery_payload_invalid"
            ) from exc
        if config.canonical_digest() != digest:
            raise PipelineConfigRecoveryError(
                "pipeline_config_recovery_digest_mismatch"
            )
        return config

    def canonical_digest(self) -> str:
        """SHA-256 over :meth:`canonical_payload`, for run provenance."""

        rendered = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = [
    "PipelineConfigRecoveryError",
    "PipelineConfig",
    "assert_step_provider_budget_funds_its_repairs",
    "step_provider_call_entitlement",
]
