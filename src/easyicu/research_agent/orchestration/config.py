"""Typed configuration object for :class:`ResearchAgentPipeline`.

The pipeline's ``__init__`` takes ~60 keyword arguments. They have
grown organically and are documented by way of the function signature
alone, which means downstream tooling cannot reason about the
configuration surface without parsing source code.

:class:`PipelineConfig` mirrors that signature as a frozen-ish
dataclass so:

* IDEs and type-checkers can autocomplete / validate construction;
* tests can build a baseline config and override only what they care
  about (``config = PipelineConfig.default().with_overrides(...)``);
* configuration can be loaded from YAML / TOML via
  :meth:`PipelineConfig.from_kwargs`, with unknown or misspelled keys
  rejected instead of silently ignored;
* future refactors that group flags (literature, runner, audits, ...)
  can add nested config objects without breaking ``__init__``.

This module is **additive**. The existing ``ResearchAgentPipeline.__init__``
keyword form continues to work; ``PipelineConfig`` is the recommended
new-code path. Call ``ResearchAgentPipeline.from_config(config)`` to
construct a pipeline from a config object, or call
``config.as_kwargs()`` to feed it back into the legacy ``__init__``.

Why a dataclass and not pydantic? ``schema.py`` already imports
pydantic for runtime-validated payloads. The pipeline-construction
surface is consumed by Python code (not by serialised pipelines)
and benefits more from being a lightweight dataclass that mirrors
``__init__`` 1:1 than from another validation layer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union


#: Field/key names whose *value* must never appear in run provenance. Matched
#: on the name, not on the value, so a key rotated to a new format stays
#: redacted.
_SECRET_FIELD_RE = re.compile(
    r"(?i)(?:^|_)(?:api_key|apikey|key|token|secret|password|passwd|credential)s?$"
)


def _is_secret_field(name: str) -> bool:
    return bool(_SECRET_FIELD_RE.search(str(name)))


def _deep_freeze(value: Any) -> Any:
    """Return an immutable view of a plain data container, recursively.

    Only the exact builtin containers are converted. A subclass (a pydantic
    model, a ``defaultdict`` a caller relies on, a numpy array) keeps its
    identity and behaviour: this exists to stop a shared ``runner_kwargs``
    dict being edited after the config was hashed, not to re-type the
    collaborators a run was handed.
    """

    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
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
    """Immutable mirror of ``ResearchAgentPipeline.__init__`` keyword args.

    Defaults intentionally match ``__init__`` so
    ``PipelineConfig(workdir=...)`` and the bare-kwargs form produce
    identical pipelines.

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
    mapping and every ``list``/``set`` a tuple/frozenset, recursively. Live
    objects (clients, factories, adapters, checkpointers) are left exactly as
    passed — they are the run's collaborators, not its configuration values.
    """

    # --- required -------------------------------------------------------
    workdir: Union[str, Path]

    # --- core LLM / runtime ---------------------------------------------
    llm: Optional[Any] = None
    timeout_seconds: float = 300.0
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
    enable_publication_figure_skill: bool = True
    enable_vlm_visual_qa: Optional[bool] = None
    vlm_client: Optional[Any] = None
    visual_qa_adapter: Optional[Any] = None
    # Uploading a rendered figure is a separate decision from authorizing the
    # provider: the image can carry per-patient marks, small-cell strata or
    # local paths that the text outbound projection would have stripped.
    allow_external_figure_upload: bool = False
    enable_llm_concept_audit: Optional[bool] = None
    llm_concept_auditor_client: Optional[Any] = None
    enable_memory: bool = True
    enable_latex: bool = True

    # --- evidence enforcement -------------------------------------------
    # "soft" (default): unsupported sentences are filtered and unresolved
    # placeholders are demoted to comments; warnings surface in findings.
    # "strict": EvidenceStore raises EvidenceEnforcementError, aborting the
    # run, so CI / final submission cannot ship a silently repaired manuscript.
    evidence_enforcement_mode: str = "soft"
    latex_venue_template: str = "article"
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
    # repair, patch, and full-rewrite fallback. Seven supports the normal
    # fail-closed semantic path: generation + three digest-bound audits + two
    # minimal patches + one Analyzer call. Successful first-pass steps do not
    # spend the additional headroom. Full rewrites and transport retries still
    # consume the same monotonic stop-loss rather than receiving a hidden budget.
    max_step_provider_calls: int = 7
    enable_deterministic_code_fallback: bool = False
    enable_deterministic_planner_fallback: bool = False
    enable_deterministic_runner_repair: bool = True
    # --- literature search backends -------------------------------------
    enable_pubmed: bool = False
    pubmed_email: Optional[str] = None
    pubmed_api_key: Optional[str] = None
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
    enable_reviewer_round: bool = True
    enable_fairness_subgroups: bool = True
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
    # Default of 12 covers probe + cohort summary + 2-3 primary models +
    # 2-3 sensitivities + figure + interpretation. Pilot run 20260515 saw
    # the planner expand a simple SOFA-2 association to 30 steps with 13
    # revisions before being killed at step 20; this cap prevents that.
    max_total_steps: int = 12
    # --- replanning convergence guards (2026-06-11) ---------------------
    # The replanner runs after the probe and after every clean step. A
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
    # Phase-1 step toward the more autonomous writer namespace.
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
    enable_coder_resources: bool = False
    enable_reviewed_memory: bool = False
    reviewed_memory_namespaces: Sequence[str] = ()
    enable_capability_workflow: bool = False
    expected_runner_image_digest: Optional[str] = None
    capability_request: Optional[Dict[str, Any]] = None
    # Operator-supplied control plane for the human-review interrupt. A live
    # object (it owns a checkpointer), so it must never be deep-copied.
    human_review_gate: Optional[Any] = None
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
    runner_factory: Optional[Callable[..., Any]] = None
    runner_kwargs: Optional[Dict[str, Any]] = None

    # --- case plugins ---------------------------------------------------
    # Opt-in deterministic-fallback plugins for specific research designs.
    # Default is empty: a pipeline constructed without case plugins carries
    # no bias toward any particular paper's column names or fallback scripts.
    # See ``easyicu.research_agent.fallback.CasePluginRegistry``. No
    # case-specific plugins are bundled; users supply their own.
    case_plugin_registry: Optional[Any] = None

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

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        """Return a new :class:`PipelineConfig` with the given fields
        replaced. Equivalent to ``dataclasses.replace``.
        """
        return replace(self, **overrides)

    def _field_values(self) -> Dict[str, Any]:
        """Return every field by *reference*.

        Deliberately not :func:`dataclasses.asdict`, which recursively
        ``copy.deepcopy``s anything that is not a dataclass or a builtin
        container. Several fields hold live objects — a provider client with an
        open ``httpx`` connection pool, a runner factory, a visual-QA adapter, a
        human-review checkpointer. Deep-copying those either raises
        (``TypeError: cannot pickle '_thread.lock' object``) or, worse,
        succeeds and hands the pipeline a *clone*, so the object the provider
        factory authorised is not the object that makes the calls.
        """

        return {f.name: getattr(self, f.name) for f in fields(self)}

    def as_kwargs(self) -> Dict[str, Any]:
        """Return a plain-dict view suitable for the legacy
        ``ResearchAgentPipeline(**config.as_kwargs())`` form.
        """
        return self._field_values()

    def canonical_payload(self) -> Dict[str, Any]:
        """Return a JSON-safe rendering of every field.

        Live objects (an ``llm`` client, a ``runner_factory``) are rendered by
        type rather than value: what matters for provenance is which kind of
        component was configured, not its memory address.

        Secret-bearing string fields are replaced by a digest: the payload is
        written into run provenance, and an API key in a manifest is a leak
        even though the key is genuinely part of the configuration.
        """

        def _render(value: Any, *, key: str = "") -> Any:
            if _is_secret_field(key) and isinstance(value, str) and value:
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

    def canonical_digest(self) -> str:
        """SHA-256 over :meth:`canonical_payload`, for run provenance."""

        rendered = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = ["PipelineConfig"]
