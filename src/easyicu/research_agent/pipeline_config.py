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
  :meth:`PipelineConfig.from_kwargs` without inspecting ``__init__``;
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

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union


@dataclass
class PipelineConfig:
    """Frozen-ish mirror of ``ResearchAgentPipeline.__init__`` keyword args.

    Defaults intentionally match ``__init__`` so
    ``PipelineConfig(workdir=...)`` and the bare-kwargs form produce
    identical pipelines.
    """

    # --- required -------------------------------------------------------
    workdir: Union[str, Path]

    # --- core LLM / runtime ---------------------------------------------
    llm: Optional[Any] = None
    timeout_seconds: float = 300.0
    python_executable: Optional[str] = None

    # --- feature toggles -------------------------------------------------
    enable_literature: bool = True
    enable_visual_qa: bool = True
    enable_publication_figure_skill: bool = True
    enable_vlm_visual_qa: Optional[bool] = None
    vlm_client: Optional[Any] = None
    visual_qa_adapter: Optional[Any] = None
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
    disable_icu_context: bool = False
    context_top_k: Optional[int] = None

    # --- code-repair / determinism --------------------------------------
    max_code_repair_attempts: int = 1
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
    enable_reviewer_round: bool = True
    enable_fairness_subgroups: bool = True
    enable_hypothesis_generator: bool = False
    hypothesis_generator_top_k: int = 5
    enable_pdf_render: bool = False

    # --- execution shape -------------------------------------------------
    max_concurrent_steps: int = 1
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
    # Hard cap on numeric-claim leaves registered per single step.
    # Prevents one step that dumps a full interaction matrix into
    # step_summary.json from creating hundreds of footnotes when its
    # numbers are referenced in the manuscript. Pilot run 20260515
    # had a step with 295 claims; 100 covers any realistic clinical
    # analysis without truncating real result quantities.
    max_numeric_claims_per_step: int = 100
    # --- writer namespace breadth ---------------------------------------
    # When False (default), the writer's evidence digest is the curated
    # ``preferred_keys`` tuple in pipeline_writer_aux. When True, the
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
    # ``easyicu.research_agent.experience.mine_experience_from_run``.
    # The bank is opt-in because (i) it changes the planner's input
    # surface and (ii) the npj DM submission run does not depend on
    # experience-bank behaviour.
    enable_experience_bank: bool = False
    experience_bank_path: Optional[Union[str, Path]] = None
    experience_bank_top_k: int = 5
    experience_bank_min_similarity: float = 0.2

    # --- code runner ----------------------------------------------------
    runner_kind: str = "subprocess"
    runner_image: Optional[str] = None
    runner_network: str = "none"
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
        """Build a config from an arbitrary kwargs dict, ignoring keys
        that don't correspond to a field. Use when loading from YAML /
        TOML / argparse where extra keys may be present.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        """Return a new :class:`PipelineConfig` with the given fields
        replaced. Equivalent to ``dataclasses.replace``.
        """
        return replace(self, **overrides)

    def as_kwargs(self) -> Dict[str, Any]:
        """Return a plain-dict view suitable for the legacy
        ``ResearchAgentPipeline(**config.as_kwargs())`` form.
        """
        return asdict(self)


__all__ = ["PipelineConfig"]
