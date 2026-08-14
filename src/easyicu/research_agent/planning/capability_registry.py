"""EasyICU research-agent capability registry — the explicit capability surface.

This module is the SINGLE SOURCE OF TRUTH for one question a reviewer (or a new
user) will ask: *for a given study-design family, which scientific result is
agent-produced, which standardized products are rendered or audited
deterministically, and where does the framework fail closed?* The answer used to
be implicit, spread across the preflight
dispatch ladder in ``execution.phase``, the ``FAMILY_RENDERERS`` table, and the
readiness gates in ``reporting.readiness``. Here it is declared once, rendered to a
matrix, and kept honest by ``tests/research_agent/test_capability_registry.py``,
which cross-checks every claim below against the code that is actually wired
(``AUXILIARY_DETERMINISTIC_RUNNERS`` and ``figures.FAMILY_RENDERERS``). If a
runner is added or removed without updating
this registry, that test fails — the matrix cannot silently rot.

Two design points this registry makes explicit:

* **Every primary analysis has one explicit owner.** A fully declared Cox or
  exact single-model adjusted-association contract is owned by a sealed host
  executor, which chooses no cohort, exposure, outcome, coding, adjustment,
  horizon or diagnostic. Free-form scientific kernels remain agent-coded.
  ``primary_analysis`` and ``figure`` stay separate.
* **A capability gap is always REPORTED, never silently filled.** When no valid
  runner/data contract exists the pipeline fails closed with a specific reason
  (see ``FAIL_CLOSED_LADDER``); it never fabricates a result or degrades a
  headline to a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from ..contracts.capability_ids import CAPABILITY_FAMILIES
from ..contracts.association_execution import association_execution_verdict
from ..contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
    EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
    exposure_outcome_distribution_execution_verdict,
)
from ..contracts.model_tokens import (
    ADJUSTED_ASSOCIATION_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
)
from .study_design_playbook import StudyDesignFamily

if TYPE_CHECKING:
    from ..schema import ResearchContext

__all__ = [
    "ScientificCapability",
    "FamilyCapability",
    "PrimaryCapabilityVerdict",
    "ScientificCapabilityAssessment",
    "resolve_primary_capability",
    "AuxiliaryRunner",
    "CAPABILITY_REGISTRY",
    "AUXILIARY_DETERMINISTIC_RUNNERS",
    "KNOWN_UNSUPPORTED_ESTIMANDS",
    "FAIL_CLOSED_LADDER",
    "get_capability",
    "get_capability_by_id",
    "get_capability_for_plan",
    "assess_scientific_capability",
    "deterministic_primary_families",
    "llm_coded_primary_families",
    "families_without_deterministic_primary",
    "render_capability_matrix_markdown",
]


@dataclass(frozen=True)
class ScientificCapability:
    """One family-owned scientific contract, not a generic agent-framework plug-in.

    The capability declares the existing execution/result boundary for a study
    family.  It deliberately does not let a registry entry invent an estimand,
    fit a substitute model, or turn an LLM result into a publication claim.  A
    future method grows by adding an owner-backed record here plus its own
    executor/validator, rather than by widening the orchestration core.
    """

    family: StudyDesignFamily
    label: str
    # primary analysis (the reported estimand)
    primary_analysis: str  # "deterministic" | "llm_coded"
    primary_estimand: str
    primary_runner: Optional[str]
    primary_runner_module: Optional[str]
    # figure
    figure: str  # "deterministic" | "llm_coded"
    figure_renderer: Optional[
        str
    ]  # FAMILY_RENDERERS key, "base_association_skill", or None
    # the data the runner/step needs, and what happens when it is absent
    data_contract: Tuple[str, ...] = field(default_factory=tuple)
    fail_closed: str = ""
    notes: str = ""
    capability_id: str = ""
    result_contract: str = ""
    required_diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    # ``analysis_only`` is an honest boundary: an agent may execute the
    # declared analysis, but EasyICU has no registered deterministic validator
    # for the scientific identification claim and therefore cannot promote it.
    scientific_validation: Literal["reportable", "analysis_only"] = "analysis_only"
    scientific_validator_owner: Optional[str] = None
    scientific_validator_contract: Optional[str] = None


# Backward-compatible public name.  New code should use ScientificCapability:
# it carries the result/diagnostic/publication contract, not merely a catalogue
# row about a study-design family.
FamilyCapability = ScientificCapability


@dataclass(frozen=True)
class ScientificCapabilityAssessment:
    """Pre-execution ceiling on what one scientific capability may support."""

    capability_id: Optional[str]
    analysis_type: Optional[str]
    question_present: bool
    question_coordinates_resolved: bool
    input_contract_resolved: bool
    # This pre-execution receipt cannot truthfully say whether source rows or
    # a provider/backend were available. Those facts belong to execution
    # receipts, so leave them unknown instead of turning a schema declaration
    # into an availability claim.
    runtime_data_available: Optional[bool]
    execution_backend_available: Optional[bool]
    scientific_validator_available: bool
    claim_ceiling: Literal["reportable", "analysis_only", "unsupported"]
    issue_code: Optional[str] = None
    reason: str = ""

    @property
    def claim_ceiling_allows_reportable(self) -> bool:
        return self.claim_ceiling == "reportable"

    @property
    def question_grounded(self) -> bool:
        """Compatibility alias; no full semantic grounding is asserted."""

        return self.question_coordinates_resolved

    @property
    def status(self) -> Literal["reportable", "analysis_only", "unsupported"]:
        """Compatibility alias for callers reading the pre-v3 receipt."""

        return self.claim_ceiling

    @property
    def publication_eligible(self) -> bool:
        """Compatibility alias; final publication readiness is a separate gate."""

        return self.claim_ceiling_allows_reportable

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-ready readiness receipt."""

        return {
            "schema_version": "easyicu.scientific_capability_assessment/3",
            "capability_id": self.capability_id,
            "analysis_type": self.analysis_type,
            "question_present": self.question_present,
            "question_coordinates_resolved": self.question_coordinates_resolved,
            "input_contract_resolved": self.input_contract_resolved,
            "runtime_data_available": self.runtime_data_available,
            "execution_backend_available": self.execution_backend_available,
            "scientific_validator_available": self.scientific_validator_available,
            "claim_ceiling": self.claim_ceiling,
            "claim_ceiling_allows_reportable": (self.claim_ceiling_allows_reportable),
            "issue_code": self.issue_code,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AuxiliaryRunner:
    """A deterministic runner that supports (not owns) a family's primary estimand."""

    name: str
    entrypoint: str
    module: str
    purpose: str
    fail_closed: str


# ---------------------------------------------------------------------------
# The registry. Every ``primary_runner`` / ``figure_renderer`` name below is
# cross-checked against the wired code by the drift test.
# ---------------------------------------------------------------------------

CAPABILITY_REGISTRY: Tuple[ScientificCapability, ...] = (
    ScientificCapability(
        family="time_to_event",
        label="Survival / time-to-event",
        primary_analysis="deterministic",
        primary_estimand="Host-computed Cox hazard ratio under an exact Planner-owned, digest-bound survival contract",
        primary_runner="survival_primary_cox",
        primary_runner_module="execution.runners.survival_primary_executor",
        figure="deterministic",
        figure_renderer="time_to_event",
        data_contract=(
            "exposure column",
            "Planner-declared, authority-bound follow-up time and unit",
            "Planner-declared, authority-bound event indicator/value",
            "Exact covariate set, horizon and complete-case policy",
        ),
        fail_closed=(
            "Plan validation refuses any primary survival contract the sealed Cox "
            "owner cannot execute; digest, fit, PH or evidence mismatch fails closed."
        ),
        notes="The sealed primary owner fits only the exact declared Cox contract; the survival renderer remains a separate consumer.",
        capability_id="survival_time_to_event_v1",
        result_contract="host-issued digest-bound survival receipt + primary CSV + PH table",
        required_diagnostics=(
            "event/censoring closure",
            "time-origin binding",
            "proportional-hazards diagnostic when Cox is declared",
        ),
        scientific_validation="reportable",
        scientific_validator_owner="execution.runners.survival_primary_executor",
        scientific_validator_contract="SurvivalPrimaryResultReceipt",
    ),
    ScientificCapability(
        family="causal_emulation",
        label="Causal inference / target-trial emulation",
        primary_analysis="llm_coded",
        primary_estimand="Agent-coded causal contrast under a declared target-trial/identification strategy; assumptions and balance checked",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="causal_emulation",
        data_contract=(
            "binarised exposure",
            "binary outcome",
            "adjustment set (config user_preferences.covariates, else demographics+severity)",
        ),
        fail_closed=(
            "The agent step fails when its declared exposure, outcome, time zero, "
            "or adjustment inputs cannot be resolved; balance, positivity, and "
            "causal-language gates reject unsupported claims."
        ),
        notes="The deterministic causal component renders registered balance/effect products only; it never selects covariates or an estimator.",
        capability_id="causal_target_trial_v1",
        result_contract="family_primary_result_requirement + registered primary CSV",
        required_diagnostics=(
            "target-trial time-zero and treatment-strategy protocol",
            "identification/refutation",
            "positivity and balance",
        ),
        scientific_validation="analysis_only",
    ),
    ScientificCapability(
        family="association",
        label="Association — graded ordinal exposure (dose-response)",
        primary_analysis="llm_coded",
        primary_estimand="Agent-coded ordered-exposure association under the declared trend method; per-stage products value-checked",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=(
            "ordinal grade exposure (>=3 ordered integer levels)",
            "binary outcome",
            "adjustment set",
        ),
        fail_closed=(
            "The ordered-product contract rejects fewer than three declared levels, "
            "invalid level ordering, cohort drift, or missing trend statistics; a "
            "binary/continuous exposure is never coerced into an ordinal gradient."
        ),
        notes="Validated ordered-trend calculation primitives are available to agent code; the execution framework does not choose the exposure, adjustment set, or model.",
        capability_id="association_ordinal_trend_v1",
        result_contract="planned_model_requirement + registered adjusted estimate",
        required_diagnostics=("declared levels", "primary contrast", "model contract"),
    ),
    ScientificCapability(
        family="association",
        label="Association — exact single-model adjusted",
        primary_analysis="deterministic",
        primary_estimand="Host-computed adjusted association under one exact Planner-owned estimator and typed model-term contract",
        primary_runner="adjusted_association_estimates",
        primary_runner_module="execution.runners.adjusted_association_executor",
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=(
            "one exposure and outcome",
            "exact covariate roster",
            "typed coding, levels, reference and transform for every model term",
            "exact supported estimator token",
        ),
        fail_closed=(
            "The owner declines names-only coding, unsupported estimators, multiple "
            "model requirements, undeclared levels, rank loss or a non-estimable "
            "fit; no estimator or predictor is substituted."
        ),
        notes="Only the exact single-model contract is deterministic; broader association kernels remain a separate LLM-coded capability. The base figure skill renders registered products.",
        capability_id="association_adjusted_v1",
        result_contract="typed planned_model_requirement + host model contract + registered adjusted estimate",
        required_diagnostics=(
            "model-term coding receipt",
            "primary model contract",
            "effect/interval reconciliation",
        ),
        scientific_validation="reportable",
        scientific_validator_owner="execution.runners.adjusted_association_executor",
        scientific_validator_contract="AssociationExecutionVerdict",
    ),
    ScientificCapability(
        family="association",
        label="Association — general / free-form",
        primary_analysis="llm_coded",
        primary_estimand="Agent-coded association for scientific kernels outside the exact single-model host contract",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=(
            "explicit scientific specification",
            "registered result product",
        ),
        fail_closed=(
            "Unsupported or under-declared scientific kernels remain agent-coded and "
            "cannot be relabelled as the deterministic adjusted-association owner."
        ),
        notes="This capability covers interactions, splines, multiple models and other free-form association kernels; implementation repair may not change their scientific design.",
        capability_id="association_freeform_v1",
        result_contract="agent-authored registered result under deterministic gates",
        required_diagnostics=(
            "method-specific contract",
            "effect/interval reconciliation",
        ),
        # Reachable and executable is not the same as publication-validated.
        # This capability has no typed exposure/outcome/contrast/adjustment
        # contract yet, so it cannot receive a higher claim ceiling than the
        # more constrained deterministic association owner.
        scientific_validation="analysis_only",
    ),
    ScientificCapability(
        family="prediction",
        label="Prediction / risk modelling",
        primary_analysis="llm_coded",
        primary_estimand="LLM-coded discrimination + calibration (AUROC, calibration); value-provenance verified",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="prediction",
        data_contract=("predictors", "binary outcome", "train/validation split"),
        fail_closed=(
            "LLM code failure -> repair -> fail-closed. manuscript_numeric_auditor "
            "catches rounded/hallucinated metrics (caught AUROC 0.766->0.7 in a pilot)."
        ),
        notes="ROC + calibration figure is deterministic; the model FIT is LLM-coded.",
        capability_id="prediction_risk_model_v1",
        result_contract="registered discrimination and calibration products",
        required_diagnostics=("split/leakage", "discrimination", "calibration"),
    ),
    ScientificCapability(
        family="prediction",
        label="Dynamic prediction / landmark early warning",
        primary_analysis="llm_coded",
        primary_estimand=(
            "Time-updated binary risks at prespecified landmarks and target "
            "horizons using only measurements available by each prediction time"
        ),
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="prediction",
        data_contract=(
            "longitudinal rows with patient/stay identity and measurement time",
            "prespecified prediction landmarks, lookback windows and target horizons",
            "event time plus follow-up/censoring time sufficient to observe each horizon",
            "patient-level development/validation split with preprocessing fitted inside each training split",
        ),
        fail_closed=(
            "Missing event/follow-up times, post-landmark feature leakage, an "
            "unobservable target horizon, or a row-level rather than patient-level "
            "split blocks the action. Static prediction is offered only as a "
            "user-confirmed alternative, never as an automatic substitute."
        ),
        notes=(
            "Landmark construction and metric evaluation use digest-bound reviewed "
            "kernels and sklearn; model fitting remains Coder-generated and "
            "analysis-only until a typed host fit/result validator is registered."
        ),
        capability_id="dynamic_prediction_landmark_v1",
        result_contract=(
            "typed scientific action + leakage-safe landmark dataset receipt + "
            "registered per-landmark discrimination/calibration products"
        ),
        required_diagnostics=(
            "prediction-time/observation-window/target-horizon separation",
            "patient-level split and leakage audit",
            "per-landmark discrimination, calibration and horizon observability",
            "temporal subgroup or drift assessment when data support it",
        ),
        scientific_validation="analysis_only",
    ),
    ScientificCapability(
        family="phenotyping",
        label="Phenotyping / clustering",
        primary_analysis="llm_coded",
        primary_estimand="Agent-planned cluster solution; outcome-by-cluster kept descriptive (not causal)",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="phenotyping",
        data_contract=("feature matrix (e.g. first-24h trajectories)", "k selection"),
        fail_closed=(
            "figure_strategy anti-pattern blocks 'clusters are causal entities'; "
            "an LLM failure fails closed to diagnostic_only."
        ),
        notes=(
            "Cluster heatmap + stability + outcome-by-cluster figure is "
            "deterministic from registered clustering products. The clustering "
            "method, feature representation, and k-selection remain agent-owned; "
            "a dedicated typed trajectory-stability step may use the supporting "
            "calculator only after the planner has fixed every resampling, refit, "
            "alignment, and decision parameter. "
            "The former SOFA-specific KMeans script is not advertised or routed "
            "as a general auxiliary capability."
        ),
        capability_id="phenotyping_cluster_v1",
        result_contract="registered clustering, profile, and stability products",
        required_diagnostics=(
            "representation",
            "cluster stability",
            "descriptive outcome use",
        ),
    ),
    ScientificCapability(
        family="descriptive",
        label="Descriptive / measurement audit",
        primary_analysis="llm_coded",
        primary_estimand="LLM-coded descriptive summaries / Table One / measurement-process audits",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=("cohort", "variables of interest"),
        fail_closed=(
            "Evidence STRICT mode blocks unbound sentences; the plausibility gate "
            "flags implausible descriptives before they reach the manuscript."
        ),
        capability_id="descriptive_measurement_v1",
        result_contract="registered summary/source-data products",
        required_diagnostics=("denominators", "measurement availability"),
    ),
    ScientificCapability(
        family="descriptive",
        label="Descriptive — typed exposure/outcome absolute risks",
        primary_analysis="deterministic",
        primary_estimand=(
            "Host-computed exposure prevalence, outcome absolute risks and an "
            "optional prespecified unadjusted risk difference under an exact "
            "typed descriptive-only contract"
        ),
        primary_runner=EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
        primary_runner_module=(
            "execution.runners.exposure_outcome_distribution_executor"
        ),
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=(
            "one digest-bound typed cohort",
            "ExposureOutcomeDistributionSpec/2 with closed typed levels",
            "DescriptiveClaimContract/1 with descriptive_only ceiling",
        ),
        fail_closed=(
            "The owner declines an untyped cohort, an incomplete distribution "
            "design, any adjusted/causal owner contract, or any claim ceiling "
            "above descriptive-only. Runtime evidence and host-derived claim "
            "metadata must reproduce from the exact registered summary bytes."
        ),
        notes=(
            "This narrow primary capability does not upgrade ordinary Table One "
            "or measurement-audit steps in the broader descriptive family."
        ),
        capability_id=DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
        result_contract=(
            "exposure_outcome_distribution/2 summary + digest-bound "
            "ScientificClaimRegistration"
        ),
        required_diagnostics=(
            "closed exposure/outcome levels and denominators",
            "interval/dependence contract",
            "descriptive-only noncausal claim ceiling",
        ),
        scientific_validation="reportable",
        scientific_validator_owner=(
            "execution.runners.exposure_outcome_distribution_executor"
        ),
        scientific_validator_contract=(
            "exposure_outcome_distribution_result_receipt_valid + "
            "ScientificClaimRegistration"
        ),
    ),
)


def _assert_capability_vocabulary_matches_registry() -> None:
    """Keep stable persisted ids synchronized with executable registrations."""

    registered = {
        capability.capability_id: capability.family
        for capability in CAPABILITY_REGISTRY
        if capability.capability_id
    }
    if registered != CAPABILITY_FAMILIES:
        raise RuntimeError(
            "scientific capability vocabulary drift: "
            f"missing={sorted(set(CAPABILITY_FAMILIES) - set(registered))!r}, "
            f"unregistered={sorted(set(registered) - set(CAPABILITY_FAMILIES))!r}, "
            "family_mismatches="
            f"{sorted(key for key in registered.keys() & CAPABILITY_FAMILIES.keys() if registered[key] != CAPABILITY_FAMILIES[key])!r}"
        )


_assert_capability_vocabulary_matches_registry()


AUXILIARY_DETERMINISTIC_RUNNERS: Tuple[AuxiliaryRunner, ...] = (
    AuxiliaryRunner(
        name="absolute_risk_context",
        entrypoint="absolute_risk_context_code",
        module="execution.runners.deterministic_descriptive",
        purpose="Render descriptive exposure prevalence and absolute-risk context from an explicit product contract.",
        fail_closed="Declines figure/primary-effect contracts and blocks when the declared descriptive columns are unavailable.",
    ),
    AuxiliaryRunner(
        name="robustness_sensitivity",
        entrypoint="robustness_sensitivity_preflight_code",
        module="execution.runners.deterministic_robustness",
        purpose="Replay an agent-locked primary model across prespecified robustness variants.",
        fail_closed="Requires a locked model/specification contract and never selects the primary exposure, outcome, cohort, or estimator.",
    ),
    AuxiliaryRunner(
        name="missingness_measurement_audit",
        entrypoint="missingness_measurement_audit_code",
        module="execution.runners.deterministic_missingness",
        purpose=(
            "Per-concept measured-vs-missing counts + structural-vs-measurement "
            "split for a missingness / measurement-process audit step (never "
            "imputes)."
        ),
        fail_closed=(
            "Blocks with a reason when no <concept>_measured columns resolve. "
            "The figure step renders the registered audit product via the "
            "data_quality->missingness renderer."
        ),
    ),
    AuxiliaryRunner(
        name="trajectory_cluster_stability",
        entrypoint="trajectory_stability_executor_code",
        module="execution.runners.trajectory_stability_executor",
        purpose=(
            "Compute a complete planner-owned, digest-bound trajectory-cluster "
            "stability specification without selecting the representation, model, "
            "cluster count, resampling design, seed policy, or decision threshold."
        ),
        fail_closed=(
            "Requires one dedicated stability owner, exact typed upstream products, "
            "and the closed supported refit contract. Unsupported or failed refits "
            "remain diagnostic and never fall back to coder repair, another method, "
            "another seed policy, or another cluster count."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Explicit boundaries: estimands the framework deliberately does NOT support.
# Recording them here (not just as a benchmark probe) keeps the capability
# surface honest — a reviewer sees the edges, and these must FAIL CLOSED rather
# than be approximated by a nearby estimand.
# ---------------------------------------------------------------------------

KNOWN_UNSUPPORTED_ESTIMANDS: Tuple[Tuple[str, str], ...] = (
    (
        "Competing-risks cumulative incidence (Fine-Gray / CIF)",
        "No deterministic runner. A cause-naive Cox HR is NOT a CIF, so a "
        "competing-risks question (for example, an event with death as a competing "
        "risk) must fail closed to diagnostic_only — not be answered with a Cox HR.",
    ),
)

# These analysis types intentionally receive the neutral *display* family in
# ``study_design`` so a brief can still be rendered.  That is not evidence of a
# validated scientific execution capability.  Keep this boundary explicit at
# the capability owner rather than silently borrowing the descriptive contract.
_DISPLAY_ONLY_ANALYSIS_TYPES = frozenset(
    {
        "multimodal",
        "reinforcement_learning",
        "cross_database_replication",
        "treatment_response",
    }
)


# ---------------------------------------------------------------------------
# The fail-closed / gap-report ladder: what happens when no valid runner or
# data contract exists. This is the answer to "what is the gap-report behavior?"
# ---------------------------------------------------------------------------

FAIL_CLOSED_LADDER: Tuple[Tuple[str, str], ...] = (
    (
        "1. Declared method, product and owner contract",
        "The Planner fixes the scientific method, cohort, exposure and outcome. "
        "Each capability declares whether the primary owner is the agent or a "
        "sealed executor; neither may substitute undeclared science.",
    ),
    (
        "2. Runner contract unmet",
        "An auxiliary runner writes status=blocked + a specific blocking_reason "
        "when its declared standardized inputs are missing or invalid. It never "
        "guesses scientific variables or a surrogate method. Optional auxiliary "
        "steps degrade to status=skipped + not_applicable when their input is "
        "legitimately absent.",
    ),
    (
        "3. Owned execution",
        "The declared primary owner executes the analysis. Code repair and "
        "statistical validators may repair agent implementation faults but never "
        "rewrite a sealed host primary or replace the declared method.",
    ),
    (
        "4. Output / validity gates (fail-closed)",
        "execution_complete (any failed step -> False); evidence_complete "
        "(STRICT: unbound citations blocked); numeric_verified (value-level "
        "provenance: hallucinated numbers blocked); analysis_validated "
        "(plausibility + survival-estimand + figure-credit + headline==primary-"
        "estimand gates); replan_budget (runaway loop -> advisory only after a "
        "clean, bound primary result, else demote).",
    ),
    (
        "5. Verdict",
        "The status ladder (publication_ready > manuscript_ready > analysis_only "
        "> diagnostic_only) and the scorecard tristate (gate_reportable / "
        "analysis_only / diagnostic_only) floor to diagnostic_only whenever a "
        "gate fails, with the specific reason surfaced. INVARIANT: a capability "
        "gap is always reported, never silently filled with a fabricated result.",
    ),
)


# ---------------------------------------------------------------------------
# Accessors + renderer
# ---------------------------------------------------------------------------


def get_capability(
    family: StudyDesignFamily,
    *,
    dose_response: bool = False,
    freeform: bool = False,
) -> Optional[ScientificCapability]:
    """Return the capability record for a family.

    ``association`` has exact, free-form and dose-response records. The default
    is the exact ``association_adjusted_v1`` capability used by
    ``association_study``; callers must explicitly request the broader path.
    """
    matches = [c for c in CAPABILITY_REGISTRY if c.family == family]
    if not matches:
        return None
    if family == "association":
        wanted = (
            "association_ordinal_trend_v1"
            if dose_response
            else "association_freeform_v1"
            if freeform
            else "association_adjusted_v1"
        )
        return next((c for c in matches if c.capability_id == wanted), None)
    return matches[0]


def get_capability_by_id(
    capability_id: Optional[str],
) -> Optional[ScientificCapability]:
    """Return one directly declared capability, rejecting unknown ids."""

    wanted = str(capability_id or "").strip()
    if not wanted:
        return None
    matches = [
        capability
        for capability in CAPABILITY_REGISTRY
        if capability.capability_id == wanted
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _declares_host_association_product_key(step: object) -> bool:
    """Whether a step names the sealed executor's product key at all.

    Asked independently of the method, because the product key -- not the
    method string -- is what ``bind_primary_output`` and the deterministic
    figure lineage read to decide what a table is.
    """

    return ADJUSTED_ASSOCIATION_OUTPUT in {
        str(value or "").strip().lower()
        for value in tuple(getattr(step, "expected_outputs", ()) or ())
    }


def _claims_host_association_product(step: object) -> bool:
    """Whether one step claims the host's exact adjusted-association product.

    This is a *claim*, not an ownership decision: the step names the method and
    the product key that belong to the sealed executor. Whether that executor
    can actually run it is :func:`association_execution_verdict`'s answer, and
    keeping the two apart is the whole point -- a step that claims the host
    product and declares an estimator the host does not implement is a
    contradiction to report, not a step to quietly re-route.
    """

    method = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(step, "method", "") or "").lower().split(" with ", 1)[0],
    ).strip("_")
    return method == PLANNED_MODEL_REQUIREMENTS_STEP_METHOD and (
        _declares_host_association_product_key(step)
    )


def _plan_declares_exact_adjusted_association(plan: object) -> bool:
    """Whether any primary step claims the deterministic association contract."""

    return any(
        _claims_host_association_product(step)
        for step in tuple(getattr(plan, "steps", ()) or ())
        if getattr(step, "planned_analysis_role", None) == "primary"
    )


@dataclass(frozen=True)
class PrimaryCapabilityVerdict:
    """One answer to "who computes the primary result, and may it be reported?"

    Four layers used to answer this separately -- the capability registry, the
    Planner's primary-result validator, runtime executor selection and
    readiness -- and on ``ba11f52`` they disagreed on a plan that is entirely
    legal to write.  A plan declaring ``adjusted_association_models``,
    ``table:adjusted_association_estimates`` and
    ``method_family='statsmodels_glm_binomial'`` (a canonical token) was
    labelled ``association_adjusted_v1`` / ``deterministic`` by the registry,
    accepted by plan validation, declined ``wrong_shape`` by the owner, passed
    over in silence by the owner-declaration gate -- and executed by the LLM
    coder.  The label reached ``run_status.json`` and readiness; the execution
    did not match it.

    ``failure_reason`` is the field that makes this type worth having: a
    verdict may be *incoherent*, and saying so is different from choosing one
    of the two coherent answers on the plan's behalf.  Re-routing a
    GLM-binomial contract to the free-form capability would let any plan escape
    the deterministic owner by naming an estimator it does not implement;
    coercing it to Logit would answer a different scientific question under the
    declared method's name.  So the resolver reports the contradiction and the
    Planner picks.
    """

    capability_id: Optional[str]
    analysis_family: Optional[str]
    execution_owner: Literal["host_deterministic", "agent_coded", "unresolved"]
    #: ``None`` when the family has no host owner to ask.
    owner_claimed: Optional[bool]
    owner_reason: str
    scientific_validation: Literal["reportable", "analysis_only", "unsupported"]
    #: A stable code, or ``None`` when the four layers agree.  Callers branch
    #: on this; the ``detail`` wording may change.
    failure_reason: Optional[str]
    detail: str
    capability: Optional[ScientificCapability] = None

    @property
    def coherent(self) -> bool:
        """Whether the declared capability and the real execution owner agree."""

        return self.failure_reason is None


def _verdict_for(
    capability: Optional[ScientificCapability],
    *,
    analysis_family: Optional[str],
    owner_claimed: Optional[bool] = None,
    owner_reason: str = "",
    failure_reason: Optional[str] = None,
    detail: str = "",
) -> PrimaryCapabilityVerdict:
    if capability is None:
        return PrimaryCapabilityVerdict(
            capability_id=None,
            analysis_family=analysis_family,
            execution_owner="unresolved",
            owner_claimed=None,
            owner_reason=owner_reason,
            scientific_validation="unsupported",
            failure_reason=failure_reason or "scientific_capability_unregistered",
            detail=detail or "No scientific capability is registered for this family.",
        )
    if failure_reason is not None:
        owner = "unresolved"
    elif capability.primary_analysis == "deterministic":
        owner = "host_deterministic"
    else:
        owner = "agent_coded"
    return PrimaryCapabilityVerdict(
        capability_id=capability.capability_id,
        analysis_family=analysis_family,
        execution_owner=owner,
        owner_claimed=owner_claimed,
        owner_reason=owner_reason,
        scientific_validation=(
            "unsupported"
            if failure_reason is not None
            else capability.scientific_validation
        ),
        failure_reason=failure_reason,
        detail=detail,
        capability=capability,
    )


def resolve_primary_capability(
    *,
    analysis_type: Optional[str],
    plan: object = None,
) -> PrimaryCapabilityVerdict:
    """Resolve capability, execution owner and reportability in one place.

    Read by plan validation, the capability assessment written into
    ``run_status.json`` and readiness.  Runtime executor selection agrees by
    construction rather than by convention: this function and the owner call
    the same :func:`association_execution_verdict`.
    """

    raw_type = str(analysis_type or "").strip()
    if not raw_type:
        return _verdict_for(
            None,
            analysis_family=None,
            failure_reason="analysis_family_unresolved",
            detail="No canonical analysis family was declared for this run.",
        )
    try:
        from .analysis_types import canonical_analysis_family, get_analysis_type

        canonical = canonical_analysis_family(raw_type)
        if canonical is None:
            raise ValueError("unknown analysis type")
        capability = get_capability_by_id(get_analysis_type(canonical).capability_id)
    except (TypeError, ValueError):
        return _verdict_for(
            None,
            analysis_family=raw_type,
            failure_reason="scientific_capability_unregistered",
            detail=f"No scientific capability is registered for {raw_type!r}.",
        )

    if capability is None or plan is None:
        return _verdict_for(capability, analysis_family=canonical)

    primary_steps = [
        step
        for step in tuple(getattr(plan, "steps", ()) or ())
        if getattr(step, "planned_analysis_role", None) == "primary"
    ]
    # No primary step: keep the conservative deterministic default so an
    # under-declared plan cannot obtain a looser publication contract.
    if not primary_steps:
        return _verdict_for(capability, analysis_family=canonical)

    primary = primary_steps[0]
    if canonical == "descriptive_epidemiology":
        descriptive_verdict = exposure_outcome_distribution_execution_verdict(
            primary
        )
        if descriptive_verdict.claimed:
            capability = get_capability_by_id(
                DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID
            )
            return _verdict_for(
                capability,
                analysis_family=canonical,
                owner_claimed=True,
                owner_reason=descriptive_verdict.reason,
            )
    declared = str(getattr(primary, "scientific_capability", "") or "").strip()
    if declared:
        declared_capability = get_capability_by_id(declared)
        if declared_capability is None:
            return _verdict_for(
                capability,
                analysis_family=canonical,
                failure_reason="scientific_capability_unknown",
                detail=(
                    f"The primary step declares unknown scientific_capability "
                    f"{declared!r}; declare a capability id from the registry."
                ),
            )
        if declared_capability.family != capability.family:
            return _verdict_for(
                capability,
                analysis_family=canonical,
                failure_reason="scientific_capability_family_mismatch",
                detail=(
                    f"scientific_capability {declared!r} belongs to family "
                    f"{declared_capability.family!r}, not {capability.family!r}."
                ),
            )

    if capability.capability_id != "association_adjusted_v1":
        if declared and declared != capability.capability_id:
            return _verdict_for(
                capability,
                analysis_family=canonical,
                failure_reason="scientific_capability_step_incompatible",
                detail=(
                    f"scientific_capability {declared!r} is not compatible with "
                    f"the primary capability {capability.capability_id!r}."
                ),
            )
        return _verdict_for(capability, analysis_family=canonical)

    if declared == "association_freeform_v1":
        # Free-form is *declared*, never inferred. Inferring it from "this step
        # does not match the exact contract" would make a feasibility audit and
        # an interaction model indistinguishable, and would hand every
        # under-declared plan the looser agent-coded obligations for free.
        if _declares_host_association_product_key(primary):
            return _verdict_for(
                capability,
                analysis_family=canonical,
                owner_claimed=None,
                owner_reason="the step declares free-form and the host product",
                failure_reason="freeform_step_claims_host_product",
                detail=(
                    "A step declaring scientific_capability="
                    "'association_freeform_v1' may not declare the sealed "
                    f"executor's product {ADJUSTED_ASSOCIATION_OUTPUT!r}. That "
                    "product key is the sealed executor's identity, read by "
                    "bind_primary_output and by the deterministic figure "
                    "lineage; a coder-written table under that name would "
                    "inherit the deterministic owner's contract without its "
                    "guarantees. Declare a result product of its own."
                ),
            )
        return _verdict_for(
            get_capability_by_id("association_freeform_v1"),
            analysis_family=canonical,
            owner_claimed=None,
            owner_reason="the primary step declares the agent-coded association kernel",
        )

    if declared and declared != capability.capability_id:
        return _verdict_for(
            capability,
            analysis_family=canonical,
            failure_reason="scientific_capability_step_incompatible",
            detail=(
                f"scientific_capability {declared!r} is registered for the "
                "association family but is not compatible with this primary "
                "step shape."
            ),
        )

    if not _claims_host_association_product(primary):
        return _verdict_for(
            capability,
            analysis_family=canonical,
            failure_reason="scientific_capability_declaration_required",
            detail=(
                "The association primary step is neither the exact host-owned "
                "adjusted-association contract nor an explicitly declared "
                "association_freeform_v1 step. Declare which registered "
                "scientific capability owns it."
            ),
        )

    verdict = association_execution_verdict(primary)
    if verdict.claimed:
        return _verdict_for(
            capability,
            analysis_family=canonical,
            owner_claimed=True,
            owner_reason=verdict.reason,
        )
    if verdict.missing_declarations:
        # Repairable: the plan-time owner-declaration gate already turns this
        # into one focused replan directive. The capability is still the
        # deterministic one, so nothing false is asserted by keeping the label.
        return _verdict_for(
            capability,
            analysis_family=canonical,
            owner_claimed=False,
            owner_reason=verdict.reason,
            failure_reason="primary_owner_declaration_incomplete",
            detail=(
                "The primary step claims the host adjusted-association product "
                "but has not declared: " + ", ".join(verdict.missing_declarations)
            ),
        )
    return _verdict_for(
        capability,
        analysis_family=canonical,
        owner_claimed=False,
        owner_reason=verdict.reason,
        failure_reason="primary_capability_owner_mismatch",
        detail=(
            "The primary step claims the host-owned deterministic product "
            f"{ADJUSTED_ASSOCIATION_OUTPUT!r}, but the sealed executor cannot "
            f"run it: {verdict.reason}. Either declare a contract this owner "
            "implements, or declare a free-form association primary step under "
            "a different result product -- the host will not substitute an "
            "estimator, and a coder-executed step may not carry a deterministic "
            "capability label."
        ),
    )


def get_capability_for_plan(
    *,
    analysis_type: Optional[str],
    plan: object = None,
) -> Optional[ScientificCapability]:
    """The capability record from :func:`resolve_primary_capability`.

    Kept as the compatibility surface for callers that only need the record.
    Callers that act on the answer should read the verdict instead: this
    function cannot express "the label and the execution owner disagree".
    """

    return resolve_primary_capability(analysis_type=analysis_type, plan=plan).capability


def assess_scientific_capability(
    *,
    analysis_type: Optional[str],
    context: "ResearchContext",
    plan: object = None,
) -> ScientificCapabilityAssessment:
    """State the maximum claim allowed by the declared pre-execution contract.

    This is a *capability* receipt, not an estimator or a data-quality check.
    ``input_contract_resolved`` means that the typed context names the input
    shape the capability requires; executors still prove row-level availability
    and backend availability in their own receipts. The assessment is
    deliberately conservative: an unregistered scientific validator keeps an
    otherwise executable run at ``analysis_only``.
    """

    question_present = bool(
        str(getattr(context, "research_question", "") or "").strip()
    )
    raw_type = str(analysis_type or "").strip()
    if not raw_type:
        return ScientificCapabilityAssessment(
            capability_id=None,
            analysis_type=None,
            question_present=question_present,
            question_coordinates_resolved=False,
            input_contract_resolved=False,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=False,
            claim_ceiling="unsupported",
            issue_code="analysis_family_unresolved",
            reason="No canonical analysis family was declared for this run.",
        )

    try:
        from .analysis_types import canonical_analysis_family

        canonical = canonical_analysis_family(raw_type)
        if canonical is None:
            raise ValueError("unknown analysis type")
        # These families currently use the neutral descriptive display brief,
        # not a scientific execution capability.  Do not mistake that UI
        # fallback for validation of multimodal/RL/transportability claims.
        if canonical in _DISPLAY_ONLY_ANALYSIS_TYPES:
            return ScientificCapabilityAssessment(
                capability_id=None,
                analysis_type=canonical,
                question_present=question_present,
                question_coordinates_resolved=False,
                input_contract_resolved=False,
                runtime_data_available=None,
                execution_backend_available=None,
                scientific_validator_available=False,
                claim_ceiling="unsupported",
                issue_code="scientific_capability_unregistered",
                reason=(
                    f"{canonical!r} has no registered scientific capability; "
                    "the descriptive display fallback is not a result validator."
                ),
            )
        verdict = resolve_primary_capability(analysis_type=canonical, plan=plan)
        capability = verdict.capability
        # Every resolver failure is fail-closed here. Planner validation is not
        # a substitute for readiness/replay safety: stale or hand-built plans
        # can reach this assessment without passing the normal parse path.
        if verdict.failure_reason is not None:
            repairable = verdict.failure_reason in {
                "primary_owner_declaration_incomplete",
                "scientific_capability_declaration_required",
            }
            return ScientificCapabilityAssessment(
                capability_id=verdict.capability_id,
                analysis_type=canonical,
                question_present=question_present,
                question_coordinates_resolved=False,
                input_contract_resolved=False,
                runtime_data_available=None,
                execution_backend_available=None,
                scientific_validator_available=False,
                claim_ceiling="analysis_only" if repairable else "unsupported",
                issue_code=verdict.failure_reason,
                reason=verdict.detail,
            )
    except (TypeError, ValueError):
        capability = None
        canonical = raw_type

    if capability is None:
        return ScientificCapabilityAssessment(
            capability_id=None,
            analysis_type=canonical,
            question_present=question_present,
            question_coordinates_resolved=False,
            input_contract_resolved=False,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=False,
            claim_ceiling="unsupported",
            issue_code="scientific_capability_unregistered",
            reason=f"No scientific capability is registered for {canonical!r}.",
        )

    exposure = str(getattr(context, "primary_exposure", "") or "").strip()
    outcome = str(getattr(context, "target_outcome", "") or "").strip()
    variables = tuple(getattr(context, "variables", ()) or ())
    # Input readiness follows the capability's own declared contract. A
    # phenotyping run has a feature matrix and no exposure/outcome by design;
    # requiring those association coordinates would silently misclassify it.
    if capability.family in {"association", "causal_emulation"}:
        input_contract_resolved = bool(exposure and outcome)
    elif capability.family == "prediction":
        input_contract_resolved = bool(outcome and variables)
    elif capability.family == "phenotyping":
        input_contract_resolved = bool(variables)
    else:
        input_contract_resolved = bool(getattr(context, "cohort", None))
    if capability.capability_id == "association_ordinal_trend_v1":
        exposure_descriptor = next(
            (
                variable
                for variable in variables
                if str(getattr(variable, "name", "") or "").strip() == exposure
            ),
            None,
        )
        levels = list(getattr(exposure_descriptor, "ordinal_levels", None) or [])
        input_contract_resolved = bool(
            exposure
            and outcome
            and bool(getattr(exposure_descriptor, "is_ordinal", False))
            and len(levels) >= 3
        )
    endpoint = getattr(context, "endpoint", None)
    if capability.family == "time_to_event":
        input_contract_resolved = bool(
            exposure
            and outcome
            and endpoint is not None
            and getattr(endpoint, "kind", None) == "time_to_event"
        )
        levels = list(getattr(endpoint, "levels", None) or [])
        if len(levels) > 2:
            return ScientificCapabilityAssessment(
                capability_id=capability.capability_id,
                analysis_type=canonical,
                question_present=question_present,
                question_coordinates_resolved=bool(
                    question_present and input_contract_resolved
                ),
                input_contract_resolved=input_contract_resolved,
                runtime_data_available=None,
                execution_backend_available=None,
                scientific_validator_available=False,
                claim_ceiling="unsupported",
                issue_code="competing_risk_estimator_unavailable",
                reason=(
                    "The endpoint declares multiple event types, but EasyICU has "
                    "no registered CIF/Fine-Gray capability. A cause-naive Cox "
                    "model must not stand in for a competing-risk claim."
                ),
            )

    if not question_present:
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_present=False,
            question_coordinates_resolved=False,
            input_contract_resolved=input_contract_resolved,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=False,
            claim_ceiling="analysis_only",
            issue_code="research_question_unresolved",
            reason="A scientific capability cannot validate an empty research question.",
        )
    if not input_contract_resolved:
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_present=True,
            question_coordinates_resolved=False,
            input_contract_resolved=False,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=False,
            claim_ceiling="analysis_only",
            issue_code="scientific_capability_data_contract_unresolved",
            reason=(
                "The typed context does not declare the exposure/outcome inputs "
                "required by this capability."
            ),
        )
    validator_registered = bool(
        capability.scientific_validation == "reportable"
        and capability.scientific_validator_owner
        and capability.scientific_validator_contract
    )
    if not validator_registered:
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_present=True,
            question_coordinates_resolved=True,
            input_contract_resolved=True,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=False,
            claim_ceiling="analysis_only",
            issue_code="scientific_validator_unavailable",
            reason=(
                "The analysis can execute, but no registered deterministic "
                "validator establishes its identification claim."
            ),
        )
    return ScientificCapabilityAssessment(
        capability_id=capability.capability_id,
        analysis_type=canonical,
        question_present=True,
        question_coordinates_resolved=True,
        input_contract_resolved=True,
        runtime_data_available=None,
        execution_backend_available=None,
        scientific_validator_available=True,
        claim_ceiling="reportable",
        reason=(
            "The registered contracts permit a reportable claim ceiling; runtime "
            "execution, evidence, numeric and manuscript gates must still pass."
        ),
    )


def deterministic_primary_families() -> Tuple[str, ...]:
    """Labels of families whose PRIMARY estimand is computed deterministically."""
    return tuple(
        c.label for c in CAPABILITY_REGISTRY if c.primary_analysis == "deterministic"
    )


def llm_coded_primary_families() -> Tuple[str, ...]:
    """Labels of families whose PRIMARY estimand is currently LLM-coded."""
    return tuple(
        c.label for c in CAPABILITY_REGISTRY if c.primary_analysis == "llm_coded"
    )


def families_without_deterministic_primary() -> frozenset:
    """StudyDesignFamily values where EVERY capability record is LLM-coded.

    This is derived per family because survival now has a sealed primary owner
    while other current families remain agent-primary.
    """
    by_family: dict = {}
    for c in CAPABILITY_REGISTRY:
        by_family.setdefault(c.family, []).append(c.primary_analysis)
    return frozenset(
        fam
        for fam, kinds in by_family.items()
        if kinds and all(k == "llm_coded" for k in kinds)
    )


def _tick(kind: str) -> str:
    return "deterministic ✅" if kind == "deterministic" else "LLM-coded ⚠️"


def render_capability_matrix_markdown() -> str:
    """Render the capability matrix + fail-closed ladder as Markdown.

    This is the human/reviewer-facing view; ``capability_matrix.md`` under
    ``docs/`` is generated from it and drift-tested for equality.
    """
    lines = [
        "# EasyICU research-agent capability matrix",
        "",
        "_Generated from `easyicu.research_agent.planning.capability_registry`. Do not edit "
        "by hand — edit the registry and regenerate._",
        "",
        "**Primary analysis** = how the reported estimand is computed. **Figure** = "
        "how the publication figure is rendered. The two are independent: a family "
        "can have a deterministic figure while its primary analysis is LLM-coded.",
        "",
        "| Study-design family | Primary analysis | Primary estimand | Runner | Figure | Fail-closed when contract unmet |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in CAPABILITY_REGISTRY:
        runner = f"`{c.primary_runner}`" if c.primary_runner else "—"
        fig = _tick(c.figure)
        if c.figure_renderer and c.figure == "deterministic":
            fig += f" (`{c.figure_renderer}`)"
        lines.append(
            f"| {c.label} | {_tick(c.primary_analysis)} | {c.primary_estimand} | "
            f"{runner} | {fig} | {c.fail_closed} |"
        )
    lines += [
        "",
        "## Scientific claim readiness",
        "",
        "A capability can execute an analysis without having a sufficient "
        "scientific validator for a publication claim. `analysis_only` is an "
        "explicit fail-closed boundary, not an error the Agent may write around.",
        "",
        "| Capability | Result contract | Validator owner | Required diagnostics | Claim ceiling |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in CAPABILITY_REGISTRY:
        status = (
            "reportable ✅"
            if c.scientific_validation == "reportable"
            else "analysis_only ⚠️"
        )
        diagnostics = "; ".join(c.required_diagnostics) or "—"
        validator = (
            f"`{c.scientific_validator_owner}` / `{c.scientific_validator_contract}`"
            if c.scientific_validator_owner and c.scientific_validator_contract
            else "—"
        )
        lines.append(
            f"| `{c.capability_id}` | {c.result_contract or '—'} | {validator} | "
            f"{diagnostics} | {status} |"
        )
    lines += [
        "",
        "## Auxiliary deterministic runners (support, not family-primary)",
        "",
        "| Runner | Purpose | Fail-closed |",
        "| --- | --- | --- |",
    ]
    for a in AUXILIARY_DETERMINISTIC_RUNNERS:
        lines.append(f"| `{a.name}` | {a.purpose} | {a.fail_closed} |")
    lines += [
        "",
        "## Known unsupported estimands (explicit boundaries)",
        "",
        "Deliberately out of scope — these must **fail closed**, not be "
        "approximated by a nearby estimand:",
        "",
    ]
    for name, why in KNOWN_UNSUPPORTED_ESTIMANDS:
        lines.append(f"- **{name}** — {why}")
    lines += [
        "",
        "## Fail-closed / gap-report ladder",
        "",
        "What happens when no valid runner or data contract exists — the pipeline "
        "fails **closed** with a surfaced reason, never open:",
        "",
    ]
    for stage, behavior in FAIL_CLOSED_LADDER:
        lines.append(f"- **{stage}** — {behavior}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual regen
    print(render_capability_matrix_markdown(), end="")
