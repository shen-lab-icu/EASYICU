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

* **The agent owns every primary scientific analysis.** Deterministic code may
  validate calculations or render a declared standardized product, but it does
  not preflight-replace the agent's cohort, exposure, outcome, method, or
  estimand. ``primary_analysis`` and ``figure`` stay separate for exactly this
  reason.
* **A capability gap is always REPORTED, never silently filled.** When no valid
  runner/data contract exists the pipeline fails closed with a specific reason
  (see ``FAIL_CLOSED_LADDER``); it never fabricates a result or degrades a
  headline to a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from .study_design_playbook import StudyDesignFamily

if TYPE_CHECKING:
    from ..schema import ResearchContext

__all__ = [
    "ScientificCapability",
    "FamilyCapability",
    "ScientificCapabilityAssessment",
    "AuxiliaryRunner",
    "CAPABILITY_REGISTRY",
    "AUXILIARY_DETERMINISTIC_RUNNERS",
    "KNOWN_UNSUPPORTED_ESTIMANDS",
    "FAIL_CLOSED_LADDER",
    "get_capability",
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
    primary_runner: Optional[str]  # reserved; primary scientific runners are not wired
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
    scientific_validation: Literal["reportable", "analysis_only"] = "reportable"


# Backward-compatible public name.  New code should use ScientificCapability:
# it carries the result/diagnostic/publication contract, not merely a catalogue
# row about a study-design family.
FamilyCapability = ScientificCapability


@dataclass(frozen=True)
class ScientificCapabilityAssessment:
    """Run-status-facing answer to what one scientific capability can support."""

    capability_id: Optional[str]
    analysis_type: Optional[str]
    question_understood: bool
    data_available: bool
    estimator_available: bool
    scientific_validator_available: bool
    status: Literal["reportable", "analysis_only", "unsupported"]
    issue_code: Optional[str] = None
    reason: str = ""

    @property
    def publication_eligible(self) -> bool:
        return self.status == "reportable"

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-ready readiness receipt."""

        return {
            "schema_version": "easyicu.scientific_capability_assessment/1",
            "capability_id": self.capability_id,
            "analysis_type": self.analysis_type,
            "question_understood": self.question_understood,
            # This means the typed context declares every column-shaped input
            # required by the capability. Row-level availability remains the
            # responsibility of the executor and its evidence receipt.
            "data_available": self.data_available,
            "estimator_available": self.estimator_available,
            "scientific_validator_available": self.scientific_validator_available,
            "status": self.status,
            "publication_eligible": self.publication_eligible,
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
        primary_analysis="llm_coded",
        primary_estimand="Agent-coded time-to-event estimand under the declared survival method; value/provenance checked",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="time_to_event",
        data_contract=(
            "exposure column",
            "Agent-declared, authority-bound follow-up time",
            "Agent-declared, authority-bound event indicator",
        ),
        fail_closed=(
            "The agent step fails when verified follow-up/event inputs are absent, "
            "and survival plausibility/provenance gates reject invalid event counts, "
            "effect scales, or unsupported estimands."
        ),
        notes="The deterministic survival component renders declared Cox/KM products; it does not choose the time origin, method, exposure, or outcome.",
        capability_id="survival_time_to_event_v1",
        result_contract="family_primary_result_requirement + registered primary CSV",
        required_diagnostics=(
            "event/censoring closure",
            "time-origin binding",
            "proportional-hazards diagnostic when Cox is declared",
        ),
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
        label="Association — general (non-graded)",
        primary_analysis="llm_coded",
        primary_estimand="LLM-coded adjusted association (logistic/linear); bound via NumericClaim + primary-effect extractor",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=("exposure", "outcome", "covariates"),
        fail_closed=(
            "LLM code failure -> mechanical code_repair only (no deterministic "
            "association refit or estimator substitution) -> if still failing the "
            "step fails, the execution gate floors the status to diagnostic_only, "
            "and the specific error is surfaced (never a silent pass)."
        ),
        notes="Base figure skill renders forest + strata + missingness deterministically.",
        capability_id="association_adjusted_v1",
        result_contract="planned_model_requirement + registered adjusted estimate",
        required_diagnostics=("primary model contract", "effect/interval reconciliation"),
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
        required_diagnostics=("representation", "cluster stability", "descriptive outcome use"),
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
)


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
        "1. Agent method and product contract",
        "The planner/coder owns the scientific method, cohort, exposure and outcome. "
        "Deterministic code is limited to validated calculation primitives or an "
        "explicit auxiliary product contract; it does not preflight-replace a "
        "primary estimand.",
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
        "3. Agent execution",
        "The agent generates the planned primary analysis; code repair and "
        "statistical validators may repair implementation faults but never replace "
        "the declared scientific method.",
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
    family: StudyDesignFamily, *, dose_response: bool = False
) -> Optional[ScientificCapability]:
    """Return the capability record for a family.

    ``association`` has two records (general + dose-response); ``dose_response``
    selects the graded-exposure one.
    """
    matches = [c for c in CAPABILITY_REGISTRY if c.family == family]
    if not matches:
        return None
    if family == "association":
        for c in matches:
            is_graded = "graded ordinal" in c.label.lower()
            if is_graded == dose_response:
                return c
    return matches[0]


def assess_scientific_capability(
    *,
    analysis_type: Optional[str],
    context: "ResearchContext",
) -> ScientificCapabilityAssessment:
    """State whether the declared question can make a reportable claim.

    This is a *capability* receipt, not an estimator or a data-quality check.
    ``data_available`` means that the typed context names the input shape the
    capability requires; executors still prove row-level availability.  The
    assessment is deliberately conservative: an unregistered scientific
    validator keeps an otherwise executable run at ``analysis_only``.
    """

    question_understood = bool(str(getattr(context, "research_question", "") or "").strip())
    raw_type = str(analysis_type or "").strip()
    if not raw_type:
        return ScientificCapabilityAssessment(
            capability_id=None,
            analysis_type=None,
            question_understood=question_understood,
            data_available=False,
            estimator_available=False,
            scientific_validator_available=False,
            status="unsupported",
            issue_code="analysis_family_unresolved",
            reason="No canonical analysis family was declared for this run.",
        )

    try:
        from .analysis_types import canonical_analysis_family
        from .study_design import study_design_family_for_analysis_type

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
                question_understood=question_understood,
                data_available=False,
                estimator_available=False,
                scientific_validator_available=False,
                status="unsupported",
                issue_code="scientific_capability_unregistered",
                reason=(
                    f"{canonical!r} has no registered scientific capability; "
                    "the descriptive display fallback is not a result validator."
                ),
            )
        capability = get_capability(study_design_family_for_analysis_type(canonical))
    except (TypeError, ValueError):
        capability = None
        canonical = raw_type

    if capability is None:
        return ScientificCapabilityAssessment(
            capability_id=None,
            analysis_type=canonical,
            question_understood=question_understood,
            data_available=False,
            estimator_available=False,
            scientific_validator_available=False,
            status="unsupported",
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
        data_available = bool(exposure and outcome)
    elif capability.family == "prediction":
        data_available = bool(outcome and variables)
    elif capability.family == "phenotyping":
        data_available = bool(variables)
    else:
        data_available = bool(getattr(context, "cohort", None))
    endpoint = getattr(context, "endpoint", None)
    if capability.family == "time_to_event":
        data_available = bool(
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
                question_understood=question_understood,
                data_available=data_available,
                estimator_available=False,
                scientific_validator_available=False,
                status="unsupported",
                issue_code="competing_risk_estimator_unavailable",
                reason=(
                    "The endpoint declares multiple event types, but EasyICU has "
                    "no registered CIF/Fine-Gray capability. A cause-naive Cox "
                    "model must not stand in for a competing-risk claim."
                ),
            )

    if not question_understood:
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_understood=False,
            data_available=data_available,
            estimator_available=True,
            scientific_validator_available=False,
            status="analysis_only",
            issue_code="research_question_unresolved",
            reason="A scientific capability cannot validate an empty research question.",
        )
    if not data_available:
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_understood=True,
            data_available=False,
            estimator_available=True,
            scientific_validator_available=False,
            status="analysis_only",
            issue_code="scientific_capability_data_contract_unresolved",
            reason=(
                "The typed context does not declare the exposure/outcome inputs "
                "required by this capability."
            ),
        )
    if capability.scientific_validation != "reportable":
        return ScientificCapabilityAssessment(
            capability_id=capability.capability_id,
            analysis_type=canonical,
            question_understood=True,
            data_available=True,
            estimator_available=True,
            scientific_validator_available=False,
            status="analysis_only",
            issue_code="scientific_validator_unavailable",
            reason=(
                "The analysis can execute, but no registered deterministic "
                "validator establishes its identification claim."
            ),
        )
    return ScientificCapabilityAssessment(
        capability_id=capability.capability_id,
        analysis_type=canonical,
        question_understood=True,
        data_available=True,
        estimator_available=True,
        scientific_validator_available=True,
        status="reportable",
        reason="The registered result and diagnostic contracts can support a claim.",
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

    Every current family is agent-primary by design, so a deterministic primary
    can never be required merely to survive the replan-budget gate. The helper
    remains explicit and will automatically exclude a family if a future
    architecture deliberately introduces a true primary owner.
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
        "| Capability | Result contract | Required diagnostics | Claim status |",
        "| --- | --- | --- | --- |",
    ]
    for c in CAPABILITY_REGISTRY:
        status = (
            "reportable ✅"
            if c.scientific_validation == "reportable"
            else "analysis_only ⚠️"
        )
        diagnostics = "; ".join(c.required_diagnostics) or "—"
        lines.append(
            f"| `{c.capability_id}` | {c.result_contract or '—'} | "
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
    print(render_capability_matrix_markdown())
