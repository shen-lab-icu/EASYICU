"""EasyICU research-agent capability registry — the explicit capability surface.

This module is the SINGLE SOURCE OF TRUTH for one question a reviewer (or a new
user) will ask: *for a given study-design family, which scientific result is
agent-produced, which standardized products are rendered or audited
deterministically, and where does the framework fail closed?* The answer used to
be implicit, spread across the preflight
dispatch ladder in ``pipeline_execute``, the ``FAMILY_RENDERERS`` table, and the
readiness gates in ``pipeline_report``. Here it is declared once, rendered to a
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
from typing import Optional, Tuple

from .study_design_playbook import StudyDesignFamily

__all__ = [
    "FamilyCapability",
    "AuxiliaryRunner",
    "CAPABILITY_REGISTRY",
    "AUXILIARY_DETERMINISTIC_RUNNERS",
    "KNOWN_UNSUPPORTED_ESTIMANDS",
    "FAIL_CLOSED_LADDER",
    "get_capability",
    "deterministic_primary_families",
    "llm_coded_primary_families",
    "families_without_deterministic_primary",
    "render_capability_matrix_markdown",
]


@dataclass(frozen=True)
class FamilyCapability:
    """How one study-design family's primary estimand and figure are produced."""

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

CAPABILITY_REGISTRY: Tuple[FamilyCapability, ...] = (
    FamilyCapability(
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
            "certified follow-up time (followup_time_hours)",
            "event indicator (event_observed)",
        ),
        fail_closed=(
            "The agent step fails when certified follow-up/event inputs are absent, "
            "and survival plausibility/provenance gates reject invalid event counts, "
            "effect scales, or unsupported estimands."
        ),
        notes="The deterministic survival component renders declared Cox/KM products; it does not choose the time origin, method, exposure, or outcome.",
    ),
    FamilyCapability(
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
    ),
    FamilyCapability(
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
    ),
    FamilyCapability(
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
    ),
    FamilyCapability(
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
    ),
    FamilyCapability(
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
    ),
    FamilyCapability(
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
    ),
)


AUXILIARY_DETERMINISTIC_RUNNERS: Tuple[AuxiliaryRunner, ...] = (
    AuxiliaryRunner(
        name="absolute_risk_context",
        entrypoint="absolute_risk_context_code",
        module="deterministic_descriptive",
        purpose="Render descriptive exposure prevalence and absolute-risk context from an explicit product contract.",
        fail_closed="Declines figure/primary-effect contracts and blocks when the declared descriptive columns are unavailable.",
    ),
    AuxiliaryRunner(
        name="robustness_sensitivity",
        entrypoint="robustness_sensitivity_preflight_code",
        module="deterministic_robustness",
        purpose="Replay an agent-locked primary model across prespecified robustness variants.",
        fail_closed="Requires a locked model/specification contract and never selects the primary exposure, outcome, cohort, or estimator.",
    ),
    AuxiliaryRunner(
        name="missingness_measurement_audit",
        entrypoint="missingness_measurement_audit_code",
        module="deterministic_missingness",
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
        module="trajectory_stability_executor",
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
) -> Optional[FamilyCapability]:
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
        "_Generated from `easyicu.research_agent.capability_registry`. Do not edit "
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
