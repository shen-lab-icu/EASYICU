"""EasyICU research-agent capability registry — the explicit capability surface.

This module is the SINGLE SOURCE OF TRUTH for one question a reviewer (or a new
user) will ask: *for a given study-design family, does EasyICU compute the
primary estimand with a deterministic runner, fall back to LLM-coded analysis,
or fail closed?* The answer used to be implicit, spread across the preflight
dispatch ladder in ``pipeline_execute``, the ``FAMILY_RENDERERS`` table, and the
readiness gates in ``pipeline_report``. Here it is declared once, rendered to a
matrix, and kept honest by ``tests/research_agent/test_capability_registry.py``,
which cross-checks every claim below against the code that is actually wired
(``_PRIMARY_DETERMINISTIC_RUNNERS`` in both pipeline modules and
``figures.FAMILY_RENDERERS``). If a runner is added or removed without updating
this registry, that test fails — the matrix cannot silently rot.

Two design points this registry makes explicit:

* **Deterministic vs LLM-coded is a per-estimand property, not per-family.** A
  family can have a deterministic FIGURE renderer while its PRIMARY analysis is
  still LLM-coded (prediction, phenotyping), or vice versa. The registry keeps
  ``primary_analysis`` and ``figure`` separate for exactly this reason.
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
    "FAIL_CLOSED_LADDER",
    "get_capability",
    "deterministic_primary_families",
    "llm_coded_primary_families",
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
    primary_runner: Optional[str]  # name in _PRIMARY_DETERMINISTIC_RUNNERS, or None
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
        primary_analysis="deterministic",
        primary_estimand="Cox proportional-hazards hazard ratio (+ Kaplan-Meier curve data)",
        primary_runner="survival_primary_cox",
        primary_runner_module="deterministic_survival",
        figure="deterministic",
        figure_renderer="time_to_event",
        data_contract=(
            "exposure column",
            "certified follow-up time (followup_time_hours)",
            "event indicator (event_observed)",
        ),
        fail_closed=(
            "Runner blocks (status=blocked) when the follow-up column is absent "
            "or uncensored; the survival-estimand integrity gate rejects any "
            "headline that is not the deterministic Cox HR."
        ),
        notes="Follow-up column certified upstream by the 01b step.",
    ),
    FamilyCapability(
        family="causal_emulation",
        label="Causal inference / target-trial emulation",
        primary_analysis="deterministic",
        primary_estimand="Stabilised-IPTW marginal odds ratio (+ covariate balance, propensity, target-trial protocol)",
        primary_runner="causal_primary_iptw",
        primary_runner_module="deterministic_causal",
        figure="deterministic",
        figure_renderer="causal_emulation",
        data_contract=(
            "binarised exposure",
            "binary outcome",
            "adjustment set (config user_preferences.covariates, else demographics+severity)",
        ),
        fail_closed=(
            "Runner blocks with 'Missing required causal columns' or 'Exposure "
            "groups too small'; positivity enforced by propensity trimming. The "
            "design schematic reads the runner's target_trial_protocol.csv."
        ),
    ),
    FamilyCapability(
        family="association",
        label="Association — graded ordinal exposure (dose-response)",
        primary_analysis="deterministic",
        primary_estimand="Adjusted odds ratio per +1 stage (trend) + per-stage forest + monotonicity",
        primary_runner="ordinal_dose_response",
        primary_runner_module="deterministic_ordinal",
        figure="deterministic",
        figure_renderer="base_association_skill",
        data_contract=(
            "ordinal grade exposure (>=3 ordered integer levels)",
            "binary outcome",
            "adjustment set",
        ),
        fail_closed=(
            "Runner blocks with 'Could not resolve a graded ordinal exposure "
            "(>=3 levels)'; a binary/continuous exposure is never coerced into a "
            "grade. Routes only on an explicit dose-response signal."
        ),
        notes="A dose-response IS an association study; no separate family exists.",
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
            "LLM code failure -> code_repair -> if still failing the step fails, "
            "the execution gate floors the status to diagnostic_only, and the "
            "specific error is surfaced (never a silent pass)."
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
        primary_estimand="LLM-coded cluster solution + stability; outcome-by-cluster kept descriptive (not causal)",
        primary_runner=None,
        primary_runner_module=None,
        figure="deterministic",
        figure_renderer="phenotyping",
        data_contract=("feature matrix (e.g. first-24h trajectories)", "k selection"),
        fail_closed=(
            "figure_strategy anti-pattern blocks 'clusters are causal entities'; "
            "an LLM failure fails closed to diagnostic_only."
        ),
        notes="Cluster heatmap + stability + outcome-by-cluster figure is deterministic.",
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
        name="cohort_definition_overlap",
        entrypoint="cohort_definition_overlap_code",
        module="deterministic_sensitivity",
        purpose="Overlap / concordance of alternative cohort definitions.",
        fail_closed="Blocks with a reason when no alternative definition is registered.",
    ),
    AuxiliaryRunner(
        name="cohort_definition_sensitivity",
        entrypoint="cohort_definition_sensitivity_comparison_code",
        module="deterministic_sensitivity",
        purpose="Re-fit the primary estimand under alternative cohort definitions.",
        fail_closed=(
            "Degrades to a CLEAN skip (status=skipped, not_applicable) when no "
            "alternative_cohort_attrition.csv exists upstream — it does NOT block "
            "(this removed the H2 'produce the missing file' replan loop)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# The fail-closed / gap-report ladder: what happens when no valid runner or
# data contract exists. This is the answer to "what is the gap-report behavior?"
# ---------------------------------------------------------------------------

FAIL_CLOSED_LADDER: Tuple[Tuple[str, str], ...] = (
    (
        "1. Deterministic runner match",
        "A family's preflight predicate fires only for its PRIMARY result step "
        "(not a figure/sensitivity step). If it fires and the data contract is "
        "met, the deterministic estimand is used and owns its step contract.",
    ),
    (
        "2. Runner contract unmet",
        "The deterministic runner writes status=blocked + a SPECIFIC "
        "blocking_reason (e.g. missing exposure/outcome column, degenerate "
        "groups, non-ordinal exposure). It never guesses a surrogate — "
        "case-specific values come from research_context.json only. Auxiliary "
        "steps degrade to status=skipped + not_applicable when their input is "
        "legitimately absent.",
    ),
    (
        "3. No deterministic runner for the family",
        "The LLM coder generates the analysis; code_repair applies deterministic "
        "post-failure repairs (KeyError strip, missing-helper restore, ...).",
    ),
    (
        "4. Output / validity gates (fail-closed)",
        "execution_complete (any failed step -> False); evidence_complete "
        "(STRICT: unbound citations blocked); numeric_verified (value-level "
        "provenance: hallucinated numbers blocked); analysis_validated "
        "(plausibility + survival-estimand + figure-credit + headline==primary-"
        "estimand gates); replan_budget (runaway loop -> advisory if "
        "converged-clean with a bound deterministic primary, else demote).",
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
        want_runner = "ordinal_dose_response" if dose_response else None
        for c in matches:
            if c.primary_runner == want_runner:
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
