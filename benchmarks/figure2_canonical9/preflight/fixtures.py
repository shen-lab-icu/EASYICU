"""Canonical9 typed ``AnalysisPlan`` fixtures + minimal synthetic cohorts.

Each fixture is derived **live** from the corresponding formal task protocol in
:func:`benchmarks.figure2_canonical9.evaluator.suite.easyicu_evaluation_protocol_suite`
(its ``expected_outputs`` and ``semantic_guardrails`` are read from the suite, so
they cannot silently drift) and is **diagnostic-only** (no paper authority; see
the package docstring).

Unlike the batch-1 fixtures, each case is **genuinely distinct**, not one shared
two-step logistic skeleton.  E1-E3 cover the basic association families; the
complex-family registry additionally exercises M2 prediction, H1 survival, H2
causal emulation, and H3 trajectory clustering without a Provider or patient
data.

* **E1** (sepsis_onset) carries an explicit *cohort-definition* step (derived
  Sepsis-3 = suspected-infection timing + SOFA, never an ICD proxy; visible
  denominator) ahead of Table 1 + the mortality association.
* **E2** (descriptive_association) carries a *within-window peak aggregation*
  step (mmol/L units) and a *missingness audit* step around the lactate-mortality
  association, and summarises skewed lactate with the median.
* **E3** (ordinal_dose_response) carries a *stage-stratified outcomes* step
  (explicit KDIGO boundaries 0-3 against LOS + mortality) and an *ordinal trend*
  step, and treats KDIGO as an ordered category (median/IQR, never mean/SD).

Each case exposes a set of :class:`GuardrailCheck` predicates — one per suite
``semantic_guardrail`` — that verify the plan/cohort *structurally* honours that
guardrail.  The preflight verifies this routing/structure, not publication-grade
science (a real model authoring correct plan/code is the Provider boundary).

Cohorts are tiny in-memory synthetic frames (no patient data).  Every column a
plan step declares as an input exists in the cohort so the deterministic Table 1
executor and the agent-owned primary both run on real data offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    TableOneSpec,
    TableOneVariableSpec,
    TrajectoryStabilitySpec,
)
from easyicu.research_agent.trajectory.plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)

from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)

# Catalog analysis-type key (see easyicu.research_agent.planning.analysis_types).
# All three E-series questions are adjusted/ordinal associations of an exposure
# with in-hospital mortality; the paper-suite ``kind`` (descriptive_association,
# ordinal_dose_response, sepsis_onset) is a *different* vocabulary and is NOT a
# planner analysis_type key.
_ASSOCIATION = "association_study"


def _suite_task(task_id: str):
    """Return the live suite task protocol (fixtures bind to it, never copy it)."""

    for task in easyicu_evaluation_protocol_suite().tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)


@dataclass(frozen=True)
class GuardrailCheck:
    """A structural predicate proving one suite guardrail is honoured by design.

    ``guardrail_index`` points into the case's ``semantic_guardrails`` tuple, so
    the mapping is checked against the live suite text; ``key`` is a short stable
    label for diagnostics; ``holds`` inspects the plan/cohort.
    """

    guardrail_index: int
    key: str
    holds: Callable[["PreflightCase"], bool]


FULFILLMENT_PRODUCED = "produced"
FULFILLMENT_PLANNED_ONLY = "planned_only"
FULFILLMENT_NOT_PRODUCED_OFFLINE = "not_produced_offline"
_FULFILLMENTS = frozenset(
    {
        FULFILLMENT_PRODUCED,
        FULFILLMENT_PLANNED_ONLY,
        FULFILLMENT_NOT_PRODUCED_OFFLINE,
    }
)


@dataclass(frozen=True)
class ProductMapping:
    """One live suite product and the honest scope of this offline harness.

    ``output_index`` is intentionally bound to the live suite tuple, rather
    than duplicating a product name in a second handwritten list.  A mapping
    may point at a plan step, but ``planned_only`` is not an artifact claim and
    ``not_produced_offline`` explicitly records products reserved for the
    paper-authority workflow (for example publication figures).  A produced
    output must also name an evidence-ID prefix; this makes the claim resolvable
    against the real run manifest instead of a hand-written status label.
    """

    output_index: int
    step_id: Optional[str]
    declared_fulfillment: str
    artifact_evidence_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.declared_fulfillment not in _FULFILLMENTS:
            raise ValueError(f"unknown fulfillment {self.declared_fulfillment!r}")
        if self.declared_fulfillment == FULFILLMENT_PRODUCED and (
            self.step_id is None or self.artifact_evidence_prefix is None
        ):
            raise ValueError(
                "a produced output must identify its producing step and evidence prefix"
            )
        if (
            self.declared_fulfillment == FULFILLMENT_PLANNED_ONLY
            and self.step_id is None
        ):
            raise ValueError("a planned-only output must identify its plan step")
        if self.declared_fulfillment == FULFILLMENT_NOT_PRODUCED_OFFLINE and (
            self.step_id is not None or self.artifact_evidence_prefix is not None
        ):
            raise ValueError(
                "an unproduced offline output must not name a plan step or artifact"
            )


@dataclass(frozen=True)
class PreflightCase:
    """One diagnostic-only preflight case bound to a live suite task."""

    task_id: str
    title: str
    analysis_type: str
    question: str
    database: str
    primary_exposure: str
    target_outcome: str
    concept_descriptions: Dict[str, str]
    deterministic_step_id: str
    primary_step_id: str
    _build_plan: Callable[[], AnalysisPlan]
    _build_cohort: Callable[[int], pd.DataFrame]
    primary_code_kind: str = "association"
    required_imports: Tuple[str, ...] = (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
    )
    guardrail_checks: Tuple[GuardrailCheck, ...] = ()
    product_map: Tuple[ProductMapping, ...] = ()
    # A minimal synthetic dev fixture intentionally does not satisfy the full
    # article display contract, so the honest fail-closed verdict is
    # ``diagnostic_only`` (execution never completes the required suite).
    expected_tristate: str = "diagnostic_only"
    diagnostic_only: bool = True

    # -- construction -----------------------------------------------------
    def build_plan(self) -> AnalysisPlan:
        return self._build_plan()

    def build_cohort(self, n: int = 80) -> pd.DataFrame:
        return self._build_cohort(n)

    # -- live suite binding ----------------------------------------------
    @property
    def expected_products(self) -> Tuple[str, ...]:
        return tuple(_suite_task(self.task_id).expected_outputs)

    @property
    def semantic_guardrails(self) -> Tuple[str, ...]:
        return tuple(_suite_task(self.task_id).semantic_guardrails)

    # -- structural accessors (for the guardrail predicates) --------------
    def plan_methods(self) -> List[str]:
        return [str(s.method or "") for s in self.build_plan().steps]

    def plan_expected_outputs(self) -> List[str]:
        out: List[str] = []
        for step in self.build_plan().steps:
            out.extend(step.expected_outputs)
        return out

    def product_mapping(self) -> Tuple[tuple[str, ProductMapping], ...]:
        """Return a one-to-one mapping against the current live suite text."""

        products = self.expected_products
        indices = [mapping.output_index for mapping in self.product_map]
        if sorted(indices) != list(range(len(products))):
            raise AssertionError(
                f"{self.task_id} product map must cover each live expected output once; "
                f"got {indices}, expected 0..{len(products) - 1}"
            )
        return tuple(
            (products[mapping.output_index], mapping) for mapping in self.product_map
        )

    def cohort_columns(self) -> List[str]:
        return [str(c) for c in self.build_cohort().columns]

    def table_one_variable(self, name: str) -> TableOneVariableSpec:
        for step in self.build_plan().steps:
            if step.table_one_spec is None:
                continue
            for var in step.table_one_spec.variables:
                if var.name == name:
                    return var
        raise AssertionError(f"{name!r} is not a Table 1 variable in {self.task_id}")


# ---------------------------------------------------------------------------
# Shared typed-step builders
# ---------------------------------------------------------------------------


def _table_one_step(
    *,
    step_id: str,
    group_by: str,
    group_levels: List[object],
    variables: List[TableOneVariableSpec],
    intent: str,
) -> AnalysisStep:
    """Typed grouped Table 1 owned by the deterministic executor.

    ``expected_outputs == ['table:table_one']`` exactly and every summarised
    variable is an explicit step input — the exact contract
    ``table_one_executor_owns_step`` requires.
    """

    inputs = [group_by, *(v.name for v in variables)]
    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=intent,
        inputs=inputs,
        expected_outputs=["table:table_one"],
        method="grouped_table_one",
        table_one_spec=TableOneSpec(
            group_by=group_by,
            group_levels=group_levels,
            variables=variables,
        ),
    )


def _primary_association_step(
    *,
    step_id: str,
    exposure: str,
    outcome: str,
    adjust: List[str],
    intent: str,
) -> AnalysisStep:
    """Agent-owned (LLM-coded) typed adjusted-association step."""

    inputs = [exposure, *adjust, outcome]
    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="primary",
        intent=intent,
        inputs=inputs,
        expected_outputs=["table:association_model_diagnostics"],
        method="agent_coded_adjusted_association",
        scientific_capability="association_freeform_v1",
    )


def _aux_step(
    *,
    step_id: str,
    method: str,
    inputs: List[str],
    expected_outputs: List[str],
    intent: str,
) -> AnalysisStep:
    """A distinct, task-specific auxiliary step (routed to the Coder offline).

    Uses ``table:``/``dataset:``/``statistic:`` products only — deliberately not
    a sealed ``figure:`` artifact — so the offline routing check never depends on
    the per-family sealed-figure renderer contract.
    """

    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=intent,
        inputs=inputs,
        expected_outputs=expected_outputs,
        method=method,
    )


def _robustness_step(*, step_id: str) -> AnalysisStep:
    """Declare article-level sensitivity products without inventing endpoints."""

    return _aux_step(
        step_id=step_id,
        method="robustness_sensitivity",
        inputs=["table:primary_adjusted_estimate"],
        expected_outputs=[
            "table:robustness_matrix",
            "statistic:robustness_summary",
        ],
        intent=(
            "Replay the primary association under the pre-specified baseline-"
            "covariate missingness strategy; never impute the exposure or outcome."
        ),
    )


def _planned_effect_step(*, step_id: str) -> AnalysisStep:
    """Keep the formal primary-estimand role visible but unclaimed offline.

    The preceding free-form Coder smoke publishes model diagnostics only.  A
    downstream lineage step declares where a formally contracted effect would
    belong, while the generic offline response deliberately cannot produce it.
    """

    return _aux_step(
        step_id=step_id,
        method="planned_primary_effect_estimate",
        inputs=["table:association_model_diagnostics"],
        expected_outputs=["table:primary_adjusted_estimate"],
        intent=(
            "Reserve the formal primary-effect result for a typed scientific "
            "contract; the diagnostic-only offline smoke must not emit it."
        ),
    )


def _baseline_missingness_robustness_spec() -> List[Dict[str, object]]:
    """One case-neutral sensitivity spec supported by all synthetic fixtures."""

    return [
        {
            "spec_id": "baseline_covariate_median",
            "axis": "missing",
            "description": (
                "Median-impute declared baseline adjustment covariates only; "
                "leave the exposure and outcome unimputed."
            ),
            "missing_override": {
                "strategy": "median_imputation",
                "scope": "baseline_adjustment_covariates_only",
                "exclude_roles": ["exposure", "outcome"],
            },
        }
    ]


def _continuous(name: str) -> TableOneVariableSpec:
    return TableOneVariableSpec(
        name=name,
        variable_kind="continuous",
        summary="median_iqr",
        test="mann_whitney_or_kruskal",
    )


def _ordinal(name: str) -> TableOneVariableSpec:
    # Ordinal stage summarised as median (IQR) with a rank test — never mean/SD,
    # honouring the E3 "treat KDIGO as an ordered category" guardrail.
    return TableOneVariableSpec(
        name=name,
        variable_kind="ordinal",
        summary="median_iqr",
        test="mann_whitney_or_kruskal",
    )


# ---------------------------------------------------------------------------
# Complex family helpers
# ---------------------------------------------------------------------------


def _trajectory_stability_spec() -> TrajectoryStabilitySpec:
    """Small, closed Planner-owned stability design for the zero-API fixture."""

    return TrajectoryStabilitySpec(
        resampling_method="subsample_without_replacement",
        n_resamples=3,
        sample_fraction=0.8,
        sample_fraction_rounding="floor",
        base_seed=1729,
        seed_derivation="numpy_seedsequence_spawn_uint32_v1",
        cross_resample_membership="distinct_membership_required",
        stability_metric="adjusted_rand_index",
        stability_aggregation="mean",
        metric_label_source="raw_refit_labels_label_invariant",
        evaluation_scope="sampled_overlap",
        label_alignment="hungarian_maximum_overlap",
        label_alignment_reference="frozen_candidate_assignments",
        label_alignment_tie_break="minimum_rank_distance_then_lexicographic_v1",
        final_assignment_policy="copy_selected_candidate_labels",
        minimum_successful_resamples=3,
        failed_refit_policy="record_once_no_retry",
        refit_engine="easyicu_observed_data_diag_gmm_v1",
        refit_initialization="random_balanced_assignments",
        refit_max_iter=60,
        refit_tolerance=1e-4,
        refit_regularization=1e-6,
        decision_mode="report_only",
        threshold_failure_action="fail_closed_require_planner_revision",
    )


# ---------------------------------------------------------------------------
# E1 — Sepsis-3 prevalence and in-hospital mortality (sepsis_onset)
# ---------------------------------------------------------------------------

_E1_TABLE_ONE = "02_table_one"
_E1_PRIMARY = "03_primary_association"


def _e1_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E1.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _aux_step(
                step_id="01_cohort_definition",
                method="cohort_definition_summary",
                inputs=["susp_infection", "sofa", "sepsis3"],
                expected_outputs=[
                    "table:cohort_definition",
                    "table:denominator_panel",
                ],
                intent=(
                    "State the explicit Sepsis-3 cohort denominator and "
                    "inclusion/exclusion: membership is derived from suspected "
                    "infection timing plus SOFA>=2 (never an ICD-code proxy); "
                    "diagnosis codes are used for membership only, not event "
                    "timing."
                ),
            ),
            _aux_step(
                step_id="01b_missingness_audit",
                method="missingness_measurement_audit",
                inputs=["sepsis3", "age", "death"],
                expected_outputs=["table:missingness_profile"],
                intent=(
                    "Report measurement availability for the exposure, outcome, "
                    "and adjustment covariates before fitting the model."
                ),
            ),
            _aux_step(
                step_id="01c_absolute_risk",
                method="binary_outcome_incidence_and_absolute_risk",
                inputs=["sepsis3", "death"],
                expected_outputs=["table:outcome_incidence"],
                intent=(
                    "Report Sepsis-3 prevalence and absolute mortality risk by "
                    "exposure group before the adjusted association."
                ),
            ),
            _table_one_step(
                step_id=_E1_TABLE_ONE,
                group_by="sepsis3",
                group_levels=[0, 1],
                variables=[_continuous("age"), _continuous("lactate")],
                intent="Baseline Table 1 grouped by Sepsis-3 cohort membership.",
            ),
            _primary_association_step(
                step_id=_E1_PRIMARY,
                exposure="sepsis3",
                outcome="death",
                adjust=["age"],
                intent="Adjusted association of Sepsis-3 with in-hospital mortality.",
            ),
            _planned_effect_step(step_id="03b_primary_effect"),
            _robustness_step(step_id="04_robustness"),
        ],
        robustness_specs=_baseline_missingness_robustness_spec(),
        rationale=(
            "E1 diagnostic-only preflight fixture: an explicit cohort-definition "
            "step (derived Sepsis-3, visible denominator) ahead of a typed "
            "Table 1 grouped by the Sepsis-3 flag and an agent-owned mortality "
            "association. Sepsis-3 is a pre-derived synthetic flag; the preflight "
            "verifies orchestration, not the derived-concept definition itself "
            "(that is the Provider/real-data boundary)."
        ),
    )


def _e1_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(101)
    age = rng.randint(45, 88, n).astype(float)
    # Derived Sepsis-3 = suspected infection AND SOFA>=2 — never an ICD lookup.
    susp_infection = rng.binomial(1, 0.55, n)
    sofa = rng.poisson(2.4, n)
    sepsis3 = ((susp_infection == 1) & (sofa >= 2)).astype(int)
    lactate = (1.2 + 1.6 * sepsis3 + rng.gamma(2.0, 0.7, n)).round(2)
    logit = -2.2 + 1.1 * sepsis3 + 0.02 * (age - 65)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "stay_id": range(1, n + 1),
            "subject_id": range(1, n + 1),
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "susp_infection": susp_infection.astype(int),
            "sofa": sofa.astype(int),
            "sepsis3": sepsis3,
            "lactate": lactate,
            "death": death,
        }
    )


_E1_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        guardrail_index=0,
        key="derived_not_icd",
        holds=lambda c: {"susp_infection", "sofa"}.issubset(c.cohort_columns())
        and not any("icd" in col.lower() for col in c.cohort_columns()),
    ),
    GuardrailCheck(
        guardrail_index=1,
        key="explicit_denominator",
        holds=lambda c: "cohort_definition_summary" in c.plan_methods()
        and "table:cohort_definition" in c.plan_expected_outputs(),
    ),
    GuardrailCheck(
        guardrail_index=2,
        key="icd_membership_not_timing",
        holds=lambda c: set(c.build_cohort()["sepsis3"].unique()).issubset({0, 1})
        and not any(
            tok in col.lower()
            for col in c.cohort_columns()
            for tok in ("offset", "onset", "_time")
        ),
    ),
)


# ---------------------------------------------------------------------------
# E2 — 24h peak lactate vs in-hospital mortality (descriptive_association)
# ---------------------------------------------------------------------------

_E2_TABLE_ONE = "01_table_one"
_E2_PRIMARY = "04_primary_association"


def _e2_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E2.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _aux_step(
                step_id="00_cohort_accounting",
                method="cohort_definition_summary",
                inputs=["lactate", "death"],
                expected_outputs=["table:denominator_panel"],
                intent=(
                    "Report the eligible denominator and complete analytic "
                    "denominator before lactate aggregation."
                ),
            ),
            _table_one_step(
                step_id=_E2_TABLE_ONE,
                group_by="death",
                group_levels=[0, 1],
                variables=[_continuous("lactate"), _continuous("age")],
                intent="Baseline Table 1 grouped by in-hospital survival status.",
            ),
            _aux_step(
                step_id="02_peak_aggregation",
                method="within_window_peak_aggregation",
                inputs=["lactate"],
                expected_outputs=["dataset:lactate_peak_first24h"],
                intent=(
                    "Aggregate lactate to the first-24h peak per ICU stay, "
                    "keeping mmol/L units (convert mg/dL if present)."
                ),
            ),
            _aux_step(
                step_id="03_missingness_audit",
                method="missingness_measurement_audit",
                inputs=["lactate"],
                expected_outputs=[
                    "table:missingness_audit",
                    "table:missingness_profile",
                ],
                intent=(
                    "Audit lactate measurement missingness and the within-window "
                    "aggregation before interpreting the association."
                ),
            ),
            _aux_step(
                step_id="03b_absolute_risk",
                method="binary_outcome_incidence_and_absolute_risk",
                inputs=["lactate", "death"],
                expected_outputs=["table:outcome_incidence"],
                intent=(
                    "Report the mortality event rate across pre-specified "
                    "first-24h peak lactate strata before adjustment."
                ),
            ),
            _primary_association_step(
                step_id=_E2_PRIMARY,
                exposure="lactate",
                outcome="death",
                adjust=["age"],
                intent="Adjusted association of first-24h peak lactate with mortality.",
            ),
            _planned_effect_step(step_id="04b_primary_effect"),
            _robustness_step(step_id="05_robustness"),
        ],
        robustness_specs=_baseline_missingness_robustness_spec(),
        rationale=(
            "E2 diagnostic-only preflight fixture: a within-window peak "
            "aggregation step (mmol/L) and a missingness audit around a typed "
            "Table 1 (skewed lactate summarised by median/IQR) and an agent-owned "
            "lactate-mortality association."
        ),
    )


def _e2_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(202)
    age = rng.randint(45, 88, n).astype(float)
    # Right-skewed lactate (gamma) so a median is the honest summary.
    lactate = rng.gamma(2.0, 1.5, n).round(2)
    logit = -2.4 + 0.28 * (lactate - 3.0) + 0.02 * (age - 65)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "stay_id": range(1, n + 1),
            "subject_id": range(1, n + 1),
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "lactate": lactate,
            "death": death,
        }
    )


_E2_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        guardrail_index=0,
        key="within_window_units",
        holds=lambda c: "within_window_peak_aggregation" in c.plan_methods()
        and "mmol/L" in c.concept_descriptions.get("lactate", ""),
    ),
    GuardrailCheck(
        guardrail_index=1,
        key="median_over_mean_for_skew",
        holds=lambda c: c.table_one_variable("lactate").summary == "median_iqr"
        and float(c.build_cohort()["lactate"].skew()) > 0.3,
    ),
)


# ---------------------------------------------------------------------------
# E3 — KDIGO AKI stage gradient vs LOS and mortality (ordinal_dose_response)
# ---------------------------------------------------------------------------

_E3_TABLE_ONE = "01_table_one"
_E3_PRIMARY = "04_primary_association"


def _e3_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E3.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _aux_step(
                step_id="00_cohort_accounting",
                method="cohort_definition_summary",
                inputs=["kdigo", "death"],
                expected_outputs=["table:denominator_panel"],
                intent=(
                    "Report the eligible and analytic denominators before "
                    "stage-stratified outcome analysis."
                ),
            ),
            _table_one_step(
                step_id=_E3_TABLE_ONE,
                group_by="death",
                group_levels=[0, 1],
                variables=[_ordinal("kdigo"), _continuous("age")],
                intent="Baseline Table 1 grouped by survival, KDIGO as ordinal.",
            ),
            _aux_step(
                step_id="02_stage_stratified",
                method="stage_stratified_outcomes",
                inputs=["kdigo", "los_icu", "death"],
                expected_outputs=[
                    "table:stage_stratified_outcomes",
                    "table:outcome_incidence",
                ],
                intent=(
                    "Stratify ICU length of stay and mortality by explicit KDIGO "
                    "stage boundaries 0, 1, 2, 3 (report each stage separately)."
                ),
            ),
            _aux_step(
                step_id="02b_missingness_audit",
                method="missingness_measurement_audit",
                inputs=["kdigo", "age", "death"],
                expected_outputs=["table:missingness_profile"],
                intent=(
                    "Report measurement availability for KDIGO stage, outcome, "
                    "and adjustment covariates before ordinal modelling."
                ),
            ),
            _aux_step(
                step_id="03_ordinal_trend",
                method="ordinal_trend_test",
                inputs=["kdigo", "death"],
                expected_outputs=["statistic:kdigo_ordinal_trend"],
                intent=(
                    "Test the ordinal trend across ordered KDIGO stages "
                    "(Cochran-Armitage style), not a continuous slope."
                ),
            ),
            _primary_association_step(
                step_id=_E3_PRIMARY,
                exposure="kdigo",
                outcome="death",
                adjust=["age"],
                intent="Adjusted ordinal KDIGO-stage gradient vs mortality.",
            ),
            _planned_effect_step(step_id="04b_primary_effect"),
            _robustness_step(step_id="05_robustness"),
        ],
        robustness_specs=_baseline_missingness_robustness_spec(),
        rationale=(
            "E3 diagnostic-only preflight fixture: a stage-stratified outcomes "
            "step (explicit KDIGO boundaries vs LOS + mortality) and an ordinal "
            "trend step around a typed Table 1 that summarises KDIGO as an "
            "ordered stage (median/IQR + rank test, never mean/SD) plus an "
            "agent-owned ordinal dose-response association."
        ),
    )


def _e3_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(303)
    age = rng.randint(45, 88, n).astype(float)
    kdigo = rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1])
    los_icu = (2.0 + 1.5 * kdigo + rng.gamma(2.0, 1.0, n)).round(2)
    logit = -2.5 + 0.5 * kdigo + 0.02 * (age - 65)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "stay_id": range(1, n + 1),
            "subject_id": range(1, n + 1),
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "kdigo": kdigo.astype(int),
            "los_icu": los_icu,
            "death": death,
        }
    )


_E3_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        guardrail_index=0,
        key="kdigo_ordinal_not_continuous",
        holds=lambda c: c.table_one_variable("kdigo").variable_kind == "ordinal"
        and c.table_one_variable("kdigo").summary != "mean_sd",
    ),
    GuardrailCheck(
        guardrail_index=1,
        key="explicit_stage_boundaries",
        holds=lambda c: "stage_stratified_outcomes" in c.plan_methods()
        and set(c.build_cohort()["kdigo"].unique()) == {0, 1, 2, 3},
    ),
)


# ---------------------------------------------------------------------------
# M2 — patient-grouped mortality prediction
# ---------------------------------------------------------------------------

_M2_TABLE_ONE = "01_table_one"
_M2_PRIMARY = "04_prediction_model_analysis"


def _m2_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=M2.question,
        analysis_type="prediction_model",
        steps=[
            _aux_step(
                step_id="00_modelling_cohort_flow",
                method="prediction_cohort_accounting",
                inputs=["patient_stay_id", "death"],
                expected_outputs=["table:modelling_cohort_flow"],
                intent=(
                    "Report eligible patients, stays, outcome-complete rows, and "
                    "the final modelling denominator before any split."
                ),
            ),
            _table_one_step(
                step_id=_M2_TABLE_ONE,
                group_by="death",
                group_levels=[0, 1],
                variables=[
                    _continuous("age"),
                    _continuous("lact"),
                    _continuous("sofa2"),
                ],
                intent="Baseline Table 1 grouped by in-hospital mortality.",
            ),
            _aux_step(
                step_id="02_patient_split_contract",
                method="patient_level_split_leakage_audit",
                inputs=["patient_stay_id", "death"],
                expected_outputs=["table:leakage_audit"],
                intent=(
                    "Derive the patient group from patient_stay_id before ':s'; "
                    "split those patient groups with zero train/test overlap and "
                    "fit preprocessing on the training partition only."
                ),
            ),
            _aux_step(
                step_id="02b_validation_design",
                method="patient_grouped_validation_design",
                inputs=["patient_stay_id", "death"],
                expected_outputs=["table:validation_design"],
                intent=(
                    "Register the patient-grouped holdout design, split seed, "
                    "training-only preprocessing rule, and leakage assertion."
                ),
            ),
            _aux_step(
                step_id="03_missingness_audit",
                method="missingness_measurement_audit",
                inputs=["age", "hr", "map", "lact", "sofa2", "death"],
                expected_outputs=["table:missingness_profile"],
                intent=(
                    "Audit first-24h predictor availability without using any "
                    "post-outcome variable."
                ),
            ),
            AnalysisStep(
                step_id=_M2_PRIMARY,
                planned_analysis_role="primary",
                intent=(
                    "Fit a first-24h mortality model on a patient-grouped split; "
                    "report held-out AUROC, average precision, recall, F1, Brier "
                    "score, calibration, and decision-curve net benefit with "
                    "registered numeric source tables."
                ),
                inputs=[
                    "patient_stay_id",
                    "age",
                    "hr",
                    "map",
                    "lact",
                    "sofa2",
                    "death",
                ],
                expected_outputs=[
                    "table:model_performance_train_test",
                    "table:model_coefficients",
                    "table:risk_predictions_test",
                    "table:roc_curve",
                    "table:calibration",
                    "table:calibration_curve",
                    "table:decision_curve",
                    "table:split_definition",
                    "figure:roc_curve",
                    "figure:calibration_curve",
                    "statistic:auc",
                ],
                method="prediction_model_analysis",
            ),
        ],
        rationale=(
            "M2 zero-Provider preflight: bind the patient-level split and "
            "anti-leakage contract before exercising the prediction Coder path."
        ),
    )


def _m2_cohort(n: int = 160) -> pd.DataFrame:
    rng = np.random.RandomState(404)
    patient = np.arange(n) // 2
    age = rng.randint(40, 89, n).astype(float)
    sofa2 = rng.poisson(3.0, n).astype(float)
    lact = rng.gamma(2.0, 1.1, n)
    hr = rng.normal(94.0, 14.0, n)
    map_value = rng.normal(72.0, 10.0, n)
    logit = (
        -3.2
        + 0.035 * (age - 60.0)
        + 0.26 * sofa2
        + 0.22 * (lact - 2.0)
        - 0.018 * (map_value - 70.0)
    )
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, n + 1),
            "subject_id": patient + 1,
            "patient_stay_id": [
                f"p{subject:04d}:s{stay}" for subject, stay in zip(patient, np.arange(n) % 2)
            ],
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "hr": hr,
            "map": map_value,
            "lact": lact,
            "sofa2": sofa2,
            "death": death,
        }
    )


_M2_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        0,
        "patient_grouped_split",
        lambda c: "patient_stay_id"
        in next(
            step.inputs
            for step in c.build_plan().steps
            if step.step_id == c.primary_step_id
        )
        and "before ':s'"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == "02_patient_split_contract"
        ),
    ),
    GuardrailCheck(
        1,
        "no_post_outcome_features",
        lambda c: not any(
            token in input_name.casefold()
            for step in c.build_plan().steps
            for input_name in step.inputs
            for token in ("post_outcome", "discharge_disposition", "death_time")
        ),
    ),
    GuardrailCheck(
        2,
        "imbalance_and_calibration",
        lambda c: {
            "table:model_performance_train_test",
            "table:calibration_curve",
        }.issubset(set(c.plan_expected_outputs()))
        and all(
            token in next(
                step.intent
                for step in c.build_plan().steps
                if step.step_id == c.primary_step_id
            )
            for token in ("average precision", "recall", "F1")
        ),
    ),
    GuardrailCheck(
        3,
        "numeric_metric_binding",
        lambda c: {
            "table:risk_predictions_test",
            "table:roc_curve",
            "statistic:auc",
        }.issubset(set(c.plan_expected_outputs())),
    ),
)


# ---------------------------------------------------------------------------
# H1 — aligned incident-exposure survival analysis
# ---------------------------------------------------------------------------

_H1_TABLE_ONE = "02_table_one"
_H1_PRIMARY = "04_survival_analysis"


def _h1_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=H1.question,
        analysis_type="survival",
        steps=[
            _aux_step(
                step_id="01_incident_alignment",
                method="incident_landmark_eligibility",
                inputs=[
                    "vent_24h_any",
                    "vent_start_hours",
                    "followup_days",
                    "event_28d",
                ],
                expected_outputs=[
                    "table:incident_eligibility",
                    "table:risk_set_flow",
                    "table:time_zero_alignment",
                ],
                intent=(
                    "Fix a 24-hour landmark, exclude ventilation prevalent "
                    "before cohort entry, and start follow-up only after exposure "
                    "classification so future exposure cannot create immortal time."
                ),
            ),
            _table_one_step(
                step_id=_H1_TABLE_ONE,
                group_by="vent_24h_any",
                group_levels=[0, 1],
                variables=[_continuous("age"), _continuous("sofa2")],
                intent="Baseline Table 1 at the fixed survival-analysis landmark.",
            ),
            _aux_step(
                step_id="03_event_censoring_audit",
                method="event_censoring_audit",
                inputs=["followup_days", "event_28d"],
                expected_outputs=[
                    "table:event_censoring_audit",
                    "table:measurement_process_audit",
                ],
                intent="Audit 28-day event and censoring definitions.",
            ),
            AnalysisStep(
                step_id=_H1_PRIMARY,
                planned_analysis_role="primary",
                intent=(
                    "Estimate Kaplan-Meier survival and an adjusted Cox model "
                    "from the fixed landmark, then register proportional-hazards "
                    "diagnostics before interpreting a single hazard ratio."
                ),
                inputs=[
                    "vent_24h_any",
                    "followup_days",
                    "event_28d",
                    "age",
                    "sofa2",
                ],
                expected_outputs=[
                    "table:survival_curve",
                    "table:cox_summary",
                    "table:survival_diagnostics",
                    "table:ph_diagnostics",
                    "statistic:hazard_ratio",
                ],
                method="cox_proportional_hazards",
            ),
        ],
        rationale=(
            "H1 zero-Provider preflight freezes time zero, incident eligibility, "
            "censoring, and PH diagnostics before the survival Coder path."
        ),
    )


def _h1_cohort(n: int = 140) -> pd.DataFrame:
    rng = np.random.RandomState(505)
    age = rng.randint(40, 90, n).astype(float)
    sofa2 = rng.poisson(3.2, n).astype(float)
    vent = rng.binomial(1, 0.42, n)
    event_rate = np.exp(-3.4 + 0.55 * vent + 0.03 * (age - 60) + 0.14 * sofa2)
    latent_event = rng.exponential(1.0 / np.maximum(event_rate, 1e-4))
    followup = np.minimum(latent_event, 28.0)
    event = (latent_event <= 28.0).astype(int)
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, n + 1),
            "subject_id": np.arange(1, n + 1),
            "age": age,
            "sofa2": sofa2,
            "vent_24h_any": vent,
            "vent_start_hours": np.where(vent == 1, rng.uniform(1.0, 23.0, n), np.nan),
            "followup_days": followup,
            "event_28d": event,
            "death": event,
        }
    )


_H1_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        0,
        "fixed_landmark_no_future_classification",
        lambda c: "24-hour landmark"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == "01_incident_alignment"
        ),
    ),
    GuardrailCheck(
        1,
        "incident_only",
        lambda c: "exclude ventilation prevalent"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == "01_incident_alignment"
        ),
    ),
    GuardrailCheck(
        2,
        "ph_diagnostics",
        lambda c: "table:ph_diagnostics" in c.plan_expected_outputs(),
    ),
)


# ---------------------------------------------------------------------------
# H2 — causal emulation with exposure capture, balance, and positivity
# ---------------------------------------------------------------------------

_H2_TABLE_ONE = "02_table_one"
_H2_PRIMARY = "04_causal_emulation"


def _h2_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=H2.question,
        analysis_type="causal_inference",
        steps=[
            _aux_step(
                step_id="01_exposure_contract",
                method="exposure_capture_authority_audit",
                inputs=["vasopressor", "vasopressor_capture_available"],
                expected_outputs=[
                    "table:causal_cohort_flow",
                    "table:exposure_capture_audit",
                ],
                intent=(
                    "Treat an absent vasopressor record as no exposure only when "
                    "the source-specific capture flag is true; otherwise mark the "
                    "exposure unavailable and stop."
                ),
            ),
            _table_one_step(
                step_id=_H2_TABLE_ONE,
                group_by="vasopressor",
                group_levels=[0, 1],
                variables=[
                    _continuous("age"),
                    _continuous("sofa2"),
                    _continuous("lact"),
                ],
                intent="Baseline Table 1 by early vasopressor exposure.",
            ),
            _aux_step(
                step_id="03_target_trial_protocol",
                method="target_trial_protocol",
                inputs=["vasopressor", "death", "age", "sofa2", "lact", "map"],
                expected_outputs=[
                    "table:target_trial_protocol",
                    "artifact:target_trial_protocol",
                ],
                intent=(
                    "Freeze eligibility, time zero, 0-24h treatment strategies, "
                    "28-day outcome, confounding set, and ATE estimand before fitting."
                ),
            ),
            AnalysisStep(
                step_id=_H2_PRIMARY,
                planned_analysis_role="primary",
                intent=(
                    "Estimate a stabilized IPTW ATE-style risk difference while "
                    "making confounding by indication explicit; register weighted "
                    "balance and positivity diagnostics and bound conclusions to "
                    "the observational target-trial assumptions."
                ),
                inputs=["vasopressor", "death", "age", "sofa2", "lact", "map"],
                expected_outputs=[
                    "table:primary_causal_contrast",
                    "table:causal_effect",
                    "table:baseline_balance",
                    "table:covariate_balance",
                    "table:positivity_diagnostics",
                    "artifact:assignment_model",
                    "statistic:risk_difference",
                ],
                method="causal_emulation",
            ),
            _aux_step(
                step_id="05_causal_sensitivity",
                method="robustness_sensitivity",
                inputs=["table:primary_causal_contrast"],
                expected_outputs=[
                    "table:causal_sensitivity",
                    "table:robustness_matrix",
                    "statistic:robustness_summary",
                ],
                intent=(
                    "Repeat the declared causal contrast under the pre-specified "
                    "baseline-covariate missingness strategy without changing "
                    "exposure, outcome, time zero, or estimand."
                ),
            ),
        ],
        robustness_specs=_baseline_missingness_robustness_spec(),
        rationale=(
            "H2 zero-Provider preflight binds exposure capture and the target-"
            "trial estimand before exercising propensity/balance control flow."
        ),
    )


def _h2_cohort(n: int = 180) -> pd.DataFrame:
    rng = np.random.RandomState(606)
    age = rng.randint(40, 90, n).astype(float)
    sofa2 = rng.poisson(3.0, n).astype(float)
    lact = rng.gamma(2.0, 1.0, n)
    map_value = rng.normal(70.0, 10.0, n)
    treatment_logit = -1.0 + 0.32 * sofa2 + 0.22 * (lact - 2.0) - 0.03 * (
        map_value - 70.0
    )
    vasopressor = rng.binomial(1, 1.0 / (1.0 + np.exp(-treatment_logit)))
    outcome_logit = (
        -3.0
        + 0.45 * vasopressor
        + 0.28 * sofa2
        + 0.18 * (lact - 2.0)
        + 0.02 * (age - 60.0)
    )
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-outcome_logit)))
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, n + 1),
            "subject_id": np.arange(1, n + 1),
            "age": age,
            "sofa2": sofa2,
            "lact": lact,
            "map": map_value,
            "vasopressor": vasopressor,
            "vasopressor_capture_available": np.ones(n, dtype=int),
            "death": death,
        }
    )


_H2_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        0,
        "confounding_balance_positivity",
        lambda c: {
            "table:covariate_balance",
            "table:positivity_diagnostics",
        }.issubset(set(c.plan_expected_outputs()))
        and "confounding by indication"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == c.primary_step_id
        ),
    ),
    GuardrailCheck(
        1,
        "bounded_causal_claim",
        lambda c: "observational target-trial assumptions"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == c.primary_step_id
        ),
    ),
)


# ---------------------------------------------------------------------------
# H3 — fixed-anchor trajectory clustering with independent stability owner
# ---------------------------------------------------------------------------

_H3_TABLE_ONE = "01_table_one"
_H3_PRIMARY = "03_candidate_selection"


def _h3_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=H3.question,
        analysis_type="trajectory_clustering",
        steps=[
            _table_one_step(
                step_id=_H3_TABLE_ONE,
                group_by="death",
                group_levels=[0, 1],
                variables=[_continuous("age"), _continuous("lact_h0_6")],
                intent="Baseline Table 1 before trajectory representation.",
            ),
            _aux_step(
                step_id="02_representation",
                method="missingness_aware_trajectory_representation",
                inputs=[
                    "lact_h0_6",
                    "lact_h6_12",
                    "lact_h12_18",
                    "lact_h18_24",
                    "map_h0_6",
                    "map_h6_12",
                    "map_h12_18",
                    "map_h18_24",
                ],
                expected_outputs=[
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                    "table:feature_availability_flow",
                    "table:feature_quality_scaling",
                    "table:trajectory_membership",
                ],
                intent=(
                    "Align all trajectories to ICU admission with fixed 0-6, "
                    "6-12, 12-18, and 18-24h windows; retain the same eligible "
                    "population instead of selecting longer observed stays."
                ),
            ),
            AnalysisStep(
                step_id=_H3_PRIMARY,
                planned_analysis_role="primary",
                intent=(
                    "Compare pre-specified candidate cluster counts on the "
                    "fixed representation, freeze one selection manifest, and "
                    "describe classes without treating them as causal groups."
                ),
                inputs=[
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                ],
                expected_outputs=[
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "manifest:cluster_selection",
                    "manifest:candidate_cluster_solution_schema",
                ],
                method="model_based_clustering",
            ),
            AnalysisStep(
                step_id="04_stability_freeze",
                planned_analysis_role="auxiliary",
                intent=(
                    "Execute the independent Planner-owned stability design and "
                    "report its fixed decision without changing seed, k, or threshold."
                ),
                inputs=sorted(STABILITY_EXECUTOR_INPUTS),
                expected_outputs=sorted(STABILITY_EXECUTOR_OUTPUTS),
                method="trajectory_cluster_stability",
                trajectory_stability_spec=_trajectory_stability_spec(),
            ),
            _aux_step(
                step_id="05_characterization",
                method="descriptive_cluster_characterization",
                inputs=["artifact:cluster_assignments"],
                expected_outputs=[
                    "table:phenotype_profiles",
                    "table:trajectory_cluster_profiles",
                    "table:outcome_by_trajectory_class",
                ],
                intent=(
                    "Describe the frozen trajectory classes and outcomes without "
                    "causal interpretation."
                ),
            ),
        ],
        rationale=(
            "H3 zero-Provider preflight separates fixed-anchor representation, "
            "candidate selection, deterministic stability, and characterization."
        ),
    )


def _h3_cohort(n: int = 150) -> pd.DataFrame:
    rng = np.random.RandomState(707)
    latent = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
    data: Dict[str, object] = {
        "stay_id": np.arange(1, n + 1),
        "subject_id": np.arange(1, n + 1),
        "age": rng.randint(40, 90, n).astype(float),
    }
    windows = ((0, 6), (6, 12), (12, 18), (18, 24))
    for index, (start, end) in enumerate(windows):
        data[f"lact_h{start}_{end}"] = (
            1.5
            + 0.8 * latent
            + 0.22 * latent * float(index)
            + rng.normal(0, 0.25, n)
        )
        data[f"map_h{start}_{end}"] = (
            78.0
            - 6.0 * latent
            - 1.8 * latent * float(index)
            + rng.normal(0, 2.0, n)
        )
    outcome_logit = -3.0 + 0.9 * latent
    data["death"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-outcome_logit)))
    return pd.DataFrame(data)


_H3_CHECKS: Tuple[GuardrailCheck, ...] = (
    GuardrailCheck(
        0,
        "fixed_anchor_no_length_bias",
        lambda c: "fixed 0-6"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == "02_representation"
        )
        and "instead of selecting longer observed stays"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == "02_representation"
        ),
    ),
    GuardrailCheck(
        1,
        "fixed_cluster_selection_noncausal",
        lambda c: "pre-specified candidate cluster counts"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == c.primary_step_id
        )
        and "without treating them as causal groups"
        in next(
            step.intent
            for step in c.build_plan().steps
            if step.step_id == c.primary_step_id
        ),
    ),
)


# ---------------------------------------------------------------------------
# Formal-output scope maps
#
# The offline preflight proves orchestration paths.  It deliberately does not
# claim to render the formal publication figures or satisfy paper authority.
# Table 1 is the only formal output actually produced by the deterministic
# executor.  The remaining case-specific analysis steps are exercised as plan
# nodes, but their offline mock code is intentionally contract-failed; sealed
# figures are not produced at all.  Keeping that distinction explicit prevents
# a green preflight from being mistaken for E1/E2/E3 completion.
# ---------------------------------------------------------------------------

_E1_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(0, "01_cohort_definition", FULFILLMENT_PLANNED_ONLY),
    ProductMapping(
        1,
        _E1_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(2, None, FULFILLMENT_NOT_PRODUCED_OFFLINE),
)

_E2_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(
        0,
        _E2_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(1, None, FULFILLMENT_NOT_PRODUCED_OFFLINE),
    ProductMapping(2, "03_missingness_audit", FULFILLMENT_PLANNED_ONLY),
)

_E3_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(
        0,
        _E3_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(1, None, FULFILLMENT_NOT_PRODUCED_OFFLINE),
    ProductMapping(2, "03_ordinal_trend", FULFILLMENT_PLANNED_ONLY),
)

_M2_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(
        0,
        _M2_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(1, _M2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(2, _M2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(3, _M2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
)

_H1_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(
        0,
        _H1_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(1, _H1_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(2, _H1_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(3, "01_incident_alignment", FULFILLMENT_PLANNED_ONLY),
)

_H2_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(
        0,
        _H2_TABLE_ONE,
        FULFILLMENT_PRODUCED,
        artifact_evidence_prefix="table_step_artifact_",
    ),
    ProductMapping(1, _H2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(2, _H2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
    ProductMapping(3, _H2_PRIMARY, FULFILLMENT_PLANNED_ONLY),
)

_H3_PRODUCTS: Tuple[ProductMapping, ...] = (
    ProductMapping(0, "05_characterization", FULFILLMENT_PLANNED_ONLY),
    ProductMapping(1, None, FULFILLMENT_NOT_PRODUCED_OFFLINE),
    ProductMapping(2, None, FULFILLMENT_NOT_PRODUCED_OFFLINE),
    ProductMapping(3, "04_stability_freeze", FULFILLMENT_PLANNED_ONLY),
)


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

E1 = PreflightCase(
    task_id="e1_sepsis3_prevalence_mortality",
    title="Sepsis-3 prevalence and in-hospital mortality",
    analysis_type=_ASSOCIATION,
    question=(
        "Estimate Sepsis-3 prevalence and its association with in-hospital "
        "mortality with a transparent, reproducible cohort definition."
    ),
    database="miiv",
    primary_exposure="sepsis3",
    target_outcome="death",
    concept_descriptions={
        "sepsis3": "Sepsis-3 cohort membership flag (0/1), derived from "
        "suspected infection timing + SOFA>=2.",
        "susp_infection": "Suspected-infection flag (0/1).",
        "sofa": "SOFA score (points).",
        "lactate": "Serum lactate (mmol/L).",
        "death": "In-hospital mortality (0/1).",
    },
    deterministic_step_id=_E1_TABLE_ONE,
    primary_step_id=_E1_PRIMARY,
    _build_plan=_e1_plan,
    _build_cohort=_e1_cohort,
    guardrail_checks=_E1_CHECKS,
    product_map=_E1_PRODUCTS,
)

E2 = PreflightCase(
    task_id="e2_lactate_mortality",
    title="24h peak lactate vs in-hospital mortality",
    analysis_type=_ASSOCIATION,
    question=(
        "Quantify the descriptive association between first-24h peak lactate "
        "and in-hospital mortality."
    ),
    database="miiv",
    primary_exposure="lactate",
    target_outcome="death",
    concept_descriptions={
        "lactate": "First-24h peak serum lactate (mmol/L).",
        "death": "In-hospital mortality (0/1).",
    },
    deterministic_step_id=_E2_TABLE_ONE,
    primary_step_id=_E2_PRIMARY,
    _build_plan=_e2_plan,
    _build_cohort=_e2_cohort,
    guardrail_checks=_E2_CHECKS,
    product_map=_E2_PRODUCTS,
)

E3 = PreflightCase(
    task_id="e3_kdigo_gradient",
    title="KDIGO AKI stage gradient vs LOS and mortality",
    analysis_type=_ASSOCIATION,
    question=(
        "Characterise the dose-response gradient of first-24h KDIGO AKI stage "
        "against ICU length of stay and mortality."
    ),
    database="miiv",
    primary_exposure="kdigo",
    target_outcome="death",
    concept_descriptions={
        "kdigo": "First-24h peak KDIGO AKI stage (ordered 0-3).",
        "los_icu": "ICU length of stay (days).",
        "death": "In-hospital mortality (0/1).",
    },
    deterministic_step_id=_E3_TABLE_ONE,
    primary_step_id=_E3_PRIMARY,
    _build_plan=_e3_plan,
    _build_cohort=_e3_cohort,
    guardrail_checks=_E3_CHECKS,
    product_map=_E3_PRODUCTS,
)

M2 = PreflightCase(
    task_id="m2_mortality_prediction",
    title="First-24h vitals+labs mortality prediction with calibration",
    analysis_type="prediction_model",
    question=(
        "Build and evaluate a first-24h in-hospital mortality prediction model "
        "with patient-grouped splitting, calibration, and clinical utility."
    ),
    database="miiv",
    primary_exposure="",
    target_outcome="death",
    concept_descriptions={
        "patient_stay_id": "Patient-grouped stay identifier: patient prefix before ':s'.",
        "death": "In-hospital mortality (0/1).",
        "lact": "First-24h peak lactate (mmol/L).",
    },
    deterministic_step_id=_M2_TABLE_ONE,
    primary_step_id=_M2_PRIMARY,
    _build_plan=_m2_plan,
    _build_cohort=_m2_cohort,
    primary_code_kind="prediction",
    required_imports=("numpy", "pandas", "scipy", "statsmodels", "sklearn"),
    guardrail_checks=_M2_CHECKS,
    product_map=_M2_PRODUCTS,
)

H1 = PreflightCase(
    task_id="h1_ventilation_survival",
    title="Mechanical ventilation duration/status vs 28-day mortality",
    analysis_type="survival",
    question=(
        "Estimate aligned incident ventilation exposure and 28-day mortality "
        "using Kaplan-Meier and Cox methods with PH diagnostics."
    ),
    database="miiv",
    primary_exposure="vent_24h_any",
    target_outcome="event_28d",
    concept_descriptions={
        "vent_24h_any": "Ventilation exposure classified within the fixed 24h landmark.",
        "followup_days": "Event/censoring time from the landmark (days).",
        "event_28d": "28-day mortality event indicator (0/1).",
    },
    deterministic_step_id=_H1_TABLE_ONE,
    primary_step_id=_H1_PRIMARY,
    _build_plan=_h1_plan,
    _build_cohort=_h1_cohort,
    primary_code_kind="survival",
    required_imports=("numpy", "pandas", "scipy", "statsmodels"),
    guardrail_checks=_H1_CHECKS,
    product_map=_H1_PRODUCTS,
)

H2 = PreflightCase(
    task_id="h2_vasopressor_causal",
    title="Early vasopressor exposure vs mortality",
    analysis_type="causal_inference",
    question=(
        "Emulate the effect of early vasopressor exposure on mortality with a "
        "frozen target-trial protocol, balance, and positivity diagnostics."
    ),
    database="miiv",
    primary_exposure="vasopressor",
    target_outcome="death",
    concept_descriptions={
        "vasopressor": "Recorded early vasopressor exposure (0/1) under a source capture contract.",
        "death": "28-day mortality (0/1).",
    },
    deterministic_step_id=_H2_TABLE_ONE,
    primary_step_id=_H2_PRIMARY,
    _build_plan=_h2_plan,
    _build_cohort=_h2_cohort,
    primary_code_kind="causal",
    required_imports=("numpy", "pandas", "scipy", "statsmodels"),
    guardrail_checks=_H2_CHECKS,
    product_map=_H2_PRODUCTS,
)

H3 = PreflightCase(
    task_id="h3_trajectory_clustering",
    title="Organ-dysfunction trajectory clustering",
    analysis_type="trajectory_clustering",
    question=(
        "Cluster fixed-window organ-dysfunction trajectories, freeze cluster "
        "selection through an independent stability design, and describe outcomes."
    ),
    database="miiv",
    primary_exposure="",
    target_outcome="death",
    concept_descriptions={
        "lact_h0_6": "Lactate in fixed ICU 0-6h window (mmol/L).",
        "map_h0_6": "MAP in fixed ICU 0-6h window (mmHg).",
        "death": "In-hospital mortality (0/1).",
    },
    deterministic_step_id=_H3_TABLE_ONE,
    primary_step_id=_H3_PRIMARY,
    _build_plan=_h3_plan,
    _build_cohort=_h3_cohort,
    primary_code_kind="trajectory",
    required_imports=("numpy", "pandas", "scipy", "sklearn"),
    guardrail_checks=_H3_CHECKS,
    product_map=_H3_PRODUCTS,
)

E1E3_CASES: Dict[str, PreflightCase] = {case.task_id: case for case in (E1, E2, E3)}
COMPLEX_CASES: Dict[str, PreflightCase] = {
    case.task_id: case for case in (M2, H1, H2, H3)
}
PREFLIGHT_CASES: Dict[str, PreflightCase] = {
    **E1E3_CASES,
    **COMPLEX_CASES,
}

__all__ = [
    "GuardrailCheck",
    "ProductMapping",
    "FULFILLMENT_PRODUCED",
    "FULFILLMENT_PLANNED_ONLY",
    "FULFILLMENT_NOT_PRODUCED_OFFLINE",
    "PreflightCase",
    "E1",
    "E2",
    "E3",
    "M2",
    "H1",
    "H2",
    "H3",
    "E1E3_CASES",
    "COMPLEX_CASES",
    "PREFLIGHT_CASES",
]
