"""E1-E3 typed ``AnalysisPlan`` fixtures + minimal synthetic cohorts.

Each fixture is derived **live** from the corresponding formal task protocol in
:func:`benchmarks.figure2_canonical9.evaluator.suite.easyicu_evaluation_protocol_suite`
(its ``expected_outputs`` and ``semantic_guardrails`` are read from the suite, so
they cannot silently drift) and is **diagnostic-only** (no paper authority; see
the package docstring).

Unlike the batch-1 fixtures, E1/E2/E3 are **genuinely distinct**, not one shared
two-step logistic skeleton:

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
    """One diagnostic-only E-series preflight case bound to a suite task."""

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
    *, step_id: str, exposure: str, outcome: str, adjust: List[str], intent: str
) -> AnalysisStep:
    """Agent-owned (LLM-coded) adjusted association step."""

    inputs = [exposure, *adjust, outcome]
    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="primary",
        intent=intent,
        inputs=inputs,
        expected_outputs=["table:primary_association"],
        method="logistic_regression",
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
                expected_outputs=["table:cohort_definition"],
                intent=(
                    "State the explicit Sepsis-3 cohort denominator and "
                    "inclusion/exclusion: membership is derived from suspected "
                    "infection timing plus SOFA>=2 (never an ICD-code proxy); "
                    "diagnosis codes are used for membership only, not event "
                    "timing."
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
        ],
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
                expected_outputs=["table:missingness_audit"],
                intent=(
                    "Audit lactate measurement missingness and the within-window "
                    "aggregation before interpreting the association."
                ),
            ),
            _primary_association_step(
                step_id=_E2_PRIMARY,
                exposure="lactate",
                outcome="death",
                adjust=["age"],
                intent="Adjusted association of first-24h peak lactate with mortality.",
            ),
        ],
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
                expected_outputs=["table:stage_stratified_outcomes"],
                intent=(
                    "Stratify ICU length of stay and mortality by explicit KDIGO "
                    "stage boundaries 0, 1, 2, 3 (report each stage separately)."
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
        ],
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

E1E3_CASES: Dict[str, PreflightCase] = {case.task_id: case for case in (E1, E2, E3)}

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
    "E1E3_CASES",
]
