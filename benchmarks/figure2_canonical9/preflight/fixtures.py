"""E1-E3 typed ``AnalysisPlan`` fixtures + minimal synthetic cohorts.

Each fixture is derived from the corresponding formal task protocol in
:func:`benchmarks.figure2_canonical9.evaluator.suite.easyicu_evaluation_protocol_suite`
and is **diagnostic-only** (no paper authority; see the package docstring).

The plans are deliberately minimal: each carries exactly the typed Table 1
contract the deterministic grouped-Table-1 executor owns plus one agent-owned
primary association step.  The pipeline's own plan shaper adds the family figure
and audit-panel steps; the preflight verifies the *routing* of those steps, not
publication-grade science (that is the Provider boundary).

Cohorts are tiny in-memory synthetic frames (no patient data).  Column names
match each plan's declared inputs so the deterministic Table 1 executor and the
agent-owned primary both run on real data offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    TableOneSpec,
    TableOneVariableSpec,
)

# Catalog analysis-type key (see easyicu.research_agent.planning.analysis_types).
# All three E-series questions are adjusted/ordinal associations of an exposure
# with in-hospital mortality; the paper-suite ``kind`` (descriptive_association,
# ordinal_dose_response, sepsis_onset) is a *different* vocabulary and is NOT a
# planner analysis_type key.
_ASSOCIATION = "association_study"

DETERMINISTIC_STEP_ID = "01_table_one"
PRIMARY_STEP_ID = "04_primary_association"


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
    _build_plan: Callable[[], AnalysisPlan]
    _build_cohort: Callable[[int], pd.DataFrame]
    deterministic_step_id: str = DETERMINISTIC_STEP_ID
    primary_step_id: str = PRIMARY_STEP_ID
    # A minimal synthetic dev fixture intentionally does not satisfy the full
    # article display contract, so the honest fail-closed verdict is
    # ``diagnostic_only`` (execution never completes the required suite).
    expected_tristate: str = "diagnostic_only"
    diagnostic_only: bool = True

    def build_plan(self) -> AnalysisPlan:
        return self._build_plan()

    def build_cohort(self, n: int = 80) -> pd.DataFrame:
        return self._build_cohort(n)


def _table_one_step(
    *,
    group_by: str,
    group_levels: List[object],
    variables: List[TableOneVariableSpec],
    intent: str,
) -> AnalysisStep:
    """Typed grouped Table 1 owned by the deterministic executor.

    ``expected_outputs == ['table:table_one']`` exactly and every summarised
    variable is an explicit step input, which is the exact contract
    ``table_one_executor_owns_step`` requires.
    """

    inputs = [group_by, *(v.name for v in variables)]
    return AnalysisStep(
        step_id=DETERMINISTIC_STEP_ID,
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
    *, exposure: str, outcome: str, adjust: List[str], intent: str
) -> AnalysisStep:
    """Agent-owned (LLM-coded) adjusted association step."""

    inputs = [exposure, *adjust, outcome]
    return AnalysisStep(
        step_id=PRIMARY_STEP_ID,
        planned_analysis_role="primary",
        intent=intent,
        inputs=inputs,
        expected_outputs=["table:primary_association"],
        method="logistic_regression",
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


def _e1_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E1.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _table_one_step(
                group_by="sepsis3",
                group_levels=[0, 1],
                variables=[_continuous("age"), _continuous("lactate")],
                intent="Baseline Table 1 grouped by Sepsis-3 cohort membership.",
            ),
            _primary_association_step(
                exposure="sepsis3",
                outcome="death",
                adjust=["age"],
                intent="Adjusted association of Sepsis-3 with in-hospital mortality.",
            ),
        ],
        rationale=(
            "E1 diagnostic-only preflight fixture: typed Table 1 grouped by the "
            "explicit Sepsis-3 cohort flag plus an agent-owned mortality "
            "association. Sepsis-3 is provided as a pre-derived synthetic flag; "
            "the preflight verifies orchestration, not the derived-concept "
            "definition (that is the Provider/real-data boundary)."
        ),
    )


def _e1_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(101)
    age = rng.randint(45, 88, n).astype(float)
    sepsis3 = rng.binomial(1, 0.4, n)
    lactate = (1.2 + 1.6 * sepsis3 + rng.gamma(2.0, 0.7, n)).round(2)
    logit = -2.2 + 1.1 * sepsis3 + 0.02 * (age - 65)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "stay_id": range(1, n + 1),
            "subject_id": range(1, n + 1),
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "lactate": lactate,
            "sepsis3": sepsis3,
            "death": death,
        }
    )


# ---------------------------------------------------------------------------
# E2 — 24h peak lactate vs in-hospital mortality (descriptive_association)
# ---------------------------------------------------------------------------


def _e2_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E2.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _table_one_step(
                group_by="death",
                group_levels=[0, 1],
                variables=[_continuous("lactate"), _continuous("age")],
                intent="Baseline Table 1 grouped by in-hospital survival status.",
            ),
            _primary_association_step(
                exposure="lactate",
                outcome="death",
                adjust=["age"],
                intent="Adjusted association of peak lactate with mortality.",
            ),
        ],
        rationale=(
            "E2 diagnostic-only preflight fixture: typed Table 1 grouped by "
            "survival plus an agent-owned lactate-mortality association."
        ),
    )


def _e2_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(202)
    age = rng.randint(45, 88, n).astype(float)
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


# ---------------------------------------------------------------------------
# E3 — KDIGO AKI stage gradient vs LOS and mortality (ordinal_dose_response)
# ---------------------------------------------------------------------------


def _e3_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=E3.question,
        analysis_type=_ASSOCIATION,
        steps=[
            _table_one_step(
                group_by="death",
                group_levels=[0, 1],
                variables=[_ordinal("kdigo"), _continuous("age")],
                intent="Baseline Table 1 grouped by survival, KDIGO as ordinal.",
            ),
            _primary_association_step(
                exposure="kdigo",
                outcome="death",
                adjust=["age"],
                intent="Adjusted ordinal KDIGO-stage gradient vs mortality.",
            ),
        ],
        rationale=(
            "E3 diagnostic-only preflight fixture: typed Table 1 summarising "
            "KDIGO as an ordered stage (median/IQR + rank test, never mean/SD) "
            "plus an agent-owned ordinal dose-response association."
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
        "sepsis3": "Sepsis-3 cohort membership flag (0/1), pre-derived.",
        "lactate": "Serum lactate (mmol/L).",
        "death": "In-hospital mortality (0/1).",
    },
    _build_plan=_e1_plan,
    _build_cohort=_e1_cohort,
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
        "lactate": "Serum lactate (mmol/L).",
        "death": "In-hospital mortality (0/1).",
    },
    _build_plan=_e2_plan,
    _build_cohort=_e2_cohort,
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
    _build_plan=_e3_plan,
    _build_cohort=_e3_cohort,
)

E1E3_CASES: Dict[str, PreflightCase] = {case.task_id: case for case in (E1, E2, E3)}

__all__ = [
    "PreflightCase",
    "E1",
    "E2",
    "E3",
    "E1E3_CASES",
    "DETERMINISTIC_STEP_ID",
    "PRIMARY_STEP_ID",
]
