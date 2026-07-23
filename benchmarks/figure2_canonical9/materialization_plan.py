"""Deterministic MIMIC-IV materialization plan for the exact Canonical9 suite.

This repository-local module owns case-specific data coordinates.  It is not
part of the installed EasyICU package and is never imported by shared Agent
prompts.  Keeping the coordinates here makes rematerialization reproducible
without asking an LLM to select a different variable universe on every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .evaluator.paper_rubric_v3 import (
    Figure2PaperRubricManifest,
    default_figure2_paper_rubric_path,
)
from .evaluator.suite import easyicu_evaluation_protocol_suite


@dataclass(frozen=True, slots=True)
class Canonical9MaterializationSpec:
    """One task's frozen extraction coordinates before input authority."""

    task_id: str
    feature_concepts: tuple[str, ...]
    static_concepts: tuple[str, ...]
    outcome_concepts: tuple[str, ...] = ("death",)
    exposure_concept: Optional[str] = None
    operational_exposure: Optional[str] = None
    emit_trajectory: bool = False
    trajectory_concepts: tuple[str, ...] = ()
    trajectory_window: Optional[tuple[float, float]] = None
    identity_mode: str = "stay"
    positive_only_event_concepts: tuple[str, ...] = ()
    notes: Optional[str] = None


_STATIC_CORE = ("age", "sex", "los_icu", "adm", "icu_readmission")
_SOFA2_COMPONENTS = (
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
)
_VASOPRESSOR_CONCEPTS = (
    "vaso_ind",
    "norepi_rate",
    "epi_rate",
    "dopa_rate",
    "dobu_rate",
    "adh_rate",
    "phn_rate",
)


CANONICAL9_MIMIC_IV_PLAN: tuple[Canonical9MaterializationSpec, ...] = (
    Canonical9MaterializationSpec(
        task_id="e1_sepsis3_prevalence_mortality",
        feature_concepts=(
            "susp_inf",
            "sep3_sofa2",
            "sofa2",
            *_SOFA2_COMPONENTS,
            "charlson",
            "lact",
            "map",
        ),
        static_concepts=_STATIC_CORE,
        exposure_concept="sepsis3",
        operational_exposure="sep3_sofa2_max",
        positive_only_event_concepts=("sep3_sofa2",),
        notes=(
            "Use the typed sep3_sofa2 concept as the Sepsis-3 criterion and "
            "susp_inf as its suspected-infection component. Report the exact "
            "operational denominator; never substitute an ICD-only proxy."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="e2_lactate_mortality",
        feature_concepts=(
            "lact",
            "charlson",
            "sep3_sofa2",
            "hr",
            "map",
            "resp",
            "temp",
            "bun",
            "wbc",
        ),
        static_concepts=_STATIC_CORE,
        exposure_concept="lactate",
        operational_exposure="lact_max",
        notes=(
            "The operational exposure is lact_max, the maximum typed lact value "
            "within ICU hours 0-24, in mmol/L. Audit measuredness and skew and "
            "do not replace this with a whole-stay or mean lactate."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="e3_kdigo_gradient",
        feature_concepts=(
            "aki_stage",
            "aki_stage_creat",
            "aki_stage_uo",
            "aki_stage_rrt",
            "crea",
            "urine24",
            "rrt",
            "sofa_cardio",
            "sofa_cns",
            "sofa_coag",
            "sofa_liver",
            "sofa_resp",
        ),
        static_concepts=_STATIC_CORE,
        exposure_concept="kdigo",
        operational_exposure="aki_stage_max",
        notes=(
            "Use aki_stage_max over ICU hours 0-24 as the ordered KDIGO stage "
            "(0-3). Retain its ordinal interpretation and report stage-specific "
            "boundaries and denominators."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="m1_hepatobiliary_missingness",
        feature_concepts=("bili", *_SOFA2_COMPONENTS),
        static_concepts=_STATIC_CORE,
        exposure_concept="bili",
        operational_exposure="bili_max",
    ),
    Canonical9MaterializationSpec(
        task_id="m2_mortality_prediction",
        feature_concepts=(
            "hr",
            "map",
            "resp",
            "temp",
            "spo2",
            "crea",
            "bun",
            "na",
            "k",
            "bicar",
            "glu",
            "lact",
            "wbc",
            "hgb",
            "plt",
            "inr_pt",
            "bili",
            "alb",
            "ph",
            "pco2",
            "po2",
            "fio2",
        ),
        static_concepts=_STATIC_CORE,
        identity_mode="patient_grouped_stay",
        notes=(
            "For the required patient-level train/test split, derive the patient "
            "group from patient_stay_id by taking the prefix before ':s'. Never "
            "split directly on the full patient_stay_id because it is unique per "
            "ICU stay."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="m3_sepsis_subphenotype",
        feature_concepts=(
            "susp_inf",
            "hr",
            "map",
            "resp",
            "temp",
            "spo2",
            "lact",
            "ph",
            "crea",
            "bili",
            "plt",
            "wbc",
            "bun",
            "na",
            "bicar",
            "glu",
            "alb",
            "inr_pt",
        ),
        static_concepts=_STATIC_CORE,
    ),
    Canonical9MaterializationSpec(
        task_id="h1_ventilation_survival",
        feature_concepts=(
            "mech_vent",
            "vent_start",
            "vent_end",
            "vent_ind",
            "mort_28d",
            "charlson",
            "sofa2_cardio",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_renal",
            "lact",
        ),
        static_concepts=(*_STATIC_CORE, "los_hosp"),
        exposure_concept="vent_24h_any",
        operational_exposure="mech_vent_max",
        positive_only_event_concepts=("mech_vent",),
        notes=(
            "Derive vent_24h_any only from typed ventilation status/timing within "
            "ICU hours 0-24. Align exposure and follow-up at a defensible landmark "
            "and explicitly audit prevalent exposure, immortal time, and PH."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="h2_vasopressor_causal",
        feature_concepts=(
            *_VASOPRESSOR_CONCEPTS,
            "charlson",
            "map",
            "hr",
            "lact",
            "mech_vent",
            "fio2",
            "crea",
            "bili",
            "plt",
            "gcs",
            "wbc",
            "temp",
            "hgb",
        ),
        static_concepts=(*_STATIC_CORE, "los_hosp"),
        exposure_concept="vasopressor",
        operational_exposure="vaso_ind_max",
        positive_only_event_concepts=("vaso_ind",),
        emit_trajectory=True,
        trajectory_concepts=(*_VASOPRESSOR_CONCEPTS, "map", "lact"),
        trajectory_window=(0.0, 24.0),
        notes=(
            "Operationalise early vasopressor exposure as any recorded "
            "vasopressor administration in ICU hours 0-24. Absence means no "
            "recorded administration in this audited inputevents-derived source, "
            "not proof that no unobserved vasopressor was given. Report positivity "
            "and covariate balance before any bounded causal interpretation."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="h3_trajectory_clustering",
        feature_concepts=("sofa2", *_SOFA2_COMPONENTS, "lact"),
        static_concepts=("age", "sex", "los_icu"),
        emit_trajectory=True,
        trajectory_concepts=("sofa2", *_SOFA2_COMPONENTS, "lact"),
        trajectory_window=(0.0, 72.0),
        notes=(
            "Build fixed-anchor ICU-hour trajectories over hours 0-72 from the "
            "typed long table. Use a common time grid, make missingness explicit, "
            "and assess cluster-count choice and stability before interpretation."
        ),
    ),
)


def validate_canonical9_mimic_iv_plan() -> None:
    """Fail if the extraction plan drifts from the frozen suite or paper rubric."""

    suite = easyicu_evaluation_protocol_suite()
    task_ids = tuple(task.task_id for task in suite.tasks)
    planned_ids = tuple(spec.task_id for spec in CANONICAL9_MIMIC_IV_PLAN)
    if planned_ids != task_ids:
        raise ValueError("Canonical9 materialization plan order drifted from suite")
    # Input materialization is explicitly development-only. Validate the frozen
    # rubric schema and task coordinates here, but leave the executable scorer
    # tree digest to the final paper-acceptance/authority gate. Otherwise an
    # unrelated scorer implementation edit blocks rebuilding diagnostic inputs.
    rubric = Figure2PaperRubricManifest.model_validate_json(
        default_figure2_paper_rubric_path().read_bytes(),
        strict=True,
    )
    rubric_exposures = tuple(
        task.validity_binding.exposure_concept for task in rubric.tasks
    )
    planned_exposures = tuple(
        spec.exposure_concept for spec in CANONICAL9_MIMIC_IV_PLAN
    )
    if planned_exposures != rubric_exposures:
        raise ValueError(
            "Canonical9 scoring exposure concepts drifted from paper rubric"
        )
    for spec in CANONICAL9_MIMIC_IV_PLAN:
        if spec.emit_trajectory != bool(
            spec.trajectory_concepts and spec.trajectory_window
        ):
            raise ValueError(f"{spec.task_id}: trajectory declaration is incomplete")
        for label, concepts in (
            ("feature", spec.feature_concepts),
            ("static", spec.static_concepts),
            ("outcome", spec.outcome_concepts),
            ("trajectory", spec.trajectory_concepts),
        ):
            if len(concepts) != len(set(concepts)):
                raise ValueError(f"{spec.task_id}: {label} concepts repeat")
        if spec.identity_mode not in {"stay", "patient_grouped_stay"}:
            raise ValueError(f"{spec.task_id}: unsupported identity mode")
        if len(spec.positive_only_event_concepts) != len(
            set(spec.positive_only_event_concepts)
        ) or not set(spec.positive_only_event_concepts).issubset(spec.feature_concepts):
            raise ValueError(
                f"{spec.task_id}: positive-only events must be unique features"
            )


__all__ = [
    "CANONICAL9_MIMIC_IV_PLAN",
    "Canonical9MaterializationSpec",
    "validate_canonical9_mimic_iv_plan",
]
