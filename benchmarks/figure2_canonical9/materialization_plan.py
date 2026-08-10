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
from .case_scientific_protocol import load_default_case_protocol
from .e1_scientific_acceptance import (
    display_label_instruction,
    measurement_products_instruction,
    sensitivity_output_instruction,
)


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
    #: Width (hours) of the uniform grid the cohort's ``<family>_h<start>_<end>``
    #: columns are summarized onto, and how each window is reduced.  ``None``
    #: emits no such columns, which is every task but the trajectory one.
    #:
    #: This is a CASE decision and belongs here rather than in the engine: the
    #: engine's ``FixedWindowGrid`` chooses no family, width, horizon or
    #: aggregate, and its parser only requires that the grid be uniform.
    trajectory_panel_width_hours: Optional[float] = None
    trajectory_panel_aggregate: str = "max"
    identity_mode: str = "stay"
    positive_only_event_concepts: tuple[str, ...] = ()
    additional_expected_outputs: tuple[str, ...] = ()
    additional_semantic_guardrails: tuple[str, ...] = ()
    task_protocol_version: Optional[str] = None
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
        positive_only_event_concepts=("susp_inf", "sep3_sofa2"),
        additional_expected_outputs=(
            "missingness and event-timing audit",
            (
                "adjusted association with timing, repeated-stay, and "
                "functional-form sensitivity table and figure"
            ),
            sensitivity_output_instruction(),
            measurement_products_instruction(),
        ),
        additional_semantic_guardrails=(
            (
                "Use ICU stays as the analysis unit. Do not call a stay count a "
                "patient count when no patient identifier is available."
            ),
            (
                "Treat an absent susp_inf row as no recorded suspected-infection "
                "event, and a missing death_time for a survivor as not applicable "
                "rather than ordinary measurement missingness."
            ),
            (
                "Audit death_time against ICU admission, report and exclude "
                "negative event times from timing-based analyses, and do not hide "
                "deaths inside the first-24-hour exposure-classification window."
            ),
            (
                "Keep full-cohort prevalence and absolute mortality descriptive. "
                "For the adjusted association, report a prespecified 24-hour "
                "landmark sensitivity among stays alive at the landmark and label "
                "the estimand as observational rather than causal."
            ),
            (
                "Because patient identity is unavailable, report a sensitivity "
                "restricted to non-readmission ICU stays instead of claiming "
                "patient-clustered inference."
            ),
            (
                "Report standardized mean differences in Table 1 and include a "
                "sensitivity allowing flexible age and Charlson functional form; "
                "do not rely on large-sample P values or linearity alone."
            ),
            (
                "Use clinical display labels such as Sepsis-3 absent/present, "
                "never Category 0/1."
            ),
            (
                "Set AnalysisPlan.cohort.selection_mode to all_input_rows with "
                "empty inclusion and exclusion predicates for the primary E1 "
                "population; completeness, timing, and readmission restrictions "
                "belong only in explicit sensitivity estimands."
            ),
            display_label_instruction(),
        ),
        task_protocol_version=(
            "easyicu_evaluation_protocol_suite/v2+"
            "e1_scientific_closure/20260728-v1"
        ),
        notes=(
            "Use the typed sep3_sofa2 concept as the Sepsis-3 criterion and "
            "susp_inf as its positive-only suspected-infection event component. "
            "Report the exact operational denominator; never substitute an "
            "ICD-only proxy. The primary prevalence denominator remains all "
            "eligible ICU stays; timing and repeated-stay restrictions are "
            "explicit sensitivity estimands, not silent cohort replacements."
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
        additional_expected_outputs=(
            "full eligible-cohort lactate measured/unmeasured audit",
            "early death/discharge exposure-opportunity audit",
            "adjusted measured-subset association with nonlinearity assessment",
        ),
        additional_semantic_guardrails=(
            (
                "The primary estimand is the adjusted descriptive association "
                "between first-24-hour peak lactate and in-hospital death among "
                "eligible ICU stays with at least one valid 0-24-hour lactate; "
                "never describe it as causal."
            ),
            (
                "Retain the full eligible cohort for measured/unmeasured counts, "
                "fractions, and standardized group differences; never zero-code "
                "unmeasured lactate."
            ),
            (
                "Audit death before 24 hours, discharge before 24 hours, time "
                "under observation, and lactate measurement count/timing so "
                "exposure opportunity is visible."
            ),
            (
                "Use age, sex, and Charlson as the prespecified primary adjustment "
                "set; report median/IQR and compare linear with prespecified "
                "nonlinear lactate modelling."
            ),
        ),
        task_protocol_version="e2_lactate_mortality/20260809-v2",
        notes=(
            "The operational exposure is lact_max, the maximum typed lact value "
            "within ICU hours 0-24, in mmol/L. Audit measuredness and skew and "
            "do not replace this with a whole-stay or mean lactate. The exact "
            "case protocol is bound in case_scientific_protocol."
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
        emit_trajectory=True,
        trajectory_concepts=(*_VASOPRESSOR_CONCEPTS, "map", "lact"),
        trajectory_window=(0.0, 24.0),
        additional_expected_outputs=(
            "source-specific medication capture and negative-evidence audit",
            "content-bound target-trial protocol table",
            "structured H2_VERIFIED_NON_USE_UNAVAILABLE feasibility result",
        ),
        additional_semantic_guardrails=(
            (
                "Do not convert an absent vaso_ind source row to zero for this "
                "causal task: absence means no recorded administration, not "
                "verified non-use."
            ),
            (
                "The current MIMIC-IV inputevents-derived capture contract sets "
                "verified_non_use_available=false and causal_contrast_authorized="
                "false; report H2_VERIFIED_NON_USE_UNAVAILABLE and do not build a "
                "binary control arm or fit PSM/IPTW."
            ),
            (
                "Preserve the frozen target-trial coordinates for future use only "
                "if a separately reviewed source contract proves verified non-use; "
                "do not infer that authority from covariate balance or positivity."
            ),
        ),
        task_protocol_version="h2_vasopressor_causal/20260809-v3",
        notes=(
            "Operationalise early vasopressor exposure as any recorded "
            "vasopressor administration in ICU hours 0-24. Absence means no "
            "recorded administration in this audited inputevents-derived source, "
            "not proof that no unobserved vasopressor was given. The current "
            "source contract therefore fails closed before a control arm, "
            "positivity analysis, or effect estimate is constructed."
        ),
    ),
    Canonical9MaterializationSpec(
        task_id="h3_trajectory_clustering",
        feature_concepts=("sofa2", *_SOFA2_COMPONENTS, "lact"),
        static_concepts=("age", "sex", "los_icu"),
        emit_trajectory=True,
        trajectory_concepts=("sofa2", *_SOFA2_COMPONENTS, "lact"),
        trajectory_window=(0.0, 72.0),
        # THE COMMON TIME GRID THE NOTE BELOW ASKS FOR, MADE EXECUTABLE.
        #
        # 12 h over 0-72 gives six points per family. Uniform, as the note
        # requires. Six is enough to separate the shapes a phenotyping study is
        # for -- rising, falling, flat, late deterioration -- while staying a
        # sub-daily resolution the data actually supports: MEASURED on the
        # sealed long table, sofa2 total is present in 100.0 / 97.7 / 92.8 /
        # 87.7 / 82.1 / 76.7 % of stays across the six windows, and 97.7 % of
        # stays have at least two. The decline is discharge and death, not
        # measurement failure -- exactly the length-biased sampling this task's
        # guardrails name, and the reason missing windows are left missing.
        #
        # Change this line to change the study's time resolution; nothing in the
        # engine hard-codes it.
        trajectory_panel_width_hours=12.0,
        trajectory_panel_aggregate="max",
        additional_expected_outputs=(
            "frozen candidate-k BIC selection ledger",
            "100-resample adjusted-Rand stability audit",
            "formal no-stable-phenotype result when the prespecified gate fails",
        ),
        additional_semantic_guardrails=(
            (
                "Use the H3 v2 frozen representation: ICU hours 0-72, 12-hour "
                "max grid, SOFA-2 total plus six components and lactate, with "
                "observed-data missingness and no zero or LOCF imputation."
            ),
            (
                "Fit observed-data diagonal Gaussian mixtures for candidate k "
                "2-6, select the minimum-BIC k without outcomes, and do not try "
                "another k if the selected solution fails cluster-size or "
                "stability gates."
            ),
            (
                "Run exactly 100 80%-subsample refits from base seed 1729, require "
                "all 100 and mean adjusted Rand index at least 0.70; otherwise "
                "report no stable phenotype solution without post-hoc rescue."
            ),
            (
                "Outcome comparisons are descriptive only after assignments are "
                "frozen; a MIMIC-IV solution is not transportable without external "
                "reproducibility using the same protocol."
            ),
        ),
        task_protocol_version="h3_trajectory_clustering/20260809-v2",
        notes=(
            "Build fixed-anchor ICU-hour trajectories over hours 0-72 from the "
            "typed long table. Use a common time grid, make missingness explicit, "
            "and assess cluster-count choice and stability before interpretation. "
            "The terminal v1 result cannot be reseeded or relaxed; this separately "
            "versioned v2 may formally conclude that no stable solution exists."
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
    for task_id in (
        "e2_lactate_mortality",
        "h2_vasopressor_causal",
        "h3_trajectory_clustering",
    ):
        protocol = load_default_case_protocol(task_id)
        spec = next(item for item in CANONICAL9_MIMIC_IV_PLAN if item.task_id == task_id)
        if spec.task_protocol_version != protocol.protocol_version:
            raise ValueError(f"{task_id}: case protocol version drifted")


__all__ = [
    "CANONICAL9_MIMIC_IV_PLAN",
    "Canonical9MaterializationSpec",
    "validate_canonical9_mimic_iv_plan",
]
