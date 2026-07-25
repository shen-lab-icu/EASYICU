from benchmarks.figure2_canonical9.evaluator.paper_rubric_v3 import (
    Figure2PaperRubricManifest,
    default_figure2_paper_rubric_path,
)
from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)
from benchmarks.figure2_canonical9.materialization_plan import (
    CANONICAL9_MIMIC_IV_PLAN,
    validate_canonical9_mimic_iv_plan,
)


def test_materialization_plan_separates_scoring_concepts_from_sealed_columns():
    validate_canonical9_mimic_iv_plan()

    suite = easyicu_evaluation_protocol_suite()
    rubric = Figure2PaperRubricManifest.model_validate_json(
        default_figure2_paper_rubric_path().read_bytes(),
        strict=True,
    )
    assert tuple(spec.task_id for spec in CANONICAL9_MIMIC_IV_PLAN) == tuple(
        task.task_id for task in suite.tasks
    )
    assert tuple(spec.exposure_concept for spec in CANONICAL9_MIMIC_IV_PLAN) == tuple(
        task.validity_binding.exposure_concept for task in rubric.tasks
    )
    by_id = {spec.task_id: spec for spec in CANONICAL9_MIMIC_IV_PLAN}
    assert by_id["e1_sepsis3_prevalence_mortality"].operational_exposure == (
        "sep3_sofa2_max"
    )
    assert by_id["e2_lactate_mortality"].operational_exposure == "lact_max"
    assert by_id["e3_kdigo_gradient"].operational_exposure == "aki_stage_max"
    assert by_id["m1_hepatobiliary_missingness"].operational_exposure == "bili_max"
    assert by_id["h1_ventilation_survival"].operational_exposure == "mech_vent_max"
    assert by_id["h2_vasopressor_causal"].operational_exposure == "vaso_ind_max"


def test_patient_split_and_trajectory_cases_have_explicit_execution_contracts():
    by_id = {spec.task_id: spec for spec in CANONICAL9_MIMIC_IV_PLAN}

    m2 = by_id["m2_mortality_prediction"]
    assert m2.identity_mode == "patient_grouped_stay"
    assert "prefix before ':s'" in str(m2.notes)

    h2 = by_id["h2_vasopressor_causal"]
    assert h2.emit_trajectory is True
    assert h2.trajectory_window == (0.0, 24.0)
    assert "vaso_ind" in h2.trajectory_concepts
    assert h2.positive_only_event_concepts == ("vaso_ind",)

    h3 = by_id["h3_trajectory_clustering"]
    assert h3.emit_trajectory is True
    assert h3.trajectory_window == (0.0, 72.0)
    assert {"sofa2", "lact"}.issubset(h3.trajectory_concepts)
