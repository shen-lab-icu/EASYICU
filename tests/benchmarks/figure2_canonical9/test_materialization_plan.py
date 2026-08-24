import hashlib
from types import SimpleNamespace

import pytest

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
from tools.materialize_canonical9_miiv import (
    _build_jsonl_row,
    _build_development_binding_receipt,
    _select_materialization_specs,
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
    e1 = by_id["e1_sepsis3_prevalence_mortality"]
    assert e1.positive_only_event_concepts == ("susp_inf", "sep3_sofa2")
    assert "e1_scientific_closure" in str(e1.task_protocol_version)
    e1_protocol = " ".join(
        [
            *e1.additional_expected_outputs,
            *e1.additional_semantic_guardrails,
        ]
    )
    for required in (
        "stay count",
        "missing death_time",
        "negative event times",
        "24-hour landmark",
        "non-readmission ICU stays",
        "standardized mean differences",
        "flexible age and Charlson",
        "Sepsis-3 absent/present",
    ):
        assert required in e1_protocol
    assert by_id["e2_lactate_mortality"].operational_exposure == "lact_max"
    e3 = by_id["e3_kdigo_gradient"]
    assert e3.operational_exposure == "aki_stage_max"
    assert e3.task_protocol_version == "e3_kdigo_gradient/20260824-v1"
    m1 = by_id["m1_hepatobiliary_missingness"]
    assert m1.operational_exposure == "bili_max"
    assert m1.task_protocol_version == "m1_hepatobiliary_missingness/20260824-v1"
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
    assert h2.positive_only_event_concepts == ()
    assert "h2_vasopressor_causal/20260810-v4" == h2.task_protocol_version
    assert h2.additional_semantic_guardrails == ()

    h3 = by_id["h3_trajectory_clustering"]
    assert h3.emit_trajectory is True
    assert h3.trajectory_window == (0.0, 72.0)
    assert "sofa2" not in h3.trajectory_concepts
    assert {"sofa2_resp", "lact"}.issubset(h3.trajectory_concepts)
    assert "sofa2" not in h3.candidate_model_concepts
    assert h3.descriptive_only_concepts == ("sofa2",)
    assert h3.task_protocol_version == "h3_trajectory_clustering/20260810-v4"
    assert h3.additional_semantic_guardrails == ()


def test_e1_materialized_item_receives_only_its_case_protocol_overlay(tmp_path):
    tasks = {
        task.task_id: task for task in easyicu_evaluation_protocol_suite().tasks
    }
    specs = {spec.task_id: spec for spec in CANONICAL9_MIMIC_IV_PLAN}
    reference = SimpleNamespace(
        file="cohort_authority.json",
        to_dict=lambda: {"file": "cohort_authority.json"},
    )
    verified = SimpleNamespace(reference=reference)

    e1_row = _build_jsonl_row(
        task=tasks["e1_sepsis3_prevalence_mortality"],
        spec=specs["e1_sepsis3_prevalence_mortality"],
        case_dir=tmp_path,
        cohort_path=tmp_path / "e1.parquet",
        cohort_verified=verified,
        trajectory_path=None,
        trajectory_verified=None,
    )
    e2_row = _build_jsonl_row(
        task=tasks["e2_lactate_mortality"],
        spec=specs["e2_lactate_mortality"],
        case_dir=tmp_path,
        cohort_path=tmp_path / "e2.parquet",
        cohort_verified=verified,
        trajectory_path=None,
        trajectory_verified=None,
    )

    assert "e1_scientific_closure" in e1_row["protocol_version"]
    assert e1_row["scientific_acceptance_contract"]["task_id"] == (
        "e1_sepsis3_prevalence_mortality"
    )
    assert e1_row["scientific_acceptance_contract"]["sensitivity_product"] == (
        "table:e1_scientific_sensitivity"
    )
    assert any("24-hour landmark" in item for item in e1_row["semantic_guardrails"])
    assert any(
        "table:e1_scientific_sensitivity" in item
        for item in e1_row["expected_outputs"]
    )
    assert any("display_labels" in item for item in e1_row["semantic_guardrails"])
    assert any(
        "functional-form sensitivity" in item for item in e1_row["expected_outputs"]
    )
    assert e1_row["scientific_acceptance_contract"][
        "primary_cohort_selection_mode"
    ] == "all_input_rows"
    assert any(
        "cohort.selection_mode to all_input_rows" in item
        for item in e1_row["semantic_guardrails"]
    )
    assert e2_row["protocol_version"] == "e2_lactate_mortality/20260810-v3"
    assert e2_row["case_scientific_protocol"]["task_id"] == "e2_lactate_mortality"
    assert len(e2_row["case_scientific_protocol_sha256"]) == 64
    assert e2_row["runtime_scientific_projection_sha256"]
    assert e2_row["expected_outputs"] == [
        "table:e2_landmark_rcs_curve",
        "table:e2_landmark_rcs_contrasts",
        "table:e2_linear_sensitivity",
        "log:e2_scientific_runtime_receipt",
    ]
    assert any("descriptive/prognostic" in item for item in e2_row["semantic_guardrails"])


def test_materializer_can_select_one_development_case_without_reordering():
    selected = _select_materialization_specs(
        ["h3_trajectory_clustering", "e1_sepsis3_prevalence_mortality"]
    )
    assert [spec.task_id for spec in selected] == [
        "e1_sepsis3_prevalence_mortality",
        "h3_trajectory_clustering",
    ]
    assert len(_select_materialization_specs([])) == 9


def test_materializer_rejects_unknown_or_duplicate_case_selection():
    with pytest.raises(ValueError, match="unknown Canonical9 task"):
        _select_materialization_specs(["not_a_task"])
    with pytest.raises(ValueError, match="must be unique"):
        _select_materialization_specs(
            [
                "e1_sepsis3_prevalence_mortality",
                "e1_sepsis3_prevalence_mortality",
            ]
        )


def test_materializer_builds_launcher_ready_nonpaper_binding(tmp_path):
    jsonl_path = (tmp_path / "canonical9_miiv.jsonl").resolve()
    jsonl_raw = b'{"key":"e1_sepsis3_prevalence_mortality"}\n'

    receipt = _build_development_binding_receipt(
        jsonl_path=jsonl_path,
        jsonl_raw=jsonl_raw,
    )

    assert receipt == {
        "schema_version": "easyicu.canonical9_development_binding_receipt/1",
        "paper_authority": False,
        "output_jsonl": str(jsonl_path),
        "output_sha256": hashlib.sha256(jsonl_raw).hexdigest(),
    }
