"""Web StudyContext -> sealed fixed-window trajectory suite projection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    FIXED_WINDOW_TRAJECTORY_DEFAULTS,
    TRAJECTORY_HOST_POLICY,
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (  # noqa: E501
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    load_trajectory_scientific_runtime_authority,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.research_launch_scientific import (
    ResearchPipelineRunError,
    validate_analysis_design_for_execution,
)
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
)
from easyicu.webserver.trajectory_runtime_projection import (
    compile_web_trajectory_runtime_projection,
    trajectory_provenance_path,
    validate_trajectory_design_declaration,
)

_COORDINATES = ("sofa2_resp", "sofa2_cardio", "sofa2_renal", "lact")


def _design(**overrides):
    payload = {
        "coordinate_concepts": list(_COORDINATES),
        "descriptive_only_concepts": ["sofa2"],
        "window_start_hours": 0,
        "window_end_hours": 48,
        "grid_width_hours": 12,
        "minimum_available_windows": 2,
        "candidate_cluster_min": 2,
        "candidate_cluster_max": 4,
    }
    payload.update(overrides)
    return payload


def _study(**overrides):
    study = {
        "analysis_design": {
            "analysis_family": "trajectory_clustering",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "trajectory_design": study_context_owner.normalize_trajectory_design(
            _design()
        ),
    }
    study.update(overrides)
    return study


def _panel(tmp_path, *, n: int = 120, concepts=_COORDINATES, window=(0.0, 48.0)):
    """A long per-timepoint panel in the exact shape the sealed owner reads."""

    rng = np.random.default_rng(20260922)
    rows = []
    for stay in range(n):
        for hour in (2.0, 14.0, 26.0, 38.0):
            for concept in concepts:
                rows.append(
                    {
                        "stay_id": stay,
                        "charttime": hour,
                        "concept": concept,
                        "value_num": float(rng.integers(0, 5)),
                        "evidence_state": "direct_observed",
                        "owner_observed": 1,
                        "owner_available": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    universe = tmp_path / "web_research_universe.parquet"
    pd.DataFrame({"stay_id": range(n)}).to_parquet(universe, index=False)
    frame.to_parquet(tmp_path / "web_research_universe_trajectory.parquet", index=False)
    trajectory_provenance_path(universe).write_text(
        json.dumps(
            {
                "trajectory_concepts_materialized": [*concepts, "sofa2"],
                "available_unobserved_concepts": [],
                "unavailable_concepts": [],
                "window": list(window),
            }
        ),
        encoding="utf-8",
    )
    return universe, frame


def _compile(universe, study=None):
    return compile_web_trajectory_runtime_projection(
        study=_study() if study is None else study,
        universe_path=universe,
        scientific_configuration_sha256="a" * 64,
    )


def test_trajectory_design_compiles_the_sealed_suite_and_executes(tmp_path):
    universe, panel = _panel(tmp_path)

    projection = _compile(universe)

    assert projection is not None
    authority = load_trajectory_scientific_runtime_authority(projection.authority)
    assert authority.coordinate_concepts == _COORDINATES
    assert authority.descriptive_only_concepts == ("sofa2",)
    assert authority.window_start_hours == 0 and authority.window_end_hours == 48
    assert authority.grid_width_hours == 12
    assert authority.candidate_cluster_counts == (2, 3, 4)
    assert authority.minimum_available_windows == 2
    assert authority.protocol_content_sha256 == "a" * 64
    assert len(authority.representation_columns) == len(_COORDINATES) * 4
    assert authority.representation_columns[0] == "sofa2_resp__h0_12"
    # Host policy, not a study field: one implementation each.
    assert authority.aggregation == TRAJECTORY_HOST_POLICY["aggregation"]
    assert authority.selection_criterion == "bic"
    assert authority.candidate_fit_base_seed == 1729
    assert authority.stability_spec.n_resamples == 100
    assert authority.stability_spec.minimum_mean_stability == 0.7
    assert authority.stability_spec.decision_mode == "minimum_mean_threshold"
    # The same configuration always signs the same contract.
    again = _compile(universe)
    assert again is not None
    assert again.projection_sha256 == projection.projection_sha256

    summary = run_trajectory_scientific_representation(
        authority=projection.authority,
        runtime_projection_sha256=projection.projection_sha256,
        trajectory_path=tmp_path / "web_research_universe_trajectory.parquet",
        out_dir=tmp_path / "out",
    )
    assert summary["status"] == "ok"
    membership = pd.read_csv(tmp_path / "out" / "trajectory_membership.csv")
    assert len(membership) == panel["stay_id"].nunique()
    assert membership["included_in_clustering"].all()


def test_a_study_without_a_trajectory_declaration_is_not_a_trajectory_study(tmp_path):
    universe, _panel_frame = _panel(tmp_path)
    association = {
        "analysis_design": {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        }
    }
    assert _compile(universe, association) is None
    assert validate_trajectory_design_declaration(association) is None
    assert validate_trajectory_design_declaration({}) is None


@pytest.mark.parametrize(
    ("study", "code", "detail_key"),
    [
        (
            {
                "analysis_design": {
                    "analysis_family": "trajectory_clustering",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                }
            },
            "web_trajectory_design_required",
            "required_fields",
        ),
        (
            {
                "analysis_design": {
                    "analysis_family": "prediction_model",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                },
                "trajectory_design": study_context_owner.normalize_trajectory_design(
                    _design()
                ),
            },
            "web_trajectory_family_mismatch",
            "required_analysis_family",
        ),
        (
            {
                "analysis_design": {
                    "analysis_family": "trajectory_clustering",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                },
                "trajectory_design": study_context_owner.normalize_trajectory_design(
                    _design()
                ),
            },
            "web_trajectory_design_unsupported",
            "supported_variance_estimator",
        ),
        (
            {
                "analysis_design": {
                    "analysis_family": "trajectory_clustering",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                },
                "trajectory_design": study_context_owner.normalize_trajectory_design(
                    _design(
                        coordinate_concepts=["lact", "map"],
                        descriptive_only_concepts=[],
                    )
                ),
            },
            "web_trajectory_eligibility_coordinate_missing",
            "required_coordinate_prefix",
        ),
    ],
)
def test_declaration_fails_closed_before_any_materialization(study, code, detail_key):
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        validate_trajectory_design_declaration(study)
    assert excinfo.value.code == code
    assert detail_key in excinfo.value.details
    # The launch gate refuses the same study with the same code, so the user
    # hears about it before an export is materialized.
    with pytest.raises(ResearchPipelineRunError) as launch:
        validate_analysis_design_for_execution(study)
    assert launch.value.code == code


def test_projection_requires_the_materialized_panel(tmp_path):
    universe, _frame = _panel(tmp_path)
    trajectory_provenance_path(universe).unlink()
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(universe)
    assert excinfo.value.code == "web_trajectory_longitudinal_panel_missing"
    assert excinfo.value.details["expected_artifact"].endswith(
        "_trajectory_provenance.json"
    )


def test_projection_refuses_a_window_outside_the_materialization(tmp_path):
    universe, _frame = _panel(tmp_path, window=(0.0, 24.0))
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(universe)
    assert excinfo.value.code == "web_trajectory_window_outside_materialization"
    assert excinfo.value.details["materialized_window_hours"] == [0.0, 24.0]


def test_projection_refuses_a_coordinate_the_panel_never_observed(tmp_path):
    universe, _frame = _panel(
        tmp_path, concepts=("sofa2_resp", "sofa2_cardio", "sofa2_renal")
    )
    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(universe)
    assert excinfo.value.code == "web_trajectory_concepts_unavailable"
    assert excinfo.value.details["missing_concepts"] == ["lact"]


def test_the_study_owner_validates_the_design_and_keeps_old_digests(tmp_path):
    normalized = study_context_owner.normalize_trajectory_design(_design())
    assert normalized["candidate_cluster_max"] == 4
    # Defaults are filled in, never guessed away.
    minimal = study_context_owner.normalize_trajectory_design(
        {"coordinate_concepts": ["sofa2_resp", "lact"]}
    )
    for field, default in FIXED_WINDOW_TRAJECTORY_DEFAULTS.items():
        assert minimal[field] == default
    assert study_context_owner.normalize_trajectory_design(None) == {}

    base = {"question": "q", "outcome": "mort_28d"}
    absent = study_context_owner.scientific_configuration_sha256(dict(base))
    empty = study_context_owner.scientific_configuration_sha256(
        {**base, "trajectory_design": {}}
    )
    declared = study_context_owner.scientific_configuration_sha256(
        {**base, "trajectory_design": normalized}
    )
    assert absent == empty
    assert declared != absent

    for payload, code in (
        ({"coordinate_concepts": ["sofa2_resp"]}, "study_trajectory_concepts_insufficient"),
        (
            {"coordinate_concepts": ["sofa2_resp", "lact"], "window_end_hours": 50},
            "study_trajectory_grid_invalid",
        ),
        (
            {
                "coordinate_concepts": ["sofa2_resp", "lact"],
                "minimum_available_windows": 99,
            },
            "study_trajectory_minimum_windows_invalid",
        ),
        (
            {
                "coordinate_concepts": ["sofa2_resp", "lact"],
                "descriptive_only_concepts": ["lact"],
            },
            "study_trajectory_descriptive_concept_is_a_coordinate",
        ),
        (
            {
                "coordinate_concepts": ["sofa2_resp", "lact"],
                "candidate_cluster_max": 2,
            },
            "study_trajectory_candidate_grid_invalid",
        ),
        (
            {
                "coordinate_concepts": ["sofa2_resp", "lact"],
                "minimum_cluster_fraction": 0.4,
            },
            "study_trajectory_minimum_cluster_fraction_invalid",
        ),
        ({"coordinate_concepts": ["SOFA2/resp", "lact"]}, "study_trajectory_concept_invalid"),
        ({"coordinate_concepts": ["sofa2_resp", "lact"], "k": 3}, "study_trajectory_design_unknown_fields"),
    ):
        with pytest.raises(study_context_owner.StudyContextError) as excinfo:
            study_context_owner.normalize_trajectory_design(payload)
        assert excinfo.value.detail["error"] == code


def test_the_sealed_body_is_built_only_from_the_design_and_host_policy():
    design = load_trajectory_design(
        study_context_owner.normalize_trajectory_design(_design())
    )
    assert design is not None
    body = sealed_trajectory_authority_body(
        design, protocol_content_sha256="b" * 64
    )
    assert "execution_contract_sha256" not in body
    assert body["representation_columns"] == list(design.representation_columns)
    assert body["candidate_cluster_counts"] == [2, 3, 4]
    assert body["upper_boundary_action"] == (
        "fail_closed_if_selected_at_upper_boundary"
    )
    assert body["evidence_state_policy"]["unavailable"] == "exclude"


def test_an_admitted_trajectory_study_always_gets_its_long_panel_emitted():
    """The projection's support gate and the panel emission must not drift.

    The signed representation owner reads the long per-timepoint panel, and the
    materializer refuses to emit one under a patient-grouped materialization.
    A design this owner admits must therefore be one the acquisition step
    actually emits a panel for, or the run would fail late with a missing
    artifact instead of early with a named design blocker.
    """

    from easyicu.webserver.research_launch_scientific import (
        _analysis_requires_longitudinal_trajectory,
        _patient_grouping_for_analysis_design,
    )

    study = _study()
    assert validate_trajectory_design_declaration(study) is not None
    validated = validate_analysis_design_for_execution(study)
    assert _patient_grouping_for_analysis_design(study) is None
    assert _analysis_requires_longitudinal_trajectory(
        study, validated_design=validated
    )


def test_a_stored_project_stays_readable_after_a_rule_tightens(tmp_path, monkeypatch):
    """Reading a saved project must not fail on a now-refused design."""

    store = tmp_path / "study_contexts.json"
    monkeypatch.setattr(study_context_owner, "_CONFIG_PATH", store)
    contradictory = {
        # A 50-hour window over a 12-hour grid: refused on write today.
        "coordinate_concepts": ["sofa2_resp", "lact"],
        "window_start_hours": 0,
        "window_end_hours": 50,
        "grid_width_hours": 12,
        "minimum_available_windows": 2,
        "candidate_cluster_min": 2,
        "candidate_cluster_max": 6,
        "stability_resamples": 100,
        "stability_sample_fraction": 0.8,
        "minimum_mean_stability": 0.7,
        "minimum_cluster_fraction": 0.05,
    }
    with pytest.raises(study_context_owner.StudyContextError):
        study_context_owner.normalize_trajectory_design(contradictory)
    store.write_text(
        json.dumps(
            {
                "contexts": [
                    {"id": "s1", "revision": 1, "trajectory_design": contradictory}
                ]
            }
        ),
        encoding="utf-8",
    )
    stored = study_context_owner.get_context("s1")
    assert stored is not None
    assert stored["trajectory_design"]["window_end_hours"] == 50


def test_copilot_can_save_the_trajectory_design_through_the_study_owner():
    """The conversation, not a separate form, is where this design is set."""

    from easyicu.webserver.pi_copilot.study_context_update import (
        _NESTED_STUDY_PATCH_FIELDS,
        _STUDY_SETUP_FIELDS,
    )
    from easyicu.webserver.pi_copilot.tool_catalog import TOOL_ARGUMENTS

    assert "trajectory_design" in _STUDY_SETUP_FIELDS
    assert "trajectory_design" in _NESTED_STUDY_PATCH_FIELDS
    declared = TOOL_ARGUMENTS["easyicu_update_study_context"]
    assert "trajectory_design" in declared.allowed
