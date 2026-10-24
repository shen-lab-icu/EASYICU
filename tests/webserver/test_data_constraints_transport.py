"""Complete study populations survive the bounded constraints transport."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import agent_pipeline_runs, primary_cohort
from easyicu.webserver.research_launch_scientific import (
    _require_export_holds_study_cohort,
)


def _ordinary_constraints() -> dict:
    """Synthetic native-export receipt with the full ordinary population."""

    return {
        "cohort": {
            "preset": "all_icu",
            "age_min": 18,
            "age_max": 100,
            "min_icu_los_hours": 24,
            "exclude_readmissions": False,
            "review": (
                "Adults admitted to an ICU, retaining the first ICU stay in each "
                "hospitalization. Exclude step-down unit stays and stays whose "
                "prior-unit history is incomplete. Read age from the demographic "
                "record at hospital admission. Apply the same rules to every "
                "selected module before planning; the outcome horizon remains "
                "separate from the feature materialization window."
            ),
            "exclusion_statement": (
                "Later ICU stays in the same hospitalization, step-down units, "
                "incomplete prior-unit records, age below 18 years, and ICU stays "
                "shorter than 24 hours are outside the declared population."
            ),
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
            "extraction_completed": True,
            "repeated_icu_stays_retained": True,
        },
        "materialization_window": {
            "role": "outer_observation_window",
            "hours": 24,
            "anchor": "ICU admission",
        },
        "event_time_semantics": {"death_time": "recorded_time"},
        "source_selection": {
            "basis": "export_contract",
            "host_applied": [
                "Keep only the first ICU stay per hospitalization, using the "
                "verified hospital and ICU admission coordinates before planning.",
                "Exclude step-down unit stays using the bound source's unit "
                "classification; unknown classifications cannot prove eligibility.",
                "Require a complete prior-unit history for the hospitalization "
                "before determining which ICU admission is the first eligible stay.",
                "Use age at hospital admission from the source demographic "
                "record; missing age does not establish adult eligibility.",
            ],
            "export_report": {
                "mode": "all_icu",
                "count_unit": "icu_stay",
                "source_total": 12,
                "selected": 4,
                "defaults_locked": True,
                "demographic_steps": [
                    {
                        "criterion": "age",
                        "parameters": {"age_min": 18, "age_max": 100},
                        "n_before": 12,
                        "n_excluded": 2,
                        "n_remaining": 10,
                        "n_excluded_missing": 1,
                    },
                    {
                        "criterion": "minimum_icu_duration",
                        "parameters": {"min_icu_los_hours": 24},
                        "n_before": 10,
                        "n_excluded": 2,
                        "n_remaining": 8,
                        "n_excluded_missing": 0,
                    },
                ],
                "host_steps": [
                    {
                        "criterion": "first_icu_stay_per_hospitalization",
                        "n_before": 8,
                        "n_excluded": 1,
                        "n_remaining": 7,
                    },
                    {
                        "criterion": "exclude_step_down_units",
                        "n_before": 7,
                        "n_excluded": 1,
                        "n_remaining": 6,
                    },
                    {
                        "criterion": "complete_prior_unit_history",
                        "n_before": 6,
                        "n_excluded": 2,
                        "n_remaining": 4,
                    },
                ],
                "module_defaults": {
                    "demographics": {"age": "hospital_admission", "sex": "static"},
                    "vitals": {
                        "window_hours": 24,
                        "aggregation": "mean",
                        "missingness_policy": "retain stays and report missing measurements",
                    },
                    "outcome": {"death": "hospital_discharge", "los_icu": "days"},
                },
                "selection_receipt": (
                    "The native export locks the selected module defaults and "
                    "records the population selection before feature extraction. "
                    "Counts refer to ICU stays in this synthetic source and are "
                    "reported in their execution order. Hospitalization selection, "
                    "unit eligibility, prior-unit completeness and demographic age "
                    "use source coordinates verified by the host. Repeated stays "
                    "across hospitalizations remain grouped by patient for inference."
                ),
            },
        },
    }


def test_an_ordinary_population_is_transported_without_elision() -> None:
    constraints = _ordinary_constraints()
    expected = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert 3_100 <= len(expected) <= 3_300

    assert agent_pipeline_runs._compile_data_constraints(constraints) == expected


@pytest.mark.parametrize("size", [4_096, 4_097])
def test_the_transport_boundary_preserves_or_refuses_the_complete_payload(size: int) -> None:
    constraints = _ordinary_constraints()
    empty_size = len(json.dumps(constraints, ensure_ascii=False, sort_keys=True))
    constraints["cohort"]["review"] += "x" * (size - empty_size)
    payload = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert len(payload) == size

    if size == 4_096:
        assert agent_pipeline_runs._compile_data_constraints(constraints) == payload
    else:
        with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
            agent_pipeline_runs._compile_data_constraints(constraints)
        assert raised.value.code == "research_pipeline_data_constraints_too_large"
        assert raised.value.details["limit_chars"] == 4_096
        assert raised.value.details["serialized_chars"] == size
        assert raised.value.details["field"] == "data_constraints"
        assert raised.value.details["section_chars"]["cohort"] == len(
            json.dumps(constraints["cohort"], ensure_ascii=False, sort_keys=True)
        )


def test_a_list_cannot_be_elided_to_fit_the_transport() -> None:
    constraints = _ordinary_constraints()
    constraints["source_selection"]["host_applied"] += [
        f"Synthetic eligibility criterion {index}: require complete source coordinates."
        for index in range(24)
    ]
    assert len(json.dumps(constraints, ensure_ascii=False, sort_keys=True)) > 4_096

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._compile_data_constraints(constraints)

    assert raised.value.code == "research_pipeline_data_constraints_too_large"


@pytest.mark.parametrize(
    "cohort",
    [
        {"preset": "unregistered", "exclude_readmissions": True},
        {"icd_enabled": "invalid", "exclude_readmissions": True},
        {"exclude_readmissions": "invalid"},
    ],
)
def test_an_invalid_cohort_declares_no_applied_restriction_and_still_refuses_execution(
    tmp_path: Path, cohort: dict,
) -> None:
    study = {"cohort": cohort}
    assert agent_pipeline_runs._host_applied_criteria(study) == []
    assert agent_pipeline_runs._exclusion_criteria(study) == []

    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"files": [], "cohort_contract": {"preset": "all_icu"}}),
        encoding="utf-8",
    )
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        _require_export_holds_study_cohort(study, str(export))

    assert raised.value.code == "research_pipeline_export_cohort_invalid"
    expected = "unsupported_cohort_preset" if "preset" in cohort else "invalid_cohort_boolean"
    assert raised.value.details == {"reason_code": expected}


def test_a_cohort_owner_error_does_not_crash_the_diagnostic_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid(_cohort: object) -> bool:
        raise primary_cohort.PrimaryCohortContractError("invalid_cohort_boolean")

    monkeypatch.setattr(primary_cohort, "first_icu_stay_only", invalid)

    assert agent_pipeline_runs._host_applied_criteria({"cohort": {}}) == []
