"""A first-stay cohort launches only with a verified first-ICU-stay coordinate.

An ICU-readmission indicator never stands in for that authority.  The launch,
the readiness review and the runner all ask the same owner, so a plan cannot
be approved against a restriction the runner would not apply.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.acquisition.first_icu_stay import FirstIcuStayBinding
from easyicu.webserver import (
    agent_pipeline_runs,
    data_package_execution_readiness,
    primary_cohort,
    research_launch_scientific,
    source_identity_authority,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


def _study(**cohort) -> dict:
    return {
        "cohort": cohort,
        "analysis_design": {"analysis_unit": "icu_stay", "variance_estimator": "model_based"},
        "data_source": {"path": "/exports/miiv", "database": "miiv"},
    }


def _binding(tmp_path: Path) -> FirstIcuStayBinding:
    return FirstIcuStayBinding(
        coordinate_path=tmp_path / "first.parquet",
        coordinate_sha256="c" * 64,
        authority_coordinates={
            "authority_ref": "export_manifest_data_path/mimic_iv/1/first_icu_stay",
            "order_rule": "earliest_order_time_per_patient",
            "stays": 3,
            "patients": 2,
            "non_first_icu_stays": 1,
        },
    )


def _resolver(monkeypatch: pytest.MonkeyPatch, result) -> list:
    calls: list = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay", resolve)
    return calls


def test_a_verified_coordinate_lets_a_first_stay_study_launch(tmp_path, monkeypatch):
    calls = _resolver(monkeypatch, _binding(tmp_path))

    design = research_launch_scientific._validate_analysis_design(
        _study(exclude_readmissions=True)
    )

    assert design == {"analysis_unit": "icu_stay", "variance_estimator": "model_based"}
    assert calls == [{"export_path": "/exports/miiv", "database": "miiv"}]


def test_without_a_coordinate_the_restriction_stays_unverified(monkeypatch):
    _resolver(monkeypatch, None)

    with pytest.raises(ResearchPipelineRunError) as exc:
        research_launch_scientific._validate_analysis_design(_study(exclude_readmissions=True))

    assert exc.value.code == "research_pipeline_first_stay_restriction_unverified"
    assert exc.value.details["first_icu_stay_reason_code"] == "first_icu_stay_authority_unavailable"
    assert exc.value.details["icu_readmission_is_first_patient_stay_authority"] is False


def test_an_unprovable_order_keeps_its_lower_layer_cause(monkeypatch):
    _resolver(
        monkeypatch,
        source_identity_authority.PatientGroupingAuthorityError(
            "raw_source_authority_first_icu_stay_invalid",
            "tied",
            details={"database": "miiv", "cause_code": "first_icu_stay_order_tied", "cause": "x"},
        ),
    )

    with pytest.raises(ResearchPipelineRunError) as exc:
        research_launch_scientific._validate_analysis_design(_study(exclude_readmissions=True))

    details = exc.value.details
    assert details["first_icu_stay_reason_code"] == "raw_source_authority_first_icu_stay_invalid"
    assert details["cause_code"] == "first_icu_stay_order_tied"
    assert "cause" not in details


def test_a_legacy_adult_first_preset_is_a_first_stay_cohort(monkeypatch):
    calls = _resolver(monkeypatch, None)

    assert primary_cohort.first_icu_stay_only({"preset": "adult_first"}) is True
    with pytest.raises(ResearchPipelineRunError):
        research_launch_scientific._validate_analysis_design(_study(preset="adult_first"))
    assert calls, "the implied restriction must be verified, not skipped"


def test_an_all_stay_cohort_never_asks_for_a_first_stay_coordinate(monkeypatch):
    calls = _resolver(monkeypatch, None)

    assert research_launch_scientific._first_icu_stay_for_cohort(_study(age_min=18)) is None
    assert calls == []


def test_the_planner_never_writes_the_host_owned_first_stay_predicate():
    mode = lambda cohort: primary_cohort.normalize_primary_cohort_scope(  # noqa: E731
        primary_cohort.planner_selectable_cohort(cohort)
    ).selection_mode

    assert mode({"exclude_readmissions": True}) == "all_input_rows"
    assert mode({"exclude_readmissions": True, "age_min": 18}) == "predicate_filtered"
    assert mode({"preset": "adult_first"}) == "predicate_filtered"  # its age axis stays
    # The execution contract itself is unchanged.
    assert (
        primary_cohort.normalize_primary_cohort_scope({"exclude_readmissions": True}).selection_mode
        == "predicate_filtered"
    )


def test_the_runner_refuses_a_universe_without_the_bound_restriction(tmp_path):
    binding = _binding(tmp_path)
    provenance = tmp_path / "universe_provenance.json"
    acquisition = SimpleNamespace(provenance_path=provenance)

    provenance.write_text(json.dumps({"columns": ["stay_id"]}), encoding="utf-8")
    with pytest.raises(ResearchPipelineRunError) as exc:
        agent_pipeline_runs._require_first_icu_stay_materialized(acquisition, binding)
    assert exc.value.code == "research_pipeline_first_stay_restriction_not_materialized"

    provenance.write_text(
        json.dumps({"first_icu_stay_restriction": {"coordinate_sha256": "c" * 64}}),
        encoding="utf-8",
    )
    agent_pipeline_runs._require_first_icu_stay_materialized(acquisition, binding)


def test_readiness_reports_the_coordinate_and_blocks_without_it(tmp_path, monkeypatch):
    review = data_package_execution_readiness._runtime_readiness_review

    _resolver(monkeypatch, None)
    blocked = review(_study(exclude_readmissions=True), source_path="/exports/miiv", catalog_by_id={})
    assert blocked["status"] == "blocked"
    assert "first_icu_stay_authority_unavailable" in blocked["required_findings"]
    assert blocked["first_icu_stay"]["status"] == "unavailable"
    # The readmission indicator still never claims the authority.
    assert blocked["readmission_indicator"]["first_patient_stay_authority"] is False

    _resolver(monkeypatch, _binding(tmp_path))
    ready = review(_study(exclude_readmissions=True), source_path="/exports/miiv", catalog_by_id={})
    assert ready["first_icu_stay"]["status"] == "ready"
    assert ready["first_icu_stay"]["non_first_icu_stays"] == 1
    assert ready["first_icu_stay"]["provider_visible_values"] is False
    assert "first_icu_stay_authority_unavailable" not in ready["required_findings"]

    all_stays = review(_study(), source_path="/exports/miiv", catalog_by_id={})
    assert "first_icu_stay" not in all_stays


def test_the_eligible_denominator_counts_only_first_stays(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        data_package_execution_readiness,
        "_read_review_concept",
        lambda _path, _concept: pd.DataFrame({"stay_id": [11, 12, 13], "age": [70.0, 71.0, 16.0]}),
    )
    flags = {"value": pd.Series({11: True, 12: False, 13: True})}
    monkeypatch.setattr(
        data_package_execution_readiness,
        "_first_icu_stay_flags",
        lambda _study, *, source_path: flags["value"],
    )
    review = data_package_execution_readiness._cohort_eligibility_review

    counted = review(
        _study(exclude_readmissions=True, age_min=18), source_path="/x", registered_denominator=3
    )
    # 11 is a first stay of an adult; 12 is a later stay; 13 is a first stay under 18.
    assert counted["count"] == 1
    assert counted["excluded_non_first_icu_stay_count"] == 1
    assert counted["basis"] == "typed_age_eligibility+first_icu_stay"

    flags["value"] = pd.Series({11: True})
    uncovered = review(
        _study(exclude_readmissions=True), source_path="/x", registered_denominator=3
    )
    assert uncovered["status"] == "unavailable"
    assert uncovered["reason_code"] == "cohort_first_icu_stay_coverage_incomplete"


def test_the_runner_asks_the_one_cohort_owner_for_both_modes(monkeypatch):
    """Execution keeps the owner's mode; the Planner's drops only the host axis."""

    from easyicu.webserver import study_contexts

    seen: list = []

    def owner(study):
        seen.append(dict(study["cohort"]))
        return "owner_mode"

    monkeypatch.setattr(study_contexts, "primary_cohort_selection_mode", owner)
    study = {"cohort": {"exclude_readmissions": True, "age_min": 18}}

    assert agent_pipeline_runs.primary_cohort_selection_mode(study) == "owner_mode"
    assert agent_pipeline_runs._planner_cohort_selection_mode(study) == "owner_mode"
    assert seen == [
        {"exclude_readmissions": True, "age_min": 18},
        {"exclude_readmissions": False, "age_min": 18},
    ]
