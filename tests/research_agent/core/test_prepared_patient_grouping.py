"""Synthetic prepared cohorts retain source grouping through run staging."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
    stage_exposure_grouped_cohort_authority,
    stage_materialized_cohort_authority,
    verified_cohort_replacement_row_identity,
)
from easyicu.research_agent.planning.dependence_authority import (
    descriptive_interval_decision,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.research_context import builder
from tests.research_agent.planning.descriptive_interval_fixtures import (
    descriptive_context,
    descriptive_plan,
)
from tests.support.typed_export import typed_export

_SOURCE_GROUPING = {
    "coordinate": "patient_stay_id",
    "group_derivation": "prefix_before_:s",
    "authority_ref": "synthetic/patient_grouping",
    "mapping_sha256": "c" * 64,
    "provider_visible_values": False,
}


def _prepared(tmp_path: Path, *, grouped: bool, exposure_groups: bool):
    source = typed_export(
        tmp_path / "source",
        labs=pd.DataFrame({
            "stay_id": [1, 2, 3, 4], "charttime": [1.0] * 4,
            "age": [50, 50, 60, 60], "lact": [1.0, 3.0, 1.5, 3.5],
            "mech_vent": [False, True, False, True],
        }),
        outcomes=pd.DataFrame({
            "stay_id": [1, 2, 3, 4], "death": [False, True, False, True],
        }),
    )
    identity = {}
    if grouped:
        mapping = tmp_path / "synthetic_grouping.parquet"
        pd.DataFrame({"stay_id": [1, 2, 3, 4], "patient_key": [7, 7, 8, 8]}).to_parquet(
            mapping, index=False,
        )
        mapping_sha256 = hashlib.sha256(mapping.read_bytes()).hexdigest()
        binding = PatientGroupingBinding(
            mapping_path=mapping,
            mapping_sha256=mapping_sha256,
            mapping_stay_column="stay_id",
            mapping_patient_column="patient_key",
            authority_coordinates={
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "synthetic/patient_grouping",
                "mapping_sha256": mapping_sha256,
                "grouping_derivation": "prefix_before_:s",
                "provider_visible_values": False,
            },
        )
        identity = binding.materializer_kwargs()
    paths = materialize_to_parquet(
        tmp_path / "prepared", stem="universe", database="miiv", data_path=source,
        static_concepts=("age",), feature_concepts=("lact",),
        outcome_concepts=("death",), **identity,
    )
    staged = tmp_path / "staged" / "cohort.parquet"
    stage_materialized_cohort_authority(
        paths["parquet"], staged, producer_implementation_sha256="a" * 64,
    )
    if exposure_groups:
        stage_exposure_grouped_cohort_authority(
            staged, producer_implementation_sha256="a" * 64,
            groupings_record_sha256="b" * 64,
            groupings=[{
                "variable": "lact_group_x1",
                "parameters": {
                    "concept": "lact",
                    "window": {"start_hours": 0.0, "end_hours": 24.0},
                    "scale": "nominal",
                    "groups": [
                        {"id": "g1", "rule": {
                            "summary": "max", "op": "<", "value": 2.5, "unit": None,
                        }},
                        {"id": "g2", "rule": "otherwise"},
                    ],
                    "unmeasured": "exclude",
                    "source_columns": {"max": "lact_max"},
                },
                "labels": {"g1": "lower", "g2": "higher"},
                "compared": {"reference": 1, "contrast": 2},
            }],
        )
    return staged


def _context(path, *, expected_grouping=False):
    template = descriptive_context()
    preferences = template.user_preferences.model_copy(deep=True)
    if expected_grouping:
        constraints = json.loads(preferences.data_constraints)
        constraints["verified_patient_grouping"] = _SOURCE_GROUPING
        preferences.data_constraints = json.dumps(constraints)
    context = builder.build_research_context(
        research_question="Describe a synthetic lactate cohort.",
        cohort=path, database="miiv", cohort_name="synthetic",
        target_outcome="death", primary_exposure="lact_max",
        user_preferences=preferences,
    )
    return template.model_copy(update={
        "cohort": context.cohort,
        "user_preferences": context.user_preferences,
    })


@pytest.mark.parametrize("exposure_groups", [False, True])
def test_preparation_preserves_grouping_and_patient_clustered_intervals(
    tmp_path, exposure_groups,
):
    context = _context(_prepared(tmp_path, grouped=True, exposure_groups=exposure_groups))
    source = load_verified_materialized_cohort_authority(tmp_path / "prepared/universe.parquet")
    source_identity = verified_cohort_replacement_row_identity(source)
    catalog = pd.DataFrame({
        "patient_stay_id": pd.Series(dtype="string"),
        "lact_max": pd.Series(dtype="float64"),
        "death": pd.Series(dtype="float64"),
    })
    catalog.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
        "patient_grouping_status": "bound",
        "replacement_row_identity": {**source_identity, "mapped_cohort_rows": 0},
    }
    candidate = _context(catalog)
    assert candidate.cohort.n_stays == 0
    assert candidate.cohort.provenance["patient_grouping_status"] == "bound"
    candidate_spec = descriptive_plan(candidate).steps[-1].exposure_outcome_distribution_spec
    assert context.cohort.provenance["replacement_row_identity"] == source_identity
    # The parent receipt, not the column's name or a Planner declaration, binds groups.
    assert context.cohort.provenance["n_patients_source"] == "patient_stay_id:prefix_before_:s"
    assert context.cohort.n_patients == 2
    assert context.cohort.provenance["patient_grouping_status"] == "bound"
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "wilson"
    assert spec.repeated_unit_interval_method == "patient_cluster_robust_wald"
    assert spec.dependence.group_source == "patient_stay_id"
    assert spec.dependence.delimiter == ":s"
    assert spec.interval_limitation is None
    assert spec.dependence == candidate_spec.dependence
    assert spec.repeated_unit_interval_method == candidate_spec.repeated_unit_interval_method


def test_a_source_without_grouping_keeps_the_existing_counts_only_limitation(tmp_path):
    context = _context(_prepared(tmp_path, grouped=False, exposure_groups=True))
    decision = descriptive_interval_decision(context, analysis_type="descriptive_epidemiology")
    assert decision.counts_only
    assert decision.dependence is None
    assert decision.limitation.reason == "repeated_stays_possible_without_verified_patient_grouping"


def test_a_prepared_package_missing_the_declared_grouping_names_that_cause(tmp_path):
    context = _context(
        _prepared(tmp_path, grouped=False, exposure_groups=False),
        expected_grouping=True,
    )
    spec = descriptive_plan(context).steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.dependence is None
    assert spec.interval_limitation.reason == "verified_patient_grouping_not_carried_by_prepared_package"
    assert "prepared package does not carry" in spec.interval_limitation.text
    assert "without verified patient grouping" not in spec.interval_limitation.text
    review = build_plan_scientific_review(context=context, plan=descriptive_plan(context))
    finding = next(f for f in review.findings if f.descriptive_interval_limitation)
    assert finding.descriptive_interval_limitation == spec.interval_limitation
    assert finding.message == spec.interval_limitation.text


@pytest.mark.parametrize("changed", [
    {"cohort_rows": 3},
    {"identity_column": "lact_max"},
    {"row_identity_sha256": "f" * 64},
])
def test_inherited_grouping_cannot_authorize_a_different_row_space(tmp_path, changed):
    path = _prepared(tmp_path, grouped=True, exposure_groups=True)
    verified = load_verified_materialized_cohort_authority(path)
    drifted = replace(verified, authority=replace(verified.authority, **changed))
    with pytest.raises(MaterializedMetadataError, match="prepared row space"):
        verified_cohort_replacement_row_identity(drifted, cohort_path=path)


@pytest.mark.parametrize("missing", [*_SOURCE_GROUPING, "not_an_object"])
def test_incomplete_source_claims_do_not_assert_verified_grouping(missing):
    context = descriptive_context()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["verified_patient_grouping"] = (
        [] if missing == "not_an_object" else {**_SOURCE_GROUPING, missing: None}
    )
    context.user_preferences.data_constraints = json.dumps(constraints)
    decision = descriptive_interval_decision(context, analysis_type="descriptive_epidemiology")
    assert decision.counts_only
    assert decision.dependence is None
    assert decision.limitation.reason == "repeated_stays_possible_without_verified_patient_grouping"


def test_metadata_only_planning_does_not_claim_a_package_has_lost_grouping():
    context = descriptive_context()
    context.cohort.provenance["evidence_stage"] = "metadata_only_planning"
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["verified_patient_grouping"] = _SOURCE_GROUPING
    context.user_preferences.data_constraints = json.dumps(constraints)
    decision = descriptive_interval_decision(context, analysis_type="descriptive_epidemiology")
    assert decision.counts_only
    assert decision.limitation.reason == "repeated_stays_possible_without_verified_patient_grouping"
