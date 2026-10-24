"""The hospital first-ICU rule stays distinct across plan, receipt and report."""

from tests.support.hospitalization import receipt
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import SELECTION_POLICY
from easyicu.research_agent.intake.materialized_metadata import (
    FIRST_ICU_STAY_PER_HOSPITALIZATION_RESTRICTION_SCHEMA as SCHEMA,
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
    MaterializedMetadataError,
    sealed_first_icu_stay_per_hospitalization_restriction,
)
from easyicu.research_agent.planning.dependence_authority import _host_restricted_to_first_icu_stays
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.research_context.builder import _planning_catalog_provenance
from easyicu.research_agent.research_context.export_selection import export_selection_counts
from easyicu.research_agent.reporting.population_selection import _step_label
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext, UserPreferences
from easyicu.webserver.primary_cohort import (
    PrimaryCohortContractError, first_icu_stay_only,
    first_icu_stay_per_hospitalization_only, normalize_primary_cohort_scope,
    normalize_execution_cohort, planner_selectable_cohort,
)

KEY = "first_icu_stay_per_hospitalization_restriction"
KIND = "first_icu_stay_per_hospitalization"




def context(provenance):
    return ResearchContext(
        variables=[],
        research_question="Adults, first ICU within each hospitalization, by sex mortality",
        cohort=CohortDescriptor(cohort_name="synthetic", database="eicu", n_stays=4,
                                provenance=provenance),
        user_preferences=UserPreferences(data_constraints=json.dumps({
            "source_selection": {"basis": "export_contract", "export_report": {
                "source_total": 10, "selected": 10,
            }},
        })),
    )


def compile_one(kind, provenance):
    spec = PopulationSpec.model_validate({"criteria": [{
        "id": "c1", "kind": kind, "role": "include", "source": "question",
        "quote": "first ICU within each hospitalization",
    }]})
    return compile_population(spec, context(provenance)).criteria[0]


def read_sealed(raw, tmp_path):
    verified = SimpleNamespace(authority=SimpleNamespace(
        producer_parameters={KEY: raw}, parent_authority_sha256=None,
    ))
    return sealed_first_icu_stay_per_hospitalization_restriction(
        verified, cohort_path=tmp_path / "cohort.parquet",
    )


def test_new_scope_never_substitutes_for_each_patient_first_stay(tmp_path):
    old = {"schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA, "coordinate_sha256": "b" * 64}
    assert compile_one(KIND, {"first_icu_stay_restriction": old}).disposition == "requires_extraction"
    new = {KEY: read_sealed(receipt(), tmp_path)}
    assert compile_one("first_icu_stay", new).disposition == "requires_extraction"
    assert _host_restricted_to_first_icu_stays(new) is False
    applied = compile_one(KIND, new)
    assert applied.disposition == "applied_by_source"
    assert applied.proof.kind == "host_first_icu_stay_per_hospitalization"
    assert applied.proof.parameters["establishes_patient_independence"] is False
    assert applied.proof.parameters["selection_rule"] == SELECTION_POLICY["selection_rule"]


@pytest.mark.parametrize("bad", [
    {"coordinate_sha256": "bad"}, {"schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA},
    {"authority_coordinates": {}},
])
def test_malformed_new_proof_does_not_pass_compilation(bad):
    raw = receipt(); raw.update(bad)
    assert compile_one(KIND, {KEY: raw}).disposition == "requires_extraction"


def test_sealed_receipt_keeps_policy_and_three_exclusion_classes_without_identity(tmp_path):
    raw = receipt()
    projected = read_sealed(raw, tmp_path)
    assert projected["non_icu_stays_removed"] == 1
    assert projected["unknown_eligibility_stays_removed"] == 2
    assert projected["icu_unit_types"] == list(SELECTION_POLICY["icu_unit_types"])
    assert "private_source_path" not in json.dumps(projected)
    result = export_selection_counts(context({KEY: projected}))
    assert result.unavailable is None
    step = result.counts.steps[-1]
    assert (step.n_before, step.n_excluded, step.n_remaining) == (10, 6, 4)
    label = _step_label(step)
    assert "within a hospitalization" in label and "non-ICU: 1" in label
    assert "unknown eligibility: 2" in label and "patient's first" not in label


@pytest.mark.parametrize("field,value", [
    ("stays_after", 5), ("non_first_icu_stays_removed", True),
    ("non_icu_stays_removed", 2), ("unknown_eligibility_stays_removed", -1),
    ("excluded_reason_counts", {"patient_123": 6}),
    ("authority_coordinates", {**SELECTION_POLICY, "establishes_patient_independence": True}),
])
def test_corrupt_aggregate_or_policy_fails_closed(tmp_path, field, value):
    raw = receipt(); raw[field] = value
    with pytest.raises(MaterializedMetadataError):
        read_sealed(raw, tmp_path)


def test_planning_projection_declares_exact_icu_rule_without_private_source_details():
    raw = receipt(); raw["provider_visible_values"] = False
    frame = pd.DataFrame()
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False, KEY: raw,
    }
    projected = _planning_catalog_provenance(frame)[KEY]
    assert projected["unknown_unit_type_policy"] == "exclude_hospitalization_with_receipt"
    assert projected["prior_unit_coverage_policy"] == "require_all_ordinals_one_through_selected_icu"
    assert "stays_before" not in projected
    assert "private_source_path" not in json.dumps(projected)
    raw["provider_visible_values"] = True
    with pytest.raises(MaterializedMetadataError):
        _planning_catalog_provenance(frame)


def test_primary_scope_is_new_and_does_not_change_old_all_stays_digest():
    base = {"preset": "all_icu"}
    assert normalize_primary_cohort_scope(base).sha256 == normalize_primary_cohort_scope({**base, KIND: False}).sha256
    requested = {**base, KIND: True}
    assert first_icu_stay_per_hospitalization_only(requested) is True
    assert first_icu_stay_only(requested) is False
    normalized = normalize_primary_cohort_scope(requested)
    assert normalized.admission_eligibility["repeated_admission_policy"] == KIND
    assert "admission_eligibility.repeated_admission_policy" in normalized.stated_fields
    with pytest.raises(PrimaryCohortContractError):
        normalize_primary_cohort_scope({**requested, "exclude_readmissions": True})


@pytest.mark.parametrize("patient_after", [10, 4])
def test_conflicting_receipts_never_establish_counts_or_an_applied_population(tmp_path, patient_after):
    hospital = read_sealed(receipt(), tmp_path)
    # Even a zero-exclusion first receipt can make both ledgers chain. The
    # scopes remain mutually exclusive; neither may silently win by ordering.
    provenance = {
        "first_icu_stay_restriction": {
            "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA,
            "coordinate_sha256": "b" * 64,
            "stays_before": 10, "stays_after": patient_after,
            "non_first_icu_stays_removed": 10 - patient_after,
        },
        KEY: hospital,
    }
    result = export_selection_counts(context(provenance))
    assert result.counts is None
    assert result.unavailable == "host_restriction_unreadable"
    for kind in ("first_icu_stay", KIND):
        assert compile_one(kind, provenance).disposition == "requires_extraction"


@pytest.mark.parametrize("raw,patient,hospital", [
    ({"preset": "adult_first"}, True, False),
    ({"preset": "adult_all", KIND: True}, False, True),
    ({"preset": "all_icu"}, False, False),
])
def test_both_scope_readers_accept_raw_and_normalized_cohorts(raw, patient, hospital):
    normalized = normalize_execution_cohort(raw)
    for cohort in (raw, normalized):
        assert first_icu_stay_only(cohort) is patient
        assert first_icu_stay_per_hospitalization_only(cohort) is hospital
    selectable = planner_selectable_cohort(normalized)
    assert selectable["exclude_readmissions"] is False
    assert selectable[KIND] is False
    assert selectable["age_min"] == normalized["age_min"]


@pytest.mark.parametrize("field", [
    "stays_after", "non_first_icu_stays_removed", "non_icu_stays_removed",
    "unknown_eligibility_stays_removed",
])
def test_report_counts_refuse_each_broken_exclusion_component(tmp_path, field):
    projected = read_sealed(receipt(), tmp_path)
    projected[field] += 1
    result = export_selection_counts(context({KEY: projected}))
    assert result.counts is None
    assert result.unavailable == "host_restriction_unreadable"
