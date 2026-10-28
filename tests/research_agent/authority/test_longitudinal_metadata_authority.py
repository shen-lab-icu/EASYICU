"""Workflow graph coordinates cannot become a run's scientific authority."""

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.execution.runners.target_trial_executor import _sealed as load_target_trial_executor_authority
from easyicu.research_agent.execution.runners.time_varying_executor import _authority as load_time_varying_executor_authority
from easyicu.research_agent.planning.family_spec.longitudinal_canonical import canonical_longitudinal_request
from easyicu.research_agent.schema import AnalysisPlan
from tests.support.target_trial import target_trial_authority_body


FAMILIES = ("target_trial_emulation", "time_varying_exposure_survival")
REASON = "workflow_metadata_not_executable"


def _metadata_owner(family_id):
    request, _spec = canonical_longitudinal_request(family_id)
    return request.sealed_target_trial or request.sealed_time_varying


def _host_body(family_id):
    if family_id == "target_trial_emulation":
        return target_trial_authority_body(development_execution_only_allowed=True, database="workflow_metadata")
    return {
        "schema_version": "easyicu.time_varying_runtime_authority/1",
        "authority_kind": "time_varying_exposure_association",
        "protocol_content_sha256": "a" * 64,
        "sensitivity_spec_id": "synthetic_time_updated",
        "specification": {
            "exposure_concept": "synthetic_measurement",
            "exposure_summary": "running_max_of_direct_measurements",
            "exposure_window_hours": 24,
            "followup": "hospital_death_or_discharge_from_icu_admission",
            "missingness_policy": "observed_state_indicator",
            "baseline_columns": ["synthetic_baseline"],
            "baseline_categorical_encodings": {},
            "interpretation": "descriptive_time_updated_association_not_causal",
        },
        "exposure_column": "synthetic_measurement",
        "outcome_column": "death",
        "identity_column": "patient_stay_id",
        "primary_cohort_selection_mode": "all_input_rows",
        "development_execution_only_allowed": True,
        "plan_method": "time_varying_exposure_model",
        "plan_intent": "Estimate a prespecified synthetic time-updated association.",
        "plan_outputs": [
            "table:time_varying_cox_estimates", "table:time_varying_input_audit", "log:time_varying_runtime_receipt",
        ],
    }


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("transport", ["mapping", "instance"])
def test_runtime_loader_rejects_metadata_with_a_stable_reason(family_id, transport):
    owner = _metadata_owner(family_id)
    assert owner.authority_origin == "workflow_metadata"
    value = owner.model_dump(mode="json") if transport == "mapping" else owner
    with pytest.raises(ValidationError) as error:
        load_current_case_scientific_runtime_authority(value)
    assert error.value.errors()[0]["type"] == REASON


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("transport", ["mapping", "instance"])
def test_executor_loaders_refuse_embedded_metadata(family_id, transport):
    owner = _metadata_owner(family_id)
    load = load_target_trial_executor_authority if family_id == "target_trial_emulation" else load_time_varying_executor_authority
    value = owner.model_dump(mode="json") if transport == "mapping" else owner
    with pytest.raises(ValueError) as error:
        load(value)
    reason = error.value.errors()[0]["type"] if isinstance(error.value, ValidationError) else error.value.type
    assert reason == REASON


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("operation", ["bind_plan", "validate_plan", "governed_step", "development_execution_only_plan"])
def test_every_execution_entry_refuses_metadata_before_plan_processing(family_id, operation):
    owner = _metadata_owner(family_id)
    plan = AnalysisPlan(research_question="Synthetic authority boundary.", analysis_type="association", steps=[])
    with pytest.raises(ValueError) as error:
        if operation == "development_execution_only_plan":
            owner.development_execution_only_plan(research_question=plan.research_question)
        else:
            getattr(owner, operation)(plan)
    assert error.value.type == REASON


@pytest.mark.parametrize("family_id", FAMILIES)
def test_metadata_is_refused_before_constructing_an_execution_plan(family_id, monkeypatch):
    owner = _metadata_owner(family_id)

    def unexpected_plan(*args, **kwargs):
        pytest.fail("Metadata reached execution plan construction.")

    monkeypatch.setattr(AnalysisPlan, "model_validate", unexpected_plan)
    with pytest.raises(ValueError) as error:
        owner.development_execution_only_plan(research_question="Synthetic authority boundary.")
    assert error.value.type == REASON


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("replacement", [None, "host_signed"])
def test_metadata_origin_cannot_be_stripped_or_relabelled_with_its_digest(family_id, replacement):
    owner = _metadata_owner(family_id)
    body = owner.model_dump(mode="json", exclude={"execution_contract_sha256"})
    if family_id == "time_varying_exposure_survival":
        body.pop("plan_literature_citation_keys")
        body.pop("plan_literature_design_bindings")
    assert body["authority_origin"] == "workflow_metadata"
    assert canonical_sha256(body) == owner.execution_contract_sha256
    payload = owner.model_dump(mode="json")
    if replacement is None:
        payload.pop("authority_origin")
    else:
        payload["authority_origin"] = replacement
    with pytest.raises(ValidationError, match="authority digest mismatch"):
        load_current_case_scientific_runtime_authority(payload)


@pytest.mark.parametrize("family_id", FAMILIES)
def test_an_unvalidated_origin_edit_cannot_bypass_plan_binding(family_id):
    owner = _metadata_owner(family_id).model_copy(update={"authority_origin": "host_signed"})
    plan = AnalysisPlan(research_question="Synthetic authority boundary.", analysis_type="association", steps=[])
    with pytest.raises(ValueError, match="authority digest mismatch"):
        owner.bind_plan(plan)


@pytest.mark.parametrize("family_id", FAMILIES)
def test_metadata_cannot_use_the_legacy_host_digest_exception(family_id):
    owner = build_current_case_scientific_runtime_authority(_host_body(family_id))
    payload = owner.model_dump(mode="json")
    payload["authority_origin"] = "workflow_metadata"
    with pytest.raises(ValidationError, match="authority digest mismatch"):
        type(owner).model_validate_json(canonical_json(payload))


@pytest.mark.parametrize("family_id", FAMILIES)
def test_authority_origin_is_a_closed_typed_choice(family_id):
    owner = _metadata_owner(family_id)
    payload = owner.model_dump(mode="json")
    payload["authority_origin"] = "arbitrary_caller"
    with pytest.raises(ValidationError) as error:
        type(owner).model_validate_json(canonical_json(payload))
    assert error.value.errors()[0]["type"] == "literal_error"


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("explicit_origin", [False, True])
def test_host_authority_digests_roundtrip_and_still_bind(family_id, explicit_origin):
    body = _host_body(family_id)
    if explicit_origin:
        body["authority_origin"] = "host_signed"
    authority = build_current_case_scientific_runtime_authority(body)
    assert authority.authority_origin == "host_signed"
    assert authority.execution_contract_sha256 == canonical_sha256(body)
    loaded = load_current_case_scientific_runtime_authority(authority.model_dump(mode="json"))
    assert loaded == authority
    assert load_current_case_scientific_runtime_authority(loaded) == authority
    load = load_target_trial_executor_authority if family_id == "target_trial_emulation" else load_time_varying_executor_authority
    assert load(loaded) == load(loaded.model_dump(mode="json")) == authority
    plan = loaded.development_execution_only_plan(research_question="Synthetic host-reviewed study.")
    loaded.validate_plan(plan)
    loaded.validate_plan(loaded.bind_plan(plan))


@pytest.mark.parametrize("origin", [None, "host_signed", "workflow_metadata"])
def test_time_varying_literature_version_preserves_its_origin_and_digest(origin):
    body = _host_body("time_varying_exposure_survival")
    body.update(
        schema_version="easyicu.time_varying_runtime_authority/2",
        plan_literature_citation_keys=["synthetic_method"],
        plan_literature_design_bindings=[{
            "citation_key": "synthetic_method", "design_elements": ["time_zero"],
            "application": "Synthetic rationale for the prespecified time origin.", "divergence": None,
        }],
    )
    if origin is not None:
        body["authority_origin"] = origin
    owner_type = type(_metadata_owner("time_varying_exposure_survival"))
    owner = owner_type.model_validate_json(canonical_json({
        **body, "execution_contract_sha256": canonical_sha256(body),
    }))
    if origin == "workflow_metadata":
        with pytest.raises(ValidationError) as error:
            load_current_case_scientific_runtime_authority(owner.model_dump(mode="json"))
        assert error.value.errors()[0]["type"] == REASON
    else:
        loaded = load_current_case_scientific_runtime_authority(owner.model_dump(mode="json"))
        loaded.validate_plan(loaded.development_execution_only_plan(research_question="Synthetic v2 study."))


def test_dynamic_coordinates_have_no_runtime_authority_dispatch():
    request, _spec = canonical_longitudinal_request("dynamic_landmark_prediction")
    with pytest.raises(ValidationError) as error:
        load_current_case_scientific_runtime_authority(request.dynamic_landmark.model_dump(mode="json"))
    assert error.value.errors()[0]["type"] == "union_tag_not_found"
