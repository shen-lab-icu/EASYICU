"""Diagnostic steps remain part of the sealed scientific identity on resume."""

import pytest

from easyicu.research_agent.authority.plan_scope import (
    _ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
    _analysis_step_scientific_authority_payload,
    _step_scientific_signature,
    completed_step_record_matches_plan,
    plan_scientific_scope_signature,
)
from easyicu.research_agent.contracts.cluster_sizes import CLUSTER_SIZE_OUTPUTS
from easyicu.research_agent.contracts.covariate_balance import BALANCE_OUTPUTS
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


BALANCE = {
    "unit_id_column": "unit", "analysis_unit": "encounter",
    "treatment_column": "treated", "weight_column": "weight",
    "covariates": [{"column": "x", "kind": "continuous"}],
}
CLUSTERS = {
    "unit_id_column": "unit", "analysis_unit": "encounter",
    "assignment_source": "table:phenotype_assignments", "assignment_unit_id_column": "unit_id",
    "label_column": "cluster", "expected_clusters": ["A", "B"],
    "missing_label_policy": "exclude",
}


def test_legacy_step_signature_preserves_absent_fields():
    step = AnalysisStep(step_id="diagnostic", intent="Review the diagnostic.")
    legacy_fields = _ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS - {
        "covariate_balance_spec", "cluster_size_spec",
    }
    assert _analysis_step_scientific_authority_payload(step) == step.model_dump(
        mode="json", include=legacy_fields,
    )
    old_record = step.model_dump(mode="json", exclude={"covariate_balance_spec", "cluster_size_spec"})
    assert _step_scientific_signature(AnalysisStep.model_validate(old_record)) == _step_scientific_signature(step)


@pytest.mark.parametrize(
    "field,spec,change",
    [
        ("covariate_balance_spec", BALANCE, {"weight_column": None}),
        ("cluster_size_spec", CLUSTERS, {"expected_clusters": ["A", "C"]}),
    ],
)
def test_diagnostic_scientific_changes_invalidate_a_completed_step(field, spec, change):
    def make(data):
        balance = field == "covariate_balance_spec"
        return AnalysisStep.model_validate({
            "step_id": "diagnostic", "intent": "Review the diagnostic.",
            "method": "covariate_balance" if balance else "cluster_sizes",
            "planned_analysis_role": "secondary",
            "scientific_action_id": "causal_emulation.covariate_balance" if balance else "phenotyping.cluster_sizes",
            "inputs": ["artifact:analysis_cohort", "unit", "treated", "weight", "x"] if balance
                      else ["artifact:analysis_cohort", "table:phenotype_assignments", "unit"],
            "expected_outputs": list(BALANCE_OUTPUTS if balance else CLUSTER_SIZE_OUTPUTS),
            field: data,
        })

    sealed = make(spec)
    same = AnalysisStep.model_validate(sealed.model_dump(mode="json"))
    changed = make({**spec, **change})
    assert _step_scientific_signature(sealed) == _step_scientific_signature(same)
    assert _step_scientific_signature(sealed) != _step_scientific_signature(changed)
    assert _step_scientific_signature(sealed) != _step_scientific_signature(
        AnalysisStep(step_id="diagnostic", intent="Review the diagnostic.")
    )
    plan = AnalysisPlan(research_question="Review the diagnostic.", steps=[sealed])
    scope = list(plan_scientific_scope_signature(plan))
    record = {
        "step_id": sealed.step_id,
        "planned_analysis_role": sealed.planned_analysis_role,
        "analysis_request": {"step": sealed.model_dump(mode="json")},
        "plan_scientific_signature": scope,
    }
    def matches(step):
        return completed_step_record_matches_plan(
            record, step=step, expected_plan_scope=scope,
            plan_scope_count=1, completed_records=[record],
        )
    assert matches(same)
    assert not matches(changed)
