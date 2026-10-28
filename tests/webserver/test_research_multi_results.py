"""Only sealed primary aggregate products may enter the side-by-side answer."""
from types import SimpleNamespace

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.methods.sensitivity import describe_estimate_heterogeneity
from easyicu.webserver.research_multi_results import sealed_primary_estimate
from tests.support.result_append import completed_source


def test_primary_projection_uses_the_existing_verified_product_resolver(tmp_path):
    wrapper, run, plan, _, _ = completed_source(tmp_path)
    target = SimpleNamespace(target=SimpleNamespace(database="miiv"), source_sha256="a" * 64,
        review=SimpleNamespace(run_id="source", study_context_id="study", plan_sha256=canonical_sha256(plan.model_dump(mode="json"))),
        decision=SimpleNamespace(substitutions=[]))
    result = sealed_primary_estimate(wrapper, target)
    assert result["population_n"] == 400
    assert result["unit"] == "odds_ratio"
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert len(result["evidence_sha256"]) == 64
    # Corrupt a registered product, leaving browser tables untouched.
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

    evidence = EvidenceStore(run).get(result["evidence_id"])
    path = verified_run_evidence_path(run, evidence)
    path.write_text("fit_status,estimate,ci_low,ci_high\nfitted,999,998,1000\n")
    with pytest.raises(ValueError):
        sealed_primary_estimate(wrapper, target)


def test_deterministic_heterogeneity_reports_range_and_direction_without_pooling():
    result = describe_estimate_heterogeneity([0.5, 1.0, 2.0], null_value=1.0)
    assert result == dict(method="estimate_range_and_null_direction_v1", scope="analysis_level", n_databases=3,
                          minimum=0.5, maximum=2.0, spread=1.5, null_value=1.0, description_code="directions_differ")
    assert describe_estimate_heterogeneity([2.0, 1.0, 0.5], null_value=1.0) == result
    assert describe_estimate_heterogeneity([-2.0, -1.0], null_value=0.0)["description_code"] == "directions_agree"


@pytest.mark.parametrize("values,null", [([1.0], 1.0), ([1.0] * 7, 1.0), ([1.0, float("nan")], 1.0),
                                        ([1.0, 2.0], float("inf")), ([-1e308, 1e308], 0.0)])
def test_invalid_comparison_numbers_are_refused(values, null):
    with pytest.raises(ValueError):
        describe_estimate_heterogeneity(values, null_value=null)


def test_sealed_projection_retains_coefficient_se_and_requires_all_numeric_gates(tmp_path):
    from tests.support.result_append import json_file

    wrapper, run, plan, _, _ = completed_source(tmp_path)
    target = SimpleNamespace(target=SimpleNamespace(database="miiv"), source_sha256="a" * 64,
        review=SimpleNamespace(run_id="source", study_context_id="study", plan_sha256=canonical_sha256(plan.model_dump(mode="json"))),
        decision=SimpleNamespace(substitutions=[]))
    assert sealed_primary_estimate(wrapper, target)["gate_passed"] is False
    gates = dict(execution_complete=True, analysis_validated=True, evidence_complete=True, numeric_verified=True)
    json_file(run / "run_status.json", {"gates": gates})
    value = sealed_primary_estimate(wrapper, target)
    assert value["gate_passed"] is True
    assert value["standard_error"] > 0
    for gate in gates:
        json_file(run / "run_status.json", {"gates": {**gates, gate: False}})
        try:
            assert sealed_primary_estimate(wrapper, target)["gate_passed"] is False
        except ValueError:
            # Execution/validation gates also block the existing result owner.
            assert gate in {"execution_complete", "analysis_validated"}


@pytest.mark.parametrize("field", ["primary_exposure", "target_outcome", "population", "adjustment", "unit", "definition"])
def test_estimand_identity_cannot_hide_changed_scientific_coordinates(tmp_path, field):
    from easyicu.webserver.research_multi_plan import estimand_identity

    _, _, plan, context, _ = completed_source(tmp_path)
    original = estimand_identity(plan, context)
    if field == "population":
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"inclusion_criteria": ["Synthetic adults only"]})})
    elif field == "adjustment":
        primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
        requirement = primary.model_requirements[0].model_copy(update={"covariates": []})
        changed = primary.model_copy(update={"model_requirements": [requirement]})
        plan = plan.model_copy(update={"steps": [changed if step == primary else step for step in plan.steps]})
    elif field in {"unit", "definition"}:
        variable = context.variable(context.primary_exposure)
        update = {"unit": "synthetic_changed_unit"} if field == "unit" else {"source_concept": "synthetic_other_definition"}
        context = context.model_copy(update={"variables": [item.model_copy(update=update) if item == variable else item
                                                           for item in context.variables]})
    else:
        context = context.model_copy(update={field: "changed_synthetic_variable"})
    assert estimand_identity(plan, context) != original


def test_native_replication_reads_verified_aggregate_and_coefficient_se(tmp_path, monkeypatch):
    from easyicu.databases import research_plan_requirements
    from easyicu.research_agent.replication.metrics import read_replication_estimate
    from tests.support.result_append import json_file

    _, run, _, _, _ = completed_source(tmp_path)
    monkeypatch.setattr(research_plan_requirements, "plan_capability_decision",
                        lambda *args: SimpleNamespace(status="exact", substitutions=[]))
    with pytest.raises(ValueError, match="result_gate_failed"):
        read_replication_estimate(run, "synthetic")
    json_file(run / "run_status.json", {"gates": dict(execution_complete=True, evidence_complete=True,
                                                      numeric_verified=True, analysis_validated=True)})
    result = read_replication_estimate(run, "synthetic")
    assert result["gate_passed"] is True
    assert result["standard_error"] > 0
    assert len(result["source_sha256"]) == len(result["estimand_sha256"]) == 64
    with pytest.raises(ValueError, match="replication_database_changed"):
        read_replication_estimate(run, "other_synthetic")
