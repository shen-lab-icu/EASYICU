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
