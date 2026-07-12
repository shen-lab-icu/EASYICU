from __future__ import annotations

from easyicu.research_agent.declared_product_contract import (
    declared_product_contract_findings,
)
from easyicu.research_agent.method_compatibility import (
    detect_forbidden_pattern_usage,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Discover trajectory phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="test", database="test", n_patients=10, n_stays=10
        ),
        variables=[],
    )


def _representation_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="representation",
        intent="Build the agent-selected trajectory representation.",
        method="missingness_aware_rank_preserving_functional_representation",
        expected_outputs=[
            "artifact:trajectory_representation",
            "table:trajectory_membership",
        ],
    )


def test_representation_code_cannot_fit_or_select_clusters():
    code = """
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=3).fit_predict(X)
pd.DataFrame(rows).to_csv(out_dir / 'cluster_selection.csv')
"""

    violations = detect_forbidden_pattern_usage(
        code,
        _context(),
        _representation_step(),
    )

    assert any(item["kind"] == "trajectory_role_scope" for item in violations)


def test_representation_summary_cannot_hide_downstream_products_as_diagnostics():
    findings = declared_product_contract_findings(
        step=_representation_step(),
        step_summary={
            "status": "ok",
            "outputs": {
                "trajectory_representation": "trajectory_representation.parquet",
                "trajectory_membership": "trajectory_membership.csv",
            },
            "diagnostic_files": {
                "cluster_selection": "cluster_selection.csv",
                "cluster_stability": "cluster_stability.csv",
                "cluster_profile": "cluster_profile.csv",
            },
        },
        effect_method_authorized=False,
    )

    assert any(
        finding.detail.get("kind") == "trajectory_role_product_out_of_scope"
        for finding in findings
    )


def test_representation_audits_and_source_companions_remain_allowed():
    findings = declared_product_contract_findings(
        step=_representation_step(),
        step_summary={
            "status": "ok",
            "outputs": {
                "trajectory_representation": "trajectory_representation.parquet",
                "trajectory_membership": "trajectory_membership.csv",
            },
            "diagnostic_files": {
                "feature_quality_audit": "feature_quality_audit.csv",
                "input_reconciliation": "input_reconciliation.csv",
            },
        },
        effect_method_authorized=False,
    )

    assert not any(finding.severity == "error" for finding in findings)
