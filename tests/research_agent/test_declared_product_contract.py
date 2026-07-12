from __future__ import annotations

from easyicu.research_agent.declared_product_contract import (
    declared_product_contract_findings,
)
from easyicu.research_agent.plan_utils import _step_contract_findings
from easyicu.research_agent.schema import AnalysisStep


def _step(*, method: str = "descriptive_summary", outputs: list[str]) -> AnalysisStep:
    return AnalysisStep(
        step_id="03_analysis",
        intent="Produce only the planned products.",
        method=method,
        expected_outputs=outputs,
    )


def _kinds(findings) -> set[str]:
    return {
        finding.detail["kind"]
        for finding in findings
        if finding.detail and "kind" in finding.detail
    }


def test_declared_file_and_statistic_products_must_be_realised_exactly():
    step = _step(outputs=["table:summary", "statistic:cohort_n"])

    valid = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "cohort_n": 120,
            "output_files": {"table:summary": "summary.csv"},
        },
        effect_method_authorized=False,
    )
    assert valid == []

    missing = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "cohort_n": 120,
            "output_files": {"table:different_summary": "different_summary.csv"},
        },
        effect_method_authorized=False,
    )
    assert "declared_product_missing" in _kinds(missing)
    finding = next(
        item for item in missing if item.detail["kind"] == "declared_product_missing"
    )
    assert finding.detail["missing_products"] == ["table:summary"]


def test_file_list_stem_can_realise_typed_product_but_companion_cannot():
    step = _step(outputs=["artifact:analysis_cohort"])
    assert (
        declared_product_contract_findings(
            step=step,
            step_summary={"output_files": ["analysis_cohort.parquet"]},
            effect_method_authorized=False,
        )
        == []
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": ["analysis_cohort_diagnostic.csv"]},
        effect_method_authorized=False,
    )
    assert "declared_product_missing" in _kinds(findings)


def test_typed_file_role_requires_a_file_like_registration():
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={"output_files": {"table:summary": "not produced"}},
        effect_method_authorized=False,
    )
    assert "declared_product_missing" in _kinds(findings)


def test_declared_file_must_exist_under_step_output_dir(tmp_path):
    step = _step(outputs=["table:summary"])
    missing = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {"table:summary": "summary.csv"}},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )
    assert "declared_product_missing" in _kinds(missing)

    (tmp_path / "summary.csv").write_text("group,n\nall,1\n", encoding="utf-8")
    valid = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {"table:summary": "summary.csv"}},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )
    assert valid == []


def test_real_execution_cannot_evade_registry_by_omitting_output_files(tmp_path):
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={"status": "ok", "n": 10},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_test_product_may_be_realised_by_a_machine_readable_table():
    findings = declared_product_contract_findings(
        step=_step(outputs=["test:ordered_trend"]),
        step_summary={"output_files": {"test:ordered_trend": "ordered_trend.csv"}},
        effect_method_authorized=False,
    )
    assert findings == []


def test_nonfigure_step_cannot_emit_publication_bundle():
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:distribution"]),
        step_summary={
            "output_files": {"table:distribution": "distribution.csv"},
            "figure": {
                "output_files": {
                    "png": "distribution.png",
                    "svg": "distribution.svg",
                    "pdf": "distribution.pdf",
                }
            },
        },
        effect_method_authorized=False,
    )
    assert "undeclared_figure_bundle" in _kinds(findings)


def test_source_data_and_single_diagnostic_companion_do_not_widen_scope():
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:distribution"]),
        step_summary={
            "output_files": {
                "table:distribution": "distribution.csv",
                "source_data": "distribution_source_data.csv",
                "diagnostic": "distribution_diagnostic.png",
                "audit_log": "distribution_audit.json",
            }
        },
        effect_method_authorized=False,
    )
    assert findings == []


def test_non_effect_method_cannot_declare_or_register_effect_products():
    declared = declared_product_contract_findings(
        step=_step(outputs=["table:association_estimates"]),
        step_summary={
            "output_files": {
                "table:association_estimates": "association_estimates.csv"
            }
        },
        effect_method_authorized=False,
    )
    assert "unauthorized_effect_product" in _kinds(declared)

    registered = declared_product_contract_findings(
        step=_step(outputs=["table:distribution"]),
        step_summary={
            "output_files": {
                "table:distribution": "distribution.csv",
                "table:risk_ratio": "risk_ratio.csv",
            }
        },
        effect_method_authorized=False,
    )
    assert "unauthorized_effect_product" in _kinds(registered)


def test_nested_effect_estimate_is_scientific_output_not_diagnostic_companion():
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:distribution"]),
        step_summary={
            "output_files": {"table:distribution": "distribution.csv"},
            "strata": [{"group": "high", "risk_ratio_vs_reference": 1.8}],
        },
        effect_method_authorized=False,
    )
    assert "unauthorized_effect_product" in _kinds(findings)


def test_effect_method_owner_may_realise_its_declared_effect():
    findings = declared_product_contract_findings(
        step=_step(
            method="adjusted_logistic_regression",
            outputs=["statistic:adjusted_or"],
        ),
        step_summary={"adjusted_or": 1.4},
        effect_method_authorized=True,
    )
    assert findings == []


def test_plan_utils_integration_fails_closed():
    findings = _step_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={"status": "ok", "output_files": []},
    )
    assert "declared_product_missing" in _kinds(findings)
