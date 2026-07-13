from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.research_agent.declared_product_contract import (
    authorize_declared_figure_product_slots,
    bind_declared_figure_products,
    declared_product_contract_findings,
    read_digest_bound_artifact_snapshot,
)
from easyicu.research_agent.plan_utils import (
    _step_contract_findings,
    effect_output_authorized,
)
from easyicu.research_agent.schema import AnalysisStep


SEALED_DISTRIBUTION_REPAIR = (
    "distribution_availability_publication_bundle_from_parent_outputs_v1"
)
SEALED_IMPLEMENTATION_DIGEST = "a" * 64
SEALED_PARENT_DIGESTS = {
    "step_summary.json": "1" * 64,
    "planned_distribution.csv": "2" * 64,
    "planned_measurement_audit.csv": "3" * 64,
}


def _authorized_slots(products: list[str]) -> dict[str, str]:
    slots: dict[str, str] = {}
    for product in products:
        if product.endswith("_distribution"):
            slots[product] = "distribution"
        elif product.endswith("_availability"):
            slots[product] = "availability"
    return slots


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
            "output_files": {"table:association_estimates": "association_estimates.csv"}
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
    step = _step(
        method="adjusted_logistic_regression with prespecified covariates",
        outputs=["statistic:adjusted_or"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={"adjusted_or": 1.4},
        effect_method_authorized=effect_output_authorized(step),
    )
    assert effect_output_authorized(step) is True
    assert findings == []


def test_inferred_effect_family_does_not_authorize_non_effect_method_output():
    step = _step(outputs=["table:cohort_summary"])
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "analysis_family": "association_study",
            "output_files": {"table:cohort_summary": "cohort_summary.csv"},
            "nested_results": {"risk_ratio_vs_reference": 1.8},
        },
    )

    assert effect_output_authorized(step) is False
    assert "unauthorized_effect_product" in _kinds(findings)


def test_effect_method_without_declared_effect_product_cannot_smuggle_effect():
    step = _step(
        method="adjusted_logistic_regression",
        outputs=["table:cohort_summary"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "output_files": {"table:cohort_summary": "cohort_summary.csv"},
            "nested_results": {"risk_ratio": 1.7},
        },
    )

    assert effect_output_authorized(step) is False
    assert "unauthorized_effect_product" in _kinds(findings)


def test_non_effect_hypothesis_test_p_value_is_not_misclassified_as_effect():
    step = _step(
        method="ordinal_stratified_descriptive_analysis",
        outputs=["statistic:trend_p_value"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={"trend_p_value": 0.03},
        effect_method_authorized=effect_output_authorized(step),
    )

    assert effect_output_authorized(step) is False
    assert "unauthorized_effect_product" not in _kinds(findings)


def test_plan_utils_integration_fails_closed():
    findings = _step_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={"status": "ok", "output_files": []},
    )
    assert "declared_product_missing" in _kinds(findings)


def test_planner_figure_roles_bind_to_one_contracted_multi_panel_bundle(tmp_path):
    (tmp_path / "planned_bundle.png").write_bytes(b"verified-renderer-output")
    (tmp_path / "planned_bundle.figure_contract.json").write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "A",
                        "role": "descriptive_result",
                        "metadata": {"planner_product_slots": ["distribution"]},
                    },
                    {
                        "panel_id": "B",
                        "role": "data_quality",
                        "metadata": {"planner_product_slots": ["availability"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "step_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "rendering_only": True,
                "figure_path": "planned_bundle.png",
                "figure_contract": "planned_bundle.figure_contract.json",
                "output_files": {"figure:publication_figure": "planned_bundle.png"},
            }
        ),
        encoding="utf-8",
    )
    products = ["figure:planned_distribution", "figure:planned_availability"]

    assert bind_declared_figure_products(
        out_dir=tmp_path,
        declared_products=products,
        authorized_product_slots=_authorized_slots(products),
        renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
        renderer_implementation_sha256=SEALED_IMPLEMENTATION_DIGEST,
        renderer_parent_digests=SEALED_PARENT_DIGESTS,
    )
    summary = json.loads((tmp_path / "step_summary.json").read_text("utf-8"))
    assert summary["planner_bound_figure_products"] == products
    assert summary["output_files"]["figure:planned_distribution"] == (
        "planned_bundle.png"
    )
    assert summary["output_files"]["figure:planned_availability"] == (
        "planned_bundle.png"
    )
    assert (
        declared_product_contract_findings(
            step=_step(outputs=products),
            step_summary=summary,
            effect_method_authorized=False,
            out_dir=tmp_path,
        )
        == []
    )


def test_planner_roles_do_not_overclaim_single_panel_bundle(tmp_path):
    (tmp_path / "planned_bundle.png").write_bytes(b"verified-renderer-output")
    (tmp_path / "planned_bundle.figure_contract.json").write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "A",
                        "metadata": {"planner_product_slots": ["distribution"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    original = {
        "rendering_only": True,
        "figure_path": "planned_bundle.png",
        "figure_contract": "planned_bundle.figure_contract.json",
    }
    (tmp_path / "step_summary.json").write_text(json.dumps(original), encoding="utf-8")

    products = [
        "figure:first_distribution",
        "figure:first_availability",
    ]
    with pytest.raises(ValueError, match="not anchored"):
        bind_declared_figure_products(
            out_dir=tmp_path,
            declared_products=products,
            authorized_product_slots=_authorized_slots(products),
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            renderer_implementation_sha256=SEALED_IMPLEMENTATION_DIGEST,
            renderer_parent_digests=SEALED_PARENT_DIGESTS,
        )
    assert json.loads((tmp_path / "step_summary.json").read_text("utf-8")) == original


@pytest.mark.parametrize("figure_path", ["../outside.png", "/tmp/outside.png"])
def test_planner_figure_binding_rejects_path_escape(tmp_path, figure_path):
    (tmp_path / "planned_bundle.figure_contract.json").write_text(
        json.dumps({"panels": [{"panel_id": "A"}]}),
        encoding="utf-8",
    )
    original = {
        "rendering_only": True,
        "figure_path": figure_path,
        "figure_contract": "planned_bundle.figure_contract.json",
    }
    (tmp_path / "step_summary.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="outside STEP_OUT_DIR"):
        bind_declared_figure_products(
            out_dir=tmp_path,
            declared_products=["figure:planned_distribution"],
            authorized_product_slots={"figure:planned_distribution": "distribution"},
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            renderer_implementation_sha256=SEALED_IMPLEMENTATION_DIGEST,
            renderer_parent_digests=SEALED_PARENT_DIGESTS,
        )
    assert json.loads((tmp_path / "step_summary.json").read_text("utf-8")) == original


def test_planner_figure_binding_rejects_output_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.png"
    outside.write_bytes(b"not-owned-by-step")
    (tmp_path / "planned_bundle.png").symlink_to(outside)
    (tmp_path / "planned_bundle.figure_contract.json").write_text(
        json.dumps({"panels": [{"panel_id": "A"}]}),
        encoding="utf-8",
    )
    original = {
        "rendering_only": True,
        "figure_path": "planned_bundle.png",
        "figure_contract": "planned_bundle.figure_contract.json",
    }
    (tmp_path / "step_summary.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="outside STEP_OUT_DIR"):
        bind_declared_figure_products(
            out_dir=tmp_path,
            declared_products=["figure:planned_distribution"],
            authorized_product_slots={"figure:planned_distribution": "distribution"},
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            renderer_implementation_sha256=SEALED_IMPLEMENTATION_DIGEST,
            renderer_parent_digests=SEALED_PARENT_DIGESTS,
        )
    assert json.loads((tmp_path / "step_summary.json").read_text("utf-8")) == original


def test_distribution_bundle_cannot_launder_unrelated_figure_roles(tmp_path):
    (tmp_path / "planned_bundle.png").write_bytes(b"verified-renderer-output")
    (tmp_path / "planned_bundle.figure_contract.json").write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "A",
                        "metadata": {"planner_product_slots": ["distribution"]},
                    },
                    {
                        "panel_id": "B",
                        "metadata": {"planner_product_slots": ["availability"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    original = {
        "rendering_only": True,
        "figure_path": "planned_bundle.png",
        "figure_contract": "planned_bundle.figure_contract.json",
    }
    (tmp_path / "step_summary.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="host product-slot authorization"):
        bind_declared_figure_products(
            out_dir=tmp_path,
            declared_products=[
                "figure:kaplan_meier_curve",
                "figure:adjusted_forest_plot",
            ],
            authorized_product_slots={},
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            renderer_implementation_sha256=SEALED_IMPLEMENTATION_DIGEST,
            renderer_parent_digests=SEALED_PARENT_DIGESTS,
        )
    assert json.loads((tmp_path / "step_summary.json").read_text("utf-8")) == original


@pytest.mark.parametrize(
    "laundered_product",
    [
        "figure:kaplan_meier_curve_distribution",
        "figure:kaplan_meier_curve_by_stage_distribution",
        "figure:roc_curve_distribution",
        "figure:roc_curve_smoothed_distribution",
        "figure:adjusted_forest_availability",
        "figure:adjusted_forest_plot_by_subgroup_availability",
    ],
)
def test_host_slot_authorization_rejects_nested_display_role_even_when_planner_anchored(
    laundered_product,
):
    with pytest.raises(ValueError, match="nests an incompatible display archetype"):
        authorize_declared_figure_product_slots(
            declared_products=[laundered_product],
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            planner_parent_anchors=[
                laundered_product.replace("figure:", "table:"),
            ],
        )


def test_host_slot_authorization_requires_planner_parent_subject():
    with pytest.raises(ValueError, match="not anchored to a verified"):
        authorize_declared_figure_product_slots(
            declared_products=["figure:unplanned_distribution"],
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            planner_parent_anchors=["table:planned_distribution"],
        )


def test_host_slot_authorization_accepts_parent_anchored_subject():
    assert authorize_declared_figure_product_slots(
        declared_products=[
            "figure:planned_distribution",
            "figure:planned_availability",
        ],
        renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
        planner_parent_anchors=[
            "table:planned_distribution",
            "table:planned_measurement_audit",
        ],
    ) == {
        "figure:planned_distribution": "distribution",
        "figure:planned_availability": "availability",
    }


def test_digest_bound_snapshot_parses_the_verified_bytes_after_path_mutation(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    summary = parent / "step_summary.json"
    table = parent / "planned_distribution.csv"
    summary.write_bytes(b'{"method":"descriptive"}')
    table.write_bytes(b"category,n\nA,2\n")
    seal = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (summary, table)
    }

    snapshot = read_digest_bound_artifact_snapshot(
        parent_out=parent,
        artifact_digests=seal,
    )
    table.write_bytes(b"category,n\nA,999\n")

    assert snapshot["planned_distribution.csv"] == b"category,n\nA,2\n"
    with pytest.raises(ValueError, match="authorized digest"):
        read_digest_bound_artifact_snapshot(
            parent_out=parent,
            artifact_digests=seal,
        )


@pytest.mark.parametrize("reported_status", ["fail_closed", "failed_closed"])
def test_reported_fail_closed_status_fails_outer_step_contract(reported_status):
    from easyicu.research_agent.pipeline_execute import (
        _step_status_from_contract_findings,
    )

    step = _step(outputs=["table:summary"])
    summary = {
        "status": reported_status,
        "output_files": {"table:summary": "summary.csv"},
    }

    declared_findings = declared_product_contract_findings(
        step=step,
        step_summary=summary,
        effect_method_authorized=False,
    )
    contract_findings = _step_contract_findings(
        step=step,
        step_summary=summary,
    )

    assert declared_findings == []
    assert any(
        finding.severity == "error"
        and finding.detail.get("reported_status") == reported_status
        for finding in contract_findings
    )
    assert (
        _step_status_from_contract_findings(
            contract_findings=contract_findings,
            figure_source_findings=[],
            stat_findings=[],
        )
        == "contract_failed"
    )
