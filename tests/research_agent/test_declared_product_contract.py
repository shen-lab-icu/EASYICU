from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.declared_product import (
    authorize_declared_figure_product_slots,
    bind_declared_figure_products,
    declared_product_contract_findings,
    effect_bearing_product,
    effect_measure_family,
    read_digest_bound_artifact_snapshot,
    typed_product,
    typed_product_binding_contract,
)
from easyicu.research_agent.plan_utils import (
    _effect_figure_source_authorized,
    _step_contract_findings,
    effect_output_authorized,
)
from easyicu.research_agent.schema import AnalysisStep

SEALED_DISTRIBUTION_REPAIR = (
    "distribution_availability_publication_bundle_from_parent_outputs_v1"
)


def test_cohort_typed_product_is_canonical_dataset_identity():
    assert typed_product("cohort:locked_analysis_rows.parquet") == (
        "dataset",
        "locked_analysis_rows",
    )


def test_cohort_typed_product_is_realised_by_its_tabular_dataset(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "cohort_locked_analysis_rows.parquet").write_bytes(
        b"parquet-placeholder"
    )
    step = AnalysisStep(
        step_id="lock_analysis_rows",
        intent="Materialize the Planner-defined analysis cohort.",
        expected_outputs=["cohort:locked_analysis_rows"],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "output_files": [
                {
                    "kind": "cohort",
                    "name": "cohort:locked_analysis_rows",
                    "relative_path": "cohort_locked_analysis_rows.parquet",
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert not [
        finding
        for finding in findings
        if finding.detail.get("kind") == "declared_product_missing"
    ]


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("input_universe_reconciliation.csv", "observed_n,expected_n\n94458,94458\n"),
        (
            "input_universe_reconciliation.json",
            '{"observed_n":94458,"expected_n":94458}',
        ),
    ],
)
def test_explicit_audit_product_is_realised_by_compatible_file(
    tmp_path, filename, payload
):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / filename).write_text(payload, encoding="utf-8")
    step = AnalysisStep(
        step_id="cohort_flow",
        intent="Reconcile the locked input universe.",
        expected_outputs=["audit:input_universe_reconciliation"],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "output_files": [
                {
                    "kind": "audit",
                    "name": "audit:input_universe_reconciliation",
                    "path": filename,
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" not in _kinds(findings)


@pytest.mark.parametrize("filename", ["input_universe_reconciliation.pkl", "plot.png"])
def test_audit_product_rejects_incompatible_physical_file(tmp_path, filename):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / filename).write_bytes(b"not-an-audit-table-or-log")
    step = AnalysisStep(
        step_id="cohort_flow",
        intent="Reconcile the locked input universe.",
        expected_outputs=["audit:input_universe_reconciliation"],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "output_files": [
                {
                    "kind": "audit",
                    "name": "audit:input_universe_reconciliation",
                    "path": filename,
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" in _kinds(findings)


@pytest.mark.parametrize(
    "output_files",
    [
        ["input_universe_reconciliation.csv"],
        [
            {
                "kind": "table",
                "name": "audit:input_universe_reconciliation",
                "path": "input_universe_reconciliation.csv",
            }
        ],
    ],
)
def test_audit_product_requires_exact_structured_audit_descriptor(
    tmp_path, output_files
):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "input_universe_reconciliation.csv").write_text(
        "observed_n,expected_n\n94458,94458\n",
        encoding="utf-8",
    )
    step = AnalysisStep(
        step_id="cohort_flow",
        intent="Reconcile the locked input universe.",
        expected_outputs=["audit:input_universe_reconciliation"],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"status": "completed", "output_files": output_files},
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" in _kinds(findings)


def _assignment_cohort(tmp_path, *, stay_ids=(101,)):
    path = tmp_path / "cohort.csv"
    path.write_text(
        "stay_id,value\n"
        + "".join(f"{stay_id},{index}\n" for index, stay_id in enumerate(stay_ids)),
        encoding="utf-8",
    )
    return path


def test_assignment_binding_contract_maps_each_declared_model_to_exact_column(
    tmp_path,
):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text(
        "row_index,propensity_source_aware,propensity_complete_case,other_numeric\n"
        "0,0.2,0.3,9\n",
        encoding="utf-8",
    )
    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                },
                {
                    "model_id": "assignment_complete_case",
                    "analysis_set": "complete_case",
                    "fit_status": "fitted",
                },
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path),
    )

    assert contract is not None
    assert [model["propensity_score_column"] for model in contract["models"]] == [
        "propensity_source_aware",
        "propensity_complete_case",
    ]
    assert contract["row_count"] == 1
    assert len(contract["row_identity_sha256"]) == 64
    assert len(contract["models"][0]["analysis_set_identity_sha256"]) == 64


def test_assignment_binding_contract_does_not_fallback_to_arbitrary_numeric_column(
    tmp_path,
):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text("row_index,estimate,score\n0,0.2,0.3\n", encoding="utf-8")
    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                }
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path),
    )

    assert contract is None


def test_assignment_binding_contract_does_not_trust_same_name_summary_mapping(
    tmp_path,
):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text("row_index,unrelated\n0,0.2\n", encoding="utf-8")

    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={"assignment_model": {"foo": "bar"}},
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path),
    )

    assert contract is None


@pytest.mark.parametrize("score", ["nan", "inf", "-0.01", "1.01"])
def test_assignment_binding_contract_rejects_invalid_propensity_values(tmp_path, score):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text(
        f"row_index,propensity_source_aware\n0,{score}\n",
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                    "n": 1,
                }
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path),
    )

    assert contract is None


def test_assignment_binding_contract_checks_row_identity_and_declared_n(tmp_path):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text(
        "row_index,propensity_source_aware\n0,0.2\n0,0.3\n",
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                    "n": 1,
                }
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path, stay_ids=(101, 102)),
    )

    assert contract is None


def test_assignment_binding_contract_rejects_identity_rows_from_another_cohort(
    tmp_path,
):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text(
        "stay_id,propensity_source_aware\n999001,0.2\n999002,0.3\n",
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                    "n": 2,
                }
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path, stay_ids=(101, 102)),
    )

    assert contract is None


def test_assignment_binding_contract_accepts_exact_cohort_identity_and_order(
    tmp_path,
):
    artifact = tmp_path / "assignment_model.csv"
    artifact.write_text(
        "stay_id,propensity_source_aware\n101,0.2\n102,0.3\n",
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="assignment_model",
        step_summary={
            "assignment_models": [
                {
                    "model_id": "assignment_source_aware",
                    "analysis_set": "source_aware",
                    "fit_status": "fitted",
                    "n": 2,
                }
            ]
        },
        artifact_path=artifact,
        authoritative_cohort_path=_assignment_cohort(tmp_path, stay_ids=(101, 102)),
    )

    assert contract is not None
    assert contract["row_identity_column"] == "stay_id"
    assert contract["row_count"] == 2
    assert contract["models"][0]["analysis_set_n"] == 2


def test_declared_diagnostic_rejects_not_computable_placeholder():
    step = AnalysisStep(
        step_id="diagnostics",
        intent="Run the Planner-owned diagnostics.",
        expected_outputs=["artifact:balance_diagnostics"],
        method="diagnostic_analysis",
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "diagnostic_status": "not_computable",
            "skipped_reason": "required model binding is ambiguous",
            "output_files": {
                "artifact:balance_diagnostics": "balance_diagnostics.json"
            },
        },
        effect_method_authorized=False,
    )

    finding = next(
        item
        for item in findings
        if item.detail
        and item.detail.get("kind") == "declared_diagnostic_not_completed"
    )
    assert finding.detail["diagnostic_status"] == "not_computable"
    assert finding.detail["skipped_reason"] == "required model binding is ambiguous"


def test_primary_exposure_binding_contract_canonicalizes_one_declared_column(
    tmp_path,
):
    artifact = tmp_path / "primary_exposure.csv"
    artifact.write_text("stay_id,treatment\n1,0\n", encoding="utf-8")
    contract = typed_product_binding_contract(
        product_name="primary_exposure_definition",
        step_summary={
            "primary_exposure_definition": {
                "column": "treatment",
                "window": "baseline",
            }
        },
        artifact_path=artifact,
    )

    assert contract is not None
    assert contract["column"] == "treatment"
    assert contract["executable_column"] == "treatment"
    assert contract["exposure_column"] == "treatment"
    assert contract["authoritative_primary_exposure"] == "treatment"
    assert contract["window"] == "baseline"
    assert contract["time_window"] == "baseline"


def test_exposure_definition_contract_joins_summary_artifact_and_cohort(tmp_path):
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("stay_id,treatment\n1,0\n2,1\n", encoding="utf-8")
    artifact = tmp_path / "exposure_definition.json"
    artifact.write_text(
        json.dumps(
            {
                "authoritative_exposure": "treatment",
                "derived_exposure": "treatment_any",
                "rule": "Use the sealed treatment indicator without recoding.",
                "locked_cohort_n": 2,
                "usable_variation": True,
                "weighted_association_feasibility": "eligible",
            }
        ),
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="exposure_definition",
        step_summary={
            "authoritative_exposure": "treatment",
            "derived_exposure": "treatment_any",
            "derived_exposure_rule": (
                "Use the sealed treatment indicator without recoding."
            ),
            "locked_cohort_n": 2,
        },
        artifact_path=artifact,
        authoritative_cohort_path=cohort,
    )

    assert contract is not None
    assert contract["executable_column"] == "treatment"
    assert contract["exposure_column"] == "treatment"
    assert contract["derived_exposure"] == "treatment_any"
    assert contract["usable_variation"] is True


def test_exposure_definition_contract_refuses_cross_surface_drift(tmp_path):
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("stay_id,treatment\n1,0\n", encoding="utf-8")
    artifact = tmp_path / "exposure_definition.json"
    artifact.write_text(
        json.dumps(
            {
                "authoritative_exposure": "different_treatment",
                "derived_exposure": "treatment_any",
                "rule": "sealed rule",
                "locked_cohort_n": 1,
            }
        ),
        encoding="utf-8",
    )

    assert (
        typed_product_binding_contract(
            product_name="exposure_definition",
            step_summary={
                "authoritative_exposure": "treatment",
                "derived_exposure": "treatment_any",
                "derived_exposure_rule": "sealed rule",
                "locked_cohort_n": 1,
            },
            artifact_path=artifact,
            authoritative_cohort_path=cohort,
        )
        is None
    )


def test_primary_exposure_binding_contract_does_not_resolve_conflicting_columns(
    tmp_path,
):
    artifact = tmp_path / "primary_exposure.csv"
    artifact.write_text("stay_id,treatment_a,treatment_b\n1,0,1\n", encoding="utf-8")
    contract = typed_product_binding_contract(
        product_name="primary_exposure_definition",
        step_summary={
            "primary_exposure_definition": {
                "column": "treatment_a",
                "executable_column": "treatment_b",
            }
        },
        artifact_path=artifact,
    )

    assert contract is None


def test_primary_exposure_binding_contract_does_not_resolve_conflicting_windows(
    tmp_path,
):
    artifact = tmp_path / "primary_exposure.csv"
    artifact.write_text("stay_id,treatment\n1,0\n", encoding="utf-8")
    contract = typed_product_binding_contract(
        product_name="primary_exposure_definition",
        step_summary={
            "primary_exposure_definition": {
                "column": "treatment",
                "window": "baseline",
                "time_window": "follow_up",
            }
        },
        artifact_path=artifact,
    )

    assert contract is None


def test_confounder_binding_contract_uses_exact_declared_covariate_field(tmp_path):
    artifact = tmp_path / "prespecified_confounder_set.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "prespecified_confounder_set",
                "selected_covariates": ["age", "severity_group"],
                "ordinal_encoding": {
                    "severity_group": {"modelling_choice": "ordered categorical"}
                },
            }
        ),
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="prespecified_confounder_set",
        step_summary={},
        artifact_path=artifact,
    )

    assert contract == {
        "covariates": ["age", "severity_group"],
        "source_field": "selected_covariates",
        "ordinal_encoding": {
            "severity_group": {"modelling_choice": "ordered categorical"}
        },
    }


def test_confounder_binding_contract_does_not_scan_arbitrary_lists(tmp_path):
    artifact = tmp_path / "prespecified_confounder_set.json"
    artifact.write_text(
        json.dumps({"candidate_columns": ["age", "first_numeric_column"]}),
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="prespecified_confounder_set",
        step_summary={},
        artifact_path=artifact,
    )

    assert contract is None


def test_confounder_binding_contract_does_not_trust_same_name_summary_mapping(
    tmp_path,
):
    artifact = tmp_path / "prespecified_confounder_set.json"
    artifact.write_text(
        json.dumps({"candidate_columns": ["age", "first_numeric_column"]}),
        encoding="utf-8",
    )

    contract = typed_product_binding_contract(
        product_name="prespecified_confounder_set",
        step_summary={"prespecified_confounder_set": {"covariates": ["age"]}},
        artifact_path=artifact,
    )

    assert contract is None


SEALED_ABSOLUTE_RISK_REPAIR = "absolute_risk_incidence_prevalence_publication_bundle_v1"
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


@pytest.mark.parametrize(
    "value",
    [
        120,
        10**1000,
    ],
)
def test_closed_inline_statistic_descriptor_registers_without_a_path(value):
    findings = declared_product_contract_findings(
        step=_step(outputs=["statistic:sample_size"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "statistic",
                    "name": "sample_size",
                    "role": "sample_size",
                    "value": value,
                },
            ],
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" not in _kinds(findings)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not computed",
        False,
        {},
        [],
        {"n": 120},
        {"n": 120, "status": "failed"},
        {"n": 120, "passed": True},
        {"n": 120, "nested": {"estimate": 1.5}},
        {"n": 120, "values": [1, 2]},
        {"n": 120, "missing": None},
        {"universe_n": 120, "analysis_n": 96},
        {"error_code": 500, "retry_count": 2},
        float("nan"),
        float("inf"),
        {"n": 120, "estimate": float("nan")},
        {"n": 120, "estimate": float("inf")},
    ],
)
def test_inline_statistic_requires_a_closed_finite_numeric_value(value):
    findings = declared_product_contract_findings(
        step=_step(outputs=["statistic:sample_size"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "statistic",
                    "name": "sample_size",
                    "role": "sample_size",
                    "value": value,
                },
            ],
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" in _kinds(findings)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"kind": "statistic", "name": "sample_size", "value": 120},
        {"kind": "statistic", "statistic:sample_size": 120},
        {"name": "sample_size", "statistic:sample_size": 120},
        {"value": 120, "statistic:sample_size": 120},
        {
            "kind": "statistic",
            "name": "sample_size",
            "role": "different_statistic",
            "value": 120,
        },
        {
            "kind": "statistic",
            "name": "sample_size",
            "role": "sample_size",
            "value": {"sample_size": 120, "other": 1},
            "status": "ok",
        },
        {
            "kind": "statistic",
            "name": "sample_size",
            "role": "sample_size",
            "value": 120,
            "error": None,
        },
        {
            "kind": "statistic",
            "name": "sample_size",
            "role": "sample_size",
            "value": {"universe_n": 120, "analysis_n": 96},
            "statistic:sample_size": 120,
        },
    ],
)
def test_inline_statistic_requires_an_exact_role_bound_envelope(descriptor):
    findings = declared_product_contract_findings(
        step=_step(outputs=["statistic:sample_size"]),
        step_summary={
            "status": "ok",
            "output_files": [descriptor],
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_legacy_single_key_inline_statistic_registry_remains_supported():
    findings = declared_product_contract_findings(
        step=_step(outputs=["statistic:sample_size"]),
        step_summary={
            "status": "ok",
            "output_files": {"statistic:sample_size": 120},
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" not in _kinds(findings)


def test_pathless_log_descriptor_is_not_registered_from_inline_metadata():
    findings = declared_product_contract_findings(
        step=_step(outputs=["log:validation_trace"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "log",
                    "name": "validation_trace",
                    "role": "validation_trace",
                    "value": {"passed": True},
                },
            ],
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_file_descriptor_still_requires_a_compatible_path():
    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {"kind": "table", "name": "summary", "value": {"rows": 4}},
            ],
        },
        effect_method_authorized=False,
    )

    assert "declared_product_missing" in _kinds(findings)


@pytest.mark.parametrize(
    ("typed_kind", "filename"),
    [
        ("table:summary", "summary.csv"),
        ("dataset:analysis_rows", "analysis_rows.parquet"),
    ],
)
def test_typed_kind_shorthand_registers_only_an_existing_compatible_file(
    tmp_path, typed_kind, filename
):
    (tmp_path / filename).write_bytes(b"placeholder")

    findings = declared_product_contract_findings(
        step=_step(outputs=[typed_kind]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": typed_kind,
                    "path": filename,
                    "role": "supporting",
                    "metadata": {"description": "generic product"},
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    assert "declared_product_missing" not in _kinds(findings)


def test_typed_kind_shorthand_accepts_consistent_identity_and_path_aliases(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "03_analysis" / "outputs"
    out_dir.mkdir(parents=True)
    filename = "summary.csv"
    output_path = out_dir / filename
    output_path.write_text("value\n1\n", encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "table:summary",
                    "name": "table:summary.csv",
                    "product_type": "table",
                    "path": str(output_path),
                    "relative_path": "steps/03_analysis/outputs/summary.csv",
                    "filename": filename,
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" not in _kinds(findings)


def test_typed_kind_shorthand_accepts_absolute_path_with_relative_output_dir(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    out_dir = Path("outputs")
    out_dir.mkdir()
    output_path = (tmp_path / out_dir / "summary.csv").absolute()
    output_path.write_text("value\n1\n", encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [{"kind": "table:summary", "path": str(output_path)}],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" not in _kinds(findings)


def test_typed_kind_shorthand_authorizes_only_its_descriptor_identity(tmp_path):
    filename = "physical_name.csv"
    (tmp_path / filename).write_text("value\n1\n", encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:logical_summary", "table:physical_name"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "table:logical_summary",
                    "path": filename,
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    missing = next(
        finding
        for finding in findings
        if finding.detail.get("kind") == "declared_product_missing"
    )
    assert missing.detail["missing_products"] == ["table:physical_name"]


@pytest.mark.parametrize(
    "descriptor",
    [
        {"kind": "table:summary", "name": "different", "path": "summary.csv"},
        {
            "kind": "table:summary",
            "name": "dataset:summary",
            "path": "summary.csv",
        },
        {
            "kind": "table:summary",
            "product_type": "dataset",
            "path": "summary.csv",
        },
        {"kind": "table:summary"},
        {
            "kind": "table:summary",
            "relative_path": "steps/03_analysis/outputs/summary.csv",
        },
        {"kind": "table:summary", "path": "summary.json"},
        {"kind": "table:summary", "path": "missing.csv"},
        {"kind": "blob:summary", "path": "summary.csv"},
        {"kind": "table:summary:extra", "path": "summary.csv"},
    ],
)
def test_typed_kind_shorthand_rejects_incomplete_or_conflicting_receipts(
    tmp_path, descriptor
):
    (tmp_path / "summary.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text('{"value": 1}', encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={"status": "ok", "output_files": [descriptor]},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_typed_kind_shorthand_rejects_output_root_escape(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (tmp_path / "outside.csv").write_text("value\n1\n", encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [{"kind": "table:summary", "path": "../outside.csv"}],
        },
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_typed_kind_shorthand_rejects_symlinked_output(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "summary.csv").symlink_to(target.name)

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [{"kind": "table:summary", "path": "summary.csv"}],
        },
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_typed_kind_shorthand_rejects_conflicting_path_aliases(tmp_path):
    (tmp_path / "first.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "second.csv").write_text("value\n2\n", encoding="utf-8")

    findings = declared_product_contract_findings(
        step=_step(outputs=["table:summary"]),
        step_summary={
            "status": "ok",
            "output_files": [
                {
                    "kind": "table:summary",
                    "path": "first.csv",
                    "filename": "second.csv",
                }
            ],
        },
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    assert "declared_product_missing" in _kinds(findings)


def test_malformed_path_only_descriptor_has_no_figure_authority():
    findings = declared_product_contract_findings(
        step=_step(outputs=[]),
        step_summary={
            "status": "ok",
            "figure_files": [{"path": "unbound_figure.png"}],
        },
        effect_method_authorized=False,
    )

    assert "undeclared_figure_bundle" not in _kinds(findings)


def test_declared_assignment_model_requires_a_successfully_fitted_model():
    step = _step(
        method="confounder_selection_and_propensity_model",
        outputs=["artifact:assignment_model"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "assignment_models": [],
            "exposure": {
                "resolution": {
                    "status": "not_available",
                    "reason": "typed binding unavailable",
                }
            },
            "output_files": [
                {
                    "kind": "artifact",
                    "name": "assignment_model",
                    "path": "assignment_model.csv",
                }
            ],
        },
        effect_method_authorized=False,
    )

    assert "assignment_model_unfitted" in _kinds(findings)
    finding = next(
        item for item in findings if item.detail["kind"] == "assignment_model_unfitted"
    )
    assert finding.detail["exposure_resolution_status"] == "not_available"


def test_declared_assignment_model_accepts_a_fitted_model():
    step = _step(
        method="confounder_selection_and_propensity_model",
        outputs=["artifact:assignment_model"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "assignment_models": [
                {
                    "model_id": "planner_model",
                    "fit_status": "fitted",
                    "n": 100,
                }
            ],
            "output_files": [
                {
                    "kind": "artifact",
                    "name": "assignment_model",
                    "path": "assignment_model.csv",
                }
            ],
        },
        effect_method_authorized=False,
    )

    assert "assignment_model_unfitted" not in _kinds(findings)


def test_declared_assignment_model_surfaces_eligibility_class_collapse():
    step = _step(
        method="propensity_score_model",
        outputs=["artifact:assignment_model"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "exposure": {
                "event_n": 120,
                "non_event_n": 880,
                "resolution": {"status": "resolved"},
            },
            "assignment_models": [
                {
                    "model_id": "planner_model",
                    "fit_status": "not_fitted",
                    "n": 90,
                    "exposure_event_n": 90,
                    "exposure_non_event_n": 0,
                    "error": "exposure has no variation in the analysis set",
                }
            ],
            "output_files": [
                {
                    "kind": "artifact",
                    "name": "assignment_model",
                    "path": "assignment_model.csv",
                }
            ],
        },
        effect_method_authorized=False,
    )

    finding = next(
        item for item in findings if item.detail["kind"] == "assignment_model_unfitted"
    )
    assert finding.detail["exposure_class_collapse_after_eligibility"] is True
    assert finding.detail["model_diagnostics"] == [
        {
            "model_id": "planner_model",
            "fit_status": "not_fitted",
            "n": 90,
            "exposure_event_n": 90,
            "exposure_non_event_n": 0,
            "error": "exposure has no variation in the analysis set",
        }
    ]
    assert "symmetrically" in finding.detail["repair_constraint"]


@pytest.mark.parametrize("noncanonical_status", ["ok", "converged"])
def test_declared_assignment_model_rejects_noncanonical_success_status(
    noncanonical_status,
):
    step = _step(
        method="propensity_score_model",
        outputs=["artifact:assignment_model"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "assignment_models": [
                {"model_id": "planner_model", "fit_status": noncanonical_status}
            ],
            "output_files": [
                {
                    "kind": "artifact",
                    "name": "assignment_model",
                    "path": "assignment_model.csv",
                }
            ],
        },
        effect_method_authorized=False,
    )

    assert "assignment_model_unfitted" in _kinds(findings)


def test_declared_assignment_model_contract_is_method_name_neutral():
    step = _step(
        method="propensity_score_model",
        outputs=["artifact:assignment_model"],
    )
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "assignment_models": [],
            "output_files": [
                {
                    "kind": "artifact",
                    "name": "assignment_model",
                    "path": "assignment_model.csv",
                }
            ],
        },
        effect_method_authorized=False,
    )

    assert "assignment_model_unfitted" in _kinds(findings)


def test_step_contract_rejects_planned_step_that_reports_skipped():
    findings = _step_contract_findings(
        step=_step(outputs=["table:diagnostics"]),
        step_summary={
            "status": "skipped",
            "error": "required upstream model unavailable",
            "output_files": {"table:diagnostics": "diagnostics.csv"},
        },
    )

    assert any(
        finding.validator == "step_contract"
        and finding.detail
        and finding.detail.get("reported_status") == "skipped"
        for finding in findings
    )


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


def test_output_artifacts_list_registers_existing_typed_file_stems(tmp_path):
    (tmp_path / "primary_exposure_definition.parquet").write_bytes(b"parquet")
    (tmp_path / "exposure_distribution.csv").write_text(
        "group,n\nexposed,1\n", encoding="utf-8"
    )
    (tmp_path / "exposure_source_status.csv").write_text(
        "source_status,count\nevent_present,1\n", encoding="utf-8"
    )
    step = _step(
        outputs=[
            "artifact:primary_exposure_definition",
            "table:exposure_distribution",
            "table:exposure_source_status",
        ]
    )
    summary = {
        "output_files": {
            "primary_exposure_definition.parquet": True,
            "exposure_distribution.csv": True,
            "exposure_source_status.csv": True,
        },
        "output_artifacts": [
            {
                "kind": "artifact",
                "name": "primary_exposure_definition",
                "filename": "primary_exposure_definition.parquet",
            },
            {
                "kind": "table",
                "name": "exposure_distribution",
                "filename": "exposure_distribution.csv",
            },
            {
                "kind": "table",
                "name": "exposure_source_status",
                "filename": "exposure_source_status.csv",
            },
        ],
    }

    assert (
        declared_product_contract_findings(
            step=step,
            step_summary=summary,
            effect_method_authorized=False,
            out_dir=tmp_path,
        )
        == []
    )


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


@pytest.mark.parametrize("status", ["deferred", "non-estimable", "not_estimated"])
def test_noncomputed_effect_status_is_not_an_effect_result(status):
    findings = declared_product_contract_findings(
        step=_step(outputs=["artifact:target_trial_protocol"]),
        step_summary={
            "output_files": {
                "artifact:target_trial_protocol": "target_trial_protocol.json"
            },
            "estimability": {
                "primary_adjusted_or_weighted_contrast": status,
                "reason": "The protocol did not estimate an effect.",
            },
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" not in _kinds(findings)


def test_explicit_false_effect_created_flag_is_not_an_effect_result():
    findings = declared_product_contract_findings(
        step=_step(outputs=["artifact:analysis_cohort"]),
        step_summary={
            "output_files": {"artifact:analysis_cohort": "analysis_cohort.parquet"},
            "output_scope": {"effect_estimates_created": False},
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" not in _kinds(findings)


def test_explicit_true_effect_created_flag_remains_fail_closed():
    findings = declared_product_contract_findings(
        step=_step(outputs=["artifact:analysis_cohort"]),
        step_summary={
            "output_files": {"artifact:analysis_cohort": "analysis_cohort.parquet"},
            "output_scope": {"effect_estimates_created": True},
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


def test_textual_or_numeric_effect_value_remains_effect_bearing():
    for value in (1.4, "OR=1.4"):
        findings = declared_product_contract_findings(
            step=_step(outputs=["artifact:target_trial_protocol"]),
            step_summary={
                "output_files": {
                    "artifact:target_trial_protocol": "target_trial_protocol.json"
                },
                "primary_adjusted_or_weighted_contrast": value,
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


def test_text_only_summary_kind_does_not_grant_effect_output_authority():
    step = _step(
        method="adjusted_logistic_regression",
        outputs=["summary:primary_association"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "summary": {
                "notes": ["Association estimate: OR=1.219 (95% CI 1.116-1.332)."]
            }
        },
    )

    assert effect_output_authorized(step) is False
    assert "unauthorized_effect_product" in _kinds(findings)


def _effect_parent_and_figure_child():
    parent = AnalysisStep(
        step_id="04_primary_association",
        intent="Estimate the prespecified primary association.",
        inputs=["exposure", "outcome"],
        expected_outputs=["table:primary_association"],
        method="logistic_regression",
    )
    child = AnalysisStep(
        step_id="04_primary_association_figure",
        intent="Render the successful direct parent's typed result.",
        inputs=["table:primary_association"],
        expected_outputs=["figure:primary_association_curve"],
        method="visualization",
    )
    record = {
        "step_id": parent.step_id,
        "status": "ok",
        "analysis_request": {"step": parent.model_dump(mode="json")},
        "step_summary": {
            "status": "ok",
            "output_files": {"table:primary_association": "primary_association.csv"},
        },
    }
    return parent, child, record


def _resolved_render_bindings(
    child: AnalysisStep,
    *,
    producer_step_id: str = "04_primary_association",
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(child.inputs or []):
        parsed = typed_product(raw)
        if parsed is None:
            continue
        bindings[str(raw)] = {
            "declared_kind": parsed[0],
            "product": parsed[1],
            "produced_by_step": producer_step_id,
            "evidence_id": f"evidence_{index}",
            "sha256": f"{index + 1:x}" * 64,
        }
    return bindings


@pytest.mark.parametrize(
    "render_method", ["visualization", "publication_figure_generation"]
)
def test_effect_named_figure_requires_successful_typed_effect_parent(render_method):
    parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(update={"method": render_method})
    bindings = _resolved_render_bindings(child)

    assert effect_output_authorized(parent) is True
    assert effect_output_authorized(child) is False
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is True
    )
    findings = _step_contract_findings(
        step=child,
        step_summary={
            "status": "ok",
            "output_files": {
                "figure:primary_association_curve": "primary_association_curve.png"
            },
        },
        completed_step_records=[record],
        resolved_input_bindings=bindings,
    )

    assert "unauthorized_effect_product" not in _kinds(findings)


def test_generic_primary_adjusted_effect_uses_verified_planner_model_roster():
    parent = AnalysisStep(
        step_id="05_primary_adjusted_association",
        intent="Fit the Planner-owned primary adjusted model.",
        method="adjusted_association_models",
        expected_outputs=[
            "table:adjusted_association_estimates",
            "artifact:primary_model_specification",
        ],
        model_requirements=[
            {
                "requirement_id": "primary_death_model",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "exposure",
                "analysis_role": "primary",
                "analysis_set": "complete_case",
                "required_for_step_success": True,
            }
        ],
    )
    child = AnalysisStep(
        step_id="05_primary_adjusted_association_figure",
        intent="Render the Planner-owned primary adjusted result.",
        method="visualization",
        inputs=["table:adjusted_association_estimates"],
        expected_outputs=["figure:primary_adjusted_effect"],
    )
    record = {
        "step_id": parent.step_id,
        "status": "ok",
        "analysis_request": {"step": parent.model_dump(mode="json")},
        "step_summary": {
            "output_files": {
                "table:adjusted_association_estimates": (
                    "adjusted_association_estimates.csv"
                )
            }
        },
    }

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(
                child,
                producer_step_id=parent.step_id,
            ),
        )
        is True
    )
    findings = _step_contract_findings(
        step=child,
        step_summary={
            "status": "ok",
            "output_files": {
                "figure:primary_adjusted_effect": "primary_adjusted_effect.png"
            },
        },
        completed_step_records=[record],
        resolved_input_bindings=_resolved_render_bindings(
            child,
            producer_step_id=parent.step_id,
        ),
    )
    assert "unauthorized_effect_product" not in _kinds(findings)


@pytest.mark.parametrize(
    "case",
    [
        "latest_parent_failed",
        "child_input_mismatch",
        "parent_not_effect_authorized",
        "child_inherits_effect_method",
    ],
)
def test_effect_named_figure_source_authority_fails_closed(case):
    parent, child, record = _effect_parent_and_figure_child()
    records = [record]
    if case == "latest_parent_failed":
        records.append({**record, "status": "contract_failed"})
    elif case == "child_input_mismatch":
        child = child.model_copy(update={"inputs": ["table:cohort_summary"]})
    elif case == "parent_not_effect_authorized":
        parent = parent.model_copy(
            update={
                "method": "descriptive_summary",
                "expected_outputs": ["table:cohort_summary"],
            }
        )
        record = {
            **record,
            "analysis_request": {"step": parent.model_dump(mode="json")},
        }
        records = [record]
    else:
        child = child.model_copy(update={"method": parent.method})
    bindings = _resolved_render_bindings(child)

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=records,
            resolved_input_bindings=bindings,
        )
        is False
    )


@pytest.mark.parametrize(
    ("parent_output", "child_input"),
    [
        ("primary_association", "table:primary_association"),
        ("table:primary_association", "primary_association"),
    ],
)
def test_effect_figure_source_authority_rejects_bare_product_lineage(
    parent_output,
    child_input,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_output]})
    child = child.model_copy(update={"inputs": [child_input]})
    record["analysis_request"]["step"] = parent.model_dump(mode="json")
    bindings = _resolved_render_bindings(child)

    assert effect_output_authorized(parent) is True
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is False
    )


def test_effect_figure_source_authority_rejects_same_name_with_different_kind():
    parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(update={"inputs": ["statistic:primary_association"]})
    bindings = _resolved_render_bindings(child)

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is False
    )


def test_effect_figure_source_authority_uses_canonical_typed_identity():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={"expected_outputs": ["Table:Primary Association.csv"]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")
    bindings = _resolved_render_bindings(child)

    assert effect_output_authorized(parent) is True
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is True
    )


@pytest.mark.parametrize(
    "case",
    ["missing_bindings", "wrong_producer", "missing_evidence", "missing_digest"],
)
def test_effect_figure_source_authority_requires_verified_direct_parent_binding(case):
    _parent, child, record = _effect_parent_and_figure_child()
    bindings = _resolved_render_bindings(child)
    if case == "missing_bindings":
        bindings = {}
    elif case == "wrong_producer":
        bindings = _resolved_render_bindings(child, producer_step_id="03_stale_model")
    elif case == "missing_evidence":
        bindings[child.inputs[0]]["evidence_id"] = ""
    else:
        bindings[child.inputs[0]]["sha256"] = ""

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is False
    )


def test_effect_figure_source_authority_rejects_mixed_typed_and_raw_inputs():
    _parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(
        update={"inputs": ["table:primary_association", "outcome"]}
    )

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_standalone_effect_figure_uses_verified_binding_instead_of_id_convention():
    _parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(update={"step_id": "05_publication_figure"})

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


def test_effect_figure_allows_only_typed_log_sidecars():
    _parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(
        update={
            "expected_outputs": [
                "figure:primary_association_curve",
                "log:primary_association_render_trace",
            ]
        }
    )
    bindings = _resolved_render_bindings(child)

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=bindings,
        )
        is True
    )
    findings = _step_contract_findings(
        step=child,
        step_summary={
            "status": "ok",
            "output_files": {
                "figure:primary_association_curve": "primary_association_curve.png",
                "log:primary_association_render_trace": (
                    "primary_association_render_trace.json"
                ),
            },
        },
        completed_step_records=[record],
        resolved_input_bindings=bindings,
    )

    assert "unauthorized_effect_product" not in _kinds(findings)


def test_effect_figure_source_authority_rejects_result_bearing_sidecar():
    _parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(
        update={
            "expected_outputs": [
                "figure:primary_association_curve",
                "table:primary_association_copy",
            ]
        }
    )

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_effect_figure_requires_effect_bearing_table_not_unrelated_table():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={
            "expected_outputs": [
                "table:cohort_summary",
                "statistic:primary_effect",
            ]
        }
    )
    child = child.model_copy(
        update={
            "inputs": ["table:cohort_summary", "statistic:primary_effect"],
            "expected_outputs": ["figure:primary_effect_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert effect_output_authorized(parent) is True
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_effect_figure_cannot_relabel_parent_effect_measure():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": ["table:primary_or"]})
    child = child.model_copy(
        update={
            "inputs": ["table:primary_or"],
            "expected_outputs": ["figure:risk_ratio_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )
    matching_child = child.model_copy(
        update={"expected_outputs": ["figure:odds_ratio_forest"]}
    )
    assert (
        _effect_figure_source_authorized(
            step=matching_child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(matching_child),
        )
        is True
    )


@pytest.mark.parametrize(
    ("parent_product", "child_product"),
    [
        ("table:primary_or", "figure:adjusted_or_forest"),
        ("table:risk_ratio", "figure:adjusted_rr_forest"),
        ("table:adjusted_or", "figure:crude_odds_ratio_forest"),
    ],
)
def test_effect_figure_cannot_invent_adjustment_qualifier(
    parent_product,
    child_product,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_product]})
    child = child.model_copy(
        update={"inputs": [parent_product], "expected_outputs": [child_product]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_effect_figure_preserves_matching_adjustment_qualifier():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": ["table:adjusted_or"]})
    child = child.model_copy(
        update={
            "inputs": ["table:adjusted_or"],
            "expected_outputs": ["figure:adjusted_odds_ratio_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


@pytest.mark.parametrize(
    "child_product",
    [
        "figure:interaction_pvalue_heatmap",
        "figure:subgroup_effects_forest",
        "figure:treatment_effect_forest",
        "figure:causal_effect_forest",
    ],
)
def test_effect_figure_cannot_invent_specialized_effect_role(child_product):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": ["table:primary_or"]})
    child = child.model_copy(
        update={"inputs": ["table:primary_or"], "expected_outputs": [child_product]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


@pytest.mark.parametrize(
    ("parent_product", "child_product"),
    [
        ("table:subgroup_effects", "figure:subgroup_effects_forest"),
        ("table:interaction_pvalue", "figure:interaction_pvalue_heatmap"),
        ("table:treatment_effect", "figure:treatment_effect_forest"),
        ("table:causal_effect", "figure:causal_effect_forest"),
    ],
)
def test_effect_figure_preserves_matching_specialized_effect_role(
    parent_product,
    child_product,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_product]})
    child = child.model_copy(
        update={
            "inputs": [parent_product],
            "expected_outputs": [child_product],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


def test_effect_figure_authority_uses_only_the_bound_parent_product():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={"expected_outputs": ["table:primary_or", "table:risk_ratio"]}
    )
    child = child.model_copy(
        update={
            "inputs": ["table:primary_or"],
            "expected_outputs": ["figure:risk_ratio_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_effect_figure_cannot_borrow_adjustment_from_unbound_sibling():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={
            "expected_outputs": [
                "table:crude_odds_ratio",
                "table:adjusted_or",
            ]
        }
    )
    child = child.model_copy(
        update={
            "inputs": ["table:crude_odds_ratio"],
            "expected_outputs": ["figure:adjusted_or_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_generic_effect_figure_can_use_generic_input_among_specialized_siblings():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={"expected_outputs": ["table:primary_or", "table:subgroup_effects"]}
    )
    child = child.model_copy(
        update={
            "inputs": ["table:primary_or"],
            "expected_outputs": ["figure:odds_ratio_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


def test_specialized_figure_can_select_matching_input_from_multi_role_parent():
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(
        update={
            "expected_outputs": [
                "table:subgroup_effects",
                "table:interaction_pvalue",
            ]
        }
    )
    child = child.model_copy(
        update={
            "inputs": ["table:subgroup_effects"],
            "expected_outputs": ["figure:subgroup_effects_forest"],
        }
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


def test_effect_figure_can_combine_verified_tables_from_multiple_parents():
    parent_a = AnalysisStep(
        step_id="04_primary_model",
        intent="Estimate the primary odds ratio.",
        expected_outputs=["table:primary_or"],
        method="logistic_regression",
    )
    parent_b = AnalysisStep(
        step_id="05_adjusted_model",
        intent="Estimate the adjusted odds ratio.",
        expected_outputs=["table:adjusted_or"],
        method="logistic_regression",
    )
    child = AnalysisStep(
        step_id="06_effect_figure",
        intent="Render both verified effect tables.",
        inputs=["table:primary_or", "table:adjusted_or"],
        expected_outputs=["figure:primary_or_forest"],
        method="visualization",
    )
    records = [
        {
            "step_id": parent.step_id,
            "status": "ok",
            "analysis_request": {"step": parent.model_dump(mode="json")},
        }
        for parent in (parent_a, parent_b)
    ]
    bindings = _resolved_render_bindings(child, producer_step_id=parent_a.step_id)
    bindings["table:adjusted_or"]["produced_by_step"] = parent_b.step_id

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=records,
            resolved_input_bindings=bindings,
        )
        is True
    )


def test_effect_figure_multi_parent_authority_requires_every_parent_current():
    parent_a = AnalysisStep(
        step_id="04_primary_model",
        intent="Estimate the primary odds ratio.",
        expected_outputs=["table:primary_or"],
        method="logistic_regression",
    )
    parent_b = AnalysisStep(
        step_id="05_adjusted_model",
        intent="Estimate the adjusted odds ratio.",
        expected_outputs=["table:adjusted_or"],
        method="logistic_regression",
    )
    child = AnalysisStep(
        step_id="06_effect_figure",
        intent="Render both effect tables.",
        inputs=["table:primary_or", "table:adjusted_or"],
        expected_outputs=["figure:primary_or_forest"],
        method="visualization",
    )
    records = [
        {
            "step_id": parent_a.step_id,
            "status": "ok",
            "analysis_request": {"step": parent_a.model_dump(mode="json")},
        },
        {
            "step_id": parent_b.step_id,
            "status": "contract_failed",
            "analysis_request": {"step": parent_b.model_dump(mode="json")},
        },
    ]
    bindings = _resolved_render_bindings(child, producer_step_id=parent_a.step_id)
    bindings["table:adjusted_or"]["produced_by_step"] = parent_b.step_id

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=records,
            resolved_input_bindings=bindings,
        )
        is False
    )


@pytest.mark.parametrize(
    ("parent_product", "child_product"),
    [
        ("table:subgroup_effects", "figure:overall_effect_forest"),
        ("table:interaction_pvalue", "figure:primary_effect_forest"),
        ("table:causal_effect", "figure:primary_association_curve"),
    ],
)
def test_specialized_parent_effect_cannot_be_relabelled_as_generic(
    parent_product,
    child_product,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_product]})
    child = child.model_copy(
        update={"inputs": [parent_product], "expected_outputs": [child_product]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


@pytest.mark.parametrize(
    ("parent_product", "figure_product"),
    [
        ("table:primary_rr", "figure:forest_primary_rr"),
        ("table:relative_risk", "figure:forest_relative_risk"),
        ("table:adjusted_hr", "figure:forest_adjusted_hr"),
        ("table:primary_rd", "figure:forest_primary_rd"),
        ("table:adjusted_odds_ratios", "figure:odds_ratio_forest"),
    ],
)
def test_effect_alias_parent_authorizes_matching_table_backed_figure(
    parent_product,
    figure_product,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_product]})
    child = child.model_copy(
        update={"inputs": [parent_product], "expected_outputs": [figure_product]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert effect_output_authorized(parent) is True
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is True
    )


@pytest.mark.parametrize(
    ("declared_output", "registered_path"),
    [
        ("table:cohort_summary", "risk_ratio.csv"),
        ("figure:overview", "risk_ratio_forest.png"),
    ],
)
def test_typed_registry_role_cannot_launder_effect_bearing_path(
    declared_output,
    registered_path,
):
    step = _step(outputs=[declared_output])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {declared_output: registered_path}},
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


def test_undeclared_effect_named_log_is_not_an_auxiliary_sidecar():
    step = _step(outputs=["table:cohort_summary"])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "output_files": {
                "table:cohort_summary": "cohort_summary.csv",
                "log:risk_ratio": "risk_ratio.json",
            }
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize("effect_log", ["risk_ratio", "primary_or"])
def test_declared_bare_effect_log_is_not_an_auxiliary_sidecar(effect_log):
    output = f"log:{effect_log}"
    step = _step(outputs=[output])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {output: f"{effect_log}.json"}},
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    ("container", "effect_key"),
    [
        ("output_files", "primary_or"),
        ("outputs", "adjusted_rr"),
        ("figure_files", "risk_ratio"),
    ],
)
def test_output_container_cannot_hide_untyped_effect_scalar(container, effect_key):
    step = _step(outputs=["table:cohort_summary"])
    summary = {"output_files": {"table:cohort_summary": "cohort_summary.csv"}}
    summary.setdefault(container, {})[effect_key] = 1.7

    findings = declared_product_contract_findings(
        step=step,
        step_summary=summary,
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    "effect_alias",
    [
        "or",
        "rr",
        "hr",
        "rd",
        "relative_risk",
        "primary_rr",
        "adjusted_rr",
        "adjusted_hr",
        "primary_rd",
        "adjusted_rd",
    ],
)
def test_explicit_effect_family_aliases_cannot_evade_effect_scope(effect_alias):
    step = _step(outputs=[f"figure:{effect_alias}_forest"])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "output_files": {
                f"figure:{effect_alias}_forest": f"{effect_alias}_forest.png"
            }
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    "figure_product",
    ["figure:forest_risk_ratio", "figure:forest_odds_ratio"],
)
def test_effect_role_is_detected_after_display_prefix(figure_product):
    step = _step(outputs=[figure_product])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "output_files": {figure_product: f"{figure_product.split(':', 1)[1]}.png"}
        },
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    "product_name",
    [
        "adjusted_association",
        "adjusted_associations",
        "adjusted_association_model",
        "adjusted_association_models",
        "adjusted_association_primary",
        "adjusted_logistic_regression_primary",
        "adjusted_or_ci",
        "primary_adjusted_association",
        "primary_effect_estimate",
        "primary_estimate",
        "effect_estimate",
        "effect_forest",
    ],
)
def test_shared_effect_vocabulary_blocks_non_effect_owner(product_name):
    output = f"table:{product_name}"
    step = _step(outputs=[output])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {output: f"{product_name}.csv"}},
        effect_method_authorized=False,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    "product_name",
    [
        "adjusted_effect_estimate",
        "subgroup_effect",
        "treatment_effect",
        "effect_estimate",
        "effect_forest",
    ],
)
def test_shared_effect_vocabulary_authorizes_effect_method_owner(product_name):
    step = _step(
        method="logistic_regression",
        outputs=[f"table:{product_name}"],
    )

    assert effect_output_authorized(step) is True


def test_prespecified_robustness_grid_authorizes_locked_effect_refits():
    step = _step(
        method="prespecified_robustness_analysis",
        outputs=["table:robustness_grid", "table:sensitivity_specification_matrix"],
    )

    assert effect_output_authorized(step) is True


@pytest.mark.parametrize(
    ("method", "outputs"),
    [
        ("descriptive_summary", ["table:robustness_grid"]),
        (
            "prespecified_robustness_analysis",
            ["table:sensitivity_specification_matrix"],
        ),
    ],
)
def test_robustness_words_without_method_and_result_product_do_not_authorize_effects(
    method,
    outputs,
):
    step = _step(method=method, outputs=outputs)

    assert effect_output_authorized(step) is False


def test_authorized_figure_role_cannot_launder_rogue_effect_figure_path():
    _parent, child, record = _effect_parent_and_figure_child()

    findings = _step_contract_findings(
        step=child,
        step_summary={
            "status": "ok",
            "output_files": {
                "figure:primary_association_curve": "risk_ratio_forest.png"
            },
        },
        completed_step_records=[record],
        resolved_input_bindings=_resolved_render_bindings(child),
    )

    assert "unauthorized_effect_product" in _kinds(findings)


@pytest.mark.parametrize(
    "extra_summary",
    [
        {"adjusted_or": 1.4},
        {"output_files": {"table:risk_ratio": "risk_ratio.csv"}},
        {"output_files": {"figure:risk_ratio_forest": "risk_ratio_forest.png"}},
    ],
)
def test_effect_figure_parent_authority_never_authorizes_numeric_or_table_effects(
    extra_summary,
):
    _parent, child, record = _effect_parent_and_figure_child()
    bindings = _resolved_render_bindings(child)
    summary = {
        "status": "ok",
        "output_files": {
            "figure:primary_association_curve": "primary_association_curve.png"
        },
    }
    if "output_files" in extra_summary:
        summary["output_files"].update(extra_summary["output_files"])
    else:
        summary.update(extra_summary)

    findings = _step_contract_findings(
        step=child,
        step_summary=summary,
        completed_step_records=[record],
        resolved_input_bindings=bindings,
    )

    assert "unauthorized_effect_product" in _kinds(findings)


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


@pytest.mark.parametrize(
    "product_name",
    [
        "included_or_excluded_counts",
        "missing_or_invalid_rows",
        "exposure_or_outcome_availability",
        "hr_trajectory",
        "rr_distribution",
    ],
)
def test_effect_abbreviations_do_not_capture_non_effect_products(product_name):
    output = f"table:{product_name}"
    step = _step(outputs=[output])

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"output_files": {output: f"{product_name}.csv"}},
        effect_method_authorized=False,
    )

    assert effect_output_authorized(step) is False
    assert "unauthorized_effect_product" not in _kinds(findings)


@pytest.mark.parametrize("abbreviation", ["or", "rr", "hr", "rd"])
def test_bare_typed_abbreviations_are_measure_hints_not_effect_authority(
    abbreviation,
):
    output = f"table:{abbreviation}"
    findings = declared_product_contract_findings(
        step=_step(outputs=[output]),
        step_summary={"output_files": {output: f"{abbreviation}.csv"}},
        effect_method_authorized=False,
    )

    # In ICU data, HR/RR are also heart/respiratory rate and OR may be ordinary
    # language. Exact abbreviations may help interpret a value column only after
    # an unambiguous typed effect product has established authority.
    assert effect_measure_family(output) is not None
    assert effect_bearing_product(output) is False
    assert "unauthorized_effect_product" not in _kinds(findings)


@pytest.mark.parametrize(
    ("parent_product", "primary_figure"),
    [
        ("table:secondary_odds_ratio", "figure:primary_odds_ratio_forest"),
        ("table:sensitivity_risk_ratio", "figure:primary_risk_ratio_forest"),
        (
            "table:corroborative_hazard_ratio",
            "figure:primary_hazard_ratio_forest",
        ),
    ],
)
def test_non_primary_effect_source_cannot_authorize_primary_figure(
    parent_product,
    primary_figure,
):
    parent, child, record = _effect_parent_and_figure_child()
    parent = parent.model_copy(update={"expected_outputs": [parent_product]})
    child = child.model_copy(
        update={"inputs": [parent_product], "expected_outputs": [primary_figure]}
    )
    record["analysis_request"]["step"] = parent.model_dump(mode="json")

    assert effect_output_authorized(parent) is True
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


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


def test_host_slot_authorization_treats_measurement_availability_as_one_role():
    assert authorize_declared_figure_product_slots(
        declared_products=[
            "figure:planned_distribution",
            "figure:planned_measurement_availability",
        ],
        renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
        planner_parent_anchors=[
            "table:planned_distribution",
            "table:planned_missingness",
            "planned_measured",
        ],
    ) == {
        "figure:planned_distribution": "distribution",
        "figure:planned_measurement_availability": "availability",
    }


def test_host_slot_authorization_rejects_unanchored_measurement_availability():
    with pytest.raises(ValueError, match="not anchored to a verified"):
        authorize_declared_figure_product_slots(
            declared_products=["figure:unplanned_measurement_availability"],
            renderer_repair_id=SEALED_DISTRIBUTION_REPAIR,
            planner_parent_anchors=["table:planned_distribution"],
        )


def test_host_slot_authorization_preserves_role_by_exposure_subject():
    assert authorize_declared_figure_product_slots(
        declared_products=["figure:absolute_risk_by_lactate"],
        renderer_repair_id=SEALED_ABSOLUTE_RISK_REPAIR,
        planner_parent_anchors=[
            "artifact:adult_lactate_complete_case",
            "lact_max",
            "death",
            "table:outcome_incidence",
            "table:exposure_prevalence",
        ],
        authoritative_display_subjects=["lact_max", "lactate"],
    ) == {"figure:absolute_risk_by_lactate": "absolute_risk"}


def test_host_slot_authorization_accepts_planned_primary_adjusted_effect():
    assert authorize_declared_figure_product_slots(
        declared_products=["figure:primary_adjusted_effect"],
        renderer_repair_id=(
            "association_publication_bundle_from_planned_model_contract_v1"
        ),
        planner_parent_anchors=[
            "table:adjusted_association_estimates",
            "artifact:primary_model_specification",
        ],
    ) == {"figure:primary_adjusted_effect": "primary_estimand"}


def test_sealed_parent_digest_receipt_is_not_a_scientific_output():
    step = _step(outputs=["figure:primary_adjusted_effect"], method="visualization")
    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "output_files": {
                "figure:primary_adjusted_effect": "primary_adjusted_effect.svg"
            },
            "sealed_renderer_parent_digests": {
                "adjusted_association_estimates.csv": "a" * 64,
                "step_summary.json": "b" * 64,
            },
        },
        effect_method_authorized=False,
        effect_figure_source_authorized=True,
    )
    assert "unauthorized_effect_product" not in _kinds(findings)


@pytest.mark.parametrize(
    "unowned_subject",
    [
        "death",
        "age",
        "unrelated_treatment",
        "lactulose",
    ],
)
def test_host_slot_authorization_rejects_role_by_nonexposure_subject(
    unowned_subject,
):
    with pytest.raises(ValueError):
        authorize_declared_figure_product_slots(
            declared_products=[f"figure:absolute_risk_by_{unowned_subject}"],
            renderer_repair_id=SEALED_ABSOLUTE_RISK_REPAIR,
            planner_parent_anchors=[
                "artifact:adult_lactate_complete_case",
                "artifact:adult_lactulose_complete_case",
                "lact_max",
                "death",
                "age",
                "unrelated_treatment",
                "table:outcome_incidence",
                "table:exposure_prevalence",
            ],
            authoritative_display_subjects=["lact_max", "lactate"],
        )


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


@pytest.mark.parametrize(
    "reported_status",
    [
        "fail_closed",
        "failed_closed",
        "fail_measurement_audit",
        "failed_provenance_audit",
    ],
)
def test_reported_fail_closed_status_fails_outer_step_contract(reported_status):
    from easyicu.research_agent.execution.phase import (
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
