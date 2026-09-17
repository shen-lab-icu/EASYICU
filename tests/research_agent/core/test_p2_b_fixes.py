"""P2-B batch acceptance: explicit degradation / transition / binding markers."""

from __future__ import annotations


def test_b_p2_1_enrichment_degraded_propagates_to_coverage() -> None:
    from easyicu.research_agent.acquisition.catalog import (
        AvailableCatalog,
        CatalogConcept,
        assess_coverage,
    )

    degraded = AvailableCatalog(
        source="mem",
        concepts=[CatalogConcept(concept_id="lact", category="labs")],
        enrichment_degraded=True,
    )
    report = assess_coverage(["lact", "missing_x"], degraded)
    assert report.enrichment_degraded is True
    assert report.to_dict()["enrichment_degraded"] is True
    assert any("enrichment" in advice.lower() for advice in report.advice)

    clean = AvailableCatalog(
        source="mem",
        concepts=[CatalogConcept(concept_id="lact", category="labs")],
        enrichment_degraded=False,
    )
    clean_report = assess_coverage(["lact"], clean)
    assert clean_report.enrichment_degraded is False
    assert clean_report.to_dict()["enrichment_degraded"] is False


def test_b_p2_1_loader_failure_marks_degraded(monkeypatch) -> None:
    import easyicu.research_agent.acquisition.catalog as catalog_mod

    def _boom():
        raise RuntimeError("dict unavailable")

    monkeypatch.setattr(
        "easyicu.concept.loader.load_concept_dict_cached", _boom
    )
    meta, degraded = catalog_mod._concept_dict_meta_with_status()
    assert meta == {}
    assert degraded is True


def test_b_p2_4_builder_marks_enrichment_degraded() -> None:
    import pandas as pd

    from easyicu.research_agent.research_context.builder import _describe_column

    df = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "totally_unknown_concept_xyz": [1.0, 2.0],
        }
    )
    descriptor = _describe_column(
        df=df,
        col="totally_unknown_concept_xyz",
        user_descriptions={},
        id_columns=["stay_id"],
        time_columns=[],
        outcome_columns=[],
    )
    assert descriptor.concept_enrichment_degraded is True
    assert "concept_enrichment_degraded" in type(descriptor).model_fields


def test_b_p2_5_transitioned_rejects_illegal_edge() -> None:
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        HumanReviewCheckpointConsumed,
        _ALLOWED_CHECKPOINT_TRANSITIONS,
    )

    assert "pending" in _ALLOWED_CHECKPOINT_TRANSITIONS
    assert "completed" not in _ALLOWED_CHECKPOINT_TRANSITIONS["pending"]

    # Build a minimal pending checkpoint via create().
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        HumanReviewCheckpoint,
    )
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest

    request = HumanReviewRequest.create(
        kind="protocol_claim",
        summary="s",
        authority_sha256="0" * 64,
        payload={},
    )
    checkpoint = HumanReviewCheckpoint.create(
        run_id="run_p2",
        pipeline_config_sha256="0" * 64,
        environment_identity={},
        llm_signature_sha256="0" * 64,
        run_input_capsule_sha256="0" * 64,
        capability_activation_sha256="0" * 64,
        runtime_capabilities=[],
        runtime_bundle=None,
        requests=[request],
        plan_handoff={},
        execution_coordinates={},
    )
    assert checkpoint.state == "pending"
    try:
        checkpoint.transitioned("completed")  # type: ignore[arg-type]
    except HumanReviewCheckpointConsumed:
        pass
    else:
        raise AssertionError("pending->completed must raise Consumed")
    # Legal edge still works (pending->failed without a decision set).
    assert checkpoint.transitioned("failed").state == "failed"


def test_c_f6_public_wrappers_match_private() -> None:
    from easyicu.research_agent.gates import visual as visual_owner
    from easyicu.research_agent.gates import step_contract as contract_owner
    from easyicu.research_agent.gates import step_result_evidence as evidence_owner
    from easyicu.research_agent.reporting import publication_bundles as bundles_owner

    for module, names in (
        (
            visual_owner,
            (
                "is_cosmetic_visual_finding",
                "demote_cosmetic_visual_findings",
                "visual_repair_request_log",
            ),
        ),
        (contract_owner, ("step_contract_findings",)),
        (
            evidence_owner,
            (
                "finite_float",
                "exposure_names_match",
                "primary_effect_from_summary",
                "primary_exposure_contract_findings",
                "primary_exposure_measurement_filter_findings",
            ),
        ),
        (
            bundles_owner,
            (
                "context_axis_label",
                "association_descriptive_context",
                "truthy_figure_value",
                "explicit_false_figure_value",
                "sensitivity_plot_label",
            ),
        ),
    ):
        assert hasattr(module, "__all__")
        for name in names:
            assert name in module.__all__, f"{module.__name__} missing {name}"
            assert callable(getattr(module, name))


def test_c_f6_no_private_imports_in_converted_callers() -> None:
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    base = repo / "src" / "easyicu" / "research_agent"
    checks = {
        "reporting/readiness.py": ("easyicu.research_agent.gates.visual",),
        "execution/phase.py": ("easyicu.research_agent.gates.visual",),
        "robustness/primary_effect.py": (
            "easyicu.research_agent.gates.step_result_evidence",
        ),
        "figures/association_prior_outputs.py": (
            "easyicu.research_agent.reporting.publication_bundles",
        ),
        "figures/sensitivity_prior_outputs.py": (
            "easyicu.research_agent.reporting.publication_bundles",
        ),
        "gates/contract.py": (
            "easyicu.research_agent.gates.step_contract",
            "easyicu.research_agent.gates.step_result_evidence",
        ),
        "gates/step_contract.py": (
            "easyicu.research_agent.gates.step_result_evidence",
        ),
    }
    for rel, _modules in checks.items():
        tree = ast.parse((base / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                ("gates.visual", "gates.step_contract", "gates.step_result_evidence", "reporting.publication_bundles")
            ):
                for alias in node.names:
                    assert not alias.name.startswith("_"), (
                        f"{rel} still imports private {alias.name} "
                        f"from {node.module}"
                    )


def test_c_f8_isolation_degraded_marker() -> None:
    from easyicu.research_agent.reporting.system_validation_report import (
        ValidationSourceBinding,
        build_system_validation_report,
    )

    assert ValidationSourceBinding.model_fields["isolation_degraded"].default is False
    base = build_system_validation_report(
        run_id="r",
        projections={},
        run_status={},
        review_checkpoint={},
        provider_usage=None,
        projection_privacy_passed=False,
    )
    assert all(b.isolation_degraded is False for b in base.source_bindings)

    degraded = build_system_validation_report(
        run_id="r",
        projections={},
        run_status={"isolation_degraded": True},
        review_checkpoint={},
        provider_usage=None,
        projection_privacy_passed=False,
    )
    markers = [b for b in degraded.source_bindings if b.isolation_degraded]
    assert len(markers) == 1
    assert markers[0].artifact == "execution_isolation"


def test_c_f9_fallback_downgraded_unless_plan_allows() -> None:
    from easyicu.research_agent.reporting.readiness import (
        _deterministic_primary_estimate_bound,
        _fallback_primary_allowed,
        _primary_records_for_readiness,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    step = AnalysisStep(
        step_id="s1",
        intent="auxiliary",
        inputs=[],
        expected_outputs=[],
        planned_analysis_role="auxiliary",
    )
    plan = AnalysisPlan(research_question="q", steps=[step])
    fallback_record = {
        "step_id": "s1",
        "generation_mode": "fallback",
        "deterministic_standard_analysis": "survival_primary_cox",
        "step_summary": {"receipt_issuer": "x"},
    }
    assert _fallback_primary_allowed(plan, "s1") is False
    assert _primary_records_for_readiness([fallback_record], plan) == []
    assert _deterministic_primary_estimate_bound([fallback_record], plan) is False

    step_allow = step.model_copy(update={"allow_fallback_as_primary": True})
    plan_allow = AnalysisPlan(research_question="q", steps=[step_allow])
    assert _fallback_primary_allowed(plan_allow, "s1") is True
    assert _primary_records_for_readiness([fallback_record], plan_allow) == [
        fallback_record
    ]


def test_c_f9_fallback_requires_compatibility_gate() -> None:
    from easyicu.research_agent.gates.method_compatibility import (
        fallback_method_compatibility_findings,
    )
    from easyicu.research_agent.reporting.readiness import (
        _fallback_method_compatibility_errors,
    )

    assert callable(fallback_method_compatibility_findings)
    record = {
        "step_id": "s1",
        "generation_mode": "fallback",
        "step_summary": {"status": "ok"},
    }
    errors = _fallback_method_compatibility_errors(
        per_step_records=[record], context=None, plan=None
    )
    assert len(errors) == 1
    assert errors[0].validator == "method_compatibility"


def test_c_f11_supersedes_without_restoring_artifact() -> None:
    from easyicu.research_agent.reporting.system_validation_report import (
        ValidationSourceBinding,
    )

    assert "supersedes" in ValidationSourceBinding.model_fields
    binding = ValidationSourceBinding(
        artifact="provider_usage_accounting",
        sha256="0" * 64,
        binding_scope="run_private_receipt",
        supersedes="provider_ledger",
    )
    assert binding.supersedes == "provider_ledger"
    assert binding.artifact == "provider_usage_accounting"
    default = ValidationSourceBinding(
        artifact="x", sha256="0" * 64, binding_scope="run_private_receipt"
    )
    assert default.supersedes is None


def test_c_f13_retry_policy_contract_and_reference() -> None:
    from easyicu.research_agent.contracts.retry_policy import (
        FAILURE_CLASS_RETRY_BUDGET,
        RETRY_DENOMINATOR_DEFINITION,
        budget_for_failure_class,
        retry_accounting_receipt,
        retry_denominator,
        retry_policy_receipt,
    )

    assert len(FAILURE_CLASS_RETRY_BUDGET) == 4
    assert "step attempts" in RETRY_DENOMINATOR_DEFINITION.lower()
    assert budget_for_failure_class("execution_timeout") is not None
    assert budget_for_failure_class("execution_timeout").max_llm_repairs == 0
    assert retry_denominator(5) == 5
    receipt = retry_policy_receipt()
    assert receipt["schema_version"] == "easyicu.retry_policy/2"
    accounting = retry_accounting_receipt(
        [
            {
                "attempt_id": "run:s1:1",
                "step_id": "s1",
                "code_repair_attempts": 0,
            },
            {
                "attempt_id": "run:s1:1",
                "step_id": "s1",
                "code_repair_attempts": 2,
                "step_llm_repair_attempts": 1,
            },
            {
                "attempt_id": "run:s1:2",
                "step_id": "s1",
                "code_repair_attempts": 1,
                "step_llm_repair_attempts": 2,
            },
            {
                "attempt_id": "run:s2:1",
                "step_id": "s2",
                "runtime_failure_class": "execution_timeout",
                "code_repair_attempts": 0,
            },
        ]
    )
    assert accounting["attempt_denominator"] == 3
    assert accounting["logical_llm_repair_attempts"] == 2
    assert accounting["code_mutation_attempts"] == 3
    assert accounting["fail_closed_attempts"] == 1
    assert accounting["failure_class_counts"] == {"execution_timeout": 1}

    import pytest

    with pytest.raises(ValueError, match="unknown runtime failure class"):
        retry_accounting_receipt(
            [{"attempt_id": "run:s3:1", "runtime_failure_class": "new_class"}]
        )

    import easyicu.research_agent.reporting.system_validation_report as report_mod

    assert hasattr(report_mod, "_budget_for_failure_class")


def test_c_f13_retry_table_governs_routing_and_reports() -> None:
    """Retry policy is enforced, not merely referenced.

    Unknown failure classes fail closed in the router; the validation
    report emits an error finding for unknown observed classes.
    """
    import pytest

    from easyicu.research_agent.contracts.retry_policy import repair_route_for

    assert repair_route_for("execution_timeout") == "fail_closed"
    assert repair_route_for("plan_data_contract") == "fail_closed"
    with pytest.raises(ValueError, match="unknown runtime failure class"):
        repair_route_for("some_future_class")


def test_c_f12_missing_vs_empty_sentinel() -> None:
    from easyicu.research_agent.reporting.system_validation_report import (
        _case_table_with_status,
        build_system_validation_report,
    )

    table, status = _case_table_with_status({"tables": []})
    assert table is None
    assert status == "missing"

    candidate = {
        "tables": [
            {
                "name": "t",
                "label": "l",
                "evidence_id": "e",
                "headers": [
                    "exposure_level",
                    "n_rows",
                    "exposure_denominator",
                    "exposure_pct",
                ],
                "rows": [],
            }
        ]
    }
    table_empty, status_empty = _case_table_with_status(candidate)
    assert table_empty is None
    assert status_empty == "empty"

    report = build_system_validation_report(
        run_id="r",
        projections={"result_tables.json": {"tables": []}},
        run_status={},
        review_checkpoint={},
        provider_usage=None,
        projection_privacy_passed=False,
    )
    assert report.case_study.primary_table_status == "missing"
    assert report.case_study.primary_table is None
