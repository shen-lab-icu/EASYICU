from easyicu.research_agent.repair_reasons import RepairReason, typed_repair_ticket
from easyicu.research_agent.schema import ValidationFinding


def test_typed_repair_ticket_uses_structured_reason_not_message():
    first = ValidationFinding(
        validator="code_preflight",
        severity="error",
        message="wording one",
        detail={"reason": "invalid_local_helper_call", "line": 12},
    )
    second = first.model_copy(update={"message": "completely different wording"})

    assert typed_repair_ticket([first])[0]["reason"] == (
        RepairReason.INVALID_HELPER_SIGNATURE.value
    )
    assert typed_repair_ticket([second])[0]["reason"] == (
        RepairReason.INVALID_HELPER_SIGNATURE.value
    )


def test_typed_repair_ticket_groups_reason_but_retains_distinct_occurrences():
    findings = [
        ValidationFinding(
            validator="code_preflight",
            severity="error",
            message=f"variant {index}",
            detail={"reason": "structural_accounting_filter", "line": index},
        )
        for index in (10, 20)
    ]

    ticket = typed_repair_ticket(findings)

    assert len(ticket) == 1
    assert ticket[0]["reason"] == RepairReason.STRUCTURAL_ACCOUNTING_INVALID.value
    assert ticket[0]["occurrence_count"] == 2
    assert [
        occurrence["detail"]["line"] for occurrence in ticket[0]["occurrences"]
    ] == [10, 20]


def test_typed_repair_ticket_folds_only_identical_occurrences():
    finding = ValidationFinding(
        validator="code_preflight",
        severity="error",
        message="same finding",
        detail={"reason": "structural_accounting_filter", "line": 10},
        evidence_ids=["script_01"],
    )

    ticket = typed_repair_ticket([finding, finding.model_copy(deep=True)])

    assert ticket[0]["occurrence_count"] == 1


def test_typed_unbound_local_ticket_keeps_same_message_at_three_locals():
    findings = [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message="A local may be unbound after a continuing branch.",
            detail={
                "reason": "branch_local_unbound",
                "name": name,
                "branch_line": line,
                "first_use_line": line + 5,
            },
        )
        for name, line in (
            ("coercion_audit", 10),
            ("provenance_audit", 20),
            ("source_status", 30),
        )
    ]

    ticket = typed_repair_ticket(findings)

    assert len(ticket) == 1
    assert ticket[0]["reason"] == RepairReason.UNBOUND_LOCAL.value
    assert ticket[0]["occurrence_count"] == 3
    assert [item["detail"]["name"] for item in ticket[0]["occurrences"]] == [
        "coercion_audit",
        "provenance_audit",
        "source_status",
    ]


def test_typed_repair_ticket_expands_nested_model_issues_and_folds_duplicates():
    issues = [
        {
            "issue": "denominator_contract_unresolvable",
            "reason": "source_variable_missing_from_authoritative_cohort",
            "model_id": model_id,
            "missing_raw_source_variables": ["group_B"],
        }
        for model_id in ("primary", "secondary", "sensitivity")
    ]
    finding = ValidationFinding(
        validator="primary_model_contract",
        severity="error",
        message="Three models have unresolved coefficient lineage.",
        detail={"step_id": "05_models", "issues": [*issues, issues[0].copy()]},
    )

    ticket = typed_repair_ticket([finding])

    assert len(ticket) == 1
    assert ticket[0]["occurrence_count"] == 3
    assert [
        occurrence["detail"]["model_id"] for occurrence in ticket[0]["occurrences"]
    ] == ["primary", "secondary", "sensitivity"]
    assert all(
        occurrence["detail"]["step_id"] == "05_models"
        for occurrence in ticket[0]["occurrences"]
    )
    assert ticket[0]["reason"] == RepairReason.TYPED_PRODUCT_BINDING_INVALID.value


def test_host_validation_swallow_has_structural_accounting_reason():
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="A host-owned validation helper error can continue.",
        detail={"reason": "host_validation_helper_error_swallowed"},
    )

    ticket = typed_repair_ticket([finding])

    assert ticket[0]["reason"] == RepairReason.STRUCTURAL_ACCOUNTING_INVALID.value


def test_llm_concept_finding_has_typed_semantics_reason_without_phrase_routing():
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="arbitrary prose",
        detail={"context": "scientific ownership changed"},
    )

    assert typed_repair_ticket([finding])[0]["reason"] == (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION.value
    )


def test_swallowed_provenance_helper_has_typed_fail_closed_reason():
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="arbitrary wording",
        detail={"reason": "provenance_helper_error_swallowed"},
    )

    assert typed_repair_ticket([finding])[0]["reason"] == (
        RepairReason.PROVENANCE_NOT_FAIL_CLOSED.value
    )


def test_uncomputed_declared_diagnostic_has_typed_reason():
    finding = ValidationFinding(
        validator="declared_product_contract",
        severity="error",
        message="wording may change",
        detail={"kind": "declared_diagnostic_not_completed"},
    )

    assert typed_repair_ticket([finding])[0]["reason"] == (
        RepairReason.DIAGNOSTIC_NOT_COMPLETED.value
    )
