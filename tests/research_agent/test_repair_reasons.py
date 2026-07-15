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


def test_typed_repair_ticket_deduplicates_same_structured_reason():
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
