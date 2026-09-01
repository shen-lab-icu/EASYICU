"""Pin that contracts.py re-exports canonical runtime types."""

from __future__ import annotations


def test_numeric_claim_is_evidence_canonical():
    from easyicu.research_agent.contracts.runtime import NumericClaim as ContractsNC
    from easyicu.research_agent.authority.evidence_store import NumericClaim as EvidenceNC

    assert ContractsNC is EvidenceNC


def test_validation_finding_is_schema_canonical():
    from easyicu.research_agent.contracts.runtime import ValidationFinding as ContractsVF
    from easyicu.research_agent.schema import ValidationFinding as SchemaVF

    assert ContractsVF is SchemaVF


def test_evidence_artifact_is_schema_evidence_record():
    from easyicu.research_agent.contracts.runtime import EvidenceArtifact
    from easyicu.research_agent.schema import EvidenceRecord

    assert EvidenceArtifact is EvidenceRecord


def test_derived_claim_aliases_numeric_claim():
    from easyicu.research_agent.contracts.runtime import DerivedClaim, NumericClaim

    assert DerivedClaim is NumericClaim


def test_no_duplicate_dataclass_shells_in_contracts():
    """Contracts must not define shadow copies of evidence/schema classes."""

    import easyicu.research_agent.contracts.runtime as c
    import easyicu.research_agent.authority.numeric_claim_identity as n
    import easyicu.research_agent.schema as s

    assert c.NumericClaim.__module__ == n.__name__
    assert c.ValidationFinding.__module__ == s.__name__
    assert c.EvidenceArtifact.__module__ == s.__name__
