"""Audit subpackage.

Groups the static / data / result / replication / manuscript audit
classes that used to live in two unrelated top-level modules
(``validators.py``, ``analysis_pattern_auditor.py``). They are now
organised here so the audit surface has a single import root:

* :mod:`.validators` — data, code, result, replication, manuscript audits
  (``CohortAuditor``, ``ConceptUsageAuditor``, ``LLMConceptAuditor``,
  ``StatisticalValidator``, ``ClinicalConstraintValidator``,
  ``StatisticalGuard``, ``ReplicationDesignAuditor``,
  ``ReplicationResultComparator``, ``PublicationClaimAuditor``,
  ``parse_llm_concept_audit_response``).
* :mod:`.patterns` — analysis-pattern static audit
  (``AnalysisPatternAuditor``).
* :mod:`.base` — shared ``Audit`` Protocol and ``AuditReport``
  dataclass for new audits.

Class names are kept as-is for backwards compatibility. New audits
SHOULD use the ``Audit`` suffix (see :mod:`.base`).
"""

from __future__ import annotations

from .base import Audit, AuditReport, AuditSeverity
from .patterns import AnalysisPatternAuditor
from .validators import (
    ClinicalConstraintValidator,
    CohortAuditor,
    ConceptUsageAuditor,
    LLMConceptAuditor,
    PublicationClaimAuditor,
    ReplicationDesignAuditor,
    ReplicationResultComparator,
    StatisticalGuard,
    StatisticalValidator,
    parse_llm_concept_audit_response,
)

__all__ = [
    # base contract
    "Audit",
    "AuditReport",
    "AuditSeverity",
    # validators
    "ClinicalConstraintValidator",
    "CohortAuditor",
    "ConceptUsageAuditor",
    "LLMConceptAuditor",
    "PublicationClaimAuditor",
    "ReplicationDesignAuditor",
    "ReplicationResultComparator",
    "StatisticalGuard",
    "StatisticalValidator",
    "parse_llm_concept_audit_response",
    # patterns
    "AnalysisPatternAuditor",
]
