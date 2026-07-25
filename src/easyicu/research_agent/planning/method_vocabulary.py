"""Single definition point for host-recognised Planner method strings.

Every deterministic capability chain (standard-executor selection contract →
compact prompt scoping → sealed-renderer registry → renderer controlled-method
check) must agree on the exact Planner method string. Before this module the
same string was retyped in up to four registries per capability, and the
recurring per-family gate blockers came precisely from one copy drifting.

Rules:
* This module stays a pure constant leaf — no imports, no logic — so any
  layer (gates, figures, execution, repair registry) may depend on it without
  creating a direction or cycle risk.
* Constants hold the exact normalised (lowercase, underscore) method head the
  Planner emits. Consumers that accept a ``_with_<rider>`` suffix normalise
  before comparing; the constant itself never carries a rider.
* Adding a deterministic capability means adding ONE constant here and
  referencing it from every registry; ``test_method_vocabulary_registry.py``
  fails when a registry retypes the literal instead.
"""

from __future__ import annotations

MISSINGNESS_SOURCE_AVAILABILITY_AUDIT = "missingness_and_source_availability_audit"
EXPOSURE_DISTRIBUTION_AND_MISSINGNESS_AUDIT = (
    "exposure_distribution_and_missingness_audit"
)
RIGHT_SKEWED_DISTRIBUTION_AND_MEASUREMENT_AVAILABILITY_AUDIT = (
    "right_skewed_distribution_and_measurement_availability_audit"
)
DISTRIBUTION_SUMMARY_AND_MISSINGNESS_AUDIT = (
    "distribution_summary_and_missingness_audit"
)
BINARY_OUTCOME_INCIDENCE_AND_ABSOLUTE_RISK = (
    "binary_outcome_incidence_and_absolute_risk"
)
ADJUSTED_ASSOCIATION_MODELS = "adjusted_association_models"
COHORT_DEFINITION = "cohort_definition"
COHORT_DEFINITION_SENSITIVITY = "cohort_definition_sensitivity"
ORDINAL_EXPOSURE_DERIVATION_AND_QUALITY_CONTROL = (
    "ordinal_exposure_derivation_and_quality_control"
)

__all__ = [
    "ADJUSTED_ASSOCIATION_MODELS",
    "BINARY_OUTCOME_INCIDENCE_AND_ABSOLUTE_RISK",
    "COHORT_DEFINITION",
    "COHORT_DEFINITION_SENSITIVITY",
    "DISTRIBUTION_SUMMARY_AND_MISSINGNESS_AUDIT",
    "EXPOSURE_DISTRIBUTION_AND_MISSINGNESS_AUDIT",
    "MISSINGNESS_SOURCE_AVAILABILITY_AUDIT",
    "ORDINAL_EXPOSURE_DERIVATION_AND_QUALITY_CONTROL",
    "RIGHT_SKEWED_DISTRIBUTION_AND_MEASUREMENT_AVAILABILITY_AUDIT",
]
