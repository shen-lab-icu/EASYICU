"""Scientific-capability receipts must expose their real validation boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.capability_registry import (
    CAPABILITY_REGISTRY,
    ScientificCapability,
    assess_scientific_capability,
)
from easyicu.research_agent.reporting.readiness import _compute_readiness_gates
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    EndpointSpec,
    ResearchContext,
    VariableRole,
)


def _context(*, endpoint: EndpointSpec | None = None) -> ResearchContext:
    return ResearchContext(
        research_question="Does exposure change mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_patients=12, n_stays=12
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                dtype="int64",
                role=VariableRole.INTERVENTION,
            ),
            ConceptDescriptor(
                name="death",
                dtype="int64",
                role=VariableRole.OUTCOME,
            ),
        ],
        primary_exposure="exposure",
        target_outcome="death",
        endpoint=endpoint,
    )


def _survival_endpoint(levels: list[int]) -> EndpointSpec:
    return EndpointSpec(
        name="death",
        kind="time_to_event",
        absence_semantics="no_absent_rows",
        levels=levels,
        event_column="event_type",
        time_column="follow_up_hours",
        time_origin="icu_admission",
        censoring_rule="administrative at 28 days",
    )


def _typed_descriptive_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe exposure prevalence and observed mortality.",
        analysis_type="descriptive_epidemiology",
        steps=[
            AnalysisStep(
                step_id="01_distribution",
                planned_analysis_role="primary",
                intent="Report prespecified unadjusted descriptive risks.",
                method="descriptive",
                inputs=["artifact:analysis_cohort", "exposure", "death"],
                expected_outputs=["table:exposure_outcome_distribution"],
                descriptive_claim={
                    "unresolved_limitations": [
                        "post_baseline_exposure_opportunity_unresolved"
                    ]
                },
                exposure_outcome_distribution_spec={
                    "exposure": "exposure",
                    "exposure_levels": [0, 1],
                    "outcome": "death",
                    "outcome_levels": [0, 1],
                    "outcome_positive_value": 1,
                    "level_match_policy": "exact_typed",
                    "denominator_policy": "all_declared_rows",
                    "missing_outcome_policy": "structural_absence_is_non_event",
                    "confidence_level": 0.95,
                },
            )
        ],
    )


def test_registry_records_machine_readable_scientific_contracts() -> None:
    for capability in CAPABILITY_REGISTRY:
        assert capability.capability_id
        assert capability.result_contract
        assert capability.required_diagnostics

    causal = next(
        capability
        for capability in CAPABILITY_REGISTRY
        if capability.capability_id == "causal_target_trial_v1"
    )

    assert isinstance(causal, ScientificCapability)
    assert causal.result_contract
    assert causal.required_diagnostics
    assert causal.scientific_validation == "analysis_only"


def test_causal_execution_is_honest_about_missing_identification_validator() -> None:
    assessment = assess_scientific_capability(
        analysis_type="causal_inference", context=_context()
    )

    assert assessment.question_present is True
    assert assessment.question_coordinates_resolved is True
    assert assessment.input_contract_resolved is True
    assert assessment.runtime_data_available is None
    assert assessment.execution_backend_available is None
    assert assessment.scientific_validator_available is False
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.issue_code == "scientific_validator_unavailable"
    assert assessment.claim_ceiling_allows_reportable is False


def test_exact_typed_descriptive_primary_has_a_registered_validator() -> None:
    assessment = assess_scientific_capability(
        analysis_type="descriptive_epidemiology",
        context=_context(),
        plan=_typed_descriptive_plan(),
    )

    assert assessment.capability_id == (
        "descriptive_exposure_outcome_distribution_v1"
    )
    assert assessment.scientific_validator_available is True
    assert assessment.claim_ceiling == "reportable"
    assert assessment.issue_code is None


def test_broad_descriptive_family_is_not_upgraded_without_the_exact_owner() -> None:
    assessment = assess_scientific_capability(
        analysis_type="descriptive_epidemiology",
        context=_context(),
    )

    assert assessment.capability_id == "descriptive_measurement_v1"
    assert assessment.scientific_validator_available is False
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.issue_code == "scientific_validator_unavailable"


def test_ordinary_survival_is_distinguished_from_a_competing_risk_endpoint() -> None:
    ordinary = assess_scientific_capability(
        analysis_type="survival",
        context=_context(endpoint=_survival_endpoint([0, 1])),
    )
    competing = assess_scientific_capability(
        analysis_type="survival",
        context=_context(endpoint=_survival_endpoint([0, 1, 2])),
    )

    assert ordinary.claim_ceiling == "reportable"
    assert ordinary.scientific_validator_available is True
    assert competing.claim_ceiling == "unsupported"
    assert competing.issue_code == "competing_risk_estimator_unavailable"
    assert competing.execution_backend_available is None


def test_ordinal_association_stays_analysis_only_without_a_validator() -> None:
    context = ResearchContext(
        research_question="Does ordered severity predict mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_patients=12, n_stays=12
        ),
        variables=[
            ConceptDescriptor(
                name="severity_stage",
                dtype="int64",
                role=VariableRole.ORDINAL_SCORE,
                is_ordinal=True,
                ordinal_levels=[0, 1, 2, 3],
            ),
            ConceptDescriptor(
                name="death", dtype="int64", role=VariableRole.OUTCOME
            ),
        ],
        primary_exposure="severity_stage",
        target_outcome="death",
    )

    assessment = assess_scientific_capability(
        analysis_type="ordinal_dose_response", context=context
    )

    assert assessment.capability_id == "association_ordinal_trend_v1"
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.scientific_validator_available is False
    assert assessment.issue_code == "scientific_validator_unavailable"


def test_ordinal_association_fails_closed_without_a_declared_ordinal_input() -> None:
    assessment = assess_scientific_capability(
        analysis_type="ordinal_dose_response", context=_context()
    )

    assert assessment.capability_id == "association_ordinal_trend_v1"
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.issue_code == "scientific_capability_data_contract_unresolved"


@pytest.mark.parametrize(
    "analysis_type",
    [
        "multimodal",
        "reinforcement_learning",
        "cross_database_replication",
        "treatment_response",
    ],
)
def test_neutral_display_fallback_is_not_mistaken_for_a_scientific_capability(
    analysis_type: str,
) -> None:
    assessment = assess_scientific_capability(
        analysis_type=analysis_type, context=_context()
    )

    assert assessment.claim_ceiling == "unsupported"
    assert assessment.issue_code == "scientific_capability_unregistered"


def test_readiness_keeps_causal_run_analysis_only_without_validator(
    tmp_path: Path,
) -> None:
    plan = AnalysisPlan(
        research_question="Does exposure change mortality?",
        analysis_type="causal_inference",
        steps=[
            AnalysisStep(
                step_id="01_primary",
                intent="Estimate the declared causal contrast.",
                planned_analysis_role="primary",
                expected_outputs=["table:causal_effect"],
            )
        ],
    )

    gates = _compute_readiness_gates(
        context=_context(),
        plan=plan,
        per_step_records=[{"step_id": "01_primary", "status": "ok"}],
        findings=[],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "manuscript.md",
        stop_after_analysis=False,
    )

    receipt = gates["scientific_capability"]
    assert receipt["claim_ceiling"] == "analysis_only"
    assert receipt["question_coordinates_resolved"] is True
    assert "status" not in receipt
    assert receipt["issue_code"] == "scientific_validator_unavailable"
    assert (
        gates["scientific_capability_claim_ceiling_allows_reportable"] is False
    )
    assert gates["analysis_validated"] is False
    assert any(
        "scientific_validator_unavailable" in error
        for error in gates["analysis_errors"]
    )
