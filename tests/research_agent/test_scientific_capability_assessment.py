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

    assert assessment.question_understood is True
    assert assessment.data_available is True
    assert assessment.estimator_available is True
    assert assessment.scientific_validator_available is False
    assert assessment.status == "analysis_only"
    assert assessment.issue_code == "scientific_validator_unavailable"
    assert assessment.publication_eligible is False


def test_ordinary_survival_is_distinguished_from_a_competing_risk_endpoint() -> None:
    ordinary = assess_scientific_capability(
        analysis_type="survival",
        context=_context(endpoint=_survival_endpoint([0, 1])),
    )
    competing = assess_scientific_capability(
        analysis_type="survival",
        context=_context(endpoint=_survival_endpoint([0, 1, 2])),
    )

    assert ordinary.status == "reportable"
    assert ordinary.scientific_validator_available is True
    assert competing.status == "unsupported"
    assert competing.issue_code == "competing_risk_estimator_unavailable"
    assert competing.estimator_available is False


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

    assert assessment.status == "unsupported"
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
    assert receipt["status"] == "analysis_only"
    assert receipt["issue_code"] == "scientific_validator_unavailable"
    assert gates["scientific_capability_reportable"] is False
    assert gates["analysis_validated"] is False
    assert any(
        "scientific_validator_unavailable" in error
        for error in gates["analysis_errors"]
    )
