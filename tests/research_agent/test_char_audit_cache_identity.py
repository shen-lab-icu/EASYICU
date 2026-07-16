"""Freeze-baseline characterization for environment-bound audit cache (G4)."""

from __future__ import annotations

from pathlib import Path


def test_concept_audit_cache_is_environment_auditor_and_run_scoped(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.concept_audit_cache import LLMConceptAuditCache

    context = ra.schema.ResearchContext(
        research_question="Does exposure associate with outcome?",
        cohort=ra.schema.CohortDescriptor(
            cohort_name="fixture",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=[],
    )
    step = ra.schema.AnalysisStep(
        step_id="01_model",
        intent="Fit the agent-planned model.",
        method="regression",
    )
    key_args = {
        "context": context,
        "step": step,
        "script_text": "result = fit_model(frame)\n",
        "audit_prompt": "Audit the planned analysis.",
        "authority_bindings": {"artifact:cohort": {"sha256": "a" * 64}},
        "validator_implementation_sha256": "b" * 64,
    }
    key_environment_a = LLMConceptAuditCache.key(
        **key_args,
        environment_sha256="environment-a",
        auditor_identity="auditor-v1",
    )
    key_environment_b = LLMConceptAuditCache.key(
        **key_args,
        environment_sha256="environment-b",
        auditor_identity="auditor-v1",
    )
    key_auditor_b = LLMConceptAuditCache.key(
        **key_args,
        environment_sha256="environment-a",
        auditor_identity="auditor-v2",
    )

    assert len({key_environment_a, key_environment_b, key_auditor_b}) == 3

    finding = ra.schema.ValidationFinding(
        validator="llm_concept_auditor",
        severity="warning",
        message="Characterized semantic warning.",
    )
    run_a = LLMConceptAuditCache(tmp_path / "run_a")
    run_a.put(key_environment_a, [finding])

    assert [item.message for item in run_a.get(key_environment_a) or []] == [
        finding.message
    ]
    assert run_a.get(key_environment_b) is None
    assert run_a.get(key_auditor_b) is None
    assert LLMConceptAuditCache(tmp_path / "run_b").get(key_environment_a) is None
