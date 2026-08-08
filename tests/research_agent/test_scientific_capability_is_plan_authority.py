"""``scientific_capability`` selects the execution owner, so it is plan authority.

The field was added to ``AnalysisStep`` to make ``association_freeform_v1``
reachable by declaration rather than by inference. It was NOT added to
``_step_scientific_signature``, and the omission was demonstrable rather than
theoretical: flipping only that field on an association primary step moved the
resolved execution owner from ``host_deterministic`` to ``agent_coded`` while
the scientific plan signature stayed byte-identical.

That is the shape this architecture exists to prevent. A step whose primary
result would be computed by the LLM coder instead of the sealed host executor
is not the same step, and a seal/resume comparison blind to the difference
would accept the substitution under an approved plan's identity -- so an
approved deterministic plan could be resumed as a coder-executed one while
every digest still said "same plan".

Note what was already sound and did not need changing: ``PlanReviewAuthority``
hashes ``plan.model_dump(mode="json")`` in full, so the human-review authority
digest always covered this field.

Migration: none required, and that is a property worth asserting rather than
assuming. ``_step_scientific_signature`` compares two signatures computed by
the same code (the sealed record is re-validated through ``AnalysisStep``
before fingerprinting), so it is never a stored digest string. A record written
before the field existed validates it to ``None`` and still matches a live plan
that declares nothing. No frozen golden was refreshed to make these pass.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.plan_scope import (
    _ANALYSIS_STEP_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
    _ANALYSIS_STEP_PRESENTATION_ONLY_FIELDS,
    _ANALYSIS_STEP_RUNTIME_ONLY_FIELDS,
    _ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
    _plan_signature,
    _step_scientific_signature,
)
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.planning.capability_registry import (
    assess_scientific_capability,
    resolve_primary_capability,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)


QUESTION = "Is sepsis associated with in-hospital mortality?"

_TERMS = [
    ModelTermSpec(
        name="sep3", role="exposure", coding="continuous", transform="identity"
    ),
    ModelTermSpec(
        name="age", role="covariate", coding="continuous", transform="identity"
    ),
]


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=QUESTION,
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=8, n_stays=8
        ),
        primary_exposure="sep3",
        target_outcome="death",
        variables=[
            ConceptDescriptor(name="sep3", description="sepsis", dtype="int64"),
            ConceptDescriptor(name="death", description="death", dtype="int64"),
            ConceptDescriptor(name="age", description="age", dtype="float64"),
        ],
    )


def _host_plan(capability=None, method_family="statsmodels_logit_mle") -> AnalysisPlan:
    step = AnalysisStep(
        step_id="06_primary",
        intent="Fit the primary adjusted association model.",
        method="adjusted_association_models",
        inputs=["table:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        planned_analysis_role="primary",
        scientific_capability=capability,
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="m1",
                analysis_role="primary",
                analysis_set="complete_case",
                required_for_step_success=True,
                exposure_source="sep3",
                outcome="death",
                outcome_type="binary",
                method_family=method_family,
                covariates=["age"],
                model_terms=_TERMS,
            )
        ],
    )
    return AnalysisPlan(
        research_question=QUESTION, analysis_type="association_study", steps=[step]
    )


def _freeform_plan(capability="association_freeform_v1") -> AnalysisPlan:
    step = AnalysisStep(
        step_id="06_primary",
        intent="Fit an exposure-by-age interaction model.",
        method="association_interaction_model",
        inputs=["table:analysis_cohort"],
        expected_outputs=["table:interaction_model_estimates"],
        planned_analysis_role="primary",
        scientific_capability=capability,
    )
    return AnalysisPlan(
        research_question=QUESTION, analysis_type="association_study", steps=[step]
    )


# --- 1. the field is load-bearing, so the digest must move with it -----------


def test_the_field_changes_the_execution_owner() -> None:
    """Establishes that it is authority, not presentation metadata."""

    without = resolve_primary_capability(
        analysis_type="association_study", plan=_freeform_plan(capability=None)
    )
    with_it = resolve_primary_capability(
        analysis_type="association_study", plan=_freeform_plan()
    )
    assert without.execution_owner == "unresolved"
    assert without.failure_reason == "scientific_capability_declaration_required"
    assert with_it.execution_owner == "agent_coded"
    assert without.capability_id != with_it.capability_id


@pytest.mark.parametrize("builder", [_freeform_plan, _host_plan])
def test_flipping_only_this_field_changes_the_scientific_plan_signature(
    builder,
) -> None:
    a = builder(capability=None)
    b = builder(capability="association_freeform_v1")

    # Nothing else differs.
    assert a.steps[0].model_dump(exclude={"scientific_capability"}) == b.steps[
        0
    ].model_dump(exclude={"scientific_capability"})

    assert _step_scientific_signature(a.steps[0]) != _step_scientific_signature(
        b.steps[0]
    )
    assert _plan_signature(a) != _plan_signature(b)


def test_every_public_step_field_has_one_explicit_authority_class() -> None:
    classes = (
        _ANALYSIS_STEP_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_STEP_PRESENTATION_ONLY_FIELDS,
        _ANALYSIS_STEP_RUNTIME_ONLY_FIELDS,
    )
    flattened = [field for fields in classes for field in fields]
    assert set(flattened) == set(AnalysisStep.model_fields)
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("covariates", []),
        ("exposure_levels", ["low", "high"]),
        ("exposure_reference_level", "low"),
        ("primary_contrast_level", "high"),
    ],
)
def test_every_nested_model_requirement_authority_change_moves_the_signature(
    field: str, value: object
) -> None:
    base = _host_plan()
    requirement = base.steps[0].model_requirements[0].model_copy(
        update={field: value}
    )
    changed_step = base.steps[0].model_copy(
        update={"model_requirements": [requirement]}
    )
    assert _step_scientific_signature(base.steps[0]) != _step_scientific_signature(
        changed_step
    )


def test_the_human_review_authority_digest_also_moves() -> None:
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority

    a = PlanReviewAuthority.create(plan=_freeform_plan(capability=None))
    b = PlanReviewAuthority.create(plan=_freeform_plan())
    assert a.plan_sha256 != b.plan_sha256


# --- 2. a sealed plan cannot switch owner and keep its identity --------------


def test_a_sealed_step_cannot_switch_between_host_and_freeform() -> None:
    """The seal/resume comparison must refuse the substitution."""

    sealed = _freeform_plan(capability=None).steps[0]
    resumed = _freeform_plan().steps[0]
    assert _step_scientific_signature(sealed) != _step_scientific_signature(resumed)


def test_a_record_written_before_the_field_existed_still_matches() -> None:
    """No stored digest is invalidated: the migration cost is zero, asserted.

    A legacy record carries no ``scientific_capability`` key at all. It must
    re-validate to ``None`` and match a live plan that declares nothing, or
    every historical resume would break.
    """

    live = _freeform_plan(capability=None).steps[0]
    legacy_payload = live.model_dump(mode="json")
    legacy_payload.pop("scientific_capability")
    legacy = AnalysisStep.model_validate(legacy_payload)

    assert legacy.scientific_capability is None
    assert _step_scientific_signature(legacy) == _step_scientific_signature(live)


# --- 3. GLM Binomial: never labelled deterministic if the coder would run ----


def test_glm_binomial_is_never_labelled_deterministic() -> None:
    plan = _host_plan(method_family="statsmodels_glm_binomial")

    from easyicu.research_agent.contracts.association_execution import (
        association_execution_verdict,
    )

    # The owner that would actually run declines...
    assert association_execution_verdict(plan.steps[0]).claimed is False
    # ...so no layer may call it deterministic, and the plan is refused.
    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)
    assert verdict.execution_owner != "host_deterministic"
    assert verdict.failure_reason == "primary_capability_owner_mismatch"

    assessment = assess_scientific_capability(
        analysis_type="association_study", context=_context(), plan=plan
    )
    assert not assessment.claim_ceiling_allows_reportable

    with pytest.raises(ValueError, match="sealed executor cannot run it"):
        validate_required_primary_result(plan=plan, context=_context())


# --- 4. declared free-form: every layer agrees it is free-form ---------------


def test_every_layer_agrees_a_declared_freeform_plan_is_freeform() -> None:
    plan = _freeform_plan()
    context = _context()

    # Planner parse
    validate_required_primary_result(plan=plan, context=context)
    # Capability
    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)
    assert verdict.capability_id == "association_freeform_v1"
    # Execution owner
    assert verdict.execution_owner == "agent_coded"
    assert verdict.coherent
    # Readiness-facing assessment
    assessment = assess_scientific_capability(
        analysis_type="association_study", context=context, plan=plan
    )
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.issue_code == "scientific_validator_unavailable"
    assert assessment.capability_id == "association_freeform_v1"
    # ... and the sealed host executor does not claim the step.
    from easyicu.research_agent.contracts.association_execution import (
        association_execution_verdict,
    )

    assert association_execution_verdict(plan.steps[0]).claimed is False


# --- 5. the authority coordinates it does and does not participate in --------
#
# Enumerated from the only two production reads of the field
# (``capability_registry.resolve_primary_capability`` and the hash site in
# ``authority.plan_scope``), traced forward:
#
#   AnalysisStep.scientific_capability
#     -> resolve_primary_capability
#          -> validate_required_primary_result   (Planner parse accepts/refuses)
#          -> get_capability_for_plan            (compatibility surface)
#          -> assess_scientific_capability
#               -> claim_ceiling / issue_code
#               -> readiness scientific_capability_errors -> analysis_validated
#               -> run_status "scientific_capability" receipt
#               -> _no_det_primary_expected -> replan-budget demotion
#
# Runtime executor selection does NOT read it: ``select_standard_executor``
# reaches ``association_execution_verdict(step)``, which reads method, expected
# outputs and model requirements. The field decides whether the plan is
# admissible and what capability the run reports, not which function runs.


def test_resume_refuses_a_plan_that_changed_the_declared_capability() -> None:
    """The real resume authority predicate, not just the signature helper."""

    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
        completed_step_record_matches_plan,
    )

    sealed_plan = _freeform_plan(capability=None)
    sealed_step = sealed_plan.steps[0]
    record = {
        "step_id": sealed_step.step_id,
        "planned_analysis_role": "primary",
        "analysis_request": {"step": sealed_step.model_dump(mode="json")},
        "plan_scientific_signature": list(
            _serializable_plan_scientific_scope_signature(sealed_plan)
        ),
    }

    def _matches(candidate_plan):
        return completed_step_record_matches_plan(
            record,
            step=candidate_plan.steps[0],
            expected_plan_scope=_serializable_plan_scientific_scope_signature(
                candidate_plan
            ),
            plan_scope_count=1,
            completed_records=[record],
        )

    # The plan that was actually sealed still resumes.
    assert _matches(sealed_plan) is True
    # The same plan with only the declared capability changed does not.
    assert _matches(_freeform_plan()) is False


def test_a_human_review_approval_cannot_be_reused_across_the_mutation() -> None:
    """An approval binds one plan payload; the mutated plan is a different one."""

    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest

    approved = PlanReviewAuthority.create(plan=_freeform_plan(capability=None))
    mutated = PlanReviewAuthority.create(plan=_freeform_plan())

    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="approve the primary association contract",
        authority_sha256=approved.plan_sha256,
        payload={"plan_review_authority": approved.model_dump(mode="json")},
    )

    # A decision is bound to the request id, which binds the authority digest.
    # Re-issuing the request for the mutated plan yields a different id, so the
    # recorded approval cannot answer it.
    mutated_request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="approve the primary association contract",
        authority_sha256=mutated.plan_sha256,
        payload={"plan_review_authority": mutated.model_dump(mode="json")},
    )
    assert request.authority_sha256 != mutated_request.authority_sha256
    assert request.review_id != mutated_request.review_id


def test_execution_identity_is_deliberately_not_a_plan_digest() -> None:
    """Why item 5 of the audit has no assertion about ExecutionIdentity.

    ``ExecutionIdentity`` fingerprints provider authorization, the prompt pack,
    the environment and the input authority -- *how* a run executes. The plan is
    not one of its fields and should not be: plan identity is carried by the
    plan digests above. Recording that here keeps the next reader from
    concluding the coordinate was forgotten.
    """

    from easyicu.research_agent.authority.execution_identity import ExecutionIdentity

    assert "plan" not in set(ExecutionIdentity.model_fields)


def test_the_final_manifest_binds_both_plan_authority_and_execution_identity():
    """Two identities that never converge would be two identities for nothing.

    ``ExecutionIdentity`` deliberately excludes the plan (previous test), so the
    guarantee "different science cannot inherit the same paper authority" only
    holds if the final authority binds *both* coordinates. It does:
    ``orchestration.finalize`` builds ``AnalysisManifest`` with
    ``current_plan_authority`` (whose payload is content-addressed by
    ``sha256_of_file(plan_path)``) **and** ``execution_identity``, and passes
    ``execution_identity.paper_eligible`` into readiness.

    Asserted against the manifest schema rather than a live run so it stays a
    structural invariant: dropping either field from the manifest fails here.
    """

    from easyicu.research_agent.schema import AnalysisManifest

    fields = set(AnalysisManifest.model_fields)
    assert "current_plan_authority" in fields
    assert "execution_identity" in fields


def test_finalize_feeds_paper_eligibility_from_the_execution_identity():
    import inspect

    from easyicu.research_agent.orchestration import finalize

    source = inspect.getsource(finalize)
    assert "execution_paper_eligible=execution_identity.paper_eligible" in source
    assert "plan_authority_verified=True" in source
    assert "plan_authority_sha256=current_plan_authority.sha256" in source
    assert 'execution_identity=execution_identity.model_dump(mode="json")' in source
    assert "current_plan_authority=current_plan_authority.to_dict()" in source

    resolve_at = source.index("current_plan_authority = resolve_registered_plan_authority")
    readiness_at = source.index("readiness, artifact_paths = write_readiness_artifacts")
    assert resolve_at < readiness_at
