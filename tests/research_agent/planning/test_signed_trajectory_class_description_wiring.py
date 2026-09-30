"""The signed trajectory suite can describe its frozen classes, wired by the host.

The four signed owners never read an outcome.  A plan that asks how an outcome
differs by class keeps one ``phenotyping.outcome_by_cluster`` step: the Planner
chooses its roster, and ``bind_plan`` wires it to the run cohort (republished
by the interpretation-free root) and to the stability owner's frozen labels and
freeze record.  Synthetic authority, context and plans only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent import pipeline as _pipeline
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.trajectory_design import (
    TRAJECTORY_OUTCOME_DESCRIPTION_RULE,
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    FROZEN_CLASS_COHORT_STEP_ID,
    FROZEN_CLASS_DESCRIPTION_STEP_ID,
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.progressive_compiler import (
    _validate_scientific_action_runtime_contract,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from easyicu.research_agent.planning.progressive_resume import (
    _migrate_installed_runtime_contract,
)
from easyicu.research_agent.planning.scientific_action_catalog import (
    scientific_action_for_id,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.bundle import (
    resolve_trajectory_bundle_plan_authority,
    trajectory_bundle_findings,
)
from easyicu.research_agent.trajectory.plan_contract import evaluate_trajectory_plan_dag
from easyicu.research_agent.trajectory.runtime_validation import (
    signed_trajectory_plan_contract_errors,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

ALLOWED = ["strobe_2007", "record_2015"]
COORDINATES = ("sofa2_cardio", "sofa2_renal", "lact")
QUESTION = (
    "Among adult ICU stays, which organ-support trajectory classes emerge over the "
    "first 72 h from cardiovascular and renal SOFA-2 scores and lactate, and how does "
    "hospital mortality differ by class?"
)
COMPARISON = "phenotyping.outcome_by_cluster"
LABELS = "table:cluster_assignments"
FREEZE = "artifact:stability_freeze"


def _authority():
    design = load_trajectory_design({"coordinate_concepts": list(COORDINATES)})
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="0" * 64)
    )


def _authorities(authority=None) -> ScientificRuntimeAuthorities:
    return ScientificRuntimeAuthorities(
        trajectory=authority or _authority(), current_case=None
    )


def _context(*, requested=("death",)) -> ResearchContext:
    return ResearchContext(
        research_question=QUESTION,
        cohort=CohortDescriptor(
            cohort_name="organ_support_fixture",
            database="miiv",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            requested_outcome_columns=list(requested),
            provenance={
                "analysis_unit": "icu_stay",
                "patient_identity_available": False,
                "stay_id_columns": ["stay_id"],
                "patient_id_columns": [],
            },
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            *[
                ConceptDescriptor(
                    name=concept,
                    description=f"{concept} coordinate",
                    role=(
                        VariableRole.ORDINAL_SCORE
                        if concept.startswith("sofa2")
                        else VariableRole.LAB
                    ),
                    dtype="float64",
                    analysis_window="icu_admission[0,72]h",
                )
                for concept in COORDINATES
            ],
            ConceptDescriptor(
                name="death",
                description="hospital mortality",
                role=VariableRole.OUTCOME,
                dtype="float64",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
        ],
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]
        ),
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering",
            covariate_selection="planner_selectable",
        ),
    )


def _comparison_step(step_id="outcome_by_class", identity="stay_id") -> dict:
    return {
        "step_id": step_id,
        "planned_analysis_role": "secondary",
        "intent": "Describe hospital mortality by frozen class.",
        "method": "descriptive_outcome_by_cluster",
        "scientific_action_id": COMPARISON,
        "inputs": [identity, "death", "artifact:analysis_cohort", LABELS, FREEZE],
        "expected_outputs": ["table:outcome_by_cluster"],
        "phenotype_comparison_spec": {
            "identity_column": identity,
            "outcome_columns": ["death"],
            "variables": [
                {
                    "name": "death",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "none_descriptive_smd_only",
                    "levels": [0, 1],
                }
            ],
        },
        "literature_citation_keys": ["strobe_2007"],
    }


def _draft(authority, *extra_steps) -> AnalysisPlan:
    owners = authority.development_execution_only_plan(research_question=QUESTION)
    payload = owners.model_dump(mode="json")
    return AnalysisPlan.model_validate(
        {**payload, "steps": [*payload["steps"], *extra_steps]}
    )


def _bound(authority=None):
    authority = authority or _authority()
    return _authorities(authority).bind_plan(_draft(authority, _comparison_step()))


def _step(plan: AnalysisPlan, step_id: str) -> AnalysisStep:
    return next(step for step in plan.steps if step.step_id == step_id)


def _replace(plan: AnalysisPlan, step_id: str, **update) -> AnalysisPlan:
    return plan.model_copy(
        update={
            "steps": [
                step.model_copy(update=update) if step.step_id == step_id else step
                for step in plan.steps
            ]
        }
    )


def test_binding_wires_the_drafts_description_to_the_run_cohort_and_frozen_labels():
    bound, findings = _bound()

    root = _step(bound, FROZEN_CLASS_COHORT_STEP_ID)
    description = _step(bound, FROZEN_CLASS_DESCRIPTION_STEP_ID)
    assert (root.method, root.planned_analysis_role, root.inputs, root.expected_outputs) == (
        "host_materialized_locked_cohort",
        "auxiliary",
        [],
        ["table:analysis_cohort"],
    )
    assert description.inputs == ["stay_id", "death", "table:analysis_cohort", LABELS, FREEZE]
    assert description.phenotype_comparison_spec.outcome_columns == ["death"]
    assert description.literature_citation_keys == ["strobe_2007"]
    assert findings[0].detail["frozen_class_description"] == {
        "carried": True,
        "draft_step_ids": ["outcome_by_class"],
        "step_ids": [FROZEN_CLASS_COHORT_STEP_ID, FROZEN_CLASS_DESCRIPTION_STEP_ID],
    }
    assert signed_trajectory_plan_contract_errors(bound) == []


def test_rebinding_a_bound_plan_keeps_the_same_description():
    authority = _authority()
    bound, _ = _bound(authority)
    rebound, findings = _authorities(authority).bind_plan(bound)
    assert rebound.steps == bound.steps
    assert findings[0].detail["frozen_class_description"]["carried"] is True


def test_a_plan_without_a_description_binds_the_four_owners_only():
    authority = _authority()
    bound, findings = _authorities(authority).bind_plan(_draft(authority))
    assert len(bound.steps) == 4
    assert findings[0].detail["frozen_class_description"] == {
        "carried": False,
        "reason": "not_requested",
    }


@pytest.mark.parametrize(
    "extra_steps,reason",
    [
        (
            [_comparison_step(), _comparison_step("second_description")],
            "ambiguous_or_unspecified",
        ),
        # The representation keys its rows by the stay; another identity
        # cannot join its frozen labels.
        ([_comparison_step(identity="hadm_id")], "signed_contract_refused"),
    ],
)
def test_a_description_that_cannot_be_wired_is_reported_not_carried(extra_steps, reason):
    authority = _authority()
    bound, findings = _authorities(authority).bind_plan(_draft(authority, *extra_steps))
    assert len(bound.steps) == 4
    detail = findings[0].detail["frozen_class_description"]
    assert (detail["carried"], detail["reason"]) == (False, reason)


def _without(plan: AnalysisPlan, step_id: str) -> AnalysisPlan:
    return plan.model_copy(
        update={"steps": [step for step in plan.steps if step.step_id != step_id]}
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda plan: _replace(
                plan,
                FROZEN_CLASS_DESCRIPTION_STEP_ID,
                inputs=["stay_id", "death", "table:analysis_cohort", "table:phenotype_assignments"],
            ),
            id="cross_sectional_labels",
        ),
        pytest.param(
            lambda plan: _replace(plan, FROZEN_CLASS_COHORT_STEP_ID, inputs=[LABELS]),
            id="root_reads_an_input",
        ),
        pytest.param(
            lambda plan: _without(plan, FROZEN_CLASS_DESCRIPTION_STEP_ID),
            id="root_without_description",
        ),
        pytest.param(
            lambda plan: _without(plan, FROZEN_CLASS_COHORT_STEP_ID),
            id="description_without_root",
        ),
        pytest.param(
            lambda plan: plan.model_copy(
                update={
                    "steps": [
                        *plan.steps,
                        _step(plan, FROZEN_CLASS_COHORT_STEP_ID).model_copy(
                            update={"step_id": "06_second_root"}
                        ),
                    ]
                }
            ),
            id="two_roots",
        ),
        pytest.param(
            lambda plan: _replace(
                plan,
                FROZEN_CLASS_DESCRIPTION_STEP_ID,
                phenotype_comparison_spec=_step(
                    plan, FROZEN_CLASS_DESCRIPTION_STEP_ID
                ).phenotype_comparison_spec.model_copy(update={"identity_column": "hadm_id"}),
                inputs=["hadm_id", "death", "table:analysis_cohort", LABELS, FREEZE],
            ),
            id="another_identity",
        ),
    ],
)
def test_the_signed_contract_admits_only_the_host_wired_description(mutate):
    bound, _ = _bound()
    assert signed_trajectory_plan_contract_errors(mutate(bound))


def test_the_trajectory_dag_and_bundle_leave_the_description_to_its_own_contract():
    bound, _ = _bound()
    context = _context()
    assert evaluate_trajectory_plan_dag(
        plan=bound, context=context, long_trajectory_bound=True
    ).findings == ()
    assert resolve_trajectory_bundle_plan_authority(
        plan=bound, context=context, long_trajectory_bound=True
    ).findings == ()

    # An agent-coded step that is not the typed comparison still may not
    # declare the characterization owner's outcome table.
    untyped = _replace(bound, FROZEN_CLASS_DESCRIPTION_STEP_ID, scientific_action_id=None)
    kinds = [
        (finding.detail or {}).get("kind")
        for finding in evaluate_trajectory_plan_dag(
            plan=untyped, context=context, long_trajectory_bound=True
        ).findings
    ]
    assert "trajectory_role_product_owner_mismatch" in kinds


def test_the_bundle_does_not_take_the_description_table_for_an_owner_product(tmp_path):
    bound, _ = _bound()
    table = tmp_path / "outcome_by_cluster.csv"
    table.write_text("variable,group\n", encoding="utf-8")
    evidence = EvidenceStore(tmp_path)
    evidence.register_file(
        kind="table",
        description="Frozen-class description.",
        source_path=table,
        produced_by_step=FROZEN_CLASS_DESCRIPTION_STEP_ID,
        evidence_id="table_frozen_class_description",
    )
    findings = trajectory_bundle_findings(
        context=_context(),
        plan=bound,
        per_step_records=[
            {
                "step_id": FROZEN_CLASS_DESCRIPTION_STEP_ID,
                "status": "ok",
                "evidence_ids": ["table_frozen_class_description"],
            }
        ],
        evidence=evidence,
        run_dir=tmp_path,
        cohort_path=tmp_path / "cohort.parquet",
        long_trajectory_bound=True,
    )
    kinds = {finding.detail["kind"] for finding in findings}
    # The owners produced nothing in this fixture; only that is reported.
    assert kinds == {"missing_current_evidence"}


def _review_codes(plan: AnalysisPlan, context=None) -> dict[str, str]:
    context = context or _context()
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=None,
        figure_strategy=build_article_figure_strategy(context),
        runtime_authority=None,
    )
    return {
        finding.code: finding.message
        for finding in review.findings
        if finding.code.startswith("PHENOTYPING_")
    }


def test_the_review_asks_the_signed_suite_for_the_requested_outcome_description():
    authority = _authority()
    owners_only, _ = _authorities(authority).bind_plan(_draft(authority))
    assert set(_review_codes(owners_only)) == {"PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE"}
    context = _context(requested=())
    no_outcome = context.model_copy(update={"target_outcome": None, "endpoint": None})
    assert _review_codes(owners_only, no_outcome) == {}

    bound, _ = _bound(authority)
    assert _review_codes(bound) == {}


def test_the_review_refuses_frozen_trajectory_labels_outside_the_signed_suite():
    bound, _ = _bound()
    description = _step(bound, FROZEN_CLASS_DESCRIPTION_STEP_ID)
    root = _step(bound, FROZEN_CLASS_COHORT_STEP_ID)
    unsigned = AnalysisPlan(
        research_question=QUESTION,
        analysis_type="trajectory_clustering",
        steps=[root, description],
    )
    codes = _review_codes(unsigned)
    assert "phenotype_comparison_trajectory_source_invalid" in codes[
        "PHENOTYPING_COMPARISON_CONTRACT_INVALID"
    ]


def _skeleton_step(product_inputs, depends_on=("cohort_accounting", "cluster_stability")):
    return ProgressiveSkeletonStep(
        step_id="outcome_by_class",
        planned_analysis_role="secondary",
        module_id="custom_analysis",
        objective="Describe hospital mortality by frozen class.",
        depends_on=list(depends_on),
        raw_inputs=["death"],
        product_inputs=[
            ProgressiveProductRef(producer_step_id=producer, product_id=product)
            for producer, product in product_inputs
        ],
        outputs=[{"product_id": "table:outcome_by_cluster", "semantic_role": "custom"}],
        scientific_action_id=COMPARISON,
        custom_method="descriptive_outcome_by_cluster",
        phenotyping_comparison_variables=[{"name": "death", "summary": "count_percent"}],
        literature_bindings=[],
    )


TRAJECTORY_INPUTS = (
    ("cohort_accounting", "artifact:analysis_cohort"),
    ("cluster_stability", LABELS),
    ("cluster_stability", FREEZE),
)


def test_the_compiler_accepts_the_frozen_label_source_as_a_whole_set():
    action = scientific_action_for_id(analysis_type="trajectory_clustering", action_id=COMPARISON)
    outputs = [("table:outcome_by_cluster", "custom")]
    _validate_scientific_action_runtime_contract(
        action=action, step=_skeleton_step(TRAJECTORY_INPUTS), step_index=5, outputs=outputs
    )
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate_scientific_action_runtime_contract(
            action=action,
            step=_skeleton_step(TRAJECTORY_INPUTS[:2]),
            step_index=5,
            outputs=outputs,
        )
    assert caught.value.reason_code == "progressive_scientific_action_inputs_mismatch"
    assert "table:phenotype_assignments" in str(caught.value)
    assert "artifact:stability_freeze" in str(caught.value)


def test_resume_keeps_an_accepted_frozen_label_source():
    step = _skeleton_step(
        TRAJECTORY_INPUTS,
        depends_on=("cohort_accounting", "cluster_stability", "cross_sectional_clusters"),
    )
    materialization = ProgressiveStepMaterialization(
        outline_step_sha256="0" * 64, foundation=None, step=step
    )
    migrated = _migrate_installed_runtime_contract(
        materialization,
        analysis_type="trajectory_clustering",
        available_product_refs=[
            ("cohort_accounting", "artifact:analysis_cohort"),
            ("cross_sectional_clusters", "table:phenotype_assignments"),
            ("cluster_stability", LABELS),
            ("cluster_stability", FREEZE),
        ],
    )
    assert migrated.step.product_inputs == step.product_inputs


def _labels_payload(request) -> dict:
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": key.replace("_", " ") + " (label)"}
            for key in dict.fromkeys(
                [*request.required_reader_label_keys, *request.level_label_keys]
            )
        ],
        "comparator_applications": [],
        "roster_decision_note": "Sealed suite: the host owns every coordinate; labels only.",
    }


def test_the_suite_template_describes_the_requested_outcome_through_the_host_path():
    authority = _authority()
    authorities = _authorities(authority)
    context = _context()
    planning_context = authorities.planning_contract_context()
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_context,
    )
    llm = ScriptedMockLLMClient([json.dumps(_labels_payload(request))])
    plan = ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED,
        direct_comparator_literature_keys=[],
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context=planning_context,
        required_primary_cohort_selection_mode="all_input_rows",
    ).output
    drafted = [step for step in plan.steps if step.scientific_action_id == COMPARISON]
    assert [step.inputs[-3:] for step in drafted] == [
        ["artifact:analysis_cohort", LABELS, FREEZE]
    ]
    # SOFA-2 components are declared ordinal, so the reviewed design names the
    # mixed-mode class model the sealed candidate owner fits.
    assert "mixed-mode latent class" in plan.design_selection.selected.primary_method
    findings: list = []
    plan = _pipeline._shape_fresh_plan(
        pipeline=SimpleNamespace(
            _scientific_runtime_authorities=authorities,
            _enable_publication_figure_skill=True,
            _max_total_steps=24,
        ),
        plan=plan,
        context=context,
        agent_context=context,
        long_trajectory_bound=False,
        findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _ = authorities.bind_plan(plan)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, findings)
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)

    assert len(llm.calls) == 1
    assert _step(bound, FROZEN_CLASS_DESCRIPTION_STEP_ID).inputs[-3:] == [
        "table:analysis_cohort",
        LABELS,
        FREEZE,
    ]
    assert signed_trajectory_plan_contract_errors(bound) == []
    review = build_plan_scientific_review(
        context=context,
        plan=bound,
        literature=None,
        figure_strategy=build_article_figure_strategy(context),
        runtime_authority=None,
    )
    assert review.approval_allowed, [finding.code for finding in review.findings]


def test_a_model_coded_trajectory_primary_is_refused_the_comparison_action():
    outline = ProgressivePlanOutline(
        analysis_type="trajectory_clustering",
        cohort_objective="Use the declared source cohort.",
        rationale="Plan the full question before materializing steps.",
        steps=[
            dict(
                step_id="cohort",
                planned_analysis_role="auxiliary",
                module_id="cohort_definition",
                objective="Account for the cohort.",
                depends_on=[],
                variable_names=["stay_id", "death"],
                literature_citation_keys=[],
            ),
            dict(
                step_id="trajectory_classes",
                planned_analysis_role="primary",
                module_id="custom_analysis",
                objective="Cluster fixed-window trajectories.",
                depends_on=["cohort"],
                variable_names=["stay_id", *COORDINATES],
                scientific_action_id="phenotyping.trajectory_feature_clustering",
                literature_citation_keys=["strobe_2007"],
            ),
            dict(
                step_id="outcome_by_class",
                planned_analysis_role="secondary",
                module_id="custom_analysis",
                objective="Describe hospital mortality by class.",
                depends_on=["cohort", "trajectory_classes"],
                variable_names=["stay_id", "death"],
                scientific_action_id=COMPARISON,
                literature_citation_keys=["strobe_2007"],
            ),
        ],
    )
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=["trajectory_clustering"],
            variable_names=[variable.name for variable in _context().variables],
            allowed_literature_citation_keys=ALLOWED,
            target_outcome="death",
        )
    assert caught.value.reason_code == "progressive_outline_trajectory_comparison_unowned"
    # The refusal says where the outcomes are described instead, in the words
    # of the Planner's own outline contract, so a retry does not move them
    # into the primary's inputs.
    assert TRAJECTORY_OUTCOME_DESCRIPTION_RULE in caught.value.details["message"]
