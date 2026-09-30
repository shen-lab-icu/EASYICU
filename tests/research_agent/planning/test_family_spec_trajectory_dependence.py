"""The fixed-window trajectory template sources its dependence decision.

When repeated ICU stays of one patient are possible, a plan must cite the
method source that governs how that dependence is handled
(``required_method_layers_for_context``).  The landmark template's primary
binds it.  The trajectory template bound only estimand, missing data,
reporting, robustness and time zero, so on an all-stay source whose bundle
carried such a source (the STROBE card) the sealed suite was refused at the
compiler boundary (``progressive_final_method_layer_unbound``) before any plan
existed.  The fixtures are generic synthetic variables.
"""

from __future__ import annotations

import json

from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.scientific_review import (
    required_method_layers_for_context,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

_COORDINATES = ("sofa2_cardio", "sofa2_renal", "lact")
_LABELS = {
    "death": "In-hospital death",
    "sofa2_resp": "SOFA-2 respiratory score",
    "sofa2_cardio": "SOFA-2 cardiovascular score",
    "sofa2_renal": "SOFA-2 renal score",
    "lact": "Lactate (mmol/L)",
}


def _disclosure() -> str:
    design = load_trajectory_design(
        {
            "coordinate_concepts": ["sofa2_resp", *_COORDINATES],
            "descriptive_only_concepts": [],
        }
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="d" * 64)
    )
    return ScientificRuntimeAuthorities(
        trajectory=authority, current_case=None
    ).planning_contract_context()


def _context() -> ResearchContext:
    # A zero-row planning catalog of ICU stays without patient identity:
    # repeated stays of one patient cannot be ruled out.
    provenance = {
        "analysis_unit": "icu_stay",
        "patient_identity_available": False,
        "stay_id_columns": ["stay_id"],
        "patient_id_columns": [],
        "evidence_stage": "metadata_only_planning",
        "patient_rows_read": False,
    }
    return ResearchContext(
        research_question=(
            "Which organ-dysfunction trajectory classes emerge over the first 72 h of an "
            "ICU stay, and how does in-hospital mortality differ by class?"
        ),
        cohort=CohortDescriptor(
            cohort_name="trajectory_synthetic", database="miiv", n_stays=0,
            id_columns=["stay_id"], outcome_columns=["death"], requested_outcome_columns=["death"],
            provenance=provenance,
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            *[
                ConceptDescriptor(
                    name=concept, description=f"{concept} coordinate",
                    role=VariableRole.ORDINAL_SCORE, dtype="float64",
                    analysis_window="icu_admission[0,72]h",
                )
                for concept in ("sofa2_resp", *_COORDINATES)
            ],
            ConceptDescriptor(
                name="death", description="in-hospital mortality", role=VariableRole.OUTCOME,
                dtype="float64", observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
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


def _plan(context: ResearchContext, citations: tuple[str, ...]):
    disclosure = _disclosure()
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=citations,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=disclosure,
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": _LABELS[key]}
            for key in request.required_reader_label_keys
        ],
        "comparator_applications": [],
        "roster_decision_note": "The sealed suite owns every coordinate.",
    }
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=citations,
        direct_comparator_literature_keys=(),
        comparison_literature_keys=(),
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context=disclosure,
        required_primary_cohort_selection_mode="all_input_rows",
    )
    assert len(llm.calls) == 1
    return result.output


def test_an_all_stay_trajectory_plan_cites_how_its_dependence_is_handled() -> None:
    context = _context()
    assert "dependence" in required_method_layers_for_context(context)

    plan = _plan(context, ("strobe_2007", "record_2015"))

    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    bound = {
        binding.citation_key: set(binding.design_elements)
        for binding in primary.literature_design_bindings
    }
    assert "dependence" in bound["strobe_2007"]
