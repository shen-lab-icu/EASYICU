"""Physical columns do not override the extraction owner's negative authority."""

import pytest

from easyicu.outcome_availability import OUTCOME_CONCEPT_SUPPORTED_DATABASES
from easyicu.concept.availability_signal import ConceptAvailabilityRecord
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.concept_availability import (
    concept_database_availability_from_load_record,
    explain_concept_availability,
)
from easyicu.research_agent.gates.plan_declared_inputs import declared_raw_input_plan_findings
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload
from easyicu.research_agent.research_context.typed import resolved_raw_input_contracts
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _outline_payload,
    _payload,
)


def _with_source(*, concept="icu_readmission", database="miiv", derived=False):
    context = _context()
    context.cohort.database = database
    context.variables.append(ConceptDescriptor(
        name="legacy_column", dtype="int64",
        source_concept=None if derived else concept,
        derived_from_concepts=[concept] if derived else [],
        source_databases=["miiv"],
        observed_domain={"is_binary": True, "levels": [0, 1], "n_unique": 2},
        missingness={"n_missing": 0, "n_total": 120, "fraction_missing": 0.0},
    ))
    return context


@pytest.mark.parametrize("concept", tuple(OUTCOME_CONCEPT_SUPPORTED_DATABASES))
@pytest.mark.parametrize("database", ("miiv", "eicu", "miiv_demo"))
def test_catalog_derived_availability_uses_the_execution_owner(concept, database):
    cell = explain_concept_availability(concept=concept, database=database)
    supported = database in OUTCOME_CONCEPT_SUPPORTED_DATABASES[concept]
    assert cell.database == database
    assert cell.available is supported
    assert cell.structural_unavailable is (not supported)
    assert cell.direct_source is False
    assert cell.reason == (
        "derived_outcome_supported" if supported
        else "outcome_concept_structurally_unavailable"
    )


def test_database_alias_keeps_supported_demo_outcomes_available():
    cell = explain_concept_availability(concept="mort_28d", database="MIMIC-IV_demo")
    assert cell.database == "miiv_demo"
    assert cell.available and not cell.direct_source


@pytest.mark.parametrize("derived", (False, True))
def test_negative_source_receipt_reaches_both_planner_projections(derived):
    context = _with_source(derived=derived)
    original = context.model_dump_json()
    payload = outbound_safe_context_payload(context, variable_names=["legacy_column"])
    cards = ProgressivePlannerAgent._retrieved_data_cards(context, ["legacy_column"])
    for card in (payload["variables"][0], cards[0]):
        receipt = card["source_unavailability"][0]
        assert receipt["concept_id"] == "icu_readmission"
        assert receipt["database"] == "miiv"
        assert receipt["reason_code"] == "outcome_concept_structurally_unavailable"
    assert context.model_dump_json() == original


@pytest.mark.parametrize("derived", (False, True))
def test_nonmissing_renamed_column_is_refused_by_the_real_consumer(derived):
    context = _with_source(derived=derived)
    original = context.model_dump_json()
    with pytest.raises(ValueError, match="outcome_concept_structurally_unavailable"):
        resolved_raw_input_contracts(context, ["legacy_column"])
    findings = declared_raw_input_plan_findings(
        plan=AnalysisPlan(research_question="test", steps=[
            AnalysisStep(step_id="baseline", intent="describe", inputs=["legacy_column"]),
        ]), context=context,
    )
    assert findings[0].detail["reason"] == "declared_raw_input_structurally_unavailable"
    assert findings[0].detail["unavailable_inputs"] == ["legacy_column"]
    assert "icu_readmission" in findings[0].message
    assert context.model_dump_json() == original


@pytest.mark.parametrize("concept", ("mort_28d", "followup_days_28d", "local_custom_measure"))
def test_supported_and_unclassified_columns_are_not_blanket_blocked(concept):
    context = _with_source(concept=concept)
    result = resolved_raw_input_contracts(context, ["legacy_column"])
    assert "legacy_column" in result["contracts"]
    card = outbound_safe_context_payload(context)["variables"][-1]
    assert "source_unavailability" not in card


def test_actual_database_outranks_the_catalog_list_of_possible_sources():
    context = _with_source(concept="mort_28d", database="eicu")
    with pytest.raises(ValueError, match="outcome_concept_structurally_unavailable"):
        resolved_raw_input_contracts(context, ["legacy_column"])


def test_saved_skeleton_is_refused_at_the_exact_raw_input_boundary():
    payload = _payload()
    payload["steps"][1]["raw_inputs"].append("legacy_column")
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload),
            context=_with_source(),
        )
    assert caught.value.reason_code == "progressive_raw_input_structurally_unavailable"
    assert caught.value.step_id == payload["steps"][1]["step_id"]
    assert caught.value.path == "raw_inputs"
    assert "icu_readmission" in str(caught.value)


def test_outline_refuses_unsupported_inputs_before_materializing_steps():
    context = _with_source()
    payload = _outline_payload()
    payload["steps"][1]["variable_names"].append("legacy_column")
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(payload),
            analysis_types=(payload["analysis_type"],),
            variable_names=tuple(variable.name for variable in context.variables),
            allowed_literature_citation_keys=(), article_context=context,
        )
    assert caught.value.reason_code == "progressive_outline_input_structurally_unavailable"
    assert caught.value.step_id == payload["steps"][1]["step_id"]
    assert caught.value.path == "variable_names"


@pytest.mark.parametrize("anchor", (False, True))
def test_preapproval_review_blocks_and_routes_source_defects(anchor):
    context = _with_source()
    if anchor:
        context = context.model_copy(update={"target_outcome": "legacy_column"})
    plan = AnalysisPlan(research_question=context.research_question, steps=[
        AnalysisStep(step_id="baseline", intent="describe", inputs=["legacy_column"]),
    ])
    review = build_plan_scientific_review(context=context, plan=plan)
    finding = next(item for item in review.findings if item.code == "PLAN_INPUT_STRUCTURALLY_UNAVAILABLE")
    assert not review.approval_allowed
    assert finding.severity == "blocker"
    assert finding.remediation_route == ("runtime_capability" if anchor else "agent_plan_revision")
    assert "legacy_column" in finding.message and "icu_readmission" in finding.message


def test_unused_legacy_column_is_retained_without_blocking_an_unrelated_plan():
    context = _with_source()
    plan = AnalysisPlan(research_question=context.research_question, steps=[
        AnalysisStep(step_id="baseline", intent="describe", inputs=["age_years"]),
    ])
    assert declared_raw_input_plan_findings(plan=plan, context=context) == []
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "PLAN_INPUT_STRUCTURALLY_UNAVAILABLE" not in {f.code for f in review.findings}
    assert context.variables[-1].name == "legacy_column"


def test_runtime_observation_record_cannot_erase_structural_source_prohibition():
    record = ConceptAvailabilityRecord(
        concept="icu_readmission", database="miiv", reason="mapped_present", n_rows=120,
        sources_defined=("legacy_export",),
    )
    cell = concept_database_availability_from_load_record(record)
    assert cell.status == "blocked" and not cell.available
    assert cell.structural_unavailable and not cell.direct_source
    assert cell.reason == "outcome_concept_structurally_unavailable"
    assert cell.runtime_reason == "mapped_present"


@pytest.mark.parametrize("inputs", (["unknown", "legacy_column"], ["legacy_column", "unknown"]))
def test_missing_and_prohibited_names_keep_distinct_causes(inputs):
    plan = AnalysisPlan(research_question="test", steps=[
        AnalysisStep(step_id="baseline", intent="describe", inputs=inputs),
    ])
    findings = declared_raw_input_plan_findings(plan=plan, context=_with_source())
    by_reason = {finding.detail["reason"]: finding for finding in findings}
    assert by_reason["declared_raw_input_structurally_unavailable"].detail["unavailable_inputs"] == ["legacy_column"]
    assert by_reason["declared_raw_input_unresolvable"].detail["unresolvable_inputs"] == ["unknown"]
    assert "lacks a context descriptor" in by_reason["declared_raw_input_unresolvable"].message


def test_typed_materialization_does_not_override_a_source_policy_withdrawal(tmp_path, monkeypatch):
    import easyicu.outcome_availability as source_owner
    from tests.research_agent.planning.test_research_context_v2_authority_join import (
        _prepare_typed_run,
    )

    _, _, context, *_ = _prepare_typed_run(tmp_path)
    # The fully validated typed fixture currently supports lactate. Simulate a
    # future owner withdrawal, not corrupt metadata or an unavailable file.
    assert "lact_max" in resolved_raw_input_contracts(context, ["lact_max"])["contracts"]
    original = context.model_dump_json()
    monkeypatch.setattr(source_owner, "OUTCOME_CONCEPT_SUPPORTED_DATABASES", {
        **source_owner.OUTCOME_CONCEPT_SUPPORTED_DATABASES, "lact": frozenset(),
    })
    with pytest.raises(ValueError, match="outcome_concept_structurally_unavailable"):
        resolved_raw_input_contracts(context, ["lact_max"])
    assert context.model_dump_json() == original
