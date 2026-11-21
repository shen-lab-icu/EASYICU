"""Pre-review ownership reads a cohort filtered on a column the run materialized.

A derived condition column (for example a sepsis flag) is known to cohort
validation only through the run's registered roster. The review-time check
must use that roster, as Execute does, or it mistakes its own typed cohort for
a step that needs model-generated code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    load_materialized_analysis_cohort_result,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from easyicu.research_agent.execution.owner_declaration import (
    HostPlanStepNeedsSandbox,
    require_host_rendered_plan,
)
from easyicu.research_agent.intake.materialized_metadata import (
    implementation_bundle_sha256,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.planning.cohort_contract import (
    cohort_concept_id_scope,
    concept_id_exists,
    sealed_cohort_concept_ids,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import (
    COHORT_DEFINITION_COHORT_OUTPUT,
    COHORT_DEFINITION_FLOW_OUTPUT,
    AnalysisPlan,
    AnalysisStep,
    CohortDefinitionSpec,
)
from tests.support.typed_export import typed_export

COLUMN = "lact_max"
QUESTION = "Which stays with a recorded lactate enter the analysis cohort?"


def _definition() -> CohortDefinition:
    return CohortDefinition(
        name="lactate_recorded",
        inclusion=(
            ConceptPredicate(
                concept_id=COLUMN,
                time_window=TimeWindow(
                    anchor="icu_admit", start_offset_hours=0, end_offset_hours=24
                ),
                aggregation="max",
                op=">=",
                value=0,
            ),
        ),
    )


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=QUESTION,
        cohort=_definition(),
        steps=[
            AnalysisStep(
                step_id="01_define_analysis_cohort",
                intent="Define the analytic cohort and report its attrition.",
                inputs=[],
                expected_outputs=[
                    COHORT_DEFINITION_COHORT_OUTPUT,
                    COHORT_DEFINITION_FLOW_OUTPUT,
                ],
                method="cohort_definition_and_attrition",
                planned_analysis_role="auxiliary",
                cohort_definition_spec=CohortDefinitionSpec(
                    identity_column="stay_id",
                    eligibility_criteria=[
                        {"criterion_id": "lactate", "description": "Lactate recorded"}
                    ],
                ),
            )
        ],
    )


def _run(tmp_path: Path, *, register_context: bool):
    source = typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    assert stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    ) is not None
    context = build_research_context(
        research_question=QUESTION,
        cohort=parent_path,
        cohort_name="lactate",
        database="miiv",
        target_outcome="death",
    )
    roster = sealed_cohort_concept_ids(context)
    evidence = EvidenceStore(run_dir)
    if register_context:
        (run_dir / "research_context.json").write_text(
            context.model_dump_json(indent=2), encoding="utf-8"
        )
        evidence.register_file(
            kind="log",
            description="Registered research_context.",
            source_path=run_dir / "research_context.json",
            evidence_id="research_context",
            producer="pipeline",
            generation_mode="system",
        )
    with cohort_concept_id_scope(roster):
        plan = _plan()
        write_locked_cohort_definition(
            run_dir=run_dir, plan=plan, evidence=evidence,
            prompt_pack_version="test", llm_signature="test",
            cohort_concept_ids=roster,
        )
        result = materialize_locked_analysis_cohort(
            run_dir=run_dir, plan=plan, universe_path=parent_path,
            cohort_concept_ids=roster,
        )
    assert result["status"] == "applied"
    assert COLUMN in roster and not concept_id_exists(COLUMN)
    return run_dir, plan


def test_review_owns_a_cohort_on_a_registered_run_column(tmp_path: Path) -> None:
    run_dir, plan = _run(tmp_path, register_context=True)

    # No caller scope: the owner reads the run's registered roster itself.
    assert load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
    require_host_rendered_plan(plan=plan, run_dir=run_dir)
    assert not concept_id_exists(COLUMN), "the roster must not leak past the check"


def test_without_a_registered_context_the_column_grants_nothing(tmp_path: Path) -> None:
    run_dir, plan = _run(tmp_path, register_context=False)

    with pytest.raises(HostPlanStepNeedsSandbox):
        require_host_rendered_plan(plan=plan, run_dir=run_dir)
