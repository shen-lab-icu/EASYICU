"""Shared synthetic fixtures; no recorded research inputs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow, materialize_locked_analysis_cohort
from easyicu.research_agent.intake.materialized_metadata import implementation_bundle_sha256, stage_materialized_cohort_authority
from easyicu.research_agent.schema import COHORT_DEFINITION_COHORT_OUTPUT, COHORT_DEFINITION_FLOW_OUTPUT, AnalysisPlan, AnalysisStep
from tests.support.typed_export import typed_export as _typed_export

def _definition() -> CohortDefinition:
    return CohortDefinition(
        name="older_adults",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=24,
                ),
                aggregation="first",
                op=">=",
                value=55,
            ),
        ),
    )


@pytest.fixture
def typed_run(tmp_path: Path):
    """A run directory holding a real typed (``/2``) materialization.

    Built with the host's own publisher, from a typed export -- the same path
    every recorded run takes.  Returns ``(run_dir, plan)``.
    """

    source = _typed_export(tmp_path / "export")
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
    parent = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert parent is not None

    plan = SimpleNamespace(cohort=_definition(), steps=[])
    result = materialize_locked_analysis_cohort(
        run_dir=run_dir, plan=plan, universe_path=parent_path
    )
    assert result["status"] == "applied"
    return run_dir, plan


def _sealable_plan():
    return AnalysisPlan(
        research_question="Which older adults enter the analysis cohort?",
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
            )
        ],
    )
