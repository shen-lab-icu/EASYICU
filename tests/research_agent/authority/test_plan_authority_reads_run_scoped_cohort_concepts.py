"""A plan whose cohort names a materialized column stays bound to its evidence.

Dev9 M3: the reviewed cohort filtered on ``sep3_sofa1_max``, a column the run's
scoped concept registration makes known.  Finalization re-read the registered
plan outside that scope, the read failed as an unknown concept, and the run
stopped with "current analysis plan is not bound to immutable EvidenceStore
authority" after every analysis step had finished.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_input_closure import (
    resolve_registered_plan_authority,
)
from easyicu.research_agent.planning.cohort_contract import (
    cohort_concept_id_scope,
    cohort_definition_concept_ids,
    concept_id_exists,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

COLUMN = "fixture_score_max"


def _plan(value: float) -> AnalysisPlan:
    with cohort_concept_id_scope([COLUMN]):
        return AnalysisPlan(
            research_question="Describe the selected stays.",
            revision=1,
            cohort={
                "name": "scoped_fixture",
                "inclusion": [
                    {
                        "concept_id": COLUMN,
                        "time_window": {
                            "anchor": "icu_admission",
                            "start_offset_hours": 0.0,
                            "end_offset_hours": 24.0,
                        },
                        "aggregation": "any",
                        "op": "==",
                        "value": value,
                    }
                ],
                "exclusion": [],
                "selection_mode": "predicate_filtered",
            },
            steps=[
                AnalysisStep(
                    step_id="describe",
                    intent="Describe the cohort.",
                    method="descriptive",
                    expected_outputs=["table:summary"],
                )
            ],
        )


def test_finalization_binds_a_plan_whose_cohort_names_a_scoped_column(
    tmp_path: Path,
) -> None:
    assert not concept_id_exists(COLUMN)
    plan = _plan(1.0)
    assert cohort_definition_concept_ids(plan.cohort) == (COLUMN,)
    plan_path = tmp_path / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(tmp_path)
    evidence.register_file(
        kind="log",
        description="Reviewed analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="pipeline",
        generation_mode="system",
    )

    # Outside any scope, as finalization and the partial-manifest flush run.
    authority = resolve_registered_plan_authority(
        run_dir=tmp_path, evidence=evidence, plan=plan, plan_path=plan_path
    )

    assert authority.evidence_id == "analysis_plan"
    assert not concept_id_exists(COLUMN)
    # The scope admits only the current plan's own ids; a plan that differs
    # from the registered one is still not bound.
    with pytest.raises(ValueError, match="not bound"):
        resolve_registered_plan_authority(
            run_dir=tmp_path, evidence=evidence, plan=_plan(0.0), plan_path=plan_path
        )
