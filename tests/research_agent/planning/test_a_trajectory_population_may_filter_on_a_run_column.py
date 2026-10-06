"""A trajectory population may filter on a column the run materialized.

A trajectory plan states whom it studies as typed cohort predicates, and the
host seals them into the fixed-window design.  A predicate may read a column
the bound export materialized -- a derived flag no packaged dictionary
defines -- which only the run's roster makes known.  Binding a draft to the
signed owners parsed the sealed population outside that roster, so such a
study failed schema validation at ``cohort`` before any review, and the
development projection failed the same way.  Each now knows the population's
own concepts while it is parsed; the run's cohort lock still decides, against
the run's roster, whether the column exists.  Fixtures are generic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort.schema import write_locked_cohort_definition
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    normalize_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortSchemaError,
    cohort_concept_id_scope,
    concept_id_exists,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

_QUESTION = (
    "Among adults with condition-a, do first-24h physiologic trajectories form "
    "distinct classes?"
)
#: A column the run materialized; no packaged dictionary defines it.
_RUN_COLUMN = "derived_condition_flag"


def _predicate(concept: str, op: str, value: float, aggregation: str) -> dict:
    return {
        "concept_id": concept,
        "time_window": {
            "anchor": "icu_admit",
            "start_offset_hours": 0,
            "end_offset_hours": 24,
        },
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


_ADULT = _predicate("age", ">=", 18, "first")
_FLAGGED = _predicate(_RUN_COLUMN, "==", 1, "max")


def _authority(**population):
    design = normalize_trajectory_design(
        {
            "coordinate_concepts": ["sofa2_resp", "sofa2_cardio"],
            "window_end_hours": 24,
            "grid_width_hours": 4,
            "population": population,
        }
    )
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            load_trajectory_design(design), protocol_content_sha256="1" * 64
        )
    )


def _draft(authority) -> AnalysisPlan:
    """A Planner draft naming the signed owners; its own cohort is no authority."""

    with cohort_concept_id_scope([_RUN_COLUMN]):
        owners = authority.development_execution_only_plan(research_question=_QUESTION)
    return AnalysisPlan.model_validate(
        {
            **owners.model_dump(mode="json"),
            "cohort": {"name": "primary", "selection_mode": "all_input_rows"},
        }
    )


def _composed(authority) -> ScientificRuntimeAuthorities:
    return ScientificRuntimeAuthorities(trajectory=authority, current_case=None)


def _canonical(predicates) -> list[dict]:
    return [predicate.to_dict() for predicate in predicates]


@pytest.fixture(autouse=True)
def _the_column_is_known_only_inside_its_run():
    # Outside a run's roster the column is unknown, before and after each test:
    # binding knows it only while the plan is parsed.
    assert not concept_id_exists(_RUN_COLUMN)
    yield
    assert not concept_id_exists(_RUN_COLUMN)


@pytest.mark.parametrize(
    "population",
    [
        {"inclusion": [_ADULT, _FLAGGED]},
        {"inclusion": [_ADULT], "exclusion": [_FLAGGED]},
    ],
    ids=["inclusion", "exclusion"],
)
def test_binding_a_draft_states_a_population_on_a_run_column(population) -> None:
    authority = _authority(**population)

    bound, [finding] = _composed(authority).bind_plan(_draft(authority))

    assert bound.cohort.selection_mode == "predicate_filtered"
    assert _canonical(bound.cohort.inclusion) == population["inclusion"]
    assert _canonical(bound.cohort.exclusion) == population.get("exclusion", [])
    assert finding.detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )
    authority.validate_plan(bound)


def test_the_development_projection_states_a_population_on_a_run_column() -> None:
    authority = _authority(inclusion=[_ADULT], exclusion=[_FLAGGED])

    plan, finding = _composed(authority).development_execution_only_plan(
        research_question=_QUESTION
    )

    assert _canonical(plan.cohort.inclusion) == [_ADULT]
    assert _canonical(plan.cohort.exclusion) == [_FLAGGED]
    assert finding.detail["analysis_only"] is True
    authority.validate_plan(plan)


def test_the_run_roster_still_decides_whether_the_column_exists(tmp_path: Path) -> None:
    authority = _authority(inclusion=[_ADULT, _FLAGGED])
    bound, _ = _composed(authority).bind_plan(_draft(authority))
    lock = {
        "run_dir": tmp_path,
        "plan": bound,
        "evidence": EvidenceStore(tmp_path),
        "prompt_pack_version": "test",
        "llm_signature": "mock",
    }

    # A run that did not materialize the column cannot lock this cohort.
    with pytest.raises(CohortSchemaError, match=f"unknown concept_id: {_RUN_COLUMN}"):
        write_locked_cohort_definition(**lock, cohort_concept_ids=("age",))
    assert not list(tmp_path.glob("cohort_locked*.json"))

    path = write_locked_cohort_definition(
        **lock, cohort_concept_ids=("age", _RUN_COLUMN)
    )
    assert f'"concept_id": "{_RUN_COLUMN}"' in path.read_text(encoding="utf-8")
