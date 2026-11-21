"""The plan card says which population condition the system added from the question.

When the Planner's attempts leave a recorded condition the question restricts
its patients to unaccounted for, the host states it itself (S1757).  The plan
review then shows that the system added it, from the exact plan under review:
the host-stated record on the sealed skeleton cohort.  Evidence that does not
bind the reviewed plan shows nothing, and a Planner cannot write the record.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.webserver.pi_copilot.population_spec_review import host_stated_population
from tests.support.bound_population_review import bound_population_review

_ADDED = {"quote": "AKI", "concept": "aki", "start_hours": 0.0, "end_hours": 24.0}


def test_the_review_shows_the_condition_the_host_stated_for_this_plan(tmp_path) -> None:
    root, pending = bound_population_review(tmp_path, cohort_fields={"host_stated_population": [_ADDED]})

    assert host_stated_population(run_dir=root, pending=pending) == [_ADDED]


def test_a_plan_the_planner_stated_alone_shows_nothing(tmp_path) -> None:
    root, pending = bound_population_review(tmp_path)

    assert host_stated_population(run_dir=root, pending=pending) == []


@pytest.mark.parametrize("filename", ["progressive_plan_skeleton.json", "progressive_planning_authority.json"])
def test_evidence_that_does_not_bind_the_plan_shows_nothing(tmp_path, filename) -> None:
    root, pending = bound_population_review(tmp_path, cohort_fields={"host_stated_population": [_ADDED]})
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")

    assert host_stated_population(run_dir=root, pending=pending) == []
    assert host_stated_population(run_dir=root, pending=SimpleNamespace(requests=[])) == []


def test_the_record_is_not_offered_to_a_planner_and_absent_when_empty() -> None:
    schema = ProgressiveCohortIntent.model_json_schema()
    assert "host_stated_population" not in schema.get("properties", {})
    plain = ProgressiveCohortIntent(name="all stays", selection_mode="all_input_rows", inclusion=[], exclusion=[])
    assert "host_stated_population" not in plain.model_dump(mode="json")
    marked = plain.model_copy(update={"host_stated_population": [_ADDED]})
    assert marked.model_dump(mode="json")["host_stated_population"] == [_ADDED]
