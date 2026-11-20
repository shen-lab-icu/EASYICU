"""A reviewed candidate's unapplied population criterion survives its execution re-plan (S1751).

UX on rc_512: the candidate could not apply 「成人 ICU 患者」 (a condition on age)
and the researcher accepted the omission; the execution run planned again and
its plan, report and paper no longer named it.  The host now carries each such
criterion verbatim: after the execution plan is compiled, one that no cohort
predicate reads and that the plan does not list is appended to the plan's
unapplied criteria, which the population review, the report and the paper read.
Synthetic plans and contexts only.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.planning import carried_population
from easyicu.research_agent.planning.carried_population import (
    bind_carried_population_limitations,
    candidate_population_limitations,
)
from easyicu.research_agent.planning.cohort_contract import CohortDefinition, ConceptPredicate
from easyicu.research_agent.reporting.population_selection import analyzed_population, writer_population_block
from tests.support.result_append import completed_source

_ADULTS = "成人 ICU 患者"
_CARRIED = {"source": "carried_from_candidate", "criteria": [{"quote": _ADULTS, "concepts": ["age"]}]}


def _plan(tmp_path, **cohort):
    _wrapper, _run, plan, context, _ = completed_source(tmp_path)
    fields = {"name": "source_cohort", "selection_mode": "all_input_rows", **cohort}
    return plan.model_copy(update={"cohort": CohortDefinition(**fields)}), context


def test_an_execution_plan_that_omits_the_criterion_still_lists_it(tmp_path) -> None:
    plan, context = _plan(tmp_path)

    bound = bind_carried_population_limitations(plan, _CARRIED)

    assert bound.cohort.unapplied_population_criteria == (_ADULTS,)
    # The report and the paper read the plan's unapplied criteria.
    population = analyzed_population(plan=bound, context=context)
    assert population.unapplied_population_criteria == (_ADULTS,)
    assert _ADULTS in writer_population_block(population)


def test_a_criterion_the_execution_plan_applies_or_lists_is_not_added(tmp_path) -> None:
    age = ConceptPredicate.from_dict({
        "concept_id": "age", "op": ">=", "value": 18, "aggregation": "first",
        "time_window": {"anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": float("inf")},
    })
    applied, _ = _plan(tmp_path / "applied", selection_mode="predicate_filtered", inclusion=(age,))
    listed, _ = _plan(tmp_path / "listed", unapplied_population_criteria=(_ADULTS,))

    assert bind_carried_population_limitations(applied, _CARRIED).cohort.unapplied_population_criteria == ()
    assert bind_carried_population_limitations(listed, _CARRIED).cohort.unapplied_population_criteria == (_ADULTS,)


def test_a_criterion_without_concepts_is_kept_unless_listed(tmp_path) -> None:
    age = ConceptPredicate.from_dict({
        "concept_id": "age", "op": ">=", "value": 18, "aggregation": "first",
        "time_window": {"anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": float("inf")},
    })
    plan, _ = _plan(tmp_path, selection_mode="predicate_filtered", inclusion=(age,))
    words_only = {"source": "carried_from_candidate", "criteria": [{"quote": "ARDS 患者", "concepts": []}]}

    assert bind_carried_population_limitations(plan, words_only).cohort.unapplied_population_criteria == ("ARDS 患者",)


def test_a_plan_without_a_cohort_is_refused_not_run_without_it(tmp_path) -> None:
    plan, _ = _plan(tmp_path)
    with pytest.raises(ValueError, match="no cohort"):
        bind_carried_population_limitations(plan.model_copy(update={"cohort": None}), _CARRIED)
    assert bind_carried_population_limitations(plan, None) is plan


def test_the_candidate_record_carries_its_words_and_reviewed_concepts(monkeypatch) -> None:
    assert candidate_population_limitations(cohort_unapplied=[]) is None
    assert candidate_population_limitations(cohort_unapplied=[" 成人  ICU 患者 "]) == {
        "source": "carried_from_candidate", "criteria": [{"quote": _ADULTS, "concepts": []}],
    }
    reviewed = {"id": "c1", "quote": _ADULTS, "source": "study_wording", "role": "include",
                "kind": "condition_present", "concepts_all_of": ["age"], "window": None, "missing_policy": "exclude"}
    monkeypatch.setattr(carried_population, "unapplied_inclusions",
                        lambda request: ({"criterion": reviewed, "disposition": "not_applied"},))
    assert candidate_population_limitations(cohort_unapplied=[_ADULTS], review_requests=[object()]) == _CARRIED


def test_the_config_issues_it_only_with_human_plan_review(tmp_path) -> None:
    with pytest.raises(ValueError, match="require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, carried_unapplied_population_criteria=_CARRIED)
    config = PipelineConfig(workdir=tmp_path, require_human_plan_review=True,
                            carried_unapplied_population_criteria=_CARRIED)
    carried = config.carried_unapplied_population_criteria
    assert carried["source"] == "carried_from_candidate"
    assert [(dict(c)["quote"], tuple(dict(c)["concepts"])) for c in carried["criteria"]] == [(_ADULTS, ("age",))]
    assert dataclasses.replace(config, carried_unapplied_population_criteria=None).carried_unapplied_population_criteria is None


def test_the_execution_plan_is_bound_by_the_pipeline_and_issued_by_the_host() -> None:
    source = inspect.getsource(pipeline_module)
    assert "plan = bind_carried_population_limitations(plan, self._config.carried_unapplied_population_criteria)" in source
    from easyicu.webserver import agent_pipeline_runs

    host = inspect.getsource(agent_pipeline_runs)
    assert "carried_unapplied_population_criteria=carried_population_limitations," in host
    assert "carried_population_limitations=_candidate_population_limitations(candidate_cohort, checkpoint)," in host


def test_an_absent_record_leaves_archived_digests_and_a_present_one_is_bound(tmp_path) -> None:
    plain = PipelineConfig(workdir=tmp_path, require_human_plan_review=True)
    carried = dataclasses.replace(plain, carried_unapplied_population_criteria=_CARRIED)

    assert "carried_unapplied_population_criteria" not in plain.canonical_payload()
    assert "carried_unapplied_population_criteria" in carried.canonical_payload()
    assert carried.canonical_digest() != plain.canonical_digest()
    restored = PipelineConfig.from_recovery_payload(
        carried.recovery_payload(), expected_digest=carried.canonical_digest()
    )
    assert restored.canonical_digest() == carried.canonical_digest()
