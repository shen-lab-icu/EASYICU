"""The host compiles the Planner's spec beside the Planner's own cohort and records both.

The audit compiles the spec the Planner wrote beside its cohort predicates
and compares the two predicate by predicate.  In step 2a the predicates
selected the rows, so each difference could be explained before the cohort
was compiled from a spec; since step 2b the spec decides the plan's cohort
and a difference is a typed finding
(``test_a_stated_population_decides_the_plans_cohort``).  Two predicates that select the same
rows are not a difference: the same threshold written as 1 or 1.0, ICU
admission under either of its anchor names, a window on a column that holds
one value per stay, and no predicates with or without a stated mode.  A
criterion on the design's exposure or outcome is listed apart.  A spec its
owner refuses is recorded with the owner's errors, and an audit that fails,
or cannot be written, is recorded or skipped while planning goes on.
Contexts, specs and plans are synthetic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.orchestration.progressive_planning import (
    run_progressive_planner,
)
from easyicu.research_agent.planning import population_shadow
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.planning.population_shadow import (
    POPULATION_SHADOW_AUDIT_FILENAME,
    population_shadow_audit,
    write_population_shadow_audit,
)
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
_DAY = "icu_admission[0,24]h"


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Which stays does the study include?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=100,
            id_columns=["stay_id"],
            outcome_columns=["outcome_flag"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            # One value per stay: the context records no window for it.
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years"
            ),
            ConceptDescriptor(
                name="outcome_flag",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain=_BINARY,
            ),
            # A status summarized over the first day.
            ConceptDescriptor(
                name="marker_flag",
                role=VariableRole.OTHER,
                dtype="int64",
                analysis_window=_DAY,
                observed_domain=_BINARY,
            ),
            ConceptDescriptor(
                name="exposure_flag",
                role=VariableRole.INTERVENTION,
                dtype="int64",
                analysis_window=_DAY,
                observed_domain=_BINARY,
            ),
        ],
        primary_exposure="exposure_flag",
        target_outcome="outcome_flag",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(
                {
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "ICU admission",
                        "hours": 24.0,
                    }
                }
            )
        ),
    )


def _criterion(index: int, **fields: Any) -> dict[str, Any]:
    return {
        "id": f"c{index}",
        "quote": f"stated restriction {index}",
        "source": "question",
        "role": "include",
        **fields,
    }


_ADULTS = _criterion(1, kind="age_years", min_years=18)
_MARKED = _criterion(
    2,
    kind="condition_present",
    concepts_all_of=["marker_flag"],
    window={"start_hours": 0, "end_hours": 24},
)


def _predicate(concept, start, end, aggregation, op, value, anchor="icu_admission"):
    return {
        "concept_id": concept,
        "time_window": {
            "anchor": anchor,
            "start_offset_hours": start,
            "end_offset_hours": end,
        },
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


def _audit(spec, plan_cohort, *, design=("exposure_flag", "outcome_flag")):
    return population_shadow_audit(
        spec=spec,
        context=_context(),
        plan_cohort=plan_cohort,
        time_zero_hours=24.0,
        design_concepts=design,
    )


def _plan(*inclusion, exclusion=()) -> dict:
    return {
        "name": "primary",
        "inclusion": list(inclusion),
        "exclusion": list(exclusion),
    }


def _counts(audit) -> dict[str, int]:
    comparison = audit["comparison"]
    return {
        part: sum(len(comparison[side][part]) for side in ("inclusion", "exclusion"))
        for part in ("both", "equivalent", "compiled_only", "plan_only")
    }


@pytest.mark.parametrize(
    "plan_cohort, mode, predicates",
    [
        ({"selection_mode": "all_input_rows"}, "all_input_rows", 0),
        (_plan(_predicate("age", 0, 24, "first", ">=", 18)), "predicate_filtered", 1),
        (None, None, 0),
    ],
    ids=["every-input-row", "predicates", "no-cohort"],
)
def test_a_plan_without_a_spec_is_recorded_with_the_cohort_it_states(
    plan_cohort, mode, predicates
) -> None:
    audit = _audit(None, plan_cohort)

    assert audit["status"] == "no_spec"
    assert audit["time_zero_hours"] == 24.0
    assert audit["design_concepts"] == ["exposure_flag", "outcome_flag"]
    # A cohort of every row needed no spec; one with predicates did, read
    # with the default mode its dict omits.
    assert audit["plan_selection_mode"] == mode
    assert audit["plan_predicate_count"] == predicates


@pytest.mark.parametrize(
    "refused",
    [
        {
            "criteria": [
                {**_ADULTS, "role": "exclude"},
                {**_MARKED, "window": "day one"},
            ]
        },
        [_ADULTS],
        "adults only",
    ],
    ids=["no-criterion-the-owner-reads", "no-object", "words"],
)
def test_a_spec_its_owner_refuses_is_recorded_with_the_owners_errors(refused) -> None:
    audit = _audit(refused, _plan())

    assert audit["status"] == "spec_invalid"
    assert audit["spec"] == refused
    assert audit["errors"]
    assert all(set(error) == {"loc", "type", "msg"} for error in audit["errors"])
    assert "day one" not in json.dumps(audit["errors"])


def test_criteria_nested_under_their_kind_are_recorded_with_the_flat_shape() -> None:
    """A Planner without a schema once wrote each kind's fields under its name."""

    common = ("id", "quote", "source", "role", "kind")

    def nested(criterion: dict[str, Any]) -> dict[str, Any]:
        fields = {key: value for key, value in criterion.items() if key not in common}
        return {**{key: criterion[key] for key in common}, criterion["kind"]: fields}

    written = {"criteria": [nested(_ADULTS), nested(_MARKED)]}

    audit = _audit(written, _plan(_predicate("age", 0.0, "inf", "first", ">=", 18)))

    assert audit["status"] == "spec_invalid"
    assert [(error["loc"], error["type"]) for error in audit["errors"]] == [
        ("criteria.0.age_years", "value_error"),
        ("criteria.1.condition_present", "value_error"),
    ]
    adults, marked = (error["msg"] for error in audit["errors"])
    adult_shape = '{"kind": "age_years", "min_years": "...", "max_years": "..."}'
    marked_shape = (
        '{"kind": "condition_present", "concepts_all_of": "...", "window": "..."}'
    )
    assert f"write {adult_shape}" in adults
    assert f"write {marked_shape}" in marked


def test_the_criteria_the_owner_reads_are_compiled_beside_those_it_refuses() -> None:
    written = {"criteria": [_ADULTS, {**_MARKED, "window": "day one"}]}

    audit = _audit(written, _plan(_predicate("age", 0.0, "inf", "first", ">=", 18)))

    assert audit["status"] == "spec_partly_invalid"
    assert audit["spec"] == written
    assert [row["id"] for row in audit["criteria"]] == ["c1"]
    assert {
        error["loc"].split(".")[:2] == ["criteria", "1"] for error in audit["errors"]
    } == {True}
    assert _counts(audit)["both"] == 1


def test_an_inclusion_the_host_cannot_apply_would_block_the_plan() -> None:
    unapplied = _criterion(
        1, kind="not_typed", why="no kind states the referral it names"
    )

    audit = _audit({"criteria": [unapplied]}, _plan())

    # The rows are the same today; after the switch the plan would stop.
    assert audit["differs"] is False
    assert audit["blocking"] == ["c1"]
    assert audit["would_block"] is True
    assert audit["comparison"]["unapplied_counts"] == {"compiled": 1, "plan": 0}


def test_the_same_predicates_written_alike_are_one() -> None:
    plan = _plan(
        _predicate("age", 0.0, "inf", "first", ">=", 18),
        _predicate("marker_flag", 0, 24, "max", "==", 1.0, anchor="icu_admit"),
    )

    audit = _audit({"criteria": [_ADULTS, _MARKED]}, plan)

    assert audit["status"] == "compiled"
    assert _counts(audit) == {
        "both": 2,
        "equivalent": 0,
        "compiled_only": 0,
        "plan_only": 0,
    }
    assert audit["differs"] is False


def test_a_window_on_a_column_of_one_value_per_stay_is_equivalent() -> None:
    plan = _plan(
        _predicate("age", 0, 24, "first", ">=", 18.0),
        _predicate("marker_flag", 0, 24, "max", "==", 1),
    )

    audit = _audit({"criteria": [_ADULTS, _MARKED]}, plan)

    assert _counts(audit) == {
        "both": 1,
        "equivalent": 1,
        "compiled_only": 0,
        "plan_only": 0,
    }
    assert audit["differs"] is False


@pytest.mark.parametrize(
    "plan_marker",
    [
        _predicate("marker_flag", 0, 12, "max", "==", 1),
        _predicate("marker_flag", 0, 24, "max", ">=", 2),
    ],
    ids=["another-window-on-a-windowed-column", "a-threshold-outside-the-status"],
)
def test_a_predicate_that_selects_other_rows_is_a_difference(plan_marker) -> None:
    audit = _audit({"criteria": [_MARKED]}, _plan(plan_marker))

    assert _counts(audit) == {
        "both": 0,
        "equivalent": 0,
        "compiled_only": 1,
        "plan_only": 1,
    }
    assert audit["differs"] is True


@pytest.mark.parametrize(
    "plan_cohort, mode",
    [(_plan(), "same"), ({"selection_mode": "all_input_rows"}, "differs")],
    ids=["no-predicate", "every-input-row"],
)
def test_a_criterion_the_plan_does_not_apply_is_a_difference(plan_cohort, mode) -> None:
    audit = _audit({"criteria": [_ADULTS, _MARKED]}, plan_cohort)

    assert _counts(audit)["compiled_only"] == 2
    assert audit["comparison"]["selection_mode"] == mode
    assert audit["differs"] is True


@pytest.mark.parametrize(
    "plan_cohort",
    [{}, {"selection_mode": "all_input_rows"}, None],
    ids=["no-mode", "every-input-row", "no-cohort"],
)
def test_no_predicate_on_either_side_selects_every_row_alike(plan_cohort) -> None:
    audit = _audit({"criteria": []}, plan_cohort)

    assert audit["comparison"]["selection_mode"] in {"same", "equivalent"}
    assert audit["differs"] is False


def test_a_criterion_on_the_designs_exposure_is_listed_apart() -> None:
    excluded = _criterion(
        3,
        role="exclude",
        kind="condition_present",
        concepts_all_of=["exposure_flag"],
        window={"start_hours": 0, "end_hours": 24},
    )

    audit = _audit({"criteria": [_ADULTS, excluded]}, _plan())

    assert audit["on_design_concepts"] == ["c3"]
    rows = {row["id"]: row["on_design_concept"] for row in audit["criteria"]}
    assert rows == {"c1": False, "c3": True}


def _plan_object(cohort: dict | None) -> SimpleNamespace:
    return SimpleNamespace(
        analysis_type="association_study",
        steps=[],
        cohort=None if cohort is None else CohortDefinition.from_dict(cohort),
    )


def _intent(spec) -> ProgressiveCohortIntent:
    return ProgressiveCohortIntent(
        name="all", selection_mode="all_input_rows", population_spec=spec
    )


def test_the_audit_is_written_beside_the_plan(tmp_path: Path) -> None:
    path = write_population_shadow_audit(
        tmp_path,
        context=_context(),
        plan=_plan_object({"selection_mode": "all_input_rows"}),
        cohort=_intent({"criteria": [_ADULTS]}),
    )

    assert path == tmp_path / POPULATION_SHADOW_AUDIT_FILENAME
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["status"] == "compiled"
    assert audit["schema_version"] == "easyicu.population_shadow_audit/1"
    assert [row["id"] for row in audit["criteria"]] == ["c1"]


def test_the_writer_reads_the_criteria_the_planner_states_but_does_not_apply(
    tmp_path: Path,
) -> None:
    # The spec decides the plan's cohort, so the audit sets the Planner's own
    # cohort beside it, with the criteria the Planner states but does not apply.
    cohort = ProgressiveCohortIntent(
        name="all",
        selection_mode="all_input_rows",
        population_criteria=[{"criterion": "patients referred by a named service"}],
        population_spec={"criteria": []},
    )

    path = write_population_shadow_audit(
        tmp_path,
        context=_context(),
        plan=_plan_object({"selection_mode": "all_input_rows"}),
        cohort=cohort,
    )

    comparison = json.loads(path.read_text(encoding="utf-8"))["comparison"]
    assert comparison["plan_unapplied"] == ["patients referred by a named service"]
    assert comparison["unapplied_counts"] == {"compiled": 0, "plan": 1}


def test_an_audit_that_fails_is_recorded_and_raises_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(population_shadow, "compile_population", broken)

    path = write_population_shadow_audit(
        tmp_path,
        context=_context(),
        plan=_plan_object(None),
        cohort=_intent({"criteria": [_ADULTS]}),
    )

    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["status"] == "audit_failed"
    assert audit["error_type"] == "RuntimeError"


def test_an_audit_it_cannot_write_is_skipped(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-run"

    assert (
        write_population_shadow_audit(
            missing,
            context=_context(),
            plan=_plan_object(None),
            cohort=_intent(None),
        )
        is None
    )
    assert not missing.exists()


class _Evidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        source = Path(str(kwargs["source_path"]))
        self.records[str(kwargs["evidence_id"])] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        return self.records[str(kwargs["evidence_id"])]


def _plan_with(run_dir: Path, spec: Any):
    foundation = _foundation_payload()
    if spec is not None:
        foundation["foundation"]["cohort"]["population_spec"] = spec
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(foundation),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic cohort")
    return run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=_planner_context(),
        run_dir=run_dir,
        evidence=_Evidence(),
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=lambda _finding: None,
    )


def _written(run_dir: Path) -> dict:
    return json.loads((run_dir / POPULATION_SHADOW_AUDIT_FILENAME).read_text("utf-8"))


def test_planning_writes_the_audit_and_applies_the_cohort_its_spec_states(
    tmp_path: Path,
) -> None:
    spec = {"criteria": []}

    plain = _plan_with(tmp_path / "plain", None)
    stated = _plan_with(tmp_path / "stated", spec)

    assert _written(tmp_path / "plain")["status"] == "no_spec"
    audit = _written(tmp_path / "stated")
    assert audit["status"] == "compiled"
    assert audit["cohort_source"] == "population_spec"
    assert audit["plan_applies_compiled"] is True
    assert audit["spec"] == PopulationSpec.model_validate(spec).model_dump(mode="json")
    # Every input row, as the Planner's own cohort also states.
    assert stated.plan.cohort.to_dict() == plain.plan.cohort.to_dict()
    assert [step.step_id for step in stated.plan.steps] == [
        step.step_id for step in plain.plan.steps
    ]


def test_an_audit_that_fails_does_not_stop_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(population_shadow, "compile_population", broken)

    result = _plan_with(tmp_path / "run", {"criteria": []})

    assert result.plan.steps
    assert _written(tmp_path / "run")["status"] == "audit_failed"
