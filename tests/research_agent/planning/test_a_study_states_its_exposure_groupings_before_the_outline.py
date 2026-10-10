"""A study states the exposure groups it forms before its plan is outlined.

When the host plans exposure groupings, the Planner is asked one short
question first (``agents.exposure_grouping_planner``): does the study form
its exposure by where one measured value falls?  The request shows the study's
own words and the values the input holds that a grouping can read.  An answer
is held to those words.  The phase owner
(``orchestration.exposure_grouping_phase``) compiles what is stated, restages
the run's exact copy in place with each applied grouping, and rebuilds the
context on it; a grouping the host cannot apply stops planning
with a typed code, and nothing is asked before a run may call a Provider.
Synthetic exports and scripted Provider answers only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.agents.exposure_grouping_planner import (
    ask_exposure_groupings,
    exposure_grouping_messages,
    parse_exposure_groupings,
)
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    canonical_parameters_sha256,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration import exposure_grouping_phase
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    resumed_cohort_is_the_sources_copy,
    run_exposure_grouping_phase,
)
from easyicu.research_agent.planning.exposure_group_compile import grouping_sources
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.providers.mocks import (
    MockLLMClient,
    ScriptedMockLLMClient,
)
from easyicu.research_agent.providers.protocol import StructuredOutputRequest
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import declared_domain_for_variable
from easyicu.research_agent.schema import ValidationFinding
from tests.support.typed_export import typed_export

_QUESTION = "Compare hospital death across lactate groups in the first day."


def _grouping(**changes: Any) -> dict[str, Any]:
    grouping = {
        "id": "x1",
        "concept": "lact",
        "window": {"start_hours": 0, "end_hours": 24},
        "scale": "nominal",
        "groups": [
            {
                "id": "g1",
                "label": "lactate below 2.5",
                "rule": {"summary": "max", "op": "<", "value": 2.5, "unit": "mmol/L"},
            },
            {"id": "g2", "label": "lactate 2.5 or above", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "own_group", "label": "not measured"},
        "reference": "g1",
        "contrast": "g2",
        "quote": "lactate groups",
        "source": "question",
    }
    grouping.update(changes)
    return grouping


def _answer(*groupings: dict[str, Any]) -> str:
    return json.dumps({"groupings": list(groupings)})


def _context(cohort: Path):
    return build_research_context(
        research_question=_QUESTION,
        cohort=cohort,
        cohort_name="synthetic",
        database="miiv",
        target_outcome="death",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )


def _run_copy(tmp_path: Path) -> Path:
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=typed_export(tmp_path / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    copy = tmp_path / "run" / "cohort.parquet"
    stage_materialized_cohort_authority(
        paths["parquet"], copy, producer_implementation_sha256="a" * 64
    )
    return copy


def _producer(cohort: Path) -> str:
    staged = load_verified_materialized_cohort_authority(cohort)
    assert staged is not None
    return staged.authority.producer


def _not_asked(phase: Any) -> str:
    """Why the phase asked nothing, as its record says; the record says it planned."""

    record = json.loads(phase.record_path.read_text(encoding="utf-8"))
    assert record["exposure_grouping_enabled"] is True
    assert "compiled" not in record
    return record["not_asked"]


def _phase(copy: Path, client: Any, **changes: Any):
    options = {
        "context": _context(copy),
        "cohort_path": copy,
        "run_dir": copy.parent,
        "planner": client,
        "rebuild_context": _context,
        "capability_review_pending": False,
        "trajectory_staged": False,
        "emit_progress": lambda *_args, **_kwargs: None,
    }
    options.update(changes)
    return run_exposure_grouping_phase(**options)


def test_the_request_shows_the_studys_words_and_what_a_grouping_can_read(
    tmp_path: Path,
) -> None:
    context = _context(_run_copy(tmp_path))

    sources = grouping_sources(context)
    request = exposure_grouping_messages(context, sources)[-1].content

    assert [source.concept for source in sources] == ["age", "lact"]
    assert f"Research question: {_QUESTION}" in request
    assert (
        "- lact: first, max, mean, min over hours [0, 24) after ICU admission; "
        "unit mmol/L"
    ) in request
    assert "- age: one value per stay (summary value, no window); unit years" in request
    # An identifier and the outcome are no values an exposure is formed from.
    assert "stay_id" not in request and "- death" not in request


def test_a_value_the_study_takes_as_its_outcome_is_not_offered_to_group(
    tmp_path: Path,
) -> None:
    context = build_research_context(
        research_question="Compare the first day's peak lactate across age groups.",
        cohort=_run_copy(tmp_path),
        cohort_name="synthetic",
        database="miiv",
        target_outcome="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("lact_max",),
    )

    # A numeric summary the study takes as its outcome is no value its
    # exposure is formed from; the concept's other summaries still are.
    assert [
        (source.concept, source.summaries) for source in grouping_sources(context)
    ] == [("age", ("value",)), ("lact", ("first", "mean", "min"))]


@pytest.mark.parametrize(
    ("grouping", "reason"),
    [
        pytest.param(
            _grouping(quote="lactate classes"), "is not written there", id="unquoted"
        ),
        pytest.param(
            _grouping(source="outline"),
            "formed in the study's words",
            id="no-study-words",
        ),
    ],
)
def test_an_answer_is_held_to_the_studys_words(
    grouping: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_exposure_groupings(_answer(grouping), study_texts=(_QUESTION,))


def test_a_refused_answer_goes_back_with_its_reason(tmp_path: Path) -> None:
    context = _context(_run_copy(tmp_path))
    client = ScriptedMockLLMClient(
        [_answer(_grouping(quote="lactate classes")), _answer(_grouping())]
    )

    answer = ask_exposure_groupings(
        client, context=context, sources=grouping_sources(context)
    )

    assert [item.id for item in answer.groupings.groupings] == ["x1"]
    assert answer.transport == "contract_text"
    retry = client.calls[1][0][-1].content
    assert "'lactate classes' is not written there" in retry


def test_a_route_that_enforces_a_schema_is_sent_the_closed_shape(
    tmp_path: Path,
) -> None:
    context = _context(_run_copy(tmp_path))
    client = ScriptedMockLLMClient([_answer()])
    client.supports_strict_json_schema = True

    answer = ask_exposure_groupings(
        client, context=context, sources=grouping_sources(context)
    )

    messages, options = client.calls[0]
    assert isinstance(options["structured_output"], StructuredOutputRequest)
    assert answer.transport == "strict_schema"
    assert answer.structured_output_authority_sha256 == (
        options["structured_output"].authority_sha256
    )
    # The enforced schema carries the shape; the request does not repeat it.
    assert "Answer with one JSON object" not in messages[-1].content


def test_a_stated_grouping_is_staged_on_the_run_copy_and_planned_on(
    tmp_path: Path,
) -> None:
    copy = _run_copy(tmp_path)
    client = ScriptedMockLLMClient([_answer(_grouping())])

    phase = _phase(copy, client)

    assert len(client.calls) == 1
    variable = phase.context.variable("lact_group_x1")
    assert variable is not None
    assert declared_domain_for_variable(variable) == (
        [1, 2, 3],
        "declared_exposure_group_levels",
    )
    record = json.loads(phase.record_path.read_text(encoding="utf-8"))
    assert record["planner"]["stated"]["groupings"][0]["quote"] == "lactate groups"
    assert [item["disposition"] for item in record["compiled"]["groupings"]] == [
        "applied"
    ]
    # The run's cohort, under the name its input capsule seals, is restaged
    # with the groups, and its authority names the record that decided them.
    staged = load_verified_materialized_cohort_authority(copy)
    assert staged is not None
    assert staged.authority.producer == EXPOSURE_GROUP_STAGE_PRODUCER
    assert staged.authority.producer_parameters["groupings_record_sha256"] == (
        hashlib.sha256(phase.record_path.read_bytes()).hexdigest()
    )


def test_a_study_forming_no_grouping_keeps_its_cohort(tmp_path: Path) -> None:
    copy = _run_copy(tmp_path)
    client = ScriptedMockLLMClient([_answer()])

    phase = _phase(copy, client)

    assert phase.context.variable("lact_group_x1") is None
    record = json.loads(phase.record_path.read_text(encoding="utf-8"))
    assert record["compiled"]["groupings"] == []
    assert _producer(copy) == "research_agent_run_stage"


@pytest.mark.parametrize(
    ("grouping", "code", "reason"),
    [
        pytest.param(
            _grouping(
                concept="glu",
                groups=[
                    {
                        "id": "g1",
                        "label": "low glucose",
                        "rule": {
                            "summary": "min",
                            "op": "<",
                            "value": 70,
                            "unit": None,
                        },
                    },
                    {"id": "g2", "label": "other", "rule": "otherwise"},
                ],
            ),
            "exposure_group_requires_extraction",
            "exposure_group_concept_not_in_export",
            id="an-extraction-would-hold-it",
        ),
        pytest.param(
            _grouping(
                groups=[
                    {
                        "id": "g1",
                        "label": "lactate below 22.5",
                        "rule": {
                            "summary": "max",
                            "op": "<",
                            "value": 22.5,
                            "unit": "mg/dL",
                        },
                    },
                    {"id": "g2", "label": "lactate 22.5 or above", "rule": "otherwise"},
                ]
            ),
            "exposure_group_not_applied",
            "exposure_group_unit_mismatch",
            id="another-unit",
        ),
    ],
)
def test_a_grouping_the_host_cannot_apply_stops_planning(
    tmp_path: Path, grouping: dict[str, Any], code: str, reason: str
) -> None:
    copy = _run_copy(tmp_path)

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _phase(copy, ScriptedMockLLMClient([_answer(grouping)]))

    assert stopped.value.reason_code == code
    assert reason in str(stopped.value)
    # What was stated is recorded; nothing is staged.
    record = json.loads((copy.parent / EXPOSURE_GROUPINGS_FILENAME).read_text())
    assert record["compiled"]["groupings"][0]["reason"] == reason
    assert _producer(copy) == "research_agent_run_stage"


def _between(low: float, high: float) -> list[dict[str, Any]]:
    """Three groups of the first day's maximum, the middle one ``[low, high)``."""

    def below(value: float, label: str, group_id: str) -> dict[str, Any]:
        rule = {"summary": "max", "op": "<", "value": value, "unit": "mmol/L"}
        return {"id": group_id, "label": label, "rule": rule}

    return [
        below(low, f"lactate below {low}", "g1"),
        below(high, f"lactate {low} to {high}", "g2"),
        {"id": "g3", "label": f"lactate {high} or above", "rule": "otherwise"},
    ]


def test_a_stated_group_no_stay_falls_in_stops_planning(tmp_path: Path) -> None:
    copy = _run_copy(tmp_path)

    # Each threshold falls between the input's maxima, but no stay's maximum
    # lies between the two: the middle group holds no stay.
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _phase(
            copy,
            ScriptedMockLLMClient([_answer(_grouping(groups=_between(2.2, 2.8)))]),
        )

    # The group is neither dropped nor merged into another: the Planner
    # revises the grouping on these rows.  Nothing is staged.
    assert stopped.value.reason_code == "exposure_group_level_empty"
    assert "x1 'lactate 2.2 to 2.8'" in str(stopped.value)
    assert _producer(copy) == "research_agent_run_stage"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        pytest.param(
            {"capability_review_pending": True},
            "capability_review_pending",
            id="capability-review",
        ),
        pytest.param(
            {"trajectory_staged": True}, "trajectory_staged", id="trajectory-staged"
        ),
    ],
)
def test_nothing_is_asked_before_a_run_may_call_a_provider_for_it(
    tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    copy = _run_copy(tmp_path)
    client = ScriptedMockLLMClient([])

    phase = _phase(copy, client, **changes)

    assert not client.calls
    assert _not_asked(phase) == reason
    assert _producer(copy) == "research_agent_run_stage"


def test_nothing_is_asked_of_an_input_without_a_value_to_group(
    tmp_path: Path,
) -> None:
    untyped = tmp_path / "run" / "cohort.parquet"
    untyped.parent.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1, 2], "age": [50, 70], "death": [0, 1]}).to_parquet(
        untyped, index=False
    )
    client = ScriptedMockLLMClient([])

    phase = _phase(untyped, client)

    assert not client.calls
    assert _not_asked(phase) == "no_value_to_group"


def test_an_answer_the_host_cannot_read_stops_planning(tmp_path: Path) -> None:
    copy = _run_copy(tmp_path)
    client = ScriptedMockLLMClient(["not json"] * 3)

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _phase(copy, client)

    assert stopped.value.reason_code == "exposure_grouping_unanswered"
    assert len(client.calls) == 3
    assert not (copy.parent / EXPOSURE_GROUPINGS_FILENAME).exists()


def test_a_resumed_run_takes_its_copy_with_the_groups_and_nothing_else(
    tmp_path: Path,
) -> None:
    copy = _run_copy(tmp_path)
    source = load_verified_materialized_cohort_authority(
        tmp_path / "materialized" / "universe.parquet"
    )
    plain = load_verified_materialized_cohort_authority(copy)
    assert source is not None and plain is not None
    _phase(copy, ScriptedMockLLMClient([_answer(_grouping())]))
    grouped = load_verified_materialized_cohort_authority(copy)
    assert grouped is not None

    assert resumed_cohort_is_the_sources_copy(plain, source)
    assert resumed_cohort_is_the_sources_copy(grouped, source)
    read_other_bytes = {
        **grouped.authority.producer_parameters,
        "source_cohort_sha256": "0" * 64,
    }
    # Other rows, a column beside the groups, or groups formed from another
    # source's bytes is another cohort.
    for other in (
        replace(grouped, authority=replace(grouped.authority, cohort_rows=99)),
        replace(
            grouped,
            authority=replace(
                grouped.authority,
                cohort_columns=(*grouped.authority.cohort_columns, "extra"),
            ),
        ),
        replace(plain, authority=replace(plain.authority, cohort_sha256="0" * 64)),
        replace(
            grouped,
            authority=replace(grouped.authority, parent_authority_sha256="0" * 64),
        ),
        replace(
            grouped,
            authority=replace(
                grouped.authority,
                producer_parameters=read_other_bytes,
                producer_parameters_sha256=canonical_parameters_sha256(
                    read_other_bytes
                ),
            ),
        ),
    ):
        assert not resumed_cohort_is_the_sources_copy(other, source)


def test_nothing_is_asked_of_a_run_its_data_gate_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = _run_copy(tmp_path)
    refused = ValidationFinding(
        validator="data_answerability_gate", severity="error", message="refused"
    )
    monkeypatch.setattr(
        exposure_grouping_phase,
        "preplan_data_findings",
        lambda **_kwargs: [refused],
    )
    client = ScriptedMockLLMClient([])

    phase = _phase(copy, client)

    assert not client.calls
    assert _not_asked(phase) == "data_gate_refused"


def test_the_offline_graph_forms_no_grouping(tmp_path: Path) -> None:
    context = _context(_run_copy(tmp_path))

    answer = ask_exposure_groupings(
        MockLLMClient(), context=context, sources=grouping_sources(context)
    )

    assert answer.groupings.groupings == []


def test_a_host_planning_no_groupings_keeps_archived_config_digests(
    tmp_path: Path,
) -> None:
    off = PipelineConfig(workdir=tmp_path)
    on = PipelineConfig(workdir=tmp_path, enable_exposure_grouping=True)

    assert "enable_exposure_grouping" not in off.canonical_payload()
    assert on.canonical_payload()["enable_exposure_grouping"] is True
    assert on.canonical_digest() != off.canonical_digest()
