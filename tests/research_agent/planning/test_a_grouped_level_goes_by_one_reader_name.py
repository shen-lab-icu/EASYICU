"""A grouped exposure's level goes by one reader name in the plan and its products.

A grouping's column holds level codes.  The plan names each level once, as
its display label ``<column>=<code>``, so its figures, tables and draft name
a level alike and the plan a reviewer approves shows the names it seals.
The host writes those names from the levels the context states: the study's
own words when a reader sentence can carry every one of them and no other
grouping uses them, and otherwise a neutral name that gives the level no
meaning its record does not.  A label the Planner wrote for a level is
replaced, and the plan says so; the Writer is told to name a level by its
label, cite its code as ``level <code>`` and give it no other meaning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.writer_display_labels import (
    reader_display_label_instruction,
)
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    run_progressive_planner,
)
from easyicu.research_agent.planning.exposure_group_reader_names import (
    ExposureGroupNamesUnreadable,
    exposure_group_display_labels,
    exposure_group_label_findings,
    exposure_group_reader_names,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ResearchContext
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
    _payload,
)


def _levels(*words: str, unmeasured: str | None = None) -> list[dict[str, Any]]:
    levels = [
        {"code": code, "group": f"g{code}", "label": word}
        for code, word in enumerate(words, start=1)
    ]
    if unmeasured is not None:
        levels.append({"code": len(levels) + 1, "group": "gU", "label": unmeasured})
    return levels


def _grouping(
    variable: str, concept: str, levels: list[dict[str, Any]], *, scale: str = "nominal"
) -> dict[str, Any]:
    return {"variable": variable, "concept": concept, "scale": scale, "levels": levels}


def _with(*groupings: Any) -> ResearchContext:
    base = _context()
    return base.model_copy(
        update={
            "cohort": base.cohort.model_copy(
                update={
                    "provenance": {
                        **base.cohort.provenance,
                        EXPOSURE_GROUP_LABELS_KEY: list(groupings),
                    }
                }
            )
        }
    )


def test_absent_groupings_add_neither_names_nor_warnings() -> None:
    assert exposure_group_reader_names(_context()) == {}
    assert exposure_group_label_findings(_context(), {}) == []


@pytest.mark.parametrize("scale", [[], {"scale": "nominal"}])
def test_malformed_scale_is_unreadable_instead_of_crashing(scale) -> None:
    context = _with(_grouping("glu_group_x1", "glu", _levels("low", "high"), scale=scale))
    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(context)
    findings = exposure_group_label_findings(context, {})
    assert findings[0].detail["reason_code"] == "exposure_group_reader_names_unreadable"


_GLUCOSE = _grouping(
    "glu_group_x1",
    "glu",
    _levels(
        "hypoglycaemia",
        "hyperglycaemia only",
        "neither",
        unmeasured="no glucose on day one",
    ),
)


def test_a_level_goes_by_the_studys_words_when_a_sentence_can_carry_them() -> None:
    assert exposure_group_reader_names(_with(_GLUCOSE)) == {
        "glu_group_x1=1": "hypoglycaemia",
        "glu_group_x1=2": "hyperglycaemia only",
        "glu_group_x1=3": "neither",
        "glu_group_x1=4": "no glucose on day one",
        "glu_group_x1": "Glucose group",
    }


@pytest.mark.parametrize(
    "words",
    [
        pytest.param(("低血糖", "高血糖", "其他"), id="not_english"),
        pytest.param(("glucose below 300", "high", "other"), id="a_traced_number"),
        pytest.param(("low", "Low", "other"), id="the_same_words_twice"),
        pytest.param(("low", "a `marked` level", "other"), id="markup"),
        pytest.param(("low", "x" * 41, "other"), id="too_long"),
    ],
)
def test_otherwise_every_level_of_the_grouping_goes_by_a_neutral_name(
    words: tuple[str, ...],
) -> None:
    grouping = _grouping("glu_group_x1", "glu", _levels(*words, unmeasured="unseen"))

    assert exposure_group_reader_names(_with(grouping)) == {
        "glu_group_x1=1": "Glucose group 1",
        "glu_group_x1=2": "Glucose group 2",
        "glu_group_x1=3": "Glucose group 3",
        "glu_group_x1=4": "Glucose not measured",
        "glu_group_x1": "Glucose group",
    }


def test_an_ordinal_groupings_neutral_names_say_its_ends() -> None:
    grouping = _grouping("lact_group_x1", "lact", _levels("低", "中", "高"), scale="ordinal")

    assert exposure_group_reader_names(_with(grouping)) == {
        "lact_group_x1=1": "Lactate group 1 (lowest)",
        "lact_group_x1=2": "Lactate group 2",
        "lact_group_x1=3": "Lactate group 3 (highest)",
        "lact_group_x1": "Lactate group",
    }


def test_words_two_groupings_share_name_neither() -> None:
    names = exposure_group_reader_names(
        _with(
            _grouping("lact_group_x1", "lact", _levels("Low", "High")),
            _grouping("glu_group_x2", "glu", _levels("low", "other")),
        )
    )

    assert names["lact_group_x1=1"] == "Lactate group 1"
    assert names["glu_group_x2=2"] == "Glucose group 2"


def test_groupings_of_one_concept_go_by_their_windows() -> None:
    def windowed(variable: str, start: int, end: int) -> dict[str, Any]:
        grouping = _grouping(variable, "glu", _levels("低", "高"))
        grouping["parameters"] = {"window": {"start_hours": start, "end_hours": end}}
        return grouping

    names = exposure_group_reader_names(
        _with(windowed("glu_group_x1", 0, 24), windowed("glu_group_x2", 24, 48))
    )

    assert names["glu_group_x1=1"] == "Glucose 0-24 h group 1"
    assert names["glu_group_x2"] == "Glucose 24-48 h group"


def test_a_context_without_groupings_names_no_level() -> None:
    assert exposure_group_reader_names(_context()) == {}


@pytest.mark.parametrize(
    "stated",
    [
        pytest.param({"variable": "x"}, id="no_levels"),
        pytest.param(_grouping("x=1", "glu", _levels("a", "b")), id="a_coded_name"),
        pytest.param(_grouping("x", "glu", _levels("a")), id="one_level"),
        pytest.param(
            _grouping("x", "glu", _levels("a", "b", unmeasured="c"), scale="ordinal"),
            id="an_unmeasured_end_on_a_scale",
        ),
        pytest.param(
            _grouping("x", "glu", [{"code": 2, "group": "g1", "label": "a"}] * 2),
            id="codes_out_of_order",
        ),
    ],
)
def test_levels_stated_in_another_shape_are_unreadable(stated: Any) -> None:
    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(_with(stated))


_PLANNER = {
    "glu_group_x1": "Glucose category",
    "glu_group_x1=1": "Low glucose",
    "glu_group_x1=9": "A level the grouping does not have",
    "age_years": "Age in years",
}


def test_the_hosts_names_replace_the_planners_level_labels() -> None:
    labels = exposure_group_display_labels(_with(_GLUCOSE), _PLANNER)

    assert labels == {
        # The Planner's name for the column stays; its level labels do not.
        "glu_group_x1": "Glucose category",
        "glu_group_x1=1": "hypoglycaemia",
        "glu_group_x1=2": "hyperglycaemia only",
        "glu_group_x1=3": "neither",
        "glu_group_x1=4": "no glucose on day one",
        "age_years": "Age in years",
    }
    (finding,) = exposure_group_label_findings(_with(_GLUCOSE), _PLANNER)
    assert finding.severity == "warning"
    assert finding.detail == {
        "reason_code": "exposure_group_planner_labels_replaced",
        "keys": ["glu_group_x1=1", "glu_group_x1=9"],
    }


def test_labels_the_host_agrees_with_say_nothing() -> None:
    agreed = {"glu_group_x1=1": "hypoglycaemia", "age_years": "Age in years"}

    assert exposure_group_label_findings(_with(_GLUCOSE), agreed) == []
    assert exposure_group_label_findings(_context(), _PLANNER) == []


def test_unreadable_levels_leave_the_planners_labels_and_say_so() -> None:
    context = _with({"variable": "x"})

    assert exposure_group_display_labels(context, _PLANNER) == _PLANNER
    (finding,) = exposure_group_label_findings(context, _PLANNER)
    assert finding.detail["reason_code"] == "exposure_group_reader_names_unreadable"


def _exposure_grouped() -> ResearchContext:
    """The fixture's exposure as a grouping of two levels the study named."""

    return _with(_grouping("exposure_flag", "lact", _levels("Lower", "Higher")))


def test_a_compiled_plan_names_each_grouped_level_by_the_hosts_name() -> None:
    plan, _receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_payload()),
        context=_exposure_grouped(),
    )

    labels = plan.display_labels
    assert (labels["exposure_flag=1"], labels["exposure_flag=2"]) == (
        "Lower",
        "Higher",
    )
    # The fixture Planner's labels: a code the grouping does not have goes;
    # the column's own name and every other label stay.
    assert "exposure_flag=0" not in labels
    assert labels["exposure_flag"] == "Exposure status"
    assert labels["sex_code"] == "Recorded sex"


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


def test_a_planning_run_reports_the_level_labels_it_replaced(tmp_path: Path) -> None:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic cohort")
    findings: list = []

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=_exposure_grouped(),
        run_dir=tmp_path,
        evidence=_Evidence(),
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=findings.append,
    )

    assert result.plan.display_labels["exposure_flag=1"] == "Lower"
    replaced = [
        finding
        for finding in findings
        if finding.validator == "exposure_group_reader_names"
    ]
    assert [finding.detail for finding in replaced] == [
        {
            "reason_code": "exposure_group_planner_labels_replaced",
            "keys": ["exposure_flag=0", "exposure_flag=1"],
        }
    ]


def test_the_writer_names_a_level_by_its_label_and_nothing_more() -> None:
    instruction = reader_display_label_instruction(
        {"glu_group_x1=1": "hypoglycaemia"}
    )

    assert "cite it as `level <code>`" in instruction
    assert (
        "Give a level no meaning, threshold or order that its label does not state."
        in instruction
    )
