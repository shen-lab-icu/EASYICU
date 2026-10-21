"""A grouped exposure's level goes by one reader name in the plan and its products.

A grouping's column holds level codes.  The plan names each level once, as
its display label ``<column>=<code>``, so its figures, tables and draft name
a level alike and the plan a reviewer approves shows the names it seals.
The host writes those names from the levels the context states: the study's
own words when a reader sentence can carry every one of them and no other
grouping uses them, and otherwise each level's rule, written in English from
the grouping's sealed derivation -- never a name by place ("Glucose group 2")
that hides what the group is.  A label the Planner wrote for a level is
replaced, and the plan says so; the Writer is told to name a level by its
label, cite its code as ``level <code>`` and give it no other meaning, and is
shown the study's words for a level whose rule states a traced threshold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.writer_display_labels import (
    reader_display_label_instruction,
)
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    planned_group_column,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
    _labels as distribution_figure_labels,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    run_progressive_planner,
)
from easyicu.research_agent.planning.exposure_group_reader_names import (
    ExposureGroupNamesUnreadable,
    exposure_group_display_labels,
    exposure_group_label_findings,
    exposure_group_reader_names,
    writer_level_labels,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import _exposure_group_projections
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
    variable: str,
    concept: str,
    levels: list[dict[str, Any]],
    *,
    scale: str = "nominal",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "variable": variable,
        "concept": concept,
        "scale": scale,
        "levels": levels,
        **({"parameters": parameters} if parameters is not None else {}),
    }


def _rule(summary: str, op: str, value: float, unit: str) -> dict[str, Any]:
    return {"summary": summary, "op": op, "value": value, "unit": unit}


def _glucose_rules(
    *, unmeasured: str = "own_group", window: tuple[int, int] = (0, 24)
) -> dict[str, Any]:
    """Low glucose, then high glucose, then every other measured stay."""

    return {
        "concept": "glu",
        "window": {"start_hours": window[0], "end_hours": window[1]},
        "scale": "nominal",
        "groups": [
            {"id": "g1", "rule": _rule("min", "<", 70, "mg/dL")},
            {"id": "g2", "rule": _rule("max", ">", 180, "mg/dL")},
            {"id": "g3", "rule": "otherwise"},
        ],
        "unmeasured": unmeasured,
        "source_columns": {"min": "glu_min", "max": "glu_max"},
    }


#: Each glucose level's rule as its reader name.
_GLUCOSE_RULE_NAMES = (
    "Minimum below 70 mg/dL",
    "Maximum above 180 mg/dL, without minimum below 70 mg/dL",
    "Neither minimum below 70 mg/dL nor maximum above 180 mg/dL",
)


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
def test_otherwise_every_level_of_the_grouping_goes_by_its_rule(
    words: tuple[str, ...],
) -> None:
    grouping = _grouping(
        "glu_group_x1",
        "glu",
        _levels(*words, unmeasured="unseen"),
        parameters=_glucose_rules(),
    )

    assert exposure_group_reader_names(_with(grouping)) == {
        "glu_group_x1=1": _GLUCOSE_RULE_NAMES[0],
        "glu_group_x1=2": _GLUCOSE_RULE_NAMES[1],
        "glu_group_x1=3": _GLUCOSE_RULE_NAMES[2],
        "glu_group_x1=4": "No value recorded",
        "glu_group_x1": "Glucose group",
    }


def test_a_name_never_stands_for_a_level_by_its_place() -> None:
    names = exposure_group_reader_names(
        _with(
            _grouping(
                "glu_group_x1",
                "glu",
                _levels("低血糖", "仅高血糖", "血糖正常"),
                parameters=_glucose_rules(unmeasured="exclude"),
            )
        )
    )

    assert not [
        name for key, name in names.items() if "=" in key and "group" in name.casefold()
    ]


def test_an_ordinal_groupings_levels_go_by_their_rules_along_its_scale() -> None:
    parameters = {
        "concept": "lact",
        "window": {"start_hours": 0, "end_hours": 24},
        "scale": "ordinal",
        # Matched in this order; the codes follow the scale.
        "groups": [
            {"id": "g1", "rule": _rule("max", "<", 2, "mmol/L")},
            {"id": "g3", "rule": _rule("max", ">=", 4, "mmol/L")},
            {"id": "g2", "rule": "otherwise"},
        ],
        "unmeasured": "exclude",
        "source_columns": {"max": "lact_max"},
    }
    grouping = _grouping(
        "lact_group_x1", "lact", _levels("低", "中", "高"), scale="ordinal", parameters=parameters
    )

    assert exposure_group_reader_names(_with(grouping)) == {
        "lact_group_x1=1": "Maximum below 2 mmol/L",
        "lact_group_x1=2": "Neither maximum below 2 mmol/L nor maximum at least 4 mmol/L",
        "lact_group_x1=3": "Maximum at least 4 mmol/L, without maximum below 2 mmol/L",
        "lact_group_x1": "Lactate group",
    }


def test_words_two_groupings_share_name_neither() -> None:
    lactate = {
        **_glucose_rules(unmeasured="exclude"),
        "concept": "lact",
        "groups": [
            {"id": "g1", "rule": _rule("max", "<", 2, "mmol/L")},
            {"id": "g2", "rule": "otherwise"},
        ],
        "source_columns": {"max": "lact_max"},
    }
    glucose = {
        **_glucose_rules(unmeasured="exclude"),
        "groups": [{"id": "g1", "rule": _rule("min", "<", 70, "mg/dL")}, {"id": "g2", "rule": "otherwise"}],
        "source_columns": {"min": "glu_min"},
    }
    names = exposure_group_reader_names(
        _with(
            _grouping("lact_group_x1", "lact", _levels("Low", "High"), parameters=lactate),
            _grouping("glu_group_x2", "glu", _levels("low", "other"), parameters=glucose),
        )
    )

    assert names["lact_group_x1=1"] == "Maximum below 2 mmol/L"
    assert names["glu_group_x2=2"] == "Not minimum below 70 mg/dL"


def test_groupings_of_one_concept_go_by_their_windows() -> None:
    def windowed(variable: str, start: int, end: int) -> dict[str, Any]:
        return _grouping(
            variable,
            "glu",
            _levels("低血糖", "仅高血糖", "血糖正常"),
            parameters=_glucose_rules(unmeasured="exclude", window=(start, end)),
        )

    names = exposure_group_reader_names(
        _with(windowed("glu_group_x1", 0, 24), windowed("glu_group_x2", 24, 48))
    )

    assert names["glu_group_x1=1"] == names["glu_group_x2=1"] == _GLUCOSE_RULE_NAMES[0]
    assert names["glu_group_x1"] == "Glucose 0-24 h group"
    assert names["glu_group_x2"] == "Glucose 24-48 h group"


def _projected(words: tuple[str, ...]) -> ResearchContext:
    """A context whose grouped levels the context builder projected from a
    planned column, as a run plans on metadata alone."""

    column = planned_group_column(
        _glucose_rules(unmeasured="exclude"),
        variable="glu_group_x1",
        labels=dict(zip(("g1", "g2", "g3"), words)),
        compared={"reference": 1, "contrast": 3},
        groupings_record_sha256="a" * 64,
    ).record()
    projected = _exposure_group_projections(None, {"exposure_groups": [column]})
    return _with(*projected[EXPOSURE_GROUP_LABELS_KEY])


def test_the_levels_the_context_builder_projects_are_named() -> None:
    # The builder states the rule wording beside each grouping's levels.
    context = _projected(("低血糖", "仅高血糖", "血糖正常"))

    assert exposure_group_label_findings(context, {}) == []
    assert exposure_group_reader_names(context) == {
        "glu_group_x1=1": _GLUCOSE_RULE_NAMES[0],
        "glu_group_x1=2": _GLUCOSE_RULE_NAMES[1],
        "glu_group_x1=3": _GLUCOSE_RULE_NAMES[2],
        "glu_group_x1": "Glucose group",
    }
    english = _projected(("hypoglycaemia", "hyperglycaemia only", "normoglycaemia"))
    assert exposure_group_reader_names(english)["glu_group_x1=2"] == "hyperglycaemia only"


@pytest.mark.parametrize("version", [0, "2", None, 3])
def test_a_rule_wording_the_host_does_not_write_is_unreadable(version: Any) -> None:
    context = _projected(("低血糖", "仅高血糖", "血糖正常"))
    (stated,) = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY]

    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(_with({**stated, "rule_wording_version": version}))


def test_words_no_sentence_carries_and_no_rule_leave_the_grouping_unreadable() -> None:
    context = _with(_grouping("glu_group_x1", "glu", _levels("低血糖", "高血糖")))

    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(context)
    (finding,) = exposure_group_label_findings(context, {})
    assert finding.detail["reason_code"] == "exposure_group_reader_names_unreadable"


def test_a_rule_that_would_read_as_markup_names_no_level() -> None:
    parameters = {
        **_glucose_rules(unmeasured="exclude"),
        "groups": [
            {"id": "g1", "rule": _rule("min", "<", 70, "mg*dL")},
            {"id": "g2", "rule": _rule("max", ">", 180, "mg*dL")},
            {"id": "g3", "rule": "otherwise"},
        ],
    }
    context = _with(
        _grouping("glu_group_x1", "glu", _levels("低血糖", "仅高血糖", "血糖正常"), parameters=parameters)
    )

    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(context)


def test_a_rule_longer_than_a_reader_name_names_no_level() -> None:
    unit = "u" * 32
    parameters = {
        **_glucose_rules(unmeasured="exclude"),
        "groups": [
            *(
                {"id": f"g{index}", "rule": _rule("max", ">=", 1000.5 * (7 - index), unit)}
                for index in range(1, 6)
            ),
            {"id": "g6", "rule": "otherwise"},
        ],
        "source_columns": {"max": "glu_max"},
    }
    context = _with(
        _grouping(
            "glu_group_x1", "glu", _levels("一", "二", "三", "四", "五", "六"), parameters=parameters
        )
    )

    with pytest.raises(ExposureGroupNamesUnreadable):
        exposure_group_reader_names(context)
    # A plan carrying them would fail its display-label cap; it shows codes.
    assert exposure_group_display_labels(context, {"death": "Death"}) == {"death": "Death"}


def test_the_figures_name_each_level_by_its_name_whatever_type_its_code_is() -> None:
    labels = exposure_group_display_labels(_projected(("低血糖", "仅高血糖", "血糖正常")), {})
    floats = pd.DataFrame({"exposure_level": [1.0, 2.0, 3.0]})
    integers = pd.DataFrame({"exposure_level": [1, 2, 3]})

    for levels in (floats, integers):
        assert distribution_figure_labels(
            levels, None, exposure="glu_group_x1", display_labels=labels
        ) == list(_GLUCOSE_RULE_NAMES)
    # Another grouping's codes are not this grouping's levels.
    assert distribution_figure_labels(
        floats, None, exposure="lact_group_x1", display_labels=labels
    ) == ["1.0", "2.0", "3.0"]


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


def test_the_writer_is_shown_the_studys_words_for_a_level_whose_rule_states_a_traced_threshold() -> None:
    context = _projected(("低血糖", "仅高血糖", "血糖正常"))
    labels = {
        **exposure_group_display_labels(context, {}),
        "age_years": "Age 65 years or older",
        "lactate_band=1": "Lactate above 2.5 mmol/L",
    }

    shown = writer_level_labels(context, labels)

    # One level's rule (70) needs no provenance, two (180) do: the whole
    # grouping goes by the study's words, which the Writer renders in English.
    assert {key: shown[key] for key in labels if key.startswith("glu_group_x1=")} == {
        "glu_group_x1=1": "低血糖",
        "glu_group_x1=2": "仅高血糖",
        "glu_group_x1=3": "血糖正常",
    }
    # Labels of no grouping, and the grouping column's own name, are shown as written.
    assert shown["lactate_band=1"] == "Lactate above 2.5 mmol/L"
    assert shown["age_years"] == "Age 65 years or older"
    assert shown["glu_group_x1"] == "Glucose group"
    instruction = reader_display_label_instruction(labels, context=context)
    assert "180" not in instruction
    assert '"glu_group_x1=2": "仅高血糖"' in instruction
    # Without the context, the mapping is shown as the plan states it.
    assert writer_level_labels(None, labels) == labels


def test_a_level_whose_words_carry_a_traced_number_too_is_left_to_its_code() -> None:
    context = _projected(("glucose < 70", "glucose 180.5 or more", "other"))
    labels = exposure_group_display_labels(context, {})

    shown = writer_level_labels(context, labels)

    assert "glu_group_x1=2" not in shown
    assert shown["glu_group_x1=1"] == "glucose < 70"
    assert shown["glu_group_x1=3"] == "other"


def test_names_that_need_no_provenance_are_shown_as_they_are() -> None:
    context = _with(_GLUCOSE)
    labels = exposure_group_display_labels(context, {})

    assert writer_level_labels(context, labels) == labels


def test_the_writer_names_a_level_by_its_label_and_nothing_more() -> None:
    instruction = reader_display_label_instruction(
        {"glu_group_x1=1": "hypoglycaemia"}
    )

    assert "cite it as `level <code>`" in instruction
    assert (
        "Give a level no meaning, threshold or order that its label does not state."
        in instruction
    )
