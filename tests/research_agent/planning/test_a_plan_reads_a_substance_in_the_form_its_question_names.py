"""A plan reads a substance in the form its question names: measured, or given.

The Web reader records each substance the question names as given or as a
measured level, with the columns of its other form
(``planning.question_substance_forms``).  A plan whose primary exposure reads
the other form -- the serum level for albumin given intravenously -- answers
another question, so planning stops with a typed reason
(``progressive_question_substance_form_substituted``) instead of analysing it.
A plan that reads the named form, or a question that names no form, passes.
Synthetic context and plan only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.orchestration.progressive_planning import (
    refuse_substance_form_substitution,
    run_progressive_planner,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

from .scientific_review_fixtures import _context, _plan

_GIVEN = {
    "concepts": ["albumin_iv"],
    "form": "given",
    "other": ["alb"],
    "evidence": "静脉输注白蛋白",
}


def _stated(*forms: Any) -> ResearchContext:
    base = _context()
    return base.model_copy(
        update={
            "variables": [
                *base.variables,
                ConceptDescriptor(
                    name="alb_max",
                    source_concept="alb",
                    role=VariableRole.OTHER,
                    dtype="float64",
                ),
                ConceptDescriptor(
                    name="albumin_iv",
                    source_concept="albumin_iv",
                    role=VariableRole.OTHER,
                    dtype="int64",
                ),
            ],
            "user_preferences": base.user_preferences.model_copy(
                update={
                    "data_constraints": json.dumps(
                        {"question_substance_forms": list(forms)}
                    )
                }
            ),
        }
    )


def _exposed_to(column: str) -> AnalysisPlan:
    plan = _plan()
    primary = next(
        step for step in plan.steps if step.planned_analysis_role == "primary"
    )
    requirement = primary.model_requirements[0].model_copy(
        update={"exposure_source": column}
    )
    primary = primary.model_copy(update={"model_requirements": [requirement]})
    return plan.model_copy(
        update={
            "steps": [
                primary if step.step_id == primary.step_id else step
                for step in plan.steps
            ]
        }
    )


def test_a_plan_that_reads_the_other_form_stops_planning() -> None:
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        refuse_substance_form_substitution(
            context=_stated(_GIVEN), plan=_exposed_to("alb_max")
        )

    assert stopped.value.reason_code == (
        "progressive_question_substance_form_substituted"
    )
    assert "'静脉输注白蛋白' as given (albumin_iv)" in str(stopped.value)
    assert "'alb_max' reads it as a measured level" in str(stopped.value)


@pytest.mark.parametrize(
    ("forms", "exposure"),
    [
        pytest.param((_GIVEN,), "albumin_iv", id="the-named-form"),
        pytest.param((), "alb_max", id="no-form-named"),
        pytest.param((_GIVEN,), "exposure", id="another-exposure"),
        pytest.param(
            (
                {
                    "concepts": ["alb"],
                    "form": "measured",
                    "other": ["albumin_iv"],
                    "evidence": "血清白蛋白",
                },
            ),
            "alb_max",
            id="the-named-level",
        ),
    ],
)
def test_a_plan_that_reads_the_named_form_passes(
    forms: tuple[Any, ...], exposure: str
) -> None:
    refuse_substance_form_substitution(
        context=_stated(*forms), plan=_exposed_to(exposure)
    )


def test_a_level_named_and_the_substance_given_planned_stops_too() -> None:
    measured = {
        "concepts": ["alb"],
        "form": "measured",
        "other": ["albumin_iv"],
        "evidence": "血清白蛋白",
    }

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        refuse_substance_form_substitution(
            context=_stated(measured), plan=_exposed_to("albumin_iv")
        )

    assert "'albumin_iv' reads it as given" in str(stopped.value)


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param({"question_substance_forms": {"form": "given"}}, id="not-a-list"),
        pytest.param(
            {"question_substance_forms": [{**_GIVEN, "form": "taken"}]},
            id="an-unknown-form",
        ),
        pytest.param(
            {"question_substance_forms": [{**_GIVEN, "other": []}]},
            id="no-other-form",
        ),
    ],
)
def test_a_malformed_record_is_refused_never_read_as_none(
    recorded: dict[str, Any],
) -> None:
    context = _stated()
    context = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(recorded)}
            )
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as refused:
        refuse_substance_form_substitution(context=context, plan=_exposed_to("alb_max"))

    assert refused.value.reason_code == "progressive_question_substance_forms_malformed"


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


def test_a_planning_run_holds_its_plan_to_the_named_form(tmp_path: Path) -> None:
    # The question named the fixture's exposure as the other form of a
    # substance given: the plan the Planner makes reads it.
    base = _planner_context()
    context = base.model_copy(
        update={
            "user_preferences": (base.user_preferences or UserPreferences()).model_copy(
                update={
                    "data_constraints": json.dumps(
                        {
                            "question_substance_forms": [
                                {
                                    "concepts": ["exposure_given"],
                                    "form": "given",
                                    "other": ["exposure_flag"],
                                    "evidence": "the exposure given",
                                }
                            ]
                        }
                    )
                }
            )
        }
    )
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(_foundation_payload()),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic cohort")

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        run_progressive_planner(
            planner=ProgressivePlannerAgent(llm),
            context=context,
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
            finding_sink=lambda _finding: None,
        )

    assert stopped.value.reason_code == (
        "progressive_question_substance_form_substituted"
    )
