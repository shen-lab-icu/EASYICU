"""A post-baseline exposure is read from the host's one materialization-window reader.

An exposure without its own window is classified by the window the Web host
materialized feature columns over.  ``post_baseline_exposure`` reads that
window through ``host_materialization_window_hours``, the reader the cohort
predicate rule uses, so the review and that rule count the same records:
``hours``, else ``observation_hours``, counted from ICU admission.  An
owner-declared admission attribute stays baseline under any window.
Fixtures are synthetic.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest

from easyicu.research_agent.planning import scientific_review
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    post_baseline_exposure,
)
from easyicu.research_agent.research_context.materialization_window import (
    host_materialization_window_hours,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

from .scientific_review_fixtures import _context, _plan

_OUTER = "outer_observation_window"


def _with_window(
    window: object | None, *, role: VariableRole = VariableRole.OTHER
) -> ResearchContext:
    base = _context()
    exposure = ConceptDescriptor(name="exposure", role=role, dtype="int64")
    constraints = None if window is None else json.dumps({"materialization_window": window})
    return base.model_copy(
        update={
            "variables": [exposure, *[v for v in base.variables if v.name != "exposure"]],
            "user_preferences": UserPreferences(covariates=["age"], data_constraints=constraints),
        }
    )


@pytest.mark.parametrize(
    ("window", "hours"),
    [
        ({"role": _OUTER, "anchor": "icu_admission", "hours": 24}, 24.0),
        ({"role": _OUTER, "anchor": "ICU admission", "hours": 48.0}, 48.0),
        ({"role": _OUTER, "anchor": "ICU admission", "observation_hours": 12}, 12.0),
        ({"role": _OUTER, "anchor": "icu_admission", "hours": 6, "observation_hours": 72}, 6.0),
        ({"role": _OUTER, "anchor": "hospital_admission", "hours": 24}, None),
        ({"role": "exposure_definition", "anchor": "icu_admission", "hours": 24}, None),
        ({"role": _OUTER, "anchor": "icu_admission", "hours": "24"}, None),
        ({"role": _OUTER, "anchor": "icu_admission"}, None),
        ([_OUTER, "icu_admission", 24], None),
        (None, None),
    ],
    ids=[
        "hours",
        "hours in another spelling of the anchor",
        "observation hours",
        "hours before observation hours",
        "another anchor",
        "another role",
        "hours as text",
        "no hours",
        "not an object",
        "no record",
    ],
)
def test_the_review_reads_the_window_the_cohort_rule_reads(
    window: object | None, hours: float | None
) -> None:
    context = _with_window(window)

    assert host_materialization_window_hours(context) == hours
    assert post_baseline_exposure(context) == (
        (False, None)
        if hours is None
        else (True, f"outer_materialization:icu_admission[0,{hours:g}]h")
    )


def test_a_window_recorded_as_observation_hours_holds_the_plan() -> None:
    context = _with_window(
        {"role": _OUTER, "anchor": "ICU admission", "observation_hours": 24}
    )

    review = build_plan_scientific_review(context=context, plan=_plan())

    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in {
        item.code for item in review.findings
    }


@pytest.mark.parametrize("key", ["hours", "observation_hours"])
def test_an_admission_attribute_stays_baseline_under_any_window(key: str) -> None:
    context = _with_window(
        {"role": _OUTER, "anchor": "ICU admission", key: 72},
        role=VariableRole.DEMOGRAPHIC,
    )

    assert post_baseline_exposure(context) == (False, None)


def test_the_review_reads_no_window_itself() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(scientific_review.post_baseline_exposure)))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }

    assert "host_materialization_window_hours" in called
    assert not {"loads", "get"} & called
