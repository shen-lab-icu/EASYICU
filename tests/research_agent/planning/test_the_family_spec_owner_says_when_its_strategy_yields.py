"""The family-spec owner says when its strategy yields to Progressive v2.

A resumed development checkpoint and a design canary (a run that stops after
its outline) always plan with Progressive v2; otherwise the strategy yields
only when no family template plans the context.  The planner records the
owner's reason as ``family_spec_fallback_reason``.  Synthetic contexts only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents import family_spec_planner
from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_fallback_reason,
)
from easyicu.research_agent.planning.family_spec import family_template_id_for_context
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlannerCheckpoint,
)
from tests.research_agent.planning.family_spec_fixtures import _context
from tests.support.survival_proposal import survival_context

_SURVIVAL_QUESTION = (
    "Among adult ICU stays, how does the first-24 h injury stage relate to the time "
    "to in-hospital death?"
)
# Only its presence matters to the owner: a checkpoint resumes a development run.
_CHECKPOINT = ProgressivePlannerCheckpoint.model_construct()


def _templated():
    context = survival_context(research_question=_SURVIVAL_QUESTION)
    assert (
        family_template_id_for_context(context, analysis_types=("survival",))
        is not None
    )
    return context


def _untemplated():
    context = _context()
    types = ("causal_inference",)
    assert family_template_id_for_context(context, analysis_types=types) is None
    return context, types


@pytest.mark.parametrize(
    ("checkpoint", "stop_after_outline", "reason"),
    [
        (_CHECKPOINT, False, "development_resume_checkpoint_uses_progressive_v2"),
        (_CHECKPOINT, True, "development_resume_checkpoint_uses_progressive_v2"),
        (None, True, "design_canary_uses_progressive_v2"),
    ],
    ids=["resumed", "resumed-canary", "canary"],
)
def test_a_resumed_run_or_a_design_canary_yields_even_with_a_template(
    checkpoint, stop_after_outline, reason
):
    assert (
        family_spec_fallback_reason(
            _templated(),
            analysis_types=("survival",),
            resume_checkpoint=checkpoint,
            stop_after_outline=stop_after_outline,
        )
        == reason
    )


def test_a_context_no_template_plans_yields_and_one_a_template_plans_keeps_the_strategy():
    context, types = _untemplated()
    assert (
        family_spec_fallback_reason(
            context,
            analysis_types=types,
            resume_checkpoint=None,
            stop_after_outline=False,
        )
        == "no_family_template_for_context"
    )
    assert (
        family_spec_fallback_reason(
            _templated(),
            analysis_types=("survival",),
            resume_checkpoint=None,
            stop_after_outline=False,
        )
        is None
    )


def test_the_template_question_reads_the_planning_contract(monkeypatch):
    seen = []

    def lookup(context, *, analysis_types, planning_contract_context=""):
        seen.append((tuple(analysis_types), planning_contract_context))
        return "a_template"

    monkeypatch.setattr(family_spec_planner, "family_template_id_for_context", lookup)
    context, types = _untemplated()
    assert (
        family_spec_fallback_reason(
            context,
            analysis_types=types,
            resume_checkpoint=None,
            stop_after_outline=False,
            planning_contract_context="sealed disclosure",
        )
        is None
    )
    assert seen == [(types, "sealed disclosure")]
