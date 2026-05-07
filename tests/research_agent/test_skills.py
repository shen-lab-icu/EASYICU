"""ClinicalSkill registry: every built-in must produce a valid plan.

The skills are how non-LLM users (and the webapp T1.7) reach the
pipeline. If a built-in skill stops emitting a plan, the demo and the
webapp regress silently, so we pin a few invariants.
"""

from __future__ import annotations

import pandas as pd


def _empty_context_for_skill(ra, skill) -> "ra.ResearchContext":
    """Build a ResearchContext with the skill's expected variables."""
    cols = {v: [0] * 3 for v in skill.expected_variables}
    cols.setdefault("stay_id", [1, 2, 3])
    df = pd.DataFrame(cols)
    return ra.build_research_context(
        research_question=skill.question_for(database="synthetic"),
        cohort=df, cohort_name=f"{skill.key}_demo", database="synthetic",
        target_outcome=skill.target_outcome,
    )


def test_every_builtin_produces_nonempty_plan(ra):
    skills = ra.list_skills()
    assert skills, "the skill registry is unexpectedly empty"
    for skill in skills:
        ctx = _empty_context_for_skill(ra, skill)
        plan = skill.plan(ctx)
        assert plan.research_question
        assert plan.steps, f"skill '{skill.key}' produced an empty plan"
        # Every step must have a step_id and a non-empty intent.
        for step in plan.steps:
            assert step.step_id and step.intent
        # Skills with a SOFA predictor must also include the SOFA-zero audit.
        if skill.primary_predictor.lower() in {"sofa", "sofa2"}:
            ids = [s.step_id for s in plan.steps]
            assert any("sofa_zero" in sid for sid in ids), (
                f"skill '{skill.key}' missing the sofa_zero audit step"
            )


def test_skill_validate_against_missing_var(ra):
    skill = ra.get_skill("sofa_mortality")
    df = pd.DataFrame({"stay_id": [1, 2], "age": [60, 70], "death": [0, 1]})
    issues = skill.validate_against(df)
    # sofa2 is required by the skill but absent from the df → issue raised
    assert any("sofa2" in s for s in issues), issues


def test_get_unknown_skill_raises(ra):
    import pytest
    with pytest.raises(KeyError):
        ra.get_skill("not_a_real_skill_v0")
