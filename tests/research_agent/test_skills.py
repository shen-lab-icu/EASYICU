"""ClinicalSkill registry invariants.

Built-in skills may describe analysis families (association, prediction,
data-quality audit), but not one concrete clinical question such as
``sofa_mortality``. User-registered skills can still bind concrete variables.
"""

from __future__ import annotations

import pandas as pd


def _generic_skill(ra):
    """A case-neutral skill built locally (never registered globally)."""
    return ra.ClinicalSkill(
        key="generic_exposure_outcome",
        name="Generic exposure → outcome association",
        description="Case-neutral association template used only in tests.",
        research_question_template=(
            "Is exposure_x associated with endpoint_y in {database}?"
        ),
        target_outcome="endpoint_y",
        primary_predictor="exposure_x",
        expected_variables=["age", "sex", "exposure_x", "endpoint_y"],
    )


def _context_for_skill(ra, skill) -> "ra.ResearchContext":
    cols = {v: [0] * 3 for v in skill.expected_variables}
    cols.setdefault("stay_id", [1, 2, 3])
    df = pd.DataFrame(cols)
    return ra.build_research_context(
        research_question=skill.question_for(database="synthetic"),
        cohort=df, cohort_name=f"{skill.key}_demo", database="synthetic",
        target_outcome=skill.target_outcome,
    )


def test_registry_ships_only_analysis_family_builtin_skills(ra):
    """Built-ins are workflow families, not score/outcome-specific questions."""
    skills = {s.key: s for s in ra.list_skills()}
    assert set(skills) == {
        "association_analysis",
        "prediction_model",
        "data_quality_audit",
    }
    forbidden_tokens = {
        "sofa",
        "aki",
        "kdigo",
        "vaso",
        "lactate",
        "mortality",
        "death",
    }
    for skill in skills.values():
        blob = " ".join([skill.key, skill.name, skill.description]).lower()
        assert not any(token in blob for token in forbidden_tokens)
        assert skill.target_outcome is None
        assert skill.primary_predictor is None
        assert skill.expected_variables == []


def test_user_registered_skill_produces_nonempty_plan(ra):
    """The registry mechanism still works for a case-neutral skill."""
    skill = _generic_skill(ra)
    ctx = _context_for_skill(ra, skill)
    plan = skill.plan(ctx)
    assert plan.research_question
    assert plan.steps, "a registered skill produced an empty plan"
    for step in plan.steps:
        assert step.step_id and step.intent


def test_no_builtin_ships_a_bespoke_canned_plan(ra):
    """Built-in skills (if any are ever re-added) must stay case-neutral.

    A skill that hard-codes a paper-specific ``AnalysisPlan`` via
    ``plan_factory`` launders human-authored analysis as autonomous agent
    output - the same integrity problem removed from ``code_repair`` and the
    bundled ``case_plugins``. Generic skills must reach a plan through the
    shared ``_default_skill_plan`` template (``plan_factory is None``).
    """
    offenders = [
        s.key for s in ra.list_skills()
        if getattr(s, "plan_factory", None) is not None
    ]
    assert offenders == [], (
        "built-in skills must not bundle a bespoke canned plan; "
        f"offenders: {offenders}"
    )


def test_skill_validate_against_missing_var(ra):
    skill = _generic_skill(ra)
    df = pd.DataFrame({"stay_id": [1, 2], "age": [60, 70], "endpoint_y": [0, 1]})
    issues = skill.validate_against(df)
    # exposure_x is required by the skill but absent from df → issue raised
    assert any("exposure_x" in s for s in issues), issues


def test_get_unknown_skill_raises(ra):
    import pytest
    with pytest.raises(KeyError):
        ra.get_skill("not_a_real_skill_v0")
