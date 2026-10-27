"""A plan claims an ordered trend only when it runs one.

The categorical landmark template words its design (primary method, figure
role, supports, methods sentences, cohort objective) from the steps it plans.
An ordered exposure with at least three levels and a binary outcome gets the
``ordinal_trend`` step -- the primary model refitted with the exposure as one
ordinal-linear term, beside a crude Cochran-Armitage test -- with or without a
secondary continuous outcome, and its text names exactly that.  An unordered or
two-level exposure gets neither the step nor any trend wording.  Synthetic,
row-free contexts only.
"""

from __future__ import annotations

import json
import re

import pytest

from easyicu.research_agent.execution.runners.ordered_stratified_executor import (
    ordered_stratified_spec_for_step,
)
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_CATEGORICAL_FAMILY_ID,
    FamilyPlanSpec,
    build_landmark_categorical_skeleton,
)
from easyicu.research_agent.planning.family_spec.contract import landmark_ordinal_trend_present
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole
from tests.research_agent.planning.family_spec_fixtures import (
    LABELS,
    _context,
    _request,
    _run,
    _spec_payload,
)

_TREND_WORDS = re.compile(
    r"trend|gradient|per[- ]level|ordinal-linear|Cochran|Jonckheere|趋势|逐级|每升高一级|有序线性",
    re.IGNORECASE,
)
_ZH_QUESTION = (
    "在成人 ICU 住院中，前 24 小时的损伤分期与 24 小时界标后的院内死亡有何关联，"
    "院内死亡风险是否随分期逐级升高？"
)


def _case(shape: str, *, zh: bool = False, risk_ratio: bool = False) -> ResearchContext:
    """One landmark context per exposure shape; only the named coordinates change."""

    base = _context()
    question = _ZH_QUESTION if zh else (
        "Among adult ICU stays, how is the injury stage in the first 24 h associated "
        "with in-hospital death after a 24 h landmark?"
    )
    if risk_ratio:
        question += "请报告相对危险度。" if zh else " Report the relative risk."
    update: dict = {"research_question": question}
    variables = list(base.variables)
    if shape != "ordered_binary_continuous":
        update["cohort"] = base.cohort.model_copy(
            update={"outcome_columns": ["death"], "requested_outcome_columns": ["death"]}
        )
        update["user_preferences"] = base.user_preferences.model_copy(update={
            "must_have_outputs": "Generate figures for the primary result.",
        })
    if shape in {"unordered", "two_level"}:
        # Alternate stage definitions share the ordered domain; keep the case to
        # the primary exposure's own shape.
        update["user_preferences"] = update["user_preferences"].model_copy(update={
            "sensitivity_specs": [
                item for item in base.user_preferences.sensitivity_specs
                if item.axis != "exposure_definition"
            ],
        })
        variables = [
            item for item in variables
            if item.name not in {"injury_stage_alt_a", "injury_stage_alt_b"}
        ]
    if shape == "unordered":
        variables = [
            ConceptDescriptor(
                name="injury_stage", role=VariableRole.OTHER, dtype="float64",
                observed_domain={"n_unique": 4, "is_binary": False, "levels": [0, 1, 2, 3]},
            )
            if item.name == "injury_stage" else item
            for item in variables
        ]
    elif shape == "two_level":
        variables = [
            item.model_copy(update={"ordinal_levels": [0, 1], "valid_range": [0.0, 1.0]})
            if item.name == "injury_stage" else item
            for item in variables
        ]
    update["variables"] = variables
    return base.model_copy(update=update)


def _payload(request, *, risk_ratio: bool) -> dict:
    if not risk_ratio:
        return _spec_payload(request)
    quote = "相对危险度" if any("一" <= ch <= "鿿" for ch in request.research_question) else "relative risk"
    return _spec_payload(
        request, effect_measure="risk_ratio",
        question_requirements=[{
            "id": "r1", "kind": "estimand", "quote": quote, "concepts": [], "coverage": "plan",
            "gap": None, "note": None, "effect_measure": "risk_ratio",
        }],
    )


def _draft(context: ResearchContext, *, risk_ratio: bool = False):
    request = _request(context)
    assert request.family_id == LANDMARK_CATEGORICAL_FAMILY_ID
    return request, build_landmark_categorical_skeleton(
        request, FamilyPlanSpec.model_validate(_payload(request, risk_ratio=risk_ratio)),
    )


def _plan_text(draft) -> list[str]:
    """Every prose field the template authors for the plan reader and the Planner."""

    texts = [draft.outline.cohort_objective]
    texts += [step.objective for step in draft.outline.steps]
    for candidate in draft.outline.design_selection.candidates:
        texts += [
            candidate.primary_method, candidate.figure_role, candidate.supports,
            candidate.estimand, candidate.cannot_prove, candidate.decision_reason,
            *(candidate.reviewable_plan or []),
        ]
    return texts


def _selected(draft):
    return next(
        item for item in draft.outline.design_selection.candidates if item.disposition == "selected"
    )


def _model_line(draft, *, zh: bool) -> str:
    starts = ("修正 Poisson", "logistic 模型") if zh else ("Modified Poisson", "Logistic model")
    [line] = [item for item in _selected(draft).reviewable_plan if item.startswith(starts)]
    return line


@pytest.mark.parametrize("zh", [False, True], ids=["en", "zh"])
@pytest.mark.parametrize("risk_ratio", [False, True], ids=["odds_ratio", "risk_ratio"])
@pytest.mark.parametrize("continuous", [False, True], ids=["binary", "binary_continuous"])
def test_an_ordered_exposure_plans_the_trend_it_describes(continuous, risk_ratio, zh):
    shape = "ordered_binary_continuous" if continuous else "ordered_binary"
    request, draft = _draft(_case(shape, zh=zh, risk_ratio=risk_ratio), risk_ratio=risk_ratio)
    assert landmark_ordinal_trend_present(request)
    assert (request.secondary_continuous_outcome == "los_icu") is continuous

    [trend] = [step for step in draft.outline.steps if step.step_id == "ordinal_trend"]
    assert trend.variable_names == ["injury_stage", "death", *(["los_icu"] if continuous else [])]
    [materialized] = [item for item in draft.materializations if item.step.step_id == "ordinal_trend"]
    assert materialized.step.raw_inputs == trend.variable_names
    report = next(item for item in draft.materializations if item.step.step_id == "report")
    assert ("ordinal_trend", "table:ordinal_trend") in {
        (ref.producer_step_id, ref.product_id) for ref in report.step.product_inputs
    }

    measure = "risk ratio" if risk_ratio else "odds ratio"
    selected = _selected(draft)
    assert "an adjusted per-level ordered trend refit" in selected.primary_method
    assert "ordered gradient" in selected.supports
    assert "trend" not in selected.figure_role  # the figure does not read the trend table
    assert f"adjusted {measure} per level increment" in trend.objective
    assert "Cochran–Armitage" in trend.objective
    assert ("Jonckheere–Terpstra" in trend.objective) is continuous

    line = _model_line(draft, zh=zh)
    if zh:
        assert "作为有序线性项" in line
        assert f"每升高一级的校正{'风险比' if risk_ratio else '比值比'}及其 95% CI 和趋势 P 值" in line
        assert "Cochran–Armitage 趋势检验" in line
        assert ("Jonckheere–Terpstra" in line) is continuous
    else:
        assert f"for the adjusted {measure} per level increment with its 95% CI and trend p-value" in line
        assert "crude Cochran–Armitage trend test" in line
        if continuous:
            assert f"a Jonckheere–Terpstra test of {LABELS['los_icu']}" in line
        else:
            assert "Jonckheere" not in line
    other = "odds ratio" if risk_ratio else "risk ratio"
    assert f"{other} per level" not in line


@pytest.mark.parametrize("zh", [False, True], ids=["en", "zh"])
@pytest.mark.parametrize("shape", ["unordered", "two_level"])
def test_an_exposure_without_an_ordered_trend_claims_none(shape, zh):
    request, draft = _draft(_case(shape, zh=zh))
    assert not landmark_ordinal_trend_present(request)
    assert request.exposure_kind == "categorical"
    assert len(request.exposure_levels) == (4 if shape == "unordered" else 2)
    assert "ordinal_trend" not in {step.step_id for step in draft.outline.steps}
    claimed = [text for text in _plan_text(draft) if _TREND_WORDS.search(text)]
    assert claimed == []


def test_the_ordered_binary_plan_compiles_a_trend_the_executor_owns():
    """End to end: the compiled step projects a logistic per-level refit with the primary's roster."""

    context = _case("ordered_binary")
    request = _request(context)
    _, result = _run(context, [json.dumps(_spec_payload(request))])
    plan = result.output
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    trend = next(step for step in plan.steps if step.step_id == "ordinal_trend")
    spec = ordered_stratified_spec_for_step(trend, plan=plan)
    assert spec is not None
    assert spec.continuous_outcome is None
    assert spec.binary_outcome == "death"
    model = spec.trend_model
    assert (model.estimator_kind, model.effect_measure) == ("logistic", "odds_ratio")
    assert model.covariates == list(primary.model_requirements[0].covariates)
    assert model.exposure_term.coding == "ordinal_linear"
    assert model.exposure_term.transform == "declared_level_index"
    assert len(model.exposure_term.levels) == len(spec.ordered_levels) == 4
    primary_terms = {
        term.name: term for term in primary.model_requirements[0].model_terms if term.role != "exposure"
    }
    assert {term.name: term for term in model.model_terms if term.role != "exposure"} == primary_terms


def _with_death(context: ResearchContext, **update) -> ResearchContext:
    return context.model_copy(update={"variables": [
        item.model_copy(update=update) if item.name == "death" else item
        for item in context.variables
    ]})


def _two_continuous_outcomes(context: ResearchContext) -> ResearchContext:
    """Two continuous outcome columns: the multi-outcome contract cannot pick one."""

    extra = ConceptDescriptor(
        name="los_hospital", description="Hospital length of stay", role=VariableRole.OUTCOME,
        dtype="float64", unit="days", source_concept="los_hospital",
    )
    return context.model_copy(update={
        "variables": [*context.variables, extra],
        "cohort": context.cohort.model_copy(update={
            "outcome_columns": ["death", "los_icu", "los_hospital"],
            "requested_outcome_columns": ["death", "los_icu"],
        }),
    })


_STRING_DEATH = {
    "dtype": "object",
    "observed_domain": {"n_unique": 2, "is_binary": True, "levels": ["0", "1"]},
}
_TEMPLATE_CASES = {
    "ordered_binary": lambda: _case("ordered_binary"),
    "ordered_binary_continuous": lambda: _case("ordered_binary_continuous"),
    "ordered_binary_risk_ratio": lambda: _case("ordered_binary", risk_ratio=True),
    "unordered": lambda: _case("unordered"),
    "two_level": lambda: _case("two_level"),
    # A 0/1 endpoint stored as text is not the closed 0/1 domain the owner binds.
    "string_levels": lambda: _with_death(_case("ordered_binary"), **_STRING_DEATH),
    "string_levels_no_endpoint": lambda: _with_death(
        _case("ordered_binary").model_copy(update={"endpoint": None}), **_STRING_DEATH,
    ),
    "string_levels_with_continuous": lambda: _with_death(
        _case("ordered_binary_continuous"), **_STRING_DEATH,
    ),
    "ambiguous_continuous": lambda: _two_continuous_outcomes(_case("ordered_binary_continuous")),
}


@pytest.mark.parametrize("name", sorted(_TEMPLATE_CASES))
def test_the_template_never_plans_a_trend_the_compiler_refuses(name):
    """The template's trend gate and the compiler read one binding, so the plan always compiles."""

    from easyicu.research_agent.planning.family_spec import FamilySpecError

    context = _TEMPLATE_CASES[name]()
    try:
        request = _request(context)
    except FamilySpecError:
        return  # no family plan is proposed, so no step is either
    risk_ratio = name.endswith("risk_ratio")
    _, result = _run(context, [json.dumps(_payload(request, risk_ratio=risk_ratio))])
    plan = result.output
    assert plan is not None, getattr(result, "failure", None) or result
    planned = "ordinal_trend" in {step.step_id for step in plan.steps}
    assert planned is landmark_ordinal_trend_present(request)
    expected = name in {"ordered_binary", "ordered_binary_continuous", "ordered_binary_risk_ratio"}
    assert planned is expected, name
