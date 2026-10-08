"""A survival design names its adjustment roster only while its estimand holds it.

The two survival templates, the binary landmark suite and the continuous-exposure
suite, wrote every covariate's reader label into the selected design's estimand,
a sentence the design contract bounds.  Ten covariates with ordinary labels
exceeded it: the Planner's spec passed, and planning ended in schema validation
(``string_too_long`` at ``estimand``) when the host built the design from that
spec.  The landmark, prediction, phenotyping and trajectory templates already
name a roster only while it fits, and the plan lists it in full.  Synthetic,
case-neutral fixtures (renal replacement therapy, the highest bilirubin, 90-day
death); zero patient rows.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.family_spec import (
    continuous_survival_template,
    survival_template,
)
from easyicu.research_agent.planning.family_spec.contract import design_field_max_length
from easyicu.research_agent.planning.family_spec.plan_language import (
    adjusted_roster_sentence,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.support import continuous_survival, survival_proposal

#: Ordinary reader labels for twelve baseline covariates, as a Planner writes them.
_LABELS = {
    "age": "Age at ICU admission (years)",
    "sex": "Sex recorded at admission",
    "weight": "Body weight at admission (kg)",
    "lact_max": "Highest lactate in the first 24 h (mmol/L)",
    "crea_max": "Highest creatinine in the first 24 h (mg/dL)",
    "plt_min": "Lowest platelet count in the first 24 h (10^3/uL)",
    "wbc_max": "Highest white cell count in the first 24 h (10^3/uL)",
    "map_min": "Lowest mean arterial pressure in the first 24 h (mmHg)",
    "hr_max": "Highest heart rate in the first 24 h (beats/min)",
    "resp_max": "Highest respiratory rate in the first 24 h (breaths/min)",
    "temp_max": "Highest temperature in the first 24 h (C)",
    "gcs_min": "Lowest Glasgow coma scale in the first 24 h",
}
_EXTRA = [
    ("weight", VariableRole.DEMOGRAPHIC, "kg"),
    ("lact_max", VariableRole.LAB, "mmol/L"),
    ("crea_max", VariableRole.LAB, "mg/dL"),
    ("plt_min", VariableRole.LAB, "10^3/uL"),
    ("wbc_max", VariableRole.LAB, "10^3/uL"),
    ("map_min", VariableRole.VITAL, "mmHg"),
    ("hr_max", VariableRole.VITAL, "beats/min"),
    ("resp_max", VariableRole.VITAL, "breaths/min"),
    ("temp_max", VariableRole.VITAL, "C"),
    ("gcs_min", VariableRole.VITAL, None),
]
#: Per family: the template module, its fixtures (context, request, spec, plan),
#: and the estimand the parent wrote for a roster of age and sex.
_FAMILIES = {
    "binary": (
        survival_template,
        (
            survival_proposal.survival_context,
            survival_proposal.survival_request,
            survival_proposal.survival_spec,
            survival_proposal.proposed_survival_plan,
        ),
        "The adjusted hazard ratio for Reader label for mort_90d through 90 days comparing stays "
        "with incident Reader label for rrt by 24 h after ICU admission against stays without it, "
        "among stays alive and event-free at the landmark, adjusted for Reader label for age, "
        "Reader label for sex; reported as a descriptive prognostic association, not a causal "
        "effect.",
    ),
    "continuous": (
        continuous_survival_template,
        (
            continuous_survival.continuous_survival_context,
            continuous_survival.continuous_survival_request,
            continuous_survival.continuous_survival_spec,
            continuous_survival.proposed_continuous_survival_plan,
        ),
        "The adjusted hazard ratio for Reader label for mort_90d through 90 days per step of "
        "Reader label for bili_max (1, 2 or 5 x 10^n mg/dL, within its IQR) (its highest value "
        "from ICU admission to the 24 h landmark), among stays alive and event-free at the "
        "landmark, adjusted for Reader label for age, Reader label for sex; the spline's 10th- "
        "and 90th-percentile contrasts replace it if linearity is rejected; a descriptive "
        "prognostic association, not a causal effect.",
    ),
}


def _covariate(name: str) -> dict:
    coding = "binary" if name == "sex" else "continuous"
    return {
        "name": name,
        "coding": coding,
        "reference_level_index": 0 if coding == "binary" else None,
        "clinical_rationale": "Recorded before the landmark; associated with both exposure and death.",
    }


def _plan(family: str, names, *, labels: dict | None = None):
    make_context, make_request, make_spec, plan = _FAMILIES[family][1]
    base = make_context()
    extra = [
        ConceptDescriptor(
            name=name,
            description=_LABELS[name].lower(),
            role=role,
            dtype="float64",
            unit=unit,
        )
        for name, role, unit in _EXTRA
    ]
    context = make_context(variables=[*base.variables, *extra])
    request = make_request(context)
    roster = [_covariate(name) for name in names]
    if labels is not None:
        labels = {
            **{
                key: f"Reader label for {key}"
                for key in [
                    *request.required_reader_label_keys,
                    *request.level_label_keys,
                ]
            },
            **{name: labels[name] for name in names},
        }
    output, llm = plan(context, make_spec(request, roster, labels=labels))
    assert len(llm.calls) == 1
    return output.design_selection.selected


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_roster_that_fits_is_named_as_before(family) -> None:
    design = _plan(family, ["age", "sex"])

    assert design.estimand == _FAMILIES[family][2]


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_long_roster_is_counted_and_the_plan_names_it_in_full(family) -> None:
    names = list(_LABELS)
    design = _plan(family, names, labels=_LABELS)

    bound = design_field_max_length("estimand")
    counted = f"adjusted for {len(names)} prespecified covariates named in the plan;"
    assert counted in design.estimand and len(design.estimand) <= bound
    # Naming the roster in place would have exceeded the design's bound.
    named = design.estimand.replace(
        counted[:-1], "adjusted for " + ", ".join(_LABELS.values())
    )
    assert len(named) > bound
    # The plan the researcher reviews still names every covariate.
    plan_text = "\n".join(design.reviewable_plan)
    assert all(label in plan_text for label in _LABELS.values())


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_bound_is_the_design_owners(family, monkeypatch) -> None:
    named = _plan(family, ["age", "sex"], labels=_LABELS).estimand
    assert (
        "adjusted for Age at ICU admission (years), Sex recorded at admission;" in named
    )
    owner = design_field_max_length
    monkeypatch.setattr(
        _FAMILIES[family][0],
        "design_field_max_length",
        lambda field: len(named) - 1 if field == "estimand" else owner(field),
    )

    design = _plan(family, ["age", "sex"], labels=_LABELS)

    assert (
        "adjusted for 2 prespecified covariates named in the plan;" in design.estimand
    )
    assert len(design.estimand) < len(named)


def test_the_shared_sentence_names_counts_or_states_no_roster() -> None:
    head, tail = "The estimate, ", "; descriptive."
    labels = ["First covariate", "Second covariate"]
    named = f"{head}adjusted for First covariate, Second covariate{tail}"

    def sentence(labels, bound, unadjusted="unused"):
        return adjusted_roster_sentence(
            head, labels, tail, bound=bound, unadjusted=unadjusted
        )

    # Named exactly at the bound, counted one character under it.
    assert sentence(labels, len(named)) == named
    assert sentence(labels, len(named) - 1) == (
        f"{head}adjusted for 2 prespecified covariates named in the plan{tail}"
    )
    assert sentence(labels[:1], 10) == (
        f"{head}adjusted for 1 prespecified covariate named in the plan{tail}"
    )
    assert sentence([], 10, "without adjustment") == f"{head}without adjustment{tail}"
