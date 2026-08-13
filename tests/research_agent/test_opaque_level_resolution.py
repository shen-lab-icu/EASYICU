"""The host asked the Planner for values its own privacy boundary withholds.

canary10's E3 step ``06_primary_mortality_association`` died the first time the
ordinal-contrast capability met a real plan.  The Planner filled every new
field correctly::

    exposure_levels        = ['__easyicu_level_1__', ..., '__easyicu_level_4__']
    exposure_reference     = '__easyicu_level_1__'
    primary_contrast_level = '__easyicu_level_4__'

Those are the host's own placeholders.  ``opaque_level_tokens`` exists to give
an external model a categorical column's cardinality "without revealing local
literals", so placeholders are the only vocabulary the Planner has for a
withheld domain -- and the executor then refused, correctly given what it was
told::

    AdjustedAssociationError: the bound cohort holds levels of 'aki_stage_max'
    the plan never declared: '0', '1', '2', '3'

Table 1 had solved this a month earlier and nothing else had.  Measured across
the 91 recorded canonical-9 plans: 171 of 256 Table 1 level declarations came
back in placeholders and Table 1 resolved every one; the two
``exposure_outcome_distribution_spec`` designs declared in placeholders both
died and were rescued only by a replan re-emitting ``[0, 1]`` -- an LLM
guessing a binary encoding, which is not a mechanism.

So the resolution belongs in the authority layer, applied to every declared
level set, sharing ONE ordering authority.  Ordering is the whole risk: token
``N`` denotes the ``N``-th observed level, so a second implementation that
ordered levels differently would attach the reference and the headline contrast
to the wrong stages and print a wrong odds ratio under the right label.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent.authority import declared_levels, table_one_binding
from easyicu.research_agent.authority.declared_levels import (
    DeclaredLevelError,
    bind_step_declared_levels,
    execution_distribution_spec,
    execution_model_requirement,
    level_spelling,
    observed_levels_for,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    adjusted_association_executor_code,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_code,
)
from easyicu.research_agent.research_context.prompt_variables import (
    opaque_level_tokens,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)

_STAGE_TOKENS = list(opaque_level_tokens(4))
_BINARY_TOKENS = list(opaque_level_tokens(2))


def _context(**domains) -> ResearchContext:
    """A context whose observed domains are the ones the cohort really had."""

    return ResearchContext(
        research_question="Does AKI stage grade in-hospital mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[
            ConceptDescriptor(name=name, dtype=dtype, observed_domain=domain)
            for name, (dtype, domain) in domains.items()
        ],
    )


def _e3_context() -> ResearchContext:
    """The exact recorded shape from canary10's ``research_context.json``."""

    return _context(
        aki_stage_max=(
            "float64",
            {
                "n_unique": 4,
                "is_constant": False,
                "is_binary": False,
                "min": 0.0,
                "max": 3.0,
                "levels": [0.0, 1.0, 2.0, 3.0],
            },
        ),
        death=("int64", {"n_unique": 2, "is_binary": True, "levels": [0, 1]}),
    )


def _requirement(**overrides) -> PlannedModelRequirement:
    payload = dict(
        requirement_id="primary_mortality_association",
        outcome="death",
        outcome_type="binary",
        method_family="logistic_regression",
        exposure_source="aki_stage_max",
        analysis_role="primary",
        analysis_set="complete_case",
        covariates=["age", "sex"],
        exposure_levels=list(_STAGE_TOKENS),
        exposure_reference_level=_STAGE_TOKENS[0],
        primary_contrast_level=_STAGE_TOKENS[-1],
    )
    payload.update(overrides)
    exposure_levels = payload["exposure_levels"]
    exposure_term = (
        {
            "name": "aki_stage_max",
            "role": "exposure",
            "coding": "categorical",
            "levels": list(exposure_levels),
            "reference_level": payload["exposure_reference_level"],
            "transform": "treatment_contrast",
        }
        if exposure_levels is not None
        else {
            "name": "aki_stage_max",
            "role": "exposure",
            "coding": "continuous",
            "transform": "identity",
        }
    )
    payload.setdefault(
        "model_terms",
        [
            exposure_term,
            {
                "name": "age",
                "role": "covariate",
                "coding": "continuous",
                "transform": "identity",
            },
            {
                "name": "sex",
                "role": "covariate",
                "coding": "binary",
                "levels": ["Female", "Male"],
                "reference_level": "Female",
                "transform": "treatment_contrast",
            },
        ],
    )
    return PlannedModelRequirement.model_validate(payload)


def _model_step(**overrides) -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "06_primary_mortality_association",
            "planned_analysis_role": "primary",
            "intent": "Estimate mortality by KDIGO stage against stage 0.",
            "inputs": ["cohort:analysis_set"],
            "expected_outputs": ["table:adjusted_association_estimates"],
            "method": "adjusted_association_models",
            "model_requirements": [_requirement(**overrides).model_dump(mode="json")],
            "input_consumption_contracts": [
                {
                    "schema_version": "easyicu.artifact_consumption/1",
                    "input_key": "cohort:analysis_set",
                    "mode": "all_rows",
                    "role_column": None,
                    "expected_roles": [],
                }
            ],
        }
    )


def _distribution_step(**overrides) -> AnalysisStep:
    spec = dict(
        schema_version="easyicu.exposure_outcome_distribution/2",
        exposure="sep3_sofa2_max",
        exposure_levels=list(_BINARY_TOKENS),
        outcome="death",
        outcome_levels=list(_BINARY_TOKENS),
        outcome_positive_value=_BINARY_TOKENS[1],
        level_match_policy="exact_typed",
        denominator_policy="all_declared_rows",
        missing_outcome_policy="fail_closed",
        confidence_level=0.95,
    )
    spec.update(overrides)
    return AnalysisStep.model_validate(
        {
            "step_id": "05_exposure_outcome_distribution",
            "planned_analysis_role": "auxiliary",
            "intent": "Cross the exposure with the outcome.",
            "inputs": ["cohort:analysis_set", "sep3_sofa2_max", "death"],
            "expected_outputs": ["table:exposure_outcome_distribution"],
            "method": "descriptive",
            "exposure_outcome_distribution_spec": spec,
            "input_consumption_contracts": [
                {
                    "schema_version": "easyicu.artifact_consumption/1",
                    "input_key": "cohort:analysis_set",
                    "mode": "all_rows",
                    "role_column": None,
                    "expected_roles": [],
                }
            ],
        }
    )


def _distribution_context() -> ResearchContext:
    return _context(
        sep3_sofa2_max=(
            "float64",
            {"n_unique": 2, "is_binary": True, "levels": [0.0, 1.0]},
        ),
        death=("int64", {"n_unique": 2, "is_binary": True, "levels": [0, 1]}),
    )


# ---------------------------------------------------------------------------
# The measured defect
# ---------------------------------------------------------------------------


def test_the_canary10_declaration_resolves_to_the_host_s_own_levels() -> None:
    """The exact shape that killed E3 step 06, end to end."""

    step = _model_step()
    bind_step_declared_levels(step, _e3_context())
    bound = execution_model_requirement(step, step.model_requirements[0])

    assert bound.exposure_levels == ["0", "1", "2", "3"]
    assert bound.exposure_reference_level == "0"
    assert bound.primary_contrast_level == "3"


def test_the_generated_model_code_carries_no_placeholder() -> None:
    """Resolving in the authority layer is worthless if the script re-reads the plan."""

    step = _model_step()
    bind_step_declared_levels(step, _e3_context())

    code = adjusted_association_executor_code(step)

    assert "__easyicu_level_" not in code
    assert "'levels': ['0', '1', '2', '3']" in code.replace('"', "'")


def test_the_public_plan_still_holds_the_placeholders() -> None:
    """The resolution is host-only; the plan that is serialised and re-sent is not.

    If the binding wrote back into the step, the next Planner or Coder prompt
    built from this plan would carry the cohort literals the placeholders exist
    to withhold.
    """

    step = _model_step()
    bind_step_declared_levels(step, _e3_context())

    assert step.model_requirements[0].exposure_levels == _STAGE_TOKENS
    assert step.model_requirements[0].exposure_reference_level == _STAGE_TOKENS[0]
    assert "__easyicu_level_" in step.model_dump_json()


# ---------------------------------------------------------------------------
# Ordering -- the whole risk
# ---------------------------------------------------------------------------


def test_one_ordering_authority_not_two_copies_that_agree_today() -> None:
    """Table 1 and the declared-level owner must be the SAME function.

    Equality of today's output would not do: two implementations that agreed on
    ``[0, 1]`` could differ on a four-level scale or a string category, and the
    only symptom would be a headline contrast attached to the wrong level.
    """

    assert table_one_binding._observed_levels is observed_levels_for
    assert table_one_binding._resolve_levels is declared_levels.resolve_typed_levels


def test_the_nth_placeholder_is_the_nth_observed_level() -> None:
    """Position, not value: a reference at index 2 must land on the 3rd level."""

    step = _model_step(
        exposure_reference_level=_STAGE_TOKENS[2],
        primary_contrast_level=_STAGE_TOKENS[1],
    )
    bind_step_declared_levels(step, _e3_context())
    bound = execution_model_requirement(step, step.model_requirements[0])

    assert bound.exposure_reference_level == "2"
    assert bound.primary_contrast_level == "1"


def test_the_spelling_rule_is_shared_with_the_executor() -> None:
    """The declaration and the cohort must be compared in one spelling.

    A float-backed stage column reaches the executor as ``3.0`` while the
    resolved declaration is the string ``"3"``. Both sides go through this one
    function, so they cannot drift into two spellings of one level.
    """

    from easyicu.research_agent.execution import model_matrix

    assert model_matrix.level_spelling is level_spelling
    assert level_spelling(3.0) == "3"
    assert level_spelling(2.5) == "2.5"
    assert level_spelling(True) == "true"


# ---------------------------------------------------------------------------
# What must NOT be resolved
# ---------------------------------------------------------------------------


def test_a_real_declaration_is_left_exactly_as_the_planner_wrote_it() -> None:
    """Levels are the Planner's to declare when it actually knows them.

    ``PlannedModelRequirement.exposure_levels`` is typed ``List[str]``, so a
    string spelling of a numeric level is the only form the field can hold.
    Forcing Table 1's typed comparison onto it would refuse every literal
    declaration the field was designed to carry.
    """

    step = _model_step(
        exposure_levels=["0", "1", "2", "3"],
        exposure_reference_level="0",
        primary_contrast_level="3",
    )
    bind_step_declared_levels(step, _e3_context())
    bound = execution_model_requirement(step, step.model_requirements[0])

    assert bound is step.model_requirements[0]
    assert bound.exposure_levels == ["0", "1", "2", "3"]


def test_a_requirement_with_no_level_set_is_untouched() -> None:
    """A binary or continuous exposure declares none of the three fields."""

    step = _model_step(
        exposure_levels=None,
        exposure_reference_level=None,
        primary_contrast_level=None,
    )
    bind_step_declared_levels(step, _e3_context())

    assert step._declared_level_binding is None
    assert (
        execution_model_requirement(step, step.model_requirements[0]).exposure_levels
        is None
    )


def test_an_unbound_step_falls_back_to_the_declaration() -> None:
    """Every accessor must work on a step nobody bound, e.g. in a unit test."""

    step = _model_step()

    bound = execution_model_requirement(step, step.model_requirements[0])

    assert bound.exposure_levels == _STAGE_TOKENS


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_a_placeholder_set_the_host_cannot_bind_is_refused() -> None:
    """Falling through would send a token into a comparison against a stage.

    The executor would then report the whole declared level set as absent from
    the cohort -- naming the host's own placeholders as if the Planner had
    invented them.
    """

    step = _model_step()
    context = _context(
        aki_stage_max=("float64", {"n_unique": 4, "is_binary": False}),
    )

    with pytest.raises(DeclaredLevelError) as excinfo:
        bind_step_declared_levels(step, context)

    assert "aki_stage_max" in str(excinfo.value)
    assert "opaque level placeholders" in str(excinfo.value)


def test_a_placeholder_count_that_does_not_match_the_column_is_refused() -> None:
    """Three placeholders against a four-level column names no level at all."""

    step = _model_step(
        exposure_levels=_STAGE_TOKENS[:3],
        exposure_reference_level=_STAGE_TOKENS[0],
        primary_contrast_level=_STAGE_TOKENS[2],
    )

    with pytest.raises(DeclaredLevelError):
        bind_step_declared_levels(step, _e3_context())


def test_a_placeholder_scalar_outside_the_declared_set_is_refused() -> None:
    """A reference the level set never listed cannot be mapped by index."""

    step = _model_step(
        exposure_levels=["0", "1", "2", "3"],
        exposure_reference_level="0",
        primary_contrast_level="3",
    )
    # Slip a placeholder into the level set only, so the scalars are literal
    # while the set is not -- a mixture the host must not silently pick from.
    step.model_requirements[0].exposure_levels = [
        _STAGE_TOKENS[0],
        "1",
        "2",
        "3",
    ]

    with pytest.raises(DeclaredLevelError):
        bind_step_declared_levels(step, _e3_context())


def test_the_refusal_names_no_cohort_literal() -> None:
    """A message that quotes the withheld values defeats the boundary it guards.

    Table 1's refusal already reports counts and types only; this one must too,
    because both messages travel back to the Planner.

    The refusal has to fire on a column that HAS a level set, or the assertion
    is vacuous -- the first version of this test used a column with no levels
    at all, so there was nothing available to leak and a mutation that appended
    the observed literals to the message passed it unchanged.
    """

    step = _model_step(
        exposure_levels=_STAGE_TOKENS[:3],
        exposure_reference_level=_STAGE_TOKENS[0],
        primary_contrast_level=_STAGE_TOKENS[2],
    )
    context = _e3_context()
    withheld = observed_levels_for(
        name="aki_stage_max",
        variables={variable.name: variable for variable in context.variables},
    )
    assert withheld, "the leak this test guards needs literals to leak"

    with pytest.raises(DeclaredLevelError) as excinfo:
        bind_step_declared_levels(step, context)

    message = str(excinfo.value)
    assert "declared_count=3" in message and "observed_level_count=4" in message
    for value in withheld:
        assert repr(value) not in message


# ---------------------------------------------------------------------------
# The second consumer: the exposure/outcome distribution design
# ---------------------------------------------------------------------------


def test_the_distribution_design_resolves_levels_and_its_positive_value() -> None:
    """The two recorded runs that were rescued only by a lucky replan."""

    step = _distribution_step()
    bind_step_declared_levels(step, _distribution_context())
    bound = execution_distribution_spec(step)

    assert bound.exposure_levels == [0.0, 1.0]
    assert bound.outcome_levels == [0, 1]
    assert bound.outcome_positive_value == 1
    assert step.exposure_outcome_distribution_spec.exposure_levels == _BINARY_TOKENS


def test_the_distribution_design_resolves_its_risk_difference_levels() -> None:
    """The host must not leave new scalar contrast fields as opaque tokens."""

    step = _distribution_step(
        risk_difference_contrast={
            "reference_exposure_level": _BINARY_TOKENS[0],
            "comparison_exposure_level": _BINARY_TOKENS[1],
        }
    )
    bind_step_declared_levels(step, _distribution_context())
    bound = execution_distribution_spec(step)

    assert bound.risk_difference_contrast.reference_exposure_level == 0.0
    assert bound.risk_difference_contrast.comparison_exposure_level == 1.0
    assert (
        step.exposure_outcome_distribution_spec.risk_difference_contrast.reference_exposure_level
        == _BINARY_TOKENS[0]
    )


def test_the_generated_distribution_code_carries_no_placeholder() -> None:
    step = _distribution_step(
        risk_difference_contrast={
            "reference_exposure_level": _BINARY_TOKENS[0],
            "comparison_exposure_level": _BINARY_TOKENS[1],
        }
    )
    bind_step_declared_levels(step, _distribution_context())

    assert "__easyicu_level_" not in exposure_outcome_distribution_executor_code(step)


def test_a_literal_distribution_design_is_left_alone() -> None:
    """17 of the 19 recorded designs declared real levels; none may be rewritten."""

    step = _distribution_step(
        exposure_levels=[0.0, 1.0],
        outcome_levels=[0, 1],
        outcome_positive_value=1,
    )
    bind_step_declared_levels(step, _distribution_context())

    assert execution_distribution_spec(step) is step.exposure_outcome_distribution_spec


# ---------------------------------------------------------------------------
# Wiring -- a resolution nobody calls resolves nothing
# ---------------------------------------------------------------------------


def _rebind_sites() -> list[tuple[str, ast.AST]]:
    """Every statement list that rebinds Table 1's private execution levels."""

    import easyicu.research_agent as package

    root = Path(package.__file__).parent
    sites: list[tuple[str, ast.AST]] = []
    for relative in (
        "agents/core.py",
        "authority/plan_authority.py",
        "execution/phase.py",
        "pipeline.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            calls = {
                statement.value.func.id
                for statement in body
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
            }
            if "bind_table_one_execution_spec" in calls:
                sites.append((relative, calls))
    return sites


def test_every_table_one_rebind_also_rebinds_the_declared_levels() -> None:
    """One rule: wherever the host restores its private Table 1 levels, these too.

    A plan is rebound after planning, after a replan, after an authority
    projection, and on resume.  A site that rebound only Table 1 would execute
    a step whose exposure levels were still placeholders -- which is the whole
    defect, reintroduced at whichever site was missed.
    """

    sites = _rebind_sites()

    assert len(sites) == 4, [name for name, _ in sites]
    for relative, calls in sites:
        assert "bind_step_declared_levels" in calls, relative


def test_the_resume_path_rebinds_too() -> None:
    """The resumed plan is the one on disk, so it still carries placeholders."""

    from easyicu.research_agent import pipeline

    source = inspect.getsource(pipeline)
    restore = source.index("restore_table_one_private_checkpoint(")
    assert "bind_step_declared_levels" in source[restore : restore + 900]
