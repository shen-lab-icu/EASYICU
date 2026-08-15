"""One owner for reading a Planner-declared level set the host wrote itself.

The host publishes ``__easyicu_level_N__`` placeholders to the Planner *on
purpose*.  Cohort literals are local evidence, so an external model is told a
categorical column's cardinality and never its values
(:func:`..research_context.prompt_variables.opaque_level_tokens`).  A level set
that comes back in that vocabulary is therefore the **expected** answer, not a
malformed one, and every consumer of a declared level set has to be able to
read it.

Table 1 could.  Nothing else could, and the corpus shows what that costs.
Across 91 recorded plans the Planner answered in placeholders 171 times out of
256 Table 1 level declarations -- Table 1 resolved every one.  The two other
declaration sites fared differently:

* ``exposure_outcome_distribution_spec`` was declared in placeholders twice.
  Both steps died on the first attempt and were rescued only because a replan
  re-emitted the design with ``[0, 1]`` -- an LLM *guessing* a binary encoding.
  That guess is not a mechanism: it happens to be right for a 0/1 flag and has
  no chance on a four-level stage scale or a string category.
* ``model_requirements[*].exposure_levels`` was declared once, on the very
  first run after the ordinal-contrast capability landed, and the step died:
  the executor was handed four placeholders and correctly refused a cohort
  holding ``0/1/2/3``.

So the capability asked the Planner for values the privacy boundary forbids it
to know, which is the host judging a counterpart against a contract it made
unsatisfiable.  This module resolves the host's own placeholders back to the
host's own observations, in the host's own layer, before any code is written.

ORDERING IS THE WHOLE RISK.  Token ``N`` denotes the ``N``-th observed level,
so a second implementation that ordered levels differently would attach the
reference and the headline contrast to the wrong stages -- a wrong odds ratio
printed under the right label, which no downstream check can catch.  There is
therefore exactly one ordering authority, :func:`observed_levels_for`, and
Table 1 imports it from here rather than keeping the copy it used to own.

What each caller may NOT share is the comparison it applies to a *literal*
declaration, because the two fields do not admit the same one.  Table 1's
levels are ``Any`` and distinguish ``1`` from ``"1"`` from ``True``, so it
matches by type and value.  ``PlannedModelRequirement.exposure_levels`` is
typed ``List[str]``: a string spelling of a numeric level is the only form the
field can hold, and its executor already reconciles the two sides through
:func:`level_spelling`.  Forcing Table 1's typed comparison onto it would
refuse every literal declaration the field was designed to carry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..contracts.model_terms import level_spelling
from ..research_context.prompt_variables import opaque_level_tokens
from ..schema import (
    AnalysisStep,
    ExposureOutcomeDistributionSpec,
    PlannedModelRequirement,
    ResearchContext,
)

OPAQUE_LEVEL_PREFIX = "__easyicu_level_"


def observed_levels_for(*, name: str, variables: Dict[str, Any]) -> List[Any]:
    """Return the host-observed levels of one column, in their canonical order.

    THE single ordering authority for every declared level set.  The order is
    the one :func:`..cohort.artifact_facts.observed_domain_for_series` sealed
    into the cohort's own record (sorted), so the ``N``-th opaque token and the
    ``N``-th observed level denote the same value for every consumer.
    """

    variable = variables.get(name)
    if variable is None or not variable.observed_domain:
        return []
    domain = variable.observed_domain
    levels = domain.get("levels")
    if isinstance(levels, list):
        return list(levels)
    if not domain.get("is_binary"):
        return []
    dtype = str(variable.dtype or "").lower()
    if dtype.startswith(("int", "uint")):
        return [0, 1]
    if dtype.startswith("bool"):
        return [False, True]
    if dtype.startswith(("float", "double")):
        return [0.0, 1.0]
    return []


def is_opaque_level(value: Any) -> bool:
    """True for one of the host's own published placeholders."""

    return isinstance(value, str) and value.startswith(OPAQUE_LEVEL_PREFIX)


class DeclaredLevelError(ValueError):
    """A declared level set the host published but cannot bind back."""


def _typed_token(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def resolve_typed_levels(
    *,
    name: str,
    declared: List[Any],
    variables: Dict[str, Any],
) -> tuple[List[Any], List[Any]]:
    """Resolve a level set declared as ``Any``; return ``(execution, observed)``.

    Used where the field distinguishes ``1`` from ``"1"`` from ``True`` --
    Table 1 and the exposure/outcome distribution design.
    """

    observed = observed_levels_for(name=name, variables=variables)
    if not observed:
        return list(declared), []
    opaque = list(opaque_level_tokens(len(observed)))
    if opaque and list(declared) == opaque:
        return observed, observed
    if {_typed_token(value) for value in declared} == {
        _typed_token(value) for value in observed
    }:
        return list(declared), observed
    # JSON has one ``number`` type even though Python/pandas distinguish
    # integral and floating scalar representations.  A Planner declaration
    # of ``[0, 1]`` therefore denotes the same closed binary domain as a
    # float-backed cohort column whose verified levels are ``[0.0, 1.0]``.
    # Canonicalise execution back to the host-observed scalar types; never
    # apply this equivalence to booleans or categorical strings.
    declared_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in declared
    )
    observed_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in observed
    )
    if (
        declared_numeric
        and observed_numeric
        and len(declared) == len(observed)
        and {float(value) for value in declared} == {float(value) for value in observed}
    ):
        return observed, observed
    safe_expected = opaque or ["<host-observed numeric scalar types>"]
    raise DeclaredLevelError(
        "Planner levels for "
        f"{name!r} must preserve the exact observed scalar types or use the "
        f"exact host-safe tokens {safe_expected!r}; expected_count="
        f"{len(observed)}, declared_count={len(declared)}, declared_types="
        f"{[type(value).__name__ for value in declared]!r}. No observed "
        "category literal is available to the Provider."
    )


def _resolve_opaque_string_levels(
    *,
    name: str,
    declared: Sequence[str],
    variables: Dict[str, Any],
) -> Optional[List[str]]:
    """Return host spellings when the declaration is the host's placeholder set.

    ``None`` means the Planner declared real levels, which are its own to
    declare and stay untouched -- the executor still checks them against the
    bound cohort.  A placeholder set the host cannot bind raises instead of
    falling through, because the executor downstream would compare a token
    against a stage and report the level set as absent from the cohort.
    """

    values = list(declared)
    if not any(is_opaque_level(value) for value in values):
        return None
    observed = observed_levels_for(name=name, variables=variables)
    expected = list(opaque_level_tokens(len(observed))) if observed else []
    if not expected or values != expected:
        raise DeclaredLevelError(
            f"the declared level set for {name!r} uses the host's own opaque "
            "level placeholders but does not match the exact published set: "
            f"declared_count={len(values)}, observed_level_count={len(observed)}. "
            "The host publishes placeholders so no cohort literal reaches the "
            "Provider, and it will not guess which level each one meant."
        )
    return [level_spelling(value) for value in observed]


def _resolve_opaque_scalar(
    *,
    name: str,
    declared: Any,
    levels: Sequence[Any],
    resolved_levels: Sequence[Any],
    field: str,
) -> Any:
    """Map one placeholder scalar through the same index its level set used."""

    if not is_opaque_level(declared):
        return declared
    try:
        index = list(levels).index(declared)
    except ValueError as exc:
        raise DeclaredLevelError(
            f"{field} for {name!r} is an opaque level placeholder that is not "
            "one of the declared exposure levels, so the host cannot tell which "
            "level it meant"
        ) from exc
    return resolved_levels[index]


class StepDeclaredLevelBinding:
    """Host-only resolved declarations; never serialised into any plan or prompt."""

    __slots__ = ("step_id", "model_requirements", "distribution_spec")

    def __init__(
        self,
        *,
        step_id: str,
        model_requirements: Dict[str, PlannedModelRequirement],
        distribution_spec: Optional[ExposureOutcomeDistributionSpec],
    ) -> None:
        self.step_id = step_id
        self.model_requirements = model_requirements
        self.distribution_spec = distribution_spec


def _bound_model_requirement(
    requirement: PlannedModelRequirement,
    variables: Dict[str, Any],
) -> Optional[PlannedModelRequirement]:
    updates: Dict[str, Any] = {}
    bound_terms = []
    terms_changed = False
    for term in requirement.model_terms or ():
        declared_term_levels = list(term.levels or ())
        if not declared_term_levels:
            bound_terms.append(term)
            continue
        resolved_term_levels = _resolve_opaque_string_levels(
            name=term.name,
            declared=declared_term_levels,
            variables=variables,
        )
        if resolved_term_levels is None:
            bound_terms.append(term)
            continue
        reference = (
            _resolve_opaque_scalar(
                name=term.name,
                declared=term.reference_level,
                levels=declared_term_levels,
                resolved_levels=resolved_term_levels,
                field="reference_level",
            )
            if term.reference_level is not None
            else None
        )
        bound_terms.append(
            term.model_copy(
                update={
                    "levels": resolved_term_levels,
                    "reference_level": reference,
                }
            )
        )
        terms_changed = True
        if term.role == "exposure" and term.transform == "treatment_contrast":
            updates["exposure_levels"] = resolved_term_levels
            updates["exposure_reference_level"] = reference
            updates["primary_contrast_level"] = _resolve_opaque_scalar(
                name=term.name,
                declared=requirement.primary_contrast_level,
                levels=declared_term_levels,
                resolved_levels=resolved_term_levels,
                field="primary_contrast_level",
            )
    if terms_changed:
        updates["model_terms"] = bound_terms

    declared = list(requirement.exposure_levels or [])
    if declared and "exposure_levels" not in updates:
        resolved = _resolve_opaque_string_levels(
            name=requirement.exposure_source,
            declared=declared,
            variables=variables,
        )
        if resolved is not None:
            updates.update(
                {
                    "exposure_levels": resolved,
                    "exposure_reference_level": _resolve_opaque_scalar(
                        name=requirement.exposure_source,
                        declared=requirement.exposure_reference_level,
                        levels=declared,
                        resolved_levels=resolved,
                        field="exposure_reference_level",
                    ),
                    "primary_contrast_level": _resolve_opaque_scalar(
                        name=requirement.exposure_source,
                        declared=requirement.primary_contrast_level,
                        levels=declared,
                        resolved_levels=resolved,
                        field="primary_contrast_level",
                    ),
                }
            )
    return requirement.model_copy(update=updates) if updates else None


def _bound_distribution_spec(
    spec: ExposureOutcomeDistributionSpec,
    variables: Dict[str, Any],
) -> Optional[ExposureOutcomeDistributionSpec]:
    exposure_levels, _ = resolve_typed_levels(
        name=spec.exposure,
        declared=list(spec.exposure_levels),
        variables=variables,
    )
    outcome_levels, _ = resolve_typed_levels(
        name=spec.outcome,
        declared=list(spec.outcome_levels),
        variables=variables,
    )
    positive = _resolve_opaque_scalar(
        name=spec.outcome,
        declared=spec.outcome_positive_value,
        levels=list(spec.outcome_levels),
        resolved_levels=outcome_levels,
        field="outcome_positive_value",
    )
    contrast = spec.risk_difference_contrast
    resolved_contrast = contrast
    if contrast is not None:
        resolved_contrast = contrast.model_copy(
            update={
                "reference_exposure_level": _resolve_opaque_scalar(
                    name=spec.exposure,
                    declared=contrast.reference_exposure_level,
                    levels=list(spec.exposure_levels),
                    resolved_levels=exposure_levels,
                    field="risk_difference_contrast.reference_exposure_level",
                ),
                "comparison_exposure_level": _resolve_opaque_scalar(
                    name=spec.exposure,
                    declared=contrast.comparison_exposure_level,
                    levels=list(spec.exposure_levels),
                    resolved_levels=exposure_levels,
                    field="risk_difference_contrast.comparison_exposure_level",
                ),
            }
        )
    if (
        exposure_levels == list(spec.exposure_levels)
        and outcome_levels == list(spec.outcome_levels)
        and _typed_token(positive) == _typed_token(spec.outcome_positive_value)
        and resolved_contrast == contrast
    ):
        return None
    return spec.model_copy(
        update={
            "exposure_levels": exposure_levels,
            "outcome_levels": outcome_levels,
            "outcome_positive_value": positive,
            "risk_difference_contrast": resolved_contrast,
        }
    )


def bind_step_declared_levels(
    step: AnalysisStep,
    context: ResearchContext,
) -> None:
    """Attach host-only resolved level sets without mutating the outbound plan."""

    variables = {variable.name: variable for variable in context.variables}
    requirements: Dict[str, PlannedModelRequirement] = {}
    for requirement in step.model_requirements or []:
        bound = _bound_model_requirement(requirement, variables)
        if bound is not None:
            requirements[requirement.requirement_id] = bound
    spec = step.exposure_outcome_distribution_spec
    bound_spec = _bound_distribution_spec(spec, variables) if spec is not None else None
    if not requirements and bound_spec is None:
        step._declared_level_binding = None
        return
    step._declared_level_binding = StepDeclaredLevelBinding(
        step_id=step.step_id,
        model_requirements=requirements,
        distribution_spec=bound_spec,
    )


def _binding(step: AnalysisStep) -> Optional[StepDeclaredLevelBinding]:
    binding = getattr(step, "_declared_level_binding", None)
    if not isinstance(binding, StepDeclaredLevelBinding):
        return None
    if binding.step_id != step.step_id:
        raise DeclaredLevelError("stale declared-level binding")
    return binding


def execution_model_requirement(
    step: AnalysisStep,
    requirement: PlannedModelRequirement,
) -> PlannedModelRequirement:
    """Return the requirement as it must execute, placeholders resolved."""

    binding = _binding(step)
    if binding is None:
        return requirement
    return binding.model_requirements.get(requirement.requirement_id, requirement)


def execution_distribution_spec(
    step: AnalysisStep,
) -> Optional[ExposureOutcomeDistributionSpec]:
    """Return the distribution design as it must execute, placeholders resolved."""

    binding = _binding(step)
    if binding is None or binding.distribution_spec is None:
        return step.exposure_outcome_distribution_spec
    return binding.distribution_spec


__all__ = [
    "OPAQUE_LEVEL_PREFIX",
    "DeclaredLevelError",
    "StepDeclaredLevelBinding",
    "bind_step_declared_levels",
    "execution_distribution_spec",
    "execution_model_requirement",
    "is_opaque_level",
    "level_spelling",
    "observed_levels_for",
    "resolve_typed_levels",
]
