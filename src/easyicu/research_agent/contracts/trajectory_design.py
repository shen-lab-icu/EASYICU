"""The user-reviewable design of a fixed-window trajectory clustering study.

Owner
-----
This module owns the small typed vocabulary a human actually reviews before a
longitudinal trajectory run: which coordinates are modelled, over which fixed
window and grid, in which population, how many clusters are admissible, and
what counts as a stable solution.  Everything else the sealed trajectory
authority needs -- the model family, the fit engine, the selection criterion,
seeds, tolerances, boundary actions and reason codes -- is host policy with
exactly one implementation, so it lives here as a constant rather than as a
slot someone has to fill in.  The model follows the coordinates' declared
measurement, not a reviewer's choice: a coordinate the host declares ordinal
(an organ score's levels) enters a mixed-mode latent class model as a
categorical indicator; a design whose coordinates are all continuous keeps the
Gaussian mixture.  The population is the cohort predicates the reviewed plan
states, unchanged; it is applied by the host's cohort materializer, so a
predicate is part of the design only when the owner can apply it.

Keeping both halves in one dependency-neutral owner is what stops the two
routes to the same executor from drifting: the benchmark protocol compiler and
the Web projection both build the same authority body from the same policy.

This module never reads patient data and never guesses a design: a declared
field that cannot be executed is a failure with a named field, not a silently
substituted default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ELIGIBILITY_COORDINATE_PREFIX",
    "FIXED_WINDOW_TRAJECTORY_DEFAULTS",
    "TRAJECTORY_HOST_POLICY",
    "TRAJECTORY_MIXED_MODE_POLICY",
    "TRAJECTORY_OUTCOME_DESCRIPTION_RULE",
    "TRAJECTORY_PRIMARY_ACTION",
    "FixedWindowTrajectoryDesign",
    "TRAJECTORY_OWNER_PLANNER_RULE",
    "TrajectoryDesignError",
    "eligibility_coordinates",
    "executable_trajectory_coordinates",
    "load_trajectory_design",
    "longitudinal_capability_note",
    "normalize_trajectory_design",
    "proposed_trajectory_coordinates",
    "sealed_trajectory_authority_body",
    "time_varying_role",
    "trajectory_coordinate_measurement",
    "trajectory_coordinate_proposal",
    "trajectory_population_design",
    "trajectory_population_issues",
    "trajectory_window_design",
]

_CONCEPT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")

# One implementation each; a second one turns the entry into a design field.
TRAJECTORY_HOST_POLICY = MappingProxyType(
    {
        "aggregation": "max",
        "scaling_method": "pooled_coordinate_wise_z_score",
        "scaling_ddof": 0,
        "scaling_zero_variance_action": "fail_closed",
        "model_family": "latent_class_diagonal_gaussian_mixture",
        "fit_method": "observed_data_em_diagonal_gaussian_mixture",
        "covariance_type": "diag",
        "selection_criterion": "bic",
        "selection_rule": "minimum",
        "bic_sample_size": "frozen_population_rows",
        "bic_parameter_count": (
            "mixture_weights_k_minus_1_plus_2_k_per_coordinate"
        ),
        "bic_tie_break": "smaller_k",
        "candidate_fit_base_seed": 1729,
        # Every candidate fit and stability refit keeps the best of ten
        # deterministic EM starts; one start can stop in a poor local optimum.
        "fit_engine": "easyicu_observed_data_diag_gmm_best_of_10_v2",
        "candidate_fit_max_iter": 200,
        "candidate_fit_tolerance": 1e-6,
        "candidate_fit_regularization": 1e-6,
        "upper_boundary_reason_code": "TRAJECTORY_NO_INTERIOR_BIC_OPTIMUM",
        "minimum_cluster_fraction_reason_code": (
            "TRAJECTORY_MINIMUM_CLUSTER_FRACTION_NOT_MET"
        ),
    }
)

#: The host policy entries that change when a coordinate is declared ordinal:
#: ordinal levels are categorical indicators and only continuous coordinates
#: are z-scored.  A Gaussian mixture on tied integer levels keeps adding
#: narrow components, so its BIC keeps falling with the class count.
TRAJECTORY_MIXED_MODE_POLICY = MappingProxyType(
    {
        "scaling_method": "continuous_coordinate_wise_z_score",
        "model_family": "latent_class_mixed_mode",
        "fit_method": "observed_data_em_mixed_mode_latent_class",
        "bic_parameter_count": (
            "mixture_weights_k_minus_1_plus_k_per_indicator_free_parameters"
        ),
        "fit_engine": "easyicu_observed_data_mixed_mode_lca_best_of_10_v1",
    }
)

# The agent's scientific defaults for the knobs a reviewer may legitimately
# move.  They are defaults, not ceilings: every one of them is validated
# against the declared window and refused rather than clipped.
FIXED_WINDOW_TRAJECTORY_DEFAULTS = MappingProxyType(
    {
        "window_start_hours": 0,
        "window_end_hours": 72,
        "grid_width_hours": 12,
        "minimum_available_windows": 2,
        "candidate_cluster_min": 2,
        "candidate_cluster_max": 6,
        "stability_resamples": 100,
        "stability_sample_fraction": 0.8,
        "minimum_mean_stability": 0.7,
        "minimum_cluster_fraction": 0.05,
    }
)

#: The Planner's primary scientific action for longitudinal trajectory
#: phenotyping.  Cross-sectional phenotype discovery shares the analysis
#: family, so this declaration is what claims classes of trajectories.
TRAJECTORY_PRIMARY_ACTION = "phenotyping.trajectory_feature_clustering"
_MAX_COORDINATE_CONCEPTS = 16
#: The signed representation owner counts a stay's eligible windows on
#: owner-available SOFA-2 evidence, the only coordinates that publish an
#: availability receipt beside each value; a design without a SOFA-2
#: coordinate would exclude every stay instead of measuring it.
ELIGIBILITY_COORDINATE_PREFIX = "sofa2"
_MAX_CANDIDATE_CLUSTERS = 12
_MAX_WINDOWS = 48
#: The anchor a population predicate counts from.  The signed owner counts
#: every trajectory window from ICU admission, so eligibility counts from it
#: too; ``icu_admission`` is the materializer's spelling of the same anchor.
_POPULATION_ANCHORS = frozenset({"icu_admit", "icu_admission"})
_POPULATION_FIELDS = frozenset({"inclusion", "exclusion"})
_ANCHOR_WORDS = {
    "icu_admit": "ICU admission",
    "icu_admission": "ICU admission",
    "hospital_admit": "hospital admission",
    "index_time": "the index time",
}


class TrajectoryDesignError(ValueError):
    """A declared trajectory design is not executable as written."""

    def __init__(self, code: str, message: str, *, field: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class FixedWindowTrajectoryDesign:
    """The reviewed scientific design of one trajectory clustering study."""

    coordinate_concepts: tuple[str, ...]
    descriptive_only_concepts: tuple[str, ...] = ()
    window_start_hours: int = 0
    window_end_hours: int = 72
    grid_width_hours: int = 12
    minimum_available_windows: int = 2
    candidate_cluster_min: int = 2
    candidate_cluster_max: int = 6
    stability_resamples: int = 100
    stability_sample_fraction: float = 0.8
    minimum_mean_stability: float = 0.7
    minimum_cluster_fraction: float = 0.05
    #: The reviewed population, as canonical cohort predicates (the CTAS JSON
    #: of ``planning.cohort_contract.ConceptPredicate``), each settled no
    #: later than the window's end.  Empty keeps every stay of the
    #: host-restricted source universe.
    population_inclusion: tuple[Mapping[str, Any], ...] = ()
    population_exclusion: tuple[Mapping[str, Any], ...] = ()

    @property
    def window_count(self) -> int:
        span = self.window_end_hours - self.window_start_hours
        return span // self.grid_width_hours

    @property
    def candidate_cluster_counts(self) -> tuple[int, ...]:
        return tuple(
            range(self.candidate_cluster_min, self.candidate_cluster_max + 1)
        )

    @property
    def representation_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{concept}__h{start}_{start + self.grid_width_hours}"
            for concept in self.coordinate_concepts
            for start in range(
                self.window_start_hours,
                self.window_end_hours,
                self.grid_width_hours,
            )
        )

    @property
    def required_concepts(self) -> tuple[str, ...]:
        return (*self.coordinate_concepts, *self.descriptive_only_concepts)

    @property
    def population_concepts(self) -> tuple[str, ...]:
        """The concepts the population's predicates read, in declared order.

        They are stay-level columns of the materialized universe, not panel
        coordinates.
        """

        return tuple(
            dict.fromkeys(
                str(predicate["concept_id"])
                for predicate in (*self.population_inclusion, *self.population_exclusion)
            )
        )

    @property
    def population_definition(self) -> dict[str, Any] | None:
        """The plan cohort the signed owners analyze; ``None`` keeps every stay."""

        if not (self.population_inclusion or self.population_exclusion):
            return None
        return {
            "name": "primary",
            "selection_mode": "predicate_filtered",
            "inclusion": [dict(predicate) for predicate in self.population_inclusion],
            "exclusion": [dict(predicate) for predicate in self.population_exclusion],
        }

    def as_study_field(self) -> dict[str, Any]:
        """The normalized StudyContext value this design round-trips through."""

        return {
            "coordinate_concepts": list(self.coordinate_concepts),
            **(
                {"descriptive_only_concepts": list(self.descriptive_only_concepts)}
                if self.descriptive_only_concepts
                else {}
            ),
            "window_start_hours": self.window_start_hours,
            "window_end_hours": self.window_end_hours,
            "grid_width_hours": self.grid_width_hours,
            "minimum_available_windows": self.minimum_available_windows,
            "candidate_cluster_min": self.candidate_cluster_min,
            "candidate_cluster_max": self.candidate_cluster_max,
            "stability_resamples": self.stability_resamples,
            "stability_sample_fraction": self.stability_sample_fraction,
            "minimum_mean_stability": self.minimum_mean_stability,
            "minimum_cluster_fraction": self.minimum_cluster_fraction,
            **(
                {
                    "population": {
                        "inclusion": [dict(item) for item in self.population_inclusion],
                        "exclusion": [dict(item) for item in self.population_exclusion],
                    }
                }
                if self.population_inclusion or self.population_exclusion
                else {}
            ),
        }


def _concepts(value: Any, *, field: str, minimum: int) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a list of concept identifiers.",
            field=field,
        )
    else:
        items = value
    concepts: list[str] = []
    for item in items:
        name = str(item or "").strip().lower()
        if not name or set(name) - _CONCEPT_CHARS or name[0].isdigit():
            raise TrajectoryDesignError(
                "study_trajectory_concept_invalid",
                (
                    f"{field} accepts lower-case concept identifiers only; "
                    "a window column built from another spelling cannot be "
                    "read back by the trajectory contract."
                ),
                field=field,
            )
        if name not in concepts:
            concepts.append(name)
    if len(concepts) < minimum:
        raise TrajectoryDesignError(
            "study_trajectory_concepts_insufficient",
            f"{field} requires at least {minimum} distinct concepts.",
            field=field,
        )
    if len(concepts) > _MAX_COORDINATE_CONCEPTS:
        raise TrajectoryDesignError(
            "study_trajectory_concepts_too_many",
            f"{field} accepts at most {_MAX_COORDINATE_CONCEPTS} concepts.",
            field=field,
        )
    return tuple(concepts)


def _integer(value: Any, *, field: str, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a whole number of hours or a count.",
            field=field,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a whole number.",
            field=field,
        ) from exc
    if number != int(number):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a whole number.",
            field=field,
        )
    return int(number)


def _fraction(value: Any, *, field: str, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a number.",
            field=field,
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            f"{field} must be a number.",
            field=field,
        ) from exc


def _canonical_predicates(value: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    """Each cohort predicate's canonical CTAS JSON, in declared order.

    Whether a concept exists is the materializing run's question (its column
    binding fails closed); the design only fixes what each predicate says.
    """

    from ..planning.cohort_contract import (
        CohortSchemaError,
        ConceptPredicate,
        cohort_concept_id_scope,
    )

    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TrajectoryDesignError(
            "study_trajectory_population_invalid",
            f"{field} must be a list of cohort predicates.",
            field=field,
        )
    canonical: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, ConceptPredicate):
            canonical.append(item.to_dict())
            continue
        if not isinstance(item, Mapping):
            raise TrajectoryDesignError(
                "study_trajectory_population_invalid",
                f"{field} must be a list of cohort predicates.",
                field=field,
            )
        concept = str(item.get("concept_id") or "").strip()
        try:
            with cohort_concept_id_scope([concept] if concept else []):
                predicate = ConceptPredicate.from_dict(
                    {
                        str(key): (dict(child) if isinstance(child, Mapping) else child)
                        for key, child in item.items()
                    }
                )
        except CohortSchemaError as exc:
            raise TrajectoryDesignError(
                "study_trajectory_population_invalid",
                f"{field}: {exc}.",
                field=field,
            ) from exc
        canonical.append(predicate.to_dict())
    return tuple(canonical)


def _population(
    value: Any,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if value is None:
        return (), ()
    if not isinstance(value, Mapping):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            "trajectory_design.population must be an object.",
            field="trajectory_design.population",
        )
    unknown = sorted(set(map(str, value)) - _POPULATION_FIELDS)
    if unknown:
        raise TrajectoryDesignError(
            "study_trajectory_design_unknown_fields",
            f"trajectory_design.population does not accept {unknown}.",
            field="trajectory_design.population",
        )
    return (
        _canonical_predicates(
            value.get("inclusion"), field="trajectory_design.population.inclusion"
        ),
        _canonical_predicates(
            value.get("exclusion"), field="trajectory_design.population.exclusion"
        ),
    )


_DESIGN_FIELDS = frozenset(
    {
        "coordinate_concepts",
        "descriptive_only_concepts",
        "population",
        *FIXED_WINDOW_TRAJECTORY_DEFAULTS,
    }
)


def normalize_trajectory_design(
    value: Any, *, enforce_design_rules: bool = True
) -> dict[str, Any]:
    """Validate a declared trajectory design without inventing one.

    An absent or empty declaration stays empty: this owner never promotes a
    study to a trajectory design, it only refuses one that cannot execute.

    ``enforce_design_rules=False`` keeps the per-field shape but skips the
    cross-field scientific rules.  Reading a stored project must not fail on
    a design that a later rule would reject: a historical contradiction has
    to stay inspectable, and new writes and execution enforce the current
    contract on their own.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TrajectoryDesignError(
            "study_trajectory_design_field_type",
            "trajectory_design must be an object.",
            field="trajectory_design",
        )
    if not value:
        return {}
    unknown = sorted(set(map(str, value)) - _DESIGN_FIELDS)
    if unknown:
        raise TrajectoryDesignError(
            "study_trajectory_design_unknown_fields",
            f"trajectory_design does not accept {unknown}.",
            field="trajectory_design",
        )
    design = FixedWindowTrajectoryDesign(
        coordinate_concepts=_concepts(
            value.get("coordinate_concepts"),
            field="trajectory_design.coordinate_concepts",
            minimum=2 if enforce_design_rules else 1,
        ),
        descriptive_only_concepts=_concepts(
            value.get("descriptive_only_concepts"),
            field="trajectory_design.descriptive_only_concepts",
            minimum=0,
        ),
        **{
            name: (
                _integer(
                    value.get(name),
                    field=f"trajectory_design.{name}",
                    default=default,
                )
                if isinstance(default, int)
                else _fraction(
                    value.get(name),
                    field=f"trajectory_design.{name}",
                    default=default,
                )
            )
            for name, default in FIXED_WINDOW_TRAJECTORY_DEFAULTS.items()
        },
    )
    inclusion, exclusion = _population(value.get("population"))
    design = replace(
        design, population_inclusion=inclusion, population_exclusion=exclusion
    )
    if enforce_design_rules:
        design = _validate(design)
    return design.as_study_field()


def load_trajectory_design(value: Any) -> FixedWindowTrajectoryDesign | None:
    """Read a normalized StudyContext value back as its typed design."""

    normalized = normalize_trajectory_design(value)
    if not normalized:
        return None
    inclusion, exclusion = _population(normalized.get("population"))
    return _validate(
        FixedWindowTrajectoryDesign(
            coordinate_concepts=tuple(normalized["coordinate_concepts"]),
            descriptive_only_concepts=tuple(
                normalized.get("descriptive_only_concepts") or ()
            ),
            **{
                name: normalized[name]
                for name in FIXED_WINDOW_TRAJECTORY_DEFAULTS
            },
            population_inclusion=inclusion,
            population_exclusion=exclusion,
        )
    )


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def proposed_trajectory_coordinates(
    plan: Any, *, excluded: Iterable[str] = ()
) -> tuple[str, ...]:
    """The concepts a trajectory plan's primary step clusters on, in plan order.

    Reads typed plan fields only (a plan object or its JSON): the primary
    step's inputs that name a bare concept, never a product reference such as
    ``artifact:`` or ``table:``, minus the identity and outcome columns the
    caller names.  A plan whose primary is not one step proposes nothing.
    """

    steps = _field(plan, "steps") or ()
    primaries = [
        step for step in steps if _field(step, "planned_analysis_role") == "primary"
    ]
    if len(primaries) != 1:
        return ()
    skip = {str(value) for value in excluded}
    concepts: list[str] = []
    for value in _field(primaries[0], "inputs") or ():
        name = str(value or "").strip()
        if not name or ":" in name or name in skip or name in concepts:
            continue
        concepts.append(name)
    return tuple(concepts)


def executable_trajectory_coordinates(concepts: Sequence[str]) -> bool:
    """Whether a proposed coordinate set can be a signed trajectory design."""

    try:
        coordinates = _concepts(
            list(concepts), field="trajectory_design.coordinate_concepts", minimum=2
        )
    except TrajectoryDesignError:
        return False
    return any(
        concept.startswith(ELIGIBILITY_COORDINATE_PREFIX) for concept in coordinates
    )


def longitudinal_capability_note(offered_components: Iterable[str]) -> str:
    """State the signed owner's rule to an agent that chooses concepts.

    ``offered_components`` are the offered concepts that carry the per-window
    observed/available receipts on which the owner counts eligible windows.
    The note is a host fact: it names the rule and those concepts and leaves
    the question-faithful choice to the agent.  It is empty when nothing
    offered can satisfy the rule.
    """

    offered = sorted(
        {
            name
            for concept in offered_components
            if (name := str(concept or "").strip()).startswith(
                ELIGIBILITY_COORDINATE_PREFIX
            )
        }
    )
    if not offered:
        return ""
    return (
        "LONGITUDINAL CAPABILITY (a host fact, not a recommendation): for a "
        "question about trajectories over ICU time, the host's signed "
        "fixed-window trajectory owner models per-window values of at least two "
        "time-varying concepts. It counts each stay's eligible windows on SOFA-2 "
        "components, which carry per-window observed/available receipts, so its "
        "design needs at least one of: "
        + ", ".join(offered)
        + ". Outcomes and one-value-per-stay concepts are never trajectory "
        "coordinates. If such a question names a score without its version and "
        "this catalog offers several versions, each is a faithful reading: "
        "select one version and say which, and why, in the rationale. Keep a "
        "version the question names, even when the owner cannot model it."
    )


#: The same rule as the Planner reads it before it chooses a primary step.
#: Where a trajectory question's outcomes are described, and so what a
#: model-coded trajectory primary leaves out of its inputs.  The Planner reads
#: it in its outline contract and when an outline adds an outcome comparison
#: after that primary (``planning.phenotype_outline_rules``).
TRAJECTORY_OUTCOME_DESCRIPTION_RULE = (
    "When a phenotyping.trajectory_feature_clustering primary's coordinates "
    "meet the signed owner's rule (at least two time-varying study variables, "
    "one of them a SOFA-2 component, a name starting with "
    f"{ELIGIBILITY_COORDINATE_PREFIX!r}), the host replaces it with the signed "
    "fixed-window suite, which describes the requested outcomes on its frozen "
    "classes: keep outcomes and one-value-per-stay characteristics out of that "
    "primary's inputs. Only a primary whose coordinates cannot meet the rule "
    "describes requested outcomes in its own characterization products."
)

TRAJECTORY_OWNER_PLANNER_RULE = (
    "The host executes a phenotyping.trajectory_feature_clustering primary with "
    "its signed fixed-window owner only when the primary's inputs are at least "
    "two time-varying study variables, one of them a SOFA-2 component (a name "
    f"starting with {ELIGIBILITY_COORDINATE_PREFIX!r}), and no outcome or "
    "one-value-per-stay variable; otherwise its classes summarize per-stay "
    "values. When the question names a score without its version, the "
    "primary's intent says which version its coordinates use. "
    + TRAJECTORY_OUTCOME_DESCRIPTION_RULE
)


#: Variable roles whose values change within a stay and can be read in fixed
#: windows.  Identity, demographics, outcomes and other one-per-stay values
#: cannot be trajectory coordinates.
_TIME_VARYING_ROLES = frozenset(
    {"vital", "lab", "intervention", "ordinal_score", "composite_score"}
)


def time_varying_role(role: Any) -> bool:
    """Whether a variable role (enum or its value) changes within a stay."""

    return str(getattr(role, "value", role) or "").strip().lower() in _TIME_VARYING_ROLES


def eligibility_coordinates(variables: Iterable[Any]) -> tuple[str, ...]:
    """A study's time-varying SOFA-2 variables, the owner's eligibility basis."""

    return tuple(
        sorted(
            name
            for variable in variables
            if (name := str(_field(variable, "name") or "")).startswith(
                ELIGIBILITY_COORDINATE_PREFIX
            )
            and time_varying_role(_field(variable, "role"))
        )
    )


def trajectory_coordinate_proposal(
    plan: Any,
    *,
    variables: Iterable[Any],
    outcomes: Iterable[str],
    excluded: Iterable[str] = (),
) -> dict[str, Any]:
    """What a trajectory plan's primary step clusters, under the owner's rule.

    The coordinates are the primary step's inputs that are study variables,
    minus ``excluded`` identity columns.  They are executable when there are
    at least two, one is a SOFA-2 component, and the step clusters no outcome
    and no one-per-stay variable; otherwise the plan's own set is not a
    design, and the remaining coordinates are never compiled on their own.
    """

    catalogue = list(variables)
    roles = {str(_field(variable, "name")): _field(variable, "role") for variable in catalogue}
    outcome_names = {str(value) for value in outcomes}
    proposed = [
        name for name in proposed_trajectory_coordinates(plan, excluded=excluded)
        if name in roles
    ]
    outcome_inputs = [name for name in proposed if name in outcome_names]
    one_per_stay_inputs = [
        name
        for name in proposed
        if name not in outcome_names and not time_varying_role(roles[name])
    ]
    coordinates = [
        name
        for name in proposed
        if name not in outcome_inputs and name not in one_per_stay_inputs
    ]
    return {
        "proposed_coordinates": coordinates,
        "outcome_inputs": outcome_inputs,
        "one_per_stay_inputs": one_per_stay_inputs,
        "eligibility_coordinate_prefix": ELIGIBILITY_COORDINATE_PREFIX,
        "study_eligibility_coordinates": list(eligibility_coordinates(catalogue)),
        # The time-varying coordinates alone would be a design; they are never
        # compiled while the step also clusters on anything else.
        "coordinates_executable": executable_trajectory_coordinates(coordinates),
        "executable": bool(
            not outcome_inputs
            and not one_per_stay_inputs
            and executable_trajectory_coordinates(coordinates)
        ),
    }


#: The grid widths, in hours, a design takes for a window its question states.
_CONVENTIONAL_GRID_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 24, 48)
_WINDOW_FIELDS = ("window_start_hours", "window_end_hours", "grid_width_hours")


def trajectory_window_design(statements: Iterable[Any]) -> dict[str, Any]:
    """The fixed window and grid a design takes from what its question states.

    ``statements`` are the windows the question states for its trajectories
    (``hours``, ``anchor`` and ``text`` fields; see
    ``research_context.temporal_semantics.trajectory_window_statements``).
    None keeps the default window.  One window from ICU admission becomes the
    design's window, on the conventional grid width that divides it into the
    number of windows nearest the default design's; a tie keeps the width
    nearest the default's.  The signed owner counts every window from ICU
    admission, so a window stated from another event, several windows, or one
    no grid divides are not executable, and say why instead of falling back
    to the default.
    """

    stated = [
        {
            "hours": float(_field(item, "hours")),
            "anchor": str(_field(item, "anchor")),
            "text": str(_field(item, "text") or ""),
        }
        for item in statements
    ]
    defaults = FIXED_WINDOW_TRAJECTORY_DEFAULTS
    if not stated:
        return {
            "source": "design_default",
            "stated": [],
            **{name: defaults[name] for name in _WINDOW_FIELDS},
            "executable": True,
            "reason": None,
        }

    def refused(reason: str) -> dict[str, Any]:
        return {"source": "question", "stated": stated, "executable": False, "reason": reason}

    other_anchors = sorted({item["anchor"] for item in stated} - {"icu_admission"})
    if other_anchors:
        return refused(
            "the question counts its trajectory window from "
            + ", ".join(other_anchors)
            + ", and the signed fixed-window owner counts every window from ICU admission"
        )
    stated_hours = sorted({item["hours"] for item in stated})
    if len(stated_hours) > 1:
        return refused(
            "the question states several trajectory windows ("
            + ", ".join(f"{value:g} h" for value in stated_hours)
            + ")"
        )
    end = stated_hours[0]
    if end != int(end):
        return refused(f"the stated {end:g}-hour trajectory window is not a whole number of hours")
    end = int(end)
    minimum = int(defaults["minimum_available_windows"])
    widths = [
        width
        for width in _CONVENTIONAL_GRID_WIDTHS
        if end % width == 0 and minimum <= end // width <= _MAX_WINDOWS
    ]
    if not widths:
        return refused(
            f"no grid of {', '.join(map(str, _CONVENTIONAL_GRID_WIDTHS))} h divides the "
            f"stated {end}-hour trajectory window into {minimum} to {_MAX_WINDOWS} windows"
        )
    default_count = defaults["window_end_hours"] // defaults["grid_width_hours"]
    grid = min(
        widths,
        key=lambda width: (
            abs(end // width - default_count),
            abs(width - defaults["grid_width_hours"]),
        ),
    )
    return {
        "source": "question",
        "stated": stated,
        "window_start_hours": 0,
        "window_end_hours": end,
        "grid_width_hours": grid,
        "executable": True,
        "reason": None,
    }


def _offset(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _predicate_label(predicate: Mapping[str, Any]) -> str:
    """One predicate as a reviewer reads it: what it tests, over which hours."""

    window = predicate.get("time_window") or {}
    op = str(predicate.get("op") or "")
    test = f"{predicate.get('concept_id')} {op}"
    if op not in {"missing", "not_missing"}:
        test += f" {predicate.get('value')!r}"
    start, end = _offset(window.get("start_offset_hours")), _offset(
        window.get("end_offset_hours")
    )
    anchor = str(window.get("anchor") or "")
    return (
        f"{test} ({predicate.get('aggregation')} over {start:g}–{end:g} h "
        f"from {_ANCHOR_WORDS.get(anchor.casefold(), anchor)})"
    )


def trajectory_population_issues(
    inclusion: Iterable[Mapping[str, Any]],
    exclusion: Iterable[Mapping[str, Any]],
    *,
    window_end_hours: float,
) -> list[str]:
    """Why the signed owner cannot apply a stated population; empty when it can.

    The owner counts every trajectory window from ICU admission, so a
    predicate counts from it too, and is settled by the end of the trajectory
    window: membership decided by what happens after the hours the classes
    describe would choose stays by their future.
    """

    issues: list[str] = []
    for kind, predicates in (("inclusion", inclusion), ("exclusion", exclusion)):
        for predicate in predicates:
            window = predicate.get("time_window") or {}
            anchor = str(window.get("anchor") or "")
            end = _offset(window.get("end_offset_hours"))
            label = _predicate_label(predicate)
            if anchor.casefold() not in _POPULATION_ANCHORS:
                issues.append(
                    f"the {kind} predicate {label} is not counted from ICU "
                    "admission, and the signed fixed-window owner counts "
                    "eligibility from ICU admission"
                )
            elif not math.isfinite(end) or end > window_end_hours:
                issues.append(
                    f"the {kind} predicate {label} is settled after the "
                    f"{window_end_hours:g}-hour trajectory window ends"
                )
    return issues


def trajectory_population_design(
    cohort: Any,
    window: Mapping[str, Any],
    *,
    static_concepts: Iterable[str] = (),
) -> dict[str, Any]:
    """The population a design takes from the cohort a trajectory plan states.

    ``cohort`` is the plan's typed cohort (``AnalysisPlan.cohort`` or its
    JSON) and ``window`` the window design (:func:`trajectory_window_design`).
    A plan that selects every input row states no population, and the owner
    keeps every stay.  Otherwise its predicates are the population,
    unchanged: the design never edits or drops one.  They are executable
    when each counts from ICU admission and is settled by the end of the
    trajectory window; otherwise the facts say why.

    ``within_trajectory_window`` names the predicates on time-varying
    concepts (every concept not in ``static_concepts``) that are evaluated
    inside the trajectory window.  Membership then depends on the hours the
    classes describe, which a plan states rather than hides.
    """

    selection_mode = str(_field(cohort, "selection_mode") or "predicate_filtered")
    inclusion = _canonical_predicates(
        _field(cohort, "inclusion") if cohort is not None else None,
        field="cohort.inclusion",
    )
    exclusion = _canonical_predicates(
        _field(cohort, "exclusion") if cohort is not None else None,
        field="cohort.exclusion",
    )
    if selection_mode == "all_input_rows" or not (inclusion or exclusion):
        return {
            "source": "none",
            "inclusion": [],
            "exclusion": [],
            "concepts": [],
            "within_trajectory_window": [],
            "executable": True,
            "reason": None,
        }
    window_end = window.get(
        "window_end_hours", FIXED_WINDOW_TRAJECTORY_DEFAULTS["window_end_hours"]
    )
    window_start = window.get(
        "window_start_hours", FIXED_WINDOW_TRAJECTORY_DEFAULTS["window_start_hours"]
    )
    issues = trajectory_population_issues(
        inclusion, exclusion, window_end_hours=float(window_end)
    )
    static = {str(value) for value in static_concepts}
    within = [
        _predicate_label(predicate)
        for predicate in (*inclusion, *exclusion)
        if str(predicate["concept_id"]) not in static
        and _offset((predicate.get("time_window") or {}).get("end_offset_hours"))
        > float(window_start)
    ]
    return {
        "source": "plan",
        "inclusion": [dict(item) for item in inclusion],
        "exclusion": [dict(item) for item in exclusion],
        "concepts": list(
            dict.fromkeys(str(item["concept_id"]) for item in (*inclusion, *exclusion))
        ),
        "within_trajectory_window": within,
        "executable": not issues,
        "reason": "; ".join(issues) or None,
    }


def _fail(code: str, message: str, *, field: str) -> None:
    raise TrajectoryDesignError(code, message, field=f"trajectory_design.{field}")


def _validate(design: FixedWindowTrajectoryDesign) -> FixedWindowTrajectoryDesign:
    overlap = sorted(
        set(design.coordinate_concepts) & set(design.descriptive_only_concepts)
    )
    if overlap:
        _fail(
            "study_trajectory_descriptive_concept_is_a_coordinate",
            (
                "A descriptive-only concept cannot also be a model coordinate; "
                f"remove {overlap} from one of the two lists."
            ),
            field="descriptive_only_concepts",
        )
    if design.window_start_hours < 0:
        _fail(
            "study_trajectory_window_invalid",
            "The trajectory window starts at or after the cohort time zero.",
            field="window_start_hours",
        )
    if design.window_end_hours <= design.window_start_hours:
        _fail(
            "study_trajectory_window_invalid",
            "The trajectory window must have a positive width.",
            field="window_end_hours",
        )
    if design.grid_width_hours <= 0:
        _fail(
            "study_trajectory_grid_invalid",
            "The fixed-window grid width must be positive.",
            field="grid_width_hours",
        )
    span = design.window_end_hours - design.window_start_hours
    if span % design.grid_width_hours:
        _fail(
            "study_trajectory_grid_invalid",
            (
                f"A {span}-hour window does not divide into complete "
                f"{design.grid_width_hours}-hour windows; a partial trailing "
                "window is a different measurement, not a shorter one."
            ),
            field="grid_width_hours",
        )
    if design.window_count > _MAX_WINDOWS:
        _fail(
            "study_trajectory_grid_invalid",
            f"The declared grid exceeds {_MAX_WINDOWS} windows.",
            field="grid_width_hours",
        )
    population_issues = trajectory_population_issues(
        design.population_inclusion,
        design.population_exclusion,
        window_end_hours=design.window_end_hours,
    )
    if population_issues:
        _fail(
            "study_trajectory_population_window_invalid",
            (
                "The signed trajectory owner cannot apply this population: "
                + "; ".join(population_issues)
                + "."
            ),
            field="population",
        )
    if not 1 <= design.minimum_available_windows <= design.window_count:
        _fail(
            "study_trajectory_minimum_windows_invalid",
            (
                "Eligibility requires between one window and the "
                f"{design.window_count} windows the grid actually has."
            ),
            field="minimum_available_windows",
        )
    if design.candidate_cluster_min < 2:
        _fail(
            "study_trajectory_candidate_grid_invalid",
            "A cluster search starts at two clusters.",
            field="candidate_cluster_min",
        )
    if design.candidate_cluster_max <= design.candidate_cluster_min:
        _fail(
            "study_trajectory_candidate_grid_invalid",
            (
                "The candidate grid needs at least two cluster counts, or the "
                "interior-optimum rule has nothing to decide."
            ),
            field="candidate_cluster_max",
        )
    if design.candidate_cluster_max > _MAX_CANDIDATE_CLUSTERS:
        _fail(
            "study_trajectory_candidate_grid_invalid",
            f"The candidate grid stops at {_MAX_CANDIDATE_CLUSTERS} clusters.",
            field="candidate_cluster_max",
        )
    if not 2 <= design.stability_resamples <= 500:
        _fail(
            "study_trajectory_stability_invalid",
            "Stability uses between 2 and 500 resamples.",
            field="stability_resamples",
        )
    if not 0.0 < design.stability_sample_fraction < 1.0:
        _fail(
            "study_trajectory_stability_invalid",
            "The stability subsample fraction is strictly between 0 and 1.",
            field="stability_sample_fraction",
        )
    if not -1.0 <= design.minimum_mean_stability <= 1.0:
        _fail(
            "study_trajectory_stability_invalid",
            "The mean adjusted-Rand threshold lies in [-1, 1].",
            field="minimum_mean_stability",
        )
    if not 0.0 < design.minimum_cluster_fraction < 1.0:
        _fail(
            "study_trajectory_minimum_cluster_fraction_invalid",
            "The minimum cluster fraction is strictly between 0 and 1.",
            field="minimum_cluster_fraction",
        )
    if design.minimum_cluster_fraction * design.candidate_cluster_max >= 1.0:
        _fail(
            "study_trajectory_minimum_cluster_fraction_invalid",
            (
                "No partition of the cohort can give every one of "
                f"{design.candidate_cluster_max} clusters at least "
                f"{design.minimum_cluster_fraction:.0%} of the rows."
            ),
            field="minimum_cluster_fraction",
        )
    return replace(design)


def trajectory_coordinate_measurement(
    concepts: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Each coordinate's measurement, from the host's declared concept scales.

    A concept the host declares ordinal with explicit levels (a SOFA-2 organ
    score: 0-4) is ordinal; every other coordinate is continuous.
    """

    from ..icu_rules import declared_ordinal_levels

    measurement: list[dict[str, Any]] = []
    for concept in concepts:
        levels = declared_ordinal_levels(concept)
        measurement.append(
            {"concept": concept, "scale": "ordinal", "levels": list(levels)}
            if levels is not None
            else {"concept": concept, "scale": "continuous"}
        )
    return tuple(measurement)


def sealed_trajectory_authority_body(
    design: FixedWindowTrajectoryDesign,
    *,
    protocol_content_sha256: str,
) -> dict[str, Any]:
    """Compile one reviewed design plus host policy into the authority body.

    The caller seals the result; this function neither hashes nor signs it, so
    there is exactly one place (``build_trajectory_scientific_runtime_authority``)
    that can mint an execution contract digest.
    """

    from ..schema import TrajectoryStabilitySpec

    measurement = trajectory_coordinate_measurement(design.coordinate_concepts)
    mixed_mode = any(entry["scale"] == "ordinal" for entry in measurement)
    policy = {
        **TRAJECTORY_HOST_POLICY,
        **(TRAJECTORY_MIXED_MODE_POLICY if mixed_mode else {}),
    }
    stability = TrajectoryStabilitySpec(
        n_resamples=design.stability_resamples,
        sample_fraction=design.stability_sample_fraction,
        base_seed=int(policy["candidate_fit_base_seed"]),
        minimum_successful_resamples=design.stability_resamples,
        refit_engine=policy["fit_engine"],
        refit_max_iter=int(policy["candidate_fit_max_iter"]),
        refit_tolerance=float(policy["candidate_fit_tolerance"]),
        refit_regularization=float(policy["candidate_fit_regularization"]),
        minimum_mean_stability=design.minimum_mean_stability,
        decision_mode="minimum_mean_threshold",
    )
    return {
        "schema_version": (
            "easyicu.trajectory_scientific_runtime_authority/2"
            if mixed_mode
            else "easyicu.trajectory_scientific_runtime_authority/1"
        ),
        "protocol_content_sha256": protocol_content_sha256,
        "coordinate_concepts": list(design.coordinate_concepts),
        "descriptive_only_concepts": list(design.descriptive_only_concepts),
        "window_start_hours": design.window_start_hours,
        "window_end_hours": design.window_end_hours,
        "grid_width_hours": design.grid_width_hours,
        "aggregation": policy["aggregation"],
        "representation_columns": list(design.representation_columns),
        "minimum_available_windows": design.minimum_available_windows,
        "coordinate_scaling": {
            "method": policy["scaling_method"],
            "ddof": policy["scaling_ddof"],
            "observed_value_policy": "direct_or_owner_locf_available",
            "missing_value_policy": "preserve_missing_exclude_from_likelihood",
            "zero_variance_action": policy["scaling_zero_variance_action"],
        },
        **({"coordinate_measurement": list(measurement)} if mixed_mode else {}),
        "evidence_state_policy": {
            "direct_observed": "include",
            "owner_locf_available": "include_and_audit",
            "unavailable": "exclude",
            "additional_clustering_stage_imputation": "none",
        },
        "representation_plan_method": (
            "signed_fixed_window_trajectory_representation"
        ),
        "representation_plan_intent": (
            "Build the digest-bound fixed-window trajectory representation "
            "exactly as declared by the scientific runtime authority."
        ),
        "representation_plan_inputs": [],
        "representation_required_outputs": [
            "artifact:trajectory_representation",
            "table:trajectory_membership",
            "manifest:trajectory_representation_schema",
        ],
        "model_family": policy["model_family"],
        "fit_method": policy["fit_method"],
        "covariance_type": policy["covariance_type"],
        "candidate_cluster_counts": list(design.candidate_cluster_counts),
        "selection_criterion": policy["selection_criterion"],
        "selection_rule": policy["selection_rule"],
        "candidate_fit_base_seed": policy["candidate_fit_base_seed"],
        "candidate_fit_max_iter": policy["candidate_fit_max_iter"],
        "candidate_fit_tolerance": policy["candidate_fit_tolerance"],
        "candidate_fit_regularization": policy["candidate_fit_regularization"],
        "bic_sample_size": policy["bic_sample_size"],
        "bic_parameter_count": policy["bic_parameter_count"],
        "bic_tie_break": policy["bic_tie_break"],
        "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
        "upper_boundary_reason_code": policy["upper_boundary_reason_code"],
        "minimum_cluster_fraction": design.minimum_cluster_fraction,
        "minimum_cluster_fraction_reason_code": (
            policy["minimum_cluster_fraction_reason_code"]
        ),
        "stability_spec": stability.model_dump(mode="json"),
        # Stated only for a design with a population, so a design without one
        # seals exactly the body (and digest) it always did.
        **(
            {
                "population": {
                    "inclusion": [dict(item) for item in design.population_inclusion],
                    "exclusion": [dict(item) for item in design.population_exclusion],
                }
            }
            if design.population_definition is not None
            else {}
        ),
    }
