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
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator, model_validator

from ..canonical_json import canonical_sha256
from .cohort_product_keys import sole_typed_cohort_input

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
    "recorded_measurement_concept",
    "sealed_trajectory_authority_body",
    "time_varying_role",
    "trajectory_coordinate_measurement",
    "trajectory_coordinate_proposal",
    "trajectory_eligibility_basis",
    "trajectory_population_design",
    "trajectory_population_issues",
    "trajectory_window_design",
    "unrecorded_trajectory_coordinates",
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
#: owner-available SOFA-2 evidence when the design has a SOFA-2 coordinate:
#: those publish an availability receipt beside each value, which tells a
#: carried-forward score from an absent one.  A design without one counts its
#: windows on its coordinates' recorded values, which only a concept read
#: directly from source rows has (``recorded_measurement_concept``).
ELIGIBILITY_COORDINATE_PREFIX = "sofa2"
_MAX_CANDIDATE_CLUSTERS = 12
_MAX_WINDOWS = 48
#: The anchor a population predicate counts from.  The signed owner counts
#: every trajectory window from ICU admission, so eligibility counts from it
#: too; ``icu_admission`` is the materializer's spelling of the same anchor.
_POPULATION_ANCHORS = frozenset({"icu_admit", "icu_admission"})
_POPULATION_FIELDS = frozenset({"inclusion", "exclusion", "unapplied_criteria"})
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
    #: Population criteria the reviewed plan states and does not apply: no
    #: allowed cohort concept expresses them, or the input does not record
    #: them over the window they state.  They select no stay; the
    #: plan on the signed owners states them so its review reports them.
    population_unapplied_criteria: tuple[str, ...] = ()

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
                        **(
                            {"unapplied_criteria": list(self.population_unapplied_criteria)}
                            if self.population_unapplied_criteria
                            else {}
                        ),
                    }
                }
                if self.population_inclusion
                or self.population_exclusion
                or self.population_unapplied_criteria
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
            f"{field} requires at least {minimum} distinct "
            + ("concept." if minimum == 1 else "concepts."),
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


def _criteria(value: Any, *, field: str) -> tuple[str, ...]:
    """Unapplied population criteria: each a non-empty string."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TrajectoryDesignError(
            "study_trajectory_population_invalid",
            f"{field} must be a list of population criteria.",
            field=field,
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TrajectoryDesignError(
            "study_trajectory_population_invalid",
            f"{field} must name each criterion as non-empty text.",
            field=field,
        )
    return tuple(dict.fromkeys(" ".join(item.split()) for item in value))


def _population(
    value: Any,
) -> tuple[
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[str, ...]
]:
    if value is None:
        return (), (), ()
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
        _criteria(
            value.get("unapplied_criteria"),
            field="trajectory_design.population.unapplied_criteria",
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
            minimum=1,
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
    inclusion, exclusion, criteria = _population(value.get("population"))
    design = replace(
        design,
        population_inclusion=inclusion,
        population_exclusion=exclusion,
        population_unapplied_criteria=criteria,
    )
    if enforce_design_rules:
        design = _validate(design)
    return design.as_study_field()


def load_trajectory_design(value: Any) -> FixedWindowTrajectoryDesign | None:
    """Read a normalized StudyContext value back as its typed design."""

    normalized = normalize_trajectory_design(value)
    if not normalized:
        return None
    inclusion, exclusion, criteria = _population(normalized.get("population"))
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
            population_unapplied_criteria=criteria,
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


@lru_cache(maxsize=1)
def _concept_dictionary() -> Any:
    from ...resources import load_dictionary

    return load_dictionary()


#: Source-item fields that make a row stand for an interval (an infusion, a
#: ventilation episode) rather than one charted value.
_INTERVAL_SOURCE_FIELDS = ("stop_var", "end_var", "win_type")


def recorded_measurement_concept(concept: str) -> bool:
    """Whether the concept dictionary reads ``concept`` as charted values.

    Such a concept is a time series read from source rows: no concept-level
    callback, components or dependencies, and no source item spans an
    interval.  Each row the materializer keeps is then one recorded value
    (``evidence_state`` ``direct_observed``).  A derived concept -- a
    composite score, an indicator -- or an interval expanded over time can
    hold values between recordings, and without a per-row receipt a window
    holding one is no evidence of a measurement.  A concept the dictionary
    does not define is not one.
    """

    definition = _concept_dictionary().get(str(concept or "").strip())
    if definition is None or not getattr(definition, "sources", None):
        return False
    if (
        getattr(definition, "callback", None)
        or getattr(definition, "sub_concepts", None)
        or getattr(definition, "depends_on", None)
        or getattr(definition, "target", None) not in (None, "ts_tbl")
    ):
        return False
    return not any(
        getattr(item, "dur_var", None)
        or getattr(item, "target", None) not in (None, "ts_tbl")
        or any((getattr(item, "params", None) or {}).get(key) for key in _INTERVAL_SOURCE_FIELDS)
        for items in definition.sources.values()
        for item in items
    )


def trajectory_eligibility_basis(coordinates: Sequence[str]) -> tuple[str, ...]:
    """The coordinates on which the signed owner counts a stay's eligible windows.

    Its SOFA-2 coordinates when the design has one, as it always did;
    otherwise every coordinate, each of which must then be a recorded
    measurement (:func:`unrecorded_trajectory_coordinates`).
    """

    sofa2 = tuple(
        concept
        for concept in coordinates
        if str(concept).startswith(ELIGIBILITY_COORDINATE_PREFIX)
    )
    return sofa2 or tuple(coordinates)


def unrecorded_trajectory_coordinates(coordinates: Sequence[str]) -> tuple[str, ...]:
    """The coordinates a design without a SOFA-2 component cannot count windows on."""

    if any(
        str(concept).startswith(ELIGIBILITY_COORDINATE_PREFIX)
        for concept in coordinates
    ):
        return ()
    return tuple(
        concept for concept in coordinates if not recorded_measurement_concept(concept)
    )


def executable_trajectory_coordinates(concepts: Sequence[str]) -> bool:
    """Whether a proposed coordinate set can be a signed trajectory design.

    One or more valid concepts whose eligibility basis carries evidence of
    observation: a SOFA-2 component among them, or else every one a recorded
    measurement.
    """

    try:
        coordinates = _concepts(
            list(concepts), field="trajectory_design.coordinate_concepts", minimum=1
        )
    except TrajectoryDesignError:
        return False
    return not unrecorded_trajectory_coordinates(coordinates)


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
        "fixed-window trajectory owner models per-window values of time-varying "
        "concepts. It counts each stay's eligible windows on SOFA-2 components, "
        "which carry per-window observed/available receipts, when its design has "
        "at least one of: "
        + ", ".join(offered)
        + ". A design without one counts them on measurements read directly from "
        "source rows, never on another derived score. Outcomes and "
        "one-value-per-stay concepts are never trajectory "
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
    "meet the signed owner's rule (time-varying study variables that include a "
    f"SOFA-2 component, a name starting with {ELIGIBILITY_COORDINATE_PREFIX!r}, "
    "or are all directly measured), the host replaces it with the signed "
    "fixed-window suite, which describes the requested outcomes on its frozen "
    "classes: keep outcomes and one-value-per-stay characteristics out of that "
    "primary's inputs. Only a primary whose coordinates cannot meet the rule "
    "describes requested outcomes in its own characterization products."
)

TRAJECTORY_OWNER_PLANNER_RULE = (
    "The host executes a phenotyping.trajectory_feature_clustering primary with "
    "its signed fixed-window owner only when the primary's inputs are one or "
    "more time-varying study variables, including a SOFA-2 component (a name "
    f"starting with {ELIGIBILITY_COORDINATE_PREFIX!r}) or all directly measured "
    "(a vital sign or laboratory value, not a derived score), and no outcome or "
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
    minus ``excluded`` identity columns.  They are executable when there is at
    least one, they include a SOFA-2 component or are all recorded
    measurements, and the step clusters no outcome and no one-per-stay
    variable; otherwise the plan's own set is not a design, and the remaining
    coordinates are never compiled on their own.
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
        # Derived coordinates a design without a SOFA-2 component cannot
        # count eligible windows on.
        "unrecorded_coordinates": list(unrecorded_trajectory_coordinates(coordinates)),
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

    ``unapplied_criteria`` names the criteria the plan states and no
    predicate applies: no allowed cohort concept expresses them, or the
    input does not record them over the window they state.  They select no
    stay; the design carries them so the plan on the signed owners states
    them too.
    """

    selection_mode = str(_field(cohort, "selection_mode") or "predicate_filtered")
    # A criterion no predicate applies selects no stay; the design
    # carries it so the plan on the signed owners still states it.
    stated = (
        _field(cohort, "unapplied_population_criteria") if cohort is not None else None
    )
    unapplied = [" ".join(str(item).split()) for item in stated or () if str(item).strip()]
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
            "unapplied_criteria": unapplied,
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
        "unapplied_criteria": unapplied,
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
        **(
            {"unapplied_population_criteria": list(design.population_unapplied_criteria)}
            if design.population_unapplied_criteria
            else {}
        ),
    }


# Repeated-outcome models retain a distinct design from latent trajectory classes.
REPEATED_CAPABILITY = "repeated_measures_v1"
REPEATED_ACTIONS = ("association.mixed_effects", "association.gee", "phenotyping.mixed_effects_growth")
REPEATED_OUTPUTS = (
    "table:repeated_measure_effects", "table:repeated_measure_diagnostics", "table:repeated_measure_profile",
)
DEATH_LIMITATION = (
    "Death truncates repeated measurements and may cause informative dropout; "
    "these observed-data estimates do not identify a trajectory after death."
)


class RepeatedMeasureSpec(BaseModel):
    """Time is a numeric offset from ICU admission, never an inferred row index."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    identity_column: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    outcome_column: str = Field(min_length=1)
    time_origin: Literal["icu_admission"]
    time_unit: Literal["hours", "days"]
    window_start: float = Field(ge=0)
    window_end: float = Field(gt=0)
    bin_width: float = Field(gt=0)
    aggregation: Literal["mean", "median", "min", "max", "sum", "first", "last"]
    model: Literal["mixed_intercept", "mixed_slope", "gee"]
    family: Literal["gaussian", "binomial"]
    time_basis: Literal["linear", "natural_cubic_spline"]
    spline_knots: tuple[float, ...] = ()
    working_correlation: Literal["exchangeable", "ar1", "independence"] | None = None
    covariates: tuple[str, ...] = ()
    dropout_limitation: Literal["death_may_cause_informative_dropout"]
    dropout_sensitivity: Literal["survivors", "joint_model_note"]
    death_column: str | None = None
    ci_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def _closed_design(self):
        columns = [self.identity_column, self.time_column, self.outcome_column, *self.covariates]
        if self.death_column is not None:
            columns.append(self.death_column)
        if len(columns) != len(set(columns)) or any(not name.strip() or name != name.strip() for name in columns):
            raise ValueError("repeated-measure columns must be distinct exact names")
        bins = (self.window_end - self.window_start) / self.bin_width
        if bins < 2 or bins > 1000 or not math.isclose(bins, round(bins), abs_tol=1e-8):
            raise ValueError("window must contain 2 to 1000 complete declared bins")
        if self.model != "gee" and (self.family != "gaussian" or self.working_correlation is not None):
            raise ValueError("mixed models require gaussian outcomes and no working correlation")
        if self.model == "gee" and self.working_correlation is None:
            raise ValueError("GEE requires a planned working_correlation")
        if self.family == "binomial" and self.aggregation not in {"first", "last", "min", "max"}:
            raise ValueError("binary repeated outcomes require a binary-preserving aggregation")
        if self.time_basis == "linear" and self.spline_knots:
            raise ValueError("linear time cannot declare spline knots")
        if self.time_basis == "natural_cubic_spline" and (
            not self.spline_knots or tuple(sorted(set(self.spline_knots))) != self.spline_knots
            or not all(self.window_start < knot < self.window_end for knot in self.spline_knots)
        ):
            raise ValueError("spline time needs ordered distinct interior knots in the declared unit")
        if self.dropout_sensitivity == "survivors" and not self.death_column:
            raise ValueError("survivor sensitivity needs an explicit binary death_column")
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.identity_column, self.time_column, self.outcome_column, *self.covariates,
                *((self.death_column,) if self.death_column else ()))

    @property
    def action_id(self) -> str:
        return "association.gee" if self.model == "gee" else "association.mixed_effects"


class RepeatedMeasurePlanFields(BaseModel):
    """Bounded progressive wire field; the scientific owner validates its JSON."""

    repeated_measure_spec: Annotated[
        RepeatedMeasureSpec | None, WithJsonSchema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
    ] = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("repeated_measure_spec", mode="before")
    @classmethod
    def _decode_repeated_measure_spec(cls, value):
        return RepeatedMeasureSpec.model_validate_json(value) if isinstance(value, str) else value


def repeated_measure_method(spec: RepeatedMeasureSpec) -> str:
    parsed = RepeatedMeasureSpec.model_validate(spec.model_dump())
    return "repeated_measures(" + json.dumps(parsed.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + ")"


def repeated_measure_spec_for_step(step: object) -> RepeatedMeasureSpec | None:
    method = str(getattr(step, "method", "") or "")
    if not method.startswith("repeated_measures("):
        return None
    if not method.endswith(")"):
        raise ValueError("unclosed repeated-measure declaration")
    spec = RepeatedMeasureSpec.model_validate_json(method[len("repeated_measures("):-1])
    action = getattr(step, "scientific_action_id", None)
    if action != spec.action_id and not (action == "phenotyping.mixed_effects_growth" and spec.model == "mixed_slope"):
        raise ValueError("repeated-measure action and model differ")
    return spec


def repeated_measure_step_matches(step: object) -> bool:
    try:
        spec = repeated_measure_spec_for_step(step)
    except ValueError:
        return False
    return bool(spec is not None and set(spec.required_columns) <= set(getattr(step, "inputs", ()))
                and getattr(step, "scientific_capability", None) in (None, REPEATED_CAPABILITY)
                and tuple(getattr(step, "expected_outputs", ())) == REPEATED_OUTPUTS
                and bool(sole_typed_cohort_input(step))
                and getattr(step, "planned_analysis_role", None) in {"primary", "secondary"}
                and not any(getattr(step, field, None) for field in (
                    "model_requirements", "table_one_spec", "trajectory_stability_spec", "cohort_definition_spec",
                    "measurement_audit_spec", "phenotype_comparison_spec", "robustness_replay_spec",
                )))


def repeated_measure_result_receipt_valid(summary: object) -> bool:
    """Reject missing, non-converged or specification-drifted core estimates."""
    if not isinstance(summary, Mapping) or summary.get("status") != "ok" or summary.get("claim_ceiling") != "analysis_only":
        return False
    receipt = summary.get("repeated_measure_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != "easyicu.repeated_measure_result/1":
        return False
    try:
        spec = RepeatedMeasureSpec.model_validate(receipt.get("specification"))
        if receipt.get("specification_sha256") != canonical_sha256(spec.model_dump(mode="json")):
            return False
        fits = receipt["fits"]
        populations = ["all_observed", "survivors"] if spec.dropout_sensitivity == "survivors" else ["all_observed"]
        if [fit["population"] for fit in fits] != populations:
            return False
        for fit in fits:
            result = fit["statistics"]
            count = len(result["var_names"])
            if result["converged"] is not True or count < 2 or result["n_obs"] <= count:
                return False
            for key in ("coef", "se", "ci_low", "ci_high"):
                if len(result[key]) != count or not all(math.isfinite(v) for v in result[key]):
                    return False
            if any(se <= 0 or low > estimate or high < estimate for se, low, high, estimate in
                   zip(result["se"], result["ci_low"], result["ci_high"], result["coef"], strict=True)):
                return False
            if spec.model == "gee":
                if result["cov_type"] != "robust" or result["family"] != spec.family or result["cov_struct"] != spec.working_correlation:
                    return False
            elif result["random_slope_var"] is None and spec.model == "mixed_slope":
                return False
    except (TypeError, ValueError, KeyError):
        return False
    return True


def _accepted_aggregation(context, outcome: str) -> str | None:
    """Read a supplied preparation choice, never guess one from the concept name.

    The optional data-constraints projection accepts a per-column aggregation
    map when an upstream preparation owner has supplied one.
    """
    raw = getattr(getattr(context, "user_preferences", None), "data_constraints", None)
    try:
        constraints = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    preparation = constraints.get("data_preparation", {}) if isinstance(constraints, dict) else {}
    aggregations = preparation.get("aggregations", {}) if isinstance(preparation, dict) else {}
    return aggregations.get(outcome) if isinstance(aggregations, dict) else None


def repeated_measure_plan_errors(context, plan, *, requested: bool = False) -> tuple[str, ...]:
    steps = [s for s in plan.steps if s.scientific_action_id in REPEATED_ACTIONS
             or str(s.method or "").startswith("repeated_measures(")]
    requested = requested or plan.analysis_type == "repeated_measures"
    if not steps:
        return ("No reviewed repeated-measure model answers the change-over-time question.",) if requested else ()
    errors = []
    known = {variable.name for variable in context.variables} | set(context.cohort.id_columns)
    for step in steps:
        try:
            spec = repeated_measure_spec_for_step(step)
            if spec is None or not repeated_measure_step_matches(step):
                raise ValueError("Require the complete time-axis, aggregation and death-dropout design with exact bound inputs and products.")
            if not set(spec.required_columns) <= known:
                raise ValueError("Repeated-measure time, identity, outcome and sensitivity columns must exist in the context.")
            aggregate = _accepted_aggregation(context, spec.outcome_column)
            if aggregate is not None and aggregate != spec.aggregation:
                raise ValueError("Repeated-measure aggregation differs from the accepted data_preparation choice.")
        except ValueError as exc:
            errors.append(f"{step.step_id}: {exc}")
    if plan.analysis_type == "repeated_measures" and not any(s.planned_analysis_role == "primary" for s in steps):
        errors.append("The repeated-measure question requires one repeated-measure primary model.")
    return tuple(errors)


def repeated_measure_primary_matches(plan) -> bool:
    primary = [s for s in plan.steps if s.planned_analysis_role == "primary"]
    return len(primary) == 1 and repeated_measure_step_matches(primary[0])


def repeated_measure_compiled_method(step):
    """Keep the scientific packet intact across Progressive compilation."""
    spec = step.repeated_measure_spec
    if spec is None:
        raise ValueError("Repeated-measure actions require repeated_measure_spec")
    if not set(spec.required_columns) <= set(step.raw_inputs):
        raise ValueError("Repeated-measure coordinates must all be explicit raw_inputs")
    method = repeated_measure_method(spec)
    # Check action/model agreement before any executable plan is produced.
    proxy = type("RepeatedStep", (), {"method": method, "scientific_action_id": step.scientific_action_id})()
    repeated_measure_spec_for_step(proxy)
    return method


def canonical_repeated_measure_spec() -> RepeatedMeasureSpec:
    """Synthetic display coordinates, never data or execution authority."""
    return RepeatedMeasureSpec(
        identity_column="stay_id", time_column="hours_from_admission", outcome_column="measurement",
        time_origin="icu_admission", time_unit="hours", window_start=0, window_end=144, bin_width=24,
        aggregation="mean", model="mixed_slope", family="gaussian", time_basis="linear",
        dropout_limitation="death_may_cause_informative_dropout", dropout_sensitivity="joint_model_note",
    )
