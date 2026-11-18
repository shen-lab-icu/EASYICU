"""Project plan-applied contracts to digest-bound v2 facts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from pydantic import ValidationError

from easyicu.concept.metadata_sidecar import TimeCoordinate

from ...canonical_json import canonical_sha256
from ...contracts.association_execution import (
    LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
    association_execution_verdict,
    baseline_association_execution_verdict,
    landmark_categorical_association_execution_verdict,
    primary_binary_effect_measure,
)
from ...contracts.descriptive_execution import exposure_outcome_distribution_execution_verdict
from ...contracts.endpoint import EndpointSpec
from ...contracts.model_terms import (
    PreparationRefusal,
    preparation_base_method,
    preparation_for_step,
    validate_model_term_roster,
)
from ...contracts.prediction_validation import external_validation_spec_for_step, external_validation_step_matches
from ...contracts.product_files import RUN_DELIVERABLES
from ...schema import AnalysisPlan, AnalysisStep, ResearchContext
from ..cohort_contract import (
    CohortSchemaError,
    cohort_concept_id_scope,
    sealed_cohort_concept_ids,
    validate_cohort_definition,
)
from ..dependence_authority import (
    DependenceAuthorityError,
    _host_restricted_to_first_icu_stays,
    context_dependence_authority,
)
from ..sensitivity_authority import EXECUTABLE_METHODS_BY_STRATEGY, PrespecifiedSensitivitySpec
from .contract import (
    Binding,
    CohortPredicate,
    Delivery,
    DictionarySnapshot,
    Facet,
    FactAuthority,
    HostFact,
    Interval,
    Method,
    NotApplicable,
    Parameter,
    Point,
    Provenance,
    RepeatStayPolicy,
    RepeatStayReporting,
    RequirementKey,
    Role,
    Time,
    TimeZero,
    Unit,
    Window,
)
from .registry import (
    STUDY_POPULATION,
    contract_registry_sha256,
    registered_anchor,
    registered_concept_id,
    registered_grading,
)

_ANALYSIS_ROLES = {"primary", "secondary", "sensitivity"}
_HOUR_UNITS = {"h", "hours"}
_TIME_UNITS = {"minutes": "min", "hours": "h", "days": "d"}


@dataclass(frozen=True)
class ProjectedFacts:
    authority: FactAuthority
    facts: tuple[HostFact, ...]
    study_choices: tuple[HostFact, ...] = ()


class _Projection:
    def __init__(
        self,
        plan: AnalysisPlan,
        context: ResearchContext | None,
        dictionary: DictionarySnapshot,
        time_coordinates: Sequence[TimeCoordinate],
    ) -> None:
        self.plan = plan
        self.context = context
        self.dictionary = dictionary
        self.authority = FactAuthority(
            plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
            context_sha256=canonical_sha256(context.model_dump(mode="json")) if context else None,
            dictionary_version=dictionary.version,
            registry_sha256=contract_registry_sha256(),
            coordinate_sha256=canonical_sha256(sorted({
                (clock.column, clock.origin, clock.unit) for clock in time_coordinates
            })) if time_coordinates else None,
        )
        self.variables = {item.name: item for item in context.variables} if context else {}
        self.variable_positions = {name: index for index, name in enumerate(self.variables)}
        self.clocks: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for clock in time_coordinates:
            self.clocks[clock.column].add((clock.origin, clock.unit))
        for name, variable in self.variables.items():
            semantics = variable.observation_semantics
            if semantics and semantics.time_origin and semantics.time_unit:
                self.clocks[name].add((semantics.time_origin, semantics.time_unit))
        self.facts: list[HostFact] = []
        self.study_choices: list[HostFact] = []

    def bind(
        self, coordinates: Sequence[tuple[str, Role]],
    ) -> tuple[Binding, ...]:
        bindings = []
        for column, role in coordinates:
            concept = registered_concept_id(column, self.variables.get(column), self.dictionary)
            if concept is None:
                return ()
            binding = Binding(concept_id=concept, role=role)
            if binding in bindings:
                return ()
            bindings.append(binding)
        return tuple(bindings)

    def clock_anchor(self, column: str | None) -> str | None:
        coordinates = self.clocks.get(column or "", set())
        # A typed hours-based contract cannot consume a different axis/unit.
        if len(coordinates) != 1:
            return None
        [(origin, unit)] = coordinates
        return registered_anchor(origin, self.dictionary) if unit in _HOUR_UNITS else None

    def add(
        self, source: str, bindings: tuple[Binding, ...], facet: Facet,
        time: Time = NotApplicable(),
        *, study_choice: bool = False,
    ) -> None:
        if not bindings:
            return
        if isinstance(time, (Interval, Point)):
            bindings += (Binding(role="time_anchor", concept_id=time.anchor),)
        target = self.study_choices if study_choice else self.facts
        target.append(HostFact(
            key=RequirementKey(bindings=bindings, time=time, facet=facet),
            provenance=Provenance(source_field=source, authority=self.authority),
        ))

    def cohort(self) -> None:
        cohort = self.plan.cohort
        if cohort is None:
            return
        try:
            with cohort_concept_id_scope(sealed_cohort_concept_ids(self.context) if self.context else ()):
                validate_cohort_definition(cohort)
        except CohortSchemaError:
            return
        for field, side, predicates in (("inclusion", "include", cohort.inclusion), ("exclusion", "exclude", cohort.exclusion)):
            for index, predicate in enumerate(predicates):
                window = predicate.time_window
                anchor = registered_anchor(window.anchor, self.dictionary)
                bindings = self.bind(((predicate.concept_id, "population"),))
                if not anchor or not bindings:
                    continue
                try:
                    facet = CohortPredicate(
                        operation=predicate.aggregation, operator=predicate.op,
                        value=predicate.value, polarity=side, missing_policy=predicate.missing_policy,
                    )
                except ValidationError:
                    continue
                self.add(
                    f"plan.cohort.{field}[{index}]", bindings, facet,
                    Interval(
                        anchor=anchor, unit="h",
                        start=window.start_offset_hours if isfinite(window.start_offset_hours) else "-inf",
                        end=window.end_offset_hours if isfinite(window.end_offset_hours) else "inf",
                        start_inclusive=isfinite(window.start_offset_hours), end_inclusive=False,
                    ),
                )

    def clinical(self, roles: Mapping[str, set[Role]]) -> None:
        if not self.context:
            return
        for column, scientific_roles in roles.items():
            variable = self.variables.get(column)
            if variable is None:
                continue
            source = f"context.variables[{self.variable_positions[column]}]"
            grading = registered_grading(variable, self.dictionary)
            for role in sorted(scientific_roles):
                bindings = self.bind(((column, role),))
                if variable.unit:
                    try:
                        unit = Unit(code=variable.unit)
                    except ValidationError:
                        pass
                    else:
                        self.add(f"{source}.unit", bindings, unit)
                if grading:
                    self.add(f"{source}.clinical_definition", bindings, grading)

    def preparation(self, step: AnalysisStep, roles: Mapping[str, set[Role]], source: str) -> None:
        try:
            preparation = preparation_for_step(step)
        except PreparationRefusal:
            return
        if preparation is None:
            return
        source = f"{source}.method.data_preparation"
        inputs = set(step.inputs)
        for index, window in enumerate(preparation.windows):
            anchor = self.clock_anchor(window.time_column)
            scientific_roles = roles.get(window.variable, set())
            if (
                anchor is None or len(scientific_roles) != 1
                or not {window.variable, window.group_column, window.time_column} <= inputs
                or not all(map(isfinite, (window.start_hours, window.end_hours)))
            ):
                continue
            time = Interval(
                anchor=anchor, unit="h", start=window.start_hours, end=window.end_hours,
                start_inclusive=True, end_inclusive=False,
            )
            field = f"{source}.windows[{index}]"
            bindings = self.bind(((window.variable, next(iter(scientific_roles))),))
            self.add(field, bindings, Window(purpose="aggregation", operation=window.function), time)
            # The maximum of an existing stage series is the stage reported
            # within this applied window. Its baseline lookback is independent.
            variable = self.variables.get(window.variable)
            if window.function == "max" and variable and bindings:
                grading = registered_grading(variable, self.dictionary)
                if (
                    grading and grading.contract_id == "kdigo_aki_2012"
                    and bindings[0].concept_id in {"kdigo_creat", "kdigo_uo", "kdigo_aki"}
                ):
                    self.add(field, bindings, grading, time)
        missing = preparation.missing_data
        if missing and missing.strategy == "multiple_imputation" and missing.variables:
            columns = sorted(item.name for item in missing.variables)
            if (
                not set(columns).union(missing.auxiliaries) <= inputs
                or any(len(roles.get(column, ())) != 1 for column in columns)
            ):
                return
            bindings = self.bind(tuple((column, next(iter(roles[column]))) for column in columns))
            if not bindings or len({item.concept_id for item in bindings}) != len(bindings):
                return
            auxiliaries = []
            for column in missing.auxiliaries:
                concept = registered_concept_id(column, self.variables.get(column), self.dictionary)
                if concept is None:
                    return
                auxiliaries.append(concept)
            concept_by_column = dict(zip(columns, (item.concept_id for item in bindings), strict=True))
            m = Parameter(name="m", value=missing.m)
            parameters = (
                m,
                Parameter(name="iterations", value=missing.iterations),
                Parameter(name="donors", value=missing.donors),
                Parameter(name="seed", value=missing.seed),
                Parameter(name="auxiliaries", value=tuple(sorted(auxiliaries))),
                *(Parameter(name=f"impute/{concept_by_column[item.name]}", value=(item.method, *item.levels))
                  for item in missing.variables),
            )
            self.add(
                f"{source}.missing_data", bindings,
                Method(action_id="multiple_imputation", parameters=parameters),
            )
            for binding in bindings:
                self.add(
                    f"{source}.missing_data", (binding,),
                    Method(action_id="multiple_imputation", parameters=(m,)),
                )

    def landmark_point(self, spec: PrespecifiedSensitivitySpec, endpoint: EndpointSpec) -> Point | None:
        if spec.strategy != "landmark" or spec.landmark_hours is None or endpoint.time_column != spec.event_time_variable:
            return None
        clock = self.variables.get(spec.event_time_variable or "")
        semantics = clock.observation_semantics if clock else None
        if semantics and semantics.event_status_column != endpoint.event_column:
            return None
        anchor = self.clock_anchor(spec.event_time_variable)
        if anchor and anchor == registered_anchor(endpoint.time_origin, self.dictionary):
            return Point(anchor=anchor, unit="h", offset=spec.landmark_hours)
        return None

    def landmark_outcome(self, endpoint: EndpointSpec) -> tuple[Binding, ...]:
        if endpoint.event_column is None:
            return ()
        bindings = self.bind(((endpoint.event_column, "outcome"),))
        if endpoint.name in self.variables or endpoint.name in self.dictionary.concept_ids:
            if self.bind(((endpoint.name, "outcome"),)) != bindings:
                return ()
        return bindings

    def landmarks(self, step: AnalysisStep, source: str) -> None:
        context = self.context
        if not context or not context.user_preferences or not context.endpoint:
            return
        endpoint = context.endpoint
        if endpoint.name != context.target_outcome or endpoint.event_column not in step.inputs:
            return
        bindings = self.landmark_outcome(endpoint)
        if not bindings:
            return
        for spec in context.user_preferences.sensitivity_specs:
            if (
                spec.spec_id not in step.sensitivity_spec_ids
                or preparation_base_method(step.method) not in EXECUTABLE_METHODS_BY_STRATEGY["landmark"]
                or spec.event_time_variable not in step.inputs
                or (spec.observation_duration_variable and spec.observation_duration_variable not in step.inputs)
            ):
                continue
            point = self.landmark_point(spec, endpoint)
            if point:
                self.add(
                    f"{source}.sensitivity_spec_ids/{spec.spec_id}",
                    bindings, TimeZero(purpose="sensitivity_landmark"), point,
                )

    def study_landmarks(self) -> None:
        context = self.context
        if not context or not context.user_preferences or not context.target_outcome or not context.endpoint:
            return
        endpoint = context.endpoint
        if endpoint.name != context.target_outcome:
            return
        bindings = self.landmark_outcome(endpoint)
        if not bindings:
            return
        for index, spec in enumerate(context.user_preferences.sensitivity_specs):
            point = self.landmark_point(spec, endpoint)
            if point:
                self.add(
                    f"context.user_preferences.sensitivity_specs[{index}]",
                    bindings, TimeZero(purpose="sensitivity_landmark"),
                    point, study_choice=True,
                )

    def model_terms(self, step: AnalysisStep, source: str) -> None:
        if not association_execution_verdict(step).claimed:
            return
        [model] = step.model_requirements
        if model.analysis_role != step.planned_analysis_role or model.model_terms is None:
            return
        try:
            validate_model_term_roster(
                terms=model.model_terms, exposure=model.exposure_source, covariates=model.covariates,
            )
        except ValueError:
            return
        if not {model.outcome, *(term.name for term in model.model_terms)} <= set(step.inputs):
            return
        for position, term in enumerate(model.model_terms):
            self.add(
                f"{source}.model_requirements[0].model_terms[{position}]",
                self.bind(((term.name, term.role), (model.outcome, "outcome"))),
                Method(action_id="model_term", parameters=(
                    Parameter(name="model_method", value=model.method_family),
                    Parameter(name="coding", value=term.coding),
                    Parameter(name="transform", value=term.transform),
                    Parameter(name="levels", value=tuple(term.levels or ())),
                    Parameter(name="reference_level", value=term.reference_level),
                )),
            )

    def endpoint_origin(self, step: AnalysisStep, source: str) -> None:
        family = step.family_primary_result_requirement
        endpoint = self.plan.endpoint
        if (
            family is None or family.analysis_family != "survival"
            or endpoint is None or endpoint.kind != "time_to_event"
            or (self.context is not None and self.context.endpoint != endpoint)
            or family.time_column != endpoint.time_column
            or family.event_column != endpoint.event_column
            or family.time_origin is None or family.time_origin != endpoint.time_origin
            or family.outcome != endpoint.name
            or not {endpoint.time_column, endpoint.event_column} <= set(step.inputs)
        ):
            return
        anchor = registered_anchor(family.time_origin, self.dictionary)
        unit = _TIME_UNITS.get(family.time_unit)
        if anchor and unit:
            self.add(
                f"{source}.family_primary_result_requirement.time_origin",
                self.bind(((family.outcome, "outcome"),)), TimeZero(),
                Point(anchor=anchor, unit=unit, offset=0),
            )

    def grouping(self, step: AnalysisStep, source: str) -> None:
        spec = step.exposure_outcome_distribution_spec
        if spec is None or not exposure_outcome_distribution_execution_verdict(step).claimed:
            return
        self.add(
            f"{source}.exposure_outcome_distribution_spec",
            self.bind(((spec.exposure, "exposure"), (spec.outcome, "outcome"))),
            Method(action_id="descriptive.exposure_outcome_distribution", parameters=(
                Parameter(name="contract_sha256", value=canonical_sha256(spec.model_dump(mode="json"))),
            )),
        )

    def repeat_stays(self, primary: tuple[int, AnalysisStep] | None) -> None:
        if self.context is None:
            return
        provenance = self.context.cohort.provenance
        first = _host_restricted_to_first_icu_stays(provenance)
        if "first_icu_stay_per_hospitalization_restriction" in provenance:
            return
        if first:
            policy = "first_icu_stay_per_patient"
            source = "context.cohort.provenance.first_icu_stay_restriction"
            self.add(source, (STUDY_POPULATION,), RepeatStayPolicy(policy=policy), study_choice=True)
        elif "first_icu_stay_restriction" not in provenance:
            try:
                authority = context_dependence_authority(self.context)
            except DependenceAuthorityError:
                return
            if authority is None:
                return
            policy = "retain_each_stay_cluster_by_patient"
            self.add(
                "context.user_preferences.data_constraints.analysis_design", (STUDY_POPULATION,),
                RepeatStayPolicy(policy=policy), study_choice=True,
            )
            if primary is None:
                return
            index, step = primary
            model = step.model_requirements
            runtime = step.runtime_dependence_contract
            execution_dependence = [item.dependence for item in model]
            if runtime is not None:
                execution_dependence.append(runtime.dependence)
            if not execution_dependence or any(
                item != authority for item in execution_dependence
            ):
                return
            field = "model_requirements" if model else "runtime_dependence_contract.dependence"
            source = f"plan.steps[{index}].{field}"
        else:
            return
        self.add(source, (STUDY_POPULATION,), RepeatStayPolicy(policy=policy))
        self.add(source, (STUDY_POPULATION,), RepeatStayReporting())

    def external_validation(self, step: AnalysisStep, source: str) -> None:
        try:
            spec = external_validation_spec_for_step(step)
            if spec is None or not external_validation_step_matches(step):
                return
        except ValueError:
            return
        bindings = self.bind((
            (spec.outcome_concept, "outcome"),
            *((column, "covariate") for column in spec.predictor_concepts),
        ))
        field = f"{source}.method.external_validation_spec"
        scope_parameters = (
            Parameter(name="validation_databases", value=tuple(sorted(spec.validation_databases))),
            Parameter(name="recalibration", value=spec.recalibration),
        )
        self.add(
            field, bindings,
            Method(action_id="prediction.external_validation", parameters=(
                *scope_parameters,
                Parameter(name="development_database", value=spec.development_database),
                Parameter(name="frozen_model_sha256", value=spec.frozen_model_sha256),
                Parameter(name="confirmed_meaning_changes", value=tuple(sorted(
                    f"{change.database}/{change.concept}" for change in spec.confirmed_changes
                ))),
            )),
        )
        self.add(field, bindings, Method(action_id="prediction.external_validation", parameters=scope_parameters))

    def binary_measures(self, step: AnalysisStep, roles: Mapping[str, set[Role]], source: str) -> None:
        measure = primary_binary_effect_measure(step)
        if not measure:
            return
        try:
            preparation_for_step(step)
        except PreparationRefusal:
            return
        if step.scientific_capability == LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID:
            field = "scientific_capability"
            coordinates = tuple(
                (column, role) for column, declared in roles.items()
                for role in sorted(declared) if role in {"exposure", "outcome"}
            )
        else:
            if (
                not association_execution_verdict(step).claimed
                and not baseline_association_execution_verdict(step).claimed
                and not landmark_categorical_association_execution_verdict(step).claimed
            ):
                return
            [requirement] = step.model_requirements
            if requirement.analysis_role != "primary" or not set(requirement.covariates or ()) <= set(step.inputs):
                return
            field = "model_requirements[0].method_family"
            coordinates = ((requirement.exposure_source, "exposure"), (requirement.outcome, "outcome"))
        if (
            len(coordinates) != 2 or {role for _, role in coordinates} != {"exposure", "outcome"}
            or any(roles.get(column) != {role} for column, role in coordinates)
        ):
            return
        bindings = self.bind(coordinates)
        if len(bindings) != 2 or len({binding.concept_id for binding in bindings}) != 2:
            return
        self.add(
            f"{source}.{field}", bindings,
            Method(action_id="binary_effect_measure", parameters=(Parameter(name="measure", value=measure),)),
        )


def _step_roles(step: AnalysisStep) -> Iterator[tuple[str, Role]]:
    for requirement in step.model_requirements:
        if requirement.analysis_role != step.planned_analysis_role:
            continue
        yield requirement.exposure_source, "exposure"
        yield requirement.outcome, "outcome"
        for column in requirement.covariates or ():
            yield column, "covariate"
    family = step.family_primary_result_requirement
    if family is not None and step.planned_analysis_role == "primary":
        yield family.exposure_source, "exposure"
        yield family.outcome, "outcome"
        for column in family.covariates or ():
            yield column, "covariate"
    distribution = step.exposure_outcome_distribution_spec
    if distribution is not None and exposure_outcome_distribution_execution_verdict(step).claimed:
        yield distribution.exposure, "exposure"
        yield distribution.outcome, "outcome"


def project_facts(
    plan: AnalysisPlan,
    *,
    context: ResearchContext | None,
    dictionary: DictionarySnapshot,
    time_coordinates: Sequence[TimeCoordinate] = (),
) -> ProjectedFacts:
    projection = _Projection(plan, context, dictionary, time_coordinates)
    projection.cohort()
    projection.study_landmarks()
    all_roles: dict[str, set[Role]] = defaultdict(set)
    primary_steps = []
    for index, step in enumerate(plan.steps):
        source = f"plan.steps[{index}]"
        projection.grouping(step, source)
        if step.planned_analysis_role not in _ANALYSIS_ROLES:
            continue
        roles: dict[str, set[Role]] = defaultdict(set)
        inputs = set(step.inputs)
        for column, role in _step_roles(step):
            if column in inputs:
                roles[column].add(role)
                all_roles[column].add(role)
        projection.preparation(step, roles, source)
        projection.model_terms(step, source)
        projection.landmarks(step, source)
        projection.external_validation(step, source)
        if step.planned_analysis_role == "primary":
            primary_steps.append((index, step))
            projection.endpoint_origin(step, source)
            projection.binary_measures(step, roles, source)
    projection.clinical(all_roles)
    projection.repeat_stays(primary_steps[0] if len(primary_steps) == 1 else None)
    for output_id in RUN_DELIVERABLES:
        projection.add(f"run_delivery.{output_id}", (STUDY_POPULATION,), Delivery(output_id=output_id))
    return ProjectedFacts(
        authority=projection.authority, facts=tuple(projection.facts),
        study_choices=tuple(projection.study_choices),
    )
