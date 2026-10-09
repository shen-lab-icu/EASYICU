"""A target trial spec compiles to the protocol it states.

The study setup states a target trial's elements; the host decides, for each,
whether an emulation over the study's input can carry it out.  Every element
gets one disposition, every reason code is reachable from a synthetic study,
the capture registry decides what an absent treatment record means, and a
database the registry or the endpoint owner does not cover is refused with a
typed reason instead of being emulated.  Fixtures are synthetic and vary the
treatment, so that no rule keys on one drug class or one benchmark question.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.outcome_availability import FixedHorizonMortalityEndpoint
from easyicu.research_agent.authority.target_trial_claim_terms import (
    TARGET_TRIAL_ASSUMPTIONS,
    TargetTrialClaimTerms,
)
from easyicu.research_agent.planning import target_trial_compile as compile_module
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.target_trial_compile import (
    CONFOUNDER_NOT_APPLIED_REASONS,
    CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
    HOST_ELEMENTS,
    NOT_APPLIED_REASONS,
    REQUIRES_EXTRACTION_REASONS,
    STATED_ELEMENTS,
    TARGET_TRIAL_COMPILE_SCHEMA_VERSION,
    CompiledElement,
    CompiledTargetTrial,
    compile_target_trial,
)
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.research_agent.planning.treatment_capture import (
    LoadedCaptureRegistry,
    TreatmentCaptureRegistry,
    packaged_treatment_capture_registry,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ObservationSemantics,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
_VASOACTIVE = ("vaso_ind", "other_vaso")


def _event_time(status: str) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=f"{status}_time",
        role=VariableRole.OTHER,
        dtype="float64",
        observation_semantics=ObservationSemantics(
            kind="conditional_event_time",
            event_status_column=status,
            representative_column=f"{status}_time",
            time_origin="icu_admission",
            time_unit="h",
        ),
    )


def _onset(concept: str, window: str) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=f"{concept}_onset_time",
        role=VariableRole.OTHER,
        dtype="float64",
        analysis_window=window,
    )


def _status(name: str, window: str) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=VariableRole.OTHER,
        dtype="int64",
        analysis_window=window,
        observed_domain=_BINARY,
    )


def _variables(
    treatments: tuple[str, ...], onset_window: str
) -> list[ConceptDescriptor]:
    return [
        ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
        ConceptDescriptor(
            name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years"
        ),
        ConceptDescriptor(
            name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit="days"
        ),
        ConceptDescriptor(
            name="death",
            role=VariableRole.OUTCOME,
            dtype="int64",
            observed_domain=_BINARY,
        ),
        _event_time("death"),
        ConceptDescriptor(
            name="mort_28d",
            role=VariableRole.OUTCOME,
            dtype="int64",
            observed_domain=_BINARY,
        ),
        ConceptDescriptor(
            name="followup_days_28d",
            role=VariableRole.OUTCOME,
            dtype="float64",
            unit="days",
        ),
        # The indication, decided over the hours before time zero.
        _status("shock", "icu_admission[0,6]h"),
        # A status decided only after a six-hour time zero.
        _status("shock_day1", "icu_admission[0,24]h"),
        # An outcome-like status: no window proves it observed by time zero.
        ConceptDescriptor(
            name="aki",
            role=VariableRole.OUTCOME,
            dtype="int64",
            observed_domain=_BINARY,
        ),
        # A laboratory value over the host's 24-hour window.
        ConceptDescriptor(
            name="lact_max",
            role=VariableRole.LAB,
            dtype="float64",
            source_concept="lact",
            unit="mmol/L",
        ),
        # A vital sign over its own window before time zero.
        ConceptDescriptor(
            name="map_min",
            role=VariableRole.VITAL,
            dtype="float64",
            source_concept="map",
            unit="mmHg",
            analysis_window="icu_admission[0,6]h",
        ),
        *(_onset(concept, onset_window) for concept in treatments),
    ]


def _ctx(
    *,
    database: str = "miiv",
    treatments: tuple[str, ...] = _VASOACTIVE,
    onset_window: str = "icu_admission[0,30]h",
    without: tuple[str, ...] = (),
    extra: tuple[ConceptDescriptor, ...] = (),
    id_columns: tuple[str, ...] = ("stay_id",),
    provenance: dict[str, Any] | None = None,
    n_stays: int = 100,
    n_patients: int | None = 100,
) -> ResearchContext:
    variables = [
        variable
        for variable in _variables(treatments, onset_window)
        if variable.name not in without
    ]
    return ResearchContext(
        research_question="Does an earlier start of the treatment change death?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database=database,
            n_stays=n_stays,
            n_patients=n_patients,
            id_columns=list(id_columns),
            outcome_columns=["mort_28d"],
            provenance=dict(provenance or {}),
        ),
        variables=[*variables, *extra],
        target_outcome="mort_28d",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(
                {
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "ICU admission",
                        "hours": 24,
                    }
                }
            )
        ),
    )


_POPULATION = (
    {
        "id": "c1",
        "quote": "patients in shock",
        "kind": "condition_present",
        "concepts_all_of": ["shock"],
        "window": {"start_hours": 0, "end_hours": 6},
    },
    {"id": "c2", "quote": "adults", "kind": "age_years", "min_years": 18},
)


def _population(*criteria: dict[str, Any]) -> PopulationSpec:
    return PopulationSpec.model_validate(
        {
            "criteria": [
                {"source": "question", "role": "include", **item}
                for item in (criteria or _POPULATION)
            ]
        }
    )


def _spec(**changes: Any) -> TargetTrialSpec:
    data: dict[str, Any] = {
        "treatment": {
            "quote": "a vasoactive drug",
            "source": "question",
            "concepts": list(_VASOACTIVE),
            "treatment_class": "vasoactive",
        },
        "strategies": {
            "quote": "start it early or not",
            "source": "question",
            "initiate_label": "Early start",
            "defer_label": "No early start",
        },
        "time_zero": {
            "quote": "six hours after ICU admission",
            "source": "question",
            "hours_after_icu_admission": 6,
        },
        "grace_period": {"quote": "within a day", "source": "question", "hours": 24},
        "outcome": {
            "quote": "death by day 28",
            "source": "question",
            "endpoint": "mort_28d",
        },
        "indication": {
            "quote": "patients in shock",
            "source": "question",
            "criterion_ids": ["c1"],
        },
        "confounders": [
            {
                "name": "age",
                "source": "question",
                "clinical_rationale": "Older patients are started later and die more often.",
            },
            {
                "name": "map_min",
                "source": "conversation",
                "clinical_rationale": "A lower blood pressure prompts the start and predicts death.",
            },
        ],
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return TargetTrialSpec.model_validate(data)


def _confounder(name: str, source: str = "question") -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "clinical_rationale": "A synthetic rationale for this test case.",
    }


def _compile(
    spec: TargetTrialSpec | None = None,
    context: ResearchContext | None = None,
    population: PopulationSpec | None = None,
    registry: LoadedCaptureRegistry | None = None,
) -> CompiledTargetTrial:
    spec = spec or _spec()
    context = context or _ctx()
    compiled = compile_population(
        population or _population(),
        context,
        time_zero_hours=spec.time_zero.hours_after_icu_admission,
    )
    return compile_target_trial(spec, context, population=compiled, registry=registry)


def _registry(data: dict[str, Any]) -> LoadedCaptureRegistry:
    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    return LoadedCaptureRegistry(
        registry=TreatmentCaptureRegistry.model_validate_json(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _packaged_data() -> dict[str, Any]:
    return packaged_treatment_capture_registry().registry.model_dump(mode="json")


def _entry(database: str, concept: str, agents: list[str], **changes: Any) -> dict:
    return {
        "database": database,
        "concept": concept,
        "agents": agents,
        "definition": {
            "dictionary": "concept-dict.json",
            "components": [
                {"component": f"{agent}_dur", "agent": agent} for agent in agents
            ],
        },
        "capture_setting": "icu",
        "absent_in_capture": "absent_row_is_no_event",
        "pre_admission_visible": False,
        "basis": "development_assumption",
        "declared_by": "a synthetic registry for this test",
        "declared_at": "2026-10-09",
        "note": "Synthetic entry.",
        **changes,
    }


# -- a trial the input supports ------------------------------------------------


def test_a_trial_the_input_supports_carries_every_element() -> None:
    trial = _compile()

    assert [item.element for item in trial.elements] == [
        *STATED_ELEMENTS,
        *HOST_ELEMENTS,
    ]
    assert {item.element: item.disposition for item in trial.elements} == {
        **{name: "applied" for name in STATED_ELEMENTS},
        **{name: "host_added" for name in HOST_ELEMENTS},
    }
    assert [(item.name, item.temporal_role) for item in trial.confounders] == [
        ("age", "baseline_static"),
        ("map_min", "at_or_before_time_zero"),
    ]
    assert trial.blocking == () and trial.approvable
    treatment = trial.element("treatment")
    assert treatment.parameters["onset_columns"] == [
        "vaso_ind_onset_time",
        "other_vaso_onset_time",
    ]
    assert set(treatment.parameters["agents"]) == set(
        packaged_treatment_capture_registry().registry.class_agents("vasoactive")
    )
    assert trial.element("grace_period").parameters["window_hours"] == [6, 30]
    assert trial.element("resampling_unit").parameters["unit"] == "icu_stay"
    assert trial.element("adjustment").parameters["confounders"] == ["age", "map_min"]
    assert [item.kind for item in trial.confirmations] == [
        "capture_assumption",
        "capture_assumption",
        "treatment_class",
        "confounder_set",
        "emulation_assumptions",
    ]


def test_the_record_is_stable_and_names_what_it_was_compiled_from() -> None:
    first, second = _compile(), _compile()
    record = first.record()

    assert first.sha256() == second.sha256()
    assert json.loads(json.dumps(record)) == record
    assert record["schema_version"] == TARGET_TRIAL_COMPILE_SCHEMA_VERSION
    assert record["evidence_ceiling"] == "analysis_only"
    assert record["capture_registry_sha256"] == (
        packaged_treatment_capture_registry().sha256
    )
    assert (
        record["population_sha256"]
        == compile_population(_population(), _ctx(), time_zero_hours=6).sha256()
    )
    assert [row["item"] for row in record["protocol"]] == [
        "eligibility",
        "treatment_strategies",
        "assignment",
        "time_zero",
        "follow_up",
        "outcome",
        "causal_contrast",
        "analysis_plan",
    ]
    # The card states what the executed protocol states: eligibility needs the
    # vital status at the horizon, which is known only after time zero.
    assert record["protocol"][0]["text"].endswith(
        "and with a known vital status 28 days after ICU admission."
    )
    assert record["materialization"]["covariate_window"] == {
        "start_hours": 0,
        "end_hours": 6,
    }
    assert record["materialization"]["treatment_onset_window"] == {
        "start_hours": 0,
        "end_hours": 30,
    }


def test_compiling_changes_neither_the_spec_nor_the_context() -> None:
    spec, context = _spec(), _ctx()
    before = (spec.model_dump(mode="json"), context.model_dump(mode="json"))

    _compile(spec, context)

    assert (spec.model_dump(mode="json"), context.model_dump(mode="json")) == before


def test_the_population_is_compiled_at_the_trials_time_zero() -> None:
    context = _ctx()
    population = compile_population(_population(), context, time_zero_hours=12)

    with pytest.raises(ValueError, match="time zero"):
        compile_target_trial(_spec(), context, population=population)
    with pytest.raises(ValueError, match="time zero"):
        compile_target_trial(
            _spec(), context, population=compile_population(_population(), context)
        )


# -- each element's reasons ----------------------------------------------------

_RENAL = {
    "quote": "dialysis",
    "concepts": ["rrt"],
    "treatment_class": "kidney_replacement",
}

#: (case, build, element, disposition, reason): each element reason code from
#: a synthetic study.
_ELEMENT_CASES: list[tuple[str, Any, str, str, str]] = [
    (
        "class the registry does not name",
        lambda: _compile(_spec(treatment={"treatment_class": "antibiotic"})),
        "treatment",
        "not_applied",
        "tte_treatment_class_unknown",
    ),
    (
        "a value concept has no start",
        lambda: _compile(
            _spec(
                treatment={
                    "concepts": ["norepi_equiv"],
                    "treatment_class": "vasopressor",
                }
            )
        ),
        "treatment",
        "not_applied",
        "tte_treatment_not_offered",
    ),
    (
        "an event concept without a capture entry",
        lambda: _compile(
            _spec(treatment={"concepts": ["milrinone"], "treatment_class": "inotrope"}),
            _ctx(treatments=("milrinone",)),
        ),
        "treatment",
        "not_applied",
        "tte_treatment_capture_undeclared",
    ),
    (
        "concepts missing drugs of the class",
        lambda: _compile(
            _spec(
                treatment={"concepts": ["vaso_ind"], "treatment_class": "vasopressor"}
            ),
            _ctx(treatments=("vaso_ind",)),
        ),
        "treatment",
        "not_applied",
        "tte_treatment_definition_partial",
    ),
    (
        "no onset column",
        lambda: _compile(_spec(), _ctx(treatments=())),
        "treatment",
        "requires_extraction",
        "tte_treatment_onset_not_materialized",
    ),
    (
        "an onset read only after admission",
        lambda: _compile(_spec(), _ctx(onset_window="icu_admission[2,30]h")),
        "treatment",
        "requires_extraction",
        "tte_treatment_onset_not_materialized",
    ),
    (
        "a dose in the strategy",
        lambda: _compile(
            _spec(
                not_typed=[
                    {
                        "quote": "at more than 0.1 per kilogram",
                        "source": "question",
                        "why": "a dose threshold no field states",
                        "affects_strategy": True,
                    }
                ]
            )
        ),
        "strategies",
        "not_applied",
        "tte_strategy_not_typed",
    ),
    (
        "time zero at admission",
        lambda: _compile(_spec(time_zero={"hours_after_icu_admission": 0})),
        "time_zero",
        "not_applied",
        "tte_time_zero_not_offered",
    ),
    (
        "time zero beyond the menu",
        lambda: _compile(_spec(time_zero={"hours_after_icu_admission": 80})),
        "time_zero",
        "not_applied",
        "tte_time_zero_not_offered",
    ),
    (
        "eligibility decided after time zero",
        lambda: _compile(
            population=_population(
                {
                    "id": "c1",
                    "quote": "shock on the first day",
                    "kind": "condition_present",
                    "concepts_all_of": ["shock_day1"],
                    "window": {"start_hours": 0, "end_hours": 24},
                }
            )
        ),
        "time_zero",
        "not_applied",
        "tte_eligibility_after_time_zero",
    ),
    (
        "a grace period beyond the menu",
        lambda: _compile(_spec(grace_period={"hours": 25})),
        "grace_period",
        "not_applied",
        "tte_grace_period_not_offered",
    ),
    (
        "an onset that ends inside the grace period",
        lambda: _compile(_spec(), _ctx(onset_window="icu_admission[0,20]h")),
        "grace_period",
        "requires_extraction",
        "tte_grace_beyond_capture",
    ),
    (
        "an endpoint with no horizon",
        lambda: _compile(_spec(outcome={"endpoint": "death"})),
        "outcome",
        "not_applied",
        "tte_endpoint_unsupported",
    ),
    (
        "an endpoint the input lacks",
        lambda: _compile(_spec(), _ctx(without=("mort_28d", "followup_days_28d"))),
        "outcome",
        "requires_extraction",
        "tte_endpoint_not_materialized",
    ),
    (
        "no indication",
        lambda: _compile(_spec(indication=None)),
        "indication",
        "not_applied",
        "tte_indication_unstated",
    ),
    (
        "an indication the population does not state",
        lambda: _compile(_spec(indication={"criterion_ids": ["c9"]})),
        "indication",
        "not_applied",
        "tte_indication_not_in_population",
    ),
    (
        "an indication that removes stays",
        lambda: _compile(
            _spec(indication={"criterion_ids": ["c3"]}),
            population=_population(
                *_POPULATION,
                {
                    "id": "c3",
                    "quote": "without shock",
                    "role": "exclude",
                    "kind": "condition_present",
                    "concepts_all_of": ["shock"],
                    "window": {"start_hours": 0, "end_hours": 6},
                },
            ),
        ),
        "indication",
        "not_applied",
        "tte_indication_not_inclusion",
    ),
    (
        "an age as the indication",
        lambda: _compile(_spec(indication={"criterion_ids": ["c2"]})),
        "indication",
        "not_applied",
        "tte_indication_not_clinical",
    ),
    (
        "an indication nothing applies",
        lambda: _compile(
            _spec(indication={"criterion_ids": ["c3"]}),
            population=_population(
                *_POPULATION,
                {
                    "id": "c3",
                    "quote": "an undefined state",
                    "kind": "condition_present",
                    "concepts_all_of": ["zzz_undefined_state"],
                    "window": {"start_hours": 0, "end_hours": 6},
                },
            ),
        ),
        "indication",
        "not_applied",
        "tte_indication_not_applied",
    ),
    (
        "an indication an extraction would apply",
        lambda: _compile(
            _spec(indication={"criterion_ids": ["c3"]}),
            population=_population(
                *_POPULATION,
                {
                    "id": "c3",
                    "quote": "on renal replacement",
                    "kind": "condition_present",
                    "concepts_all_of": ["rrt"],
                    "window": {"start_hours": 0, "end_hours": 6},
                },
            ),
        ),
        "indication",
        "requires_extraction",
        "tte_indication_requires_extraction",
    ),
    (
        "a death time read by date",
        lambda: _compile(_spec(), _ctx(database="aumc")),
        "death_time",
        "not_applied",
        "tte_death_time_not_hourly",
    ),
    (
        "no death time in the input",
        lambda: _compile(_spec(), _ctx(without=("death_time",))),
        "death_time",
        "requires_extraction",
        "tte_death_time_not_materialized",
    ),
    (
        "no death status beside its time",
        lambda: _compile(_spec(), _ctx(without=("death",))),
        "death_time",
        "requires_extraction",
        "tte_death_time_not_materialized",
    ),
    (
        "no ICU length of stay in the input",
        lambda: _compile(_spec(), _ctx(without=("los_icu",))),
        "icu_exit",
        "requires_extraction",
        "tte_icu_exit_unavailable",
    ),
    (
        "repeated stays and no patient identity",
        lambda: _compile(_spec(), _ctx(n_stays=120, n_patients=100)),
        "resampling_unit",
        "requires_extraction",
        "tte_patient_identity_unavailable",
    ),
    (
        "no confounder stated",
        lambda: _compile(_spec(confounders=[])),
        "adjustment",
        "not_applied",
        "tte_no_confounder_carried",
    ),
    (
        "only a confounder nothing carries",
        lambda: _compile(_spec(confounders=[_confounder("aki")])),
        "adjustment",
        "not_applied",
        "tte_no_confounder_carried",
    ),
    (
        "only a confounder an extraction carries",
        lambda: _compile(_spec(confounders=[_confounder("crea")])),
        "adjustment",
        "requires_extraction",
        "tte_confounders_require_extraction",
    ),
]


@pytest.mark.parametrize(
    ("case", "build", "element", "disposition", "reason"),
    _ELEMENT_CASES,
    ids=[case[0] for case in _ELEMENT_CASES],
)
def test_each_element_stops_with_its_reason(
    case: str, build: Any, element: str, disposition: str, reason: str
) -> None:
    trial = build()
    compiled = trial.element(element)

    assert (compiled.disposition, compiled.reason) == (disposition, reason), (
        compiled.detail
    )
    assert compiled in trial.blocking and not trial.approvable
    # Every element still gets exactly one disposition.
    assert [item.element for item in trial.elements] == [
        *STATED_ELEMENTS,
        *HOST_ELEMENTS,
    ]


def test_an_inclusion_the_population_owner_does_not_apply_holds_approval() -> None:
    trial = _compile(
        population=_population(
            *_POPULATION,
            {
                "id": "c3",
                "quote": "an undefined state",
                "kind": "condition_present",
                "concepts_all_of": ["zzz_undefined_state"],
                "window": {"start_hours": 0, "end_hours": 6},
            },
        )
    )

    assert trial.blocking == () and trial.confounders_waiting == ()
    assert trial.population_blocking and not trial.approvable
    assert trial.record()["population_blocking"] is True


def test_a_horizon_inside_the_grace_period_is_refused(monkeypatch) -> None:
    short = FixedHorizonMortalityEndpoint(
        event_concept="mort_1d", followup_concept="followup_days_1d", horizon_days=1
    )
    monkeypatch.setattr(
        compile_module,
        "fixed_horizon_mortality_endpoint",
        lambda name: short if name == "mort_1d" else None,
    )
    monkeypatch.setattr(
        compile_module,
        "OUTCOME_CONCEPT_SUPPORTED_DATABASES",
        {"mort_1d": frozenset({"miiv"})},
    )

    outcome = _compile(_spec(outcome={"endpoint": "mort_1d"})).element("outcome")

    assert (outcome.disposition, outcome.reason) == (
        "not_applied",
        "tte_horizon_within_grace",
    )


def test_an_icu_exit_no_dictionary_defines_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        compile_module, "_extraction_defines", lambda concept, db: False
    )

    exit_ = _compile(_spec(), _ctx(without=("los_icu",))).element("icu_exit")

    assert (exit_.disposition, exit_.reason) == (
        "not_applied",
        "tte_icu_exit_undefined",
    )


def test_patient_identity_resamples_patients() -> None:
    context = _ctx(
        id_columns=("stay_id", "subject_id"),
        provenance={"patient_id_columns": ["subject_id"]},
        n_stays=120,
        n_patients=100,
    )

    unit = _compile(_spec(), context).element("resampling_unit")

    assert unit.disposition == "host_added"
    assert dict(unit.parameters) == {
        "unit": "patient",
        "group_source": "subject_id",
        "group_derivation": "identity",
        "delimiter": None,
    }


# -- confounders ----------------------------------------------------------------

_CONFOUNDER_CASES = [
    ("zzz_never_defined", "not_applied", "tte_confounder_unavailable"),
    ("vaso_ind_onset_time", "not_applied", "tte_confounder_is_design_concept"),
    ("mort_28d", "not_applied", "tte_confounder_is_design_concept"),
    ("los_icu", "not_applied", "tte_confounder_is_design_concept"),
    ("aki", "not_applied", "tte_confounder_after_time_zero"),
    ("crea", "requires_extraction", "tte_confounder_not_in_export"),
    ("lact_max", "requires_extraction", "tte_confounder_window_after_time_zero"),
]


@pytest.mark.parametrize(("name", "disposition", "reason"), _CONFOUNDER_CASES)
def test_each_confounder_is_carried_at_time_zero_or_listed(
    name: str, disposition: str, reason: str
) -> None:
    # Age is carried, so the weights adjust for something whatever the case.
    trial = _compile(_spec(confounders=[_confounder("age"), _confounder(name)]))
    _, confounder = trial.confounders

    assert (confounder.disposition, confounder.reason) == (disposition, reason)
    # A confounder waiting for data holds approval; one nothing carries is listed.
    assert trial.approvable is (disposition == "not_applied")
    (line,) = [item for item in trial.confirmations if item.kind == "confounder_set"]
    assert line.text.startswith("Adjusted for at time zero: age.")
    listed = "Not adjusted for: " if disposition == "not_applied" else "also: "
    assert f"{listed}{name}" in line.text
    # An extraction asks for a confounder it can carry and for no other; the
    # design's own columns are asked for as the design's.
    if reason != "tte_confounder_is_design_concept":
        columns = trial.record()["materialization"]["columns"]
        assert (name in columns) is (disposition == "requires_extraction")


def test_the_adjustment_set_and_its_assumption_are_confirmed_for_every_trial() -> None:
    trial = _compile(
        _spec(
            confounders=[
                _confounder("age"),
                _confounder("map_min", source="design_choice"),
                _confounder("crea"),
                _confounder("aki"),
            ]
        )
    )
    unadjusted = _compile(_spec(confounders=[]))

    def line(compiled: CompiledTargetTrial) -> tuple[str, str]:
        (item,) = [
            item for item in compiled.confirmations if item.kind == "confounder_set"
        ]
        return item.element, item.text

    assert line(trial) == (
        "confounders",
        "Adjusted for at time zero: age, map_min. After an extraction, also: crea. "
        "Not adjusted for: aki (not observed by time zero). Proposed by the setup, "
        "not stated by the study: map_min. The comparison assumes no confounding "
        "beyond the adjusted set.",
    )
    assert line(unadjusted) == (
        "confounders",
        "Adjusted for at time zero: nothing. The comparison assumes no "
        "confounding beyond the adjusted set.",
    )
    assert not unadjusted.approvable


def test_the_researcher_confirms_every_assumption_the_estimates_state() -> None:
    terms = TargetTrialClaimTerms.model_validate(
        {
            "schema_version": "easyicu.target_trial_claim_terms/1",
            "measure": "risk_difference",
            "strategy": None,
            "weighting": "stabilized",
            "initiate_label": "Early start",
            "defer_label": "No early start",
            "outcome_label": "Death",
            "horizon_days": 28,
            "analysis_unit_label": "ICU stays",
            "truncation_percentiles": None,
        }
    )
    stated = (
        terms.result_sentence(point="-4.00", low="-7.50", high="-0.50", confidence="95"),
        terms.conclusion_sentence("negative"),
    )
    # Every trial carries the line, whether or not it can be approved yet.
    for trial in (_compile(), _compile(_spec(confounders=[]))):
        (line,) = [
            item
            for item in trial.confirmations
            if item.kind == "emulation_assumptions"
        ]
        assert line.element == "analysis"
        for assumption in TARGET_TRIAL_ASSUMPTIONS:
            assert assumption in line.text
            assert all(assumption in sentence for sentence in stated)


def test_every_reason_code_is_reachable_from_a_synthetic_study() -> None:
    reached = {case[4] for case in _ELEMENT_CASES} | {
        "tte_horizon_within_grace",
        "tte_icu_exit_undefined",
    }

    assert reached == set(NOT_APPLIED_REASONS) | set(REQUIRES_EXTRACTION_REASONS)
    assert {reason for _, _, reason in _CONFOUNDER_CASES} == set(
        CONFOUNDER_NOT_APPLIED_REASONS
    ) | set(CONFOUNDER_REQUIRES_EXTRACTION_REASONS)


def test_every_reason_a_confounder_is_not_carried_has_a_card_label() -> None:
    assert set(compile_module._NOT_ADJUSTED_FOR) == set(CONFOUNDER_NOT_APPLIED_REASONS)


# -- databases outside what the owners cover -------------------------------------


@pytest.mark.parametrize("database", ["eicu", "sic", "hirid", "aumc"])
def test_a_database_without_a_capture_statement_is_refused(database: str) -> None:
    trial = _compile(_spec(), _ctx(database=database))
    treatment = trial.element("treatment")

    assert (treatment.disposition, treatment.reason) == (
        "not_applied",
        "tte_treatment_capture_undeclared",
    )
    assert not trial.approvable
    assert trial.capture_entries == ()


def test_eicu_is_refused_for_its_endpoint_and_its_death_time_as_well() -> None:
    # The context carries mort_28d, its follow-up and a death time, as an
    # export with structural placeholder columns does: the owners, not the
    # columns' presence, decide what the database supports.
    context = _ctx(database="eicu")
    assert {"mort_28d", "followup_days_28d", "death_time"} <= {
        variable.name for variable in context.variables
    }

    trial = _compile(_spec(), context)

    assert {item.element: item.reason for item in trial.blocking} == {
        "treatment": "tte_treatment_capture_undeclared",
        "outcome": "tte_endpoint_unsupported",
        "death_time": "tte_death_time_not_hourly",
    }


# -- the registry decides what an absent record means ----------------------------


@pytest.mark.parametrize("absent", ["absent_row_is_unmeasured", "unknown"])
def test_an_absence_not_read_as_no_treatment_stops_the_trial(absent: str) -> None:
    data = _packaged_data()
    data["entries"] = [
        {**entry, "absent_in_capture": absent} for entry in data["entries"]
    ]

    treatment = _compile(registry=_registry(data)).element("treatment")

    assert (treatment.disposition, treatment.reason) == (
        "not_applied",
        "tte_treatment_capture_undeclared",
    )


def test_a_development_assumption_is_confirmed_at_approval() -> None:
    trial = _compile()
    capture = [
        item for item in trial.confirmations if item.kind == "capture_assumption"
    ]

    assert [item.element for item in capture] == ["treatment", "treatment"]
    assert all("ICU" in item.text and "not visible" in item.text for item in capture)
    assert "pre_admission_use_not_visible" in dict(trial.limitations)

    data = _packaged_data()
    data["entries"] = [
        {**entry, "basis": "data_owner_attested"} for entry in data["entries"]
    ]
    attested = _compile(registry=_registry(data))
    assert not [
        item for item in attested.confirmations if item.kind == "capture_assumption"
    ]


def test_a_treatment_wider_than_its_class_is_confirmed_not_hidden() -> None:
    trial = _compile(_spec(treatment={"treatment_class": "vasopressor"}))
    lines = {
        item.kind: item.text
        for item in trial.confirmations
        if item.kind in {"treatment_class", "treatment_outside_class"}
    }
    agents = packaged_treatment_capture_registry().registry.class_agents("vasopressor")

    assert trial.element("treatment").disposition == "applied"
    assert lines == {
        # Which drugs a class counts is a development reading, confirmed with it.
        "treatment_class": f"The vasopressor class is read as "
        f"{', '.join(sorted(agents))}; EasyICU development set this composition.",
        "treatment_outside_class": "A stay that starts only dobutamine or "
        "milrinone counts as starting the treatment, though outside the "
        "vasopressor class.",
    }


def test_a_coordinate_the_study_did_not_state_is_confirmed() -> None:
    trial = _compile(
        _spec(
            time_zero={"source": "design_choice"},
            outcome={"source": "design_choice"},
        )
    )
    choices = [
        item.element for item in trial.confirmations if item.kind == "design_choice"
    ]

    assert choices == ["time_zero", "outcome"]


def test_a_non_drug_treatment_compiles_through_the_same_rules() -> None:
    data = _packaged_data()
    data["treatment_classes"]["kidney_replacement"] = {
        "agents": ["rrt"],
        "note": "Any renal replacement therapy.",
    }
    data["entries"].append(_entry("miiv", "rrt", ["rrt"]))
    registry = _registry(data)

    trial = _compile(
        _spec(treatment=_RENAL),
        _ctx(treatments=("rrt",)),
        registry=registry,
    )

    assert trial.element("treatment").disposition == "applied"
    assert trial.approvable
    assert trial.record()["materialization"]["treatment_onset_columns"] == [
        "rrt_onset_time"
    ]


def test_an_element_that_changes_a_strategy_is_confirmed_and_stops_the_trial() -> None:
    # Whether an element changes a strategy is the spec's own statement, so
    # the card shows every such element whichever way it is stated.
    trial = _compile(
        _spec(
            not_typed=[
                {
                    "quote": "kept on it for two days",
                    "source": "question",
                    "why": "a continued use no field states",
                    "affects_strategy": True,
                }
            ]
        )
    )

    assert [
        item.element for item in trial.confirmations if item.kind == "not_typed"
    ] == ["strategies"]
    assert trial.element("strategies").reason == "tte_strategy_not_typed"
    assert not trial.approvable
    assert "not_typed" not in dict(trial.limitations)


def test_an_element_no_field_states_is_kept_as_a_limitation() -> None:
    trial = _compile(
        _spec(
            not_typed=[
                {
                    "quote": "in a teaching hospital",
                    "source": "question",
                    "why": "a setting no field of the spec states",
                    "affects_strategy": False,
                }
            ]
        )
    )

    assert trial.element("strategies").disposition == "applied"
    assert any(
        code == "not_typed" and "teaching hospital" in text
        for code, text in trial.limitations
    )
    assert [
        (item.element, item.text)
        for item in trial.confirmations
        if item.kind == "not_typed"
    ] == [
        (
            "not_typed",
            "Not emulated: 'in a teaching hospital' (a setting no field of the "
            "spec states).",
        )
    ]
    assert dict(trial.limitations)["evidence_ceiling"].startswith(
        "The evidence ceiling is analysis_only"
    )


# -- the element record ----------------------------------------------------------


def test_an_element_keeps_its_disposition_consistent() -> None:
    with pytest.raises(ValueError, match="never host_added"):
        CompiledElement(
            element="treatment",
            disposition="host_added",
            reason=None,
            detail="",
            quote="a drug",
            source="question",
        )
    with pytest.raises(ValueError, match="never applied"):
        CompiledElement(
            element="death_time", disposition="applied", reason=None, detail=""
        )
    with pytest.raises(ValueError, match="does not fit"):
        CompiledElement(
            element="death_time",
            disposition="not_applied",
            reason="tte_confounder_unavailable",
            detail="",
        )
    with pytest.raises(ValueError, match="no stated words"):
        CompiledElement(
            element="icu_exit",
            disposition="host_added",
            reason=None,
            detail="",
            quote="ICU exit",
        )


# -- the spec ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"treatment": {"concepts": ["vaso_ind", "vaso_ind"]}},
        {"strategies": {"initiate_label": "Start", "defer_label": " start "}},
        {"grace_period": {"hours": 0}},
        {"time_zero": {"hours_after_icu_admission": 169}},
        {"treatment": {"treatment_class": "Vasoactive"}},
        {
            "confounders": [
                {
                    "name": "age",
                    "source": "question",
                    "clinical_rationale": "Age changes both, twice over.",
                },
                {
                    "name": "age",
                    "source": "question",
                    "clinical_rationale": "Age changes both, once again.",
                },
            ]
        },
        # A confounder says whether the study or its setup named it.
        {
            "confounders": [
                {"name": "age", "clinical_rationale": "Age changes both, twice over."}
            ]
        },
        {"outcome": {"endpoint": "mort_28d", "horizon": 28}},
    ],
)
def test_the_spec_refuses_what_it_cannot_state(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _spec(**changes)


def test_a_confounder_rationale_is_kept_in_single_spaces() -> None:
    spec = _spec(
        confounders=[
            {
                "name": "age",
                "source": "question",
                "clinical_rationale": "Older   patients\nare started later.",
            }
        ]
    )

    assert spec.confounders[0].clinical_rationale == (
        "Older patients are started later."
    )
