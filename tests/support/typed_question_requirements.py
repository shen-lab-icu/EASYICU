"""Shared typed requirement coordinates, synthetic plans, and shadow capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pytest

from easyicu.research_agent.contracts.model_terms import PlannedModelRequirement, preparation_method
from easyicu.research_agent.planning.requirement_typing import shadow
from easyicu.research_agent.planning.requirement_typing.contract import (
    AtomicPart,
    Binding,
    CoordinateRelation,
    Delivery,
    DictionarySnapshot,
    Facet,
    FacetName,
    FactAuthority,
    Grading,
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
    SourceSpan,
    Time,
    TimeZero,
    Unit,
    Window,
)
from easyicu.research_agent.planning.requirement_typing.registry import STUDY_POPULATION
from easyicu.research_agent.schema import (
    AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext,
)


ICU = "icu_admission"
HOSPITAL = "hospital_admission"
DICTIONARY = DictionarySnapshot(
    version="review-regression/1",
    concept_ids={
        "urine_albumin", "kdigo_creat", "kdigo_aki", "icu_readmission", "death",
        "lact", "sofa", "mg", "sep3", "af_onset", "rrt", "acute_rrt_input",
        "vent_end", "vent_ind", "vaso_ind", "crea", "blood_culture_negative", "k",
    },
    anchor_ids={ICU, HOSPITAL, "icu_discharge", "sep3", "af_onset"},
    population_ids={"study_population"},
    event_specializations=(CoordinateRelation(
        fact_concept_id="acute_rrt_input", request_concept_id="rrt",
    ),),
)
AUTHORITY = FactAuthority(
    dictionary_version=DICTIONARY.version, plan_sha256="1" * 64, context_sha256="2" * 64,
)


def bind(concept: str, role: Role = "exposure") -> Binding:
    return Binding(concept_id=concept, role=role)


def key(facet: Facet, *bindings: Binding, time: Time = NotApplicable()) -> RequirementKey:
    if isinstance(time, (Point, Interval)):
        bindings += (bind(time.anchor, "time_anchor"),)
    return RequirementKey(bindings=bindings, time=time, facet=facet)


def point(hours: int, *scope: Binding, anchor: str = ICU) -> RequirementKey:
    return key(TimeZero(), *scope, time=Point(anchor=anchor, unit="h", offset=hours))


def interval(
    start: int, end: int, anchor: str = ICU, *, unit="h",
    start_inclusive: bool = True, end_inclusive: bool = False,
) -> Interval:
    return Interval(
        anchor=anchor, unit=unit, start=start, end=end,
        start_inclusive=start_inclusive, end_inclusive=end_inclusive,
    )


def measurement(concept: str, start: int, end: int, anchor: str = ICU) -> RequirementKey:
    return key(Window(), bind(concept), time=interval(start, end, anchor))


def fact(typed_key: RequirementKey, authority: FactAuthority = AUTHORITY) -> HostFact:
    return HostFact(
        key=typed_key, provenance=Provenance(
            source_field="typed_fixture.applied_contract", authority=authority,
        ),
    )


@dataclass(frozen=True)
class Case:
    id: str
    quote: str
    facet: FacetName
    requested: RequirementKey | None
    rejected: tuple[RequirementKey, ...] = ()
    accepted: RequirementKey | None = None

    def atomic_part(self) -> AtomicPart:
        return AtomicPart(
            part_id=self.id, source_span=SourceSpan(start=0, end=len(self.quote)),
            facet_hint=self.facet, key=self.requested,
        )


ALBUMIN = key(Unit(code="mg/24h"), bind("urine_albumin"))
KDIGO_48 = key(
    Grading(contract_id="kdigo_aki_2012", version="2012"),
    bind("kdigo_creat"), time=interval(0, 48),
)
READMISSION_30 = key(
    Method(action_id="readmission_followup"), bind("icu_readmission", "outcome"),
    time=interval(0, 30, "icu_discharge", unit="d"),
)
EXTERNAL = key(
    Method(action_id="prediction.external_validation", parameters=(
        Parameter(name="validation_databases", value=("eicu",)),
        Parameter(name="recalibration", value="none"),
    )), bind("death", "outcome"), bind("lact", "covariate"),
)
IMPUTATION = key(
    Method(action_id="multiple_imputation", parameters=(Parameter(name="m", value=20),)),
    bind("lact", "covariate"),
)
ICU8 = point(8, STUDY_POPULATION)
ICU18 = point(18, STUDY_POPULATION)
MG11 = measurement("mg", 0, 11)
LAC12 = measurement("lact", 0, 12)
SEPSIS = point(0, bind("sep3"))
AF = point(0, bind("af_onset"))
AF_WINDOW = measurement("af_onset", -13, 0, "af_onset")
REPORT_STAYS = key(RepeatStayReporting(), STUDY_POPULATION)
FIRST_ONLY = key(RepeatStayPolicy(policy="first_icu_stay_per_patient"), STUDY_POPULATION)
CLUSTER = key(RepeatStayPolicy(policy="retain_each_stay_cluster_by_patient"), STUDY_POPULATION)
DIALYSIS12 = point(12, bind("rrt"))
DIALYSIS16 = point(16, bind("rrt"))
EXTUB18 = point(18, bind("vent_end"))
DEATH18 = point(18, bind("death", "outcome"))
EXTUB_DEATH18 = point(18, bind("vent_end"), bind("death", "outcome"))
MORTALITY_SEPSIS = point(0, bind("death", "outcome"), anchor="sep3")
VASOPRESSOR16 = point(16, bind("vaso_ind"))
CREAT24 = measurement("crea", 0, 24)
MAX_POTASSIUM = key(
    Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 6),
)
DELIVERY = key(Delivery(output_id="result_tables"), STUDY_POPULATION)


CASES = (
    Case("jdd/albumin-unit", "Albumin units in mg/24 hours", "unit", ALBUMIN,
         (key(Unit(code="mg/dL"), bind("urine_albumin")),)),
    Case("jdd/kdigo-48h", "KDIGO 48-hour creatinine staging", "grading", KDIGO_48, (
        key(Grading(contract_id="kdigo_aki_2012", version="2012"), bind("kdigo_creat"), time=interval(0, 168)),
        key(Grading(contract_id="kdigo_aki_2012", version="2012"), bind("kdigo_aki"), time=interval(0, 48)),
        key(Grading(contract_id="kdigo_aki_2012", version="2012"), bind("kdigo_creat"), time=interval(0, 48, HOSPITAL)),
    )),
    Case("jdd/30-day-readmission", "30-day ICU readmissions", "method", READMISSION_30, (
        key(Method(action_id="readmission_followup"), bind("icu_readmission", "outcome"), time=interval(0, 30)),
    )),
    Case("jdd/external-validation", "External validation", "method", EXTERNAL, (
        key(Method(action_id="prediction.internal_validation"), bind("death", "outcome"), bind("lact", "covariate")),
    )),
    Case("jdd/imputation", "Multiple imputation", "method", IMPUTATION, (
        key(Method(action_id="complete_case"), bind("lact", "covariate")),
        key(Method(action_id="multiple_imputation", parameters=(Parameter(name="m", value=5),)), bind("lact", "covariate")),
    )),
    Case("jdd/cohort-8h", "Cohort time zero at 8 hours", "time_zero", ICU8, (
        point(8, bind("lact")), point(8, bind("lact"), bind("sofa")), point(8, STUDY_POPULATION, anchor=HOSPITAL),
        point(-8, STUDY_POPULATION),
    )),
    Case("jdd/magnesium-ehr", "Specify magnesium within 11 hours from EHR records", "window", MG11,
         (measurement("mg", -11, 0), measurement("mg", 0, 11, HOSPITAL))),
    Case("jdd/magnesium-bedside", "Specify magnesium within 11 hours at bedside", "window", MG11,
         (measurement("lact", 0, 11),)),
    Case("jdd/magnesium-first", "Use the first 11 hours of serum magnesium", "window", MG11,
         (measurement("mg", 0, 12),)),
    Case("jdd/lactate-unspecified", "Specify the lactate window within 12 hours", "window", None, (LAC12,)),
    Case("jdd/lactate-confirmed", "Specify the lactate window within 12 hours", "window", LAC12,
         (measurement("lact", 0, 12, HOSPITAL),)),
    Case("jdd/sepsis-onset", "Document sepsis onset", "time_zero", SEPSIS,
         (point(0, bind("sep3", "outcome")),)),
    Case("jdd/af-onset", "Specify atrial fibrillation onset", "time_zero", AF, (SEPSIS,)),
    Case("jdd/af-negative-window", "Specify the AF window −13 to 0 h", "window", AF_WINDOW,
         (measurement("af_onset", 0, 13, "af_onset"), measurement("af_onset", -13, 0, ICU))),
    Case("jde/repeat-report", "Repeat-stay handling for included ICU admissions", "repeat_stay_reporting",
         REPORT_STAYS, (FIRST_ONLY, CLUSTER)),
    Case("jde/first-only", "Do not retain repeated stays; include only the first", "repeat_stay_policy",
         FIRST_ONLY, (CLUSTER,)),
    Case("jde/open-repeat-choice", "Specify whether to include only the first or cluster", "repeat_stay_policy",
         None, (FIRST_ONLY, CLUSTER)),
    Case("jde/cluster-repeats", "Retain repeated ICU stays and cluster by patient", "repeat_stay_policy",
         CLUSTER, (FIRST_ONLY,)),
    Case("jde/dialysis-en", "Define dialysis time zero at 12 hours", "time_zero",
         DIALYSIS12, (point(12, STUDY_POPULATION),)),
    Case("jdf/dialysis-zh", "定义透析时间零点为12小时", "time_zero",
         DIALYSIS12, (point(12, STUDY_POPULATION),)),
    Case("jdf/dialysis-unbound", "定义透析时间零点为12小时", "time_zero", None, (DIALYSIS12,)),
    Case("jdf/extubation-no-label", "Document extubation time zero at 18 hours", "time_zero",
         EXTUB18, (ICU18, point(18, bind("vent_ind")))),
    Case("jde/unsigned-window", "Specify the measurement window in 24 hours", "window", None, (
        measurement("death", 0, 24), measurement("lact", 0, 24),
    )),
    Case("jdf/unsigned-window-zh", "明确测量窗为30小时", "window", None, (
        measurement("blood_culture_negative", 0, 30), measurement("lact", 0, 30),
    )),
    Case("jdg/window-with-sibling", "Specify the measurement window in 24 hours and model lactate",
         "window", None, (measurement("lact", 0, 24),)),
    Case("jdg/creatinine-24h", "Specify in 24 hours the sampling window for creatinine", "window", CREAT24, (
        measurement("death", 0, 24), measurement("lact", 0, 24),
    )),
    Case("jdg/lactate-and-mortality", "Compare lactate and mortality in ICU", "method",
         key(Method(action_id="compare"), bind("lact"), bind("death", "outcome")),
         (key(Method(action_id="compare"), bind("lact")),
          key(Method(action_id="compare"), bind("death", "outcome")))),
    Case("jde/mortality-sepsis", "Specify mortality follow-up at sepsis onset", "time_zero", MORTALITY_SEPSIS, (
        point(0, bind("sep3")), point(0, bind("death", "outcome")),
    )),
    Case("jdf/cohort-18h", "Cohort time zero 18 hours after ICU admission", "time_zero",
         ICU18, (DEATH18, EXTUB18)),
    Case("jdf/mortality-18h", "Cohort time zero at 18 hours for mortality", "time_zero",
         DEATH18, (ICU18, EXTUB18)),
    Case("jdf/extubation-18h", "Document extubation 18 hours after ICU admission", "time_zero",
         EXTUB18, (ICU18, point(18, bind("vent_ind")))),
    Case("jdg/extubation-for-mortality", "Document extubation for mortality time zero at 18 hours",
         "time_zero", EXTUB_DEATH18, (EXTUB18, DEATH18, ICU18)),
    Case("jdg/extubation-unresolved", "Document extubation for mortality time zero at 18 hours",
         "time_zero", None, (EXTUB18, DEATH18)),
    Case("jdg/lactate-dialysis-sibling", "Compare lactate and dialysis; define dialysis time zero at 16 hours",
         "time_zero", DIALYSIS16, (point(16, bind("lact")),)),
    Case("jdg/the-dialysis-time-zero", "Specify the dialysis time zero at 16 hours",
         "time_zero", DIALYSIS16, (VASOPRESSOR16,), accepted=point(16, bind("acute_rrt_input"))),
    Case("jdg/vasopressor-16h", "Clarify vasopressor initiation time zero at 16 hours", "time_zero",
         VASOPRESSOR16, (DIALYSIS16, point(16, STUDY_POPULATION))),
    Case("jdg/vasopressor-conflicting-proposal", "Clarify vasopressor initiation time zero at 16 hours",
         "time_zero", None, (DIALYSIS16,)),
    Case("new/peak-potassium-en", "Use peak potassium in the first six hours after ICU entry", "window",
         MAX_POTASSIUM, (
             key(Window(purpose="aggregation", operation="mean"), bind("k"), time=interval(0, 6)),
             key(Window(purpose="aggregation", operation="max"), bind("k", "covariate"), time=interval(0, 6)),
             key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 6, HOSPITAL)),
             key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 8)),
         )),
    Case("new/peak-potassium-zh", "入ICU后前六小时取最高血钾", "window", MAX_POTASSIUM, (
        key(Window(purpose="aggregation", operation="mean"), bind("k"), time=interval(0, 6)),
        key(Window(purpose="aggregation", operation="max"), bind("k", "outcome"), time=interval(0, 6)),
        key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(-6, 0)),
    )),
)


def _context(*, preferences=None, first=False, variables=None) -> ResearchContext:
    provenance = {}
    if first:
        provenance["first_icu_stay_restriction"] = {
            "schema_version": "easyicu.first_icu_stay_restriction/1", "coordinate_sha256": "b" * 64,
        }
    if preferences and preferences.data_constraints:
        provenance["patient_id_columns"] = ["patient_id"]
    return ResearchContext(
        research_question="Use peak potassium in the first six hours after ICU entry.",
        cohort=CohortDescriptor(
            cohort_name="test_cohort", database="miiv", n_stays=4,
            time_columns=["clock"], id_columns=["patient_id", "stay_id"], provenance=provenance,
        ),
        primary_exposure="k", target_outcome="death",
        variables=variables if variables is not None else [
            ConceptDescriptor(name="k", dtype="float64", unit="mmol/L"),
            ConceptDescriptor(name="death", dtype="bool"),
        ],
        user_preferences=preferences,
    )


def _model(*, exposure="k", dependence=None) -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="m1", outcome="death", outcome_type="binary",
        method_family="logistic_regression", exposure_source=exposure,
        analysis_role="primary", analysis_set="complete_case", covariates=[],
        dependence=dependence,
    )


def _model_plan(preparation=None, *, model=True) -> AnalysisPlan:
    method = "adjusted_association_models"
    return AnalysisPlan(
        research_question="Use peak potassium in the first six hours after ICU entry.",
        steps=[AnalysisStep(
            step_id="model", intent="fit a model", inputs=["k", "death", "clock", "stay_id"],
            method=preparation_method(method, preparation) if preparation is not None else method,
            expected_outputs=["table:adjusted_association_estimates"],
            planned_analysis_role="primary", model_requirements=[_model()] if model else [],
        )],
    )


def proposal(part_id, quote, asked, *, facet=None, alternatives=(), start=0):
    return {
        "part_id": part_id, "source_span": {"start": start, "end": start + len(quote)},
        "facet_hint": facet or (asked.facet.kind if asked else None),
        "key": asked.model_dump(mode="json") if asked else None,
        "alternatives": [choice.model_dump(mode="json") for choice in alternatives],
    }


@pytest.fixture
def shadow_events(monkeypatch):
    comparisons = []
    evaluate = shadow._observe

    def capture(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        comparisons.append(result)
        return result

    monkeypatch.setattr(shadow, "_observe", capture)
    return comparisons


def _shadow_event(shadow_events):
    [comparison] = shadow_events
    return {"parts": [{
        "v1": row.v1,
        "v1_reason": row.v1_reason,
        "v2": row.decision.state,
        "v2_reason": getattr(row.decision, "reason", None),
        "source_span": row.result.part.source_span.model_dump() if row.result else None,
        "result": row.result.model_dump(mode="json") if row.result else None,
        "repair": [asdict(issue) for issue in row.repair],
    } for row in comparison.parts]}
