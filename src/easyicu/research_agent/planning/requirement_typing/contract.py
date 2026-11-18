"""Wire contract for atomic requirements and their plan-bound evidence."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_serializer,
    model_validator,
)

from ...contracts.product_files import DeliverableKind

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Role = Literal["exposure", "outcome", "covariate", "population", "time_anchor"]
TimeUnit = Literal["s", "min", "h", "d"]
Aggregation = Literal["first", "last", "min", "max", "mean", "median", "any", "all", "count", "sum"]
RepeatPolicy = Literal["first_icu_stay_per_patient", "retain_each_stay_cluster_by_patient"]
FacetName = Literal[
    "time_zero", "window", "grading", "unit", "method", "cohort_predicate",
    "repeat_stay_reporting", "repeat_stay_policy", "delivery",
]
PartOrigin = Literal["study_choice", "plan_contract", "model_proposal_host_validated"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceSpan(_Record):
    """Half-open Unicode character offsets in the source question."""

    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def _nonempty(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("a source span must have positive length")
        return self


class Binding(_Record):
    concept_id: Identifier
    role: Role


class NotApplicable(_Record):
    kind: Literal["not_applicable"] = "not_applicable"


class Point(_Record):
    kind: Literal["point"] = "point"
    anchor: Identifier
    unit: TimeUnit
    offset: Decimal


class Interval(_Record):
    kind: Literal["interval"] = "interval"
    anchor: Identifier
    unit: TimeUnit
    start: Decimal | Literal["-inf"]
    end: Decimal | Literal["inf"]
    start_inclusive: StrictBool
    end_inclusive: StrictBool

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if (
            isinstance(self.start, Decimal) and isinstance(self.end, Decimal)
            and self.start >= self.end
        ):
            raise ValueError("an interval must have positive length")
        if (self.start == "-inf" and self.start_inclusive) or (
            self.end == "inf" and self.end_inclusive
        ):
            raise ValueError("an infinite interval endpoint is open")
        return self


Time = Annotated[NotApplicable | Point | Interval, Field(discriminator="kind")]


class TimeZero(_Record):
    kind: Literal["time_zero"] = "time_zero"
    purpose: Literal["analysis_origin", "sensitivity_landmark"] = "analysis_origin"


class Window(_Record):
    kind: Literal["window"] = "window"
    purpose: Literal["measurement", "observation", "aggregation"] = "measurement"
    operation: Aggregation | None = None

    @model_validator(mode="after")
    def _operation_matches_purpose(self) -> Window:
        if (self.operation is not None) != (self.purpose == "aggregation"):
            raise ValueError("an aggregation window names its operation")
        return self


class Grading(_Record):
    kind: Literal["grading"] = "grading"
    contract_id: Identifier
    version: Identifier


class Unit(_Record):
    kind: Literal["unit"] = "unit"
    code: Annotated[str, Field(pattern=r"^[A-Za-z0-9%][A-Za-z0-9%./_\[\]{}*^-]{0,63}$")]


Scalar = StrictBool | StrictInt | StrictFloat | StrictStr | None
ParameterValue = Scalar | tuple[Scalar, ...]


class Parameter(_Record):
    name: Identifier
    value: ParameterValue


class Method(_Record):
    kind: Literal["method"] = "method"
    action_id: Identifier
    parameters: tuple[Parameter, ...] = ()

    @model_validator(mode="after")
    def _distinct_parameter_names(self) -> Method:
        if len({parameter.name for parameter in self.parameters}) != len(self.parameters):
            raise ValueError("method parameter names must be distinct")
        return self


class CohortPredicate(_Record):
    kind: Literal["cohort_predicate"] = "cohort_predicate"
    operation: Aggregation
    operator: Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "missing", "not_missing"]
    value: ParameterValue
    polarity: Literal["include", "exclude"]
    missing_policy: Literal["exclude", "retain", "stop"] | None = None


class RepeatStayReporting(_Record):
    kind: Literal["repeat_stay_reporting"] = "repeat_stay_reporting"


class RepeatStayPolicy(_Record):
    kind: Literal["repeat_stay_policy"] = "repeat_stay_policy"
    policy: RepeatPolicy


class Delivery(_Record):
    kind: Literal["delivery"] = "delivery"
    output_id: DeliverableKind


Facet = Annotated[
    TimeZero | Window | Grading | Unit | Method | CohortPredicate
    | RepeatStayReporting | RepeatStayPolicy | Delivery,
    Field(discriminator="kind"),
]


class RequirementKey(_Record):
    bindings: tuple[Binding, ...] = Field(min_length=1)
    time: Time
    facet: Facet

    @model_validator(mode="after")
    def _closed_coordinates(self) -> RequirementKey:
        if len({(item.role, item.concept_id) for item in self.bindings}) != len(self.bindings):
            raise ValueError("bindings must be distinct")
        anchors = tuple(item.concept_id for item in self.bindings if item.role == "time_anchor")
        scope = tuple(item for item in self.bindings if item.role != "time_anchor")
        expected = () if isinstance(self.time, NotApplicable) else (self.time.anchor,)
        if anchors != expected or not scope or sum(item.role == "population" for item in scope) > 1:
            raise ValueError("a key needs a scope and its exact temporal anchor binding, if applicable")
        if isinstance(self.facet, TimeZero) and not isinstance(self.time, Point):
            raise ValueError("time zero needs an anchored point")
        if isinstance(self.facet, (Window, CohortPredicate)) and not isinstance(self.time, Interval):
            raise ValueError("a window or cohort predicate needs an anchored interval")
        if isinstance(self.facet, (Delivery, RepeatStayReporting, RepeatStayPolicy)) and (
            len(scope) != 1 or scope[0].role != "population"
            or not isinstance(self.time, NotApplicable)
        ):
            raise ValueError("delivery and repeat-stay policies apply to an explicit study population")
        return self


class CoordinateRelation(_Record):
    """A dictionary-declared event specialization, directed from fact to request."""

    fact_concept_id: Identifier
    request_concept_id: Identifier


class DictionarySnapshot(_Record):
    version: Identifier
    concept_ids: frozenset[Identifier]
    anchor_ids: frozenset[Identifier]
    population_ids: frozenset[Identifier]
    event_specializations: tuple[CoordinateRelation, ...] = ()

    @field_serializer("concept_ids", "anchor_ids", "population_ids")
    def _ordered_ids(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def _known_relatives(self) -> DictionarySnapshot:
        if any(
            edge.fact_concept_id not in self.concept_ids
            or edge.request_concept_id not in self.concept_ids
            or edge.fact_concept_id == edge.request_concept_id
            for edge in self.event_specializations
        ):
            raise ValueError("dictionary relations must connect distinct registered concepts")
        return self


class FactAuthority(_Record):
    plan_sha256: Digest
    context_sha256: Digest | None
    dictionary_version: Identifier
    coordinate_sha256: Digest | None = None
    registry_sha256: Digest | None = None


class Provenance(_Record):
    source_field: str = Field(pattern=r"^[a-zA-Z0-9_.:/\[\]-]{1,512}$")
    authority: FactAuthority


class HostFact(_Record):
    key: RequirementKey
    provenance: Provenance


class VerifiedGap(_Record):
    key: RequirementKey
    provenance: Provenance
    reason: Literal["concept_unavailable", "definition_unavailable", "method_unavailable", "delivery_unavailable"]


class Covered(_Record):
    state: Literal["covered"] = "covered"
    next_action: Literal["none"] = "none"
    provenance: Provenance


class NotCovered(_Record):
    state: Literal["not_covered"] = "not_covered"
    next_action: Literal["edit_plan"] = "edit_plan"
    reason: Literal["no_matching_fact"] = "no_matching_fact"


class Unknown(_Record):
    state: Literal["unknown"] = "unknown"
    next_action: Literal["confirm_requirements"] = "confirm_requirements"
    reason: Literal[
        "missing_typed_coordinates", "unregistered_coordinate",
        "dictionary_snapshot_mismatch", "conflicting_evidence", "source_span_unavailable",
        "plan_contract_unavailable",
        "invalid_typing_proposal", "unresolved_alternatives", "unbound_source_coordinates",
    ] = "missing_typed_coordinates"


class CapabilityGap(_Record):
    state: Literal["capability_gap"] = "capability_gap"
    next_action: Literal["verified_alternative_or_question_revision"] = "verified_alternative_or_question_revision"
    gap: VerifiedGap


Decision = Annotated[Covered | NotCovered | Unknown | CapabilityGap, Field(discriminator="state")]


class AtomicPart(_Record):
    part_id: Identifier
    source_span: SourceSpan
    key: RequirementKey | None
    facet_hint: FacetName | None = None
    origin: PartOrigin | None = None
    alternatives: tuple[RequirementKey, ...] = ()

    @model_validator(mode="after")
    def _facet_agrees(self) -> AtomicPart:
        facets = {key.facet.kind for key in self.alternatives}
        if self.key is not None:
            facets.add(self.key.facet.kind)
        if self.facet_hint is not None:
            facets.add(self.facet_hint)
        if len(facets) > 1:
            raise ValueError("the key, facet hint and alternatives must agree on the facet")
        return self


class PartResult(_Record):
    part: AtomicPart
    decision: Decision

    @model_validator(mode="after")
    def _typed_decision_has_key(self) -> PartResult:
        if self.part.key is None and self.decision.state != "unknown":
            raise ValueError("an untyped part has an unknown decision")
        return self


class CoverageRecord(_Record):
    schema_version: Literal["easyicu.question_requirement_coverage/2"] = "easyicu.question_requirement_coverage/2"
    authority: FactAuthority
    dictionary: DictionarySnapshot
    parts: tuple[PartResult, ...]

    @model_validator(mode="after")
    def _unique_parts_and_bound_receipts(self) -> CoverageRecord:
        if len({row.part.part_id for row in self.parts}) != len(self.parts):
            raise ValueError("atomic part IDs must be unique")
        if self.authority.dictionary_version != self.dictionary.version:
            raise ValueError("the coverage record must bind its dictionary snapshot")
        for row in self.parts:
            decision = row.decision
            if isinstance(decision, Covered) and decision.provenance.authority != self.authority:
                raise ValueError("the coverage decision must bind to the record's authority")
            if isinstance(decision, CapabilityGap) and (
                decision.gap.provenance.authority != self.authority
                or decision.gap.key != row.part.key
            ):
                raise ValueError("the capability gap must bind to this authority and key")
        return self

    @property
    def complete(self) -> bool:
        return all(row.decision.state == "covered" for row in self.parts)
