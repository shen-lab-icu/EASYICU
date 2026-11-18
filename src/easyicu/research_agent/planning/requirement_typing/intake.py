"""Validate experimental question typings beside the legacy requirement owner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, ValidationError, model_validator

from .contract import (
    AtomicPart,
    CoverageRecord,
    Decision,
    Delivery,
    DictionarySnapshot,
    FacetName,
    Identifier,
    Interval,
    PartOrigin,
    PartResult,
    Point,
    RequirementKey,
    SourceSpan,
    TimeZero,
    Unknown,
    _Record,
)
from .facts import ProjectedFacts
from .matcher import match
from .registry import contract_registry_sha256, registered_unit_codes, unregistered_fields
from .source import has_source_alternatives, source_supports

if TYPE_CHECKING:
    from ...schema import ResearchContext

PROPOSALS_FIELD = "question_requirement_typings"
_MAX_PROPOSALS = 24
_MAX_WIRE_BYTES = 24_576
_MAX_REPAIR_FIELDS = 8


class TypingProposal(_Record):
    part_id: Identifier
    source_span: SourceSpan
    facet_hint: FacetName | None
    key: RequirementKey | None
    alternatives: Annotated[tuple[RequirementKey, ...], Field(max_length=3)]

    @model_validator(mode="after")
    def _closed_proposal(self) -> TypingProposal:
        if self.key is not None and self.alternatives:
            raise ValueError("put unresolved choices only in alternatives")
        keys = (self.key,) if self.key is not None else self.alternatives
        facets = {key.facet.kind for key in keys}
        if self.facet_hint is not None:
            facets.add(self.facet_hint)
        if len(facets) > 1:
            raise ValueError("facet_hint, key and alternatives must identify one facet")
        for key in keys:
            time = key.time
            values = (time.offset,) if isinstance(time, Point) else (
                (time.start, time.end) if isinstance(time, Interval) else ()
            )
            if any(isinstance(value, Decimal) and (
                len(value.as_tuple().digits) > 32 or abs(value.as_tuple().exponent) > 32
            ) for value in values):
                raise ValueError("time coordinates exceed the transport precision")
        return self


def typing_proposals_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    item = TypingProposal.model_json_schema()
    definitions = item.pop("$defs")

    def provider_unions(node: Any) -> None:
        if isinstance(node, dict):
            if node.pop("discriminator", None) is not None:
                node["anyOf"] = node.pop("oneOf")
            for child in node.values():
                provider_unions(child)
        elif isinstance(node, list):
            for child in node:
                provider_unions(child)

    provider_unions(item)
    provider_unions(definitions)
    return {"type": "array", "items": item, "maxItems": _MAX_PROPOSALS}, definitions


def typing_proposals_instruction(*, include_schema: bool) -> str:
    instruction = (
        f'- "{PROPOSALS_FIELD}": experiment for typing the request independently of plan choices. '
        'Return [] when there are none. Use the question’s zero-based Unicode character offsets '
        '(start inclusive, end exclusive) for the complete atomic clause. Use the requirement ID '
        'or, for a multi-clause requirement, <id>/clause/<one-based index>. For several delivery outputs '
        'in one clause use <id>/delivery/<output ID>. Set key to null and '
        'list explicit unresolved alternatives, or use an empty alternatives array for a single key. '
        'Use registered IDs and explicit roles, time anchors, units and interval boundaries. '
        'Abstain when a coordinate is unspecified. Never infer a request from notes, intent or a plan fact.'
    )
    if not include_schema:
        return instruction
    schema, definitions = typing_proposals_schema()
    return instruction + ' Use this shape: ' + json.dumps({**schema, "$defs": definitions}, separators=(",", ":"))


@dataclass(frozen=True)
class RepairField:
    path: str
    code: str


@dataclass(frozen=True)
class _Submission:
    index: int
    value: TypingProposal | None
    repair: tuple[RepairField, ...]

    def field(self, field: str, code: str) -> RepairField:
        suffix = f".{field}" if field else ""
        return RepairField(f"{PROPOSALS_FIELD}[{self.index}]{suffix}", code)


def _schema_repairs(index: int, error: ValidationError) -> tuple[RepairField, ...]:
    result = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False)[:_MAX_REPAIR_FIELDS]:
        location = issue["loc"]
        if issue["type"] == "extra_forbidden":
            location = location[:-1]  # Do not log an untrusted field name.
        path = "".join(f"[{step}]" if isinstance(step, int) else f".{step}" for step in location)
        result.append(RepairField(f"{PROPOSALS_FIELD}[{index}]{path}", issue["type"]))
    return tuple(dict.fromkeys(result))


class TypingIntake:
    """One question, one current projection, and one bounded proposal batch."""

    def __init__(
        self, *, question: str, dictionary: DictionarySnapshot, projected: ProjectedFacts,
        context: ResearchContext | None, proposals: object, part_ids: frozenset[str],
    ) -> None:
        self.question = question
        self.dictionary = dictionary
        self.projected = projected
        self.context = context
        self.units = registered_unit_codes(context.cohort.database if context else None)
        self.submissions: dict[str, _Submission] = {}
        self.repair: list[RepairField] = []
        self.invalid_batch = False
        if proposals != ():
            self._read_batch(proposals, part_ids)

    def _read_batch(self, proposals: object, part_ids: frozenset[str]) -> None:
        try:
            within_budget = len(json.dumps(proposals, ensure_ascii=True)) <= _MAX_WIRE_BYTES
        except (TypeError, ValueError, RecursionError):
            within_budget = False
        if not isinstance(proposals, list) or len(proposals) > _MAX_PROPOSALS or not within_budget:
            self.invalid_batch = True
            self.repair.append(RepairField(PROPOSALS_FIELD, "invalid_proposal_batch"))
            return
        for index, raw in enumerate(proposals):
            part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
            entry = _Submission(index, None, ())
            if not isinstance(part_id, str) or part_id not in part_ids:
                self.repair.append(entry.field("part_id", "unknown_part"))
                continue
            if part_id in self.submissions:
                entry = _Submission(index, None, (entry.field("part_id", "duplicate_part"),))
            else:
                try:
                    entry = _Submission(index, TypingProposal.model_validate(raw), ())
                except ValidationError as error:
                    entry = _Submission(index, None, _schema_repairs(index, error))
            self.submissions[part_id] = entry

    def _source_supports(self, source: str, key: RequirementKey) -> bool:
        return source_supports(source, key, context=self.context, dictionary=self.dictionary)

    def _decide(self, key: RequirementKey) -> Decision:
        return match(
            key, authority=self.projected.authority, dictionary=self.dictionary,
            facts=self.projected.facts,
        )

    def _registry_issues(self, key: RequirementKey) -> tuple[str, ...]:
        return unregistered_fields(key, self.dictionary, units=self.units)

    def _seed(self, part: AtomicPart, source: str, output_id: str | None) -> PartResult:
        candidates: dict[RequirementKey, PartOrigin] = {}
        for facts, origin in (
            (self.projected.facts, "plan_contract"), (self.projected.study_choices, "study_choice"),
        ):
            for fact in facts:
                if (
                    fact.provenance.authority == self.projected.authority
                    and part.facet_hint in (None, fact.key.facet.kind)
                    and (output_id is None or isinstance(fact.key.facet, Delivery) and fact.key.facet.output_id == output_id)
                    and not self._registry_issues(fact.key)
                    and self._source_supports(source, fact.key)
                ):
                    candidates[fact.key] = origin
        if len(candidates) > 1:
            alternatives = tuple(candidates) if len({key.facet.kind for key in candidates}) == 1 else ()
            return PartResult(
                part=part.model_copy(update={"alternatives": alternatives}),
                decision=Unknown(reason="conflicting_evidence"),
            )
        if candidates:
            [(key, origin)] = candidates.items()
            return PartResult(
                part=part.model_copy(update={"key": key, "origin": origin}),
                decision=self._decide(key),
            )
        return PartResult(part=part, decision=Unknown())

    def judge(self, part: AtomicPart, *, output_id: str | None = None) -> tuple[PartResult, tuple[RepairField, ...]]:
        if part.source_span.end > len(self.question):
            return PartResult(part=part, decision=Unknown(reason="source_span_unavailable")), ()
        authority = self.projected.authority
        if authority.dictionary_version != self.dictionary.version or authority.registry_sha256 != contract_registry_sha256():
            return PartResult(part=part, decision=Unknown(reason="dictionary_snapshot_mismatch")), ()
        if self.invalid_batch:
            return PartResult(part=part, decision=Unknown(reason="invalid_typing_proposal")), ()
        source = self.question[part.source_span.start:part.source_span.end]
        seeded = self._seed(part, source, output_id)
        submission = self.submissions.get(part.part_id)
        if submission is None:
            return seeded, ()
        proposal = submission.value
        if proposal is None:
            return PartResult(part=part, decision=Unknown(reason="invalid_typing_proposal")), submission.repair
        if proposal.source_span != part.source_span:
            issue = submission.field("source_span", "span_does_not_bind_part")
            return PartResult(part=part, decision=Unknown(reason="invalid_typing_proposal")), (issue,)
        keys = (("key", proposal.key),) if proposal.key else tuple(
            (f"alternatives[{index}]", key) for index, key in enumerate(proposal.alternatives)
        )
        hint = proposal.facet_hint or (keys[0][1].facet.kind if keys else part.facet_hint)
        if part.facet_hint not in (None, hint):
            issue = submission.field("facet_hint", "facet_does_not_bind_part")
            return PartResult(part=part, decision=Unknown(reason="invalid_typing_proposal")), (issue,)
        for prefix, key in keys:
            if output_id is not None and (not isinstance(key.facet, Delivery) or key.facet.output_id != output_id):
                issue = submission.field(f"{prefix}.facet.output_id", "output_does_not_bind_part")
                return PartResult(part=part, decision=Unknown(reason="invalid_typing_proposal")), (issue,)
        part = part.model_copy(update={"facet_hint": hint})
        issues = tuple(
            submission.field(f"{prefix}.{field}", "unregistered_coordinate")
            for prefix, key in keys for field in self._registry_issues(key)
        )[:_MAX_REPAIR_FIELDS]
        if issues:
            return PartResult(part=part, decision=Unknown(reason="unregistered_coordinate")), issues
        if proposal.alternatives or has_source_alternatives(source):
            if seeded.part.key is not None:
                return seeded, (submission.field("alternatives", "coordinates_need_source_binding"),)
            return PartResult(
                part=part.model_copy(update={"alternatives": proposal.alternatives}),
                decision=Unknown(reason="unresolved_alternatives"),
            ), ()
        key = proposal.key
        if key is None:
            return seeded, ()
        decision = self._decide(key)
        differs_from_seed = seeded.part.key not in (None, key)
        # A time origin is an event definition. Keep its coordinates unknown
        # when neither the question nor the study binds the event and clock.
        if (
            (differs_from_seed and decision.state != "covered")
            or ((decision.state == "covered" or isinstance(key.facet, TimeZero)) and not self._source_supports(source, key))
        ):
            issue = submission.field("key", "coordinates_need_source_binding")
            if seeded.part.key is not None:
                return seeded, (issue,)
            return PartResult(part=part, decision=Unknown(reason="unbound_source_coordinates")), (issue,)
        return PartResult(
            part=part.model_copy(update={"key": key, "origin": "model_proposal_host_validated"}),
            decision=decision,
        ), ()

    def record(self, parts: tuple[PartResult, ...]) -> CoverageRecord:
        return CoverageRecord(authority=self.projected.authority, dictionary=self.dictionary, parts=parts)
