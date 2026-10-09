"""How the Planner is asked for its population spec, with and without a schema.

Owner
-----
Step 2a of the population spec design asks the progressive Planner's
foundation for the population twice in one call: as the cohort intent it
already writes, and as a typed :class:`.population_spec.PopulationSpec`
(``cohort.population_spec``).  This module owns both forms of that request.

* A Provider with strict JSON schema receives the spec owner's own schema,
  made strict (:func:`bind_population_spec_transport`): the kinds become
  closed alternatives, each concept is one of the run's allowed cohort
  concepts, and a kind that states the stays kept can only include.  An
  optional spec is a required, nullable property, as every optional field of
  a strict schema is; a caller-bound cohort of every input row states none.
* A Provider without one never receives that schema.  It reads the same
  shape, kinds and rules in the foundation contract's text
  (:func:`population_spec_shape`, :func:`population_spec_contract`).

The spec is a shadow in this step: the host compiles it beside the cohort
intent and records how the two compare (``planning.population_shadow``),
while the intent's predicates still select the rows.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Sequence

from ..planning.population_spec import CRITERION_MODELS, KEPT_KINDS, PopulationSpec

__all__ = [
    "PopulationSpecTransportError",
    "bind_population_spec_transport",
    "population_spec_contract",
    "population_spec_shape",
]

_SPEC = "PopulationSpec"
#: Criterion fields that name a study concept, and the fields that hold one.
_CONCEPT_FIELDS = ("concept", "concepts_all_of")


class PopulationSpecTransportError(ValueError):
    """The spec's schema cannot be bound into the foundation transport."""


def _concept_enum(concept_ids: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(concept_ids)}


def bind_population_spec_transport(
    definitions: dict[str, Any],
    cohort_properties: dict[str, Any],
    *,
    concept_ids: Sequence[str],
    stated: bool,
) -> None:
    """Bind ``cohort.population_spec`` into one foundation transport schema.

    ``stated`` is false for a caller-bound cohort of every input row, which
    states no population.  With no allowed concept, the kinds that name one
    are not offered.
    """

    if not stated:
        cohort_properties["population_spec"] = {"type": "null"}
        return
    schema = copy.deepcopy(PopulationSpec.model_json_schema(mode="validation"))
    spec_definitions = schema.pop("$defs", None)
    if not isinstance(spec_definitions, dict):
        raise PopulationSpecTransportError("the population spec schema has no $defs")
    clashes = sorted(set(definitions) & {_SPEC, *spec_definitions})
    if clashes:
        raise PopulationSpecTransportError(
            f"population spec definitions clash with the foundation's: {clashes}"
        )
    offered = []
    for kind, model in CRITERION_MODELS.items():
        definition = spec_definitions.get(model.__name__)
        properties = (
            definition.get("properties") if isinstance(definition, dict) else None
        )
        if not isinstance(properties, dict) or "role" not in properties:
            raise PopulationSpecTransportError(
                f"population criterion kind {kind!r} has no closed definition"
            )
        named = [field for field in _CONCEPT_FIELDS if field in properties]
        if named and not concept_ids:
            spec_definitions.pop(model.__name__)
            continue
        for field in named:
            if properties[field].get("type") == "array":
                properties[field]["items"] = _concept_enum(concept_ids)
            else:
                properties[field] = _concept_enum(concept_ids)
        if kind in KEPT_KINDS:
            properties["role"] = {"type": "string", "const": "include"}
        offered.append({"$ref": f"#/$defs/{model.__name__}"})
    # One closed alternative per kind: the discriminator mapping and oneOf are
    # not part of the strict subset, and each kind's const already decides.
    schema["properties"]["criteria"]["items"] = {"anyOf": offered}
    definitions.update(spec_definitions)
    definitions[_SPEC] = schema
    cohort_properties["population_spec"] = {
        "anyOf": [{"$ref": f"#/$defs/{_SPEC}"}, {"type": "null"}]
    }


_WINDOW = {"start_hours": "<number>", "end_hours": "<greater number>"}
_CONCEPT = "<copy an allowed cohort concept id>"
_BOUND = "<number or null>"
#: Each kind's own fields, as the contract text shows them.
_KIND_FIELDS: dict[str, dict[str, Any]] = {
    "age_years": {"min_years": _BOUND, "max_years": _BOUND},
    "first_icu_stay": {},
    "icu_stay_hours": {"min_hours": _BOUND, "max_hours": _BOUND},
    "condition_present": {"concepts_all_of": [_CONCEPT], "window": _WINDOW},
    "diagnosis_codes": {"system": "<icd9|icd10>", "codes": ["<code or code prefix>"]},
    "measurement": {
        "concept": _CONCEPT,
        "summary": "<max|min|first|last|mean>",
        "window": _WINDOW,
        "op": "<>|>=|<|<=|==|!=>",
        "value": "<number>",
        "unit": "<the stated unit or null>",
    },
    "alive_at": {"hours": "<number>"},
    "event_absent": {"concept": _CONCEPT, "window": _WINDOW},
    "not_typed": {"why": "<8-240 characters: why no kind expresses it>"},
}


def population_spec_shape() -> dict[str, Any]:
    """The spec as the foundation template shows it, one criterion long."""

    return {
        "criteria": [
            {
                "id": "<c1, c2, ...>",
                "quote": "<the words that state it, 2-160 characters>",
                "source": "<question|study_wording|outline|preset>",
                "role": "<include|exclude>",
                "kind": "<one kind listed below>",
            }
        ]
    }


def population_spec_contract(concept_ids: Sequence[str]) -> str:
    """The spec's rules in the foundation contract's text, naming no case."""

    kinds = {
        kind: fields
        for kind, fields in _KIND_FIELDS.items()
        if concept_ids or not any(field in fields for field in _CONCEPT_FIELDS)
    }
    if set(_KIND_FIELDS) != set(CRITERION_MODELS):
        raise PopulationSpecTransportError(
            "the contract text and the spec owner name different kinds"
        )
    return (
        "\npopulation_spec states the same population again as typed criteria. "
        "The host compiles it beside this cohort and records how the two "
        "compare; the predicates above still select the rows. Write "
        '{"criteria":[]} when the study includes every input row. Each '
        "criterion keeps the words that state it (quote, verbatim; criteria "
        "stated in one sentence share it), where they come from (source: the "
        "question, study_wording for the study's own cohort wording, the "
        "outline, or a preset), and its role (include keeps the stays that "
        "meet it, exclude removes them), and adds exactly the fields of its "
        "kind:\n"
        + json.dumps(kinds, ensure_ascii=False, separators=(",", ":"))
        + "\nWindows are hours after ICU admission, [start_hours, end_hours); "
        "a condition_present or event_absent window may be null, meaning the "
        "whole stay. Age and stay-length bounds are inclusive: give at least "
        "one, and null for an open end. "
        + ", ".join(sorted(KEPT_KINDS & set(kinds)))
        + " state the stays kept, so their role is include; an excluding "
        "condition_present names one concept. A condition is present or "
        "absent, never compared with a number: state it with condition_present "
        "or event_absent, and keep measurement for a value with a threshold. "
        "State a restriction no kind expresses as not_typed with why; never "
        "drop one. A restriction that the exposure itself defines, such as "
        "where the exposure starts relative to time zero or the exclusion of "
        "stays already exposed, belongs to the design's exposure definition, "
        "not to population_spec."
    )
