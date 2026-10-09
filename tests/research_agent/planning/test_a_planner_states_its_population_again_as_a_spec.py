"""A Planner that chooses its cohort states the population again as a typed spec.

Step 2a of the population spec design asks the foundation for the population
twice in one call: the cohort intent it already writes, and the same
population as typed criteria (``cohort.population_spec``).  A Provider with
strict JSON schema receives the spec owner's schema made strict: one closed
alternative per kind, the run's allowed concepts, an inclusion-only role where
a kind states the stays kept, and a required but nullable spec.  Every other
strict request built from the plan models sees the field as a closed null.  A
Provider without one reads the same shape and rules in the contract text.  The
spec is a shadow: the cohort keeps it as written, so a spec its owner would
refuse does not stop the foundation.  Rosters and concepts are synthetic.
"""

from __future__ import annotations

import json
import math
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.population_spec_transport import (
    bind_population_spec_transport,
    population_spec_contract,
)
from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_foundation_materialization,
    progressive_foundation_structured_output_request,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
    progressive_structured_output_request,
)
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.population_spec import (
    CRITERION_MODELS,
    KEPT_KINDS,
    PopulationSpec,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveOutlineStep,
)
from easyicu.research_agent.providers.strict_json_schema import (
    assert_closed_json_schema,
)

_CONCEPTS = ("age_years", "marker_flag", "level_value")
_COMMON = {"id", "quote", "source", "role", "kind"}
_WINDOW = {"start_hours": 0, "end_hours": 24}
#: One criterion of each kind as a strict reply writes it: every field present.
_ONE_OF_EACH: dict[str, dict[str, Any]] = {
    "age_years": {"min_years": 18, "max_years": None},
    "first_icu_stay": {},
    "icu_stay_hours": {"min_hours": 24, "max_hours": None},
    "condition_present": {"concepts_all_of": ["marker_flag"], "window": _WINDOW},
    "diagnosis_codes": {"system": "icd10", "codes": ["X12"]},
    "measurement": {
        "concept": "level_value",
        "summary": "max",
        "window": _WINDOW,
        "op": ">=",
        "value": 2,
        "unit": None,
    },
    "alive_at": {"hours": 24},
    "event_absent": {"concept": "marker_flag", "window": None},
    "not_typed": {"why": "no kind states a referral the stay received"},
}


def _request(concepts=_CONCEPTS, mode=None) -> dict:
    request = progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=("age_years", "marker_flag", "level_value"),
        cohort_concept_ids=concepts,
        required_cohort_selection_mode=mode,
    )
    return json.loads(request.schema_json)


def _criterion(index: int, kind: str, **fields) -> dict:
    return {
        "id": f"c{index}",
        "quote": f"stated restriction {index}",
        "source": "question",
        "role": "include",
        "kind": kind,
        **_ONE_OF_EACH[kind],
        **fields,
    }


def _reply(spec: Any) -> dict:
    return {
        "schema_version": "easyicu.progressive_plan_foundation/1",
        "outline_sha256": "a" * 64,
        "foundation": {
            "cohort": {
                "name": "every input stay",
                "selection_mode": "all_input_rows",
                "inclusion": [],
                "exclusion": [],
                "population_criteria": [],
                "population_spec": spec,
            },
            "display_labels": [],
            "robustness_intents": [],
            "know_how_decisions": [],
        },
    }


def _valid(schema: dict, reply: dict) -> bool:
    try:
        jsonschema.validate(reply, schema, cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError:
        return False
    return True


def test_the_strict_foundation_asks_for_the_spec_as_closed_kinds() -> None:
    schema = _request()
    definitions = schema["$defs"]

    assert definitions["ProgressiveCohortIntent"]["properties"]["population_spec"] == {
        "anyOf": [{"$ref": "#/$defs/PopulationSpec"}, {"type": "null"}]
    }
    assert "population_spec" in definitions["ProgressiveCohortIntent"]["required"]
    offered = definitions["PopulationSpec"]["properties"]["criteria"]["items"]
    assert offered == {
        "anyOf": [
            {"$ref": f"#/$defs/{model.__name__}"} for model in CRITERION_MODELS.values()
        ]
    }
    text = json.dumps(schema)
    assert '"oneOf"' not in text and '"discriminator"' not in text
    concept = {"type": "string", "enum": list(_CONCEPTS)}
    assert definitions["Measurement"]["properties"]["concept"] == concept
    assert definitions["EventAbsent"]["properties"]["concept"] == concept
    assert definitions["ConditionPresent"]["properties"]["concepts_all_of"][
        "items"
    ] == (concept)
    for kind, model in CRITERION_MODELS.items():
        role = definitions[model.__name__]["properties"]["role"]
        if kind in KEPT_KINDS:
            assert role == {"type": "string", "const": "include"}
        else:
            assert role["enum"] == ["include", "exclude"]
        # Strict: every field is required, an optional one as nullable.
        assert set(definitions[model.__name__]["required"]) == set(model.model_fields)


@pytest.mark.parametrize("kinds", [list(_ONE_OF_EACH)[:5], list(_ONE_OF_EACH)[5:]])
def test_a_spec_the_strict_schema_accepts_is_one_its_owner_reads(kinds) -> None:
    schema = _request()
    spec = {
        "criteria": [_criterion(index, kind) for index, kind in enumerate(kinds, 1)]
    }

    assert _valid(schema, _reply(spec))
    assert _valid(schema, _reply(None))
    assert _valid(schema, _reply({"criteria": []}))
    parsed = parse_progressive_foundation_materialization(
        json.dumps(_reply(spec)), host_cohort=None, outline_sha256="a" * 64
    )
    typed = PopulationSpec.model_validate(parsed.foundation.cohort.population_spec)
    assert [item.kind for item in typed.criteria] == kinds


@pytest.mark.parametrize(
    "criterion",
    [
        _criterion(1, "age_years", role="exclude"),
        _criterion(1, "measurement", concept="unlisted_value"),
        _criterion(1, "condition_present", concepts_all_of=["unlisted_flag"]),
        _criterion(1, "measurement", threshold=2),
        {
            key: value
            for key, value in _criterion(1, "age_years").items()
            if key != "max_years"
        },
        {**_criterion(1, "condition_present"), "kind": "status_at_least"},
    ],
    ids=[
        "a-kept-kind-excluding",
        "a-concept-outside-the-roster",
        "a-condition-outside-the-roster",
        "a-field-its-kind-lacks",
        "an-optional-field-left-out",
        "an-unknown-kind",
    ],
)
def test_the_strict_schema_refuses_what_its_kinds_do_not_state(criterion) -> None:
    assert not _valid(_request(), _reply({"criteria": [criterion]}))


def test_with_no_allowed_concept_the_kinds_that_name_one_are_not_offered() -> None:
    # The foundation request falls back to the variable roster; the binder
    # itself offers no kind it could not fill.
    definitions: dict = {}
    cohort: dict = {}
    bind_population_spec_transport(definitions, cohort, concept_ids=(), stated=True)
    offered = {
        ref["$ref"].rsplit("/", 1)[-1]
        for ref in definitions["PopulationSpec"]["properties"]["criteria"]["items"][
            "anyOf"
        ]
    }

    assert offered == {
        "AgeYears",
        "FirstIcuStay",
        "IcuStayHours",
        "DiagnosisCodes",
        "AliveAt",
        "NotTyped",
    }
    assert not {"ConditionPresent", "Measurement", "EventAbsent"} & set(definitions)
    text = population_spec_contract(())
    assert '"measurement"' not in text and '"age_years"' in text


def test_a_cohort_of_every_input_row_bound_by_the_caller_states_no_spec() -> None:
    definitions = _request(mode="all_input_rows")["$defs"]

    assert definitions["ProgressiveCohortIntent"]["properties"]["population_spec"] == {
        "type": "null"
    }
    assert "PopulationSpec" not in definitions
    contract = foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=None,
        required_cohort_selection_mode="all_input_rows",
        cohort_concept_ids=_CONCEPTS,
    )
    assert "population_spec" not in contract


_ACTION = "association.adjusted_association"
_STEP = ProgressiveOutlineStep(
    step_id="primary_model",
    module_id="custom_analysis",
    planned_analysis_role="primary",
    objective="Fit the prespecified model.",
    variable_names=list(_CONCEPTS),
    scientific_action_id=_ACTION,
)


def _strict_request(name: str):
    rosters = {"variable_names": _CONCEPTS, "scientific_action_ids": (_ACTION,)}
    if name in {"initial", "suffix"}:
        return progressive_structured_output_request(
            analysis_types=("association_study",),
            cohort_concept_ids=_CONCEPTS,
            suffix=name == "suffix",
            **rosters,
        )
    if name == "outline":
        return progressive_outline_structured_output_request(
            analysis_types=("association_study",), **rosters
        )
    if name == "step":
        return progressive_step_materialization_request(
            outline_step=_STEP,
            outline_step_sha256=canonical_sha256(_STEP.model_dump(mode="json")),
            **rosters,
        )
    mode = {"foundation": None, "foundation-every-row": "all_input_rows"}[name]
    return progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=_CONCEPTS,
        cohort_concept_ids=_CONCEPTS,
        required_cohort_selection_mode=mode,
    )


@pytest.mark.parametrize(
    "name",
    ["initial", "suffix", "outline", "foundation", "foundation-every-row", "step"],
)
def test_every_strict_progressive_request_still_closes(name) -> None:
    # The cohort intent keeps a spec of any JSON shape, but its own schema is a
    # closed null: each strict request built from the plan models closes, and
    # only a foundation that states its population offers the spec.
    schema = json.loads(_strict_request(name).schema_json)

    assert_closed_json_schema(schema)
    assert ("PopulationSpec" in schema["$defs"]) == (name == "foundation")
    intent = schema["$defs"].get("ProgressiveCohortIntent")
    if intent is not None and name != "foundation":
        assert intent["properties"]["population_spec"] == {"type": "null"}


@pytest.mark.parametrize("required_mode", [None, "predicate_filtered"])
def test_a_planner_without_a_schema_reads_each_kind_and_its_fields(
    required_mode,
) -> None:
    contract = foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=None,
        required_cohort_selection_mode=required_mode,
        cohort_concept_ids=_CONCEPTS,
    )
    template = json.loads(contract.split("\n")[1])
    shown = template["foundation"]["cohort"]["population_spec"]["criteria"][0]
    kinds_line = next(
        line for line in contract.split("\n") if line.startswith('{"age_years"')
    )
    kinds = json.loads(kinds_line)

    assert set(shown) == _COMMON
    assert list(kinds) == list(CRITERION_MODELS)
    for kind, model in CRITERION_MODELS.items():
        assert set(kinds[kind]) == set(model.model_fields) - _COMMON
    assert "population_spec states the same population again as typed criteria" in (
        contract
    )
    host = foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=ProgressiveCohortIntent(
            name="bound", selection_mode="all_input_rows"
        ),
        cohort_concept_ids=_CONCEPTS,
    )
    assert "population_spec" not in host


def test_the_spec_rules_name_no_case() -> None:
    text = population_spec_contract(("zz_alpha", "zz_beta"))

    assert (
        "A restriction that the exposure itself defines, such as where the exposure "
        "starts relative to time zero or the exclusion of stays already exposed, "
        "belongs to the design's exposure definition, not to population_spec."
    ) in text
    for word in ("sepsis", "shock", "sofa", "lactate", "ventilat", "mimic", "eicu"):
        assert word not in text.casefold()
    assert "zz_alpha" not in text


@pytest.mark.parametrize(
    "refused",
    [
        {"criteria": [_criterion(1, "age_years", role="exclude")]},
        [_criterion(1, "age_years")],
        "adults only",
        7,
    ],
    ids=[
        "a-kept-kind-excluding",
        "the-criteria-without-their-object",
        "words",
        "a-number",
    ],
)
def test_a_spec_of_any_shape_does_not_stop_the_foundation(refused) -> None:
    # A Provider without a schema may write what the owner refuses, in any shape.
    with pytest.raises(ValidationError):
        PopulationSpec.model_validate(refused)

    parsed = parse_progressive_foundation_materialization(
        json.dumps(_reply(refused)), host_cohort=None, outline_sha256="a" * 64
    )

    assert parsed.foundation.cohort.population_spec == refused
    assert parsed.foundation.cohort.selection_mode == "all_input_rows"


def test_a_spec_is_left_out_of_the_digest_when_absent() -> None:
    plain = ProgressiveCohortIntent(name="all", selection_mode="all_input_rows")
    stated = ProgressiveCohortIntent(
        name="all", selection_mode="all_input_rows", population_spec={"criteria": []}
    )

    assert "population_spec" not in plain.model_dump(mode="json")
    assert stated.model_dump(mode="json")["population_spec"] == {"criteria": []}


@pytest.mark.parametrize(
    "spec",
    [
        {"criteria": [{"value": math.nan}]},
        {"criteria": [{"quote": "x" * 40_000}]},
        {"criteria": [{"concepts_all_of": {"not", "json"}}]},
    ],
    ids=["not-finite", "oversized", "not-json"],
)
def test_a_spec_that_is_not_a_few_criteria_of_plain_json_is_refused(spec) -> None:
    with pytest.raises(ValidationError):
        ProgressiveCohortIntent(
            name="all", selection_mode="all_input_rows", population_spec=spec
        )
