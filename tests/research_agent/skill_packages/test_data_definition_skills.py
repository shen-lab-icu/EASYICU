"""Metadata-only skill views must be exact projections of their owners."""

from dataclasses import asdict
import inspect

import pytest

from easyicu.clinical_contracts import load_clinical_contracts
from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES
from easyicu.databases.profiles import public_database_keys
from easyicu.resources import load_dictionary
from easyicu.research_agent.concept_availability import explain_concept_availability
from easyicu.research_agent.concept_catalog import (
    DefinitionProjectionError, clinical_definitions, extraction_projection,
)


def test_all_composites_are_projected_from_the_current_owners():
    dictionary = load_dictionary(include_sofa2=True)
    contracts = load_clinical_contracts()
    rows = {row["concept"]: row for row in clinical_definitions(database="miiv")["items"]}
    registered = {name for name, definition in dictionary.items()
                  if definition.sub_concepts or definition.depends_on}
    registered |= {name for contract in contracts.values() for name in contract.concepts}
    assert set(rows) == registered | set(COMPOSITE_CONCEPT_OUTPUT_SOURCES)
    for name in ("sofa2", "sep3", "kdigo_aki"):
        definition = dictionary.get(name)
        assert rows[name]["components"] == list(dict.fromkeys(definition.sub_concepts + definition.depends_on))
        assert rows[name]["description"] == definition.description
        assert rows[name]["callback"] == definition.callback
        expected = asdict(contracts[definition.clinical_contract_id])
        for key in ("definition", "version", "definition_time_anchor", "database_conformance"):
            assert rows[name]["clinical_contract"][key] == expected[key]


def test_availability_and_aki_profiles_equal_database_owners():
    from easyicu.scores.aki_profiles import list_aki_profiles

    rows = {row["concept"]: row for row in clinical_definitions(["sofa2", "sep3", "kdigo_aki"])["items"]}
    for name, row in rows.items():
        assert set(row["databases"]) == set(public_database_keys())
        for db, cell in row["databases"].items():
            assert cell["availability"] == explain_concept_availability(concept=name, database=db).model_dump(mode="json")
            if name == "kdigo_aki":
                assert cell["aki_profiles"] == [dict(p.payload, profile_id=p.profile_id) for p in list_aki_profiles(db)]
    assert rows["sofa2"]["databases"]["sic"]["usable"] is False
    assert rows["sofa2"]["databases"]["sic"]["structurally_unavailable_components"] == ["sofa2_cns"]


def test_windows_and_demo_substitutions_follow_owner_defaults():
    from easyicu.scores.sepsis import sep3
    from easyicu.databases.profiles import get_packaged_registry

    row = clinical_definitions(["sep3"], database="eicu_demo")["items"][0]
    defaults = row["time_window"]["score_owner_defaults"]
    assert defaults["si_lwr"] == str(inspect.signature(sep3).parameters["si_lwr"].default)
    assert defaults["si_upr"] == str(inspect.signature(sep3).parameters["si_upr"].default)
    assert row["time_window"]["definition_time_anchor"] == "suspected_infection_onset"
    assert row["time_window"]["study_observation_window"] == "requires_explicit_selection"
    assert row["databases"]["eicu_demo"]["substitution_policy"] == "owner_declared_only_no_automatic_substitution"
    bindings = row["databases"]["eicu_demo"]["owner_source_bindings"]
    source = get_packaged_registry().get("eicu_demo")
    definition = load_dictionary(include_sofa2=True).get("sofa")
    # These bindings are owner mappings, not a hand-authored replacement table.
    import json
    assert bindings["sofa"] == json.loads(json.dumps(
        [asdict(item) for item in definition.for_data_source(source)], default=str,
    ))


def test_extraction_prefill_projects_all_six_owners_without_executing(monkeypatch):
    from easyicu.api import extraction

    def forbidden(*args, **kwargs):
        pytest.fail("metadata projection executed extraction")

    monkeypatch.setattr(extraction, "extract_database", forbidden)
    result = extraction_projection()
    assert {row["database"] for row in result["databases"]} == set(public_database_keys())
    for row in result["databases"]:
        assert {k: v["concepts"] for k, v in row["modules"].items()} == extraction.EXTRACT_MODULES
        assert row["use_action"]["prefill"]["database"] == row["database"]
        assert row["use_action"]["auto_submit"] is False
        for module in row["modules"].values():
            for concept, cell in module["availability"].items():
                assert cell == explain_concept_availability(concept=concept, database=row["database"]).model_dump(mode="json")


@pytest.mark.parametrize("concepts,db,code", [
    ([], None, "definition.concepts_required"),
    ("sofa2", None, "definition.concepts_required"),
    (["unregistered_composite"], None, "definition.concept_unknown"),
    (["sofa2"], "unregistered_database", "definition.database_unknown"),
])
def test_invalid_definition_requests_stop(concepts, db, code):
    with pytest.raises(DefinitionProjectionError, match=code):
        clinical_definitions(concepts, database=db)


def test_new_owner_definition_appears_without_a_copied_skill_table(monkeypatch):
    from easyicu import resources
    from easyicu.concept.schema import ConceptDefinition, ConceptDictionary
    from easyicu.research_agent.concept_availability import _explain_concept_availability_cached

    original = resources.load_dictionary(include_sofa2=True)
    definition = ConceptDefinition(
        name="synthetic_composite_marker", sources={}, description="Synthetic combined marker",
        sub_concepts=["hr", "map"], callback="synthetic_owner_callback",
    )
    dictionary = ConceptDictionary({definition.name: definition,
                                    "hr": original.get("hr"), "map": original.get("map")})
    monkeypatch.setattr(resources, "load_dictionary", lambda **kwargs: dictionary)
    _explain_concept_availability_cached.cache_clear()
    try:
        row = clinical_definitions([definition.name], database="miiv")["items"][0]
        assert row["components"] == ["hr", "map"]
        assert row["description"] == definition.description
        assert row["callback"] == definition.callback
    finally:
        _explain_concept_availability_cached.cache_clear()


@pytest.mark.parametrize("concept", ["sofa2_cns", "sep3_sofa2"])
def test_structural_score_absence_is_retained_by_derived_definitions(concept):
    row = clinical_definitions([concept], database="sic")["items"][0]
    assert row["databases"]["sic"]["structurally_unavailable_components"] == ["sofa2_cns"]
    assert row["databases"]["sic"]["usable"] is False
