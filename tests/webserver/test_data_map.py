"""The data map projects packaged owners; no patient data or export scans."""

import json
from pathlib import Path

import pytest

from easyicu.concept import catalog
from easyicu.databases.profiles import iter_database_profiles
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, assess_study
from easyicu.research_agent.planning.analysis_types import list_analysis_types
from easyicu.webserver import sources
from easyicu.webserver.catalog_metadata import catalog_definitions, concept_audience


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")

    def forbidden(*args, **kwargs):
        pytest.fail("the map must not discover, describe, or read an export")

    monkeypatch.setattr(sources, "load_registry", forbidden)
    monkeypatch.setattr(sources, "_source_from_path", forbidden)
    monkeypatch.setattr(sources, "_autodiscovered_paths", forbidden)
    monkeypatch.setattr(sources.dataio, "describe_export_source", forbidden)


def build(**kwargs):
    from easyicu.webserver.data_map import build_data_map

    return build_data_map(**kwargs).model_dump(mode="json")


def test_map_projects_the_six_profiles_without_claiming_real_data_acceptance():
    result = build(concept="hr")
    profiles = iter_database_profiles(public_only=True)
    assert [row["database"] for row in result["databases"]] == [p.key for p in profiles]
    assert result["basis"] == "packaged_metadata"
    for row, profile in zip(result["databases"], profiles, strict=True):
        facts = research_capability_profile(profile.key)
        assert row["profile"] == profile.model_dump(mode="json")
        assert row["capabilities"] == facts.model_dump(mode="json", exclude={"concepts"})
        assert row["real_data_acceptance"] == "not_accepted"
        for name in ("years", "population_size", "sampling_resolution"):
            assert row[name] == {"value": None, "reason_code": "not_declared_by_metadata_owners"}
        assert row["sources"] == []


@pytest.mark.parametrize("database", [p.key for p in iter_database_profiles(public_only=True)])
def test_all_concept_pages_equal_the_catalog_and_capability_owner(database):
    profile = research_capability_profile(database)
    definitions = catalog_definitions()
    seen = []
    for offset in range(0, len(catalog.CONCEPT_DICTIONARY), 100):
        result = build(database=database, concept_offset=offset, concept_limit=100)
        assert result["concept_total"] == len(catalog.CONCEPT_DICTIONARY)
        for cell in result["databases"][0]["concepts"]:
            name = cell["concept"]
            seen.append(name)
            fact = profile.concept(name)
            assert cell["capability"] == fact.model_dump(mode="json")
            assert cell["unit"] == catalog.CONCEPT_DICTIONARY[name][2]
            assert cell["audience"] == concept_audience(name, definitions.get(name, {}))
            representation = getattr(profile, name, None) if name in {"age", "weight"} else None
            if name == "death" and profile.mortality and profile.mortality.meaning != "hospital_death":
                assert cell["status"] == "substituted"
                assert cell["substitute"] == profile.mortality.model_dump(mode="json")
            elif fact.mapped and representation and representation.kind not in {"exact", "unavailable"}:
                assert cell["status"] == "substituted"
                assert cell["substitute"] == representation.model_dump(mode="json")
            else:
                assert cell["status"] == ("available" if fact.mapped else "missing")
                assert cell["substitute"] is None
    assert seen == sorted(catalog.CONCEPT_DICTIONARY)


@pytest.mark.parametrize("database", [p.key for p in iter_database_profiles(public_only=True)])
def test_catalog_hospital_death_label_keeps_the_actual_source_meaning(database):
    profile = research_capability_profile(database)
    expected = assess_study(profile, StudyRequirements(family="descriptive_epidemiology", outcome="hospital_death"))
    cell = build(database=database, concept="death")["databases"][0]["concepts"][0]
    assert cell["status"] == ("substituted" if expected.substitutions else "available")
    if expected.substitutions:
        assert cell["substitute"] == profile.mortality.model_dump(mode="json")
        assert cell["substitute"]["meaning"] == expected.substitutions[0].substitute


def test_partial_derivation_is_not_invented_as_an_available_substitute():
    cell = build(database="miiv", concept="sofa2")["databases"][0]["concepts"][0]
    assert cell["capability"]["derivation_status"] == "degraded"
    assert cell["status"] == "missing"
    assert cell["reason_code"] == "partial_dependency_availability"
    assert cell["substitute"] is None


def test_study_families_retain_the_owners_decision_and_requirements():
    states = set()
    for row in build(concept="hr")["databases"]:
        assert [s["family"] for s in row["studies"]] == [s.key for s in list_analysis_types()]
        for study in row["studies"]:
            from easyicu.webserver.research_capability_scope import study_requirements
            from easyicu.databases.research_requirements import family_requirements

            requirements = (study_requirements({"execution_concepts": {"outcome": "death"}}, family=study["family"])
                            if "event_time" in family_requirements(study["family"]) else StudyRequirements(family=study["family"]))
            expected = assess_study(research_capability_profile(row["database"]), requirements)
            assert study["decision"] == expected.model_dump(mode="json")
            assert study["status"] == ("refused" if expected.status == "refusal" else expected.status)
            assert study["scope"] == "family_minimum_requirements"
            states.add(study["status"])
    assert states == {"exact", "refused"}


@pytest.mark.parametrize("database,status", [
    ("mimic", "exact"), ("miiv", "exact"), ("eicu", "refused"),
    ("hirid", "refused"), ("aumc", "refused"), ("sic", "refused"),
])
def test_survival_map_agrees_with_default_death_question_and_plan(database, status):
    from easyicu.databases.research_plan_requirements import plan_capability_decision
    from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext
    from easyicu.webserver.research_capability_scope import study_requirements
    from easyicu.webserver.study_family_scope import study_family_database_gap

    question = "Estimate time to death."
    study = {"analysis_design": {"analysis_family": "survival"}, "execution_concepts": {"outcome": "death"}}
    cell, = build(database=database, concept="death", family="survival")["databases"][0]["studies"]
    decision = assess_study(research_capability_profile(database), study_requirements(study, family="survival", question=question))
    context = ResearchContext(research_question=question, target_outcome="death",
        cohort=CohortDescriptor(cohort_name="synthetic", database=database, n_stays=0), variables=[])
    plan = AnalysisPlan(research_question=question, analysis_type="survival", steps=[])
    assert cell["status"] == status
    assert cell["decision"] == decision.model_dump(mode="json") == plan_capability_decision(context, plan).model_dump(mode="json")
    gap = study_family_database_gap(study, question=question, database=database)
    assert (gap is not None) == (status == "refused")
    if gap is not None:
        assert gap["missing_capabilities"] == cell["decision"]["missing"]


def test_filters_aliases_pagination_and_empty_tail():
    result = build(database="MIMIC-IV", concept="hr", family="survival")
    assert [r["database"] for r in result["databases"]] == ["miiv"]
    assert [c["concept"] for c in result["databases"][0]["concepts"]] == ["hr"]
    assert [s["family"] for s in result["databases"][0]["studies"]] == ["survival"]
    assert result["concept_total"] == 1
    assert result["concept_next_offset"] is None
    first = build(concept_limit=2)
    assert first["concept_next_offset"] == 2
    assert all(len(row["concepts"]) == 2 for row in first["databases"])
    tail = build(concept_offset=10000)
    assert tail["concept_next_offset"] is None
    assert all(not row["concepts"] for row in tail["databases"])


@pytest.mark.parametrize("filters,code", [
    ({"database": "synthetic_unknown"}, "data_map_unknown_database"),
    ({"database": "eicu_demo"}, "data_map_database_not_public"),
    ({"concept": "synthetic_unknown"}, "data_map_unknown_concept"),
    ({"family": "synthetic_unknown"}, "data_map_unknown_family"),
    ({"concept_limit": 101}, "data_map_invalid_page"),
    ({"concept_limit": 0}, "data_map_invalid_page"),
    ({"concept_offset": -1}, "data_map_invalid_page"),
    ({"source_limit": 101}, "data_map_invalid_page"),
    ({"source_offset": -1}, "data_map_invalid_page"),
])
def test_invalid_selectors_fail_with_codes_before_reading_registrations(monkeypatch, filters, code):
    from easyicu.webserver.data_map import DataMapError

    monkeypatch.setattr(sources, "_read_raw", lambda: pytest.fail("invalid selector read registry"))
    with pytest.raises(DataMapError) as error:
        build(**filters)
    assert error.value.code == code


def test_registration_identity_and_flags_are_path_free_bounded_snapshots():
    from easyicu.webserver.demo_source_contracts import get_source

    demo = get_source("mimic_iv_demo_v2_2")
    records = [
        {"id": "src_a", "database": "miiv", "path": "/private/synthetic", "label": "Private label", "source_kind": "synthetic"},
        {"id": "src_b", "database": "miiv", "path": "/private/demo", "label": f"{demo.title} v{demo.version}",
         "source_id": demo.id, "version": demo.version, "archive_sha256": "a" * 64},
        {"id": "src_c", "database": "eicu_demo", "path": "/private/legacy", "label": "Demo without receipt"},
        {"id": "src_d", "database": "miiv", "path": "/private/unrecorded", "label": "Private label"},
        {"id": "src_removed", "database": "miiv", "path": "/private/removed"},
        {"id": "src_unknown", "database": "synthetic_unknown", "path": "/private/unknown"},
    ]
    sources._CONFIG_PATH.write_text(json.dumps({"sources": records, "removed_paths": ["/private/removed"]}))
    result = build(concept="hr", source_limit=1)
    miiv = next(row for row in result["databases"] if row["database"] == "miiv")
    eicu = next(row for row in result["databases"] if row["database"] == "eicu")
    assert miiv["source_total"] == 3
    assert miiv["source_next_offset"] == 1
    assert miiv["sources"][0]["identity"] == {"database": "miiv", "registration_id": "src_a"}
    assert miiv["sources"][0]["kind"] == "synthetic"
    assert miiv["sources"][0]["is_demo"] is True
    assert eicu["sources"][0]["kind"] == "public_demo"
    assert eicu["sources"][0]["identity"] == {"database": "eicu_demo", "registration_id": "src_c"}
    next_page = build(database="miiv", concept="hr", source_offset=1, source_limit=1)["databases"][0]
    official = next_page["sources"][0]
    assert official["identity"] == {"database": "miiv", "registration_id": "src_b", "version": demo.version,
                                   "source_id": demo.id, "archive_sha256": "a" * 64}
    assert official["kind"] == "public_demo"
    last = build(database="miiv", concept="hr", source_offset=2)["databases"][0]
    assert last["sources"][0]["kind"] == "unrecorded"
    assert last["source_next_offset"] is None
    assert "/private" not in json.dumps(result)
    assert "Private label" not in json.dumps(result)
    assert all(s["basis"] == "stored_registration" for s in miiv["sources"])


def test_projection_never_changes_the_sealed_dictionary():
    path = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
    before = path.read_bytes()
    build()
    assert path.read_bytes() == before
