"""The hub reads workflow owners without opening a study or patient rows."""

import ast
import json
from pathlib import Path
import re

import pytest

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, assess_study
from easyicu.research_agent.method_skills import (
    MethodSkillPackageNotFound,
    method_skill_catalog,
    method_skill_package,
    method_skill_workflow_metadata,
)
from easyicu.research_agent.planning.progressive_contract import PROGRESSIVE_HOST_COMPILED_OUTPUTS
from easyicu.webserver import research_workflows as projection
from easyicu.webserver.research_workflows import research_workflow_catalog
from easyicu.webserver.research_workflow_shapes import canonical_workflow_shape
from easyicu.webserver.raw_source_authority import source_identity_capability_profiles
from tests.support.node import run_node


@pytest.fixture(scope="module")
def workflows():
    return research_workflow_catalog(enabled=True)["items"]


def test_every_complete_study_projects_products_and_a_real_graph(workflows):
    expected = {row["id"] for row in method_skill_catalog(enabled=True)["items"]
                if row["layer"] == "research_workflow"}
    rows = workflows
    assert {row["id"] for row in rows} == expected
    for row in rows:
        assert row["steps"] and row["outputs"]
        ids = {step["id"] for step in row["steps"]}
        assert len(ids) == len(row["steps"])
        assert all(edge["source"] in ids and edge["target"] in ids for edge in row.get("edges", []))
        assert len(row["databases"]) == 6
        assert row["read_only"] is True


def test_every_graph_product_and_dependency_comes_from_the_template_or_public_package(workflows):
    for row in workflows:
        shape = canonical_workflow_shape(row["id"])
        if shape is None:
            metadata = method_skill_workflow_metadata(row["id"], enabled=True)
            assert [step["id"] for step in row["steps"]] == [step["id"] for step in metadata["steps"]]
            assert {item["id"] for item in row["outputs"]} == set(metadata["output_files"])
            assert "edges" not in row  # No owner declares their ordering.
            assert row["canonical"] is False
            assert all("kind" not in step for step in row["steps"])
            assert all(item["planned"] == metadata["planned_filenames"] for item in row["outputs"])
            continue
        request, draft, optional = shape
        assert row["canonical"] is True
        assert row["graph_source"] == request.family_id
        assert [step["id"] for step in row["steps"]] == [item.step.step_id for item in draft.materializations]
        expected_products, expected_edges = set(), set()
        consumed = {}
        for item in draft.materializations:
            step = item.step
            expected_products.update(product for product, _ in PROGRESSIVE_HOST_COMPILED_OUTPUTS.get(step.module_id, ()))
            expected_products.update(output.product_id for output in step.outputs)
            expected_edges.update((parent, step.step_id) for parent in step.depends_on)
            expected_edges.update((ref.producer_step_id, step.step_id) for ref in step.product_inputs)
            for ref in step.product_inputs:
                assert ref.product_id in next(node["outputs"] for node in row["steps"] if node["id"] == ref.producer_step_id)
                consumed.setdefault((ref.producer_step_id, step.step_id), []).append(ref.product_id)
        assert {item["id"] for item in row["outputs"]} == expected_products
        assert {(edge["source"], edge["target"]) for edge in row["edges"]} == expected_edges
        for edge in row["edges"]:
            assert edge["products"] == consumed.get((edge["source"], edge["target"]), [])
        assert {node["id"] for node in row["steps"] if node["optional"]} == optional


def test_database_chips_preserve_the_profile_decision_and_typed_reasons(workflows):
    for row in workflows:
        requirements = StudyRequirements(**method_skill_workflow_metadata(row["id"], enabled=True)["requirements"])
        for database in row["databases"]:
            profile = research_capability_profile(database["id"])
            expected = assess_study(profile, requirements).model_dump(mode="json")
            assert database["decision"] == expected
            status = {"exact": "supported", "substitution": "degraded", "refusal": "unavailable"}[expected["status"]]
            if expected["unassessed_concepts"] and status == "supported":
                status = "degraded"
            assert database["status"] == status
            assert database["analysis_units"] == list(profile.analysis_units)
            assert database["identity_profiles"] == source_identity_capability_profiles(database["id"])
            assert database["real_data_acceptance"] == "not_verified"


def test_inputs_are_typed_owner_roles_and_never_universal_prose(workflows):
    from easyicu.databases.research_requirements import family_requirements

    for row in workflows:
        metadata = method_skill_workflow_metadata(row["id"], enabled=True)
        assert all(isinstance(item, dict) and "role" in item for item in row["inputs"])
        assert set(family_requirements(metadata["requirements"]["family"])) <= {item["role"] for item in row["inputs"]}
        assert {item["concept"] for item in row["inputs"] if item["role"] == "concept"} == set(metadata["requirements"]["concepts"])
        assert len({(item["role"], item.get("concept")) for item in row["inputs"]}) == len(row["inputs"])
        for item in row["inputs"]:
            if item.get("concept") in metadata["input_labels"]:
                assert item["label_zh"] == metadata["input_labels"][item["concept"]]["label_zh"]
        shape = canonical_workflow_shape(row["id"])
        if shape and shape[0].exposure_kind != "none":
            exposure = next(item for item in row["inputs"] if item["role"] == "exposure")
            assert exposure["kind"] == shape[0].exposure_kind
            assert exposure["ordered"] == shape[0].exposure_is_ordered
        if shape and (shape[0].observation_duration_column or shape[0].survival_suite):
            assert any(item["role"] == "followup" for item in row["inputs"])


def test_master_switch_disables_every_workflow_without_hiding_the_metadata():
    rows = research_workflow_catalog(enabled=False)["items"]
    assert rows and all(row["enabled"] is False and row["steps"] and row["outputs"] for row in rows)


def test_workflow_version_references_and_example_ids_are_owner_bound(workflows):
    static = Path(projection.__file__).parent / "static" / "js"
    starters = (static / "screens-guided-pi-starters.js").read_text()
    example_ids = set(re.findall(r"id: '(example-[^']+)'", starters))
    for row in workflows:
        package = method_skill_package(row["id"], enabled=True)
        assert row["example"] in example_ids
        assert row["references"] == [item["path"] for item in package["files"] if item["path"].startswith("references/")]
        assert row["version"] == next(item["version"] for item in method_skill_catalog(enabled=True)["items"] if item["id"] == row["id"])


def test_projection_contains_no_literal_scientific_input_or_output_copy():
    tree = ast.parse(Path(projection.__file__).read_text())
    literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any(re.match(r"(?:table|figure|report|artifact|manifest):", value) for value in literals)
    assert not any(" " in value for value in literals if value != ast.get_docstring(tree, clean=False))


def test_component_methods_cannot_acquire_a_workflow_projection():
    with pytest.raises(MethodSkillPackageNotFound):
        method_skill_workflow_metadata("cohort-characterization-table-one", enabled=True)


def test_read_only_route_delegates_to_the_owner_and_honors_the_master_switch(monkeypatch):
    from easyicu.webserver.routes import system

    monkeypatch.setattr(system.capabilities, "capability_settings", lambda: {"science_skills_enabled": False})
    payload = system.get_research_workflows()
    assert payload["items"] and all(row["enabled"] is False for row in payload["items"])
    assert any(route.path == "/api/capabilities/research-workflows" and route.methods == {"GET"}
               for route in system.router.routes)


def test_every_projected_product_has_a_shared_reader_label(workflows):
    vocabulary = Path(projection.__file__).parent / "static/js/screens-agent-reader-vocab.js"
    script = """
global.window = { t: (en, zh) => zh };
require(process.argv[1]);
const ids = JSON.parse(process.argv[2]);
const missing = ids.filter(id => !window.AGENT_READER_VOCAB.tableTitle(id));
if (missing.length) { console.error(missing); process.exitCode = 1; }
"""
    result = run_node("node", script, str(vocabulary), json.dumps(
        sorted({item["id"] for row in workflows for item in row["outputs"]})), check=False)
    assert result.returncode == 0, result.stderr


def test_metadata_projection_never_opens_patient_tables(monkeypatch):
    import pandas as pd
    import pyarrow.parquet as pq

    def forbidden(*args, **kwargs):
        pytest.fail("metadata opened a patient table")

    for owner, names in ((pd, ("read_csv", "read_parquet")), (pq, ("read_table", "read_pandas"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    assert research_workflow_catalog(enabled=True)["items"]


def test_identity_recipes_are_copies_and_never_claim_a_local_binding():
    mimic = source_identity_capability_profiles("miiv")
    assert mimic["first_icu_stay"]["order_column"] == "intime"
    mimic["patient_grouping"]["table"] = "changed"
    assert source_identity_capability_profiles("miiv")["patient_grouping"]["table"] == "icustays"
    assert source_identity_capability_profiles("eicu")["first_icu_stay"] is None
    assert source_identity_capability_profiles("sic")["first_icu_stay"] is None
    assert source_identity_capability_profiles("hirid")["patient_identity_absent"] is True
    assert source_identity_capability_profiles("unknown")["patient_grouping"] is None
