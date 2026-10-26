"""Clinician tasks are discoverable without granting research authority."""

from dataclasses import replace
import json

import pytest

from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
from easyicu.research_agent.method_skills import OWNER_SKILLS, validate_owner_skill


def test_clinician_tasks_have_separate_authority_and_packages():
    catalog = method_skill_catalog(enabled=True)
    rows = {row["id"]: row for row in catalog["items"]}
    for name in ("six-database-cohort-extraction", "clinical-concept-definitions"):
        row = rows[name]
        assert row["layer"] == "data_definition"
        assert "claim_ceiling" not in row
        assert row["capability_id"] is None
        assert row["action_ids"] == []
        assert row["use_action"]["kind"] == "open_owner"
        assert row["use_action"]["auto_submit"] is False
        assert method_skill_package(name, enabled=True)["execution_boundary"] == "owner_projection_no_execution"
    assert rows["sample-size-and-power"]["layer"] == "planning_aid"
    assert rows["sample-size-and-power"]["claim_ceiling"] == "analysis_only"
    assert rows["treatment-patterns"]["claim_ceiling"] == "analysis_only"
    assert catalog["data_definition_count"] == 2
    assert catalog["planning_aid_count"] == 1
    for name in ("six-database-cohort-extraction", "clinical-concept-definitions",
                 "sample-size-and-power", "treatment-patterns"):
        package = method_skill_package(name, enabled=True)
        files = {row["path"]: row["content"] for row in package["files"]}
        assert "When to use" in files["SKILL.md"]
        assert "Not for" in files["SKILL.md"]
        assert "references/stops.json" in files
        assert "references/example.md" in files


def test_disabled_clinician_tasks_do_not_offer_an_enabled_action():
    catalog = method_skill_catalog(enabled=False)
    assert catalog["active_skill_ids"] == []
    for row in catalog["items"]:
        if "use_action" in row:
            assert row["use_action"]["enabled"] is False


@pytest.mark.parametrize("changes,code", [
    ({"layer": "analysis_module"}, "skill.owner_binding_invalid"),
    ({"owner": "pretend"}, "skill.owner_binding_invalid"),
    ({"entry": "/api/execute"}, "skill.owner_binding_invalid"),
    ({"skill_id": "unknown"}, "skill.owner_binding_invalid"),
    ({"capability_id": "descriptive_measurement_v1"}, "skill.projection_cannot_execute_analysis"),
    ({"action_ids": ("descriptive.table_one",)}, "skill.projection_cannot_execute_analysis"),
    ({"included_module_ids": ("cohort-characterization-table-one",)}, "skill.projection_cannot_execute_analysis"),
    ({"claim_ceiling": "analysis_only"}, "skill.definition_cannot_claim"),
])
def test_data_layer_rejects_execution_and_claims(changes, code):
    with pytest.raises(RuntimeError, match=code):
        validate_owner_skill(replace(OWNER_SKILLS[0], **changes))


@pytest.mark.parametrize("ceiling", [None, "reportable"])
def test_planning_cannot_claim_findings(ceiling):
    with pytest.raises(RuntimeError, match="skill.planning_analysis_only"):
        validate_owner_skill(replace(OWNER_SKILLS[2], claim_ceiling=ceiling))


def test_analysis_spec_cannot_bypass_new_layer_validation(monkeypatch):
    from easyicu.research_agent import method_skills

    row = next(row for row in method_skills.METHOD_SKILLS if row.skill_id == "treatment-patterns")
    monkeypatch.setattr(method_skills, "METHOD_SKILLS", (replace(row, layer="data_definition"),))
    with pytest.raises(RuntimeError, match="skill.analysis_layer_invalid"):
        method_skills._validate_method_skills()


def test_registry_and_projection_both_validate_owner_skills(monkeypatch):
    from easyicu.research_agent import method_skills

    invalid = replace(OWNER_SKILLS[0], claim_ceiling="reportable")
    monkeypatch.setattr(method_skills, "METHOD_SKILLS", (invalid,))
    with pytest.raises(RuntimeError, match="skill.definition_cannot_claim"):
        method_skills._validate_method_skills()
    with pytest.raises(RuntimeError, match="skill.definition_cannot_claim"):
        invalid.to_dict(enabled=True)


def test_clinician_packages_are_stable_and_siblings_exist():
    from easyicu.research_agent.method_skills import CLINICIAN_SKILL_IDS

    ids = {row["id"] for row in method_skill_catalog(enabled=True)["items"]}
    for name in CLINICIAN_SKILL_IDS:
        one = method_skill_package(name, enabled=True)
        assert one == method_skill_package(name, enabled=True)
        files = {row["path"]: row["content"] for row in one["files"]}
        assert json.loads(files["references/stops.json"])
        for line in files["SKILL.md"].splitlines():
            if " -> " in line and not line.startswith("##"):
                sibling = line.split(" -> ")[1].split()[0].rstrip(".,")
                assert sibling in ids
