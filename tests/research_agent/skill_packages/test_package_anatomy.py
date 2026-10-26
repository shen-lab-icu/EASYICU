"""Reader packages describe reviewed tasks, not bare execution coordinates."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent import method_skills as skills


WORKFLOWS = tuple(row for row in skills.METHOD_SKILLS if row.layer == "research_workflow")
HEADINGS = (
    "Purpose", "When to use", "When not to use", "How it works",
    "What the plan will propose for your review", "Steps and their checks",
    "Outputs", "Per-database notes", "Common stops", "References", "Boundary",
)
HEADINGS_ZH = (
    "用途", "何时使用", "何时不适用", "工作原理", "计划将提出哪些内容供您审阅",
    "步骤与检查", "输出", "各数据库说明", "常见停止原因", "参考依据", "使用边界",
)


@pytest.mark.parametrize("row", WORKFLOWS, ids=lambda row: row.skill_id)
def test_workflow_has_clinical_bilingual_task_package(row):
    package = skills.method_skill_package(row.skill_id, enabled=True)
    assert package["schema_version"] == "easyicu.method-skill-package/4"
    files = {item["path"]: item["content"] for item in package["files"]}
    assert row.not_for
    assert "SKILL.zh.md" in files
    for path, headings in (("SKILL.md", HEADINGS), ("SKILL.zh.md", HEADINGS_ZH)):
        body = files[path]
        assert tuple(line[3:] for line in body.splitlines() if line.startswith("## ")) == headings
        assert row.capability_id not in body
        assert not any(word in body for word in ("契约", "门禁", "封闭", "冻结", "研究因素"))
    assert "examples/public_demo.md" in files
    assert "examples/public_demo.zh.md" in files
    assert "The plan proposes" in files["SKILL.md"] and "you review" in files["SKILL.md"]
    assert "计划将提出" in files["SKILL.zh.md"] and "一次审阅" in files["SKILL.zh.md"]
    assert package == skills.method_skill_package(row.skill_id, enabled=True)


@pytest.mark.parametrize("target", ["unknown-sibling", "survival-time-to-event"])
def test_registry_rejects_bad_sibling(monkeypatch, target):
    rows = tuple(
        replace(row, not_for=(("A changing exposure", "暴露随时间变化", target),))
        if row.skill_id == "survival-time-to-event" else row
        for row in skills.METHOD_SKILLS
    )
    monkeypatch.setattr(skills, "METHOD_SKILLS", rows)
    with pytest.raises(RuntimeError, match="sibling"):
        skills._validate_method_skills()


def test_registry_requires_a_workflow_exclusion(monkeypatch):
    rows = tuple(replace(row, not_for=()) if row.layer == "research_workflow" else row
                 for row in skills.METHOD_SKILLS)
    monkeypatch.setattr(skills, "METHOD_SKILLS", rows)
    with pytest.raises(RuntimeError, match="not_for"):
        skills._validate_method_skills()


@pytest.mark.parametrize("not_for", [(("", "不适用", None),), (("Not suitable", "", None),)])
def test_registry_requires_both_exclusion_languages(monkeypatch, not_for):
    rows = tuple(replace(row, not_for=not_for) if row.layer == "research_workflow" else row
                 for row in skills.METHOD_SKILLS)
    monkeypatch.setattr(skills, "METHOD_SKILLS", rows)
    with pytest.raises(RuntimeError, match="bilingual"):
        skills._validate_method_skills()


def test_sibling_projection_preserves_titles_and_optional_absence():
    catalog = {row["id"]: row for row in skills.method_skill_catalog(enabled=True)["items"]}
    for row in WORKFLOWS:
        package = skills.method_skill_package(row.skill_id, enabled=True)
        assert package["not_for"] == catalog[row.skill_id]["not_for"]
        for item, (en, zh, sibling) in zip(package["not_for"], row.not_for, strict=True):
            assert item == dict(situation_en=en, situation_zh=zh, sibling_skill_id=sibling)
            assert sibling is None or sibling in catalog and sibling != row.skill_id
    assert any(sibling is None for row in WORKFLOWS for _, _, sibling in row.not_for)


@pytest.mark.parametrize("owner,code", [
    ("typed_input_binding", "invented_stop"),
    ("invented_owner", "digest_mismatch"),
    ("question_requirements", "digest_mismatch"),
])
def test_stop_code_validation_rejects_unknown_or_wrong_owner(owner, code):
    stop = replace(skills.PACKAGE_STOPS[0], owner=owner, code=code)
    with pytest.raises(RuntimeError, match="stop_code_invalid"):
        skills.validate_package_stops((stop,))


def test_every_stop_is_registered_and_has_a_next_action():
    from easyicu.research_agent.execution.runners.typed_input_binding import BINDING_REASON_CODES
    from easyicu.research_agent.planning.question_requirements import QUESTION_REQUIREMENT_STOP_CODES

    owners = {"typed_input_binding": BINDING_REASON_CODES,
              "question_requirements": QUESTION_REQUIREMENT_STOP_CODES}
    skills.validate_package_stops(skills.PACKAGE_STOPS)
    for stop in skills.PACKAGE_STOPS:
        assert stop.code in owners[stop.owner]
        assert all(stop.situation) and all(stop.next_step)


def test_package_refuses_stale_stop_copy(monkeypatch):
    monkeypatch.setattr(skills, "PACKAGE_STOPS", (replace(skills.PACKAGE_STOPS[0], code="invented_stop"),))
    with pytest.raises(RuntimeError, match="stop_code_invalid"):
        skills.method_skill_package(WORKFLOWS[0].skill_id, enabled=True)


@pytest.mark.parametrize("row", WORKFLOWS, ids=lambda row: row.skill_id)
def test_database_notes_are_live_db6a_decisions(row):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import assess_study
    from easyicu.webserver.research_capability_scope import eligible_family_databases

    requirements = skills.WORKFLOW_GUIDES[row.skill_id].requirements
    notes = skills.package_database_notes(row.to_dict(enabled=True))
    assert {note["decision"]["database"] for note in notes} == {"mimic", "miiv", "eicu", "hirid", "aumc", "sic"}
    for note in notes:
        database = note["decision"]["database"]
        assert note["decision"] == assess_study(research_capability_profile(database), requirements).model_dump(mode="json")
        assert note["real_data_acceptance"] == "not_verified"
        assert all(item["code"] != "family_requirements_missing" for item in note["decision"]["missing"])
    assert {note["decision"]["database"] for note in notes if note["decision"]["status"] != "refusal"} == set(
        eligible_family_databases(requirements.family, requirements=requirements)
    )


def test_database_copy_changes_when_owner_facts_change(monkeypatch):
    original = skills.research_capability_profile
    monkeypatch.setattr(skills, "research_capability_profile", lambda database: original(database).model_copy(
        update={"mortality": None, "mortality_outcomes": ()}) if database == "miiv" else original(database))
    row = next(row for row in WORKFLOWS if row.skill_id == "survival-time-to-event").to_dict(enabled=True)
    note = next(note for note in skills.package_database_notes(row) if note["decision"]["database"] == "miiv")
    assert note["decision"]["status"] == "refusal"
    assert {item["code"] for item in note["decision"]["missing"]} >= {"mortality_status_unavailable", "event_time_unavailable"}
    assert "MIMIC-IV — Not available" in skills._package_skill_markdown("research_workflow", row)
    assert "缺少：事件时间" in skills._package_skill_markdown("research_workflow", row, language=1)


def test_substitution_names_the_new_definition_and_unassessed_inputs_stay_visible(monkeypatch):
    row = WORKFLOWS[0].to_dict(enabled=True)
    english = skills._package_skill_markdown("research_workflow", row)
    chinese = skills._package_skill_markdown("research_workflow", row, language=1)
    assert "death before hospital discharge → death before ICU discharge" in english
    assert "出院前死亡 → 出 ICU 后 72 小时内死亡" in chinese
    guide = skills.WORKFLOW_GUIDES[row["id"]]
    monkeypatch.setitem(skills.WORKFLOW_GUIDES, row["id"], replace(guide, requirements=guide.requirements.model_copy(
        update={"concepts": ("unregistered_demo_measurement",)})))
    assert "Still unassessed:" in skills._package_skill_markdown("research_workflow", row)


@pytest.mark.parametrize("row", WORKFLOWS, ids=lambda row: row.skill_id)
def test_steps_follow_included_modules_actions_and_actual_diagnostics(row):
    from easyicu.research_agent.planning.capability_registry import CAPABILITY_REGISTRY

    capabilities = {item.capability_id: item for item in CAPABILITY_REGISTRY}
    specs = {item.skill_id: item for item in skills.METHOD_SKILLS}
    steps = skills.package_steps(row.to_dict(enabled=True))
    assert tuple(step.skill_id for step in steps) == (*row.included_module_ids, row.skill_id)
    for step in steps:
        assert step.action_ids == specs[step.skill_id].action_ids
        capability = capabilities[step.capability_id]
        assert step.checks == capability.required_diagnostics
        assert step.validator_owner == capability.scientific_validator_owner
        assert all(key in skills._DIAGNOSTIC_LABELS for key in step.checks)


def test_examples_only_describe_public_demo_plans_without_results():
    assert set(skills.WORKFLOW_GUIDES) == {row.skill_id for row in WORKFLOWS}
    for row in WORKFLOWS:
        guide = skills.WORKFLOW_GUIDES[row.skill_id]
        assert guide.demo_source in {"miiv_demo", "eicu_demo"}
        assert all(guide.question) and all(guide.example_plan) and guide.output_files
        for language in (0, 1):
            example = skills._package_example_markdown(row.to_dict(enabled=True), language)
            assert "MIMIC-IV demo" in example or "eICU demo" in example
            assert ("not a recorded run" if language == 0 else "并非已记录的运行") in example
            assert all(name in example for name in guide.output_files)


def test_contract_holds_machine_coordinates_but_reader_files_do_not():
    for row in WORKFLOWS:
        package = skills.method_skill_package(row.skill_id, enabled=True)
        files = {item["path"]: item["content"] for item in package["files"]}
        contract = json.loads(files["references/workflow_contract.md"].split("```json\n")[1].split("\n```")[0])
        assert contract["capability_id"] == row.capability_id
        assert {item["code"] for item in contract["common_stops"]} == {stop.code for stop in skills.PACKAGE_STOPS}
        notes = skills.package_database_notes(row.to_dict(enabled=True))
        assert package["database_notes"] == notes
        assert contract["database_assessment_sha256"] == hashlib.sha256(
            json.dumps(notes, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        for path in ("SKILL.md", "SKILL.zh.md", "references/composition.md", "references/validation_framework.md"):
            assert row.capability_id not in files[path]
            assert not any(code in files[path] for code in row.action_ids)
            assert not any(stop.code in files[path] for stop in skills.PACKAGE_STOPS)
        assert package["overview_paths"] == {"en": "SKILL.md", "zh": "SKILL.zh.md"}


def test_package_digest_changes_when_reader_content_changes(monkeypatch):
    row = WORKFLOWS[0]
    before = skills.method_skill_package(row.skill_id, enabled=True)
    guide = skills.WORKFLOW_GUIDES[row.skill_id]
    monkeypatch.setitem(skills.WORKFLOW_GUIDES, row.skill_id,
                        replace(guide, philosophy=(guide.philosophy[0] + " Read the diagnostics.", guide.philosophy[1])))
    after = skills.method_skill_package(row.skill_id, enabled=True)
    assert after["package_sha256"] != before["package_sha256"]
    for item in after["files"]:
        assert item["sha256"] == hashlib.sha256(item["content"].encode()).hexdigest()


def test_checked_in_landmark_reader_is_generated_from_the_same_typed_content():
    root = Path(skills.__file__).parent / "skill_packages" / "landmark_categorical_association"
    row = next(row for row in WORKFLOWS if row.skill_id == "fixed-landmark-association-study").to_dict(enabled=True)
    for language, suffix in ((0, ""), (1, ".zh")):
        assert (root / f"SKILL{suffix}.md").read_text() == skills._package_skill_markdown("research_workflow", row, language=language)
        assert (root / f"examples/public_demo{suffix}.md").read_text() == skills._package_example_markdown(row, language)
    reporting = "# Reporting references / 报告依据\n\n" + "\n".join(
        skills._reference_lines(row, 0) + [""] + skills._reference_lines(row, 1)) + "\n"
    assert (root / "references/reporting_checklist.md").read_text() == reporting


def test_landmark_bilingual_guides_and_examples_are_in_wheel_data_patterns():
    import fnmatch
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    root = Path(skills.__file__).resolve().parents[3]
    patterns = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "easyicu.research_agent.skill_packages"]
    for path in ("landmark_categorical_association/SKILL.zh.md",
                 "landmark_categorical_association/examples/public_demo.md",
                 "landmark_categorical_association/examples/public_demo.zh.md"):
        assert any(fnmatch.fnmatch(path, pattern) for pattern in patterns), path


@pytest.mark.parametrize("skill_id", ["cohort-characterization-table-one", "prediction.decision_curve"])
def test_supporting_methods_do_not_invent_a_workflow_example_or_database_scope(skill_id):
    package = skills.method_skill_package(skill_id, enabled=True)
    assert package["database_notes"] == []
    for item in package["files"]:
        if item["path"] in {"SKILL.md", "SKILL.zh.md"}:
            assert "examples/public_demo" not in item["content"]
            assert "per-database assessment" in item["content"] or "数据库评估" in item["content"]
