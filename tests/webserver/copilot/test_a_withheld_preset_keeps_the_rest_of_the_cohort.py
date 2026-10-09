"""A setup slot Copilot may not save yet costs nothing else, and is named.

A conversational update withholds what the researcher has not chosen: a
population preset (the reviewed plan proposes a restricted population, and the
choice between all stays and one stay per patient is the researcher's), and an
outcome, exposure or analysis goal the turn names only as intent. It used to
withhold a preset by putting the whole cohort back, so the age and ICU-stay
restrictions written beside it were lost, and it recorded the preset alone; an
outcome took the proposed feature modules and its execution concept with it
unrecorded. In probe 3 (2a, 10-09) the saved cohort was empty, and the reply the
host finalized never said so. Now the update withholds the preset and the
words written to name its population, keeps every other cohort field, lists
each field it does not save with the reason, and the finalized reply names
them. The receipt projects every field the tool accepts, so the same check
reads a real receipt.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.projections import project_study_context
from tests.support.node import run_node

APP_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
)
CURRENT = {
    "id": "study-withheld-preset",
    "revision": 1,
    "active_job_id": None,
    "question": "",
    "cohort": {},
    "outcome": "",
    "primary_exposure": "",
    "analysis_goal": "",
}
WORKFLOWS = {
    "needs_source": {
        "next_action_code": "study_setup_incomplete",
        "missing_setup_fields": ["data_source"],
        "study_setup_receipt": {"configuration": {"data_source": {}}},
    },
    "ready": {
        "next_action_code": "provider_ready_to_generate_plan",
        "missing_setup_fields": [],
        "study_setup_receipt": {"configuration": {"data_source": {"database": "synthetic"}}},
    },
}
# Probe 3's call (14:50:44Z) and the researcher's own words, verbatim.
PROBE_3_MESSAGE = (
    "在 MIMIC-IV 中纳入年龄 ≥ 18 岁、ICU 住院时长 ≥ 48 小时且符合 Sepsis-3 的 ICU 入住，"
    "基于纵向 SOFA 分项或乳酸轨迹进行器官功能障碍轨迹聚类，并描述轨迹类别与结局的关联。"
    "所有轨迹须对齐固定时间锚点与预先设定窗口，显式处理不等长随访、早死、缺失和长度偏倚；"
    "说明特征表示、距离或模型、聚类数选择及重抽样稳定性。轨迹类别不是因果组。"
)
PROBE_3_CALL = {
    "question": PROBE_3_MESSAGE,
    "purpose": (
        "在符合 Sepsis-3 的成年 ICU 入住者中识别器官功能障碍纵向轨迹类别，并描述其与临床结局的关联；"
        "该轨迹分类用于描述性和关联性分析，不作因果分组解释。"
    ),
    "cohort": {
        "preset": "sepsis3",
        "label": "年龄 ≥ 18 岁、ICU 住院时长 ≥ 48 小时且符合 Sepsis-3 的 ICU 入住",
        "age_min": 18,
        "min_icu_los_hours": 48,
    },
    "outcome": "轨迹类别与临床结局的关联，具体结局定义由候选研究计划提出并供研究者审阅。",
    "analysis_goal": (
        "进行器官功能障碍纵向轨迹聚类，比较轨迹类别的结局分布或关联；固定时间锚点和预先设定窗口，"
        "处理不等长随访、早死、缺失及长度偏倚，并报告特征表示、距离或模型、聚类数选择和重抽样稳定性。"
        "轨迹类别不解释为因果组。"
    ),
    "comparator": "不同器官功能障碍轨迹类别之间的描述性结局比较；不作因果比较。",
}
MESSAGE = "成年 ICU 入住中，condition-x 和 marker-y 与 outcome-z 有什么关系？"
QUESTION = "How do condition-x and marker-y relate to outcome-z in adult ICU stays?"


def _update(monkeypatch, params: dict[str, Any], *, current=None, message=MESSAGE, workflow="needs_source"):
    current = dict(current or CURRENT)
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module, "_workflow_snapshot", lambda *_args, **_kwargs: dict(WORKFLOWS[workflow])
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**current, **raw, "revision": 2},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-withheld-preset",
            binding=AuthorityBinding(study_context_id=current["id"], study_revision=current["revision"]),
        ),
        user_message=message,
        allowed_actions={"configure"},
    )
    return tool_module.execute_tool("easyicu_update_study_context", params, context), writes


_NESTED = {"cohort", "execution_concepts", "time_window", "analysis_design", "trajectory_design"}


def _dropped_without_a_record(params: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Each proposed field neither saved as proposed nor listed as not saved.

    It reads only the receipt: the study it projects and its omissions.
    """

    study = result["details"]["study"]
    omitted = set(result["details"]["omitted_unconfirmed_fields"])
    silent: list[str] = []

    def check(field: str, proposed: Any, saved: Any) -> None:
        if field in omitted:
            return
        if isinstance(proposed, list):
            same = sorted(map(str, proposed)) == sorted(map(str, saved or []))
        elif isinstance(proposed, str):
            same = proposed.strip() == str(saved or "").strip()
        else:
            same = proposed == saved
        if not same:
            silent.append(field)

    for key, value in params.items():
        if key in _NESTED and isinstance(value, dict):
            for field, proposed in value.items():
                check(f"{key}.{field}", proposed, (study.get(key) or {}).get(field))
        else:
            check(key, value, study.get(key))
    return silent


def _omitted(result: dict[str, Any]) -> list[tuple[str, str]]:
    return [(item["field"], item["code"]) for item in result["details"]["unconfirmed_omissions"]]


def test_probe_3_keeps_its_restrictions_and_names_what_it_withholds(monkeypatch):
    result, writes = _update(monkeypatch, PROBE_3_CALL, message=PROBE_3_MESSAGE)

    assert (result["status"], result["code"]) == ("ok", "study_context_updated")
    assert writes[0]["cohort"] == {"age_min": 18, "min_icu_los_hours": 48}
    assert _omitted(result) == [
        ("cohort.preset", "study_cohort_population_requires_plan"),
        ("cohort.label", "study_cohort_population_requires_plan"),
        ("outcome", "study_primary_outcome_confirmation_required"),
        ("analysis_goal", "study_analysis_goal_confirmation_required"),
    ]
    assert _dropped_without_a_record(PROBE_3_CALL, result) == []


BRANCHES = {
    "population_preset": (
        {
            "question": QUESTION,
            "cohort": {
                "preset": "sepsis3",
                "label": "Adult ICU stays with condition-x",
                "review": "adults with condition-x",
                "age_min": 18,
                "exclusion_statement": "stays transferred from another ICU",
            },
        },
        {"age_min": 18, "exclusion_statement": "stays transferred from another ICU"},
        ["cohort.preset", "cohort.label", "cohort.review"],
        "study_cohort_population_requires_plan",
    ),
    "all_stays_preset": (
        {"question": QUESTION, "cohort": {"preset": "adult_all", "label": "Adult ICU stays", "min_icu_los_hours": 24}},
        {"min_icu_los_hours": 24},
        ["cohort.preset", "cohort.label"],
        "study_cohort_all_stays_confirmation_required",
    ),
    "first_stay_preset": (
        {"question": QUESTION, "cohort": {"preset": "adult_first", "label": "First adult ICU stays", "age_max": 90}},
        {"age_max": 90},
        ["cohort.preset", "cohort.label"],
        "study_cohort_first_stay_confirmation_required",
    ),
    "outcome": (
        {
            "question": QUESTION,
            "outcome": "Death within 28 days of ICU admission",
            "execution_concepts": {"outcome": "death"},
            "modules": ["demographics", "outcome"],
        },
        None,
        ["outcome", "execution_concepts.outcome", "modules"],
        "study_primary_outcome_confirmation_required",
    ),
    "exposure": (
        {
            "question": QUESTION,
            "primary_exposure": "Peak marker-y in the first 24 hours",
            "execution_concepts": {"primary_exposure": "lact"},
        },
        None,
        ["primary_exposure", "execution_concepts.primary_exposure"],
        "study_primary_exposure_confirmation_required",
    ),
    "analysis_goal": (
        {"question": QUESTION, "analysis_goal": "Estimate the adjusted association of marker-y with outcome-z."},
        None,
        ["analysis_goal"],
        "study_analysis_goal_confirmation_required",
    ),
}


@pytest.mark.parametrize("branch", list(BRANCHES))
def test_each_withheld_slot_is_listed_and_everything_else_is_saved(monkeypatch, branch):
    params, saved_cohort, fields, code = BRANCHES[branch]
    result, writes = _update(monkeypatch, params)

    assert (result["status"], result["code"]) == ("ok", "study_context_updated")
    assert writes[0]["question"] == QUESTION
    if saved_cohort is not None:
        assert writes[0]["cohort"] == saved_cohort
    assert _omitted(result) == [(field, code) for field in fields]
    assert _dropped_without_a_record(params, result) == []
    # The summary the model reads names each field and why.
    for field in fields:
        assert field in result["summary"]
    assert f"({code}: " in result["summary"]
    assert "The rest of the setup is saved." in result["summary"]
    assert "Tell the researcher, one line each, which of these were not saved and why." in result["summary"]


def test_a_cohort_restriction_alone_is_a_change_to_save(monkeypatch):
    """Withholding the preset leaves the age bound, so the update saves it."""

    params = {"cohort": {"preset": "sepsis3", "label": "Adults with condition-x", "age_min": 18}}
    result, writes = _update(monkeypatch, params)

    assert (result["status"], writes[0]["cohort"]) == ("ok", {"age_min": 18})
    assert [field for field, _ in _omitted(result)] == ["cohort.preset", "cohort.label"]
    assert _dropped_without_a_record(params, result) == []


def test_a_restriction_also_saves_beside_an_unconfirmed_outcome(monkeypatch):
    params = {"cohort": {"preset": "sepsis3", "age_min": 18}, "outcome": "Death within 28 days"}
    result, writes = _update(monkeypatch, params)

    assert (result["status"], writes[0]["cohort"]) == ("ok", {"age_min": 18})
    assert [field for field, _ in _omitted(result)] == ["cohort.preset", "outcome"]


@pytest.mark.parametrize(
    "cohort",
    [{"preset": "sepsis3"}, {"preset": "sepsis3", "label": "Adults with condition-x", "review": "condition-x"}],
    ids=["preset_alone", "preset_and_its_words"],
)
def test_a_preset_with_nothing_else_to_save_is_refused_whole(monkeypatch, cohort):
    result, writes = _update(monkeypatch, {"cohort": cohort})

    assert (result["status"], result["code"], writes) == (
        "blocked",
        "study_cohort_population_requires_plan",
        [],
    )


def test_words_the_study_already_holds_are_not_withheld_news(monkeypatch):
    current = {**CURRENT, "cohort": {"label": "Adult ICU stays"}}
    params = {"cohort": {"preset": "sepsis3", "label": "Adult ICU stays", "age_min": 18}}
    result, writes = _update(monkeypatch, params, current=current)

    assert writes[0]["cohort"] == {"label": "Adult ICU stays", "age_min": 18}
    assert [field for field, _ in _omitted(result)] == ["cohort.preset"]


def _finalized_reply(result: dict[str, Any], params: dict[str, Any], language: str) -> str:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    module = APP_DIR / "src" / "post-tool-finalization.mjs"
    script = f"""
      import {{ hostPostToolFinalization }} from {json.dumps(module.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const user = {{ role: 'user', content: [{{ type: 'text', text: {json.dumps(PROBE_3_MESSAGE)} }}] }};
      const assistant = {{ role: 'assistant', content: [{{
        type: 'toolCall', id: 'call-update', name: 'easyicu_update_study_context',
        arguments: {json.dumps(params)},
      }}] }};
      const result = {{ role: 'toolResult', toolCallId: 'call-update',
        toolName: 'easyicu_update_study_context', isError: false, content: [],
        details: {json.dumps(result)} }};
      const stream = hostPostToolFinalization(model, {{ messages: [user, assistant, result] }}, {json.dumps(language)});
      if (!stream) throw new Error('expected a host-finalized reply');
      const message = await stream.result();
      console.log(JSON.stringify(message.content[0].text));
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


@pytest.mark.parametrize("workflow", ["needs_source", "ready"])
def test_the_finalized_reply_names_each_withheld_setting(monkeypatch, workflow):
    result, _ = _update(monkeypatch, PROBE_3_CALL, message=PROBE_3_MESSAGE, workflow=workflow)

    zh = _finalized_reply(result, PROBE_3_CALL, "zh")
    assert "这次没有保存的设置：" in zh
    assert "- 人群限定没有保存为研究设置（人群预设和为它写的名称）" in zh
    assert "- 主要结局没有保存：" in zh
    assert "- 分析目标没有保存：" in zh
    en = _finalized_reply(result, PROBE_3_CALL, "en")
    assert "Not saved this time:" in en
    assert "- The population restriction was not saved to the study" in en
    assert "- The primary outcome was not saved:" in en
    assert "- The analysis goal was not saved:" in en
    if workflow == "needs_source":
        assert zh.index("这次没有保存的设置") < zh.index("**下一步：**")


def test_the_reply_folds_the_withheld_modules_into_their_slot(monkeypatch):
    params = BRANCHES["outcome"][0]
    result, _ = _update(monkeypatch, params, workflow="ready")

    zh = _finalized_reply(result, params, "zh")
    assert "- 主要结局没有保存：" in zh and "本轮一起提议的特征模块也没有保存。" in zh
    assert zh.count("\n- ") == 1


# The receipt projects every field the tool accepts --------------------------

_MAIN = APP_DIR / "src" / "main.mjs"
_ACTIONS = {"bind_active_export", "bind_source_id"}


def _object_keys(source: str, opener: str) -> set[str]:
    """The property names of the first ``Type.Object({...})`` after ``opener``."""

    index = source.index("Type.Object({", source.index(opener)) + len("Type.Object({")
    depth, quote, top = 0, "", []
    while True:
        char = source[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            top.append(char)
        index += 1
    return set(re.findall(r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", "".join(top)))


def _schemas() -> dict[str, set[str]]:
    source = _MAIN.read_text(encoding="utf-8")
    return {
        "tool": _object_keys(source, 'name: "easyicu_update_study_context"'),
        "cohort": _object_keys(source, "const studyCohort = "),
        "time_window": _object_keys(source, "const studyWindow = "),
        "execution_concepts": _object_keys(source, "const executionConcepts = "),
        "analysis_design": _object_keys(source, "const analysisDesign = "),
        "trajectory_design": _object_keys(source, "const trajectoryDesign = "),
        "sensitivity_spec": _object_keys(source, "const sensitivitySpec = "),
    }


def _a_value(key: str) -> Any:
    if key in {"include_diagnoses", "exclude_diagnoses", "coordinate_concepts", "descriptive_only_concepts",
               "execution_variables", "modules", "covariates"}:
        return ["alpha", "beta"]
    if key in {"exclude_readmissions", "require_alive_at_landmark", "exclude_negative_event_times"}:
        return True
    if key in {"stability_sample_fraction", "minimum_cluster_fraction", "minimum_mean_stability"}:
        return 0.5
    if key.endswith(("_hours", "_min", "_max", "_windows", "_resamples", "max_patients")):
        return 24
    return f"value-{key}"


def test_the_receipt_projects_every_field_the_tool_accepts():
    schemas = _schemas()
    assert {"cohort", "trajectory_design", "covariate_rationales"} <= schemas["tool"]
    assert {"review", "comparison", "source_type", "include_diagnoses"} <= schemas["cohort"]
    nested = {name: {key: _a_value(key) for key in schemas[name]} for name in _NESTED}
    nested["analysis_design"] = {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust",
        "cluster_unit": "patient",
    }
    assert set(nested["analysis_design"]) == schemas["analysis_design"]
    study = {"id": "study-every-field", "revision": 3}
    for key in schemas["tool"] - _ACTIONS:
        if key in nested:
            study[key] = nested[key]
        elif key == "sensitivity_specs":
            study[key] = [{field: _a_value(field) for field in schemas["sensitivity_spec"]}]
        elif key in {"covariate_rationales", "covariate_operationalizations", "covariate_temporal_roles"}:
            study[key] = {"age": f"{key} for age"}
        elif key == "confirmations":
            study[key] = {"repeated_icu_stays_retained": True}
        else:
            study[key] = _a_value(key)

    projected = project_study_context(study)

    assert schemas["tool"] - _ACTIONS <= set(projected)
    for name in _NESTED:
        assert schemas[name] <= set(projected[name]), name
    assert schemas["sensitivity_spec"] <= set(projected["sensitivity_specs"][0])
