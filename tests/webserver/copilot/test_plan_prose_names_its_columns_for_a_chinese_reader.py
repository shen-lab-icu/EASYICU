"""Plan prose names the columns it quotes for a Chinese reader.

A plan step's intent is the planner's sentence and quotes column ids.  On
pilot1 (10-22) the run review read 「估计 年龄分组 与院内 death 的 sex
调整观察性关联」: the grouped exposure was named, the outcome and the
covariate were not.  The plan-prose owner names every whole identifier that
has a Chinese name (grouping, the plan's Chinese label, else the catalog's),
does not repeat the words before it, and leaves anything without a Chinese
name, an English sentence and an English page as the plan wrote them.  The
plan card, the run review and the analysis report all read steps through it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver import agent_pipeline_runs
from tests.support.node import run_node

STATIC = Path(agent_pipeline_runs.__file__).parent / "static"
GROUPS = {
    "status": "shown",
    "groupings": [{
        "variable": "age_group_x1", "concept": "age", "concept_label_en": "Age", "concept_label_zh": "年龄",
        "status": "labelled", "reference": 1,
        "levels": [{"code": 1, "label": "<65 岁"}, {"code": 2, "label": "≥65 岁"}],
    }],
}
CATALOG = {"death": ["In-hospital Mortality", "院内死亡", "boolean"], "sex": ["Sex", "性别", ""],
           "lact": ["Lactate", "乳酸", "mmol/L"]}


def _named(texts, *, language="zh", labels=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    script = f"""
global.window = global;
window.EU_LANG = {json.dumps(language)};
window.EU_CATALOG = {{ dict: {json.dumps(CATALOG)} }};
require({json.dumps(str(STATIC / "js/screens-agent-exposure-levels.js"))});
require({json.dumps(str(STATIC / "js/screens-agent-plan-prose.js"))});
const tr = (en, zh) => ({json.dumps(language)} === 'zh' ? zh : en);
const levels = window.AGENT_EXPOSURE_LEVELS.reader({{ exposure_groups: {json.dumps(GROUPS)} }}, tr);
const labels = {json.dumps(labels or {})};
process.stdout.write(JSON.stringify({json.dumps(texts)}.map(text => window.AGENT_PLAN_PROSE.namedZh(text, {{ labels, levels }}))));
"""
    result = run_node(node, script)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_pilot_sentence_reads_in_chinese():
    (named,) = _named(["估计 age_group_x1 与院内 death 的 sex 调整观察性关联，作为主要科学结果。"])

    assert named == "估计年龄分组与院内死亡的性别调整观察性关联，作为主要科学结果。"


def test_a_registered_chinese_label_is_used_without_its_definition():
    named = _named(["按 lact_max 分组比较 death"], labels={"lact_max": "最高乳酸（入 ICU 24 小时内）"})

    assert named == ["按最高乳酸分组比较院内死亡"]


def test_what_has_no_chinese_name_keeps_its_words():
    assert _named(["比较 lact_max 与 death，报告 OR 与 95% CI"]) == ["比较 lact_max 与院内死亡，报告 OR 与 95% CI"]
    # A sentence with no Chinese is the plan's own wording; only groups are named.
    assert _named(["Estimate age_group_x1 and death"]) == ["Estimate 年龄分组 and death"]


def test_an_english_page_names_only_the_groups():
    assert _named(["估计 age_group_x1 与 death 的关联"], language="en") == ["估计 Age groups 与 death 的关联"]


def test_every_plan_step_reader_uses_the_owner_after_it_loads():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    owner = index.index("js/screens-agent-plan-prose.js?")
    for reader in ("screens-agent-render.js", "screens-guided-pi-analysis-report.js", "screens-guided-pi-run-answer.js"):
        assert owner < index.index(f"js/{reader}?")
        assert "AGENT_PLAN_PROSE" in (STATIC / "js" / reader).read_text(encoding="utf-8")
