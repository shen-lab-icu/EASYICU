"""A question-requirements record reads requirement by requirement.

The planning owner keeps two records of what the question asks of a plan: the
planning record, judged on the plan the Planner compiled, and the judgment of
the plan offered for review, which an approval rests on.  A card that refuses
approval links the judgment, and opened as a run file it read "0 fields".  The
shared artifact reader now shows each requirement in the question's words,
how the host judged the plan on it, why, and whether EasyICU verified it; on a
route that lists no requirements, each concept the question names and what
reads it.  A record file it cannot read says so instead of showing an empty
judgment.

The rows are the planning owner's own (``judge_question_requirements``,
``outline_route_unstated``) on a small plan, and each record is opened as the
run-file route opens it (``agent_runs.read_run_artifact``), so the reader is
held to the shape the owner writes and the route serves.  Synthetic plans and
generic wording only.
"""

from __future__ import annotations

import html as html_module
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCapabilityGap,
)
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
    QUESTION_REQUIREMENTS_SCHEMA_VERSION,
    JudgedRequirement,
    NamedQuestionConcept,
    QuestionRequirement,
    UnstatedConcept,
    judge_question_requirements,
    outline_route_unstated,
)
from easyicu.webserver import agent_runs
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
UNCHECKABLE = "the host has no typed evidence about estimand_unsupported"


def _step(step_id: str, inputs: Sequence[str], action: str, role: str = "primary"):
    return SimpleNamespace(
        step_id=step_id,
        planned_analysis_role=role,
        inputs=list(inputs),
        scientific_action_id=action,
    )


#: A model that reads the severity score as a predictor, and a subgroup step.
PLAN = SimpleNamespace(
    steps=[
        _step("s2", ["score_pred", "lactate"], "prediction.logistic_model"),
        _step("s3", ["sofa"], "association.effect_modification", "secondary"),
    ],
    cohort=SimpleNamespace(
        inclusion=[SimpleNamespace(concept_id="age")], exclusion=[]
    ),
)

REQUIREMENTS = [
    QuestionRequirement(
        id="r1",
        kind="benchmark",
        quote="compare it with the admission severity score",
        concepts=["score_pred"],
        coverage="plan",
    ),
    QuestionRequirement(
        id="r2",
        kind="subgroup",
        quote="in stays with a high SOFA score",
        concepts=["sofa"],
        coverage="plan",
    ),
    QuestionRequirement(id="r3", kind="estimand", quote="its AUROC", coverage="plan"),
    QuestionRequirement(
        id="r4",
        kind="analysis",
        quote="excluding readmissions",
        concepts=["readmission"],
        coverage="plan",
    ),
    QuestionRequirement(
        id="r5",
        kind="estimand",
        quote="its net benefit",
        coverage="capability_gap",
        gap=ProgressiveCapabilityGap(
            requirement="estimand_unsupported",
            element="analysis",
            detail="This plan cannot estimate net benefit at named thresholds.",
        ),
    ),
    QuestionRequirement(
        id="r6",
        kind="definition",
        quote="septic shock",
        concepts=["lactate"],
        coverage="definition_only",
        note="the cohort's shock definition",
    ),
]


def _judged(*, family_template: bool = True) -> tuple[JudgedRequirement, ...]:
    return judge_question_requirements(
        REQUIREMENTS,
        plan=PLAN,
        family_template=family_template,
        relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: ("unverifiable", UNCHECKABLE),
    )


def _record(
    judged: Sequence[JudgedRequirement] = (),
    unstated: Sequence[UnstatedConcept] = (),
    *,
    route: str = "family_template",
    review: bool = True,
) -> dict[str, Any]:
    """A record with the owner's envelope keys and the owner's rows."""

    head: dict[str, Any] = (
        {
            "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
            "plan_sha256": "a" * 64,
            "planning_record_sha256": "b" * 64,
        }
        if review
        else {
            "schema_version": QUESTION_REQUIREMENTS_SCHEMA_VERSION,
            "compiled_plan_sha256": "a" * 64,
        }
    )
    return {
        **head,
        "route": route,
        "judged": [entry.row() for entry in judged],
        "unstated": [item.row() for item in unstated],
        "coverage": {},
    }


def _opened(record: dict[str, Any], name: str, run_dir: Path) -> Any:
    """The record as the run-file route serves it, from a run directory."""

    (run_dir / name).write_text(json.dumps(record), encoding="utf-8")
    loaded = agent_runs.read_run_artifact(str(run_dir), name)
    assert loaded["ok"], loaded
    return loaded["payload"]


def _render(name: str, payload: Any, *, lang: str = "zh") -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = f"""
      global.window = {{ t: (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en, icon: () => '' }};
      eval({json.dumps((STATIC / "js" / "html-escape.js").read_text(encoding="utf-8"))});
      eval({json.dumps((STATIC / "js" / "screens-agent-reader-vocab.js").read_text(encoding="utf-8"))});
      eval({json.dumps((STATIC / "js" / "screens-agent-question-requirements.js").read_text(encoding="utf-8"))});
      eval({json.dumps((STATIC / "js" / "screens-agent-render.js").read_text(encoding="utf-8"))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView({json.dumps(name)}, {json.dumps(payload)}));
    """
    return run_node(node, script, check=True).stdout


def _tables(html: str) -> list[list[list[str]]]:
    """Each table's body rows, cell text unescaped."""

    tables = []
    for body in re.findall(r"<tbody>(.*?)</tbody>", html, re.S):
        tables.append(
            [
                [html_module.unescape(cell) for cell in re.findall(r"<td>(.*?)</td>", row, re.S)]
                for row in re.findall(r"<tr>(.*?)</tr>", body, re.S)
            ]
        )
    return tables


def test_each_requirement_reads_with_its_disposition_reason_and_source(
    tmp_path: Path,
) -> None:
    judged = _judged()
    record = _record(judged)
    served = _opened(record, QUESTION_REQUIREMENTS_REVIEW_FILENAME, tmp_path)

    html = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, served)

    # The route serves the record as the owner wrote it.
    assert served == record

    assert "0 字段" not in html
    assert "在提交审阅的计划上的判定，批准依据这一份" in html
    assert "3 项没有回答或做不到" in html
    (rows,) = _tables(html)
    assert rows == [
        [
            "r1 · 与已有评分或模型比较：「compare it with the admission severity score」",
            "这份计划做不到 · 不能批准",
            "这份计划所用的模板没有在同一批行上把模型与已有评分或模型比较的步骤",
            "EasyICU 已核实",
        ],
        [
            "r2 · 指定亚组：「in stays with a high SOFA score」",
            "已回答",
            "由步骤 s3 回答",
            "EasyICU 已核实",
        ],
        [
            "r3 · 指定估计量或指标：「its AUROC」",
            "计划声明已回答",
            "它没有指明概念",
            "计划声明，EasyICU 未核实",
        ],
        [
            "r4 · 其他指定分析：「excluding readmissions」",
            "没有回答 · 不能批准",
            "它的概念并非都有分析步骤读取（readmission）",
            "EasyICU 已核实",
        ],
        [
            "r5 · 指定估计量或指标：「its net benefit」",
            "这份计划做不到 · 不能批准",
            "规划的说明：This plan cannot estimate net benefit at named thresholds.",
            "规划声明，EasyICU 无法核对",
        ],
        [
            "r6 · 定义：「septic shock」",
            "只用于定义",
            "计划说它只用于定义：the cohort's shock definition",
            "计划声明，EasyICU 未核实",
        ],
    ]
    # The verification column is the owner's verdict, row by row.
    assert [row[3] == "EasyICU 已核实" for row in rows] == [
        entry.verified_by_host for entry in judged
    ]


def test_english_readers_keep_the_hosts_own_gap_detail(tmp_path: Path) -> None:
    judged = _judged()
    served = _opened(_record(judged), QUESTION_REQUIREMENTS_REVIEW_FILENAME, tmp_path)

    (rows,) = _tables(_render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, served, lang="en"))

    assert rows[0][1] == "This plan cannot do it · stops approval"
    assert rows[0][2] == judged[0].gap.detail
    assert rows[0][3] == "Verified by EasyICU"


def test_a_benchmark_read_as_a_predictor_is_not_its_answer(tmp_path: Path) -> None:
    judged = _judged(family_template=False)
    served = _opened(_record(judged), QUESTION_REQUIREMENTS_REVIEW_FILENAME, tmp_path)

    (rows,) = _tables(_render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, served))

    assert judged[0].disposition == "not_covered"
    assert rows[0][1:] == [
        "没有回答 · 不能批准",
        "没有步骤在同一批行上把模型与它比较（score_pred）；读取它的分析步骤：s2",
        "EasyICU 已核实",
    ]


def test_only_the_judgment_of_the_plan_under_review_says_approval_stops(
    tmp_path: Path,
) -> None:
    served = _opened(_record(_judged(), review=False), QUESTION_REQUIREMENTS_FILENAME, tmp_path)

    html = _render(QUESTION_REQUIREMENTS_FILENAME, served)

    (rows,) = _tables(html)
    assert "在规划所得计划上的判定。计划在审阅前还会加工，提交审阅的计划另有一份判定" in html
    assert [row[1] for row in rows] == [
        "这份计划做不到",
        "已回答",
        "计划声明已回答",
        "没有回答",
        "这份计划做不到",
        "只用于定义",
    ]


def test_a_route_without_requirements_lists_each_named_concept_and_what_reads_it(
    tmp_path: Path,
) -> None:
    named = [
        NamedQuestionConcept(concepts=["score_pred"], evidence="the admission severity score"),
        NamedQuestionConcept(concepts=["age"], evidence="adult"),
        NamedQuestionConcept(concepts=["readmission"], evidence="readmissions"),
        NamedQuestionConcept(concepts=["death"], evidence="in-hospital mortality"),
    ]
    unstated = outline_route_unstated(named, plan=PLAN, sealed=("death",))

    served = _opened(
        _record((), unstated, route="outline"),
        QUESTION_REQUIREMENTS_REVIEW_FILENAME,
        tmp_path,
    )

    html = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, served)

    assert "3 项待核对" in html
    # The outline route judges no requirement, so only the concepts are shown.
    (rows,) = _tables(html)
    assert rows == [
        ["「the admission severity score」", "score_pred", "s2"],
        ["「adult」", "age", "人群条件"],
        ["「readmissions」", "readmission", "没有步骤读取，请核对计划是否遗漏了它"],
    ]


def test_a_record_that_asks_nothing_beyond_the_design_says_so() -> None:
    family = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, _record())
    outline = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, _record(route="outline"))

    assert "题面没有在研究设计之外提出要求。" in family
    assert "题面提到的概念都已在研究设计中。" in outline
    assert "没有待处理的要求" in family and "没有待处理的要求" in outline


@pytest.mark.parametrize("payload", [{}, {"schema_version": "easyicu.other/1"}, []])
def test_a_record_file_that_cannot_be_read_says_so(payload: Any) -> None:
    html = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, payload)

    assert "这份记录无法按题面要求读取，未显示任何判定" in html
    assert "0 字段" not in html
    assert "<tbody>" not in html


def test_question_and_model_text_is_clipped_and_escaped() -> None:
    (row,) = _record(_judged()[:1])["judged"]
    record = _record()
    record["judged"] = [
        {
            **row,
            "quote": "<img src=x onerror=alert(1)> " + "q" * 400,
            "disposition": "definition_only",
            "note": "n" * 900,
        }
    ]

    html = _render(QUESTION_REQUIREMENTS_REVIEW_FILENAME, record)

    assert "<img" not in html and "&lt;img src=x onerror=alert(1)&gt;" in html
    ((requirement, _, reason, _),) = _tables(html)[0]
    assert requirement.endswith("…」") and len(requirement.split("「", 1)[1]) == 241
    assert reason == "计划说它只用于定义：" + "n" * 299 + "…"


def test_other_artifacts_are_not_read_as_question_requirements() -> None:
    html = _render("cohort_summary.json", {"summary": "1,234 stays", "judged": []})

    assert "题面要求" not in html
    assert "可读产物摘要" in html


def test_the_reader_loads_before_the_renderer_that_asks_it() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    renderer = (STATIC / "js" / "screens-agent-render.js").read_text(encoding="utf-8")

    owner = index.index("js/screens-agent-question-requirements.js?v=")
    assert index.index("js/screens-agent-reader-vocab.js?v=") < owner
    assert owner < index.index("js/screens-agent-render.js?v=")
    assert "window.AGENT_QUESTION_REQUIREMENTS.view(n, p, { artifactTable, esc, context })" in renderer
