"""A run's exposure groups are named in the study's words wherever they are shown.

A study can group a concept's value into named levels; the analysis models
each level by a code (1, 2, 3), and the plan, its tables and its estimates
carry the codes.  The host reads each applied grouping from the run's
grouping record (``recorded_exposure_group_labels``: ``[]`` when the run made
none, ``None`` when the record cannot be read) and projects it with the
study's label for each code (``exposure_groups``): on the plan review card,
in the run's context for the result readers, and beside a Table 1 the column
it is grouped by.

Every reader then names a grouped value by the record's label for that
variable's code, and a grouped column by its reader name ("乳酸分组"); a code
the record does not label keeps its own text, and a record that cannot be
read is said to be unreadable, never shown as a plan without groups.

Synthetic records only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.exposure_group_notes import (
    UNREADABLE_REASON,
    project_exposure_groups,
)
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node
from tests.webserver.copilot.research_workflow_fixtures import complete_study

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
JS = STATIC / "js"
UNREADABLE_ZH = "分组记录读不出，分组按编码显示，各编码的含义无法核对。"


def _grouping(**extra: Any) -> dict[str, Any]:
    return {
        "variable": "lact_max_group",
        "concept": "lact",
        "scale": "ordinal",
        "status": "labelled",
        "levels": [
            {"code": 1, "group": "g1", "label": "乳酸 <2.5", "unmeasured": False,
             "rule": "Lactate over 0-24 h after ICU admission: maximum < 2.5 mmol/L"},
            {"code": 2, "group": "g2", "label": "乳酸 ≥2.5", "unmeasured": False,
             "rule": "Lactate over 0-24 h after ICU admission: maximum >= 2.5 mmol/L"},
            {"code": 3, "group": "unmeasured", "label": "前 24 小时未测乳酸", "unmeasured": True,
             "rule": "Lactate over 0-24 h after ICU admission: no value recorded"},
        ],
        "reference": 1,
        "contrast": 2,
        "groupings_record_sha256": "a" * 64,
        **extra,
    }


SHOWN = project_exposure_groups([_grouping()])
UNREADABLE = project_exposure_groups(None)


# --- the projection ---------------------------------------------------------------


def test_a_grouping_is_projected_with_its_concepts_names() -> None:
    assert SHOWN is not None and SHOWN["status"] == "shown" and SHOWN["reason_code"] is None
    (row,) = SHOWN["groupings"]

    assert row["variable"] == "lact_max_group"
    assert (row["concept"], row["concept_label_en"], row["concept_label_zh"]) == ("lact", "Lactate", "乳酸")
    assert (row["scale"], row["status"], row["reference"], row["contrast"]) == ("ordinal", "labelled", 1, 2)
    assert [level["label"] for level in row["levels"]] == ["乳酸 <2.5", "乳酸 ≥2.5", "前 24 小时未测乳酸"]
    assert [level["unmeasured"] for level in row["levels"]] == [False, False, True]


def test_a_record_without_labels_keeps_its_codes_only() -> None:
    (row,) = project_exposure_groups([_grouping(status="codes_only")])["groupings"]

    assert [level["label"] for level in row["levels"]] == ["", "", ""]


def test_a_run_without_groupings_adds_nothing_and_an_unreadable_record_says_so() -> None:
    assert project_exposure_groups([]) is None
    assert UNREADABLE == {"status": "unavailable", "reason_code": UNREADABLE_REASON, "groupings": []}


@pytest.mark.parametrize(
    "broken",
    [
        {"scale": "interval"},
        {"status": "guessed"},
        {"variable": ""},
        {"levels": [{"code": 1, "label": "only"}]},
        {"levels": [{"code": 1, "label": "a"}, {"code": 1, "label": "b"}]},
        {"levels": [{"code": 1, "label": "a"}, {"code": "2", "label": "b"}]},
        {"levels": [{"code": 1, "label": "a"}, {"code": 2, "label": ""}]},
        # The compared levels are measured ones of the grouping.
        {"contrast": 3},
        {"reference": 9},
        {"levels": [{"code": index, "label": f"g{index}"} for index in range(1, 9)]},
    ],
)
def test_a_record_with_a_malformed_grouping_is_unreadable_as_a_whole(broken: Mapping[str, Any]) -> None:
    # Keeping the well-formed rows would show a plan with fewer groupings
    # than its record states.
    assert project_exposure_groups([_grouping(variable="sofa_group", concept="sofa"), _grouping(**broken)]) == UNREADABLE


def test_no_more_groupings_than_a_study_can_state() -> None:
    rows = [_grouping(variable=f"lact_group_{index}") for index in range(4)]

    assert project_exposure_groups(rows) == UNREADABLE
    assert project_exposure_groups(rows[:3])["status"] == "shown"
    assert project_exposure_groups([_grouping(), _grouping()]) == UNREADABLE
    assert project_exposure_groups({"variable": "x"}) == UNREADABLE


# --- the plan review summary ------------------------------------------------------


def _snapshot(review_extra: Mapping[str, Any]):
    study = complete_study()
    digest = study_context_owner.scientific_configuration_sha256(dict(study))
    run = {
        "run_id": "run-groups",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
    }
    review = {
        "run_id": "run-groups",
        "resumable_here": True,
        "scientific_configuration_sha256": digest,
        "budget_mode": "full_reviewed",
        "requests": [
            {
                "review_id": "review-0",
                "kind": "scientific_stop",
                "summary": "Review the plan.",
                "authority_sha256": "b" * 64,
                "reason_code": "operator_plan_approval_required",
                "approval_allowed": True,
            }
        ],
        "plan_approval_allowed": True,
        "scientific_plan_review": {
            "status": "ready_for_approval",
            "approval_allowed": True,
            "score": 90,
            "findings": [],
        },
        **review_extra,
    }
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=run,
        plan_review_authority=review,
    )


def test_the_review_summary_carries_the_groupings_or_their_unreadable_record() -> None:
    def summary(extra: Mapping[str, Any]) -> Mapping[str, Any]:
        return _snapshot(extra).plan_review_summary

    shown = summary({"exposure_group_labels": [_grouping()], "exposure_group_labels_recorded": True})
    assert [row["variable"] for row in shown["exposure_groups"]["groupings"]] == ["lact_max_group"]
    unreadable = {"exposure_group_labels": None, "exposure_group_labels_recorded": True}
    assert summary(unreadable)["exposure_groups"] == UNREADABLE
    # A run that made no grouping, or wrote no record, adds nothing.
    for extra in (
        {"exposure_group_labels": [], "exposure_group_labels_recorded": True},
        {"exposure_group_labels": None, "exposure_group_labels_recorded": False},
        {},
    ):
        assert "exposure_groups" not in summary(extra)


# --- the host's run projection ----------------------------------------------------


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: Any) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    read: list[Path] = []

    def recorded_labels(directory: Path) -> Any:
        read.append(directory)
        return recorded

    monkeypatch.setattr(agent_pipeline_runs, "recorded_exposure_group_labels", recorded_labels)
    wrapper = tmp_path / "wrapper"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=SimpleNamespace(
            selection=SimpleNamespace(selected_concepts=[]),
            coverage=SimpleNamespace(sufficient=True),
            materialized_concepts=[],
        ),
        run_dir=run_dir,
    )
    # The pipeline run directory is the one the grouping record is read from.
    assert read and all(directory == run_dir for directory in read)
    return json.loads((wrapper / "run_context.json").read_text(encoding="utf-8"))


def test_the_run_context_carries_the_groupings_its_readers_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _project(tmp_path, monkeypatch, [_grouping()])["exposure_groups"] == SHOWN
    assert _project(tmp_path, monkeypatch, None)["exposure_groups"] == UNREADABLE
    assert "exposure_groups" not in _project(tmp_path, monkeypatch, [])


def test_a_grouped_table_one_carries_the_column_it_is_grouped_by(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    evidence = run_dir / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "table_one.csv").write_text(
        "variable,group,group_order,denominator_n\nage,1,1,40\nage,2,2,60\n", encoding="utf-8"
    )
    (evidence / "estimates.csv").write_text("exposure,estimate\nlact_max_group,1.4\n", encoding="utf-8")
    (evidence / "evidence_index.json").write_text(
        json.dumps([
            {"kind": "table", "evidence_id": "t1", "description": "Table 1.",
             "relative_path": "evidence/table_one.csv", "produced_by_step": "baseline"},
            {"kind": "table", "evidence_id": "t2", "description": "Estimates.",
             "relative_path": "evidence/estimates.csv", "produced_by_step": "primary_model"},
        ]),
        encoding="utf-8",
    )
    plan = {"steps": [
        {"step_id": "baseline", "table_one_spec": {"group_by": "lact_max_group"}},
        {"step_id": "primary_model", "method": "logistic_regression"},
    ]}

    tables = {table["name"]: table for table in agent_pipeline_runs._table_projection(run_dir, plan=plan)["tables"]}

    assert tables["table_one.csv"]["group_by"] == "lact_max_group"
    assert "group_by" not in tables["estimates.csv"]
    assert "group_by" not in agent_pipeline_runs._table_projection(run_dir)["tables"][0]


# --- the shared reader ------------------------------------------------------------


def _node(script: str, *files: str, args: tuple[str, ...] = ()) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    # The owners register on ``window``, which a script may replace after them.
    loader = "".join(f"require({json.dumps(str((JS / name).resolve()))});\n" for name in files)
    result = run_node(node, "global.window = global;\n" + loader + script, *args, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _reader(context: Any, call: str, language: str = "zh") -> Any:
    script = f"""
const zh = {json.dumps(language)} === 'zh';
const reader = window.AGENT_EXPOSURE_LEVELS.reader({json.dumps(context)}, (en, value) => (zh ? value : en));
process.stdout.write(JSON.stringify({call}));
"""
    return json.loads(_node(script, "screens-agent-exposure-levels.js"))


CONTEXT = {"exposure_groups": SHOWN}


def test_a_grouped_value_is_named_by_its_records_label() -> None:
    assert _reader(CONTEXT, "[reader.levelName('lact_max_group', '2'), reader.levelName('lact_max_group', 1), reader.levelName('lact_max_group', '3.0')]") == [
        "乳酸 ≥2.5", "乳酸 <2.5", "前 24 小时未测乳酸",
    ]
    assert _reader(CONTEXT, "[reader.variableName('lact_max_group'), reader.contrastName('lact_max_group', '2 vs 1')]") == [
        "乳酸分组", "乳酸 ≥2.5 vs 乳酸 <2.5",
    ]
    assert _reader(CONTEXT, "reader.variableName('lact_max_group')", "en") == "Lactate groups"


def test_a_value_the_record_does_not_label_keeps_its_own_text() -> None:
    names = "[reader.levelName('lact_max_group', '4'), reader.levelName('lact_max_group', 'high'), reader.levelName('sofa', '1'), reader.variableName('sofa'), reader.contrastName('lact_max_group', '4 vs 1')]"

    assert _reader(CONTEXT, names) == ["", "", "", "", ""]
    codes_only = {"exposure_groups": project_exposure_groups([_grouping(status="codes_only")])}
    assert _reader(codes_only, "[reader.levelName('lact_max_group', '1'), reader.variableName('lact_max_group')]") == ["", "乳酸分组"]
    # The reader goes by the record's status, not by a label a codes-only row carries.
    carried = project_exposure_groups([_grouping(status="codes_only")])
    carried["groupings"][0]["levels"][0]["label"] = "乳酸 <2.5"
    assert _reader({"exposure_groups": carried}, "[reader.levelName('lact_max_group', '1')]") == [""]
    assert _reader({}, "[reader.unreadable, reader.levelName('lact_max_group', '1'), reader.unreadableText()]") == [False, "", ""]


def test_an_unreadable_record_names_nothing_and_says_why() -> None:
    assert _reader({"exposure_groups": UNREADABLE}, "[reader.unreadable, reader.levelName('lact_max_group', '1'), reader.unreadableText()]") == [
        True, "", UNREADABLE_ZH,
    ]


def test_plan_prose_names_a_grouped_variable_only_as_a_whole_identifier() -> None:
    text = "Compare death across lact_max_group levels; lact_max_group_n and xlact_max_group stay."

    assert _reader(CONTEXT, f"reader.inText({json.dumps(text)})") == (
        "Compare death across 乳酸分组 levels; lact_max_group_n and xlact_max_group stay."
    )


def test_every_reader_that_names_groups_is_listed_for_its_viewers() -> None:
    names = ["agent_plan.json", "result_tables.json", "question_requirements.json",
             "question_requirements_review.json", "run_context.json", "figure_gallery.json"]
    script = f"process.stdout.write(JSON.stringify({json.dumps(names)}.map(window.AGENT_EXPOSURE_LEVELS.namedIn)));"

    assert json.loads(_node(script, "screens-agent-exposure-levels.js")) == [True, True, True, True, False, False]


def test_a_result_table_names_its_grouped_cells() -> None:
    headers = ["exposure", "exposure_level", "reference_level", "contrast", "lact_max_group", "group", "n"]
    rows = [
        ["lact_max_group", "2", "1", "2 vs 1", "3", "1", "7"],
        ["age", "2", "1", "2 vs 1", "9", "1", "7"],
    ]
    call = f"[reader.tableRows({{group_by: 'lact_max_group'}}, {json.dumps(headers)}, {json.dumps(rows)}), reader.tableRows({{}}, {json.dumps(headers)}, {json.dumps(rows)})[0][5]]"

    named, ungrouped_table = _reader(CONTEXT, call)

    assert named[0] == ["乳酸分组", "乳酸 ≥2.5", "乳酸 <2.5", "乳酸 ≥2.5 vs 乳酸 <2.5", "前 24 小时未测乳酸", "乳酸 <2.5", "7"]
    # Another exposure's levels are not this grouping's codes; the grouped
    # column itself still is.
    assert named[1] == ["age", "2", "1", "2 vs 1", "9", "乳酸 <2.5", "7"]
    # A Table 1 grouped by another column keeps its group codes.
    assert ungrouped_table == "1"


# --- the plan review card ---------------------------------------------------------


def _card(summary: Mapping[str, Any], language: str) -> str:
    script = r"""
let api = null;
window.EasyICU = { guidedPi: { declare: (_name, value) => { api = value; } } };
require(process.argv[3]);
const zh = process.argv[2] === 'zh';
process.stdout.write(api.notesHtml(JSON.parse(process.argv[1]), {
  tr: (en, value) => (zh ? value : en),
  esc: window.EU_HTML.esc,
}));
"""
    return _node(
        script,
        "html-escape.js",
        "screens-agent-exposure-levels.js",
        args=(json.dumps(summary), language, str((JS / "screens-guided-pi-exposure-groups.js").resolve())),
    )


def test_the_card_names_each_level_in_the_studys_words() -> None:
    html = _card({"exposure_groups": SHOWN}, "zh")

    assert "<strong>暴露分组</strong>" in html
    assert "乳酸：乳酸 &lt;2.5（参照） / 乳酸 ≥2.5（比较） / 前 24 小时未测乳酸" in html
    # The host's English rules open below, each under its level's label.
    assert "<summary>分组规则</summary>" in html
    assert '<span>乳酸 &lt;2.5：</span><span lang="en">Lactate over 0-24 h after ICU admission: maximum &lt; 2.5 mmol/L</span>' in html
    assert ">1<" not in html and "（1）" not in html


def test_the_english_card_marks_the_compared_levels() -> None:
    html = _card({"exposure_groups": SHOWN}, "en")

    assert "Lactate: 乳酸 &lt;2.5 (reference) / 乳酸 ≥2.5 (compared) / 前 24 小时未测乳酸" in html
    assert "<summary>Grouping rules</summary>" in html


def test_a_record_without_labels_is_shown_by_its_codes() -> None:
    html = _card({"exposure_groups": project_exposure_groups([_grouping(status="codes_only")])}, "zh")

    assert "乳酸：1（参照） / 2（比较） / 3（记录没有标签）" in html


def test_an_unordered_grouping_reads_like_an_ordered_one() -> None:
    nominal = _grouping(
        scale="nominal",
        levels=[
            {"code": 1, "group": "g1", "label": "内科", "rule": "admission type medical", "unmeasured": False},
            {"code": 2, "group": "g2", "label": "外科", "rule": "admission type surgical", "unmeasured": False},
        ],
        concept="adm",
        variable="adm_group",
    )

    html = _card({"exposure_groups": project_exposure_groups([nominal])}, "zh")

    assert "入院类型：内科（参照） / 外科（比较）" in html


def test_the_card_says_when_the_grouping_record_cannot_be_read() -> None:
    html = _card({"exposure_groups": UNREADABLE}, "zh")

    assert f"<strong>暴露分组</strong><p>{UNREADABLE_ZH}</p>" in html
    assert _card({}, "zh") == ""


# --- the result and plan readers --------------------------------------------------


def _artifact(name: str, payload: Mapping[str, Any], context: Any, language: str = "zh") -> str:
    script = f"""
global.window = {{
  EU_HTML: {{
    esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    escAttr: value => String(value ?? ''),
  }},
  t: (en, zh) => {json.dumps(language)} === 'zh' ? zh : en,
  EU_LANG: {json.dumps(language)},
  icon: () => '',
}};
"""
    call = f"process.stdout.write(window.AGENT_RENDER.artifactStructuredView({json.dumps(name)}, {json.dumps(payload)}, {json.dumps(context)}));"
    return _node(
        script + "".join(f"require({json.dumps(str((JS / file).resolve()))});\n" for file in (
            "screens-agent-reader-vocab.js", "screens-agent-question-requirements.js",
            "screens-agent-exposure-levels.js", "screens-agent-render.js",
        )) + call,
    )


RESULT_TABLES = {"tables": [
    {"name": "step__exposure_outcome_distribution.csv",
     "label": "Table exposure_outcome_distribution from step 03_distribution.",
     "headers": ["row_role", "exposure_level", "n_rows", "exposure_column"],
     "rows": [["exposure_level", "1", "40", "lact_max_group"], ["exposure_level", "2", "60", "lact_max_group"]]},
    {"name": "step__table_one.csv", "label": "Table table_one from step baseline.", "group_by": "lact_max_group",
     "headers": ["variable", "group", "denominator_n"], "rows": [["age", "1", "40"], ["age", "2", "60"]]},
]}


def test_the_result_tables_name_the_groups_in_the_studys_words() -> None:
    html = _artifact("result_tables.json", RESULT_TABLES, CONTEXT)

    assert "<td>乳酸 &lt;2.5</td>" in html and "<td>乳酸 ≥2.5</td>" in html
    assert "<td>乳酸分组</td>" in html
    assert "<td>1</td>" not in html and "<td>2</td>" not in html


def test_without_the_runs_context_the_tables_keep_their_codes() -> None:
    html = _artifact("result_tables.json", RESULT_TABLES, None)

    assert "<td>1</td>" in html and "乳酸 &lt;2.5" not in html
    assert UNREADABLE_ZH not in html


def test_the_result_tables_say_when_the_grouping_record_cannot_be_read() -> None:
    html = _artifact("result_tables.json", RESULT_TABLES, {"exposure_groups": UNREADABLE})

    assert f'<p class="ag-result-reader-note">{UNREADABLE_ZH}</p>' in html
    assert "<td>1</td>" in html


def test_a_grouped_column_header_reads_by_its_group_name() -> None:
    tables = {"tables": [{"name": "step__counts.csv", "label": "Counts.", "headers": ["lact_max_group", "n"],
                          "rows": [["1", "40"], ["3", "5"]]}]}

    html = _artifact("result_tables.json", tables, CONTEXT)

    assert "<th>乳酸分组</th>" in html
    assert "<td>前 24 小时未测乳酸</td>" in html


PLAN = {
    "display_labels": {"death": "院内死亡"},
    "design_selection": {"candidates": [{"disposition": "selected", "required_variables": ["lact_max_group", "death"]}]},
    "steps": [
        {"step_id": "primary_model", "method": "logistic_regression", "planned_analysis_role": "primary",
         "intent": "按 lact_max_group 各组比较院内死亡。",
         "model_requirements": [{"exposure_source": "lact_max_group", "analysis_role": "primary",
                                 "exposure_levels": ["1", "2"], "exposure_reference_level": "1",
                                 "primary_contrast_level": "2"}]},
    ],
}


def test_the_plan_names_its_grouped_variable_and_levels() -> None:
    html = _artifact("agent_plan.json", PLAN, CONTEXT)

    assert "<span>乳酸分组</span>" in html
    assert "主模型中的暴露水平：</strong>乳酸 &lt;2.5、乳酸 ≥2.5（参照 乳酸 &lt;2.5）；主要对比 乳酸 ≥2.5 vs 乳酸 &lt;2.5" in html
    assert "按 乳酸分组 各组比较院内死亡。" in html


def test_a_label_the_plan_registers_is_kept() -> None:
    plan = {**PLAN, "display_labels": {**PLAN["display_labels"], "lact_max_group": "最高乳酸分层"}}

    html = _artifact("agent_plan.json", plan, CONTEXT)

    assert "<span>最高乳酸分层</span>" in html and "<span>乳酸分组</span>" not in html


def test_without_the_runs_context_the_plan_keeps_its_codes() -> None:
    html = _artifact("agent_plan.json", PLAN, None)

    assert "主模型中的暴露水平：</strong>1、2（参照 1）" in html


REQUIREMENTS = {
    "schema_version": "easyicu.question_requirements/1",
    "route": "outline",
    "judged": [{"id": "R1", "kind": "estimand", "quote": "乳酸分组", "disposition": "not_covered",
                "concepts": ["lact_max_group"], "reading_step_ids": []}],
    "unstated": [{"evidence": "乳酸", "concepts": ["lact_max_group", "lact_max"], "reading_step_ids": ["s1"]}],
}


def test_the_question_requirements_name_a_grouped_column_by_its_group_name() -> None:
    html = _artifact("question_requirements.json", REQUIREMENTS, CONTEXT)

    assert "（乳酸分组）" in html
    assert "<td>乳酸分组, lact_max</td>" in html
    assert "lact_max_group" not in html
    assert "（lact_max_group）" in _artifact("question_requirements.json", REQUIREMENTS, None)


# --- the run answer ---------------------------------------------------------------


def _answer(context: Mapping[str, Any]) -> str:
    payloads = {
        "result_tables.json": {"tables": [{
            "name": "step__exposure_outcome_distribution.csv", "label": "Table x.", "evidence_id": "t",
            "headers": ["row_role", "exposure_level", "n_rows", "exposure_denominator", "exposure_pct",
                        "outcome_events", "outcome_denominator", "outcome_rate_pct"],
            "rows": [
                ["exposure_level", "1", "600", "1000", "60.0", "60", "600", "10.0"],
                ["exposure_level", "2", "300", "1000", "30.0", "60", "300", "20.0"],
                ["exposure_level", "3", "100", "1000", "10.0", "5", "100", "5.0"],
                ["overall", "", "1000", "1000", "100.0", "125", "1000", "12.5"],
            ],
        }]},
        "agent_plan.json": {
            "display_labels": {"death": "院内死亡"},
            "steps": [{
                "step_id": "distribution", "method": "descriptive", "planned_analysis_role": "primary",
                "intent": "Compare death across lact_max_group levels.",
                "exposure_outcome_distribution_spec": {"exposure": "lact_max_group", "outcome": "death", "exposure_levels": [1, 2, 3]},
            }],
        },
        "run_context.json": {"question": "乳酸分组与院内死亡", "source": {"label": "eICU"}, **context},
        "figure_gallery.json": {"figures": []},
        "manuscript_provenance.json": {"claims": []},
    }
    script = r"""
const path = require('node:path');
global.window = global;
global.EU_LANG = 'zh';
global.t = (en, zh) => zh;
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
global.EU_CATALOG = { dict: { death: ['In-hospital Mortality', '院内死亡', 'boolean'] } };
const [payloadText, ...files] = process.argv.slice(1);
for (const file of files) require(path.resolve(file));
const payloads = JSON.parse(payloadText);
const modules = global.EasyICU.guidedPi;
const tr = (en, zh) => zh;
const esc = global.EU_HTML.esc;
const resources = modules.require('resources').create({ esc });
const ref = artifact => ({ kind: 'research_artifact', run_id: 'run_g', artifact, sha256: 'a'.repeat(64) });
const latest = {
  present: true, analysis_results_available: true, analysis_validated: true, numeric_verified: true,
  run_id: 'run_g', figure_count: 0,
  artifact_refs: Object.keys(payloads).map(ref),
};
const workflow = { current_stage: 'interpretation', stages: [{ id: 'analysis', status: 'complete' }] };
const api = { loadPiCopilotResearchArtifact: async (_project, _run, artifact) => ({ ok: true, payload: payloads[artifact] }) };
const outcome = modules.require('runOutcome').create({
  tr, esc, iconHtml: () => '', resourceButton: resources.button, api: () => api, projectId: () => 'project_g',
  host: () => null,
});
(async () => {
  await outcome.loadScientificReview(latest, workflow);
  for (let turn = 0; turn < 5; turn += 1) await new Promise(resolve => setImmediate(resolve));
  process.stdout.write(outcome.render(latest, workflow));
})().catch(error => { console.error(error); process.exit(1); });
"""
    files = (
        "screens-guided-pi-modules.js", "screens-guided-pi-resources.js", "screens-agent-exposure-levels.js",
        "screens-guided-pi-result-summary.js", "screens-guided-pi-run-answer.js", "screens-guided-pi-run-outcome.js",
        "screens-guided-pi-activity.js", "screens-guided-pi-transcript.js",
    )
    return _node(script, args=(json.dumps(payloads), *[str((JS / name).resolve()) for name in files]))


def test_the_run_answer_names_the_groups_in_the_studys_words() -> None:
    card = _answer({"exposure_groups": SHOWN})

    assert "eICU 中按乳酸分组的院内死亡：「乳酸 &lt;2.5」10.00%（60/600）；「乳酸 ≥2.5」20.00%（60/300）；「前 24 小时未测乳酸」5.00%（5/100）。" in card
    assert "「1」" not in card and UNREADABLE_ZH not in card


def test_the_run_answer_says_when_the_grouping_record_cannot_be_read() -> None:
    card = _answer({"exposure_groups": UNREADABLE})

    assert "「1」10.00%" in card
    assert UNREADABLE_ZH in card


def test_the_summary_names_an_estimates_compared_groups() -> None:
    tables = {"tables": [{
        "name": "step__adjusted_association.csv", "label": "Estimates.",
        "headers": ["exposure", "exposure_level", "reference_level", "contrast", "is_primary_contrast",
                    "fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "n"],
        "rows": [["lact_max_group", "2", "1", "2 vs 1", "true", "fitted", "1.8", "1.2", "2.6", "odds_ratio", "900"]],
    }]}
    script = r"""
global.t = (en, zh) => zh;
let api = null;
window.EasyICU = { guidedPi: { declare: (_name, value) => { api = value; } } };
require(process.argv[3]);
const summary = api.summarize(JSON.parse(process.argv[1]), {}, JSON.parse(process.argv[2]));
process.stdout.write(JSON.stringify([summary.estimates[0].label, summary.groupsUnreadable]));
"""

    def summarize(context: Any) -> list[Any]:
        return json.loads(_node(
            script, "screens-agent-exposure-levels.js",
            args=(json.dumps(tables), json.dumps(context), str((JS / "screens-guided-pi-result-summary.js").resolve())),
        ))

    assert summarize(CONTEXT) == ["乳酸 ≥2.5 vs 乳酸 <2.5", False]
    assert summarize({"exposure_groups": UNREADABLE}) == ["2 vs 1", True]


# --- the page ---------------------------------------------------------------------


def test_every_reader_of_a_runs_groups_is_given_its_context() -> None:
    def source(name: str) -> str:
        return (JS / name).read_text(encoding="utf-8")

    run_files = source("screens-guided-pi-run-files.js")
    preview = source("screens-guided-pi-preview.js")
    assert "api.loadAgentRunArtifact(review.project_dir, 'run_context.json')" in run_files
    assert "window.AGENT_EXPOSURE_LEVELS.namedIn(name)" in run_files
    assert "artifactStructuredView(state.artifact.name, state.artifact.payload || {}, state.artifact.context)" in run_files
    assert "loadPiCopilotResearchArtifact(state.projectId, state.resource.run_id, 'run_context.json')" in preview
    assert "window.AGENT_EXPOSURE_LEVELS.namedIn(state.resource.artifact)" in preview
    assert "artifactStructuredView(state.resource.artifact, state.payload || {}, state.runContext)" in preview
    assert "AGENT_QUESTION_REQUIREMENTS.view(n, p, { artifactTable, esc, context })" in source("screens-agent-render.js")
    assert "summarize(payload('result_tables.json'), plan, context)" in source("screens-guided-pi-run-answer.js")
    assert "summarize(p.result_tables || {}, p.plan, context)" in source("screens-guided-pi-analysis-report.js")


def test_the_card_and_page_load_the_readers() -> None:
    confirmation = (JS / "screens-guided-pi-confirmation.js").read_text(encoding="utf-8")
    index = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "${trialBody}${groupNotes}${questionNotes}" in confirmation
    assert "window.EasyICU.guidedPi.optional('exposureGroups')" in confirmation
    token = "v=20261010-exposure-labels1"
    for script in (
        "screens-agent-exposure-levels.js", "screens-agent-render.js", "screens-guided-pi-exposure-groups.js",
        "screens-guided-pi-confirmation.js", "screens-guided-pi-result-summary.js", "screens-guided-pi-run-answer.js",
        "screens-guided-pi-run-files.js", "screens-guided-pi-analysis-report.js",
        "screens-agent-question-requirements.js",
    ):
        assert f'src="js/{script}?{token}"' in index
    # The preview moved on with the folder selection's close (source-gate1).
    assert 'src="js/screens-guided-pi-preview.js?v=20261010-source-gate1"' in index
    # The card's stylesheet moved on once: its rules toggle got a 24px hit height.
    assert 'href="css/guided-pi-exposure-groups.css?v=20261010-exposure-labels2"' in index
    # Every reader finds the owner when it renders.
    assert index.index("js/screens-agent-exposure-levels.js?") < index.index("js/screens-agent-render.js?")
