"""The card, result reader and Writer share a synthetic plan's contrast."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.exposure_group_labels import exposure_group_label_rows
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.workflow import HumanReviewPending, HumanReviewRequest
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.exposure_group_notes import project_exposure_groups
from tests.support.grouped_contrast_fixtures import _bmi, _record, grouped_context, skeleton
from tests.support.node import run_node
from tests.webserver.copilot.research_workflow_fixtures import complete_study


def _views(groups, language):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    js = Path(agent_pipeline_runs.__file__).parent / "static/js"
    script = """
global.window = global;
let card;
window.EasyICU = {guidedPi: {declare: (_name, api) => {card = api;}}};
"""
    for name in ("html-escape.js", "screens-agent-exposure-levels.js", "screens-guided-pi-exposure-groups.js"):
        script += f"require({json.dumps(str(js / name))});\n"
    script += f"""
const source = {{exposure_groups: {json.dumps(groups)}}};
const tr = (en, zh) => ({json.dumps(language)} === 'zh' ? zh : en);
const reader = window.AGENT_EXPOSURE_LEVELS.reader(source, tr);
process.stdout.write(JSON.stringify({{
    card: card.notesHtml(source, {{tr, esc: window.EU_HTML.esc}}),
    contrast: reader.contrastName('exposure_flag', '1 vs 3'),
}}));
"""
    result = run_node(node, script)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _source():
    plan, _ = compile_progressive_plan(skeleton=skeleton(3, 1), context=grouped_context(stated=[]))
    record = _record({**_bmi(), "reference": None, "contrast": None})
    record["compiled"]["groupings"][0].update(
        variable="exposure_flag", labels={"g1": "组甲", "g2": "组乙", "g3": "组丙"},
    )
    return plan, record


@pytest.mark.parametrize("language,reference,comparison", [
    ("zh", "组丙（参照）", "组甲（比较）"),
    ("en", "Band C (reference)", "Band A (compared)"),
])
def test_card_and_result_direction_use_the_compiled_plan(language, reference, comparison):
    plan, record = _source()
    if language == "en":
        record["compiled"]["groupings"][0]["labels"] = {
            "g1": "Band A", "g2": "Band B", "g3": "Band C",
        }
    groups = project_exposure_groups(exposure_group_label_rows(record, record_sha256="a" * 64, plan=plan))
    views = _views(groups, language)
    html = views["card"]
    assert reference in html and comparison in html
    text = views["contrast"]
    assert text == ("Band A vs " if language == "en" else "组甲 vs ") + reference


def test_no_plan_means_no_unstated_reference_badge():
    _, record = _source()
    groups = project_exposure_groups(exposure_group_label_rows(record, record_sha256="a" * 64))
    for language in ("en", "zh"):
        html = _views(groups, language)["card"]
        assert "(reference)" not in html and "（参照）" not in html
        assert "(compared)" not in html and "（比较）" not in html


def test_digest_bound_pending_plan_and_current_plan_feed_the_same_projection(tmp_path, monkeypatch):
    plan, record = _source()
    (tmp_path / "exposure_groupings.json").write_text(json.dumps(record))
    authority = PlanReviewAuthority.create(plan=plan)
    pending = HumanReviewPending(
        run_id="synthetic-contrast", thread_id="synthetic", run_dir=str(tmp_path),
        requests=(HumanReviewRequest.create(
            kind="scientific_stop", summary="Review the compiled contrast",
            authority_sha256="a" * 64, payload={
                "reason": "operator_plan_approval_required",
                "plan_review_authority": authority.model_dump(mode="json"),
            },
        ),),
    )
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(agent_pipeline_runs._PendingRun(
        pipeline=SimpleNamespace(), pending=pending, wrapper_dir=tmp_path / "wrapper",
        study=complete_study(), provider={}, acquisition=SimpleNamespace(), created_at=1.0,
    ))
    review = agent_pipeline_runs.pending_review(pending.run_id)
    (row,) = review["exposure_group_labels"]
    assert (row["reference"], row["contrast"]) == (3, 1)

    monkeypatch.setattr(agent_pipeline_runs, "load_current_plan_authority",
                        lambda _: plan.model_dump(mode="json"))
    wrapper = tmp_path / "wrapper"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper, study=complete_study(), provider={},
        acquisition=SimpleNamespace(
            selection=SimpleNamespace(selected_concepts=[]),
            coverage=SimpleNamespace(sufficient=True), materialized_concepts=[],
        ),
        run_dir=tmp_path,
    )
    context = json.loads((wrapper / "run_context.json").read_text())
    assert context["exposure_groups"] == project_exposure_groups(review["exposure_group_labels"])


def test_a_contrast_conflict_has_a_reader_facing_planning_stop():
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

    message = agent_pipeline_runs._progressive_compile_failure_message(
        ProgressivePlanCompileError("progressive_exposure_group_contrast_mismatch", "private-column"),
    )
    assert "reference or comparison" in message and "conflicting contrasts" in message
    assert "private-column" not in message
